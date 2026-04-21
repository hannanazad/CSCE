"""Q-Drifting agent: offline RL via a one-step drifting-based policy.

Algorithm overview
------------------
Q-Drifting trains a policy that generates actions in a single forward pass by
combining three forces into a "drifting field":

  V(x) = V_attract(x)  —  V_repulse(x)  +  α · ∇Q(x)

  V_attract  pulls generated actions toward the frozen BC policy output,
             keeping the learned policy grounded in the behavioral distribution.

  V_repulse  pushes generated actions apart from each other within each state,
             preventing mode collapse to a single action.

  α · ∇Q    steers actions toward higher Q-value regions, providing the RL
             improvement signal on top of the BC prior.

Training has two phases (orchestrated by main.py):

  Phase 1 — BC Pretraining
    `bc_update` trains `bc_actor` via MSE loss on dataset actions.
    Goal: learn a smooth obs->action mapping that generalises across states.

  Phase 2 — Offline RL
    `update` jointly trains the critic (TD backup) and `drifting_actor`.
    The drifting actor learns to output actions that are close to
    (x_gen + V(x_gen)), where V is computed with the frozen BC policy.
    bc_actor receives zero gradient during this phase and stays fixed.

Inference
---------
At test time, `sample_actions` does a single forward pass:
  noise ~ N(0, I)  →  drifting_actor(obs, noise)  →  clip to [-1, 1]
Best-of-N selection (N=1 by default) picks the action with the highest
mean Q-value across the critic ensemble.

Reference papers
----------------
- Q-Learning with Adjoint Matching (Li & Levine, 2026)
- Generative Modeling via Drifting (Deng et al., 2026)
"""
import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import BCPolicy, DriftingActor, Value


class QDriftAgent(flax.struct.PyTreeNode):
    """JAX-compatible offline RL agent using a one-step drifting policy.

    All methods decorated with `@jax.jit` are traced once and then execute
    on-device without Python overhead in the training loop.

    Fields
    ------
    rng     : JAX PRNG key (split before each stochastic operation).
    network : TrainState wrapping the ModuleDict of all sub-networks.
    config  : Frozen hyperparameter dict (non-pytree field).
    """

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    # ==================================================================
    # Critic loss  (pessimistic ensemble TD backup)
    # ==================================================================

    def critic_loss(self, batch, grad_params, rng):
        """Compute the pessimistic TD-error loss for the critic ensemble.

        Uses the drifting actor to generate next-state actions and forms
        the pessimistic target:

            Q_target = r + γ^H · mask · (mean(Q_next) − ρ · std(Q_next))

        The pessimism term (ρ · std) discourages overestimating Q in
        out-of-distribution regions, which is the key challenge in offline RL.

        Args:
            batch:       Sampled batch with keys observations, actions, rewards,
                         masks, next_observations, valid (all n-step expanded).
            grad_params: Differentiable parameter dict (passed by apply_loss_fn).
            rng:         PRNG key for next-action sampling.

        Returns:
            (loss scalar, info dict)
        """
        # Take the first action in the sequence (index 0 along the horizon axis)
        batch_actions = batch['actions'][..., 0, :]

        next_actions = self.sample_actions(
            batch['next_observations'][..., -1, :], rng=rng
        )
        next_actions = jnp.clip(next_actions, -1, 1)

        # Pessimistic bootstrap target
        next_qs = self.network.select('target_critic')(
            batch['next_observations'][..., -1, :], next_actions
        )
        next_q = next_qs.mean(axis=0) - self.config['rho'] * next_qs.std(axis=0)

        target_q = batch['rewards'][..., -1] + (
            (self.config['discount'] ** self.config['horizon_length'])
            * batch['masks'][..., -1]
            * next_q
        )

        q = self.network.select('critic')(
            batch['observations'], batch_actions, params=grad_params
        )
        td_error = jnp.abs(q - target_q)          # (num_qs, B)
        valid_mask = batch['valid'][..., -1]       # (B,)
        critic_loss = (jnp.square(q - target_q) * valid_mask).mean()

        return critic_loss, {
            'critic/critic_loss': critic_loss,
            # Q-value statistics on the current (live) critic
            'critic/q_mean': q.mean(),
            'critic/q_max': q.max(),
            'critic/q_min': q.min(),
            'critic/q_std': q.std(axis=0).mean(),   # mean ensemble spread
            # TD-error statistics (lower over time = critic converging)
            'critic/td_error_mean': td_error.mean(),
            'critic/td_error_max': td_error.max(),
            # Target Q-value statistics (sanity-checks reward / discount scale)
            'critic/target_q_mean': target_q.mean(),
            'critic/target_q_max': target_q.max(),
            'critic/target_q_min': target_q.min(),
            # Batch reward statistics (verify dataset rewards are as expected)
            'batch/reward_mean': batch['rewards'][..., -1].mean(),
            'batch/reward_std': batch['rewards'][..., -1].std(),
            'batch/mask_mean': batch['masks'][..., -1].mean(),  # fraction non-terminal
        }

    # ==================================================================
    # Drifting field  (the three-force combination)
    # ==================================================================

    @staticmethod
    def _compute_drifting_field(
        x_gen,          # (B, K, A)  — generated actions
        a_bc,           # (B, A)     — frozen BC policy actions (attraction target)
        q_grad,         # (B, K, A)  — Q-gradient at generated actions
        drift_temp,     # scalar     — repulsion kernel temperature
        q_drift_scale,  # scalar     — weight on Q-gradient term (α)
    ):
        """Combine the three drifting forces into a single field V.

        V = V_attract − V_repulse + α · ∇Q

        V_attract  = a_bc − x_gen
            Points from each generated action toward the BC policy output.
            Acts as a soft anchor to the behavioral distribution.

        V_repulse  = Σ_j  w_ij · (x_gen_j − x_gen_i)   (mean-shift, negated)
            Weighted by softmax(−dist/T) so nearby generated actions repel
            more strongly.  Prevents all K samples from collapsing to the
            same mode.

        α · ∇Q    = α · ∂Q/∂action
            Pushes each generated action in the direction of increasing Q.
            This is the RL signal that lifts the policy above pure BC.

        Args:
            x_gen:         Generated actions, shape (B, K, A).
            a_bc:          BC policy actions (frozen), shape (B, A).
            q_grad:        Gradient of mean-Q w.r.t. x_gen, shape (B, K, A).
            drift_temp:    Temperature T for the repulsion kernel (higher T →
                           smoother, more uniform repulsion).
            q_drift_scale: Weight α on the Q-gradient term.

        Returns:
            V:          Combined drifting field, shape (B, K, A).
            V_attract:  Attraction component, shape (B, K, A).
            V_repulse:  Repulsion component (positive = away from neighbours),
                        shape (B, K, A).
            V_q:        Scaled Q-gradient component, shape (B, K, A).
        """
        B, K, A = x_gen.shape

        # --- Attraction: pull each generated action toward BC output ---
        a_pos = a_bc[:, None, :]            # (B, 1, A)  broadcast over K
        V_attract = a_pos - x_gen           # (B, K, A)

        # --- Repulsion: push generated actions away from each other ---
        if K > 1:
            # diff[b, i, j] = x_gen[b, j] − x_gen[b, i]   shape (B, K, K, A)
            diff = x_gen[:, None, :, :] - x_gen[:, :, None, :]
            dist = jnp.linalg.norm(diff, axis=-1, keepdims=True)  # (B, K, K, 1)

            # Softmax weights: nearby actions repel more (lower dist → higher weight)
            logits = -dist.squeeze(-1) / drift_temp          # (B, K, K)
            mask = jnp.eye(K)[None, :, :]                    # mask out self
            logits = jnp.where(mask > 0.5, -1e9, logits)
            weights = jax.nn.softmax(logits, axis=2)         # (B, K, K)

            # V_repulse[b, i] = Σ_j  w[b,i,j] · (x_gen[b,j] − x_gen[b,i])
            # diff is indexed [b, i, j] = x_gen[j] − x_gen[i], so einsum
            # over j gives the weighted mean-shift *toward* other actions,
            # which we negate (subtract) to get repulsion.
            V_repulse = jnp.einsum('bkj,bkja->bka', weights, diff)
        else:
            V_repulse = jnp.zeros_like(x_gen)

        # --- Q-gradient drift ---
        V_q = q_drift_scale * q_grad        # (B, K, A)

        return V_attract - V_repulse + V_q, V_attract, V_repulse, V_q

    # ==================================================================
    # BC pretraining loss
    # ==================================================================

    def bc_loss(self, batch, grad_params, rng):
        """MSE behaviour cloning loss for the bc_actor.

        Trains bc_actor to predict dataset actions from observations.
        Only bc_actor parameters receive non-zero gradients during bc_update.

        Args:
            batch:       Training batch (only observations and actions used).
            grad_params: Differentiable parameter dict.
            rng:         Unused (kept for uniform method signature).

        Returns:
            (loss scalar, info dict)
        """
        batch_actions = batch['actions'][..., 0, :]  # (B, A)
        obs = batch['observations']                   # (B, obs_dim)

        pred_actions = self.network.select('bc_actor')(obs, params=grad_params)
        per_dim_err  = jnp.abs(pred_actions - batch_actions)          # (B, A)
        mse          = jnp.square(pred_actions - batch_actions).sum(axis=-1)  # (B,)
        valid        = batch['valid'][..., -1]                         # (B,)
        loss         = (mse * valid).mean()

        return loss, {
            'bc/loss': loss,
            # Mean absolute error across all (B, A) entries
            'bc/action_err': per_dim_err.mean(),
            # Max per-dimension error (shows worst-case prediction)
            'bc/action_err_max': per_dim_err.max(),
            # L2 norm of each predicted action (should match dataset action scale)
            'bc/pred_action_norm': jnp.linalg.norm(pred_actions, axis=-1).mean(),
            # L2 norm of each dataset action (reference scale)
            'bc/dataset_action_norm': jnp.linalg.norm(batch_actions, axis=-1).mean(),
            # Fraction of valid samples in the batch
            'bc/valid_fraction': valid.mean(),
        }

    @staticmethod
    def _bc_update(agent, batch):
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.bc_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def bc_update(self, batch):
        """Single BC pretraining step (jit-compiled)."""
        return self._bc_update(self, batch)

    # ==================================================================
    # Drifting actor loss
    # ==================================================================

    def actor_loss(self, batch, grad_params, rng):
        """Drifting actor loss: match generated actions to (x_gen + V(x_gen)).

        Procedure:
          1. Sample K noise vectors per state.
          2. Forward pass through drifting_actor → x_gen  (B, K, A).
          3. Compute Q-gradient at x_gen (stop-grad through actor).
          4. Query frozen bc_actor for the attraction target.
          5. Compute drifting field V = V_attract − V_repulse + α·∇Q.
          6. MSE loss: ||x_gen − clip(x_gen + V)||².

        The actor learns to output actions that are already "drifted" in the
        direction V, effectively internalising the drifting field into its
        weights so that inference requires only one forward pass.

        Args:
            batch:       Training batch.
            grad_params: Differentiable parameter dict.
            rng:         PRNG key for noise sampling.

        Returns:
            (loss scalar, info dict)
        """
        batch_size = batch['observations'].shape[0]
        action_dim = batch['actions'].shape[-1]
        obs = batch['observations']                   # (B, obs_dim)
        K = self.config['num_drift_samples']

        rng, noise_rng = jax.random.split(rng)
        noise = jax.random.normal(noise_rng, (batch_size, K, action_dim))

        # Expand obs over the K sample dimension
        obs_K = jnp.repeat(obs[:, None, :], K, axis=1)  # (B, K, obs_dim)

        # --- Generate K actions per state ---
        x_gen = self.network.select('drifting_actor')(
            obs_K, noise, params=grad_params
        )  # (B, K, A), tanh-squashed

        # --- Q-gradient at generated actions (stop-grad through actor params) ---
        x_gen_sg = jax.lax.stop_gradient(x_gen)
        critic_name = 'target_critic' if self.config['use_target_grad'] else 'critic'

        def q_fn(actions):
            qs = self.network.select(critic_name)(obs_K, actions)  # (num_qs, B, K)
            return qs.mean(axis=0).sum()

        q_grad = jax.grad(q_fn)(x_gen_sg)              # (B, K, A)
        q_grad = jnp.clip(q_grad, -1.0, 1.0)           # clip for stability

        # --- BC policy attraction target (frozen after Phase 1) ---
        bc_actions = jax.lax.stop_gradient(
            self.network.select('bc_actor')(obs)
        )  # (B, A)

        # --- Drifting field (returns combined V and all three components) ---
        V, V_attract, V_repulse, V_q = self._compute_drifting_field(
            x_gen_sg,
            bc_actions,
            q_grad,
            self.config['drift_temperature'],
            self.config['q_drift_scale'],
        )

        # --- Regression target: x_gen + V, clipped to valid action range ---
        target = jax.lax.stop_gradient(jnp.clip(x_gen_sg + V, -1, 1))

        sq_err = jnp.square(x_gen - target).sum(axis=-1)  # (B, K)
        drift_loss = jnp.mean(sq_err * batch['valid'][..., -1, None])

        # Per-sample norms of each force component (for interpretability)
        attract_mag  = jnp.linalg.norm(V_attract,  axis=-1).mean()  # (scalar)
        repulse_mag  = jnp.linalg.norm(V_repulse,  axis=-1).mean()
        q_mag        = jnp.linalg.norm(V_q,         axis=-1).mean()
        drift_mag    = jnp.linalg.norm(V,           axis=-1).mean()

        return drift_loss, {
            'actor/drift_loss': drift_loss,
            # --- Overall field magnitude ---
            'actor/drift_magnitude': drift_mag,
            # --- Per-force magnitudes (key for ablation and debugging) ---
            # v_attract_mag should decrease as the actor learns to track bc_actor.
            # v_repulse_mag should be > 0 if K > 1 and samples aren't identical.
            # v_q_mag should be proportional to q_drift_scale * ||∇Q||.
            'actor/v_attract_mag': attract_mag,
            'actor/v_repulse_mag': repulse_mag,
            'actor/v_q_mag': q_mag,
            # --- Force balance ratio: how much Q-guidance vs BC-anchoring ---
            # Values close to 1 → forces roughly balanced.
            # Very large values → Q-gradient dominating (potential instability).
            'actor/q_attract_ratio': q_mag / (attract_mag + 1e-8),
            # --- Generated action statistics ---
            'actor/gen_action_mean': jnp.abs(x_gen).mean(),
            'actor/gen_action_std': x_gen.std(),
            # BC policy action norm (sanity-check that bc_actor is non-trivial)
            'actor/bc_action_norm': jnp.linalg.norm(bc_actions, axis=-1).mean(),
            # Per-sample squared error before masking (measure of field size)
            'actor/sq_err_mean': sq_err.mean(),
        }

    # ==================================================================
    # Combined offline RL loss
    # ==================================================================

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Sum critic and actor losses; used by `update`."""
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        info.update(critic_info)

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        info.update(actor_info)

        return critic_loss + actor_loss, info

    # ==================================================================
    # Target network update  (exponential moving average)
    # ==================================================================

    def target_update(self, network, module_name):
        """Polyak-average the live network into the target network.

        τ controls how quickly the target follows the live network.
        Small τ (e.g. 0.005) keeps the target stable, which reduces
        the "moving target" problem in TD learning.
        """
        new_target = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target

    # ==================================================================
    # Update steps
    # ==================================================================

    @staticmethod
    def _update(agent, batch):
        """Single gradient step on critic + drifting actor."""
        new_rng, rng = jax.random.split(agent.rng)

        def loss_fn(grad_params):
            return agent.total_loss(batch, grad_params, rng=rng)

        new_network, info = agent.network.apply_loss_fn(loss_fn=loss_fn)
        agent.target_update(new_network, 'critic')

        return agent.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def update(self, batch):
        """Single offline RL update step (jit-compiled)."""
        return self._update(self, batch)

    @jax.jit
    def batch_update(self, batch):
        """Multiple update steps via lax.scan (for UTD ratio > 1)."""
        agent, infos = jax.lax.scan(self._update, self, batch)
        return agent, jax.tree_util.tree_map(lambda x: x.mean(), infos)

    # ==================================================================
    # Sampling  (inference — one forward pass, no ODE)
    # ==================================================================

    @jax.jit
    def sample_actions(self, observations, rng):
        """Sample an action for each observation in the batch.

        Draws `best_of_n` noise samples, generates that many candidate
        actions, and returns the one with the highest mean Q-value.
        With best_of_n=1 (default) this is a single forward pass.

        Args:
            observations: Observation array, shape (..., obs_dim).
            rng:          JAX PRNG key.

        Returns:
            Selected action, shape (..., action_dim).
        """
        action_dim = self.config['action_dim']
        N = self.config['best_of_n']

        noises = jax.random.normal(
            rng,
            (*observations.shape[: -len(self.config['ob_dims'])], N, action_dim),
        )
        obs_N = jnp.repeat(observations[..., None, :], N, axis=-2)

        actions = jnp.clip(
            self.network.select('drifting_actor')(obs_N, noises), -1, 1
        )

        # Best-of-N: pick action with highest mean Q across ensemble
        q = self.network.select('critic')(obs_N, actions).mean(axis=0)
        indices = jnp.argmax(q, axis=-1)

        bshape = indices.shape
        bsize = indices.reshape(-1).shape[0]
        actions = jnp.reshape(actions, (-1, N, action_dim))[
            jnp.arange(bsize), indices.reshape(-1), :
        ].reshape(bshape + (action_dim,))

        return actions

    # ==================================================================
    # Construction
    # ==================================================================

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        """Initialise the agent and all sub-networks.

        Args:
            seed:            Integer random seed.
            ex_observations: Example observation array (shape only, not values).
            ex_actions:      Example action array (shape only, not values).
            config:          Agent hyperparameter ConfigDict.

        Returns:
            Initialised QDriftAgent.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng)

        ob_dims = ex_observations.shape
        action_dim = ex_actions.shape[-1]

        # Critic ensemble: Q(obs, action) → (num_qs,) scalars
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
            num_ensembles=config['num_qs'],
        )

        # Drifting actor: (obs, noise) → action  (one-step generator)
        actor_def = DriftingActor(
            action_dim=action_dim,
            hidden_dims=config['actor_hidden_dims'],
            layer_norm=config['actor_layer_norm'],
        )

        # BC policy: obs → action  (pretrained then frozen)
        bc_def = BCPolicy(
            action_dim=action_dim,
            hidden_dims=config['bc_hidden_dims'],
            layer_norm=config['bc_layer_norm'],
        )

        network_info = {
            'critic':        (critic_def,              (ex_observations, ex_actions)),
            'target_critic': (copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            'drifting_actor': (actor_def,              (ex_observations, ex_actions)),
            'bc_actor':      (bc_def,                  (ex_observations,)),
        }

        network_def = ModuleDict({k: v[0] for k, v in network_info.items()})
        network_args = {k: v[1] for k, v in network_info.items()}

        if config['clip_grad']:
            tx = optax.chain(
                optax.clip_by_global_norm(max_norm=1.0),
                optax.adam(learning_rate=config['lr']),
            )
        else:
            tx = optax.adam(learning_rate=config['lr'])

        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=tx)

        # Initialise target critic as a copy of the live critic
        network.params['modules_target_critic'] = network.params['modules_critic']

        config = dict(config)
        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


# ==================================================================
# Default hyperparameters
# ==================================================================

def get_config():
    """Return the default agent hyperparameter config.

    Override individual values on the command line with --agent.KEY=VALUE.
    """
    config = ml_collections.ConfigDict(dict(
        agent_name='qdrift',

        # Placeholders filled in by QDriftAgent.create()
        ob_dims=ml_collections.config_dict.placeholder(list),
        action_dim=ml_collections.config_dict.placeholder(int),

        # ---- Optimisation ----
        lr=3e-4,
        batch_size=256,

        # ---- Network architecture ----
        actor_hidden_dims=(512, 512, 512, 512),
        actor_layer_norm=False,
        value_hidden_dims=(512, 512, 512, 512),
        value_layer_norm=True,
        bc_hidden_dims=(512, 512, 512, 512),
        bc_layer_norm=False,

        # ---- N-step returns ----
        # horizon_length is set by main.py from --horizon_length
        horizon_length=ml_collections.config_dict.placeholder(int),

        # ---- Offline RL ----
        num_qs=10,          # critic ensemble size
        rho=0.5,            # pessimism coefficient
        discount=0.99,      # γ  (use 0.995 for giant mazes)
        tau=0.005,          # target network EMA rate

        # ---- Inference ----
        best_of_n=1,        # best-of-N sampling at test time (1 = no resampling)

        # ---- Drifting field ----
        num_drift_samples=8,    # K: generated actions per state during training
        drift_temperature=1.0,  # T: repulsion kernel temperature
        q_drift_scale=0.1,      # α: weight on Q-gradient term

        # ---- Stability ----
        clip_grad=True,         # clip global gradient norm to 1.0
        use_target_grad=True,   # use target critic for Q-gradient (more stable)
    ))
    return config
