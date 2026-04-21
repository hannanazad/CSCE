"""Neural network building blocks for Q-Drifting.

Network hierarchy
-----------------
MLP                 — shared backbone: Linear → (activation → LayerNorm)* → Linear
  └─ Value          — critic ensemble: Q(obs, action) → (num_qs,) scalars
  └─ BCPolicy       — behaviour cloning policy: obs → action  (no noise)
  └─ DriftingActor  — one-step stochastic actor: (obs, noise) → action

All action outputs are tanh-squashed to (-1, 1).

Design notes
------------
- Layer norm is ON by default in Value (stabilises large Q ensembles) and
  OFF by default in BCPolicy/DriftingActor (typically not needed for actors).
- The critic uses `nn.vmap` (via `ensemblize`) to share a single forward
  pass across the ensemble dimension, which is more memory-efficient than
  looping and gives vectorised gradients.
- DriftingActor takes a noise vector alongside the observation so the same
  network can generate diverse actions from different noise samples during
  training (the K-sample repulsion mechanism).
"""
from typing import Any, Sequence

import flax.linen as nn
import jax.numpy as jnp

default_init = nn.initializers.xavier_uniform


def default_init_(scale=1.0):
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    """Vectorise a Flax module over an ensemble dimension using nn.vmap.

    Each ensemble member has independent parameters (split_rngs=True).
    The output gains a leading axis of size `num_qs`.

    Args:
        cls:     Flax Module class to vectorise.
        num_qs:  Number of ensemble members.
        in_axes: Which input axes to map over (None = broadcast all inputs).
        out_axes: Which output axis receives the ensemble dimension.

    Returns:
        A new Module class that runs `num_qs` independent copies in parallel.
    """
    return nn.vmap(
        cls,
        variable_axes={'params': 0, 'intermediates': 0},
        split_rngs={'params': True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class MLP(nn.Module):
    """Multi-layer perceptron with optional layer normalisation.

    When `y` is provided, it is concatenated with `x` before the first layer,
    which is how both Value and DriftingActor combine their two inputs
    (e.g. obs+action, or obs+noise).

    Args:
        hidden_dims:    Sequence of hidden layer sizes (including the output).
        activations:    Activation function applied after each non-final layer.
        activate_final: If True, apply activation after the last layer too.
        kernel_init:    Weight initialiser.
        layer_norm:     If True, apply LayerNorm after each activation.
    """
    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x, y=None):
        if y is not None:
            x = jnp.concatenate([x, y], axis=-1)
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
        return x


class Value(nn.Module):
    """Critic network: Q(observations, actions) → scalar per ensemble member.

    Implements a vectorised ensemble so that all Q-values are computed in a
    single forward pass.  Output shape is (num_ensembles, *batch_shape).

    Pessimistic backup uses:
        Q_target = mean(Q_ensemble) − ρ · std(Q_ensemble)

    which discourages the policy from exploiting overestimated values in
    out-of-distribution state-action pairs.

    Args:
        hidden_dims:   MLP hidden sizes (output size 1 is appended internally).
        layer_norm:    Layer norm (recommended True for large ensembles).
        num_ensembles: Number of Q-network copies (default 10).
    """
    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2

    def setup(self):
        mlp_cls = MLP if self.num_ensembles <= 1 else ensemblize(MLP, self.num_ensembles)
        self.value_net = mlp_cls(
            (*self.hidden_dims, 1),
            activate_final=False,
            layer_norm=self.layer_norm,
        )

    def __call__(self, observations, actions=None):
        inputs = jnp.concatenate(
            [observations] + ([actions] if actions is not None else []),
            axis=-1,
        )
        return self.value_net(inputs).squeeze(-1)  # (...) or (num_qs, ...)


class BCPolicy(nn.Module):
    """Deterministic behaviour cloning policy: obs → tanh(action).

    Takes only the observation as input (no noise).  Trained in Phase 1
    via MSE loss on dataset actions, then frozen for the rest of training.

    The frozen output is used as the attraction target in the drifting field:
        V_attract = bc_actor(obs) − x_gen

    Using a learned, smoothed BC policy rather than raw dataset actions
    gives a more generalisable attraction target — one that interpolates
    sensibly to observations not seen in the dataset.

    Args:
        action_dim:  Dimensionality of the action space.
        hidden_dims: MLP hidden layer sizes.
        layer_norm:  Layer norm (False by default for actor networks).
    """
    action_dim: int
    hidden_dims: Sequence[int] = (512, 512, 512, 512)
    layer_norm: bool = False

    def setup(self):
        self.mlp = MLP(
            (*self.hidden_dims, self.action_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )

    def __call__(self, observations):
        return jnp.tanh(self.mlp(observations))


class DriftingActor(nn.Module):
    """One-step stochastic actor: (obs, noise) → tanh(action).

    Maps a Gaussian noise vector to an action conditioned on the observation,
    in a single forward pass.  Unlike flow-matching policies, there is no
    iterative ODE solve at inference time — one call = one action.

    During training, K noise samples are drawn per observation to populate
    the repulsion set.  At inference, `best_of_n` noise samples are drawn
    and the one with the highest Q-value is returned.

    The concatenation of [obs, noise] as input means the network can map
    different noise seeds to different modes of the action distribution,
    providing the diversity needed for the repulsion term.

    Args:
        action_dim:  Dimensionality of the action space.
        hidden_dims: MLP hidden layer sizes.
        layer_norm:  Layer norm (False by default).
    """
    action_dim: int
    hidden_dims: Sequence[int] = (512, 512, 512, 512)
    layer_norm: bool = False

    def setup(self):
        self.mlp = MLP(
            (*self.hidden_dims, self.action_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )

    def __call__(self, observations, noise):
        x = jnp.concatenate([observations, noise], axis=-1)
        return jnp.tanh(self.mlp(x))
