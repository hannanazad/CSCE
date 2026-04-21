"""Dataset and replay buffer utilities for Q-Drifting.

Dataset
-------
A frozen dict of numpy arrays (observations, actions, rewards, masks,
next_observations, terminals) loaded from an OGBench/D4RL offline dataset.
Arrays are write-protected after construction to prevent accidental mutation.

The key method is `sample_sequence`, which returns batches with an extra
"horizon" dimension for n-step return computation:

  observations  (B, obs_dim)       — start of the sequence
  actions       (B, horizon, A)    — actions taken at each step
  rewards       (B, horizon)       — cumulative discounted reward up to step i
  masks         (B, horizon)       — 0 if the episode ended before step i
  terminals     (B, horizon)       — 1 if the episode ended at or before step i
  valid         (B, horizon)       — 1 if step i is within the episode boundary
  next_observations (B, horizon, obs_dim) — observation after each step

With horizon_length=1 (default), this collapses to a standard single-step
batch — the extra dimension is just size 1.

ReplayBuffer
------------
Extends Dataset to support online transition insertion (circular buffer).
Used only if online training is re-enabled; currently unused in offline-only mode.
"""
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict


def get_size(data) -> int:
    """Return the number of transitions in the dataset (length of first axis)."""
    sizes = jax.tree_util.tree_map(lambda arr: len(arr), data)
    return max(jax.tree_util.tree_leaves(sizes))


class Dataset(FrozenDict):
    """Immutable offline RL dataset backed by a frozen dict of numpy arrays.

    Constructed via `Dataset.create(**fields)` where each field is a numpy
    array whose first axis is the transition index.

    Key attributes
    --------------
    size          : Number of transitions.
    terminal_locs : Indices where `terminals > 0` (end of episode).
    initial_locs  : Indices where episodes begin (i.e. terminal_locs shifted by 1).
    """

    @classmethod
    def create(cls, freeze: bool = True, **fields):
        """Build a Dataset from keyword-argument arrays.

        Args:
            freeze: If True, mark all arrays as read-only to catch mutations early.
            **fields: Must include at minimum 'observations'.  Typical keys:
                      observations, actions, rewards, masks, next_observations, terminals.
        """
        assert 'observations' in fields, "Dataset must contain 'observations'."
        if freeze:
            jax.tree_util.tree_map(lambda arr: arr.setflags(write=False), fields)
        return cls(fields)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.size = get_size(self._dict)
        self.terminal_locs = np.nonzero(self['terminals'] > 0)[0]
        self.initial_locs = np.concatenate([[0], self.terminal_locs[:-1] + 1])

    def get_random_idxs(self, num_idxs: int) -> np.ndarray:
        return np.random.randint(self.size, size=num_idxs)

    def sample(self, batch_size: int, idxs=None) -> dict:
        """Sample a batch of individual transitions (no sequence structure)."""
        if idxs is None:
            idxs = self.get_random_idxs(batch_size)
        return self.get_subset(idxs)

    def sample_sequence(
        self,
        batch_size: int,
        sequence_length: int,
        discount: float,
    ) -> dict:
        """Sample a batch of contiguous transition sequences for n-step returns.

        For each starting index, `sequence_length` consecutive transitions are
        gathered.  The cumulative discounted reward and validity masks are
        computed so the agent can use n-step returns directly.

        Terminal boundaries are respected: once an episode ends, subsequent
        steps in the window are marked as invalid and their rewards do not
        accumulate further.

        Args:
            batch_size:      Number of sequences to sample.
            sequence_length: Length of each sequence (horizon H).
            discount:        Discount factor γ for reward accumulation.

        Returns:
            Dict with keys: observations, actions, masks, rewards, terminals,
            valid, next_observations.  All except observations have a trailing
            horizon dimension of size `sequence_length`.
        """
        # Sample start indices, leaving room for the full sequence
        idxs = np.random.randint(self.size - sequence_length + 1, size=batch_size)
        data = {k: v[idxs] for k, v in self.items()}

        B = batch_size
        obs_dim = data['observations'].shape[-1]
        act_dim = data['actions'].shape[-1]
        H = sequence_length

        rewards       = np.zeros((B, H), dtype=float)
        masks         = np.ones((B, H),  dtype=float)
        valid         = np.ones((B, H),  dtype=float)
        terminals     = np.zeros((B, H), dtype=float)
        observations  = np.zeros((B, H, obs_dim), dtype=float)
        next_obs      = np.zeros((B, H, obs_dim), dtype=float)
        actions       = np.zeros((B, H, act_dim), dtype=float)

        for i in range(H):
            cur = idxs + i
            if i == 0:
                rewards[..., 0]   = self['rewards'][cur]
                masks[..., 0]     = self['masks'][cur]
                terminals[..., 0] = self['terminals'][cur]
            else:
                # A step is valid only if the previous step was not terminal
                valid[..., i] = 1.0 - terminals[..., i - 1]
                rewards[..., i] = (
                    rewards[..., i - 1]
                    + self['rewards'][cur] * (discount ** i) * valid[..., i]
                )
                masks[..., i] = (
                    np.minimum(masks[..., i - 1], self['masks'][cur]) * valid[..., i]
                    + masks[..., i - 1] * (1.0 - valid[..., i])
                )
                terminals[..., i] = np.maximum(terminals[..., i - 1], self['terminals'][cur])

            actions[..., i, :]    = self['actions'][cur]
            observations[..., i, :] = self['observations'][cur]
            # Carry forward the last valid next_observation past terminal boundaries
            next_obs[..., i, :] = (
                self['next_observations'][cur] * valid[..., i:i+1]
                + (next_obs[..., i-1, :] if i > 0 else 0.0) * (1.0 - valid[..., i:i+1])
            )

        return dict(
            observations=data['observations'].copy(),
            actions=actions,
            masks=masks,
            rewards=rewards,
            terminals=terminals,
            valid=valid,
            next_observations=next_obs,
        )

    def get_subset(self, idxs: np.ndarray) -> dict:
        """Return a dict of arrays indexed by `idxs` (no Dataset wrapping)."""
        return jax.tree_util.tree_map(lambda arr: arr[idxs], self._dict)


class ReplayBuffer(Dataset):
    """Circular replay buffer that extends Dataset with online transition insertion.

    Supports two construction modes:
      - `create(transition, size)` — empty buffer initialised from an example transition.
      - `create_from_initial_dataset(init_dataset, size)` — pre-filled with offline data
        so that early online training can sample from both offline and online data.

    The buffer wraps around when full (pointer returns to 0).
    """

    @classmethod
    def create(cls, transition: dict, size: int):
        """Create an empty buffer sized for `size` transitions.

        Args:
            transition: Example transition dict (used only for shapes/dtypes).
            size:       Maximum number of transitions the buffer can hold.
        """
        def make_zeros(example):
            example = np.array(example)
            return np.zeros((size, *example.shape), dtype=example.dtype)

        return cls(jax.tree_util.tree_map(make_zeros, transition))

    @classmethod
    def create_from_initial_dataset(cls, init_dataset: dict, size: int):
        """Create a buffer pre-loaded with an offline dataset.

        The buffer is allocated with `size` slots; the first `len(init_dataset)`
        slots are filled with the offline data.  Online transitions appended later
        wrap around, eventually overwriting old offline data.

        Args:
            init_dataset: Dict of arrays to pre-fill (e.g. dict(train_dataset)).
            size:         Total buffer capacity (must be >= len of init_dataset).
        """
        def make_buffer(init_arr):
            buf = np.zeros((size, *init_arr.shape[1:]), dtype=init_arr.dtype)
            buf[:len(init_arr)] = init_arr
            return buf

        buf_dict = jax.tree_util.tree_map(make_buffer, init_dataset)
        ds = cls(buf_dict)
        ds.size = ds.pointer = get_size(init_dataset)
        return ds

    def __init__(self, *args, pointer: int = 0, size: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_size = get_size(self._dict)
        self.size = size
        self.pointer = pointer

    def add_transition(self, transition: dict):
        """Insert one transition into the buffer at the current pointer position.

        Advances the pointer and updates `size` (capped at `max_size`).
        """
        def set_idx(buf, elem):
            buf[self.pointer] = elem

        jax.tree_util.tree_map(set_idx, self._dict, transition)
        self.pointer = (self.pointer + 1) % self.max_size
        self.size = max(self.pointer, self.size)

    def clear(self):
        """Reset the buffer to empty (does not zero the underlying arrays)."""
        self.size = self.pointer = 0
