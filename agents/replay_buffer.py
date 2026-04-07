"""
Experience Replay Buffer with optional Prioritized Experience Replay (PER).

Two variants:
  - UniformReplayBuffer  — standard random sampling (used by SAC / TD3)
  - PrioritizedReplayBuffer — samples proportional to TD-error (PER)
    Reference: Schaul et al., "Prioritized Experience Replay" (2016)
               https://arxiv.org/abs/1511.05952

Both implement the same `.add()` / `.sample()` / `.__len__()` interface.
"""

from typing import Dict, NamedTuple, Optional, Tuple

import numpy as np


class Batch(NamedTuple):
    """A mini-batch of transitions sampled from the replay buffer."""
    obs:        np.ndarray   # (B, obs_dim)
    actions:    np.ndarray   # (B, act_dim)
    rewards:    np.ndarray   # (B,)
    next_obs:   np.ndarray   # (B, obs_dim)
    dones:      np.ndarray   # (B,)   float32 {0, 1}
    weights:    np.ndarray   # (B,)   importance weights (all 1s for uniform)
    indices:    np.ndarray   # (B,)   indices into buffer (for PER updates)


# ──────────────────────────────────────────────────────────────────────────────
# Uniform Replay Buffer
# ──────────────────────────────────────────────────────────────────────────────

class UniformReplayBuffer:
    """
    Ring-buffer that stores (s, a, r, s', done) transitions.

    Uses pre-allocated numpy arrays for memory efficiency.
    Thread-safety is NOT guaranteed (single-process use only).
    """

    def __init__(
        self,
        capacity:  int,
        obs_shape: Tuple,
        act_shape: Tuple,
        seed:      int = 0,
    ):
        self.capacity  = capacity
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self._rng      = np.random.default_rng(seed)

        self._obs      = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._actions  = np.zeros((capacity, *act_shape), dtype=np.float32)
        self._rewards  = np.zeros((capacity,),             dtype=np.float32)
        self._next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._dones    = np.zeros((capacity,),             dtype=np.float32)

        self._ptr  = 0
        self._size = 0

    def add(
        self,
        obs:      np.ndarray,
        action:   np.ndarray,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ) -> None:
        i = self._ptr
        self._obs[i]      = obs
        self._actions[i]  = action
        self._rewards[i]  = reward
        self._next_obs[i] = next_obs
        self._dones[i]    = float(done)

        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        assert self._size >= batch_size, (
            f"Buffer has {self._size} samples but {batch_size} requested"
        )
        idx = self._rng.integers(0, self._size, size=batch_size)
        return Batch(
            obs      = self._obs[idx],
            actions  = self._actions[idx],
            rewards  = self._rewards[idx],
            next_obs = self._next_obs[idx],
            dones    = self._dones[idx],
            weights  = np.ones(batch_size, dtype=np.float32),
            indices  = idx,
        )

    def __len__(self) -> int:
        return self._size

    @property
    def is_ready(self) -> bool:
        return self._size >= 256   # minimum before we start training


# ──────────────────────────────────────────────────────────────────────────────
# Sum-Tree for Prioritized Experience Replay
# ──────────────────────────────────────────────────────────────────────────────

class _SumTree:
    """
    Binary tree where each leaf stores a priority value.
    Parent nodes store the sum of their children.
    Supports O(log N) priority updates and O(log N) stratified sampling.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._tree  = np.zeros(2 * capacity, dtype=np.float64)
        self._data_ptr = 0

    def _propagate(self, idx: int, change: float) -> None:
        """Iterative (non-recursive) priority propagation."""
        while idx > 0:
            parent = (idx - 1) // 2
            self._tree[parent] += change
            idx = parent

    def update(self, idx: int, priority: float) -> None:
        """Update the priority at leaf index idx."""
        tree_idx = idx + self.capacity - 1
        change   = priority - self._tree[tree_idx]
        self._tree[tree_idx] = priority
        self._propagate(tree_idx, change)

    def get(self, s: float) -> Tuple[int, float]:
        """
        Retrieve the leaf index whose prefix sum contains s.
        Returns (data_idx, priority).
        """
        idx = 0
        while idx < self.capacity - 1:
            left  = 2 * idx + 1
            right = left + 1
            if s <= self._tree[left]:
                idx = left
            else:
                s  -= self._tree[left]
                idx = right
        data_idx = idx - self.capacity + 1
        return data_idx, self._tree[idx]

    @property
    def total(self) -> float:
        return float(self._tree[0])

    @property
    def min_priority(self) -> float:
        leaves = self._tree[self.capacity - 1: self.capacity - 1 + self.capacity]
        nonzero = leaves[leaves > 0]
        return float(nonzero.min()) if len(nonzero) else 1.0


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.

    Transitions with higher TD-error are sampled more frequently.
    Importance sampling weights correct for the sampling bias.

    Args:
        capacity:  Maximum number of transitions stored.
        alpha:     Priority exponent (0 = uniform, 1 = full prioritization).
        beta_init: IS-weight exponent (start value, annealed to 1).
        beta_steps: Total steps over which beta is annealed to 1.0.
    """

    def __init__(
        self,
        capacity:   int,
        obs_shape:  Tuple,
        act_shape:  Tuple,
        alpha:      float = 0.6,
        beta_init:  float = 0.4,
        beta_steps: int   = 1_000_000,
        epsilon:    float = 1e-6,
        seed:       int   = 0,
    ):
        self.capacity   = capacity
        self.obs_shape  = obs_shape
        self.act_shape  = act_shape
        self.alpha      = alpha
        self.beta_init  = beta_init
        self.beta_steps = beta_steps
        self.epsilon    = epsilon
        self._rng       = np.random.default_rng(seed)
        self._step      = 0

        self._tree     = _SumTree(capacity)
        self._max_prio = 1.0

        self._obs      = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._actions  = np.zeros((capacity, *act_shape), dtype=np.float32)
        self._rewards  = np.zeros((capacity,),             dtype=np.float32)
        self._next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._dones    = np.zeros((capacity,),             dtype=np.float32)

        self._ptr  = 0
        self._size = 0

    def add(
        self,
        obs:      np.ndarray,
        action:   np.ndarray,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ) -> None:
        i = self._ptr
        self._obs[i]      = obs
        self._actions[i]  = action
        self._rewards[i]  = reward
        self._next_obs[i] = next_obs
        self._dones[i]    = float(done)

        self._tree.update(i, self._max_prio ** self.alpha)
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        assert self._size >= batch_size
        self._step += 1
        beta = min(
            1.0,
            self.beta_init + (1.0 - self.beta_init) * self._step / self.beta_steps,
        )

        indices = np.zeros(batch_size, dtype=np.int64)
        prios   = np.zeros(batch_size, dtype=np.float64)

        # Stratified sampling across priority distribution
        segment = self._tree.total / batch_size
        for i in range(batch_size):
            s = self._rng.uniform(i * segment, (i + 1) * segment)
            idx, prio = self._tree.get(s)
            indices[i] = idx
            prios[i]   = prio

        # Importance sampling weights
        probs   = (prios + self.epsilon) / (self._tree.total + self.epsilon * self._size)
        weights = (self._size * probs) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)

        return Batch(
            obs      = self._obs[indices],
            actions  = self._actions[indices],
            rewards  = self._rewards[indices],
            next_obs = self._next_obs[indices],
            dones    = self._dones[indices],
            weights  = weights,
            indices  = indices,
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities after computing new TD-errors."""
        prios = (np.abs(td_errors) + self.epsilon) ** self.alpha
        for idx, prio in zip(indices, prios):
            self._tree.update(int(idx), float(prio))
            self._max_prio = max(self._max_prio, float(prio))

    def __len__(self) -> int:
        return self._size

    @property
    def is_ready(self) -> bool:
        return self._size >= 256
