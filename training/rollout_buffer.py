"""
Rollout buffer with Generalized Advantage Estimation (GAE-λ).

Stores a fixed number of environment steps, then computes
advantages and discounted returns for PPO updates.
"""

from typing import Tuple
import numpy as np


class RolloutBuffer:
    """
    Fixed-size buffer for on-policy data collection.

    Call `add()` for each environment step, then `compute_returns()`
    after the rollout, then `get()` to retrieve tensors for training.
    """

    def __init__(
        self,
        n_steps: int,
        obs_shape: Tuple,
        act_shape: Tuple,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.n_steps    = n_steps
        self.obs_shape  = obs_shape
        self.act_shape  = act_shape
        self.gamma      = gamma
        self.gae_lambda = gae_lambda

        self._obs      = np.zeros((n_steps, *obs_shape), dtype=np.float32)
        self._actions  = np.zeros((n_steps, *act_shape), dtype=np.float32)
        self._rewards  = np.zeros((n_steps,),             dtype=np.float32)
        self._dones    = np.zeros((n_steps,),             dtype=np.float32)
        self._values   = np.zeros((n_steps,),             dtype=np.float32)
        self._log_probs = np.zeros((n_steps,),            dtype=np.float32)

        self._returns   = np.zeros((n_steps,),            dtype=np.float32)
        self._advantages = np.zeros((n_steps,),           dtype=np.float32)

        self._ptr = 0
        self._full = False

    def reset(self) -> None:
        self._ptr  = 0
        self._full = False

    def add(
        self,
        obs:      np.ndarray,
        action:   np.ndarray,
        reward:   float,
        done:     bool,
        value:    float,
        log_prob: float,
    ) -> None:
        assert self._ptr < self.n_steps, "Buffer is full — call reset() or compute_returns() first."
        i = self._ptr
        self._obs[i]       = obs
        self._actions[i]   = action
        self._rewards[i]   = reward
        self._dones[i]     = float(done)
        self._values[i]    = value
        self._log_probs[i] = log_prob
        self._ptr += 1
        if self._ptr == self.n_steps:
            self._full = True

    def compute_returns(self, last_value: float) -> None:
        """
        Compute GAE-λ advantages and discounted returns.

        Args:
            last_value: V(s_{T+1}), bootstrapped value of the state
                        after the last collected step.
        """
        n   = self._ptr
        gae = 0.0
        for t in reversed(range(n)):
            next_val  = last_value if t == n - 1 else self._values[t + 1]
            next_done = 0.0        if t == n - 1 else self._dones[t + 1]

            delta = (
                self._rewards[t]
                + self.gamma * next_val * (1.0 - self._dones[t])
                - self._values[t]
            )
            gae = delta + self.gamma * self.gae_lambda * (1.0 - self._dones[t]) * gae
            self._advantages[t] = gae
            self._returns[t]    = gae + self._values[t]

    def get(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Return buffered data as numpy arrays.

        Returns:
            obs, actions, log_probs, returns, advantages
        """
        n = self._ptr
        return (
            self._obs[:n],
            self._actions[:n],
            self._log_probs[:n],
            self._returns[:n],
            self._advantages[:n],
        )

    @property
    def size(self) -> int:
        return self._ptr
