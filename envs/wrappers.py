"""
Environment wrappers — modular transformations applied on top of base environments.

Usage:
    from envs.wrappers import wrap
    env = wrap(make_env("Hopper"),
               time_limit=500,
               obs_noise=0.01,
               action_repeat=2,
               record_stats=True)

Available wrappers:
    TimeLimitWrapper     — truncate episodes at a max step count
    ClipActionWrapper    — hard-clip actions to action_space bounds
    ObsNormWrapper       — running-mean normalise observations (Welford)
    RewardScaleWrapper   — multiply rewards by a scalar
    ObsNoiseWrapper      — add Gaussian noise to observations (domain randomisation)
    ActionRepeatWrapper  — repeat each action k times, sum rewards
    RecordEpisodeStats   — track episode reward/length in info dict
    GoalWrapper          — augment obs with goal vector (for HER-compatible envs)
    FrameStackWrapper    — stack last k observations (for recurrent policies)
    RandomisePhysics     — randomise body masses/friction at each reset (DR)
"""

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


# ──────────────────────────────────────────────────────────────────────────────
# TimeLimitWrapper
# ──────────────────────────────────────────────────────────────────────────────

class TimeLimitWrapper(gym.Wrapper):
    """Override the episode time limit."""

    def __init__(self, env: gym.Env, max_episode_steps: int):
        super().__init__(env)
        self._max_steps   = max_episode_steps
        self._elapsed     = 0

    def reset(self, **kwargs):
        self._elapsed = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed += 1
        if self._elapsed >= self._max_steps:
            truncated = True
        return obs, reward, terminated, truncated, info


# ──────────────────────────────────────────────────────────────────────────────
# ClipActionWrapper
# ──────────────────────────────────────────────────────────────────────────────

class ClipActionWrapper(gym.Wrapper):
    """Silently clip actions to the declared action space."""

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return self.env.step(action)


# ──────────────────────────────────────────────────────────────────────────────
# ObsNormWrapper
# ──────────────────────────────────────────────────────────────────────────────

class ObsNormWrapper(gym.ObservationWrapper):
    """
    Normalise observations using a running Welford estimate.
    Clip normalised values to ±clip_range for stability.
    """

    def __init__(self, env: gym.Env, clip_range: float = 10.0, epsilon: float = 1e-8):
        super().__init__(env)
        shape         = env.observation_space.shape
        self._mean    = np.zeros(shape, dtype=np.float64)
        self._var     = np.ones(shape,  dtype=np.float64)
        self._count   = 1e-4
        self._clip    = clip_range
        self._eps     = epsilon

    def observation(self, obs: np.ndarray) -> np.ndarray:
        self._update(obs)
        normed = (obs - self._mean) / np.sqrt(self._var + self._eps)
        return np.clip(normed, -self._clip, self._clip).astype(np.float32)

    def _update(self, x: np.ndarray) -> None:
        self._count += 1
        delta        = x - self._mean
        self._mean  += delta / self._count
        delta2       = x - self._mean
        self._var   += (delta * delta2 - self._var) / self._count


# ──────────────────────────────────────────────────────────────────────────────
# RewardScaleWrapper
# ──────────────────────────────────────────────────────────────────────────────

class RewardScaleWrapper(gym.RewardWrapper):
    """Multiply all rewards by a constant factor."""

    def __init__(self, env: gym.Env, scale: float):
        super().__init__(env)
        self._scale = scale

    def reward(self, reward: float) -> float:
        return reward * self._scale


# ──────────────────────────────────────────────────────────────────────────────
# ObsNoiseWrapper  (Domain Randomisation)
# ──────────────────────────────────────────────────────────────────────────────

class ObsNoiseWrapper(gym.ObservationWrapper):
    """
    Add zero-mean Gaussian noise to every observation.
    Useful for domain randomisation and robustness testing.
    """

    def __init__(self, env: gym.Env, noise_std: float, seed: int = 0):
        super().__init__(env)
        self._std = noise_std
        self._rng = np.random.default_rng(seed)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        noise = self._rng.normal(0.0, self._std, size=obs.shape).astype(obs.dtype)
        return obs + noise


# ──────────────────────────────────────────────────────────────────────────────
# ActionRepeatWrapper
# ──────────────────────────────────────────────────────────────────────────────

class ActionRepeatWrapper(gym.Wrapper):
    """
    Repeat each action k times and accumulate rewards.
    Reduces effective control frequency, useful for some locomotion tasks.
    """

    def __init__(self, env: gym.Env, k: int):
        super().__init__(env)
        assert k >= 1
        self._k = k

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._k):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


# ──────────────────────────────────────────────────────────────────────────────
# RecordEpisodeStats
# ──────────────────────────────────────────────────────────────────────────────

class RecordEpisodeStats(gym.Wrapper):
    """
    Accumulate per-episode reward and length and inject them into info.
    After each episode end: info["episode"] = {"r": total_reward, "l": length}
    """

    def __init__(self, env: gym.Env, deque_size: int = 100):
        super().__init__(env)
        self._ep_reward   = 0.0
        self._ep_length   = 0
        self.episode_returns: deque = deque(maxlen=deque_size)
        self.episode_lengths: deque = deque(maxlen=deque_size)

    def reset(self, **kwargs):
        self._ep_reward = 0.0
        self._ep_length = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._ep_reward += reward
        self._ep_length += 1

        if terminated or truncated:
            ep_info = {"r": self._ep_reward, "l": self._ep_length}
            info["episode"] = ep_info
            self.episode_returns.append(self._ep_reward)
            self.episode_lengths.append(self._ep_length)

        return obs, reward, terminated, truncated, info

    @property
    def mean_episode_return(self) -> float:
        if not self.episode_returns:
            return 0.0
        return float(np.mean(self.episode_returns))


# ──────────────────────────────────────────────────────────────────────────────
# FrameStackWrapper
# ──────────────────────────────────────────────────────────────────────────────

class FrameStackWrapper(gym.ObservationWrapper):
    """
    Stack the last k observations into a single flat vector.
    Useful for partially observable environments or adding velocity context.
    """

    def __init__(self, env: gym.Env, k: int):
        super().__init__(env)
        self._k      = k
        self._frames: deque = deque(maxlen=k)
        single_shape = env.observation_space.shape
        stacked_shape = (single_shape[0] * k,)
        self.observation_space = spaces.Box(
            low=np.tile(env.observation_space.low,  k),
            high=np.tile(env.observation_space.high, k),
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self._k):
            self._frames.append(obs)
        return self.observation(obs), info

    def observation(self, obs: np.ndarray) -> np.ndarray:
        self._frames.append(obs)
        return np.concatenate(list(self._frames), axis=0).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# RandomisePhysicsWrapper  (Domain Randomisation)
# ──────────────────────────────────────────────────────────────────────────────

class RandomisePhysicsWrapper(gym.Wrapper):
    """
    At each reset, randomly perturb body masses and geom friction.
    Trains policies that generalise across physical variations.

    Args:
        mass_range:     (low, high) multiplicative scale for body masses
        friction_range: (low, high) multiplicative scale for geom friction
    """

    def __init__(
        self,
        env: gym.Env,
        mass_range:     Tuple[float, float] = (0.8, 1.2),
        friction_range: Tuple[float, float] = (0.8, 1.2),
        seed: int = 0,
    ):
        super().__init__(env)
        self._mass_range     = mass_range
        self._friction_range = friction_range
        self._rng            = np.random.default_rng(seed)

        # Store original values so we can scale from them
        self._orig_masses    = env.unwrapped.model.body_mass.copy()
        self._orig_friction  = env.unwrapped.model.geom_friction.copy()

    def reset(self, **kwargs):
        model = self.env.unwrapped.model
        lo_m, hi_m = self._mass_range
        lo_f, hi_f = self._friction_range

        mass_scale = self._rng.uniform(lo_m, hi_m, size=self._orig_masses.shape)
        fric_scale = self._rng.uniform(lo_f, hi_f, size=self._orig_friction.shape)

        model.body_mass[:]    = self._orig_masses    * mass_scale
        model.geom_friction[:] = self._orig_friction * fric_scale

        return self.env.reset(**kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Convenience factory
# ──────────────────────────────────────────────────────────────────────────────

def wrap(
    env: gym.Env,
    time_limit:       Optional[int]   = None,
    clip_actions:     bool            = True,
    obs_noise:        float           = 0.0,
    action_repeat:    int             = 1,
    reward_scale:     float           = 1.0,
    record_stats:     bool            = False,
    frame_stack:      int             = 1,
    obs_norm:         bool            = False,
    randomise_physics: bool           = False,
    physics_mass_range:    Tuple[float, float] = (0.8, 1.2),
    physics_friction_range: Tuple[float, float] = (0.8, 1.2),
    seed:             int             = 0,
) -> gym.Env:
    """
    Apply a stack of wrappers to an environment.

    Wrappers are applied in this order (innermost → outermost):
        ClipAction → TimeLimitWrapper → ActionRepeat → ObsNoise →
        FrameStack → ObsNorm → RewardScale → RecordStats
    """
    if clip_actions:
        env = ClipActionWrapper(env)
    if time_limit is not None:
        env = TimeLimitWrapper(env, time_limit)
    if action_repeat > 1:
        env = ActionRepeatWrapper(env, action_repeat)
    if obs_noise > 0.0:
        env = ObsNoiseWrapper(env, obs_noise, seed=seed)
    if frame_stack > 1:
        env = FrameStackWrapper(env, frame_stack)
    if obs_norm:
        env = ObsNormWrapper(env)
    if reward_scale != 1.0:
        env = RewardScaleWrapper(env, reward_scale)
    if randomise_physics:
        env = RandomisePhysicsWrapper(
            env,
            mass_range=physics_mass_range,
            friction_range=physics_friction_range,
            seed=seed,
        )
    if record_stats:
        env = RecordEpisodeStats(env)
    return env
