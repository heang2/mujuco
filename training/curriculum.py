"""
Curriculum Learning — progressively increase task difficulty during training.

Curriculum strategies:
  - LinearCurriculum     — linearly interpolate difficulty over steps
  - SuccessCurriculum    — advance when success rate exceeds a threshold
  - RewardCurriculum     — advance when mean reward exceeds a threshold
  - StagedCurriculum     — discrete difficulty stages

Each curriculum modifies environment parameters (e.g. goal distance, noise level,
initial state perturbation) to make training easier early on.

Usage:
    from training.curriculum import LinearCurriculum, CurriculumWrapper

    schedule = LinearCurriculum(
        param_name="goal_dist_max",
        start_value=0.05,   # start easy: small goal radius
        end_value=0.22,     # end hard: full range
        total_steps=500_000,
    )
    env = CurriculumWrapper(make_env("Reacher"), [schedule])
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Curriculum schedules
# ──────────────────────────────────────────────────────────────────────────────

class BaseCurriculum(ABC):
    """Base class for curriculum difficulty schedules."""

    @abstractmethod
    def get_value(self, step: int, **stats) -> float:
        """Return the current difficulty value given training step and stats."""

    @property
    @abstractmethod
    def param_name(self) -> str:
        """The environment parameter this schedule controls."""


class LinearCurriculum(BaseCurriculum):
    """
    Linearly interpolate a parameter from start_value to end_value
    over total_steps training steps.
    """

    def __init__(
        self,
        param_name: str,
        start_value: float,
        end_value: float,
        total_steps: int,
        warmup_steps: int = 0,
    ):
        self._param   = param_name
        self.start    = start_value
        self.end      = end_value
        self.total    = total_steps
        self.warmup   = warmup_steps

    @property
    def param_name(self) -> str:
        return self._param

    def get_value(self, step: int, **kwargs) -> float:
        if step < self.warmup:
            return self.start
        frac = min(1.0, (step - self.warmup) / max(self.total - self.warmup, 1))
        return self.start + frac * (self.end - self.start)


class CosineCurriculum(BaseCurriculum):
    """Cosine-annealed difficulty schedule (slow → fast → slow transitions)."""

    def __init__(
        self,
        param_name: str,
        start_value: float,
        end_value: float,
        total_steps: int,
    ):
        self._param = param_name
        self.start  = start_value
        self.end    = end_value
        self.total  = total_steps

    @property
    def param_name(self) -> str:
        return self._param

    def get_value(self, step: int, **kwargs) -> float:
        frac = min(1.0, step / self.total)
        cosine = 0.5 * (1 - np.cos(np.pi * frac))
        return self.start + cosine * (self.end - self.start)


class SuccessCurriculum(BaseCurriculum):
    """
    Advance to a harder difficulty level when recent success rate
    exceeds `advance_threshold`.

    Args:
        levels:            List of (value, name) tuples, easiest first
        advance_threshold: Fraction of successes needed to advance
        window_size:       Episodes to average over
        min_episodes:      Minimum episodes before first advancement
    """

    def __init__(
        self,
        param_name: str,
        levels: List[float],
        advance_threshold: float = 0.8,
        window_size: int = 100,
        min_episodes: int = 200,
    ):
        self._param     = param_name
        self.levels     = levels
        self.threshold  = advance_threshold
        self.window     = window_size
        self.min_ep     = min_episodes
        self._level_idx = 0
        self._successes: List[bool] = []

    @property
    def param_name(self) -> str:
        return self._param

    def update(self, success: bool) -> bool:
        """Update with latest episode result. Returns True if level advanced."""
        self._successes.append(success)
        if len(self._successes) < self.min_ep:
            return False

        recent = self._successes[-self.window:]
        if np.mean(recent) >= self.threshold and self._level_idx < len(self.levels) - 1:
            self._level_idx += 1
            return True
        return False

    def get_value(self, step: int, **kwargs) -> float:
        return self.levels[self._level_idx]

    @property
    def current_level(self) -> int:
        return self._level_idx

    @property
    def n_levels(self) -> int:
        return len(self.levels)


class StagedCurriculum(BaseCurriculum):
    """
    Discrete stages triggered at fixed training steps.

    stages = [(0, 0.05), (100_000, 0.10), (300_000, 0.20), (600_000, 0.22)]
    means: at step 0 → 0.05, at step 100k → 0.10, etc.
    """

    def __init__(self, param_name: str, stages: List[Tuple[int, float]]):
        assert len(stages) > 0 and stages[0][0] == 0, \
            "First stage must start at step 0"
        self._param  = param_name
        self._stages = sorted(stages, key=lambda x: x[0])

    @property
    def param_name(self) -> str:
        return self._param

    def get_value(self, step: int, **kwargs) -> float:
        value = self._stages[0][1]
        for trigger_step, v in self._stages:
            if step >= trigger_step:
                value = v
            else:
                break
        return value


# ──────────────────────────────────────────────────────────────────────────────
# CurriculumWrapper
# ──────────────────────────────────────────────────────────────────────────────

class CurriculumWrapper(gym.Wrapper):
    """
    Wrapper that applies curriculum schedules to environment parameters.

    The wrapper maintains a global step counter and after each episode reset,
    updates the environment parameters by setting attributes on the unwrapped env.

    Example:
        env = CurriculumWrapper(
            make_env("Reacher"),
            schedules=[
                LinearCurriculum("SUCCESS_DIST", start_value=0.08, end_value=0.025,
                                 total_steps=500_000),
            ]
        )
        # After 250k steps, env.SUCCESS_DIST ≈ 0.05 (halfway)
    """

    def __init__(self, env: gym.Env, schedules: List[BaseCurriculum]):
        super().__init__(env)
        self.schedules     = schedules
        self._global_step  = 0
        self._episode_step = 0
        self._curriculum_log: List[Dict] = []

    def reset(self, **kwargs):
        # Apply current curriculum values before reset
        self._apply_schedules()
        obs, info = self.env.reset(**kwargs)
        self._episode_step = 0

        # Attach curriculum state to info
        info["curriculum"] = self._get_curriculum_state()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._global_step  += 1
        self._episode_step += 1
        return obs, reward, terminated, truncated, info

    def _apply_schedules(self) -> None:
        base = self.env.unwrapped
        for sched in self.schedules:
            value = sched.get_value(self._global_step)
            if hasattr(base, sched.param_name):
                setattr(base, sched.param_name, value)

    def _get_curriculum_state(self) -> Dict[str, float]:
        return {
            sched.param_name: sched.get_value(self._global_step)
            for sched in self.schedules
        }

    @property
    def curriculum_values(self) -> Dict[str, float]:
        return self._get_curriculum_state()


# ──────────────────────────────────────────────────────────────────────────────
# CurriculumTrainer helper
# ──────────────────────────────────────────────────────────────────────────────

class CurriculumTrainer:
    """
    Coordinates curriculum advancement and logging during training.

    Tracks success rates per difficulty level and logs advancement events.
    """

    def __init__(self, success_curriculum: Optional[SuccessCurriculum] = None):
        self.success_curriculum  = success_curriculum
        self.advancement_history: List[Dict] = []

    def on_episode_end(
        self, step: int, success: bool, reward: float, length: int
    ) -> Optional[str]:
        """
        Call at end of each episode.
        Returns a log message if the difficulty level advanced, else None.
        """
        if self.success_curriculum is None:
            return None

        advanced = self.success_curriculum.update(success)
        if advanced:
            event = {
                "step":        step,
                "new_level":   self.success_curriculum.current_level,
                "n_levels":    self.success_curriculum.n_levels,
                "new_value":   self.success_curriculum.get_value(step),
            }
            self.advancement_history.append(event)
            return (
                f"[Curriculum] Advanced to level "
                f"{event['new_level']}/{event['n_levels'] - 1} "
                f"(value={event['new_value']:.4f}) at step {step:,}"
            )
        return None
