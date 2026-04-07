"""
Base MuJoCo environment providing common utilities for all robot environments.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces


MODELS_DIR = Path(__file__).parent.parent / "models"


class BaseMuJoCoEnv(gym.Env, ABC):
    """
    Abstract base class for MuJoCo environments.

    Wraps the MuJoCo Python bindings with a clean gymnasium interface.
    Subclasses must implement `step`, `reset`, `_get_obs`, and `_get_info`.
    """

    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 50}

    def __init__(
        self,
        model_filename: str,
        frame_skip: int = 4,
        render_mode: Optional[str] = None,
        width: int = 480,
        height: int = 480,
        camera_name: Optional[str] = None,
    ):
        super().__init__()

        self.model_path = MODELS_DIR / model_filename
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model XML not found: {self.model_path}")

        self.frame_skip = frame_skip
        self.render_mode = render_mode
        self.width = width
        self.height = height
        self.camera_name = camera_name

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)

        # Renderer (lazy init)
        self._renderer: Optional[mujoco.Renderer] = None

        # Spaces will be set in subclasses
        self.observation_space: spaces.Space
        self.action_space: spaces.Space

        # Build action space from actuator limits
        n_acts = self.model.nu
        ctrl_low  = self.model.actuator_ctrlrange[:, 0].copy()
        ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()
        self.action_space = spaces.Box(
            low=ctrl_low, high=ctrl_high, dtype=np.float32
        )

        self._step_count = 0

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _get_obs(self) -> np.ndarray:
        """Return current observation vector."""

    @abstractmethod
    def _get_info(self) -> Dict[str, Any]:
        """Return auxiliary info dict."""

    @abstractmethod
    def _compute_reward(self, action: np.ndarray) -> Tuple[float, bool, bool]:
        """Return (reward, terminated, truncated)."""

    @abstractmethod
    def _reset_model(self) -> np.ndarray:
        """Reset model state; return initial observation."""

    # ------------------------------------------------------------------
    # Core gym interface
    # ------------------------------------------------------------------

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.data.ctrl[:] = action

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        obs = self._get_obs()
        reward, terminated, truncated = self._compute_reward(action)
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._step_count = 0

        mujoco.mj_resetData(self.model, self.data)
        obs = self._reset_model()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, info

    def render(self):
        if self.render_mode is None:
            return None

        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, self.height, self.width)

        self._renderer.update_scene(self.data, camera=self.camera_name)

        if self.render_mode == "rgb_array":
            return self._renderer.render()
        elif self.render_mode == "depth_array":
            self._renderer.enable_depth_rendering()
            depth = self._renderer.render()
            self._renderer.disable_depth_rendering()
            return depth
        elif self.render_mode == "human":
            # For headless environments, fall back to rgb_array
            return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @property
    def dt(self) -> float:
        """Simulation timestep × frame_skip."""
        return self.model.opt.timestep * self.frame_skip

    def get_body_pos(self, body_name: str) -> np.ndarray:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self.data.xpos[body_id].copy()

    def get_body_vel(self, body_name: str) -> np.ndarray:
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body_id, vel, 0)
        return vel.copy()

    def get_site_pos(self, site_name: str) -> np.ndarray:
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        return self.data.site_xpos[site_id].copy()

    def set_state(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        assert qpos.shape == (self.model.nq,), f"Expected qpos shape {self.model.nq}"
        assert qvel.shape == (self.model.nv,), f"Expected qvel shape {self.model.nv}"
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

    def _noisy_reset(self, scale: float = 0.01) -> None:
        """Add small noise to qpos and qvel around zero."""
        self.data.qpos[:] = self.model.qpos0 + self.np_random.uniform(
            low=-scale, high=scale, size=self.model.nq
        )
        self.data.qvel[:] = self.np_random.uniform(
            low=-scale, high=scale, size=self.model.nv
        )
        mujoco.mj_forward(self.model, self.data)
