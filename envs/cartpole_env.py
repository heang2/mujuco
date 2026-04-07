"""
CartPole-v2 (MuJoCo) — balance an inverted pendulum on a sliding cart.

Observation (4,):
    [cart_pos, cart_vel, pole_angle, pole_angular_vel]

Action (1,):
    [horizontal force applied to cart]  ∈ [-1, 1]

Reward:
    +1 every step the pole remains within ±12° and cart within ±2.4 units.
    Shaping bonus for small angle and small cart displacement.

Episode ends when pole falls past ±24° OR cart goes out of ±2.4 range.
"""

from typing import Any, Dict, Optional, Tuple

import mujoco
import numpy as np
from gymnasium import spaces

from envs.base_env import BaseMuJoCoEnv


class CartPoleEnv(BaseMuJoCoEnv):

    ANGLE_LIMIT = np.deg2rad(24)   # terminate if |pole angle| > 24°
    CART_LIMIT  = 2.4              # terminate if |cart pos| > 2.4 m
    MAX_STEPS   = 1000

    def __init__(self, render_mode: Optional[str] = None, **kwargs):
        super().__init__(
            model_filename="cartpole.xml",
            frame_skip=2,
            render_mode=render_mode,
            **kwargs,
        )
        # obs: cart_pos, cart_vel, pole_angle, pole_angular_vel
        high = np.array([self.CART_LIMIT * 2, np.inf, np.pi, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        slider_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "slider")
        hinge_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "hinge")

        cart_pos  = self.data.qpos[slider_id]
        pole_ang  = self.data.qpos[hinge_id]
        cart_vel  = self.data.qvel[slider_id]
        pole_vel  = self.data.qvel[hinge_id]

        return np.array([cart_pos, cart_vel, pole_ang, pole_vel], dtype=np.float32)

    def _compute_reward(self, action: np.ndarray) -> Tuple[float, bool, bool]:
        obs = self._get_obs()
        cart_pos, _, pole_ang, _ = obs

        terminated = bool(
            abs(pole_ang)  > self.ANGLE_LIMIT or
            abs(cart_pos)  > self.CART_LIMIT
        )
        truncated = self._step_count >= self.MAX_STEPS

        if terminated:
            reward = 0.0
        else:
            # Alive bonus + shaping terms
            angle_shaping = 1.0 - (abs(pole_ang) / self.ANGLE_LIMIT) ** 2
            pos_shaping   = 1.0 - (abs(cart_pos) / self.CART_LIMIT)  ** 2
            reward = 1.0 + 0.5 * angle_shaping + 0.2 * pos_shaping

        return reward, terminated, truncated

    def _get_info(self) -> Dict[str, Any]:
        obs = self._get_obs()
        return {
            "cart_pos":  float(obs[0]),
            "cart_vel":  float(obs[1]),
            "pole_angle_deg": float(np.rad2deg(obs[2])),
            "pole_vel":  float(obs[3]),
            "step":      self._step_count,
        }

    def _reset_model(self) -> np.ndarray:
        # Small random perturbation around upright
        self._noisy_reset(scale=0.05)
        return self._get_obs()
