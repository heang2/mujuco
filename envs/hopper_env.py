"""
Hopper — 1-legged robot must hop forward as fast as possible.

Observation (11,):
    z-height, rooty angle,
    thigh_joint, leg_joint, foot_joint,   (positions)
    rootx_vel, rootz_vel, rooty_vel,
    thigh_vel, leg_vel, foot_vel           (velocities)

Action (3,):
    [thigh_torque, leg_torque, foot_torque]  ∈ [-1, 1]

Reward:
    forward_vel − 0.001 * ctrl_cost − 1e-3 * contact_cost
    Alive bonus +1 per step if healthy

Episode ends when torso falls below 0.7 m or tilts beyond 0.2 rad.
"""

from typing import Any, Dict, Optional, Tuple

import mujoco
import numpy as np
from gymnasium import spaces

from envs.base_env import BaseMuJoCoEnv


class HopperEnv(BaseMuJoCoEnv):

    HEALTHY_Z_MIN  = 0.7       # m, minimum torso height
    HEALTHY_ANG_MAX = 0.20     # rad, max torso tilt (rooty)
    MAX_STEPS       = 1000
    CTRL_COST_WEIGHT = 1e-3
    HEALTHY_REWARD   = 1.0
    FORWARD_REWARD_WEIGHT = 1.25

    def __init__(
        self,
        render_mode: Optional[str] = None,
        terminate_when_unhealthy: bool = True,
        **kwargs,
    ):
        super().__init__(
            model_filename="hopper.xml",
            frame_skip=4,
            render_mode=render_mode,
            **kwargs,
        )
        self.terminate_when_unhealthy = terminate_when_unhealthy

        # obs: z, rooty, 3 joint pos, rootx_vel, rootz_vel, rooty_vel, 3 joint vel
        obs_high = np.array(
            [np.inf] * 11, dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        self._rootx_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "rootx")
        self._rootz_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "rootz")
        self._rooty_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "rooty")

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos.copy()
        qvel = np.clip(self.data.qvel.copy(), -10, 10)

        # joints: rootx(0), rootz(1), rooty(2), thigh(3), leg(4), foot(5)
        # obs excludes rootx (x position), includes z height and angle
        obs = np.concatenate([
            qpos[1:],   # z, rooty, thigh, leg, foot  (5 values)
            qvel[:],    # rootx_vel, rootz_vel, rooty_vel, thigh_vel, leg_vel, foot_vel  (6 values)
        ]).astype(np.float32)
        return obs

    def _is_healthy(self) -> bool:
        # Use absolute body position (qpos for slide joint is relative displacement)
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        z   = float(self.data.xpos[torso_id][2])
        ang = float(self.data.qpos[self._rooty_id])
        return (self.HEALTHY_Z_MIN < z) and (abs(ang) < self.HEALTHY_ANG_MAX)

    def _compute_reward(self, action: np.ndarray) -> Tuple[float, bool, bool]:
        x_before = float(self.data.qpos[self._rootx_id])

        # forward velocity (already stepped)
        x_vel = float(self.data.qvel[self._rootx_id])

        forward_reward = self.FORWARD_REWARD_WEIGHT * x_vel
        ctrl_cost      = self.CTRL_COST_WEIGHT * float(np.sum(action ** 2))
        healthy_reward = self.HEALTHY_REWARD if self._is_healthy() else 0.0

        reward = forward_reward - ctrl_cost + healthy_reward

        terminated = (not self._is_healthy()) and self.terminate_when_unhealthy
        truncated  = self._step_count >= self.MAX_STEPS

        return reward, terminated, truncated

    def _get_info(self) -> Dict[str, Any]:
        torso_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        return {
            "x_pos":     float(self.data.xpos[torso_id][0]),
            "z_height":  float(self.data.xpos[torso_id][2]),
            "x_vel":     float(self.data.qvel[self._rootx_id]),
            "is_healthy": self._is_healthy(),
            "step":       self._step_count,
        }

    def _reset_model(self) -> np.ndarray:
        self._noisy_reset(scale=0.005)
        return self._get_obs()
