"""
Ant — 4-legged robot must move forward (along +x) as fast as possible.

Observation (27,):
    torso z-height (1),
    torso orientation quaternion (4),
    8 joint positions,
    torso linear + angular velocity (6),
    8 joint velocities,
    → total = 1 + 4 + 8 + 6 + 8 = 27

Action (8,):
    [hip1, ankle1, hip2, ankle2, hip3, ankle3, hip4, ankle4]  ∈ [-1, 1]

Reward:
    forward_vel × weight − ctrl_cost − contact_cost + alive_bonus

Episode ends when torso falls below 0.2 m or rises above 1.0 m.
"""

from typing import Any, Dict, Optional, Tuple

import mujoco
import numpy as np
from gymnasium import spaces

from envs.base_env import BaseMuJoCoEnv


class AntEnv(BaseMuJoCoEnv):

    HEALTHY_Z_RANGE  = (0.2, 1.0)
    MAX_STEPS        = 1000
    FORWARD_WEIGHT   = 1.0
    CTRL_COST_WEIGHT = 0.5e-4
    CONTACT_COST_WEIGHT = 5e-4
    CONTACT_FORCE_RANGE = (-1.0, 1.0)
    ALIVE_BONUS      = 1.0

    def __init__(
        self,
        render_mode: Optional[str] = None,
        terminate_when_unhealthy: bool = True,
        use_contact_forces: bool = True,
        **kwargs,
    ):
        super().__init__(
            model_filename="ant.xml",
            frame_skip=5,
            render_mode=render_mode,
            **kwargs,
        )
        self.terminate_when_unhealthy = terminate_when_unhealthy
        self.use_contact_forces = use_contact_forces

        # Observation size computed dynamically from the model
        # qpos: 7 (free joint) + n_joints; excluding x,y → (nq - 2) values
        # qvel: nv values
        # contact forces: nbody × 6 (if enabled)
        obs_size = (self.model.nq - 2) + self.model.nv
        if use_contact_forces:
            obs_size += self.model.nbody * 6
        obs_high = np.inf * np.ones(obs_size, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        # Find torso free-joint address in qpos / qvel
        # free joint: 7 qpos dofs (3 pos + 4 quat), 6 qvel dofs (3 lin + 3 ang)
        self._torso_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso"
        )

    def _is_healthy(self) -> bool:
        # qpos[2] is absolute z for the free joint root body
        z = float(self.data.xpos[self._torso_body_id][2])
        return self.HEALTHY_Z_RANGE[0] < z < self.HEALTHY_Z_RANGE[1]

    def _get_obs(self) -> np.ndarray:
        # qpos: [x, y, z, qw, qx, qy, qz, joint0..joint7]
        # qvel: [vx, vy, vz, wx, wy, wz, jv0..jv7]
        position = self.data.qpos.flat.copy()
        velocity = np.clip(self.data.qvel.flat.copy(), -10, 10)

        # Exclude global x, y position (indices 0, 1); keep z and orientation
        obs_parts = [position[2:], velocity]

        if self.use_contact_forces:
            cfrc = np.clip(
                self.data.cfrc_ext.flat.copy(),
                *self.CONTACT_FORCE_RANGE,
            )
            obs_parts.append(cfrc)

        return np.concatenate(obs_parts).astype(np.float32)

    def _compute_reward(self, action: np.ndarray) -> Tuple[float, bool, bool]:
        x_vel = float(self.data.qvel[0])   # global x velocity

        forward_reward = self.FORWARD_WEIGHT * x_vel
        ctrl_cost      = self.CTRL_COST_WEIGHT * float(np.sum(action ** 2))
        contact_cost   = self.CONTACT_COST_WEIGHT * float(
            np.sum(np.square(np.clip(self.data.cfrc_ext, *self.CONTACT_FORCE_RANGE)))
        )
        alive_bonus    = self.ALIVE_BONUS if self._is_healthy() else 0.0

        reward = forward_reward - ctrl_cost - contact_cost + alive_bonus

        terminated = (not self._is_healthy()) and self.terminate_when_unhealthy
        truncated  = self._step_count >= self.MAX_STEPS

        return reward, terminated, truncated

    def _get_info(self) -> Dict[str, Any]:
        pos = self.data.xpos[self._torso_body_id]
        return {
            "x_pos":      float(pos[0]),
            "y_pos":      float(pos[1]),
            "z_height":   float(pos[2]),
            "x_vel":      float(self.data.qvel[0]),
            "is_healthy": self._is_healthy(),
            "step":       self._step_count,
        }

    def _reset_model(self) -> np.ndarray:
        self._noisy_reset(scale=0.1)
        return self._get_obs()
