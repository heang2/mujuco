"""
Walker2D — bipedal robot must walk forward as fast as possible.

Observation (17,):
    z-height, rooty angle (2),
    6 joint positions (right & left hip, knee, ankle),
    rootx_vel, rootz_vel, rooty_vel (3),
    6 joint velocities
    → 2 + 6 + 3 + 6 = 17

Action (6,):
    [right_hip, right_knee, right_ankle, left_hip, left_knee, left_ankle] ∈ [-1, 1]

Reward:
    forward_vel × weight − ctrl_cost + alive_bonus

Harder than Hopper: bilateral symmetry, coordination of two legs.
"""

from typing import Any, Dict, Optional, Tuple

import mujoco
import numpy as np
from gymnasium import spaces

from envs.base_env import BaseMuJoCoEnv


class Walker2DEnv(BaseMuJoCoEnv):

    HEALTHY_Z_MIN   = 0.8     # minimum torso height
    HEALTHY_Z_MAX   = 2.5     # maximum torso height (falling over backwards)
    HEALTHY_ANG_MAX = 1.0     # max |rooty| before falling
    MAX_STEPS       = 1000
    FORWARD_WEIGHT  = 1.25
    CTRL_COST_WEIGHT = 1e-3
    HEALTHY_REWARD  = 1.0

    def __init__(
        self,
        render_mode: Optional[str] = None,
        terminate_when_unhealthy: bool = True,
        **kwargs,
    ):
        super().__init__(
            model_filename="walker2d.xml",
            frame_skip=4,
            render_mode=render_mode,
            **kwargs,
        )
        self.terminate_when_unhealthy = terminate_when_unhealthy

        # 17-dimensional observation
        obs_high = np.full(17, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        self._torso_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self._rootx_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "rootx")
        self._rooty_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "rooty")

    def _is_healthy(self) -> bool:
        z   = float(self.data.xpos[self._torso_id][2])
        ang = float(self.data.qpos[self._rooty_id])
        return (
            self.HEALTHY_Z_MIN < z < self.HEALTHY_Z_MAX
            and abs(ang) < self.HEALTHY_ANG_MAX
        )

    def _get_obs(self) -> np.ndarray:
        # qpos: rootx(0), rootz(1), rooty(2), rh(3), rk(4), ra(5), lh(6), lk(7), la(8)
        # qvel: vx(0), vz(1), vy(2), rh_v(3)...la_v(8)
        qpos = self.data.qpos.copy()
        qvel = np.clip(self.data.qvel.copy(), -10.0, 10.0)

        # Exclude rootx; include rootz, rooty, 6 joint positions
        obs = np.concatenate([
            qpos[1:],    # rootz, rooty, 6 joints = 8
            qvel[:],     # vx, vz, vy, 6 joint vels = 9
        ]).astype(np.float32)
        return obs

    def _compute_reward(self, action: np.ndarray) -> Tuple[float, bool, bool]:
        x_vel = float(self.data.qvel[self._rootx_id])

        forward_r = self.FORWARD_WEIGHT * x_vel
        ctrl_cost = self.CTRL_COST_WEIGHT * float(np.sum(action ** 2))
        healthy_r = self.HEALTHY_REWARD if self._is_healthy() else 0.0

        reward = forward_r - ctrl_cost + healthy_r

        terminated = (not self._is_healthy()) and self.terminate_when_unhealthy
        truncated  = self._step_count >= self.MAX_STEPS

        return reward, terminated, truncated

    def _get_info(self) -> Dict[str, Any]:
        torso_pos = self.data.xpos[self._torso_id]
        return {
            "x_pos":      float(torso_pos[0]),
            "z_height":   float(torso_pos[2]),
            "x_vel":      float(self.data.qvel[self._rootx_id]),
            "is_healthy": self._is_healthy(),
            "step":       self._step_count,
        }

    def _reset_model(self) -> np.ndarray:
        self._noisy_reset(scale=0.005)
        return self._get_obs()
