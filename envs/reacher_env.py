"""
Reacher — 2-DOF planar arm must reach a randomly placed target.

Observation (9,):
    [cos(θ_shoulder), sin(θ_shoulder),
     cos(θ_elbow),    sin(θ_elbow),
     ω_shoulder,      ω_elbow,
     Δx, Δy, Δdist]           (fingertip → target vector + distance)

Action (2,):
    [shoulder_torque, elbow_torque]  ∈ [-1, 1]

Reward:
    -dist(fingertip, target) − 0.01 * ||action||²
    Sparse bonus +1 when dist < 0.02 m (success)

Episode resets target and arm pose; max 200 steps.
"""

from typing import Any, Dict, Optional, Tuple

import mujoco
import numpy as np
from gymnasium import spaces

from envs.base_env import BaseMuJoCoEnv


class ReacherEnv(BaseMuJoCoEnv):

    MAX_STEPS   = 200
    SUCCESS_DIST = 0.025   # meters, "close enough"

    def __init__(self, render_mode: Optional[str] = None, **kwargs):
        super().__init__(
            model_filename="reacher.xml",
            frame_skip=2,
            render_mode=render_mode,
            **kwargs,
        )
        obs_high = np.ones(9, dtype=np.float32) * np.inf
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        # Cache joint / site indices
        self._shoulder_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder")
        self._elbow_id    = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "elbow")
        self._target_x_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_x")
        self._target_y_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "target_y")

    def _fingertip_pos(self) -> np.ndarray:
        return self.get_site_pos("fingertip_site")[:2]   # (x, y)

    def _target_pos(self) -> np.ndarray:
        return self.get_site_pos("target_site")[:2]

    def _get_obs(self) -> np.ndarray:
        shoulder_ang = self.data.qpos[self._shoulder_id]
        elbow_ang    = self.data.qpos[self._elbow_id]
        shoulder_vel = self.data.qvel[self._shoulder_id]
        elbow_vel    = self.data.qvel[self._elbow_id]

        fp  = self._fingertip_pos()
        tgt = self._target_pos()
        delta = tgt - fp
        dist  = np.linalg.norm(delta)

        obs = np.array([
            np.cos(shoulder_ang), np.sin(shoulder_ang),
            np.cos(elbow_ang),    np.sin(elbow_ang),
            shoulder_vel,         elbow_vel,
            delta[0],             delta[1],
            dist,
        ], dtype=np.float32)
        return obs

    def _compute_reward(self, action: np.ndarray) -> Tuple[float, bool, bool]:
        fp   = self._fingertip_pos()
        tgt  = self._target_pos()
        dist = float(np.linalg.norm(tgt - fp))

        reward    = -dist - 0.01 * float(np.sum(action ** 2))
        if dist < self.SUCCESS_DIST:
            reward += 1.0

        terminated = False
        truncated  = self._step_count >= self.MAX_STEPS

        return reward, terminated, truncated

    def _get_info(self) -> Dict[str, Any]:
        fp   = self._fingertip_pos()
        tgt  = self._target_pos()
        dist = float(np.linalg.norm(tgt - fp))
        return {
            "dist_to_target": dist,
            "success":        dist < self.SUCCESS_DIST,
            "fingertip":      fp.tolist(),
            "target":         tgt.tolist(),
            "step":           self._step_count,
        }

    def _reset_model(self) -> np.ndarray:
        # Random arm configuration
        self.data.qpos[self._shoulder_id] = self.np_random.uniform(-np.pi, np.pi)
        self.data.qpos[self._elbow_id]    = self.np_random.uniform(-np.pi, np.pi)
        self.data.qvel[self._shoulder_id] = 0.0
        self.data.qvel[self._elbow_id]    = 0.0

        # Random target within reach (total arm length = 0.12 + 0.10 = 0.22 m)
        # Sample in polar coords to avoid corners
        r     = self.np_random.uniform(0.05, 0.20)
        theta = self.np_random.uniform(-np.pi, np.pi)
        self.data.qpos[self._target_x_id] = r * np.cos(theta)
        self.data.qpos[self._target_y_id] = r * np.sin(theta)

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()
