"""
Pusher — 7-DOF robot arm must push a puck to a goal position on a table.

This environment is significantly more complex than Reacher:
  • 7 joint robot (vs 2)
  • Object interaction (contact forces with puck)
  • Two separate subtasks: reach puck, then push puck to goal

Observation (23,):
    7 joint positions,
    7 joint velocities,
    end-effector position (3),
    puck-to-ee delta (3),
    puck-to-goal delta (3)

Action (7,):
    joint torques ∈ [-1, 1]

Reward:
    −dist(ee, puck) × w1
    − dist(puck, goal) × w2
    − ctrl_cost
    + success_bonus if dist(puck, goal) < 0.05

Dense reward encourages reaching puck first, then pushing.
"""

from typing import Any, Dict, Optional, Tuple

import mujoco
import numpy as np
from gymnasium import spaces

from envs.base_env import BaseMuJoCoEnv

# Joint names for the 7-DOF arm
JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "upper_arm_roll",
    "elbow_flex", "forearm_roll", "wrist_flex", "wrist_roll",
]


class PusherEnv(BaseMuJoCoEnv):

    MAX_STEPS         = 200
    SUCCESS_DIST      = 0.05    # puck-to-goal distance for success
    REACH_WEIGHT      = 0.5     # weight for ee-to-puck distance
    PUSH_WEIGHT       = 1.25    # weight for puck-to-goal distance
    CTRL_COST_WEIGHT  = 1e-4
    SUCCESS_BONUS     = 5.0
    PUCK_GOAL_RANGE   = (0.25, 0.55)   # goal radius range from origin

    def __init__(self, render_mode: Optional[str] = None, **kwargs):
        super().__init__(
            model_filename="pusher.xml",
            frame_skip=5,
            render_mode=render_mode,
            **kwargs,
        )
        # 23-dim obs
        obs_high = np.full(23, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)

        # Cache joint indices
        self._joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            for n in JOINT_NAMES
        ]
        self._puck_x_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "puck_x")
        self._puck_y_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "puck_y")
        self._goal_x_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "goal_x")
        self._goal_y_id  = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "goal_y")

    def _ee_pos(self) -> np.ndarray:
        return self.get_site_pos("ee_site")

    def _puck_pos(self) -> np.ndarray:
        return self.get_site_pos("puck_site")

    def _goal_pos(self) -> np.ndarray:
        return self.get_site_pos("goal_site")

    def _get_obs(self) -> np.ndarray:
        # Joint positions and velocities
        q_pos = np.array([self.data.qpos[i] for i in self._joint_ids], dtype=np.float32)
        q_vel = np.clip(
            [self.data.qvel[i] for i in self._joint_ids], -10.0, 10.0
        ).astype(np.float32)

        ee   = self._ee_pos()
        puck = self._puck_pos()
        goal = self._goal_pos()

        ee_to_puck   = (puck - ee)[:3]
        puck_to_goal = (goal - puck)[:3]

        obs = np.concatenate([
            q_pos,          # 7
            q_vel,          # 7
            ee[:3],         # 3  (absolute ee position)
            ee_to_puck,     # 3
            puck_to_goal,   # 3
        ]).astype(np.float32)
        return obs

    def _compute_reward(self, action: np.ndarray) -> Tuple[float, bool, bool]:
        ee   = self._ee_pos()
        puck = self._puck_pos()
        goal = self._goal_pos()

        dist_reach = float(np.linalg.norm(puck[:2] - ee[:2]))
        dist_push  = float(np.linalg.norm(goal[:2] - puck[:2]))
        ctrl_cost  = self.CTRL_COST_WEIGHT * float(np.sum(action ** 2))

        reward = (
            -self.REACH_WEIGHT * dist_reach
            - self.PUSH_WEIGHT * dist_push
            - ctrl_cost
        )

        success = dist_push < self.SUCCESS_DIST
        if success:
            reward += self.SUCCESS_BONUS

        terminated = False
        truncated  = self._step_count >= self.MAX_STEPS
        return reward, terminated, truncated

    def _get_info(self) -> Dict[str, Any]:
        puck = self._puck_pos()
        goal = self._goal_pos()
        ee   = self._ee_pos()
        return {
            "dist_puck_goal": float(np.linalg.norm(goal[:2] - puck[:2])),
            "dist_ee_puck":   float(np.linalg.norm(puck[:2] - ee[:2])),
            "success":        float(np.linalg.norm(goal[:2] - puck[:2])) < self.SUCCESS_DIST,
            "puck_pos":       puck[:2].tolist(),
            "goal_pos":       goal[:2].tolist(),
            "step":           self._step_count,
        }

    def _reset_model(self) -> np.ndarray:
        # Random arm configuration (near zero to avoid self-collision)
        for jid in self._joint_ids:
            self.data.qpos[jid] = self.np_random.uniform(-0.3, 0.3)
            self.data.qvel[jid] = 0.0

        # Random puck position on table
        puck_r     = self.np_random.uniform(0.10, 0.25)
        puck_theta = self.np_random.uniform(-np.pi / 3, np.pi / 3)
        self.data.qpos[self._puck_x_id] = puck_r * np.cos(puck_theta)
        self.data.qpos[self._puck_y_id] = puck_r * np.sin(puck_theta)

        # Random goal position
        lo, hi = self.PUCK_GOAL_RANGE
        goal_r     = self.np_random.uniform(lo, hi)
        goal_theta = self.np_random.uniform(-np.pi / 3, np.pi / 3)
        self.data.qpos[self._goal_x_id] = goal_r * np.cos(goal_theta) - 0.45
        self.data.qpos[self._goal_y_id] = goal_r * np.sin(goal_theta)

        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()
