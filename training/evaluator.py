"""
Policy evaluator — runs deterministic rollouts and computes statistics.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from envs import make_env


class Evaluator:
    """
    Evaluate a trained policy over multiple episodes.

    Supports:
      - Mean / std / min / max episode reward
      - Episode length statistics
      - Success rate (for environments that report it)
      - Frame capture for video generation
    """

    def __init__(
        self,
        env_name: str,
        n_episodes: int = 20,
        deterministic: bool = True,
        seed: int = 0,
        render_mode: Optional[str] = None,
    ):
        self.env_name     = env_name
        self.n_episodes   = n_episodes
        self.deterministic = deterministic
        self.seed          = seed
        self.env = make_env(env_name, render_mode=render_mode)

    def evaluate(self, agent) -> Dict[str, Any]:
        """
        Run `n_episodes` deterministic episodes.

        Args:
            agent: Any object with a `.predict(obs) -> action` method.

        Returns:
            Dictionary of statistics.
        """
        rewards:  List[float] = []
        lengths:  List[int]   = []
        successes: List[bool] = []

        for ep in range(self.n_episodes):
            obs, _ = self.env.reset(seed=self.seed + ep)
            ep_reward, ep_length = 0.0, 0
            terminated = truncated = False
            ep_success = False

            while not (terminated or truncated):
                action = agent.predict(obs, deterministic=self.deterministic)
                obs, reward, terminated, truncated, info = self.env.step(action)
                ep_reward += float(reward)
                ep_length += 1

                # Track success if environment reports it
                if info.get("success", False):
                    ep_success = True

            rewards.append(ep_reward)
            lengths.append(ep_length)
            successes.append(ep_success)

        return {
            "mean_reward":  float(np.mean(rewards)),
            "std_reward":   float(np.std(rewards)),
            "min_reward":   float(np.min(rewards)),
            "max_reward":   float(np.max(rewards)),
            "mean_length":  float(np.mean(lengths)),
            "success_rate": float(np.mean(successes)) if any(successes) else None,
            "n_episodes":   self.n_episodes,
            "all_rewards":  rewards,
        }

    def record_episode(self, agent) -> List[np.ndarray]:
        """
        Run one episode and return a list of RGB frames for video.

        Args:
            agent: Policy with `.predict()` method.

        Returns:
            List of (H, W, 3) uint8 numpy arrays.
        """
        rec_env = make_env(self.env_name, render_mode="rgb_array")
        obs, _  = rec_env.reset(seed=self.seed)
        frames  = [rec_env.render()]
        terminated = truncated = False

        while not (terminated or truncated):
            action = agent.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = rec_env.step(action)
            frame = rec_env.render()
            if frame is not None:
                frames.append(frame)

        rec_env.close()
        return frames

    def close(self) -> None:
        self.env.close()
