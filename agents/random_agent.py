"""Random baseline agent — samples uniformly from the action space."""

import numpy as np
import gymnasium as gym


class RandomAgent:
    """
    Baseline that takes random actions each step.
    Useful as a sanity-check lower bound.
    """

    def __init__(self, env: gym.Env, seed: int = 0):
        self.env = env
        self.rng = np.random.default_rng(seed)

    def predict(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        return self.rng.uniform(
            low=self.env.action_space.low,
            high=self.env.action_space.high,
        ).astype(np.float32)

    def evaluate(self, n_episodes: int = 10) -> dict:
        rewards, lengths = [], []
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            total, steps = 0.0, 0
            terminated = truncated = False
            while not (terminated or truncated):
                action = self.predict(obs)
                obs, r, terminated, truncated, _ = self.env.step(action)
                total += r
                steps += 1
            rewards.append(total)
            lengths.append(steps)
        return {
            "mean_reward": float(np.mean(rewards)),
            "std_reward":  float(np.std(rewards)),
            "mean_length": float(np.mean(lengths)),
        }
