#!/usr/bin/env python
"""
Compare trained PPO agents against the random baseline across all environments.
Generates a bar chart saved to assets/benchmark.png.

Usage:
    # Evaluate random agents only (no trained checkpoints needed):
    python scripts/benchmark.py --random-only

    # Evaluate trained agents (must provide checkpoint directory):
    python scripts/benchmark.py --log-dir logs/
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from envs          import REGISTRY, make_env
from agents.random_agent import RandomAgent
from agents.ppo    import PPO, PPOConfig
from utils.plotting import plot_multi_env_comparison


def eval_random(env_name: str, n_episodes: int = 20) -> float:
    env   = make_env(env_name)
    agent = RandomAgent(env, seed=0)
    res   = agent.evaluate(n_episodes)
    env.close()
    return res["mean_reward"]


def find_checkpoint(log_dir: Path, env_name: str) -> Path | None:
    """Look for the most recent final_model.pt for this env."""
    candidates = sorted(log_dir.glob(f"{env_name}*/final_model.pt"))
    if candidates:
        return candidates[-1]
    # Fall back to any .pt checkpoint
    candidates = sorted(log_dir.glob(f"{env_name}*/**/*.pt"))
    return candidates[-1] if candidates else None


def eval_ppo(env_name: str, ckpt_path: Path, n_episodes: int = 20) -> float:
    env   = make_env(env_name)
    agent = PPO(env, PPOConfig())
    agent.load(str(ckpt_path))

    rewards = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        total  = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            action = agent.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            total += r
        rewards.append(total)

    env.close()
    return float(np.mean(rewards))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--random-only", action="store_true",
                   help="Skip PPO evaluation (no checkpoints needed)")
    p.add_argument("--log-dir",  type=str, default="logs",
                   help="Directory containing training run subdirectories")
    p.add_argument("--episodes", type=int, default=20)
    args = p.parse_args()

    log_dir = Path(args.log_dir)
    results = {}

    print(f"\n{'ENV':<12}  {'Random':>10}  {'PPO':>10}")
    print("─" * 38)

    for env_name in REGISTRY:
        rand_rew = eval_random(env_name, args.episodes)

        if args.random_only:
            ppo_rew = None
            print(f"{env_name:<12}  {rand_rew:>10.2f}  {'N/A':>10}")
        else:
            ckpt = find_checkpoint(log_dir, env_name)
            if ckpt is None:
                print(f"{env_name:<12}  {rand_rew:>10.2f}  {'no ckpt':>10}")
                ppo_rew = rand_rew  # placeholder for chart
            else:
                ppo_rew = eval_ppo(env_name, ckpt, args.episodes)
                print(f"{env_name:<12}  {rand_rew:>10.2f}  {ppo_rew:>10.2f}  ← {ckpt.name}")

        results[env_name] = {
            "random": rand_rew,
            "ppo":    ppo_rew if ppo_rew is not None else rand_rew,
        }

    out = Path("assets") / "benchmark.png"
    plot_multi_env_comparison(results, str(out))
    print(f"\nSaved benchmark chart → {out}")


if __name__ == "__main__":
    main()
