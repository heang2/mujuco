#!/usr/bin/env python
"""
Evaluate a trained PPO checkpoint.

Usage:
    python scripts/evaluate.py --checkpoint logs/Hopper_*/final_model.pt --env Hopper
    python scripts/evaluate.py --checkpoint logs/CartPole_*/final_model.pt --env CartPole --episodes 50
    python scripts/evaluate.py --checkpoint path/to/model.pt --env Ant --record
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from agents.ppo    import PPO, PPOConfig
from envs          import make_env
from training.evaluator import Evaluator


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained PPO checkpoint")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to .pt checkpoint file")
    p.add_argument("--env", type=str, required=True,
                   choices=["CartPole", "Reacher", "Hopper", "Ant"])
    p.add_argument("--episodes", type=int, default=20,
                   help="Number of evaluation episodes")
    p.add_argument("--seed",     type=int, default=0)
    p.add_argument("--stochastic", action="store_true",
                   help="Use stochastic policy (default: deterministic)")
    p.add_argument("--record",     action="store_true",
                   help="Save one episode as video frames (requires imageio)")
    p.add_argument("--record-out", type=str, default="episode.mp4",
                   help="Output video file path")
    return p.parse_args()


def load_agent(checkpoint_path: str, env_name: str) -> PPO:
    """Rebuild agent from checkpoint (env needed for obs/act dims)."""
    env     = make_env(env_name)
    agent   = PPO(env, PPOConfig())   # config will be overwritten by load
    agent.load(checkpoint_path)
    env.close()
    return agent


def main():
    args = parse_args()

    print(f"\nLoading checkpoint: {args.checkpoint}")
    agent = load_agent(args.checkpoint, args.env)
    print(f"Loaded model (trained for {agent._global_step:,} steps)\n")

    evaluator = Evaluator(
        args.env,
        n_episodes=args.episodes,
        deterministic=not args.stochastic,
        seed=args.seed,
    )

    print(f"Evaluating {args.episodes} episodes on {args.env}...")
    result = evaluator.evaluate(agent)

    print(f"\n{'='*50}")
    print(f"  Environment  : {args.env}")
    print(f"  Episodes     : {args.episodes}")
    print(f"  Mode         : {'stochastic' if args.stochastic else 'deterministic'}")
    print(f"  Mean Reward  : {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
    print(f"  Min / Max    : {result['min_reward']:.2f} / {result['max_reward']:.2f}")
    print(f"  Mean Length  : {result['mean_length']:.1f}")
    if result["success_rate"] is not None:
        print(f"  Success Rate : {result['success_rate']*100:.1f}%")
    print(f"{'='*50}\n")

    if args.record:
        try:
            import imageio
        except ImportError:
            print("imageio not installed — run: pip install imageio[ffmpeg]")
            return

        print(f"Recording episode to {args.record_out}...")
        frames = evaluator.record_episode(agent)
        imageio.mimsave(args.record_out, frames, fps=50)
        print(f"Saved video → {args.record_out}")

    evaluator.close()


if __name__ == "__main__":
    main()
