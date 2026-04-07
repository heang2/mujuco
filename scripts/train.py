#!/usr/bin/env python
"""
Train a PPO agent on a MuJoCo environment.

Usage:
    # From the project root:
    python scripts/train.py --config configs/hopper_ppo.yaml
    python scripts/train.py --env Ant --steps 2000000 --seed 1
    python scripts/train.py --env CartPole --steps 200000 --lr 5e-4
"""

import sys
import argparse
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.trainer import Trainer


def parse_args():
    p = argparse.ArgumentParser(description="Train PPO on a MuJoCo environment")

    # Config file (overrides all other args if provided)
    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML config file (e.g. configs/hopper_ppo.yaml)")

    # Quick-launch overrides
    p.add_argument("--env",   type=str, default="Hopper",
                   choices=["CartPole", "Reacher", "Hopper", "Ant"],
                   help="Environment name")
    p.add_argument("--steps", type=int, default=None, help="Total training timesteps")
    p.add_argument("--seed",  type=int, default=42,   help="Random seed")
    p.add_argument("--lr",    type=float, default=None, help="Learning rate")
    p.add_argument("--run-name", type=str, default=None,
                   help="Custom run name for logging")

    return p.parse_args()


def build_config_from_args(args) -> dict:
    """Build a minimal config dict from CLI args."""
    config_path = Path(__file__).parent.parent / "configs" / f"{args.env.lower()}_ppo.yaml"

    import yaml
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        config = {"env_name": args.env, "ppo": {}}

    if args.steps:
        config["total_timesteps"] = args.steps
    if args.seed:
        config["seed"] = args.seed
    if args.lr:
        config.setdefault("ppo", {})["lr"] = args.lr
    if args.run_name:
        config["run_name"] = args.run_name

    return config


def main():
    args = parse_args()

    if args.config:
        trainer = Trainer.from_yaml(args.config)
    else:
        config  = build_config_from_args(args)
        trainer = Trainer(config)

    trainer.train()


if __name__ == "__main__":
    main()
