#!/usr/bin/env python
"""
Hyperparameter sweep — grid search or random search.

Usage:
    # Grid search over lr and clip_eps:
    python scripts/sweep.py \\
        --config configs/hopper_ppo.yaml \\
        --params "ppo.lr=1e-4,3e-4,1e-3" "ppo.clip_eps=0.1,0.2,0.3" \\
        --seeds 3

    # Random search (20 trials):
    python scripts/sweep.py \\
        --config configs/hopper_ppo.yaml \\
        --random --n-trials 20 --seeds 2

    # Lightweight quick sweep:
    python scripts/sweep.py \\
        --config configs/cartpole_ppo.yaml \\
        --params "ppo.lr=3e-4,1e-3" --seeds 1
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from utils.sweep import GridSweep, RandomSweep


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",   type=str, required=True)
    p.add_argument("--params",   nargs="+", default=None,
                   help='Grid params: "key=v1,v2,v3" ...')
    p.add_argument("--random",   action="store_true", help="Use random search")
    p.add_argument("--n-trials", type=int, default=20, help="Random search trials")
    p.add_argument("--seeds",    type=int, default=1,  help="Seeds per config")
    p.add_argument("--sweep-dir", type=str, default="sweeps")
    p.add_argument("--eval-episodes", type=int, default=5)
    return p.parse_args()


def parse_param_grid(param_strs: list) -> dict:
    """Parse 'key=v1,v2,v3' strings into a search space dict."""
    space = {}
    for s in (param_strs or []):
        key, vals_str = s.split("=", 1)
        values = []
        for v in vals_str.split(","):
            v = v.strip()
            try:
                values.append(float(v))
            except ValueError:
                values.append(v)
        space[key.strip()] = values
    return space


def main():
    args  = parse_args()
    space = parse_param_grid(args.params)

    if not space and not args.random:
        print("No --params specified and not using --random. Nothing to sweep.")
        return

    if args.random:
        # Convert list params to samplers for random search
        random_space = {}
        for k, v in space.items():
            if isinstance(v, list):
                random_space[k] = v   # RandomSweep handles lists by uniform sampling
            else:
                random_space[k] = v

        sweep = RandomSweep(
            base_config=args.config,
            search_space=random_space,
            n_trials=args.n_trials,
            n_seeds=args.seeds,
            sweep_dir=args.sweep_dir,
            eval_episodes=args.eval_episodes,
        )
    else:
        sweep = GridSweep(
            base_config=args.config,
            search_space=space,
            n_seeds=args.seeds,
            sweep_dir=args.sweep_dir,
            eval_episodes=args.eval_episodes,
        )

    results = sweep.run()

    out = Path(args.sweep_dir) / sweep._sweep_name / "sweep_plot.png"
    sweep.plot_results(str(out))


if __name__ == "__main__":
    main()
