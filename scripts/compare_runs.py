#!/usr/bin/env python
"""
Compare multiple training runs — algorithms, seeds, or hyperparameters.

Usage:
    # Auto-discover PPO, SAC, TD3 runs for Hopper:
    python scripts/compare_runs.py --env Hopper

    # Manually specify groups:
    python scripts/compare_runs.py \\
        --groups "PPO:Hopper_PPO_1,Hopper_PPO_2" "SAC:Hopper_SAC_1,Hopper_SAC_2"

    # Generate full comparison report:
    python scripts/compare_runs.py --env Hopper --full-report
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.comparison import RunComparison


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env",        type=str,   default=None,
                   help="Auto-discover runs for this environment name")
    p.add_argument("--groups",     nargs="+",  default=None,
                   help='Manual groups: "Label:dir1,dir2" ...')
    p.add_argument("--log-dir",    type=str,   default="logs")
    p.add_argument("--out-dir",    type=str,   default="assets")
    p.add_argument("--algos",      nargs="+",  default=["PPO", "SAC", "TD3"])
    p.add_argument("--full-report",action="store_true",
                   help="Generate all comparison plots")
    return p.parse_args()


def main():
    args = parse_args()
    comp = RunComparison(log_dir=args.log_dir)

    if args.groups:
        for group_str in args.groups:
            label, dirs = group_str.split(":", 1)
            comp.add_group(label, dirs.split(","))
    elif args.env:
        comp.auto_discover(args.env, algorithms=args.algos)
    else:
        print("Provide --env or --groups. Run with --help for usage.")
        return

    comp.print_table()

    out = Path(args.out_dir)
    comp.plot_comparison(str(out / "comparison_learning_curves.png"))
    comp.plot_sample_efficiency(str(out / "comparison_sample_efficiency.png"))

    if args.full_report:
        print("\nFull report generated.")


if __name__ == "__main__":
    main()
