"""
Multi-run and multi-algorithm comparison tools.

Loads CSV logs from multiple training runs and produces:
  - Learning curves with confidence intervals (multiple seeds)
  - Algorithm comparison (PPO vs SAC vs TD3 on same env)
  - Sample efficiency comparison (reward per 1000 env steps)
  - Hyperparameter sensitivity plots
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_csv(path: str, x_col: str, y_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load two columns from a CSV file, return (x, y) numpy arrays."""
    xs, ys = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                xs.append(float(row[x_col]))
                ys.append(float(row[y_col]))
            except (KeyError, ValueError):
                pass
    return np.array(xs), np.array(ys)


def _smooth(y: np.ndarray, window: int) -> np.ndarray:
    if len(y) < window:
        return y
    return np.convolve(y, np.ones(window) / window, mode="valid")


def _interpolate_to_common_steps(
    runs: List[Tuple[np.ndarray, np.ndarray]],
    n_points: int = 200,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Interpolate multiple (steps, rewards) curves onto a common x-axis.
    Returns (common_steps, matrix of shape (n_runs, n_points)).
    """
    max_step = min(r[0][-1] for r in runs if len(r[0]) > 0)
    xs       = np.linspace(0, max_step, n_points)
    ys_mat   = np.zeros((len(runs), n_points))

    for i, (run_x, run_y) in enumerate(runs):
        ys_mat[i] = np.interp(xs, run_x, run_y)

    return xs, ys_mat


# ──────────────────────────────────────────────────────────────────────────────
# RunComparison
# ──────────────────────────────────────────────────────────────────────────────

class RunComparison:
    """
    Compare multiple training runs.

    Each run is identified by a directory under log_dir that contains
    an eval.csv file.

    Usage:
        comp = RunComparison(log_dir="logs/")
        comp.add_group("PPO",  ["PPO_Hopper_s0", "PPO_Hopper_s1", "PPO_Hopper_s2"])
        comp.add_group("SAC",  ["SAC_Hopper_s0", "SAC_Hopper_s1"])
        comp.plot_comparison("comparison.png")
        comp.plot_sample_efficiency("sample_eff.png")
        comp.print_table()
    """

    COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]

    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self._groups: Dict[str, List[str]] = {}   # label → list of run dirs

    def add_group(self, label: str, run_dirs: List[str]) -> "RunComparison":
        """Add a group of runs (e.g. multiple seeds of the same algorithm)."""
        self._groups[label] = run_dirs
        return self

    def auto_discover(
        self,
        env_name: str,
        algorithms: Optional[List[str]] = None,
    ) -> "RunComparison":
        """
        Auto-discover runs in log_dir matching '{algo}_{env_name}*'.

        Args:
            env_name:   Environment to match (e.g. "Hopper")
            algorithms: Algorithms to look for (default: ["PPO", "SAC", "TD3"])
        """
        algorithms = algorithms or ["PPO", "SAC", "TD3"]
        for algo in algorithms:
            matches = sorted(self.log_dir.glob(f"{env_name}*"))
            # Filter by presence of eval.csv and algo prefix
            valid = [str(p.name) for p in matches
                     if (p / "eval.csv").exists() and algo.lower() in p.name.lower()]
            if valid:
                self.add_group(algo, valid)
        return self

    def _load_group(
        self, run_dirs: List[str], csv_name: str = "eval.csv"
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        runs = []
        for d in run_dirs:
            csv_path = self.log_dir / d / csv_name
            if not csv_path.exists():
                print(f"  Warning: {csv_path} not found, skipping")
                continue
            steps, rewards = _load_csv(str(csv_path), "step", "mean_reward")
            if len(steps) > 0:
                runs.append((steps, rewards))
        return runs

    def plot_comparison(
        self,
        save_path: str,
        smooth_window: int = 5,
        shade_std: bool = True,
        show: bool = False,
    ) -> None:
        """
        Plot mean ± std learning curves for each algorithm group.
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title("Algorithm Comparison — Evaluation Reward", fontsize=13, fontweight="bold")

        for idx, (label, run_dirs) in enumerate(self._groups.items()):
            color = self.COLORS[idx % len(self.COLORS)]
            runs  = self._load_group(run_dirs)
            if not runs:
                continue

            if len(runs) == 1:
                xs, ys = runs[0]
                ax.plot(xs, _smooth(ys, smooth_window), label=label, color=color, linewidth=2)
            else:
                xs, ys_mat = _interpolate_to_common_steps(runs)
                mean = ys_mat.mean(axis=0)
                std  = ys_mat.std(axis=0)
                ax.plot(xs, mean, label=f"{label} (n={len(runs)})",
                        color=color, linewidth=2)
                if shade_std:
                    ax.fill_between(xs, mean - std, mean + std,
                                    alpha=0.2, color=color)

        ax.set_xlabel("Environment Steps")
        ax.set_ylabel("Mean Episode Reward (eval)")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        if show: plt.show()
        plt.close(fig)
        print(f"Saved comparison plot → {save_path}")

    def plot_sample_efficiency(
        self,
        save_path: str,
        reward_threshold: Optional[float] = None,
        show: bool = False,
    ) -> None:
        """
        Bar chart: area under the learning curve (proxy for sample efficiency).
        """
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        fig.suptitle("Sample Efficiency Comparison", fontsize=13, fontweight="bold")

        labels, aucs, final_rewards = [], [], []
        colors = []

        for idx, (label, run_dirs) in enumerate(self._groups.items()):
            color = self.COLORS[idx % len(self.COLORS)]
            runs  = self._load_group(run_dirs)
            if not runs:
                continue
            xs, ys_mat = _interpolate_to_common_steps(runs)
            auc     = np.trapz(ys_mat.mean(axis=0), xs)
            final_r = ys_mat[:, -1].mean()

            labels.append(label)
            aucs.append(auc)
            final_rewards.append(final_r)
            colors.append(color)

        x = np.arange(len(labels))
        w = 0.6

        # AUC bars
        bars = axes[0].bar(x, aucs, width=w, color=colors, edgecolor="white", linewidth=1.2)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels)
        axes[0].set_ylabel("Area Under Curve")
        axes[0].set_title("AUC (higher = more sample efficient)")
        axes[0].grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, aucs):
            axes[0].text(bar.get_x() + bar.get_width() / 2, v,
                         f"{v:.0f}", ha="center", va="bottom", fontsize=9)

        # Final reward bars
        bars2 = axes[1].bar(x, final_rewards, width=w, color=colors, edgecolor="white", linewidth=1.2)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels)
        axes[1].set_ylabel("Final Eval Reward")
        axes[1].set_title("Final Performance")
        axes[1].grid(axis="y", alpha=0.3)
        for bar, v in zip(bars2, final_rewards):
            axes[1].text(bar.get_x() + bar.get_width() / 2, v,
                         f"{v:.1f}", ha="center", va="bottom", fontsize=9)

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        if show: plt.show()
        plt.close(fig)
        print(f"Saved sample efficiency plot → {save_path}")

    def plot_hyperparameter_sensitivity(
        self,
        param_name: str,
        param_values: List[Any],
        run_dir_groups: List[List[str]],
        save_path: str,
        show: bool = False,
    ) -> None:
        """
        Show how final performance varies with a single hyperparameter.

        Args:
            param_name:      Name of the hyperparameter (for axis label)
            param_values:    List of hyperparameter values tested
            run_dir_groups:  List of run_dir lists, one per param value
        """
        fig, ax = plt.subplots(figsize=(8, 4))
        final_means, final_stds = [], []

        for run_dirs in run_dir_groups:
            runs = self._load_group(run_dirs)
            if not runs:
                final_means.append(np.nan)
                final_stds.append(np.nan)
                continue
            finals = np.array([r[1][-1] for r in runs])
            final_means.append(finals.mean())
            final_stds.append(finals.std() if len(finals) > 1 else 0.0)

        ax.errorbar(
            range(len(param_values)),
            final_means,
            yerr=final_stds,
            fmt="o-", color="#4C72B0", capsize=5, linewidth=2, markersize=8,
        )
        ax.set_xticks(range(len(param_values)))
        ax.set_xticklabels([str(v) for v in param_values])
        ax.set_xlabel(param_name)
        ax.set_ylabel("Final Eval Reward")
        ax.set_title(f"Hyperparameter Sensitivity: {param_name}")
        ax.grid(True, alpha=0.3)

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        if show: plt.show()
        plt.close(fig)
        print(f"Saved sensitivity plot → {save_path}")

    def print_table(self) -> None:
        """Print a formatted comparison table."""
        print(f"\n{'Algorithm':<12}  {'Runs':>5}  {'Final Rew':>10}  {'±':>8}")
        print("─" * 42)
        for label, run_dirs in self._groups.items():
            runs = self._load_group(run_dirs)
            if not runs:
                print(f"{label:<12}  {'N/A':>5}")
                continue
            finals = np.array([r[1][-1] for r in runs])
            print(
                f"{label:<12}  {len(runs):>5}  "
                f"{finals.mean():>10.2f}  {finals.std():>8.2f}"
            )
        print()
