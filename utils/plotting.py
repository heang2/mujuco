"""
Plotting utilities for training diagnostics.

All functions save PNG files to disk and optionally display them.
"""

from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")           # headless backend — safe on servers
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ---- Color palette ----
COLORS = {
    "CartPole": "#4C72B0",
    "Reacher":  "#DD8452",
    "Hopper":   "#55A868",
    "Ant":      "#C44E52",
    "default":  "#8172B2",
}


def _smooth(values: List[float], window: int = 20) -> np.ndarray:
    """Simple moving average for reward curves."""
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def plot_training_curves(
    reward_history: List[Dict],
    eval_history:   List[Dict],
    env_name: str,
    save_path: str,
    show: bool = False,
) -> None:
    """
    Plot episode rewards + evaluation scores over training steps.

    Args:
        reward_history: List of dicts with keys "step", "reward", "length"
        eval_history:   List of dicts with keys "step", "mean_reward", etc.
        env_name:       Name of the environment (for title / colour)
        save_path:      Where to save the PNG
        show:           Whether to call plt.show()
    """
    color = COLORS.get(env_name, COLORS["default"])

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle(f"{env_name} — PPO Training", fontsize=16, fontweight="bold")

    # ---- 1. Episode rewards (raw + smoothed) ----
    ax1 = fig.add_subplot(gs[0, :])
    if reward_history:
        steps   = [r["step"]   for r in reward_history]
        rewards = [r["reward"] for r in reward_history]
        ax1.plot(steps, rewards, alpha=0.25, color=color, linewidth=0.6, label="episode reward")
        if len(rewards) >= 5:
            smoothed = _smooth(rewards, window=min(20, len(rewards) // 5 + 1))
            trim     = len(steps) - len(smoothed)
            ax1.plot(steps[trim:], smoothed, color=color, linewidth=2.0, label="smoothed (MA-20)")
    if eval_history:
        eval_steps  = [e["step"]        for e in eval_history]
        eval_means  = [e["mean_reward"] for e in eval_history]
        eval_stds   = [e["std_reward"]  for e in eval_history]
        ax1.plot(eval_steps, eval_means, "o--", color="black", linewidth=1.5,
                 markersize=5, label="eval mean", zorder=5)
        ax1.fill_between(
            eval_steps,
            np.array(eval_means) - np.array(eval_stds),
            np.array(eval_means) + np.array(eval_stds),
            alpha=0.15, color="black",
        )
    ax1.set_xlabel("Environment Steps")
    ax1.set_ylabel("Episode Reward")
    ax1.set_title("Reward over Training")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ---- 2. Episode length ----
    ax2 = fig.add_subplot(gs[1, 0])
    if reward_history:
        lengths  = [r["length"] for r in reward_history]
        ax2.plot(steps, lengths, alpha=0.3, color=color, linewidth=0.6)
        if len(lengths) >= 5:
            smoothed_len = _smooth(lengths, window=min(20, len(lengths) // 5 + 1))
            ax2.plot(steps[len(steps) - len(smoothed_len):], smoothed_len,
                     color=color, linewidth=1.5)
    ax2.set_xlabel("Environment Steps")
    ax2.set_ylabel("Episode Length")
    ax2.set_title("Episode Length over Training")
    ax2.grid(True, alpha=0.3)

    # ---- 3. Eval reward box-style ----
    ax3 = fig.add_subplot(gs[1, 1])
    if eval_history:
        ax3.errorbar(
            range(len(eval_history)),
            eval_means,
            yerr=eval_stds,
            fmt="o-", color=color, capsize=4, linewidth=1.5,
        )
        ax3.set_xlabel("Evaluation Number")
        ax3.set_ylabel("Mean Reward ± Std")
        ax3.set_title("Periodic Evaluation Results")
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "No evaluation data yet",
                 transform=ax3.transAxes, ha="center", va="center",
                 fontsize=11, color="grey")
        ax3.set_title("Periodic Evaluation Results")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved training curves → {save_path}")


def plot_multi_env_comparison(
    results: Dict[str, Dict],
    save_path: str,
    show: bool = False,
) -> None:
    """
    Bar chart comparing PPO vs Random across environments.

    Args:
        results: {env_name: {"ppo": mean_rew, "random": mean_rew}}
        save_path: output PNG path
    """
    envs    = list(results.keys())
    ppo_r   = [results[e]["ppo"]    for e in envs]
    rand_r  = [results[e]["random"] for e in envs]

    x     = np.arange(len(envs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width/2, ppo_r,  width, label="PPO",    color="#4C72B0", edgecolor="white")
    bars2 = ax.bar(x + width/2, rand_r, width, label="Random", color="#DD8452", edgecolor="white")

    ax.set_xlabel("Environment")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("PPO vs. Random Agent — Mean Reward per Environment")
    ax.set_xticks(x)
    ax.set_xticklabels(envs)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Value labels on bars
    for bar in [*bars1, *bars2]:
        h = bar.get_height()
        ax.annotate(
            f"{h:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=8,
        )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    print(f"Saved comparison chart → {save_path}")


def plot_value_distribution(
    values:    List[float],
    returns:   List[float],
    save_path: str,
    show:      bool = False,
) -> None:
    """
    Compare predicted V(s) versus actual discounted returns.
    Useful for diagnosing value function quality.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle("Value Function Diagnostics", fontsize=13, fontweight="bold")

    # Scatter: predicted vs actual
    axes[0].scatter(values, returns, alpha=0.3, s=5, color="#4C72B0")
    lo = min(min(values), min(returns))
    hi = max(max(values), max(returns))
    axes[0].plot([lo, hi], [lo, hi], "r--", linewidth=1, label="perfect")
    axes[0].set_xlabel("Predicted V(s)")
    axes[0].set_ylabel("Actual Return")
    axes[0].set_title("Predicted vs Actual Return")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Distribution of advantages
    advantages = np.array(returns) - np.array(values)
    axes[1].hist(advantages, bins=50, color="#55A868", edgecolor="white", alpha=0.8)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Advantage")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Advantage Distribution  (μ={np.mean(advantages):.3f})")
    axes[1].grid(True, alpha=0.3)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
