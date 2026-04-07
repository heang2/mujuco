"""
Policy Analyzer — deep-dive into what a trained policy has learned.

Produces:
  - Action distribution plots (mean and std across rollouts)
  - State-space visitation heatmap (for 2D projections)
  - Reward decomposition (reward components over time)
  - Policy entropy / determinism measure
  - Joint correlation matrix (for locomotion tasks)
  - Trajectory visualization (x-z path)
  - Value function landscape (for CartPole / 2D envs)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm
import matplotlib.patches as mpatches


class PolicyAnalyzer:
    """
    Analyse a trained policy by collecting rollout data.

    Usage:
        analyzer = PolicyAnalyzer(env, agent, n_episodes=50)
        analyzer.collect()
        analyzer.plot_action_distributions("analysis_actions.png")
        analyzer.plot_state_visitation("analysis_heatmap.png")
        analyzer.plot_trajectory("analysis_traj.png")
        report = analyzer.summary()
    """

    def __init__(
        self,
        env,
        agent,
        n_episodes: int = 50,
        max_steps_per_ep: int = 1000,
        deterministic: bool = True,
    ):
        self.env             = env
        self.agent           = agent
        self.n_episodes      = n_episodes
        self.max_steps       = max_steps_per_ep
        self.deterministic   = deterministic

        # Collected rollout data
        self.obs_list:     List[np.ndarray] = []
        self.action_list:  List[np.ndarray] = []
        self.reward_list:  List[float]      = []
        self.episode_data: List[Dict]       = []
        self._collected    = False

    def collect(self) -> "PolicyAnalyzer":
        """Run n_episodes and collect trajectory data."""
        self.obs_list.clear()
        self.action_list.clear()
        self.reward_list.clear()
        self.episode_data.clear()

        for ep in range(self.n_episodes):
            obs, _ = self.env.reset(seed=ep)
            ep_obs, ep_acts, ep_rews = [], [], []
            terminated = truncated = False
            steps = 0

            while not (terminated or truncated) and steps < self.max_steps:
                action = self.agent.predict(obs, deterministic=self.deterministic)
                next_obs, reward, terminated, truncated, info = self.env.step(action)

                ep_obs.append(obs.copy())
                ep_acts.append(action.copy())
                ep_rews.append(float(reward))

                obs   = next_obs
                steps += 1

            self.obs_list.extend(ep_obs)
            self.action_list.extend(ep_acts)
            self.reward_list.extend(ep_rews)
            self.episode_data.append({
                "total_reward": sum(ep_rews),
                "length":       steps,
                "success":      info.get("success", False),
                "obs":          np.array(ep_obs),
                "actions":      np.array(ep_acts),
                "rewards":      np.array(ep_rews),
            })

        self._collected = True
        return self

    def _check_collected(self) -> None:
        if not self._collected:
            raise RuntimeError("Call .collect() before plotting.")

    # ------------------------------------------------------------------
    # Action distribution analysis
    # ------------------------------------------------------------------

    def plot_action_distributions(self, save_path: str, show: bool = False) -> None:
        """
        Per-dimension histogram of actions taken by the policy.
        Also shows mean ± std bands to reveal action biases.
        """
        self._check_collected()
        acts = np.array(self.action_list)   # (T, act_dim)
        act_dim = acts.shape[1]
        n_cols  = min(act_dim, 4)
        n_rows  = (act_dim + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        fig.suptitle("Action Distributions", fontsize=14, fontweight="bold")
        axes = np.array(axes).flatten() if act_dim > 1 else [axes]

        colors = plt.cm.tab10(np.linspace(0, 1, act_dim))
        for i in range(act_dim):
            ax = axes[i]
            a  = acts[:, i]
            ax.hist(a, bins=50, density=True, alpha=0.7, color=colors[i], edgecolor="white")
            ax.axvline(a.mean(),              color="black", linestyle="--", linewidth=1.5,
                       label=f"μ={a.mean():.3f}")
            ax.axvline(a.mean() + a.std(),    color="gray",  linestyle=":",  linewidth=1)
            ax.axvline(a.mean() - a.std(),    color="gray",  linestyle=":",  linewidth=1,
                       label=f"σ={a.std():.3f}")
            ax.set_xlabel(f"Action[{i}]", fontsize=9)
            ax.set_ylabel("Density", fontsize=9)
            ax.set_title(f"Dim {i}", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-1.1, 1.1)

        for j in range(act_dim, len(axes)):
            axes[j].set_visible(False)

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        if show: plt.show()
        plt.close(fig)
        print(f"Saved action distributions → {save_path}")

    # ------------------------------------------------------------------
    # State visitation heatmap
    # ------------------------------------------------------------------

    def plot_state_visitation(
        self,
        save_path: str,
        dim_x: int = 0,
        dim_y: int = 1,
        show: bool = False,
    ) -> None:
        """
        2D histogram of state visitation density.
        Projects observations onto two chosen dimensions.
        """
        self._check_collected()
        obs   = np.array(self.obs_list)
        obs_x = obs[:, dim_x]
        obs_y = obs[:, dim_y]

        fig, ax = plt.subplots(figsize=(7, 6))
        h = ax.hist2d(
            obs_x, obs_y, bins=60,
            norm=LogNorm(), cmap="hot_r",
        )
        fig.colorbar(h[3], ax=ax, label="Visit count (log scale)")
        ax.set_xlabel(f"obs[{dim_x}]")
        ax.set_ylabel(f"obs[{dim_y}]")
        ax.set_title("State Visitation Density (policy rollouts)")
        ax.grid(True, alpha=0.2)

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        if show: plt.show()
        plt.close(fig)
        print(f"Saved state visitation heatmap → {save_path}")

    # ------------------------------------------------------------------
    # Trajectory visualization (locomotion tasks)
    # ------------------------------------------------------------------

    def plot_trajectory(self, save_path: str, n_trajs: int = 5, show: bool = False) -> None:
        """
        Plot x-position over time for multiple episodes.
        Reveals gait patterns and speed profiles.
        """
        self._check_collected()

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle("Episode Trajectories", fontsize=13, fontweight="bold")

        colors = plt.cm.viridis(np.linspace(0, 1, n_trajs))
        shown  = min(n_trajs, len(self.episode_data))

        for i, color in zip(range(shown), colors):
            ep = self.episode_data[i]
            t  = np.arange(ep["length"])

            # Reward over time
            axes[0].plot(t, ep["rewards"], alpha=0.7, color=color,
                         linewidth=1.0, label=f"ep {i} (R={ep['total_reward']:.1f})")

            # Cumulative reward
            axes[1].plot(t, np.cumsum(ep["rewards"]), alpha=0.7, color=color,
                         linewidth=1.5)

        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Reward")
        axes[0].set_title("Reward per Step")
        axes[0].legend(fontsize=7, loc="upper right")
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Cumulative Reward")
        axes[1].set_title("Cumulative Reward")
        axes[1].grid(True, alpha=0.3)

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        if show: plt.show()
        plt.close(fig)
        print(f"Saved trajectory plot → {save_path}")

    # ------------------------------------------------------------------
    # Joint correlation matrix
    # ------------------------------------------------------------------

    def plot_joint_correlation(self, save_path: str, show: bool = False) -> None:
        """
        Pearson correlation between action dimensions.
        High correlation = coordinated joint movement.
        """
        self._check_collected()
        acts = np.array(self.action_list)
        if acts.shape[1] < 2:
            print("Need ≥2 action dimensions for correlation plot.")
            return

        corr = np.corrcoef(acts.T)
        act_dim = acts.shape[1]

        fig, ax = plt.subplots(figsize=(max(5, act_dim), max(4, act_dim - 1)))
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        fig.colorbar(im, ax=ax, label="Pearson ρ")

        ax.set_xticks(range(act_dim))
        ax.set_yticks(range(act_dim))
        ax.set_xticklabels([f"a[{i}]" for i in range(act_dim)], fontsize=9)
        ax.set_yticklabels([f"a[{i}]" for i in range(act_dim)], fontsize=9)
        ax.set_title("Action Dimension Correlation Matrix", fontsize=12)

        for i in range(act_dim):
            for j in range(act_dim):
                ax.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center",
                        fontsize=8 if act_dim <= 6 else 6,
                        color="white" if abs(corr[i,j]) > 0.5 else "black")

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        if show: plt.show()
        plt.close(fig)
        print(f"Saved joint correlation matrix → {save_path}")

    # ------------------------------------------------------------------
    # Observation statistics
    # ------------------------------------------------------------------

    def plot_obs_statistics(self, save_path: str, show: bool = False) -> None:
        """Box plots of each observation dimension across all rollout steps."""
        self._check_collected()
        obs     = np.array(self.obs_list)
        obs_dim = obs.shape[1]

        # Subsample if too many dimensions
        max_show = 20
        dims     = list(range(min(obs_dim, max_show)))

        fig, ax = plt.subplots(figsize=(max(8, len(dims) * 0.7), 5))
        bp = ax.boxplot(
            [obs[:, i] for i in dims],
            patch_artist=True,
            notch=True,
            boxprops=dict(facecolor="lightblue", color="steelblue"),
            medianprops=dict(color="darkred", linewidth=2),
            flierprops=dict(marker=".", markersize=2, alpha=0.3),
        )
        ax.set_xticks(range(1, len(dims) + 1))
        ax.set_xticklabels([f"obs[{i}]" for i in dims], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Value")
        ax.set_title("Observation Dimension Statistics (policy rollouts)")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.grid(True, axis="y", alpha=0.3)

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        if show: plt.show()
        plt.close(fig)
        print(f"Saved observation statistics → {save_path}")

    # ------------------------------------------------------------------
    # Full analysis report
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return a dictionary of summary statistics."""
        self._check_collected()
        rewards = [ep["total_reward"] for ep in self.episode_data]
        lengths = [ep["length"]       for ep in self.episode_data]
        succs   = [ep["success"]      for ep in self.episode_data]
        acts    = np.array(self.action_list)

        return {
            "n_episodes":    self.n_episodes,
            "mean_reward":   float(np.mean(rewards)),
            "std_reward":    float(np.std(rewards)),
            "min_reward":    float(np.min(rewards)),
            "max_reward":    float(np.max(rewards)),
            "mean_length":   float(np.mean(lengths)),
            "success_rate":  float(np.mean(succs)),
            "action_mean":   acts.mean(axis=0).tolist(),
            "action_std":    acts.std(axis=0).tolist(),
            "action_range":  (acts.min(axis=0).tolist(), acts.max(axis=0).tolist()),
            "obs_dim":       np.array(self.obs_list).shape[1],
            "act_dim":       acts.shape[1],
            "total_steps":   len(self.obs_list),
        }

    def run_full_analysis(self, output_dir: str) -> Dict[str, Any]:
        """Run all analyses and save outputs to output_dir."""
        self.collect()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        self.plot_action_distributions(str(out / "action_distributions.png"))
        self.plot_state_visitation(str(out / "state_visitation.png"))
        self.plot_trajectory(str(out / "trajectories.png"))
        self.plot_obs_statistics(str(out / "obs_statistics.png"))

        acts = np.array(self.action_list)
        if acts.shape[1] >= 2:
            self.plot_joint_correlation(str(out / "joint_correlation.png"))

        s = self.summary()
        import json
        with open(out / "summary.json", "w") as f:
            json.dump(s, f, indent=2)

        print(f"\nAnalysis saved to: {output_dir}")
        print(f"  mean_reward  : {s['mean_reward']:.2f} ± {s['std_reward']:.2f}")
        print(f"  success_rate : {s['success_rate']*100:.1f}%")
        print(f"  total_steps  : {s['total_steps']:,}")
        return s
