"""
Hyperparameter sweep — grid search and random search.

Runs multiple training experiments with different hyperparameter values
and collects final performance metrics.

Usage:
    from utils.sweep import GridSweep, RandomSweep

    sweep = GridSweep(
        base_config="configs/hopper_ppo.yaml",
        search_space={
            "ppo.lr":        [1e-4, 3e-4, 1e-3],
            "ppo.clip_eps":  [0.1, 0.2, 0.3],
            "ppo.n_epochs":  [5, 10],
        },
        n_seeds=3,
    )
    results = sweep.run(max_parallel=1)
    sweep.plot_results("sweep_results.png")
"""

import copy
import itertools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml


# ──────────────────────────────────────────────────────────────────────────────
# Config utilities
# ──────────────────────────────────────────────────────────────────────────────

def _set_nested(d: Dict, key_path: str, value: Any) -> None:
    """
    Set a value in a nested dict using dot notation.
    e.g. _set_nested(cfg, "ppo.lr", 1e-4) → cfg["ppo"]["lr"] = 1e-4
    """
    keys = key_path.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def _get_nested(d: Dict, key_path: str) -> Any:
    keys = key_path.split(".")
    for k in keys:
        d = d[k]
    return d


def _load_config(path: str) -> Dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _save_config(config: Dict, path: str) -> None:
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


# ──────────────────────────────────────────────────────────────────────────────
# Sweep result
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SweepResult:
    params:        Dict[str, Any]
    seed:          int
    final_reward:  float
    std_reward:    float
    run_dir:       str
    elapsed_sec:   float
    success:       bool = True
    error:         Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Base sweep
# ──────────────────────────────────────────────────────────────────────────────

class BaseSweep:

    def __init__(
        self,
        base_config: str,
        search_space: Dict[str, List[Any]],
        n_seeds: int = 1,
        sweep_dir: str = "sweeps",
        eval_episodes: int = 10,
    ):
        self.base_config   = base_config
        self.search_space  = search_space
        self.n_seeds       = n_seeds
        self.sweep_dir     = Path(sweep_dir)
        self.eval_episodes = eval_episodes
        self.results:      List[SweepResult] = []
        self._sweep_name   = f"sweep_{int(time.time())}"

    def _make_configs(self) -> Generator[Tuple[Dict, Dict], None, None]:
        """Yield (params_dict, full_config_dict) for each combination."""
        raise NotImplementedError

    def run(self, verbose: bool = True) -> List[SweepResult]:
        """Execute all sweep configurations sequentially."""
        from training.trainer import Trainer

        configs_list = list(self._make_configs())
        total = len(configs_list) * self.n_seeds
        print(f"\nStarting sweep: {len(configs_list)} configs × {self.n_seeds} seeds = {total} runs")
        print(f"Sweep dir: {self.sweep_dir / self._sweep_name}\n")

        run_idx = 0
        for params, config in configs_list:
            for seed in range(self.n_seeds):
                run_idx += 1
                run_cfg  = copy.deepcopy(config)
                run_cfg["seed"]     = seed
                run_name = f"run_{run_idx:03d}_s{seed}"
                run_cfg["run_name"] = run_name
                run_cfg["log_dir"]  = str(self.sweep_dir / self._sweep_name)
                run_cfg["eval_episodes"] = self.eval_episodes

                if verbose:
                    param_str = ", ".join(f"{k}={v}" for k, v in params.items())
                    print(f"[{run_idx}/{total}] seed={seed}  {param_str}")

                t0 = time.time()
                try:
                    trainer = Trainer(run_cfg)
                    trainer.train()

                    # Extract final eval reward from logger
                    if trainer.logger.eval_history:
                        last = trainer.logger.eval_history[-1]
                        final_r = last["mean_reward"]
                        std_r   = last["std_reward"]
                    else:
                        final_r = std_r = float("nan")

                    result = SweepResult(
                        params=params, seed=seed,
                        final_reward=final_r, std_reward=std_r,
                        run_dir=str(trainer.log_dir),
                        elapsed_sec=time.time() - t0,
                    )
                except Exception as e:
                    result = SweepResult(
                        params=params, seed=seed,
                        final_reward=float("nan"), std_reward=float("nan"),
                        run_dir="", elapsed_sec=time.time() - t0,
                        success=False, error=str(e),
                    )
                    print(f"  ERROR: {e}")

                self.results.append(result)

        self._save_results()
        self._print_best()
        return self.results

    def _save_results(self) -> None:
        out = self.sweep_dir / self._sweep_name / "sweep_results.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {
                "params":       r.params,
                "seed":         r.seed,
                "final_reward": r.final_reward,
                "std_reward":   r.std_reward,
                "run_dir":      r.run_dir,
                "elapsed_sec":  r.elapsed_sec,
                "success":      r.success,
                "error":        r.error,
            }
            for r in self.results
        ]
        with open(out, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nResults saved → {out}")

    def _print_best(self) -> None:
        valid = [r for r in self.results if r.success and not np.isnan(r.final_reward)]
        if not valid:
            print("No valid results to summarise.")
            return
        best = max(valid, key=lambda r: r.final_reward)
        print(f"\n{'='*50}")
        print(f"  Best result: {best.final_reward:.2f} ± {best.std_reward:.2f}")
        print(f"  Params:      {best.params}")
        print(f"  Seed:        {best.seed}")
        print(f"{'='*50}\n")

    def plot_results(self, save_path: str, show: bool = False) -> None:
        """Scatter plot: each parameter vs. final reward."""
        valid = [r for r in self.results if r.success and not np.isnan(r.final_reward)]
        if not valid:
            print("No valid results to plot.")
            return

        params = list(self.search_space.keys())
        n_params = len(params)
        if n_params == 0:
            return

        n_cols = min(n_params, 3)
        n_rows = (n_params + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        fig.suptitle("Hyperparameter Sweep Results", fontsize=14, fontweight="bold")
        axes = np.array(axes).flatten() if n_params > 1 else [axes]

        rewards = np.array([r.final_reward for r in valid])
        r_min, r_max = rewards.min(), rewards.max()

        for i, param in enumerate(params):
            ax = axes[i]
            xs = [r.params.get(param, np.nan) for r in valid]
            ys = [r.final_reward for r in valid]

            sc = ax.scatter(xs, ys, c=ys, cmap="viridis", vmin=r_min, vmax=r_max,
                            s=60, edgecolors="white", linewidths=0.5, zorder=3)
            fig.colorbar(sc, ax=ax, label="Reward")
            ax.set_xlabel(param, fontsize=9)
            ax.set_ylabel("Final Reward", fontsize=9)
            ax.set_title(param, fontsize=10)
            ax.grid(True, alpha=0.3)

        for j in range(n_params, len(axes)):
            axes[j].set_visible(False)

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        if show: plt.show()
        plt.close(fig)
        print(f"Saved sweep results → {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Grid search
# ──────────────────────────────────────────────────────────────────────────────

class GridSweep(BaseSweep):
    """
    Exhaustive grid search over all parameter combinations.
    Total runs = product(len(v) for v in search_space.values()) × n_seeds
    """

    def _make_configs(self) -> Generator[Tuple[Dict, Dict], None, None]:
        base = _load_config(self.base_config)
        keys  = list(self.search_space.keys())
        vals  = list(self.search_space.values())
        for combo in itertools.product(*vals):
            params = dict(zip(keys, combo))
            cfg    = copy.deepcopy(base)
            for k, v in params.items():
                _set_nested(cfg, k, v)
            yield params, cfg


# ──────────────────────────────────────────────────────────────────────────────
# Random search
# ──────────────────────────────────────────────────────────────────────────────

class RandomSweep(BaseSweep):
    """
    Random search — sample n_trials random configurations.

    search_space values should be callables (samplers) or lists:
      - List  → uniform sample from list
      - Callable → call it to get a value

    Example:
        RandomSweep(
            base_config="configs/hopper_ppo.yaml",
            search_space={
                "ppo.lr":      lambda: 10 ** np.random.uniform(-4, -3),
                "ppo.clip_eps": [0.1, 0.15, 0.2, 0.25, 0.3],
            },
            n_trials=20,
            n_seeds=2,
        )
    """

    def __init__(self, *args, n_trials: int = 20, rng_seed: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_trials = n_trials
        self._rng     = np.random.default_rng(rng_seed)

    def _sample_params(self) -> Dict[str, Any]:
        params = {}
        for key, sampler in self.search_space.items():
            if callable(sampler):
                params[key] = sampler()
            elif isinstance(sampler, list):
                params[key] = self._rng.choice(sampler)
            else:
                params[key] = sampler
        return params

    def _make_configs(self) -> Generator[Tuple[Dict, Dict], None, None]:
        base = _load_config(self.base_config)
        for _ in range(self.n_trials):
            params = self._sample_params()
            cfg    = copy.deepcopy(base)
            for k, v in params.items():
                _set_nested(cfg, k, float(v) if isinstance(v, np.floating) else v)
            yield params, cfg
