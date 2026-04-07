"""
Lightweight training logger.

Records episode rewards, evaluation metrics, and loss values
to CSV files for later analysis or plotting.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class Logger:
    """
    Logs training data to CSV + JSON.

    Files created in `log_dir`:
      - rewards.csv       — per-episode rewards during training
      - eval.csv          — periodic evaluation results
      - losses.csv        — per-update loss statistics
      - run_info.json     — metadata (start time, config hash, etc.)
    """

    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._reward_file = self._open_csv(
            "rewards.csv", ["step", "episode", "reward", "length", "timestamp"]
        )
        self._eval_file = self._open_csv(
            "eval.csv",
            ["step", "mean_reward", "std_reward", "min_reward", "max_reward",
             "mean_length", "success_rate", "timestamp"]
        )
        self._loss_file = self._open_csv(
            "losses.csv",
            ["step", "pg_loss", "vf_loss", "entropy", "approx_kl",
             "clip_fraction", "lr", "timestamp"]
        )

        # In-memory history for plotting
        self.reward_history: List[Dict] = []
        self.eval_history:   List[Dict] = []
        self.loss_history:   List[Dict] = []

        self._episode_count = 0

        # Write run metadata
        meta = {
            "start_time": datetime.now().isoformat(),
            "log_dir":    str(self.log_dir),
        }
        with open(self.log_dir / "run_info.json", "w") as f:
            json.dump(meta, f, indent=2)

    def log_episode(self, step: int, reward: float, length: int) -> None:
        self._episode_count += 1
        row = {
            "step":      step,
            "episode":   self._episode_count,
            "reward":    round(reward, 4),
            "length":    length,
            "timestamp": datetime.now().isoformat(),
        }
        self._reward_file.writerow(row)
        self._reward_fh.flush()
        self.reward_history.append(row)

    def log_eval(self, result: Dict[str, Any], step: int) -> None:
        row = {
            "step":         step,
            "mean_reward":  round(result["mean_reward"], 4),
            "std_reward":   round(result["std_reward"],  4),
            "min_reward":   round(result["min_reward"],  4),
            "max_reward":   round(result["max_reward"],  4),
            "mean_length":  round(result["mean_length"], 2),
            "success_rate": result.get("success_rate"),
            "timestamp":    datetime.now().isoformat(),
        }
        self._eval_file.writerow(row)
        self._eval_fh.flush()
        self.eval_history.append(row)

    def log_losses(self, step: int, info: Dict[str, float]) -> None:
        row = {
            "step":          step,
            "pg_loss":       round(info.get("pg_loss",       0), 6),
            "vf_loss":       round(info.get("vf_loss",       0), 6),
            "entropy":       round(info.get("entropy",       0), 6),
            "approx_kl":     round(info.get("approx_kl",     0), 6),
            "clip_fraction": round(info.get("clip_fraction", 0), 4),
            "lr":            info.get("lr", 0),
            "timestamp":     datetime.now().isoformat(),
        }
        self._loss_file.writerow(row)
        self._loss_fh.flush()
        self.loss_history.append(row)

    def close(self) -> None:
        for fh in [self._reward_fh, self._eval_fh, self._loss_fh]:
            fh.close()

    # ------------------------------------------------------------------

    def _open_csv(self, filename: str, fieldnames: List[str]):
        fh     = open(self.log_dir / filename, "w", newline="")
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        # stash handle so we can flush/close it
        setattr(self, f"_{filename.split('.')[0]}_fh", fh)
        return writer

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
