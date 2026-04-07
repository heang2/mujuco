"""
TensorBoard integration — optional logging to TensorBoard.

Falls back gracefully to no-op if tensorboard is not installed.

Usage:
    writer = TensorBoardWriter("logs/MyRun")
    writer.add_scalar("train/reward", value=42.0, step=1000)
    writer.add_histogram("actor/weights", tensor, step=1000)
    writer.close()

Install: pip install tensorboard
View:    tensorboard --logdir logs/
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TB = True
except ImportError:
    HAS_TB = False


class TensorBoardWriter:
    """
    Thin wrapper around PyTorch's SummaryWriter with graceful fallback.

    If tensorboard is not installed, all calls are silently ignored.
    This keeps the rest of the codebase clean — no try/except everywhere.
    """

    def __init__(self, log_dir: str, comment: str = ""):
        self.log_dir  = Path(log_dir)
        self._enabled = HAS_TB

        if HAS_TB:
            self._writer = SummaryWriter(str(log_dir), comment=comment)
            print(f"TensorBoard writer → {log_dir}")
            print(f"  View with: tensorboard --logdir {Path(log_dir).parent}")
        else:
            self._writer = None
            # Silent: no warning spam

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        if self._writer:
            self._writer.add_scalar(tag, value, step)

    def add_scalars(self, main_tag: str, values: Dict[str, float], step: int) -> None:
        if self._writer:
            self._writer.add_scalars(main_tag, values, step)

    def add_histogram(self, tag: str, values, step: int) -> None:
        if self._writer:
            try:
                import torch
                if not isinstance(values, torch.Tensor):
                    import torch as _torch
                    values = _torch.as_tensor(values)
                self._writer.add_histogram(tag, values, step)
            except Exception:
                pass

    def add_image(self, tag: str, img: np.ndarray, step: int) -> None:
        """img: HWC uint8 or CHW float32"""
        if self._writer:
            if img.ndim == 3 and img.shape[-1] in (1, 3, 4):
                img = img.transpose(2, 0, 1)   # → CHW
            self._writer.add_image(tag, img, step)

    def add_hparams(self, hparam_dict: Dict, metric_dict: Dict) -> None:
        if self._writer:
            self._writer.add_hparams(hparam_dict, metric_dict)

    def add_text(self, tag: str, text: str, step: int) -> None:
        if self._writer:
            self._writer.add_text(tag, text, step)

    def add_video(self, tag: str, frames: np.ndarray, step: int, fps: int = 30) -> None:
        """frames: (T, H, W, C) uint8"""
        if self._writer:
            try:
                import torch
                t = torch.as_tensor(frames).permute(0, 3, 1, 2).unsqueeze(0)  # NTCHW
                self._writer.add_video(tag, t, step, fps=fps)
            except Exception:
                pass

    def flush(self) -> None:
        if self._writer:
            self._writer.flush()

    def close(self) -> None:
        if self._writer:
            self._writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def enabled(self) -> bool:
        return self._enabled


# ──────────────────────────────────────────────────────────────────────────────
# Convenience: log all standard training metrics at once
# ──────────────────────────────────────────────────────────────────────────────

class TrainingMonitor:
    """
    Combines TensorBoardWriter with CSV Logger for comprehensive tracking.

    Provides a single `.log()` call that routes to both backends.
    """

    def __init__(self, log_dir: str, run_name: str = ""):
        from utils.logger import Logger
        self.log_dir  = Path(log_dir)
        self.tb       = TensorBoardWriter(str(self.log_dir), comment=run_name)
        self.csv      = Logger(str(self.log_dir))
        self._step    = 0

    def log_train(self, step: int, info: Dict[str, float]) -> None:
        """Log training step metrics."""
        self._step = step
        for key, val in info.items():
            self.tb.add_scalar(f"train/{key}", val, step)
        self.csv.log_losses(step, info)

    def log_episode(self, step: int, reward: float, length: int) -> None:
        """Log episode completion."""
        self.tb.add_scalar("rollout/ep_reward", reward, step)
        self.tb.add_scalar("rollout/ep_length", length, step)
        self.csv.log_episode(step, reward, length)

    def log_eval(self, step: int, result: Dict[str, Any]) -> None:
        """Log evaluation results."""
        self.tb.add_scalar("eval/mean_reward", result["mean_reward"], step)
        self.tb.add_scalar("eval/std_reward",  result["std_reward"],  step)
        if result.get("success_rate") is not None:
            self.tb.add_scalar("eval/success_rate", result["success_rate"], step)
        self.csv.log_eval(result, step)

    def close(self) -> None:
        self.tb.close()
        self.csv.close()
