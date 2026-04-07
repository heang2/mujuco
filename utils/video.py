"""
Video recording with annotated overlays.

Records policy rollouts as MP4 or GIF, with optional overlays:
  - Episode step counter
  - Current reward and cumulative reward
  - Action bar charts (what is the policy doing)
  - Info dict values (speed, height, etc.)

Requires: imageio[ffmpeg]  →  pip install imageio[ffmpeg]
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def _draw_text_cv2(
    frame: np.ndarray,
    lines: List[str],
    origin: Tuple[int, int] = (10, 20),
    font_scale: float = 0.45,
    color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Optional[Tuple] = (0, 0, 0),
    line_height: int = 18,
) -> np.ndarray:
    """Overlay text lines onto a BGR image using OpenCV."""
    img = frame.copy()
    x, y = origin
    for i, line in enumerate(lines):
        py = y + i * line_height
        if bg_color is not None:
            # Draw a semi-transparent background rectangle
            (tw, th), _ = cv2.getTextSize(
                line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            cv2.rectangle(img, (x - 2, py - th - 2), (x + tw + 2, py + 2),
                          bg_color, -1)
        cv2.putText(
            img, line,
            (x, py),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, color, 1, cv2.LINE_AA,
        )
    return img


def _draw_action_bars(
    frame: np.ndarray,
    actions: np.ndarray,
    bar_width: int = 12,
    max_bar_height: int = 40,
) -> np.ndarray:
    """Draw vertical bars at the bottom showing action magnitudes."""
    img    = frame.copy()
    h, w   = img.shape[:2]
    n_acts = len(actions)
    margin = 10
    total_w = n_acts * (bar_width + 2) + 2 * margin
    start_x = w // 2 - total_w // 2

    for i, a in enumerate(actions):
        x0  = start_x + margin + i * (bar_width + 2)
        bar_h = int(abs(a) * max_bar_height)
        bar_h = max(1, min(bar_h, max_bar_height))
        color = (80, 200, 80) if a >= 0 else (80, 80, 200)   # BGR
        y1 = h - 5
        y0 = y1 - bar_h
        # Background
        cv2.rectangle(img, (x0, h - max_bar_height - 5), (x0 + bar_width, h - 5),
                      (40, 40, 40), -1)
        # Bar
        cv2.rectangle(img, (x0, y0), (x0 + bar_width, y1), color, -1)

    return img


class VideoRecorder:
    """
    Record a trained policy to video with optional annotations.

    Usage:
        recorder = VideoRecorder(env, agent, fps=50)
        recorder.record(n_episodes=1, save_path="demo.mp4", annotate=True)
    """

    def __init__(
        self,
        env,
        agent,
        fps: int = 50,
        width: int = 480,
        height: int = 480,
        annotate: bool = True,
    ):
        if not HAS_IMAGEIO:
            raise ImportError(
                "imageio not installed. Run: pip install imageio[ffmpeg]"
            )
        self.env      = env
        self.agent    = agent
        self.fps      = fps
        self.width    = width
        self.height   = height
        self.annotate = annotate and HAS_CV2

    def record(
        self,
        n_episodes: int = 1,
        save_path: str = "recording.mp4",
        deterministic: bool = True,
        seed: int = 0,
        max_steps: int = 2000,
    ) -> Dict[str, Any]:
        """
        Record n_episodes and save to video.

        Returns dict of episode statistics.
        """
        from envs import make_env
        rec_env = make_env(
            type(self.env).__name__.replace("Env", ""),
            render_mode="rgb_array",
        )

        all_frames = []
        episode_rewards = []
        episode_lengths = []

        for ep in range(n_episodes):
            obs, _ = rec_env.reset(seed=seed + ep)
            ep_reward = 0.0
            ep_step   = 0
            terminated = truncated = False
            ep_frames  = []

            while not (terminated or truncated) and ep_step < max_steps:
                action = self.agent.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = rec_env.step(action)

                ep_reward += reward
                ep_step   += 1

                # Render frame
                frame = rec_env.render()
                if frame is None:
                    continue

                # BGR for OpenCV
                frame_bgr = frame[:, :, ::-1].copy()

                if self.annotate and HAS_CV2:
                    lines = [
                        f"Ep {ep+1}/{n_episodes}  Step {ep_step}",
                        f"Rew  {reward:+.3f}  (total {ep_reward:+.1f})",
                    ]
                    for k, v in list(info.items())[:3]:
                        if isinstance(v, (int, float)):
                            lines.append(f"{k}: {v:.3f}")

                    frame_bgr = _draw_text_cv2(frame_bgr, lines)
                    if len(action) <= 12:
                        frame_bgr = _draw_action_bars(frame_bgr, action)

                # Back to RGB
                ep_frames.append(frame_bgr[:, :, ::-1])

            all_frames.extend(ep_frames)
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_step)

        rec_env.close()

        # Write video
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        ext = save_path_obj.suffix.lower()

        if ext == ".gif":
            imageio.mimsave(str(save_path), all_frames, fps=self.fps, loop=0)
        else:
            writer = imageio.get_writer(str(save_path), fps=self.fps, codec="libx264",
                                        quality=8)
            for f in all_frames:
                writer.append_data(f)
            writer.close()

        stats = {
            "n_episodes":   n_episodes,
            "n_frames":     len(all_frames),
            "mean_reward":  float(np.mean(episode_rewards)),
            "mean_length":  float(np.mean(episode_lengths)),
            "save_path":    str(save_path),
        }
        print(
            f"Saved {n_episodes}-episode video → {save_path}  "
            f"({len(all_frames)} frames, "
            f"mean_rew={stats['mean_reward']:.1f})"
        )
        return stats


def frames_to_gif(
    frames: List[np.ndarray],
    save_path: str,
    fps: int = 30,
    scale: float = 1.0,
) -> None:
    """Convert a list of RGB frames to a GIF or MP4."""
    if not HAS_IMAGEIO:
        raise ImportError("imageio required: pip install imageio[ffmpeg]")

    if scale != 1.0 and HAS_CV2:
        scaled = []
        for f in frames:
            h, w = int(f.shape[0] * scale), int(f.shape[1] * scale)
            scaled.append(cv2.resize(f, (w, h)))
        frames = scaled

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(save_path, frames, fps=fps)
    print(f"Saved {len(frames)}-frame video → {save_path}")
