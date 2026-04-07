"""
DreamerV3 hyper-parameter configuration.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class DreamerConfig:
    # ── General ───────────────────────────────────────────────────────────
    total_steps: int   = 1_000_000
    seed:        int   = 42
    log_dir:     str   = "logs"
    device:      str   = "cpu"           # "cpu" | "cuda" | "mps"

    # ── World model ───────────────────────────────────────────────────────
    # RSSM dimensions
    deter_dim:   int   = 512             # deterministic hidden state h_t
    stoch_dim:   int   = 32             # stochastic latent z categories
    stoch_cats:  int   = 32             # number of categories (discrete z)
    embed_dim:   int   = 1024            # encoder output dimension
    hidden_dim:  int   = 512            # hidden units in MLP components

    # Encoder / Decoder
    enc_hidden:  List[int] = field(default_factory=lambda: [512, 512])
    dec_hidden:  List[int] = field(default_factory=lambda: [512, 512])
    reward_hidden: List[int] = field(default_factory=lambda: [512, 512])

    # World-model learning
    wm_lr:       float = 1e-4
    wm_grad_clip: float = 1000.0
    kl_free:     float = 1.0            # free nats for KL balancing
    kl_balance:  float = 0.8            # KL balancing factor (DreamerV3 trick)

    # ── Replay buffer ─────────────────────────────────────────────────────
    buffer_capacity: int  = 1_000_000
    batch_size:      int  = 16          # episode sequences per batch
    seq_len:         int  = 64          # sequence length for world model
    prefill_steps:   int  = 2_500       # random steps before training

    # ── Actor-Critic (in imagination) ─────────────────────────────────────
    actor_hidden: List[int] = field(default_factory=lambda: [512, 512, 512, 512])
    critic_hidden: List[int] = field(default_factory=lambda: [512, 512, 512, 512])

    actor_lr:     float = 3e-5
    critic_lr:    float = 3e-5
    ac_grad_clip: float = 100.0

    # Imagined rollout
    imagine_horizon: int   = 15         # H-step imagined trajectory
    gamma:           float = 0.997
    lambda_:         float = 0.95      # λ-return

    # Return normalization (DreamerV3 percentile trick)
    return_ema_decay: float = 0.99

    # Entropy regularization
    actor_ent_scale: float = 3e-4

    # ── Training schedule ─────────────────────────────────────────────────
    train_ratio:  int  = 512            # gradient steps per env step
    eval_interval: int = 10_000
    eval_episodes: int = 5
    save_interval: int = 100_000

    # ── Logging ───────────────────────────────────────────────────────────
    log_interval: int  = 1_000
    video_interval: int = 50_000       # record video every N env steps
