"""
DreamerV3 World Model.

Components:
  - Encoder:      obs → embed
  - RSSM:         (h_{t-1}, z_{t-1}, embed_t) → (h_t, z_t)  [Recurrent State Space Model]
  - Decoder:      (h_t, z_t) → obs_hat
  - Reward head:  (h_t, z_t) → reward_hat
  - Continue head:(h_t, z_t) → gamma (discount)

The RSSM factorizes the latent state as:
    s_t = (h_t, z_t)
where
    h_t = f(h_{t-1}, z_{t-1}, a_{t-1})   [deterministic GRU]
    z_t ~ q(z_t | h_t, x_t)              [posterior, given observation]
    z_t ~ p(z_t | h_t)                    [prior, without observation]

DreamerV3 uses straight-through categorical distributions for z_t.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from agents.dreamer.config import DreamerConfig


# ── Helpers ──────────────────────────────────────────────────────────────────

def mlp(dims: List[int], act: nn.Module = nn.SiLU, norm: bool = True) -> nn.Sequential:
    """Build a MLP with optional LayerNorm."""
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if norm and i < len(dims) - 2:
            layers.append(nn.LayerNorm(dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(act())
    return nn.Sequential(*layers)


class StraightThroughCategorical(nn.Module):
    """
    Straight-through estimator for discrete categorical latents.

    DreamerV3 uses a 32×32 one-hot categorical (stoch_dim categories,
    stoch_cats classes each) → 1024-d embedding after flattening.
    """

    def __init__(self, in_dim: int, stoch_dim: int, stoch_cats: int):
        super().__init__()
        self.stoch_dim  = stoch_dim
        self.stoch_cats = stoch_cats
        self.logit_head = nn.Linear(in_dim, stoch_dim * stoch_cats)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            z_hard: straight-through one-hot  (B, stoch_dim * stoch_cats)
            kl_loss: KL( q || p=Uniform ) — per element, mean over batch
        """
        logits = self.logit_head(x)                              # (B, D*C)
        logits = logits.view(-1, self.stoch_dim, self.stoch_cats)  # (B, D, C)
        # Straight-through via gumbel-softmax with τ→0
        z_soft = F.gumbel_softmax(logits, tau=1.0, hard=False)   # (B, D, C)
        z_hard = F.gumbel_softmax(logits, tau=1.0, hard=True)    # (B, D, C)
        z_st   = z_soft + (z_hard - z_soft).detach()             # straight-through
        z_flat = z_st.view(-1, self.stoch_dim * self.stoch_cats) # (B, D*C)

        # KL against uniform prior
        log_q  = F.log_softmax(logits, dim=-1)                   # (B, D, C)
        log_p  = torch.full_like(log_q, -math.log(self.stoch_cats))
        kl     = (torch.exp(log_q) * (log_q - log_p)).sum(-1).mean()

        return z_flat, kl


# ── RSSM ─────────────────────────────────────────────────────────────────────

class RSSM(nn.Module):
    """
    Recurrent State Space Model.

    State: (h, z) where h = GRU hidden, z = straight-through categorical.
    """

    def __init__(self, cfg: DreamerConfig, obs_dim: int, act_dim: int):
        super().__init__()
        self.deter_dim  = cfg.deter_dim
        self.stoch_dim  = cfg.stoch_dim
        self.stoch_cats = cfg.stoch_cats
        self.z_dim      = cfg.stoch_dim * cfg.stoch_cats   # flat z size

        # Recurrent core
        self.gru_input_proj = nn.Linear(self.z_dim + act_dim, cfg.deter_dim)
        self.gru_cell       = nn.GRUCell(cfg.deter_dim, cfg.deter_dim)

        # Prior p(z | h)
        self.prior_head = StraightThroughCategorical(
            cfg.deter_dim, cfg.stoch_dim, cfg.stoch_cats
        )

        # Posterior q(z | h, embed)
        self.post_proj  = nn.Linear(cfg.deter_dim + cfg.embed_dim, cfg.deter_dim)
        self.post_head  = StraightThroughCategorical(
            cfg.deter_dim, cfg.stoch_dim, cfg.stoch_cats
        )

    def initial_state(self, batch: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(batch, self.deter_dim, device=device)
        z = torch.zeros(batch, self.z_dim,     device=device)
        return h, z

    def observe(
        self,
        embed: torch.Tensor,   # (B, embed_dim)
        action: torch.Tensor,  # (B, act_dim)
        h: torch.Tensor,       # (B, deter_dim)
        z: torch.Tensor,       # (B, z_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One step: update h, then sample posterior z."""
        h = self._recurrent(action, z, h)
        post_in = torch.cat([h, embed], dim=-1)
        z, kl   = self.post_head(self.post_proj(post_in))
        return h, z, kl

    def imagine(
        self,
        action: torch.Tensor,  # (B, act_dim)
        h: torch.Tensor,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One step: update h, sample prior z (no observation)."""
        h        = self._recurrent(action, z, h)
        z, kl    = self.prior_head(h)
        return h, z, kl

    def _recurrent(self, action: torch.Tensor, z: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([z, action], dim=-1)
        inp = F.silu(self.gru_input_proj(inp))
        return self.gru_cell(inp, h)

    @property
    def state_dim(self) -> int:
        return self.deter_dim + self.z_dim


# ── Encoder ───────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """MLP encoder: obs → embed."""

    def __init__(self, obs_dim: int, cfg: DreamerConfig):
        super().__init__()
        dims = [obs_dim] + cfg.enc_hidden + [cfg.embed_dim]
        self.net = mlp(dims)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ── Decoder ───────────────────────────────────────────────────────────────────

class Decoder(nn.Module):
    """MLP decoder: (h, z) → obs_hat. Uses MSE reconstruction."""

    def __init__(self, obs_dim: int, cfg: DreamerConfig):
        super().__init__()
        in_dim = cfg.deter_dim + cfg.stoch_dim * cfg.stoch_cats
        dims   = [in_dim] + cfg.dec_hidden + [obs_dim]
        self.net = mlp(dims)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)


# ── Reward / Continue heads ───────────────────────────────────────────────────

class RewardHead(nn.Module):
    """Predict reward from latent state."""

    def __init__(self, cfg: DreamerConfig):
        super().__init__()
        in_dim = cfg.deter_dim + cfg.stoch_dim * cfg.stoch_cats
        dims   = [in_dim] + cfg.reward_hidden + [1]
        self.net = mlp(dims)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat).squeeze(-1)   # (B,)


class ContinueHead(nn.Module):
    """Predict discount γ (binary: done → 0, alive → 1)."""

    def __init__(self, cfg: DreamerConfig):
        super().__init__()
        in_dim = cfg.deter_dim + cfg.stoch_dim * cfg.stoch_cats
        dims   = [in_dim] + cfg.reward_hidden + [1]
        self.net = mlp(dims)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(feat)).squeeze(-1)   # (B,) in [0,1]


# ── World Model ───────────────────────────────────────────────────────────────

class WorldModel(nn.Module):
    """
    Full DreamerV3 world model.

    Wraps Encoder, RSSM, Decoder, RewardHead, ContinueHead.
    """

    def __init__(self, obs_dim: int, act_dim: int, cfg: DreamerConfig):
        super().__init__()
        self.cfg      = cfg
        self.rssm     = RSSM(cfg, obs_dim, act_dim)
        self.encoder  = Encoder(obs_dim, cfg)
        self.decoder  = Decoder(obs_dim, cfg)
        self.reward   = RewardHead(cfg)
        self.cont     = ContinueHead(cfg)

    @property
    def state_dim(self) -> int:
        return self.rssm.state_dim

    def feat(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Concatenate deterministic and stochastic parts."""
        return torch.cat([h, z], dim=-1)

    def observe_sequence(
        self,
        obs:    torch.Tensor,   # (T, B, obs_dim)
        actions: torch.Tensor,  # (T, B, act_dim)
    ) -> Dict[str, torch.Tensor]:
        """
        Encode a sequence and compute world-model losses.

        Returns a dict with:
          feats     — (T, B, state_dim) latent features
          kl        — scalar KL loss
          rec_loss  — scalar reconstruction loss
          rew_loss  — scalar reward prediction loss
          cont_loss — scalar continue prediction loss
        """
        T, B, _ = obs.shape
        device   = obs.device
        h, z     = self.rssm.initial_state(B, device)

        feats    = []
        kl_total = torch.zeros(1, device=device)

        for t in range(T):
            embed       = self.encoder(obs[t])
            h, z, kl    = self.rssm.observe(embed, actions[t], h, z)
            feats.append(torch.cat([h, z], dim=-1))
            kl_total   = kl_total + kl

        feats = torch.stack(feats, dim=0)   # (T, B, state_dim)
        return {
            "feats": feats,
            "kl":    kl_total / T,
        }

    def imagine_rollout(
        self,
        initial_feat: torch.Tensor,   # (B, state_dim)
        actor_fn,                      # callable: feat → action
        horizon: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Roll out H steps in imagination using the prior.

        Returns:
          feats    — (H, B, state_dim)
          rewards  — (H, B)
          conts    — (H, B)
          actions  — (H, B, act_dim)
        """
        B      = initial_feat.shape[0]
        device = initial_feat.device
        h = initial_feat[:, :self.rssm.deter_dim]
        z = initial_feat[:, self.rssm.deter_dim:]

        feats   = []
        rewards = []
        conts   = []
        actions = []

        for _ in range(horizon):
            feat   = self.feat(h, z)
            action = actor_fn(feat)
            h, z, _ = self.rssm.imagine(action, h, z)
            next_feat = self.feat(h, z)

            feats.append(next_feat)
            rewards.append(self.reward(next_feat))
            conts.append(self.cont(next_feat))
            actions.append(action)

        return {
            "feats":   torch.stack(feats,   dim=0),
            "rewards": torch.stack(rewards, dim=0),
            "conts":   torch.stack(conts,   dim=0),
            "actions": torch.stack(actions, dim=0),
        }
