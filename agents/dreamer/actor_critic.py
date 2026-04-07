"""
DreamerV3 Actor and Critic operating in latent space.

Both networks operate on latent features (h, z) rather than raw observations.
The actor is trained by backpropagating through imagined trajectories.
The critic uses λ-returns with exponential moving average return normalization.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from agents.dreamer.config import DreamerConfig
from agents.dreamer.world_model import mlp


class DreamerActor(nn.Module):
    """
    Latent-space actor for continuous action spaces.

    Outputs a diagonal Gaussian action distribution.
    Actions are squashed with tanh to respect action bounds.
    """

    def __init__(self, state_dim: int, act_dim: int, cfg: DreamerConfig):
        super().__init__()
        self.act_dim = act_dim
        dims = [state_dim] + cfg.actor_hidden + [2 * act_dim]  # mean + log_std
        self.net = mlp(dims, norm=True)
        # Initialize final layer with small weights for stable early training
        nn.init.uniform_(self.net[-1].weight, -1e-4, 1e-4)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            action:   tanh-squashed action (B, act_dim)
            log_prob: log probability of action under the policy
        """
        out     = self.net(feat)
        mean, log_std = out.chunk(2, dim=-1)
        log_std = log_std.clamp(-10, 2)
        std     = log_std.exp()

        # Reparameterization trick
        eps    = torch.randn_like(mean)
        raw    = mean + std * eps
        action = torch.tanh(raw)

        # Log probability with tanh correction
        log_prob = (
            -0.5 * ((raw - mean) / std).pow(2)
            - log_std
            - 0.5 * math.log(2 * math.pi)
            - torch.log(1 - action.pow(2) + 1e-6)
        ).sum(-1)

        return action, log_prob

    def get_action(self, feat: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get an action (deterministic = use mean, stochastic = sample)."""
        out = self.net(feat)
        mean, log_std = out.chunk(2, dim=-1)
        if deterministic:
            return torch.tanh(mean)
        std  = log_std.clamp(-10, 2).exp()
        return torch.tanh(mean + std * torch.randn_like(mean))

    def entropy(self, feat: torch.Tensor) -> torch.Tensor:
        """Approximate differential entropy of the squashed Gaussian."""
        out = self.net(feat)
        _, log_std = out.chunk(2, dim=-1)
        log_std = log_std.clamp(-10, 2)
        # Gaussian entropy (ignoring tanh correction — approximate)
        return (log_std + 0.5 * math.log(2 * math.pi * math.e)).sum(-1)


class DreamerCritic(nn.Module):
    """
    Twin-critic value network operating in latent space.

    Uses two separate networks and takes the minimum for pessimism.
    Predicts λ-return targets.
    """

    def __init__(self, state_dim: int, cfg: DreamerConfig):
        super().__init__()
        dims = [state_dim] + cfg.critic_hidden + [1]
        self.v1 = mlp(dims, norm=True)
        self.v2 = mlp(dims, norm=True)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Return the minimum of both critics."""
        return torch.min(self.v1(feat), self.v2(feat)).squeeze(-1)

    def both(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return both critic values separately."""
        return self.v1(feat).squeeze(-1), self.v2(feat).squeeze(-1)


class ReturnNormalizer:
    """
    DreamerV3 percentile-based return normalizer.

    Maintains EMA of the 5th and 95th percentile of returns.
    Normalizes returns to have ~unit scale without clipping.
    """

    def __init__(self, decay: float = 0.99, low: float = 5.0, high: float = 95.0):
        self.decay = decay
        self.low   = low
        self.high  = high
        self._lo   = 0.0
        self._hi   = 1.0

    def update(self, returns: torch.Tensor) -> None:
        vals   = returns.detach().cpu()
        lo_new = torch.quantile(vals, self.low  / 100).item()
        hi_new = torch.quantile(vals, self.high / 100).item()
        self._lo = self.decay * self._lo + (1 - self.decay) * lo_new
        self._hi = self.decay * self._hi + (1 - self.decay) * hi_new

    def normalize(self, returns: torch.Tensor) -> torch.Tensor:
        scale = max(self._hi - self._lo, 1.0)
        return (returns - self._lo) / scale

    def state_dict(self):
        return {"lo": self._lo, "hi": self._hi}

    def load_state_dict(self, d):
        self._lo = d["lo"]
        self._hi = d["hi"]


def compute_lambda_returns(
    rewards:  torch.Tensor,   # (H, B)
    values:   torch.Tensor,   # (H+1, B) — includes bootstrap
    conts:    torch.Tensor,   # (H, B)  discount per step (≈ gamma * (1-done))
    gamma:    float = 0.997,
    lambda_:  float = 0.95,
) -> torch.Tensor:
    """
    Compute TD-λ returns for imagined trajectories.

    G_t = r_t + γ·c_t·[(1-λ)·V_{t+1} + λ·G_{t+1}]

    Returns:
        targets: (H, B)
    """
    H = rewards.shape[0]
    targets = torch.zeros_like(rewards)
    last    = values[-1]

    for t in reversed(range(H)):
        bootstrap  = (1 - lambda_) * values[t + 1] + lambda_ * last
        targets[t] = rewards[t] + gamma * conts[t] * bootstrap
        last       = targets[t]

    return targets
