"""
Neural network architectures for Actor-Critic RL.

- MLP feature extractor (shared or separate trunk)
- GaussianActor  — continuous action, diagonal Normal distribution
- CriticV        — state value estimator V(s)
- ActorCritic    — combined module used by PPO
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import List, Optional, Tuple


def _init_weights(module: nn.Module, gain: float = 1.0) -> nn.Module:
    """Orthogonal weight init (a la PPO paper)."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=gain)
            nn.init.constant_(m.bias, 0.0)
    return module


def build_mlp(
    input_dim: int,
    hidden_sizes: List[int],
    activation: type = nn.Tanh,
    output_dim: Optional[int] = None,
    output_gain: float = 1.0,
) -> nn.Sequential:
    """Build a fully-connected MLP."""
    layers: list = []
    in_dim = input_dim
    for h in hidden_sizes:
        layers += [nn.Linear(in_dim, h), activation()]
        in_dim = h
    if output_dim is not None:
        out_layer = nn.Linear(in_dim, output_dim)
        _init_weights(out_layer, gain=output_gain)
        layers.append(out_layer)
    net = nn.Sequential(*layers)
    _init_weights(net)
    return net


class GaussianActor(nn.Module):
    """
    Actor network that outputs a diagonal Gaussian distribution over actions.

    The log-std is a learned parameter (not state-dependent), clamped to
    [log_std_min, log_std_max] for stability.
    """

    LOG_STD_MIN = -4.0
    LOG_STD_MAX =  2.0

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: List[int] = (256, 256),
        activation: type = nn.Tanh,
    ):
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_sizes, activation)
        feat_dim = hidden_sizes[-1]

        self.mean_layer = nn.Linear(feat_dim, act_dim)
        _init_weights(self.mean_layer, gain=0.01)

        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor) -> Normal:
        feat    = self.net(obs)
        mean    = self.mean_layer(feat)
        log_std = self.log_std.clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std     = log_std.exp()
        return Normal(mean, std)

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist   = self(obs)
        action = dist.mean if deterministic else dist.sample()
        log_p  = dist.log_prob(action).sum(dim=-1)
        return action, log_p

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist    = self(obs)
        log_p   = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_p, entropy


class CriticV(nn.Module):
    """State-value estimator V(s)."""

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: List[int] = (256, 256),
        activation: type = nn.Tanh,
    ):
        super().__init__()
        self.net = build_mlp(obs_dim, hidden_sizes, activation, output_dim=1, output_gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class ActorCritic(nn.Module):
    """
    Combined Actor-Critic module.

    Shares no parameters between actor and critic (separate trunks).
    This is typical for PPO to allow different learning rates or
    gradient scales if needed.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        actor_hidden: List[int] = (256, 256),
        critic_hidden: List[int] = (256, 256),
        activation: type = nn.Tanh,
    ):
        super().__init__()
        self.actor  = GaussianActor(obs_dim, act_dim, actor_hidden, activation)
        self.critic = CriticV(obs_dim, critic_hidden, activation)

        self.obs_dim = obs_dim
        self.act_dim = act_dim

    @torch.no_grad()
    def predict(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Numpy in → numpy out for environment interaction.

        Returns:
            actions, log_probs, values   (all numpy)
        """
        obs_t   = torch.as_tensor(obs, dtype=torch.float32)
        actions, log_p = self.actor.get_action(obs_t, deterministic)
        values  = self.critic(obs_t)
        return (
            actions.cpu().numpy(),
            log_p.cpu().numpy(),
            values.cpu().numpy(),
        )

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for PPO update.

        Returns:
            log_probs, values, entropy
        """
        log_p, entropy = self.actor.evaluate_actions(obs, actions)
        values         = self.critic(obs)
        return log_p, values, entropy

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str, device: str = "cpu") -> None:
        state = torch.load(path, map_location=device)
        self.load_state_dict(state)
