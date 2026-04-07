"""
Twin Delayed Deep Deterministic Policy Gradient (TD3) — from scratch.

Reference:
    Fujimoto et al., "Addressing Function Approximation Error in
    Actor-Critic Methods" (2018)  https://arxiv.org/abs/1802.09477

TD3 improvements over DDPG:
  1. Twin Q-networks — take min to prevent overestimation
  2. Delayed actor updates — update actor/targets every d steps
  3. Target policy smoothing — add clipped noise to target actions

Deterministic policy (vs SAC's stochastic), exploration via
external noise (Ornstein-Uhlenbeck or Gaussian).
"""

import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.replay_buffer import UniformReplayBuffer


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TD3Config:
    hidden_sizes:      List[int] = field(default_factory=lambda: [256, 256])

    gamma:             float = 0.99
    tau:               float = 5e-3
    lr_actor:          float = 3e-4
    lr_critic:         float = 3e-4
    batch_size:        int   = 256
    replay_capacity:   int   = 1_000_000
    learning_starts:   int   = 10_000
    update_every:      int   = 1
    policy_delay:      int   = 2           # update actor every policy_delay critic steps

    # Exploration noise
    expl_noise:        float = 0.1         # std of action-space Gaussian noise
    noise_clip:        float = 0.5         # clip target policy smoothing noise
    policy_noise:      float = 0.2         # std of target policy smoothing noise

    total_timesteps:   int   = 1_000_000
    device:            str   = "cpu"
    seed:              int   = 42


# ──────────────────────────────────────────────────────────────────────────────
# Networks
# ──────────────────────────────────────────────────────────────────────────────

def _mlp(in_dim: int, hidden: List[int], out_dim: int) -> nn.Sequential:
    layers: list = []
    dim = in_dim
    for h in hidden:
        layers += [nn.Linear(dim, h), nn.ReLU()]
        dim = h
    layers.append(nn.Linear(dim, out_dim))
    for m in layers:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)
    return nn.Sequential(*layers)


class DeterministicActor(nn.Module):
    """Deterministic policy: μ(s) ∈ [-1, 1]^act_dim"""

    def __init__(self, obs_dim: int, act_dim: int, hidden: List[int]):
        super().__init__()
        self.net = _mlp(obs_dim, hidden, act_dim)
        # Smaller init for the output layer
        nn.init.orthogonal_(list(self.net.children())[-1].weight, gain=0.01)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).tanh()


class TwinQNetwork(nn.Module):
    """Twin Q-networks: min(Q1, Q2) for conservative value estimates."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: List[int]):
        super().__init__()
        self.q1 = _mlp(obs_dim + act_dim, hidden, 1)
        self.q2 = _mlp(obs_dim + act_dim, hidden, 1)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x  = torch.cat([obs, action], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def q1_only(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.q1(torch.cat([obs, action], dim=-1)).squeeze(-1)


# ──────────────────────────────────────────────────────────────────────────────
# Ornstein-Uhlenbeck Noise (for correlated action noise)
# ──────────────────────────────────────────────────────────────────────────────

class OUNoise:
    """
    Ornstein-Uhlenbeck process for temporally correlated exploration noise.
    More effective than independent Gaussian noise for physical control tasks.
    """

    def __init__(
        self,
        shape: tuple,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
        seed: int = 0,
    ):
        self.shape = shape
        self.mu    = mu * np.ones(shape)
        self.theta = theta
        self.sigma = sigma
        self._rng  = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        self.state = np.copy(self.mu)

    def sample(self) -> np.ndarray:
        x    = self.state
        dx   = self.theta * (self.mu - x) + self.sigma * self._rng.standard_normal(self.shape)
        self.state = x + dx
        return self.state.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# TD3 Agent
# ──────────────────────────────────────────────────────────────────────────────

class TD3:
    """
    TD3 agent.

    Example:
        from agents.td3 import TD3, TD3Config
        agent = TD3(env, TD3Config(total_timesteps=300_000))
        agent.learn(log_interval=5000)
    """

    def __init__(self, env, config: Optional[TD3Config] = None):
        self.env    = env
        self.config = config or TD3Config()
        cfg         = self.config

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
        self.act_dim = act_dim

        # Networks
        self.actor        = DeterministicActor(obs_dim, act_dim, cfg.hidden_sizes).to(cfg.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic       = TwinQNetwork(obs_dim, act_dim, cfg.hidden_sizes).to(cfg.device)
        self.critic_target = copy.deepcopy(self.critic)

        for net in [self.actor_target, self.critic_target]:
            for p in net.parameters():
                p.requires_grad_(False)

        # Optimisers
        self.actor_optim  = optim.Adam(self.actor.parameters(),  lr=cfg.lr_actor)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)

        # Replay buffer
        self.buffer = UniformReplayBuffer(
            capacity=cfg.replay_capacity,
            obs_shape=env.observation_space.shape,
            act_shape=env.action_space.shape,
            seed=cfg.seed,
        )

        # Exploration noise
        self.expl_noise = OUNoise(
            shape=(act_dim,),
            sigma=cfg.expl_noise,
            seed=cfg.seed,
        )

        self._global_step    = 0
        self._critic_updates = 0
        self._episode_rewards: List[float] = []
        self._episode_lengths: List[int]   = []
        self._current_ep_rew = 0.0
        self._current_ep_len = 0
        self.callbacks: List[Callable] = []

    # ------------------------------------------------------------------

    def predict(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t  = torch.as_tensor(obs, dtype=torch.float32, device=self.config.device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs_t).squeeze(0).cpu().numpy()
        if not deterministic:
            action = action + self.expl_noise.sample()
        return np.clip(action, self.env.action_space.low, self.env.action_space.high)

    def learn(
        self,
        total_timesteps: Optional[int] = None,
        log_interval: int = 5000,
        save_dir: Optional[str] = None,
        save_freq: int = 100_000,
    ) -> "TD3":
        cfg             = self.config
        total_timesteps = total_timesteps or cfg.total_timesteps
        obs, _          = self.env.reset(seed=cfg.seed)
        t_start         = time.time()

        while self._global_step < total_timesteps:
            if self._global_step < cfg.learning_starts:
                action = self.env.action_space.sample()
            else:
                action = self.predict(obs, deterministic=False)

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.buffer.add(obs, action, float(reward), next_obs, terminated)

            self._current_ep_rew += float(reward)
            self._current_ep_len += 1
            self._global_step    += 1

            if done:
                self._episode_rewards.append(self._current_ep_rew)
                self._episode_lengths.append(self._current_ep_len)
                self._current_ep_rew = 0.0
                self._current_ep_len = 0
                obs, _ = self.env.reset()
                self.expl_noise.reset()
            else:
                obs = next_obs

            if (
                self._global_step >= cfg.learning_starts
                and self._global_step % cfg.update_every == 0
                and self.buffer.is_ready
            ):
                self._update()

            if self._global_step % log_interval == 0:
                elapsed = time.time() - t_start
                self._log(self._global_step / max(elapsed, 1))

            if save_dir and self._global_step % save_freq < 1:
                path = Path(save_dir) / f"td3_{self._global_step}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                self.save(str(path))

            for cb in self.callbacks:
                cb()

        return self

    def save(self, path: str) -> None:
        torch.save({
            "actor":         self.actor.state_dict(),
            "critic":        self.critic.state_dict(),
            "actor_target":  self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "step":          self._global_step,
            "config":        self.config,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.config.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self._global_step = ckpt["step"]

    # ------------------------------------------------------------------

    def _update(self) -> None:
        cfg   = self.config
        batch = self.buffer.sample(cfg.batch_size)

        obs_t      = torch.as_tensor(batch.obs,      dtype=torch.float32, device=cfg.device)
        acts_t     = torch.as_tensor(batch.actions,  dtype=torch.float32, device=cfg.device)
        rews_t     = torch.as_tensor(batch.rewards,  dtype=torch.float32, device=cfg.device)
        next_obs_t = torch.as_tensor(batch.next_obs, dtype=torch.float32, device=cfg.device)
        dones_t    = torch.as_tensor(batch.dones,    dtype=torch.float32, device=cfg.device)

        # ── Critic update ─────────────────────────────────────────────
        with torch.no_grad():
            # Target policy smoothing: add clipped noise
            noise = (torch.randn_like(acts_t) * cfg.policy_noise).clamp(
                -cfg.noise_clip, cfg.noise_clip
            )
            next_acts = (self.actor_target(next_obs_t) + noise).clamp(-1.0, 1.0)
            q1_targ, q2_targ = self.critic_target(next_obs_t, next_acts)
            q_targ  = torch.min(q1_targ, q2_targ)
            y       = rews_t + cfg.gamma * (1.0 - dones_t) * q_targ

        q1, q2     = self.critic(obs_t, acts_t)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optim.step()
        self._critic_updates += 1

        # ── Delayed actor & target updates ───────────────────────────
        if self._critic_updates % cfg.policy_delay == 0:
            # Actor loss: maximise Q1(s, μ(s))
            actor_loss = -self.critic.q1_only(obs_t, self.actor(obs_t)).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
            self.actor_optim.step()

            # Soft updates
            for p, p_t in zip(self.actor.parameters(),  self.actor_target.parameters()):
                p_t.data.mul_(1 - cfg.tau).add_(cfg.tau * p.data)
            for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_t.data.mul_(1 - cfg.tau).add_(cfg.tau * p.data)

    def _log(self, fps: float) -> None:
        recent = self._episode_rewards[-20:] if self._episode_rewards else [0.0]
        lens   = self._episode_lengths[-20:] if self._episode_lengths else [0]
        print(
            f"[TD3 {self._global_step:>8,}]  "
            f"ep_rew={np.mean(recent):>8.2f}  "
            f"ep_len={int(np.mean(lens)):>5}  "
            f"buf={len(self.buffer):>7,}  "
            f"fps={fps:>5.0f}"
        )
