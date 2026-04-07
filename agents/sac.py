"""
Soft Actor-Critic (SAC) — from scratch.

Reference:
    Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL
    with a Stochastic Actor" (2018)  https://arxiv.org/abs/1801.01290
    Haarnoja et al., "Soft Actor-Critic Algorithms and Applications" (2019)
    https://arxiv.org/abs/1812.05905  (auto-tuned temperature)

SAC is an off-policy, maximum-entropy RL algorithm for continuous control.
Key ideas:
  1. Actor maximises Q-value AND entropy simultaneously
  2. Twin Q-networks (critic) prevent overestimation bias
  3. Target Q-networks (EMA) stabilise training
  4. Automatic entropy tuning adjusts temperature α so that
     E[−log π(a|s)] ≈ target_entropy = −dim(A)
  5. Reparameterization trick for low-variance policy gradients

Compared to PPO:
  - Off-policy (far more sample efficient)
  - Continuous actions only
  - Typically converges in fewer env steps but requires a replay buffer
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
from torch.distributions import Normal

from agents.replay_buffer import UniformReplayBuffer, PrioritizedReplayBuffer


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SACConfig:
    # Network
    hidden_sizes: List[int] = field(default_factory=lambda: [256, 256])

    # SAC hypers
    gamma:              float = 0.99
    tau:                float = 5e-3      # EMA rate for target networks
    lr_actor:           float = 3e-4
    lr_critic:          float = 3e-4
    lr_alpha:           float = 3e-4
    batch_size:         int   = 256
    replay_capacity:    int   = 1_000_000
    learning_starts:    int   = 10_000    # steps before first gradient update
    update_every:       int   = 1         # gradient steps per env step
    gradient_steps:     int   = 1         # gradient steps per update call

    # Entropy
    auto_tune_alpha:    bool  = True
    init_alpha:         float = 0.2
    target_entropy:     Optional[float] = None  # set to -act_dim if None

    # PER
    use_per:            bool  = False
    per_alpha:          float = 0.6
    per_beta_init:      float = 0.4
    per_beta_steps:     int   = 1_000_000

    # Training
    total_timesteps:    int   = 1_000_000
    device:             str   = "cpu"
    seed:               int   = 42


# ──────────────────────────────────────────────────────────────────────────────
# Network definitions
# ──────────────────────────────────────────────────────────────────────────────

def _mlp(in_dim: int, hidden: List[int], out_dim: int, act=nn.ReLU) -> nn.Sequential:
    layers: list = []
    dim = in_dim
    for h in hidden:
        layers += [nn.Linear(dim, h), act()]
        dim = h
    layers.append(nn.Linear(dim, out_dim))
    # Orthogonal init
    for m in layers:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)
    return nn.Sequential(*layers)


class SquashedGaussianActor(nn.Module):
    """
    Actor with tanh-squashed Gaussian output (reparameterization trick).

    Action = tanh(mean + std * ε),  ε ~ N(0, I)

    Log-prob corrected for tanh squashing:
        log π(a|s) = log N(u|μ, σ) − Σ log(1 − tanh²(u))
    """

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, obs_dim: int, act_dim: int, hidden: List[int]):
        super().__init__()
        self.net       = _mlp(obs_dim, hidden, hidden[-1])
        self.mean_head = nn.Linear(hidden[-1], act_dim)
        self.lstd_head = nn.Linear(hidden[-1], act_dim)

        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.constant_(self.mean_head.bias, 0)
        nn.init.orthogonal_(self.lstd_head.weight, gain=0.01)
        nn.init.constant_(self.lstd_head.bias, 0)

    def forward(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            action:   tanh-squashed action  (B, act_dim)
            log_prob: log π(a|s)             (B,)
        """
        feat  = self.net(obs)
        mean  = self.mean_head(feat)
        lstd  = self.lstd_head(feat).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std   = lstd.exp()

        if deterministic:
            u      = mean
            # Guard: replace NaN means with zeros (unstable simulation guard)
            u      = torch.nan_to_num(u, nan=0.0)
            action = u.tanh()
            return action, torch.zeros(obs.shape[0], device=obs.device)

        dist = Normal(mean, std, validate_args=False)
        u    = dist.rsample()          # reparameterization
        action = u.tanh()

        # Corrected log-prob: Σ [log N(u) − log(1 − tanh²(u))]
        log_prob = dist.log_prob(u) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)

        return action, log_prob


class QNetwork(nn.Module):
    """Twin Q-networks Q1(s,a) and Q2(s,a)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden: List[int]):
        super().__init__()
        self.q1 = _mlp(obs_dim + act_dim, hidden, 1)
        self.q2 = _mlp(obs_dim + act_dim, hidden, 1)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x  = torch.cat([obs, action], dim=-1)
        q1 = self.q1(x).squeeze(-1)
        q2 = self.q2(x).squeeze(-1)
        return q1, q2

    def q_min(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.forward(obs, action)
        return torch.min(q1, q2)


# ──────────────────────────────────────────────────────────────────────────────
# SAC Agent
# ──────────────────────────────────────────────────────────────────────────────

class SAC:
    """
    Soft Actor-Critic agent.

    Example:
        from agents.sac import SAC, SACConfig
        from envs import make_env

        env    = make_env("Hopper")
        config = SACConfig(total_timesteps=500_000)
        agent  = SAC(env, config)
        agent.learn(log_interval=5000)
    """

    def __init__(self, env, config: Optional[SACConfig] = None):
        self.env    = env
        self.config = config or SACConfig()
        cfg         = self.config

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Actor
        self.actor = SquashedGaussianActor(obs_dim, act_dim, cfg.hidden_sizes).to(cfg.device)

        # Critics + targets
        self.critic        = QNetwork(obs_dim, act_dim, cfg.hidden_sizes).to(cfg.device)
        self.critic_target = copy.deepcopy(self.critic)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        # Optimisers
        self.actor_optim  = optim.Adam(self.actor.parameters(),  lr=cfg.lr_actor)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)

        # Entropy temperature α
        if cfg.auto_tune_alpha:
            self.target_entropy = cfg.target_entropy or -float(act_dim)
            self.log_alpha      = torch.tensor(
                np.log(cfg.init_alpha), dtype=torch.float32,
                device=cfg.device, requires_grad=True,
            )
            self.alpha_optim    = optim.Adam([self.log_alpha], lr=cfg.lr_alpha)
        else:
            self.log_alpha      = torch.tensor(np.log(cfg.init_alpha), device=cfg.device)

        # Replay buffer
        buf_kwargs = dict(
            capacity=cfg.replay_capacity,
            obs_shape=env.observation_space.shape,
            act_shape=env.action_space.shape,
            seed=cfg.seed,
        )
        if cfg.use_per:
            self.buffer = PrioritizedReplayBuffer(
                **buf_kwargs,
                alpha=cfg.per_alpha,
                beta_init=cfg.per_beta_init,
                beta_steps=cfg.per_beta_steps,
            )
        else:
            self.buffer = UniformReplayBuffer(**buf_kwargs)

        self._global_step       = 0
        self._episode_rewards:  List[float] = []
        self._episode_lengths:  List[int]   = []
        self._current_ep_rew    = 0.0
        self._current_ep_len    = 0

        self.callbacks: List[Callable] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.exp().item())

    def predict(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        # Guard against NaN/Inf observations (unstable simulation)
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        obs_t  = torch.as_tensor(obs, dtype=torch.float32, device=self.config.device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        with torch.no_grad():
            action, _ = self.actor(obs_t, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()

    def learn(
        self,
        total_timesteps: Optional[int] = None,
        log_interval: int = 5000,
        save_dir: Optional[str] = None,
        save_freq: int = 100_000,
    ) -> "SAC":
        cfg             = self.config
        total_timesteps = total_timesteps or cfg.total_timesteps

        obs, _ = self.env.reset(seed=cfg.seed)
        t_start = time.time()

        while self._global_step < total_timesteps:
            # ── Collect one step ──────────────────────────────────────
            if self._global_step < cfg.learning_starts:
                action = self.env.action_space.sample()
            else:
                action = self.predict(obs, deterministic=False)

            action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            # Skip storing transitions with non-finite obs (unstable simulation)
            if np.all(np.isfinite(obs)) and np.all(np.isfinite(next_obs)):
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
            else:
                obs = next_obs

            # ── Gradient updates ──────────────────────────────────────
            if (
                self._global_step >= cfg.learning_starts
                and self._global_step % cfg.update_every == 0
                and self.buffer.is_ready
            ):
                for _ in range(cfg.gradient_steps):
                    train_info = self._update()

            # ── Logging ───────────────────────────────────────────────
            if self._global_step % log_interval == 0:
                elapsed = time.time() - t_start
                fps     = self._global_step / max(elapsed, 1)
                self._log(fps)

            # ── Checkpointing ─────────────────────────────────────────
            if save_dir and self._global_step % save_freq < 1:
                path = Path(save_dir) / f"sac_{self._global_step}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                self.save(str(path))

            # ── Callbacks ─────────────────────────────────────────────
            for cb in self.callbacks:
                cb()

        return self

    def save(self, path: str) -> None:
        torch.save({
            "actor":        self.actor.state_dict(),
            "critic":       self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha":    self.log_alpha,
            "step":         self._global_step,
            "config":       self.config,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.config.device)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.log_alpha      = ckpt["log_alpha"]
        self._global_step   = ckpt["step"]

    # ------------------------------------------------------------------
    # Internal training
    # ------------------------------------------------------------------

    def _update(self) -> Dict[str, float]:
        cfg   = self.config
        batch = self.buffer.sample(cfg.batch_size)
        alpha = self.alpha

        # Sanitize batch data — NaN can arise from unstable simulations
        obs_t      = torch.as_tensor(np.nan_to_num(batch.obs,      0.0), dtype=torch.float32, device=cfg.device)
        acts_t     = torch.as_tensor(np.nan_to_num(batch.actions,  0.0), dtype=torch.float32, device=cfg.device)
        rews_t     = torch.as_tensor(np.nan_to_num(batch.rewards,  0.0), dtype=torch.float32, device=cfg.device)
        next_obs_t = torch.as_tensor(np.nan_to_num(batch.next_obs, 0.0), dtype=torch.float32, device=cfg.device)
        dones_t    = torch.as_tensor(batch.dones,    dtype=torch.float32, device=cfg.device)
        weights_t  = torch.as_tensor(batch.weights,  dtype=torch.float32, device=cfg.device)

        # Skip update if batch still contains non-finite values
        if not (torch.all(torch.isfinite(obs_t)) and torch.all(torch.isfinite(rews_t))):
            return {"critic_loss": 0.0, "actor_loss": 0.0, "alpha_loss": 0.0,
                    "alpha": self.alpha, "log_pi_mean": 0.0}

        # ── Critic update ─────────────────────────────────────────────
        with torch.no_grad():
            next_actions, next_log_pi = self.actor(next_obs_t)
            q1_targ, q2_targ = self.critic_target(next_obs_t, next_actions)
            q_targ  = torch.min(q1_targ, q2_targ) - alpha * next_log_pi
            y       = rews_t + cfg.gamma * (1.0 - dones_t) * q_targ

        q1, q2 = self.critic(obs_t, acts_t)
        td1     = (q1 - y).abs().detach()
        td2     = (q2 - y).abs().detach()
        td_err  = (td1 + td2) / 2.0

        critic_loss = (
            (weights_t * F.mse_loss(q1, y, reduction="none")).mean()
            + (weights_t * F.mse_loss(q2, y, reduction="none")).mean()
        )

        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optim.step()

        # Update PER priorities
        if cfg.use_per and hasattr(self.buffer, "update_priorities"):
            self.buffer.update_priorities(batch.indices, td_err.cpu().numpy())

        # ── Actor update ──────────────────────────────────────────────
        new_actions, log_pi = self.actor(obs_t)
        q_pi = self.critic.q_min(obs_t, new_actions)

        actor_loss = (alpha * log_pi - q_pi).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optim.step()

        # ── Alpha update (auto-tune) ───────────────────────────────────
        alpha_loss_val = 0.0
        if cfg.auto_tune_alpha:
            # Guard: skip if log_pi contains NaN (from unstable obs)
            if torch.all(torch.isfinite(log_pi)):
                alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                nn.utils.clip_grad_norm_([self.log_alpha], 1.0)
                self.alpha_optim.step()
                alpha_loss_val = alpha_loss.item()
                # Clamp log_alpha to prevent divergence
                with torch.no_grad():
                    self.log_alpha.clamp_(-5.0, 2.0)

        # ── Soft target update ────────────────────────────────────────
        for p, p_targ in zip(self.critic.parameters(), self.critic_target.parameters()):
            p_targ.data.mul_(1 - cfg.tau)
            p_targ.data.add_(cfg.tau * p.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
            "alpha_loss":  alpha_loss_val,
            "alpha":       self.alpha,
            "log_pi_mean": log_pi.mean().item(),
        }

    def _log(self, fps: float) -> None:
        recent  = self._episode_rewards[-20:] if self._episode_rewards else [0.0]
        lens    = self._episode_lengths[-20:] if self._episode_lengths else [0]
        buf_pct = len(self.buffer) / self.buffer.capacity * 100
        print(
            f"[SAC {self._global_step:>8,}]  "
            f"ep_rew={np.mean(recent):>8.2f}  "
            f"ep_len={int(np.mean(lens)):>5}  "
            f"α={self.alpha:.4f}  "
            f"buf={buf_pct:>5.1f}%  "
            f"fps={fps:>5.0f}"
        )
