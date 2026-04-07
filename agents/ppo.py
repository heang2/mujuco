"""
Proximal Policy Optimization (PPO-Clip) — from scratch.

Reference:
    Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
    https://arxiv.org/abs/1707.06347

Key features:
  - Clipped surrogate objective
  - Generalized Advantage Estimation (GAE-λ)
  - Shared or separate Actor/Critic networks
  - Gradient norm clipping
  - Observation normalization (running stats)
  - Entropy bonus for exploration
  - Linear LR annealing
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from agents.networks import ActorCritic
from training.rollout_buffer import RolloutBuffer


@dataclass
class PPOConfig:
    # ------- Network -------
    actor_hidden:  List[int] = field(default_factory=lambda: [256, 256])
    critic_hidden: List[int] = field(default_factory=lambda: [256, 256])

    # ------- PPO hypers -------
    n_steps:        int   = 2048     # steps per rollout per env
    n_epochs:       int   = 10       # gradient update epochs per rollout
    mini_batch_size: int  = 64
    clip_eps:       float = 0.2      # clipping parameter ε
    vf_coef:        float = 0.5      # value function loss coefficient
    ent_coef:       float = 0.0      # entropy bonus coefficient
    max_grad_norm:  float = 0.5      # gradient clipping
    target_kl:     Optional[float] = 0.015  # early stop if KL > target

    # ------- GAE -------
    gamma:     float = 0.99
    gae_lambda: float = 0.95

    # ------- Optimiser -------
    lr:            float = 3e-4
    lr_anneal:     bool  = True     # linearly anneal to 0
    total_timesteps: int = 1_000_000

    # ------- Normalisation -------
    normalize_obs:      bool = True
    normalize_rewards:  bool = False

    # ------- Misc -------
    device: str = "cpu"
    seed:   int = 42


class RunningMeanStd:
    """Welford online algorithm for mean and variance."""

    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var  = np.ones(shape,  dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        batch_mean = x.mean(axis=0)
        batch_var  = x.var(axis=0)
        batch_n    = x.shape[0]

        delta       = batch_mean - self.mean
        total_n     = self.count + batch_n
        new_mean    = self.mean + delta * batch_n / total_n
        m_a         = self.var * self.count
        m_b         = batch_var * batch_n
        new_var     = (m_a + m_b + delta ** 2 * self.count * batch_n / total_n) / total_n

        self.mean   = new_mean
        self.var    = new_var
        self.count  = total_n

    def normalize(self, x: np.ndarray, clip: float = 10.0) -> np.ndarray:
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -clip, clip)


class PPO:
    """
    PPO-Clip agent.

    Example usage:
        from agents.ppo import PPO, PPOConfig
        from envs import make_env

        env    = make_env("Hopper")
        config = PPOConfig(total_timesteps=500_000)
        agent  = PPO(env, config)
        agent.learn(log_interval=10)
    """

    def __init__(self, env, config: Optional[PPOConfig] = None):
        self.env    = env
        self.config = config or PPOConfig()
        cfg         = self.config

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))

        self.policy = ActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            actor_hidden=cfg.actor_hidden,
            critic_hidden=cfg.critic_hidden,
        ).to(cfg.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=cfg.lr, eps=1e-5)

        self.buffer = RolloutBuffer(
            n_steps=cfg.n_steps,
            obs_shape=env.observation_space.shape,
            act_shape=env.action_space.shape,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
        )

        # Normalizers
        self.obs_rms = RunningMeanStd(shape=env.observation_space.shape)
        self.rew_rms = RunningMeanStd(shape=())

        self._global_step = 0
        self._episode_rewards: List[float] = []
        self._episode_lengths: List[int]   = []
        self._current_ep_reward = 0.0
        self._current_ep_length = 0

        # Callbacks
        self.callbacks: List[Callable] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def learn(
        self,
        total_timesteps: Optional[int] = None,
        log_interval: int = 1,
        save_dir: Optional[str] = None,
        save_freq: int = 50_000,
    ) -> "PPO":
        cfg = self.config
        total_timesteps = total_timesteps or cfg.total_timesteps

        obs, _ = self.env.reset(seed=cfg.seed)
        rollout_count = 0
        t_start = time.time()

        while self._global_step < total_timesteps:
            # ---- Collect rollout ----
            self._collect_rollout(obs)
            rollout_count += 1
            obs = self._last_obs   # updated inside _collect_rollout

            # ---- LR annealing ----
            if cfg.lr_anneal:
                frac = 1.0 - self._global_step / total_timesteps
                for pg in self.optimizer.param_groups:
                    pg["lr"] = cfg.lr * frac

            # ---- Update policy ----
            train_info = self._update()

            # ---- Logging ----
            if rollout_count % log_interval == 0:
                elapsed = time.time() - t_start
                fps     = self._global_step / max(elapsed, 1)
                self._log(rollout_count, train_info, fps)

            # ---- Save checkpoint ----
            if save_dir and self._global_step % save_freq < cfg.n_steps:
                path = Path(save_dir) / f"ckpt_{self._global_step}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                self.save(str(path))

        return self

    def predict(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Single observation → single action (numpy)."""
        if self.config.normalize_obs:
            obs = self.obs_rms.normalize(obs)
        obs = obs[np.newaxis] if obs.ndim == 1 else obs
        actions, _, _ = self.policy.predict(obs, deterministic)
        return actions[0]

    def save(self, path: str) -> None:
        torch.save({
            "policy":    self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "obs_rms":   self.obs_rms,
            "rew_rms":   self.rew_rms,
            "step":      self._global_step,
            "config":    self.config,
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.config.device, weights_only=False)
        self.policy.load_state_dict(ckpt["policy"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.obs_rms      = ckpt["obs_rms"]
        self.rew_rms      = ckpt["rew_rms"]
        self._global_step = ckpt["step"]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _collect_rollout(self, obs: np.ndarray) -> None:
        cfg = self.config
        self.buffer.reset()
        self.policy.eval()

        for _ in range(cfg.n_steps):
            # Normalise observation
            if cfg.normalize_obs:
                self.obs_rms.update(obs[np.newaxis])
                obs_norm = self.obs_rms.normalize(obs)
            else:
                obs_norm = obs

            actions, log_probs, values = self.policy.predict(obs_norm[np.newaxis])
            action   = actions[0]
            log_prob = log_probs[0]
            value    = values[0]

            # Clip to action space
            clipped_action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
            next_obs, reward, terminated, truncated, info = self.env.step(clipped_action)

            # Reward normalisation
            if cfg.normalize_rewards:
                self.rew_rms.update(np.array([reward]))
                reward_norm = reward / np.sqrt(self.rew_rms.var + 1e-8)
            else:
                reward_norm = float(reward)

            done = terminated or truncated

            self.buffer.add(
                obs=obs_norm,
                action=action,
                reward=reward_norm,
                done=done,
                value=value,
                log_prob=log_prob,
            )

            # Track episode stats
            self._current_ep_reward += float(reward)
            self._current_ep_length += 1
            self._global_step       += 1

            if done:
                self._episode_rewards.append(self._current_ep_reward)
                self._episode_lengths.append(self._current_ep_length)
                self._current_ep_reward = 0.0
                self._current_ep_length = 0
                obs, _ = self.env.reset()
            else:
                obs = next_obs

        # Bootstrap last value
        if cfg.normalize_obs:
            obs_norm = self.obs_rms.normalize(obs)
        else:
            obs_norm = obs
        _, _, last_val = self.policy.predict(obs_norm[np.newaxis])
        self.buffer.compute_returns(last_val[0])

        self._last_obs = obs

    def _update(self) -> Dict[str, float]:
        cfg = self.config
        self.policy.train()

        pg_losses, vf_losses, ent_bonuses, kl_divs = [], [], [], []
        clip_fractions = []

        obs, acts, log_probs_old, returns, advantages = self.buffer.get()

        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t        = torch.as_tensor(obs,          dtype=torch.float32, device=cfg.device)
        acts_t       = torch.as_tensor(acts,         dtype=torch.float32, device=cfg.device)
        logp_old_t   = torch.as_tensor(log_probs_old, dtype=torch.float32, device=cfg.device)
        returns_t    = torch.as_tensor(returns,      dtype=torch.float32, device=cfg.device)
        advantages_t = torch.as_tensor(advantages,   dtype=torch.float32, device=cfg.device)

        n = obs_t.shape[0]

        for epoch in range(cfg.n_epochs):
            indices = torch.randperm(n, device=cfg.device)
            for start in range(0, n, cfg.mini_batch_size):
                idx  = indices[start: start + cfg.mini_batch_size]

                logp, values, entropy = self.policy.evaluate(obs_t[idx], acts_t[idx])

                ratio     = (logp - logp_old_t[idx]).exp()
                adv_mb    = advantages_t[idx]

                # Clipped surrogate loss
                surr1 = ratio * adv_mb
                surr2 = ratio.clamp(1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv_mb
                pg_loss = -torch.min(surr1, surr2).mean()

                # Value function loss
                vf_loss = 0.5 * nn.functional.mse_loss(values, returns_t[idx])

                # Entropy bonus
                ent_loss = -entropy.mean()

                loss = pg_loss + cfg.vf_coef * vf_loss + cfg.ent_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    clip_frac = ((ratio - 1).abs() > cfg.clip_eps).float().mean().item()
                    approx_kl = 0.5 * (logp_old_t[idx] - logp).pow(2).mean().item()

                pg_losses.append(pg_loss.item())
                vf_losses.append(vf_loss.item())
                ent_bonuses.append(-ent_loss.item())
                kl_divs.append(approx_kl)
                clip_fractions.append(clip_frac)

            # Early stopping on KL divergence
            if cfg.target_kl and np.mean(kl_divs[-10:]) > cfg.target_kl:
                break

        return {
            "pg_loss":       float(np.mean(pg_losses)),
            "vf_loss":       float(np.mean(vf_losses)),
            "entropy":       float(np.mean(ent_bonuses)),
            "approx_kl":     float(np.mean(kl_divs)),
            "clip_fraction": float(np.mean(clip_fractions)),
            "lr":            self.optimizer.param_groups[0]["lr"],
        }

    def _log(self, rollout: int, info: Dict[str, float], fps: float) -> None:
        recent = self._episode_rewards[-20:] if self._episode_rewards else [0.0]
        lens   = self._episode_lengths[-20:] if self._episode_lengths else [0]
        print(
            f"[Rollout {rollout:>5}]  "
            f"steps={self._global_step:>8,}  "
            f"ep_rew={np.mean(recent):>8.2f}  "
            f"ep_len={int(np.mean(lens)):>5}  "
            f"pg={info['pg_loss']:>7.4f}  "
            f"vf={info['vf_loss']:>7.4f}  "
            f"ent={info['entropy']:>6.4f}  "
            f"kl={info['approx_kl']:>6.4f}  "
            f"fps={fps:>5.0f}"
        )
