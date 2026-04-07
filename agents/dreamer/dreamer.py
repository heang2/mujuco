"""
DreamerV3 — main training loop.

Training alternates between:
  1. Environment interaction (collect experience into episode buffer)
  2. World model learning (encode sequences, minimize reconstruction + KL)
  3. Actor-Critic learning (optimize AC in imagination via λ-returns)

Algorithm sketch per env step:
  - Execute action from actor in the real env
  - Store complete episodes in replay buffer
  - For each gradient step:
      a. Sample sequence batch from buffer
      b. Observe sequence → latent features (RSSM posterior)
      c. Decode → reconstruct obs, predict rewards and continuations
      d. Compute world-model loss: rec + KL + reward + continue
      e. Imagine H-step rollouts from random latent states
      f. Compute λ-returns, normalize with ReturnNormalizer
      g. Update actor to maximize normalized returns + entropy
      h. Update critic to match λ-return targets
"""

import time
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.dreamer.config      import DreamerConfig
from agents.dreamer.world_model import WorldModel
from agents.dreamer.actor_critic import (
    DreamerActor, DreamerCritic, ReturnNormalizer, compute_lambda_returns
)

logger = logging.getLogger(__name__)


class DreamerV3:
    """
    DreamerV3 agent.

    Supports any environment that follows the gymnasium.Env interface
    with a continuous observation and action space.

    Args:
        env:    gymnasium environment
        config: DreamerConfig (use defaults for a reasonable starting point)
    """

    def __init__(self, env, config: Optional[DreamerConfig] = None):
        self.env = env
        self.cfg = config or DreamerConfig()
        cfg      = self.cfg

        # Infer dims from env
        obs_dim = int(np.prod(env.observation_space.shape))
        act_dim = int(np.prod(env.action_space.shape))
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.device = torch.device(cfg.device)

        # ── World model ───────────────────────────────────────────────────
        self.wm = WorldModel(obs_dim, act_dim, cfg).to(self.device)
        self.wm_opt = torch.optim.Adam(
            self.wm.parameters(), lr=cfg.wm_lr, eps=1e-8
        )

        # ── Actor-Critic ──────────────────────────────────────────────────
        state_dim = self.wm.state_dim
        self.actor  = DreamerActor(state_dim, act_dim, cfg).to(self.device)
        self.critic = DreamerCritic(state_dim, cfg).to(self.device)
        self.critic_target = DreamerCritic(state_dim, cfg).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt  = torch.optim.Adam(
            self.actor.parameters(),  lr=cfg.actor_lr,  eps=1e-8
        )
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=cfg.critic_lr, eps=1e-8
        )

        # ── Return normalizer ─────────────────────────────────────────────
        self.return_norm = ReturnNormalizer(decay=cfg.return_ema_decay)

        # ── Episode buffer ────────────────────────────────────────────────
        # Simple Python episode buffer (Rust EpisodeBuffer optional)
        self._episodes: List[Dict] = []
        self._ep_ptr   = 0
        self._buf_size = 0

        # ── Stats ─────────────────────────────────────────────────────────
        self.total_steps   = 0
        self.total_updates = 0
        self._ep_rewards: List[float] = []

    # ── Episode buffer helpers ────────────────────────────────────────────────

    def _store_episode(self, ep: Dict) -> None:
        """Store one complete episode in the ring buffer."""
        if len(self._episodes) < self.cfg.buffer_capacity:
            self._episodes.append(ep)
        else:
            self._episodes[self._ep_ptr] = ep
        self._ep_ptr  = (self._ep_ptr + 1) % self.cfg.buffer_capacity
        self._buf_size = len(self._episodes)

    def _sample_sequences(self) -> Optional[Dict[str, torch.Tensor]]:
        """Sample a batch of fixed-length sequences from stored episodes."""
        if self._buf_size == 0:
            return None
        cfg = self.cfg
        obs_list = []
        act_list = []
        rew_list = []
        don_list = []

        for _ in range(cfg.batch_size):
            ep   = self._episodes[np.random.randint(self._buf_size)]
            T    = len(ep["rewards"])
            sl   = min(cfg.seq_len, T)
            start = np.random.randint(0, max(1, T - sl + 1))
            obs_list.append(ep["obs"][start:start + sl])
            act_list.append(ep["actions"][start:start + sl])
            rew_list.append(ep["rewards"][start:start + sl])
            don_list.append(ep["dones"][start:start + sl])

        def pad_and_stack(seqs, pad_dim):
            max_t = max(s.shape[0] for s in seqs)
            padded = [
                np.concatenate([s, np.zeros((max_t - s.shape[0], pad_dim), dtype=np.float32)])
                if len(s.shape) > 1
                else np.concatenate([s, np.zeros(max_t - s.shape[0], dtype=np.float32)])
                for s in seqs
            ]
            return np.stack(padded, axis=1)   # (T, B, dim)

        obs_np  = pad_and_stack(obs_list, self.obs_dim)   # (T, B, obs_dim)
        act_np  = pad_and_stack(act_list, self.act_dim)
        rew_np  = np.stack([ep[:obs_np.shape[0]] if len(ep) >= obs_np.shape[0]
                            else np.pad(ep, (0, obs_np.shape[0]-len(ep)))
                            for ep in rew_list], axis=1)  # (T, B)
        don_np  = np.stack([ep[:obs_np.shape[0]] if len(ep) >= obs_np.shape[0]
                            else np.pad(ep, (0, obs_np.shape[0]-len(ep)))
                            for ep in don_list], axis=1)

        to = lambda x: torch.from_numpy(x).float().to(self.device)
        return {
            "obs":     to(obs_np),
            "actions": to(act_np),
            "rewards": to(rew_np),
            "dones":   to(don_np),
        }

    # ── Core update ───────────────────────────────────────────────────────────

    def _update_world_model(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        obs     = batch["obs"]       # (T, B, obs_dim)
        actions = batch["actions"]   # (T, B, act_dim)
        rewards = batch["rewards"]   # (T, B)
        dones   = batch["dones"]     # (T, B)

        result  = self.wm.observe_sequence(obs, actions)
        feats   = result["feats"]    # (T, B, state_dim)
        kl_loss = result["kl"]

        # Flatten (T, B) → (T*B,) for prediction heads
        feat_flat = feats.reshape(-1, feats.shape[-1])
        obs_flat  = obs.reshape(-1, self.obs_dim)

        # Reconstruction
        obs_hat   = self.wm.decoder(feat_flat)
        rec_loss  = F.mse_loss(obs_hat, obs_flat)

        # Reward prediction (symlog as in DreamerV3 paper)
        rew_hat   = self.wm.reward(feat_flat)
        rew_flat  = rewards.reshape(-1)
        rew_loss  = F.mse_loss(rew_hat, rew_flat)

        # Continue prediction
        cont_hat  = self.wm.cont(feat_flat)
        cont_target = (1.0 - dones.reshape(-1))
        cont_loss = F.binary_cross_entropy(cont_hat.clamp(1e-6, 1-1e-6), cont_target)

        # KL balancing: scale free bits
        kl_loss = torch.clamp(kl_loss, min=self.cfg.kl_free)
        loss    = rec_loss + kl_loss + rew_loss + cont_loss

        self.wm_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.wm.parameters(), self.cfg.wm_grad_clip)
        self.wm_opt.step()

        return {
            "wm/rec_loss":  rec_loss.item(),
            "wm/kl_loss":   kl_loss.item(),
            "wm/rew_loss":  rew_loss.item(),
            "wm/cont_loss": cont_loss.item(),
            "wm/loss":      loss.item(),
        }

    def _update_actor_critic(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        cfg = self.cfg

        # Detach latent features — actor-critic update must not backprop into WM
        with torch.no_grad():
            result = self.wm.observe_sequence(batch["obs"], batch["actions"])
            feats  = result["feats"].detach()   # (T, B, state_dim)

        # Use all latent states as starting points for imagined rollouts
        T, B, _ = feats.shape
        start_feats = feats.reshape(T * B, -1)  # (T*B, state_dim)

        # Imagined rollout
        rollout = self.wm.imagine_rollout(
            start_feats,
            actor_fn=lambda f: self.actor.get_action(f),
            horizon=cfg.imagine_horizon,
        )
        imag_feats   = rollout["feats"]    # (H, T*B, state_dim)
        imag_rewards = rollout["rewards"]  # (H, T*B)
        imag_conts   = rollout["conts"]    # (H, T*B)

        # Critic values for λ-return (include bootstrap at H)
        H = imag_feats.shape[0]
        all_feats = torch.cat([start_feats.unsqueeze(0), imag_feats], dim=0)  # (H+1, T*B, D)
        with torch.no_grad():
            values = self.critic_target(all_feats.reshape(-1, all_feats.shape[-1]))
            values = values.reshape(H + 1, T * B)  # (H+1, T*B)

        conts = imag_conts * cfg.gamma
        targets = compute_lambda_returns(
            imag_rewards, values, conts, cfg.gamma, cfg.lambda_
        )   # (H, T*B)

        # Normalize returns
        self.return_norm.update(targets)
        targets_norm = self.return_norm.normalize(targets)

        # ── Critic update ─────────────────────────────────────────────────
        v1, v2   = self.critic.both(imag_feats.reshape(-1, imag_feats.shape[-1]))
        v1 = v1.reshape(H, T * B)
        v2 = v2.reshape(H, T * B)
        targets_det = targets_norm.detach()
        c_loss = 0.5 * (F.mse_loss(v1, targets_det) + F.mse_loss(v2, targets_det))

        self.critic_opt.zero_grad()
        c_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.ac_grad_clip)
        self.critic_opt.step()

        # Soft update target critic
        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
                pt.data.mul_(0.98).add_(0.02 * p.data)

        # ── Actor update ──────────────────────────────────────────────────
        # Re-roll with gradient tracking
        rollout_g = self.wm.imagine_rollout(
            start_feats,
            actor_fn=lambda f: self.actor.get_action(f),
            horizon=cfg.imagine_horizon,
        )
        imag_feats_g = rollout_g["feats"]
        actor_values = self.critic(imag_feats_g.reshape(-1, imag_feats_g.shape[-1]))
        actor_values = actor_values.reshape(H, T * B)

        entropy = self.actor.entropy(imag_feats_g.reshape(-1, imag_feats_g.shape[-1]))
        entropy = entropy.reshape(H, T * B).mean()

        a_loss = -(actor_values.mean() + cfg.actor_ent_scale * entropy)

        self.actor_opt.zero_grad()
        a_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.ac_grad_clip)
        self.actor_opt.step()

        return {
            "ac/critic_loss": c_loss.item(),
            "ac/actor_loss":  a_loss.item(),
            "ac/entropy":     entropy.item(),
            "ac/target_mean": targets.mean().item(),
        }

    # ── Main training loop ────────────────────────────────────────────────────

    def learn(self, callback=None) -> None:
        cfg  = self.cfg
        log_dir = Path(cfg.log_dir) / f"DreamerV3_{int(time.time())}"
        log_dir.mkdir(parents=True, exist_ok=True)

        obs, _ = self.env.reset(seed=cfg.seed)
        ep_obs, ep_acts, ep_rews, ep_done = [], [], [], []
        ep_return = 0.0

        t0 = time.time()

        while self.total_steps < cfg.total_steps:
            # ── Collect experience ─────────────────────────────────────────
            if self.total_steps < cfg.prefill_steps:
                action = self.env.action_space.sample()
            else:
                feat = self._get_current_feat(obs)
                action = self.actor.get_action(feat).detach().cpu().numpy()

            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            ep_obs.append(obs.astype(np.float32))
            ep_acts.append(np.asarray(action, dtype=np.float32))
            ep_rews.append(float(reward))
            ep_done.append(float(done))
            ep_return += reward
            self.total_steps += 1

            if done:
                self._store_episode({
                    "obs":     np.array(ep_obs),
                    "actions": np.array(ep_acts),
                    "rewards": np.array(ep_rews, dtype=np.float32),
                    "dones":   np.array(ep_done, dtype=np.float32),
                })
                self._ep_rewards.append(ep_return)
                ep_obs, ep_acts, ep_rews, ep_done = [], [], [], []
                ep_return = 0.0
                obs, _ = self.env.reset()
            else:
                obs = next_obs

            # ── Gradient updates ──────────────────────────────────────────
            if (self.total_steps >= cfg.prefill_steps and
                    self._buf_size > 0 and
                    self.total_steps % max(1, cfg.train_ratio // 512) == 0):
                for _ in range(cfg.train_ratio // 512):
                    batch = self._sample_sequences()
                    if batch is None:
                        break
                    wm_stats = self._update_world_model(batch)
                    ac_stats = self._update_actor_critic(batch)
                    self.total_updates += 1

            # ── Logging ───────────────────────────────────────────────────
            if self.total_steps % cfg.log_interval == 0:
                mean_ret = np.mean(self._ep_rewards[-20:]) if self._ep_rewards else float("nan")
                elapsed  = time.time() - t0
                logger.info(
                    f"step={self.total_steps:>8d}  "
                    f"ep_return={mean_ret:>8.1f}  "
                    f"updates={self.total_updates}  "
                    f"elapsed={elapsed:.0f}s"
                )
                if callback:
                    callback(self.total_steps, {"mean_return": mean_ret})

            # ── Save ──────────────────────────────────────────────────────
            if self.total_steps % cfg.save_interval == 0:
                self.save(log_dir / f"checkpoint_{self.total_steps}.pt")

        self.save(log_dir / "final_model.pt")
        logger.info("Training complete.")

    def _get_current_feat(self, obs: np.ndarray) -> torch.Tensor:
        """Encode a single observation to a latent feature (greedy, no history)."""
        obs_t  = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        embed  = self.wm.encoder(obs_t)
        h, z   = self.wm.rssm.initial_state(1, self.device)
        act    = torch.zeros(1, self.act_dim, device=self.device)
        h, z, _ = self.wm.rssm.observe(embed, act, h, z)
        return self.wm.feat(h, z)

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Return an action for a given observation."""
        with torch.no_grad():
            feat   = self._get_current_feat(obs)
            action = self.actor.get_action(feat, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "wm":         self.wm.state_dict(),
            "actor":      self.actor.state_dict(),
            "critic":     self.critic.state_dict(),
            "wm_opt":     self.wm_opt.state_dict(),
            "actor_opt":  self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "total_steps": self.total_steps,
            "cfg":        asdict(self.cfg),
        }, path)
        logger.info(f"Saved checkpoint to {path}")

    def load(self, path) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.wm.load_state_dict(ckpt["wm"])
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.wm_opt.load_state_dict(ckpt["wm_opt"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])
        self.total_steps = ckpt["total_steps"]
        logger.info(f"Loaded checkpoint from {path} (step {self.total_steps})")
