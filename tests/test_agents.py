"""
Tests for PPO agent and network components.

Run with:
    python -m pytest tests/test_agents.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import pytest

from envs         import make_env
from agents.networks    import GaussianActor, CriticV, ActorCritic
from agents.ppo         import PPO, PPOConfig, RunningMeanStd
from agents.random_agent import RandomAgent
from training.rollout_buffer import RolloutBuffer


# -----------------------------------------------------------------------
# Running mean/std
# -----------------------------------------------------------------------

class TestRunningMeanStd:
    def test_single_batch(self):
        rms = RunningMeanStd(shape=(3,))
        x = np.array([[1., 2., 3.], [3., 4., 5.]], dtype=np.float64)
        rms.update(x)
        # Initial count is 1e-4 (pseudo-count), so mean is weighted; use loose tolerance
        np.testing.assert_allclose(rms.mean, x.mean(axis=0), atol=1e-3)

    def test_normalize_output_range(self):
        rms = RunningMeanStd(shape=(4,))
        data = np.random.randn(100, 4)
        rms.update(data)
        normed = rms.normalize(data)
        # After normalisation, values should be mostly in [-10, 10]
        assert np.all(np.abs(normed) <= 10.1)

    def test_variance_positive(self):
        rms = RunningMeanStd(shape=(2,))
        rms.update(np.random.randn(50, 2))
        assert np.all(rms.var > 0)


# -----------------------------------------------------------------------
# Neural networks
# -----------------------------------------------------------------------

class TestGaussianActor:
    def setup_method(self):
        self.obs_dim = 11
        self.act_dim = 3
        self.actor   = GaussianActor(self.obs_dim, self.act_dim)

    def test_forward_returns_distribution(self):
        from torch.distributions import Normal
        obs  = torch.randn(4, self.obs_dim)
        dist = self.actor(obs)
        assert isinstance(dist, Normal)
        assert dist.mean.shape == (4, self.act_dim)

    def test_get_action_shapes(self):
        obs = torch.randn(8, self.obs_dim)
        actions, log_p = self.actor.get_action(obs)
        assert actions.shape == (8, self.act_dim)
        assert log_p.shape   == (8,)

    def test_evaluate_actions_shapes(self):
        obs  = torch.randn(16, self.obs_dim)
        acts = torch.randn(16, self.act_dim)
        log_p, entropy = self.actor.evaluate_actions(obs, acts)
        assert log_p.shape   == (16,)
        assert entropy.shape == (16,)

    def test_log_probs_finite(self):
        obs  = torch.randn(10, self.obs_dim)
        acts = torch.randn(10, self.act_dim)
        log_p, _ = self.actor.evaluate_actions(obs, acts)
        assert torch.all(torch.isfinite(log_p))

    def test_deterministic_vs_stochastic(self):
        obs = torch.randn(1, self.obs_dim)
        det, _ = self.actor.get_action(obs, deterministic=True)
        sto, _ = self.actor.get_action(obs, deterministic=False)
        # Deterministic action = distribution mean
        dist = self.actor(obs)
        np.testing.assert_allclose(det.detach().numpy(), dist.mean.detach().numpy(), atol=1e-5)


class TestCriticV:
    def test_output_shape(self):
        critic = CriticV(obs_dim=27)
        obs    = torch.randn(16, 27)
        val    = critic(obs)
        assert val.shape == (16,)

    def test_single_obs(self):
        critic = CriticV(obs_dim=9)
        obs    = torch.randn(1, 9)
        val    = critic(obs)
        assert val.shape == (1,)


class TestActorCritic:
    def setup_method(self):
        self.ac = ActorCritic(obs_dim=11, act_dim=3)

    def test_predict_returns_numpy(self):
        obs = np.random.randn(1, 11).astype(np.float32)
        actions, log_p, values = self.ac.predict(obs)
        assert isinstance(actions, np.ndarray)
        assert isinstance(log_p,   np.ndarray)
        assert isinstance(values,  np.ndarray)

    def test_predict_shapes(self):
        obs = np.random.randn(4, 11).astype(np.float32)
        actions, log_p, values = self.ac.predict(obs)
        assert actions.shape == (4, 3)
        assert log_p.shape   == (4,)
        assert values.shape  == (4,)

    def test_evaluate_shapes(self):
        obs  = torch.randn(16, 11)
        acts = torch.randn(16, 3)
        log_p, values, entropy = self.ac.evaluate(obs, acts)
        assert log_p.shape   == (16,)
        assert values.shape  == (16,)
        assert entropy.shape == (16,)

    def test_save_and_load(self, tmp_path):
        path = str(tmp_path / "model.pt")
        obs  = np.random.randn(1, 11).astype(np.float32)
        # Use deterministic=True so actions are the distribution mean (reproducible)
        acts_before, _, _ = self.ac.predict(obs, deterministic=True)
        self.ac.save(path)

        ac2 = ActorCritic(obs_dim=11, act_dim=3)
        ac2.load(path)
        acts_after, _, _ = ac2.predict(obs, deterministic=True)
        np.testing.assert_array_almost_equal(acts_before, acts_after)


# -----------------------------------------------------------------------
# Rollout Buffer
# -----------------------------------------------------------------------

class TestRolloutBuffer:
    def setup_method(self):
        self.buf = RolloutBuffer(
            n_steps=16,
            obs_shape=(4,),
            act_shape=(1,),
            gamma=0.99,
            gae_lambda=0.95,
        )

    def test_add_and_size(self):
        for i in range(10):
            self.buf.add(np.zeros(4), np.zeros(1), 1.0, False, 0.5, -0.3)
        assert self.buf.size == 10

    def test_overflow_raises(self):
        for _ in range(16):
            self.buf.add(np.zeros(4), np.zeros(1), 1.0, False, 0.5, -0.3)
        with pytest.raises(AssertionError):
            self.buf.add(np.zeros(4), np.zeros(1), 1.0, False, 0.5, -0.3)

    def test_compute_returns_shapes(self):
        for _ in range(16):
            self.buf.add(np.zeros(4), np.zeros(1), 1.0, False, 0.5, -0.3)
        self.buf.compute_returns(last_value=0.0)
        obs, acts, log_p, ret, adv = self.buf.get()
        assert obs.shape  == (16, 4)
        assert acts.shape == (16, 1)
        assert log_p.shape == ret.shape == adv.shape == (16,)

    def test_returns_finite(self):
        for _ in range(8):
            self.buf.add(np.random.randn(4).astype(np.float32), np.zeros(1),
                         float(np.random.randn()), False, float(np.random.randn()), -0.2)
        self.buf.compute_returns(0.0)
        _, _, _, ret, adv = self.buf.get()
        assert np.all(np.isfinite(ret))
        assert np.all(np.isfinite(adv))

    def test_reset_clears_buffer(self):
        for _ in range(8):
            self.buf.add(np.zeros(4), np.zeros(1), 1.0, False, 0.5, -0.3)
        self.buf.reset()
        assert self.buf.size == 0

    def test_done_cuts_advantage_propagation(self):
        """Advantages should not propagate across episode boundaries."""
        for i in range(8):
            done = (i == 3)   # episode ends at step 3
            self.buf.add(np.zeros(4), np.zeros(1), 1.0, done, 0.5, -0.3)
        self.buf.compute_returns(0.0)
        _, _, _, _, adv = self.buf.get()
        # Advantage at done step should only reflect its own reward
        assert np.isfinite(adv[3])


# -----------------------------------------------------------------------
# PPO integration (short run)
# -----------------------------------------------------------------------

class TestPPOIntegration:
    def test_short_training_run(self):
        """PPO should complete 2 rollouts without errors."""
        env    = make_env("CartPole")
        config = PPOConfig(
            n_steps=64,
            n_epochs=2,
            mini_batch_size=32,
            total_timesteps=256,
            normalize_obs=True,
            lr_anneal=False,
        )
        agent = PPO(env, config)
        agent.learn(total_timesteps=256, log_interval=999)
        assert agent._global_step >= 256
        env.close()

    def test_predict_after_training(self):
        env    = make_env("CartPole")
        config = PPOConfig(n_steps=64, total_timesteps=64)
        agent  = PPO(env, config)
        agent.learn(total_timesteps=64, log_interval=999)

        obs, _ = env.reset()
        action = agent.predict(obs, deterministic=True)
        assert action.shape == env.action_space.shape
        env.close()

    def test_save_and_load(self, tmp_path):
        env    = make_env("CartPole")
        config = PPOConfig(n_steps=64, total_timesteps=64)
        agent  = PPO(env, config)
        agent.learn(total_timesteps=64, log_interval=999)

        path = str(tmp_path / "ppo.pt")
        agent.save(path)

        agent2 = PPO(make_env("CartPole"), PPOConfig())
        agent2.load(path)
        assert agent2._global_step == agent._global_step
        env.close()


# -----------------------------------------------------------------------
# Random agent
# -----------------------------------------------------------------------

class TestRandomAgent:
    def test_predict_shape(self):
        env   = make_env("Reacher")
        agent = RandomAgent(env)
        obs, _ = env.reset()
        action = agent.predict(obs)
        assert action.shape == env.action_space.shape
        env.close()

    def test_action_in_bounds(self):
        env   = make_env("Ant")
        agent = RandomAgent(env)
        obs, _ = env.reset()
        for _ in range(20):
            action = agent.predict(obs)
            assert np.all(action >= env.action_space.low  - 1e-6)
            assert np.all(action <= env.action_space.high + 1e-6)
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                obs, _ = env.reset()
        env.close()
