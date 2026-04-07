"""
Tests for environment wrappers and vectorized environments.

Run with:
    python -m pytest tests/test_wrappers.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import gymnasium as gym

from envs          import make_env
from envs.wrappers import (
    TimeLimitWrapper, ClipActionWrapper, ObsNormWrapper,
    RewardScaleWrapper, ObsNoiseWrapper, ActionRepeatWrapper,
    RecordEpisodeStats, FrameStackWrapper, RandomisePhysicsWrapper, wrap,
)
from training.vec_env import DummyVecEnv, make_vec_env


# ──────────────────────────────────────────────────────────────────────────────
# TimeLimitWrapper
# ──────────────────────────────────────────────────────────────────────────────

class TestTimeLimitWrapper:

    def test_truncates_at_limit(self):
        env = TimeLimitWrapper(make_env("Reacher"), max_episode_steps=10)
        obs, _ = env.reset()
        truncated = False
        for _ in range(15):
            _, _, _, truncated, _ = env.step(env.action_space.sample())
            if truncated:
                break
        assert truncated
        env.close()

    def test_resets_counter(self):
        env = TimeLimitWrapper(make_env("CartPole"), max_episode_steps=5)
        env.reset()
        for _ in range(5):
            env.step(env.action_space.sample())
        # After reset, counter should restart
        env.reset()
        for _ in range(4):
            _, _, _, trunc, _ = env.step(env.action_space.sample())
            assert not trunc, "Should not truncate before step 5 after reset"
        env.close()


# ──────────────────────────────────────────────────────────────────────────────
# ClipActionWrapper
# ──────────────────────────────────────────────────────────────────────────────

class TestClipActionWrapper:

    def test_clips_large_actions(self):
        env = ClipActionWrapper(make_env("CartPole"))
        env.reset()
        huge = np.array([1e10], dtype=np.float32)
        obs, _, _, _, _ = env.step(huge)
        assert np.all(np.isfinite(obs))
        env.close()

    def test_clips_negative(self):
        env = ClipActionWrapper(make_env("Hopper"))
        env.reset()
        tiny = np.full(env.action_space.shape, -1e10, dtype=np.float32)
        obs, _, _, _, _ = env.step(tiny)
        assert np.all(np.isfinite(obs))
        env.close()


# ──────────────────────────────────────────────────────────────────────────────
# ObsNormWrapper
# ──────────────────────────────────────────────────────────────────────────────

class TestObsNormWrapper:

    def test_obs_dtype(self):
        env = ObsNormWrapper(make_env("CartPole"))
        obs, _ = env.reset()
        assert obs.dtype == np.float32
        env.close()

    def test_obs_clipped(self):
        env = ObsNormWrapper(make_env("CartPole"), clip_range=5.0)
        obs, _ = env.reset()
        assert np.all(np.abs(obs) <= 5.0 + 1e-6)
        for _ in range(50):
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            assert np.all(np.abs(obs) <= 5.0 + 1e-6)
            if term or trunc:
                obs, _ = env.reset()
        env.close()

    def test_running_stats_update(self):
        env = ObsNormWrapper(make_env("Reacher"))
        env.reset()
        for _ in range(100):
            obs, _, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc:
                env.reset()
        # Mean should have moved from zero
        assert not np.allclose(env._mean, 0.0)
        env.close()


# ──────────────────────────────────────────────────────────────────────────────
# RewardScaleWrapper
# ──────────────────────────────────────────────────────────────────────────────

class TestRewardScaleWrapper:

    def test_scale_10(self):
        base = make_env("CartPole")
        env  = RewardScaleWrapper(base, scale=10.0)
        base2 = make_env("CartPole")

        obs, _ = env.reset(seed=0)
        obs2, _ = base2.reset(seed=0)
        action = env.action_space.sample()

        _, r_scaled, _, _, _ = env.step(action)
        _, r_base, _, _, _   = base2.step(action)

        np.testing.assert_allclose(r_scaled, r_base * 10.0, rtol=1e-4)
        env.close()
        base2.close()

    def test_scale_zero(self):
        env = RewardScaleWrapper(make_env("CartPole"), scale=0.0)
        env.reset()
        _, r, _, _, _ = env.step(env.action_space.sample())
        assert r == 0.0
        env.close()


# ──────────────────────────────────────────────────────────────────────────────
# ObsNoiseWrapper
# ──────────────────────────────────────────────────────────────────────────────

class TestObsNoiseWrapper:

    def test_adds_noise(self):
        base = make_env("CartPole")
        noisy = ObsNoiseWrapper(make_env("CartPole"), noise_std=1.0, seed=0)

        obs_base, _ = base.reset(seed=42)
        obs_noisy, _ = noisy.reset(seed=42)

        # Observations should differ due to noise
        assert not np.allclose(obs_base, obs_noisy)
        base.close()
        noisy.close()

    def test_noise_zero_preserves(self):
        base  = make_env("CartPole")
        noisy = ObsNoiseWrapper(make_env("CartPole"), noise_std=0.0, seed=0)
        obs1, _ = base.reset(seed=7)
        obs2, _ = noisy.reset(seed=7)
        np.testing.assert_array_almost_equal(obs1, obs2)
        base.close()
        noisy.close()


# ──────────────────────────────────────────────────────────────────────────────
# ActionRepeatWrapper
# ──────────────────────────────────────────────────────────────────────────────

class TestActionRepeatWrapper:

    def test_k1_same_as_base(self):
        base = make_env("CartPole")
        rep  = ActionRepeatWrapper(make_env("CartPole"), k=1)
        base.reset(seed=0)
        rep.reset(seed=0)
        action = base.action_space.sample()
        _, r1, _, _, _ = base.step(action)
        _, r2, _, _, _ = rep.step(action)
        np.testing.assert_allclose(r1, r2, rtol=1e-4)
        base.close()
        rep.close()

    def test_k2_accumulates_reward(self):
        env = ActionRepeatWrapper(make_env("CartPole"), k=2)
        env.reset(seed=0)
        base = make_env("CartPole")
        base.reset(seed=0)
        action = env.action_space.sample()

        _, r_rep, _, _, _  = env.step(action)
        _, r1, _, _, _     = base.step(action)
        _, r2, _, _, _     = base.step(action)
        np.testing.assert_allclose(r_rep, r1 + r2, rtol=1e-4)
        env.close()
        base.close()


# ──────────────────────────────────────────────────────────────────────────────
# RecordEpisodeStats
# ──────────────────────────────────────────────────────────────────────────────

class TestRecordEpisodeStats:

    def test_episode_info_added(self):
        env = RecordEpisodeStats(make_env("CartPole"))
        env.reset()
        ep_info_seen = False
        for _ in range(2000):
            _, _, term, trunc, info = env.step(env.action_space.sample())
            if "episode" in info:
                ep_info_seen = True
                assert "r" in info["episode"]
                assert "l" in info["episode"]
                assert info["episode"]["l"] > 0
                break
            if term or trunc:
                env.reset()
        assert ep_info_seen
        env.close()

    def test_mean_episode_return(self):
        env = RecordEpisodeStats(make_env("CartPole"))
        env.reset()
        for _ in range(2000):
            _, _, term, trunc, _ = env.step(env.action_space.sample())
            if term or trunc:
                env.reset()
        assert env.mean_episode_return != 0.0
        env.close()


# ──────────────────────────────────────────────────────────────────────────────
# FrameStackWrapper
# ──────────────────────────────────────────────────────────────────────────────

class TestFrameStackWrapper:

    def test_obs_shape(self):
        base = make_env("CartPole")
        env  = FrameStackWrapper(base, k=3)
        obs_shape = env.observation_space.shape[0]
        assert obs_shape == 4 * 3   # 4-dim obs × 3 frames
        env.close()

    def test_reset_fills_buffer(self):
        env = FrameStackWrapper(make_env("CartPole"), k=4)
        obs, _ = env.reset()
        assert obs.shape == (4 * 4,)
        assert obs.dtype == np.float32
        env.close()

    def test_step_shape(self):
        env = FrameStackWrapper(make_env("Reacher"), k=2)
        env.reset()
        obs, _, _, _, _ = env.step(env.action_space.sample())
        assert obs.shape == (9 * 2,)
        env.close()


# ──────────────────────────────────────────────────────────────────────────────
# wrap() factory
# ──────────────────────────────────────────────────────────────────────────────

class TestWrapFactory:

    def test_wrap_time_limit(self):
        env = wrap(make_env("Reacher"), time_limit=20)
        env.reset()
        for _ in range(25):
            _, _, _, trunc, _ = env.step(env.action_space.sample())
            if trunc:
                break
        assert trunc
        env.close()

    def test_wrap_frame_stack_changes_obs_shape(self):
        base  = make_env("CartPole")
        orig_dim = base.observation_space.shape[0]
        env   = wrap(make_env("CartPole"), frame_stack=3)
        obs, _ = env.reset()
        assert obs.shape[0] == orig_dim * 3
        base.close()
        env.close()

    def test_wrap_record_stats(self):
        env = wrap(make_env("CartPole"), record_stats=True, time_limit=50)
        env.reset()
        for _ in range(200):
            _, _, term, trunc, info = env.step(env.action_space.sample())
            if term or trunc:
                env.reset()
            if "episode" in info:
                break
        env.close()


# ──────────────────────────────────────────────────────────────────────────────
# DummyVecEnv
# ──────────────────────────────────────────────────────────────────────────────

class TestDummyVecEnv:

    def setup_method(self):
        from envs import make_env as _make
        self.vec = DummyVecEnv([lambda: _make("CartPole") for _ in range(4)])

    def teardown_method(self):
        self.vec.close()

    def test_reset_shape(self):
        obs, infos = self.vec.reset(seed=0)
        assert obs.shape == (4, 4)
        assert len(infos) == 4

    def test_step_shapes(self):
        self.vec.reset()
        actions = np.random.randn(4, 1).astype(np.float32)
        obs, rews, terms, truncs, infos = self.vec.step(actions)
        assert obs.shape   == (4, 4)
        assert rews.shape  == (4,)
        assert terms.shape == (4,)
        assert truncs.shape == (4,)
        assert len(infos)  == 4

    def test_auto_reset_on_done(self):
        """Environments that finish should auto-reset."""
        self.vec.reset(seed=0)
        # Run until at least one env terminates
        for _ in range(2000):
            actions = np.zeros((4, 1), dtype=np.float32)
            obs, _, terms, truncs, infos = self.vec.step(actions)
            if any(terms) or any(truncs):
                # Next obs should be from a reset state (valid finite obs)
                assert np.all(np.isfinite(obs))
                break

    def test_n_envs(self):
        assert self.vec.n_envs == 4

    def test_make_vec_env_factory(self):
        venv = make_vec_env("Reacher", n_envs=2, vec_cls="dummy", seed=0)
        obs, _ = venv.reset()
        assert obs.shape == (2, 9)
        venv.close()
