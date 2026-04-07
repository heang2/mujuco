"""
Tests for all MuJoCo environments.

Run with:
    python -m pytest tests/test_envs.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
import gymnasium as gym

from envs import make_env, REGISTRY


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------

@pytest.fixture(params=list(REGISTRY.keys()))
def env_name(request):
    return request.param


@pytest.fixture
def env(env_name):
    e = make_env(env_name)
    yield e
    e.close()


# -----------------------------------------------------------------------
# Parametrised tests — run for every environment
# -----------------------------------------------------------------------

class TestEnvironmentInterface:

    def test_make_env(self, env_name):
        e = make_env(env_name)
        assert e is not None
        e.close()

    def test_observation_space_type(self, env):
        assert isinstance(env.observation_space, gym.spaces.Box)

    def test_action_space_type(self, env):
        assert isinstance(env.action_space, gym.spaces.Box)

    def test_reset_returns_obs_and_info(self, env):
        obs, info = env.reset(seed=0)
        assert isinstance(obs,  np.ndarray)
        assert isinstance(info, dict)

    def test_obs_shape_matches_space(self, env):
        obs, _ = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape, (
            f"obs.shape={obs.shape}, space.shape={env.observation_space.shape}"
        )

    def test_obs_dtype_is_float32(self, env):
        obs, _ = env.reset(seed=0)
        assert obs.dtype == np.float32, f"Expected float32, got {obs.dtype}"

    def test_step_with_zero_action(self, env):
        env.reset(seed=0)
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert isinstance(obs,       np.ndarray)
        assert isinstance(reward,    float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated,  bool)
        assert isinstance(info,       dict)

    def test_step_with_random_action(self, env):
        env.reset(seed=1)
        rng    = np.random.default_rng(1)
        action = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape

    def test_step_obs_in_space(self, env):
        """Observation should lie within declared observation space bounds (or close)."""
        obs, _ = env.reset(seed=2)
        rng = np.random.default_rng(2)
        for _ in range(20):
            action = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
        # Only check finite-bounded spaces
        lo, hi = env.observation_space.low, env.observation_space.high
        finite_mask = np.isfinite(lo) & np.isfinite(hi)
        if finite_mask.any():
            assert np.all(obs[finite_mask] >= lo[finite_mask] - 1e-3)
            assert np.all(obs[finite_mask] <= hi[finite_mask] + 1e-3)

    def test_action_clipping(self, env):
        """Environment should accept out-of-range actions (clipped internally)."""
        env.reset(seed=3)
        big_action = np.ones(env.action_space.shape, dtype=np.float32) * 1e6
        obs, reward, terminated, truncated, info = env.step(big_action)
        assert np.all(np.isfinite(obs))

    def test_deterministic_reset(self, env):
        """Same seed → same initial observation."""
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)

    def test_multiple_resets(self, env):
        """Should be able to reset repeatedly without error."""
        for seed in range(5):
            obs, _ = env.reset(seed=seed)
            assert obs.shape == env.observation_space.shape

    def test_full_episode(self, env):
        """Run one full episode to completion."""
        obs, _ = env.reset(seed=7)
        rng = np.random.default_rng(7)
        steps   = 0
        done    = False
        rewards = []

        while not done and steps < 2000:
            action = rng.uniform(env.action_space.low, env.action_space.high).astype(np.float32)
            obs, r, terminated, truncated, _ = env.step(action)
            rewards.append(r)
            done  = terminated or truncated
            steps += 1

        assert steps > 0
        assert all(np.isfinite(r) for r in rewards), "Non-finite rewards encountered"

    def test_dt_positive(self, env):
        assert env.dt > 0

    def test_info_has_step_key(self, env):
        env.reset()
        _, _, _, _, info = env.step(np.zeros(env.action_space.shape))
        assert "step" in info


# -----------------------------------------------------------------------
# Environment-specific tests
# -----------------------------------------------------------------------

class TestCartPole:
    def test_obs_dim(self):
        env = make_env("CartPole")
        assert env.observation_space.shape == (4,)
        env.close()

    def test_act_dim(self):
        env = make_env("CartPole")
        assert env.action_space.shape == (1,)
        env.close()

    def test_terminates_on_fall(self):
        from envs.cartpole_env import CartPoleEnv
        env = CartPoleEnv()
        env.reset(seed=0)
        # Force pole to large angle
        import mujoco
        hinge_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "hinge")
        env.data.qpos[hinge_id] = 1.0   # > 24° threshold
        mujoco.mj_forward(env.model, env.data)
        _, _, terminated, _, _ = env.step(np.zeros(1))
        assert terminated
        env.close()


class TestReacher:
    def test_obs_dim(self):
        env = make_env("Reacher")
        assert env.observation_space.shape == (9,)
        env.close()

    def test_act_dim(self):
        env = make_env("Reacher")
        assert env.action_space.shape == (2,)
        env.close()

    def test_distance_in_info(self):
        env = make_env("Reacher")
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(2))
        assert "dist_to_target" in info
        assert info["dist_to_target"] >= 0
        env.close()


class TestHopper:
    def test_obs_dim(self):
        env = make_env("Hopper")
        assert env.observation_space.shape == (11,)
        env.close()

    def test_act_dim(self):
        env = make_env("Hopper")
        assert env.action_space.shape == (3,)
        env.close()

    def test_health_info(self):
        env = make_env("Hopper")
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.zeros(3))
        assert "is_healthy" in info
        env.close()


class TestAnt:
    def test_act_dim(self):
        env = make_env("Ant")
        assert env.action_space.shape == (8,)
        env.close()

    def test_obs_has_contact_forces(self):
        env = make_env("Ant")
        obs, _ = env.reset(seed=0)
        # obs = (nq-2) + nv + nbody*6  (contact forces) or (nq-2) + nv without
        base = (env.model.nq - 2) + env.model.nv
        with_cf = base + env.model.nbody * 6
        assert obs.shape[0] in (base, with_cf), (
            f"obs dim {obs.shape[0]} not in ({base}, {with_cf})"
        )
        env.close()
