"""
Vectorised environments — run N environments in parallel for faster data collection.

Two implementations:
  DummyVecEnv  — runs envs sequentially in the same process.
                 Zero overhead, great for debugging.
  SubprocVecEnv — runs each env in a separate subprocess.
                  True parallelism; 3-8× speedup on multi-core machines.

Both implement the same interface:
    obs, infos = vec_env.reset()
    obs, rews, terminateds, truncateds, infos = vec_env.step(actions)

Usage:
    from training.vec_env import make_vec_env

    # 8 parallel Hopper environments
    vec_env = make_vec_env("Hopper", n_envs=8, vec_cls="subprocess")
    obs, _ = vec_env.reset()

    # actions shape: (8, act_dim)
    actions = np.random.randn(8, vec_env.action_space.shape[0])
    obs, rews, terms, truncs, infos = vec_env.step(actions)
"""

import multiprocessing as mp
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Abstract base
# ──────────────────────────────────────────────────────────────────────────────

class VecEnv(ABC):
    """Abstract vectorised environment."""

    def __init__(self, n_envs: int, observation_space: gym.Space, action_space: gym.Space):
        self.n_envs            = n_envs
        self.observation_space = observation_space
        self.action_space      = action_space

    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, List[Dict]]:
        """Reset all envs. Returns (obs_batch, info_list)."""

    @abstractmethod
    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Step all envs. Returns (obs, rews, terminateds, truncateds, infos)."""

    @abstractmethod
    def close(self) -> None:
        """Clean up resources."""

    @property
    def obs_shape(self) -> Tuple:
        return self.observation_space.shape

    @property
    def act_shape(self) -> Tuple:
        return self.action_space.shape


# ──────────────────────────────────────────────────────────────────────────────
# DummyVecEnv
# ──────────────────────────────────────────────────────────────────────────────

class DummyVecEnv(VecEnv):
    """
    Sequential vectorised environment (no multiprocessing).

    Advantages:
      - Zero setup overhead
      - Easy to debug (same process, no pickling)
      - Works with any environment

    Disadvantages:
      - No true parallelism
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        self._envs = [fn() for fn in env_fns]
        super().__init__(
            n_envs=len(self._envs),
            observation_space=self._envs[0].observation_space,
            action_space=self._envs[0].action_space,
        )
        obs_shape = self.observation_space.shape
        self._obs_buf  = np.zeros((self.n_envs, *obs_shape), dtype=np.float32)
        self._rew_buf  = np.zeros(self.n_envs, dtype=np.float32)
        self._term_buf = np.zeros(self.n_envs, dtype=bool)
        self._trunc_buf = np.zeros(self.n_envs, dtype=bool)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, List[Dict]]:
        infos = []
        for i, env in enumerate(self._envs):
            s = seed + i if seed is not None else None
            obs, info = env.reset(seed=s)
            self._obs_buf[i] = obs
            infos.append(info)
        return self._obs_buf.copy(), infos

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        infos = []
        for i, env in enumerate(self._envs):
            obs, rew, term, trunc, info = env.step(actions[i])
            if term or trunc:
                info["terminal_obs"] = obs
                obs, _ = env.reset()
            self._obs_buf[i]  = obs
            self._rew_buf[i]  = rew
            self._term_buf[i] = term
            self._trunc_buf[i] = trunc
            infos.append(info)
        return (
            self._obs_buf.copy(),
            self._rew_buf.copy(),
            self._term_buf.copy(),
            self._trunc_buf.copy(),
            infos,
        )

    def close(self) -> None:
        for env in self._envs:
            env.close()

    def render(self) -> Optional[np.ndarray]:
        return self._envs[0].render()


# ──────────────────────────────────────────────────────────────────────────────
# SubprocVecEnv worker
# ──────────────────────────────────────────────────────────────────────────────

def _worker(conn, env_fn_pickle):
    """Subprocess worker: receives commands, sends responses."""
    import cloudpickle
    env_fn = cloudpickle.loads(env_fn_pickle)
    env    = env_fn()

    try:
        while True:
            cmd, data = conn.recv()

            if cmd == "step":
                obs, rew, term, trunc, info = env.step(data)
                if term or trunc:
                    info["terminal_obs"] = obs
                    obs, _ = env.reset()
                conn.send((obs, rew, term, trunc, info))

            elif cmd == "reset":
                obs, info = env.reset(**data)
                conn.send((obs, info))

            elif cmd == "spaces":
                conn.send((env.observation_space, env.action_space))

            elif cmd == "close":
                env.close()
                conn.close()
                break

            elif cmd == "render":
                conn.send(env.render())

            else:
                raise ValueError(f"Unknown command: {cmd}")
    except EOFError:
        pass


class SubprocVecEnv(VecEnv):
    """
    True parallel vectorised environment using subprocesses.

    Each environment runs in its own process; communication via Pipes.
    Requires cloudpickle: pip install cloudpickle

    On Windows, uses 'spawn' start method (slower startup but stable).
    On Unix, uses 'fork' (fast startup).
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        try:
            import cloudpickle
        except ImportError:
            raise ImportError(
                "SubprocVecEnv requires cloudpickle. "
                "Install with: pip install cloudpickle"
            )

        ctx    = mp.get_context("spawn")
        self._conns:    List[mp.connection.Connection] = []
        self._processes: List[mp.Process] = []

        for fn in env_fns:
            parent_conn, child_conn = ctx.Pipe()
            p = ctx.Process(
                target=_worker,
                args=(child_conn, cloudpickle.dumps(fn)),
                daemon=True,
            )
            p.start()
            child_conn.close()
            self._conns.append(parent_conn)
            self._processes.append(p)

        # Fetch spaces from first worker
        self._conns[0].send(("spaces", None))
        obs_space, act_space = self._conns[0].recv()

        super().__init__(
            n_envs=len(env_fns),
            observation_space=obs_space,
            action_space=act_space,
        )

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, List[Dict]]:
        for i, conn in enumerate(self._conns):
            s = {"seed": seed + i} if seed is not None else {}
            conn.send(("reset", s))
        results  = [conn.recv() for conn in self._conns]
        obs_list = [r[0] for r in results]
        info_list = [r[1] for r in results]
        return np.stack(obs_list).astype(np.float32), info_list

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        for conn, action in zip(self._conns, actions):
            conn.send(("step", action))
        results = [conn.recv() for conn in self._conns]
        obs_arr  = np.stack([r[0] for r in results]).astype(np.float32)
        rew_arr  = np.array([r[1] for r in results], dtype=np.float32)
        term_arr = np.array([r[2] for r in results], dtype=bool)
        trunc_arr = np.array([r[3] for r in results], dtype=bool)
        info_list = [r[4] for r in results]
        return obs_arr, rew_arr, term_arr, trunc_arr, info_list

    def close(self) -> None:
        for conn in self._conns:
            conn.send(("close", None))
        for p in self._processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

    def render(self) -> Optional[np.ndarray]:
        self._conns[0].send(("render", None))
        return self._conns[0].recv()


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def make_vec_env(
    env_name: str,
    n_envs: int = 4,
    vec_cls: str = "dummy",
    seed: int = 0,
    **env_kwargs,
) -> VecEnv:
    """
    Create a vectorised environment.

    Args:
        env_name:  Environment name (e.g. "Hopper")
        n_envs:    Number of parallel environments
        vec_cls:   "dummy" or "subprocess"
        seed:      Base random seed (each env gets seed + i)
        **env_kwargs: Passed to make_env()

    Returns:
        VecEnv instance
    """
    from envs import make_env

    def make_fn(i: int) -> Callable:
        def _fn():
            env = make_env(env_name, **env_kwargs)
            env.reset(seed=seed + i)
            return env
        return _fn

    fns = [make_fn(i) for i in range(n_envs)]

    if vec_cls == "dummy":
        return DummyVecEnv(fns)
    elif vec_cls == "subprocess":
        return SubprocVecEnv(fns)
    else:
        raise ValueError(f"Unknown vec_cls '{vec_cls}'. Choose 'dummy' or 'subprocess'.")
