#!/usr/bin/env python
"""
Quick demo — runs all 4 environments for a few steps with a random policy
and prints observations, rewards, and infos to confirm everything works.

Usage:
    python scripts/demo.py
    python scripts/demo.py --env Hopper --steps 100
    python scripts/demo.py --benchmark          # time 1000 env steps for each env
"""

import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from envs import make_env, REGISTRY
from agents.random_agent import RandomAgent


BANNER = """
╔══════════════════════════════════════════════════════╗
║          MuJoCo Robotics Playground — Demo           ║
╚══════════════════════════════════════════════════════╝
"""


def demo_env(env_name: str, n_steps: int = 50) -> None:
    print(f"\n{'─'*54}")
    print(f"  {env_name}")
    print(f"{'─'*54}")

    env   = make_env(env_name)
    agent = RandomAgent(env)

    obs, info = env.reset(seed=0)
    print(f"  obs_space  : {env.observation_space}")
    print(f"  act_space  : {env.action_space}")
    print(f"  obs shape  : {obs.shape}  dtype={obs.dtype}")
    print(f"  obs sample : {obs[:5].round(4)}{'...' if len(obs) > 5 else ''}")

    total_r, ep_steps, ep_count = 0.0, 0, 0
    t0 = time.time()

    for _ in range(n_steps):
        action = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_r  += reward
        ep_steps += 1

        if terminated or truncated:
            ep_count += 1
            obs, _ = env.reset()
            ep_steps = 0

    elapsed = time.time() - t0
    fps     = n_steps / max(elapsed, 1e-6)

    print(f"\n  {n_steps} steps in {elapsed:.3f}s  ({fps:.0f} fps)")
    print(f"  total reward : {total_r:.3f}")
    print(f"  episodes     : {ep_count}")
    print(f"  last info    : {info}")

    env.close()


def benchmark(n_steps: int = 1000) -> None:
    print(f"\nBenchmarking {n_steps} steps per environment...")
    print(f"{'ENV':<12}  {'FPS':>8}  {'ms/step':>10}")
    print("─" * 35)

    for name in REGISTRY:
        env   = make_env(name)
        agent = RandomAgent(env)
        obs, _ = env.reset(seed=0)

        t0 = time.perf_counter()
        for _ in range(n_steps):
            action = agent.predict(obs)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
        elapsed = time.perf_counter() - t0

        fps = n_steps / elapsed
        ms  = elapsed / n_steps * 1000
        print(f"{name:<12}  {fps:>8.0f}  {ms:>10.3f}")
        env.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env",       type=str, default=None,
                   help="Single env to demo (default: all)")
    p.add_argument("--steps",     type=int, default=50)
    p.add_argument("--benchmark", action="store_true",
                   help="Run speed benchmark instead of demo")
    p.add_argument("--bench-steps", type=int, default=1000)
    return p.parse_args()


def main():
    print(BANNER)
    args = parse_args()

    if args.benchmark:
        benchmark(args.bench_steps)
        return

    envs = [args.env] if args.env else list(REGISTRY.keys())
    for name in envs:
        demo_env(name, n_steps=args.steps)

    print(f"\n{'═'*54}")
    print("  All environments verified successfully.")
    print(f"{'═'*54}\n")


if __name__ == "__main__":
    main()
