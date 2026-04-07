#!/usr/bin/env python
"""
Full policy analysis — action distributions, state visitation, trajectories.

Usage:
    python scripts/analyze_policy.py \\
        --checkpoint logs/Hopper_PPO_*/final_model.pt \\
        --env Hopper \\
        --algo ppo \\
        --output analysis/Hopper_PPO

    python scripts/analyze_policy.py \\
        --checkpoint logs/Reacher_SAC_*/final_model.pt \\
        --env Reacher --algo sac
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from envs          import make_env
from agents.ppo    import PPO, PPOConfig
from agents.sac    import SAC, SACConfig
from agents.td3    import TD3, TD3Config
from analysis.policy_analyzer import PolicyAnalyzer


def load_agent(checkpoint: str, env_name: str, algo: str):
    env = make_env(env_name)
    if algo == "ppo":
        agent = PPO(env, PPOConfig())
        agent.load(checkpoint)
    elif algo == "sac":
        agent = SAC(env, SACConfig())
        agent.load(checkpoint)
    elif algo == "td3":
        agent = TD3(env, TD3Config())
        agent.load(checkpoint)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    env.close()
    return agent


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--env",        type=str, required=True)
    p.add_argument("--algo",       type=str, default="ppo",
                   choices=["ppo", "sac", "td3"])
    p.add_argument("--episodes",   type=int, default=50)
    p.add_argument("--output",     type=str, default=None)
    p.add_argument("--stochastic", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    out  = args.output or f"analysis/{args.env}_{args.algo.upper()}"

    print(f"Loading {args.algo.upper()} from: {args.checkpoint}")
    agent = load_agent(args.checkpoint, args.env, args.algo)
    env   = make_env(args.env)

    analyzer = PolicyAnalyzer(
        env=env,
        agent=agent,
        n_episodes=args.episodes,
        deterministic=not args.stochastic,
    )
    summary = analyzer.run_full_analysis(out)

    env.close()


if __name__ == "__main__":
    main()
