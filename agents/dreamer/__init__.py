"""
DreamerV3 — Model-based RL via world model learning.

Reference:
    Hafner et al., "Mastering Diverse Domains through World Models" (2023)
    https://arxiv.org/abs/2301.04104

Usage:
    from agents.dreamer import DreamerV3, DreamerConfig

    env    = make_env("Hopper")
    config = DreamerConfig(total_steps=1_000_000)
    agent  = DreamerV3(env, config)
    agent.learn()
"""

from agents.dreamer.config  import DreamerConfig
from agents.dreamer.world_model import WorldModel, RSSM
from agents.dreamer.actor_critic import DreamerActor, DreamerCritic
from agents.dreamer.dreamer import DreamerV3

__all__ = [
    "DreamerV3",
    "DreamerConfig",
    "WorldModel",
    "RSSM",
    "DreamerActor",
    "DreamerCritic",
]
