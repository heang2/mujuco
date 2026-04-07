from agents.ppo            import PPO, PPOConfig
from agents.sac            import SAC, SACConfig
from agents.td3            import TD3, TD3Config
from agents.random_agent   import RandomAgent
from agents.networks       import ActorCritic, GaussianActor, CriticV
from agents.replay_buffer  import UniformReplayBuffer, PrioritizedReplayBuffer
from agents.dreamer        import DreamerV3, DreamerConfig

ALGORITHM_REGISTRY = {
    "ppo":     (PPO,      PPOConfig),
    "sac":     (SAC,      SACConfig),
    "td3":     (TD3,      TD3Config),
    "dreamer": (DreamerV3, DreamerConfig),
}

__all__ = [
    "PPO", "PPOConfig",
    "SAC", "SACConfig",
    "TD3", "TD3Config",
    "DreamerV3", "DreamerConfig",
    "RandomAgent",
    "ActorCritic", "GaussianActor", "CriticV",
    "UniformReplayBuffer", "PrioritizedReplayBuffer",
    "ALGORITHM_REGISTRY",
]
