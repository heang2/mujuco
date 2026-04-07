"""
MuJoCo robot environments.

Usage:
    from envs import make_env
    env = make_env("CartPole")
"""

from envs.cartpole_env  import CartPoleEnv
from envs.reacher_env   import ReacherEnv
from envs.hopper_env    import HopperEnv
from envs.ant_env       import AntEnv
from envs.walker2d_env  import Walker2DEnv
from envs.pusher_env    import PusherEnv

REGISTRY: dict = {
    "CartPole": CartPoleEnv,
    "Reacher":  ReacherEnv,
    "Hopper":   HopperEnv,
    "Ant":      AntEnv,
    "Walker2D": Walker2DEnv,
    "Pusher":   PusherEnv,
}


def make_env(env_name: str, **kwargs):
    """
    Create an environment by name.

    Args:
        env_name: One of "CartPole", "Reacher", "Hopper", "Ant"
        **kwargs: Passed to the environment constructor.

    Returns:
        A gymnasium.Env instance.
    """
    if env_name not in REGISTRY:
        raise ValueError(
            f"Unknown environment '{env_name}'. "
            f"Available: {list(REGISTRY.keys())}"
        )
    return REGISTRY[env_name](**kwargs)


__all__ = [
    "CartPoleEnv", "ReacherEnv", "HopperEnv", "AntEnv",
    "Walker2DEnv", "PusherEnv",
    "REGISTRY", "make_env",
]
