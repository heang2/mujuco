#!/usr/bin/env python
"""
Train a SAC or TD3 agent on a MuJoCo environment.

Usage:
    python scripts/train_sac.py --config configs/hopper_sac.yaml
    python scripts/train_sac.py --env Hopper --algo sac --steps 500000
    python scripts/train_sac.py --env Reacher --algo td3 --steps 300000 --seed 1
"""

import sys
import argparse
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml

from envs          import make_env
from agents.sac    import SAC, SACConfig
from agents.td3    import TD3, TD3Config
from training.evaluator import Evaluator
from utils.logger  import Logger
from utils.plotting import plot_training_curves


def parse_args():
    p = argparse.ArgumentParser(description="Train SAC or TD3 on a MuJoCo environment")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--env",    type=str, default="Hopper",
                   choices=["CartPole", "Reacher", "Hopper", "Ant", "Walker2D", "Pusher"])
    p.add_argument("--algo",   type=str, default="sac", choices=["sac", "td3"])
    p.add_argument("--steps",  type=int, default=None)
    p.add_argument("--seed",   type=int, default=42)
    p.add_argument("--lr",     type=float, default=None)
    p.add_argument("--no-per", action="store_true", help="Disable PER for SAC")
    return p.parse_args()


def build_sac_config_from_dict(d: dict, total_steps: int) -> SACConfig:
    s = d.get("sac", {})
    return SACConfig(
        hidden_sizes      = s.get("hidden_sizes",     [256, 256]),
        gamma             = s.get("gamma",            0.99),
        tau               = s.get("tau",              5e-3),
        lr_actor          = float(s.get("lr_actor",   3e-4)),
        lr_critic         = float(s.get("lr_critic",  3e-4)),
        lr_alpha          = float(s.get("lr_alpha",   3e-4)),
        batch_size        = s.get("batch_size",       256),
        replay_capacity   = s.get("replay_capacity",  1_000_000),
        learning_starts   = s.get("learning_starts",  10_000),
        update_every      = s.get("update_every",     1),
        gradient_steps    = s.get("gradient_steps",   1),
        auto_tune_alpha   = s.get("auto_tune_alpha",  True),
        init_alpha        = float(s.get("init_alpha", 0.2)),
        use_per           = s.get("use_per",          False),
        per_alpha         = s.get("per_alpha",        0.6),
        per_beta_init     = s.get("per_beta_init",    0.4),
        per_beta_steps    = s.get("per_beta_steps",   1_000_000),
        total_timesteps   = total_steps,
        seed              = d.get("seed", 42),
    )


def build_td3_config_from_dict(d: dict, total_steps: int) -> TD3Config:
    t = d.get("td3", {})
    return TD3Config(
        hidden_sizes    = t.get("hidden_sizes",   [256, 256]),
        gamma           = t.get("gamma",          0.99),
        tau             = t.get("tau",            5e-3),
        lr_actor        = float(t.get("lr_actor", 3e-4)),
        lr_critic       = float(t.get("lr_critic", 3e-4)),
        batch_size      = t.get("batch_size",     256),
        replay_capacity = t.get("replay_capacity", 1_000_000),
        learning_starts = t.get("learning_starts", 10_000),
        update_every    = t.get("update_every",   1),
        policy_delay    = t.get("policy_delay",   2),
        expl_noise      = t.get("expl_noise",     0.1),
        noise_clip      = t.get("noise_clip",     0.5),
        policy_noise    = t.get("policy_noise",   0.2),
        total_timesteps = total_steps,
        seed            = d.get("seed", 42),
    )


def main():
    args = parse_args()

    # Load config
    if args.config:
        with open(args.config) as f:
            config_dict = yaml.safe_load(f)
    else:
        config_dict = {
            "env_name":         args.env,
            "algorithm":        args.algo,
            "total_timesteps":  args.steps or 500_000,
            "seed":             args.seed,
            "eval_episodes":    10,
            "eval_interval":    25_000,
        }

    env_name  = config_dict["env_name"]
    algo      = config_dict.get("algorithm", args.algo)
    total_steps = args.steps or config_dict.get("total_timesteps", 500_000)
    seed      = args.seed or config_dict.get("seed", 42)
    run_name  = f"{env_name}_{algo.upper()}_{int(time.time())}"
    log_dir   = Path(config_dict.get("log_dir", "logs")) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  {algo.upper()} Training — {env_name}")
    print(f"  Steps    : {total_steps:,}")
    print(f"  Seed     : {seed}")
    print(f"  Log dir  : {log_dir}")
    print(f"{'='*60}\n")

    env = make_env(env_name)

    if algo == "sac":
        sac_cfg = build_sac_config_from_dict(config_dict, total_steps)
        if args.lr:
            sac_cfg.lr_actor = sac_cfg.lr_critic = sac_cfg.lr_alpha = args.lr
        if args.no_per:
            sac_cfg.use_per = False
        agent = SAC(env, sac_cfg)
    elif algo == "td3":
        td3_cfg = build_td3_config_from_dict(config_dict, total_steps)
        if args.lr:
            td3_cfg.lr_actor = td3_cfg.lr_critic = args.lr
        agent = TD3(env, td3_cfg)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    # Setup logger
    logger    = Logger(str(log_dir))
    evaluator = Evaluator(env_name, n_episodes=config_dict.get("eval_episodes", 10))
    eval_interval = config_dict.get("eval_interval", 25_000)
    last_eval     = 0

    def eval_callback():
        nonlocal last_eval
        step = agent._global_step
        if step - last_eval >= eval_interval:
            last_eval = step
            result = evaluator.evaluate(agent)
            logger.log_eval(result, step)
            print(
                f"  [Eval @ {step:>8,}]  "
                f"mean={result['mean_reward']:.2f} ± {result['std_reward']:.2f}"
            )

    agent.callbacks.append(eval_callback)
    agent.learn(
        total_timesteps=total_steps,
        log_interval=5000,
        save_dir=str(log_dir / "checkpoints"),
        save_freq=config_dict.get("save_interval", 100_000),
    )

    # Final eval
    final = evaluator.evaluate(agent)
    print(f"\nFinal:  mean={final['mean_reward']:.2f} ± {final['std_reward']:.2f}")

    agent.save(str(log_dir / "final_model.pt"))

    # Plot
    plot_training_curves(
        logger.reward_history, logger.eval_history,
        env_name, str(log_dir / "training_curves.png"),
    )

    env.close()
    evaluator.close()


if __name__ == "__main__":
    main()
