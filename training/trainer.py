"""
High-level training coordinator.

Wires together PPO, Evaluator, Logger, and checkpointing into a single
`Trainer` class that can be driven from a config dict or YAML file.
"""

import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

from agents.ppo     import PPO, PPOConfig
from envs           import make_env
from training.evaluator import Evaluator
from utils.logger   import Logger
from utils.plotting import plot_training_curves


class Trainer:
    """
    Orchestrates a complete PPO training run.

    Usage:
        trainer = Trainer.from_yaml("configs/hopper_ppo.yaml")
        trainer.train()
    """

    def __init__(self, config: Dict[str, Any]):
        self.config   = config
        self.env_name = config["env_name"]
        self.run_name = config.get("run_name", f"{self.env_name}_{int(time.time())}")
        self.log_dir  = Path(config.get("log_dir", "logs")) / self.run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Save the config used for this run
        with open(self.log_dir / "config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        # Build environment
        self.env = make_env(self.env_name)

        # Build PPO config from the training section
        ppo_cfg = config.get("ppo", {})
        self.ppo_config = PPOConfig(
            actor_hidden     = ppo_cfg.get("actor_hidden",     [256, 256]),
            critic_hidden    = ppo_cfg.get("critic_hidden",    [256, 256]),
            n_steps          = ppo_cfg.get("n_steps",          2048),
            n_epochs         = ppo_cfg.get("n_epochs",         10),
            mini_batch_size  = ppo_cfg.get("mini_batch_size",  64),
            clip_eps         = ppo_cfg.get("clip_eps",         0.2),
            vf_coef          = ppo_cfg.get("vf_coef",          0.5),
            ent_coef         = ppo_cfg.get("ent_coef",         0.0),
            max_grad_norm    = ppo_cfg.get("max_grad_norm",    0.5),
            gamma            = ppo_cfg.get("gamma",            0.99),
            gae_lambda       = ppo_cfg.get("gae_lambda",       0.95),
            lr               = float(ppo_cfg.get("lr",         3e-4)),
            lr_anneal        = ppo_cfg.get("lr_anneal",        True),
            total_timesteps  = config.get("total_timesteps",   1_000_000),
            normalize_obs    = ppo_cfg.get("normalize_obs",    True),
            normalize_rewards = ppo_cfg.get("normalize_rewards", False),
            seed             = config.get("seed", 42),
        )

        self.agent    = PPO(self.env, self.ppo_config)
        self.logger   = Logger(str(self.log_dir))
        self.evaluator = Evaluator(
            self.env_name,
            n_episodes=config.get("eval_episodes", 10),
        )

        self._eval_interval = config.get("eval_interval", 50_000)
        self._save_interval = config.get("save_interval", 100_000)
        self._last_eval_step = 0

    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str) -> "Trainer":
        with open(path) as f:
            config = yaml.safe_load(f)
        return cls(config)

    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training loop."""
        print(f"\n{'='*60}")
        print(f"  Training {self.env_name}  →  {self.ppo_config.total_timesteps:,} steps")
        print(f"  Run name : {self.run_name}")
        print(f"  Log dir  : {self.log_dir}")
        print(f"{'='*60}\n")

        self.agent.callbacks.append(self._eval_callback)

        self.agent.learn(
            total_timesteps=self.ppo_config.total_timesteps,
            log_interval=1,
            save_dir=str(self.log_dir / "checkpoints"),
            save_freq=self._save_interval,
        )

        # Final evaluation
        print("\n--- Final Evaluation ---")
        eval_result = self.evaluator.evaluate(self.agent)
        self._print_eval(eval_result, self.agent._global_step)
        self.logger.log_eval(eval_result, self.agent._global_step)

        # Save final model
        final_path = self.log_dir / "final_model.pt"
        self.agent.save(str(final_path))
        print(f"Saved final model → {final_path}")

        # Plot training curves
        curve_path = self.log_dir / "training_curves.png"
        plot_training_curves(
            self.logger.reward_history,
            self.logger.eval_history,
            self.env_name,
            str(curve_path),
        )
        print(f"Saved training curves → {curve_path}")

        self.env.close()
        self.evaluator.close()

    # ------------------------------------------------------------------

    def _eval_callback(self) -> None:
        """Called periodically from the training loop."""
        step = self.agent._global_step
        if step - self._last_eval_step < self._eval_interval:
            return
        self._last_eval_step = step

        result = self.evaluator.evaluate(self.agent)
        self._print_eval(result, step)
        self.logger.log_eval(result, step)

        ckpt = self.log_dir / "checkpoints" / f"eval_{step}.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        self.agent.save(str(ckpt))

    @staticmethod
    def _print_eval(result: Dict, step: int) -> None:
        print(
            f"  [Eval @ {step:>8,}]  "
            f"mean_rew={result['mean_reward']:>8.2f} ± {result['std_reward']:.2f}  "
            f"min={result['min_reward']:.2f}  max={result['max_reward']:.2f}"
        )


__all__ = ["Trainer"]
