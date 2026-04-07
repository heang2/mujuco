# MuJoCo Robotics Playground

A self-contained reinforcement learning suite built on **MuJoCo 3** and **PyTorch**.  
Four custom robot environments, PPO implemented from scratch, a complete training pipeline, and visualization tools — all ready to clone and run.

---

## Environments

| Environment | Task | Obs | Act | Difficulty |
|-------------|------|-----|-----|------------|
| **CartPole** | Balance an inverted pendulum on a sliding cart | 4 | 1 | Easy |
| **Reacher** | Move a 2-DOF planar arm to a random target | 9 | 2 | Medium |
| **Hopper** | 1-legged robot hops forward as fast as possible | 11 | 3 | Hard |
| **Walker2D** | 2-legged planar walker (+x direction) | 17 | 6 | Hard |
| **Pusher** | Robotic arm pushes an object to a goal | 23 | 7 | Hard |
| **Ant** | 4-legged robot locomotion (+x direction) | 27/111 | 8 | Very Hard |

All environments implement the `gymnasium.Env` interface and can be used with any compatible RL library.

---

## Project Structure

```
mujoco-robotics-playground/
├── models/              # MuJoCo XML robot definitions
│   ├── cartpole.xml
│   ├── reacher.xml
│   ├── hopper.xml
│   └── ant.xml
├── envs/                # Gymnasium-compatible environments
│   ├── base_env.py      # Abstract base class
│   ├── cartpole_env.py
│   ├── reacher_env.py
│   ├── hopper_env.py
│   └── ant_env.py
├── agents/              # RL algorithms
│   ├── networks.py      # Actor-Critic neural networks
│   ├── ppo.py           # PPO-Clip from scratch
│   ├── sac.py           # Soft Actor-Critic (off-policy)
│   ├── td3.py           # Twin-Delayed DDPG
│   ├── replay_buffer.py # Uniform + Prioritized replay
│   ├── random_agent.py  # Random baseline
│   └── dreamer/         # DreamerV3 world-model RL
│       ├── config.py
│       ├── world_model.py   # Encoder, RSSM, Decoder, heads
│       ├── actor_critic.py  # Latent-space AC + ReturnNormalizer
│       └── dreamer.py       # Main training loop
├── training/
│   ├── rollout_buffer.py  # GAE experience buffer
│   ├── trainer.py         # Training coordinator
│   └── evaluator.py       # Evaluation utilities
├── utils/
│   ├── logger.py          # CSV logging
│   └── plotting.py        # Training curve plots
├── configs/             # YAML hyperparameter configs
│   ├── cartpole_ppo.yaml
│   ├── reacher_ppo.yaml
│   ├── hopper_ppo.yaml
│   └── ant_ppo.yaml
├── scripts/             # Entry-point scripts
│   ├── train.py
│   ├── evaluate.py
│   ├── demo.py
│   └── benchmark.py
├── notebooks/           # Jupyter tutorials
│   └── 01_quickstart.ipynb
├── src/
│   ├── c/               # C extensions (fast GAE, running stats)
│   ├── cython/          # Cython GAE extension
│   │   └── fast_gae.pyx
│   └── rust/            # Rust replay buffer (PyO3)
│       └── src/
│           ├── uniform.rs
│           ├── per.rs       # Prioritized Experience Replay
│           ├── episode.rs   # Episode buffer for DreamerV3
│           └── sum_tree.rs
└── tests/               # pytest test suite
    ├── test_envs.py
    ├── test_agents.py
    ├── test_sac_td3.py
    └── test_wrappers.py
```

---

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
# or editable install:
pip install -e ".[dev]"
```

### 2. Verify everything works

```bash
python scripts/demo.py
```

Expected output:
```
  CartPole   obs_space: Box([-inf ...], [inf ...], (4,), float32)   ✓
  Reacher    obs_space: Box([-inf ...], [inf ...], (9,), float32)   ✓
  Hopper     obs_space: Box([-inf ...], [inf ...], (11,), float32)  ✓
  Ant        obs_space: Box([-inf ...], [inf ...], (111,), float32) ✓
```

### 3. Run the test suite

```bash
python -m pytest tests/ -v
```

### 4. Train an agent

```bash
# Using a pre-made config:
python scripts/train.py --config configs/hopper_ppo.yaml

# Quick CLI override:
python scripts/train.py --env CartPole --steps 200000 --seed 1
```

### 5. Evaluate a checkpoint

```bash
python scripts/evaluate.py \
    --checkpoint logs/CartPole_*/final_model.pt \
    --env CartPole \
    --episodes 50
```

### 6. Benchmark all environments

```bash
# Compare PPO vs random baseline:
python scripts/benchmark.py --log-dir logs/

# Speed test only (no trained models needed):
python scripts/benchmark.py --random-only
```

---

## PPO Implementation

The PPO agent (`agents/ppo.py`) is implemented entirely from scratch using PyTorch, with no dependency on external RL libraries.

**Key features:**
- Clipped surrogate objective (ε = 0.2)
- Generalized Advantage Estimation (GAE-λ)
- Separate Actor and Critic networks (orthogonal initialization)
- Diagonal Gaussian action distribution (learned log-std)
- Running observation normalization (Welford online algorithm)
- Gradient norm clipping
- Linear learning rate annealing
- Early stopping based on approximate KL divergence

**Architecture:**
```
Observation → [256 → Tanh → 256 → Tanh] → Mean (Actor)
                                          → LogStd (learnable param)
           → [256 → Tanh → 256 → Tanh → 1] (Critic)
```

---

## Training Details

### CartPole (200k steps, ~2 min)
- Converges to near-perfect balance within ~50k steps
- Expected final reward: ~1700 / episode

### Reacher (500k steps, ~5 min)
- Distance-based dense reward; success defined as dist < 2.5 cm
- Expected final success rate: ~90%

### Hopper (1M steps, ~15 min)
- Reward = forward velocity + alive bonus − control cost
- Expected final reward: ~2500 / episode

### Ant (3M steps, ~45 min)
- High-dimensional locomotion (8 DOF, 111-dim observation with contact forces)
- Expected final reward: ~5000 / episode

---

## Using Environments Standalone

```python
from envs import make_env
import numpy as np

env = make_env("Hopper")
obs, info = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

## Using the PPO Agent

```python
from envs        import make_env
from agents.ppo  import PPO, PPOConfig

env    = make_env("CartPole")
config = PPOConfig(
    total_timesteps=200_000,
    n_steps=1024,
    lr=3e-4,
    normalize_obs=True,
)
agent = PPO(env, config)
agent.learn(log_interval=5)

# Evaluate
obs, _ = env.reset()
for _ in range(500):
    action = agent.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
```

---

## Configuration Reference

All YAML config keys (all optional — sane defaults are built in):

```yaml
env_name:          Hopper          # environment to train on
total_timesteps:   1_000_000       # total env steps
seed:              42
eval_episodes:     10              # episodes per evaluation
eval_interval:     50_000          # steps between evaluations
save_interval:     200_000         # steps between checkpoints
log_dir:           logs

ppo:
  actor_hidden:      [256, 256]    # MLP hidden layer sizes
  critic_hidden:     [256, 256]
  n_steps:           2048          # rollout length
  n_epochs:          10            # update epochs per rollout
  mini_batch_size:   64
  clip_eps:          0.2           # PPO ε
  vf_coef:           0.5           # value loss weight
  ent_coef:          0.0           # entropy bonus weight
  max_grad_norm:     0.5
  gamma:             0.99          # discount factor
  gae_lambda:        0.95          # GAE λ
  lr:                3.0e-4
  lr_anneal:         true          # linear annealing to 0
  normalize_obs:     true          # running obs normalization
  normalize_rewards: false
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| mujoco | ≥ 3.0 | Physics simulation |
| gymnasium | ≥ 1.0 | Environment interface |
| torch | ≥ 2.0 | Neural networks |
| numpy | ≥ 1.24 | Numerics |
| matplotlib | ≥ 3.7 | Plotting |
| pyyaml | ≥ 6.0 | Config files |
| imageio | ≥ 2.31 *(optional)* | Video recording |

---

## License

MIT
