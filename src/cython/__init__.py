"""
Cython extensions for performance-critical RL computations.

Available after building:
    cd src/cython && python setup.py build_ext --inplace

Modules:
    fast_gae — Generalized Advantage Estimation in Cython (3-5x faster than NumPy)
"""

try:
    from fast_gae import compute_gae, compute_gae_2d, normalize_advantages
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False

    import numpy as np

    def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95, next_value=0.0):
        """Pure NumPy fallback for compute_gae."""
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            nv   = values[t + 1] if t < T - 1 else next_value
            delta = rewards[t] + gamma * nv * mask - values[t]
            last_gae = delta + gamma * gae_lambda * mask * last_gae
            advantages[t] = last_gae
        return advantages, advantages + values

    def normalize_advantages(advantages, eps=1e-8):
        return (advantages - advantages.mean()) / (advantages.std() + eps)
