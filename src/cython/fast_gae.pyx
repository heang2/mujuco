# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
"""
fast_gae.pyx — Cython implementation of Generalized Advantage Estimation.

Computes GAE-λ advantages and TD-λ returns in a tight C loop, avoiding
Python object overhead for the sequential backward pass.

Build:
    python src/cython/setup.py build_ext --inplace

Python usage:
    from fast_gae import compute_gae
    advantages, returns = compute_gae(
        rewards, values, dones,
        gamma=0.99, gae_lambda=0.95
    )
"""

import numpy as np
cimport numpy as np
from libc.math cimport fabs

ctypedef np.float32_t DTYPE_t
DTYPE = np.float32


def compute_gae(
    np.ndarray[DTYPE_t, ndim=1] rewards,
    np.ndarray[DTYPE_t, ndim=1] values,
    np.ndarray[DTYPE_t, ndim=1] dones,
    float gamma     = 0.99,
    float gae_lambda = 0.95,
    float next_value = 0.0,
):
    """
    Compute GAE-λ advantages and TD-λ returns.

    Args:
        rewards:     (T,) observed rewards
        values:      (T,) critic value estimates V(s_t)
        dones:       (T,) episode termination flags (1=done, 0=alive)
        gamma:       discount factor γ
        gae_lambda:  GAE λ parameter
        next_value:  bootstrap value V(s_{T+1}), 0 for terminal states

    Returns:
        advantages: (T,) advantage estimates A_t = GAE(γ, λ)
        returns:    (T,) discounted returns R_t = A_t + V(s_t)
    """
    cdef int T = rewards.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] advantages = np.zeros(T, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] returns     = np.zeros(T, dtype=DTYPE)

    cdef float last_gae = 0.0
    cdef float delta, mask
    cdef int t

    for t in range(T - 1, -1, -1):
        mask  = 1.0 - dones[t]
        if t == T - 1:
            delta = rewards[t] + gamma * next_value * mask - values[t]
        else:
            delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
        last_gae = delta + gamma * gae_lambda * mask * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


def compute_gae_2d(
    np.ndarray[DTYPE_t, ndim=2] rewards,
    np.ndarray[DTYPE_t, ndim=2] values,
    np.ndarray[DTYPE_t, ndim=2] dones,
    float gamma     = 0.99,
    float gae_lambda = 0.95,
    np.ndarray[DTYPE_t, ndim=1] next_values = None,
):
    """
    Vectorized GAE for multiple parallel environments.

    Args:
        rewards:     (T, N) rewards from N parallel envs
        values:      (T, N) value estimates
        dones:       (T, N) done flags
        gamma:       discount factor
        gae_lambda:  λ parameter
        next_values: (N,) bootstrap values; zeros if None

    Returns:
        advantages: (T, N)
        returns:    (T, N)
    """
    cdef int T = rewards.shape[0]
    cdef int N = rewards.shape[1]
    cdef np.ndarray[DTYPE_t, ndim=2] advantages = np.zeros((T, N), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=2] returns     = np.zeros((T, N), dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] last_gae    = np.zeros(N, dtype=DTYPE)
    cdef np.ndarray[DTYPE_t, ndim=1] _next_vals

    if next_values is None:
        _next_vals = np.zeros(N, dtype=DTYPE)
    else:
        _next_vals = next_values

    cdef float delta, mask
    cdef int t, n

    for t in range(T - 1, -1, -1):
        for n in range(N):
            mask = 1.0 - dones[t, n]
            if t == T - 1:
                delta = rewards[t, n] + gamma * _next_vals[n] * mask - values[t, n]
            else:
                delta = rewards[t, n] + gamma * values[t + 1, n] * mask - values[t, n]
            last_gae[n] = delta + gamma * gae_lambda * mask * last_gae[n]
            advantages[t, n] = last_gae[n]

    returns = advantages + values
    return advantages, returns


def normalize_advantages(
    np.ndarray[DTYPE_t, ndim=1] advantages,
    float eps = 1e-8,
):
    """
    Normalize advantages to zero mean, unit variance.

    Args:
        advantages: (T,) or (T*N,) raw advantage estimates
        eps:        numerical stability epsilon

    Returns:
        normalized: (T,) normalized advantages
    """
    cdef int n = advantages.shape[0]
    cdef float mean = 0.0
    cdef float var  = 0.0
    cdef float std
    cdef int i

    for i in range(n):
        mean += advantages[i]
    mean /= n

    for i in range(n):
        var += (advantages[i] - mean) ** 2
    var /= n
    std = (var + eps) ** 0.5

    cdef np.ndarray[DTYPE_t, ndim=1] out = np.empty(n, dtype=DTYPE)
    for i in range(n):
        out[i] = (advantages[i] - mean) / std

    return out
