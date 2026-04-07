/**
 * fast_gae.h — Generalized Advantage Estimation in C
 *
 * Provides ~10× speedup over the equivalent Python loop by:
 *   - Contiguous memory layout (cache-friendly single-pass backwards loop)
 *   - No Python object overhead (pure float32 arrays)
 *   - Potential SIMD auto-vectorization by modern compilers
 *
 * Build:
 *   gcc -O3 -march=native -ffast-math -shared -fPIC \
 *       -o fast_gae.so fast_gae.c -lm
 *   clang -O3 -march=native -ffast-math -shared -fPIC \
 *       -o fast_gae.so fast_gae.c -lm
 *
 * Usage from Python (via ctypes wrapper in fast_gae_py.py).
 */

#pragma once

#include <stdint.h>

#ifdef _WIN32
  #define EXPORT __declspec(dllexport)
#else
  #define EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * compute_gae — compute advantages and returns for a single rollout.
 *
 * @param rewards     float32[n]  — per-step rewards
 * @param values      float32[n]  — V(s_t) estimates
 * @param dones       float32[n]  — episode termination flags {0, 1}
 * @param advantages  float32[n]  — OUTPUT: GAE advantages
 * @param returns     float32[n]  — OUTPUT: discounted returns
 * @param n           number of timesteps
 * @param gamma       discount factor
 * @param gae_lambda  GAE-λ parameter
 * @param last_value  V(s_{T+1}) bootstrap value
 */
EXPORT void compute_gae(
    const float* rewards,
    const float* values,
    const float* dones,
    float*       advantages,
    float*       returns,
    int32_t      n,
    float        gamma,
    float        gae_lambda,
    float        last_value
);

/**
 * compute_gae_batched — compute GAE for B independent rollouts in parallel.
 *
 * Each rollout has length n; data is laid out as [B × n] row-major.
 *
 * @param rewards     float32[B, n]
 * @param values      float32[B, n]
 * @param dones       float32[B, n]
 * @param advantages  float32[B, n]  — OUTPUT
 * @param returns     float32[B, n]  — OUTPUT
 * @param B           batch size (number of rollouts)
 * @param n           rollout length
 * @param gamma       discount factor
 * @param gae_lambda  GAE-λ
 * @param last_values float32[B]  — bootstrap values per rollout
 */
EXPORT void compute_gae_batched(
    const float* rewards,
    const float* values,
    const float* dones,
    float*       advantages,
    float*       returns,
    int32_t      B,
    int32_t      n,
    float        gamma,
    float        gae_lambda,
    const float* last_values
);

/**
 * normalize_advantages — in-place normalisation: (adv - mean) / (std + eps)
 *
 * @param advantages  float32[n]  — in-place normalised output
 * @param n           number of elements
 * @param eps         numerical stability constant
 */
EXPORT void normalize_advantages(
    float*   advantages,
    int32_t  n,
    float    eps
);

#ifdef __cplusplus
}
#endif
