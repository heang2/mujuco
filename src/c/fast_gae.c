/**
 * fast_gae.c — Generalized Advantage Estimation in C
 *
 * Single-pass backwards loop, cache-friendly, auto-vectorisable.
 * See fast_gae.h for documentation.
 */

#include "fast_gae.h"
#include <math.h>
#include <string.h>

/* ── Single rollout ──────────────────────────────────────────────────────── */

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
) {
    float gae = 0.0f;

    for (int32_t t = n - 1; t >= 0; t--) {
        /* Bootstrap from next value (or provided last_value at the boundary) */
        float next_val  = (t == n - 1) ? last_value : values[t + 1];
        float next_done = (t == n - 1) ? 0.0f       : dones[t + 1];

        /* TD-error δ_t = r_t + γ·V(s_{t+1})·(1-done_t) − V(s_t) */
        float delta = rewards[t]
                    + gamma * next_val * (1.0f - dones[t])
                    - values[t];

        /* Recursive GAE: Â_t = δ_t + γλ·(1-done_t)·Â_{t+1} */
        gae = delta + gamma * gae_lambda * (1.0f - dones[t]) * gae;

        advantages[t] = gae;
        returns[t]    = gae + values[t];
    }
}

/* ── Batched rollouts ────────────────────────────────────────────────────── */

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
) {
    /*
     * Each rollout b occupies a contiguous block [b*n … b*n + n-1].
     * We process rollouts independently; the compiler can vectorise the
     * inner loop across rollouts if B is known at compile time, but the
     * backwards dependency within each rollout prevents full SIMD here.
     */
    for (int32_t b = 0; b < B; b++) {
        int32_t      offset   = b * n;
        const float* r_b      = rewards    + offset;
        const float* v_b      = values     + offset;
        const float* d_b      = dones      + offset;
        float*       adv_b    = advantages + offset;
        float*       ret_b    = returns    + offset;
        float        last_v   = last_values[b];

        compute_gae(r_b, v_b, d_b, adv_b, ret_b, n, gamma, gae_lambda, last_v);
    }
}

/* ── Advantage normalisation ─────────────────────────────────────────────── */

EXPORT void normalize_advantages(
    float*   advantages,
    int32_t  n,
    float    eps
) {
    /* Welford single-pass mean + variance */
    double mean = 0.0, M2 = 0.0;
    for (int32_t i = 0; i < n; i++) {
        double delta = advantages[i] - mean;
        mean += delta / (i + 1);
        M2   += delta * (advantages[i] - mean);
    }
    double var = (n > 1) ? M2 / (n - 1) : 1.0;
    float  std = (float) sqrt(var + (double) eps);
    float  m   = (float) mean;

    /* In-place normalisation — auto-vectorisable */
    for (int32_t i = 0; i < n; i++) {
        advantages[i] = (advantages[i] - m) / std;
    }
}
