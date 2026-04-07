/**
 * running_stats.c — Welford online mean/variance in C.
 *
 * Uses Welford's algorithm for numerical stability:
 *   https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
 *
 * Batch updates use the parallel/combined Welford update so that a single
 * call processes B observations efficiently without per-step Python overhead.
 */

#include "running_stats.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/* ── Internal struct ─────────────────────────────────────────────────────── */

struct RunningStats {
    int32_t  dim;
    double*  mean;   /* [dim] */
    double*  M2;     /* [dim]  running sum of squared deviations */
    double   count;
};

/* ── Lifecycle ───────────────────────────────────────────────────────────── */

EXPORT RunningStats* running_stats_create(int32_t dim) {
    RunningStats* rs = (RunningStats*) malloc(sizeof(RunningStats));
    rs->dim          = dim;
    rs->mean         = (double*) calloc(dim, sizeof(double));
    rs->M2           = (double*) calloc(dim, sizeof(double));
    rs->count        = 1e-4;   /* pseudo-count avoids division by zero */
    return rs;
}

EXPORT void running_stats_destroy(RunningStats* rs) {
    if (rs) {
        free(rs->mean);
        free(rs->M2);
        free(rs);
    }
}

/* ── Update (single observation) ─────────────────────────────────────────── */

static inline void _update_one(RunningStats* rs, const float* x) {
    rs->count += 1.0;
    for (int32_t d = 0; d < rs->dim; d++) {
        double val   = (double) x[d];
        double delta = val - rs->mean[d];
        rs->mean[d] += delta / rs->count;
        rs->M2[d]   += delta * (val - rs->mean[d]);
    }
}

/* ── Update (batch — more efficient than looping _update_one) ─────────────── */

EXPORT void running_stats_update(RunningStats* rs,
                                   const float*  obs,
                                   int32_t       batch) {
    /*
     * Parallel Welford update:
     * Combine existing statistics (mean_a, M2_a, count_a) with
     * batch statistics (mean_b, M2_b, count_b) to get merged result.
     */
    int32_t  dim = rs->dim;
    double   n_a = rs->count;

    /* Compute batch mean and variance */
    double* mean_b = (double*) calloc(dim, sizeof(double));
    double* M2_b   = (double*) calloc(dim, sizeof(double));
    double  n_b    = (double) batch;

    for (int32_t i = 0; i < batch; i++) {
        const float* x = obs + i * dim;
        for (int32_t d = 0; d < dim; d++) {
            double val   = (double) x[d];
            double delta = val - mean_b[d];
            mean_b[d]   += delta / (i + 1);
            M2_b[d]     += delta * (val - mean_b[d]);
        }
    }

    /* Merge */
    double n_total = n_a + n_b;
    for (int32_t d = 0; d < dim; d++) {
        double delta    = mean_b[d] - rs->mean[d];
        rs->mean[d]    += delta * n_b / n_total;
        rs->M2[d]      += M2_b[d] + delta * delta * n_a * n_b / n_total;
    }
    rs->count = n_total;

    free(mean_b);
    free(M2_b);
}

/* ── Normalise ────────────────────────────────────────────────────────────── */

EXPORT void running_stats_normalize(RunningStats*  rs,
                                      float*          obs,
                                      int32_t         batch,
                                      float           eps,
                                      float           clip) {
    int32_t dim = rs->dim;

    /* Pre-compute std per dimension */
    double* std_d = (double*) malloc(dim * sizeof(double));
    for (int32_t d = 0; d < dim; d++) {
        double var = (rs->count > 1.0) ? rs->M2[d] / (rs->count - 1.0) : 1.0;
        std_d[d]   = sqrt(var + (double) eps);
    }

    /* Apply normalisation with optional clipping */
    for (int32_t i = 0; i < batch; i++) {
        float* x = obs + i * dim;
        for (int32_t d = 0; d < dim; d++) {
            float normed = (float)((x[d] - rs->mean[d]) / std_d[d]);
            if (clip > 0.0f) {
                normed = (normed >  clip) ?  clip : normed;
                normed = (normed < -clip) ? -clip : normed;
            }
            x[d] = normed;
        }
    }

    free(std_d);
}

/* ── Accessors ───────────────────────────────────────────────────────────── */

EXPORT void running_stats_get_mean(const RunningStats* rs, float* out) {
    for (int32_t d = 0; d < rs->dim; d++)
        out[d] = (float) rs->mean[d];
}

EXPORT void running_stats_get_var(const RunningStats* rs, float* out) {
    double denom = (rs->count > 1.0) ? rs->count - 1.0 : 1.0;
    for (int32_t d = 0; d < rs->dim; d++)
        out[d] = (float)(rs->M2[d] / denom);
}

EXPORT double running_stats_get_count(const RunningStats* rs) {
    return rs->count;
}

EXPORT void running_stats_set_state(RunningStats*  rs,
                                      const float*   mean,
                                      const float*   var,
                                      double         count) {
    rs->count = count;
    double denom = (count > 1.0) ? count - 1.0 : 1.0;
    for (int32_t d = 0; d < rs->dim; d++) {
        rs->mean[d] = (double) mean[d];
        rs->M2[d]   = (double) var[d] * denom;
    }
}
