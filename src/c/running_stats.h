/**
 * running_stats.h — Welford online mean/variance for observation normalisation.
 *
 * Thread-safe (lock-free) when used with a single writer.
 * Designed to be called every environment step with minimal overhead.
 *
 * Build alongside fast_gae.c:
 *   gcc -O3 -march=native -shared -fPIC -o fast_gae.so \
 *       fast_gae.c running_stats.c -lm
 */

#pragma once
#include "fast_gae.h"    /* for EXPORT macro */
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Opaque handle returned by running_stats_create(). */
typedef struct RunningStats RunningStats;

/**
 * running_stats_create — allocate a new RunningStats object.
 *
 * @param dim  observation dimensionality
 * @return     pointer to heap-allocated object (must be freed with _destroy)
 */
EXPORT RunningStats* running_stats_create(int32_t dim);

/** Free resources. */
EXPORT void running_stats_destroy(RunningStats* rs);

/**
 * running_stats_update — update statistics with a batch of observations.
 *
 * @param rs        RunningStats handle
 * @param obs       float32[batch × dim] row-major batch of observations
 * @param batch     number of observations in this update
 */
EXPORT void running_stats_update(RunningStats* rs,
                                  const float*  obs,
                                  int32_t       batch);

/**
 * running_stats_normalize — normalise a batch of observations in-place.
 *
 * Applies:  out[i] = clip((obs[i] - mean) / sqrt(var + eps), -clip, clip)
 *
 * @param rs      RunningStats handle
 * @param obs     float32[batch × dim] — normalised in-place
 * @param batch   number of observations
 * @param eps     numerical stability (default 1e-8)
 * @param clip    clipping range (default 10.0, use 0 to disable)
 */
EXPORT void running_stats_normalize(RunningStats*  rs,
                                     float*          obs,
                                     int32_t         batch,
                                     float           eps,
                                     float           clip);

/** Accessors for exporting state to Python (for checkpointing). */
EXPORT void running_stats_get_mean(const RunningStats* rs, float* out);
EXPORT void running_stats_get_var (const RunningStats* rs, float* out);
EXPORT double running_stats_get_count(const RunningStats* rs);

/** Restore from checkpoint. */
EXPORT void running_stats_set_state(RunningStats*  rs,
                                     const float*   mean,
                                     const float*   var,
                                     double         count);

#ifdef __cplusplus
}
#endif
