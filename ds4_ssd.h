#ifndef DS4_SSD_H
#define DS4_SSD_H

#include <stdbool.h>
#include <stdint.h>

/* Residency is deliberately tri-state.  AUTO is zero so callers that
 * zero-initialize ds4_engine_options get the platform policy without having
 * to opt in to a backend-specific default. */
typedef enum {
    DS4_RESIDENCY_AUTO = 0,
    DS4_RESIDENCY_RESIDENT,
    DS4_RESIDENCY_SSD,
} ds4_residency_mode;

typedef enum {
    DS4_RESIDENCY_REASON_EXPLICIT_RESIDENT = 0,
    DS4_RESIDENCY_REASON_EXPLICIT_SSD,
    DS4_RESIDENCY_REASON_NON_METAL_AUTO,
    DS4_RESIDENCY_REASON_METAL_FITS,
    DS4_RESIDENCY_REASON_METAL_EXCEEDS,
    DS4_RESIDENCY_REASON_METAL_BUDGET_UNAVAILABLE,
    DS4_RESIDENCY_REASON_INSPECT_ONLY,
} ds4_residency_reason;

typedef struct {
    ds4_residency_mode requested;
    ds4_residency_mode resolved;
    ds4_residency_reason reason;
    uint64_t recommended_bytes;
    uint64_t external_reserved_bytes;
    uint64_t budget_bytes;
    uint64_t model_bytes;
    uint64_t runtime_bytes;
    uint64_t headroom_bytes;
    uint64_t required_bytes;
} ds4_residency_plan;

typedef struct {
    void *ptr;
    uint64_t bytes;
} ds4_ssd_memory_lock;

typedef struct {
    uint64_t model_target_bytes;
    uint64_t cache_bytes;
    uint64_t effective_cache_bytes;
    uint32_t cache_experts;
} ds4_ssd_cache_plan;

bool ds4_parse_gib_arg(const char *s, uint64_t *bytes);
bool ds4_parse_streaming_cache_experts_arg(const char *s,
                                           uint32_t   *experts,
                                           uint64_t   *bytes);

uint32_t ds4_ssd_cache_experts_for_byte_budget(uint64_t bytes,
                                               uint64_t per_expert_bytes);
bool ds4_ssd_auto_cache_plan(uint64_t            recommended_bytes,
                             uint64_t            non_routed_bytes,
                             uint64_t            per_expert_bytes,
                             uint64_t            max_model_experts,
                             ds4_ssd_cache_plan *out);
bool ds4_ssd_cache_plan_for_model_target(uint64_t            model_target_bytes,
                                         uint64_t            non_routed_bytes,
                                         uint64_t            per_expert_bytes,
                                         uint64_t            max_model_experts,
                                         ds4_ssd_cache_plan *out);
bool ds4_ssd_working_set_after_reserve(uint64_t  recommended_bytes,
                                       uint64_t  runtime_bytes,
                                       uint64_t  external_reserved_bytes,
                                       uint64_t *available_bytes,
                                       uint64_t *reserved_bytes);

const char *ds4_residency_mode_name(ds4_residency_mode mode);
const char *ds4_residency_reason_name(ds4_residency_reason reason);
bool ds4_residency_plan_make(bool                   metal_backend,
                             ds4_residency_mode     requested,
                             uint64_t               model_bytes,
                             uint64_t               runtime_bytes,
                             uint64_t               recommended_bytes,
                             uint64_t               external_reserved_bytes,
                             ds4_residency_plan    *out);

bool ds4_ssd_memory_lock_acquire(ds4_ssd_memory_lock *lock,
                                 uint64_t             bytes);
void ds4_ssd_memory_lock_release(ds4_ssd_memory_lock *lock);

#endif
