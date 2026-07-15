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
    DS4_RESIDENCY_REASON_METAL_CURRENT_PRESSURE,
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

/* Point-in-time guard for AUTO full-model mapped mode. Metal's fixed
 * recommended working-set limit does not shrink when another process consumes
 * unified memory, so the live reclaimable budget must pass independently. */
typedef struct {
    uint64_t physical_bytes;
    uint64_t reclaimable_bytes;
    uint64_t inactive_credit_bytes;
    uint64_t current_headroom_bytes;
    uint64_t pressure_margin_bytes;
    uint64_t required_bytes;
    bool pressure_status_available;
    bool pressure_normal;
    bool fits;
} ds4_ssd_resident_pressure_plan;

typedef struct {
    uint64_t working_set_experts;
    uint64_t minimum_cache_experts;
    uint64_t minimum_cache_bytes;
    uint64_t warning_cache_experts;
} ds4_ssd_expert_cache_floor;

/* Point-in-time host memory state used by the SSD expert-cache planner.
 * The platform backend owns collection; the planner below is pure so policy
 * can be tested without depending on live machine pressure. */
typedef struct {
    uint64_t physical_bytes;
    uint64_t recommended_bytes;
    uint64_t task_footprint_bytes;
    /* Directly reclaimable free pages.  On Darwin, Mach free_count already
     * includes speculative pages; collectors must not add them again. */
    uint64_t free_bytes;
    uint64_t purgeable_bytes;
    uint64_t inactive_bytes;
    uint64_t file_backed_bytes;
    /* Darwin's current system pressure state. The resident planner and the
     * bounded Qwen low-RAM policy may use a larger inactive working-set credit
     * while pressure is normal. */
    bool pressure_status_available;
    bool pressure_normal;
} ds4_ssd_host_memory;

typedef struct {
    ds4_ssd_expert_cache_floor floor;
    uint64_t reclaimable_bytes;
    /* Pageable static weights compete with the wired expert cache for the
     * same unified-memory budget. AUTO preserves this charge unless the model
     * policy explicitly lets pageable pages share ordinary system headroom or
     * those weights were already pinned and visible in the live snapshot. */
    uint64_t pageable_static_reserve_bytes;
    /* recommendedMaxWorkingSetSize is a fixed platform limit, so pinned
     * static bytes remain charged here even though a post-pin current-memory
     * snapshot already reflects them. */
    uint64_t platform_static_reserve_bytes;
    uint64_t current_headroom_bytes;
    uint64_t pressure_margin_bytes;
    uint64_t platform_headroom_bytes;
    uint64_t current_wire_budget_bytes;
    uint64_t platform_wire_budget_bytes;
    uint64_t safety_wire_budget_bytes;
    uint64_t cache_envelope_bytes;
    uint64_t wire_budget_bytes;
    uint64_t cache_bytes;
    bool low_ram_floor_ceiling_active;
    uint32_t cache_experts;
} ds4_ssd_adaptive_cache_plan;

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
bool ds4_ssd_expert_cache_floor_make(
        uint64_t                    cacheable_routed_layers,
        uint64_t                    experts_per_token,
        uint64_t                    per_expert_bytes,
        ds4_ssd_expert_cache_floor *out);
bool ds4_ssd_adaptive_cache_plan_make(
        const ds4_ssd_host_memory  *memory,
        uint64_t                    runtime_bytes,
        uint64_t                    cacheable_routed_layers,
        uint64_t                    experts_per_token,
        uint64_t                    per_expert_bytes,
        uint64_t                    max_cacheable_experts,
        ds4_ssd_adaptive_cache_plan *out);
bool ds4_ssd_adaptive_cache_plan_make_with_static_reserve(
        const ds4_ssd_host_memory  *memory,
        uint64_t                    runtime_bytes,
        uint64_t                    static_working_set_bytes,
        bool                        static_already_pinned,
        uint64_t                    cacheable_routed_layers,
        uint64_t                    experts_per_token,
        uint64_t                    per_expert_bytes,
        uint64_t                    max_cacheable_experts,
        ds4_ssd_adaptive_cache_plan *out);
bool ds4_ssd_adaptive_cache_plan_make_strict_with_static_reserve(
        const ds4_ssd_host_memory  *memory,
        uint64_t                    runtime_bytes,
        uint64_t                    static_working_set_bytes,
        bool                        static_already_pinned,
        uint64_t                    cacheable_routed_layers,
        uint64_t                    experts_per_token,
        uint64_t                    per_expert_bytes,
        uint64_t                    max_cacheable_experts,
        ds4_ssd_adaptive_cache_plan *out);
bool ds4_ssd_resident_pressure_plan_make(
        const ds4_ssd_host_memory      *memory,
        uint64_t                        model_bytes,
        uint64_t                        runtime_bytes,
        ds4_ssd_resident_pressure_plan *out);
bool ds4_ssd_low_ram_cache_policy(uint64_t physical_bytes);
bool ds4_ssd_static_pin_host_supported(uint64_t physical_bytes);
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
