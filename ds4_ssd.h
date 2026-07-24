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
    DS4_RESIDENCY_REASON_MODEL_REQUIRES_SSD,
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

/* Qwen Metal memory policy is continuous in the measured host budgets, while
 * the profile label makes the finite Apple unified-memory cuts observable and
 * independently testable.  Values between named cuts use the next containing
 * profile without changing the byte-based safety arithmetic. */
typedef struct {
    uint32_t profile_gib;
    uint64_t physical_bytes;
    uint64_t recommended_bytes;
    uint64_t resident_headroom_bytes;
} ds4_qwen_metal_hardware_policy;

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

/* Initial LFU policy for the streaming expert hotlist. Adaptive keeps every
 * ordered seed at priority one so live selections take over immediately.
 * Legacy preserves the historical per-entry priorities (built-in rank or
 * file hit count), while fixed is an explicit numeric A/B override. */
typedef enum {
    DS4_STREAMING_HOTLIST_PRIORITY_ADAPTIVE = 0,
    DS4_STREAMING_HOTLIST_PRIORITY_LEGACY,
    DS4_STREAMING_HOTLIST_PRIORITY_FIXED,
} ds4_streaming_hotlist_priority_mode;

typedef struct {
    ds4_streaming_hotlist_priority_mode mode;
    uint32_t priority;
} ds4_streaming_hotlist_priority_policy;

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
    /* Darwin's current system pressure state. The resident planner and Qwen
     * SSD policy may use a larger inactive working-set credit while pressure
     * is normal. */
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
    /* Qwen <=16 GiB: pageable static weights share ordinary headroom and
     * normal pressure permits full bounded file-inactive credit. */
    bool low_ram_shared_static_headroom_active;
    /* DeepSeek <=16 GiB: measured performance policy caps AUTO at its floor. */
    bool low_ram_floor_ceiling_active;
    /* Qwen on all profiles, and DeepSeek on its measured 64 GiB tier: normal
     * pressure allows full bounded file-backed inactive credit. */
    bool normal_pressure_full_file_credit_active;
    uint32_t cache_experts;
} ds4_ssd_adaptive_cache_plan;

bool ds4_parse_gib_arg(const char *s, uint64_t *bytes);
bool ds4_parse_streaming_cache_experts_arg(const char *s,
                                           uint32_t   *experts,
                                           uint64_t   *bytes);
bool ds4_parse_streaming_hotlist_priority_policy(
        const char                            *s,
        ds4_streaming_hotlist_priority_policy *out);

uint32_t ds4_ssd_cache_experts_for_byte_budget(uint64_t bytes,
                                               uint64_t per_expert_bytes);
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
/* GLM ExpertMajor is intentionally a model-specific runtime contract. On the
 * measured 64 GiB Metal tier, one complete route plus the in-flight safety
 * slot preserves more per-layer SSD concurrency than a larger hit-rate-driven
 * cache. Larger hosts keep the adaptive candidate until they are measured. */
uint32_t ds4_ssd_glm_expert_major_auto_cache_target(
        const ds4_ssd_host_memory         *memory,
        const ds4_ssd_adaptive_cache_plan *plan);
/* DeepSeek ExpertMajor keeps the generic pressure-derived plan as a hard
 * ceiling. The measured 64 GiB tier starts from a bounded number of complete
 * route cycles so pageable static weights and the macOS file cache retain
 * useful headroom; genuine pressure may still select a smaller tier. */
uint32_t ds4_ssd_deepseek_expert_major_auto_cache_target(
        const ds4_ssd_host_memory         *memory,
        const ds4_ssd_adaptive_cache_plan *plan);
/* Long prompts need a resident hot set before decode starts. On the measured
 * 64 GiB tier, retain sixteen complete route cycles through 32K and eight at
 * 65K+ instead of restoring an empty decode cache. The generic pressure plan
 * remains the ceiling. Returning zero means that no measured override applies. */
uint32_t ds4_ssd_deepseek_long_context_cache_target(
        const ds4_ssd_host_memory         *memory,
        const ds4_ssd_adaptive_cache_plan *plan,
        uint32_t                           decode_target,
        uint32_t                           n_tokens);
/* Pure transition policy used by cold and resumed sessions. The work size
 * selects the prefill schedule; the resulting total context selects the
 * bounded long-context decode tier, with a 128-token guard before each hard
 * frontier to avoid grow-then-shrink churn. A zero prefill target means no
 * transition is required before evaluating the supplied work. */
uint32_t ds4_ssd_deepseek_prefill_phase_cache_target(
        uint32_t prefill_tokens,
        uint32_t resulting_context_tokens,
        uint32_t batched_prefill_max_tokens,
        uint32_t prefill_target,
        uint32_t long_context_target,
        uint32_t extended_context_target);
uint32_t ds4_ssd_deepseek_post_prefill_cache_target(
        uint32_t resulting_context_tokens,
        uint32_t long_context_target,
        uint32_t extended_context_target,
        uint32_t decode_target);
bool ds4_ssd_deepseek_post_prefill_seed_allowed(
        bool     prefill_succeeded,
        bool     pressure_allows_seed,
        bool     cache_changed,
        uint32_t target,
        uint32_t cache_after);
/* Batched victim-buffer reuse is a GLM scheduling policy. Keeping the family
 * gate in this pure helper prevents another shared-backend optimization from
 * silently changing DeepSeek decode. */
bool ds4_ssd_glm_streaming_batch_reuse_allowed(
        bool     glm_model,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes);
bool ds4_ssd_resident_pressure_plan_make(
        const ds4_ssd_host_memory      *memory,
        uint64_t                        model_bytes,
        uint64_t                        runtime_bytes,
        ds4_ssd_resident_pressure_plan *out);
bool ds4_qwen_metal_hardware_policy_make(
        uint64_t                         physical_bytes,
        uint64_t                         recommended_bytes,
        ds4_qwen_metal_hardware_policy *out);
bool ds4_residency_plan_apply_qwen_metal_hardware_policy(
        const ds4_qwen_metal_hardware_policy *policy,
        ds4_residency_plan                   *plan);
bool ds4_ssd_low_ram_cache_policy(uint64_t physical_bytes);
/* Qwen's 16/24 GiB SSD tiers use a measured cache ceiling and require an
 * affirmative normal-pressure signal before allocating or growing cache
 * storage.  Keep this separate from the generic <=16 GiB policy because the
 * latter also controls DeepSeek-specific behavior. */
bool ds4_ssd_qwen_guarded_cache_policy(uint64_t physical_bytes);
/* A guarded Qwen phase rechecks pressure even when its configured cache budget
 * is unchanged: lazy expert slabs can still become physically populated during
 * later work. `cache_budget_changed` is explicit so tests keep both paths
 * covered; it must never weaken the pressure decision. */
bool ds4_ssd_qwen_phase_pressure_allowed(
        bool guarded,
        bool cache_budget_changed,
        bool snapshot_available,
        bool pressure_status_available,
        bool pressure_normal);
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

/* Apply a model contract that permits SSD streaming only. The caller may pass
 * an already-populated generic plan; accounting fields are preserved while
 * requested/resolved/reason are made authoritative for the model. Explicit
 * resident requests return false so inference and inspection fail closed. */
bool ds4_residency_plan_apply_ssd_only(ds4_residency_mode  requested,
                                       ds4_residency_plan *plan);

bool ds4_ssd_memory_lock_acquire(ds4_ssd_memory_lock *lock,
                                 uint64_t             bytes);
void ds4_ssd_memory_lock_release(ds4_ssd_memory_lock *lock);

#endif
