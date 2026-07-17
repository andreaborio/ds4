#ifndef DS4_QWEN_EXPERT_GROUP_H
#define DS4_QWEN_EXPERT_GROUP_H

#include <stdint.h>

/* The router exposes selected experts as token-major slots.  Grouped kernels
 * want the opposite traversal (expert-major), but the final reduction must
 * still observe the original token-slot order.  canonical_index is therefore
 * the contract between the two layouts:
 *
 *   canonical_index = token_row * routes_per_token + route_slot
 *
 * A grouped kernel may execute entries in any expert-major order, provided it
 * scatters each result to route_output[canonical_index].  The reducer then
 * visits route slots 0..routes_per_token-1 for every token, exactly as the
 * baseline does.  This deliberately keeps scheduling order separate from
 * floating-point addition order.
 */
typedef struct {
    uint32_t expert;
    uint32_t token_row;
    uint32_t route_slot;
    uint32_t canonical_index;
} ds4_qwen_expert_group_route;

/* Metal can upload this array verbatim: make accidental host-ABI padding a
 * compile-time failure instead of a silently shifted shader field. */
typedef char ds4_qwen_expert_group_route_must_be_16_bytes[
    sizeof(ds4_qwen_expert_group_route) == 4u * sizeof(uint32_t) ? 1 : -1];

typedef enum {
    DS4_QWEN_EXPERT_GROUP_OK = 0,
    DS4_QWEN_EXPERT_GROUP_INVALID_ARGUMENT = 1,
    DS4_QWEN_EXPERT_GROUP_OVERFLOW = 2,
    DS4_QWEN_EXPERT_GROUP_EXPERT_OUT_OF_RANGE = 3,
    DS4_QWEN_EXPERT_GROUP_OUT_OF_MEMORY = 4,
} ds4_qwen_expert_group_result;

typedef struct {
    uint32_t n_tokens;
    uint32_t routes_per_token;
    uint32_t n_experts;
    uint32_t route_count;
    uint32_t active_expert_count;

    uint32_t expert_capacity;
    uint32_t route_capacity;

    /* expert_offsets has n_experts + 1 elements.  Expert e owns the half-open
     * grouped range [expert_offsets[e], expert_offsets[e + 1]).  Empty experts
     * retain an empty range, making direct expert-ID lookup branch-free. */
    uint32_t *expert_offsets;

    /* Ascending expert IDs with a non-empty range.  Ascending order is chosen
     * explicitly so the plan is deterministic across platforms and runs. */
    uint32_t *active_experts;

    /* Stable expert-major route list.  Within one expert, entries retain their
     * canonical token-major/slot-major order, including duplicate expert IDs. */
    ds4_qwen_expert_group_route *grouped_routes;

    /* Maps every canonical token-slot index to its grouped_routes index.  This
     * makes the permutation explicit and lets Metal validate/gather either
     * layout without deriving an inverse on the hot path. */
    uint32_t *canonical_to_grouped;

    /* One reusable allocation backs all four arrays above.  The plan owns it
     * and is not copyable.  Callers must use init/destroy rather than accessing
     * this ownership field directly. */
    void *storage;
} ds4_qwen_expert_group_plan;

void ds4_qwen_expert_group_plan_init(ds4_qwen_expert_group_plan *plan);
void ds4_qwen_expert_group_plan_destroy(ds4_qwen_expert_group_plan *plan);

/* Reserve reusable storage without constructing a route plan.  This is useful
 * at graph creation time so no allocation occurs between router readback and
 * Metal encoding.  A failed reserve leaves the previous plan and storage
 * untouched; a successful growth invalidates old route metadata because the
 * backing addresses change. */
ds4_qwen_expert_group_result ds4_qwen_expert_group_plan_reserve(
        ds4_qwen_expert_group_plan *plan,
        uint32_t                    expert_capacity,
        uint32_t                    route_capacity);

/* Build a stable counting-sort permutation from signed router IDs.  Duplicate
 * IDs are valid routes and are never deduplicated: each slot may carry a
 * distinct router weight and must contribute separately in canonical order.
 *
 * Dimensions and every expert ID are validated before the plan is mutated or
 * enlarged.  On any error, the previous complete plan remains available.
 */
ds4_qwen_expert_group_result ds4_qwen_expert_group_plan_build(
        ds4_qwen_expert_group_plan *plan,
        const int32_t              *selected,
        uint32_t                    n_tokens,
        uint32_t                    routes_per_token,
        uint32_t                    n_experts);

#endif
