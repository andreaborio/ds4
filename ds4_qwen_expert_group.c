#include "ds4_qwen_expert_group.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static bool checked_array_bytes(
        size_t *total,
        uint64_t count,
        size_t element_size) {
    if (!total || element_size == 0 || count > SIZE_MAX / element_size) {
        return false;
    }
    const size_t bytes = (size_t)count * element_size;
    if (*total > SIZE_MAX - bytes) return false;
    *total += bytes;
    return true;
}

void ds4_qwen_expert_group_plan_init(ds4_qwen_expert_group_plan *plan) {
    if (plan) memset(plan, 0, sizeof(*plan));
}

void ds4_qwen_expert_group_plan_destroy(ds4_qwen_expert_group_plan *plan) {
    if (!plan) return;
    free(plan->storage);
    free(plan->route_tile_storage);
    memset(plan, 0, sizeof(*plan));
}

static uint32_t build_route_tiles_unchecked(
        ds4_qwen_expert_group_plan *plan,
        uint32_t                    n_experts,
        uint32_t                    max_routes_per_tile) {
    uint32_t route_tile_count = 0;
    for (uint32_t expert = 0; expert < n_experts; expert++) {
        const uint32_t end = plan->expert_offsets[expert + 1u];
        for (uint32_t begin = plan->expert_offsets[expert]; begin < end;) {
            const uint32_t remaining = end - begin;
            const uint32_t count = remaining < max_routes_per_tile
                                     ? remaining
                                     : max_routes_per_tile;
            ds4_expert_group_route_tile *tile =
                &plan->route_tiles[route_tile_count++];
            tile->expert = expert;
            tile->route_begin = begin;
            tile->route_count = count;
            tile->reserved = 0;
            begin += count;
        }
    }
    return route_tile_count;
}

ds4_qwen_expert_group_result ds4_qwen_expert_group_plan_retile(
        ds4_qwen_expert_group_plan *plan,
        uint32_t                    max_routes_per_tile) {
    if (!plan || max_routes_per_tile == 0 ||
        max_routes_per_tile > DS4_EXPERT_GROUP_ROUTE_TILE_SIZE ||
        !plan->storage || !plan->expert_offsets || !plan->grouped_routes ||
        !plan->canonical_to_grouped || !plan->route_tiles ||
        plan->n_tokens == 0 || plan->routes_per_token == 0 ||
        plan->n_experts == 0 || plan->route_count == 0 ||
        plan->n_tokens > UINT32_MAX / plan->routes_per_token ||
        plan->n_tokens * plan->routes_per_token != plan->route_count ||
        plan->n_experts > plan->expert_capacity ||
        plan->route_count > plan->route_capacity ||
        plan->route_count > plan->route_tile_capacity ||
        plan->expert_offsets[0] != 0 ||
        plan->expert_offsets[plan->n_experts] != plan->route_count) {
        return DS4_QWEN_EXPERT_GROUP_INVALID_ARGUMENT;
    }

    /* Validate every source range before overwriting the first tile.  Besides
     * making malformed plans fail transactionally, this makes the expert
     * boundary invariant explicit at the public API rather than relying on a
     * caller to have used build immediately beforehand. */
    for (uint32_t expert = 0; expert < plan->n_experts; expert++) {
        const uint32_t begin = plan->expert_offsets[expert];
        const uint32_t end = plan->expert_offsets[expert + 1u];
        if (begin > end || end > plan->route_count) {
            return DS4_QWEN_EXPERT_GROUP_INVALID_ARGUMENT;
        }
        for (uint32_t grouped = begin; grouped < end; grouped++) {
            const ds4_qwen_expert_group_route route =
                plan->grouped_routes[grouped];
            if (route.expert != expert ||
                route.canonical_index >= plan->route_count ||
                plan->canonical_to_grouped[route.canonical_index] != grouped) {
                return DS4_QWEN_EXPERT_GROUP_INVALID_ARGUMENT;
            }
        }
    }

    plan->route_tile_count = build_route_tiles_unchecked(
        plan, plan->n_experts, max_routes_per_tile);
    return DS4_QWEN_EXPERT_GROUP_OK;
}

ds4_qwen_expert_group_result ds4_qwen_expert_group_plan_reserve(
        ds4_qwen_expert_group_plan *plan,
        uint32_t                    expert_capacity,
        uint32_t                    route_capacity) {
    if (!plan || expert_capacity == 0 || route_capacity == 0) {
        return DS4_QWEN_EXPERT_GROUP_INVALID_ARGUMENT;
    }
    if (expert_capacity == UINT32_MAX) {
        /* expert_offsets needs one sentinel beyond the last expert. */
        return DS4_QWEN_EXPERT_GROUP_OVERFLOW;
    }
    if (plan->storage &&
        plan->expert_capacity >= expert_capacity &&
        plan->route_capacity >= route_capacity) {
        return DS4_QWEN_EXPERT_GROUP_OK;
    }

    /* Grow both axes monotonically.  A later small micro-batch therefore does
     * not churn the allocator or discard the larger macro-prefill capacity. */
    if (plan->expert_capacity > expert_capacity) {
        expert_capacity = plan->expert_capacity;
    }
    if (plan->route_capacity > route_capacity) {
        route_capacity = plan->route_capacity;
    }

    size_t storage_bytes = 0;
    if (!checked_array_bytes(&storage_bytes,
                             (uint64_t)expert_capacity + 1u,
                             sizeof(uint32_t)) ||
        !checked_array_bytes(&storage_bytes,
                             expert_capacity,
                             sizeof(uint32_t)) ||
        !checked_array_bytes(&storage_bytes,
                             route_capacity,
                             sizeof(ds4_qwen_expert_group_route)) ||
        !checked_array_bytes(&storage_bytes,
                             route_capacity,
                             sizeof(uint32_t))) {
        return DS4_QWEN_EXPERT_GROUP_OVERFLOW;
    }

    void *storage = malloc(storage_bytes);
    if (!storage) return DS4_QWEN_EXPERT_GROUP_OUT_OF_MEMORY;

    uint8_t *cursor = storage;
    uint32_t *expert_offsets = (uint32_t *)cursor;
    cursor += ((size_t)expert_capacity + 1u) * sizeof(uint32_t);
    uint32_t *active_experts = (uint32_t *)cursor;
    cursor += (size_t)expert_capacity * sizeof(uint32_t);
    ds4_qwen_expert_group_route *grouped_routes =
        (ds4_qwen_expert_group_route *)cursor;
    cursor += (size_t)route_capacity *
              sizeof(ds4_qwen_expert_group_route);
    uint32_t *canonical_to_grouped = (uint32_t *)cursor;

    /* Core growth must not discard an independently reserved tile buffer.
     * Preserve its ownership while publishing the replacement permutation. */
    void *const route_tile_storage = plan->route_tile_storage;
    ds4_expert_group_route_tile *const route_tiles = plan->route_tiles;
    const uint32_t route_tile_capacity = plan->route_tile_capacity;

    /* Publish the replacement only after all size checks and allocation have
     * succeeded.  This keeps the old, complete plan usable after an OOM. */
    free(plan->storage);
    memset(plan, 0, sizeof(*plan));
    plan->expert_capacity = expert_capacity;
    plan->route_capacity = route_capacity;
    plan->expert_offsets = expert_offsets;
    plan->active_experts = active_experts;
    plan->grouped_routes = grouped_routes;
    plan->canonical_to_grouped = canonical_to_grouped;
    plan->route_tiles = route_tiles;
    plan->storage = storage;
    plan->route_tile_storage = route_tile_storage;
    plan->route_tile_capacity = route_tile_capacity;
    return DS4_QWEN_EXPERT_GROUP_OK;
}

ds4_qwen_expert_group_result ds4_qwen_expert_group_plan_reserve_route_tiles(
        ds4_qwen_expert_group_plan *plan,
        uint32_t                    route_capacity) {
    if (!plan || route_capacity == 0) {
        return DS4_QWEN_EXPERT_GROUP_INVALID_ARGUMENT;
    }
    if (plan->route_tile_storage &&
        plan->route_tile_capacity >= route_capacity) {
        return DS4_QWEN_EXPERT_GROUP_OK;
    }

    if ((uint64_t)route_capacity >
        SIZE_MAX / sizeof(ds4_expert_group_route_tile)) {
        return DS4_QWEN_EXPERT_GROUP_OVERFLOW;
    }
    const size_t bytes =
        (size_t)route_capacity * sizeof(ds4_expert_group_route_tile);
    void *storage = malloc(bytes);
    if (!storage) return DS4_QWEN_EXPERT_GROUP_OUT_OF_MEMORY;

    free(plan->route_tile_storage);
    plan->route_tile_storage = storage;
    plan->route_tiles = (ds4_expert_group_route_tile *)storage;
    plan->route_tile_capacity = route_capacity;
    plan->route_tile_count = 0;
    return DS4_QWEN_EXPERT_GROUP_OK;
}

ds4_qwen_expert_group_result ds4_qwen_expert_group_plan_build(
        ds4_qwen_expert_group_plan *plan,
        const int32_t              *selected,
        uint32_t                    n_tokens,
        uint32_t                    routes_per_token,
        uint32_t                    n_experts) {
    if (!plan || !selected || n_tokens == 0 ||
        routes_per_token == 0 || n_experts == 0) {
        return DS4_QWEN_EXPERT_GROUP_INVALID_ARGUMENT;
    }
    if (n_experts == UINT32_MAX ||
        n_tokens > UINT32_MAX / routes_per_token) {
        return DS4_QWEN_EXPERT_GROUP_OVERFLOW;
    }
    const uint32_t route_count = n_tokens * routes_per_token;

    /* Validate the complete route before reserving or touching plan metadata.
     * In particular, negative int32 IDs must not become huge unsigned bucket
     * indexes.  This pass is also what gives build its transactional failure
     * behavior for malformed router output. */
    for (uint64_t canonical = 0; canonical < route_count; canonical++) {
        const int32_t expert = selected[(size_t)canonical];
        if (expert < 0 || (uint32_t)expert >= n_experts) {
            return DS4_QWEN_EXPERT_GROUP_EXPERT_OUT_OF_RANGE;
        }
    }

    const ds4_qwen_expert_group_result reserve_result =
        ds4_qwen_expert_group_plan_reserve(plan, n_experts, route_count);
    if (reserve_result != DS4_QWEN_EXPERT_GROUP_OK) return reserve_result;

    memset(plan->expert_offsets, 0,
           ((size_t)n_experts + 1u) * sizeof(uint32_t));

    /* Counts live one cell to the right so an in-place prefix sum produces
     * both range starts and the final route-count sentinel. */
    for (uint64_t canonical = 0; canonical < route_count; canonical++) {
        const uint32_t expert = (uint32_t)selected[(size_t)canonical];
        plan->expert_offsets[expert + 1u]++;
    }
    for (uint32_t expert = 0; expert < n_experts; expert++) {
        plan->expert_offsets[expert + 1u] +=
            plan->expert_offsets[expert];
        /* active_experts temporarily holds each bucket's write cursor.  It is
         * rebuilt as a compact ID list after the stable scatter. */
        plan->active_experts[expert] = plan->expert_offsets[expert];
    }

    /* Scanning the source permutation in canonical order makes this counting
     * sort stable.  Two slots selecting the same expert remain two distinct
     * entries and appear in token/slot order within that expert's bucket. */
    for (uint64_t canonical = 0; canonical < route_count; canonical++) {
        const uint32_t canonical_index = (uint32_t)canonical;
        const uint32_t expert =
            (uint32_t)selected[(size_t)canonical];
        const uint32_t grouped = plan->active_experts[expert]++;
        ds4_qwen_expert_group_route *route =
            &plan->grouped_routes[grouped];
        route->expert = expert;
        route->token_row = canonical_index / routes_per_token;
        route->route_slot = canonical_index % routes_per_token;
        route->canonical_index = canonical_index;
        plan->canonical_to_grouped[canonical_index] = grouped;
    }

    uint32_t active_count = 0;
    for (uint32_t expert = 0; expert < n_experts; expert++) {
        if (plan->expert_offsets[expert] !=
            plan->expert_offsets[expert + 1u]) {
            plan->active_experts[active_count++] = expert;
        }
    }

    /* Metadata is committed last.  Every public dimension now describes a
     * fully populated permutation rather than an intermediate counting pass.
     * Tile construction remains an explicit opt-in after this commit. */
    plan->n_tokens = n_tokens;
    plan->routes_per_token = routes_per_token;
    plan->n_experts = n_experts;
    plan->route_count = route_count;
    plan->active_expert_count = active_count;
    plan->route_tile_count = 0;
    return DS4_QWEN_EXPERT_GROUP_OK;
}
