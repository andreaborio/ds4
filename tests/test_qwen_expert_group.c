#include "../ds4_qwen_expert_group.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", \
                __FILE__, __LINE__, #condition); \
        return false; \
    } \
} while (0)

static uint32_t float_bits(float value) {
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static float route_value(
        uint32_t expert,
        uint32_t token,
        uint32_t slot) {
    /* Values span several exponents so a changed accumulation order would be
     * visible in the final bit comparison on ordinary IEEE-754 hosts. */
    static const float scale[] = {
        0x1.0p+12f, 0x1.0p-9f, -0x1.0p+8f,
        0x1.0p-15f, 0x1.8p+3f, -0x1.0p-4f,
    };
    return scale[expert] + (float)(token * 11u + slot) * 0x1.0p-18f;
}

static float canonical_reduce(const float *values, uint32_t n) {
    /* The production Metal reducer has the same sequential slot dependency.
     * Volatile keeps this model-free check meaningful even under ds4's global
     * -ffast-math build flag, which would otherwise be allowed to reassociate
     * one source loop differently from another. */
    volatile float sum = 0.0f;
    for (uint32_t i = 0; i < n; i++) sum += values[i];
    return sum;
}

static bool check_route_tiles(
        const ds4_qwen_expert_group_plan *plan,
        uint32_t                          max_routes_per_tile) {
    CHECK(plan != NULL);
    CHECK(plan->route_tiles != NULL);
    CHECK(plan->route_tile_count > 0);

    uint32_t covered = 0;
    for (uint32_t tile_index = 0;
         tile_index < plan->route_tile_count;
         tile_index++) {
        const ds4_expert_group_route_tile tile =
            plan->route_tiles[tile_index];
        CHECK(tile.route_begin == covered);
        CHECK(tile.route_count > 0);
        CHECK(tile.route_count <= max_routes_per_tile);
        CHECK(tile.reserved == 0);
        CHECK(tile.expert < plan->n_experts);
        CHECK(tile.route_begin >= plan->expert_offsets[tile.expert]);
        CHECK(tile.route_begin + tile.route_count <=
              plan->expert_offsets[tile.expert + 1u]);

        for (uint32_t local = 0; local < tile.route_count; local++) {
            CHECK(plan->grouped_routes[tile.route_begin + local].expert ==
                  tile.expert);
        }

        /* Only the last tile of an expert bucket may be shorter than the
         * requested bound.
         * This proves that a tail does not create a gap or reorder routes. */
        const uint32_t tile_end = tile.route_begin + tile.route_count;
        if (tile_end < plan->expert_offsets[tile.expert + 1u]) {
            CHECK(tile.route_count == max_routes_per_tile);
            CHECK(tile_index + 1u < plan->route_tile_count);
            CHECK(plan->route_tiles[tile_index + 1u].expert == tile.expert);
        }
        covered = tile_end;
    }
    CHECK(covered == plan->route_count);
    return true;
}

static bool test_stable_group_and_canonical_reduce(void) {
    enum { N_TOKEN = 4, N_ROUTE = 4, N_EXPERT = 6 };
    const int32_t selected[N_TOKEN * N_ROUTE] = {
        3, 1, 3, 5,
        1, 0, 5, 1,
        2, 2, 2, 4,
        5, 0, 4, 3,
    };
    const uint32_t expected_offsets[N_EXPERT + 1] = {
        0, 2, 5, 8, 11, 13, 16,
    };
    const uint32_t expected_canonical[N_TOKEN * N_ROUTE] = {
        5, 13,
        1, 4, 7,
        8, 9, 10,
        0, 2, 15,
        11, 14,
        3, 6, 12,
    };

    ds4_qwen_expert_group_plan plan;
    ds4_qwen_expert_group_plan_init(&plan);
    CHECK(ds4_qwen_expert_group_plan_build(
              &plan, selected, N_TOKEN, N_ROUTE, N_EXPERT) ==
          DS4_QWEN_EXPERT_GROUP_OK);
    CHECK(plan.n_tokens == N_TOKEN);
    CHECK(plan.routes_per_token == N_ROUTE);
    CHECK(plan.n_experts == N_EXPERT);
    CHECK(plan.route_count == N_TOKEN * N_ROUTE);
    CHECK(plan.active_expert_count == N_EXPERT);
    CHECK(plan.route_tile_count == 0);
    CHECK(plan.route_tile_capacity == 0);
    CHECK(plan.route_tiles == NULL);
    CHECK(plan.route_tile_storage == NULL);
    CHECK(memcmp(plan.expert_offsets, expected_offsets,
                 sizeof(expected_offsets)) == 0);
    for (uint32_t expert = 0; expert < N_EXPERT; expert++) {
        CHECK(plan.active_experts[expert] == expert);
    }

    bool canonical_seen[N_TOKEN * N_ROUTE] = { false };
    for (uint32_t grouped = 0; grouped < plan.route_count; grouped++) {
        const ds4_qwen_expert_group_route *route =
            &plan.grouped_routes[grouped];
        CHECK(route->canonical_index == expected_canonical[grouped]);
        CHECK(route->token_row == route->canonical_index / N_ROUTE);
        CHECK(route->route_slot == route->canonical_index % N_ROUTE);
        CHECK((uint32_t)selected[route->canonical_index] == route->expert);
        CHECK(!canonical_seen[route->canonical_index]);
        canonical_seen[route->canonical_index] = true;
        CHECK(plan.canonical_to_grouped[route->canonical_index] == grouped);

        if (grouped > plan.expert_offsets[route->expert]) {
            const ds4_qwen_expert_group_route *previous =
                &plan.grouped_routes[grouped - 1u];
            CHECK(previous->expert == route->expert);
            CHECK(previous->canonical_index < route->canonical_index);
        }
    }
    for (uint32_t canonical = 0; canonical < plan.route_count; canonical++) {
        CHECK(canonical_seen[canonical]);
    }

    /* Simulate expert-major execution into a per-route buffer, followed by the
     * only permitted reduction: token-major, increasing route slot. */
    float baseline_output[N_TOKEN * N_ROUTE] = { 0.0f };
    float grouped_output[N_TOKEN * N_ROUTE] = { 0.0f };
    for (uint32_t canonical = 0; canonical < plan.route_count; canonical++) {
        const uint32_t token = canonical / N_ROUTE;
        const uint32_t slot = canonical % N_ROUTE;
        baseline_output[canonical] =
            route_value((uint32_t)selected[canonical], token, slot);
    }
    for (uint32_t grouped = 0; grouped < plan.route_count; grouped++) {
        const ds4_qwen_expert_group_route route =
            plan.grouped_routes[grouped];
        grouped_output[route.canonical_index] =
            route_value(route.expert, route.token_row, route.route_slot);
    }
    for (uint32_t token = 0; token < N_TOKEN; token++) {
        for (uint32_t slot = 0; slot < N_ROUTE; slot++) {
            const uint32_t canonical = token * N_ROUTE + slot;
            CHECK(float_bits(grouped_output[canonical]) ==
                  float_bits(baseline_output[canonical]));
        }
        const float baseline = canonical_reduce(
            baseline_output + token * N_ROUTE, N_ROUTE);
        const float grouped = canonical_reduce(
            grouped_output + token * N_ROUTE, N_ROUTE);
        CHECK(float_bits(grouped) == float_bits(baseline));
    }

    ds4_qwen_expert_group_plan_destroy(&plan);
    return true;
}

static bool test_inactive_experts_and_duplicate_slots(void) {
    const int32_t selected[] = {
        4, 4, 4, 1,
        4, 1, 4, 4,
    };
    ds4_qwen_expert_group_plan plan;
    ds4_qwen_expert_group_plan_init(&plan);
    CHECK(ds4_qwen_expert_group_plan_reserve_route_tiles(
              &plan, (uint32_t)(sizeof(selected) / sizeof(selected[0]))) ==
          DS4_QWEN_EXPERT_GROUP_OK);
    CHECK(ds4_qwen_expert_group_plan_build(
              &plan, selected, 2, 4, 7) ==
          DS4_QWEN_EXPERT_GROUP_OK);
    CHECK(plan.route_tile_count == 0);
    CHECK(ds4_qwen_expert_group_plan_retile(
              &plan, DS4_EXPERT_GROUP_ROUTE_TILE_SIZE) ==
          DS4_QWEN_EXPERT_GROUP_OK);

    const uint32_t expected_offsets[] = { 0, 0, 2, 2, 2, 8, 8, 8 };
    CHECK(memcmp(plan.expert_offsets, expected_offsets,
                 sizeof(expected_offsets)) == 0);
    CHECK(plan.active_expert_count == 2);
    CHECK(plan.active_experts[0] == 1);
    CHECK(plan.active_experts[1] == 4);
    CHECK(plan.route_tile_count == 3);
    CHECK(check_route_tiles(&plan, DS4_EXPERT_GROUP_ROUTE_TILE_SIZE));
    CHECK(plan.route_tiles[0].expert == 1);
    CHECK(plan.route_tiles[0].route_begin == 0);
    CHECK(plan.route_tiles[0].route_count == 2);
    CHECK(plan.route_tiles[1].expert == 4);
    CHECK(plan.route_tiles[1].route_begin == 2);
    CHECK(plan.route_tiles[1].route_count == 4);
    CHECK(plan.route_tiles[2].expert == 4);
    CHECK(plan.route_tiles[2].route_begin == 6);
    CHECK(plan.route_tiles[2].route_count == 2);

    /* Six occurrences of expert 4 remain six routes.  Deduplicating them
     * would silently discard distinct top-k slots/weights. */
    CHECK(plan.expert_offsets[5] - plan.expert_offsets[4] == 6);
    const uint32_t expected_expert4_canonical[] = { 0, 1, 2, 4, 6, 7 };
    for (uint32_t i = 0; i < 6; i++) {
        CHECK(plan.grouped_routes[plan.expert_offsets[4] + i]
                  .canonical_index == expected_expert4_canonical[i]);
    }

    ds4_qwen_expert_group_plan_destroy(&plan);
    return true;
}

static bool test_configurable_retile_preserves_permutation(void) {
    enum { N_TOKEN = 5, N_ROUTE = 4, N_EXPERT = 6, ROUTE_COUNT = 20 };
    const int32_t selected[ROUTE_COUNT] = {
        4, 2, 5, 3,
        1, 4, 3, 5,
        2, 4, 0, 5,
        3, 4, 2, 5,
        1, 4, 3, 5,
    };

    ds4_qwen_expert_group_plan plan;
    ds4_qwen_expert_group_plan_init(&plan);
    CHECK(ds4_qwen_expert_group_plan_reserve_route_tiles(
              &plan, ROUTE_COUNT) == DS4_QWEN_EXPERT_GROUP_OK);
    CHECK(ds4_qwen_expert_group_plan_build(
              &plan, selected, N_TOKEN, N_ROUTE, N_EXPERT) ==
          DS4_QWEN_EXPERT_GROUP_OK);
    CHECK(plan.route_count == ROUTE_COUNT);
    CHECK(plan.route_tile_count == 0);
    CHECK(ds4_qwen_expert_group_plan_retile(
              &plan, DS4_EXPERT_GROUP_ROUTE_TILE_SIZE) ==
          DS4_QWEN_EXPERT_GROUP_OK);
    CHECK(plan.route_tile_count == 8);
    CHECK(check_route_tiles(&plan, DS4_EXPERT_GROUP_ROUTE_TILE_SIZE));

    const void *const storage = plan.storage;
    const void *const tile_storage = plan.route_tile_storage;
    ds4_qwen_expert_group_route grouped_before[ROUTE_COUNT];
    uint32_t canonical_before[ROUTE_COUNT];
    ds4_expert_group_route_tile default_tiles[ROUTE_COUNT];
    memcpy(grouped_before, plan.grouped_routes, sizeof(grouped_before));
    memcpy(canonical_before, plan.canonical_to_grouped,
           sizeof(canonical_before));
    memcpy(default_tiles, plan.route_tiles,
           (size_t)plan.route_tile_count * sizeof(default_tiles[0]));
    const uint32_t default_tile_count = plan.route_tile_count;

    CHECK(ds4_qwen_expert_group_plan_retile(&plan, 2) ==
          DS4_QWEN_EXPERT_GROUP_OK);
    CHECK(plan.storage == storage);
    CHECK(plan.route_tile_storage == tile_storage);
    CHECK(plan.route_tiles == tile_storage);
    CHECK(plan.route_tile_count == 12);
    CHECK(check_route_tiles(&plan, 2));
    CHECK(memcmp(plan.grouped_routes, grouped_before,
                 sizeof(grouped_before)) == 0);
    CHECK(memcmp(plan.canonical_to_grouped, canonical_before,
                 sizeof(canonical_before)) == 0);

    CHECK(ds4_qwen_expert_group_plan_retile(&plan, 1) ==
          DS4_QWEN_EXPERT_GROUP_OK);
    CHECK(plan.route_tile_count == ROUTE_COUNT);
    CHECK(check_route_tiles(&plan, 1));
    CHECK(ds4_qwen_expert_group_plan_retile(&plan, 2) ==
          DS4_QWEN_EXPERT_GROUP_OK);
    CHECK(plan.route_tile_count == 12);
    CHECK(check_route_tiles(&plan, 2));

    /* Invalid widths and malformed source metadata must leave the complete
     * two-route tile view untouched. */
    ds4_expert_group_route_tile tiles_at_two[ROUTE_COUNT];
    memcpy(tiles_at_two, plan.route_tiles,
           (size_t)plan.route_tile_count * sizeof(tiles_at_two[0]));
    const uint32_t tiles_at_two_count = plan.route_tile_count;
    CHECK(ds4_qwen_expert_group_plan_retile(&plan, 0) ==
          DS4_QWEN_EXPERT_GROUP_INVALID_ARGUMENT);
    CHECK(ds4_qwen_expert_group_plan_retile(
              &plan, DS4_EXPERT_GROUP_ROUTE_TILE_SIZE + 1u) ==
          DS4_QWEN_EXPERT_GROUP_INVALID_ARGUMENT);
    CHECK(plan.route_tile_count == tiles_at_two_count);
    CHECK(memcmp(plan.route_tiles, tiles_at_two,
                 (size_t)tiles_at_two_count * sizeof(tiles_at_two[0])) == 0);

    const uint32_t first_expert = plan.grouped_routes[0].expert;
    plan.grouped_routes[0].expert = (first_expert + 1u) % N_EXPERT;
    CHECK(ds4_qwen_expert_group_plan_retile(
              &plan, DS4_EXPERT_GROUP_ROUTE_TILE_SIZE) ==
          DS4_QWEN_EXPERT_GROUP_INVALID_ARGUMENT);
    CHECK(plan.route_tile_count == tiles_at_two_count);
    CHECK(memcmp(plan.route_tiles, tiles_at_two,
                 (size_t)tiles_at_two_count * sizeof(tiles_at_two[0])) == 0);
    plan.grouped_routes[0].expert = first_expert;

    CHECK(ds4_qwen_expert_group_plan_retile(
              &plan, DS4_EXPERT_GROUP_ROUTE_TILE_SIZE) ==
          DS4_QWEN_EXPERT_GROUP_OK);
    CHECK(plan.route_tile_count == default_tile_count);
    CHECK(check_route_tiles(&plan, DS4_EXPERT_GROUP_ROUTE_TILE_SIZE));
    CHECK(memcmp(plan.route_tiles, default_tiles,
                 (size_t)default_tile_count * sizeof(default_tiles[0])) == 0);
    CHECK(memcmp(plan.grouped_routes, grouped_before,
                 sizeof(grouped_before)) == 0);
    CHECK(memcmp(plan.canonical_to_grouped, canonical_before,
                 sizeof(canonical_before)) == 0);

    ds4_qwen_expert_group_plan_destroy(&plan);
    return true;
}

static bool test_reuse_and_transactional_validation(void) {
    const int32_t selected_a[] = { 2, 0, 1, 2, 3, 1, 0, 3 };
    const int32_t selected_b[] = { 1, 0 };
    const int32_t selected_negative[] = { 1, -1 };
    const int32_t selected_too_large[] = { 1, 4 };
    ds4_qwen_expert_group_plan plan;
    ds4_qwen_expert_group_plan_init(&plan);
    CHECK(ds4_qwen_expert_group_plan_reserve_route_tiles(
              &plan, (uint32_t)(sizeof(selected_a) /
                               sizeof(selected_a[0]))) ==
          DS4_QWEN_EXPERT_GROUP_OK);
    CHECK(ds4_qwen_expert_group_plan_build(
              &plan, selected_a, 2, 4, 4) ==
          DS4_QWEN_EXPERT_GROUP_OK);
    CHECK(ds4_qwen_expert_group_plan_retile(
              &plan, DS4_EXPERT_GROUP_ROUTE_TILE_SIZE) ==
          DS4_QWEN_EXPERT_GROUP_OK);
    void *const storage = plan.storage;
    const uint32_t old_route_count = plan.route_count;
    const uint32_t old_first_canonical =
        plan.grouped_routes[0].canonical_index;
    const uint32_t old_route_tile_count = plan.route_tile_count;
    const ds4_expert_group_route_tile old_first_tile = plan.route_tiles[0];

    CHECK(ds4_qwen_expert_group_plan_build(
              &plan, selected_negative, 1, 2, 4) ==
          DS4_QWEN_EXPERT_GROUP_EXPERT_OUT_OF_RANGE);
    CHECK(plan.storage == storage);
    CHECK(plan.route_count == old_route_count);
    CHECK(plan.grouped_routes[0].canonical_index == old_first_canonical);
    CHECK(plan.route_tile_count == old_route_tile_count);
    CHECK(memcmp(&plan.route_tiles[0], &old_first_tile,
                 sizeof(old_first_tile)) == 0);

    CHECK(ds4_qwen_expert_group_plan_build(
              &plan, selected_too_large, 1, 2, 4) ==
          DS4_QWEN_EXPERT_GROUP_EXPERT_OUT_OF_RANGE);
    CHECK(plan.storage == storage);
    CHECK(plan.route_count == old_route_count);

    CHECK(ds4_qwen_expert_group_plan_build(
              &plan, selected_b, 1, 2, 4) ==
          DS4_QWEN_EXPERT_GROUP_OK);
    CHECK(plan.storage == storage);
    CHECK(plan.route_count == 2);
    CHECK(plan.route_tile_count == 0);
    CHECK(ds4_qwen_expert_group_plan_retile(
              &plan, DS4_EXPERT_GROUP_ROUTE_TILE_SIZE) ==
          DS4_QWEN_EXPERT_GROUP_OK);
    CHECK(plan.route_tile_count == 2);
    CHECK(check_route_tiles(&plan, DS4_EXPERT_GROUP_ROUTE_TILE_SIZE));
    CHECK(plan.expert_capacity >= 4);
    CHECK(plan.route_capacity >= 8);

    ds4_qwen_expert_group_plan_destroy(&plan);
    return true;
}

static bool test_argument_and_overflow_guards(void) {
    const int32_t selected[] = { 0 };
    ds4_qwen_expert_group_plan plan;
    ds4_qwen_expert_group_plan_init(&plan);

    CHECK(ds4_qwen_expert_group_plan_retile(&plan, 2) ==
          DS4_QWEN_EXPERT_GROUP_INVALID_ARGUMENT);
    CHECK(ds4_qwen_expert_group_plan_reserve_route_tiles(NULL, 1) ==
          DS4_QWEN_EXPERT_GROUP_INVALID_ARGUMENT);
    CHECK(ds4_qwen_expert_group_plan_reserve_route_tiles(&plan, 0) ==
          DS4_QWEN_EXPERT_GROUP_INVALID_ARGUMENT);

    CHECK(ds4_qwen_expert_group_plan_build(
              NULL, selected, 1, 1, 1) ==
          DS4_QWEN_EXPERT_GROUP_INVALID_ARGUMENT);
    CHECK(ds4_qwen_expert_group_plan_build(
              &plan, NULL, 1, 1, 1) ==
          DS4_QWEN_EXPERT_GROUP_INVALID_ARGUMENT);
    CHECK(ds4_qwen_expert_group_plan_build(
              &plan, selected, 0, 1, 1) ==
          DS4_QWEN_EXPERT_GROUP_INVALID_ARGUMENT);
    CHECK(ds4_qwen_expert_group_plan_build(
              &plan, selected, 1, 0, 1) ==
          DS4_QWEN_EXPERT_GROUP_INVALID_ARGUMENT);
    CHECK(ds4_qwen_expert_group_plan_build(
              &plan, selected, 1, 1, 0) ==
          DS4_QWEN_EXPERT_GROUP_INVALID_ARGUMENT);
    CHECK(ds4_qwen_expert_group_plan_build(
              &plan, selected, UINT32_MAX, 2, 1) ==
          DS4_QWEN_EXPERT_GROUP_OVERFLOW);
    CHECK(ds4_qwen_expert_group_plan_build(
              &plan, selected, 1, 1, UINT32_MAX) ==
          DS4_QWEN_EXPERT_GROUP_OVERFLOW);
    CHECK(ds4_qwen_expert_group_plan_reserve(
              &plan, UINT32_MAX, 1) ==
          DS4_QWEN_EXPERT_GROUP_OVERFLOW);

    ds4_qwen_expert_group_plan_destroy(&plan);
    return true;
}

static bool test_qwen_long_context_capacity(void) {
    enum { N_TOKEN = 8192, N_ROUTE = 8, N_EXPERT = 256 };
    const uint32_t route_count = N_TOKEN * N_ROUTE;
    int32_t *selected = malloc((size_t)route_count * sizeof(selected[0]));
    CHECK(selected != NULL);
    for (uint32_t token = 0; token < N_TOKEN; token++) {
        for (uint32_t slot = 0; slot < N_ROUTE; slot++) {
            selected[token * N_ROUTE + slot] =
                (int32_t)((token * 17u + slot * 31u) % N_EXPERT);
        }
    }

    ds4_qwen_expert_group_plan plan;
    ds4_qwen_expert_group_plan_init(&plan);
    CHECK(ds4_qwen_expert_group_plan_build(
              &plan, selected, N_TOKEN, N_ROUTE, N_EXPERT) ==
          DS4_QWEN_EXPERT_GROUP_OK);
    CHECK(plan.route_count == route_count);
    CHECK(plan.active_expert_count == N_EXPERT);
    CHECK(plan.expert_offsets[0] == 0);
    CHECK(plan.expert_offsets[N_EXPERT] == route_count);
    /* This is the Qwen-sized hot path that motivated lazy tiles: 65,536
     * routes must not allocate an otherwise unused extra 1 MiB tile array. */
    CHECK(plan.route_tile_count == 0);
    CHECK(plan.route_tile_capacity == 0);
    CHECK(plan.route_tiles == NULL);
    CHECK(plan.route_tile_storage == NULL);

    for (uint32_t expert = 0; expert < N_EXPERT; expert++) {
        CHECK(plan.active_experts[expert] == expert);
        uint32_t previous = 0;
        bool have_previous = false;
        for (uint32_t grouped = plan.expert_offsets[expert];
             grouped < plan.expert_offsets[expert + 1u];
             grouped++) {
            const ds4_qwen_expert_group_route route =
                plan.grouped_routes[grouped];
            CHECK(route.expert == expert);
            CHECK(!have_previous || route.canonical_index > previous);
            CHECK(plan.canonical_to_grouped[route.canonical_index] == grouped);
            previous = route.canonical_index;
            have_previous = true;
        }
    }

    ds4_qwen_expert_group_plan_destroy(&plan);
    free(selected);
    return true;
}

int main(void) {
    if (!test_stable_group_and_canonical_reduce() ||
        !test_inactive_experts_and_duplicate_slots() ||
        !test_configurable_retile_preserves_permutation() ||
        !test_reuse_and_transactional_validation() ||
        !test_argument_and_overflow_guards() ||
        !test_qwen_long_context_capacity()) {
        return 1;
    }
    puts("test_qwen_expert_group: OK");
    return 0;
}
