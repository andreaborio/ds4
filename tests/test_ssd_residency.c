#include "ds4_ssd.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#define GIB (UINT64_C(1024) * 1024u * 1024u)
#define MIB (UINT64_C(1024) * 1024u)

static ds4_ssd_host_memory memory_for_raw_experts_on_host(
        uint64_t physical_bytes,
        uint64_t recommended_bytes,
        uint64_t raw_experts,
        uint64_t per_expert_bytes) {
    uint64_t current_headroom = physical_bytes / 16u;
    if (current_headroom < 2 * GIB) current_headroom = 2 * GIB;
    uint64_t pressure_margin = physical_bytes / 64u;
    if (pressure_margin < GIB / 4u) pressure_margin = GIB / 4u;
    ds4_ssd_host_memory memory = {
        .physical_bytes = physical_bytes,
        .recommended_bytes = recommended_bytes,
        .free_bytes = current_headroom + pressure_margin +
                      raw_experts * per_expert_bytes,
    };
    return memory;
}

static ds4_ssd_host_memory memory_for_raw_experts(uint64_t raw_experts,
                                                   uint64_t per_expert_bytes) {
    return memory_for_raw_experts_on_host(64 * GIB,
                                          64 * GIB,
                                          raw_experts,
                                          per_expert_bytes);
}

static ds4_residency_plan plan(bool metal,
                               ds4_residency_mode requested,
                               uint64_t model,
                               uint64_t runtime,
                               uint64_t recommended,
                               uint64_t external) {
    ds4_residency_plan p;
    assert(ds4_residency_plan_make(metal,
                                   requested,
                                   model,
                                   runtime,
                                   recommended,
                                   external,
                                   &p));
    return p;
}

int main(void) {
    ds4_residency_plan p = plan(true,
                                DS4_RESIDENCY_RESIDENT,
                                UINT64_MAX,
                                UINT64_MAX,
                                1,
                                1);
    assert(p.resolved == DS4_RESIDENCY_RESIDENT);
    assert(p.reason == DS4_RESIDENCY_REASON_EXPLICIT_RESIDENT);

    p = plan(false, DS4_RESIDENCY_SSD, 1, 1, 0, 0);
    assert(p.resolved == DS4_RESIDENCY_SSD);
    assert(p.reason == DS4_RESIDENCY_REASON_EXPLICIT_SSD);

    p = plan(false, DS4_RESIDENCY_AUTO, UINT64_MAX, UINT64_MAX, 0, 0);
    assert(p.resolved == DS4_RESIDENCY_RESIDENT);
    assert(p.reason == DS4_RESIDENCY_REASON_NON_METAL_AUTO);

    /* At 10 GiB the fixed minimum and 20% headroom are both 2 GiB.
     * Model + runtime + headroom exactly equals the budget. */
    p = plan(true, DS4_RESIDENCY_AUTO, 6 * GIB, 2 * GIB, 10 * GIB, 0);
    assert(p.required_bytes == 10 * GIB);
    assert(p.budget_bytes == 10 * GIB);
    assert(p.resolved == DS4_RESIDENCY_RESIDENT);
    assert(p.reason == DS4_RESIDENCY_REASON_METAL_FITS);

    p = plan(true, DS4_RESIDENCY_AUTO, 6 * GIB + 1, 2 * GIB,
             10 * GIB, 0);
    assert(p.resolved == DS4_RESIDENCY_SSD);
    assert(p.reason == DS4_RESIDENCY_REASON_METAL_EXCEEDS);

    p = plan(true, DS4_RESIDENCY_AUTO, 1, 1, 10 * GIB, 10 * GIB);
    assert(p.budget_bytes == 0);
    assert(p.resolved == DS4_RESIDENCY_SSD);

    p = plan(true, DS4_RESIDENCY_AUTO, 1, 1, 10 * GIB, 11 * GIB);
    assert(p.budget_bytes == 0);
    assert(p.resolved == DS4_RESIDENCY_SSD);

    p = plan(true, DS4_RESIDENCY_AUTO, UINT64_MAX, UINT64_MAX,
             UINT64_MAX, 1);
    assert(p.required_bytes == UINT64_MAX);
    assert(p.resolved == DS4_RESIDENCY_SSD);

    p = plan(true, DS4_RESIDENCY_AUTO, 1, 1, 0, 0);
    assert(p.resolved == DS4_RESIDENCY_SSD);
    assert(p.reason == DS4_RESIDENCY_REASON_METAL_BUDGET_UNAVAILABLE);

    assert(!ds4_residency_plan_make(true,
                                    (ds4_residency_mode)99,
                                    0, 0, 0, 0, &p));

    uint64_t available = 0;
    uint64_t reserved = 0;
    assert(ds4_ssd_working_set_after_reserve(10 * GIB,
                                             2 * GIB,
                                             1 * GIB,
                                             &available,
                                             &reserved));
    assert(reserved == 3 * GIB);
    assert(available == 7 * GIB);
    assert(!ds4_ssd_working_set_after_reserve(10 * GIB,
                                              9 * GIB,
                                              1 * GIB,
                                              &available,
                                              &reserved));
    assert(available == 0);
    assert(reserved == 10 * GIB);
    assert(!ds4_ssd_working_set_after_reserve(UINT64_MAX,
                                              UINT64_MAX,
                                              1,
                                              &available,
                                              &reserved));
    assert(reserved == UINT64_MAX);

    ds4_ssd_cache_plan cache;
    assert(ds4_ssd_cache_plan_for_model_target(10 * GIB,
                                                2 * GIB,
                                                1 * GIB,
                                                100,
                                                &cache));
    assert(cache.model_target_bytes == 10 * GIB);
    assert(cache.cache_experts == 8);
    assert(cache.effective_cache_bytes == 8 * GIB);
    assert(!ds4_ssd_cache_plan_for_model_target(2 * GIB,
                                                 2 * GIB,
                                                 1 * GIB,
                                                 100,
                                                 &cache));
    assert(!ds4_ssd_cache_plan_for_model_target(2 * GIB + GIB / 2,
                                                 2 * GIB,
                                                 1 * GIB,
                                                 100,
                                                 &cache));

    ds4_ssd_expert_cache_floor floor;
    assert(ds4_ssd_expert_cache_floor_make(43,
                                            6,
                                            UINT64_C(7077888),
                                            &floor));
    assert(floor.working_set_experts == 258);
    assert(floor.minimum_cache_experts == 259);
    assert(floor.minimum_cache_bytes == UINT64_C(1833172992));
    assert(floor.warning_cache_experts == 516);

    /* PRO has 61 routed layers.  Mixed-precision layers which bypass the
     * uniform expert slab are excluded by passing the cacheable count. */
    assert(ds4_ssd_expert_cache_floor_make(61, 6, 1, &floor));
    assert(floor.working_set_experts == 366);
    assert(floor.minimum_cache_experts == 367);
    assert(floor.warning_cache_experts == 732);
    assert(ds4_ssd_expert_cache_floor_make(55, 6, 1, &floor));
    assert(floor.working_set_experts == 330);
    assert(floor.minimum_cache_experts == 331);

    assert(!ds4_ssd_expert_cache_floor_make(0, 6, 1, &floor));
    assert(!ds4_ssd_expert_cache_floor_make(1, 0, 1, &floor));
    assert(!ds4_ssd_expert_cache_floor_make(1, 1, 0, &floor));
    assert(!ds4_ssd_expert_cache_floor_make(UINT64_MAX, 2, 1, &floor));
    assert(!ds4_ssd_expert_cache_floor_make(UINT64_MAX, 1, 1, &floor));
    assert(!ds4_ssd_expert_cache_floor_make(UINT64_MAX / 2u + 1u,
                                             1,
                                             1,
                                             &floor));
    assert(!ds4_ssd_expert_cache_floor_make(1,
                                             1,
                                             UINT64_MAX,
                                             &floor));
    assert(!ds4_ssd_expert_cache_floor_make(1, 1, 1, NULL));

    const uint64_t flash_expert_bytes = UINT64_C(7077888);
    const uint64_t flash_max_cacheable = UINT64_C(43) * 256u;
    ds4_ssd_host_memory memory = {
        .physical_bytes = 16 * GIB,
        .recommended_bytes = 12 * GIB,
        .task_footprint_bytes = 21 * GIB / 2u,
        .free_bytes = 11 * GIB / 2u,
    };
    ds4_ssd_adaptive_cache_plan adaptive;
    assert(ds4_ssd_adaptive_cache_plan_make(&memory,
                                             512 * MIB,
                                             43,
                                             6,
                                             flash_expert_bytes,
                                             flash_max_cacheable,
                                             &adaptive));
    assert(adaptive.reclaimable_bytes == 11 * GIB / 2u);
    assert(adaptive.current_headroom_bytes == 2 * GIB);
    assert(adaptive.pressure_margin_bytes == GIB / 4u);
    assert(adaptive.platform_headroom_bytes == 2 * GIB);
    assert(adaptive.current_wire_budget_bytes == 11 * GIB / 4u);
    assert(adaptive.platform_wire_budget_bytes == 19 * GIB / 2u);
    assert(adaptive.wire_budget_bytes == 11 * GIB / 4u);
    assert(adaptive.cache_experts == 259);
    assert(adaptive.cache_bytes == UINT64_C(259) * flash_expert_bytes);
    assert(adaptive.floor.working_set_experts == 258);
    assert(adaptive.low_ram_floor_ceiling_active);

    /* High free memory does not buy the slower second tier on a 16 GiB host.
     * The same policy still fails closed when even the correctness floor does
     * not fit. */
    memory = memory_for_raw_experts_on_host(16 * GIB,
                                            12 * GIB,
                                            517,
                                            flash_expert_bytes);
    assert(ds4_ssd_adaptive_cache_plan_make(&memory, 0, 43, 6,
                                             flash_expert_bytes,
                                             flash_max_cacheable,
                                             &adaptive));
    assert(adaptive.low_ram_floor_ceiling_active);
    assert(adaptive.wire_budget_bytes / flash_expert_bytes == 517);
    assert(adaptive.cache_experts == 259);
    memory = memory_for_raw_experts_on_host(16 * GIB,
                                            12 * GIB,
                                            258,
                                            flash_expert_bytes);
    assert(!ds4_ssd_adaptive_cache_plan_make(&memory, 0, 43, 6,
                                              flash_expert_bytes,
                                              flash_max_cacheable,
                                              &adaptive));
    assert(adaptive.low_ram_floor_ceiling_active);

    /* The planner never recreates AUTO=119.  Below the correctness floor it
     * fails closed; above it, capacity advances only in complete 258-entry
     * per-token working sets. */
    memory = memory_for_raw_experts(119, flash_expert_bytes);
    assert(!ds4_ssd_adaptive_cache_plan_make(&memory, 0, 43, 6,
                                              flash_expert_bytes,
                                              flash_max_cacheable,
                                              &adaptive));
    memory = memory_for_raw_experts(258, flash_expert_bytes);
    assert(!ds4_ssd_adaptive_cache_plan_make(&memory, 0, 43, 6,
                                              flash_expert_bytes,
                                              flash_max_cacheable,
                                              &adaptive));
    memory = memory_for_raw_experts(259, flash_expert_bytes);
    assert(ds4_ssd_adaptive_cache_plan_make(&memory, 0, 43, 6,
                                             flash_expert_bytes,
                                             flash_max_cacheable,
                                             &adaptive));
    assert(adaptive.cache_experts == 259);
    memory = memory_for_raw_experts(516, flash_expert_bytes);
    assert(ds4_ssd_adaptive_cache_plan_make(&memory, 0, 43, 6,
                                             flash_expert_bytes,
                                             flash_max_cacheable,
                                             &adaptive));
    assert(adaptive.cache_experts == 259);
    memory = memory_for_raw_experts(517, flash_expert_bytes);
    assert(ds4_ssd_adaptive_cache_plan_make(&memory, 0, 43, 6,
                                             flash_expert_bytes,
                                             flash_max_cacheable,
                                             &adaptive));
    assert(adaptive.cache_experts == 517);
    assert(!adaptive.low_ram_floor_ceiling_active);

    memory = memory_for_raw_experts_on_host(32 * GIB,
                                            24 * GIB,
                                            517,
                                            flash_expert_bytes);
    assert(ds4_ssd_adaptive_cache_plan_make(&memory, 0, 43, 6,
                                             flash_expert_bytes,
                                             flash_max_cacheable,
                                             &adaptive));
    assert(!adaptive.low_ram_floor_ceiling_active);
    assert(adaptive.cache_experts == 517);

    /* Engine-open planning happens before the session allocates its modeled
     * runtime footprint.  Use a host above the low-RAM ceiling so this test
     * isolates the runtime reserve: a future 512 MiB session must prevent the
     * 517-entry tier. */
    memory = memory_for_raw_experts_on_host(32 * GIB,
                                            24 * GIB,
                                            517,
                                            flash_expert_bytes);
    assert(ds4_ssd_adaptive_cache_plan_make(&memory, 512 * MIB, 43, 6,
                                             flash_expert_bytes,
                                             flash_max_cacheable,
                                             &adaptive));
    assert(!adaptive.low_ram_floor_ceiling_active);
    assert(adaptive.cache_experts == 259);
    assert(adaptive.current_wire_budget_bytes ==
           UINT64_C(517) * flash_expert_bytes - 512 * MIB);
    memory = memory_for_raw_experts(775, flash_expert_bytes);
    assert(ds4_ssd_adaptive_cache_plan_make(&memory, 0, 43, 6,
                                             flash_expert_bytes,
                                             flash_max_cacheable,
                                             &adaptive));
    assert(adaptive.cache_experts == 775);
    assert((adaptive.cache_experts - 1u) % 258u == 0);

    /* Model capacity remains a hard upper bound before cycle rounding. */
    memory = memory_for_raw_experts(517, flash_expert_bytes);
    assert(ds4_ssd_adaptive_cache_plan_make(&memory, 0, 43, 6,
                                             flash_expert_bytes,
                                             259,
                                             &adaptive));
    assert(adaptive.cache_experts == 259);

    /* A budget that fits every cacheable expert is terminal: rounding to
     * 1+k*working_set would needlessly leave one partial cycle uncached. */
    memory = memory_for_raw_experts(6, 1);
    assert(ds4_ssd_adaptive_cache_plan_make(&memory, 0, 2, 1,
                                             1, 6, &adaptive));
    assert(adaptive.floor.working_set_experts == 2);
    assert(adaptive.cache_experts == 6);
    assert(adaptive.cache_bytes == 6);

    memory = (ds4_ssd_host_memory){
        .physical_bytes = UINT64_MAX,
        .recommended_bytes = UINT64_MAX,
        .free_bytes = UINT64_MAX,
        .purgeable_bytes = UINT64_MAX,
        .inactive_bytes = UINT64_MAX,
        .file_backed_bytes = UINT64_MAX,
    };
    assert(ds4_ssd_adaptive_cache_plan_make(&memory, 0, 43, 6,
                                             flash_expert_bytes,
                                             flash_max_cacheable,
                                             &adaptive));
    assert(adaptive.reclaimable_bytes == UINT64_MAX);
    assert(adaptive.cache_experts == flash_max_cacheable);

    memory = (ds4_ssd_host_memory){0};
    assert(!ds4_ssd_adaptive_cache_plan_make(&memory, 0, 43, 6,
                                              flash_expert_bytes,
                                              flash_max_cacheable,
                                              &adaptive));
    memory.physical_bytes = 16 * GIB;
    assert(!ds4_ssd_adaptive_cache_plan_make(&memory, 0, 43, 6,
                                              flash_expert_bytes,
                                              flash_max_cacheable,
                                              &adaptive));
    memory = (ds4_ssd_host_memory){
        .physical_bytes = 64 * GIB,
        .recommended_bytes = 12 * GIB,
        .free_bytes = 40 * GIB,
    };
    assert(!ds4_ssd_adaptive_cache_plan_make(&memory, 10 * GIB, 43, 6,
                                              flash_expert_bytes,
                                              flash_max_cacheable,
                                              &adaptive));
    assert(adaptive.platform_wire_budget_bytes == 0);
    assert(!ds4_ssd_adaptive_cache_plan_make(&memory, UINT64_MAX, 43, 6,
                                              flash_expert_bytes,
                                              flash_max_cacheable,
                                              &adaptive));
    assert(adaptive.current_wire_budget_bytes == 0);
    assert(adaptive.platform_wire_budget_bytes == 0);
    assert(!ds4_ssd_adaptive_cache_plan_make(&memory, 0, 43, 6,
                                              flash_expert_bytes,
                                              0,
                                              &adaptive));
    assert(!ds4_ssd_adaptive_cache_plan_make(NULL, 0, 43, 6,
                                              flash_expert_bytes,
                                              flash_max_cacheable,
                                              &adaptive));
    assert(!ds4_ssd_adaptive_cache_plan_make(&memory, 0, 43, 6,
                                              flash_expert_bytes,
                                              flash_max_cacheable,
                                              NULL));
    puts("ssd residency resolver: ok");
    return 0;
}
