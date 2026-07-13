#include "ds4_ssd.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#define GIB (UINT64_C(1024) * 1024u * 1024u)

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
    puts("ssd residency resolver: ok");
    return 0;
}
