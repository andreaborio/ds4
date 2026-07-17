#include "ds4_ssd.h"
#include "ds4_qwen.h"

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

    ds4_ssd_resident_pressure_plan pressure = {0};
    ds4_ssd_host_memory pressure_memory = {
        .physical_bytes = 64 * GIB,
        .free_bytes = 30 * GIB,
        .purgeable_bytes = 2 * GIB,
        .inactive_bytes = 8 * GIB,
        .file_backed_bytes = 10 * GIB,
    };
    assert(ds4_ssd_resident_pressure_plan_make(
        &pressure_memory, 20 * GIB, 1 * GIB, &pressure));
    assert(pressure.inactive_credit_bytes == 4 * GIB);
    assert(pressure.reclaimable_bytes == 34 * GIB);
    assert(pressure.current_headroom_bytes == 4 * GIB);
    assert(pressure.pressure_margin_bytes == 1 * GIB);
    assert(pressure.required_bytes == 26 * GIB);
    assert(pressure.fits);

    /* Normal Darwin pressure allows the complete bounded inactive working-set
     * proxy as resident admission credit. Elevated or unavailable pressure
     * retains conservative half credit. */
    pressure_memory = (ds4_ssd_host_memory){
        .physical_bytes = 64 * GIB,
        .free_bytes = 4 * GIB,
        .inactive_bytes = 24 * GIB,
        .file_backed_bytes = 20 * GIB,
        .pressure_status_available = true,
        .pressure_normal = true,
    };
    assert(ds4_ssd_resident_pressure_plan_make(
        &pressure_memory, 18 * GIB, 1 * GIB, &pressure));
    assert(pressure.inactive_credit_bytes == 20 * GIB);
    assert(pressure.reclaimable_bytes == 24 * GIB);
    assert(pressure.pressure_normal);
    assert(pressure.fits);
    pressure_memory.pressure_normal = false;
    assert(ds4_ssd_resident_pressure_plan_make(
        &pressure_memory, 18 * GIB, 1 * GIB, &pressure));
    assert(pressure.inactive_credit_bytes == 10 * GIB);
    assert(pressure.reclaimable_bytes == 14 * GIB);
    assert(!pressure.pressure_normal);
    assert(!pressure.fits);
    pressure_memory.pressure_status_available = false;
    pressure_memory.pressure_normal = true;
    assert(ds4_ssd_resident_pressure_plan_make(
        &pressure_memory, 18 * GIB, 1 * GIB, &pressure));
    assert(pressure.inactive_credit_bytes == 10 * GIB);
    assert(!pressure.pressure_normal);
    assert(!pressure.fits);

    /* Purgeable pages may already belong to an inactive queue. The resident
     * gate uses the larger reclaimable pool and never adds both. */
    pressure_memory = (ds4_ssd_host_memory){
        .physical_bytes = 64 * GIB,
        .free_bytes = 4 * GIB,
        .purgeable_bytes = 12 * GIB,
        .inactive_bytes = 20 * GIB,
        .file_backed_bytes = 20 * GIB,
        .pressure_status_available = true,
        .pressure_normal = true,
    };
    assert(ds4_ssd_resident_pressure_plan_make(
        &pressure_memory, 1 * GIB, 0, &pressure));
    assert(pressure.inactive_credit_bytes == 20 * GIB);
    assert(pressure.reclaimable_bytes == 24 * GIB);
    pressure_memory = (ds4_ssd_host_memory){
        .physical_bytes = 16 * GIB,
        .free_bytes = 16 * GIB,
        .inactive_bytes = 16 * GIB,
        .file_backed_bytes = 16 * GIB,
        .pressure_status_available = true,
        .pressure_normal = true,
    };
    assert(ds4_ssd_resident_pressure_plan_make(
        &pressure_memory, 19 * GIB, 0, &pressure));
    assert(pressure.reclaimable_bytes == 16 * GIB);
    assert(!pressure.fits);

    /* Live pressure is an independent residency gate: equality is safe, but
     * one byte less reclaimable memory fails closed even on a 64 GiB host. */
    pressure_memory = (ds4_ssd_host_memory){
        .physical_bytes = 64 * GIB,
        .free_bytes = 26 * GIB,
    };
    assert(ds4_ssd_resident_pressure_plan_make(
        &pressure_memory, 20 * GIB, 1 * GIB, &pressure));
    assert(pressure.fits);
    pressure_memory.free_bytes--;
    assert(ds4_ssd_resident_pressure_plan_make(
        &pressure_memory, 20 * GIB, 1 * GIB, &pressure));
    assert(!pressure.fits);
    assert(!ds4_ssd_resident_pressure_plan_make(
        NULL, 1, 0, &pressure));
    assert(!ds4_ssd_resident_pressure_plan_make(
        &pressure_memory, 0, 0, &pressure));
    assert(!ds4_ssd_resident_pressure_plan_make(
        &pressure_memory, UINT64_MAX, 1, &pressure));
    assert(!ds4_ssd_resident_pressure_plan_make(
        &pressure_memory, 1, 0, NULL));

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

    /* Qwen's supported Q4_K artifact has three equally sized expert slabs.
     * Each block stores 256 values in 144 bytes: gate/up are 2048x512 and
     * down is 512x2048, so every selected expert occupies 3 x 589824 bytes.
     * One complete token route spans 40 layers x top-8; the extra cache slot
     * prevents the first load of the next token from evicting a still-live
     * expert in the current route. */
    const uint64_t qwen_q4_k_block_bytes = 144u;
    const uint64_t qwen_gate_row_bytes =
        (QWEN35_N_EMBD / 256u) * qwen_q4_k_block_bytes;
    const uint64_t qwen_down_row_bytes =
        (QWEN35_N_FF_EXP / 256u) * qwen_q4_k_block_bytes;
    const uint64_t qwen_gate_expert_bytes =
        qwen_gate_row_bytes * QWEN35_N_FF_EXP;
    const uint64_t qwen_down_expert_bytes =
        qwen_down_row_bytes * QWEN35_N_EMBD;
    const uint64_t qwen_expert_bytes =
        2u * qwen_gate_expert_bytes + qwen_down_expert_bytes;
    const uint64_t qwen_max_cacheable =
        (uint64_t)QWEN35_N_LAYER * QWEN35_N_EXPERT;
    assert(qwen_gate_row_bytes == UINT64_C(1152));
    assert(qwen_down_row_bytes == UINT64_C(288));
    assert(qwen_gate_expert_bytes == UINT64_C(589824));
    assert(qwen_down_expert_bytes == UINT64_C(589824));
    assert(qwen_expert_bytes == UINT64_C(1769472));
    assert(qwen_max_cacheable == UINT64_C(10240));
    assert(ds4_ssd_expert_cache_floor_make(QWEN35_N_LAYER,
                                            QWEN35_N_EXPERT_USED,
                                            qwen_expert_bytes,
                                            &floor));
    assert(floor.working_set_experts == 320);
    assert(floor.minimum_cache_experts == 321);
    assert(floor.minimum_cache_bytes == UINT64_C(568000512));
    assert(floor.warning_cache_experts == 640);

    /* Qwen keeps the complete static mapping charged on 16 GiB, but those
     * unpinned pages share ordinary headroom because macOS can reclaim and
     * stream them again. Unlike DeepSeek's measured low-RAM performance cap,
     * Qwen consumes the largest complete tier admitted by its safety budget. */
    ds4_ssd_host_memory qwen_memory = {
        .physical_bytes = 16 * GIB,
        .recommended_bytes = 12 * GIB,
        .free_bytes = 15 * GIB,
        .pressure_status_available = true,
        .pressure_normal = true,
    };
    ds4_ssd_adaptive_cache_plan qwen_adaptive = {0};
    assert(ds4_ssd_adaptive_cache_plan_make_strict_with_static_reserve(
        &qwen_memory,
        512 * MIB,
        5 * GIB / 2u,
        false,
        QWEN35_N_LAYER,
        QWEN35_N_EXPERT_USED,
        qwen_expert_bytes,
        qwen_max_cacheable,
        &qwen_adaptive));
    assert(qwen_adaptive.low_ram_shared_static_headroom_active);
    assert(!qwen_adaptive.low_ram_floor_ceiling_active);
    assert(qwen_adaptive.pageable_static_reserve_bytes == 5 * GIB / 2u);
    assert(qwen_adaptive.platform_static_reserve_bytes == 5 * GIB / 2u);
    /* The 1.75 GiB field plus the separately recorded 0.25 GiB pressure
     * margin is the policy's single 2 GiB request reserve.  Pageable static
     * pages may occupy that reserve, so they are not charged a second time. */
    assert(qwen_adaptive.current_headroom_bytes == 7 * GIB / 4u);
    assert(qwen_adaptive.pressure_margin_bytes == GIB / 4u);
    assert(qwen_adaptive.platform_headroom_bytes == 2 * GIB);
    assert(qwen_adaptive.platform_wire_budget_bytes == 19 * GIB / 2u);
    assert(qwen_adaptive.cache_envelope_bytes ==
           qwen_adaptive.safety_wire_budget_bytes);
    assert(qwen_adaptive.cache_experts == 5761);
    assert(qwen_adaptive.cache_bytes ==
           UINT64_C(5761) * qwen_expert_bytes);
    assert(qwen_adaptive.cache_bytes <=
           qwen_adaptive.safety_wire_budget_bytes);

    /* Physical M1 Pro 16 GiB snapshot captured after the original Qwen AUTO
     * launch was rejected despite green pressure. With normal pressure, full
     * bounded file-backed credit admits four complete working-set cycles plus
     * the safety slot. The same page counts must still fail closed when the
     * pressure signal is elevated or unavailable, even if the arithmetic
     * budget alone could hold the minimum tier. */
    const uint64_t darwin_page_bytes = UINT64_C(16384);
    qwen_memory = (ds4_ssd_host_memory){
        .physical_bytes = 16 * GIB,
        .recommended_bytes = 12 * GIB,
        .free_bytes = UINT64_C(136760) * darwin_page_bytes,
        .purgeable_bytes = UINT64_C(15366) * darwin_page_bytes,
        .inactive_bytes = UINT64_C(291866) * darwin_page_bytes,
        .file_backed_bytes = UINT64_C(172380) * darwin_page_bytes,
        .pressure_status_available = true,
        .pressure_normal = true,
    };
    ds4_ssd_adaptive_cache_plan qwen_m1_normal = {0};
    assert(ds4_ssd_adaptive_cache_plan_make_strict_with_static_reserve(
        &qwen_memory,
        3 * GIB / 8u,
        5 * GIB / 2u,
        false,
        QWEN35_N_LAYER,
        QWEN35_N_EXPERT_USED,
        qwen_expert_bytes,
        qwen_max_cacheable,
        &qwen_m1_normal));
    assert(qwen_m1_normal.reclaimable_bytes ==
           qwen_memory.free_bytes + qwen_memory.file_backed_bytes);
    assert(qwen_m1_normal.current_headroom_bytes == 7 * GIB / 4u);
    assert(qwen_m1_normal.pressure_margin_bytes == GIB / 4u);
    assert(qwen_m1_normal.current_wire_budget_bytes >=
           floor.minimum_cache_bytes);
    assert(qwen_m1_normal.low_ram_shared_static_headroom_active);
    assert(!qwen_m1_normal.low_ram_floor_ceiling_active);
    assert(qwen_m1_normal.cache_experts == 1281);
    assert(qwen_m1_normal.cache_bytes ==
           UINT64_C(1281) * qwen_expert_bytes);

    qwen_memory.pressure_normal = false;
    ds4_ssd_adaptive_cache_plan qwen_m1_elevated = {0};
    assert(!ds4_ssd_adaptive_cache_plan_make_strict_with_static_reserve(
        &qwen_memory,
        3 * GIB / 8u,
        5 * GIB / 2u,
        false,
        QWEN35_N_LAYER,
        QWEN35_N_EXPERT_USED,
        qwen_expert_bytes,
        qwen_max_cacheable,
        &qwen_m1_elevated));
    assert(qwen_m1_elevated.reclaimable_bytes ==
           qwen_memory.free_bytes + qwen_memory.purgeable_bytes +
               qwen_memory.file_backed_bytes / 2u);
    assert(qwen_m1_elevated.wire_budget_bytes >= floor.minimum_cache_bytes);
    assert(qwen_m1_elevated.cache_experts == 641);

    qwen_memory.pressure_status_available = false;
    qwen_memory.pressure_normal = true;
    ds4_ssd_adaptive_cache_plan qwen_m1_unknown = {0};
    assert(!ds4_ssd_adaptive_cache_plan_make_strict_with_static_reserve(
        &qwen_memory,
        3 * GIB / 8u,
        5 * GIB / 2u,
        false,
        QWEN35_N_LAYER,
        QWEN35_N_EXPERT_USED,
        qwen_expert_bytes,
        qwen_max_cacheable,
        &qwen_m1_unknown));
    assert(qwen_m1_unknown.reclaimable_bytes ==
           qwen_m1_elevated.reclaimable_bytes);
    assert(qwen_m1_unknown.wire_budget_bytes >= floor.minimum_cache_bytes);
    assert(qwen_m1_unknown.cache_experts == 641);

    qwen_memory = (ds4_ssd_host_memory){
        .physical_bytes = 64 * GIB,
        .recommended_bytes = 52 * GIB,
        .free_bytes = 60 * GIB,
    };
    assert(ds4_ssd_adaptive_cache_plan_make_strict_with_static_reserve(
        &qwen_memory,
        512 * MIB,
        5 * GIB / 2u,
        false,
        QWEN35_N_LAYER,
        QWEN35_N_EXPERT_USED,
        qwen_expert_bytes,
        qwen_max_cacheable,
        &qwen_adaptive));
    assert(!qwen_adaptive.low_ram_floor_ceiling_active);
    assert(!qwen_adaptive.low_ram_shared_static_headroom_active);
    assert(qwen_adaptive.current_headroom_bytes == 4 * GIB);
    assert(qwen_adaptive.platform_headroom_bytes == 8 * GIB);
    assert(qwen_adaptive.cache_experts == qwen_max_cacheable);

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
    assert(!ds4_ssd_low_ram_cache_policy(0));
    assert(ds4_ssd_low_ram_cache_policy(16 * GIB));
    assert(!ds4_ssd_low_ram_cache_policy(16 * GIB + 1u));
    assert(!ds4_ssd_static_pin_host_supported(0));
    assert(!ds4_ssd_static_pin_host_supported(16 * GIB));
    assert(!ds4_ssd_static_pin_host_supported(64 * GIB - 1u));
    assert(ds4_ssd_static_pin_host_supported(64 * GIB));
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
    assert(adaptive.pageable_static_reserve_bytes == 0);

    /* Static pinning is never part of the low-RAM policy.  Even if a caller
     * supplies Flash's full always-used static set, the 16 GiB tier leaves it
     * pageable and preserves only the measured 259-expert correctness floor. */
    memory = (ds4_ssd_host_memory){
        .physical_bytes = 16 * GIB,
        .recommended_bytes = 12 * GIB,
        .free_bytes = 11 * GIB / 2u,
    };
    assert(ds4_ssd_adaptive_cache_plan_make_with_static_reserve(
        &memory, 512 * MIB, 8 * GIB, false, 43, 6,
        flash_expert_bytes, flash_max_cacheable, &adaptive));
    assert(adaptive.low_ram_floor_ceiling_active);
    assert(adaptive.pageable_static_reserve_bytes == 0);
    assert(adaptive.current_headroom_bytes == 2 * GIB);
    assert(adaptive.cache_experts == 259);

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

    /* Above 16 GiB, AUTO grows with the point-in-time reclaimable budget but
     * first protects the unpinned static working set.  max(static, baseline)
     * is intentional: adding both reserves would underfill the cache. */
    memory = (ds4_ssd_host_memory){
        .physical_bytes = 24 * GIB,
        .recommended_bytes = 18 * GIB,
        .free_bytes = 18 * GIB,
    };
    assert(ds4_ssd_adaptive_cache_plan_make_with_static_reserve(
        &memory, 512 * MIB, 8 * GIB, false, 43, 6,
        flash_expert_bytes, flash_max_cacheable, &adaptive));
    assert(!adaptive.low_ram_floor_ceiling_active);
    assert(adaptive.pageable_static_reserve_bytes == 8 * GIB);
    assert(adaptive.current_headroom_bytes == 8 * GIB);
    assert(adaptive.platform_headroom_bytes == 8 * GIB);
    assert(adaptive.cache_experts == 1291);

    memory = (ds4_ssd_host_memory){
        .physical_bytes = 32 * GIB,
        .recommended_bytes = 24 * GIB,
        .free_bytes = 24 * GIB,
    };
    assert(ds4_ssd_adaptive_cache_plan_make_with_static_reserve(
        &memory, 512 * MIB, 8 * GIB, false, 43, 6,
        flash_expert_bytes, flash_max_cacheable, &adaptive));
    assert(adaptive.pageable_static_reserve_bytes == 8 * GIB);
    assert(adaptive.current_headroom_bytes == 8 * GIB);
    assert(adaptive.platform_headroom_bytes == 8 * GIB);
    assert(adaptive.cache_envelope_bytes == 27 * GIB / 2u);
    assert(adaptive.cache_experts == 1807);

    memory.recommended_bytes = 32 * GIB;
    assert(ds4_ssd_adaptive_cache_plan_make_with_static_reserve(
        &memory, 512 * MIB, 8 * GIB, false, 43, 6,
        flash_expert_bytes, flash_max_cacheable, &adaptive));
    assert(adaptive.cache_envelope_bytes == 18 * GIB);
    assert(adaptive.cache_experts == 2065);

    /* A smaller static set is covered by the ordinary host headroom; it is
     * not added a second time. */
    memory = (ds4_ssd_host_memory){
        .physical_bytes = 64 * GIB,
        .recommended_bytes = 52 * GIB,
        .free_bytes = 40 * GIB,
    };
    assert(ds4_ssd_adaptive_cache_plan_make_with_static_reserve(
        &memory, 0, 2 * GIB, false, 43, 6,
        flash_expert_bytes, flash_max_cacheable, &adaptive));
    assert(adaptive.pageable_static_reserve_bytes == 2 * GIB);
    assert(adaptive.current_headroom_bytes == 4 * GIB);
    assert(adaptive.platform_headroom_bytes == 8 * GIB);

    /* Reproduce the bounded M5 canary snapshot.  The safety budget alone can
     * fit the old 4903-expert tier, while the stable envelope and the strict
     * static reserve independently hold AUTO at the nearby measured sweet
     * spot, 4387 experts / 28.92 GiB. */
    memory = (ds4_ssd_host_memory){
        .physical_bytes = 64 * GIB,
        .recommended_bytes = 51 * GIB + 84 * GIB / 100u,
        .free_bytes = 39 * GIB + 76 * GIB / 100u,
    };
    assert(ds4_ssd_adaptive_cache_plan_make(&memory,
                                             86 * GIB / 100u,
                                             43, 6,
                                             flash_expert_bytes,
                                             flash_max_cacheable,
                                             &adaptive));
    assert(adaptive.safety_wire_budget_bytes /
           flash_expert_bytes > 4903);
    assert(adaptive.cache_experts == 4387);
    assert(ds4_ssd_adaptive_cache_plan_make_with_static_reserve(
        &memory, 86 * GIB / 100u, 8 * GIB + GIB / 5u, false, 43, 6,
        flash_expert_bytes, flash_max_cacheable, &adaptive));
    assert(adaptive.pageable_static_reserve_bytes ==
           8 * GIB + GIB / 5u);
    assert(adaptive.cache_experts == 4387);

    /* A warmer launch exposes more file-backed pages as reclaimable.  The
     * safety budget can now fit 4645, but the envelope prevents startup-order
     * feedback from growing the wired cache. */
    memory.free_bytes = 41 * GIB + 27 * GIB / 100u;
    assert(ds4_ssd_adaptive_cache_plan_make_with_static_reserve(
        &memory, 86 * GIB / 100u, 8 * GIB + GIB / 5u, false, 43, 6,
        flash_expert_bytes, flash_max_cacheable, &adaptive));
    assert(adaptive.safety_wire_budget_bytes /
           flash_expert_bytes > 4645);
    assert(adaptive.cache_experts == 4387);

    /* The envelope is a ceiling, not a fixed allocation: genuine pressure
     * still shrinks AUTO by complete working-set tiers. */
    memory.free_bytes = 38 * GIB;
    assert(ds4_ssd_adaptive_cache_plan_make_with_static_reserve(
        &memory, 86 * GIB / 100u, 8 * GIB + GIB / 5u, false, 43, 6,
        flash_expert_bytes, flash_max_cacheable, &adaptive));
    assert(adaptive.safety_wire_budget_bytes <
           adaptive.cache_envelope_bytes);
    assert(adaptive.cache_experts == 4129);

    /* Once the static set is pinned, the live snapshot reflects it in the
     * current-pressure constraint.  The fixed platform working-set limit must
     * still retain the static charge. */
    memory = (ds4_ssd_host_memory){
        .physical_bytes = 64 * GIB,
        .recommended_bytes = 32 * GIB,
        .free_bytes = 40 * GIB,
    };
    assert(ds4_ssd_adaptive_cache_plan_make_with_static_reserve(
        &memory, 1 * GIB, 10 * GIB, true, 43, 6,
        flash_expert_bytes, flash_max_cacheable, &adaptive));
    assert(adaptive.pageable_static_reserve_bytes == 0);
    assert(adaptive.platform_static_reserve_bytes == 10 * GIB);
    assert(adaptive.current_headroom_bytes == 0);
    assert(adaptive.platform_headroom_bytes == 18 * GIB);
    assert(adaptive.platform_wire_budget_bytes == 13 * GIB);
    assert(adaptive.safety_wire_budget_bytes == 13 * GIB);

    /* If the post-pin snapshot falls by exactly the pinned set, current
     * pressure must yield the same budget as the equivalent pre-pin plan. */
    memory = (ds4_ssd_host_memory){
        .physical_bytes = 64 * GIB,
        .recommended_bytes = 52 * GIB,
        .free_bytes = 40 * GIB,
    };
    assert(ds4_ssd_adaptive_cache_plan_make_with_static_reserve(
        &memory, 1 * GIB, 8 * GIB, false, 43, 6,
        flash_expert_bytes, flash_max_cacheable, &adaptive));
    const uint64_t pre_pin_current_budget =
        adaptive.current_wire_budget_bytes;
    assert(adaptive.current_headroom_bytes == 8 * GIB);
    memory.free_bytes -= 8 * GIB;
    assert(ds4_ssd_adaptive_cache_plan_make_with_static_reserve(
        &memory, 1 * GIB, 8 * GIB, true, 43, 6,
        flash_expert_bytes, flash_max_cacheable, &adaptive));
    assert(adaptive.current_headroom_bytes == 0);
    assert(adaptive.current_wire_budget_bytes == pre_pin_current_budget);

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
