#include "ds4.h"
#include "ds4_ssd.h"
#include "ds4_qwen.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

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
    assert(DS4_BACKEND_METAL == 0);
    assert(DS4_BACKEND_CPU == 2);
    assert(strcmp(ds4_residency_reason_name(
                      DS4_RESIDENCY_REASON_MODEL_REQUIRES_SSD),
                  "the model family is qualified only for SSD streaming") == 0);
    assert(strcmp(ds4_residency_reason_name(
                      DS4_RESIDENCY_REASON_HARDWARE_REQUIRES_SSD),
                  "this hardware tier is qualified only for SSD streaming") == 0);

    /* GLM ExpertMajor v2 applies this policy after the generic planner has
     * populated its memory accounting. AUTO is deterministic even when a
     * hypothetical resident plan fits a very large Metal budget. */
    ds4_residency_plan glm = plan(true,
                                  DS4_RESIDENCY_AUTO,
                                  1 * GIB,
                                  1 * GIB,
                                  128 * GIB,
                                  0);
    assert(glm.resolved == DS4_RESIDENCY_RESIDENT);
    const uint64_t glm_required = glm.required_bytes;
    assert(ds4_residency_plan_apply_ssd_only(DS4_RESIDENCY_AUTO, &glm));
    assert(glm.requested == DS4_RESIDENCY_AUTO);
    assert(glm.resolved == DS4_RESIDENCY_SSD);
    assert(glm.reason == DS4_RESIDENCY_REASON_MODEL_REQUIRES_SSD);
    assert(glm.required_bytes == glm_required);

    memset(&glm, 0, sizeof(glm));
    assert(!ds4_residency_plan_apply_ssd_only(
        DS4_RESIDENCY_RESIDENT, &glm));
    assert(glm.requested == DS4_RESIDENCY_RESIDENT);
    assert(glm.resolved == DS4_RESIDENCY_SSD);
    assert(glm.reason == DS4_RESIDENCY_REASON_MODEL_REQUIRES_SSD);

    memset(&glm, 0, sizeof(glm));
    assert(ds4_residency_plan_apply_ssd_only(DS4_RESIDENCY_SSD, &glm));
    assert(glm.requested == DS4_RESIDENCY_SSD);
    assert(glm.resolved == DS4_RESIDENCY_SSD);
    assert(glm.reason == DS4_RESIDENCY_REASON_EXPLICIT_SSD);
    assert(!ds4_residency_plan_apply_ssd_only(
        (ds4_residency_mode)99, &glm));
    assert(!ds4_residency_plan_apply_ssd_only(
        DS4_RESIDENCY_AUTO, NULL));

    ds4_streaming_hotlist_priority_policy hotlist_priority;
    assert(ds4_parse_streaming_hotlist_priority_policy(NULL,
                                                        &hotlist_priority));
    assert(hotlist_priority.mode ==
           DS4_STREAMING_HOTLIST_PRIORITY_ADAPTIVE);
    assert(hotlist_priority.priority == 1);
    assert(ds4_parse_streaming_hotlist_priority_policy("",
                                                        &hotlist_priority));
    assert(hotlist_priority.mode ==
           DS4_STREAMING_HOTLIST_PRIORITY_ADAPTIVE);
    assert(hotlist_priority.priority == 1);
    assert(ds4_parse_streaming_hotlist_priority_policy("adaptive",
                                                        &hotlist_priority));
    assert(hotlist_priority.mode ==
           DS4_STREAMING_HOTLIST_PRIORITY_ADAPTIVE);
    assert(hotlist_priority.priority == 1);
    assert(ds4_parse_streaming_hotlist_priority_policy("legacy",
                                                        &hotlist_priority));
    assert(hotlist_priority.mode ==
           DS4_STREAMING_HOTLIST_PRIORITY_LEGACY);
    assert(hotlist_priority.priority == 0);
    assert(ds4_parse_streaming_hotlist_priority_policy("17",
                                                        &hotlist_priority));
    assert(hotlist_priority.mode == DS4_STREAMING_HOTLIST_PRIORITY_FIXED);
    assert(hotlist_priority.priority == 17);
    assert(ds4_parse_streaming_hotlist_priority_policy("4294967295",
                                                        &hotlist_priority));
    assert(hotlist_priority.mode == DS4_STREAMING_HOTLIST_PRIORITY_FIXED);
    assert(hotlist_priority.priority == UINT32_MAX);
    assert(!ds4_parse_streaming_hotlist_priority_policy("0",
                                                         &hotlist_priority));
    assert(!ds4_parse_streaming_hotlist_priority_policy("-1",
                                                         &hotlist_priority));
    assert(!ds4_parse_streaming_hotlist_priority_policy("1x",
                                                         &hotlist_priority));
    assert(!ds4_parse_streaming_hotlist_priority_policy("4294967296",
                                                         &hotlist_priority));
    assert(!ds4_parse_streaming_hotlist_priority_policy("LEGACY",
                                                         &hotlist_priority));
    assert(!ds4_parse_streaming_hotlist_priority_policy("adaptive", NULL));

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

    /* DeepSeek/Qwen keep using the generic planner. At 10 GiB the fixed
     * minimum and 20% headroom are both 2 GiB; model + runtime + headroom
     * exactly equals the budget and remains resident. */
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

    /* Qwen's Metal reserve follows physical memory continuously, while every
     * Apple unified-memory cut has an observable profile label.  The 24, 36,
     * and 48 GiB cuts are first-class policies rather than aliases of 32/64. */
    static const uint32_t qwen_profile_gib[] = {
        16u, 24u, 32u, 36u, 48u, 64u, 96u, 128u,
    };
    /* Recommended budgets here are monotonic synthetic fixtures. Production
     * always consumes the value reported by the active Metal device. */
    static const uint64_t qwen_recommended_gib[] = {
        12u, 18u, 25u, 28u, 38u, 52u, 78u, 104u,
    };
    static const uint64_t qwen_reserve_sixteenths[] = {
        36u, 38u, 40u, 45u, 60u, 80u, 120u, 160u,
    };
    ds4_qwen_metal_hardware_policy qwen_hw = {0};
    uint64_t previous_reserve = 0;
    for (size_t i = 0;
         i < sizeof(qwen_profile_gib) / sizeof(qwen_profile_gib[0]); i++) {
        assert(ds4_qwen_metal_hardware_policy_make(
            (uint64_t)qwen_profile_gib[i] * GIB,
            qwen_recommended_gib[i] * GIB,
            &qwen_hw));
        assert(qwen_hw.profile_gib == qwen_profile_gib[i]);
        const uint64_t expected_reserve =
            qwen_reserve_sixteenths[i] * GIB / 16u;
        assert(qwen_hw.resident_headroom_bytes == expected_reserve);
        assert(qwen_hw.resident_headroom_bytes > previous_reserve);
        previous_reserve = qwen_hw.resident_headroom_bytes;
    }
    assert(ds4_qwen_metal_hardware_policy_make(
        20 * GIB, 15 * GIB, &qwen_hw));
    assert(qwen_hw.profile_gib == 24u);
    assert(!ds4_qwen_metal_hardware_policy_make(0, 1, &qwen_hw));
    assert(!ds4_qwen_metal_hardware_policy_make(1, 0, &qwen_hw));
    assert(!ds4_qwen_metal_hardware_policy_make(1, 1, NULL));

    /* A conservative synthetic 24 GiB working-set fixture rejects the current
     * Qwen-sized payload. A 32 GiB M1-class fixture admits the shorter runtime
     * but rejects the next context tier using the same 2.5 GiB reserve as the
     * live-pressure gate. Production uses each device's reported limit. */
    assert(ds4_qwen_metal_hardware_policy_make(
        24 * GIB, 18 * GIB, &qwen_hw));
    ds4_residency_plan qwen_residency = plan(
        true, DS4_RESIDENCY_AUTO, 20 * GIB, 0, 18 * GIB, 0);
    assert(ds4_residency_plan_apply_qwen_metal_hardware_policy(
        &qwen_hw, &qwen_residency));
    assert(qwen_residency.resolved == DS4_RESIDENCY_SSD);
    assert(ds4_qwen_metal_hardware_policy_make(
        32 * GIB, 25 * GIB, &qwen_hw));
    qwen_residency = plan(
        true, DS4_RESIDENCY_AUTO, 20 * GIB, 2 * GIB, 25 * GIB, 0);
    assert(ds4_residency_plan_apply_qwen_metal_hardware_policy(
        &qwen_hw, &qwen_residency));
    assert(qwen_residency.headroom_bytes == 5 * GIB / 2u);
    assert(qwen_residency.required_bytes == 49 * GIB / 2u);
    assert(qwen_residency.resolved == DS4_RESIDENCY_RESIDENT);
    qwen_residency.runtime_bytes = 3 * GIB;
    assert(ds4_residency_plan_apply_qwen_metal_hardware_policy(
        &qwen_hw, &qwen_residency));
    assert(qwen_residency.required_bytes == 51 * GIB / 2u);
    assert(qwen_residency.resolved == DS4_RESIDENCY_SSD);
    assert(!ds4_residency_plan_apply_qwen_metal_hardware_policy(
        NULL, &qwen_residency));
    assert(!ds4_residency_plan_apply_qwen_metal_hardware_policy(
        &qwen_hw, NULL));

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

    /* The selected Q2_K_XL artifact has three routed layer classes. Size the
     * single SSD slab for the largest IQ3_XXS/IQ4_XS expert so every layer is
     * cacheable: gate/up are 2048x512 with 98-byte blocks and down is
     * 512x2048 with 136-byte blocks. One complete token route spans 40 layers
     * x top-8; the extra cache slot prevents the first load of the next token
     * from evicting a still-live expert in the current route. */
    const uint64_t qwen_iq3_xxs_block_bytes = 98u;
    const uint64_t qwen_iq4_xs_block_bytes = 136u;
    const uint64_t qwen_gate_row_bytes =
        (QWEN35_N_EMBD / 256u) * qwen_iq3_xxs_block_bytes;
    const uint64_t qwen_down_row_bytes =
        (QWEN35_N_FF_EXP / 256u) * qwen_iq4_xs_block_bytes;
    const uint64_t qwen_gate_expert_bytes =
        qwen_gate_row_bytes * QWEN35_N_FF_EXP;
    const uint64_t qwen_down_expert_bytes =
        qwen_down_row_bytes * QWEN35_N_EMBD;
    const uint64_t qwen_expert_bytes =
        2u * qwen_gate_expert_bytes + qwen_down_expert_bytes;
    const uint64_t qwen_max_cacheable =
        (uint64_t)QWEN35_N_LAYER * QWEN35_N_EXPERT;
    assert(qwen_gate_row_bytes == UINT64_C(784));
    assert(qwen_down_row_bytes == UINT64_C(272));
    assert(qwen_gate_expert_bytes == UINT64_C(401408));
    assert(qwen_down_expert_bytes == UINT64_C(557056));
    assert(qwen_expert_bytes == UINT64_C(1359872));
    assert(qwen_max_cacheable == UINT64_C(10240));
    assert(ds4_ssd_expert_cache_floor_make(QWEN35_N_LAYER,
                                            QWEN35_N_EXPERT_USED,
                                            qwen_expert_bytes,
                                            &floor));
    assert(floor.working_set_experts == 320);
    assert(floor.minimum_cache_experts == 321);
    assert(floor.minimum_cache_bytes == UINT64_C(436518912));
    assert(floor.warning_cache_experts == 640);

    /* GLM ExpertMajor keeps one complete route plus its in-flight safety slot
     * on the measured 64 GiB Metal tier. The 96+ GiB policy remains the
     * ordinary adaptive candidate until it has its own hardware evidence. */
    ds4_ssd_adaptive_cache_plan glm_plan = {
        .floor = {
            .minimum_cache_experts = 601,
        },
        .cache_experts = 1801,
    };
    ds4_ssd_host_memory glm_memory = {
        .physical_bytes = 64 * GIB,
    };
    assert(ds4_ssd_glm_expert_major_auto_cache_target(
               &glm_memory, &glm_plan) == 601);
    glm_memory.physical_bytes = 95 * GIB;
    assert(ds4_ssd_glm_expert_major_auto_cache_target(
               &glm_memory, &glm_plan) == 601);
    glm_memory.physical_bytes = 96 * GIB;
    assert(ds4_ssd_glm_expert_major_auto_cache_target(
               &glm_memory, &glm_plan) == 1801);
    glm_memory.physical_bytes = 128 * GIB;
    assert(ds4_ssd_glm_expert_major_auto_cache_target(
               &glm_memory, &glm_plan) == 1801);
    assert(ds4_ssd_glm_expert_major_auto_cache_target(NULL, &glm_plan) == 0);
    assert(ds4_ssd_glm_expert_major_auto_cache_target(
               &glm_memory, NULL) == 0);

    /* Under normal pressure, equivalent cold and warm GGUF page states must
     * yield the same Qwen cache on every named memory profile.  Capacity is
     * monotonic with the hardware budget and never falls at a tier boundary. */
    uint32_t previous_cache_experts = 0;
    for (size_t i = 0;
         i < sizeof(qwen_profile_gib) / sizeof(qwen_profile_gib[0]); i++) {
        const uint64_t physical = (uint64_t)qwen_profile_gib[i] * GIB;
        ds4_ssd_host_memory cold = {
            .physical_bytes = physical,
            .recommended_bytes = qwen_recommended_gib[i] * GIB,
            .free_bytes = physical * 3u / 4u,
            .pressure_status_available = true,
            .pressure_normal = true,
        };
        ds4_ssd_host_memory warm = {
            .physical_bytes = physical,
            .recommended_bytes = qwen_recommended_gib[i] * GIB,
            .free_bytes = physical / 8u,
            .inactive_bytes = physical * 5u / 8u,
            .file_backed_bytes = physical * 5u / 8u,
            .pressure_status_available = true,
            .pressure_normal = true,
        };
        ds4_ssd_adaptive_cache_plan cold_plan = {0};
        ds4_ssd_adaptive_cache_plan warm_plan = {0};
        assert(ds4_ssd_adaptive_cache_plan_make_strict_with_static_reserve(
            &cold, 512 * MIB, 5 * GIB / 2u, false,
            QWEN35_N_LAYER, QWEN35_N_EXPERT_USED,
            qwen_expert_bytes, qwen_max_cacheable, &cold_plan));
        assert(ds4_ssd_adaptive_cache_plan_make_strict_with_static_reserve(
            &warm, 512 * MIB, 5 * GIB / 2u, false,
            QWEN35_N_LAYER, QWEN35_N_EXPERT_USED,
            qwen_expert_bytes, qwen_max_cacheable, &warm_plan));
        assert(cold_plan.normal_pressure_full_file_credit_active);
        assert(warm_plan.normal_pressure_full_file_credit_active);
        assert(cold_plan.reclaimable_bytes == warm_plan.reclaimable_bytes);
        assert(cold_plan.cache_experts == warm_plan.cache_experts);
        if (qwen_profile_gib[i] <= 24u) {
            assert(ds4_ssd_qwen_guarded_cache_policy(physical));
            assert(cold_plan.cache_experts == 3521u);
        } else {
            assert(!ds4_ssd_qwen_guarded_cache_policy(physical));
        }
        assert(cold_plan.cache_experts >= previous_cache_experts);
        previous_cache_experts = cold_plan.cache_experts;
    }

    /* Qwen keeps the complete static mapping charged on 16 GiB, but those
     * unpinned pages share ordinary headroom because macOS can reclaim and
     * stream them again. The guarded 16/24 GiB profiles then apply their
     * independent 3,521-expert ceiling after the safety budget is computed. */
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
    /* The 2.25 GiB field plus the separately recorded 0.25 GiB pressure
     * margin is the 2.5 GiB static/request reserve. Pageable static pages are
     * not charged a second time. */
    assert(qwen_adaptive.current_headroom_bytes == 9 * GIB / 4u);
    assert(qwen_adaptive.pressure_margin_bytes == GIB / 4u);
    assert(qwen_adaptive.platform_headroom_bytes == 5 * GIB / 2u);
    assert(qwen_adaptive.platform_wire_budget_bytes == 9 * GIB);
    assert(qwen_adaptive.cache_envelope_bytes ==
           qwen_adaptive.safety_wire_budget_bytes);
    assert(qwen_adaptive.cache_experts == 3521);
    assert(qwen_adaptive.cache_bytes ==
           UINT64_C(3521) * qwen_expert_bytes);
    assert(qwen_adaptive.cache_bytes <=
           qwen_adaptive.safety_wire_budget_bytes);

    /* Physical M1 Pro 16 GiB snapshot captured after the original Qwen AUTO
     * launch was rejected despite green pressure. With normal pressure, full
     * bounded file-backed credit admits four complete Q2_K_XL working-set
     * cycles plus the safety slot. The same page counts must still fail closed
     * when the pressure signal is elevated or unavailable, even if the
     * arithmetic budget alone could hold the minimum tier. */
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
    assert(qwen_m1_normal.current_headroom_bytes == 9 * GIB / 4u);
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
    assert(qwen_m1_elevated.cache_experts == 321);

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
    assert(qwen_m1_unknown.cache_experts == 321);

    /* A 24 GiB Qwen host previously consumed every complete route admitted by
     * its synthetic 18 GiB Metal budget (14.24 GiB in this fixture).  The
     * guarded profile reuses the proven 5.80 GiB ceiling and fails closed when
     * pressure is elevated or unavailable, before phase cache growth. */
    qwen_memory = (ds4_ssd_host_memory){
        .physical_bytes = 24 * GIB,
        .recommended_bytes = 18 * GIB,
        .free_bytes = 20 * GIB,
        .pressure_status_available = true,
        .pressure_normal = true,
    };
    ds4_ssd_adaptive_cache_plan qwen_24g = {0};
    assert(ds4_ssd_adaptive_cache_plan_make_strict_with_static_reserve(
        &qwen_memory,
        512 * MIB,
        5 * GIB / 2u,
        false,
        QWEN35_N_LAYER,
        QWEN35_N_EXPERT_USED,
        qwen_expert_bytes,
        qwen_max_cacheable,
        &qwen_24g));
    assert(qwen_24g.cache_experts == 3521);
    assert(qwen_24g.cache_bytes == UINT64_C(3521) * qwen_expert_bytes);
    assert(qwen_24g.cache_bytes < 6 * GIB);
    assert(!qwen_24g.low_ram_shared_static_headroom_active);
    assert(!qwen_24g.low_ram_floor_ceiling_active);

    qwen_memory.pressure_normal = false;
    assert(!ds4_ssd_adaptive_cache_plan_make_strict_with_static_reserve(
        &qwen_memory,
        512 * MIB,
        5 * GIB / 2u,
        false,
        QWEN35_N_LAYER,
        QWEN35_N_EXPERT_USED,
        qwen_expert_bytes,
        qwen_max_cacheable,
        &qwen_24g));
    qwen_memory.pressure_status_available = false;
    qwen_memory.pressure_normal = true;
    assert(!ds4_ssd_adaptive_cache_plan_make_strict_with_static_reserve(
        &qwen_memory,
        512 * MIB,
        5 * GIB / 2u,
        false,
        QWEN35_N_LAYER,
        QWEN35_N_EXPERT_USED,
        qwen_expert_bytes,
        qwen_max_cacheable,
        &qwen_24g));

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
    assert(!ds4_ssd_qwen_guarded_cache_policy(0));
    assert(ds4_ssd_qwen_guarded_cache_policy(16 * GIB));
    assert(ds4_ssd_qwen_guarded_cache_policy(24 * GIB));
    assert(!ds4_ssd_qwen_guarded_cache_policy(24 * GIB + 1u));
    ds4_residency_plan qwen_guarded_residency = {
        .requested = DS4_RESIDENCY_AUTO,
        .resolved = DS4_RESIDENCY_RESIDENT,
        .reason = DS4_RESIDENCY_REASON_METAL_FITS,
    };
    assert(ds4_residency_plan_apply_qwen_guarded_ssd_only(
        21 * GIB, DS4_RESIDENCY_AUTO, &qwen_guarded_residency));
    assert(qwen_guarded_residency.requested == DS4_RESIDENCY_AUTO);
    assert(qwen_guarded_residency.resolved == DS4_RESIDENCY_SSD);
    assert(qwen_guarded_residency.reason ==
           DS4_RESIDENCY_REASON_HARDWARE_REQUIRES_SSD);
    assert(!ds4_residency_plan_apply_qwen_guarded_ssd_only(
        24 * GIB, DS4_RESIDENCY_RESIDENT, &qwen_guarded_residency));
    qwen_guarded_residency.resolved = DS4_RESIDENCY_RESIDENT;
    qwen_guarded_residency.reason = DS4_RESIDENCY_REASON_METAL_FITS;
    assert(ds4_residency_plan_apply_qwen_guarded_ssd_only(
        24 * GIB + 1u, DS4_RESIDENCY_AUTO, &qwen_guarded_residency));
    assert(qwen_guarded_residency.resolved == DS4_RESIDENCY_RESIDENT);
    assert(qwen_guarded_residency.reason ==
           DS4_RESIDENCY_REASON_METAL_FITS);
    assert(!ds4_residency_plan_apply_qwen_guarded_ssd_only(
        0, DS4_RESIDENCY_AUTO, &qwen_guarded_residency));
    assert(!ds4_residency_plan_apply_qwen_guarded_ssd_only(
        21 * GIB, (ds4_residency_mode)(DS4_RESIDENCY_SSD + 1),
        &qwen_guarded_residency));
    assert(!ds4_residency_plan_apply_qwen_guarded_ssd_only(
        21 * GIB, DS4_RESIDENCY_AUTO, NULL));
    const bool qwen_16g_guarded =
        ds4_ssd_qwen_guarded_cache_policy(16 * GIB);
    const bool qwen_32g_guarded =
        ds4_ssd_qwen_guarded_cache_policy(32 * GIB);
    for (int changed = 0; changed <= 1; changed++) {
        assert(ds4_ssd_qwen_phase_pressure_allowed(
            qwen_16g_guarded, changed != 0, true, true, true));
        assert(!ds4_ssd_qwen_phase_pressure_allowed(
            qwen_16g_guarded, changed != 0, true, true, false));
        assert(!ds4_ssd_qwen_phase_pressure_allowed(
            qwen_16g_guarded, changed != 0, true, false, true));
        assert(!ds4_ssd_qwen_phase_pressure_allowed(
            qwen_16g_guarded, changed != 0, false, true, true));
        assert(ds4_ssd_qwen_phase_pressure_allowed(
            qwen_32g_guarded, changed != 0, false, false, false));
    }

    /* A synthetic 21 GiB host exercises the 24 GiB containing profile without
     * pretending that process limits reproduce physical unified memory. The
     * incremental guard charges one proposed 321-expert slab against live host
     * headroom and reconstructs the complete fixed Metal envelope. */
    const uint64_t qwen_slab_bytes =
        UINT64_C(321) * qwen_expert_bytes;
    ds4_ssd_host_memory qwen_21g_memory = {
        .physical_bytes = 21 * GIB,
        .recommended_bytes = 16 * GIB,
        .free_bytes = 6 * GIB,
        .purgeable_bytes = 2 * GIB,
        .inactive_bytes = 3 * GIB,
        .file_backed_bytes = 3 * GIB,
        .pressure_status_available = true,
        .pressure_normal = true,
    };
    ds4_ssd_qwen_slab_growth_plan qwen_21g_growth = {0};
    assert(ds4_ssd_qwen_guarded_cache_policy(21 * GIB));
    assert(ds4_ssd_qwen_slab_growth_plan_make(
        &qwen_21g_memory, 2 * GIB, 5 * GIB / 2u, 5 * GIB,
        qwen_slab_bytes,
        &qwen_21g_growth));
    assert(qwen_21g_growth.pressure_normal);
    assert(qwen_21g_growth.host_fits);
    assert(qwen_21g_growth.platform_fits);
    assert(qwen_21g_growth.allowed);
    assert(qwen_21g_growth.slab_bytes == qwen_slab_bytes);

    /* Equality is admitted: the guard rejects only an actual overrun. */
    const uint64_t qwen_21g_equal_host =
        qwen_21g_growth.host_required_bytes;
    const uint64_t qwen_21g_equal_metal =
        qwen_21g_growth.platform_required_bytes;
    qwen_21g_memory.recommended_bytes = qwen_21g_equal_metal;
    qwen_21g_memory.free_bytes = qwen_21g_equal_host;
    qwen_21g_memory.purgeable_bytes = 0;
    qwen_21g_memory.inactive_bytes = 0;
    qwen_21g_memory.file_backed_bytes = 0;
    assert(ds4_ssd_qwen_slab_growth_plan_make(
        &qwen_21g_memory, 2 * GIB, 5 * GIB / 2u, 5 * GIB,
        qwen_slab_bytes,
        &qwen_21g_growth));
    assert(qwen_21g_growth.host_required_bytes ==
           qwen_21g_memory.free_bytes);
    assert(qwen_21g_growth.platform_required_bytes ==
           qwen_21g_memory.recommended_bytes);
    assert(qwen_21g_growth.allowed);

    qwen_21g_memory.recommended_bytes = 16 * GIB;
    qwen_21g_memory.free_bytes = 6 * GIB;
    qwen_21g_memory.purgeable_bytes = 2 * GIB;
    qwen_21g_memory.inactive_bytes = 3 * GIB;
    qwen_21g_memory.file_backed_bytes = 3 * GIB;
    qwen_21g_memory.pressure_normal = false;
    assert(ds4_ssd_qwen_slab_growth_plan_make(
        &qwen_21g_memory, 2 * GIB, 5 * GIB / 2u, 5 * GIB,
        qwen_slab_bytes,
        &qwen_21g_growth));
    assert(!qwen_21g_growth.allowed);
    qwen_21g_memory.pressure_status_available = false;
    qwen_21g_memory.pressure_normal = true;
    assert(ds4_ssd_qwen_slab_growth_plan_make(
        &qwen_21g_memory, 2 * GIB, 5 * GIB / 2u, 5 * GIB,
        qwen_slab_bytes,
        &qwen_21g_growth));
    assert(!qwen_21g_growth.allowed);

    qwen_21g_memory.pressure_status_available = true;
    qwen_21g_memory.free_bytes = 2 * GIB;
    qwen_21g_memory.purgeable_bytes = 0;
    qwen_21g_memory.inactive_bytes = 0;
    qwen_21g_memory.file_backed_bytes = 0;
    assert(ds4_ssd_qwen_slab_growth_plan_make(
        &qwen_21g_memory, 2 * GIB, 5 * GIB / 2u, 5 * GIB,
        qwen_slab_bytes,
        &qwen_21g_growth));
    assert(!qwen_21g_growth.host_fits);
    assert(!qwen_21g_growth.allowed);

    qwen_21g_memory.free_bytes = 6 * GIB;
    qwen_21g_memory.recommended_bytes = 10 * GIB;
    assert(ds4_ssd_qwen_slab_growth_plan_make(
        &qwen_21g_memory, 2 * GIB, 5 * GIB / 2u, 5 * GIB,
        qwen_slab_bytes,
        &qwen_21g_growth));
    assert(qwen_21g_growth.host_fits);
    assert(!qwen_21g_growth.platform_fits);
    assert(!qwen_21g_growth.allowed);

    qwen_21g_memory.recommended_bytes = UINT64_MAX;
    assert(ds4_ssd_qwen_slab_growth_plan_make(
        &qwen_21g_memory, UINT64_MAX, 5 * GIB / 2u, 5 * GIB,
        qwen_slab_bytes,
        &qwen_21g_growth));
    assert(qwen_21g_growth.platform_required_bytes == UINT64_MAX);
    assert(!qwen_21g_growth.platform_fits);
    assert(!qwen_21g_growth.allowed);
    assert(!ds4_ssd_qwen_slab_growth_plan_make(
        NULL, 2 * GIB, 5 * GIB / 2u, 5 * GIB, qwen_slab_bytes,
        &qwen_21g_growth));
    assert(!ds4_ssd_qwen_slab_growth_plan_make(
        &qwen_21g_memory, 0, 5 * GIB / 2u, 5 * GIB,
        qwen_slab_bytes, &qwen_21g_growth));
    assert(!ds4_ssd_qwen_slab_growth_plan_make(
        &qwen_21g_memory, 2 * GIB, 0, 5 * GIB,
        qwen_slab_bytes, &qwen_21g_growth));
    assert(!ds4_ssd_qwen_slab_growth_plan_make(
        &qwen_21g_memory, 2 * GIB, 5 * GIB / 2u, 5 * GIB,
        0, &qwen_21g_growth));
    assert(!ds4_ssd_qwen_slab_growth_plan_make(
        &qwen_21g_memory, 2 * GIB, 5 * GIB / 2u, 5 * GIB,
        qwen_slab_bytes, NULL));

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
    assert(ds4_ssd_deepseek_expert_major_auto_cache_target(
               &memory, &adaptive) == 4387);

    /* The 64 GiB DeepSeek target is a pressure-bounded route-cycle policy,
     * not a fixed allocation: a smaller admitted candidate wins. Hosts above
     * the measured tier retain the generic adaptive candidate. */
    ds4_ssd_adaptive_cache_plan deepseek_tier = adaptive;
    deepseek_tier.cache_experts = 5500;
    assert(ds4_ssd_deepseek_expert_major_auto_cache_target(
               &memory, &deepseek_tier) == 4387);
    deepseek_tier.cache_experts = 2839;
    assert(ds4_ssd_deepseek_expert_major_auto_cache_target(
               &memory, &deepseek_tier) == 2839);
    deepseek_tier.cache_experts = 5500;
    memory.physical_bytes = 95 * GIB;
    assert(ds4_ssd_deepseek_expert_major_auto_cache_target(
               &memory, &deepseek_tier) == 4387);
    memory.physical_bytes = 96 * GIB;
    assert(ds4_ssd_deepseek_expert_major_auto_cache_target(
               &memory, &deepseek_tier) == 5500);
    memory.physical_bytes = 128 * GIB;
    assert(ds4_ssd_deepseek_expert_major_auto_cache_target(
               &memory, &deepseek_tier) == 5500);
    memory.physical_bytes = 63 * GIB;
    assert(ds4_ssd_deepseek_expert_major_auto_cache_target(
               &memory, &deepseek_tier) == 5500);
    memory.physical_bytes = 64 * GIB;
    deepseek_tier.cache_experts = UINT32_MAX;
    deepseek_tier.floor.working_set_experts = UINT32_MAX / 17u + 1u;
    assert(ds4_ssd_deepseek_expert_major_auto_cache_target(
               &memory, &deepseek_tier) == 0);
    deepseek_tier.floor.working_set_experts = UINT64_MAX / 17u + 1u;
    assert(ds4_ssd_deepseek_expert_major_auto_cache_target(
               &memory, &deepseek_tier) == 0);
    assert(ds4_ssd_deepseek_expert_major_auto_cache_target(NULL,
                                                            &adaptive) == 0);
    assert(ds4_ssd_deepseek_expert_major_auto_cache_target(&memory,
                                                            NULL) == 0);

    /* At 8K and above, the measured 64 GiB tier keeps sixteen complete
     * DeepSeek route cycles hot across prefill and decode. The target remains
     * pressure bounded and does not silently affect unmeasured host tiers. */
    deepseek_tier = adaptive;
    assert(ds4_ssd_deepseek_long_context_cache_target(
               &memory, &deepseek_tier, 4387, 8191) == 0);
    assert(ds4_ssd_deepseek_long_context_cache_target(
               &memory, &deepseek_tier, 4387, 8192) == 4129);
    assert(ds4_ssd_deepseek_long_context_cache_target(
               &memory, &deepseek_tier, 2839, 32768) == 2839);
    assert(ds4_ssd_deepseek_long_context_cache_target(
               &memory, &deepseek_tier, 4387, 65535) == 4129);
    assert(ds4_ssd_deepseek_long_context_cache_target(
               &memory, &deepseek_tier, 4387, 65536) == 2065);
    assert(ds4_ssd_deepseek_long_context_cache_target(
               &memory, &deepseek_tier, 4387, 100000) == 2065);
    deepseek_tier.cache_experts = 2065;
    assert(ds4_ssd_deepseek_long_context_cache_target(
               &memory, &deepseek_tier, 4387, 32768) == 2065);
    deepseek_tier.cache_experts = adaptive.cache_experts;
    memory.physical_bytes = 95 * GIB;
    assert(ds4_ssd_deepseek_long_context_cache_target(
               &memory, &deepseek_tier, 4387, 32768) == 4129);
    memory.physical_bytes = 96 * GIB;
    assert(ds4_ssd_deepseek_long_context_cache_target(
               &memory, &deepseek_tier, 4387, 32768) == 0);
    memory.physical_bytes = 63 * GIB;
    assert(ds4_ssd_deepseek_long_context_cache_target(
               &memory, &deepseek_tier, 4387, 32768) == 0);
    memory.physical_bytes = 64 * GIB;
    deepseek_tier.floor.working_set_experts = UINT64_MAX / 16u + 1u;
    assert(ds4_ssd_deepseek_long_context_cache_target(
               &memory, &deepseek_tier, UINT32_MAX, 8192) == 0);
    assert(ds4_ssd_deepseek_long_context_cache_target(
               NULL, &deepseek_tier, 4387, 8192) == 0);
    assert(ds4_ssd_deepseek_long_context_cache_target(
               &memory, NULL, 4387, 8192) == 0);

    /* Resume decisions use the work size for the batching floor and the total
     * resulting context for the post-prefill cap. */
    assert(ds4_ssd_deepseek_prefill_phase_cache_target(
               8192, 8192, 760, 259, 4129, 2065) == 259);
    assert(ds4_ssd_deepseek_prefill_phase_cache_target(
               2048, 9000, 760, 259, 4129, 2065) == 259);
    assert(ds4_ssd_deepseek_prefill_phase_cache_target(
               4096, 8192, 760, 259, 4129, 2065) == 259);
    assert(ds4_ssd_deepseek_prefill_phase_cache_target(
               292, 8192, 760, 259, 4129, 2065) == 4129);
    assert(ds4_ssd_deepseek_prefill_phase_cache_target(
               292, 65536, 760, 259, 4129, 2065) == 2065);
    assert(ds4_ssd_deepseek_prefill_phase_cache_target(
               128, 128, 760, 259, 4129, 2065) == 0);
    assert(ds4_ssd_deepseek_prefill_phase_cache_target(
               32, 32, 0, 259, 4129, 2065) == 259);
    assert(ds4_ssd_deepseek_prefill_phase_cache_target(
               31, 8192, 0, 259, 4129, 2065) == 4129);
    assert(ds4_ssd_deepseek_prefill_phase_cache_target(
               31, 8063, 0, 259, 4129, 2065) == 0);
    assert(ds4_ssd_deepseek_prefill_phase_cache_target(
               31, 8064, 0, 259, 4129, 2065) == 4129);
    assert(ds4_ssd_deepseek_post_prefill_cache_target(
               8063, 4129, 2065, 4387) == 4387);
    assert(ds4_ssd_deepseek_post_prefill_cache_target(
               8064, 4129, 2065, 4387) == 4129);
    assert(ds4_ssd_deepseek_post_prefill_cache_target(
               8191, 4129, 2065, 4387) == 4129);
    assert(ds4_ssd_deepseek_post_prefill_cache_target(
               8192, 4129, 2065, 4387) == 4129);
    assert(ds4_ssd_deepseek_post_prefill_cache_target(
               65407, 4129, 2065, 4387) == 4129);
    assert(ds4_ssd_deepseek_post_prefill_cache_target(
               65408, 4129, 2065, 4387) == 2065);
    assert(ds4_ssd_deepseek_post_prefill_cache_target(
               65536, 4129, 2065, 4387) == 2065);
    assert(ds4_ssd_deepseek_post_prefill_cache_target(
               32768, 0, 0, 4387) == 4387);
    assert(ds4_ssd_deepseek_post_prefill_seed_allowed(
               true, true, true, 2065, 2065));
    assert(ds4_ssd_deepseek_post_prefill_seed_allowed(
               true, true, true, 2065, 4129));
    assert(!ds4_ssd_deepseek_post_prefill_seed_allowed(
               true, false, true, 259, 259));
    assert(!ds4_ssd_deepseek_post_prefill_seed_allowed(
               false, true, true, 2065, 2065));
    assert(!ds4_ssd_deepseek_post_prefill_seed_allowed(
               true, true, false, 2065, 2065));
    assert(!ds4_ssd_deepseek_post_prefill_seed_allowed(
               true, true, true, 2065, 259));
    assert(!ds4_ssd_deepseek_post_prefill_seed_allowed(
               true, true, true, 0, 2065));

    /* The buffer-reuse optimization introduced with GLM must never spill
     * into DeepSeek merely because its record is below the byte threshold. */
    assert(ds4_ssd_glm_streaming_batch_reuse_allowed(true,
                                                       4 * MIB,
                                                       4 * MIB));
    assert(!ds4_ssd_glm_streaming_batch_reuse_allowed(false,
                                                        4 * MIB,
                                                        4 * MIB));
    assert(!ds4_ssd_glm_streaming_batch_reuse_allowed(true,
                                                        8 * MIB,
                                                        1));
    assert(!ds4_ssd_glm_streaming_batch_reuse_allowed(true,
                                                        UINT64_MAX,
                                                        1));
    assert(!ds4_ssd_glm_streaming_batch_reuse_allowed(true, 0, 1));
    assert(!ds4_ssd_glm_streaming_batch_reuse_allowed(true, 1, 0));

    /* A real 64 GiB post-pin snapshot can have little immediately free RAM but
     * a large bounded file-backed inactive set. Normal pressure makes those
     * pages reclaimable; full credit reaches the same 9/16 envelope instead of
     * changing AUTO tiers according to which GGUF ran first. */
    memory = (ds4_ssd_host_memory){
        .physical_bytes = 64 * GIB,
        .recommended_bytes = 51 * GIB + 84 * GIB / 100u,
        .free_bytes = 13 * GIB + 30 * GIB / 100u,
        .purgeable_bytes = 11 * GIB / 100u,
        .inactive_bytes = 19 * GIB + 87 * GIB / 100u,
        .file_backed_bytes = 22 * GIB + 43 * GIB / 100u,
        .pressure_status_available = true,
        .pressure_normal = true,
    };
    assert(ds4_ssd_adaptive_cache_plan_make_with_static_reserve(
        &memory, 86 * GIB / 100u, 8 * GIB + GIB / 5u, true, 43, 6,
        flash_expert_bytes, flash_max_cacheable, &adaptive));
    assert(adaptive.normal_pressure_full_file_credit_active);
    assert(adaptive.reclaimable_bytes ==
           memory.free_bytes + memory.inactive_bytes);
    assert(adaptive.cache_experts == 4387);

    /* The unpinned launch must choose the same tier when a preceding model run
     * has converted immediately-free pages into reclaimable GGUF cache. The
     * independent Metal envelope still leaves ample platform headroom. */
    memory = (ds4_ssd_host_memory){
        .physical_bytes = 64 * GIB,
        .recommended_bytes = 51 * GIB + 84 * GIB / 100u,
        .free_bytes = 5 * GIB + 57 * GIB / 100u,
        .purgeable_bytes = 3 * GIB / 100u,
        .inactive_bytes = 26 * GIB + 66 * GIB / 100u,
        .file_backed_bytes = 42 * GIB + 60 * GIB / 100u,
        .pressure_status_available = true,
        .pressure_normal = true,
    };
    assert(ds4_ssd_adaptive_cache_plan_make_with_static_reserve(
        &memory, 19 * GIB / 100u, 8 * GIB + GIB / 5u, false, 43, 6,
        flash_expert_bytes, flash_max_cacheable, &adaptive));
    assert(adaptive.normal_pressure_full_file_credit_active);
    assert(adaptive.current_headroom_bytes == GIB);
    assert(adaptive.pressure_margin_bytes == GIB);
    assert(adaptive.cache_experts == 4387);

    /* The new credit is a measured 64 GiB policy, not a global relaxation.
     * Larger hosts retain their prior half-credit accounting. */
    memory.physical_bytes = 128 * GIB;
    memory.recommended_bytes = 104 * GIB;
    assert(ds4_ssd_adaptive_cache_plan_make_with_static_reserve(
        &memory, 86 * GIB / 100u, 8 * GIB + GIB / 5u, true, 43, 6,
        flash_expert_bytes, flash_max_cacheable, &adaptive));
    assert(!adaptive.normal_pressure_full_file_credit_active);
    assert(adaptive.reclaimable_bytes ==
           memory.free_bytes + memory.purgeable_bytes +
               memory.inactive_bytes / 2u);

    /* A warmer launch exposes more file-backed pages as reclaimable.  The
     * safety budget can now fit 4645, but the envelope prevents startup-order
     * feedback from growing the wired cache. */
    memory = (ds4_ssd_host_memory){
        .physical_bytes = 64 * GIB,
        .recommended_bytes = 51 * GIB + 84 * GIB / 100u,
        .free_bytes = 41 * GIB + 27 * GIB / 100u,
    };
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
