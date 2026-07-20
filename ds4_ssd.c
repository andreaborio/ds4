#include "ds4_ssd.h"

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif

static const uint64_t DS4_GIB = 1024ull * 1024ull * 1024ull;

static uint64_t saturating_add_u64(uint64_t a, uint64_t b) {
    return a > UINT64_MAX - b ? UINT64_MAX : a + b;
}

const char *ds4_residency_mode_name(ds4_residency_mode mode) {
    switch (mode) {
    case DS4_RESIDENCY_AUTO:     return "auto";
    case DS4_RESIDENCY_RESIDENT: return "resident";
    case DS4_RESIDENCY_SSD:      return "ssd";
    }
    return "invalid";
}

const char *ds4_residency_reason_name(ds4_residency_reason reason) {
    switch (reason) {
    case DS4_RESIDENCY_REASON_EXPLICIT_RESIDENT:
        return "explicit resident override";
    case DS4_RESIDENCY_REASON_EXPLICIT_SSD:
        return "explicit SSD override";
    case DS4_RESIDENCY_REASON_NON_METAL_AUTO:
        return "AUTO currently preserves resident mode outside Metal";
    case DS4_RESIDENCY_REASON_METAL_FITS:
        return "estimated Metal residency plan fits the conservative budget";
    case DS4_RESIDENCY_REASON_METAL_EXCEEDS:
        return "estimated Metal residency plan exceeds the conservative budget";
    case DS4_RESIDENCY_REASON_METAL_CURRENT_PRESSURE:
        return "live-pressure budget cannot admit full-model mapped mode";
    case DS4_RESIDENCY_REASON_METAL_BUDGET_UNAVAILABLE:
        return "Metal recommended working-set budget is unavailable";
    case DS4_RESIDENCY_REASON_MODEL_REQUIRES_SSD:
        return "the model family is qualified only for SSD streaming";
    case DS4_RESIDENCY_REASON_INSPECT_ONLY:
        return "model inspection defers runtime residency selection";
    }
    return "unknown reason";
}

bool ds4_residency_plan_make(bool                metal_backend,
                             ds4_residency_mode  requested,
                             uint64_t            model_bytes,
                             uint64_t            runtime_bytes,
                             uint64_t            recommended_bytes,
                             uint64_t            external_reserved_bytes,
                             ds4_residency_plan *out) {
    if (!out || requested < DS4_RESIDENCY_AUTO ||
        requested > DS4_RESIDENCY_SSD) {
        return false;
    }

    memset(out, 0, sizeof(*out));
    out->requested = requested;
    out->recommended_bytes = recommended_bytes;
    out->external_reserved_bytes = external_reserved_bytes;
    out->model_bytes = model_bytes;
    out->runtime_bytes = runtime_bytes;

    if (recommended_bytes > external_reserved_bytes) {
        out->budget_bytes = recommended_bytes - external_reserved_bytes;
    }

    if (metal_backend && recommended_bytes != 0) {
        out->headroom_bytes = recommended_bytes / 5u;
        if (out->headroom_bytes < 2u * DS4_GIB) {
            out->headroom_bytes = 2u * DS4_GIB;
        }
    }
    out->required_bytes = saturating_add_u64(model_bytes, runtime_bytes);
    out->required_bytes = saturating_add_u64(out->required_bytes,
                                             out->headroom_bytes);

    if (requested == DS4_RESIDENCY_RESIDENT) {
        out->resolved = DS4_RESIDENCY_RESIDENT;
        out->reason = DS4_RESIDENCY_REASON_EXPLICIT_RESIDENT;
        return true;
    }
    if (requested == DS4_RESIDENCY_SSD) {
        out->resolved = DS4_RESIDENCY_SSD;
        out->reason = DS4_RESIDENCY_REASON_EXPLICIT_SSD;
        return true;
    }
    if (!metal_backend) {
        out->resolved = DS4_RESIDENCY_RESIDENT;
        out->reason = DS4_RESIDENCY_REASON_NON_METAL_AUTO;
        return true;
    }
    if (recommended_bytes == 0) {
        /* AUTO is conservative: if Metal cannot report a budget, do not risk
         * attempting to map a model whose residency has not been proven. */
        out->resolved = DS4_RESIDENCY_SSD;
        out->reason = DS4_RESIDENCY_REASON_METAL_BUDGET_UNAVAILABLE;
        return true;
    }

    if (out->required_bytes <= out->budget_bytes) {
        out->resolved = DS4_RESIDENCY_RESIDENT;
        out->reason = DS4_RESIDENCY_REASON_METAL_FITS;
    } else {
        out->resolved = DS4_RESIDENCY_SSD;
        out->reason = DS4_RESIDENCY_REASON_METAL_EXCEEDS;
    }
    return true;
}

bool ds4_residency_plan_apply_ssd_only(ds4_residency_mode  requested,
                                       ds4_residency_plan *plan) {
    if (!plan || requested < DS4_RESIDENCY_AUTO ||
        requested > DS4_RESIDENCY_SSD) {
        return false;
    }

    plan->requested = requested;
    plan->resolved = DS4_RESIDENCY_SSD;
    if (requested == DS4_RESIDENCY_RESIDENT) {
        plan->reason = DS4_RESIDENCY_REASON_MODEL_REQUIRES_SSD;
        return false;
    }
    plan->reason = requested == DS4_RESIDENCY_AUTO ?
        DS4_RESIDENCY_REASON_MODEL_REQUIRES_SSD :
        DS4_RESIDENCY_REASON_EXPLICIT_SSD;
    return true;
}

bool ds4_parse_gib_arg(const char *s, uint64_t *bytes) {
    if (bytes) *bytes = 0;
    if (!s || !s[0] || !bytes) return false;

    size_t len = strlen(s);
    if (len > 2 &&
        (s[len - 2] == 'g' || s[len - 2] == 'G') &&
        (s[len - 1] == 'b' || s[len - 1] == 'B')) {
        len -= 2;
    }
    if (len == 0) return false;
    for (size_t i = 0; i < len; i++) {
        if (!isdigit((unsigned char)s[i])) return false;
    }

    char numbuf[32];
    if (len >= sizeof(numbuf)) return false;
    memcpy(numbuf, s, len);
    numbuf[len] = '\0';

    errno = 0;
    unsigned long long v = strtoull(numbuf, NULL, 10);
    if (errno != 0 || v == 0 || v > UINT64_MAX / DS4_GIB) return false;

    *bytes = (uint64_t)v * DS4_GIB;
    return true;
}

bool ds4_parse_streaming_cache_experts_arg(const char *s,
                                           uint32_t   *experts,
                                           uint64_t   *bytes) {
    if (experts) *experts = 0;
    if (bytes) *bytes = 0;
    if (!s || !s[0] || !experts || !bytes) return false;

    const size_t len = strlen(s);
    if (len > 2 &&
        (s[len - 2] == 'g' || s[len - 2] == 'G') &&
        (s[len - 1] == 'b' || s[len - 1] == 'B')) {
        return ds4_parse_gib_arg(s, bytes);
    }

    for (size_t i = 0; i < len; i++) {
        if (!isdigit((unsigned char)s[i])) return false;
    }

    errno = 0;
    unsigned long v = strtoul(s, NULL, 10);
    if (errno != 0 || v == 0 || v > UINT32_MAX) return false;

    *experts = (uint32_t)v;
    return true;
}

bool ds4_parse_streaming_hotlist_priority_policy(
        const char                            *s,
        ds4_streaming_hotlist_priority_policy *out) {
    if (!out) return false;
    *out = (ds4_streaming_hotlist_priority_policy){
        .mode = DS4_STREAMING_HOTLIST_PRIORITY_ADAPTIVE,
        .priority = 1u,
    };
    if (!s || !s[0] || strcmp(s, "adaptive") == 0) return true;
    if (strcmp(s, "legacy") == 0) {
        out->mode = DS4_STREAMING_HOTLIST_PRIORITY_LEGACY;
        out->priority = 0;
        return true;
    }

    const size_t len = strlen(s);
    for (size_t i = 0; i < len; i++) {
        if (!isdigit((unsigned char)s[i])) return false;
    }
    errno = 0;
    const unsigned long value = strtoul(s, NULL, 10);
    if (errno != 0 || value == 0 || value > UINT32_MAX) return false;
    out->mode = DS4_STREAMING_HOTLIST_PRIORITY_FIXED;
    out->priority = (uint32_t)value;
    return true;
}

uint32_t ds4_ssd_cache_experts_for_byte_budget(uint64_t bytes,
                                               uint64_t per_expert_bytes) {
    if (bytes == 0 || per_expert_bytes == 0) return 0;
    const uint64_t experts = bytes / per_expert_bytes;
    if (experts == 0 || experts > UINT32_MAX) return 0;
    return (uint32_t)experts;
}

bool ds4_ssd_cache_plan_for_model_target(uint64_t            model_target_bytes,
                                         uint64_t            non_routed_bytes,
                                         uint64_t            per_expert_bytes,
                                         uint64_t            max_model_experts,
                                         ds4_ssd_cache_plan *out) {
    if (!out) return false;
    memset(out, 0, sizeof(*out));
    if (model_target_bytes == 0 || per_expert_bytes == 0) return false;

    out->model_target_bytes = model_target_bytes;
    if (out->model_target_bytes <= non_routed_bytes) return false;
    out->cache_bytes = out->model_target_bytes - non_routed_bytes;

    uint64_t cache_experts = out->cache_bytes / per_expert_bytes;
    if (cache_experts == 0) return false;
    if (max_model_experts != 0 && cache_experts > max_model_experts) {
        cache_experts = max_model_experts;
    }
    if (cache_experts > UINT32_MAX) cache_experts = UINT32_MAX;

    out->cache_experts = (uint32_t)cache_experts;
    out->effective_cache_bytes = cache_experts * per_expert_bytes;
    return out->cache_experts != 0;
}

bool ds4_ssd_expert_cache_floor_make(
        uint64_t                    cacheable_routed_layers,
        uint64_t                    experts_per_token,
        uint64_t                    per_expert_bytes,
        ds4_ssd_expert_cache_floor *out) {
    if (!out) return false;
    memset(out, 0, sizeof(*out));
    if (cacheable_routed_layers == 0 ||
        experts_per_token == 0 ||
        per_expert_bytes == 0 ||
        cacheable_routed_layers > UINT64_MAX / experts_per_token) {
        return false;
    }

    const uint64_t working_set =
        cacheable_routed_layers * experts_per_token;
    if (working_set == UINT64_MAX || working_set > UINT64_MAX / 2u) {
        return false;
    }
    const uint64_t minimum_cache = working_set + 1u;
    if (minimum_cache > UINT64_MAX / per_expert_bytes) return false;

    out->working_set_experts = working_set;
    out->minimum_cache_experts = minimum_cache;
    out->minimum_cache_bytes = minimum_cache * per_expert_bytes;
    out->warning_cache_experts = working_set * 2u;
    return true;
}

bool ds4_ssd_low_ram_cache_policy(uint64_t physical_bytes) {
    return physical_bytes != 0 && physical_bytes <= 16u * DS4_GIB;
}

bool ds4_ssd_resident_pressure_plan_make(
        const ds4_ssd_host_memory      *memory,
        uint64_t                        model_bytes,
        uint64_t                        runtime_bytes,
        ds4_ssd_resident_pressure_plan *out) {
    if (!out) return false;
    memset(out, 0, sizeof(*out));
    if (!memory || memory->physical_bytes == 0 || model_bytes == 0) {
        return false;
    }

    out->physical_bytes = memory->physical_bytes;
    const uint64_t bounded_inactive_bytes =
        memory->inactive_bytes < memory->file_backed_bytes ?
            memory->inactive_bytes : memory->file_backed_bytes;
    out->pressure_status_available = memory->pressure_status_available;
    out->pressure_normal = memory->pressure_status_available &&
                           memory->pressure_normal;
    /* inactive_bytes and file_backed_bytes are separate VM aggregates, not a
     * proven intersection. Their minimum is only a bounded working-set proxy.
     * macOS can reclaim that proxy aggressively while system pressure is
     * normal; elevated or unavailable pressure keeps the previous half-credit
     * fail-closed policy. Purgeable pages may overlap inactive queues, so take
     * the larger pool instead of adding both. The SSD cache planner below
     * normally stays on half-credit because it wires new cache pages; its
     * bounded Qwen low-RAM path may take full credit only while pressure is
     * explicitly normal and AUTO remains capped at the safe expert floor. */
    out->inactive_credit_bytes = out->pressure_normal ?
        bounded_inactive_bytes : bounded_inactive_bytes / 2u;
    const uint64_t reclaimable_working_set =
        memory->purgeable_bytes > out->inactive_credit_bytes ?
            memory->purgeable_bytes : out->inactive_credit_bytes;
    uint64_t reclaimable = saturating_add_u64(memory->free_bytes,
                                               reclaimable_working_set);
    if (reclaimable > memory->physical_bytes) {
        reclaimable = memory->physical_bytes;
    }
    out->reclaimable_bytes = reclaimable;

    out->current_headroom_bytes = memory->physical_bytes / 16u;
    if (out->current_headroom_bytes < 2u * DS4_GIB) {
        out->current_headroom_bytes = 2u * DS4_GIB;
    }
    out->pressure_margin_bytes = memory->physical_bytes / 64u;
    if (out->pressure_margin_bytes < DS4_GIB / 4u) {
        out->pressure_margin_bytes = DS4_GIB / 4u;
    }

    if (model_bytes > UINT64_MAX - runtime_bytes) return false;
    uint64_t required = model_bytes + runtime_bytes;
    if (required > UINT64_MAX - out->current_headroom_bytes) return false;
    required += out->current_headroom_bytes;
    if (required > UINT64_MAX - out->pressure_margin_bytes) return false;
    required += out->pressure_margin_bytes;
    out->required_bytes = required;
    out->fits = required <= reclaimable;
    return true;
}

bool ds4_ssd_static_pin_host_supported(uint64_t physical_bytes) {
    /* The static pin is an experimental performance arm, not a correctness
     * requirement.  It has only been validated on a 64 GiB unified-memory
     * host; smaller machines keep these weights pageable. */
    return physical_bytes >= 64u * DS4_GIB;
}

static bool ds4_ssd_adaptive_cache_plan_make_policy(
        const ds4_ssd_host_memory   *memory,
        uint64_t                     runtime_bytes,
        uint64_t                     static_working_set_bytes,
        bool                         static_already_pinned,
        bool                         apply_deepseek_tuning,
        uint64_t                     cacheable_routed_layers,
        uint64_t                     experts_per_token,
        uint64_t                     per_expert_bytes,
        uint64_t                     max_cacheable_experts,
        ds4_ssd_adaptive_cache_plan *out) {
    if (!out) return false;
    memset(out, 0, sizeof(*out));
    if (!memory ||
        memory->physical_bytes == 0 ||
        memory->recommended_bytes == 0 ||
        max_cacheable_experts == 0 ||
        !ds4_ssd_expert_cache_floor_make(cacheable_routed_layers,
                                          experts_per_token,
                                          per_expert_bytes,
                                          &out->floor)) {
        return false;
    }
    const bool low_ram_host =
        ds4_ssd_low_ram_cache_policy(memory->physical_bytes);
    const bool qwen_low_ram_policy =
        !apply_deepseek_tuning && low_ram_host;
    out->low_ram_shared_static_headroom_active = qwen_low_ram_policy;
    out->low_ram_floor_ceiling_active =
        apply_deepseek_tuning && low_ram_host;
    /* DeepSeek Flash's measured <=16 GiB policy leaves its complete static set
     * pageable and later caps AUTO at the minimum expert tier. Strict Qwen
     * retains the static charge on the same host class; the overlap with
     * ordinary headroom is accounted for separately below. Larger hosts keep
     * the strict static reserve independent of headroom. */
    const uint64_t static_reserve_bytes =
        apply_deepseek_tuning && low_ram_host ? 0 :
                                                static_working_set_bytes;
    out->pageable_static_reserve_bytes =
        static_already_pinned ? 0 : static_reserve_bytes;
    out->platform_static_reserve_bytes = static_reserve_bytes;

    const uint64_t file_inactive_bytes =
        memory->inactive_bytes < memory->file_backed_bytes ?
            memory->inactive_bytes : memory->file_backed_bytes;
    const bool deepseek_64g_normal_credit =
        apply_deepseek_tuning &&
        memory->physical_bytes >= 64u * DS4_GIB &&
        memory->physical_bytes < 96u * DS4_GIB &&
        memory->pressure_status_available && memory->pressure_normal;
    out->normal_pressure_full_file_credit_active =
        deepseek_64g_normal_credit;
    uint64_t reclaimable = 0;
    if ((qwen_low_ram_policy || deepseek_64g_normal_credit) &&
        memory->pressure_status_available && memory->pressure_normal) {
        /* Under a qualified normal-pressure policy, give the bounded
         * file-backed inactive set full credit so warm GGUF pages are not
         * mistaken for irreversibly used RAM. Purgeable pages may overlap that
         * set, so take max(), not a sum.
         *
         * Qwen uses this only for its bounded <=16 GiB floor. DeepSeek uses it
         * only on the measured [64,96) GiB tier, where the independent 9/16
         * cache envelope still caps wired growth at the retained M5 sweet
         * spot. Missing/elevated pressure, smaller Macs, and >=96 GiB hosts
         * preserve the half-credit policy below. */
        const uint64_t reclaimable_working_set =
            memory->purgeable_bytes > file_inactive_bytes ?
                memory->purgeable_bytes : file_inactive_bytes;
        reclaimable = saturating_add_u64(memory->free_bytes,
                                          reclaimable_working_set);
    } else {
        reclaimable = saturating_add_u64(memory->free_bytes,
                                          memory->purgeable_bytes);
        reclaimable = saturating_add_u64(reclaimable,
                                          file_inactive_bytes / 2u);
    }
    if (reclaimable > memory->physical_bytes) {
        reclaimable = memory->physical_bytes;
    }
    out->reclaimable_bytes = reclaimable;

    const uint64_t two_gib = 2u * DS4_GIB;
    const uint64_t quarter_gib = DS4_GIB / 4u;
    out->current_headroom_bytes = memory->physical_bytes / 16u;
    if (out->current_headroom_bytes < two_gib) {
        out->current_headroom_bytes = two_gib;
    }
    /* DeepSeek's pageable static set already shares ordinary headroom.  The
     * Qwen strict policy keeps the static charge independent on larger hosts,
     * but on a 16 GiB host it must share headroom: the pages are not pinned and
     * can be reclaimed by macOS while DS4 streams them again from the GGUF. */
    const bool pageable_static_overlaps_headroom =
        apply_deepseek_tuning || qwen_low_ram_policy;
    if (pageable_static_overlaps_headroom && static_already_pinned) {
        /* The live snapshot has already fallen by the pinned bytes.  Preserve
         * the same pre-pin max(base, static) policy by charging only the
         * residual base slack not covered by that drop. */
        out->current_headroom_bytes =
            out->current_headroom_bytes > static_reserve_bytes ?
                out->current_headroom_bytes - static_reserve_bytes : 0;
    } else if (pageable_static_overlaps_headroom && !qwen_low_ram_policy &&
               !deepseek_64g_normal_credit &&
               out->current_headroom_bytes <
               out->pageable_static_reserve_bytes) {
        out->current_headroom_bytes = out->pageable_static_reserve_bytes;
    }
    out->pressure_margin_bytes = memory->physical_bytes / 64u;
    if (out->pressure_margin_bytes < quarter_gib) {
        out->pressure_margin_bytes = quarter_gib;
    }
    if (deepseek_64g_normal_credit) {
        /* On the qualified 64 GiB tier, the independent Metal envelope leaves
         * roughly 14 GiB below recommendedMaxWorkingSetSize after static and
         * runtime pages. Keeping another 8.2 GiB current-pressure reserve made
         * AUTO depend on whether the previous GGUF was still in the file
         * cache. Freeze one 2 GiB live-pressure reserve here (including this
         * pressure margin); elevated or unavailable pressure never enters the
         * branch. */
        out->current_headroom_bytes =
            two_gib > out->pressure_margin_bytes
                ? two_gib - out->pressure_margin_bytes : 0;
    }
    if (qwen_low_ram_policy) {
        /* Qwen's bounded <=16 GiB policy freezes one 2 GiB request reserve,
         * including the pressure allowance.  The pageable static mapping may
         * occupy that reserve and be reclaimed; charging max(static, 2 GiB)
         * and then another pressure margin left useful RAM idle and violated
         * the policy's single-headroom contract. */
        out->current_headroom_bytes =
            two_gib > out->pressure_margin_bytes
                ? two_gib - out->pressure_margin_bytes : 0;
    }
    out->platform_headroom_bytes = memory->physical_bytes / 8u;
    if (out->platform_headroom_bytes < two_gib) {
        out->platform_headroom_bytes = two_gib;
    }
    if (pageable_static_overlaps_headroom && static_already_pinned) {
        /* Pinned pages are irreversible pressure and therefore additive to
         * the ordinary platform slack.  Pageable static pages can instead
         * occupy that slack and remain reclaimable, so max() is sufficient. */
        out->platform_headroom_bytes = saturating_add_u64(
            out->platform_headroom_bytes,
            out->platform_static_reserve_bytes);
    } else if (pageable_static_overlaps_headroom && !qwen_low_ram_policy &&
               out->platform_headroom_bytes <
               out->platform_static_reserve_bytes) {
        out->platform_headroom_bytes = out->platform_static_reserve_bytes;
    }

    uint64_t current_reserve =
        saturating_add_u64(out->current_headroom_bytes,
                           out->pressure_margin_bytes);
    if (!pageable_static_overlaps_headroom) {
        /* Qwen's strict bounded mode treats the mapped static page set as an
         * independent charge.  It is not allowed to consume the ordinary
         * host-pressure headroom that protects the rest of the system. */
        current_reserve = saturating_add_u64(
            current_reserve,
            out->pageable_static_reserve_bytes);
    }
    /* The host snapshot is taken while the engine is opened, before a session
     * allocates its modeled KV/cache/scratch footprint.  Reserve that future
     * allocation in both independent safety constraints: subtracting it only
     * from the platform working-set limit can overcommit whenever current
     * memory pressure is the tighter bound. */
    current_reserve = saturating_add_u64(current_reserve, runtime_bytes);
    if (reclaimable > current_reserve) {
        out->current_wire_budget_bytes = reclaimable - current_reserve;
    }

    uint64_t platform_reserve =
        saturating_add_u64(runtime_bytes, out->platform_headroom_bytes);
    if (!pageable_static_overlaps_headroom) {
        platform_reserve = saturating_add_u64(
            platform_reserve,
            out->platform_static_reserve_bytes);
    }
    if (memory->recommended_bytes > platform_reserve) {
        out->platform_wire_budget_bytes =
            memory->recommended_bytes - platform_reserve;
    }

    out->safety_wire_budget_bytes =
        out->current_wire_budget_bytes < out->platform_wire_budget_bytes ?
            out->current_wire_budget_bytes : out->platform_wire_budget_bytes;
    /* A warm model mapping can turn useful file-backed pages into apparently
     * reclaimable memory and make AUTO grow between identical launches.  The
     * 9/16 envelope is just above the measured M5 Flash winner (55.21% of the
     * Metal recommended set), but below the larger tier which consumed 3.57
     * GiB more wired memory without improving decode.  Current pressure can
     * still shrink below this hardware-scaled ceiling. */
    out->cache_envelope_bytes = apply_deepseek_tuning ?
        (memory->recommended_bytes / 16u) * 9u +
            ((memory->recommended_bytes % 16u) * 9u) / 16u :
        out->safety_wire_budget_bytes;
    out->wire_budget_bytes =
        out->safety_wire_budget_bytes < out->cache_envelope_bytes ?
            out->safety_wire_budget_bytes : out->cache_envelope_bytes;
    uint64_t raw_experts = out->wire_budget_bytes / per_expert_bytes;
    if (raw_experts > max_cacheable_experts) {
        raw_experts = max_cacheable_experts;
    }
    if (raw_experts > UINT32_MAX) raw_experts = UINT32_MAX;
    if (raw_experts < out->floor.minimum_cache_experts) return false;

    /* DeepSeek measurements on M1 16 GiB show that the second complete cache
     * tier loses end-to-end time as page-cache displacement dominates. Keep
     * that model-specific performance ceiling; Qwen instead consumes every
     * complete tier admitted by the pressure and platform budgets above. */
    if (out->low_ram_floor_ceiling_active &&
        raw_experts > out->floor.minimum_cache_experts) {
        raw_experts = out->floor.minimum_cache_experts;
    }

    /* Grow only by complete per-token working sets.  Besides leaving useful
     * pressure slack, this prevents small changes in free pages from buying a
     * cache which still cannot retain one more token's routed-expert cycle. */
    /* Once every cacheable expert fits, retain the exact full-model count.
     * There is no eviction cycle to round away in that terminal state. */
    const uint64_t cache_experts =
        raw_experts == max_cacheable_experts ?
            raw_experts :
            1u + out->floor.working_set_experts *
                     ((raw_experts - 1u) / out->floor.working_set_experts);
    if (cache_experts < out->floor.minimum_cache_experts ||
        cache_experts > UINT32_MAX ||
        cache_experts > UINT64_MAX / per_expert_bytes) {
        return false;
    }

    out->cache_experts = (uint32_t)cache_experts;
    out->cache_bytes = cache_experts * per_expert_bytes;
    if (qwen_low_ram_policy &&
        (!memory->pressure_status_available || !memory->pressure_normal)) {
        /* The 2 GiB envelope deliberately spends more of a small host's free
         * budget.  It is therefore valid only with an affirmative normal
         * pressure signal; missing telemetry is not evidence of headroom. */
        return false;
    }
    return true;
}

bool ds4_ssd_adaptive_cache_plan_make_with_static_reserve(
        const ds4_ssd_host_memory   *memory,
        uint64_t                     runtime_bytes,
        uint64_t                     static_working_set_bytes,
        bool                         static_already_pinned,
        uint64_t                     cacheable_routed_layers,
        uint64_t                     experts_per_token,
        uint64_t                     per_expert_bytes,
        uint64_t                     max_cacheable_experts,
        ds4_ssd_adaptive_cache_plan *out) {
    return ds4_ssd_adaptive_cache_plan_make_policy(
        memory,
        runtime_bytes,
        static_working_set_bytes,
        static_already_pinned,
        true,
        cacheable_routed_layers,
        experts_per_token,
        per_expert_bytes,
        max_cacheable_experts,
        out);
}

bool ds4_ssd_adaptive_cache_plan_make_strict_with_static_reserve(
        const ds4_ssd_host_memory   *memory,
        uint64_t                     runtime_bytes,
        uint64_t                     static_working_set_bytes,
        bool                         static_already_pinned,
        uint64_t                     cacheable_routed_layers,
        uint64_t                     experts_per_token,
        uint64_t                     per_expert_bytes,
        uint64_t                     max_cacheable_experts,
        ds4_ssd_adaptive_cache_plan *out) {
    return ds4_ssd_adaptive_cache_plan_make_policy(
        memory,
        runtime_bytes,
        static_working_set_bytes,
        static_already_pinned,
        false,
        cacheable_routed_layers,
        experts_per_token,
        per_expert_bytes,
        max_cacheable_experts,
        out);
}

bool ds4_ssd_adaptive_cache_plan_make(
        const ds4_ssd_host_memory   *memory,
        uint64_t                     runtime_bytes,
        uint64_t                     cacheable_routed_layers,
        uint64_t                     experts_per_token,
        uint64_t                     per_expert_bytes,
        uint64_t                     max_cacheable_experts,
        ds4_ssd_adaptive_cache_plan *out) {
    return ds4_ssd_adaptive_cache_plan_make_with_static_reserve(
        memory,
        runtime_bytes,
        0,
        false,
        cacheable_routed_layers,
        experts_per_token,
        per_expert_bytes,
        max_cacheable_experts,
        out);
}

uint32_t ds4_ssd_glm_expert_major_auto_cache_target(
        const ds4_ssd_host_memory         *memory,
        const ds4_ssd_adaptive_cache_plan *plan) {
    if (!memory || !plan || plan->cache_experts == 0 ||
        plan->floor.minimum_cache_experts == 0 ||
        plan->floor.minimum_cache_experts > UINT32_MAX ||
        plan->floor.minimum_cache_experts > plan->cache_experts) {
        return 0;
    }

    const bool measured_64g_tier =
        memory->physical_bytes >= 64u * DS4_GIB &&
        memory->physical_bytes < 96u * DS4_GIB;
    return measured_64g_tier ?
        (uint32_t)plan->floor.minimum_cache_experts : plan->cache_experts;
}

uint32_t ds4_ssd_deepseek_expert_major_auto_cache_target(
        const ds4_ssd_host_memory         *memory,
        const ds4_ssd_adaptive_cache_plan *plan) {
    if (!memory || !plan || plan->cache_experts == 0 ||
        plan->floor.working_set_experts == 0 ||
        plan->floor.minimum_cache_experts == 0 ||
        plan->floor.minimum_cache_experts > plan->cache_experts) {
        return 0;
    }

    const bool measured_64g_tier =
        memory->physical_bytes >= 64u * DS4_GIB &&
        memory->physical_bytes < 96u * DS4_GIB;
    if (!measured_64g_tier) return plan->cache_experts;

    /* Seventeen complete token working sets are the largest stable tier
     * admitted by the 9/16 envelope on the measured 64 GiB M5 Pro.  After
     * isolating GLM-only batch reuse this tier recovered decode throughput;
     * the next practical tier crossed the runtime pressure guard.  Express
     * the target in route cycles rather than one artifact-specific count and
     * retain the generic point-in-time plan as the safety ceiling, so AUTO
     * still contracts when the host has less reclaimable memory. */
    const uint64_t measured_cycles = 17u;
    if (plan->floor.working_set_experts >
        (UINT64_MAX - 1u) / measured_cycles) {
        return 0;
    }
    const uint64_t measured_target =
        1u + measured_cycles * plan->floor.working_set_experts;
    if (measured_target > UINT32_MAX) return 0;
    return plan->cache_experts < measured_target ?
        plan->cache_experts : (uint32_t)measured_target;
}

bool ds4_ssd_glm_streaming_batch_reuse_allowed(
        bool     glm_model,
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes) {
    if (!glm_model || gate_expert_bytes == 0 || down_expert_bytes == 0 ||
        gate_expert_bytes > (UINT64_MAX - down_expert_bytes) / 2u) {
        return false;
    }
    const uint64_t slot_bytes =
        gate_expert_bytes * 2u + down_expert_bytes;
    return slot_bytes <= 16u * 1024u * 1024u;
}

bool ds4_ssd_working_set_after_reserve(uint64_t  recommended_bytes,
                                       uint64_t  runtime_bytes,
                                       uint64_t  external_reserved_bytes,
                                       uint64_t *available_bytes,
                                       uint64_t *reserved_bytes) {
    if (available_bytes) *available_bytes = 0;
    if (reserved_bytes) *reserved_bytes = 0;
    if (!available_bytes || !reserved_bytes || recommended_bytes == 0) {
        return false;
    }

    const uint64_t reserved = saturating_add_u64(runtime_bytes,
                                                  external_reserved_bytes);
    *reserved_bytes = reserved;
    if (reserved >= recommended_bytes) return false;
    *available_bytes = recommended_bytes - reserved;
    return true;
}

bool ds4_ssd_memory_lock_acquire(ds4_ssd_memory_lock *lock,
                                 uint64_t             bytes) {
    if (!lock) return false;
    lock->ptr = NULL;
    lock->bytes = 0;
    if (bytes == 0) return true;
    if (bytes > (uint64_t)SIZE_MAX) {
        fprintf(stderr,
                "ds4: --simulate-used-memory is too large for this process\n");
        return false;
    }

    void *ptr = mmap(NULL,
                     (size_t)bytes,
                     PROT_READ | PROT_WRITE,
                     MAP_PRIVATE | MAP_ANONYMOUS,
                     -1,
                     0);
    if (ptr == MAP_FAILED) {
        fprintf(stderr,
                "ds4: --simulate-used-memory mmap %.2f GiB failed: %s\n",
                (double)bytes / (double)DS4_GIB,
                strerror(errno));
        return false;
    }

    const long page_long = sysconf(_SC_PAGESIZE);
    const uint64_t page = page_long > 0 ? (uint64_t)page_long : 4096ull;
    const uint64_t chunk_bytes = 256ull * 1024ull * 1024ull;
    volatile unsigned char *p = (volatile unsigned char *)ptr;

    /*
     * Touch and lock in bounded chunks.  A single very large mlock() is harder
     * to diagnose when it fails and can create long uninterruptible VM work on
     * macOS; chunking mirrors the standalone diagnostic utility.
     */
    uint64_t locked = 0;
    for (uint64_t off = 0; off < bytes; off += chunk_bytes) {
        uint64_t len = bytes - off;
        if (len > chunk_bytes) len = chunk_bytes;

        for (uint64_t pos = off; pos < off + len; pos += page) {
            p[pos] = (unsigned char)(pos / page);
        }
        if (len != 0) p[off + len - 1u] = 1;

        if (mlock((void *)(p + off), (size_t)len) != 0) {
            fprintf(stderr,
                    "ds4: --simulate-used-memory mlock failed after %.2f/%.2f GiB: %s\n",
                    (double)locked / (double)DS4_GIB,
                    (double)bytes / (double)DS4_GIB,
                    strerror(errno));
            if (locked != 0) munlock(ptr, (size_t)locked);
            munmap(ptr, (size_t)bytes);
            return false;
        }
        locked += len;
    }

    lock->ptr = ptr;
    lock->bytes = bytes;
    fprintf(stderr,
            "ds4: simulated used memory: locked %.2f GiB before model load\n",
            (double)bytes / (double)DS4_GIB);
    return true;
}

void ds4_ssd_memory_lock_release(ds4_ssd_memory_lock *lock) {
    if (!lock || !lock->ptr || lock->bytes == 0) return;
    munlock(lock->ptr, (size_t)lock->bytes);
    munmap(lock->ptr, (size_t)lock->bytes);
    lock->ptr = NULL;
    lock->bytes = 0;
}
