/* DeepSeek ExpertMajor cache transitions around batched prefill.
 *
 * This is a textual implementation partition of ds4.c.  It deliberately stays
 * in that translation unit so the session path keeps static linkage and the
 * compiler sees the complete cache transition sequence.
 */

#if defined(__APPLE__) && !defined(DS4_NO_GPU)
static bool ds4_session_deepseek_phase_cache_enabled(
        const ds4_session *s) {
    if (!s || !s->engine ||
        s->engine->model.family != DS4_MODEL_FAMILY_DEEPSEEK4 ||
        !s->engine->model.native_expert_store_v2 ||
        !s->engine->ssd_streaming ||
        !ds4_ssd_deepseek_phase_cache_targets_enabled(
            s->engine->deepseek_prefill_cache_experts,
            s->engine->deepseek_decode_cache_experts)) {
        return false;
    }
    return true;
}

static int ds4_session_deepseek_prepare_prefill_cache(
        ds4_session *s,
        uint32_t     prefill_tokens,
        uint32_t     resulting_context_tokens,
        bool        *changed,
        char        *err,
        size_t       errlen) {
    if (changed) *changed = false;
    if (!ds4_session_deepseek_phase_cache_enabled(s)) return 0;
    const uint32_t selected_max =
        metal_graph_stream_prefill_batch_selected_addr_auto_max();
    const uint32_t target =
        ds4_ssd_deepseek_prefill_phase_cache_target(
            prefill_tokens,
            resulting_context_tokens,
            selected_max,
            s->engine->deepseek_prefill_cache_experts,
            s->engine->deepseek_long_context_cache_experts,
            s->engine->deepseek_extended_context_cache_experts);
    if (target == 0) return 0;
    const uint32_t current =
        ds4_gpu_stream_expert_cache_configured_count();
    /* Batched native prefill bypasses the decode cache. Keep only its safe
     * floor during that work, then install the long-context hot set once at
     * commit. A tiny suffix that crosses 8K only needs the bounded long tier. */
    if (current == target) return 0;
    if (!ds4_gpu_reconfigure_streaming_expert_cache_budget(target)) {
        if (errlen) {
            snprintf(err, errlen,
                     "failed to shrink DeepSeek cache for long prefill");
        }
        return 1;
    }
    if (changed) *changed = true;
    if (getenv("DS4_METAL_MEMORY_REPORT") != NULL) {
        fprintf(stderr,
                "ds4: DeepSeek native cache phase switched %u -> %u experts "
                "for %u-token prefill (%u-token resulting context)\n",
                current, target, prefill_tokens, resulting_context_tokens);
    }
    return 0;
}

static int ds4_session_deepseek_finish_prefill_cache(
        ds4_session *s,
        uint32_t     resulting_context_tokens,
        bool         changed,
        bool         prefill_succeeded,
        char        *err,
        size_t       errlen) {
    /* An explicit cache budget bypasses AUTO and therefore has no phase
     * endpoints.  Leave that configured budget untouched after prefill. */
    if (!ds4_session_deepseek_phase_cache_enabled(s)) return 0;
    uint32_t current =
        ds4_gpu_stream_expert_cache_configured_count();
    const uint32_t cache_before = current;
    const uint32_t long_target =
        s->engine->deepseek_long_context_cache_experts;
    uint32_t target =
        ds4_ssd_deepseek_post_prefill_cache_target(
            resulting_context_tokens,
            long_target,
            s->engine->deepseek_extended_context_cache_experts,
            s->engine->deepseek_decode_cache_experts);
    ds4_ssd_host_memory memory = {0};
    bool pressure_allows_seed = true;
    const bool cache_changed = changed || target != cache_before;
    if ((target > current || (prefill_succeeded && cache_changed)) &&
        (!ds4_gpu_host_memory_snapshot(&memory) ||
         !memory.pressure_status_available || !memory.pressure_normal)) {
        if (getenv("DS4_METAL_MEMORY_REPORT") != NULL) {
            if (target > current) {
                fprintf(stderr,
                        "ds4: DeepSeek cache growth held at %u experts "
                        "because post-prefill memory pressure is not normal\n",
                        current);
            } else {
                fprintf(stderr,
                        "ds4: DeepSeek hotlist seed skipped because "
                        "post-prefill memory pressure is not normal\n");
            }
        }
        pressure_allows_seed = false;
        if (target > current) target = current;
    }
    if (target > current &&
        !ds4_gpu_grow_streaming_expert_cache_budget(target)) {
        if (errlen) {
            snprintf(err, errlen,
                     "failed to restore DeepSeek decode cache after prefill");
        }
        return 1;
    }
    if (target < current &&
        !ds4_gpu_reconfigure_streaming_expert_cache_budget(target)) {
        if (errlen) {
            snprintf(err, errlen,
                     "failed to reduce DeepSeek decode cache at context tier");
        }
        return 1;
    }
    if (target != current && getenv("DS4_METAL_MEMORY_REPORT") != NULL) {
        fprintf(stderr,
                "ds4: DeepSeek native cache phase set %u -> %u experts "
                "for decode at %u-token context%s\n",
                current, target, resulting_context_tokens,
                changed ? "" : " (short-context recovery)");
    }
    current = ds4_gpu_stream_expert_cache_configured_count();
    if (ds4_ssd_deepseek_post_prefill_seed_allowed(
            prefill_succeeded, pressure_allows_seed,
            cache_changed, target, current) &&
        !metal_graph_seed_streaming_expert_cache_from_hotlist(
            &s->graph, &s->engine->model, &s->engine->weights)) {
        if (errlen) {
            snprintf(err, errlen,
                     "failed to seed DeepSeek long-context decode cache");
        }
        return 1;
    }
    return 0;
}

/* Keep cache policy attached to committed context, including one-token resume
 * and normal decode. This closes the 8K/65K frontier without requiring a
 * batched prefill transition. Reconfiguration drains queued Metal work. */
static int ds4_session_deepseek_apply_context_cache_tier(
        ds4_session *s,
        uint32_t     resulting_context_tokens,
        char        *err,
        size_t       errlen) {
    if (!ds4_session_deepseek_phase_cache_enabled(s)) return 0;
    const uint32_t target =
        ds4_ssd_deepseek_post_prefill_cache_target(
            resulting_context_tokens,
            s->engine->deepseek_long_context_cache_experts,
            s->engine->deepseek_extended_context_cache_experts,
            s->engine->deepseek_decode_cache_experts);
    if (target == 0 ||
        target == ds4_gpu_stream_expert_cache_configured_count()) {
        return 0;
    }
    return ds4_session_deepseek_finish_prefill_cache(
        s, resulting_context_tokens, true, true, err, errlen);
}
#endif
