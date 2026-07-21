/* DeepSeek ExpertMajor cache transitions around batched prefill.
 *
 * This is a textual implementation partition of ds4.c.  It deliberately stays
 * in that translation unit so the session path keeps static linkage and the
 * compiler sees the complete cache transition sequence.
 */

#if defined(__APPLE__) && !defined(DS4_NO_GPU)
static bool ds4_session_deepseek_long_prefill_phase_needed(
        const ds4_session *s,
        uint32_t           n_tokens) {
    if (!s || !s->engine ||
        s->engine->model.family != DS4_MODEL_FAMILY_DEEPSEEK4 ||
        !s->engine->model.native_expert_store_v2 ||
        !s->engine->ssd_streaming ||
        s->engine->deepseek_prefill_cache_experts == 0 ||
        s->engine->deepseek_decode_cache_experts <=
            s->engine->deepseek_prefill_cache_experts ||
        getenv("DS4_METAL_DISABLE_DEEPSEEK_PHASE_CACHE") != NULL) {
        return false;
    }
    const uint32_t selected_max =
        metal_graph_stream_prefill_batch_selected_addr_auto_max();
    return selected_max == 0 ? n_tokens >= 32u : n_tokens > selected_max;
}

static DS4_MAYBE_UNUSED int ds4_session_deepseek_prepare_prefill_cache(
        ds4_session *s,
        uint32_t     n_tokens,
        bool        *changed,
        char        *err,
        size_t       errlen) {
    if (changed) *changed = false;
    if (!ds4_session_deepseek_long_prefill_phase_needed(s, n_tokens)) {
        return 0;
    }
    const uint32_t current =
        ds4_gpu_stream_expert_cache_configured_count();
    const uint32_t target = s->engine->deepseek_prefill_cache_experts;
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
                "for %u-token prefill\n",
                current, target, n_tokens);
    }
    return 0;
}

static DS4_MAYBE_UNUSED int ds4_session_deepseek_finish_prefill_cache(
        ds4_session *s,
        bool         changed,
        char        *err,
        size_t       errlen) {
    if (!changed) return 0;
    const uint32_t current =
        ds4_gpu_stream_expert_cache_configured_count();
    const uint32_t target = s->engine->deepseek_decode_cache_experts;
    if (target <= current) return 0;
    if (!ds4_gpu_grow_streaming_expert_cache_budget(target)) {
        if (errlen) {
            snprintf(err, errlen,
                     "failed to restore DeepSeek decode cache after prefill");
        }
        return 1;
    }
    if (getenv("DS4_METAL_MEMORY_REPORT") != NULL) {
        fprintf(stderr,
                "ds4: DeepSeek native cache phase restored %u -> %u experts "
                "for decode\n",
                current, target);
    }
    return 0;
}
#else
static DS4_MAYBE_UNUSED int ds4_session_deepseek_prepare_prefill_cache(
        ds4_session *s,
        uint32_t     n_tokens,
        bool        *changed,
        char        *err,
        size_t       errlen) {
    (void)s;
    (void)n_tokens;
    (void)err;
    (void)errlen;
    if (changed) *changed = false;
    return 0;
}

static DS4_MAYBE_UNUSED int ds4_session_deepseek_finish_prefill_cache(
        ds4_session *s,
        bool         changed,
        char        *err,
        size_t       errlen) {
    (void)s;
    (void)changed;
    (void)err;
    (void)errlen;
    return 0;
}
#endif
