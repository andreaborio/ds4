/* Textual GLM graph implementation partition owned by ds4.c.
 * Keep it in the same translation unit and under the caller's DS4_NO_GPU guard. */

typedef struct {
    uint32_t ctx_size;
    uint32_t ctx_cap;
    uint32_t normal_layers;
    uint32_t layer_start;
    uint32_t layer_end;
    uint32_t layer_count;
    uint64_t q_dim;
    uint64_t q_nope;
    uint64_t heads_dim;
    uint64_t kv_raw_dim;
    uint64_t dense_hidden_max;
    uint64_t ffn_mid_elems;

    ds4_gpu_tensor *cur;
    ds4_gpu_tensor *next;
    ds4_gpu_tensor *attn_norm;
    ds4_gpu_tensor *q_rank;
    ds4_gpu_tensor *q_rank_norm;
    ds4_gpu_tensor *q;
    ds4_gpu_tensor *kv_raw;
    ds4_gpu_tensor *kv_norm;
    ds4_gpu_tensor *k_nope;
    ds4_gpu_tensor *value;
    ds4_gpu_tensor *heads;
    ds4_gpu_tensor *attn_out;
    ds4_gpu_tensor *after_attn;
    ds4_gpu_tensor *ffn_norm;
    ds4_gpu_tensor *ffn_gate;
    ds4_gpu_tensor *ffn_up;
    ds4_gpu_tensor *ffn_mid;
    ds4_gpu_tensor *routed_gate;
    ds4_gpu_tensor *routed_up;
    ds4_gpu_tensor *routed_down;
    ds4_gpu_tensor *ffn_out;
    ds4_gpu_tensor *ffn_sum;
    ds4_gpu_tensor *router_logits;
    ds4_gpu_tensor *router_probs;
    ds4_gpu_tensor *router_selected;
    ds4_gpu_tensor *router_weights;
    ds4_gpu_tensor *output_norm;
    ds4_gpu_tensor *logits;
    ds4_gpu_tensor *batch_router_logits;
    ds4_gpu_tensor *batch_router_probs;
    ds4_gpu_tensor *batch_router_selected;
    ds4_gpu_tensor *batch_router_weights;

    ds4_gpu_tensor *prefill_tokens;
    ds4_gpu_tensor *batch_cur;
    ds4_gpu_tensor *batch_next;
    ds4_gpu_tensor *batch_attn_norm;
    ds4_gpu_tensor *batch_q_rank;
    ds4_gpu_tensor *batch_q_rank_norm;
    ds4_gpu_tensor *batch_q;
    ds4_gpu_tensor *batch_kv_raw;
    ds4_gpu_tensor *batch_kv_norm;
    ds4_gpu_tensor *batch_k_nope;
    ds4_gpu_tensor *batch_value;
    ds4_gpu_tensor *batch_heads;
    ds4_gpu_tensor *batch_attn_out;
    ds4_gpu_tensor *batch_after_attn;
    ds4_gpu_tensor *batch_ffn_norm;
    ds4_gpu_tensor *batch_ffn_gate;
    ds4_gpu_tensor *batch_ffn_up;
    ds4_gpu_tensor *batch_shared_mid;
    ds4_gpu_tensor *batch_ffn_mid;
    ds4_gpu_tensor *batch_routed_gate;
    ds4_gpu_tensor *batch_routed_up;
    ds4_gpu_tensor *batch_routed_down;
    ds4_gpu_tensor *batch_ffn_out;
    bool batch_routed_mid_is_f16;

    uint32_t compact_cache_cap;
    uint32_t indexed_prefill_cap;
    uint32_t indexed_prefill_score_cap;
    uint32_t indexer_full_layers;
    ds4_gpu_tensor *indexer_k;
    ds4_gpu_tensor *indexer_q;
    ds4_gpu_tensor *indexer_weights;
    ds4_gpu_tensor *indexer_scores;
    ds4_gpu_tensor *indexer_selected;
    ds4_gpu_tensor *qk_low;
    ds4_gpu_tensor *attn_partial_lora;
    ds4_gpu_tensor *attn_partial_ms;
    ds4_gpu_tensor *batch_indexer_k;
    ds4_gpu_tensor *batch_indexer_q;
    ds4_gpu_tensor *batch_indexer_weights;
    ds4_gpu_tensor *batch_indexer_scores;
    ds4_gpu_tensor *batch_indexer_selected;
    ds4_gpu_tensor *batch_qk_low;
    ds4_gpu_tensor *batch_attn_lora;
    ds4_gpu_tensor *layer_kv_lora_cache[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_k_rope_cache[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_indexer_key_cache[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_key_cache[DS4_MAX_LAYER];
    ds4_gpu_tensor *layer_value_cache[DS4_MAX_LAYER];
    bool full_kv_cache;
    bool has_token_embd;
    bool has_output_head;
    bool quality;
    bool ssd_streaming;
    bool ssd_streaming_cold;
    bool glm_indexed_prefill_prepare;
    ds4_glm_router_ahead_mode glm_router_ahead;
    uint32_t glm_router_ahead_lookahead;
    bool generic_routed_moe;
    bool streaming_static_decode_map_current;
} ds4_glm_gpu_graph;

static uint32_t glm_graph_model_context_limit(void) {
    if (DS4_ROPE_ORIG_CTX > UINT32_MAX) return UINT32_MAX;
    return (uint32_t)DS4_ROPE_ORIG_CTX;
}

static double glm_graph_bytes_to_gib(uint64_t bytes) {
    return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

static uint64_t glm_graph_saturating_add_u64(uint64_t a, uint64_t b) {
    return a > UINT64_MAX - b ? UINT64_MAX : a + b;
}

static bool glm_graph_env_disabled(const char *name) {
    const char *env = getenv(name);
    if (!env || !env[0]) return false;
    return strcmp(env, "0") == 0 ||
           strcasecmp(env, "false") == 0 ||
           strcasecmp(env, "off") == 0 ||
           strcasecmp(env, "no") == 0;
}

static double glm_graph_env_double(
        const char *name,
        double      fallback,
        double      min_value,
        double      max_value) {
    const char *env = getenv(name);
    if (!env || !env[0]) return fallback;
    char *end = NULL;
    errno = 0;
    const double v = strtod(env, &end);
    if (end == env || errno != 0 || !isfinite(v)) return fallback;
    if (v < min_value) return min_value;
    if (v > max_value) return max_value;
    return v;
}

static uint64_t glm_graph_host_memory_bytes(void) {
#if defined(__APPLE__)
    uint64_t mem = 0;
    size_t len = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &len, NULL, 0) != 0) return 0;
    return mem;
#else
    return 0;
#endif
}

static uint64_t glm_graph_streaming_active_model_bytes(
        const ds4_weights *weights) {
    if (!weights) return 0;

    uint64_t max_bytes = 0;
    ds4_model_map_span_vec spans;

    if (weights_model_map_token_spans(weights, &spans)) {
        max_bytes = model_map_span_vec_total_bytes(&spans);
        free(spans.v);
    }
    if (weights_model_map_output_spans(weights, &spans)) {
        const uint64_t bytes = model_map_span_vec_total_bytes(&spans);
        if (bytes > max_bytes) max_bytes = bytes;
        free(spans.v);
    }
    for (uint32_t il = 0; il < DS4_N_LAYER; il++) {
        if (!weights_model_map_spans(weights, il, il, false, &spans)) {
            continue;
        }
        const uint64_t bytes = model_map_span_vec_total_bytes(&spans);
        if (bytes > max_bytes) max_bytes = bytes;
        free(spans.v);
    }

    return max_bytes;
}

static uint64_t glm_graph_model_bytes_for_guard(
        const ds4_model   *model,
        const ds4_weights *weights,
        bool               ssd_streaming) {
    if (!model) return 0;
    if (!ssd_streaming) return model->size;
    const uint64_t active_bytes = glm_graph_streaming_active_model_bytes(weights);
    return active_bytes != 0 ? active_bytes : model->size;
}

static double glm_graph_memory_guard_default_reserve_gib(
        uint64_t budget_base,
        uint64_t model_bytes) {
    const double base_gib = glm_graph_bytes_to_gib(budget_base);
    const double model_gib = glm_graph_bytes_to_gib(model_bytes);
    if (base_gib >= 480.0 &&
        base_gib <= 640.0 &&
        model_gib >= base_gib * 0.80) {
        return 24.0;
    }
    return 32.0;
}

static bool glm_graph_memory_guard_for_compact_cap(
        const ds4_model   *model,
        const ds4_weights *weights,
        bool               ssd_streaming,
        uint32_t           ctx_size,
        uint32_t           compact_cap,
        uint64_t           transient_extra_bytes,
        const char        *phase) {
    if (!model || glm_graph_env_disabled("DS4_GLM_MEMORY_GUARD")) return true;

    const uint64_t host_bytes = glm_graph_host_memory_bytes();
    uint64_t budget_base = host_bytes;
    if (budget_base == 0) {
        budget_base = ds4_gpu_recommended_working_set_size();
    }
    if (budget_base == 0) return true;

    const uint32_t work_ctx =
        glm_graph_full_attention_cap(ctx_size, ssd_streaming);
    const ds4_context_memory mem =
        glm_graph_context_memory_estimate_for_compact_cap(ctx_size,
                                                          work_ctx,
                                                          compact_cap,
                                                          ssd_streaming);
    const uint64_t graph_bytes = mem.total_bytes;
    const uint64_t model_bytes =
        glm_graph_model_bytes_for_guard(model,
                                        weights,
                                        ssd_streaming);
    uint64_t required = glm_graph_saturating_add_u64(model_bytes, graph_bytes);
    required = glm_graph_saturating_add_u64(required, transient_extra_bytes);

    const double fraction =
        glm_graph_env_double("DS4_GLM_MEMORY_GUARD_FRACTION", 0.99, 0.50, 1.00);
    const double default_reserve_gib =
        glm_graph_memory_guard_default_reserve_gib(budget_base, model_bytes);
    const double reserve_gib =
        glm_graph_env_double("DS4_GLM_MEMORY_GUARD_RESERVE_GB",
                             default_reserve_gib,
                             0.0,
                             1024.0);
    const uint64_t fraction_budget = (uint64_t)((double)budget_base * fraction);
    const uint64_t reserve_bytes =
        (uint64_t)(reserve_gib * 1024.0 * 1024.0 * 1024.0);
    const uint64_t reserve_budget =
        reserve_bytes >= budget_base ? 0 : budget_base - reserve_bytes;
    uint64_t budget = fraction_budget;
    if (reserve_bytes != 0 && reserve_budget < budget) budget = reserve_budget;

    if (required <= budget) {
        const char *report = getenv("DS4_GLM_MEMORY_GUARD_REPORT");
        if (report && report[0]) {
            fprintf(stderr,
                    "ds4: GLM memory guard ctx=%u compact_cap=%u required=%.2f GiB "
                    "budget=%.2f GiB (model %.2f GiB, graph %.2f GiB, transient %.2f GiB)\n",
                    ctx_size,
                    mem.comp_cap,
                    glm_graph_bytes_to_gib(required),
                    glm_graph_bytes_to_gib(budget),
                    glm_graph_bytes_to_gib(model_bytes),
                    glm_graph_bytes_to_gib(graph_bytes),
                    glm_graph_bytes_to_gib(transient_extra_bytes));
            if (ssd_streaming && model_bytes != model->size) {
                fprintf(stderr,
                        "ds4:   GLM streaming guard uses active model span %.2f GiB "
                        "(full GGUF %.2f GiB)\n",
                        glm_graph_bytes_to_gib(model_bytes),
                        glm_graph_bytes_to_gib(model->size));
            }
        }
        return true;
    }

    fprintf(stderr,
            "ds4: GLM memory guard refused ctx=%u compact_cap=%u %s\n",
            ctx_size,
            mem.comp_cap,
            phase ? phase : "before Metal graph allocation");
    if (ssd_streaming && model_bytes != model->size) {
        fprintf(stderr,
                "ds4:   streamed active model map: %.2f GiB "
                "(full GGUF %.2f GiB)\n",
                glm_graph_bytes_to_gib(model_bytes),
                glm_graph_bytes_to_gib(model->size));
    } else {
        fprintf(stderr,
                "ds4:   model map: %.2f GiB\n",
                glm_graph_bytes_to_gib(model_bytes));
    }
    fprintf(stderr,
            "ds4:   graph cache/scratch: %.2f GiB "
            "(full KV %.2f GiB, compact DSA %.2f GiB, scratch %.2f GiB)\n",
            glm_graph_bytes_to_gib(graph_bytes),
            glm_graph_bytes_to_gib(mem.raw_bytes),
            glm_graph_bytes_to_gib(mem.compressed_bytes),
            glm_graph_bytes_to_gib(mem.scratch_bytes));
    fprintf(stderr,
            "ds4:   required model+graph: %.2f GiB; guard budget: %.2f GiB "
            "(base %.2f GiB, fraction %.2f, reserve %.2f GiB, transient %.2f GiB)\n",
            glm_graph_bytes_to_gib(required),
            glm_graph_bytes_to_gib(budget),
            glm_graph_bytes_to_gib(budget_base),
            fraction,
            reserve_gib,
            glm_graph_bytes_to_gib(transient_extra_bytes));
    fprintf(stderr,
            "ds4:   set DS4_GLM_MEMORY_GUARD=0 to bypass, use a smaller --ctx, "
            "or use SSD streaming\n");
    return false;
}

static bool glm_graph_memory_guard(
        const ds4_model   *model,
        const ds4_weights *weights,
        bool               ssd_streaming,
        uint32_t           ctx_size) {
    const uint32_t work_ctx =
        glm_graph_full_attention_cap(ctx_size, ssd_streaming);
    const uint32_t compact_cap =
        glm_graph_compact_cache_initial_cap(ctx_size, work_ctx);
    return glm_graph_memory_guard_for_compact_cap(
            model,
            weights,
            ssd_streaming,
            ctx_size,
            compact_cap,
            0,
            "before Metal graph allocation");
}

static uint32_t glm_graph_full_attention_cap(uint32_t ctx_size,
                                             bool     ssd_streaming) {
    uint32_t cap = ssd_streaming ?
        DS4_GLM_METAL_STREAMING_FULL_ATTN_CONTEXT :
        DS4_GLM_METAL_FULL_ATTN_DEFAULT_CONTEXT;
    if (ctx_size >= DS4_GLM_METAL_LONG_CONTEXT_THRESHOLD &&
        cap > DS4_GLM_METAL_LONG_CONTEXT_FULL_ATTN_CONTEXT) {
        cap = DS4_GLM_METAL_LONG_CONTEXT_FULL_ATTN_CONTEXT;
    }
    if (ctx_size > 0 && cap > ctx_size) cap = ctx_size;
    if (cap == 0) cap = 1;
    return cap;
}

static uint32_t glm_graph_full_prefill_layer_flush_interval(
        uint32_t n_tokens,
        uint32_t command_rows,
        bool     logits_requested) {
    return (n_tokens > DS4_GLM_METAL_FULL_ATTN_LAYER_FLUSH_CONTEXT ||
            command_rows > DS4_GLM_METAL_FULL_ATTN_LAYER_FLUSH_CONTEXT ||
            logits_requested) ? 1u : 0u;
}

static uint32_t glm_graph_prefill_progress_flush_interval(
        uint32_t                layer_flush_interval,
        uint32_t                n_tokens,
        ds4_session_progress_fn display_progress,
        uint32_t                work_total) {
    if (layer_flush_interval != 0) return layer_flush_interval;
    (void)n_tokens;
    (void)display_progress;
    (void)work_total;
    return 0;
}

static void glm_graph_report_prefill_display_progress(
        ds4_session_progress_fn display_progress,
        void                   *display_progress_ud,
        uint32_t                absolute_base,
        uint32_t                work_done_base,
        uint32_t                n_tokens,
        uint32_t                layer_done,
        uint32_t                normal_layers,
        uint32_t                work_total,
        bool                    allow_complete) {
    if (!display_progress || work_total == 0) return;

    uint64_t chunk_done = 0;
    if (normal_layers == 0 || layer_done >= normal_layers) {
        chunk_done = n_tokens;
    } else {
        chunk_done = (uint64_t)n_tokens * (uint64_t)layer_done /
                     (uint64_t)normal_layers;
    }

    uint64_t done = (uint64_t)work_done_base + chunk_done;
    if (done > (uint64_t)work_total) done = work_total;
    if (!allow_complete && done >= (uint64_t)work_total) {
        done = work_total > 0 ? (uint64_t)work_total - 1u : 0u;
    }
    display_progress(display_progress_ud,
                     "prefill_display",
                     (int)((uint64_t)absolute_base + done),
                     (int)((uint64_t)absolute_base + (uint64_t)work_total));
}

static bool glm_graph_small_prefill_stage_sync(
        uint32_t n_tokens,
        bool     logits_requested) {
    return logits_requested &&
           n_tokens > 0 &&
           n_tokens <= DS4_GLM_METAL_SMALL_PREFILL_STAGE_SYNC_TOKENS;
}

static uint32_t glm_graph_indexed_decode_split_min_block_rows(void) {
    return 32u;
}

static uint32_t glm_graph_indexed_decode_split_blocks(void) {
    const uint32_t block_rows = glm_graph_indexed_decode_split_min_block_rows();
    const uint32_t top_k = glm_graph_indexer_top_k_limit();
    return (top_k + block_rows - 1u) / block_rows;
}

static uint32_t glm_graph_indexed_decode_split_block_rows_for(uint32_t n_selected) {
    return n_selected <= 1024u ? 32u : 128u;
}

static bool glm_graph_indexed_decode_split_group8_available(uint32_t n_selected) {
    const uint32_t block_rows = glm_graph_indexed_decode_split_block_rows_for(n_selected);
    const uint32_t needed_blocks =
        block_rows != 0u ? (n_selected + block_rows - 1u) / block_rows : 0u;
    return n_selected > 512u &&
           block_rows > 0 &&
           needed_blocks > 0 &&
           needed_blocks <= glm_graph_indexed_decode_split_blocks() &&
           glm_graph_indexed_decode_split_blocks() <= 64u &&
           (DS4_N_HEAD % 8u) == 0 &&
           DS4_N_KV_LORA == 512u &&
           DS4_N_ROT == 64u &&
           glm_graph_compact_cache_is_f16();
}

static bool glm_graph_prefill_stage_sync_boundary(void) {
    if (ds4_gpu_end_commands() == 0) return false;
    return ds4_gpu_begin_commands() != 0;
}

static bool glm_graph_indexed_prefill_batch_available(
        const ds4_glm_gpu_graph *g) {
    return g &&
           g->compact_cache_cap != 0 &&
           g->indexed_prefill_cap != 0 &&
           g->indexed_prefill_score_cap != 0 &&
           g->batch_indexer_q &&
           g->batch_indexer_weights &&
           g->batch_indexer_scores &&
           g->batch_indexer_selected &&
           g->batch_qk_low &&
           g->batch_attn_lora;
}

static bool glm_graph_indexed_prefill_batch_ready(
        const ds4_glm_gpu_graph *g,
        uint32_t                 pos) {
    return glm_graph_indexed_prefill_batch_available(g) &&
           (!g->full_kv_cache || pos >= g->ctx_cap);
}

static uint32_t glm_graph_limit_indexed_prefill_chunk(
        uint32_t pos,
        uint32_t chunk) {
    const uint32_t top_k = glm_graph_indexer_top_k_limit();
    if (pos < top_k) {
        const uint32_t bridge = top_k - pos;
        if (bridge != 0 && chunk > bridge) chunk = bridge;
    }
    return chunk;
}

static uint32_t glm_graph_indexed_prefill_chunk_tokens(
        uint32_t full_attention_cap,
        uint32_t compact_cap) {
    (void)full_attention_cap;
    uint32_t chunk = DS4_GLM_METAL_INDEXED_PREFILL_CHUNK_TOKENS;
    if (compact_cap > 0 && chunk > compact_cap) chunk = compact_cap;
    if (chunk == 0) chunk = 1;
    return chunk;
}

static uint32_t glm_graph_indexed_prefill_score_tokens(
        uint32_t indexed_prefill_cap,
        uint32_t compact_cap) {
    if (indexed_prefill_cap == 0 || compact_cap == 0) return 0;
    const uint32_t scratch_mb = DS4_GLM_METAL_INDEXED_PREFILL_SCORE_SCRATCH_MB;
    const uint64_t budget_bytes = (uint64_t)scratch_mb * 1024ull * 1024ull;
    uint64_t budget_rows = budget_bytes / ((uint64_t)compact_cap * sizeof(float));
    if (budget_rows == 0) budget_rows = 1;
    if (budget_rows > indexed_prefill_cap) budget_rows = indexed_prefill_cap;
    if (budget_rows > UINT32_MAX) budget_rows = UINT32_MAX;
    return (uint32_t)budget_rows;
}

static bool glm_graph_context_request(int ctx_size, uint32_t *ctx_out) {
    if (!ctx_out || ctx_size <= 0) return false;
    const uint32_t model_ctx = glm_graph_model_context_limit();
    if ((uint64_t)(uint32_t)ctx_size > (uint64_t)model_ctx) {
        fprintf(stderr,
                "ds4: GLM context %d exceeds model context %u\n",
                ctx_size,
                model_ctx);
        return false;
    }
    *ctx_out = (uint32_t)ctx_size;
    return true;
}

static bool glm_graph_span_fits_context(
        const ds4_glm_gpu_graph *g,
        uint32_t                 pos0,
        uint32_t                 n_tokens) {
    return g && n_tokens > 0 && pos0 < g->ctx_size && n_tokens <= g->ctx_size - pos0;
}

static bool glm_graph_span_fits_full_attention(
        const ds4_glm_gpu_graph *g,
        uint32_t                 pos0,
        uint32_t                 n_tokens) {
    return g && n_tokens > 0 && pos0 < g->ctx_cap && n_tokens <= g->ctx_cap - pos0;
}

static void glm_graph_log_full_attention_limit(
        const ds4_glm_gpu_graph *g,
        uint32_t                 pos0,
        uint32_t                 n_tokens) {
    const uint32_t end = pos0 + n_tokens;
    fprintf(stderr,
            "ds4: GLM Metal full-attention work cap is %u tokens; "
            "requested span [%u,%u) in ctx %u needs compact indexed attention\n",
            g ? g->ctx_cap : 0,
            pos0,
            end,
            g ? g->ctx_size : 0);
}

static bool glm_graph_tensor_layout(
        const ds4_tensor *t,
        uint32_t          type,
        uint32_t          ndim,
        uint64_t          dim0,
        uint64_t          dim1,
        uint64_t          dim2) {
    if (!t || t->type != type || t->ndim != ndim) return false;
    if (ndim > 0 && t->dim[0] != dim0) return false;
    if (ndim > 1 && t->dim[1] != dim1) return false;
    if (ndim > 2 && t->dim[2] != dim2) return false;
    return true;
}

static bool glm_graph_layer_uses_generic_routed_moe(
        const ds4_layer_weights *l) {
    return l &&
           l->ffn_gate_exps &&
           l->ffn_up_exps &&
           l->ffn_down_exps &&
           l->ffn_gate_exps->type == DS4_TENSOR_IQ2_XXS;
}

static bool glm_graph_stream_map_token(
        ds4_glm_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights) {
    if (!g || !g->ssd_streaming) return true;
    g->streaming_static_decode_map_current = false;
    return metal_graph_stream_map_token(model, weights);
}

static bool glm_graph_stream_layer_expert_cache_supported(
        const ds4_weights       *weights,
        const ds4_layer_weights *l,
        uint32_t                 il) {
    if (!weights || !l) return false;
    if (il < DS4_N_LEADING_DENSE) return true;
    return glm_stream_decode_experts_are_streamed(weights, l, il);
}

static bool glm_graph_stream_prefill_expert_addr_supported(
        const ds4_weights       *weights,
        const ds4_layer_weights *l,
        uint32_t                 il,
        uint32_t                 n_tokens) {
    if (il < DS4_N_LEADING_DENSE) return true;
    if (n_tokens <= 1) return false;
    return glm_stream_expert_cache_addr_supported(weights, l, il);
}

static bool glm_graph_stream_map_decode_layer(
        ds4_glm_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        uint32_t           il) {
    if (!g || !g->ssd_streaming) return true;
    g->streaming_static_decode_map_current = false;
    if (getenv("DS4_GLM_STREAMING_DECODE_FULL_LAYER_MAP") != NULL) {
        return metal_graph_stream_map_layer(model, weights, il);
    }
    if (weights && il < DS4_N_LAYER &&
        glm_graph_stream_layer_expert_cache_supported(weights,
                                                      &weights->layer[il],
                                                      il)) {
        return metal_graph_stream_map_layer_decode(model, weights, il);
    }
    return metal_graph_stream_map_layer(model, weights, il);
}

static bool glm_graph_stream_map_prefill_layer(
        ds4_glm_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        uint32_t           il,
        uint32_t           n_tokens,
        bool               full_layer_prefill) {
    if (!g || !g->ssd_streaming) return true;
    g->streaming_static_decode_map_current = false;
    if (full_layer_prefill) {
        const char *trace = getenv("DS4_METAL_STREAMING_MAP_TRACE");
        if (trace && trace[0] && strcmp(trace, "0") != 0) {
            fprintf(stderr,
                    "ds4: GLM SSD prefill map layer=%u tokens=%u mode=full-prefill\n",
                    il,
                    n_tokens);
        }
        return metal_graph_stream_map_layer(model, weights, il);
    }
    const bool addr_supported =
        weights && il < DS4_N_LAYER &&
        glm_graph_stream_prefill_expert_addr_supported(weights,
                                                       &weights->layer[il],
                                                       il,
                                                       n_tokens);
    const char *trace = getenv("DS4_METAL_STREAMING_MAP_TRACE");
    if (trace && trace[0] && strcmp(trace, "0") != 0) {
        fprintf(stderr,
                "ds4: GLM SSD prefill map layer=%u tokens=%u mode=%s\n",
                il,
                n_tokens,
                addr_supported ? "decode-expert-cache" : "full-layer");
    }
    if (addr_supported) {
        return metal_graph_stream_map_layer_decode(model, weights, il);
    }
    return metal_graph_stream_map_layer(model, weights, il);
}

enum { DS4_GLM_STREAM_PREFILL_FULL_LAYER_MIN_TOKENS = 64 };

static uint32_t glm_graph_stream_prefill_full_layer_min_tokens(void) {
    const char *env =
        getenv("DS4_METAL_GLM_STREAMING_PREFILL_FULL_LAYER_MIN_TOKENS");
    if (!env || !env[0]) return DS4_GLM_STREAM_PREFILL_FULL_LAYER_MIN_TOKENS;
    char *end = NULL;
    errno = 0;
    unsigned long v = strtoul(env, &end, 10);
    if (end == env || errno != 0 || v == 0 || v > UINT32_MAX) {
        return DS4_GLM_STREAM_PREFILL_FULL_LAYER_MIN_TOKENS;
    }
    return (uint32_t)v;
}

static bool glm_graph_stream_prefill_full_layer_enabled(
        const ds4_glm_gpu_graph *g,
        uint32_t                 n_tokens) {
    if (!g || !g->ssd_streaming) return false;
    if (getenv("DS4_METAL_DISABLE_GLM_STREAMING_PREFILL_FULL_LAYER") != NULL) {
        return false;
    }
    if (getenv("DS4_METAL_GLM_STREAMING_PREFILL_FULL_LAYER") != NULL) {
        return true;
    }
    return n_tokens >= glm_graph_stream_prefill_full_layer_min_tokens();
}

static bool glm_graph_stream_prefill_full_layer_prepare_enabled(
        const ds4_glm_gpu_graph *g,
        bool                     full_layer_prefill) {
    return g &&
           g->ssd_streaming &&
           full_layer_prefill &&
           getenv("DS4_METAL_DISABLE_GLM_STREAMING_PREFILL_FULL_LAYER_PREPARE") == NULL;
}

static bool glm_graph_stream_map_output(
        ds4_glm_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights) {
    if (!g || !g->ssd_streaming) return true;
    g->streaming_static_decode_map_current = false;
    return metal_graph_stream_map_output(model, weights);
}

static bool glm_graph_validate_expert_layout(
        const ds4_model  *model,
        const ds4_tensor *gate,
        const ds4_tensor *up,
        const ds4_tensor *down,
        uint64_t         *gate_row_bytes,
        uint64_t         *up_row_bytes,
        uint64_t         *down_row_bytes) {
    if (!gate || !up || !down) return false;
    if (!glm_graph_gate_pair_type_supported(gate->type, up->type) ||
        !glm_graph_down_type_supported(down->type) ||
        gate->ndim != 3 || up->ndim != 3 || down->ndim != 3 ||
        gate->dim[0] != DS4_N_EMBD ||
        gate->dim[1] != DS4_N_FF_EXP ||
        gate->dim[2] != DS4_N_EXPERT ||
        up->dim[0] != DS4_N_EMBD ||
        up->dim[1] != DS4_N_FF_EXP ||
        up->dim[2] != DS4_N_EXPERT ||
        down->dim[0] != DS4_N_FF_EXP ||
        down->dim[1] != DS4_N_EMBD ||
        down->dim[2] != DS4_N_EXPERT) {
        return false;
    }

    uint64_t gate_in = 0, gate_out = 0;
    uint64_t up_in = 0, up_out = 0;
    uint64_t down_in = 0, down_out = 0;
    (void)tensor_expert_bytes(model, gate, 0, &gate_in, &gate_out, gate_row_bytes);
    (void)tensor_expert_bytes(model, up, 0, &up_in, &up_out, up_row_bytes);
    (void)tensor_expert_bytes(model, down, 0, &down_in, &down_out, down_row_bytes);
    return gate_in == DS4_N_EMBD &&
           up_in == DS4_N_EMBD &&
           down_in == DS4_N_FF_EXP &&
           gate_out == DS4_N_FF_EXP &&
           up_out == DS4_N_FF_EXP &&
           down_out == DS4_N_EMBD;
}

static bool glm_graph_validate_layer_layout(
        const ds4_model         *model,
        const ds4_layer_weights *l,
        uint32_t                 il,
        uint64_t                 q_dim,
        uint64_t                 q_nope,
        uint64_t                 heads_dim,
        uint64_t                *kv_raw_dim_out,
        uint64_t                *dense_hidden_max) {
    if (!l) return false;
    const uint64_t kv_raw_dim = l->attn_kv_a_mqa ? l->attn_kv_a_mqa->dim[1] : 0;
    const uint64_t indexer_q_dim = (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM;
    if (!glm_graph_tensor_layout(l->attn_norm, DS4_TENSOR_F32, 1, DS4_N_EMBD, 0, 0) ||
        !glm_graph_tensor_layout(l->attn_q_a, DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_LORA_Q, 0) ||
        !glm_graph_tensor_layout(l->attn_q_a_norm, DS4_TENSOR_F32, 1, DS4_N_LORA_Q, 0, 0) ||
        !glm_graph_tensor_layout(l->attn_q_b, DS4_TENSOR_Q8_0, 2, DS4_N_LORA_Q, q_dim, 0) ||
        !l->attn_kv_a_mqa ||
        l->attn_kv_a_mqa->type != DS4_TENSOR_Q8_0 ||
        l->attn_kv_a_mqa->ndim != 2 ||
        l->attn_kv_a_mqa->dim[0] != DS4_N_EMBD ||
        kv_raw_dim < (uint64_t)DS4_N_KV_LORA + DS4_N_ROT ||
        !glm_graph_tensor_layout(l->attn_kv_a_norm, DS4_TENSOR_F32, 1, DS4_N_KV_LORA, 0, 0) ||
        !glm_graph_tensor_layout(l->attn_k_b, DS4_TENSOR_Q8_0, 3, q_nope, DS4_N_KV_LORA, DS4_N_HEAD) ||
        !glm_graph_tensor_layout(l->attn_v_b, DS4_TENSOR_Q8_0, 3, DS4_N_KV_LORA, DS4_N_VALUE_MLA, DS4_N_HEAD) ||
        !glm_graph_tensor_layout(l->attn_output, DS4_TENSOR_Q8_0, 2, heads_dim, DS4_N_EMBD, 0) ||
        !glm_graph_tensor_layout(l->ffn_norm, DS4_TENSOR_F32, 1, DS4_N_EMBD, 0, 0)) {
        fprintf(stderr, "ds4: GLM Metal graph found unexpected attention layout in layer %u\n", il);
        return false;
    }
    if (glm_dsa_layer_has_indexer(il) &&
        (!glm_graph_tensor_layout(l->indexer_attn_q_b, DS4_TENSOR_Q8_0, 2, DS4_N_LORA_Q, indexer_q_dim, 0) ||
         !glm_graph_tensor_layout(l->indexer_attn_k, DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_INDEXER_HEAD_DIM, 0) ||
         !glm_graph_tensor_layout(l->indexer_k_norm, DS4_TENSOR_F32, 1, DS4_N_INDEXER_HEAD_DIM, 0, 0) ||
         !glm_graph_tensor_layout(l->indexer_k_norm_b, DS4_TENSOR_F32, 1, DS4_N_INDEXER_HEAD_DIM, 0, 0) ||
         !glm_graph_tensor_layout(l->indexer_proj, DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_INDEXER_HEAD, 0))) {
        fprintf(stderr, "ds4: GLM Metal graph found unexpected indexer layout in layer %u\n", il);
        return false;
    }
    if (kv_raw_dim > *kv_raw_dim_out) *kv_raw_dim_out = kv_raw_dim;

    if (il < DS4_N_LEADING_DENSE) {
        const uint64_t hidden = l->ffn_gate ? l->ffn_gate->dim[1] : 0;
        if (!l->ffn_gate ||
            !glm_graph_tensor_layout(l->ffn_gate, DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, hidden, 0) ||
            !glm_graph_tensor_layout(l->ffn_up, DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, hidden, 0) ||
            !glm_graph_tensor_layout(l->ffn_down, DS4_TENSOR_Q8_0, 2, hidden, DS4_N_EMBD, 0)) {
            fprintf(stderr, "ds4: GLM Metal graph found unexpected dense FFN layout in layer %u\n", il);
            return false;
        }
        if (hidden > *dense_hidden_max) *dense_hidden_max = hidden;
    } else {
        uint64_t gate_row_bytes = 0, up_row_bytes = 0, down_row_bytes = 0;
        if (!glm_graph_tensor_layout(l->ffn_gate_inp, DS4_TENSOR_F16, 2, DS4_N_EMBD, DS4_N_EXPERT, 0) ||
            !glm_graph_tensor_layout(l->ffn_exp_probs_b, DS4_TENSOR_F32, 1, DS4_N_EXPERT, 0, 0) ||
            !glm_graph_validate_expert_layout(model,
                                              l->ffn_gate_exps,
                                              l->ffn_up_exps,
                                              l->ffn_down_exps,
                                              &gate_row_bytes,
                                              &up_row_bytes,
                                              &down_row_bytes) ||
            !glm_graph_tensor_layout(l->ffn_gate_shexp, DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_FF_EXP, 0) ||
            !glm_graph_tensor_layout(l->ffn_up_shexp, DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_FF_EXP, 0) ||
            !glm_graph_tensor_layout(l->ffn_down_shexp, DS4_TENSOR_Q8_0, 2, DS4_N_FF_EXP, DS4_N_EMBD, 0)) {
            fprintf(stderr, "ds4: GLM Metal graph found unexpected sparse FFN layout in layer %u\n", il);
            return false;
        }
        (void)gate_row_bytes;
        (void)up_row_bytes;
        (void)down_row_bytes;
    }
    return true;
}

static bool glm_graph_token_embedding_layout(const ds4_tensor *t) {
    return glm_graph_tensor_layout(t, DS4_TENSOR_Q8_0, 2, DS4_N_EMBD, DS4_N_VOCAB, 0) ||
           glm_graph_tensor_layout(t, DS4_TENSOR_F16,  2, DS4_N_EMBD, DS4_N_VOCAB, 0);
}

static int glm_graph_embed_tokens_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *tokens,
        const ds4_model      *model,
        const ds4_tensor     *token_embd,
        uint32_t              n_tokens) {
    if (!out || !tokens || !model || !token_embd || n_tokens == 0) return 0;
    if (token_embd->type == DS4_TENSOR_Q8_0) {
        return ds4_gpu_embed_tokens_q8_0_tensor(out,
                                                tokens,
                                                model->map,
                                                model->size,
                                                token_embd->abs_offset,
                                                DS4_N_VOCAB,
                                                n_tokens,
                                                DS4_N_EMBD);
    }
    if (token_embd->type == DS4_TENSOR_F16) {
        return ds4_gpu_embed_tokens_hc_tensor(out,
                                              tokens,
                                              model->map,
                                              model->size,
                                              token_embd->abs_offset,
                                              DS4_N_VOCAB,
                                              n_tokens,
                                              DS4_N_EMBD,
                                              1);
    }
    return 0;
}

static int glm_graph_embed_token_tensor(
        ds4_gpu_tensor   *out,
        const ds4_model  *model,
        const ds4_tensor *token_embd,
        uint32_t          token) {
    if (!out || !model || !token_embd) return 0;
    if (token_embd->type == DS4_TENSOR_Q8_0) {
        return ds4_gpu_embed_token_q8_0_tensor(out,
                                               model->map,
                                               model->size,
                                               token_embd->abs_offset,
                                               DS4_N_VOCAB,
                                               token,
                                               DS4_N_EMBD);
    }
    if (token_embd->type == DS4_TENSOR_F16) {
        return ds4_gpu_embed_token_hc_tensor(out,
                                             model->map,
                                             model->size,
                                             token_embd->abs_offset,
                                             DS4_N_VOCAB,
                                             token,
                                             DS4_N_EMBD,
                                             1);
    }
    return 0;
}

static bool glm_graph_validate_layout(
        const ds4_model   *model,
        const ds4_weights *weights,
        ds4_glm_gpu_graph *g,
        uint32_t           layer_start,
        uint32_t           layer_end,
        bool               require_token_embd,
        bool               require_output_head) {
    if (!model || !weights || !g) return false;
    const uint32_t normal_layers = glm_graph_normal_layer_count();
    if (normal_layers == 0 || DS4_N_ROT >= DS4_N_KEY_MLA) {
        fprintf(stderr, "ds4: GLM Metal graph found unsupported layer/key dimensions\n");
        return false;
    }
    if (layer_end == UINT32_MAX) layer_end = normal_layers - 1u;
    if (layer_start > layer_end || layer_end >= normal_layers) {
        fprintf(stderr,
                "ds4: GLM Metal graph found invalid layer slice %u:%u for %u normal layers\n",
                layer_start,
                layer_end,
                normal_layers);
        return false;
    }

    g->has_token_embd = weights->token_embd != NULL;
    g->has_output_head = weights_have_output_head(weights);
    if (require_token_embd && !g->has_token_embd) {
        fprintf(stderr, "ds4: GLM Metal graph layer slice requires token embeddings\n");
        return false;
    }
    if (g->has_token_embd &&
        !glm_graph_token_embedding_layout(weights->token_embd)) {
        fprintf(stderr, "ds4: GLM Metal graph found unexpected token embedding layout\n");
        return false;
    }
    if (require_output_head && !g->has_output_head) {
        fprintf(stderr, "ds4: GLM Metal graph layer slice requires the output head\n");
        return false;
    }
    if (weights_have_partial_output_head(weights) && !g->has_output_head) {
        fprintf(stderr, "ds4: GLM Metal graph found partial output head\n");
        return false;
    }
    if (g->has_output_head &&
        (!glm_graph_tensor_layout(weights->output_norm, DS4_TENSOR_F32, 1, DS4_N_EMBD, 0, 0) ||
         !glm_graph_tensor_layout(weights->output, DS4_TENSOR_Q8_0, 2,
                                  DS4_N_EMBD, DS4_N_VOCAB, 0))) {
        fprintf(stderr, "ds4: GLM Metal graph found unexpected output head layout\n");
        return false;
    }

    g->normal_layers = normal_layers;
    g->layer_start = layer_start;
    g->layer_end = layer_end;
    g->layer_count = layer_end - layer_start + 1u;
    g->q_dim = (uint64_t)DS4_N_HEAD * DS4_N_KEY_MLA;
    g->q_nope = (uint64_t)DS4_N_KEY_MLA - DS4_N_ROT;
    g->heads_dim = (uint64_t)DS4_N_HEAD * DS4_N_VALUE_MLA;
    g->dense_hidden_max = DS4_N_FF_EXP;
    g->kv_raw_dim = 0;
    for (uint32_t il = g->layer_start; il <= g->layer_end; il++) {
        if (!glm_graph_validate_layer_layout(model,
                                             &weights->layer[il],
                                             il,
                                             g->q_dim,
                                             g->q_nope,
                                             g->heads_dim,
                                             &g->kv_raw_dim,
                                             &g->dense_hidden_max)) {
            return false;
        }
        if (glm_graph_layer_uses_generic_routed_moe(&weights->layer[il])) {
            g->generic_routed_moe = true;
        }
    }
    const uint64_t sparse_mid_elems = (uint64_t)DS4_N_EXPERT_USED * DS4_N_FF_EXP;
    g->ffn_mid_elems =
        g->dense_hidden_max > sparse_mid_elems ? g->dense_hidden_max : sparse_mid_elems;
    return g->layer_count > 0 &&
           g->kv_raw_dim >= (uint64_t)DS4_N_KV_LORA + DS4_N_ROT;
}

static void glm_graph_free(ds4_glm_gpu_graph *g) {
    if (!g) return;
    for (uint32_t il = 0; il < DS4_MAX_LAYER; il++) {
        ds4_gpu_tensor_free(g->layer_indexer_key_cache[il]);
        ds4_gpu_tensor_free(g->layer_k_rope_cache[il]);
        ds4_gpu_tensor_free(g->layer_kv_lora_cache[il]);
        ds4_gpu_tensor_free(g->layer_value_cache[il]);
        ds4_gpu_tensor_free(g->layer_key_cache[il]);
    }
    ds4_gpu_tensor_free(g->logits);
    ds4_gpu_tensor_free(g->batch_router_weights);
    ds4_gpu_tensor_free(g->batch_router_selected);
    ds4_gpu_tensor_free(g->batch_router_probs);
    ds4_gpu_tensor_free(g->batch_router_logits);
    ds4_gpu_tensor_free(g->batch_routed_down);
    ds4_gpu_tensor_free(g->batch_routed_up);
    ds4_gpu_tensor_free(g->batch_routed_gate);
    ds4_gpu_tensor_free(g->batch_ffn_out);
    ds4_gpu_tensor_free(g->batch_ffn_mid);
    ds4_gpu_tensor_free(g->batch_shared_mid);
    ds4_gpu_tensor_free(g->batch_ffn_up);
    ds4_gpu_tensor_free(g->batch_ffn_gate);
    ds4_gpu_tensor_free(g->batch_ffn_norm);
    ds4_gpu_tensor_free(g->batch_after_attn);
    ds4_gpu_tensor_free(g->batch_attn_out);
    ds4_gpu_tensor_free(g->batch_heads);
    ds4_gpu_tensor_free(g->batch_value);
    ds4_gpu_tensor_free(g->batch_k_nope);
    ds4_gpu_tensor_free(g->batch_kv_norm);
    ds4_gpu_tensor_free(g->batch_kv_raw);
    ds4_gpu_tensor_free(g->batch_attn_lora);
    ds4_gpu_tensor_free(g->batch_qk_low);
    ds4_gpu_tensor_free(g->batch_indexer_selected);
    ds4_gpu_tensor_free(g->batch_indexer_scores);
    ds4_gpu_tensor_free(g->batch_indexer_weights);
    ds4_gpu_tensor_free(g->batch_indexer_q);
    ds4_gpu_tensor_free(g->batch_indexer_k);
    ds4_gpu_tensor_free(g->batch_q);
    ds4_gpu_tensor_free(g->batch_q_rank_norm);
    ds4_gpu_tensor_free(g->batch_q_rank);
    ds4_gpu_tensor_free(g->batch_attn_norm);
    ds4_gpu_tensor_free(g->batch_next);
    ds4_gpu_tensor_free(g->batch_cur);
    ds4_gpu_tensor_free(g->prefill_tokens);
    ds4_gpu_tensor_free(g->output_norm);
    ds4_gpu_tensor_free(g->router_weights);
    ds4_gpu_tensor_free(g->router_selected);
    ds4_gpu_tensor_free(g->router_probs);
    ds4_gpu_tensor_free(g->router_logits);
    ds4_gpu_tensor_free(g->ffn_sum);
    ds4_gpu_tensor_free(g->ffn_out);
    ds4_gpu_tensor_free(g->routed_down);
    ds4_gpu_tensor_free(g->routed_up);
    ds4_gpu_tensor_free(g->routed_gate);
    ds4_gpu_tensor_free(g->ffn_mid);
    ds4_gpu_tensor_free(g->ffn_up);
    ds4_gpu_tensor_free(g->ffn_gate);
    ds4_gpu_tensor_free(g->ffn_norm);
    ds4_gpu_tensor_free(g->after_attn);
    ds4_gpu_tensor_free(g->attn_out);
    ds4_gpu_tensor_free(g->heads);
    ds4_gpu_tensor_free(g->value);
    ds4_gpu_tensor_free(g->k_nope);
    ds4_gpu_tensor_free(g->kv_norm);
    ds4_gpu_tensor_free(g->kv_raw);
    ds4_gpu_tensor_free(g->attn_partial_ms);
    ds4_gpu_tensor_free(g->attn_partial_lora);
    ds4_gpu_tensor_free(g->qk_low);
    ds4_gpu_tensor_free(g->indexer_selected);
    ds4_gpu_tensor_free(g->indexer_scores);
    ds4_gpu_tensor_free(g->indexer_weights);
    ds4_gpu_tensor_free(g->indexer_q);
    ds4_gpu_tensor_free(g->indexer_k);
    ds4_gpu_tensor_free(g->q);
    ds4_gpu_tensor_free(g->q_rank_norm);
    ds4_gpu_tensor_free(g->q_rank);
    ds4_gpu_tensor_free(g->attn_norm);
    ds4_gpu_tensor_free(g->next);
    ds4_gpu_tensor_free(g->cur);
    memset(g, 0, sizeof(*g));
}

static bool glm_graph_ensure_compact_cache(
        const ds4_glm_gpu_graph *g,
        uint32_t                 needed_rows) {
    if (!g || needed_rows == 0) return false;
    if (needed_rows <= g->ctx_cap && g->compact_cache_cap == 0) return true;
    if (needed_rows <= g->compact_cache_cap) return true;
    fprintf(stderr,
            "ds4: GLM compact DSA cache capacity %u is smaller than required row %u "
            "(ctx=%u, full_cap=%u)\n",
            g->compact_cache_cap,
            needed_rows,
            g->ctx_size,
            g->ctx_cap);
    return false;
}

static bool glm_graph_warm_compact_indexer_store(
        ds4_glm_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        uint32_t           warm_pos) {
    if (!g || !model || !weights) return false;
    if (g->compact_cache_cap == 0 || g->indexer_full_layers == 0) return true;
    if (!g->indexer_k) return false;
    if (warm_pos >= g->compact_cache_cap) warm_pos = g->compact_cache_cap - 1u;

    if (ds4_gpu_tensor_fill_f32(g->indexer_k,
                                0.0f,
                                DS4_N_INDEXER_HEAD_DIM) == 0) {
        return false;
    }

    const bool profile = false;
    const double t0 = 0.0;
    bool ok = ds4_gpu_begin_commands() != 0;
    uint32_t warmed = 0;
    for (uint32_t il = g->layer_start; ok && il <= g->layer_end; il++) {
        if (!glm_graph_layer_uses_full_indexer(il)) continue;
        const ds4_layer_weights *l = &weights->layer[il];
        if (!g->layer_indexer_key_cache[il] ||
            !l->indexer_k_norm ||
            !l->indexer_k_norm_b) {
            ok = false;
            break;
        }
        const float rope_base = layer_rope_freq_base(il);
        const float rope_scale = layer_rope_freq_scale(il);
        ok = ds4_gpu_glm_store_indexer_k_tensor(
                g->layer_indexer_key_cache[il],
                g->indexer_k,
                model->map,
                model->size,
                l->indexer_k_norm->abs_offset,
                l->indexer_k_norm_b->abs_offset,
                warm_pos,
                1,
                g->compact_cache_cap,
                DS4_N_INDEXER_HEAD_DIM,
                DS4_N_ROT,
                0,
                1.0e-6f,
                rope_base,
                rope_scale,
                0.0f,
                1.0f,
                DS4_ROPE_YARN_BETA_FAST,
                DS4_ROPE_YARN_BETA_SLOW,
                glm_graph_compact_cache_is_f16()) != 0;
        if (ok) warmed++;
    }
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();

    if (profile) {
        fprintf(stderr,
                "ds4: GLM compact indexer warmup pos=%u layers=%u %.3f ms\n",
                warm_pos,
                warmed,
                (now_sec() - t0) * 1000.0);
    }
    return ok;
}

static bool glm_graph_alloc(
        ds4_glm_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        int                ctx_size,
        bool               ssd_streaming,
        bool               ssd_streaming_cold) {
    if (!g || !model || !weights || ctx_size <= 0) return false;
    const uint32_t normal_layers = glm_graph_normal_layer_count();
    if (normal_layers == 0) {
        fprintf(stderr, "ds4: GLM Metal graph found no normal transformer layers\n");
        return false;
    }
    memset(g, 0, sizeof(*g));
    g->ssd_streaming = ssd_streaming;
    g->ssd_streaming_cold = ssd_streaming_cold;

    if (!glm_graph_context_request(ctx_size, &g->ctx_size)) return false;
    if (!glm_graph_memory_guard(model,
                                weights,
                                g->ssd_streaming,
                                g->ctx_size)) {
        return false;
    }
    if (!glm_graph_validate_layout(model,
                                   weights,
                                   g,
                                   0,
                                   normal_layers - 1u,
                                   true,
                                   true)) {
        return false;
    }
    g->ctx_cap = glm_graph_full_attention_cap(g->ctx_size,
                                              g->ssd_streaming);
    g->full_kv_cache = glm_graph_expanded_kv_cache_enabled(g->ssd_streaming);
    g->compact_cache_cap =
        glm_graph_compact_cache_initial_cap(g->ctx_size, g->ctx_cap);
    g->indexed_prefill_cap =
        g->compact_cache_cap != 0 ?
            glm_graph_indexed_prefill_chunk_tokens(g->ctx_cap, g->compact_cache_cap) :
            0;
    g->indexed_prefill_score_cap =
        glm_graph_indexed_prefill_score_tokens(g->indexed_prefill_cap,
                                               g->compact_cache_cap);
    g->indexer_full_layers =
        glm_graph_full_indexer_layer_count_range(g->layer_start,
                                                 g->layer_end);
    if (g->ctx_size > g->ctx_cap) {
        fprintf(stderr,
                "ds4: GLM Metal session ctx=%u (model max=%u); "
                "full-attention prefill/work cap=%u; compact indexed decode is used beyond the cap\n",
                g->ctx_size,
                glm_graph_model_context_limit(),
                g->ctx_cap);
    }

    const uint64_t emb_bytes = (uint64_t)DS4_N_EMBD * sizeof(float);
    const uint64_t q_rank_bytes = (uint64_t)DS4_N_LORA_Q * sizeof(float);
    const uint64_t q_bytes = g->q_dim * sizeof(float);
    const uint64_t kv_raw_bytes = g->kv_raw_dim * sizeof(float);
    const uint64_t kv_norm_bytes = (uint64_t)DS4_N_KV_LORA * sizeof(float);
    const uint64_t k_nope_bytes = (uint64_t)DS4_N_HEAD * g->q_nope * sizeof(float);
    const uint64_t heads_bytes = g->heads_dim * sizeof(float);
    const uint64_t indexer_k_bytes = (uint64_t)DS4_N_INDEXER_HEAD_DIM * sizeof(float);
    const uint64_t indexer_q_bytes =
        (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM * sizeof(float);
    const uint64_t indexer_weights_bytes =
        (uint64_t)DS4_N_INDEXER_HEAD * sizeof(float);
    const uint64_t indexer_work_cap =
        g->compact_cache_cap != 0 ? g->compact_cache_cap : g->ctx_cap;
    const uint64_t indexer_scores_bytes = indexer_work_cap * sizeof(float);
    const uint32_t indexer_top_k = glm_graph_indexer_top_k_limit();
    const uint64_t indexer_selected_bytes =
        (uint64_t)indexer_top_k * sizeof(uint32_t);
    const uint64_t qk_low_bytes =
        (uint64_t)DS4_N_HEAD * DS4_N_KV_LORA * sizeof(float);
    const uint32_t split_attn_blocks = glm_graph_indexed_decode_split_blocks();
    const uint64_t attn_partial_lora_bytes =
        (uint64_t)split_attn_blocks * qk_low_bytes;
    const uint64_t attn_partial_ms_bytes =
        (uint64_t)split_attn_blocks * DS4_N_HEAD * 2u * sizeof(float);
    const uint64_t logits_bytes = (uint64_t)DS4_N_VOCAB * sizeof(float);
    const uint64_t full_kv_elem_bytes = glm_graph_full_kv_cache_elem_bytes();
    const uint64_t key_cache_bytes = g->full_kv_cache ?
        (uint64_t)g->ctx_cap * g->q_dim * full_kv_elem_bytes : 0;
    const uint64_t value_cache_bytes = g->full_kv_cache ?
        (uint64_t)g->ctx_cap * g->heads_dim * full_kv_elem_bytes : 0;
    const uint64_t compact_kv_lora_bytes =
        (uint64_t)g->compact_cache_cap * DS4_N_KV_LORA *
        glm_graph_compact_cache_elem_bytes();
    const uint64_t compact_k_rope_bytes =
        (uint64_t)g->compact_cache_cap * DS4_N_ROT *
        glm_graph_compact_cache_elem_bytes();
    const uint64_t compact_indexer_key_bytes =
        (uint64_t)g->compact_cache_cap * DS4_N_INDEXER_HEAD_DIM *
        glm_graph_compact_cache_elem_bytes();
    const uint64_t batch_rows = g->ctx_cap;
    const uint64_t indexed_batch_rows = g->indexed_prefill_cap;
    const uint64_t indexed_score_rows = g->indexed_prefill_score_cap;
    const uint64_t batch_indexer_q_bytes = indexed_batch_rows * indexer_q_bytes;
    const uint64_t batch_indexer_weights_bytes = indexed_batch_rows * indexer_weights_bytes;
    const uint64_t batch_indexer_scores_bytes =
        indexed_score_rows * indexer_work_cap * sizeof(float);
    const uint64_t batch_indexer_selected_bytes =
        indexed_batch_rows * indexer_top_k * sizeof(uint32_t);
    const uint64_t batch_qk_low_bytes = indexed_batch_rows * qk_low_bytes;
    const uint64_t batch_attn_lora_bytes =
        indexed_batch_rows * (uint64_t)DS4_N_HEAD * DS4_N_KV_LORA * sizeof(float);
    const uint64_t routed_mid_bytes =
        (uint64_t)DS4_N_EXPERT_USED * DS4_N_FF_EXP * sizeof(float);
    const uint64_t routed_down_bytes =
        (uint64_t)DS4_N_EXPERT_USED * DS4_N_EMBD * sizeof(float);
    const double cache_gib =
        (double)(g->layer_count * (key_cache_bytes + value_cache_bytes)) /
        (1024.0 * 1024.0 * 1024.0);
    if (g->full_kv_cache) {
        fprintf(stderr,
                "ds4: GLM Metal graph allocating full-attention KV cache: work_ctx=%u layers=%u:%u (%u) %s %.2f GiB\n",
                g->ctx_cap,
                g->layer_start,
                g->layer_end,
                g->layer_count,
                "f16",
                cache_gib);
    } else {
        fprintf(stderr,
                "ds4: GLM Metal graph using compact DSA KV only; expanded full-attention KV cache is skipped\n");
    }
    if (g->compact_cache_cap != 0) {
        const uint64_t compact_kv_total =
            (uint64_t)g->layer_count * (compact_kv_lora_bytes + compact_k_rope_bytes);
        const uint64_t compact_indexer_total =
            (uint64_t)g->indexer_full_layers * compact_indexer_key_bytes;
        const double compact_gib =
            (double)(compact_kv_total + compact_indexer_total) /
            (1024.0 * 1024.0 * 1024.0);
        fprintf(stderr,
                "ds4: GLM Metal graph allocating compact DSA cache: rows=%u logical_ctx=%u kv_layers=%u indexer_layers=%u %s %.2f GiB\n",
                g->compact_cache_cap,
                g->ctx_size,
                g->layer_count,
                g->indexer_full_layers,
                glm_graph_compact_cache_is_f16() ? "f16" : "f32",
                compact_gib);
        fprintf(stderr,
                "ds4: GLM compact indexed prefill chunk=%u score_rows=%u score_scratch=%.2f MiB\n",
                g->indexed_prefill_cap,
                g->indexed_prefill_score_cap,
                (double)batch_indexer_scores_bytes / (1024.0 * 1024.0));
    }

    bool ok = true;
#define DS4_GLM_GRAPH_ALLOC_TENSOR(var, bytes_) \
    do { \
        (var) = ds4_gpu_tensor_alloc((bytes_)); \
        if (!(var)) { \
            fprintf(stderr, "ds4: GLM Metal graph could not allocate %s\n", #var); \
            ok = false; \
        } \
    } while (0)

    DS4_GLM_GRAPH_ALLOC_TENSOR(g->cur, emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->next, emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->attn_norm, emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->q_rank, q_rank_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->q_rank_norm, q_rank_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->q, q_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->indexer_k, indexer_k_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->indexer_q, indexer_q_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->indexer_weights, indexer_weights_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->indexer_scores, indexer_scores_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->indexer_selected, indexer_selected_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->qk_low, qk_low_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->attn_partial_lora, attn_partial_lora_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->attn_partial_ms, attn_partial_ms_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->kv_raw, kv_raw_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->kv_norm, kv_norm_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->k_nope, k_nope_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->value, heads_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->heads, heads_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->attn_out, emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->after_attn, emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->ffn_norm, emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->ffn_gate, g->dense_hidden_max * sizeof(float));
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->ffn_up, g->dense_hidden_max * sizeof(float));
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->ffn_mid, g->ffn_mid_elems * sizeof(float));
    if (g->generic_routed_moe) {
        DS4_GLM_GRAPH_ALLOC_TENSOR(g->routed_gate, routed_mid_bytes);
        DS4_GLM_GRAPH_ALLOC_TENSOR(g->routed_up, routed_mid_bytes);
        DS4_GLM_GRAPH_ALLOC_TENSOR(g->routed_down, routed_down_bytes);
    }
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->ffn_out, emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->ffn_sum, emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->router_logits, (uint64_t)DS4_N_EXPERT * sizeof(float));
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->router_probs, (uint64_t)DS4_N_EXPERT * sizeof(float));
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->router_selected, (uint64_t)DS4_N_EXPERT_USED * sizeof(int32_t));
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->router_weights, (uint64_t)DS4_N_EXPERT_USED * sizeof(float));
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->output_norm, emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->logits, logits_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_router_logits, batch_rows * DS4_N_EXPERT * sizeof(float));
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_router_probs, batch_rows * DS4_N_EXPERT * sizeof(float));
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_router_selected, batch_rows * DS4_N_EXPERT_USED * sizeof(int32_t));
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_router_weights, batch_rows * DS4_N_EXPERT_USED * sizeof(float));

    DS4_GLM_GRAPH_ALLOC_TENSOR(g->prefill_tokens, batch_rows * sizeof(int32_t));
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_cur, batch_rows * emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_next, batch_rows * emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_attn_norm, batch_rows * emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_q_rank, batch_rows * q_rank_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_q_rank_norm, batch_rows * q_rank_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_q, batch_rows * q_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_indexer_k, batch_rows * indexer_k_bytes);
    if (g->compact_cache_cap != 0 && g->indexed_prefill_cap != 0) {
        DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_indexer_q, batch_indexer_q_bytes);
        DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_indexer_weights, batch_indexer_weights_bytes);
        DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_indexer_scores, batch_indexer_scores_bytes);
        DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_indexer_selected, batch_indexer_selected_bytes);
        DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_qk_low, batch_qk_low_bytes);
        DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_attn_lora, batch_attn_lora_bytes);
    }
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_kv_raw, batch_rows * kv_raw_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_kv_norm, batch_rows * kv_norm_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_k_nope, batch_rows * k_nope_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_value, batch_rows * heads_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_heads, batch_rows * heads_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_attn_out, batch_rows * emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_after_attn, batch_rows * emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_ffn_norm, batch_rows * emb_bytes);
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_ffn_gate, batch_rows * g->dense_hidden_max * sizeof(float));
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_ffn_up, batch_rows * g->dense_hidden_max * sizeof(float));
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_shared_mid, batch_rows * DS4_N_FF_EXP * sizeof(float));
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_ffn_mid, batch_rows * g->ffn_mid_elems * sizeof(float));
    if (g->generic_routed_moe) {
        DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_routed_gate, batch_rows * routed_mid_bytes);
        DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_routed_up, batch_rows * routed_mid_bytes);
        DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_routed_down, batch_rows * routed_down_bytes);
    }
    DS4_GLM_GRAPH_ALLOC_TENSOR(g->batch_ffn_out, batch_rows * emb_bytes);

    for (uint32_t il = g->layer_start; ok && il <= g->layer_end; il++) {
        if (g->full_kv_cache) {
            DS4_GLM_GRAPH_ALLOC_TENSOR(g->layer_key_cache[il], key_cache_bytes);
            DS4_GLM_GRAPH_ALLOC_TENSOR(g->layer_value_cache[il], value_cache_bytes);
        }
        if (g->compact_cache_cap != 0) {
            DS4_GLM_GRAPH_ALLOC_TENSOR(g->layer_kv_lora_cache[il], compact_kv_lora_bytes);
            DS4_GLM_GRAPH_ALLOC_TENSOR(g->layer_k_rope_cache[il], compact_k_rope_bytes);
            if (glm_graph_layer_uses_full_indexer(il)) {
                DS4_GLM_GRAPH_ALLOC_TENSOR(g->layer_indexer_key_cache[il],
                                           compact_indexer_key_bytes);
            }
        }
    }
#undef DS4_GLM_GRAPH_ALLOC_TENSOR

    if (!ok) {
        glm_graph_free(g);
        return false;
    }
    return true;
}

static int glm_graph_matmul_q8_0_decode_tensor(
        ds4_gpu_tensor       *out,
        const ds4_model      *model,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        bool                  ssd_streaming) {
    if (!model) return 0;
    if (ssd_streaming) {
        return ds4_gpu_matmul_q8_0_tensor(out,
                                          model->map,
                                          model->size,
                                          weight_offset,
                                          in_dim,
                                          out_dim,
                                          x,
                                          1);
    }
    return ds4_gpu_matmul_q8_0_decode_mpp_model_view_tensor(out,
                                                            model->map,
                                                            model->size,
                                                            weight_offset,
                                                            in_dim,
                                                            out_dim,
                                                            x,
                                                            1);
}

static bool glm_graph_q8_decode_profile_enabled(uint32_t il, const char *label) {
    (void)il;
    (void)label;
    return false;
}

static int glm_graph_matmul_q8_0_decode_profiled_tensor(
        ds4_gpu_tensor       *out,
        const ds4_model      *model,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              il,
        uint32_t              pos,
        const char           *label,
        bool                  ssd_streaming) {
    const bool profile = glm_graph_q8_decode_profile_enabled(il, label);
    if (profile) {
        if (ds4_gpu_end_commands() == 0) return 0;
        if (ds4_gpu_begin_commands() == 0) return 0;
    }
    const double t0 = profile ? now_sec() : 0.0;
    int ok = glm_graph_matmul_q8_0_decode_tensor(out,
                                                 model,
                                                 weight_offset,
                                                 in_dim,
                                                 out_dim,
                                                 x,
                                                 ssd_streaming);
    if (profile) {
        if (ok) ok = ds4_gpu_end_commands() != 0;
        else (void)ds4_gpu_synchronize();
        const double now = now_sec();
        fprintf(stderr,
                "ds4: GLM Q8 decode profile layer=%u pos=%u label=%s in=%llu out=%llu %.3f ms\n",
                il,
                pos,
                label ? label : "?",
                (unsigned long long)in_dim,
                (unsigned long long)out_dim,
                (now - t0) * 1000.0);
        if (ok) ok = ds4_gpu_begin_commands() != 0;
    }
    return ok;
}

static bool glm_graph_encode_output_head_from(
        ds4_glm_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        const ds4_gpu_tensor *hidden) {
    bool ok = ds4_gpu_rms_norm_weight_tensor(g->output_norm,
                                             hidden,
                                             model->map,
                                             model->size,
                                             weights->output_norm->abs_offset,
                                             DS4_N_EMBD,
                                             DS4_RMS_EPS) != 0;
    if (ok) ok = glm_graph_matmul_q8_0_decode_tensor(g->logits,
                                                     model,
                                                     weights->output->abs_offset,
                                                     DS4_N_EMBD,
                                                     DS4_N_VOCAB,
                                                     g->output_norm,
                                                     g->ssd_streaming) != 0;
    return ok;
}

static bool glm_graph_encode_output_head(
        ds4_glm_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights) {
    return glm_graph_encode_output_head_from(g, model, weights, g->cur);
}

static bool glm_graph_forward_output_head(
        ds4_glm_gpu_graph    *g,
        const ds4_model      *model,
        const ds4_weights    *weights,
        const ds4_gpu_tensor *hidden,
        float                *logits_out) {
    if (!g || !model || !weights || !hidden || !logits_out) return false;
    bool ok = ds4_gpu_begin_commands() != 0;
    if (ok) ok = glm_graph_encode_output_head_from(g, model, weights, hidden);
    if (ok) ok = ds4_gpu_end_commands() != 0;
    else (void)ds4_gpu_synchronize();
    if (ok) {
        ok = ds4_gpu_tensor_read(g->logits,
                                 0,
                                 logits_out,
                                 (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
    }
    return ok;
}

static bool glm_graph_profile_stage(
        bool        enabled,
        const char *part,
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        double     *stage_t0) {
    if (!enabled) return true;
    if (!stage_t0) return false;
    return metal_graph_layer_stage_profile_boundary(part, stage, il, pos0, n_tokens, stage_t0);
}

static bool glm_graph_profile_router_selection(
        ds4_glm_gpu_graph       *g,
        const ds4_layer_weights *layer,
        uint32_t                 il,
        uint32_t                 pos) {
    if (!g_expert_profile.active) return true;
    if (!g || !layer || !g->router_selected || !g->router_weights) return false;

    if (ds4_gpu_end_commands() == 0) {
        fprintf(stderr,
                "ds4: failed to end GLM Metal command batch for expert profile readback\n");
        return false;
    }

    int32_t selected[DS4_MAX_EXPERT_USED] = {0};
    float weights[DS4_MAX_EXPERT_USED] = {0};
    const bool read_ok =
        ds4_gpu_tensor_read(g->router_selected,
                            0,
                            selected,
                            (uint64_t)DS4_N_EXPERT_USED * sizeof(selected[0])) != 0 &&
        ds4_gpu_tensor_read(g->router_weights,
                            0,
                            weights,
                            (uint64_t)DS4_N_EXPERT_USED * sizeof(weights[0])) != 0;

    if (ds4_gpu_begin_commands() == 0) {
        fprintf(stderr,
                "ds4: failed to resume GLM Metal command batch after expert profile readback\n");
        return false;
    }
    if (!read_ok) {
        fprintf(stderr, "ds4: failed to read GLM Metal router tensors for expert profile\n");
        return false;
    }

    ds4_expert_profile_record(il,
                              pos,
                              selected,
                              weights,
                              layer->ffn_gate_tid2eid != NULL);
    return true;
}

static bool glm_graph_profile_router_selection_batch(
        ds4_glm_gpu_graph       *g,
        const ds4_layer_weights *layer,
        uint32_t                 il,
        uint32_t                 pos0,
        uint32_t                 n_tokens) {
    if (!g_expert_profile.active) return true;
    if (!g || !layer || !g->batch_router_selected ||
        !g->batch_router_weights || n_tokens == 0) {
        return false;
    }

    const size_t selected_count = (size_t)n_tokens * DS4_N_EXPERT_USED;
    if (n_tokens != 0 && selected_count / n_tokens != DS4_N_EXPERT_USED) {
        return false;
    }
    if (selected_count > SIZE_MAX / sizeof(int32_t) ||
        selected_count > SIZE_MAX / sizeof(float)) {
        return false;
    }

    if (ds4_gpu_end_commands() == 0) {
        fprintf(stderr,
                "ds4: failed to end GLM Metal command batch for batch expert profile readback\n");
        return false;
    }

    int32_t *selected = xmalloc(selected_count * sizeof(selected[0]));
    float *weights = xmalloc(selected_count * sizeof(weights[0]));
    const bool read_ok =
        ds4_gpu_tensor_read(g->batch_router_selected,
                            0,
                            selected,
                            (uint64_t)selected_count * sizeof(selected[0])) != 0 &&
        ds4_gpu_tensor_read(g->batch_router_weights,
                            0,
                            weights,
                            (uint64_t)selected_count * sizeof(weights[0])) != 0;

    if (ds4_gpu_begin_commands() == 0) {
        free(weights);
        free(selected);
        fprintf(stderr,
                "ds4: failed to resume GLM Metal command batch after batch expert profile readback\n");
        return false;
    }
    if (!read_ok) {
        free(weights);
        free(selected);
        fprintf(stderr, "ds4: failed to read GLM Metal batch router tensors for expert profile\n");
        return false;
    }

    for (uint32_t t = 0; t < n_tokens; t++) {
        const size_t off = (size_t)t * DS4_N_EXPERT_USED;
        ds4_expert_profile_record(il,
                                  pos0 + t,
                                  selected + off,
                                  weights + off,
                                  layer->ffn_gate_tid2eid != NULL);
    }

    free(weights);
    free(selected);
    return true;
}

static bool glm_graph_prefill_stage_boundary(
        bool        stage_profile,
        bool        stage_sync,
        const char *part,
        const char *stage,
        uint32_t    il,
        uint32_t    pos0,
        uint32_t    n_tokens,
        double     *stage_t0) {
    if (stage_profile) {
        return glm_graph_profile_stage(true, part, stage, il, pos0, n_tokens, stage_t0);
    }
    if (stage_sync) return glm_graph_prefill_stage_sync_boundary();
    return true;
}

static int glm_graph_routed_moe_one_dispatch(
        const ds4_glm_gpu_graph *g,
        const ds4_model         *model,
        const ds4_layer_weights *l,
        uint32_t                 il,
        ds4_gpu_tensor          *out,
        ds4_gpu_tensor          *mid,
        uint64_t                 gate_expert_bytes,
        uint64_t                 gate_row_bytes,
        uint64_t                 up_expert_bytes,
        uint64_t                 up_row_bytes,
        uint64_t                 down_expert_bytes,
        uint64_t                 down_row_bytes,
        const ds4_gpu_tensor    *selected,
        const ds4_gpu_tensor    *weights,
        const ds4_gpu_tensor    *x,
        bool                     force_resident) {
    if (!g || !model || !l) return 0;
    if (glm_graph_layer_uses_generic_routed_moe(l)) {
        if (!g->routed_gate || !g->routed_up || !g->routed_down ||
            l->ffn_gate_exps->type != l->ffn_up_exps->type) {
            return 0;
        }
        return ds4_gpu_routed_moe_one_tensor(out,
                                             g->routed_gate,
                                             g->routed_up,
                                             mid,
                                             g->routed_down,
                                             model->map,
                                             model->size,
                                             l->ffn_gate_exps->abs_offset,
                                             l->ffn_up_exps->abs_offset,
                                             l->ffn_down_exps->abs_offset,
                                             l->ffn_gate_exps->type,
                                             l->ffn_down_exps->type,
                                             gate_expert_bytes,
                                             gate_row_bytes,
                                             down_expert_bytes,
                                             down_row_bytes,
                                             DS4_N_EMBD,
                                             DS4_N_FF_EXP,
                                             DS4_N_EMBD,
                                             selected,
                                             weights,
                                             DS4_N_EXPERT,
                                             DS4_N_EXPERT_USED,
                                             0.0f,
                                             x,
                                             il,
                                             force_resident);
    }

    return ds4_gpu_glm_routed_moe_one_tensor(out,
                                             mid,
                                             model->map,
                                             model->size,
                                             l->ffn_gate_exps->abs_offset,
                                             l->ffn_up_exps->abs_offset,
                                             l->ffn_down_exps->abs_offset,
                                             l->ffn_gate_exps->type,
                                             l->ffn_up_exps->type,
                                             l->ffn_down_exps->type,
                                             gate_expert_bytes,
                                             gate_row_bytes,
                                             up_expert_bytes,
                                             up_row_bytes,
                                             down_expert_bytes,
                                             down_row_bytes,
                                             DS4_N_EMBD,
                                             DS4_N_FF_EXP,
                                             DS4_N_EMBD,
                                             selected,
                                             weights,
                                             DS4_N_EXPERT,
                                             DS4_N_EXPERT_USED,
                                             il,
                                             x,
                                             force_resident);
}

static int glm_graph_routed_moe_batch_dispatch(
        ds4_glm_gpu_graph       *g,
        const ds4_model         *model,
        const ds4_layer_weights *l,
        uint32_t                 il,
        ds4_gpu_tensor          *out,
        ds4_gpu_tensor          *mid,
        uint64_t                 gate_expert_bytes,
        uint64_t                 gate_row_bytes,
        uint64_t                 up_expert_bytes,
        uint64_t                 up_row_bytes,
        uint64_t                 down_expert_bytes,
        uint64_t                 down_row_bytes,
        const ds4_gpu_tensor    *selected,
        const ds4_gpu_tensor    *weights,
        const ds4_gpu_tensor    *x,
        uint32_t                 n_tokens,
        uint32_t                 mid_token_stride,
        bool                     direct_scalar_q4) {
    if (!g || !model || !l) return 0;
    g->batch_routed_mid_is_f16 = false;

    if (glm_graph_layer_uses_generic_routed_moe(l)) {
        if (!g->batch_routed_gate || !g->batch_routed_up || !g->batch_routed_down ||
            l->ffn_gate_exps->type != l->ffn_up_exps->type) {
            return 0;
        }
        return ds4_gpu_routed_moe_batch_tensor(out,
                                               g->batch_routed_gate,
                                               g->batch_routed_up,
                                               mid,
                                               g->batch_routed_down,
                                               model->map,
                                               model->size,
                                               l->ffn_gate_exps->abs_offset,
                                               l->ffn_up_exps->abs_offset,
                                               l->ffn_down_exps->abs_offset,
                                               l->ffn_gate_exps->type,
                                               l->ffn_down_exps->type,
                                               gate_expert_bytes,
                                               gate_row_bytes,
                                               down_expert_bytes,
                                               down_row_bytes,
                                               DS4_N_EMBD,
                                               DS4_N_FF_EXP,
                                               DS4_N_EMBD,
                                               selected,
                                               weights,
                                               DS4_N_EXPERT,
                                               DS4_N_EXPERT_USED,
                                               0.0f,
                                               x,
                                               il,
                                               n_tokens,
                                               &g->batch_routed_mid_is_f16);
    }

    if (direct_scalar_q4) {
        return ds4_gpu_glm_routed_moe_batch_direct_scalar_q4_tensor(
                out,
                mid,
                model->map,
                model->size,
                l->ffn_gate_exps->abs_offset,
                l->ffn_up_exps->abs_offset,
                l->ffn_down_exps->abs_offset,
                l->ffn_gate_exps->type,
                l->ffn_up_exps->type,
                l->ffn_down_exps->type,
                gate_expert_bytes,
                gate_row_bytes,
                up_expert_bytes,
                up_row_bytes,
                down_expert_bytes,
                down_row_bytes,
                DS4_N_EMBD,
                DS4_N_FF_EXP,
                DS4_N_EMBD,
                selected,
                weights,
                DS4_N_EXPERT,
                DS4_N_EXPERT_USED,
                il,
                x,
                n_tokens,
                mid_token_stride);
    }

    return ds4_gpu_glm_routed_moe_batch_tensor(
            out,
            mid,
            model->map,
            model->size,
            l->ffn_gate_exps->abs_offset,
            l->ffn_up_exps->abs_offset,
            l->ffn_down_exps->abs_offset,
            l->ffn_gate_exps->type,
            l->ffn_up_exps->type,
            l->ffn_down_exps->type,
            gate_expert_bytes,
            gate_row_bytes,
            up_expert_bytes,
            up_row_bytes,
            down_expert_bytes,
            down_row_bytes,
            DS4_N_EMBD,
            DS4_N_FF_EXP,
            DS4_N_EMBD,
            selected,
            weights,
            DS4_N_EXPERT,
            DS4_N_EXPERT_USED,
            il,
            x,
            n_tokens,
            mid_token_stride);
}

static bool glm_graph_disable_add3_residual(void);

static bool glm_graph_use_streaming_selected_async_load(
        const ds4_glm_gpu_graph *g) {
    return g &&
           g->ssd_streaming &&
           getenv("DS4_METAL_ENABLE_GLM_STREAMING_SELECTED_ASYNC_LOAD") != NULL &&
           getenv("DS4_METAL_DISABLE_STREAMING_SELECTED_ASYNC_LOAD") == NULL &&
           getenv("DS4_METAL_DISABLE_GLM_STREAMING_SELECTED_ASYNC_LOAD") == NULL;
}

typedef struct glm_graph_streaming_async_profile {
    uint64_t async_calls;
    uint64_t sync_calls;
    double async_total_ms;
    double sync_total_ms;
    double async_signal_start_ms;
    double async_flush_router_ms;
    double async_shared_ms;
    double async_flush_shared_ms;
    double async_finish_ms;
    double async_routed_ms;
    double async_post_ms;
    double sync_early_load_ms;
    double sync_shared_ms;
    double sync_routed_ms;
    double sync_post_ms;
} glm_graph_streaming_async_profile;

static glm_graph_streaming_async_profile g_glm_streaming_async_profile;
static bool g_glm_streaming_async_profile_registered;

static bool glm_graph_streaming_async_profile_enabled(void) {
    return getenv("DS4_METAL_GLM_STREAMING_ASYNC_PROFILE") != NULL;
}

static void glm_graph_streaming_async_profile_print(void) {
    const glm_graph_streaming_async_profile *p =
        &g_glm_streaming_async_profile;
    if (p->async_calls == 0 && p->sync_calls == 0) return;
    const double async_calls = p->async_calls ? (double)p->async_calls : 1.0;
    const double sync_calls = p->sync_calls ? (double)p->sync_calls : 1.0;
    fprintf(stderr,
            "ds4: GLM streaming async profile async_calls=%llu "
            "total=%.3f ms avg=%.3f ms signal_start=%.3f ms "
            "flush_router=%.3f ms shared=%.3f ms flush_shared=%.3f ms "
            "finish=%.3f ms routed=%.3f ms post=%.3f ms\n",
            (unsigned long long)p->async_calls,
            p->async_total_ms,
            p->async_total_ms / async_calls,
            p->async_signal_start_ms,
            p->async_flush_router_ms,
            p->async_shared_ms,
            p->async_flush_shared_ms,
            p->async_finish_ms,
            p->async_routed_ms,
            p->async_post_ms);
    fprintf(stderr,
            "ds4: GLM streaming sync profile calls=%llu "
            "total=%.3f ms avg=%.3f ms early_load=%.3f ms "
            "shared=%.3f ms routed=%.3f ms post=%.3f ms\n",
            (unsigned long long)p->sync_calls,
            p->sync_total_ms,
            p->sync_total_ms / sync_calls,
            p->sync_early_load_ms,
            p->sync_shared_ms,
            p->sync_routed_ms,
            p->sync_post_ms);
}

static void glm_graph_streaming_async_profile_register(void) {
    if (g_glm_streaming_async_profile_registered) return;
    if (!glm_graph_streaming_async_profile_enabled()) return;
    atexit(glm_graph_streaming_async_profile_print);
    g_glm_streaming_async_profile_registered = true;
}

static double glm_graph_streaming_async_profile_ms(void) {
    return now_sec() * 1000.0;
}

/* Router-ahead prefetch (GLM streaming decode).
 *
 * While layer L's selected experts stream in, a worker thread predicts layer
 * L+1's selection by running L+1's router on L's ffn_norm activation (the
 * residual-stream similarity of consecutive gating inputs makes this a good
 * proxy) and issues advisory OS readahead for the predicted experts.  The
 * prediction never touches cache or compute state: mispredicted hints only
 * waste SSD bandwidth, and the real load path is unchanged.  The effective
 * engine profile selects advisory or off once at startup. */
typedef struct {
    const ds4_model   *model;
    const ds4_weights *weights;
    uint32_t           il;
    float              x[DS4_MAX_EMBD];
} glm_router_ahead_job;

static pthread_mutex_t g_glm_router_ahead_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_glm_router_ahead_cond = PTHREAD_COND_INITIALIZER;
static glm_router_ahead_job g_glm_router_ahead_job;
static bool g_glm_router_ahead_job_pending;
static bool g_glm_router_ahead_worker_started;
static const ds4_weights *g_glm_router_ahead_weights;

static bool glm_router_ahead_prefetch_enabled(
        const ds4_glm_gpu_graph *g) {
    return g && g->glm_router_ahead == DS4_GLM_ROUTER_AHEAD_ADVISORY;
}

/* CPU router selection from GPU logits with the prune mask applied.  Mirrors
 * the GLM select kernel math (sigmoid probs, +bias selection score, top-k,
 * weights normalized from probs and scaled): only near-tie float rounding can
 * differ.  Ends the open command buffer to read the logits, writes the
 * selection back, and reopens the buffer. */
static bool glm_graph_cpu_masked_router_select_batch(
        ds4_gpu_tensor          *logits_t,
        ds4_gpu_tensor          *selected_t,
        ds4_gpu_tensor          *weights_t,
        ds4_gpu_tensor          *probs_t,
        const ds4_model         *model,
        const ds4_layer_weights *l,
        uint32_t                 il,
        uint32_t                 n_tokens) {
    if (!logits_t || !selected_t || !weights_t || !model || !l ||
        il >= DS4_MAX_LAYER || n_tokens == 0) {
        return false;
    }
    const uint64_t logits_elems = (uint64_t)n_tokens * DS4_N_EXPERT;
    float *logits = malloc(logits_elems * sizeof(float));
    if (!logits) return false;
    bool ok = ds4_gpu_end_commands() != 0;
    if (ok) {
        ok = ds4_gpu_tensor_read(logits_t,
                                 0,
                                 logits,
                                 logits_elems * sizeof(float)) != 0;
    }
    const float *bias = ok ? tensor_data(model, l->ffn_exp_probs_b) : NULL;
    if (ok && !bias) ok = false;

    for (uint32_t t = 0; ok && t < n_tokens; t++) {
        const float *tl = logits + (uint64_t)t * DS4_N_EXPERT;
        float probs[DS4_MAX_EXPERT];
        float selection[DS4_MAX_EXPERT];
        for (uint32_t i = 0; i < DS4_N_EXPERT; i++) {
            probs[i] = sigmoid_stable(tl[i]);
            selection[i] = probs[i] + bias[i];
        }
        ds4_expert_prune_mask_apply_selection(il, selection, DS4_N_EXPERT);
        int selected[DS4_MAX_EXPERT_USED];
        topk_desc(selection, (int)DS4_N_EXPERT, (int)DS4_N_EXPERT_USED, selected);
        float weights[DS4_MAX_EXPERT_USED];
        int32_t selected_i32[DS4_MAX_EXPERT_USED];
        float sum = 0.0f;
        for (uint32_t i = 0; ok && i < DS4_N_EXPERT_USED; i++) {
            if (selected[i] < 0 || (uint32_t)selected[i] >= DS4_N_EXPERT) {
                ok = false;
                break;
            }
            weights[i] = probs[selected[i]];
            sum += weights[i];
        }
        if (!ok) break;
        if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
        for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
            weights[i] = weights[i] / sum * DS4_EXPERT_WEIGHT_SCALE;
            selected_i32[i] = (int32_t)selected[i];
        }
        ok = ds4_gpu_tensor_write(selected_t,
                                  (uint64_t)t * DS4_N_EXPERT_USED *
                                      sizeof(int32_t),
                                  selected_i32,
                                  sizeof(selected_i32)) != 0 &&
             ds4_gpu_tensor_write(weights_t,
                                  (uint64_t)t * DS4_N_EXPERT_USED *
                                      sizeof(float),
                                  weights,
                                  sizeof(weights)) != 0;
        if (ok && probs_t) {
            ok = ds4_gpu_tensor_write(probs_t,
                                      (uint64_t)t * DS4_N_EXPERT *
                                          sizeof(float),
                                      probs,
                                      (uint64_t)DS4_N_EXPERT *
                                          sizeof(float)) != 0;
        }
    }
    free(logits);
    if (ok) ok = ds4_gpu_begin_commands() != 0;
    return ok;
}

/* Advisory mode: how many layers ahead the worker predicts and warms.
 * 1 gives the OS ~one layer of readahead runway, 2 doubles it at a small
 * prediction-accuracy cost (the gating-input proxy drifts with distance). */
static uint32_t glm_router_ahead_prefetch_lookahead(
        const ds4_glm_gpu_graph *g) {
    if (!g || g->glm_router_ahead_lookahead < 1) return 1;
    return g->glm_router_ahead_lookahead > 4 ? 4 :
        g->glm_router_ahead_lookahead;
}

static void *glm_router_ahead_worker(void *arg) {
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&g_glm_router_ahead_mutex);
        while (!g_glm_router_ahead_job_pending) {
            pthread_cond_wait(&g_glm_router_ahead_cond,
                              &g_glm_router_ahead_mutex);
        }
        glm_router_ahead_job job = g_glm_router_ahead_job;
        g_glm_router_ahead_job_pending = false;
        pthread_mutex_unlock(&g_glm_router_ahead_mutex);

        const ds4_layer_weights *l = &job.weights->layer[job.il];
        int selected[DS4_MAX_EXPERT_USED];
        float expert_weight[DS4_MAX_EXPERT_USED];
        layer_glm_router_selected_experts(selected,
                                          expert_weight,
                                          job.model,
                                          l,
                                          job.il,
                                          job.x);
        uint64_t gate_in = 0, gate_out = 0, gate_row_bytes = 0;
        uint64_t down_in = 0, down_out = 0, down_row_bytes = 0;
        (void)tensor_expert_bytes(job.model, l->ffn_gate_exps, 0,
                                  &gate_in, &gate_out, &gate_row_bytes);
        (void)tensor_expert_bytes(job.model, l->ffn_down_exps, 0,
                                  &down_in, &down_out, &down_row_bytes);
        int32_t ids[DS4_MAX_EXPERT_USED];
        for (uint32_t i = 0; i < DS4_N_EXPERT_USED; i++) {
            ids[i] = (int32_t)selected[i];
        }
        (void)ds4_gpu_glm_stream_expert_prefetch_hint(
                job.il,
                ids,
                DS4_N_EXPERT_USED,
                job.model->map,
                job.model->size,
                l->ffn_gate_exps->abs_offset,
                l->ffn_up_exps->abs_offset,
                l->ffn_down_exps->abs_offset,
                gate_out * gate_row_bytes,
                down_out * down_row_bytes);
    }
    return NULL;
}

static bool glm_router_ahead_ensure_worker(void) {
    if (g_glm_router_ahead_worker_started) return true;
    pthread_t tid;
    if (pthread_create(&tid, NULL, glm_router_ahead_worker, NULL) != 0) {
        return false;
    }
    pthread_detach(tid);
    g_glm_router_ahead_worker_started = true;
    return true;
}

static void glm_router_ahead_prefetch_submit(
        const ds4_glm_gpu_graph *g,
        const ds4_model      *model,
        uint32_t              il_next,
        const ds4_gpu_tensor *ffn_norm) {
    if (!glm_router_ahead_prefetch_enabled(g)) return;
    const ds4_weights *weights = g_glm_router_ahead_weights;
    if (!model || !weights || !ffn_norm) return;
    if (il_next < DS4_N_LEADING_DENSE) return;
    if (il_next >= glm_graph_normal_layer_count()) return;
    const ds4_layer_weights *l = &weights->layer[il_next];
    if (!l->ffn_gate_inp || !l->ffn_exp_probs_b ||
        !l->ffn_gate_exps || !l->ffn_up_exps || !l->ffn_down_exps) {
        return;
    }
    if (!glm_router_ahead_ensure_worker()) return;

    pthread_mutex_lock(&g_glm_router_ahead_mutex);
    if (g_glm_router_ahead_job_pending) {
        /* Worker still busy with an older prediction: drop, best-effort. */
        pthread_mutex_unlock(&g_glm_router_ahead_mutex);
        return;
    }
    const bool read_ok = ds4_gpu_tensor_read(
            (ds4_gpu_tensor *)ffn_norm,
            0,
            g_glm_router_ahead_job.x,
            (uint64_t)DS4_N_EMBD * sizeof(float)) != 0;
    if (read_ok) {
        g_glm_router_ahead_job.model = model;
        g_glm_router_ahead_job.weights = weights;
        g_glm_router_ahead_job.il = il_next;
        g_glm_router_ahead_job_pending = true;
        pthread_cond_signal(&g_glm_router_ahead_cond);
    }
    pthread_mutex_unlock(&g_glm_router_ahead_mutex);
}

static bool glm_graph_encode_sparse_ffn_one(
        ds4_glm_gpu_graph       *g,
        const ds4_model         *model,
        const ds4_layer_weights *l,
        uint32_t                 il,
        uint32_t                 pos,
        const ds4_gpu_tensor    *ffn_norm,
        const ds4_gpu_tensor    *after_attn,
        ds4_gpu_tensor          *next,
        ds4_gpu_tensor          *ffn_gate,
        ds4_gpu_tensor          *ffn_up,
        ds4_gpu_tensor          *ffn_mid,
        ds4_gpu_tensor          *ffn_out,
        ds4_gpu_tensor          *ffn_sum,
        ds4_gpu_tensor          *tmp,
        bool                     stage_profile,
        double                  *stage_t0) {
    uint64_t gate_in = 0, gate_out = 0, gate_row_bytes = 0;
    uint64_t up_in = 0, up_out = 0, up_row_bytes = 0;
    uint64_t down_in = 0, down_out = 0, down_row_bytes = 0;
    (void)tensor_expert_bytes(model, l->ffn_gate_exps, 0, &gate_in, &gate_out, &gate_row_bytes);
    (void)tensor_expert_bytes(model, l->ffn_up_exps, 0, &up_in, &up_out, &up_row_bytes);
    (void)tensor_expert_bytes(model, l->ffn_down_exps, 0, &down_in, &down_out, &down_row_bytes);
    (void)gate_in;
    (void)up_in;
    (void)down_in;
    (void)ffn_gate;
    (void)ffn_up;

    bool ok = metal_graph_matmul_plain_tensor(g->router_logits,
                                              model,
                                              l->ffn_gate_inp,
                                              DS4_N_EMBD,
                                              DS4_N_EXPERT,
                                              ffn_norm,
                                              1);
    ds4_expert_prune_mask_ensure();
    if (ok && g_expert_prune_mask_state == 1) {
        ok = glm_graph_cpu_masked_router_select_batch(g->router_logits,
                                                      g->router_selected,
                                                      g->router_weights,
                                                      g->router_probs,
                                                      model,
                                                      l,
                                                      il,
                                                      1);
    } else if (ok) {
        ok = ds4_gpu_glm_router_select_tensor(g->router_selected,
                                              g->router_weights,
                                              g->router_probs,
                                              model->map,
                                              model->size,
                                              l->ffn_exp_probs_b->abs_offset,
                                              g->router_logits,
                                              DS4_N_EXPERT,
                                              DS4_N_EXPERT_USED,
                                              DS4_EXPERT_WEIGHT_SCALE) != 0;
    }
    if (ok) ok = glm_graph_profile_stage(stage_profile,
                                         "glm_decode_ffn",
                                         "router",
                                         il,
                                         pos,
                                         1,
                                         stage_t0);
    if (ok) ok = glm_graph_profile_router_selection(g, l, il, pos);
    const bool generic_streaming_selected_cache =
        g->ssd_streaming &&
        glm_graph_layer_uses_generic_routed_moe(l);
    const bool uniform_streaming_selected_cache =
        g->ssd_streaming &&
        l->ffn_gate_exps->type == l->ffn_up_exps->type &&
        l->ffn_gate_exps->type == l->ffn_down_exps->type &&
        (l->ffn_gate_exps->type == DS4_TENSOR_Q2_K ||
         l->ffn_gate_exps->type == DS4_TENSOR_Q4_K);
    const bool streaming_selected_cache =
        generic_streaming_selected_cache ||
        uniform_streaming_selected_cache;
    const bool shared_first = streaming_selected_cache;
    metal_graph_selected_async_load async_load = {0};
    bool async_load_started = false;
    const bool async_profile =
        streaming_selected_cache &&
        glm_graph_streaming_async_profile_enabled();
    if (async_profile) glm_graph_streaming_async_profile_register();
    const double stream_total_t0 =
        async_profile ? glm_graph_streaming_async_profile_ms() : 0.0;
    double stream_t0 = stream_total_t0;
    bool async_path_profiled = false;
    if (ok && streaming_selected_cache) {
        const ds4_gpu_stream_expert_table table = {
            .model_map = model->map,
            .model_size = model->size,
            .layer = il,
            .n_total_expert = DS4_N_EXPERT,
            .gate_offset = l->ffn_gate_exps->abs_offset,
            .up_offset = l->ffn_up_exps->abs_offset,
            .down_offset = l->ffn_down_exps->abs_offset,
            .gate_expert_bytes = gate_out * gate_row_bytes,
            .down_expert_bytes = down_out * down_row_bytes,
        };
        const bool async_selected_load =
            glm_graph_layer_uses_generic_routed_moe(l) &&
            glm_graph_use_streaming_selected_async_load(g);
        async_path_profiled = async_profile && async_selected_load;
        if (async_selected_load) {
            uint64_t selected_event = 0;
            ok = ds4_gpu_signal_selected_readback_ready(&selected_event) != 0;
            if (ok) {
                ok = metal_graph_selected_async_load_start_tensor(
                        &async_load,
                        g->router_selected,
                        model,
                        l,
                        il,
                        selected_event,
                        gate_out * gate_row_bytes,
                        down_out * down_row_bytes);
                async_load_started = ok;
            }
            if (async_profile) {
                g_glm_streaming_async_profile.async_signal_start_ms +=
                    glm_graph_streaming_async_profile_ms() - stream_t0;
                stream_t0 = glm_graph_streaming_async_profile_ms();
            }
            if (ok) {
                ok = ds4_gpu_flush_commands() != 0;
            }
            if (async_profile) {
                g_glm_streaming_async_profile.async_flush_router_ms +=
                    glm_graph_streaming_async_profile_ms() - stream_t0;
                stream_t0 = glm_graph_streaming_async_profile_ms();
            }
        } else {
            ok = ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor(
                    &table,
                    g->router_selected,
                    DS4_N_EXPERT_USED) != 0;
            if (ok && glm_router_ahead_prefetch_enabled(g)) {
                /* Commands were just synced by the early-load readback, so
                 * this ffn_norm read is cheap; predict ahead off-thread. */
                const uint32_t lookahead =
                    glm_router_ahead_prefetch_lookahead(g);
                glm_router_ahead_prefetch_submit(g,
                                                 model,
                                                 il + lookahead,
                                                 ffn_norm);
            }
            if (async_profile) {
                g_glm_streaming_async_profile.sync_early_load_ms +=
                    glm_graph_streaming_async_profile_ms() - stream_t0;
                stream_t0 = glm_graph_streaming_async_profile_ms();
            }
        }
    }
    if (ok && shared_first) {
        ok = ds4_gpu_shared_mid_swiglu_q8_0_tensor(
                ffn_mid,
                model->map,
                model->size,
                l->ffn_gate_shexp->abs_offset,
                l->ffn_up_shexp->abs_offset,
                DS4_N_EMBD,
                DS4_N_FF_EXP,
                ffn_norm,
                0.0f) != 0;
        if (ok) ok = glm_graph_profile_stage(stage_profile,
                                             "glm_decode_ffn",
                                             "shared_gate_up_swiglu",
                                             il,
                                             pos,
                                             1,
                                             stage_t0);
        if (ok) ok = glm_graph_matmul_q8_0_decode_profiled_tensor(ffn_sum,
                                                                  model,
                                                                  l->ffn_down_shexp->abs_offset,
                                                                  DS4_N_FF_EXP,
                                                                  DS4_N_EMBD,
                                                                  ffn_mid,
                                                                  il,
                                                                  pos,
                                                                  "shared_down",
                                                                  g->ssd_streaming) != 0;
        if (ok) ok = glm_graph_profile_stage(stage_profile,
                                             "glm_decode_ffn",
                                             "shared_down",
                                             il,
                                             pos,
                                             1,
                                             stage_t0);
    }
    if (async_profile) {
        const double now_ms = glm_graph_streaming_async_profile_ms();
        if (async_path_profiled) {
            g_glm_streaming_async_profile.async_shared_ms +=
                now_ms - stream_t0;
        } else {
            g_glm_streaming_async_profile.sync_shared_ms +=
                now_ms - stream_t0;
        }
        stream_t0 = now_ms;
    }
    if (async_load_started) {
        const bool flush_ok = ds4_gpu_flush_commands() != 0;
        if (async_profile) {
            g_glm_streaming_async_profile.async_flush_shared_ms +=
                glm_graph_streaming_async_profile_ms() - stream_t0;
            stream_t0 = glm_graph_streaming_async_profile_ms();
        }
        const bool finish_ok = metal_graph_selected_async_load_finish(&async_load);
        ok = ok && flush_ok && finish_ok;
        if (async_profile) {
            g_glm_streaming_async_profile.async_finish_ms +=
                glm_graph_streaming_async_profile_ms() - stream_t0;
            stream_t0 = glm_graph_streaming_async_profile_ms();
        }
    }
    if (ok) ok = glm_graph_routed_moe_one_dispatch(
            g,
            model,
            l,
            il,
            ffn_out,
            ffn_mid,
            gate_out * gate_row_bytes,
            gate_row_bytes,
            up_out * up_row_bytes,
            up_row_bytes,
            down_out * down_row_bytes,
            down_row_bytes,
            g->router_selected,
            g->router_weights,
            ffn_norm,
            false) != 0;
    if (async_profile) {
        const double now_ms = glm_graph_streaming_async_profile_ms();
        if (async_path_profiled) {
            g_glm_streaming_async_profile.async_routed_ms +=
                now_ms - stream_t0;
        } else {
            g_glm_streaming_async_profile.sync_routed_ms +=
                now_ms - stream_t0;
        }
        stream_t0 = now_ms;
    }
    if (ok) ok = glm_graph_profile_stage(stage_profile,
                                         "glm_decode_ffn",
                                         "routed_moe",
                                         il,
                                         pos,
                                         1,
                                         stage_t0);
    if (ok && !shared_first) {
        ok = ds4_gpu_shared_mid_swiglu_q8_0_tensor(
                ffn_mid,
                model->map,
                model->size,
                l->ffn_gate_shexp->abs_offset,
                l->ffn_up_shexp->abs_offset,
                DS4_N_EMBD,
                DS4_N_FF_EXP,
                ffn_norm,
                0.0f) != 0;
        if (ok) ok = glm_graph_profile_stage(stage_profile,
                                             "glm_decode_ffn",
                                             "shared_gate_up_swiglu",
                                             il,
                                             pos,
                                             1,
                                             stage_t0);
        if (ok) ok = glm_graph_matmul_q8_0_decode_profiled_tensor(ffn_sum,
                                                                  model,
                                                                  l->ffn_down_shexp->abs_offset,
                                                                  DS4_N_FF_EXP,
                                                                  DS4_N_EMBD,
                                                                  ffn_mid,
                                                                  il,
                                                                  pos,
                                                                  "shared_down",
                                                                  g->ssd_streaming) != 0;
        if (ok) ok = glm_graph_profile_stage(stage_profile,
                                             "glm_decode_ffn",
                                             "shared_down",
                                             il,
                                             pos,
                                             1,
                                             stage_t0);
    }
    if (ok && !glm_graph_disable_add3_residual()) {
        ok = ds4_gpu_add3_tensor(next,
                                 after_attn,
                                 ffn_out,
                                 ffn_sum,
                                 DS4_N_EMBD) != 0;
    } else if (ok) {
        ok = ds4_gpu_add_tensor(tmp,
                                ffn_out,
                                ffn_sum,
                                DS4_N_EMBD) != 0;
        if (ok) ok = ds4_gpu_add_tensor(next,
                                        after_attn,
                                        tmp,
                                        DS4_N_EMBD) != 0;
    }
    if (ok) ok = glm_graph_profile_stage(stage_profile,
                                         "glm_decode_ffn",
                                         "residual",
                                         il,
                                         pos,
                                         1,
                                         stage_t0);
    if (async_profile) {
        const double now_ms = glm_graph_streaming_async_profile_ms();
        if (async_path_profiled) {
            g_glm_streaming_async_profile.async_calls++;
            g_glm_streaming_async_profile.async_post_ms += now_ms - stream_t0;
            g_glm_streaming_async_profile.async_total_ms +=
                now_ms - stream_total_t0;
        } else {
            g_glm_streaming_async_profile.sync_calls++;
            g_glm_streaming_async_profile.sync_post_ms += now_ms - stream_t0;
            g_glm_streaming_async_profile.sync_total_ms +=
                now_ms - stream_total_t0;
        }
    }
    return ok;
}

static bool glm_graph_encode_ffn_one_normed_from(
        ds4_glm_gpu_graph       *g,
        const ds4_model         *model,
        const ds4_layer_weights *l,
        uint32_t                 il,
        uint32_t                 pos,
        const ds4_gpu_tensor    *ffn_norm,
        const ds4_gpu_tensor    *after_attn,
        ds4_gpu_tensor          *next,
        ds4_gpu_tensor          *ffn_gate,
        ds4_gpu_tensor          *ffn_up,
        ds4_gpu_tensor          *ffn_mid,
        ds4_gpu_tensor          *ffn_out,
        ds4_gpu_tensor          *ffn_sum,
        ds4_gpu_tensor          *tmp,
        bool                     stage_profile,
        double                  *stage_t0) {
    if (!g || !model || !l || !ffn_norm || !after_attn || !next ||
        !ffn_gate || !ffn_up || !ffn_mid || !ffn_out ||
        !ffn_sum || !tmp) {
        return false;
    }

    if (il < DS4_N_LEADING_DENSE) {
        const uint64_t hidden = l->ffn_gate->dim[1];
        const bool fused_gate_up = ds4_gpu_shared_gate_up_swiglu_q8_0_model_view_tensor(
                ffn_gate,
                ffn_up,
                ffn_mid,
                model->map,
                model->size,
                l->ffn_gate->abs_offset,
                l->ffn_up->abs_offset,
                DS4_N_EMBD,
                hidden,
                ffn_norm,
                0.0f) != 0;
        bool ok = fused_gate_up;
        if (fused_gate_up) {
            ok = glm_graph_profile_stage(stage_profile,
                                         "glm_decode_ffn",
                                         "dense_gate_up_swiglu",
                                         il,
                                         pos,
                                         1,
                                         stage_t0);
        } else {
            ok = glm_graph_matmul_q8_0_decode_profiled_tensor(ffn_gate,
                                                              model,
                                                              l->ffn_gate->abs_offset,
                                                              DS4_N_EMBD,
                                                              hidden,
                                                              ffn_norm,
                                                              il,
                                                              pos,
                                                              "dense_gate",
                                                              g->ssd_streaming) != 0;
            if (ok) ok = glm_graph_matmul_q8_0_decode_profiled_tensor(ffn_up,
                                                                      model,
                                                                      l->ffn_up->abs_offset,
                                                                      DS4_N_EMBD,
                                                                      hidden,
                                                                      ffn_norm,
                                                                      il,
                                                                      pos,
                                                                      "dense_up",
                                                                      g->ssd_streaming) != 0;
            if (ok) ok = glm_graph_profile_stage(stage_profile,
                                                 "glm_decode_ffn",
                                                 "dense_gate_up",
                                                 il,
                                                 pos,
                                                 1,
                                                 stage_t0);
            if (ok) ok = ds4_gpu_swiglu_tensor(ffn_mid,
                                               ffn_gate,
                                               ffn_up,
                                               (uint32_t)hidden,
                                               0.0f,
                                               1.0f) != 0;
            if (ok) ok = glm_graph_profile_stage(stage_profile,
                                                 "glm_decode_ffn",
                                                 "dense_swiglu",
                                                 il,
                                                 pos,
                                                 1,
                                                 stage_t0);
        }
        if (ok) ok = glm_graph_matmul_q8_0_decode_profiled_tensor(ffn_out,
                                                                  model,
                                                                  l->ffn_down->abs_offset,
                                                                  hidden,
                                                                  DS4_N_EMBD,
                                                                  ffn_mid,
                                                                  il,
                                                                  pos,
                                                                  "dense_down",
                                                                  g->ssd_streaming) != 0;
        if (ok) ok = glm_graph_profile_stage(stage_profile,
                                             "glm_decode_ffn",
                                             "dense_down",
                                             il,
                                             pos,
                                             1,
                                             stage_t0);
        if (ok) ok = ds4_gpu_add_tensor(next,
                                        after_attn,
                                        ffn_out,
                                        DS4_N_EMBD) != 0;
        if (ok) ok = glm_graph_profile_stage(stage_profile,
                                             "glm_decode_ffn",
                                             "residual",
                                             il,
                                             pos,
                                             1,
                                             stage_t0);
        return ok;
    }

    return glm_graph_encode_sparse_ffn_one(g,
                                           model,
                                           l,
                                           il,
                                           pos,
                                           ffn_norm,
                                           after_attn,
                                           next,
                                           ffn_gate,
                                           ffn_up,
                                           ffn_mid,
                                           ffn_out,
                                           ffn_sum,
                                           tmp,
                                           stage_profile,
                                           stage_t0);
}

static bool glm_graph_encode_ffn_one_from(
        ds4_glm_gpu_graph       *g,
        const ds4_model         *model,
        const ds4_layer_weights *l,
        uint32_t                 il,
        uint32_t                 pos,
        const ds4_gpu_tensor    *after_attn,
        ds4_gpu_tensor          *next,
        ds4_gpu_tensor          *ffn_norm,
        ds4_gpu_tensor          *ffn_gate,
        ds4_gpu_tensor          *ffn_up,
        ds4_gpu_tensor          *ffn_mid,
        ds4_gpu_tensor          *ffn_out,
        ds4_gpu_tensor          *ffn_sum,
        ds4_gpu_tensor          *tmp,
        bool                     stage_profile,
        double                  *stage_t0) {
    if (!g || !model || !l || !after_attn || !next ||
        !ffn_norm || !ffn_gate || !ffn_up || !ffn_mid || !ffn_out ||
        !ffn_sum || !tmp) {
        return false;
    }

    bool ok = ds4_gpu_rms_norm_weight_tensor(ffn_norm,
                                             after_attn,
                                             model->map,
                                             model->size,
                                             l->ffn_norm->abs_offset,
                                             DS4_N_EMBD,
                                             DS4_RMS_EPS) != 0;
    if (ok) ok = glm_graph_profile_stage(stage_profile,
                                         "glm_decode_ffn",
                                         "ffn_norm",
                                         il,
                                         pos,
                                         1,
                                         stage_t0);
    if (!ok) return false;

    return glm_graph_encode_ffn_one_normed_from(g,
                                                model,
                                                l,
                                                il,
                                                pos,
                                                ffn_norm,
                                                after_attn,
                                                next,
                                                ffn_gate,
                                                ffn_up,
                                                ffn_mid,
                                                ffn_out,
                                                ffn_sum,
                                                tmp,
                                                stage_profile,
                                                stage_t0);
}

static ds4_gpu_tensor *glm_graph_tensor_row_view_strided(
        ds4_gpu_tensor *base,
        uint32_t        row,
        uint64_t        stride_values,
        uint64_t        row_values) {
    return ds4_gpu_tensor_view(base,
                               (uint64_t)row * stride_values * sizeof(float),
                               row_values * sizeof(float));
}

static uint32_t glm_graph_q8_stripe_tokens(void) {
    return 2048u;
}

static bool glm_graph_flash_attention_prefill_enabled(void) {
    return true;
}

static uint32_t glm_graph_flash_attention_prefill_min_tokens(void) {
    return 24u;
}

static bool glm_graph_use_flash_attention_prefill(uint32_t n_tokens) {
    return glm_graph_flash_attention_prefill_enabled() &&
           n_tokens >= glm_graph_flash_attention_prefill_min_tokens();
}

static bool glm_graph_use_flash_attention_staged_kv(
        uint32_t pos0,
        uint32_t n_tokens,
        uint32_t cache_len) {
    return pos0 == 0 &&
           n_tokens == cache_len;
}

static bool glm_graph_force_indexed_decode(void) {
    return false;
}

static bool glm_graph_disable_indexed_decode(void) {
    return false;
}

static bool glm_graph_decode_uses_indexed_attention(const ds4_glm_gpu_graph *g,
                                                    uint32_t                 pos,
                                                    const float             *logits_out) {
    return g && g->compact_cache_cap != 0 &&
           (!g->full_kv_cache ||
            pos >= g->ctx_cap ||
            glm_graph_force_indexed_decode() ||
            (logits_out != NULL && !glm_graph_disable_indexed_decode()));
}

static bool glm_graph_decode_updates_dense_cache(const ds4_glm_gpu_graph *g,
                                                 uint32_t                 pos,
                                                 const float             *logits_out) {
    return g && pos < g->ctx_cap &&
           !glm_graph_decode_uses_indexed_attention(g, pos, logits_out);
}

static bool glm_graph_indexed_prefill_scalar_kernels(void) {
    return false;
}

static bool glm_graph_indexed_prefill_scalar_indexer(void) {
    return false;
}

static bool glm_graph_indexed_prefill_batch_indexer(void) {
    return true;
}

static bool glm_graph_indexed_prefill_scalar_attn(void) {
    return false;
}

static bool glm_graph_indexed_prefill_batch_qk_low(void) {
    return true;
}

static bool glm_graph_indexed_prefill_batch_attn_kernel(void) {
    return true;
}

static uint32_t glm_graph_indexed_prefill_batch_attn_slice_tokens(void) {
    return 2048u;
}

static bool glm_graph_indexer_qat(void) {
    return false;
}

static bool glm_graph_indexed_prefill_batch_ffn(void) {
    return true;
}

static bool glm_graph_indexed_prefill_batch_ffn_norm(void) {
    return true;
}

static bool glm_graph_indexed_prefill_batch_routed_moe(void) {
    return true;
}

static bool glm_graph_indexed_prefill_batch_router_select(void) {
    return true;
}

static bool glm_graph_indexed_prefill_batch_residual(void) {
    return true;
}

static bool glm_graph_indexed_prefill_batch_f32_rows(void) {
    return true;
}

static bool glm_graph_indexed_prefill_batch_q8_rows(void) {
    return true;
}

static bool glm_graph_indexed_prefill_batch_shared_expert(void) {
    return true;
}

static bool glm_graph_matmul_q8_0_tensor(
        ds4_gpu_tensor       *out,
        const ds4_model      *model,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_tokens) {
    if (!out || !model || !x || n_tokens == 0) return false;

    const uint32_t q8_stripe_tokens = glm_graph_q8_stripe_tokens();
    if (n_tokens <= q8_stripe_tokens) {
        return ds4_gpu_matmul_q8_0_tensor(out,
                                          model->map,
                                          model->size,
                                          weight_offset,
                                          in_dim,
                                          out_dim,
                                          x,
                                          n_tokens) != 0;
    }
    if (in_dim > UINT64_MAX / sizeof(float) ||
        out_dim > UINT64_MAX / sizeof(float)) {
        return false;
    }

    uint32_t done = 0;
    while (done < n_tokens) {
        uint32_t chunk = n_tokens - done;
        if (chunk > q8_stripe_tokens) chunk = q8_stripe_tokens;
        /*
         * The Q8 prefill TensorOps path needs token counts divisible by 32.
         * For a final chunk like 1736 rows, keep the aligned 1728 rows on that
         * path and leave only the tiny tail to the small-batch kernel.  Avoid
         * splitting small batches where the extra launch would dominate.
         */
        if (chunk >= 256u && chunk == n_tokens - done) {
            const uint32_t tail = chunk & 31u;
            if (tail != 0u && tail <= 16u) chunk -= tail;
        }

        ds4_gpu_tensor *x_view = ds4_gpu_tensor_view(
                x,
                (uint64_t)done * in_dim * sizeof(float),
                (uint64_t)chunk * in_dim * sizeof(float));
        ds4_gpu_tensor *out_view = ds4_gpu_tensor_view(
                out,
                (uint64_t)done * out_dim * sizeof(float),
                (uint64_t)chunk * out_dim * sizeof(float));
        const bool ok = x_view && out_view &&
            ds4_gpu_matmul_q8_0_tensor(out_view,
                                       model->map,
                                       model->size,
                                       weight_offset,
                                       in_dim,
                                       out_dim,
                                       x_view,
                                       chunk) != 0;
        ds4_gpu_tensor_free(out_view);
        ds4_gpu_tensor_free(x_view);
        if (!ok) return false;
        done += chunk;
    }
    return true;
}

static bool glm_graph_matmul_q8_0_rows_scalar(
        ds4_gpu_tensor       *out,
        const ds4_model      *model,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_tokens) {
    if (!out || !model || !x || n_tokens == 0) return false;
    if (in_dim > UINT64_MAX / sizeof(float) ||
        out_dim > UINT64_MAX / sizeof(float)) {
        return false;
    }

    if (glm_graph_indexed_prefill_batch_q8_rows() &&
        ds4_gpu_matmul_q8_0_rows_scalar_tensor(out,
                                               model->map,
                                               model->size,
                                               weight_offset,
                                               in_dim,
                                               out_dim,
                                               x,
                                               n_tokens) != 0) {
        return true;
    }

    for (uint32_t t = 0; t < n_tokens; t++) {
        ds4_gpu_tensor *x_view = ds4_gpu_tensor_view(
                (ds4_gpu_tensor *)x,
                (uint64_t)t * in_dim * sizeof(float),
                in_dim * sizeof(float));
        ds4_gpu_tensor *out_view = ds4_gpu_tensor_view(
                out,
                (uint64_t)t * out_dim * sizeof(float),
                out_dim * sizeof(float));
        const bool ok = x_view && out_view &&
            ds4_gpu_matmul_q8_0_tensor(out_view,
                                       model->map,
                                       model->size,
                                       weight_offset,
                                       in_dim,
                                       out_dim,
                                       x_view,
                                       1) != 0;
        ds4_gpu_tensor_free(out_view);
        ds4_gpu_tensor_free(x_view);
        if (!ok) return false;
    }
    return true;
}

static bool glm_graph_matmul_f32_rows_scalar(
        ds4_gpu_tensor       *out,
        const ds4_model      *model,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_tokens) {
    if (!out || !model || !x || n_tokens == 0) return false;
    if (in_dim > UINT64_MAX / sizeof(float) ||
        out_dim > UINT64_MAX / sizeof(float)) {
        return false;
    }

    if (glm_graph_indexed_prefill_batch_f32_rows()) {
        return ds4_gpu_matmul_f32_tensor(out,
                                         model->map,
                                         model->size,
                                         weight_offset,
                                         in_dim,
                                         out_dim,
                                         x,
                                         n_tokens) != 0;
    }

    for (uint32_t t = 0; t < n_tokens; t++) {
        ds4_gpu_tensor *x_view = ds4_gpu_tensor_view(
                (ds4_gpu_tensor *)x,
                (uint64_t)t * in_dim * sizeof(float),
                in_dim * sizeof(float));
        ds4_gpu_tensor *out_view = ds4_gpu_tensor_view(
                out,
                (uint64_t)t * out_dim * sizeof(float),
                out_dim * sizeof(float));
        const bool ok = x_view && out_view &&
            ds4_gpu_matmul_f32_tensor(out_view,
                                      model->map,
                                      model->size,
                                      weight_offset,
                                      in_dim,
                                      out_dim,
                                      x_view,
                                      1) != 0;
        ds4_gpu_tensor_free(out_view);
        ds4_gpu_tensor_free(x_view);
        if (!ok) return false;
    }
    return true;
}

/* Row-scalar matmul against a weight tensor whose GGUF type varies across
 * model exports (F32, F16 or Q8_0), e.g. the GLM router and indexer proj. */
static bool glm_graph_matmul_plain_rows_scalar(
        ds4_gpu_tensor       *out,
        const ds4_model      *model,
        const ds4_tensor     *w,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_tokens) {
    if (!out || !model || !w || !x || n_tokens == 0) return false;
    if (w->type == DS4_TENSOR_Q8_0) {
        return glm_graph_matmul_q8_0_rows_scalar(out,
                                                 model,
                                                 w->abs_offset,
                                                 in_dim,
                                                 out_dim,
                                                 x,
                                                 n_tokens);
    }
    if (w->type == DS4_TENSOR_F32) {
        return glm_graph_matmul_f32_rows_scalar(out,
                                                model,
                                                w->abs_offset,
                                                in_dim,
                                                out_dim,
                                                x,
                                                n_tokens);
    }
    if (w->type == DS4_TENSOR_F16) {
        return metal_graph_matmul_plain_tensor(out,
                                               model,
                                               w,
                                               in_dim,
                                               out_dim,
                                               x,
                                               n_tokens);
    }
    fprintf(stderr, "ds4: GLM rows-scalar matmul does not support %s\n",
            tensor_type_name(w->type));
    return false;
}

static bool glm_graph_shared_gate_up_swiglu_q8_0_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const ds4_model      *model,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_tokens,
        float                 clamp) {
    if (!gate || !up || !mid || !model || !x || n_tokens == 0) return false;

    const uint32_t q8_stripe_tokens = glm_graph_q8_stripe_tokens();
    if (n_tokens == 1) {
        return ds4_gpu_shared_gate_up_swiglu_q8_0_model_view_tensor(gate,
                                                                    up,
                                                                    mid,
                                                                    model->map,
                                                                    model->size,
                                                                    gate_offset,
                                                                    up_offset,
                                                                    in_dim,
                                                                    out_dim,
                                                                    x,
                                                                    clamp) != 0;
    }
    if (n_tokens <= q8_stripe_tokens) {
        return ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor(gate,
                                                              up,
                                                              mid,
                                                              model->map,
                                                              model->size,
                                                              gate_offset,
                                                              up_offset,
                                                              in_dim,
                                                              out_dim,
                                                              x,
                                                              n_tokens,
                                                              clamp) != 0;
    }
    if (in_dim > UINT64_MAX / sizeof(float) ||
        out_dim > UINT64_MAX / sizeof(float)) {
        return false;
    }

    uint32_t done = 0;
    while (done < n_tokens) {
        uint32_t chunk = n_tokens - done;
        if (chunk > q8_stripe_tokens) chunk = q8_stripe_tokens;

        ds4_gpu_tensor *x_view = ds4_gpu_tensor_view(
                x,
                (uint64_t)done * in_dim * sizeof(float),
                (uint64_t)chunk * in_dim * sizeof(float));
        ds4_gpu_tensor *gate_view = ds4_gpu_tensor_view(
                gate,
                (uint64_t)done * out_dim * sizeof(float),
                (uint64_t)chunk * out_dim * sizeof(float));
        ds4_gpu_tensor *up_view = ds4_gpu_tensor_view(
                up,
                (uint64_t)done * out_dim * sizeof(float),
                (uint64_t)chunk * out_dim * sizeof(float));
        ds4_gpu_tensor *mid_view = ds4_gpu_tensor_view(
                mid,
                (uint64_t)done * out_dim * sizeof(float),
                (uint64_t)chunk * out_dim * sizeof(float));
        const bool ok = x_view && gate_view && up_view && mid_view &&
            ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor(gate_view,
                                                           up_view,
                                                           mid_view,
                                                           model->map,
                                                           model->size,
                                                           gate_offset,
                                                           up_offset,
                                                           in_dim,
                                                           out_dim,
                                                           x_view,
                                                           chunk,
                                                           clamp) != 0;
        ds4_gpu_tensor_free(mid_view);
        ds4_gpu_tensor_free(up_view);
        ds4_gpu_tensor_free(gate_view);
        ds4_gpu_tensor_free(x_view);
        if (!ok) return false;
        done += chunk;
    }
    return true;
}

static bool glm_graph_indexed_prefill_grouped_moe_default(
        const ds4_glm_gpu_graph *g) {
    return g &&
           !g->quality;
}

static bool glm_graph_disable_add3_residual(void);

static bool glm_graph_encode_sparse_ffn_indexed_batch_routed_moe(
        ds4_glm_gpu_graph       *g,
        const ds4_model         *model,
        const ds4_layer_weights *l,
        uint32_t                 il,
        uint32_t                 pos0,
        const ds4_gpu_tensor    *after_attn,
        ds4_gpu_tensor          *next,
        uint32_t                 n_tokens,
        bool                     stage_profile,
        bool                     stage_sync,
        double                  *stage_t0) {
    if (!g || !model || !l || !after_attn || !next ||
        !g->batch_ffn_norm ||
        !g->batch_router_logits ||
        !g->batch_router_probs ||
        !g->batch_router_selected ||
        !g->batch_router_weights ||
        !g->batch_ffn_out ||
        !g->batch_ffn_mid ||
        n_tokens <= 1 ||
        il < DS4_N_LEADING_DENSE ||
        g->ffn_mid_elems > UINT32_MAX) {
        return false;
    }

    uint64_t gate_in = 0, gate_out = 0, gate_row_bytes = 0;
    uint64_t up_in = 0, up_out = 0, up_row_bytes = 0;
    uint64_t down_in = 0, down_out = 0, down_row_bytes = 0;
    (void)tensor_expert_bytes(model, l->ffn_gate_exps, 0, &gate_in, &gate_out, &gate_row_bytes);
    (void)tensor_expert_bytes(model, l->ffn_up_exps, 0, &up_in, &up_out, &up_row_bytes);
    (void)tensor_expert_bytes(model, l->ffn_down_exps, 0, &down_in, &down_out, &down_row_bytes);
    (void)gate_in;
    (void)up_in;
    (void)down_in;

    bool ok = glm_graph_matmul_plain_rows_scalar(g->batch_router_logits,
                                                 model,
                                                 l->ffn_gate_inp,
                                                 DS4_N_EMBD,
                                                 DS4_N_EXPERT,
                                                 g->batch_ffn_norm,
                                                 n_tokens);
    ds4_expert_prune_mask_ensure();
    const bool masked_router_select = g_expert_prune_mask_state == 1;
    if (ok && masked_router_select) {
        ok = glm_graph_cpu_masked_router_select_batch(g->batch_router_logits,
                                                      g->batch_router_selected,
                                                      g->batch_router_weights,
                                                      g->batch_router_probs,
                                                      model,
                                                      l,
                                                      il,
                                                      n_tokens);
    }
    const bool use_batch_router_select =
        !masked_router_select &&
        glm_graph_indexed_prefill_batch_router_select();
    if (ok && use_batch_router_select) {
        ok = ds4_gpu_glm_router_select_batch_tensor(g->batch_router_selected,
                                                    g->batch_router_weights,
                                                    g->batch_router_probs,
                                                    model->map,
                                                    model->size,
                                                    l->ffn_exp_probs_b->abs_offset,
                                                    g->batch_router_logits,
                                                    DS4_N_EXPERT,
                                                    DS4_N_EXPERT_USED,
                                                    DS4_EXPERT_WEIGHT_SCALE,
                                                    n_tokens) != 0;
    }
    for (uint32_t t = 0;
         ok && !masked_router_select && !use_batch_router_select &&
         t < n_tokens;
         t++) {
        ds4_gpu_tensor *logits_view =
            glm_graph_tensor_row_view_strided(g->batch_router_logits,
                                              t,
                                              DS4_N_EXPERT,
                                              DS4_N_EXPERT);
        ds4_gpu_tensor *probs_view =
            glm_graph_tensor_row_view_strided(g->batch_router_probs,
                                              t,
                                              DS4_N_EXPERT,
                                              DS4_N_EXPERT);
        ds4_gpu_tensor *selected_view =
            ds4_gpu_tensor_view(g->batch_router_selected,
                                (uint64_t)t * DS4_N_EXPERT_USED * sizeof(int32_t),
                                (uint64_t)DS4_N_EXPERT_USED * sizeof(int32_t));
        ds4_gpu_tensor *weights_view =
            glm_graph_tensor_row_view_strided(g->batch_router_weights,
                                              t,
                                              DS4_N_EXPERT_USED,
                                              DS4_N_EXPERT_USED);
        ok = logits_view && probs_view && selected_view && weights_view;
        if (ok) {
            ok = ds4_gpu_glm_router_select_tensor(selected_view,
                                                  weights_view,
                                                  probs_view,
                                                  model->map,
                                                  model->size,
                                                  l->ffn_exp_probs_b->abs_offset,
                                                  logits_view,
                                                  DS4_N_EXPERT,
                                                  DS4_N_EXPERT_USED,
                                                  DS4_EXPERT_WEIGHT_SCALE) != 0;
        }
        ds4_gpu_tensor_free(weights_view);
        ds4_gpu_tensor_free(selected_view);
        ds4_gpu_tensor_free(probs_view);
        ds4_gpu_tensor_free(logits_view);
    }
    if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                  stage_sync,
                                                  "glm_indexed_ffn",
                                                  "router",
                                                  il,
                                                  pos0,
                                                  n_tokens,
                                                  stage_t0);
    if (ok) ok = glm_graph_profile_router_selection_batch(g,
                                                          l,
                                                          il,
                                                          pos0,
                                                          n_tokens);
    metal_graph_debug_dump_tensor("glm_indexed_router_logits",
                                  g->batch_router_logits,
                                  (uint64_t)n_tokens * DS4_N_EXPERT,
                                  il,
                                  pos0);
    metal_graph_debug_dump_i32_tensor("glm_indexed_router_selected",
                                      g->batch_router_selected,
                                      (uint64_t)n_tokens * DS4_N_EXPERT_USED,
                                      il,
                                      pos0);
    metal_graph_debug_dump_tensor("glm_indexed_router_weights",
                                  g->batch_router_weights,
                                  (uint64_t)n_tokens * DS4_N_EXPERT_USED,
                                  il,
                                  pos0);

    if (ok) {
        const bool use_grouped_moe =
            glm_graph_indexed_prefill_grouped_moe_default(g);
        ok = glm_graph_routed_moe_batch_dispatch(
                g,
                model,
                l,
                il,
                g->batch_ffn_out,
                g->batch_ffn_mid,
                gate_out * gate_row_bytes,
                gate_row_bytes,
                up_out * up_row_bytes,
                up_row_bytes,
                down_out * down_row_bytes,
                down_row_bytes,
                g->batch_router_selected,
                g->batch_router_weights,
                g->batch_ffn_norm,
                n_tokens,
                (uint32_t)g->ffn_mid_elems,
                !use_grouped_moe) != 0;
    }
    if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                  stage_sync,
                                                  "glm_indexed_ffn",
                                                  "routed_moe",
                                                  il,
                                                  pos0,
                                                  n_tokens,
                                                  stage_t0);
    metal_graph_debug_dump_tensor("glm_indexed_routed_out",
                                  g->batch_ffn_out,
                                  (uint64_t)n_tokens * DS4_N_EMBD,
                                  il,
                                  pos0);

    const bool use_batch_residual =
        glm_graph_indexed_prefill_batch_residual();
    const uint64_t residual_elems = (uint64_t)n_tokens * DS4_N_EMBD;
    if (use_batch_residual && residual_elems > UINT32_MAX) return false;

    bool shared_expert_done = false;
    if (ok &&
        use_batch_residual &&
        glm_graph_indexed_prefill_batch_shared_expert() &&
        g->batch_ffn_gate &&
        g->batch_ffn_up &&
        g->batch_shared_mid &&
        ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor(
                g->batch_ffn_gate,
                g->batch_ffn_up,
                g->batch_shared_mid,
                model->map,
                model->size,
                l->ffn_gate_shexp->abs_offset,
                l->ffn_up_shexp->abs_offset,
                DS4_N_EMBD,
                DS4_N_FF_EXP,
                g->batch_ffn_norm,
                n_tokens,
                0.0f) != 0) {
        shared_expert_done =
            glm_graph_matmul_q8_0_rows_scalar(g->batch_attn_out,
                                              model,
                                              l->ffn_down_shexp->abs_offset,
                                              DS4_N_FF_EXP,
                                              DS4_N_EMBD,
                                              g->batch_shared_mid,
                                              n_tokens);
    }

    for (uint32_t t = 0; ok && !shared_expert_done && t < n_tokens; t++) {
        ds4_gpu_tensor *ffn_norm_view =
            glm_graph_tensor_row_view_strided(g->batch_ffn_norm,
                                              t,
                                              DS4_N_EMBD,
                                              DS4_N_EMBD);
        ds4_gpu_tensor *shared_out_view = use_batch_residual ?
            glm_graph_tensor_row_view_strided(g->batch_attn_out,
                                              t,
                                              DS4_N_EMBD,
                                              DS4_N_EMBD) :
            NULL;
        ds4_gpu_tensor *after_attn_view = !use_batch_residual ?
            glm_graph_tensor_row_view_strided((ds4_gpu_tensor *)after_attn,
                                              t,
                                              DS4_N_EMBD,
                                              DS4_N_EMBD) :
            NULL;
        ds4_gpu_tensor *routed_out_view = !use_batch_residual ?
            glm_graph_tensor_row_view_strided(g->batch_ffn_out,
                                              t,
                                              DS4_N_EMBD,
                                              DS4_N_EMBD) :
            NULL;
        ds4_gpu_tensor *next_view = !use_batch_residual ?
            glm_graph_tensor_row_view_strided(next,
                                              t,
                                              DS4_N_EMBD,
                                              DS4_N_EMBD) :
            NULL;
        ok = ffn_norm_view &&
             (use_batch_residual ? (shared_out_view != NULL) :
                                   (after_attn_view && routed_out_view && next_view));
        if (ok) {
            ok = ds4_gpu_shared_gate_up_swiglu_q8_0_model_view_tensor(
                    g->ffn_gate,
                    g->ffn_up,
                    g->ffn_mid,
                    model->map,
                    model->size,
                    l->ffn_gate_shexp->abs_offset,
                    l->ffn_up_shexp->abs_offset,
                    DS4_N_EMBD,
                    DS4_N_FF_EXP,
                    ffn_norm_view,
                    0.0f) != 0;
        }
        if (ok) ok = ds4_gpu_matmul_q8_0_tensor(use_batch_residual ?
                                                    shared_out_view :
                                                    g->ffn_sum,
                                                model->map,
                                                model->size,
                                                l->ffn_down_shexp->abs_offset,
                                                DS4_N_FF_EXP,
                                                DS4_N_EMBD,
                                                g->ffn_mid,
                                                1) != 0;
        if (ok && !use_batch_residual) {
            ok = ds4_gpu_add_tensor(g->attn_out,
                                    routed_out_view,
                                    g->ffn_sum,
                                    DS4_N_EMBD) != 0;
        }
        if (ok && !use_batch_residual) {
            ok = ds4_gpu_add_tensor(next_view,
                                    after_attn_view,
                                    g->attn_out,
                                    DS4_N_EMBD) != 0;
        }

        ds4_gpu_tensor_free(next_view);
        ds4_gpu_tensor_free(routed_out_view);
        ds4_gpu_tensor_free(shared_out_view);
        ds4_gpu_tensor_free(ffn_norm_view);
        ds4_gpu_tensor_free(after_attn_view);
    }
    if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                  stage_sync,
                                                  "glm_indexed_ffn",
                                                  "shared_expert",
                                                  il,
                                                  pos0,
                                                  n_tokens,
                                                  stage_t0);
    if (ok && use_batch_residual) {
        if (!glm_graph_disable_add3_residual()) {
            ok = ds4_gpu_add3_tensor(next,
                                     after_attn,
                                     g->batch_ffn_out,
                                     g->batch_attn_out,
                                     (uint32_t)residual_elems) != 0;
        } else {
            ok = ds4_gpu_add_tensor(g->batch_heads,
                                    g->batch_ffn_out,
                                    g->batch_attn_out,
                                    (uint32_t)residual_elems) != 0;
            if (ok) ok = ds4_gpu_add_tensor(next,
                                            after_attn,
                                            g->batch_heads,
                                            (uint32_t)residual_elems) != 0;
        }
    }
    if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                  stage_sync,
                                                  "glm_indexed_ffn",
                                                  "residual",
                                                  il,
                                                  pos0,
                                                  n_tokens,
                                                  stage_t0);
    metal_graph_debug_dump_tensor("glm_indexed_next",
                                  next,
                                  (uint64_t)n_tokens * DS4_N_EMBD,
                                  il,
                                  pos0);

    (void)pos0;
    return ok;
}

static bool glm_graph_upload_tokens(
        ds4_gpu_tensor *out_tokens,
        const int      *tokens,
        uint32_t        n_tokens) {
    if (!out_tokens || !tokens || n_tokens == 0) return false;

    int32_t *ids = xmalloc((size_t)n_tokens * sizeof(ids[0]));
    for (uint32_t i = 0; i < n_tokens; i++) ids[i] = (int32_t)tokens[i];
    const bool ok = ds4_gpu_tensor_write(out_tokens,
                                         0,
                                         ids,
                                         (uint64_t)n_tokens * sizeof(ids[0])) != 0;
    free(ids);
    return ok;
}

static bool glm_graph_disable_add3_residual(void) {
    return false;
}

static bool glm_graph_encode_ffn_batch(
        ds4_glm_gpu_graph       *g,
        const ds4_model         *model,
        const ds4_layer_weights *l,
        uint32_t                 il,
        uint32_t                 pos0,
        ds4_gpu_tensor          *after_attn,
        ds4_gpu_tensor          *next,
        uint32_t                 n_tokens,
        bool                     stage_profile,
        bool                     stage_sync,
        double                  *stage_t0) {
    if (!g || !model || !l || !after_attn || !next || n_tokens == 0) return false;

    bool ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_ffn_norm,
                                                  after_attn,
                                                  model->map,
                                                  model->size,
                                                  l->ffn_norm->abs_offset,
                                                  DS4_N_EMBD,
                                                  n_tokens,
                                                  DS4_RMS_EPS) != 0;
    if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                  stage_sync,
                                                  "glm_ffn",
                                                  "ffn_norm",
                                                  il,
                                                  pos0,
                                                  n_tokens,
                                                  stage_t0);
    if (!ok) return false;

    if (il < DS4_N_LEADING_DENSE) {
        const uint64_t hidden = l->ffn_gate->dim[1];
        if (hidden == 0 || hidden > UINT32_MAX / n_tokens) return false;
        const uint32_t mid_elems = (uint32_t)(hidden * n_tokens);
        const uint64_t residual_elems = (uint64_t)n_tokens * DS4_N_EMBD;
        if (residual_elems > UINT32_MAX) return false;

        const bool fused_gate_up = glm_graph_shared_gate_up_swiglu_q8_0_tensor(
                g->batch_ffn_gate,
                g->batch_ffn_up,
                g->batch_ffn_mid,
                model,
                l->ffn_gate->abs_offset,
                l->ffn_up->abs_offset,
                DS4_N_EMBD,
                hidden,
                g->batch_ffn_norm,
                n_tokens,
                0.0f);
        if (fused_gate_up) {
            ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                  stage_sync,
                                                  "glm_ffn",
                                                  "dense_gate_up_swiglu",
                                                  il,
                                                  pos0,
                                                  n_tokens,
                                                  stage_t0);
        } else {
            ok = glm_graph_matmul_q8_0_tensor(g->batch_ffn_gate,
                                              model,
                                              l->ffn_gate->abs_offset,
                                              DS4_N_EMBD,
                                              hidden,
                                              g->batch_ffn_norm,
                                              n_tokens);
            if (ok) ok = glm_graph_matmul_q8_0_tensor(g->batch_ffn_up,
                                                      model,
                                                      l->ffn_up->abs_offset,
                                                      DS4_N_EMBD,
                                                      hidden,
                                                      g->batch_ffn_norm,
                                                      n_tokens);
            if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                          stage_sync,
                                                          "glm_ffn",
                                                          "dense_gate_up",
                                                          il,
                                                          pos0,
                                                          n_tokens,
                                                          stage_t0);
            if (ok) ok = ds4_gpu_swiglu_tensor(g->batch_ffn_mid,
                                               g->batch_ffn_gate,
                                               g->batch_ffn_up,
                                               mid_elems,
                                               0.0f,
                                               1.0f) != 0;
            if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                          stage_sync,
                                                          "glm_ffn",
                                                          "dense_swiglu",
                                                          il,
                                                          pos0,
                                                          n_tokens,
                                                          stage_t0);
        }
        if (ok) ok = glm_graph_matmul_q8_0_tensor(g->batch_ffn_out,
                                                  model,
                                                  l->ffn_down->abs_offset,
                                                  hidden,
                                                  DS4_N_EMBD,
                                                  g->batch_ffn_mid,
                                                  n_tokens);
        if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                      stage_sync,
                                                      "glm_ffn",
                                                      "dense_down",
                                                      il,
                                                      pos0,
                                                      n_tokens,
                                                      stage_t0);
        if (ok) ok = ds4_gpu_add_tensor(next,
                                        after_attn,
                                        g->batch_ffn_out,
                                        (uint32_t)residual_elems) != 0;
        if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                      stage_sync,
                                                      "glm_ffn",
                                                      "residual",
                                                      il,
                                                      pos0,
                                                      n_tokens,
                                                      stage_t0);
        return ok;
    }

    if (g->ffn_mid_elems > UINT32_MAX ||
        g->dense_hidden_max < DS4_N_FF_EXP ||
        (uint64_t)n_tokens > UINT32_MAX / DS4_N_FF_EXP) {
        return false;
    }
    const uint64_t residual_elems = (uint64_t)n_tokens * DS4_N_EMBD;
    if (residual_elems > UINT32_MAX) return false;

    uint64_t gate_in = 0, gate_out = 0, gate_row_bytes = 0;
    uint64_t up_in = 0, up_out = 0, up_row_bytes = 0;
    uint64_t down_in = 0, down_out = 0, down_row_bytes = 0;
    (void)tensor_expert_bytes(model, l->ffn_gate_exps, 0, &gate_in, &gate_out, &gate_row_bytes);
    (void)tensor_expert_bytes(model, l->ffn_up_exps, 0, &up_in, &up_out, &up_row_bytes);
    (void)tensor_expert_bytes(model, l->ffn_down_exps, 0, &down_in, &down_out, &down_row_bytes);
    (void)gate_in;
    (void)up_in;
    (void)down_in;

    ok = metal_graph_matmul_plain_tensor(g->batch_router_logits,
                                         model,
                                         l->ffn_gate_inp,
                                         DS4_N_EMBD,
                                         DS4_N_EXPERT,
                                         g->batch_ffn_norm,
                                         n_tokens);
    ds4_expert_prune_mask_ensure();
    if (ok && g_expert_prune_mask_state == 1) {
        ok = glm_graph_cpu_masked_router_select_batch(g->batch_router_logits,
                                                      g->batch_router_selected,
                                                      g->batch_router_weights,
                                                      g->batch_router_probs,
                                                      model,
                                                      l,
                                                      il,
                                                      n_tokens);
    } else if (ok) {
        ok = ds4_gpu_glm_router_select_batch_tensor(g->batch_router_selected,
                                                    g->batch_router_weights,
                                                    g->batch_router_probs,
                                                    model->map,
                                                    model->size,
                                                    l->ffn_exp_probs_b->abs_offset,
                                                    g->batch_router_logits,
                                                    DS4_N_EXPERT,
                                                    DS4_N_EXPERT_USED,
                                                    DS4_EXPERT_WEIGHT_SCALE,
                                                    n_tokens) != 0;
    }
    if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                  stage_sync,
                                                  "glm_ffn",
                                                  "router",
                                                  il,
                                                  pos0,
                                                  n_tokens,
                                                  stage_t0);
    if (ok) ok = glm_graph_profile_router_selection_batch(g,
                                                          l,
                                                          il,
                                                          pos0,
                                                          n_tokens);
    if (ok) ok = glm_graph_routed_moe_batch_dispatch(
            g,
            model,
            l,
            il,
            g->batch_ffn_out,
            g->batch_ffn_mid,
            gate_out * gate_row_bytes,
            gate_row_bytes,
            up_out * up_row_bytes,
            up_row_bytes,
            down_out * down_row_bytes,
            down_row_bytes,
            g->batch_router_selected,
            g->batch_router_weights,
            g->batch_ffn_norm,
            n_tokens,
            (uint32_t)g->ffn_mid_elems,
            false) != 0;
    if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                  stage_sync,
                                                  "glm_ffn",
                                                  "routed_moe",
                                                  il,
                                                  pos0,
                                                  n_tokens,
                                                  stage_t0);
    if (ok) {
        const bool fused_shared = glm_graph_shared_gate_up_swiglu_q8_0_tensor(
                g->batch_ffn_gate,
                g->batch_ffn_up,
                g->batch_shared_mid,
                model,
                l->ffn_gate_shexp->abs_offset,
                l->ffn_up_shexp->abs_offset,
                DS4_N_EMBD,
                DS4_N_FF_EXP,
                g->batch_ffn_norm,
                n_tokens,
                0.0f);
        if (fused_shared) {
            ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                  stage_sync,
                                                  "glm_ffn",
                                                  "shared_gate_up_swiglu",
                                                  il,
                                                  pos0,
                                                  n_tokens,
                                                  stage_t0);
        } else {
            ok = glm_graph_matmul_q8_0_tensor(g->batch_ffn_gate,
                                              model,
                                              l->ffn_gate_shexp->abs_offset,
                                              DS4_N_EMBD,
                                              DS4_N_FF_EXP,
                                              g->batch_ffn_norm,
                                              n_tokens);
            if (ok) ok = glm_graph_matmul_q8_0_tensor(g->batch_ffn_up,
                                                      model,
                                                      l->ffn_up_shexp->abs_offset,
                                                      DS4_N_EMBD,
                                                      DS4_N_FF_EXP,
                                                      g->batch_ffn_norm,
                                                      n_tokens);
            if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                          stage_sync,
                                                          "glm_ffn",
                                                          "shared_gate_up",
                                                          il,
                                                          pos0,
                                                          n_tokens,
                                                          stage_t0);
            if (ok) ok = ds4_gpu_swiglu_tensor(g->batch_shared_mid,
                                               g->batch_ffn_gate,
                                               g->batch_ffn_up,
                                               n_tokens * DS4_N_FF_EXP,
                                               0.0f,
                                               1.0f) != 0;
            if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                          stage_sync,
                                                          "glm_ffn",
                                                          "shared_swiglu",
                                                          il,
                                                          pos0,
                                                          n_tokens,
                                                          stage_t0);
        }
    }
    if (ok) ok = glm_graph_matmul_q8_0_tensor(g->batch_attn_out,
                                              model,
                                              l->ffn_down_shexp->abs_offset,
                                              DS4_N_FF_EXP,
                                              DS4_N_EMBD,
                                              g->batch_shared_mid,
                                              n_tokens);
    if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                  stage_sync,
                                                  "glm_ffn",
                                                  "shared_down",
                                                  il,
                                                  pos0,
                                                  n_tokens,
                                                  stage_t0);
    if (ok && !glm_graph_disable_add3_residual()) {
        ok = ds4_gpu_add3_tensor(next,
                                 after_attn,
                                 g->batch_ffn_out,
                                 g->batch_attn_out,
                                 (uint32_t)residual_elems) != 0;
    } else if (ok) {
        ok = ds4_gpu_add_tensor(g->batch_heads,
                                g->batch_ffn_out,
                                g->batch_attn_out,
                                (uint32_t)residual_elems) != 0;
        if (ok) ok = ds4_gpu_add_tensor(next,
                                        after_attn,
                                        g->batch_heads,
                                        (uint32_t)residual_elems) != 0;
    }
    if (ok) ok = glm_graph_prefill_stage_boundary(stage_profile,
                                                  stage_sync,
                                                  "glm_ffn",
                                                  "residual",
                                                  il,
                                                  pos0,
                                                  n_tokens,
                                                  stage_t0);
    return ok;
}

static bool glm_graph_forward_token(
        ds4_glm_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        int                token,
        const float       *input_hc,
        uint32_t           pos,
        float             *output_hc,
        float             *logits_out);

static bool glm_graph_forward_tokens(
        ds4_glm_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        const int         *tokens,
        const float       *input_hc,
        uint32_t           pos0,
        uint32_t           n_tokens,
        float             *output_hc,
        float             *logits_out,
        ds4_session_progress_fn display_progress,
        void              *display_progress_ud,
        uint32_t           display_absolute_base,
        uint32_t           work_done_base,
        uint32_t           work_total) {
    if (!g || !model || !weights || !tokens ||
        n_tokens == 0 ||
        g->layer_count == 0 ||
        !glm_graph_span_fits_context(g, pos0, n_tokens)) {
        return false;
    }
    if (!glm_graph_span_fits_full_attention(g, pos0, n_tokens)) {
        glm_graph_log_full_attention_limit(g, pos0, n_tokens);
        return false;
    }
    if (!g->full_kv_cache) {
        fprintf(stderr,
                "ds4: GLM full-attention prefill was requested without an expanded KV cache\n");
        return false;
    }
    for (uint32_t i = 0; i < n_tokens; i++) {
        if (tokens[i] < 0 || tokens[i] >= (int)DS4_N_VOCAB) return false;
    }
    if (!input_hc && !g->has_token_embd) return false;
    if (logits_out && !g->has_output_head) return false;
    const uint32_t n_rows = pos0 + n_tokens;
    const bool trace = glm_graph_full_prefill_trace_enabled();
    const bool trace_all = trace && glm_graph_full_prefill_trace_all();
    const double trace_slow_ms = trace ?
        (double)glm_graph_full_prefill_trace_slow_ms() : 0.0;
    const double trace_chunk_t0 = trace ? now_sec() : 0.0;
    if (trace) {
        glm_graph_full_prefill_tracef(
                "chunk begin pos=%u tokens=%u rows=%u compact_cap=%u work_base=%u work_total=%u",
                pos0,
                n_tokens,
                n_rows,
                g->compact_cache_cap,
                work_done_base,
                work_total);
    }
    if (g->compact_cache_cap != 0) {
        const double trace_cache_t0 = trace ? now_sec() : 0.0;
        if (!glm_graph_ensure_compact_cache(g, n_rows)) {
            if (trace) {
                glm_graph_full_prefill_tracef(
                        "ensure_cache failed pos=%u tokens=%u rows=%u compact_cap=%u",
                        pos0,
                        n_tokens,
                        n_rows,
                        g->compact_cache_cap);
            }
            return false;
        }
        if (trace) {
            const double ms = (now_sec() - trace_cache_t0) * 1000.0;
            if (trace_all || ms >= trace_slow_ms) {
                glm_graph_full_prefill_tracef(
                        "ensure_cache done pos=%u tokens=%u rows=%u compact_cap=%u %.3f ms",
                        pos0,
                        n_tokens,
                        n_rows,
                        g->compact_cache_cap,
                        ms);
            }
        }
    }

    const double trace_upload_t0 = trace ? now_sec() : 0.0;
    bool ok = glm_graph_upload_tokens(g->prefill_tokens, tokens, n_tokens);
    if (trace) {
        const double ms = (now_sec() - trace_upload_t0) * 1000.0;
        if (trace_all || ms >= trace_slow_ms || !ok) {
            glm_graph_full_prefill_tracef(
                    "upload_tokens %s pos=%u tokens=%u %.3f ms",
                    ok ? "done" : "failed",
                    pos0,
                    n_tokens,
                    ms);
        }
    }
    ds4_gpu_tensor *cur = g->batch_cur;
    ds4_gpu_tensor *next = g->batch_next;
    ds4_gpu_tensor *last_hidden = NULL;

    glm_graph_report_prefill_display_progress(display_progress,
                                              display_progress_ud,
                                              display_absolute_base,
                                              work_done_base,
                                              n_tokens,
                                              0,
                                              g->layer_count,
                                              work_total,
                                              true);

    const bool stage_sync =
        glm_graph_small_prefill_stage_sync(n_tokens, logits_out != NULL);
    const uint32_t layer_flush_interval = stage_sync ? 0u :
        glm_graph_full_prefill_layer_flush_interval(n_tokens,
                                                    n_rows,
                                                    logits_out != NULL);
    const uint32_t progress_flush_interval =
        glm_graph_prefill_progress_flush_interval(layer_flush_interval,
                                                  n_tokens,
                                                  display_progress,
                                                  work_total);
    const bool progress_requested = display_progress && work_total > 0;
    const uint32_t drain_interval =
        (progress_requested && progress_flush_interval != 0) ?
        glm_graph_full_prefill_drain_interval() : 0u;
    metal_graph_stream_prepare_slot layer_prepare_slots[DS4_STREAM_PREFILL_MAX_PREPARE_AHEAD];
    memset(layer_prepare_slots, 0, sizeof(layer_prepare_slots));
    const bool full_layer_prefill =
        glm_graph_stream_prefill_full_layer_enabled(g, n_tokens);
    ds4_gpu_set_glm_streaming_prefill_full_layer(full_layer_prefill);
    const bool full_layer_prepare_base =
        glm_graph_stream_prefill_full_layer_prepare_enabled(g,
                                                            full_layer_prefill);
    const bool layer_pagein =
        full_layer_prepare_base &&
        getenv("DS4_METAL_ENABLE_STREAMING_PREFILL_LAYER_PAGEIN") != NULL &&
        getenv("DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PAGEIN") == NULL;
    const bool layer_readahead =
        full_layer_prepare_base &&
        !layer_pagein &&
        getenv("DS4_METAL_ENABLE_STREAMING_PREFILL_LAYER_READAHEAD") != NULL &&
        getenv("DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_READAHEAD") == NULL &&
        getenv("DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREPARE") == NULL;
    const bool layer_pread =
        full_layer_prepare_base &&
        !layer_pagein &&
        !layer_readahead &&
        getenv("DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREAD") == NULL &&
        getenv("DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREPARE") == NULL;
    const bool layer_madvise =
        full_layer_prepare_base &&
        !layer_pagein &&
        !layer_pread &&
        !layer_readahead &&
        getenv("DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_PREPARE") == NULL &&
        getenv("DS4_METAL_DISABLE_STREAMING_PREFILL_LAYER_MADVISE") == NULL;
    const bool layer_prepare =
        layer_pagein || layer_pread || layer_readahead || layer_madvise;
    const bool layer_prepare_overlap =
        layer_prepare &&
        metal_graph_stream_prefill_layer_pagein_overlap_enabled();
    const bool full_layer_flush_intermediate = full_layer_prefill;
    const uint32_t layer_prepare_ahead =
        layer_prepare && layer_prepare_overlap ?
        metal_graph_stream_prefill_layer_prepare_ahead() : 1u;
    if (trace) {
        glm_graph_full_prefill_tracef(
                "mode pos=%u tokens=%u stage_sync=%u layer_flush_interval=%u progress_flush_interval=%u drain_interval=%u",
                pos0,
                n_tokens,
                stage_sync ? 1u : 0u,
                layer_flush_interval,
                progress_flush_interval,
                drain_interval);
    }
    if (ok && layer_prepare && g->layer_count > 0 &&
        !metal_graph_stream_prepare_start_if_needed(NULL,
                                                    model,
                                                    weights,
                                                    g->layer_start,
                                                    n_tokens,
                                                    layer_madvise,
                                                    layer_pread,
                                                    layer_readahead,
                                                    false,
                                                    layer_prepare_slots,
                                                    layer_prepare_ahead)) {
        ok = false;
    }
    if (ok) {
        const double t0 = trace ? now_sec() : 0.0;
        if (input_hc) {
            ok = ds4_gpu_tensor_write(cur,
                                      0,
                                      input_hc,
                                      (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float)) != 0;
        } else {
            ok = glm_graph_stream_map_token(g, model, weights);
        }
        if (ok) ok = ds4_gpu_begin_commands() != 0;
        if (trace) {
            const double ms = (now_sec() - t0) * 1000.0;
            if (trace_all || ms >= trace_slow_ms || !ok) {
                glm_graph_full_prefill_tracef(
                        "begin_commands%s %s pos=%u tokens=%u %.3f ms",
                        input_hc ? "_from_hidden" : "",
                        ok ? "done" : "failed",
                        pos0,
                        n_tokens,
                        ms);
            }
        }
    }
    if (ok && !input_hc) {
        const double t0 = trace ? now_sec() : 0.0;
        ok = glm_graph_embed_tokens_tensor(cur,
                                           g->prefill_tokens,
                                           model,
                                           weights->token_embd,
                                           n_tokens) != 0;
        if (trace) {
            const double ms = (now_sec() - t0) * 1000.0;
            if (trace_all || ms >= trace_slow_ms || !ok) {
                glm_graph_full_prefill_tracef(
                        "embed %s pos=%u tokens=%u %.3f ms",
                        ok ? "done" : "failed",
                        pos0,
                        n_tokens,
                        ms);
            }
        }
    }
    if (ok && g->ssd_streaming) ok = ds4_gpu_end_commands() != 0;
#define DS4_GLM_PROFILE_PREFILL_STAGE(part_, name_) do { \
        if (ok && layer_stage_profile) { \
            ok = metal_graph_layer_stage_profile_boundary((part_), (name_), il, pos0, n_tokens, &layer_stage_t0); \
        } else if (ok && stage_sync) { \
            ok = glm_graph_prefill_stage_sync_boundary(); \
        } \
    } while (0)
    for (uint32_t il = g->layer_start; ok && il <= g->layer_end; il++) {
        const uint32_t slice_layer_done = il - g->layer_start + 1u;
        if (layer_prepare &&
            !metal_graph_stream_prepare_join_layer(NULL,
                                                   model,
                                                   weights,
                                                   il,
                                                   n_tokens,
                                                   layer_madvise,
                                                   layer_pread,
                                                   layer_readahead,
                                                   false,
                                                   layer_prepare_slots,
                                                   layer_prepare_ahead)) {
            ok = false;
            break;
        }
        if (g->ssd_streaming) {
            ok = glm_graph_stream_map_prefill_layer(g,
                                                    model,
                                                    weights,
                                                    il,
                                                    n_tokens,
                                                    full_layer_prefill);
            if (ok && layer_prepare && layer_prepare_overlap) {
                bool started_future = false;
                for (uint32_t ahead = 1; ahead <= layer_prepare_ahead; ahead++) {
                    if (il + ahead > g->layer_end) break;
                    started_future = true;
                    if (!metal_graph_stream_prepare_start_if_needed(NULL,
                                                                    model,
                                                                    weights,
                                                                    il + ahead,
                                                                    n_tokens,
                                                                    layer_madvise,
                                                                    layer_pread,
                                                                    layer_readahead,
                                                                    false,
                                                                    layer_prepare_slots,
                                                                    layer_prepare_ahead)) {
                        ok = false;
                        break;
                    }
                }
                if (ok && !started_future && logits_out) {
                    metal_graph_stream_readahead_output(model, weights);
                }
            }
            if (ok && !ds4_gpu_commands_active()) {
                ok = ds4_gpu_begin_commands() != 0;
            }
        }
        const ds4_layer_weights *l = &weights->layer[il];
        const uint32_t kv_raw_dim = (uint32_t)l->attn_kv_a_mqa->dim[1];
        const float rope_base = layer_rope_freq_base(il);
        const float rope_scale = layer_rope_freq_scale(il);
        const uint32_t cache_len = pos0 + n_tokens;
        const uint64_t residual_elems = (uint64_t)n_tokens * DS4_N_EMBD;
        const bool layer_stage_profile = metal_graph_layer_stage_profile_enabled(il);
        double layer_stage_t0 = layer_stage_profile ? now_sec() : 0.0;
        const double trace_layer_t0 = trace ? now_sec() : 0.0;
        bool trace_layer_flushed = false;
        if (trace && trace_all) {
            glm_graph_full_prefill_tracef(
                    "layer begin layer=%u pos=%u tokens=%u rows=%u",
                    il,
                    pos0,
                    n_tokens,
                    n_rows);
        }
        if (residual_elems > UINT32_MAX) {
            ok = false;
            break;
        }
        if (layer_stage_profile) {
            ok = metal_graph_layer_stage_profile_boundary("glm_attn",
                                                          NULL,
                                                          il,
                                                          pos0,
                                                          n_tokens,
                                                          &layer_stage_t0);
        }

        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_attn_norm,
                                                         cur,
                                                         model->map,
                                                         model->size,
                                                         l->attn_norm->abs_offset,
                                                         DS4_N_EMBD,
                                                         n_tokens,
                                                         DS4_RMS_EPS) != 0;
        DS4_GLM_PROFILE_PREFILL_STAGE("glm_attn", "attn_norm");
        if (ok) ok = glm_graph_matmul_q8_0_tensor(g->batch_q_rank,
                                                  model,
                                                  l->attn_q_a->abs_offset,
                                                  DS4_N_EMBD,
                                                  DS4_N_LORA_Q,
                                                  g->batch_attn_norm,
                                                  n_tokens);
        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_q_rank_norm,
                                                         g->batch_q_rank,
                                                         model->map,
                                                         model->size,
                                                         l->attn_q_a_norm->abs_offset,
                                                         DS4_N_LORA_Q,
                                                         n_tokens,
                                                         DS4_RMS_EPS) != 0;
        if (ok) ok = glm_graph_matmul_q8_0_tensor(g->batch_q,
                                                  model,
                                                  l->attn_q_b->abs_offset,
                                                  DS4_N_LORA_Q,
                                                  g->q_dim,
                                                  g->batch_q_rank_norm,
                                                  n_tokens);
        if (ok) ok = ds4_gpu_rope_tail_tensor(g->batch_q,
                                              n_tokens,
                                              DS4_N_HEAD,
                                              DS4_N_KEY_MLA,
                                              DS4_N_ROT,
                                              pos0,
                                              0,
                                              false,
                                              rope_base,
                                              rope_scale,
                                              0.0f,
                                              1.0f,
                                              DS4_ROPE_YARN_BETA_FAST,
                                              DS4_ROPE_YARN_BETA_SLOW) != 0;
        DS4_GLM_PROFILE_PREFILL_STAGE("glm_attn", "q_path");
        if (ok && g->compact_cache_cap != 0 && glm_graph_layer_uses_full_indexer(il)) {
            ok = glm_graph_matmul_q8_0_tensor(g->batch_indexer_k,
                                              model,
                                              l->indexer_attn_k->abs_offset,
                                              DS4_N_EMBD,
                                              DS4_N_INDEXER_HEAD_DIM,
                                              cur,
                                              n_tokens);
            if (ok) {
                ok = ds4_gpu_glm_store_indexer_k_tensor(
                        g->layer_indexer_key_cache[il],
                        g->batch_indexer_k,
                        model->map,
                        model->size,
                        l->indexer_k_norm->abs_offset,
                        l->indexer_k_norm_b->abs_offset,
                        pos0,
                        n_tokens,
                        g->compact_cache_cap,
                        DS4_N_INDEXER_HEAD_DIM,
                        DS4_N_ROT,
                        0,
                        1.0e-6f,
                        rope_base,
                        rope_scale,
                        0.0f,
                        1.0f,
                        DS4_ROPE_YARN_BETA_FAST,
                        DS4_ROPE_YARN_BETA_SLOW,
                        glm_graph_compact_cache_is_f16()) != 0;
            }
        }
        DS4_GLM_PROFILE_PREFILL_STAGE("glm_attn", "indexer_k");
        if (ok) ok = glm_graph_matmul_q8_0_tensor(g->batch_kv_raw,
                                                  model,
                                                  l->attn_kv_a_mqa->abs_offset,
                                                  DS4_N_EMBD,
                                                  kv_raw_dim,
                                                  g->batch_attn_norm,
                                                  n_tokens);
        if (ok) ok = ds4_gpu_glm_kv_lora_rms_norm_tensor(g->batch_kv_norm,
                                                         g->batch_kv_raw,
                                                         model->map,
                                                         model->size,
                                                         l->attn_kv_a_norm->abs_offset,
                                                         n_tokens,
                                                         kv_raw_dim,
                                                         DS4_N_KV_LORA,
                                                         DS4_RMS_EPS) != 0;
        if (ok && g->compact_cache_cap != 0) {
            ok = ds4_gpu_glm_store_compact_kv_tensor(g->layer_kv_lora_cache[il],
                                                     g->layer_k_rope_cache[il],
                                                     g->batch_kv_norm,
                                                     g->batch_kv_raw,
                                                     pos0,
                                                     n_tokens,
                                                     g->compact_cache_cap,
                                                     kv_raw_dim,
                                                     DS4_N_KV_LORA,
                                                     DS4_N_ROT,
                                                     glm_graph_compact_cache_is_f16()) != 0;
        }
        if (ok) ok = ds4_gpu_glm_k_b_project_tensor(g->batch_k_nope,
                                                    g->batch_kv_norm,
                                                    model->map,
                                                    model->size,
                                                    l->attn_k_b->abs_offset,
                                                    n_tokens,
                                                    DS4_N_KV_LORA,
                                                    (uint32_t)g->q_nope,
                                                    DS4_N_HEAD) != 0;
        if (ok) ok = glm_graph_matmul_q8_0_tensor(g->batch_value,
                                                  model,
                                                  l->attn_v_b->abs_offset,
                                                  DS4_N_KV_LORA,
                                                  g->heads_dim,
                                                  g->batch_kv_norm,
                                                  n_tokens);
        DS4_GLM_PROFILE_PREFILL_STAGE("glm_attn", "kv_path");
        const bool flash_requested = glm_graph_use_flash_attention_prefill(n_tokens);
        const bool use_staged_flash_kv =
            flash_requested && glm_graph_use_flash_attention_staged_kv(pos0, n_tokens, cache_len);
        const bool use_flash_attn = flash_requested;
        if (ok) {
            if (use_staged_flash_kv) {
                ok = ds4_gpu_glm_build_kv_cache_flash_tensor(g->layer_key_cache[il],
                                                             g->layer_value_cache[il],
                                                             g->batch_kv_raw,
                                                             g->batch_k_nope,
                                                             g->batch_value,
                                                             pos0,
                                                             n_tokens,
                                                             g->ctx_cap,
                                                             DS4_N_HEAD,
                                                             kv_raw_dim,
                                                             DS4_N_KV_LORA,
                                                             (uint32_t)g->q_nope,
                                                             DS4_N_ROT,
                                                             DS4_N_VALUE_MLA,
                                                             0,
                                                             rope_base,
                                                             rope_scale,
                                                             0.0f,
                                                             1.0f,
                                                             DS4_ROPE_YARN_BETA_FAST,
                                                             DS4_ROPE_YARN_BETA_SLOW,
                                                             true) != 0;
            } else {
                ok = ds4_gpu_glm_build_kv_cache_tensor(g->layer_key_cache[il],
                                                       g->layer_value_cache[il],
                                                       g->batch_kv_raw,
                                                       g->batch_k_nope,
                                                       g->batch_value,
                                                       pos0,
                                                       n_tokens,
                                                       g->ctx_cap,
                                                       DS4_N_HEAD,
                                                       kv_raw_dim,
                                                       DS4_N_KV_LORA,
                                                       (uint32_t)g->q_nope,
                                                       DS4_N_ROT,
                                                       DS4_N_VALUE_MLA,
                                                       0,
                                                       rope_base,
                                                       rope_scale,
                                                       0.0f,
                                                       1.0f,
                                                       DS4_ROPE_YARN_BETA_FAST,
                                                       DS4_ROPE_YARN_BETA_SLOW,
                                                       true) != 0;
            }
        }
        DS4_GLM_PROFILE_PREFILL_STAGE("glm_attn", "kv_cache");
        if (ok) {
            if (use_flash_attn) {
                if (use_staged_flash_kv) {
                    ok = ds4_gpu_glm_attention_flash_staged_tensor(g->batch_heads,
                                                                   g->batch_q,
                                                                   g->layer_key_cache[il],
                                                                   g->layer_value_cache[il],
                                                                   pos0,
                                                                   n_tokens,
                                                                   cache_len,
                                                                   g->ctx_cap,
                                                                   DS4_N_HEAD,
                                                                   DS4_N_KEY_MLA,
                                                                   DS4_N_VALUE_MLA,
                                                                   true) != 0;
                } else {
                    ok = ds4_gpu_glm_attention_flash_tensor(g->batch_heads,
                                                            g->batch_q,
                                                            g->layer_key_cache[il],
                                                            g->layer_value_cache[il],
                                                            pos0,
                                                            n_tokens,
                                                            cache_len,
                                                            g->ctx_cap,
                                                            DS4_N_HEAD,
                                                            DS4_N_KEY_MLA,
                                                            DS4_N_VALUE_MLA,
                                                            true) != 0;
                }
            } else {
                ok = ds4_gpu_glm_attention_full_tensor(g->batch_heads,
                                                       g->batch_q,
                                                       g->layer_key_cache[il],
                                                       g->layer_value_cache[il],
                                                       pos0,
                                                       n_tokens,
                                                       cache_len,
                                                       g->ctx_cap,
                                                       DS4_N_HEAD,
                                                       DS4_N_KEY_MLA,
                                                       DS4_N_VALUE_MLA,
                                                       true) != 0;
            }
        }
        DS4_GLM_PROFILE_PREFILL_STAGE("glm_attn", "attention");
        if (ok) ok = glm_graph_matmul_q8_0_tensor(g->batch_attn_out,
                                                  model,
                                                  l->attn_output->abs_offset,
                                                  g->heads_dim,
                                                  DS4_N_EMBD,
                                                  g->batch_heads,
                                                  n_tokens);
        if (ok) ok = ds4_gpu_add_tensor(g->batch_after_attn,
                                        cur,
                                        g->batch_attn_out,
                                        (uint32_t)residual_elems) != 0;
        DS4_GLM_PROFILE_PREFILL_STAGE("glm_attn", "attn_output");
        if (ok) ok = glm_graph_encode_ffn_batch(g,
                                                model,
                                                l,
                                                il,
                                                pos0,
                                                g->batch_after_attn,
                                                next,
                                                n_tokens,
                                                layer_stage_profile,
                                                stage_sync,
                                                layer_stage_profile ? &layer_stage_t0 : NULL);
        if (ok) {
            ds4_gpu_tensor *tmp = cur;
            cur = next;
            next = tmp;
        }
        if (ok &&
            !g->ssd_streaming &&
            progress_flush_interval != 0 &&
            (il < g->layer_end || progress_requested) &&
            (slice_layer_done % progress_flush_interval) == 0) {
            const uint32_t work_done =
                work_done_base + (uint32_t)(((uint64_t)n_tokens * slice_layer_done) / g->layer_count);
            const bool drain_now =
                drain_interval != 0 &&
                il < g->layer_end &&
                (slice_layer_done % drain_interval) == 0;
            const char *command_action = drain_now ? "drain" : "flush";
            const double trace_command_t0 = trace ? now_sec() : 0.0;
            if (trace && (trace_all || drain_now)) {
                glm_graph_full_prefill_tracef(
                        "layer %s begin layer=%u pos=%u tokens=%u work=%u/%u",
                        command_action,
                        il,
                        pos0,
                        n_tokens,
                        work_done,
                        work_total);
            }
            if (drain_now) {
                ok = ds4_gpu_end_commands() != 0;
                if (ok) ok = ds4_gpu_begin_commands() != 0;
            } else {
                ok = ds4_gpu_flush_commands() != 0;
            }
            if (trace) {
                const double trace_command_done = now_sec();
                const double command_ms = (trace_command_done - trace_command_t0) * 1000.0;
                const double layer_ms = (trace_command_done - trace_layer_t0) * 1000.0;
                trace_layer_flushed = true;
                if (trace_all || drain_now ||
                    command_ms >= trace_slow_ms || layer_ms >= trace_slow_ms || !ok) {
                    glm_graph_full_prefill_tracef(
                            "layer %s %s layer=%u pos=%u tokens=%u command=%.3f ms layer_total=%.3f ms work=%u/%u",
                            command_action,
                            ok ? "done" : "failed",
                            il,
                            pos0,
                            n_tokens,
                            command_ms,
                            layer_ms,
                            work_done,
                            work_total);
                }
            }
            if (ok) {
                const bool progress_completed =
                    drain_interval == 0 || drain_now;
                if (progress_completed) {
                    glm_graph_report_prefill_display_progress(display_progress,
                                                              display_progress_ud,
                                                              display_absolute_base,
                                                              work_done_base,
                                                              n_tokens,
                                                              slice_layer_done,
                                                              g->layer_count,
                                                              work_total,
                                                              logits_out == NULL && output_hc == NULL);
                }
            }
        }
        if (g->ssd_streaming) {
            if (ok && full_layer_flush_intermediate &&
                il < g->layer_end) {
                ok = ds4_gpu_flush_commands() != 0;
            } else if (ok) {
                ok = ds4_gpu_end_commands() != 0;
            }
            else (void)ds4_gpu_synchronize();
            if (ok) {
                glm_graph_report_prefill_display_progress(display_progress,
                                                          display_progress_ud,
                                                          display_absolute_base,
                                                          work_done_base,
                                                          n_tokens,
                                                          slice_layer_done,
                                                          g->layer_count,
                                                          work_total,
                                                          logits_out == NULL && output_hc == NULL);
            }
        }
        if (ok &&
            g->ssd_streaming &&
            layer_prepare &&
            !layer_prepare_overlap) {
            if (il < g->layer_end) {
                if (!metal_graph_stream_prepare_start_if_needed(NULL,
                                                                model,
                                                                weights,
                                                                il + 1u,
                                                                n_tokens,
                                                                layer_madvise,
                                                                layer_pread,
                                                                layer_readahead,
                                                                false,
                                                                layer_prepare_slots,
                                                                layer_prepare_ahead)) {
                    ok = false;
                }
            } else if (logits_out) {
                metal_graph_stream_readahead_output(model, weights);
            }
        }
        if (trace && ok) {
            const double layer_ms = (now_sec() - trace_layer_t0) * 1000.0;
            if (trace_all || (!trace_layer_flushed && layer_ms >= trace_slow_ms)) {
                glm_graph_full_prefill_tracef(
                        "layer end layer=%u pos=%u tokens=%u flushed=%u layer_total=%.3f ms",
                        il,
                        pos0,
                        n_tokens,
                        trace_layer_flushed ? 1u : 0u,
                        layer_ms);
            }
        }
    }
#undef DS4_GLM_PROFILE_PREFILL_STAGE
    if (ok && !g->ssd_streaming) {
        const double trace_end_t0 = trace ? now_sec() : 0.0;
        if (trace) {
            glm_graph_full_prefill_tracef(
                    "chunk end_commands begin pos=%u tokens=%u",
                    pos0,
                    n_tokens);
        }
        ok = ds4_gpu_end_commands() != 0;
        if (trace) {
            const double end_ms = (now_sec() - trace_end_t0) * 1000.0;
            const double chunk_ms = (now_sec() - trace_chunk_t0) * 1000.0;
            glm_graph_full_prefill_tracef(
                    "chunk end_commands %s pos=%u tokens=%u end=%.3f ms chunk_total=%.3f ms",
                    ok ? "done" : "failed",
                    pos0,
                    n_tokens,
                    end_ms,
                    chunk_ms);
        }
    } else if (!ok) {
        if (trace) {
            glm_graph_full_prefill_tracef(
                    "chunk failed before end pos=%u tokens=%u elapsed=%.3f ms",
                    pos0,
                    n_tokens,
                    (now_sec() - trace_chunk_t0) * 1000.0);
        }
        if (layer_prepare) {
            (void)metal_graph_stream_prepare_join_all(layer_prepare_slots,
                                                      layer_prepare_ahead);
        }
        (void)ds4_gpu_synchronize();
    }
    if (ok && layer_prepare &&
        !metal_graph_stream_prepare_join_all(layer_prepare_slots,
                                             layer_prepare_ahead)) {
        ok = false;
    }
    if (ok) {
        glm_graph_report_prefill_display_progress(display_progress,
                                                  display_progress_ud,
                                                  display_absolute_base,
                                                  work_done_base,
                                                  n_tokens,
                                                  g->layer_count,
                                                  g->layer_count,
                                                  work_total,
                                                  logits_out == NULL && output_hc == NULL);
    }
    if (ok && output_hc) {
        ok = ds4_gpu_tensor_read(cur,
                                 0,
                                 output_hc,
                                 (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float)) != 0;
    }
    if (ok && logits_out) {
        last_hidden = glm_graph_tensor_row_view_strided(cur,
                                                        n_tokens - 1u,
                                                        DS4_N_EMBD,
                                                        DS4_N_EMBD);
        ok = last_hidden != NULL;
        if (ok && g->ssd_streaming) ok = glm_graph_stream_map_output(g, model, weights);
        if (ok) ok = glm_graph_forward_output_head(g, model, weights, last_hidden, logits_out);
        if (ok) {
            glm_graph_report_prefill_display_progress(display_progress,
                                                      display_progress_ud,
                                                      display_absolute_base,
                                                      work_done_base,
                                                      n_tokens,
                                                      g->layer_count,
                                                      g->layer_count,
                                                      work_total,
                                                      true);
        }
    }
    ds4_gpu_tensor_free(last_hidden);
    ds4_gpu_set_glm_streaming_prefill_full_layer(false);
    return ok;
}

static uint32_t glm_graph_prefill_chunk_tokens(uint32_t full_attention_cap) {
    return full_attention_cap ? full_attention_cap : DS4_GLM_METAL_FULL_ATTN_DEFAULT_CONTEXT;
}

static bool glm_graph_forward_indexed_tokens(
        ds4_glm_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        const int         *tokens,
        const float       *input_hc,
        uint32_t           pos0,
        uint32_t           n_tokens,
        float             *output_hc,
        float             *logits_out,
        ds4_session_progress_fn display_progress,
        void              *display_progress_ud,
        uint32_t           display_absolute_base,
        uint32_t           work_done_base,
        uint32_t           work_total) {
    if (!g || !model || !weights || !tokens ||
        g->compact_cache_cap == 0 ||
        g->indexed_prefill_cap == 0 ||
        g->indexed_prefill_score_cap == 0 ||
        !g->batch_indexer_q ||
        !g->batch_indexer_weights ||
        !g->batch_indexer_scores ||
        !g->batch_indexer_selected ||
        !g->batch_qk_low ||
        n_tokens == 0 ||
        g->layer_count == 0 ||
        n_tokens > g->indexed_prefill_cap ||
        !glm_graph_span_fits_context(g, pos0, n_tokens)) {
        return false;
    }
    const uint32_t n_rows = pos0 + n_tokens;
    const uint32_t indexer_top_k = glm_graph_indexer_top_k_limit();
    if (pos0 < indexer_top_k && n_rows > indexer_top_k) {
        return false;
    }
    const uint32_t indexed_selected_count =
        n_rows <= indexer_top_k ? n_rows : indexer_top_k;
    const bool use_causal_range_select = n_rows <= indexer_top_k;
    const bool trace = glm_graph_indexed_prefill_trace_enabled();
    const bool trace_all = trace && glm_graph_indexed_prefill_trace_all();
    const double trace_slow_ms = trace ?
        (double)glm_graph_indexed_prefill_trace_slow_ms() : 0.0;
    const double trace_chunk_t0 = trace ? now_sec() : 0.0;
    if (trace) {
        glm_graph_indexed_prefill_tracef(
                "chunk begin pos=%u tokens=%u rows=%u selected=%u compact_cap=%u score_cap=%u work_base=%u work_total=%u",
                pos0,
                n_tokens,
                n_rows,
                indexed_selected_count,
                g->compact_cache_cap,
                g->indexed_prefill_score_cap,
                work_done_base,
                work_total);
    }
    const double trace_cache_t0 = trace ? now_sec() : 0.0;
    if (!glm_graph_ensure_compact_cache(g, n_rows)) {
        if (trace) {
            glm_graph_indexed_prefill_tracef(
                    "ensure_cache failed pos=%u tokens=%u rows=%u compact_cap=%u",
                    pos0,
                    n_tokens,
                    n_rows,
                    g->compact_cache_cap);
        }
        return false;
    }
    if (trace) {
        const double ms = (now_sec() - trace_cache_t0) * 1000.0;
        if (trace_all || ms >= trace_slow_ms) {
            glm_graph_indexed_prefill_tracef(
                    "ensure_cache done pos=%u tokens=%u rows=%u compact_cap=%u %.3f ms",
                    pos0,
                    n_tokens,
                    n_rows,
                    g->compact_cache_cap,
                    ms);
        }
    }
    for (uint32_t i = 0; i < n_tokens; i++) {
        if (tokens[i] < 0 || tokens[i] >= (int)DS4_N_VOCAB) return false;
    }
    if (!input_hc && !g->has_token_embd) return false;
    if (logits_out && !g->has_output_head) return false;

    const double trace_upload_t0 = trace ? now_sec() : 0.0;
    bool ok = glm_graph_upload_tokens(g->prefill_tokens, tokens, n_tokens);
    if (trace) {
        const double ms = (now_sec() - trace_upload_t0) * 1000.0;
        if (trace_all || ms >= trace_slow_ms || !ok) {
            glm_graph_indexed_prefill_tracef(
                    "upload_tokens %s pos=%u tokens=%u %.3f ms",
                    ok ? "done" : "failed",
                    pos0,
                    n_tokens,
                    ms);
        }
    }
    ds4_gpu_tensor *cur = g->batch_cur;
    ds4_gpu_tensor *next = g->batch_next;
    ds4_gpu_tensor *last_hidden = NULL;

    glm_graph_report_prefill_display_progress(display_progress,
                                              display_progress_ud,
                                              display_absolute_base,
                                              work_done_base,
                                              n_tokens,
                                              0,
                                              g->layer_count,
                                              work_total,
                                              true);

    const bool use_all_scalar_kernels =
        n_tokens == 1u && glm_graph_indexed_prefill_scalar_kernels();
    const bool use_scalar_indexer =
        use_all_scalar_kernels ||
        glm_graph_indexed_prefill_scalar_indexer() ||
        !glm_graph_indexed_prefill_batch_indexer();
    const bool force_scalar_attn =
        use_all_scalar_kernels || glm_graph_indexed_prefill_scalar_attn();
    const bool use_batch_qk_low =
        !force_scalar_attn && glm_graph_indexed_prefill_batch_qk_low();
    const bool use_batch_attn_kernel =
        !force_scalar_attn && glm_graph_indexed_prefill_batch_attn_kernel();
    const bool use_split_value_proj =
        use_batch_attn_kernel &&
        g->batch_attn_lora;
    const bool use_batch_q_rank_proj = true;
    const bool use_batch_q_proj = true;
    const bool use_batch_indexer_k_proj = true;
    const bool use_batch_kv_proj = true;
    const bool use_batch_indexer_q_proj = true;
    const bool use_batch_indexer_weights_proj = true;
    const bool use_batch_attn_out_proj = true;
    const bool use_batch_ffn = glm_graph_indexed_prefill_batch_ffn();
    const bool stage_sync =
        glm_graph_small_prefill_stage_sync(n_tokens, logits_out != NULL);
    const uint32_t layer_flush_interval = stage_sync ? 0u :
        glm_graph_full_prefill_layer_flush_interval(n_tokens,
                                                    n_tokens,
                                                    logits_out != NULL);
    const uint32_t progress_flush_interval =
        glm_graph_prefill_progress_flush_interval(layer_flush_interval,
                                                  n_tokens,
                                                  display_progress,
                                                  work_total);
    const uint32_t drain_interval =
        progress_flush_interval != 0 ? glm_graph_indexed_prefill_drain_interval() : 0u;
    const bool progress_requested = display_progress && work_total > 0;
    /* Indexed prefill must take the same full-layer path as plain prefill once
     * a chunk can select more experts than the streaming cache can hold. */
    const bool full_layer_prefill =
        glm_graph_stream_prefill_full_layer_enabled(g, n_tokens);
    ds4_gpu_set_glm_streaming_prefill_full_layer(full_layer_prefill);

    if (trace) {
        glm_graph_indexed_prefill_tracef(
                "mode pos=%u tokens=%u scalar_indexer=%u batch_qk_low=%u batch_attn=%u split_value=%u batch_ffn=%u progress_flush_interval=%u drain_interval=%u",
                pos0,
                n_tokens,
                use_scalar_indexer ? 1u : 0u,
                use_batch_qk_low ? 1u : 0u,
                use_batch_attn_kernel ? 1u : 0u,
                use_split_value_proj ? 1u : 0u,
                use_batch_ffn ? 1u : 0u,
                progress_flush_interval,
                drain_interval);
    }

    if (ok) {
        const double t0 = trace ? now_sec() : 0.0;
        if (input_hc) {
            ok = ds4_gpu_tensor_write(cur,
                                      0,
                                      input_hc,
                                      (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float)) != 0;
        } else {
            ok = glm_graph_stream_map_token(g, model, weights);
        }
        if (ok) ok = ds4_gpu_begin_commands() != 0;
        if (trace) {
            const double ms = (now_sec() - t0) * 1000.0;
            if (trace_all || ms >= trace_slow_ms || !ok) {
                glm_graph_indexed_prefill_tracef(
                        "begin_commands%s %s pos=%u tokens=%u %.3f ms",
                        input_hc ? "_from_hidden" : "",
                        ok ? "done" : "failed",
                        pos0,
                        n_tokens,
                        ms);
            }
        }
    }
    if (ok && !input_hc) {
        const double t0 = trace ? now_sec() : 0.0;
        ok = glm_graph_embed_tokens_tensor(cur,
                                           g->prefill_tokens,
                                           model,
                                           weights->token_embd,
                                           n_tokens) != 0;
        if (trace) {
            const double ms = (now_sec() - t0) * 1000.0;
            if (trace_all || ms >= trace_slow_ms || !ok) {
                glm_graph_indexed_prefill_tracef(
                        "embed %s pos=%u tokens=%u %.3f ms",
                        ok ? "done" : "failed",
                        pos0,
                        n_tokens,
                        ms);
            }
        }
    }
    if (ok && g->ssd_streaming) ok = ds4_gpu_end_commands() != 0;

#define DS4_GLM_PROFILE_INDEXED_STAGE(part_, name_) do { \
        if (ok && trace) { \
            const double _trace_stage_now = now_sec(); \
            const double _trace_stage_ms = (_trace_stage_now - trace_stage_t0) * 1000.0; \
            if (trace_all || _trace_stage_ms >= trace_slow_ms) { \
                glm_graph_indexed_prefill_tracef( \
                        "stage layer=%u pos=%u tokens=%u %s.%s encode %.3f ms", \
                        il, \
                        pos0, \
                        n_tokens, \
                        (part_), \
                        (name_), \
                        _trace_stage_ms); \
            } \
            trace_stage_t0 = _trace_stage_now; \
        } \
        if (ok && layer_stage_profile) { \
            ok = metal_graph_layer_stage_profile_boundary((part_), (name_), il, pos0, n_tokens, &layer_stage_t0); \
        } else if (ok && stage_sync) { \
            ok = glm_graph_prefill_stage_sync_boundary(); \
        } \
    } while (0)
    ds4_gpu_tensor *last_indexer_selected = NULL;
    uint32_t last_indexer_selected_count = 0;
    /* The indexed prefill maps each full layer and then pays the sweep as GPU
     * demand faults during the kernels (~2.6 GB/s measured on an M5 Pro
     * against ~7 GB/s the SSD can sustain).  Reuse the full-attention layer
     * prepare (pread pool warming the next layer while this one computes):
     * measured 3.5 -> 8.5 t/s prefill on a 271-token prompt, decode
     * unaffected, identical greedy output.  Opt-in while it soaks. */
    metal_graph_stream_prepare_slot idx_prepare_slots[DS4_STREAM_PREFILL_MAX_PREPARE_AHEAD];
    memset(idx_prepare_slots, 0, sizeof(idx_prepare_slots));
    const bool idx_prepare =
        g->ssd_streaming &&
        full_layer_prefill &&
        g->glm_indexed_prefill_prepare;
    for (uint32_t il = g->layer_start; ok && il <= g->layer_end; il++) {
        const uint32_t slice_layer_done = il - g->layer_start + 1u;
        if (g->ssd_streaming) {
            if (idx_prepare &&
                !metal_graph_stream_prepare_join_layer(NULL,
                                                       model,
                                                       weights,
                                                       il,
                                                       n_tokens,
                                                       false,
                                                       true,
                                                       false,
                                                       false,
                                                       idx_prepare_slots,
                                                       DS4_STREAM_PREFILL_MAX_PREPARE_AHEAD)) {
                ok = false;
                break;
            }
            ok = glm_graph_stream_map_prefill_layer(g,
                                                    model,
                                                    weights,
                                                    il,
                                                    n_tokens,
                                                    full_layer_prefill);
            if (ok && idx_prepare && il < g->layer_end) {
                if (!metal_graph_stream_prepare_start_if_needed(
                            NULL,
                            model,
                            weights,
                            il + 1,
                            n_tokens,
                            false,
                            true,
                            false,
                            false,
                            idx_prepare_slots,
                            DS4_STREAM_PREFILL_MAX_PREPARE_AHEAD)) {
                    ok = false;
                }
            }
            if (ok) ok = ds4_gpu_begin_commands() != 0;
        }
        const ds4_layer_weights *l = &weights->layer[il];
        const uint32_t kv_raw_dim = (uint32_t)l->attn_kv_a_mqa->dim[1];
        const float rope_base = layer_rope_freq_base(il);
        const float rope_scale = layer_rope_freq_scale(il);
        const uint64_t residual_elems = (uint64_t)n_tokens * DS4_N_EMBD;
        const bool layer_stage_profile = metal_graph_layer_stage_profile_enabled(il);
        double layer_stage_t0 = layer_stage_profile ? now_sec() : 0.0;
        double trace_stage_t0 = trace ? now_sec() : 0.0;
        const double trace_layer_t0 = trace_stage_t0;
        const bool trace_full_indexer = glm_graph_layer_uses_full_indexer(il);
        bool trace_layer_flushed = false;
        if (trace && (trace_all || trace_full_indexer)) {
            glm_graph_indexed_prefill_tracef(
                    "layer begin layer=%u pos=%u tokens=%u rows=%u selected=%u full_indexer=%u",
                    il,
                    pos0,
                    n_tokens,
                    n_rows,
                    indexed_selected_count,
                    trace_full_indexer ? 1u : 0u);
        }
        if (residual_elems > UINT32_MAX) {
            ok = false;
            break;
        }
        if (layer_stage_profile) {
            ok = metal_graph_layer_stage_profile_boundary("glm_indexed_attn",
                                                          NULL,
                                                          il,
                                                          pos0,
                                                          n_tokens,
                                                          &layer_stage_t0);
        }

        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_attn_norm,
                                                         cur,
                                                         model->map,
                                                         model->size,
                                                         l->attn_norm->abs_offset,
                                                         DS4_N_EMBD,
                                                         n_tokens,
                                                         DS4_RMS_EPS) != 0;
        DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_attn", "attn_norm");
        if (ok) {
            ok = (use_batch_q_rank_proj ?
                  glm_graph_matmul_q8_0_tensor(g->batch_q_rank,
                                               model,
                                               l->attn_q_a->abs_offset,
                                               DS4_N_EMBD,
                                               DS4_N_LORA_Q,
                                               g->batch_attn_norm,
                                               n_tokens) :
                  glm_graph_matmul_q8_0_rows_scalar(g->batch_q_rank,
                                                    model,
                                                    l->attn_q_a->abs_offset,
                                                    DS4_N_EMBD,
                                                    DS4_N_LORA_Q,
                                                    g->batch_attn_norm,
                                                    n_tokens));
        }
        if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_q_rank_norm,
                                                         g->batch_q_rank,
                                                         model->map,
                                                         model->size,
                                                         l->attn_q_a_norm->abs_offset,
                                                         DS4_N_LORA_Q,
                                                         n_tokens,
                                                         DS4_RMS_EPS) != 0;
        if (ok) {
            ok = (use_batch_q_proj ?
                  glm_graph_matmul_q8_0_tensor(g->batch_q,
                                               model,
                                               l->attn_q_b->abs_offset,
                                               DS4_N_LORA_Q,
                                               g->q_dim,
                                               g->batch_q_rank_norm,
                                               n_tokens) :
                  glm_graph_matmul_q8_0_rows_scalar(g->batch_q,
                                                    model,
                                                    l->attn_q_b->abs_offset,
                                                    DS4_N_LORA_Q,
                                                    g->q_dim,
                                                    g->batch_q_rank_norm,
                                                    n_tokens));
        }
        if (ok) ok = ds4_gpu_rope_tail_tensor(g->batch_q,
                                              n_tokens,
                                              DS4_N_HEAD,
                                              DS4_N_KEY_MLA,
                                              DS4_N_ROT,
                                              pos0,
                                              0,
                                              false,
                                              rope_base,
                                              rope_scale,
                                              0.0f,
                                              1.0f,
                                              DS4_ROPE_YARN_BETA_FAST,
                                              DS4_ROPE_YARN_BETA_SLOW) != 0;
        DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_attn", "q_path");

        if (ok && glm_graph_layer_uses_full_indexer(il)) {
            ok = (use_batch_indexer_k_proj ?
                  glm_graph_matmul_q8_0_tensor(g->batch_indexer_k,
                                               model,
                                               l->indexer_attn_k->abs_offset,
                                               DS4_N_EMBD,
                                               DS4_N_INDEXER_HEAD_DIM,
                                               cur,
                                               n_tokens) :
                  glm_graph_matmul_q8_0_rows_scalar(g->batch_indexer_k,
                                                    model,
                                                    l->indexer_attn_k->abs_offset,
                                                    DS4_N_EMBD,
                                                    DS4_N_INDEXER_HEAD_DIM,
                                                    cur,
                                                    n_tokens));
            if (ok) {
                ok = ds4_gpu_glm_store_indexer_k_tensor(
                        g->layer_indexer_key_cache[il],
                        g->batch_indexer_k,
                        model->map,
                        model->size,
                        l->indexer_k_norm->abs_offset,
                        l->indexer_k_norm_b->abs_offset,
                        pos0,
                        n_tokens,
                        g->compact_cache_cap,
                        DS4_N_INDEXER_HEAD_DIM,
                        DS4_N_ROT,
                        0,
                        1.0e-6f,
                        rope_base,
                        rope_scale,
                        0.0f,
                        1.0f,
                        DS4_ROPE_YARN_BETA_FAST,
                        DS4_ROPE_YARN_BETA_SLOW,
                        glm_graph_compact_cache_is_f16()) != 0;
            }
        }
        DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_attn", "indexer_k");

        if (ok) {
            ok = (use_batch_kv_proj ?
                  glm_graph_matmul_q8_0_tensor(g->batch_kv_raw,
                                               model,
                                               l->attn_kv_a_mqa->abs_offset,
                                               DS4_N_EMBD,
                                               kv_raw_dim,
                                               g->batch_attn_norm,
                                               n_tokens) :
                  glm_graph_matmul_q8_0_rows_scalar(g->batch_kv_raw,
                                                    model,
                                                    l->attn_kv_a_mqa->abs_offset,
                                                    DS4_N_EMBD,
                                                    kv_raw_dim,
                                                    g->batch_attn_norm,
                                                    n_tokens));
        }
        if (ok) ok = ds4_gpu_glm_kv_lora_rms_norm_tensor(g->batch_kv_norm,
                                                         g->batch_kv_raw,
                                                         model->map,
                                                         model->size,
                                                         l->attn_kv_a_norm->abs_offset,
                                                         n_tokens,
                                                         kv_raw_dim,
                                                         DS4_N_KV_LORA,
                                                         DS4_RMS_EPS) != 0;
        if (ok) {
            ok = ds4_gpu_glm_store_compact_kv_tensor(g->layer_kv_lora_cache[il],
                                                     g->layer_k_rope_cache[il],
                                                     g->batch_kv_norm,
                                                     g->batch_kv_raw,
                                                     pos0,
                                                     n_tokens,
                                                     g->compact_cache_cap,
                                                     kv_raw_dim,
                                                     DS4_N_KV_LORA,
                                                     DS4_N_ROT,
                                                     glm_graph_compact_cache_is_f16()) != 0;
        }
        DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_attn", "kv_path");

        if (ok && glm_graph_layer_uses_full_indexer(il)) {
            if (ok && !use_causal_range_select) {
                ok = (use_batch_indexer_q_proj ?
                      glm_graph_matmul_q8_0_tensor(g->batch_indexer_q,
                                                   model,
                                                   l->indexer_attn_q_b->abs_offset,
                                                   DS4_N_LORA_Q,
                                                   (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM,
                                                   g->batch_q_rank_norm,
                                                   n_tokens) :
                      glm_graph_matmul_q8_0_rows_scalar(g->batch_indexer_q,
                                                        model,
                                                        l->indexer_attn_q_b->abs_offset,
                                                        DS4_N_LORA_Q,
                                                        (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM,
                                                        g->batch_q_rank_norm,
                                                        n_tokens));
                if (ok) ok = ds4_gpu_glm_indexer_rope_tail_tensor(g->batch_indexer_q,
                                                                  n_tokens,
                                                                  DS4_N_INDEXER_HEAD,
                                                                  DS4_N_INDEXER_HEAD_DIM,
                                                                  DS4_N_ROT,
                                                                  pos0,
                                                                  0,
                                                                  rope_base,
                                                                  rope_scale,
                                                                  0.0f,
                                                                  1.0f,
                                                                  DS4_ROPE_YARN_BETA_FAST,
                                                                  DS4_ROPE_YARN_BETA_SLOW) != 0;
                if (ok && glm_graph_indexer_qat()) {
                    ok = ds4_gpu_dsv4_indexer_qat_tensor(g->batch_indexer_q,
                                                         n_tokens * DS4_N_INDEXER_HEAD,
                                                         DS4_N_INDEXER_HEAD_DIM) != 0;
                }
                if (ok) {
                    ok = (use_batch_indexer_weights_proj ?
                          metal_graph_matmul_plain_tensor(g->batch_indexer_weights,
                                                          model,
                                                          l->indexer_proj,
                                                          DS4_N_EMBD,
                                                          DS4_N_INDEXER_HEAD,
                                                          cur,
                                                          n_tokens) :
                          glm_graph_matmul_plain_rows_scalar(g->batch_indexer_weights,
                                                             model,
                                                             l->indexer_proj,
                                                             DS4_N_EMBD,
                                                             DS4_N_INDEXER_HEAD,
                                                             cur,
                                                             n_tokens));
                }
                DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_attn", "indexer_q_weights");
            }
            if (ok) {
                if (use_causal_range_select) {
                    ok = ds4_gpu_glm_fill_selected_range_batch_tensor(
                                 g->batch_indexer_selected,
                                 n_tokens,
                                 pos0,
                                 indexed_selected_count,
                                 g->compact_cache_cap) != 0;
                    DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_attn", "indexer_range");
                } else if (use_scalar_indexer) {
                    const float indexer_scale =
                        1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
                    for (uint32_t t = 0; ok && t < n_tokens; t++) {
                        const uint32_t visible = pos0 + t + 1u;
                        ds4_gpu_tensor *indexer_q_view =
                            glm_graph_tensor_row_view_strided(
                                    g->batch_indexer_q,
                                    t,
                                    (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM,
                                    (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM);
                        ds4_gpu_tensor *indexer_weights_view =
                            glm_graph_tensor_row_view_strided(g->batch_indexer_weights,
                                                              t,
                                                              DS4_N_INDEXER_HEAD,
                                                              DS4_N_INDEXER_HEAD);
                        ds4_gpu_tensor *scores_view =
                            ds4_gpu_tensor_view(g->batch_indexer_scores,
                                                0,
                                                (uint64_t)visible * sizeof(float));
                        ds4_gpu_tensor *selected_view =
                            ds4_gpu_tensor_view(g->batch_indexer_selected,
                                                (uint64_t)t * indexed_selected_count * sizeof(uint32_t),
                                                (uint64_t)indexed_selected_count * sizeof(uint32_t));
                        ok = indexer_q_view && indexer_weights_view && scores_view && selected_view;
                        if (!ok) fprintf(stderr, "ds4: GLM scalar indexed prefill failed to create indexer row views at layer %u token %u\n", il, t);
                        if (ok) {
                            int rc = ds4_gpu_glm_indexer_score_one_tensor(
                                             scores_view,
                                             indexer_q_view,
                                             indexer_weights_view,
                                             g->layer_indexer_key_cache[il],
                                             visible,
                                             DS4_N_INDEXER_HEAD,
                                             DS4_N_INDEXER_HEAD_DIM,
                                             indexer_scale,
                                             glm_graph_compact_cache_is_f16());
                            ok = rc != 0;
                            if (!ok) fprintf(stderr, "ds4: GLM scalar indexed prefill indexer scores failed at layer %u token %u\n", il, t);
                        }
                        DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_attn", "indexer_score_scalar");
                        if (ok) {
                            int rc = ds4_gpu_indexer_topk_tensor(selected_view,
                                                                 scores_view,
                                                                 visible,
                                                                 1,
                                                                 indexed_selected_count);
                            ok = rc != 0;
                            if (!ok) fprintf(stderr, "ds4: GLM scalar indexed prefill topk failed at layer %u token %u\n", il, t);
                        }
                        DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_attn", "indexer_topk_scalar");
                        ds4_gpu_tensor_free(selected_view);
                        ds4_gpu_tensor_free(scores_view);
                        ds4_gpu_tensor_free(indexer_weights_view);
                        ds4_gpu_tensor_free(indexer_q_view);
                    }
                } else {
                    const float indexer_scale =
                        1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
                    const uint32_t score_cap =
                        g->indexed_prefill_score_cap != 0 ?
                            g->indexed_prefill_score_cap :
                            g->indexed_prefill_cap;
                    const uint64_t indexer_q_row_bytes =
                        (uint64_t)DS4_N_INDEXER_HEAD *
                        DS4_N_INDEXER_HEAD_DIM *
                        sizeof(float);
                    const uint64_t indexer_weights_row_bytes =
                        (uint64_t)DS4_N_INDEXER_HEAD * sizeof(float);
                    const uint64_t selected_row_bytes =
                        (uint64_t)indexed_selected_count * sizeof(uint32_t);
                    for (uint32_t t0 = 0; ok && t0 < n_tokens; ) {
                        uint32_t slice = n_tokens - t0;
                        if (slice > score_cap) slice = score_cap;
                        if (slice == 0) {
                            ok = false;
                            break;
                        }

                        ds4_gpu_tensor *indexer_q_view =
                            ds4_gpu_tensor_view(g->batch_indexer_q,
                                                (uint64_t)t0 * indexer_q_row_bytes,
                                                (uint64_t)slice * indexer_q_row_bytes);
                        ds4_gpu_tensor *indexer_weights_view =
                            ds4_gpu_tensor_view(g->batch_indexer_weights,
                                                (uint64_t)t0 * indexer_weights_row_bytes,
                                                (uint64_t)slice * indexer_weights_row_bytes);
                        ds4_gpu_tensor *selected_view =
                            ds4_gpu_tensor_view(g->batch_indexer_selected,
                                                (uint64_t)t0 * selected_row_bytes,
                                                (uint64_t)slice * selected_row_bytes);
                        ok = indexer_q_view && indexer_weights_view && selected_view;
                        if (!ok) {
                            fprintf(stderr,
                                    "ds4: GLM indexed prefill failed to create indexer score slice views at layer %u token %u\n",
                                    il,
                                    t0);
                        }
                        if (ok) {
                            ok = ds4_gpu_glm_indexer_scores_batch_tensor(
                                         g->batch_indexer_scores,
                                         indexer_q_view,
                                         indexer_weights_view,
                                         g->layer_indexer_key_cache[il],
                                         n_rows,
                                         slice,
                                         pos0 + t0,
                                         DS4_N_INDEXER_HEAD,
                                         DS4_N_INDEXER_HEAD_DIM,
                                         indexer_scale,
                                         glm_graph_compact_cache_is_f16()) != 0;
                        }
                        DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_attn", "indexer_score");
                        if (ok) {
                            ok = ds4_gpu_indexer_topk_tensor(selected_view,
                                                             g->batch_indexer_scores,
                                                             n_rows,
                                                             slice,
                                                             indexed_selected_count) != 0;
                        }
                        DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_attn", "indexer_topk");
                        ds4_gpu_tensor_free(selected_view);
                        ds4_gpu_tensor_free(indexer_weights_view);
                        ds4_gpu_tensor_free(indexer_q_view);
                        t0 += slice;
                    }
                }
            }
            if (ok) {
                last_indexer_selected = g->batch_indexer_selected;
                last_indexer_selected_count = indexed_selected_count;
            }
        } else if (ok && (!last_indexer_selected || last_indexer_selected_count == 0)) {
            ok = false;
        }
        DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_attn", "indexer_select");
        if (use_batch_qk_low) {
            if (ok) ok = ds4_gpu_glm_qk_lowrank_q8_0_batch_tensor(g->batch_qk_low,
                                                                  g->batch_q,
                                                                  model->map,
                                                                  model->size,
                                                                  l->attn_k_b->abs_offset,
                                                                  n_tokens,
                                                                  DS4_N_HEAD,
                                                                  DS4_N_KV_LORA,
                                                                  (uint32_t)g->q_nope,
                                                                  DS4_N_KEY_MLA) != 0;
        } else {
            for (uint32_t t = 0; ok && t < n_tokens; t++) {
                ds4_gpu_tensor *q_view =
                    glm_graph_tensor_row_view_strided(g->batch_q,
                                                      t,
                                                      g->q_dim,
                                                      g->q_dim);
                ds4_gpu_tensor *qk_low_view =
                    glm_graph_tensor_row_view_strided(
                            g->batch_qk_low,
                            t,
                            (uint64_t)DS4_N_HEAD * DS4_N_KV_LORA,
                            (uint64_t)DS4_N_HEAD * DS4_N_KV_LORA);
                ok = q_view && qk_low_view;
                if (!ok) fprintf(stderr, "ds4: GLM scalar indexed prefill failed to create qk-low row views at layer %u token %u\n", il, t);
                if (ok) {
                    int rc = ds4_gpu_glm_qk_lowrank_q8_0_tensor(qk_low_view,
                                                                q_view,
                                                                model->map,
                                                                model->size,
                                                                l->attn_k_b->abs_offset,
                                                                DS4_N_HEAD,
                                                                DS4_N_KV_LORA,
                                                                (uint32_t)g->q_nope,
                                                                DS4_N_KEY_MLA);
                    ok = rc != 0;
                    if (!ok) fprintf(stderr, "ds4: GLM scalar indexed prefill qk-low failed at layer %u token %u\n", il, t);
                }
                ds4_gpu_tensor_free(qk_low_view);
                ds4_gpu_tensor_free(q_view);
            }
        }
        DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_attn", "qk_low");
        if (ok && use_batch_attn_kernel) ok = ds4_gpu_flush_encoder() != 0;

        if (use_batch_attn_kernel) {
            const uint32_t attn_slice_cap =
                glm_graph_indexed_prefill_batch_attn_slice_tokens();
            for (uint32_t t0 = 0; ok && t0 < n_tokens; ) {
                uint32_t slice = n_tokens - t0;
                if (slice > attn_slice_cap) slice = attn_slice_cap;

                ds4_gpu_tensor *q_view =
                    ds4_gpu_tensor_view(g->batch_q,
                                        (uint64_t)t0 * g->q_dim * sizeof(float),
                                        (uint64_t)slice * g->q_dim * sizeof(float));
                ds4_gpu_tensor *qk_low_view =
                    ds4_gpu_tensor_view(g->batch_qk_low,
                                        (uint64_t)t0 * DS4_N_HEAD * DS4_N_KV_LORA * sizeof(float),
                                        (uint64_t)slice * DS4_N_HEAD * DS4_N_KV_LORA * sizeof(float));
                ds4_gpu_tensor *heads_view =
                    ds4_gpu_tensor_view(g->batch_heads,
                                        (uint64_t)t0 * g->heads_dim * sizeof(float),
                                        (uint64_t)slice * g->heads_dim * sizeof(float));
                ds4_gpu_tensor *attn_lora_view = use_split_value_proj ?
                    ds4_gpu_tensor_view(g->batch_attn_lora,
                                        (uint64_t)t0 * DS4_N_HEAD * DS4_N_KV_LORA * sizeof(float),
                                        (uint64_t)slice * DS4_N_HEAD * DS4_N_KV_LORA * sizeof(float)) :
                    NULL;
                ds4_gpu_tensor *selected_view =
                    ds4_gpu_tensor_view(last_indexer_selected,
                                        (uint64_t)t0 * last_indexer_selected_count * sizeof(uint32_t),
                                        (uint64_t)slice * last_indexer_selected_count * sizeof(uint32_t));
                ok = q_view && qk_low_view && heads_view && selected_view &&
                     (!use_split_value_proj || attn_lora_view);
                if (!ok) {
                    fprintf(stderr, "ds4: GLM sliced indexed prefill failed to create attention views at layer %u token %u\n", il, t0);
                }
                if (ok && use_split_value_proj) {
                    int rc = 0;
                    if (use_causal_range_select) {
                        rc = ds4_gpu_glm_attention_indexed_batch_lora_causal_tensor(
                                attn_lora_view,
                                q_view,
                                qk_low_view,
                                g->layer_kv_lora_cache[il],
                                g->layer_k_rope_cache[il],
                                slice,
                                pos0 + t0,
                                last_indexer_selected_count,
                                g->compact_cache_cap,
                                glm_graph_compact_cache_is_f16(),
                                DS4_N_HEAD,
                                DS4_N_KV_LORA,
                                (uint32_t)g->q_nope,
                                DS4_N_ROT,
                                0,
                                rope_base,
                                rope_scale,
                                0.0f,
                                1.0f,
                                DS4_ROPE_YARN_BETA_FAST,
                                DS4_ROPE_YARN_BETA_SLOW);
                    } else {
                        rc = ds4_gpu_glm_attention_indexed_batch_lora_valid_tensor(
                                attn_lora_view,
                                q_view,
                                qk_low_view,
                                g->layer_kv_lora_cache[il],
                                g->layer_k_rope_cache[il],
                                selected_view,
                                slice,
                                last_indexer_selected_count,
                                g->compact_cache_cap,
                                glm_graph_compact_cache_is_f16(),
                                DS4_N_HEAD,
                                DS4_N_KV_LORA,
                                (uint32_t)g->q_nope,
                                DS4_N_ROT,
                                0,
                                rope_base,
                                rope_scale,
                                0.0f,
                                1.0f,
                                DS4_ROPE_YARN_BETA_FAST,
                                DS4_ROPE_YARN_BETA_SLOW);
                    }
                    ok = rc != 0;
                    if (!ok) fprintf(stderr, "ds4: GLM sliced indexed prefill attention-lora failed at layer %u token %u\n", il, t0);
                    if (ok && layer_stage_profile) {
                        ok = metal_graph_layer_stage_profile_boundary("glm_indexed_attn",
                                                                      "attention_lora",
                                                                      il,
                                                                      pos0 + t0,
                                                                      slice,
                                                                      &layer_stage_t0);
                    }
                    if (ok) {
                        rc = ds4_gpu_glm_value_project_q8_0_batch_heads_tensor(
                                    heads_view,
                                    attn_lora_view,
                                    model->map,
                                    model->size,
                                    l->attn_v_b->abs_offset,
                                    slice,
                                    DS4_N_HEAD,
                                    DS4_N_KV_LORA,
                                    DS4_N_VALUE_MLA);
                        ok = rc != 0;
                        if (!ok) fprintf(stderr, "ds4: GLM sliced indexed prefill value project failed at layer %u token %u\n", il, t0);
                    }
                    if (ok && layer_stage_profile) {
                        ok = metal_graph_layer_stage_profile_boundary("glm_indexed_attn",
                                                                      "value_project",
                                                                      il,
                                                                      pos0 + t0,
                                                                      slice,
                                                                      &layer_stage_t0);
                    }
                } else if (ok) {
                    int rc = ds4_gpu_glm_attention_indexed_batch_tensor(heads_view,
                                                                        q_view,
                                                                        qk_low_view,
                                                                        g->layer_kv_lora_cache[il],
                                                                        g->layer_k_rope_cache[il],
                                                                        model->map,
                                                                        model->size,
                                                                        l->attn_v_b->abs_offset,
                                                                        selected_view,
                                                                        slice,
                                                                        last_indexer_selected_count,
                                                                        g->compact_cache_cap,
                                                                        glm_graph_compact_cache_is_f16(),
                                                                        DS4_N_HEAD,
                                                                        DS4_N_KV_LORA,
                                                                        (uint32_t)g->q_nope,
                                                                        DS4_N_ROT,
                                                                        DS4_N_VALUE_MLA,
                                                                        0,
                                                                        rope_base,
                                                                        rope_scale,
                                                                        0.0f,
                                                                        1.0f,
                                                                        DS4_ROPE_YARN_BETA_FAST,
                                                                        DS4_ROPE_YARN_BETA_SLOW);
                    ok = rc != 0;
                    if (!ok) fprintf(stderr, "ds4: GLM sliced indexed prefill indexed attention failed at layer %u token %u\n", il, t0);
                    if (ok && layer_stage_profile) {
                        ok = metal_graph_layer_stage_profile_boundary("glm_indexed_attn",
                                                                      "attention_fused",
                                                                      il,
                                                                      pos0 + t0,
                                                                      slice,
                                                                      &layer_stage_t0);
                    }
                }
                ds4_gpu_tensor_free(selected_view);
                ds4_gpu_tensor_free(attn_lora_view);
                ds4_gpu_tensor_free(heads_view);
                ds4_gpu_tensor_free(qk_low_view);
                ds4_gpu_tensor_free(q_view);
                t0 += slice;
            }
            if (ok) ok = ds4_gpu_flush_encoder() != 0;
        } else {
            for (uint32_t t = 0; ok && t < n_tokens; t++) {
                ds4_gpu_tensor *q_view =
                    glm_graph_tensor_row_view_strided(g->batch_q,
                                                      t,
                                                      g->q_dim,
                                                      g->q_dim);
                ds4_gpu_tensor *qk_low_view =
                    glm_graph_tensor_row_view_strided(
                            g->batch_qk_low,
                            t,
                            (uint64_t)DS4_N_HEAD * DS4_N_KV_LORA,
                            (uint64_t)DS4_N_HEAD * DS4_N_KV_LORA);
                ds4_gpu_tensor *heads_view =
                    glm_graph_tensor_row_view_strided(g->batch_heads,
                                                      t,
                                                      g->heads_dim,
                                                      g->heads_dim);
                ds4_gpu_tensor *selected_view =
                    ds4_gpu_tensor_view(last_indexer_selected,
                                        (uint64_t)t * last_indexer_selected_count * sizeof(uint32_t),
                                        (uint64_t)last_indexer_selected_count * sizeof(uint32_t));
                ok = q_view && qk_low_view && heads_view && selected_view;
                if (!ok) fprintf(stderr, "ds4: GLM scalar indexed prefill failed to create attention row views at layer %u token %u\n", il, t);
                if (ok) {
                    int rc = ds4_gpu_glm_attention_indexed_decode_tensor(heads_view,
                                                                          q_view,
                                                                          qk_low_view,
                                                                          g->layer_kv_lora_cache[il],
                                                                          g->layer_k_rope_cache[il],
                                                                          model->map,
                                                                          model->size,
                                                                          l->attn_v_b->abs_offset,
                                                                          selected_view,
                                                                          last_indexer_selected_count,
                                                                          g->compact_cache_cap,
                                                                          glm_graph_compact_cache_is_f16(),
                                                                          DS4_N_HEAD,
                                                                          DS4_N_KV_LORA,
                                                                          (uint32_t)g->q_nope,
                                                                          DS4_N_ROT,
                                                                          DS4_N_VALUE_MLA,
                                                                          0,
                                                                          rope_base,
                                                                          rope_scale,
                                                                          0.0f,
                                                                          1.0f,
                                                                          DS4_ROPE_YARN_BETA_FAST,
                                                                          DS4_ROPE_YARN_BETA_SLOW);
                    ok = rc != 0;
                    if (!ok) fprintf(stderr, "ds4: GLM scalar indexed prefill indexed attention failed at layer %u token %u\n", il, t);
                }
                ds4_gpu_tensor_free(selected_view);
                ds4_gpu_tensor_free(heads_view);
                ds4_gpu_tensor_free(qk_low_view);
                ds4_gpu_tensor_free(q_view);
            }
        }
        DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_attn", "attention");
        metal_graph_debug_dump_tensor("glm_indexed_heads",
                                      g->batch_heads,
                                      (uint64_t)n_tokens * g->heads_dim,
                                      il,
                                      pos0);
        if (ok) {
            ok = (use_batch_attn_out_proj ?
                  glm_graph_matmul_q8_0_tensor(g->batch_attn_out,
                                               model,
                                               l->attn_output->abs_offset,
                                               g->heads_dim,
                                               DS4_N_EMBD,
                                               g->batch_heads,
                                               n_tokens) :
                  glm_graph_matmul_q8_0_rows_scalar(g->batch_attn_out,
                                                    model,
                                                    l->attn_output->abs_offset,
                                                    g->heads_dim,
                                                    DS4_N_EMBD,
                                                    g->batch_heads,
                                                    n_tokens));
        }
        if (ok) ok = ds4_gpu_add_tensor(g->batch_after_attn,
                                        cur,
                                        g->batch_attn_out,
                                        (uint32_t)residual_elems) != 0;
        DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_attn", "attn_output");
        metal_graph_debug_dump_tensor("glm_indexed_after_attn",
                                      g->batch_after_attn,
                                      (uint64_t)n_tokens * DS4_N_EMBD,
                                      il,
                                      pos0);
        metal_graph_debug_dump_tensor("glm_indexed_attn_out",
                                      g->batch_attn_out,
                                      (uint64_t)n_tokens * DS4_N_EMBD,
                                      il,
                                      pos0);
        if (ok && use_batch_ffn) {
            ok = glm_graph_encode_ffn_batch(g,
                                            model,
                                            l,
                                            il,
                                            pos0,
                                            g->batch_after_attn,
                                            next,
                                            n_tokens,
                                            layer_stage_profile,
                                            stage_sync,
                                            layer_stage_profile ? &layer_stage_t0 : NULL);
        } else if (ok) {
            const bool use_batch_ffn_norm =
                n_tokens > 1 && glm_graph_indexed_prefill_batch_ffn_norm();
            if (use_batch_ffn_norm) {
                ok = ds4_gpu_rms_norm_weight_rows_tensor(g->batch_ffn_norm,
                                                          g->batch_after_attn,
                                                          model->map,
                                                          model->size,
                                                          l->ffn_norm->abs_offset,
                                                          DS4_N_EMBD,
                                                          n_tokens,
                                                          DS4_RMS_EPS) != 0;
            }
            if (use_batch_ffn_norm) {
                DS4_GLM_PROFILE_INDEXED_STAGE("glm_indexed_ffn", "ffn_norm");
            }
            if (ok &&
                use_batch_ffn_norm &&
                il >= DS4_N_LEADING_DENSE &&
                glm_graph_indexed_prefill_batch_routed_moe()) {
                ok = glm_graph_encode_sparse_ffn_indexed_batch_routed_moe(
                        g,
                        model,
                        l,
                        il,
                        pos0,
                        g->batch_after_attn,
                        next,
                        n_tokens,
                        layer_stage_profile,
                        stage_sync,
                        layer_stage_profile ? &layer_stage_t0 : NULL);
            } else for (uint32_t t = 0; ok && t < n_tokens; t++) {
                ds4_gpu_tensor *after_attn_view =
                    glm_graph_tensor_row_view_strided(g->batch_after_attn,
                                                      t,
                                                      DS4_N_EMBD,
                                                      DS4_N_EMBD);
                ds4_gpu_tensor *ffn_norm_view = use_batch_ffn_norm ?
                    glm_graph_tensor_row_view_strided(g->batch_ffn_norm,
                                                      t,
                                                      DS4_N_EMBD,
                                                      DS4_N_EMBD) :
                    NULL;
                ds4_gpu_tensor *next_view =
                    glm_graph_tensor_row_view_strided(next,
                                                      t,
                                                      DS4_N_EMBD,
                                                      DS4_N_EMBD);
                ok = after_attn_view && next_view &&
                     (!use_batch_ffn_norm || ffn_norm_view);
                if (ok && use_batch_ffn_norm) {
                    ok = glm_graph_encode_ffn_one_normed_from(g,
                                                              model,
                                                              l,
                                                              il,
                                                              pos0 + t,
                                                              ffn_norm_view,
                                                              after_attn_view,
                                                              next_view,
                                                              g->ffn_gate,
                                                              g->ffn_up,
                                                              g->ffn_mid,
                                                              g->ffn_out,
                                                              g->ffn_sum,
                                                              g->attn_out,
                                                              false,
                                                              NULL);
                } else if (ok) {
                    ok = glm_graph_encode_ffn_one_from(g,
                                                       model,
                                                       l,
                                                       il,
                                                       pos0 + t,
                                                       after_attn_view,
                                                       next_view,
                                                       g->ffn_norm,
                                                       g->ffn_gate,
                                                       g->ffn_up,
                                                       g->ffn_mid,
                                                       g->ffn_out,
                                                       g->ffn_sum,
                                                       g->attn_out,
                                                       false,
                                                       NULL);
                }
                ds4_gpu_tensor_free(next_view);
                ds4_gpu_tensor_free(ffn_norm_view);
                ds4_gpu_tensor_free(after_attn_view);
            }
        }
        if (ok) {
            ds4_gpu_tensor *tmp = cur;
            cur = next;
            next = tmp;
        }
        if (ok &&
            !g->ssd_streaming &&
            progress_flush_interval != 0 &&
            (il < g->layer_end || progress_requested) &&
            (slice_layer_done % progress_flush_interval) == 0) {
            const uint32_t work_done =
                work_done_base + (uint32_t)(((uint64_t)n_tokens * slice_layer_done) / g->layer_count);
            const bool drain_now =
                drain_interval != 0 &&
                il < g->layer_end &&
                (slice_layer_done % drain_interval) == 0;
            const char *command_action = drain_now ? "drain" : "flush";
            const double trace_command_t0 = trace ? now_sec() : 0.0;
            if (trace && (trace_all || trace_full_indexer || drain_now)) {
                glm_graph_indexed_prefill_tracef(
                        "layer %s begin layer=%u pos=%u tokens=%u work=%u/%u",
                        command_action,
                        il,
                        pos0,
                        n_tokens,
                        work_done,
                        work_total);
            }
            if (drain_now) {
                ok = ds4_gpu_end_commands() != 0;
                if (ok) ok = ds4_gpu_begin_commands() != 0;
            } else {
                ok = ds4_gpu_flush_commands() != 0;
            }
            if (trace) {
                const double trace_command_done = now_sec();
                const double command_ms = (trace_command_done - trace_command_t0) * 1000.0;
                const double layer_ms = (trace_command_done - trace_layer_t0) * 1000.0;
                trace_layer_flushed = true;
                if (trace_all || trace_full_indexer || drain_now ||
                    command_ms >= trace_slow_ms || layer_ms >= trace_slow_ms || !ok) {
                    glm_graph_indexed_prefill_tracef(
                            "layer %s %s layer=%u pos=%u tokens=%u command=%.3f ms layer_total=%.3f ms work=%u/%u",
                            command_action,
                            ok ? "done" : "failed",
                            il,
                            pos0,
                            n_tokens,
                            command_ms,
                            layer_ms,
                            work_done,
                            work_total);
                }
            }
            if (ok) {
                const bool progress_completed =
                    drain_interval == 0 || drain_now;
                if (progress_completed) {
                    glm_graph_report_prefill_display_progress(display_progress,
                                                              display_progress_ud,
                                                              display_absolute_base,
                                                              work_done_base,
                                                              n_tokens,
                                                              slice_layer_done,
                                                              g->layer_count,
                                                              work_total,
                                                              logits_out == NULL && output_hc == NULL);
                }
            }
        }
        if (g->ssd_streaming) {
            if (ok) ok = ds4_gpu_end_commands() != 0;
            else (void)ds4_gpu_synchronize();
            if (ok) {
                glm_graph_report_prefill_display_progress(display_progress,
                                                          display_progress_ud,
                                                          display_absolute_base,
                                                          work_done_base,
                                                          n_tokens,
                                                          slice_layer_done,
                                                          g->layer_count,
                                                          work_total,
                                                          logits_out == NULL && output_hc == NULL);
            }
        }
        if (trace && ok) {
            const double layer_ms = (now_sec() - trace_layer_t0) * 1000.0;
            if (trace_all || (!trace_layer_flushed && layer_ms >= trace_slow_ms)) {
                glm_graph_indexed_prefill_tracef(
                        "layer end layer=%u pos=%u tokens=%u flushed=%u layer_total=%.3f ms",
                        il,
                        pos0,
                        n_tokens,
                        trace_layer_flushed ? 1u : 0u,
                        layer_ms);
            }
        }
    }
#undef DS4_GLM_PROFILE_INDEXED_STAGE
    if (ok && !g->ssd_streaming) {
        const double trace_end_t0 = trace ? now_sec() : 0.0;
        if (trace) {
            glm_graph_indexed_prefill_tracef(
                    "chunk end_commands begin pos=%u tokens=%u",
                    pos0,
                    n_tokens);
        }
        ok = ds4_gpu_end_commands() != 0;
        if (trace) {
            const double end_ms = (now_sec() - trace_end_t0) * 1000.0;
            const double chunk_ms = (now_sec() - trace_chunk_t0) * 1000.0;
            glm_graph_indexed_prefill_tracef(
                    "chunk end_commands %s pos=%u tokens=%u end=%.3f ms chunk_total=%.3f ms",
                    ok ? "done" : "failed",
                    pos0,
                    n_tokens,
                    end_ms,
                    chunk_ms);
        }
    } else if (!ok) {
        if (trace) {
            glm_graph_indexed_prefill_tracef(
                    "chunk failed before end pos=%u tokens=%u elapsed=%.3f ms",
                    pos0,
                    n_tokens,
                    (now_sec() - trace_chunk_t0) * 1000.0);
        }
        (void)ds4_gpu_synchronize();
    }
    if (idx_prepare &&
        !metal_graph_stream_prepare_join_all(idx_prepare_slots,
                                             DS4_STREAM_PREFILL_MAX_PREPARE_AHEAD)) {
        ok = false;
    }
    if (ok) {
        glm_graph_report_prefill_display_progress(display_progress,
                                                  display_progress_ud,
                                                  display_absolute_base,
                                                  work_done_base,
                                                  n_tokens,
                                                  g->layer_count,
                                                  g->layer_count,
                                                  work_total,
                                                  logits_out == NULL && output_hc == NULL);
    }
    if (ok && output_hc) {
        ok = ds4_gpu_tensor_read(cur,
                                 0,
                                 output_hc,
                                 (uint64_t)n_tokens * DS4_N_EMBD * sizeof(float)) != 0;
    }
    if (ok && logits_out) {
        last_hidden = glm_graph_tensor_row_view_strided(cur,
                                                        n_tokens - 1u,
                                                        DS4_N_EMBD,
                                                        DS4_N_EMBD);
        ok = last_hidden != NULL;
        if (ok && g->ssd_streaming) ok = glm_graph_stream_map_output(g, model, weights);
        if (ok) ok = glm_graph_forward_output_head(g, model, weights, last_hidden, logits_out);
        if (ok) {
            glm_graph_report_prefill_display_progress(display_progress,
                                                      display_progress_ud,
                                                      display_absolute_base,
                                                      work_done_base,
                                                      n_tokens,
                                                      g->layer_count,
                                                      g->layer_count,
                                                      work_total,
                                                      true);
        }
    }
    ds4_gpu_tensor_free(last_hidden);
    ds4_gpu_set_glm_streaming_prefill_full_layer(false);
    return ok;
}

static bool glm_graph_use_streaming_token_prefill(
        const ds4_glm_gpu_graph *g,
        uint32_t                 pos0,
        uint32_t                 n_tokens);
static bool glm_graph_prefill_token_major(
        ds4_glm_gpu_graph      *g,
        const ds4_model        *model,
        const ds4_weights      *weights,
        const int              *tokens,
        uint32_t                pos0,
        uint32_t                n_tokens,
        float                  *logits_out,
        ds4_session_progress_fn display_progress,
        void                   *display_progress_ud,
        uint32_t                display_absolute_base,
        uint32_t                work_done_base,
        uint32_t                work_total);

static bool glm_graph_prefill_range(
        ds4_glm_gpu_graph      *g,
        const ds4_model        *model,
        const ds4_weights      *weights,
        const int              *tokens,
        uint32_t                pos0,
        uint32_t                n_tokens,
        float                  *logits_out,
        ds4_session_progress_fn progress,
        void                   *progress_ud,
        uint32_t                progress_total) {
    if (n_tokens == 0) return true;
    if (!glm_graph_span_fits_context(g, pos0, n_tokens)) return false;
    const uint32_t chunk_max = glm_graph_prefill_chunk_tokens(g->ctx_cap);
    uint32_t done = 0;
    while (done < n_tokens) {
        const uint32_t pos = pos0 + done;
        if (!g->full_kv_cache) {
            uint32_t chunk = n_tokens - done;
            if (chunk > g->indexed_prefill_cap) chunk = g->indexed_prefill_cap;
            if (glm_graph_use_streaming_token_prefill(g, pos, chunk)) {
                float *dst_logits = (done + chunk == n_tokens) ? logits_out : NULL;
                if (!glm_graph_prefill_token_major(g,
                                                   model,
                                                   weights,
                                                   tokens + done,
                                                   pos,
                                                   chunk,
                                                   dst_logits,
                                                   progress,
                                                   progress_ud,
                                                   pos0,
                                                   done,
                                                   n_tokens)) {
                    return false;
                }
            } else if (glm_graph_indexed_prefill_batch_ready(g, pos)) {
                chunk = glm_graph_limit_indexed_prefill_chunk(pos, chunk);
                if (chunk == 0) chunk = 1;
                float *dst_logits = (done + chunk == n_tokens) ? logits_out : NULL;
                if (!glm_graph_forward_indexed_tokens(g,
                                                      model,
                                                      weights,
                                                      tokens + done,
                                                      NULL,
                                                      pos,
                                                      chunk,
                                                      NULL,
                                                      dst_logits,
                                                      progress,
                                                      progress_ud,
                                                      pos0,
                                                      done,
                                                      n_tokens)) {
                    return false;
                }
            } else {
                return false;
            }
            done += chunk;
            if (progress) {
                const uint32_t current = pos0 + done;
                progress(progress_ud,
                         "prefill_chunk",
                         current,
                         progress_total ? progress_total : pos0 + n_tokens);
            }
            continue;
        }
        if (pos >= g->ctx_cap) {
            if (g->compact_cache_cap == 0) {
                glm_graph_log_full_attention_limit(g, pos, n_tokens - done);
                return false;
            }
            while (done < n_tokens) {
                const uint32_t cur_pos = pos0 + done;
                const bool use_indexed_batch =
                    glm_graph_indexed_prefill_batch_ready(g, cur_pos);
                if (use_indexed_batch) {
                    uint32_t chunk = n_tokens - done;
                    if (chunk > g->indexed_prefill_cap) chunk = g->indexed_prefill_cap;
                    chunk = glm_graph_limit_indexed_prefill_chunk(cur_pos, chunk);
                    float *dst_logits = (done + chunk == n_tokens) ? logits_out : NULL;
                    if (!glm_graph_forward_indexed_tokens(g,
                                                          model,
                                                          weights,
                                                          tokens + done,
                                                          NULL,
                                                          cur_pos,
                                                          chunk,
                                                          NULL,
                                                          dst_logits,
                                                          progress,
                                                          progress_ud,
                                                          pos0,
                                                          done,
                                                          n_tokens)) {
                        return false;
                    }
                    done += chunk;
                } else {
                    float *dst_logits = (done + 1u == n_tokens) ? logits_out : NULL;
                    if (!glm_graph_forward_token(g,
                                                 model,
                                                 weights,
                                                 tokens[done],
                                                 NULL,
                                                 cur_pos,
                                                 NULL,
                                                 dst_logits)) {
                        return false;
                    }
                    done++;
                }
                if (progress) {
                    const uint32_t current = pos0 + done;
                    progress(progress_ud,
                             "prefill_chunk",
                             current,
                             progress_total ? progress_total : pos0 + n_tokens);
                }
            }
            return true;
        }
        uint32_t chunk = n_tokens - done;
        const uint32_t full_remaining = g->ctx_cap - pos;
        if (chunk > full_remaining) chunk = full_remaining;
        if (chunk > chunk_max) chunk = chunk_max;
        float *dst_logits = (done + chunk == n_tokens) ? logits_out : NULL;
        if (glm_graph_use_streaming_token_prefill(g, pos, chunk)) {
            if (!glm_graph_prefill_token_major(g,
                                               model,
                                               weights,
                                               tokens + done,
                                               pos,
                                               chunk,
                                               dst_logits,
                                               progress,
                                               progress_ud,
                                               pos0,
                                               done,
                                               n_tokens)) {
                return false;
            }
        } else if (!glm_graph_forward_tokens(g,
                                             model,
                                             weights,
                                             tokens + done,
                                             NULL,
                                             pos,
                                             chunk,
                                             NULL,
                                             dst_logits,
                                             progress,
                                             progress_ud,
                                             pos0,
                                             done,
                                             n_tokens)) {
            return false;
        }
        done += chunk;
        if (progress) {
            const uint32_t current = pos0 + done;
            progress(progress_ud,
                     "prefill_chunk",
                     current,
                     progress_total ? progress_total : pos0 + n_tokens);
        }
    }
    return true;
}

/*
 * For very short GLM SSD-streaming prefills, the canonical batch path pays the
 * full streamed-prefill setup cost and leaves the decode expert cache cold for
 * the first generated token. The token-major path reuses the normal decode
 * graph, which is not strict batch-equivalent but is the latency/throughput
 * path users expect for interactive prompts. --quality and the env escape hatch
 * keep canonical prefill available for strict-vector diagnostics.
 */
enum { DS4_GLM_STREAM_PREFILL_TOKEN_MAJOR_MAX_TOKENS = 64 };

static bool glm_graph_use_streaming_token_prefill(
        const ds4_glm_gpu_graph *g,
        uint32_t                 pos0,
        uint32_t                 n_tokens) {
    if (!g || !g->ssd_streaming || g->quality || n_tokens == 0) return false;
    if (getenv("DS4_GLM_DISABLE_STREAMING_TOKEN_PREFILL") != NULL ||
        getenv("DS4_METAL_GLM_DISABLE_STREAMING_TOKEN_PREFILL") != NULL) {
        return false;
    }
    if (!glm_graph_span_fits_full_attention(g, pos0, n_tokens)) return false;
    return n_tokens <= DS4_GLM_STREAM_PREFILL_TOKEN_MAJOR_MAX_TOKENS;
}

static bool glm_graph_prefill_token_major(
        ds4_glm_gpu_graph      *g,
        const ds4_model        *model,
        const ds4_weights      *weights,
        const int              *tokens,
        uint32_t                pos0,
        uint32_t                n_tokens,
        float                  *logits_out,
        ds4_session_progress_fn display_progress,
        void                   *display_progress_ud,
        uint32_t                display_absolute_base,
        uint32_t                work_done_base,
        uint32_t                work_total) {
    if (!g || !model || !weights || !tokens || n_tokens == 0) return false;
    for (uint32_t i = 0; i < n_tokens; i++) {
        const bool last = i + 1u == n_tokens;
        float *dst_logits = (last && logits_out) ? logits_out : NULL;
        if (!glm_graph_forward_token(g,
                                     model,
                                     weights,
                                     tokens[i],
                                     NULL,
                                     pos0 + i,
                                     NULL,
                                     dst_logits)) {
            return false;
        }
        glm_graph_report_prefill_display_progress(display_progress,
                                                  display_progress_ud,
                                                  display_absolute_base,
                                                  work_done_base + i + 1u,
                                                  n_tokens,
                                                  0,
                                                  1,
                                                  work_total,
                                                  false);
    }
    return true;
}

static bool glm_graph_maybe_warm_compact_indexer_after_prefill(
        ds4_glm_gpu_graph      *g,
        const ds4_model        *model,
        const ds4_weights      *weights,
        uint32_t                next_pos) {
    if (!g || !model || !weights) return false;
    if (g->compact_cache_cap == 0 ||
        g->indexer_full_layers == 0 ||
        next_pos < g->ctx_cap) {
        return true;
    }
    if (next_pos >= g->ctx_size) return true;
    return glm_graph_warm_compact_indexer_store(g, model, weights, next_pos);
}

static bool glm_graph_begin_commands_if_needed(void) {
    return ds4_gpu_commands_active() || ds4_gpu_begin_commands() != 0;
}

static bool glm_graph_end_commands_if_active(void) {
    return !ds4_gpu_commands_active() || ds4_gpu_end_commands() != 0;
}

static bool glm_graph_forward_token(
        ds4_glm_gpu_graph *g,
        const ds4_model   *model,
        const ds4_weights *weights,
        int                token,
        const float       *input_hc,
        uint32_t           pos,
        float             *output_hc,
        float             *logits_out) {
    if (!g || !model || !weights ||
        token < 0 || token >= (int)DS4_N_VOCAB ||
        g->layer_count == 0 ||
        pos >= g->ctx_size) {
        return false;
    }
    if (!input_hc && !g->has_token_embd) return false;
    if (logits_out && !g->has_output_head) return false;
    g_glm_router_ahead_weights = weights;
    const bool use_indexed_attention =
        glm_graph_decode_uses_indexed_attention(g, pos, logits_out);
    uint32_t decode_layer_flush_interval = 0;
    if (logits_out != NULL) {
        decode_layer_flush_interval = use_indexed_attention ? 4u : 32u;
        if (decode_layer_flush_interval > g->layer_count) {
            decode_layer_flush_interval = g->layer_count;
        }
    }
    if (pos >= g->ctx_cap && !use_indexed_attention) {
        glm_graph_log_full_attention_limit(g, pos, 1);
        return false;
    }
    if (g->compact_cache_cap != 0 &&
        !glm_graph_ensure_compact_cache(g, pos + 1u)) {
        return false;
    }

    const bool decode_output_profile = false;
    const bool merge_indexed_output =
        logits_out != NULL && use_indexed_attention && !decode_output_profile;
    double decode_output_stage_t0 = decode_output_profile ? now_sec() : 0.0;
    const bool decode_flush_profile = false;
    uint32_t decode_flush_layer0 = 0;
    double decode_flush_stage_t0 = decode_flush_profile ? now_sec() : 0.0;

    const bool static_decode_map =
        !input_hc &&
        g->has_token_embd &&
        g->ssd_streaming &&
        metal_graph_stream_decode_static_map_enabled();
    const bool static_map_state_cache =
        static_decode_map && metal_graph_stream_decode_static_map_state_cache_enabled();
    bool ok = true;
    if (input_hc) {
        ok = ds4_gpu_tensor_write(g->cur,
                                  0,
                                  input_hc,
                                  (uint64_t)DS4_N_EMBD * sizeof(float)) != 0;
    } else if (static_decode_map) {
        if (!static_map_state_cache || !g->streaming_static_decode_map_current) {
            ok = metal_graph_stream_map_decode_static_all(model, weights);
            if (ok) g->streaming_static_decode_map_current = static_map_state_cache;
        }
    } else {
        ok = glm_graph_stream_map_token(g, model, weights);
    }
    if (ok) ok = glm_graph_begin_commands_if_needed();
    if (ok && !input_hc) {
        ok = glm_graph_embed_token_tensor(g->cur,
                                          model,
                                          weights->token_embd,
                                          (uint32_t)token) != 0;
    }
    if (ok && g->ssd_streaming && !static_decode_map) {
        ok = ds4_gpu_end_commands() != 0;
    }
    const uint32_t indexer_top_k = glm_graph_indexer_top_k_limit();
    ds4_gpu_tensor *last_indexer_selected = NULL;
    uint32_t last_indexer_selected_count = 0;
#define DS4_GLM_PROFILE_DECODE_STAGE(part_, name_) do { \
        if (ok && decode_stage_profile) { \
            ok = metal_graph_layer_stage_profile_boundary((part_), (name_), il, pos, 1, &decode_stage_t0); \
        } \
    } while (0)
    for (uint32_t il = g->layer_start; ok && il <= g->layer_end; il++) {
        const uint32_t slice_layer_done = il - g->layer_start + 1u;
        if (g->ssd_streaming) {
            if (!static_decode_map) {
                ok = glm_graph_stream_map_decode_layer(g, model, weights, il);
            }
            if (ok) ok = glm_graph_begin_commands_if_needed();
        }
        const ds4_layer_weights *l = &weights->layer[il];
        const uint32_t kv_raw_dim = (uint32_t)l->attn_kv_a_mqa->dim[1];
        const float rope_base = layer_rope_freq_base(il);
        const float rope_scale = layer_rope_freq_scale(il);
        const bool decode_stage_profile = metal_graph_decode_stage_profile_enabled(il);
        double decode_stage_t0 = decode_stage_profile ? now_sec() : 0.0;
        if (decode_stage_profile) {
            ok = metal_graph_layer_stage_profile_boundary("glm_decode_attn",
                                                          NULL,
                                                          il,
                                                          pos,
                                                          1,
                                                          &decode_stage_t0);
        }

        if (ok) ok = ds4_gpu_rms_norm_weight_tensor(g->attn_norm,
                                                    g->cur,
                                                    model->map,
                                                    model->size,
                                                    l->attn_norm->abs_offset,
                                                    DS4_N_EMBD,
                                                    DS4_RMS_EPS) != 0;
        DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "attn_norm");
        if (ok) ok = glm_graph_matmul_q8_0_decode_profiled_tensor(g->q_rank,
                                                                  model,
                                                                  l->attn_q_a->abs_offset,
                                                                  DS4_N_EMBD,
                                                                  DS4_N_LORA_Q,
                                                                  g->attn_norm,
                                                                  il,
                                                                  pos,
                                                                  "attn_q_a",
                                                                  g->ssd_streaming) != 0;
        const bool fuse_qkv_norm_store = use_indexed_attention &&
                                         !decode_stage_profile &&
                                         g->compact_cache_cap != 0;
        const bool fuse_qkv_norm = !decode_stage_profile && !fuse_qkv_norm_store;
        if (ok && fuse_qkv_norm_store) {
            ok = glm_graph_matmul_q8_0_decode_profiled_tensor(g->kv_raw,
                                                              model,
                                                              l->attn_kv_a_mqa->abs_offset,
                                                              DS4_N_EMBD,
                                                              kv_raw_dim,
                                                              g->attn_norm,
                                                              il,
                                                              pos,
                                                              "attn_kv_a_store",
                                                              g->ssd_streaming) != 0;
            if (ok) {
                ok = ds4_gpu_glm_qkv_norm_store_compact_kv_tensor(
                        g->q_rank_norm,
                        g->q_rank,
                        model->map,
                        model->size,
                        l->attn_q_a_norm->abs_offset,
                        DS4_N_LORA_Q,
                        g->layer_kv_lora_cache[il],
                        g->layer_k_rope_cache[il],
                        g->kv_raw,
                        l->attn_kv_a_norm->abs_offset,
                        pos,
                        1,
                        g->compact_cache_cap,
                        kv_raw_dim,
                        DS4_N_KV_LORA,
                        DS4_N_ROT,
                        glm_graph_compact_cache_is_f16(),
                        DS4_RMS_EPS) != 0;
            }
        } else if (ok && fuse_qkv_norm) {
            ok = glm_graph_matmul_q8_0_decode_profiled_tensor(g->kv_raw,
                                                              model,
                                                              l->attn_kv_a_mqa->abs_offset,
                                                              DS4_N_EMBD,
                                                              kv_raw_dim,
                                                              g->attn_norm,
                                                              il,
                                                              pos,
                                                              "attn_kv_a_norm",
                                                              g->ssd_streaming) != 0;
            if (ok) {
                ok = ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(g->q_rank_norm,
                                                           g->q_rank,
                                                           model->map,
                                                           model->size,
                                                           l->attn_q_a_norm->abs_offset,
                                                           DS4_N_LORA_Q,
                                                           g->kv_norm,
                                                           g->kv_raw,
                                                           l->attn_kv_a_norm->abs_offset,
                                                           DS4_N_KV_LORA,
                                                           1,
                                                           DS4_RMS_EPS) != 0;
            }
        } else if (ok) {
            ok = ds4_gpu_rms_norm_weight_tensor(g->q_rank_norm,
                                                g->q_rank,
                                                model->map,
                                                model->size,
                                                l->attn_q_a_norm->abs_offset,
                                                DS4_N_LORA_Q,
                                                DS4_RMS_EPS) != 0;
        }
        if (ok) ok = glm_graph_matmul_q8_0_decode_profiled_tensor(g->q,
                                                                  model,
                                                                  l->attn_q_b->abs_offset,
                                                                  DS4_N_LORA_Q,
                                                                  g->q_dim,
                                                                  g->q_rank_norm,
                                                                  il,
                                                                  pos,
                                                                  "attn_q_b",
                                                                  g->ssd_streaming) != 0;
        if (ok) ok = ds4_gpu_glm_rope_tail_tensor(g->q,
                                                  1,
                                                  DS4_N_HEAD,
                                                  DS4_N_KEY_MLA,
                                                  DS4_N_ROT,
                                                  pos,
                                                  0,
                                                  rope_base,
                                                  rope_scale,
                                                  0.0f,
                                                  1.0f,
                                                  DS4_ROPE_YARN_BETA_FAST,
                                                  DS4_ROPE_YARN_BETA_SLOW) != 0;
        DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "q_path");
        if (ok && g->compact_cache_cap != 0 && glm_graph_layer_uses_full_indexer(il)) {
            ok = glm_graph_matmul_q8_0_decode_profiled_tensor(g->indexer_k,
                                                              model,
                                                              l->indexer_attn_k->abs_offset,
                                                              DS4_N_EMBD,
                                                              DS4_N_INDEXER_HEAD_DIM,
                                                              g->cur,
                                                              il,
                                                              pos,
                                                              "indexer_k",
                                                              g->ssd_streaming) != 0;
            DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "indexer_k_proj");
            if (ok) {
                ok = ds4_gpu_glm_store_indexer_k_tensor(
                        g->layer_indexer_key_cache[il],
                        g->indexer_k,
                        model->map,
                        model->size,
                        l->indexer_k_norm->abs_offset,
                        l->indexer_k_norm_b->abs_offset,
                        pos,
                        1,
                        g->compact_cache_cap,
                        DS4_N_INDEXER_HEAD_DIM,
                        DS4_N_ROT,
                        0,
                        1.0e-6f,
                        rope_base,
                        rope_scale,
                        0.0f,
                        1.0f,
                        DS4_ROPE_YARN_BETA_FAST,
                        DS4_ROPE_YARN_BETA_SLOW,
                        glm_graph_compact_cache_is_f16()) != 0;
            }
            DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "indexer_k_store");
        }
        DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "indexer_k");
        if (ok && !fuse_qkv_norm && !fuse_qkv_norm_store) {
            ok = glm_graph_matmul_q8_0_decode_profiled_tensor(g->kv_raw,
                                                              model,
                                                              l->attn_kv_a_mqa->abs_offset,
                                                              DS4_N_EMBD,
                                                              kv_raw_dim,
                                                              g->attn_norm,
                                                              il,
                                                              pos,
                                                              "attn_kv_a",
                                                              g->ssd_streaming) != 0;
            if (ok) ok = ds4_gpu_glm_kv_lora_rms_norm_tensor(g->kv_norm,
                                                             g->kv_raw,
                                                             model->map,
                                                             model->size,
                                                             l->attn_kv_a_norm->abs_offset,
                                                             1,
                                                             kv_raw_dim,
                                                             DS4_N_KV_LORA,
                                                             DS4_RMS_EPS) != 0;
        }
        if (ok && g->compact_cache_cap != 0 && !fuse_qkv_norm_store) {
            ok = ds4_gpu_glm_store_compact_kv_tensor(g->layer_kv_lora_cache[il],
                                                     g->layer_k_rope_cache[il],
                                                     g->kv_norm,
                                                     g->kv_raw,
                                                     pos,
                                                     1,
                                                     g->compact_cache_cap,
                                                     kv_raw_dim,
                                                     DS4_N_KV_LORA,
                                                     DS4_N_ROT,
                                                     glm_graph_compact_cache_is_f16()) != 0;
        }
        if (use_indexed_attention) {
            if (ok && glm_graph_layer_uses_full_indexer(il)) {
                const uint32_t visible = pos + 1u;
                if (ok && visible <= indexer_top_k) {
                    ok = ds4_gpu_glm_fill_selected_range_tensor(g->indexer_selected,
                                                                visible) != 0;
                    DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "indexer_fill");
                    last_indexer_selected_count = visible;
                } else if (ok) {
                    ok = glm_graph_matmul_q8_0_decode_profiled_tensor(g->indexer_q,
                                                                      model,
                                                                      l->indexer_attn_q_b->abs_offset,
                                                                      DS4_N_LORA_Q,
                                                                      (uint64_t)DS4_N_INDEXER_HEAD * DS4_N_INDEXER_HEAD_DIM,
                                                                      g->q_rank_norm,
                                                                      il,
                                                                      pos,
                                                                      "indexer_q",
                                                                      g->ssd_streaming) != 0;
                    if (ok) ok = ds4_gpu_glm_indexer_rope_tail_tensor(g->indexer_q,
                                                                      1,
                                                                      DS4_N_INDEXER_HEAD,
                                                                      DS4_N_INDEXER_HEAD_DIM,
                                                                      DS4_N_ROT,
                                                                      pos,
                                                                      0,
                                                                      rope_base,
                                                                      rope_scale,
                                                                      0.0f,
                                                                      1.0f,
                                                                      DS4_ROPE_YARN_BETA_FAST,
                                                                      DS4_ROPE_YARN_BETA_SLOW) != 0;
                    if (ok && glm_graph_indexer_qat()) {
                        ok = ds4_gpu_dsv4_indexer_qat_tensor(g->indexer_q,
                                                             DS4_N_INDEXER_HEAD,
                                                             DS4_N_INDEXER_HEAD_DIM) != 0;
                    }
                    DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "indexer_q");
                    if (ok) ok = metal_graph_matmul_plain_tensor(g->indexer_weights,
                                                                 model,
                                                                 l->indexer_proj,
                                                                 DS4_N_EMBD,
                                                                 DS4_N_INDEXER_HEAD,
                                                                 g->cur,
                                                                 1);
                    DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "indexer_weights");
                    const float indexer_scale =
                        1.0f / sqrtf((float)(DS4_N_INDEXER_HEAD_DIM * DS4_N_INDEXER_HEAD));
                    ok = ds4_gpu_glm_indexer_score_one_tensor(g->indexer_scores,
                                                              g->indexer_q,
                                                              g->indexer_weights,
                                                              g->layer_indexer_key_cache[il],
                                                              visible,
                                                              DS4_N_INDEXER_HEAD,
                                                              DS4_N_INDEXER_HEAD_DIM,
                                                              indexer_scale,
                                                              glm_graph_compact_cache_is_f16()) != 0;
                    DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "indexer_scores");
                    if (ok) ok = ds4_gpu_indexer_topk_tensor(g->indexer_selected,
                                                             g->indexer_scores,
                                                             visible,
                                                             1,
                                                             indexer_top_k) != 0;
                    DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "indexer_topk");
                    last_indexer_selected_count = indexer_top_k;
                }
                if (ok) last_indexer_selected = g->indexer_selected;
            } else if (ok && (!last_indexer_selected || last_indexer_selected_count == 0)) {
                ok = false;
            }
            DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "indexer_select");
            if (ok) ok = ds4_gpu_glm_qk_lowrank_q8_0_tensor(g->qk_low,
                                                            g->q,
                                                            model->map,
                                                            model->size,
                                                            l->attn_k_b->abs_offset,
                                                            DS4_N_HEAD,
                                                            DS4_N_KV_LORA,
                                                            (uint32_t)g->q_nope,
                                                            DS4_N_KEY_MLA) != 0;
            DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "kv_path");
            if (ok && glm_graph_indexed_decode_split_group8_available(last_indexer_selected_count)) {
                const uint32_t split_block_rows =
                    glm_graph_indexed_decode_split_block_rows_for(last_indexer_selected_count);
                const uint32_t split_blocks =
                    (last_indexer_selected_count + split_block_rows - 1u) / split_block_rows;
                ok = ds4_gpu_glm_attention_indexed_decode_split_group8_tensor(g->heads,
                                                                              g->attn_partial_lora,
                                                                              g->attn_partial_ms,
                                                                              g->q,
                                                                              g->qk_low,
                                                                              g->layer_kv_lora_cache[il],
                                                                              g->layer_k_rope_cache[il],
                                                                              model->map,
                                                                              model->size,
                                                                              l->attn_v_b->abs_offset,
                                                                              last_indexer_selected,
                                                                              last_indexer_selected_count,
                                                                              true,
                                                                              g->compact_cache_cap,
                                                                              glm_graph_compact_cache_is_f16(),
                                                                              DS4_N_HEAD,
                                                                              DS4_N_KV_LORA,
                                                                              (uint32_t)g->q_nope,
                                                                              DS4_N_ROT,
                                                                              DS4_N_VALUE_MLA,
                                                                              0,
                                                                              split_block_rows,
                                                                              split_blocks,
                                                                              rope_base,
                                                                              rope_scale,
                                                                              0.0f,
                                                                              1.0f,
                                                                              DS4_ROPE_YARN_BETA_FAST,
                                                                              DS4_ROPE_YARN_BETA_SLOW) != 0;
            } else if (ok) {
                ok = ds4_gpu_glm_attention_indexed_decode_tensor(g->heads,
                                                                  g->q,
                                                                  g->qk_low,
                                                                  g->layer_kv_lora_cache[il],
                                                                  g->layer_k_rope_cache[il],
                                                                  model->map,
                                                                  model->size,
                                                                  l->attn_v_b->abs_offset,
                                                                  last_indexer_selected,
                                                                  last_indexer_selected_count,
                                                                  g->compact_cache_cap,
                                                                  glm_graph_compact_cache_is_f16(),
                                                                  DS4_N_HEAD,
                                                                  DS4_N_KV_LORA,
                                                                  (uint32_t)g->q_nope,
                                                                  DS4_N_ROT,
                                                                  DS4_N_VALUE_MLA,
                                                                  0,
                                                                  rope_base,
                                                                  rope_scale,
                                                                  0.0f,
                                                                  1.0f,
                                                                  DS4_ROPE_YARN_BETA_FAST,
                                                                  DS4_ROPE_YARN_BETA_SLOW) != 0;
            }
        } else {
            if (ok) ok = ds4_gpu_glm_k_b_project_tensor(g->k_nope,
                                                        g->kv_norm,
                                                        model->map,
                                                        model->size,
                                                        l->attn_k_b->abs_offset,
                                                        1,
                                                        DS4_N_KV_LORA,
                                                        (uint32_t)g->q_nope,
                                                        DS4_N_HEAD) != 0;
            if (ok) ok = glm_graph_matmul_q8_0_decode_profiled_tensor(g->value,
                                                                      model,
                                                                      l->attn_v_b->abs_offset,
                                                                      DS4_N_KV_LORA,
                                                                      g->heads_dim,
                                                                      g->kv_norm,
                                                                      il,
                                                                      pos,
                                                                      "attn_v_b",
                                                                      g->ssd_streaming) != 0;
            DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "kv_path");
            if (ok) ok = ds4_gpu_glm_build_kv_cache_tensor(g->layer_key_cache[il],
                                                           g->layer_value_cache[il],
                                                           g->kv_raw,
                                                           g->k_nope,
                                                           g->value,
                                                           pos,
                                                           1,
                                                           g->ctx_cap,
                                                           DS4_N_HEAD,
                                                           kv_raw_dim,
                                                           DS4_N_KV_LORA,
                                                           (uint32_t)g->q_nope,
                                                           DS4_N_ROT,
                                                           DS4_N_VALUE_MLA,
                                                           0,
                                                           rope_base,
                                                           rope_scale,
                                                           0.0f,
                                                           1.0f,
                                                           DS4_ROPE_YARN_BETA_FAST,
                                                           DS4_ROPE_YARN_BETA_SLOW,
                                                           true) != 0;
            DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "kv_cache");
            if (ok) ok = ds4_gpu_glm_attention_full_tensor(g->heads,
                                                           g->q,
                                                           g->layer_key_cache[il],
                                                           g->layer_value_cache[il],
                                                           pos,
                                                           1,
                                                           pos + 1u,
                                                           g->ctx_cap,
                                                           DS4_N_HEAD,
                                                           DS4_N_KEY_MLA,
                                                           DS4_N_VALUE_MLA,
                                                           true) != 0;
        }
        DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "attention");
        if (ok) ok = glm_graph_matmul_q8_0_decode_profiled_tensor(g->attn_out,
                                                                  model,
                                                                  l->attn_output->abs_offset,
                                                                  g->heads_dim,
                                                                  DS4_N_EMBD,
                                                                  g->heads,
                                                                  il,
                                                                  pos,
                                                                  "attn_o",
                                                                  g->ssd_streaming) != 0;
        DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_attn", "attn_output");
        if (ok) ok = ds4_gpu_add_rms_norm_weight_tensor(g->ffn_norm,
                                                        g->after_attn,
                                                        g->cur,
                                                        g->attn_out,
                                                        model->map,
                                                        model->size,
                                                        l->ffn_norm->abs_offset,
                                                        DS4_N_EMBD,
                                                        DS4_RMS_EPS) != 0;
        DS4_GLM_PROFILE_DECODE_STAGE("glm_decode_ffn", "ffn_norm");
        if (ok) ok = glm_graph_encode_ffn_one_normed_from(g,
                                                          model,
                                                          l,
                                                          il,
                                                          pos,
                                                          g->ffn_norm,
                                                          g->after_attn,
                                                          g->next,
                                                          g->ffn_gate,
                                                          g->ffn_up,
                                                          g->ffn_mid,
                                                          g->ffn_out,
                                                          g->ffn_sum,
                                                          g->attn_out,
                                                          decode_stage_profile,
                                                          decode_stage_profile ? &decode_stage_t0 : NULL);
        if (ok) {
            ds4_gpu_tensor *tmp = g->cur;
            g->cur = g->next;
            g->next = tmp;
        }
        if (ok &&
            !g->ssd_streaming &&
            decode_layer_flush_interval != 0 &&
            il < g->layer_end &&
            (slice_layer_done % decode_layer_flush_interval) == 0) {
            if (decode_flush_profile) {
                ok = ds4_gpu_flush_commands() != 0;
                if (ok) ok = ds4_gpu_synchronize() != 0;
                if (ok) {
                    const double now = now_sec();
                    fprintf(stderr,
                            "ds4: GLM decode layer flush pos=%u layers=%u..%u %.3f ms\n",
                            pos,
                            decode_flush_layer0,
                            il,
                            (now - decode_flush_stage_t0) * 1000.0);
                    decode_flush_layer0 = il + 1u;
                    decode_flush_stage_t0 = now;
                    ok = ds4_gpu_begin_commands() != 0;
                }
            } else {
                ok = ds4_gpu_flush_commands() != 0;
            }
        }
        if (g->ssd_streaming && !static_decode_map) {
            if (ok) ok = ds4_gpu_end_commands() != 0;
            else (void)ds4_gpu_synchronize();
        }
    }
#undef DS4_GLM_PROFILE_DECODE_STAGE
    if (ok && merge_indexed_output) {
        if (g->ssd_streaming) {
            if (!static_decode_map) {
                ok = glm_graph_stream_map_output(g, model, weights);
            }
            if (ok) ok = glm_graph_begin_commands_if_needed();
        }
        ok = glm_graph_encode_output_head(g, model, weights);
        if (g->ssd_streaming) {
            if (ok) ok = glm_graph_end_commands_if_active();
            else (void)ds4_gpu_synchronize();
        }
    }
    if (!g->ssd_streaming) {
        if (ok) ok = ds4_gpu_end_commands() != 0;
        else (void)ds4_gpu_synchronize();
    } else if (!ok) {
        (void)ds4_gpu_synchronize();
    }
    if (decode_output_profile) {
        const double now = now_sec();
        fprintf(stderr,
                "ds4: GLM decode output profile pos=%u layers=%.3f ms\n",
                pos,
                (now - decode_output_stage_t0) * 1000.0);
        decode_output_stage_t0 = now;
    }
    if (ok && output_hc) {
        ok = ds4_gpu_tensor_read(g->cur,
                                 0,
                                 output_hc,
                                 (uint64_t)DS4_N_EMBD * sizeof(float)) != 0;
    }
    if (ok && logits_out) {
        if (use_indexed_attention) {
            if (!merge_indexed_output) {
                if (g->ssd_streaming && !static_decode_map) {
                    ok = glm_graph_stream_map_output(g, model, weights);
                }
                if (ok) ok = glm_graph_begin_commands_if_needed();
                if (ok) ok = glm_graph_encode_output_head(g, model, weights);
                if (ok) ok = glm_graph_end_commands_if_active();
                else (void)ds4_gpu_synchronize();
                if (decode_output_profile) {
                    const double now = now_sec();
                    fprintf(stderr,
                            "ds4: GLM decode output profile pos=%u output_head=%.3f ms\n",
                            pos,
                            (now - decode_output_stage_t0) * 1000.0);
                    decode_output_stage_t0 = now;
                }
            }
            if (ok) {
                ok = ds4_gpu_tensor_read(g->logits,
                                         0,
                                         logits_out,
                                         (uint64_t)DS4_N_VOCAB * sizeof(float)) != 0;
                if (decode_output_profile) {
                    const double now = now_sec();
                    fprintf(stderr,
                            "ds4: GLM decode output profile pos=%u logits_read=%.3f ms\n",
                            pos,
                            (now - decode_output_stage_t0) * 1000.0);
                }
            }
        } else {
            if (g->ssd_streaming && !static_decode_map) {
                ok = glm_graph_stream_map_output(g, model, weights);
            }
            if (ok && ds4_gpu_commands_active()) {
                ok = ds4_gpu_end_commands() != 0;
            }
            if (ok) ok = glm_graph_forward_output_head(g, model, weights, g->cur, logits_out);
            if (decode_output_profile) {
                const double now = now_sec();
                fprintf(stderr,
                        "ds4: GLM decode output profile pos=%u fallback_output=%.3f ms\n",
                        pos,
                        (now - decode_output_stage_t0) * 1000.0);
            }
        }
    }
    if (ok && !logits_out && g->ssd_streaming) {
        ok = glm_graph_end_commands_if_active();
    } else if (!ok) {
        (void)ds4_gpu_synchronize();
    }
    return ok;
}

static int generate_glm_metal_argmax(
        const ds4_model   * model,
        const ds4_vocab   * vocab,
        const ds4_weights * weights,
        const token_vec   * prompt,
        int                 n_predict,
        int                 ctx_size,
        bool                quality,
        bool                ssd_streaming,
        bool                ssd_streaming_cold,
        uint32_t            ssd_streaming_preload_experts,
        const ds4_metal_ssd_profile *profile,
        ds4_token_emit_fn   emit,
        ds4_generation_done_fn done,
        void              * emit_ud,
        ds4_session_progress_fn progress,
        void              * progress_ud) {
    fprintf(stderr, "ds4: using GLM Metal full-attention argmax generation path\n");

    if (!prompt || prompt->len <= 0 || prompt->len > ctx_size) {
        fprintf(stderr, "ds4: prompt is empty or exceeds context size\n");
        return 1;
    }
    if (n_predict <= 0) {
        if (done) done(emit_ud);
        return 0;
    }

    ds4_glm_gpu_graph g;
    if (!glm_graph_alloc(&g,
                         model,
                         weights,
                         ctx_size,
                         ssd_streaming,
                         ssd_streaming_cold)) {
        fprintf(stderr, "ds4: failed to allocate GLM Metal graph runtime\n");
        return 1;
    }
    g.quality = quality;
    if (profile) {
        g.glm_indexed_prefill_prepare =
            profile->glm_indexed_prefill_prepare;
        g.glm_router_ahead = profile->glm_router_ahead;
        g.glm_router_ahead_lookahead =
            profile->glm_router_ahead_lookahead;
    }
    if ((uint32_t)prompt->len >= g.ctx_size) {
        fprintf(stderr,
                "ds4: prompt length %d leaves no GLM Metal context room (ctx %u)\n",
                prompt->len,
                g.ctx_size);
        glm_graph_free(&g);
        return 1;
    }
    const bool memory_report = getenv("DS4_METAL_MEMORY_REPORT") != NULL;
    if (memory_report) ds4_gpu_print_memory_report("after GLM graph alloc");

    float *logits = xmalloc((size_t)DS4_N_VOCAB * sizeof(logits[0]));
    bool ok = true;
    const char *hotlist_path = getenv("DS4_METAL_STREAMING_EXPERT_HOTLIST");
    const bool seed_before_prefill =
        ssd_streaming &&
        getenv("DS4_METAL_GLM_DISABLE_STREAMING_SEED_BEFORE_PREFILL") == NULL &&
        (ssd_streaming_preload_experts != 0 ||
         (hotlist_path && hotlist_path[0]));
    const double t_prefill0 = now_sec();
    if (seed_before_prefill) {
        ds4_gpu_graph seed_graph;
        memset(&seed_graph, 0, sizeof(seed_graph));
        seed_graph.quality = quality;
        seed_graph.ssd_streaming = ssd_streaming;
        seed_graph.ssd_streaming_cold = ssd_streaming_cold;
        seed_graph.streaming_preload_experts = ssd_streaming_preload_experts;
        ok = metal_graph_seed_streaming_expert_cache_from_hotlist(&seed_graph,
                                                                  model,
                                                                  weights);
    }
    if (ok) {
        ok = glm_graph_prefill_range(&g,
                                     model,
                                     weights,
                                     prompt->v,
                                     0,
                                     (uint32_t)prompt->len,
                                     logits,
                                     progress,
                                     progress_ud,
                                     (uint32_t)prompt->len);
    }
    const double t_prefill1 = now_sec();
    if (memory_report) ds4_gpu_print_memory_report("after GLM prefill");
    if (!ok) {
        fprintf(stderr, "ds4: GLM Metal prefill failed\n");
        free(logits);
        glm_graph_free(&g);
        return 1;
    }

    int n_generated = 0;
    int n_decode_eval = 0;
    uint32_t pos = (uint32_t)prompt->len;
    const bool token_timing = getenv("DS4_TOKEN_TIMING") != NULL;
    const double t_decode0 = now_sec();
    for (int i = 0; i < n_predict && pos < g.ctx_size; i++) {
        if (getenv("DS4_TRACE_TOP") != NULL) {
            char label[64];
            snprintf(label, sizeof(label), "GLM step %d", i);
            print_top_logits(stderr, label, vocab, logits, DS4_N_VOCAB, 10);
        }
        const int token = sample_argmax(logits, DS4_N_VOCAB);
        if (vocab_token_is_generation_stop(vocab, token)) break;
        if (emit) emit(emit_ud, token);
        n_generated++;

        if (i == n_predict - 1 || pos + 1u >= g.ctx_size) {
            pos++;
            break;
        }

        const double t_eval0 = token_timing ? now_sec() : 0.0;
        ok = glm_graph_forward_token(&g, model, weights, token, NULL, pos, NULL, logits);
        if (!ok) {
            fprintf(stderr, "ds4: GLM Metal decode failed at position %u\n", pos);
            free(logits);
            glm_graph_free(&g);
            return 1;
        }
        if (token_timing) {
            const double t_eval1 = now_sec();
            fprintf(stderr,
                    "ds4: GLM decode eval %d took %.3f ms\n",
                    n_decode_eval + 1,
                    (t_eval1 - t_eval0) * 1000.0);
        }
        n_decode_eval++;
        pos++;
    }
    const double t_decode1 = now_sec();
    if (done) done(emit_ud);

    const double prefill_s = t_prefill1 - t_prefill0;
    const double decode_s = t_decode1 - t_decode0;
    ds4_log(stderr,
            DS4_LOG_TIMING,
            "ds4: GLM prefill: %.2f t/s, generation: %.2f t/s\n",
            prefill_s > 0.0 ? (double)prompt->len / prefill_s : 0.0,
            decode_s > 0.0 ? (double)n_generated / decode_s : 0.0);

    if (memory_report) ds4_gpu_print_memory_report("before GLM graph free");
    free(logits);
    glm_graph_free(&g);
    return 0;
}
