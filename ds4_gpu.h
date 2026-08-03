#ifndef DS4_GPU_H
#define DS4_GPU_H

#include <stdbool.h>
#include <stdint.h>

#include "ds4_ssd.h"

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 * GPU Tensor and Command Lifetime.
 * =========================================================================
 *
 * Opaque device tensor used by the DS4-specific GPU executor.
 *
 * The public GPU API is tensor-resident: activations, KV state, and scratch
 * buffers stay device-owned across the whole prefill/decode command sequence.
 */
typedef struct ds4_gpu_tensor ds4_gpu_tensor;

int ds4_gpu_init(void);
void ds4_gpu_cleanup(void);

ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes);
ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed(uint64_t bytes);
ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base, uint64_t offset, uint64_t bytes);
void ds4_gpu_tensor_free(ds4_gpu_tensor *tensor);
uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *tensor);
void *ds4_gpu_tensor_contents(ds4_gpu_tensor *tensor);
int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *tensor, float value, uint64_t count);
int ds4_gpu_tensor_write(ds4_gpu_tensor *tensor, uint64_t offset, const void *data, uint64_t bytes);
int ds4_gpu_tensor_read(const ds4_gpu_tensor *tensor, uint64_t offset, void *data, uint64_t bytes);
int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t dst_offset,
                          const ds4_gpu_tensor *src, uint64_t src_offset,
                          uint64_t bytes);
/* Validate both exact byte ranges before encoding either copy, then record
 * both operations in one blit encoder on the active command batch. */
int ds4_gpu_tensor_copy_pair(
        ds4_gpu_tensor       *dst0,
        uint64_t              dst_offset0,
        const ds4_gpu_tensor *src0,
        uint64_t              src_offset0,
        uint64_t              bytes0,
        ds4_gpu_tensor       *dst1,
        uint64_t              dst_offset1,
        const ds4_gpu_tensor *src1,
        uint64_t              src_offset1,
        uint64_t              bytes1);
int ds4_gpu_tensor_copy_f32_to_f16(ds4_gpu_tensor *dst, uint64_t dst_offset,
                                   const ds4_gpu_tensor *src, uint64_t src_offset,
                                   uint64_t count);

int ds4_gpu_begin_commands(void);
int ds4_gpu_flush_encoder(void);
int ds4_gpu_flush_commands(void);
int ds4_gpu_commands_active(void);
int ds4_gpu_signal_selected_readback_ready(uint64_t *event_value);
int ds4_gpu_commit_and_wait_selected_readback(uint64_t event_value, const char *label);
int ds4_gpu_wait_selected_readback_ready(uint64_t event_value, const char *label);
int ds4_gpu_end_commands(void);
int ds4_gpu_synchronize(void);

int ds4_gpu_set_model_map(const void *model_map, uint64_t model_size);
int ds4_gpu_set_model_fd(int fd);
int ds4_gpu_set_model_fd_for_map(int fd, const void *model_map);
typedef struct ds4_gpu_expert_store_layer_v2 {
    uint32_t layer;
    uint32_t reserved;
    uint64_t data_offset;
    uint64_t data_size;
    uint64_t record_bytes;
    uint64_t component_offset[3];
    uint64_t component_bytes[3];
} ds4_gpu_expert_store_layer_v2;
/* Expert stores are separate identity domains. TARGET remains zero so every
 * zero-initialized legacy table keeps the historical target-only behavior. */
typedef enum ds4_gpu_expert_store_id {
    DS4_GPU_EXPERT_STORE_TARGET = 0,
    DS4_GPU_EXPERT_STORE_SUPPORT = 1,
    DS4_GPU_EXPERT_STORE_COUNT = 2,
} ds4_gpu_expert_store_id;
/* Install a validated expert-major store embedded in the model GGUF. layer is
 * the real model-layer id, so inventories may omit a dense prefix. */
int ds4_gpu_expert_store_v2_install(
        int                                  fd,
        uint64_t                             file_size,
        uint32_t                             n_layer,
        uint32_t                             n_expert,
        uint32_t                             storage_format,
        uint32_t                             group_size,
        const ds4_gpu_expert_store_layer_v2 *layers);
int ds4_gpu_expert_store_v2_bind_layer(
        uint32_t layer,
        uint64_t model_size,
        uint64_t gate_offset,
        uint64_t up_offset,
        uint64_t down_offset);
/* Return the physical expert-major payload for one embedded v2 layer.  SSD
 * mapping code uses this to replace the three virtual canonical expert spans
 * with the one record-interleaved span consumed by routed ID kernels. */
int ds4_gpu_expert_store_v2_layer_span(
        uint32_t layer,
        uint64_t model_size,
        uint64_t *offset,
        uint64_t *size);
int ds4_gpu_expert_store_v2_enable_resident(void);
void ds4_gpu_expert_store_v2_clear(void);
/* Store-aware variants form the DSpark seam. The legacy entry points above
 * are exact TARGET wrappers. */
int ds4_gpu_expert_store_v2_install_for_store(
        ds4_gpu_expert_store_id                store_id,
        int                                    fd,
        uint64_t                               file_size,
        uint32_t                               n_layer,
        uint32_t                               n_expert,
        uint32_t                               storage_format,
        uint32_t                               group_size,
        const ds4_gpu_expert_store_layer_v2   *layers);
int ds4_gpu_expert_store_v2_bind_layer_for_store(
        ds4_gpu_expert_store_id store_id,
        uint32_t                layer,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset);
int ds4_gpu_expert_store_v2_layer_span_for_store(
        ds4_gpu_expert_store_id store_id,
        uint32_t                layer,
        uint64_t                model_size,
        uint64_t               *offset,
        uint64_t               *size);
void ds4_gpu_expert_store_v2_clear_for_store(
        ds4_gpu_expert_store_id store_id);
int ds4_gpu_set_model_map_range(const void *model_map, uint64_t model_size, uint64_t map_offset, uint64_t map_size, uint64_t max_tensor_bytes);
int ds4_gpu_set_model_map_spans(const void *model_map, uint64_t model_size, const uint64_t *offsets, const uint64_t *sizes, uint32_t count, uint64_t max_tensor_bytes);
int ds4_gpu_pro_q4_expert_table_auto_available(void);
int ds4_gpu_preload_q4_expert_tables(const void *model_map, uint64_t model_size,
                                     uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
                                     uint64_t gate_expert_bytes, uint64_t down_expert_bytes,
                                     uint32_t n_total_expert);
int ds4_gpu_should_use_managed_kv_cache(uint64_t kv_cache_bytes, uint64_t context_bytes);
void ds4_gpu_set_quality(bool quality);

/* Test hook: force the pre-M5 exact Qwen router kernel on the current device.
 * Production code never enables this override. */
void ds4_gpu_internal_force_qwen35_exact_router_for_test(bool enabled);
void ds4_gpu_set_glm_model(bool enabled);
void ds4_gpu_set_ssd_streaming(bool enabled);
void ds4_gpu_set_streaming_expert_readahead(bool enabled);
void ds4_gpu_set_glm_streaming_prefill_full_layer(bool enabled);
void ds4_gpu_set_streaming_expert_cache_budget(uint32_t experts);
/* Increase the live budget without discarding resident expert slots. */
int ds4_gpu_grow_streaming_expert_cache_budget(uint32_t experts);
/* Change the SSD cache phase. Growth preserves resident entries; shrinkage
 * releases resident entries and their backing slabs after synchronizing all
 * users. The model-lifetime required floor is never relaxed. */
int ds4_gpu_reconfigure_streaming_expert_cache_budget(uint32_t experts);
/* Optional model-lifetime fail-closed floor; zero disables the contract. */
void ds4_gpu_set_streaming_expert_cache_required_floor(uint32_t experts);
void ds4_gpu_set_streaming_expert_cache_expert_bytes(uint64_t bytes);
/* Optional model-specific slab-growth target; zero restores the backend
 * default. An explicit DS4_METAL_STREAMING_EXPERT_SLAB_MB wins only outside
 * guarded model tiers whose measured slab geometry is part of admission. */
void ds4_gpu_set_streaming_expert_cache_slab_target_bytes(uint64_t bytes);
/* Guarded streaming tiers admit every additional TARGET slab against a fresh
 * host snapshot. runtime_bytes must include the maximum prefill/decode
 * envelope plus any separately owned cache such as DSpark SUPPORT;
 * static_page_bytes is the complete pageable non-routed coverage. */
void ds4_gpu_set_streaming_expert_cache_growth_guard(
        bool     enabled,
        uint64_t runtime_bytes,
        uint64_t static_page_bytes);
uint64_t ds4_gpu_recommended_working_set_size(void);
int ds4_gpu_host_memory_snapshot(ds4_ssd_host_memory *out);
/* Effective combined cache budget after TARGET caps and an optional DSpark
 * SUPPORT split. Equal to configured_count() for target-only models. */
uint32_t ds4_gpu_stream_expert_cache_effective_parent_count(void);
uint32_t ds4_gpu_stream_expert_cache_configured_count(void);
uint32_t ds4_gpu_stream_expert_cache_current_count(void);
/* Complete gate + up + down payload for one logical routed expert.  This is
 * exposed separately from the cumulative counters so benchmark tooling can
 * report storage read amplification without changing the versioned stats ABI. */
uint64_t ds4_gpu_stream_expert_cache_logical_expert_bytes(void);
/* Benchmark-only window for unique (layer, expert) storage demand.  begin()
 * clears a small backend bitmap; end() returns its cardinality and logical
 * payload bytes, then disables the extra accounting.  Normal inference never
 * enables this branch. */
void ds4_gpu_stream_expert_io_measurement_begin(void);
void ds4_gpu_stream_expert_io_measurement_end(uint64_t *unique_experts,
                                              uint64_t *unique_bytes);

/* Versioned cumulative snapshot for campaign attribution. Integer counters
 * saturate at UINT64_MAX rather than wrapping. expert_loads counts complete
 * logical expert records (gate + up + down). pread_syscalls counts actual
 * pread(2) calls that returned at least one byte; EINTR/error/EOF returns do
 * not count, while multiple successful partial reads count separately.
 * pread_wall_ms is parallel-read wall time, not the sum of worker times. */
#define DS4_GPU_STREAM_EXPERT_CACHE_STATS_VERSION_V1 1u
typedef struct ds4_gpu_stream_expert_cache_stats_v1 {
    uint32_t version;
    uint32_t struct_size;
    uint64_t hits;
    uint64_t misses;
    uint64_t evictions;
    uint64_t expert_loads;
    uint64_t pread_syscalls;
    uint64_t pread_bytes;
    double   pread_wall_ms;
    double   split_resident_wait_ms;
} ds4_gpu_stream_expert_cache_stats_v1;

/* Returns zero only for a NULL output. Non-streaming backends return a valid,
 * versioned all-zero payload so callers never need backend-specific branches. */
int ds4_gpu_stream_expert_cache_snapshot_v1(
        ds4_gpu_stream_expert_cache_stats_v1 *out);

/* Legacy subset retained for source and ABI compatibility. Any out pointer may
 * be NULL. Values have the same cumulative lifetime as snapshot_v1(). */
void ds4_gpu_stream_expert_cache_stats(uint64_t *hits, uint64_t *misses,
                                       uint64_t *pread_bytes, double *pread_ms,
                                       double *split_resident_wait_ms);

typedef struct ds4_gpu_stream_expert_table {
    const void *model_map;
    uint64_t    model_size;
    uint32_t    layer;
    uint32_t    n_total_expert;
    uint64_t    gate_offset;
    uint64_t    up_offset;
    uint64_t    down_offset;
    uint64_t    gate_expert_bytes;
    uint64_t    down_expert_bytes;
} ds4_gpu_stream_expert_table;
/* Reset only the prompt-local eviction heuristic.  The resident SSD expert
 * cache itself is intentionally kept warm across sessions. */
void ds4_gpu_stream_expert_cache_reset_route_hotness(void);
void ds4_gpu_stream_expert_cache_release_resident(void);
/* A layer lease pins only experts that are loaded while the lease is active;
 * acquiring one never preloads the layer. The opaque non-zero id is bound to
 * the acquiring thread and must be released by that same owner. */
int ds4_gpu_stream_expert_cache_acquire_layer_lease(uint32_t layer,
                                                    uint64_t *lease_id);
int ds4_gpu_stream_expert_cache_release_layer_lease(uint64_t lease_id);
uint32_t ds4_gpu_stream_expert_cache_budget_for_expert_size(
        uint64_t gate_expert_bytes,
        uint64_t down_expert_bytes);
int ds4_gpu_stream_expert_cache_seed_selected(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_selected);
int ds4_gpu_stream_expert_cache_begin_selected_load(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *selected_ids,
        uint32_t                           n_selected);

/* SSD routed MoE can expose a completed GPU route before encoding the shared
 * expert. The backend then owns one bounded pread generation while the encoder
 * submits shared-expert work. A zero generation means the capability was
 * unavailable and the caller must use the exact synchronous routed-MoE
 * fallback. */
typedef struct ds4_gpu_stream_io_ticket {
    uint64_t generation;
    uint32_t unique_experts;
    uint32_t missing_experts;
    uint32_t max_inflight_reads;
    uint32_t asynchronous;
} ds4_gpu_stream_io_ticket;

/* Family-neutral API used by both Qwen top-8 Q4 and DeepSeek top-6 IQ2/Q2.
 * Types are passed explicitly so capability checks cannot accidentally
 * advertise a selected-address kernel for the wrong model geometry. */
int ds4_gpu_stream_io_overlap_capable(
        uint32_t n_tokens,
        uint32_t n_total_expert,
        uint32_t n_selected,
        uint32_t gate_type,
        uint32_t down_type);
/* True only when the ordinary SSD selected-address batch path is eligible
 * under the current runtime policy.  Auto scheduling uses this probe so
 * supported fallback/debug knobs keep selecting their historical path;
 * explicit schedule requests remain fail-closed at the caller. */
int ds4_gpu_stream_selected_addr_capable(
        uint32_t n_tokens,
        uint32_t n_total_expert,
        uint32_t n_selected,
        uint32_t gate_type,
        uint32_t down_type);
int ds4_gpu_stream_batch_route_ready_select(
        const ds4_gpu_stream_expert_table *table,
        const ds4_gpu_tensor              *selected,
        uint32_t                           n_tokens,
        uint32_t                           n_selected,
        uint32_t                           gate_type,
        uint32_t                           down_type,
        int                                request_expert_group,
        ds4_gpu_stream_io_ticket          *ticket);
int ds4_gpu_stream_batch_finish(uint64_t generation);
int ds4_gpu_stream_batch_abort(uint64_t generation);

/* Compatibility names retained for the integrated Qwen exact-stack API. */
typedef ds4_gpu_stream_io_ticket ds4_gpu_qwen35_stream_io_ticket;

int ds4_gpu_qwen35_stream_io_overlap_capable(
        uint32_t gate_type,
        uint32_t down_type);
int ds4_gpu_qwen35_stream_batch_route_ready(
        const ds4_gpu_stream_expert_table *table,
        const ds4_gpu_tensor              *selected,
        uint32_t                           n_tokens,
        uint32_t                           n_selected,
        uint32_t                           gate_type,
        uint32_t                           down_type,
        ds4_gpu_qwen35_stream_io_ticket   *ticket);
/* Select variant used by the integrated Qwen scheduler.  When expert grouping
 * is requested, the backend builds the stable expert-major permutation from
 * the ID array already made visible for I/O.  The legacy entry point above is
 * exactly equivalent to passing request_expert_group=0. */
int ds4_gpu_qwen35_stream_batch_route_ready_select(
        const ds4_gpu_stream_expert_table *table,
        const ds4_gpu_tensor              *selected,
        uint32_t                           n_tokens,
        uint32_t                           n_selected,
        uint32_t                           gate_type,
        uint32_t                           down_type,
        int                                request_expert_group,
        ds4_gpu_qwen35_stream_io_ticket   *ticket);
/* Called immediately after shared-expert encoding, finish() flushes that
 * command buffer without waiting, then waits for pread and installs completed
 * staging buffers on the encoder owner. The following routed-MoE call consumes
 * the generation. abort() waits for workers but never installs staging. */
int ds4_gpu_qwen35_stream_batch_finish(uint64_t generation);
int ds4_gpu_qwen35_stream_batch_abort(uint64_t generation);

/* Router-ahead prefetch: advisory OS readahead for the experts a predictor
 * expects the NEXT layer to select.  Never touches cache state; safe to call
 * from a worker thread while the decode thread streams the current layer. */
int ds4_gpu_glm_stream_expert_prefetch_hint(
        uint32_t        layer,
        const int32_t  *expert_ids,
        uint32_t        n_experts,
        const void     *model_map,
        uint64_t        model_size,
        uint64_t        gate_offset,
        uint64_t        up_offset,
        uint64_t        down_offset,
        uint64_t        gate_expert_bytes,
        uint64_t        down_expert_bytes);
int ds4_gpu_stream_expert_cache_seed_experts(
        const ds4_gpu_stream_expert_table *table,
        const int32_t                     *expert_ids,
        const uint32_t                    *expert_priorities,
        uint32_t                           n_experts);
void ds4_gpu_print_memory_report(const char *label);

/* =========================================================================
 * Embeddings and Indexer Helpers.
 * =========================================================================
 *
 * These kernels seed HC state from token embeddings and implement the ratio-4
 * compressed-attention indexer that chooses visible compressed rows.
 */

int ds4_gpu_embed_token_hc_tensor(
        ds4_gpu_tensor *out_hc,
        const void       *model_map,
        uint64_t          model_size,
        uint64_t          weight_offset,
        uint32_t          n_vocab,
        uint32_t          token,
        uint32_t          n_embd,
        uint32_t          n_hc);

int ds4_gpu_embed_tokens_hc_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *tokens,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_embed_token_q8_0_tensor(
        ds4_gpu_tensor *out,
        const void       *model_map,
        uint64_t          model_size,
        uint64_t          weight_offset,
        uint32_t          n_vocab,
        uint32_t          token,
        uint32_t          n_embd);

int ds4_gpu_embed_tokens_q8_0_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *tokens,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n_vocab,
        uint32_t                n_tokens,
        uint32_t                n_embd);

int ds4_gpu_indexer_score_one_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_head,
        uint32_t                head_dim,
        float                   scale);

int ds4_gpu_indexer_scores_prefill_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale);

int ds4_gpu_indexer_scores_decode_batch_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale);

int ds4_gpu_indexer_topk_tensor(
        ds4_gpu_tensor       *selected,
        const ds4_gpu_tensor *scores,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k);

/* GPU argmax over n_vocab F32 logits. Writes the winning index as int32 at
 * out_idx[0]. Tie-break: lower index wins (matches host sample_argmax). */
int ds4_gpu_argmax_tensor(
        ds4_gpu_tensor       *out_idx,
        const ds4_gpu_tensor *logits,
        uint32_t                n_vocab);

int ds4_gpu_dsv4_topk_mask_tensor(
        ds4_gpu_tensor       *mask,
        const ds4_gpu_tensor *topk,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k);

/* =========================================================================
 * Dense Projections, Norms, RoPE, and KV Rounding.
 * =========================================================================
 *
 * The graph uses these primitives for Q/KV projections, HC/output projections,
 * attention output projections, and DS4's tail-only RoPE.
 */

int ds4_gpu_matmul_q8_0_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_ggml_k_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                weight_bytes,
        uint32_t                weight_type,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_q8_0_decode_mpp_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_q8_0_decode_mpp_model_view_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_q8_0_rows_scalar_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

/* Optional fused GPU operations.
 *
 * These are acceleration hooks, not required backend primitives.  A backend
 * that does not provide the fused kernel must still define the symbol and
 * return 0.  Callers then use the portable sequence of required primitives.
 * Backends that return nonzero from a fused half-output operation must also
 * implement the matching half-input HC expansion helpers below.
 */
int ds4_gpu_is_m5_family(void);
int ds4_gpu_matmul_q8_0_pair_tensor(
        ds4_gpu_tensor       *out0,
        ds4_gpu_tensor       *out1,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight0_offset,
        uint64_t                weight1_offset,
        uint64_t                in_dim,
        uint64_t                out0_dim,
        uint64_t                out1_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_q8_0_f16_out_tensor(
        ds4_gpu_tensor       *out_h,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp);

int ds4_gpu_shared_mid_swiglu_q8_0_tensor(
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp);

int ds4_gpu_shared_gate_up_swiglu_q8_0_model_view_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp);

int ds4_gpu_shared_gate_up_swiglu_q8_0_rows_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok,
        float                   clamp);

int ds4_gpu_shared_gate_up_swiglu_q8_0_rows_scalar_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok,
        float                   clamp);

int ds4_gpu_matmul_f16_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_f16_pair_tensor(
        ds4_gpu_tensor       *out_a,
        ds4_gpu_tensor       *out_b,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_a_offset,
        uint64_t                weight_b_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_matmul_f32_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok);

int ds4_gpu_repeat_hc_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *row,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_rms_norm_plain_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        uint32_t                n,
        float                   eps);

int ds4_gpu_rms_norm_plain_rows_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        uint32_t                n,
        uint32_t                rows,
        float                   eps);

int ds4_gpu_rms_norm_weight_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        float                   eps);

int ds4_gpu_rms_norm_weight_rows_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *x,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        uint32_t                rows,
        float                   eps);

int ds4_gpu_add_rms_norm_weight_tensor(
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *sum_out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                n,
        float                   eps);

int ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
        ds4_gpu_tensor       *q_out,
        const ds4_gpu_tensor *q,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                q_weight_offset,
        uint32_t                q_n,
        ds4_gpu_tensor       *kv_out,
        const ds4_gpu_tensor *kv,
        uint64_t                kv_weight_offset,
        uint32_t                kv_n,
        uint32_t                rows,
        float                   eps);

int ds4_gpu_head_rms_norm_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        float             eps);

int ds4_gpu_head_rms_norm_rope_tail_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        uint32_t          n_rot,
        uint32_t          pos0,
        uint32_t          n_ctx_orig,
        bool              inverse,
        float             freq_base,
        float             freq_scale,
        float             ext_factor,
        float             attn_factor,
        float             beta_fast,
        float             beta_slow,
        float             eps);

int ds4_gpu_attn_q_b_f16_head_rms_rope_tail_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *q_half,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        uint32_t              n_tok,
        uint32_t              n_head,
        uint32_t              head_dim,
        uint32_t              n_rot,
        uint32_t              pos0,
        uint32_t              n_ctx_orig,
        bool                  inverse,
        float                 freq_base,
        float                 freq_scale,
        float                 ext_factor,
        float                 attn_factor,
        float                 beta_fast,
        float                 beta_slow,
        float                 eps);

int ds4_gpu_dsv4_fp8_kv_quantize_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          head_dim,
        uint32_t          n_rot);

/* Round F32 lanes to BF16 with round-to-nearest-even, retaining F32 storage.
 * Infinities remain bit-exact; NaNs retain sign/high payload and are quieted. */
int ds4_gpu_bf16_round_f32_tensor(ds4_gpu_tensor *x, uint64_t count);

int ds4_gpu_dsv4_indexer_qat_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_rows,
        uint32_t          head_dim);

int ds4_gpu_rope_tail_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_tok,
        uint32_t          n_head,
        uint32_t          head_dim,
        uint32_t          n_rot,
        uint32_t          pos0,
        uint32_t          n_ctx_orig,
        bool              inverse,
        float             freq_base,
        float             freq_scale,
        float             ext_factor,
        float             attn_factor,
        float             beta_fast,
        float             beta_slow);

/* Qwen3.6 resident primitives.  Decode uses the one-token entry points while
 * layer-major prefill uses the batch/sequence forms below.  Recurrent and
 * full-attention state remain resident in the model's F32 correctness layout.
 * Tensor arguments must not overlap unless the function exposes a single
 * in-place tensor, as RoPE does. */
int ds4_gpu_qwen35_split_q_gate_tensor(
        ds4_gpu_tensor       *query,
        ds4_gpu_tensor       *gate,
        const ds4_gpu_tensor *projection,
        uint32_t              n_query_head,
        uint32_t              head_dim);

int ds4_gpu_qwen35_split_q_gate_batch_tensor(
        ds4_gpu_tensor       *query,
        ds4_gpu_tensor       *gate,
        const ds4_gpu_tensor *projection,
        uint32_t              n_token,
        uint32_t              n_query_head,
        uint32_t              head_dim);

int ds4_gpu_qwen35_split_q_gate_rms_norm_batch_tensor(
        ds4_gpu_tensor       *query,
        ds4_gpu_tensor       *gate,
        const ds4_gpu_tensor *projection,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              norm_weight_offset,
        uint32_t              n_token,
        uint32_t              n_query_head,
        uint32_t              head_dim,
        float                 eps);

int ds4_gpu_qwen35_sigmoid_mul_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *input,
        const ds4_gpu_tensor *gate,
        uint32_t              n_value,
        bool                  broadcast_gate);

/* Row-wise scalar gate used by batched shared experts.  input/output are
 * [n_row][row_width], while gate contains one scalar per row. */
int ds4_gpu_qwen35_sigmoid_mul_rows_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *input,
        const ds4_gpu_tensor *gate,
        uint32_t              n_row,
        uint32_t              row_width);

int ds4_gpu_qwen35_rope_prefix_tensor(
        ds4_gpu_tensor *values,
        uint32_t        n_head,
        uint32_t        head_dim,
        uint32_t        n_rot,
        uint32_t        position,
        float           theta);

/* positions is a packed uint32 tensor with one absolute position per token. */
int ds4_gpu_qwen35_rope_prefix_batch_tensor(
        ds4_gpu_tensor       *values,
        const ds4_gpu_tensor *positions,
        uint32_t              n_token,
        uint32_t              n_head,
        uint32_t              head_dim,
        uint32_t              n_rot,
        float                 theta);

int ds4_gpu_qwen35_causal_conv_step_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *state,
        const ds4_gpu_tensor *input,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_channel,
        uint32_t              kernel_size);

/* Advances a complete token chunk in order while keeping each channel's
 * short convolution history in registers. */
int ds4_gpu_qwen35_causal_conv_sequence_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *state,
        const ds4_gpu_tensor *input,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_token,
        uint32_t              n_channel,
        uint32_t              kernel_size);

int ds4_gpu_qwen35_gated_delta_step_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *state,
        const ds4_gpu_tensor *query,
        const ds4_gpu_tensor *key,
        const ds4_gpu_tensor *value,
        const ds4_gpu_tensor *log_decay,
        const ds4_gpu_tensor *beta,
        uint32_t              n_key_head,
        uint32_t              n_value_head,
        uint32_t              key_dim,
        uint32_t              value_dim);

/* Qwen's fixed [Q,K,V]=[16x128,16x128,32x128] projection layout.  The
 * recurrent state is advanced token-serially inside each row-parallel Metal
 * threadgroup and committed once at the end of the chunk. */
int ds4_gpu_qwen35_gated_delta_sequence_128_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *state,
        const ds4_gpu_tensor *projection,
        const ds4_gpu_tensor *log_decay,
        const ds4_gpu_tensor *beta,
        uint32_t              n_token,
        uint32_t              n_key_head,
        uint32_t              n_value_head);

int ds4_gpu_qwen35_rmsnorm_gated_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *input,
        const ds4_gpu_tensor *gate,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint32_t              n_vector,
        uint32_t              dim,
        float                 epsilon);

int ds4_gpu_qwen35_dequant_embedding_q5_k_tensor(
        ds4_gpu_tensor *out,
        const void     *model_map,
        uint64_t        model_size,
        uint64_t        embedding_offset,
        uint32_t        row_index,
        uint32_t        n_embd);

int ds4_gpu_qwen35_dequant_embedding_q8_0_tensor(
        ds4_gpu_tensor *out,
        const void     *model_map,
        uint64_t        model_size,
        uint64_t        embedding_offset,
        uint32_t        row_index,
        uint32_t        n_embd);

int ds4_gpu_qwen35_dequant_embedding_q5_k_batch_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *token_ids,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              embedding_offset,
        uint32_t              n_token,
        uint32_t              n_row,
        uint32_t              n_embd);

int ds4_gpu_qwen35_dequant_embedding_q8_0_batch_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *token_ids,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              embedding_offset,
        uint32_t              n_token,
        uint32_t              n_row,
        uint32_t              n_embd);

int ds4_gpu_qwen35_gated_delta_controls_tensor(
        ds4_gpu_tensor       *log_decay,
        ds4_gpu_tensor       *beta,
        const ds4_gpu_tensor *alpha_logit,
        const ds4_gpu_tensor *beta_logit,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              ssm_a_offset,
        uint64_t              dt_bias_offset,
        uint32_t              n_value_head);

int ds4_gpu_qwen35_gated_delta_controls_batch_tensor(
        ds4_gpu_tensor       *log_decay,
        ds4_gpu_tensor       *beta,
        const ds4_gpu_tensor *alpha_logit,
        const ds4_gpu_tensor *beta_logit,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              ssm_a_offset,
        uint64_t              dt_bias_offset,
        uint32_t              n_token,
        uint32_t              n_value_head);

int ds4_gpu_qwen35_gqa_decode_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *query,
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        uint32_t              n_kv,
        uint32_t              n_query_head,
        uint32_t              n_kv_head,
        uint32_t              head_dim);

int ds4_gpu_qwen35_gqa_prefill_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *query,
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        uint32_t              position0,
        uint32_t              n_token,
        uint32_t              n_query_head,
        uint32_t              n_kv_head,
        uint32_t              head_dim);

/* Stable telemetry IDs returned through reuse_used.  Path 0 is the serial
 * legacy oracle, path 1 is the exact F32 K/V-reuse fallback, and path 2 is the
 * direct-F32 FlashAttention prefill specialization.  request_reuse remains a
 * boolean policy input: zero forces path 0, while a nonzero value permits the
 * selector to try path 2 and then path 1 when Flash is unavailable before any
 * GPU work is submitted. */
enum {
    DS4_GPU_QWEN35_GQA_PATH_LEGACY = 0,
    DS4_GPU_QWEN35_GQA_PATH_EXACT_REUSE = 1,
    DS4_GPU_QWEN35_GQA_PATH_FLASH_F32 = 2,
};

/* K/V-tile reuse specialization for Qwen's 8:1 grouped-query geometry.  The
 * capability is deliberately separate from policy: an engine resolves it
 * once, combines it with its per-request feature mask, then passes that result
 * as request_reuse.  The select wrappers preserve the existing kernel for a
 * disabled feature, incompatible geometry, short context, or a device whose
 * pipeline cannot host the specialization.  reuse_used reports one of the
 * DS4_GPU_QWEN35_GQA_PATH_* values above and is suitable for strict campaign
 * telemetry.
 *
 * A selected reuse dispatch is never retried through the legacy kernel after
 * a GPU error: doing so could conceal a partial write and invalidate exact-run
 * accounting. */
int ds4_gpu_qwen35_gqa_reuse_capable(
        uint32_t n_query_head,
        uint32_t n_kv_head,
        uint32_t head_dim);

int ds4_gpu_qwen35_gqa_decode_select_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *query,
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        uint32_t              n_kv,
        uint32_t              n_query_head,
        uint32_t              n_kv_head,
        uint32_t              head_dim,
        int                   request_reuse,
        int                  *reuse_used);

int ds4_gpu_qwen35_gqa_prefill_select_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *query,
        const ds4_gpu_tensor *key_cache,
        const ds4_gpu_tensor *value_cache,
        uint32_t              position0,
        uint32_t              n_token,
        uint32_t              n_query_head,
        uint32_t              n_kv_head,
        uint32_t              head_dim,
        int                   request_reuse,
        int                  *reuse_used);

/* Fixed 256-expert Qwen router: stable full softmax, deterministic top-8
 * selection (lower expert ID wins ties), then top-8 renormalization.
 * Non-finite diagnostic input writes eight {-1, 0} ID/weight pairs while a
 * successful GPU dispatch still returns nonzero. */
int ds4_gpu_qwen35_router_softmax_top8_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *selected_weight,
        const ds4_gpu_tensor *logits);

int ds4_gpu_qwen35_router_softmax_top8_batch_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *selected_weight,
        const ds4_gpu_tensor *logits,
        uint32_t              n_token);

/* Release decode fused KV finalizer: after the standalone RoPE kernel, this
 * performs DS4's FP8 non-RoPE KV round trip and writes the F16-rounded raw
 * attention cache row in one dispatch. */
int ds4_gpu_kv_fp8_store_raw_tensor(
        ds4_gpu_tensor *kv,
        ds4_gpu_tensor *raw_cache,
        uint32_t          raw_cap,
        uint32_t          row,
        uint32_t          head_dim,
        uint32_t          n_rot);

/* Reference/raw-cache primitive kept for prefill and diagnostics.  Decode uses
 * ds4_gpu_kv_fp8_store_raw_tensor unless a diagnostic reference path is
 * explicitly selected by the graph driver. */
int ds4_gpu_store_raw_kv_tensor(
        ds4_gpu_tensor       *raw_cache,
        const ds4_gpu_tensor *kv,
        uint32_t                raw_cap,
        uint32_t                row,
        uint32_t                head_dim);

int ds4_gpu_store_raw_kv_batch_tensor(
        ds4_gpu_tensor       *raw_cache,
        const ds4_gpu_tensor *kv,
        uint32_t                raw_cap,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                head_dim);

/* =========================================================================
 * KV Compression and Attention.
 * =========================================================================
 *
 * Compressed layers maintain rolling score/KV state and append pooled rows at
 * ratio boundaries.  Attention kernels consume raw SWA rows, compressed rows,
 * and optional indexer masks.
 */

int ds4_gpu_compressor_update_tensor(
        const ds4_gpu_tensor *kv_cur,
        const ds4_gpu_tensor *sc_cur,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        ds4_gpu_tensor       *comp_cache,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos,
        uint32_t                comp_row,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps);

int ds4_gpu_compressor_store_batch_tensor(
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens);

int ds4_gpu_compressor_prefill_tensor(
        ds4_gpu_tensor       *comp_cache,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps);

int ds4_gpu_compressor_prefill_ratio4_replay_tensor(
        ds4_gpu_tensor       *comp_cache,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps);

int ds4_gpu_compressor_prefill_state_ratio4_tensor(
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv_tail,
        const ds4_gpu_tensor *sc_tail,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                pos0);

int ds4_gpu_attention_decode_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                n_comp,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_mask,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_gpu_attention_prefill_raw_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_gpu_attention_decode_raw_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                window,
        uint32_t                n_head,
        uint32_t                head_dim);

/* Final-0731 DSpark attention: five query rows attend non-causally to the
 * committed ring in pinned physical-cache order plus exactly five transient
 * draft rows. q, committed_kv, and transient_draft_kv use F32 storage but
 * must already contain BF16-rounded values reopened as F32; this primitive
 * does not round its inputs. The ring already contains current main_kv. */
int ds4_gpu_dspark_attention_two_source_f32_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *committed_kv,
        const ds4_gpu_tensor *transient_draft_kv,
        uint32_t                committed_count,
        uint32_t                committed_cap,
        uint32_t                committed_start,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_gpu_attention_decode_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_comp_mask,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *topk,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        uint32_t                n_comp,
        uint32_t                top_k,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_gpu_attention_prefill_static_mixed_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_gpu_attention_prefill_masked_mixed_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim);

int ds4_gpu_attention_output_q8_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *low,
        ds4_gpu_tensor       *group_tmp,
        ds4_gpu_tensor       *low_tmp,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens);

int ds4_gpu_attention_output_q8_batch_f16_tensor(
        ds4_gpu_tensor       *out_h,
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens);

int ds4_gpu_attention_output_low_q8_tensor(
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        const ds4_gpu_tensor *heads);

/* =========================================================================
 * Router, Shared Expert, and Routed MoE.
 * =========================================================================
 *
 * These kernels implement the FFN body: router probabilities/top-k or hash
 * routing, shared SwiGLU, and the IQ2_XXS/Q2_K/Q4_K routed experts.
 */

int ds4_gpu_swiglu_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *gate,
        const ds4_gpu_tensor *up,
        uint32_t                n,
        float                   clamp,
        float                   weight);

int ds4_gpu_add_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        uint32_t                n);

int ds4_gpu_add3_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *a,
        const ds4_gpu_tensor *b,
        const ds4_gpu_tensor *c,
        uint32_t                n);

int ds4_gpu_directional_steering_project_tensor(
        ds4_gpu_tensor       *x,
        const ds4_gpu_tensor *directions,
        uint32_t                layer,
        uint32_t                width,
        uint32_t                rows,
        float                   scale);

int ds4_gpu_router_select_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        uint64_t                hash_offset,
        uint32_t                hash_rows,
        uint32_t                token,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_expert_groups,
        uint32_t                n_group_used,
        bool                    has_bias,
        bool                    hash_mode,
        const ds4_gpu_tensor *logits);

int ds4_gpu_router_select_batch_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        uint64_t                hash_offset,
        uint32_t                hash_rows,
        uint32_t                n_expert_groups,
        uint32_t                n_group_used,
        bool                    has_bias,
        bool                    hash_mode,
        const ds4_gpu_tensor *logits,
        const ds4_gpu_tensor *tokens,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_tokens);

int ds4_gpu_glm_router_select_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        const ds4_gpu_tensor *logits,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale);

int ds4_gpu_glm_router_select_batch_tensor(
        ds4_gpu_tensor       *selected,
        ds4_gpu_tensor       *weights,
        ds4_gpu_tensor       *probs,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                bias_offset,
        const ds4_gpu_tensor *logits,
        uint32_t                n_expert,
        uint32_t                n_expert_used,
        float                   expert_weight_scale,
        uint32_t                n_tokens);

int ds4_gpu_glm_routed_moe_one_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                up_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                up_expert_bytes,
        uint64_t                up_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        uint32_t                layer_index,
        const ds4_gpu_tensor *x,
        bool                    force_resident);

int ds4_gpu_glm_routed_moe_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                up_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                up_expert_bytes,
        uint64_t                up_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        uint32_t                layer_index,
        const ds4_gpu_tensor *x,
        uint32_t                n_tokens,
        uint32_t                mid_token_stride);

int ds4_gpu_glm_routed_moe_batch_direct_scalar_q4_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                up_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                up_expert_bytes,
        uint64_t                up_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        uint32_t                layer_index,
        const ds4_gpu_tensor *x,
        uint32_t                n_tokens,
        uint32_t                mid_token_stride);

int ds4_gpu_routed_moe_set_selected_override(const int32_t *selected, uint32_t n_selected);

int ds4_gpu_routed_moe_one_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *experts,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   clamp,
        const ds4_gpu_tensor *x,
        uint32_t                layer_index,
        bool                    force_resident);

/* Qwen3.6 routes one token to eight experts. Resident Metal consumes a trusted
 * GPU route as one top-8 pass and reduces all expert outputs in one dispatch.
 * SSD streaming keeps the complete route pinned as one cache unit, executes
 * two ordered top-4 selected-slot passes, and adds their partial outputs. The
 * caller owns the persistent offset-0/16 half views, partial outputs, and an
 * eight-expert activation scratch shared by both modes. trusted_gpu_route may
 * only be true when the IDs were produced by the top-8 router in the current
 * command batch; trace/replay observability deliberately retains readback. */
int ds4_gpu_qwen35_routed_moe_top8_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *partial0,
        ds4_gpu_tensor       *partial1,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *experts,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected_top8,
        const ds4_gpu_tensor *weights_top8,
        const ds4_gpu_tensor *selected_half0,
        const ds4_gpu_tensor *selected_half1,
        const ds4_gpu_tensor *weights_half0,
        const ds4_gpu_tensor *weights_half1,
        uint32_t                n_total_expert,
        float                   clamp,
        const ds4_gpu_tensor *x,
        uint32_t                layer_index,
        bool                    trusted_gpu_route);

/* Test/profiling counters for the resident Qwen route boundary. */
void ds4_gpu_internal_qwen35_resident_route_stats_reset(void);
void ds4_gpu_internal_qwen35_resident_route_stats_add(uint64_t calls);
uint64_t ds4_gpu_internal_qwen35_resident_gpu_route_calls(void);
uint64_t ds4_gpu_internal_qwen35_resident_host_readbacks(void);
void ds4_gpu_internal_qwen35_gdn128_stats_reset(void);
uint64_t ds4_gpu_internal_qwen35_gdn128_parallel_calls(void);

int ds4_gpu_routed_moe_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *experts,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   clamp,
        const ds4_gpu_tensor *x,
        uint32_t                layer_index,
        uint32_t                n_tokens,
        bool                   *mid_is_f16);

/* Family-neutral scheduling request bits.  GROUP asks for the stable
 * expert-major permutation; ROUTE_TILE additionally asks a compatible backend
 * to batch consecutive routes for the same expert.  Callers may combine bits;
 * schedule_used returns the exact subset actually encoded, so a stricter
 * caller can require `(used & request) == request`. */
enum {
    DS4_GPU_EXPERT_SCHEDULE_GROUP = 1 << 0,
    DS4_GPU_EXPERT_SCHEDULE_ROUTE_TILE = 1 << 1,
};

int ds4_gpu_routed_moe_batch_select_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *experts,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   clamp,
        const ds4_gpu_tensor *x,
        uint32_t                layer_index,
        uint32_t                n_tokens,
        bool                   *mid_is_f16,
        int                     schedule_request,
        int                    *schedule_used);

int ds4_gpu_qwen35_routed_moe_batch_select_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *experts,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   clamp,
        const ds4_gpu_tensor *x,
        uint32_t                layer_index,
        uint32_t                n_tokens,
        bool                   *mid_is_f16,
        int                     request_expert_group,
        int                    *expert_group_used);

/* Model-free ABI/permutation regression used by --metal-kernels. */
int ds4_gpu_internal_qwen35_expert_group_test(void);
/* Model-free ownership regression for overlap router-ID reuse. */
int hebrus_gpu_internal_qwen35_stream_selected_ids_ownership_test(void);
/* Model-free invariant regression for bounded/indexed expert-cache scans and
 * the Qwen layer-staleness eviction tie-break. */
int hebrus_gpu_internal_stream_expert_cache_scan_limit_test(void);
/* Model-free source-translation/fail-closed regression used by
 * --metal-expert-pack. */
int ds4_gpu_internal_qwen35_expert_pack_test(void);
/* Canonical-vs-embedded GLM Q2 regression for direct and grouped execution. */
int ds4_gpu_internal_expert_store_v2_kernel_test(void);
#ifdef DS4_TEST_HOOKS
/* Candidate final-0731 Q-B primitive.  Keep the disconnected wrapper and API
 * hook-private until the stage graph provides its first production caller. */
int ds4_gpu_dspark_q_head_norm_bf16_tensor(
        ds4_gpu_tensor *x,
        uint32_t          rows,
        float             eps);
/* Model-free rollback and authenticated lease-error unwind regressions. */
int ds4_gpu_internal_qwen35_stream_staging_rollback_test(void);
int ds4_gpu_internal_qwen35_lease_error_unwind_test(void);
/* TARGET/SUPPORT descriptor namespace and offset-isolation regression. */
int ds4_gpu_internal_dspark_dual_store_test(void);
/* Separate SUPPORT SSD cache quota, namespace, I/O and teardown regression. */
int ds4_gpu_internal_dspark_support_cache_test(void);
/* Boundary-correct compact-mid routed MoE through one SUPPORT transaction. */
int ds4_gpu_internal_dspark_support_routed_moe_test(void);
/* Device-only post-layer HC mean regression for the DSpark tap. */
int ds4_gpu_internal_dspark_hc_mean_test(void);
/* Device ring append/publication regression for DSpark capture history. */
int ds4_gpu_internal_dspark_history_test(void);
/* Physical Metal BF16 round/reopen bit-semantics and range regression. */
int ds4_gpu_internal_bf16_round_f32_test(void);
/* Final-0731 per-head Q normalization BF16-publication regression. */
int ds4_gpu_internal_dspark_q_head_norm_bf16_test(void);
/* Final-0731 DSpark two-source non-causal F32 attention regression. */
int ds4_gpu_internal_dspark_two_source_attention_test(void);
/* Payload-first final-0731 stage-zero 32/32/4 physical Metal white box. */
int ds4_gpu_internal_dspark_stage_zero_physical_test(void);
/* Fixed 5x256/top-6 DSpark router oracle; disconnected from production. */
int ds4_gpu_internal_dspark_router_f32_test(void);
#endif

/* =========================================================================
 * Hyper-Connection Kernels.
 * =========================================================================
 *
 * HC kernels reduce four residual streams before a sublayer and expand the
 * sublayer output back into four streams afterward.
 */

int ds4_gpu_hc_split_sinkhorn_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *mix,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps);

int ds4_gpu_hc_weighted_sum_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *weights,
        uint32_t                n_embd,
        uint32_t                n_hc);

/* DSpark capture primitive: average the HC lanes after a selected target
 * decoder layer without host readback or a learned HC-head transform. */
int ds4_gpu_hc_mean_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *residual_hc,
        uint32_t                n_embd,
        uint32_t                n_hc);

/* Encode one selected-stage DSpark history update. Geometry comes from the
 * graph's checked absolute-ring plan. Single-row decode writes directly to
 * current; multi-row prefill uses scratch for the retained HC means. */
int ds4_gpu_dspark_capture_history_tensor(
        ds4_gpu_tensor       *current,
        ds4_gpu_tensor       *history,
        ds4_gpu_tensor       *scratch,
        const ds4_gpu_tensor *post_layer_hc,
        uint32_t              active_tokens,
        uint32_t              input_skip,
        uint32_t              retained_rows,
        uint32_t              first_physical_row,
        uint32_t              first_rows,
        uint32_t              second_rows);

/* Publish one selected stage from verifier shadow storage. Only the accepted
 * prefix reaches history and only its final row reaches current. */
int ds4_gpu_dspark_publish_history_tensor(
        ds4_gpu_tensor       *current,
        ds4_gpu_tensor       *history,
        const ds4_gpu_tensor *candidate,
        uint32_t              committed_rows,
        uint32_t              first_physical_row,
        uint32_t              first_rows,
        uint32_t              second_rows);

int ds4_gpu_hc_weighted_sum_split_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

/* Release decode fused HC pre-sublayer operation: split the HC mixer and
 * immediately reduce four HC streams into the active 4096-wide sublayer row. */
int ds4_gpu_hc_split_weighted_sum_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps);

int ds4_gpu_hc_split_weighted_sum_norm_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *norm_out,
        ds4_gpu_tensor       *split,
        const ds4_gpu_tensor *mix,
        const ds4_gpu_tensor *residual_hc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint64_t                norm_weight_offset,
        uint32_t                n_embd,
        uint32_t                n_hc,
        uint32_t                sinkhorn_iters,
        float                   eps,
        float                   norm_eps);

int ds4_gpu_output_hc_weights_tensor(
        ds4_gpu_tensor       *out,
        const ds4_gpu_tensor *pre,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                scale_offset,
        uint64_t                base_offset,
        uint32_t                n_hc,
        float                   eps);

int ds4_gpu_hc_expand_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *post,
        const ds4_gpu_tensor *comb,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_hc_expand_split_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_hc_expand_split_half_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out_h,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_hc_expand_add_split_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_hc_expand_add_split_half_add_tensor(
        ds4_gpu_tensor       *out_hc,
        const ds4_gpu_tensor *block_out,
        const ds4_gpu_tensor *block_add_h,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_shared_down_hc_expand_q8_0_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *shared_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *shared_mid,
        const ds4_gpu_tensor *routed_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

int ds4_gpu_matmul_q8_0_hc_expand_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *block_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc);

/* GLM-5.2 DSA Metal primitives.  These stay model-specific so the existing
 * DeepSeek and Qwen execution contracts do not gain conditional semantics. */
int ds4_gpu_glm_stream_expert_cache_begin_selected_load_tensor(
        const ds4_gpu_stream_expert_table *table,
        const ds4_gpu_tensor              *selected,
        uint32_t                           n_selected);

int ds4_gpu_glm_rope_tail_tensor(
        ds4_gpu_tensor *x,
        uint32_t n_tokens, uint32_t n_head, uint32_t head_dim,
        uint32_t rot_dim, uint32_t pos0, uint32_t n_ctx_orig,
        float freq_base, float freq_scale, float ext_factor,
        float attn_factor, float beta_fast, float beta_slow);

int ds4_gpu_glm_kv_lora_rms_norm_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *kv_raw,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t n_tokens, uint32_t kv_raw_dim, uint32_t kv_lora_dim,
        float eps);

int ds4_gpu_glm_k_b_project_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *kv_norm,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t n_tokens, uint32_t kv_lora_dim, uint32_t qk_nope,
        uint32_t n_head);

int ds4_gpu_glm_store_compact_kv_tensor(
        ds4_gpu_tensor *kv_lora_cache, ds4_gpu_tensor *k_rope_cache,
        const ds4_gpu_tensor *kv_norm, const ds4_gpu_tensor *kv_raw,
        uint32_t pos0, uint32_t n_tokens, uint32_t cache_cap,
        uint32_t kv_raw_dim, uint32_t kv_lora_dim, uint32_t qk_rope,
        bool cache_f16);

int ds4_gpu_glm_qkv_norm_store_compact_kv_tensor(
        ds4_gpu_tensor *q_out, const ds4_gpu_tensor *q,
        const void *model_map, uint64_t model_size, uint64_t q_weight_offset,
        uint32_t q_n, ds4_gpu_tensor *kv_lora_cache,
        ds4_gpu_tensor *k_rope_cache, const ds4_gpu_tensor *kv_raw,
        uint64_t kv_weight_offset, uint32_t pos0, uint32_t n_tokens,
        uint32_t cache_cap, uint32_t kv_raw_dim, uint32_t kv_lora_dim,
        uint32_t qk_rope, bool cache_f16, float eps);

int ds4_gpu_glm_store_indexer_k_tensor(
        ds4_gpu_tensor *indexer_key_cache, const ds4_gpu_tensor *raw_k,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint64_t bias_offset, uint32_t pos0, uint32_t n_tokens,
        uint32_t cache_cap, uint32_t head_dim, uint32_t rot_dim,
        uint32_t n_ctx_orig, float eps, float freq_base, float freq_scale,
        float ext_factor, float attn_factor, float beta_fast, float beta_slow,
        bool cache_f16);

int ds4_gpu_glm_build_kv_cache_tensor(
        ds4_gpu_tensor *key_cache, ds4_gpu_tensor *value_cache,
        const ds4_gpu_tensor *kv_raw, const ds4_gpu_tensor *k_nope,
        const ds4_gpu_tensor *value, uint32_t pos0, uint32_t n_tokens,
        uint32_t cache_cap, uint32_t n_head, uint32_t kv_raw_dim,
        uint32_t kv_lora_dim, uint32_t qk_nope, uint32_t qk_rope,
        uint32_t value_dim, uint32_t n_ctx_orig, float freq_base,
        float freq_scale, float ext_factor, float attn_factor,
        float beta_fast, float beta_slow, bool cache_f16);

int ds4_gpu_glm_build_kv_cache_flash_tensor(
        ds4_gpu_tensor *key_cache, ds4_gpu_tensor *value_cache,
        const ds4_gpu_tensor *kv_raw, const ds4_gpu_tensor *k_nope,
        const ds4_gpu_tensor *value, uint32_t pos0, uint32_t n_tokens,
        uint32_t cache_cap, uint32_t n_head, uint32_t kv_raw_dim,
        uint32_t kv_lora_dim, uint32_t qk_nope, uint32_t qk_rope,
        uint32_t value_dim, uint32_t n_ctx_orig, float freq_base,
        float freq_scale, float ext_factor, float attn_factor,
        float beta_fast, float beta_slow, bool cache_f16);

int ds4_gpu_glm_attention_full_tensor(
        ds4_gpu_tensor *heads, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *key_cache, const ds4_gpu_tensor *value_cache,
        uint32_t pos0, uint32_t n_tokens, uint32_t cache_len,
        uint32_t cache_cap, uint32_t n_head, uint32_t qk_dim,
        uint32_t value_dim, bool cache_f16);

int ds4_gpu_glm_fill_selected_range_tensor(
        ds4_gpu_tensor *selected, uint32_t n_selected);
int ds4_gpu_glm_fill_selected_range_batch_tensor(
        ds4_gpu_tensor *selected, uint32_t n_tokens, uint32_t pos0,
        uint32_t n_selected, uint32_t pad_row);

int ds4_gpu_glm_indexer_rope_tail_tensor(
        ds4_gpu_tensor *x, uint32_t n_tokens, uint32_t n_head,
        uint32_t head_dim, uint32_t rot_dim, uint32_t pos0,
        uint32_t n_ctx_orig, float freq_base, float freq_scale,
        float ext_factor, float attn_factor, float beta_fast,
        float beta_slow);

int ds4_gpu_glm_indexer_score_one_tensor(
        ds4_gpu_tensor *scores, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *indexer_key_cache, uint32_t n_rows,
        uint32_t n_head, uint32_t head_dim, float scale, bool cache_f16);

int ds4_gpu_glm_indexer_scores_batch_tensor(
        ds4_gpu_tensor *scores, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *indexer_key_cache, uint32_t n_rows,
        uint32_t n_tokens, uint32_t pos0, uint32_t n_head,
        uint32_t head_dim, float scale, bool cache_f16);

int ds4_gpu_glm_qk_lowrank_q8_0_tensor(
        ds4_gpu_tensor *qk_low, const ds4_gpu_tensor *q,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t n_head, uint32_t kv_lora_dim, uint32_t qk_nope,
        uint32_t qk_dim);

int ds4_gpu_glm_qk_lowrank_q8_0_batch_tensor(
        ds4_gpu_tensor *qk_low, const ds4_gpu_tensor *q,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t n_tokens, uint32_t n_head, uint32_t kv_lora_dim,
        uint32_t qk_nope, uint32_t qk_dim);

int ds4_gpu_glm_value_project_q8_0_batch_heads_tensor(
        ds4_gpu_tensor *heads, const ds4_gpu_tensor *lora,
        const void *model_map, uint64_t model_size, uint64_t weight_offset,
        uint32_t n_tokens, uint32_t n_head, uint32_t kv_lora_dim,
        uint32_t value_dim);

int ds4_gpu_glm_attention_indexed_decode_tensor(
        ds4_gpu_tensor *heads, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache, const void *model_map,
        uint64_t model_size, uint64_t value_weight_offset,
        const ds4_gpu_tensor *selected, uint32_t n_selected,
        uint32_t cache_cap, bool cache_f16, uint32_t n_head,
        uint32_t kv_lora_dim, uint32_t qk_nope, uint32_t qk_rope,
        uint32_t value_dim, uint32_t n_ctx_orig, float freq_base,
        float freq_scale, float ext_factor, float attn_factor,
        float beta_fast, float beta_slow);

int ds4_gpu_glm_attention_indexed_decode_split_group8_tensor(
        ds4_gpu_tensor *heads, ds4_gpu_tensor *partial_lora,
        ds4_gpu_tensor *partial_ms, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache, const void *model_map,
        uint64_t model_size, uint64_t value_weight_offset,
        const ds4_gpu_tensor *selected, uint32_t n_selected,
        bool selected_rows_valid, uint32_t cache_cap, bool cache_f16,
        uint32_t n_head, uint32_t kv_lora_dim, uint32_t qk_nope,
        uint32_t qk_rope, uint32_t value_dim, uint32_t n_ctx_orig,
        uint32_t block_rows, uint32_t n_blocks, float freq_base,
        float freq_scale, float ext_factor, float attn_factor,
        float beta_fast, float beta_slow);

int ds4_gpu_glm_attention_indexed_batch_tensor(
        ds4_gpu_tensor *heads, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache, const void *model_map,
        uint64_t model_size, uint64_t value_weight_offset,
        const ds4_gpu_tensor *selected, uint32_t n_tokens,
        uint32_t n_selected, uint32_t cache_cap, bool cache_f16,
        uint32_t n_head, uint32_t kv_lora_dim, uint32_t qk_nope,
        uint32_t qk_rope, uint32_t value_dim, uint32_t n_ctx_orig,
        float freq_base, float freq_scale, float ext_factor,
        float attn_factor, float beta_fast, float beta_slow);

int ds4_gpu_glm_attention_indexed_batch_lora_causal_tensor(
        ds4_gpu_tensor *lora_out, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache, uint32_t n_tokens,
        uint32_t pos0, uint32_t n_selected, uint32_t cache_cap,
        bool cache_f16, uint32_t n_head, uint32_t kv_lora_dim,
        uint32_t qk_nope, uint32_t qk_rope, uint32_t n_ctx_orig,
        float freq_base, float freq_scale, float ext_factor,
        float attn_factor, float beta_fast, float beta_slow);

int ds4_gpu_glm_attention_indexed_batch_lora_valid_tensor(
        ds4_gpu_tensor *lora_out, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *qk_low,
        const ds4_gpu_tensor *kv_lora_cache,
        const ds4_gpu_tensor *k_rope_cache,
        const ds4_gpu_tensor *selected, uint32_t n_tokens,
        uint32_t n_selected, uint32_t cache_cap, bool cache_f16,
        uint32_t n_head, uint32_t kv_lora_dim, uint32_t qk_nope,
        uint32_t qk_rope, uint32_t n_ctx_orig, float freq_base,
        float freq_scale, float ext_factor, float attn_factor,
        float beta_fast, float beta_slow);

int ds4_gpu_glm_attention_flash_staged_tensor(
        ds4_gpu_tensor *heads, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *key_cache, const ds4_gpu_tensor *value_cache,
        uint32_t pos0, uint32_t n_tokens, uint32_t cache_len,
        uint32_t cache_cap, uint32_t n_head, uint32_t qk_dim,
        uint32_t value_dim, bool cache_f16);

int ds4_gpu_glm_attention_flash_tensor(
        ds4_gpu_tensor *heads, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *key_cache, const ds4_gpu_tensor *value_cache,
        uint32_t pos0, uint32_t n_tokens, uint32_t cache_len,
        uint32_t cache_cap, uint32_t n_head, uint32_t qk_dim,
        uint32_t value_dim, bool cache_f16);

#ifdef __cplusplus
}
#endif

#endif
