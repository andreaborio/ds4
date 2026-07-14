#ifndef DS4_QWEN_H
#define DS4_QWEN_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* The only Qwen shape accepted by ds4.  Keeping these values outside the
 * DeepSeek shape profile prevents Qwen state from inheriting HC/compressor
 * assumptions while giving the loader and CPU cache one shared contract. */
enum {
    QWEN35_N_LAYER                    = 40,
    QWEN35_N_TENSOR                   = 733,
    QWEN35_N_EMBD                     = 2048,
    QWEN35_N_VOCAB                    = 248320,
    QWEN35_N_MERGE                    = 247587,
    QWEN35_N_HEAD                     = 16,
    QWEN35_N_HEAD_KV                  = 2,
    QWEN35_N_HEAD_DIM                 = 256,
    QWEN35_N_ROT                      = 64,
    QWEN35_N_EXPERT                   = 256,
    QWEN35_N_EXPERT_USED              = 8,
    QWEN35_N_FF_EXP                   = 512,
    QWEN35_N_FF_SHARED                = 512,
    QWEN35_SSM_CONV_KERNEL            = 4,
    QWEN35_SSM_STATE                  = 128,
    QWEN35_SSM_GROUP                  = 16,
    QWEN35_SSM_VALUE_HEAD             = 32,
    QWEN35_SSM_DT_RANK                = 32,
    QWEN35_SSM_INNER                  = 4096,
    QWEN35_SSM_CONV_CHANNEL           = 8192,
    QWEN35_FULL_ATTENTION_INTERVAL    = 4,
    QWEN35_FULL_ATTENTION_LAYER_COUNT = 10,
    QWEN35_RECURRENT_LAYER_COUNT      = 30,
    QWEN35_CONTEXT_LENGTH             = 262144,
    QWEN35_BOS_PAD_ID                 = 248044,
    QWEN35_EOS_ID                     = 248046,
    QWEN35_MODEL_ID                   = 2,
};

typedef struct {
    uint64_t gdn_conv_bytes;
    uint64_t gdn_recurrent_bytes;
    uint64_t fixed_bytes;
    uint64_t kv_bytes_per_token;
    uint64_t max_kv_bytes;
    uint64_t max_total_bytes;
} ds4_qwen35_cpu_cache_plan;

typedef struct {
    /* Full-attention layers only: [context][kv_head][head_dim]. */
    float *key;
    float *value;

    /* Gated DeltaNet layers only.  Conv history is oldest to newest; the
     * recurrent layout is [value_head][value_dim][key_dim]. */
    float *conv;
    float *recurrent;
} ds4_qwen35_cpu_layer_state;

typedef struct {
    ds4_qwen35_cpu_layer_state layer[QWEN35_N_LAYER];
    ds4_qwen35_cpu_cache_plan plan;
    uint32_t ctx_capacity;
    uint32_t kv_capacity;
    uint32_t n_tokens;
} ds4_qwen35_cpu_cache;

bool ds4_qwen35_layer_is_full_attention(uint32_t layer);

bool ds4_qwen35_cpu_cache_plan_make(
        uint32_t                     ctx_size,
        ds4_qwen35_cpu_cache_plan   *plan);

/* The initial correctness path keeps both full-attention K/V and GDN state in
 * F32.  Init commits only the fixed GDN state; K/V grows explicitly through
 * reserve(), outside the allocation-free token forward.  The cache must be
 * zero/uninitialized; free a live cache before calling init again. */
bool ds4_qwen35_cpu_cache_init(
        ds4_qwen35_cpu_cache *cache,
        uint32_t              ctx_capacity);

/* Ensure K/V space for required_tokens without changing the live length.
 * Existing row pointers remain valid only until the next successful reserve. */
bool ds4_qwen35_cpu_cache_reserve(
        ds4_qwen35_cpu_cache *cache,
        uint32_t              required_tokens);

void ds4_qwen35_cpu_cache_reset(ds4_qwen35_cpu_cache *cache);
void ds4_qwen35_cpu_cache_free(ds4_qwen35_cpu_cache *cache);

/* Commit completely evaluated tokens.  This never allocates and fails if the
 * rows were not reserved or would exceed the configured context. */
bool ds4_qwen35_cpu_cache_advance(
        ds4_qwen35_cpu_cache *cache,
        uint32_t              n_tokens);

uint64_t ds4_qwen35_cpu_cache_allocated_bytes(
        const ds4_qwen35_cpu_cache *cache);

#endif
