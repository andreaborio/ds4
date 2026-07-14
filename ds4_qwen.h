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
    uint64_t float_bytes;
    uint64_t quant_bytes;
    uint64_t fixed_bytes;
    uint64_t score_bytes;
    uint64_t total_bytes;
} ds4_qwen35_cpu_scratch_plan;

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

/* One allocation owns every temporary used by the scalar CPU token forward.
 * Capacities follow the fixed Qwen shape: projection=8192, gate/query/value/
 * heads=4096, key=2048, routed_mid=8x512, and score=ctx_capacity.  The raw
 * Q8_K regions use ds4's private 292-byte block layout and are cast only in
 * ds4.c, keeping quant implementation details out of the operator API. */
typedef struct {
    void *arena;
    uint64_t arena_bytes;
    uint32_t ctx_capacity;
    uint32_t score_cap;

    float *hidden[2];
    float *norm;
    float *projection;
    float *gate;
    float *query;
    float *key;
    float *value;
    float *alpha_logit;
    float *beta_logit;
    float *log_decay;
    float *beta;
    float *heads;
    float *attn_out;
    float *score;

    float *router_logits;
    float *router_probability;
    int32_t selected[QWEN35_N_EXPERT_USED];
    float selected_weight[QWEN35_N_EXPERT_USED];
    float *routed_mid;
    float *moe_out;
    float *shared_gate;
    float *shared_up;
    float *shared_mid;
    float *shared_out;

    int8_t *dense_q8;
    float *dense_q8_scale;
    uint8_t *routed_q8k;
    uint8_t *routed_mid_q8k;
} ds4_qwen35_cpu_scratch;

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

bool ds4_qwen35_cpu_scratch_plan_make(
        uint32_t                       ctx_size,
        ds4_qwen35_cpu_scratch_plan   *plan);

/* Scratch is fully allocated up front so the token loop never grows it.  As
 * with the cache, initialize only a zero/uninitialized object and free it
 * before reinitializing. */
bool ds4_qwen35_cpu_scratch_init(
        ds4_qwen35_cpu_scratch *scratch,
        uint32_t                ctx_capacity);

void ds4_qwen35_cpu_scratch_free(ds4_qwen35_cpu_scratch *scratch);

uint64_t ds4_qwen35_cpu_scratch_allocated_bytes(
        const ds4_qwen35_cpu_scratch *scratch);

/* Allocation-free post-projection CPU operators.  Dimensions are explicit so
 * the model-free tests can compare the production path with small independent
 * fixtures; the runtime calls them with the fixed Qwen3.6 geometry above.
 * Conv output may alias input, delta output may alias value, and both gated
 * operators may alias input. */
bool ds4_qwen35_cpu_causal_conv_step_f32(
        float       *output,
        float       *state,
        const float *input,
        const float *weight,
        size_t       n_channel,
        size_t       kernel);

bool ds4_qwen35_cpu_gated_delta_controls_f32(
        float       *log_decay,
        float       *beta,
        const float *alpha_logit,
        const float *beta_logit,
        const float *ssm_a,
        const float *dt_bias,
        size_t       n_value_head);

bool ds4_qwen35_cpu_gated_delta_step_f32(
        float       *output,
        float       *state,
        const float *query,
        const float *key,
        const float *value,
        const float *log_decay,
        const float *beta,
        size_t       n_key_head,
        size_t       n_value_head,
        size_t       key_dim,
        size_t       value_dim);

bool ds4_qwen35_cpu_rmsnorm_gated_f32(
        float       *output,
        const float *input,
        const float *gate,
        const float *weight,
        size_t       n_vector,
        size_t       dim,
        float        epsilon);

bool ds4_qwen35_cpu_sigmoid_gate_f32(
        float       *output,
        const float *input,
        const float *gate_logit,
        size_t       n_vector,
        size_t       dim);

bool ds4_qwen35_cpu_softmax_top8_f32(
        int32_t     selected[QWEN35_N_EXPERT_USED],
        float       selected_weight[QWEN35_N_EXPERT_USED],
        float       probability[QWEN35_N_EXPERT],
        const float logits[QWEN35_N_EXPERT]);

bool ds4_qwen35_cpu_split_q_gate_f32(
        float       *query,
        float       *gate,
        const float *projection,
        size_t       n_query_head,
        size_t       head_dim);

bool ds4_qwen35_cpu_head_rms_norm_f32(
        float       *output,
        const float *input,
        const float *weight,
        size_t       n_head,
        size_t       head_dim,
        float        epsilon);

bool ds4_qwen35_cpu_text_rope_f32(
        float    *values,
        uint32_t  position,
        size_t    n_head,
        size_t    head_dim,
        size_t    n_rot,
        float     theta);

/* Decode one query row against cache rows [0, n_kv).  score may be reused for
 * every query head.  Projection/input/output buffers must not overlap. */
bool ds4_qwen35_cpu_gqa_decode_f32(
        float       *output,
        float       *score,
        size_t       score_cap,
        const float *query,
        const float *key,
        const float *value,
        size_t       n_kv,
        size_t       n_query_head,
        size_t       n_kv_head,
        size_t       head_dim);

bool ds4_qwen35_cpu_sigmoid_gate_elements_f32(
        float       *output,
        const float *input,
        const float *gate_logit,
        size_t       n_value);

#endif
