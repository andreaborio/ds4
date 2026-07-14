#include "ds4_qwen.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

static bool qwen35_u64_mul(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out || (a != 0 && b > UINT64_MAX / a)) return false;
    *out = a * b;
    return true;
}

static bool qwen35_u64_add(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out || b > UINT64_MAX - a) return false;
    *out = a + b;
    return true;
}

bool ds4_qwen35_layer_is_full_attention(uint32_t layer) {
    return layer < QWEN35_N_LAYER &&
           ((layer + 1u) % QWEN35_FULL_ATTENTION_INTERVAL) == 0;
}

bool ds4_qwen35_cpu_cache_plan_make(
        uint32_t                   ctx_size,
        ds4_qwen35_cpu_cache_plan *plan) {
    if (!plan || ctx_size == 0 || ctx_size > QWEN35_CONTEXT_LENGTH) {
        return false;
    }

    uint64_t conv_values = 0;
    uint64_t recurrent_values = 0;
    uint64_t gdn_conv_bytes = 0;
    uint64_t gdn_recurrent_bytes = 0;
    uint64_t fixed_bytes = 0;
    uint64_t kv_values_per_token = 0;
    uint64_t kv_bytes_per_token = 0;
    uint64_t max_kv_bytes = 0;
    uint64_t max_total_bytes = 0;

    if (!qwen35_u64_mul(QWEN35_SSM_CONV_CHANNEL,
                        QWEN35_SSM_CONV_KERNEL - 1u,
                        &conv_values) ||
        !qwen35_u64_mul(conv_values, QWEN35_RECURRENT_LAYER_COUNT,
                        &conv_values) ||
        !qwen35_u64_mul(conv_values, sizeof(float),
                        &gdn_conv_bytes) ||
        !qwen35_u64_mul(QWEN35_SSM_VALUE_HEAD, QWEN35_SSM_STATE,
                        &recurrent_values) ||
        !qwen35_u64_mul(recurrent_values, QWEN35_SSM_STATE,
                        &recurrent_values) ||
        !qwen35_u64_mul(recurrent_values, QWEN35_RECURRENT_LAYER_COUNT,
                        &recurrent_values) ||
        !qwen35_u64_mul(recurrent_values, sizeof(float),
                        &gdn_recurrent_bytes) ||
        !qwen35_u64_add(gdn_conv_bytes, gdn_recurrent_bytes,
                        &fixed_bytes) ||
        !qwen35_u64_mul(2u * QWEN35_N_HEAD_KV, QWEN35_N_HEAD_DIM,
                        &kv_values_per_token) ||
        !qwen35_u64_mul(kv_values_per_token,
                        QWEN35_FULL_ATTENTION_LAYER_COUNT,
                        &kv_values_per_token) ||
        !qwen35_u64_mul(kv_values_per_token, sizeof(float),
                        &kv_bytes_per_token) ||
        !qwen35_u64_mul(kv_bytes_per_token, ctx_size,
                        &max_kv_bytes) ||
        !qwen35_u64_add(fixed_bytes, max_kv_bytes,
                        &max_total_bytes)) {
        return false;
    }

    *plan = (ds4_qwen35_cpu_cache_plan){
        .gdn_conv_bytes = gdn_conv_bytes,
        .gdn_recurrent_bytes = gdn_recurrent_bytes,
        .fixed_bytes = fixed_bytes,
        .kv_bytes_per_token = kv_bytes_per_token,
        .max_kv_bytes = max_kv_bytes,
        .max_total_bytes = max_total_bytes,
    };
    return true;
}

void ds4_qwen35_cpu_cache_free(ds4_qwen35_cpu_cache *cache) {
    if (!cache) return;
    for (uint32_t layer = 0; layer < QWEN35_N_LAYER; layer++) {
        free(cache->layer[layer].key);
        free(cache->layer[layer].value);
        free(cache->layer[layer].conv);
        free(cache->layer[layer].recurrent);
    }
    memset(cache, 0, sizeof(*cache));
}

bool ds4_qwen35_cpu_cache_init(
        ds4_qwen35_cpu_cache *cache,
        uint32_t              ctx_capacity) {
    if (!cache) return false;
    memset(cache, 0, sizeof(*cache));

    ds4_qwen35_cpu_cache_plan plan;
    if (!ds4_qwen35_cpu_cache_plan_make(ctx_capacity, &plan) ||
        plan.fixed_bytes > SIZE_MAX) {
        return false;
    }

    const uint64_t conv_values =
        (uint64_t)QWEN35_SSM_CONV_CHANNEL *
        (QWEN35_SSM_CONV_KERNEL - 1u);
    const uint64_t recurrent_values =
        (uint64_t)QWEN35_SSM_VALUE_HEAD * QWEN35_SSM_STATE *
        QWEN35_SSM_STATE;
    if (conv_values > SIZE_MAX / sizeof(float) ||
        recurrent_values > SIZE_MAX / sizeof(float)) {
        return false;
    }

    const size_t conv_bytes = (size_t)conv_values * sizeof(float);
    const size_t recurrent_bytes =
        (size_t)recurrent_values * sizeof(float);
    for (uint32_t layer = 0; layer < QWEN35_N_LAYER; layer++) {
        if (ds4_qwen35_layer_is_full_attention(layer)) continue;
        ds4_qwen35_cpu_layer_state *state = &cache->layer[layer];
        state->conv = malloc(conv_bytes);
        state->recurrent = malloc(recurrent_bytes);
        if (!state->conv || !state->recurrent) {
            ds4_qwen35_cpu_cache_free(cache);
            return false;
        }
        memset(state->conv, 0, conv_bytes);
        memset(state->recurrent, 0, recurrent_bytes);
    }

    cache->plan = plan;
    cache->ctx_capacity = ctx_capacity;
    return true;
}

bool ds4_qwen35_cpu_cache_reserve(
        ds4_qwen35_cpu_cache *cache,
        uint32_t              required_tokens) {
    if (!cache || cache->ctx_capacity == 0 ||
        required_tokens > cache->ctx_capacity) {
        return false;
    }
    if (required_tokens <= cache->kv_capacity) return true;

    uint32_t new_capacity = cache->kv_capacity;
    if (new_capacity == 0) {
        new_capacity = cache->ctx_capacity < 64u ? cache->ctx_capacity : 64u;
    }
    while (new_capacity < required_tokens) {
        if (new_capacity > cache->ctx_capacity / 2u) {
            new_capacity = cache->ctx_capacity;
        } else {
            new_capacity *= 2u;
        }
    }

    const uint64_t row_values =
        (uint64_t)QWEN35_N_HEAD_KV * QWEN35_N_HEAD_DIM;
    uint64_t new_values = 0;
    uint64_t old_values = 0;
    if (!qwen35_u64_mul(new_capacity, row_values, &new_values) ||
        !qwen35_u64_mul(cache->kv_capacity, row_values, &old_values) ||
        new_values > SIZE_MAX / sizeof(float)) {
        return false;
    }
    const size_t new_bytes = (size_t)new_values * sizeof(float);
    const size_t old_bytes = (size_t)old_values * sizeof(float);

    float *new_key[QWEN35_N_LAYER] = {0};
    float *new_value[QWEN35_N_LAYER] = {0};
    for (uint32_t layer = 0; layer < QWEN35_N_LAYER; layer++) {
        if (!ds4_qwen35_layer_is_full_attention(layer)) continue;
        new_key[layer] = malloc(new_bytes);
        new_value[layer] = malloc(new_bytes);
        if (!new_key[layer] || !new_value[layer]) {
            for (uint32_t prior = 0; prior <= layer; prior++) {
                free(new_key[prior]);
                free(new_value[prior]);
            }
            return false;
        }

        ds4_qwen35_cpu_layer_state *state = &cache->layer[layer];
        if (old_bytes != 0) {
            memcpy(new_key[layer], state->key, old_bytes);
            memcpy(new_value[layer], state->value, old_bytes);
        }
        memset((uint8_t *)new_key[layer] + old_bytes, 0,
               new_bytes - old_bytes);
        memset((uint8_t *)new_value[layer] + old_bytes, 0,
               new_bytes - old_bytes);
    }

    for (uint32_t layer = 0; layer < QWEN35_N_LAYER; layer++) {
        if (!ds4_qwen35_layer_is_full_attention(layer)) continue;
        ds4_qwen35_cpu_layer_state *state = &cache->layer[layer];
        free(state->key);
        free(state->value);
        state->key = new_key[layer];
        state->value = new_value[layer];
    }
    cache->kv_capacity = new_capacity;
    return true;
}

void ds4_qwen35_cpu_cache_reset(ds4_qwen35_cpu_cache *cache) {
    if (!cache || cache->ctx_capacity == 0) return;

    const size_t conv_values =
        (size_t)QWEN35_SSM_CONV_CHANNEL *
        (QWEN35_SSM_CONV_KERNEL - 1u);
    const size_t recurrent_values =
        (size_t)QWEN35_SSM_VALUE_HEAD * QWEN35_SSM_STATE *
        QWEN35_SSM_STATE;
    for (uint32_t layer = 0; layer < QWEN35_N_LAYER; layer++) {
        ds4_qwen35_cpu_layer_state *state = &cache->layer[layer];
        if (!ds4_qwen35_layer_is_full_attention(layer)) {
            memset(state->conv, 0, conv_values * sizeof(state->conv[0]));
            memset(state->recurrent, 0,
                   recurrent_values * sizeof(state->recurrent[0]));
        }
    }
    /* K/V rows are overwritten before n_tokens exposes them.  Keeping both
     * allocation and contents makes reset O(fixed GDN state), not O(context). */
    cache->n_tokens = 0;
}

bool ds4_qwen35_cpu_cache_advance(
        ds4_qwen35_cpu_cache *cache,
        uint32_t              n_tokens) {
    if (!cache || cache->ctx_capacity == 0 ||
        cache->n_tokens > cache->kv_capacity ||
        cache->n_tokens > cache->ctx_capacity ||
        n_tokens > cache->kv_capacity - cache->n_tokens ||
        n_tokens > cache->ctx_capacity - cache->n_tokens) {
        return false;
    }
    cache->n_tokens += n_tokens;
    return true;
}

uint64_t ds4_qwen35_cpu_cache_allocated_bytes(
        const ds4_qwen35_cpu_cache *cache) {
    if (!cache || cache->ctx_capacity == 0) return 0;
    uint64_t kv_bytes = 0;
    uint64_t total_bytes = 0;
    if (!qwen35_u64_mul(cache->plan.kv_bytes_per_token,
                        cache->kv_capacity, &kv_bytes) ||
        !qwen35_u64_add(cache->plan.fixed_bytes, kv_bytes,
                        &total_bytes)) {
        return 0;
    }
    return total_bytes;
}

enum {
    QWEN35_CPU_Q8_0_CAP = QWEN35_SSM_INNER,
    QWEN35_CPU_Q8_0_BLOCK = 32,
    QWEN35_CPU_QK_K = 256,
    QWEN35_CPU_Q8_K_BLOCK_BYTES = 292,
};

bool ds4_qwen35_cpu_scratch_plan_make(
        uint32_t                     ctx_size,
        ds4_qwen35_cpu_scratch_plan *plan) {
    if (!plan || ctx_size == 0 || ctx_size > QWEN35_CONTEXT_LENGTH) {
        return false;
    }

    uint64_t float_values = 0;
    uint64_t float_bytes = 0;
    uint64_t quant_bytes = 0;
    uint64_t fixed_bytes = 0;
    uint64_t score_bytes = 0;
    uint64_t total_bytes = 0;

    const uint64_t fixed_float_values =
        2u * QWEN35_N_EMBD +                 /* hidden ping-pong */
        QWEN35_N_EMBD +                      /* norm */
        QWEN35_SSM_CONV_CHANNEL +            /* largest projection */
        QWEN35_SSM_INNER +                   /* gate */
        QWEN35_N_HEAD * QWEN35_N_HEAD_DIM +  /* query */
        QWEN35_SSM_GROUP * QWEN35_SSM_STATE +/* key */
        QWEN35_SSM_VALUE_HEAD * QWEN35_SSM_STATE + /* value */
        4u * QWEN35_SSM_DT_RANK +            /* alpha/beta controls */
        QWEN35_N_HEAD * QWEN35_N_HEAD_DIM +  /* attention heads */
        QWEN35_N_EMBD +                      /* attention output */
        2u * QWEN35_N_EXPERT +               /* router logits/prob */
        QWEN35_N_EXPERT_USED * QWEN35_N_FF_EXP + /* routed mid */
        QWEN35_N_EMBD +                      /* routed output */
        3u * QWEN35_N_FF_SHARED +            /* shared gate/up/mid */
        QWEN35_N_EMBD;                       /* shared output */
    const uint64_t dense_scale_values =
        (QWEN35_CPU_Q8_0_CAP + QWEN35_CPU_Q8_0_BLOCK - 1u) /
        QWEN35_CPU_Q8_0_BLOCK;
    const uint64_t routed_input_blocks =
        QWEN35_N_EMBD / QWEN35_CPU_QK_K;
    const uint64_t routed_mid_blocks =
        QWEN35_N_EXPERT_USED * QWEN35_N_FF_EXP / QWEN35_CPU_QK_K;

    if (!qwen35_u64_add(float_values, fixed_float_values, &float_values) ||
        !qwen35_u64_mul(float_values, sizeof(float), &float_bytes) ||
        !qwen35_u64_mul(dense_scale_values, sizeof(float), &quant_bytes) ||
        !qwen35_u64_add(quant_bytes, QWEN35_CPU_Q8_0_CAP,
                        &quant_bytes)) {
        return false;
    }
    uint64_t routed_quant_bytes = 0;
    if (!qwen35_u64_add(routed_input_blocks, routed_mid_blocks,
                        &routed_quant_bytes) ||
        !qwen35_u64_mul(routed_quant_bytes,
                        QWEN35_CPU_Q8_K_BLOCK_BYTES,
                        &routed_quant_bytes) ||
        !qwen35_u64_add(quant_bytes, routed_quant_bytes, &quant_bytes) ||
        !qwen35_u64_add(float_bytes, quant_bytes, &fixed_bytes) ||
        !qwen35_u64_mul(ctx_size, sizeof(float), &score_bytes) ||
        !qwen35_u64_add(fixed_bytes, score_bytes, &total_bytes)) {
        return false;
    }

    *plan = (ds4_qwen35_cpu_scratch_plan){
        .float_bytes = float_bytes,
        .quant_bytes = quant_bytes,
        .fixed_bytes = fixed_bytes,
        .score_bytes = score_bytes,
        .total_bytes = total_bytes,
    };
    return true;
}

void ds4_qwen35_cpu_scratch_free(ds4_qwen35_cpu_scratch *scratch) {
    if (!scratch) return;
    free(scratch->arena);
    memset(scratch, 0, sizeof(*scratch));
}

bool ds4_qwen35_cpu_scratch_init(
        ds4_qwen35_cpu_scratch *scratch,
        uint32_t                ctx_capacity) {
    if (!scratch) return false;
    memset(scratch, 0, sizeof(*scratch));

    ds4_qwen35_cpu_scratch_plan plan;
    if (!ds4_qwen35_cpu_scratch_plan_make(ctx_capacity, &plan) ||
        plan.total_bytes > SIZE_MAX) {
        return false;
    }

    uint8_t *arena = malloc((size_t)plan.total_bytes);
    if (!arena) return false;
    memset(arena, 0, (size_t)plan.total_bytes);

    uint8_t *cursor = arena;
#define QWEN35_TAKE_FLOAT(field_, count_) do {                               \
        scratch->field_ = (float *)cursor;                                  \
        cursor += (size_t)(count_) * sizeof(float);                         \
    } while (0)
    QWEN35_TAKE_FLOAT(hidden[0], QWEN35_N_EMBD);
    QWEN35_TAKE_FLOAT(hidden[1], QWEN35_N_EMBD);
    QWEN35_TAKE_FLOAT(norm, QWEN35_N_EMBD);
    QWEN35_TAKE_FLOAT(projection, QWEN35_SSM_CONV_CHANNEL);
    QWEN35_TAKE_FLOAT(gate, QWEN35_SSM_INNER);
    QWEN35_TAKE_FLOAT(query, QWEN35_N_HEAD * QWEN35_N_HEAD_DIM);
    QWEN35_TAKE_FLOAT(key, QWEN35_SSM_GROUP * QWEN35_SSM_STATE);
    QWEN35_TAKE_FLOAT(value,
                      QWEN35_SSM_VALUE_HEAD * QWEN35_SSM_STATE);
    QWEN35_TAKE_FLOAT(alpha_logit, QWEN35_SSM_DT_RANK);
    QWEN35_TAKE_FLOAT(beta_logit, QWEN35_SSM_DT_RANK);
    QWEN35_TAKE_FLOAT(log_decay, QWEN35_SSM_DT_RANK);
    QWEN35_TAKE_FLOAT(beta, QWEN35_SSM_DT_RANK);
    QWEN35_TAKE_FLOAT(heads, QWEN35_N_HEAD * QWEN35_N_HEAD_DIM);
    QWEN35_TAKE_FLOAT(attn_out, QWEN35_N_EMBD);
    QWEN35_TAKE_FLOAT(router_logits, QWEN35_N_EXPERT);
    QWEN35_TAKE_FLOAT(router_probability, QWEN35_N_EXPERT);
    QWEN35_TAKE_FLOAT(routed_mid,
                      QWEN35_N_EXPERT_USED * QWEN35_N_FF_EXP);
    QWEN35_TAKE_FLOAT(moe_out, QWEN35_N_EMBD);
    QWEN35_TAKE_FLOAT(shared_gate, QWEN35_N_FF_SHARED);
    QWEN35_TAKE_FLOAT(shared_up, QWEN35_N_FF_SHARED);
    QWEN35_TAKE_FLOAT(shared_mid, QWEN35_N_FF_SHARED);
    QWEN35_TAKE_FLOAT(shared_out, QWEN35_N_EMBD);
#undef QWEN35_TAKE_FLOAT

    scratch->dense_q8 = (int8_t *)cursor;
    cursor += QWEN35_CPU_Q8_0_CAP;
    scratch->dense_q8_scale = (float *)cursor;
    cursor += (QWEN35_CPU_Q8_0_CAP / QWEN35_CPU_Q8_0_BLOCK) *
              sizeof(float);
    scratch->routed_q8k = cursor;
    cursor += (QWEN35_N_EMBD / QWEN35_CPU_QK_K) *
              QWEN35_CPU_Q8_K_BLOCK_BYTES;
    scratch->routed_mid_q8k = cursor;
    cursor += (QWEN35_N_EXPERT_USED * QWEN35_N_FF_EXP /
               QWEN35_CPU_QK_K) * QWEN35_CPU_Q8_K_BLOCK_BYTES;
    scratch->score = (float *)cursor;
    cursor += (size_t)ctx_capacity * sizeof(float);

    if ((uint64_t)(cursor - arena) != plan.total_bytes) {
        free(arena);
        memset(scratch, 0, sizeof(*scratch));
        return false;
    }
    scratch->arena = arena;
    scratch->arena_bytes = plan.total_bytes;
    scratch->ctx_capacity = ctx_capacity;
    scratch->score_cap = ctx_capacity;
    return true;
}

uint64_t ds4_qwen35_cpu_scratch_allocated_bytes(
        const ds4_qwen35_cpu_scratch *scratch) {
    return scratch ? scratch->arena_bytes : 0;
}

static float qwen35_sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    const float e = expf(x);
    return e / (1.0f + e);
}

/* ds4 builds with -ffast-math, which lets the compiler assume that ordinary
 * isfinite() calls are always true.  Model weights are binary32 on every
 * supported backend, so inspect the exponent bits for fail-closed guards. */
static bool qwen35_f32_is_finite(float value) {
    typedef char qwen35_requires_binary32[
        sizeof(float) == sizeof(uint32_t) ? 1 : -1];
    (void)sizeof(qwen35_requires_binary32);
    uint32_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}

static float qwen35_softplus(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

bool ds4_qwen35_cpu_causal_conv_step_f32(
        float       *output,
        float       *state,
        const float *input,
        const float *weight,
        size_t       n_channel,
        size_t       kernel) {
    if (!output || !state || !input || !weight || n_channel == 0 ||
        kernel < 2 || n_channel > SIZE_MAX / (kernel - 1u) ||
        n_channel > SIZE_MAX / kernel) {
        return false;
    }

    const size_t history_len = kernel - 1u;
    for (size_t channel = 0; channel < n_channel; channel++) {
        float *history = state + channel * history_len;
        const float *filter = weight + channel * kernel;
        const float current = input[channel];
        float sum = current * filter[kernel - 1u];
        for (size_t k = 0; k < history_len; k++) {
            sum += history[k] * filter[k];
        }
        output[channel] = sum * qwen35_sigmoid(sum);

        if (history_len > 1u) {
            memmove(history, history + 1,
                    (history_len - 1u) * sizeof(history[0]));
        }
        history[history_len - 1u] = current;
    }
    return true;
}

bool ds4_qwen35_cpu_gated_delta_controls_f32(
        float       *log_decay,
        float       *beta,
        const float *alpha_logit,
        const float *beta_logit,
        const float *ssm_a,
        const float *dt_bias,
        size_t       n_value_head) {
    if (!log_decay || !beta || !alpha_logit || !beta_logit || !ssm_a ||
        !dt_bias || n_value_head == 0) {
        return false;
    }
    for (size_t head = 0; head < n_value_head; head++) {
        beta[head] = qwen35_sigmoid(beta_logit[head]);
        /* GGUF already stores -exp(A_log); it must not be exponentiated a
         * second time before scaling the positive timestep. */
        log_decay[head] = ssm_a[head] *
            qwen35_softplus(alpha_logit[head] + dt_bias[head]);
    }
    return true;
}

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
        size_t       value_dim) {
    if (!output || !state || !query || !key || !value || !log_decay ||
        !beta || n_key_head == 0 || n_value_head == 0 || key_dim == 0 ||
        value_dim == 0 || n_value_head % n_key_head != 0 ||
        key_dim > SIZE_MAX / value_dim ||
        n_value_head > SIZE_MAX / (key_dim * value_dim)) {
        return false;
    }

    const float query_scale = 1.0f / sqrtf((float)key_dim);
    const size_t state_stride = key_dim * value_dim;
    for (size_t value_head = 0; value_head < n_value_head; value_head++) {
        /* The pinned converter tiles V-side quantities.  Modulo is therefore
         * intentional and differs from the full-attention contiguous GQA map. */
        const size_t key_head = value_head % n_key_head;
        const float *q = query + key_head * key_dim;
        const float *k = key + key_head * key_dim;
        const float *v = value + value_head * value_dim;
        float *head_state = state + value_head * state_stride;
        float *head_output = output + value_head * value_dim;

        float q_square = 0.0f;
        float k_square = 0.0f;
        for (size_t i = 0; i < key_dim; i++) {
            q_square += q[i] * q[i];
            k_square += k[i] * k[i];
        }
        const float q_inv = query_scale / sqrtf(q_square + 1.0e-6f);
        const float k_inv = 1.0f / sqrtf(k_square + 1.0e-6f);
        const float decay = expf(log_decay[value_head]);
        const float step = beta[value_head];

        for (size_t i = 0; i < state_stride; i++) head_state[i] *= decay;
        for (size_t j = 0; j < value_dim; j++) {
            float memory = 0.0f;
            for (size_t i = 0; i < key_dim; i++) {
                memory += head_state[j * key_dim + i] * (k[i] * k_inv);
            }
            const float delta = (v[j] - memory) * step;
            for (size_t i = 0; i < key_dim; i++) {
                head_state[j * key_dim + i] += (k[i] * k_inv) * delta;
            }
        }
        for (size_t j = 0; j < value_dim; j++) {
            float sum = 0.0f;
            for (size_t i = 0; i < key_dim; i++) {
                sum += head_state[j * key_dim + i] * (q[i] * q_inv);
            }
            head_output[j] = sum;
        }
    }
    return true;
}

bool ds4_qwen35_cpu_rmsnorm_gated_f32(
        float       *output,
        const float *input,
        const float *gate,
        const float *weight,
        size_t       n_vector,
        size_t       dim,
        float        epsilon) {
    if (!output || !input || !gate || !weight || n_vector == 0 || dim == 0 ||
        !(epsilon > 0.0f) || !qwen35_f32_is_finite(epsilon) ||
        n_vector > SIZE_MAX / dim) {
        return false;
    }
    for (size_t vector = 0; vector < n_vector; vector++) {
        const size_t base = vector * dim;
        float square = 0.0f;
        for (size_t i = 0; i < dim; i++) {
            square += input[base + i] * input[base + i];
        }
        const float inv_rms = 1.0f / sqrtf(square / (float)dim + epsilon);
        for (size_t i = 0; i < dim; i++) {
            output[base + i] = input[base + i] * inv_rms * weight[i] *
                               (gate[base + i] * qwen35_sigmoid(gate[base + i]));
        }
    }
    return true;
}

bool ds4_qwen35_cpu_sigmoid_gate_f32(
        float       *output,
        const float *input,
        const float *gate_logit,
        size_t       n_vector,
        size_t       dim) {
    if (!output || !input || !gate_logit || n_vector == 0 || dim == 0 ||
        n_vector > SIZE_MAX / dim) {
        return false;
    }
    for (size_t vector = 0; vector < n_vector; vector++) {
        const float gate = qwen35_sigmoid(gate_logit[vector]);
        for (size_t i = 0; i < dim; i++) {
            output[vector * dim + i] = input[vector * dim + i] * gate;
        }
    }
    return true;
}

bool ds4_qwen35_cpu_softmax_top8_f32(
        int32_t     selected[QWEN35_N_EXPERT_USED],
        float       selected_weight[QWEN35_N_EXPERT_USED],
        float       probability[QWEN35_N_EXPERT],
        const float logits[QWEN35_N_EXPERT]) {
    if (!selected || !selected_weight || !probability || !logits) return false;

    float maximum = logits[0];
    if (!qwen35_f32_is_finite(maximum)) return false;
    for (size_t expert = 1; expert < QWEN35_N_EXPERT; expert++) {
        if (!qwen35_f32_is_finite(logits[expert])) return false;
        if (logits[expert] > maximum) maximum = logits[expert];
    }

    float total = 0.0f;
    for (size_t expert = 0; expert < QWEN35_N_EXPERT; expert++) {
        probability[expert] = expf(logits[expert] - maximum);
        total += probability[expert];
    }
    if (!(total > 0.0f) || !qwen35_f32_is_finite(total)) return false;
    for (size_t expert = 0; expert < QWEN35_N_EXPERT; expert++) {
        probability[expert] /= total;
    }

    for (size_t slot = 0; slot < QWEN35_N_EXPERT_USED; slot++) {
        size_t best = QWEN35_N_EXPERT;
        for (size_t expert = 0; expert < QWEN35_N_EXPERT; expert++) {
            bool used = false;
            for (size_t prior = 0; prior < slot; prior++) {
                if (selected[prior] == (int32_t)expert) {
                    used = true;
                    break;
                }
            }
            if (used) continue;
            if (best == QWEN35_N_EXPERT ||
                probability[expert] > probability[best] ||
                (probability[expert] == probability[best] && expert < best)) {
                best = expert;
            }
        }
        selected[slot] = (int32_t)best;
        selected_weight[slot] = probability[best];
    }

    float selected_total = 0.0f;
    for (size_t slot = 0; slot < QWEN35_N_EXPERT_USED; slot++) {
        selected_total += selected_weight[slot];
    }
    if (!(selected_total > 0.0f) ||
        !qwen35_f32_is_finite(selected_total)) return false;
    for (size_t slot = 0; slot < QWEN35_N_EXPERT_USED; slot++) {
        selected_weight[slot] /= selected_total;
    }
    return true;
}

bool ds4_qwen35_cpu_split_q_gate_f32(
        float       *query,
        float       *gate,
        const float *projection,
        size_t       n_query_head,
        size_t       head_dim) {
    if (!query || !gate || !projection || n_query_head == 0 || head_dim == 0 ||
        n_query_head > SIZE_MAX / head_dim ||
        n_query_head * head_dim > SIZE_MAX / 2u) {
        return false;
    }
    for (size_t head = 0; head < n_query_head; head++) {
        const size_t projection_base = head * 2u * head_dim;
        const size_t output_base = head * head_dim;
        memcpy(query + output_base, projection + projection_base,
               head_dim * sizeof(query[0]));
        memcpy(gate + output_base, projection + projection_base + head_dim,
               head_dim * sizeof(gate[0]));
    }
    return true;
}

bool ds4_qwen35_cpu_head_rms_norm_f32(
        float       *output,
        const float *input,
        const float *weight,
        size_t       n_head,
        size_t       head_dim,
        float        epsilon) {
    if (!output || !input || !weight || n_head == 0 || head_dim == 0 ||
        !(epsilon > 0.0f) || !qwen35_f32_is_finite(epsilon) ||
        n_head > SIZE_MAX / head_dim) {
        return false;
    }
    for (size_t head = 0; head < n_head; head++) {
        const size_t base = head * head_dim;
        float square = 0.0f;
        for (size_t i = 0; i < head_dim; i++) {
            square += input[base + i] * input[base + i];
        }
        const float inv_rms =
            1.0f / sqrtf(square / (float)head_dim + epsilon);
        for (size_t i = 0; i < head_dim; i++) {
            /* The converter already applied the zero-centred norm's +1. */
            output[base + i] = input[base + i] * inv_rms * weight[i];
        }
    }
    return true;
}

bool ds4_qwen35_cpu_text_rope_f32(
        float    *values,
        uint32_t  position,
        size_t    n_head,
        size_t    head_dim,
        size_t    n_rot,
        float     theta) {
    if (!values || n_head == 0 || head_dim == 0 || n_rot == 0 ||
        n_rot > head_dim || (n_rot & 1u) || !(theta > 0.0f) ||
        !qwen35_f32_is_finite(theta) || n_head > SIZE_MAX / head_dim) {
        return false;
    }

    const size_t half = n_rot / 2u;
    for (size_t i = 0; i < half; i++) {
        const float exponent = (2.0f * (float)i) / (float)n_rot;
        const float angle = (float)position / powf(theta, exponent);
        const float cosine = cosf(angle);
        const float sine = sinf(angle);
        for (size_t head = 0; head < n_head; head++) {
            float *x = values + head * head_dim;
            const float a = x[i];
            const float b = x[i + half];
            x[i] = a * cosine - b * sine;
            x[i + half] = b * cosine + a * sine;
        }
    }
    return true;
}

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
        size_t       head_dim) {
    if (!output || !score || !query || !key || !value || n_kv == 0 ||
        n_kv > score_cap ||
        n_query_head == 0 || n_kv_head == 0 || head_dim == 0 ||
        n_query_head % n_kv_head != 0 ||
        n_query_head > SIZE_MAX / head_dim ||
        n_kv_head > SIZE_MAX / head_dim ||
        n_kv > SIZE_MAX / (n_kv_head * head_dim)) {
        return false;
    }

    const size_t query_per_kv = n_query_head / n_kv_head;
    const size_t kv_stride = n_kv_head * head_dim;
    const float scale = 1.0f / sqrtf((float)head_dim);
    for (size_t query_head = 0; query_head < n_query_head; query_head++) {
        const size_t kv_head = query_head / query_per_kv;
        const float *q = query + query_head * head_dim;
        float maximum = -FLT_MAX;
        for (size_t token = 0; token < n_kv; token++) {
            const float *k = key + token * kv_stride + kv_head * head_dim;
            float dot = 0.0f;
            for (size_t i = 0; i < head_dim; i++) dot += q[i] * k[i];
            score[token] = dot * scale;
            if (score[token] > maximum) maximum = score[token];
        }

        float denominator = 0.0f;
        for (size_t token = 0; token < n_kv; token++) {
            score[token] = expf(score[token] - maximum);
            denominator += score[token];
        }
        if (!(denominator > 0.0f) ||
            !qwen35_f32_is_finite(denominator)) return false;

        float *out = output + query_head * head_dim;
        memset(out, 0, head_dim * sizeof(out[0]));
        for (size_t token = 0; token < n_kv; token++) {
            const float probability = score[token] / denominator;
            const float *v = value + token * kv_stride + kv_head * head_dim;
            for (size_t i = 0; i < head_dim; i++) {
                out[i] += probability * v[i];
            }
        }
    }
    return true;
}

bool ds4_qwen35_cpu_sigmoid_gate_elements_f32(
        float       *output,
        const float *input,
        const float *gate_logit,
        size_t       n_value) {
    if (!output || !input || !gate_logit || n_value == 0) return false;
    for (size_t i = 0; i < n_value; i++) {
        output[i] = input[i] * qwen35_sigmoid(gate_logit[i]);
    }
    return true;
}
