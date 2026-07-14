#include "ds4_qwen.h"

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
