#include "ds4_qwen.h"

#include <stdio.h>
#include <string.h>

static int failures;

#define CHECK(condition) do {                                                \
    if (!(condition)) {                                                      \
        fprintf(stderr, "qwen state check failed at %s:%d: %s\n",          \
                __FILE__, __LINE__, #condition);                             \
        failures++;                                                          \
    }                                                                        \
} while (0)

static void test_layer_pattern(void) {
    uint32_t full = 0;
    uint32_t recurrent = 0;
    for (uint32_t layer = 0; layer < QWEN35_N_LAYER; layer++) {
        const bool expected =
            layer == 3 || layer == 7 || layer == 11 || layer == 15 ||
            layer == 19 || layer == 23 || layer == 27 || layer == 31 ||
            layer == 35 || layer == 39;
        CHECK(ds4_qwen35_layer_is_full_attention(layer) == expected);
        if (expected) full++;
        else recurrent++;
    }
    CHECK(full == QWEN35_FULL_ATTENTION_LAYER_COUNT);
    CHECK(recurrent == QWEN35_RECURRENT_LAYER_COUNT);
    CHECK(!ds4_qwen35_layer_is_full_attention(QWEN35_N_LAYER));
}

static void test_memory_plan(void) {
    ds4_qwen35_cpu_cache_plan plan = {0};
    CHECK(!ds4_qwen35_cpu_cache_plan_make(0, &plan));
    CHECK(!ds4_qwen35_cpu_cache_plan_make(
        QWEN35_CONTEXT_LENGTH + 1u, &plan));
    CHECK(!ds4_qwen35_cpu_cache_plan_make(UINT32_MAX, &plan));
    CHECK(!ds4_qwen35_cpu_cache_plan_make(1, NULL));
    CHECK(ds4_qwen35_cpu_cache_plan_make(1, &plan));
    CHECK(plan.gdn_conv_bytes == UINT64_C(2949120));
    CHECK(plan.gdn_recurrent_bytes == UINT64_C(62914560));
    CHECK(plan.fixed_bytes == UINT64_C(65863680));
    CHECK(plan.kv_bytes_per_token == UINT64_C(40960));
    CHECK(plan.max_kv_bytes == UINT64_C(40960));
    CHECK(plan.max_total_bytes == UINT64_C(65904640));

    CHECK(ds4_qwen35_cpu_cache_plan_make(32768, &plan));
    CHECK(plan.fixed_bytes == UINT64_C(65863680));
    CHECK(plan.max_kv_bytes == UINT64_C(1342177280));
    CHECK(plan.max_total_bytes == UINT64_C(1408040960));
}

static void test_scratch_plan(void) {
    ds4_qwen35_cpu_scratch_plan plan = {0};
    CHECK(!ds4_qwen35_cpu_scratch_plan_make(0, &plan));
    CHECK(!ds4_qwen35_cpu_scratch_plan_make(
        QWEN35_CONTEXT_LENGTH + 1u, &plan));
    CHECK(!ds4_qwen35_cpu_scratch_plan_make(UINT32_MAX, &plan));
    CHECK(!ds4_qwen35_cpu_scratch_plan_make(1, NULL));
    CHECK(ds4_qwen35_cpu_scratch_plan_make(1, &plan));
    CHECK(plan.float_bytes == UINT64_C(180736));
    CHECK(plan.quant_bytes == UINT64_C(11616));
    CHECK(plan.fixed_bytes == UINT64_C(192352));
    CHECK(plan.score_bytes == UINT64_C(4));
    CHECK(plan.total_bytes == UINT64_C(192356));

    CHECK(ds4_qwen35_cpu_scratch_plan_make(32768, &plan));
    CHECK(plan.fixed_bytes == UINT64_C(192352));
    CHECK(plan.score_bytes == UINT64_C(131072));
    CHECK(plan.total_bytes == UINT64_C(323424));
}

static void test_scratch_lifecycle(void) {
    ds4_qwen35_cpu_scratch scratch;
    memset(&scratch, 0xa5, sizeof(scratch));
    CHECK(!ds4_qwen35_cpu_scratch_init(&scratch, 0));

    memset(&scratch, 0, sizeof(scratch));
    CHECK(ds4_qwen35_cpu_scratch_init(&scratch, 130));
    CHECK(scratch.arena != NULL);
    CHECK(scratch.ctx_capacity == 130);
    CHECK(scratch.score_cap == 130);
    CHECK(ds4_qwen35_cpu_scratch_allocated_bytes(&scratch) ==
          UINT64_C(192872));
    CHECK(scratch.hidden[0] != NULL);
    CHECK(scratch.hidden[1] != NULL);
    CHECK(scratch.norm != NULL);
    CHECK(scratch.projection != NULL);
    CHECK(scratch.gate != NULL);
    CHECK(scratch.query != NULL);
    CHECK(scratch.key != NULL);
    CHECK(scratch.value != NULL);
    CHECK(scratch.alpha_logit != NULL);
    CHECK(scratch.beta_logit != NULL);
    CHECK(scratch.log_decay != NULL);
    CHECK(scratch.beta != NULL);
    CHECK(scratch.heads != NULL);
    CHECK(scratch.attn_out != NULL);
    CHECK(scratch.score != NULL);
    CHECK(scratch.router_logits != NULL);
    CHECK(scratch.router_probability != NULL);
    CHECK(scratch.routed_mid != NULL);
    CHECK(scratch.moe_out != NULL);
    CHECK(scratch.shared_gate != NULL);
    CHECK(scratch.shared_up != NULL);
    CHECK(scratch.shared_mid != NULL);
    CHECK(scratch.shared_out != NULL);
    CHECK(scratch.dense_q8 != NULL);
    CHECK(scratch.dense_q8_scale != NULL);
    CHECK(scratch.routed_q8k != NULL);
    CHECK(scratch.routed_mid_q8k != NULL);

    scratch.hidden[0][0] = 1.0f;
    scratch.hidden[1][QWEN35_N_EMBD - 1u] = 2.0f;
    scratch.projection[QWEN35_SSM_CONV_CHANNEL - 1u] = 3.0f;
    scratch.routed_mid[
        QWEN35_N_EXPERT_USED * QWEN35_N_FF_EXP - 1u] = 4.0f;
    scratch.score[129] = 5.0f;
    CHECK(scratch.hidden[0][0] == 1.0f);
    CHECK(scratch.hidden[1][QWEN35_N_EMBD - 1u] == 2.0f);
    CHECK(scratch.projection[QWEN35_SSM_CONV_CHANNEL - 1u] == 3.0f);
    CHECK(scratch.routed_mid[
        QWEN35_N_EXPERT_USED * QWEN35_N_FF_EXP - 1u] == 4.0f);
    CHECK(scratch.score[129] == 5.0f);

    ds4_qwen35_cpu_scratch_free(&scratch);
    ds4_qwen35_cpu_scratch empty = {0};
    CHECK(memcmp(&scratch, &empty, sizeof(scratch)) == 0);
    ds4_qwen35_cpu_scratch_free(&scratch);
    CHECK(ds4_qwen35_cpu_scratch_allocated_bytes(&scratch) == 0);
}

static void test_cache_lifecycle(void) {
    ds4_qwen35_cpu_cache cache;
    memset(&cache, 0xa5, sizeof(cache));
    CHECK(!ds4_qwen35_cpu_cache_init(&cache, 0));

    memset(&cache, 0, sizeof(cache));
    CHECK(ds4_qwen35_cpu_cache_init(&cache, 130));
    CHECK(cache.ctx_capacity == 130);
    CHECK(cache.kv_capacity == 0);
    CHECK(cache.n_tokens == 0);
    CHECK(cache.plan.fixed_bytes == UINT64_C(65863680));
    CHECK(cache.plan.max_kv_bytes == UINT64_C(5324800));
    CHECK(ds4_qwen35_cpu_cache_allocated_bytes(&cache) ==
          UINT64_C(65863680));

    const size_t conv_values =
        (size_t)QWEN35_SSM_CONV_CHANNEL *
        (QWEN35_SSM_CONV_KERNEL - 1u);
    const size_t recurrent_values =
        (size_t)QWEN35_SSM_VALUE_HEAD * QWEN35_SSM_STATE *
        QWEN35_SSM_STATE;
    for (uint32_t layer = 0; layer < QWEN35_N_LAYER; layer++) {
        ds4_qwen35_cpu_layer_state *state = &cache.layer[layer];
        if (ds4_qwen35_layer_is_full_attention(layer)) {
            CHECK(state->key == NULL);
            CHECK(state->value == NULL);
            CHECK(state->conv == NULL);
            CHECK(state->recurrent == NULL);
        } else {
            CHECK(state->key == NULL);
            CHECK(state->value == NULL);
            CHECK(state->conv != NULL);
            CHECK(state->recurrent != NULL);
            state->conv[0] = 1.0f;
            state->conv[conv_values - 1] = 2.0f;
            state->recurrent[0] = 3.0f;
            state->recurrent[recurrent_values - 1] = 4.0f;
        }
    }

    CHECK(ds4_qwen35_cpu_cache_reserve(&cache, 0));
    CHECK(cache.kv_capacity == 0);
    CHECK(ds4_qwen35_cpu_cache_reserve(&cache, 1));
    CHECK(cache.kv_capacity == 64);
    CHECK(ds4_qwen35_cpu_cache_allocated_bytes(&cache) ==
          UINT64_C(68485120));
    for (uint32_t layer = 0; layer < QWEN35_N_LAYER; layer++) {
        ds4_qwen35_cpu_layer_state *state = &cache.layer[layer];
        if (ds4_qwen35_layer_is_full_attention(layer)) {
            CHECK(state->key != NULL);
            CHECK(state->value != NULL);
            state->key[0] = 9.0f + (float)layer;
            state->value[0] = -4.0f - (float)layer;
            const size_t last_old =
                (size_t)64 * QWEN35_N_HEAD_KV * QWEN35_N_HEAD_DIM - 1u;
            state->key[last_old] = 19.0f + (float)layer;
            state->value[last_old] = -14.0f - (float)layer;
        }
    }
    CHECK(ds4_qwen35_cpu_cache_advance(&cache, 1));

    CHECK(ds4_qwen35_cpu_cache_reserve(&cache, 65));
    CHECK(cache.kv_capacity == 128);
    for (uint32_t layer = 0; layer < QWEN35_N_LAYER; layer++) {
        ds4_qwen35_cpu_layer_state *state = &cache.layer[layer];
        if (ds4_qwen35_layer_is_full_attention(layer)) {
            CHECK(state->key[0] == 9.0f + (float)layer);
            CHECK(state->value[0] == -4.0f - (float)layer);
            const size_t last_old =
                (size_t)64 * QWEN35_N_HEAD_KV * QWEN35_N_HEAD_DIM - 1u;
            CHECK(state->key[last_old] == 19.0f + (float)layer);
            CHECK(state->value[last_old] == -14.0f - (float)layer);
            const size_t new_row =
                (size_t)64 * QWEN35_N_HEAD_KV * QWEN35_N_HEAD_DIM;
            CHECK(state->key[new_row] == 0.0f);
            CHECK(state->value[new_row] == 0.0f);
        }
    }
    CHECK(ds4_qwen35_cpu_cache_advance(&cache, 127));
    CHECK(cache.n_tokens == 128);
    CHECK(!ds4_qwen35_cpu_cache_advance(&cache, 1));
    CHECK(cache.n_tokens == 128);
    float *key_before_failure = cache.layer[3].key;
    float *value_before_failure = cache.layer[3].value;
    CHECK(!ds4_qwen35_cpu_cache_reserve(&cache, 131));
    CHECK(cache.kv_capacity == 128);
    CHECK(cache.layer[3].key == key_before_failure);
    CHECK(cache.layer[3].value == value_before_failure);
    float *key_before_growth = cache.layer[3].key;
    float *value_before_growth = cache.layer[3].value;
    CHECK(ds4_qwen35_cpu_cache_reserve(&cache, 130));
    CHECK(cache.kv_capacity == 130);
    CHECK(cache.layer[3].key != key_before_growth);
    CHECK(cache.layer[3].value != value_before_growth);
    CHECK(cache.layer[3].key[0] == 12.0f);
    CHECK(cache.layer[3].value[0] == -7.0f);
    CHECK(ds4_qwen35_cpu_cache_advance(&cache, 2));
    CHECK(cache.n_tokens == 130);
    CHECK(!ds4_qwen35_cpu_cache_advance(&cache, 1));
    CHECK(cache.n_tokens == 130);

    ds4_qwen35_cpu_cache_reset(&cache);
    CHECK(cache.n_tokens == 0);
    CHECK(cache.kv_capacity == 130);
    CHECK(ds4_qwen35_cpu_cache_allocated_bytes(&cache) ==
          UINT64_C(71188480));
    CHECK(cache.layer[3].key[0] == 12.0f);
    CHECK(cache.layer[3].value[0] == -7.0f);
    for (uint32_t layer = 0; layer < QWEN35_N_LAYER; layer++) {
        ds4_qwen35_cpu_layer_state *state = &cache.layer[layer];
        if (!ds4_qwen35_layer_is_full_attention(layer)) {
            CHECK(state->conv[0] == 0.0f);
            CHECK(state->conv[conv_values - 1] == 0.0f);
            CHECK(state->recurrent[0] == 0.0f);
            CHECK(state->recurrent[recurrent_values - 1] == 0.0f);
        }
    }

    ds4_qwen35_cpu_cache_free(&cache);
    ds4_qwen35_cpu_cache empty = {0};
    CHECK(memcmp(&cache, &empty, sizeof(cache)) == 0);
    ds4_qwen35_cpu_cache_free(&cache);
    CHECK(!ds4_qwen35_cpu_cache_reserve(&cache, 1));
    CHECK(!ds4_qwen35_cpu_cache_advance(&cache, 1));
    CHECK(ds4_qwen35_cpu_cache_allocated_bytes(&cache) == 0);
}

int main(void) {
    test_layer_pattern();
    test_memory_plan();
    test_scratch_plan();
    test_cache_lifecycle();
    test_scratch_lifecycle();
    if (failures) {
        fprintf(stderr, "qwen CPU state tests: %d failure(s)\n", failures);
        return 1;
    }
    puts("qwen CPU state tests: OK");
    return 0;
}
