#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_qwen4exp_ref.h"
#include "runtime/ds4_qwen4exp_graph.h"
#include "qwen4exp/qwen4exp_graph_golden.inc"

/* Standalone compile lane: production integration textually includes the same
 * implementation from ds4.c, without exposing any ds4 private struct here. */
#include "runtime/ds4_qwen4exp_graph.inc"

enum {
    FIXTURE_TOKENS = 6,
};

typedef struct {
    size_t fail_after;
    size_t nonfinite_after;
    bool capture;
    float embedding[FIXTURE_TOKENS * DS4_Q4E_GRAPH_HIDDEN];
    uint32_t ple_rows[FIXTURE_TOKENS * DS4_Q4E_GRAPH_PLE_HEADS];
    float ple_output[FIXTURE_TOKENS * DS4_Q4E_GRAPH_WIDE];
    float attention_mixed[FIXTURE_TOKENS * DS4_Q4E_GRAPH_LAYERS * DS4_Q4E_GRAPH_HIDDEN];
    float attention_output[FIXTURE_TOKENS * DS4_Q4E_GRAPH_LAYERS * DS4_Q4E_GRAPH_HIDDEN];
    float router_logits[FIXTURE_TOKENS * DS4_Q4E_GRAPH_LAYERS * DS4_Q4E_GRAPH_EXPERTS];
    uint32_t route_ids[FIXTURE_TOKENS * DS4_Q4E_GRAPH_LAYERS * DS4_Q4E_GRAPH_EXPERTS_USED];
    float route_weights[FIXTURE_TOKENS * DS4_Q4E_GRAPH_LAYERS * DS4_Q4E_GRAPH_EXPERTS_USED];
    float moe_output[FIXTURE_TOKENS * DS4_Q4E_GRAPH_LAYERS * DS4_Q4E_GRAPH_HIDDEN];
    float post_layer_wide[FIXTURE_TOKENS * DS4_Q4E_GRAPH_LAYERS * DS4_Q4E_GRAPH_WIDE];
    float final_hidden[FIXTURE_TOKENS * DS4_Q4E_GRAPH_HIDDEN];
    float final_logits[FIXTURE_TOKENS * DS4_Q4E_GRAPH_VOCAB];
    float state_wide[DS4_Q4E_GRAPH_WIDE];
    float state_activation[DS4_Q4E_GRAPH_HIDDEN];
    float state_gdn_conv[DS4_Q4E_GRAPH_GDN_LAYERS *
                         DS4_Q4E_GRAPH_GDN_CONV_CHANNELS *
                         (DS4_Q4E_GRAPH_GDN_CONV_KERNEL - 1)];
    float state_gdn_recurrent[DS4_Q4E_GRAPH_GDN_LAYERS *
                              DS4_Q4E_GRAPH_GDN_VALUE_HEADS *
                              DS4_Q4E_GRAPH_GDN_KEY_DIM *
                              DS4_Q4E_GRAPH_GDN_VALUE_DIM];
    float state_qsa_key[DS4_Q4E_GRAPH_CONTEXT * DS4_Q4E_GRAPH_QSA_HEAD_DIM];
    float state_qsa_value[DS4_Q4E_GRAPH_CONTEXT * DS4_Q4E_GRAPH_QSA_HEAD_DIM];
    float state_qsa_raw_index[DS4_Q4E_GRAPH_CONTEXT * DS4_Q4E_GRAPH_QSA_INDEX_DIM];
    uint32_t state_qsa_position[DS4_Q4E_GRAPH_CONTEXT];
    uint32_t state_ple_history[3];
    float state_ple_conv[DS4_Q4E_GRAPH_WIDE * DS4_Q4E_GRAPH_PLE_CONV_STATE];
} test_backend;

static int failures = 0;

#define CHECK(expression)                                                     \
    do {                                                                      \
        if (!(expression)) {                                                  \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,          \
                    #expression);                                             \
            ++failures;                                                       \
        }                                                                     \
    } while (0)

static uint64_t test_splitmix64(uint64_t value) {
    value += UINT64_C(0x9E3779B97F4A7C15);
    value = (value ^ (value >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94D049BB133111EB);
    return value ^ (value >> 31);
}

static float test_weight(uint64_t role, size_t index) {
    uint64_t mixed = UINT64_C(0x4E585134) ^
        role * UINT64_C(0xD6E8FEB86659FD93);
    int32_t signed_value;
    mixed ^= (uint64_t)index * UINT64_C(0xA0761D6478BD642F);
    mixed = test_splitmix64(mixed);
    signed_value = (int32_t)((mixed >> 40) & UINT64_C(0xFFFF)) - 32768;
    return (float)signed_value * (0.12f / 32768.0f);
}

static void test_vector(float *output, uint64_t role, size_t count) {
    size_t i;
    for (i = 0u; i < count; ++i) output[i] = test_weight(role, i);
}

static void test_dense(
        float *output, const float *input, size_t input_count,
        size_t output_count, uint64_t role) {
    size_t row;
    for (row = 0u; row < output_count; ++row) {
        float sum = 0.0f;
        size_t column;
        for (column = 0u; column < input_count; ++column) {
            sum += test_weight(role, row * input_count + column) * input[column];
        }
        output[row] = sum;
    }
}

static float test_sigmoid(float value) {
    if (value >= 0.0f) return 1.0f / (1.0f + expf(-value));
    {
        const float exponential = expf(value);
        return exponential / (1.0f + exponential);
    }
}

static float test_silu(float value) {
    return value * test_sigmoid(value);
}

static bool test_zcrms_full(
        float *output, const float *input, uint64_t role,
        size_t n_vector, size_t dim) {
    size_t vector;
    for (vector = 0u; vector < n_vector; ++vector) {
        float weights[DS4_Q4E_GRAPH_WIDE];
        size_t i;
        for (i = 0u; i < dim; ++i) {
            weights[i] = test_weight(role, vector * dim + i);
        }
        if (!ds4_qwen4exp_ref_zero_centered_rmsnorm_f32(
                output + vector * dim, input + vector * dim, weights,
                1u, dim, 1.0e-6f)) {
            return false;
        }
    }
    return true;
}

static bool test_zcrms_shared(
        float *output, const float *input, uint64_t role,
        size_t n_vector, size_t dim) {
    float weights[DS4_Q4E_GRAPH_WIDE];
    test_vector(weights, role, dim);
    return ds4_qwen4exp_ref_zero_centered_rmsnorm_f32(
        output, input, weights, n_vector, dim, 1.0e-6f);
}

static void test_gr_roles(
        size_t layer, size_t phase,
        uint64_t *norm, uint64_t *down, uint64_t *up, uint64_t *inject) {
    const uint64_t base = UINT64_C(1000) + (uint64_t)layer * 100u +
                          (uint64_t)phase * 10u;
    *norm = base;
    *down = base + 1u;
    *up = base + 2u;
    *inject = base + 3u;
}

static bool test_gr_prepare(
        ds4_qwen4exp_graph_stage_io *io, size_t phase) {
    float norm_weight[DS4_Q4E_GRAPH_WIDE];
    float down[DS4_Q4E_GRAPH_GR_RANK * DS4_Q4E_GRAPH_WIDE];
    float up[DS4_Q4E_GRAPH_WIDE * DS4_Q4E_GRAPH_GR_RANK];
    float inject[DS4_Q4E_GRAPH_STREAMS * DS4_Q4E_GRAPH_WIDE];
    uint64_t norm_role;
    uint64_t down_role;
    uint64_t up_role;
    uint64_t inject_role;
    test_gr_roles(io->layer, phase, &norm_role, &down_role, &up_role,
                  &inject_role);
    test_vector(norm_weight, norm_role, DS4_Q4E_GRAPH_WIDE);
    test_vector(down, down_role,
                DS4_Q4E_GRAPH_GR_RANK * DS4_Q4E_GRAPH_WIDE);
    test_vector(up, up_role,
                DS4_Q4E_GRAPH_WIDE * DS4_Q4E_GRAPH_GR_RANK);
    test_vector(inject, inject_role,
                DS4_Q4E_GRAPH_STREAMS * DS4_Q4E_GRAPH_WIDE);
    return ds4_qwen4exp_ref_gr_prepare_f32(
        io->activation, io->injection, io->wide, norm_weight, down, up,
        inject, DS4_Q4E_GRAPH_STREAMS, DS4_Q4E_GRAPH_HIDDEN,
        DS4_Q4E_GRAPH_GR_RANK, 1.0e-6f);
}

static bool test_gr_apply(ds4_qwen4exp_graph_stage_io *io) {
    return ds4_qwen4exp_ref_gr_apply_f32(
        io->wide, io->block_output, io->injection,
        DS4_Q4E_GRAPH_STREAMS, DS4_Q4E_GRAPH_HIDDEN);
}

static size_t test_layer_hidden_index(const ds4_qwen4exp_graph_stage_io *io) {
    return (io->position * DS4_Q4E_GRAPH_LAYERS + io->layer) *
           DS4_Q4E_GRAPH_HIDDEN;
}

static size_t test_layer_wide_index(const ds4_qwen4exp_graph_stage_io *io) {
    return (io->position * DS4_Q4E_GRAPH_LAYERS + io->layer) *
           DS4_Q4E_GRAPH_WIDE;
}

static bool test_ple(ds4_qwen4exp_graph_stage_io *io, test_backend *backend) {
    float embedding[DS4_Q4E_GRAPH_PLE_HEADS];
    float key_raw[DS4_Q4E_GRAPH_WIDE];
    float key[DS4_Q4E_GRAPH_WIDE];
    float value[DS4_Q4E_GRAPH_HIDDEN];
    float query[DS4_Q4E_GRAPH_WIDE];
    float gated[DS4_Q4E_GRAPH_WIDE];
    float normalized[DS4_Q4E_GRAPH_WIDE];
    float conv[DS4_Q4E_GRAPH_WIDE];
    float conv_weight[DS4_Q4E_GRAPH_WIDE * DS4_Q4E_GRAPH_PLE_CONV_KERNEL];
    size_t i;
    for (i = 0u; i < DS4_Q4E_GRAPH_PLE_HEADS; ++i) {
        embedding[i] = test_weight(2000u, io->ple_row[i]);
    }
    test_dense(key_raw, embedding, DS4_Q4E_GRAPH_PLE_HEADS,
               DS4_Q4E_GRAPH_WIDE, 2001u);
    test_dense(value, embedding, DS4_Q4E_GRAPH_PLE_HEADS,
               DS4_Q4E_GRAPH_HIDDEN, 2002u);
    if (!test_zcrms_full(key, key_raw, 2003u,
                         DS4_Q4E_GRAPH_STREAMS, DS4_Q4E_GRAPH_HIDDEN) ||
        !test_zcrms_full(query, io->wide, 2004u,
                         DS4_Q4E_GRAPH_STREAMS, DS4_Q4E_GRAPH_HIDDEN) ||
        !ds4_qwen4exp_ref_ple_gate_f32(
            gated, query, key, value,
            DS4_Q4E_GRAPH_STREAMS, DS4_Q4E_GRAPH_HIDDEN) ||
        !test_zcrms_full(normalized, gated, 2005u,
                         DS4_Q4E_GRAPH_STREAMS, DS4_Q4E_GRAPH_HIDDEN)) {
        return false;
    }
    test_vector(conv_weight, 2006u,
                DS4_Q4E_GRAPH_WIDE * DS4_Q4E_GRAPH_PLE_CONV_KERNEL);
    if (!ds4_qwen4exp_ref_dilated_conv1d_silu_f32(
            conv, io->ple_conv, normalized, conv_weight, 1u,
            DS4_Q4E_GRAPH_WIDE, DS4_Q4E_GRAPH_PLE_CONV_KERNEL,
            DS4_Q4E_GRAPH_PLE_CONV_DILATION)) {
        return false;
    }
    for (i = 0u; i < DS4_Q4E_GRAPH_WIDE; ++i) {
        const float output = gated[i] + conv[i];
        io->wide[i] += output;
        if (backend->capture && io->position < FIXTURE_TOKENS) {
            backend->ple_output[io->position * DS4_Q4E_GRAPH_WIDE + i] =
                output;
        }
    }
    if (backend->capture && io->position < FIXTURE_TOKENS) {
        memcpy(backend->ple_rows +
                   io->position * DS4_Q4E_GRAPH_PLE_HEADS,
               io->ple_row,
               DS4_Q4E_GRAPH_PLE_HEADS * sizeof(io->ple_row[0]));
    }
    if (io->token_id == DS4_Q4E_GRAPH_PLE_PAD_TOKEN) {
        io->ple_history[0] = DS4_Q4E_GRAPH_PLE_PAD_TOKEN;
        io->ple_history[1] = DS4_Q4E_GRAPH_PLE_PAD_TOKEN;
        *io->ple_history_count = 0u;
    } else {
        const uint32_t previous = io->ple_history[0];
        const size_t previous_count = *io->ple_history_count;
        io->ple_history[0] = io->token_id;
        io->ple_history[1] = previous_count > 0u
            ? previous : DS4_Q4E_GRAPH_PLE_PAD_TOKEN;
        *io->ple_history_count = previous_count < 2u
            ? previous_count + 1u : 2u;
    }
    return true;
}

static bool test_gdn(ds4_qwen4exp_graph_stage_io *io) {
    const uint64_t base = UINT64_C(3000) + (uint64_t)io->layer * 20u;
    float qkv[DS4_Q4E_GRAPH_GDN_CONV_CHANNELS];
    float convolved[DS4_Q4E_GRAPH_GDN_CONV_CHANNELS];
    float z[DS4_Q4E_GRAPH_GDN_VALUE_HEADS * DS4_Q4E_GRAPH_GDN_VALUE_DIM];
    float alpha[DS4_Q4E_GRAPH_GDN_VALUE_HEADS];
    float beta_logit[DS4_Q4E_GRAPH_GDN_VALUE_HEADS];
    float log_decay[DS4_Q4E_GRAPH_GDN_VALUE_HEADS];
    float beta[DS4_Q4E_GRAPH_GDN_VALUE_HEADS];
    float a_log[DS4_Q4E_GRAPH_GDN_VALUE_HEADS];
    float dt_bias[DS4_Q4E_GRAPH_GDN_VALUE_HEADS];
    float conv_weight[DS4_Q4E_GRAPH_GDN_CONV_CHANNELS *
                      DS4_Q4E_GRAPH_GDN_CONV_KERNEL];
    float recurrent_output[DS4_Q4E_GRAPH_GDN_VALUE_HEADS *
                           DS4_Q4E_GRAPH_GDN_VALUE_DIM];
    float gated[DS4_Q4E_GRAPH_GDN_VALUE_HEADS *
                DS4_Q4E_GRAPH_GDN_VALUE_DIM];
    float norm_weight[DS4_Q4E_GRAPH_GDN_VALUE_DIM];
    size_t i;
    test_dense(qkv, io->activation, DS4_Q4E_GRAPH_HIDDEN,
               DS4_Q4E_GRAPH_GDN_CONV_CHANNELS, base);
    test_dense(z, io->activation, DS4_Q4E_GRAPH_HIDDEN,
               DS4_Q4E_GRAPH_GDN_VALUE_HEADS *
               DS4_Q4E_GRAPH_GDN_VALUE_DIM, base + 1u);
    test_dense(beta_logit, io->activation, DS4_Q4E_GRAPH_HIDDEN,
               DS4_Q4E_GRAPH_GDN_VALUE_HEADS, base + 2u);
    test_dense(alpha, io->activation, DS4_Q4E_GRAPH_HIDDEN,
               DS4_Q4E_GRAPH_GDN_VALUE_HEADS, base + 3u);
    test_vector(conv_weight, base + 4u,
                DS4_Q4E_GRAPH_GDN_CONV_CHANNELS *
                DS4_Q4E_GRAPH_GDN_CONV_KERNEL);
    if (!ds4_qwen4exp_ref_causal_conv1d_silu_f32(
            convolved, io->gdn_conv, qkv, conv_weight, 1u,
            DS4_Q4E_GRAPH_GDN_CONV_CHANNELS,
            DS4_Q4E_GRAPH_GDN_CONV_KERNEL)) {
        return false;
    }
    test_vector(a_log, base + 5u, DS4_Q4E_GRAPH_GDN_VALUE_HEADS);
    test_vector(dt_bias, base + 6u, DS4_Q4E_GRAPH_GDN_VALUE_HEADS);
    if (!ds4_qwen4exp_ref_gdn_controls_f32(
            log_decay, beta, alpha, beta_logit, a_log, dt_bias,
            1u, DS4_Q4E_GRAPH_GDN_VALUE_HEADS) ||
        !ds4_qwen4exp_ref_gdn_f32(
            recurrent_output, io->gdn_recurrent,
            convolved, convolved + 2u, convolved + 4u,
            log_decay, beta, 1u,
            DS4_Q4E_GRAPH_GDN_KEY_HEADS,
            DS4_Q4E_GRAPH_GDN_VALUE_HEADS,
            DS4_Q4E_GRAPH_GDN_KEY_DIM,
            DS4_Q4E_GRAPH_GDN_VALUE_DIM)) {
        return false;
    }
    for (i = 0u; i < DS4_Q4E_GRAPH_GDN_VALUE_DIM; ++i) {
        norm_weight[i] = 1.0f + test_weight(base + 7u, i);
    }
    if (!ds4_qwen4exp_ref_sigmoid_gated_rmsnorm_f32(
            gated, recurrent_output, z, norm_weight,
            DS4_Q4E_GRAPH_GDN_VALUE_HEADS,
            DS4_Q4E_GRAPH_GDN_VALUE_DIM, 1.0e-6f)) {
        return false;
    }
    test_dense(io->block_output, gated,
               DS4_Q4E_GRAPH_GDN_VALUE_HEADS *
               DS4_Q4E_GRAPH_GDN_VALUE_DIM,
               DS4_Q4E_GRAPH_HIDDEN, base + 8u);
    return true;
}

static void test_rope_one(float values[2], size_t position) {
    const float cosine = cosf((float)position);
    const float sine = sinf((float)position);
    const float first = values[0];
    const float second = values[1];
    values[0] = first * cosine - second * sine;
    values[1] = second * cosine + first * sine;
}

static bool test_qsa(ds4_qwen4exp_graph_stage_io *io) {
    const uint64_t base = UINT64_C(4000);
    float query_raw[DS4_Q4E_GRAPH_QSA_QUERY_HEADS *
                    DS4_Q4E_GRAPH_QSA_HEAD_DIM];
    float query[DS4_Q4E_GRAPH_QSA_QUERY_HEADS *
                DS4_Q4E_GRAPH_QSA_HEAD_DIM];
    float gate[DS4_Q4E_GRAPH_QSA_QUERY_HEADS *
               DS4_Q4E_GRAPH_QSA_HEAD_DIM];
    float key_raw[DS4_Q4E_GRAPH_QSA_HEAD_DIM];
    float key[DS4_Q4E_GRAPH_QSA_HEAD_DIM];
    float value[DS4_Q4E_GRAPH_QSA_HEAD_DIM];
    float index_query_raw[DS4_Q4E_GRAPH_QSA_INDEX_HEADS *
                          DS4_Q4E_GRAPH_QSA_INDEX_DIM];
    float index_query[DS4_Q4E_GRAPH_QSA_INDEX_HEADS *
                      DS4_Q4E_GRAPH_QSA_INDEX_DIM];
    float raw_index[DS4_Q4E_GRAPH_QSA_INDEX_DIM];
    float group_key[(DS4_Q4E_GRAPH_CONTEXT / DS4_Q4E_GRAPH_QSA_COMPRESSION) *
                    DS4_Q4E_GRAPH_QSA_INDEX_DIM];
    float score[DS4_Q4E_GRAPH_CONTEXT / DS4_Q4E_GRAPH_QSA_COMPRESSION];
    uint32_t selected[DS4_Q4E_GRAPH_CONTEXT];
    size_t selected_count = 0u;
    const size_t complete = (io->position + 1u) /
                            DS4_Q4E_GRAPH_QSA_COMPRESSION;
    float core[DS4_Q4E_GRAPH_QSA_QUERY_HEADS *
               DS4_Q4E_GRAPH_QSA_HEAD_DIM];
    float gated[DS4_Q4E_GRAPH_QSA_QUERY_HEADS *
                DS4_Q4E_GRAPH_QSA_HEAD_DIM];
    size_t head;
    test_dense(query_raw, io->activation, DS4_Q4E_GRAPH_HIDDEN,
               DS4_Q4E_GRAPH_QSA_QUERY_HEADS *
               DS4_Q4E_GRAPH_QSA_HEAD_DIM, base);
    test_dense(gate, io->activation, DS4_Q4E_GRAPH_HIDDEN,
               DS4_Q4E_GRAPH_QSA_QUERY_HEADS *
               DS4_Q4E_GRAPH_QSA_HEAD_DIM, base + 3u);
    test_dense(key_raw, io->activation, DS4_Q4E_GRAPH_HIDDEN,
               DS4_Q4E_GRAPH_QSA_HEAD_DIM, base + 1u);
    test_dense(value, io->activation, DS4_Q4E_GRAPH_HIDDEN,
               DS4_Q4E_GRAPH_QSA_HEAD_DIM, base + 2u);
    if (!test_zcrms_shared(query, query_raw, base + 10u,
                            DS4_Q4E_GRAPH_QSA_QUERY_HEADS,
                            DS4_Q4E_GRAPH_QSA_HEAD_DIM) ||
        !test_zcrms_shared(key, key_raw, base + 11u, 1u,
                            DS4_Q4E_GRAPH_QSA_HEAD_DIM)) {
        return false;
    }
    for (head = 0u; head < DS4_Q4E_GRAPH_QSA_QUERY_HEADS; ++head) {
        test_rope_one(query + head * DS4_Q4E_GRAPH_QSA_HEAD_DIM,
                      io->position);
    }
    test_rope_one(key, io->position);
    memcpy(io->qsa_key + io->position * DS4_Q4E_GRAPH_QSA_HEAD_DIM,
           key, sizeof(key));
    memcpy(io->qsa_value + io->position * DS4_Q4E_GRAPH_QSA_HEAD_DIM,
           value, sizeof(value));

    test_dense(index_query_raw, io->activation, DS4_Q4E_GRAPH_HIDDEN,
               DS4_Q4E_GRAPH_QSA_INDEX_HEADS *
               DS4_Q4E_GRAPH_QSA_INDEX_DIM, base + 4u);
    if (!test_zcrms_shared(index_query, index_query_raw, base + 6u,
                            DS4_Q4E_GRAPH_QSA_INDEX_HEADS,
                            DS4_Q4E_GRAPH_QSA_INDEX_DIM)) {
        return false;
    }
    for (head = 0u; head < DS4_Q4E_GRAPH_QSA_INDEX_HEADS; ++head) {
        test_rope_one(index_query + head * DS4_Q4E_GRAPH_QSA_INDEX_DIM,
                      io->position);
    }
    test_dense(raw_index, io->activation, DS4_Q4E_GRAPH_HIDDEN,
               DS4_Q4E_GRAPH_QSA_INDEX_DIM, base + 5u);
    memcpy(io->qsa_raw_index +
               io->position * DS4_Q4E_GRAPH_QSA_INDEX_DIM,
           raw_index, sizeof(raw_index));
    io->qsa_position[io->position] = (uint32_t)io->position;
    *io->qsa_count = io->position + 1u;

    if (complete > 0u) {
        float norm_weight[DS4_Q4E_GRAPH_QSA_INDEX_DIM];
        test_vector(norm_weight, base + 7u,
                    DS4_Q4E_GRAPH_QSA_INDEX_DIM);
        if (!ds4_qwen4exp_ref_qsa_group_keys_f32(
                group_key, io->qsa_raw_index, norm_weight, complete,
                DS4_Q4E_GRAPH_QSA_COMPRESSION,
                DS4_Q4E_GRAPH_QSA_INDEX_DIM,
                DS4_Q4E_GRAPH_QSA_INDEX_DIM, 1.0f, 1.0e-6f) ||
            !ds4_qwen4exp_ref_qsa_scores_f32(
                score, index_query, group_key, complete,
                DS4_Q4E_GRAPH_QSA_INDEX_HEADS,
                DS4_Q4E_GRAPH_QSA_INDEX_DIM)) {
            return false;
        }
    } else {
        score[0] = 0.0f;
    }
    if (!ds4_qwen4exp_ref_qsa_select_positions(
            selected, DS4_Q4E_GRAPH_CONTEXT, &selected_count, score,
            io->position + 1u, DS4_Q4E_GRAPH_QSA_COMPRESSION,
            DS4_Q4E_GRAPH_QSA_BLOCK_BUDGET)) {
        return false;
    }
    memset(core, 0, sizeof(core));
    for (head = 0u; head < DS4_Q4E_GRAPH_QSA_QUERY_HEADS; ++head) {
        float attention[DS4_Q4E_GRAPH_CONTEXT];
        float maximum = -INFINITY;
        float total = 0.0f;
        size_t item;
        for (item = 0u; item < selected_count; ++item) {
            const float *cached_key = io->qsa_key +
                selected[item] * DS4_Q4E_GRAPH_QSA_HEAD_DIM;
            float dot = 0.0f;
            size_t dim;
            for (dim = 0u; dim < DS4_Q4E_GRAPH_QSA_HEAD_DIM; ++dim) {
                dot += query[head * DS4_Q4E_GRAPH_QSA_HEAD_DIM + dim] *
                       cached_key[dim];
            }
            attention[item] = dot / sqrtf((float)DS4_Q4E_GRAPH_QSA_HEAD_DIM);
            if (attention[item] > maximum) maximum = attention[item];
        }
        for (item = 0u; item < selected_count; ++item) {
            attention[item] = expf(attention[item] - maximum);
            total += attention[item];
        }
        for (item = 0u; item < selected_count; ++item) {
            const float probability = attention[item] / total;
            const float *cached_value = io->qsa_value +
                selected[item] * DS4_Q4E_GRAPH_QSA_HEAD_DIM;
            size_t dim;
            for (dim = 0u; dim < DS4_Q4E_GRAPH_QSA_HEAD_DIM; ++dim) {
                core[head * DS4_Q4E_GRAPH_QSA_HEAD_DIM + dim] +=
                    probability * cached_value[dim];
            }
        }
    }
    for (head = 0u; head < sizeof(gated) / sizeof(gated[0]); ++head) {
        gated[head] = core[head] * test_sigmoid(gate[head]);
    }
    test_dense(io->block_output, gated,
               DS4_Q4E_GRAPH_QSA_QUERY_HEADS *
               DS4_Q4E_GRAPH_QSA_HEAD_DIM,
               DS4_Q4E_GRAPH_HIDDEN, base + 9u);
    return true;
}

static bool test_router(ds4_qwen4exp_graph_stage_io *io) {
    test_dense(io->router_logits, io->activation, DS4_Q4E_GRAPH_HIDDEN,
               DS4_Q4E_GRAPH_EXPERTS, 5000u + io->layer);
    return ds4_qwen4exp_ref_softmax_topk_f32(
        io->route_id, io->route_weight, io->router_logits,
        DS4_Q4E_GRAPH_EXPERTS, DS4_Q4E_GRAPH_EXPERTS_USED);
}

static bool test_moe(ds4_qwen4exp_graph_stage_io *io) {
    float routed[DS4_Q4E_GRAPH_HIDDEN] = {0.0f};
    float gate[DS4_Q4E_GRAPH_EXPERT_DIM];
    float up[DS4_Q4E_GRAPH_EXPERT_DIM];
    float product[DS4_Q4E_GRAPH_EXPERT_DIM];
    float down[DS4_Q4E_GRAPH_HIDDEN];
    float shared_gate[DS4_Q4E_GRAPH_EXPERT_DIM];
    float shared_up[DS4_Q4E_GRAPH_EXPERT_DIM];
    float shared[DS4_Q4E_GRAPH_HIDDEN];
    float shared_scale[1];
    size_t slot;
    size_t i;
    for (slot = 0u; slot < DS4_Q4E_GRAPH_EXPERTS_USED; ++slot) {
        const uint64_t base = UINT64_C(6000) +
            (uint64_t)io->layer * 2000u +
            (uint64_t)io->route_id[slot] * 3u;
        test_dense(gate, io->activation, DS4_Q4E_GRAPH_HIDDEN,
                   DS4_Q4E_GRAPH_EXPERT_DIM, base);
        test_dense(up, io->activation, DS4_Q4E_GRAPH_HIDDEN,
                   DS4_Q4E_GRAPH_EXPERT_DIM, base + 1u);
        for (i = 0u; i < DS4_Q4E_GRAPH_EXPERT_DIM; ++i) {
            product[i] = test_silu(gate[i]) * up[i];
        }
        test_dense(down, product, DS4_Q4E_GRAPH_EXPERT_DIM,
                   DS4_Q4E_GRAPH_HIDDEN, base + 2u);
        for (i = 0u; i < DS4_Q4E_GRAPH_HIDDEN; ++i) {
            routed[i] += io->route_weight[slot] * down[i];
        }
    }
    {
        const uint64_t base = UINT64_C(15000) +
                              (uint64_t)io->layer * 10u;
        test_dense(shared_gate, io->activation, DS4_Q4E_GRAPH_HIDDEN,
                   DS4_Q4E_GRAPH_EXPERT_DIM, base);
        test_dense(shared_up, io->activation, DS4_Q4E_GRAPH_HIDDEN,
                   DS4_Q4E_GRAPH_EXPERT_DIM, base + 1u);
        for (i = 0u; i < DS4_Q4E_GRAPH_EXPERT_DIM; ++i) {
            product[i] = test_silu(shared_gate[i]) * shared_up[i];
        }
        test_dense(shared, product, DS4_Q4E_GRAPH_EXPERT_DIM,
                   DS4_Q4E_GRAPH_HIDDEN, base + 2u);
        test_dense(shared_scale, io->activation, DS4_Q4E_GRAPH_HIDDEN,
                   1u, base + 3u);
    }
    for (i = 0u; i < DS4_Q4E_GRAPH_HIDDEN; ++i) {
        io->block_output[i] = routed[i] +
            test_sigmoid(shared_scale[0]) * shared[i];
    }
    return true;
}

static bool test_final_mix(ds4_qwen4exp_graph_stage_io *io) {
    float norm_weight[DS4_Q4E_GRAPH_WIDE];
    float down[DS4_Q4E_GRAPH_GR_RANK * DS4_Q4E_GRAPH_WIDE];
    float up[DS4_Q4E_GRAPH_WIDE * DS4_Q4E_GRAPH_GR_RANK];
    uint64_t norm_role;
    uint64_t down_role;
    uint64_t up_role;
    uint64_t inject_role;
    test_gr_roles(DS4_Q4E_GRAPH_LAYERS, 2u, &norm_role, &down_role,
                  &up_role, &inject_role);
    (void)inject_role;
    test_vector(norm_weight, norm_role, DS4_Q4E_GRAPH_WIDE);
    test_vector(down, down_role,
                DS4_Q4E_GRAPH_GR_RANK * DS4_Q4E_GRAPH_WIDE);
    test_vector(up, up_role,
                DS4_Q4E_GRAPH_WIDE * DS4_Q4E_GRAPH_GR_RANK);
    return ds4_qwen4exp_ref_gr_final_mix_f32(
        io->activation, io->wide, norm_weight, down, up,
        DS4_Q4E_GRAPH_STREAMS, DS4_Q4E_GRAPH_HIDDEN,
        DS4_Q4E_GRAPH_GR_RANK, 1.0e-6f);
}

static bool test_stage(void *context, ds4_qwen4exp_graph_stage_io *io) {
    test_backend *backend = (test_backend *)context;
    bool success = false;
    const size_t hidden_index = test_layer_hidden_index(io);
    const size_t wide_index = test_layer_wide_index(io);
    switch (io->stage) {
        case DS4_Q4E_STAGE_EMBEDDING:
            {
                size_t dim;
                for (dim = 0u; dim < DS4_Q4E_GRAPH_HIDDEN; ++dim) {
                    io->activation[dim] = test_weight(
                        100u, io->token_id * DS4_Q4E_GRAPH_HIDDEN + dim);
                }
            }
            if (backend->capture && io->position < FIXTURE_TOKENS) {
                memcpy(backend->embedding +
                           io->position * DS4_Q4E_GRAPH_HIDDEN,
                       io->activation,
                       DS4_Q4E_GRAPH_HIDDEN * sizeof(float));
            }
            success = true;
            break;
        case DS4_Q4E_STAGE_FOUR_STREAM_RESIDUAL: {
            size_t stream;
            for (stream = 0u; stream < DS4_Q4E_GRAPH_STREAMS; ++stream) {
                memcpy(io->wide + stream * DS4_Q4E_GRAPH_HIDDEN,
                       io->activation,
                       DS4_Q4E_GRAPH_HIDDEN * sizeof(float));
            }
            success = true;
            break;
        }
        case DS4_Q4E_STAGE_PLE:
            success = test_ple(io, backend);
            break;
        case DS4_Q4E_STAGE_ATTN_GR_PREPARE:
            success = test_gr_prepare(io, 0u);
            if (success && backend->capture && io->position < FIXTURE_TOKENS) {
                memcpy(backend->attention_mixed + hidden_index,
                       io->activation,
                       DS4_Q4E_GRAPH_HIDDEN * sizeof(float));
            }
            break;
        case DS4_Q4E_STAGE_GDN:
            success = test_gdn(io);
            if (success && backend->capture && io->position < FIXTURE_TOKENS) {
                const size_t conv_per_layer =
                    DS4_Q4E_GRAPH_GDN_CONV_CHANNELS *
                    (DS4_Q4E_GRAPH_GDN_CONV_KERNEL - 1);
                const size_t recurrent_per_layer =
                    DS4_Q4E_GRAPH_GDN_VALUE_HEADS *
                    DS4_Q4E_GRAPH_GDN_KEY_DIM *
                    DS4_Q4E_GRAPH_GDN_VALUE_DIM;
                memcpy(backend->attention_output + hidden_index,
                       io->block_output,
                       DS4_Q4E_GRAPH_HIDDEN * sizeof(float));
                memcpy(backend->state_gdn_conv +
                           io->gdn_layer * conv_per_layer,
                       io->gdn_conv, conv_per_layer * sizeof(float));
                memcpy(backend->state_gdn_recurrent +
                           io->gdn_layer * recurrent_per_layer,
                       io->gdn_recurrent,
                       recurrent_per_layer * sizeof(float));
            }
            break;
        case DS4_Q4E_STAGE_QSA:
            success = test_qsa(io);
            if (success && backend->capture && io->position < FIXTURE_TOKENS) {
                memcpy(backend->attention_output + hidden_index,
                       io->block_output,
                       DS4_Q4E_GRAPH_HIDDEN * sizeof(float));
            }
            break;
        case DS4_Q4E_STAGE_ATTN_GR_APPLY:
            success = test_gr_apply(io);
            break;
        case DS4_Q4E_STAGE_MOE_GR_PREPARE:
            success = test_gr_prepare(io, 1u);
            break;
        case DS4_Q4E_STAGE_ROUTER:
            success = test_router(io);
            if (success && backend->capture && io->position < FIXTURE_TOKENS) {
                memcpy(backend->router_logits +
                           (io->position * DS4_Q4E_GRAPH_LAYERS + io->layer) *
                           DS4_Q4E_GRAPH_EXPERTS,
                       io->router_logits,
                       DS4_Q4E_GRAPH_EXPERTS * sizeof(float));
                memcpy(backend->route_ids +
                           (io->position * DS4_Q4E_GRAPH_LAYERS + io->layer) *
                           DS4_Q4E_GRAPH_EXPERTS_USED,
                       io->route_id,
                       DS4_Q4E_GRAPH_EXPERTS_USED * sizeof(uint32_t));
                memcpy(backend->route_weights +
                           (io->position * DS4_Q4E_GRAPH_LAYERS + io->layer) *
                           DS4_Q4E_GRAPH_EXPERTS_USED,
                       io->route_weight,
                       DS4_Q4E_GRAPH_EXPERTS_USED * sizeof(float));
            }
            break;
        case DS4_Q4E_STAGE_ROUTED_SHARED_MOE:
            success = test_moe(io);
            if (success && backend->capture && io->position < FIXTURE_TOKENS) {
                memcpy(backend->moe_output + hidden_index,
                       io->block_output,
                       DS4_Q4E_GRAPH_HIDDEN * sizeof(float));
            }
            break;
        case DS4_Q4E_STAGE_MOE_GR_APPLY:
            success = test_gr_apply(io);
            if (success && backend->capture && io->position < FIXTURE_TOKENS) {
                memcpy(backend->post_layer_wide + wide_index,
                       io->wide, DS4_Q4E_GRAPH_WIDE * sizeof(float));
            }
            break;
        case DS4_Q4E_STAGE_FINAL_GR_MIXER:
            success = test_final_mix(io);
            if (success && backend->capture && io->position < FIXTURE_TOKENS) {
                memcpy(backend->final_hidden +
                           io->position * DS4_Q4E_GRAPH_HIDDEN,
                       io->activation,
                       DS4_Q4E_GRAPH_HIDDEN * sizeof(float));
            }
            break;
        case DS4_Q4E_STAGE_OUTPUT_HEAD:
            test_dense(io->logits, io->activation,
                       DS4_Q4E_GRAPH_HIDDEN, DS4_Q4E_GRAPH_VOCAB, 17000u);
            if (backend->capture && io->position < FIXTURE_TOKENS) {
                memcpy(backend->final_logits +
                           io->position * DS4_Q4E_GRAPH_VOCAB,
                       io->logits,
                       DS4_Q4E_GRAPH_VOCAB * sizeof(float));
                memcpy(backend->state_wide, io->wide,
                       sizeof(backend->state_wide));
                memcpy(backend->state_activation, io->activation,
                       sizeof(backend->state_activation));
                memcpy(backend->state_qsa_key, io->qsa_key,
                       sizeof(backend->state_qsa_key));
                memcpy(backend->state_qsa_value, io->qsa_value,
                       sizeof(backend->state_qsa_value));
                memcpy(backend->state_qsa_raw_index, io->qsa_raw_index,
                       sizeof(backend->state_qsa_raw_index));
                memcpy(backend->state_qsa_position, io->qsa_position,
                       sizeof(backend->state_qsa_position));
                backend->state_ple_history[0] = io->ple_history[0];
                backend->state_ple_history[1] = io->ple_history[1];
                backend->state_ple_history[2] =
                    (uint32_t)*io->ple_history_count;
                memcpy(backend->state_ple_conv, io->ple_conv,
                       sizeof(backend->state_ple_conv));
            }
            success = true;
            break;
        case DS4_Q4E_STAGE_COUNT:
            success = false;
            break;
    }
    if (!success) return false;
    if (backend->nonfinite_after == io->stage_ordinal) {
        io->wide[0] = NAN;
    }
    return backend->fail_after != io->stage_ordinal;
}

static void test_backend_init(test_backend *backend, bool capture) {
    memset(backend, 0, sizeof(*backend));
    backend->fail_after = SIZE_MAX;
    backend->nonfinite_after = SIZE_MAX;
    backend->capture = capture;
}

static ds4_qwen4exp_graph *test_graph_create(test_backend *backend) {
    ds4_qwen4exp_graph_geometry geometry;
    ds4_qwen4exp_graph *graph = NULL;
    ds4_qwen4exp_graph_backend graph_backend;
    ds4_qwen4exp_graph_geometry_frozen(&geometry);
    graph_backend.stage = test_stage;
    graph_backend.context = backend;
    CHECK(ds4_qwen4exp_graph_create(&graph, &geometry, graph_backend));
    return graph;
}

static bool test_close_array(
        const float *actual, const float *expected, size_t count,
        float atol, float rtol, const char *name) {
    size_t i;
    for (i = 0u; i < count; ++i) {
        const float difference = fabsf(actual[i] - expected[i]);
        const float limit = atol + rtol * fabsf(expected[i]);
        if (!isfinite(actual[i]) || difference > limit) {
            fprintf(stderr,
                    "FAIL %s[%zu]: actual %.9g expected %.9g diff %.9g limit %.9g\n",
                    name, i, actual[i], expected[i], difference, limit);
            ++failures;
            return false;
        }
    }
    return true;
}

#define CHECK_FLOAT_ARRAY(actual, expected)                                    \
    (void)test_close_array((actual), (expected),                              \
        sizeof(expected) / sizeof((expected)[0]),                             \
        DS4_Q4E_GOLDEN_ATOL, DS4_Q4E_GOLDEN_RTOL, #expected)

static void test_golden_and_chunks(void) {
    test_backend one_shot_backend;
    test_backend token_backend;
    test_backend chunk_backend;
    ds4_qwen4exp_graph *one_shot;
    ds4_qwen4exp_graph *token_graph;
    ds4_qwen4exp_graph *chunk_graph;
    float output[DS4_Q4E_GRAPH_VOCAB];
    float token_output[DS4_Q4E_GRAPH_VOCAB];
    float chunk_output[DS4_Q4E_GRAPH_VOCAB];
    uint64_t one_shot_digest = 0u;
    uint64_t token_digest = 0u;
    uint64_t chunk_digest = 0u;
    size_t i;
    test_backend_init(&one_shot_backend, true);
    test_backend_init(&token_backend, false);
    test_backend_init(&chunk_backend, false);
    one_shot = test_graph_create(&one_shot_backend);
    token_graph = test_graph_create(&token_backend);
    chunk_graph = test_graph_create(&chunk_backend);
    if (!one_shot || !token_graph || !chunk_graph) goto cleanup;

    CHECK(ds4_qwen4exp_graph_run(
        one_shot, ds4_q4e_golden_tokens, FIXTURE_TOKENS,
        output, DS4_Q4E_GRAPH_VOCAB));
    CHECK(ds4_qwen4exp_graph_frontier(one_shot) == FIXTURE_TOKENS);
    CHECK_FLOAT_ARRAY(one_shot_backend.embedding, ds4_q4e_golden_embedding);
    CHECK(memcmp(one_shot_backend.ple_rows, ds4_q4e_golden_ple_rows,
                 sizeof(ds4_q4e_golden_ple_rows)) == 0);
    CHECK_FLOAT_ARRAY(one_shot_backend.ple_output, ds4_q4e_golden_ple_output);
    CHECK_FLOAT_ARRAY(one_shot_backend.attention_mixed, ds4_q4e_golden_attention_mixed);
    CHECK_FLOAT_ARRAY(one_shot_backend.attention_output, ds4_q4e_golden_attention_output);
    CHECK_FLOAT_ARRAY(one_shot_backend.router_logits, ds4_q4e_golden_router_logits);
    CHECK(memcmp(one_shot_backend.route_ids, ds4_q4e_golden_route_ids,
                 sizeof(ds4_q4e_golden_route_ids)) == 0);
    CHECK_FLOAT_ARRAY(one_shot_backend.route_weights, ds4_q4e_golden_route_weights);
    CHECK_FLOAT_ARRAY(one_shot_backend.moe_output, ds4_q4e_golden_moe_output);
    CHECK_FLOAT_ARRAY(one_shot_backend.post_layer_wide, ds4_q4e_golden_post_layer_wide);
    CHECK_FLOAT_ARRAY(one_shot_backend.final_hidden, ds4_q4e_golden_final_hidden);
    CHECK_FLOAT_ARRAY(one_shot_backend.final_logits, ds4_q4e_golden_final_logits);
    CHECK_FLOAT_ARRAY(one_shot_backend.state_wide, ds4_q4e_golden_final_wide);
    CHECK_FLOAT_ARRAY(one_shot_backend.state_activation, ds4_q4e_golden_final_activation);
    CHECK_FLOAT_ARRAY(one_shot_backend.state_gdn_conv, ds4_q4e_golden_final_gdn_conv);
    CHECK_FLOAT_ARRAY(one_shot_backend.state_gdn_recurrent, ds4_q4e_golden_final_gdn_recurrent);
    CHECK_FLOAT_ARRAY(one_shot_backend.state_qsa_key, ds4_q4e_golden_final_qsa_key);
    CHECK_FLOAT_ARRAY(one_shot_backend.state_qsa_value, ds4_q4e_golden_final_qsa_value);
    CHECK_FLOAT_ARRAY(one_shot_backend.state_qsa_raw_index, ds4_q4e_golden_final_qsa_raw_index);
    CHECK(memcmp(one_shot_backend.state_qsa_position,
                 ds4_q4e_golden_final_qsa_position,
                 sizeof(ds4_q4e_golden_final_qsa_position)) == 0);
    CHECK(memcmp(one_shot_backend.state_ple_history,
                 ds4_q4e_golden_final_ple_history,
                 sizeof(ds4_q4e_golden_final_ple_history)) == 0);
    CHECK_FLOAT_ARRAY(one_shot_backend.state_ple_conv, ds4_q4e_golden_final_ple_conv);
    (void)test_close_array(output,
        ds4_q4e_golden_final_logits +
            (FIXTURE_TOKENS - 1u) * DS4_Q4E_GRAPH_VOCAB,
        DS4_Q4E_GRAPH_VOCAB, DS4_Q4E_GOLDEN_ATOL,
        DS4_Q4E_GOLDEN_RTOL, "one-shot output");

    for (i = 0u; i < FIXTURE_TOKENS; ++i) {
        CHECK(ds4_qwen4exp_graph_run(
            token_graph, ds4_q4e_golden_tokens + i, 1u,
            token_output, DS4_Q4E_GRAPH_VOCAB));
        (void)test_close_array(token_output,
            ds4_q4e_golden_final_logits + i * DS4_Q4E_GRAPH_VOCAB,
            DS4_Q4E_GRAPH_VOCAB, DS4_Q4E_GOLDEN_ATOL,
            DS4_Q4E_GOLDEN_RTOL, "token output");
    }
    CHECK(ds4_qwen4exp_graph_run(
        chunk_graph, ds4_q4e_golden_tokens, 2u,
        chunk_output, DS4_Q4E_GRAPH_VOCAB));
    CHECK(ds4_qwen4exp_graph_run(
        chunk_graph, ds4_q4e_golden_tokens + 2u, 1u,
        chunk_output, DS4_Q4E_GRAPH_VOCAB));
    CHECK(ds4_qwen4exp_graph_run(
        chunk_graph, ds4_q4e_golden_tokens + 3u, 3u,
        chunk_output, DS4_Q4E_GRAPH_VOCAB));
    (void)test_close_array(chunk_output,
        ds4_q4e_golden_final_logits +
            (FIXTURE_TOKENS - 1u) * DS4_Q4E_GRAPH_VOCAB,
        DS4_Q4E_GRAPH_VOCAB, DS4_Q4E_GOLDEN_ATOL,
        DS4_Q4E_GOLDEN_RTOL, "chunk output");
    CHECK(ds4_qwen4exp_graph_public_digest(one_shot, &one_shot_digest));
    CHECK(ds4_qwen4exp_graph_public_digest(token_graph, &token_digest));
    CHECK(ds4_qwen4exp_graph_public_digest(chunk_graph, &chunk_digest));
    CHECK(one_shot_digest == token_digest);
    CHECK(one_shot_digest == chunk_digest);

cleanup:
    ds4_qwen4exp_graph_destroy(one_shot);
    ds4_qwen4exp_graph_destroy(token_graph);
    ds4_qwen4exp_graph_destroy(chunk_graph);
}

static void test_ple_eos_ubatch_transitions(void) {
    const uint32_t sequence[] = {
        1u, 2u, DS4_Q4E_GRAPH_PLE_PAD_TOKEN, 3u,
    };
    const uint32_t eos[] = {DS4_Q4E_GRAPH_PLE_PAD_TOKEN};
    const uint32_t after_eos[] = {3u};
    test_backend one_shot_backend;
    test_backend chunk_backend;
    test_backend eos_backend;
    test_backend after_eos_backend;
    ds4_qwen4exp_graph *one_shot;
    ds4_qwen4exp_graph *chunk;
    ds4_qwen4exp_graph *fresh_eos;
    ds4_qwen4exp_graph *fresh_after_eos;
    float one_shot_output[DS4_Q4E_GRAPH_VOCAB];
    float chunk_output[DS4_Q4E_GRAPH_VOCAB];
    float scratch_output[DS4_Q4E_GRAPH_VOCAB];
    uint64_t one_shot_digest = 0u;
    uint64_t chunk_digest = 0u;
    size_t i;
    bool nonzero_conv = false;
    test_backend_init(&one_shot_backend, true);
    test_backend_init(&chunk_backend, true);
    test_backend_init(&eos_backend, true);
    test_backend_init(&after_eos_backend, true);
    one_shot = test_graph_create(&one_shot_backend);
    chunk = test_graph_create(&chunk_backend);
    fresh_eos = test_graph_create(&eos_backend);
    fresh_after_eos = test_graph_create(&after_eos_backend);
    if (!one_shot || !chunk || !fresh_eos || !fresh_after_eos) goto cleanup;

    CHECK(ds4_qwen4exp_graph_run(
        one_shot, sequence, sizeof(sequence) / sizeof(sequence[0]),
        one_shot_output, DS4_Q4E_GRAPH_VOCAB));
    CHECK(ds4_qwen4exp_graph_run(
        chunk, sequence, 1u, chunk_output, DS4_Q4E_GRAPH_VOCAB));
    CHECK(ds4_qwen4exp_graph_run(
        chunk, sequence + 1u, 2u, chunk_output, DS4_Q4E_GRAPH_VOCAB));
    CHECK(ds4_qwen4exp_graph_run(
        chunk, sequence + 3u, 1u, chunk_output, DS4_Q4E_GRAPH_VOCAB));
    CHECK(ds4_qwen4exp_graph_run(
        fresh_eos, eos, 1u, scratch_output, DS4_Q4E_GRAPH_VOCAB));
    CHECK(ds4_qwen4exp_graph_run(
        fresh_after_eos, after_eos, 1u, scratch_output,
        DS4_Q4E_GRAPH_VOCAB));

    CHECK(memcmp(one_shot_backend.ple_rows +
                     2u * DS4_Q4E_GRAPH_PLE_HEADS,
                 eos_backend.ple_rows,
                 DS4_Q4E_GRAPH_PLE_HEADS * sizeof(uint32_t)) != 0);
    CHECK(memcmp(one_shot_backend.ple_rows +
                     3u * DS4_Q4E_GRAPH_PLE_HEADS,
                 after_eos_backend.ple_rows,
                 DS4_Q4E_GRAPH_PLE_HEADS * sizeof(uint32_t)) == 0);
    CHECK(memcmp(one_shot_backend.ple_rows, chunk_backend.ple_rows,
                 4u * DS4_Q4E_GRAPH_PLE_HEADS * sizeof(uint32_t)) == 0);
    CHECK(memcmp(one_shot_backend.state_ple_conv,
                 chunk_backend.state_ple_conv,
                 sizeof(one_shot_backend.state_ple_conv)) == 0);
    for (i = 0u; i < DS4_Q4E_GRAPH_WIDE *
                         DS4_Q4E_GRAPH_PLE_CONV_STATE; ++i) {
        if (one_shot_backend.state_ple_conv[i] != 0.0f) nonzero_conv = true;
    }
    CHECK(nonzero_conv);
    CHECK(memcmp(one_shot_output, chunk_output,
                 sizeof(one_shot_output)) == 0);
    CHECK(ds4_qwen4exp_graph_public_digest(one_shot, &one_shot_digest));
    CHECK(ds4_qwen4exp_graph_public_digest(chunk, &chunk_digest));
    CHECK(one_shot_digest == chunk_digest);

cleanup:
    ds4_qwen4exp_graph_destroy(one_shot);
    ds4_qwen4exp_graph_destroy(chunk);
    ds4_qwen4exp_graph_destroy(fresh_eos);
    ds4_qwen4exp_graph_destroy(fresh_after_eos);
}

static void test_plan_reset_and_rejections(void) {
    ds4_qwen4exp_graph_geometry geometry;
    ds4_qwen4exp_graph_geometry invalid;
    ds4_qwen4exp_graph_plan plan;
    ds4_qwen4exp_graph_plan untouched;
    test_backend backend;
    ds4_qwen4exp_graph *graph;
    float output[DS4_Q4E_GRAPH_VOCAB];
    float public_logits[DS4_Q4E_GRAPH_VOCAB];
    uint64_t reset_digest = 0u;
    uint64_t second_reset_digest = 0u;
    union {
        uint32_t token[DS4_Q4E_GRAPH_VOCAB];
        float logits[DS4_Q4E_GRAPH_VOCAB];
    } aliased;
    size_t i;
    ds4_qwen4exp_graph_geometry_frozen(&geometry);
    CHECK(ds4_qwen4exp_graph_geometry_validate(&geometry));
    CHECK(ds4_qwen4exp_graph_plan_make(&geometry, &plan));
    CHECK(plan.wide_residual_bytes == 16u * sizeof(float));
    CHECK(plan.private_activation_bytes == 12u * sizeof(float));
    CHECK(plan.gdn_conv_bytes == 3u * 10u * 3u * sizeof(float));
    CHECK(DS4_Q4E_GRAPH_GDN_CONV_CHANNELS ==
          2u * DS4_Q4E_GRAPH_GDN_KEY_HEADS *
              DS4_Q4E_GRAPH_GDN_KEY_DIM +
          DS4_Q4E_GRAPH_GDN_VALUE_HEADS *
              DS4_Q4E_GRAPH_GDN_VALUE_DIM);
    CHECK(plan.gdn_recurrent_bytes == 3u * 3u * 2u * 2u * sizeof(float));
    CHECK(plan.qsa_kv_bytes == 8u * 2u * 2u * sizeof(float));
    CHECK(plan.qsa_raw_index_bytes == 8u * 2u * sizeof(float));
    CHECK(plan.qsa_position_bytes == 8u * sizeof(uint32_t));
    CHECK(plan.ple_conv_bytes == 16u * 9u * sizeof(float));
    CHECK(plan.route_bytes ==
          512u * sizeof(float) + 10u * sizeof(uint32_t) +
          10u * sizeof(float));
    CHECK(plan.logits_bytes == 13u * sizeof(float));
    CHECK(plan.bank_bytes > plan.wide_residual_bytes);
    CHECK(plan.allocated_bytes == sizeof(ds4_qwen4exp_graph) +
                                  2u * plan.bank_bytes);

    memset(&untouched, 0xA5, sizeof(untouched));
    invalid = geometry;
    invalid.streams = SIZE_MAX;
    CHECK(!ds4_qwen4exp_graph_plan_make(&invalid, &untouched));
    {
        const unsigned char *bytes = (const unsigned char *)&untouched;
        for (i = 0u; i < sizeof(untouched); ++i) CHECK(bytes[i] == 0xA5u);
    }
    invalid = geometry;
    invalid.context = DS4_Q4E_GRAPH_DENSE_QSA_LIMIT + 1u;
    CHECK(!ds4_qwen4exp_graph_plan_make(&invalid, &untouched));
    /* Phase 5 is deliberately dense-QSA-only.  Exercise the eventual long
     * context frontier explicitly so 262K cannot silently fall through this
     * resident scaffold before the sparse Phase-6 cache is qualified. */
    invalid = geometry;
    invalid.context = 262143u;
    CHECK(!ds4_qwen4exp_graph_plan_make(&invalid, &untouched));
    invalid.context = 262144u;
    CHECK(!ds4_qwen4exp_graph_plan_make(&invalid, &untouched));
    invalid.context = 262145u;
    CHECK(!ds4_qwen4exp_graph_plan_make(&invalid, &untouched));

    test_backend_init(&backend, false);
    graph = test_graph_create(&backend);
    if (!graph) return;
    CHECK(ds4_qwen4exp_graph_frontier(graph) == 0u);
    CHECK(ds4_qwen4exp_graph_public_digest(graph, &reset_digest));
    CHECK(!ds4_qwen4exp_graph_public_logits(
        graph, (float *)graph, DS4_Q4E_GRAPH_VOCAB));
    CHECK(!ds4_qwen4exp_graph_public_logits(
        graph, (float *)graph->bank[0], DS4_Q4E_GRAPH_VOCAB));
    CHECK(!ds4_qwen4exp_graph_public_digest(graph, (uint64_t *)graph));
    CHECK(!ds4_qwen4exp_graph_public_digest(
        graph, (uint64_t *)graph->bank[1]));
    CHECK(ds4_qwen4exp_graph_public_digest(graph, &second_reset_digest));
    CHECK(reset_digest == second_reset_digest);
    for (i = 0u; i < DS4_Q4E_GRAPH_VOCAB; ++i) output[i] = 77.0f;
    CHECK(!ds4_qwen4exp_graph_run(graph, ds4_q4e_golden_tokens, 0u,
                                  output, DS4_Q4E_GRAPH_VOCAB));
    CHECK(!ds4_qwen4exp_graph_run(graph, ds4_q4e_golden_tokens, SIZE_MAX,
                                  output, DS4_Q4E_GRAPH_VOCAB));
    CHECK(!ds4_qwen4exp_graph_run(graph, ds4_q4e_golden_tokens, 1u,
                                  output, DS4_Q4E_GRAPH_VOCAB - 1u));
    aliased.token[0] = 1u;
    CHECK(!ds4_qwen4exp_graph_run(graph, aliased.token, 1u,
                                  aliased.logits, DS4_Q4E_GRAPH_VOCAB));
    {
        const uint32_t invalid_token = DS4_Q4E_GRAPH_VOCAB;
        CHECK(!ds4_qwen4exp_graph_run(graph, &invalid_token, 1u,
                                      output, DS4_Q4E_GRAPH_VOCAB));
    }
    CHECK(ds4_qwen4exp_graph_run(graph, ds4_q4e_golden_tokens,
                                  FIXTURE_TOKENS, output,
                                  DS4_Q4E_GRAPH_VOCAB));
    CHECK(!ds4_qwen4exp_graph_run(graph, ds4_q4e_golden_tokens, 3u,
                                   output, DS4_Q4E_GRAPH_VOCAB));
    CHECK(ds4_qwen4exp_graph_reset(graph));
    CHECK(ds4_qwen4exp_graph_frontier(graph) == 0u);
    CHECK(ds4_qwen4exp_graph_public_logits(
        graph, public_logits, DS4_Q4E_GRAPH_VOCAB));
    for (i = 0u; i < DS4_Q4E_GRAPH_VOCAB; ++i) CHECK(public_logits[i] == 0.0f);
    CHECK(ds4_qwen4exp_graph_public_digest(graph, &second_reset_digest));
    CHECK(reset_digest == second_reset_digest);
    CHECK(ds4_qwen4exp_graph_byte_report(graph) != NULL);
    ds4_qwen4exp_graph_destroy(graph);
}

static void test_failure_transactions(void) {
    const size_t stages = ds4_qwen4exp_graph_stages_per_token();
    size_t failed_stage;
    CHECK(stages == 33u);
    for (failed_stage = 0u; failed_stage < stages; ++failed_stage) {
        test_backend target_backend;
        test_backend fresh_backend;
        ds4_qwen4exp_graph *target;
        ds4_qwen4exp_graph *fresh;
        float prefix_output[DS4_Q4E_GRAPH_VOCAB];
        float failed_output[DS4_Q4E_GRAPH_VOCAB];
        float target_output[DS4_Q4E_GRAPH_VOCAB];
        float fresh_output[DS4_Q4E_GRAPH_VOCAB];
        float old_logits[DS4_Q4E_GRAPH_VOCAB];
        uint64_t before = 0u;
        uint64_t after_failure = 0u;
        uint64_t target_digest = 0u;
        uint64_t fresh_digest = 0u;
        size_t i;
        test_backend_init(&target_backend, false);
        test_backend_init(&fresh_backend, false);
        target = test_graph_create(&target_backend);
        fresh = test_graph_create(&fresh_backend);
        if (!target || !fresh) {
            ds4_qwen4exp_graph_destroy(target);
            ds4_qwen4exp_graph_destroy(fresh);
            continue;
        }
        CHECK(ds4_qwen4exp_graph_run(
            target, ds4_q4e_golden_tokens, 2u, prefix_output,
            DS4_Q4E_GRAPH_VOCAB));
        CHECK(ds4_qwen4exp_graph_run(
            fresh, ds4_q4e_golden_tokens, 2u, prefix_output,
            DS4_Q4E_GRAPH_VOCAB));
        CHECK(ds4_qwen4exp_graph_public_digest(target, &before));
        CHECK(ds4_qwen4exp_graph_public_logits(
            target, old_logits, DS4_Q4E_GRAPH_VOCAB));
        for (i = 0u; i < DS4_Q4E_GRAPH_VOCAB; ++i) failed_output[i] = 123.0f;
        target_backend.fail_after = failed_stage;
        CHECK(!ds4_qwen4exp_graph_run(
            target, ds4_q4e_golden_tokens + 2u, 1u,
            failed_output, DS4_Q4E_GRAPH_VOCAB));
        CHECK(ds4_qwen4exp_graph_frontier(target) == 2u);
        CHECK(ds4_qwen4exp_graph_public_digest(target, &after_failure));
        CHECK(before == after_failure);
        for (i = 0u; i < DS4_Q4E_GRAPH_VOCAB; ++i) {
            CHECK(failed_output[i] == 123.0f);
        }
        CHECK(ds4_qwen4exp_graph_public_logits(
            target, prefix_output, DS4_Q4E_GRAPH_VOCAB));
        CHECK(memcmp(prefix_output, old_logits, sizeof(old_logits)) == 0);

        target_backend.fail_after = SIZE_MAX;
        CHECK(ds4_qwen4exp_graph_run(
            target, ds4_q4e_golden_tokens + 2u, 1u,
            target_output, DS4_Q4E_GRAPH_VOCAB));
        CHECK(ds4_qwen4exp_graph_run(
            fresh, ds4_q4e_golden_tokens + 2u, 1u,
            fresh_output, DS4_Q4E_GRAPH_VOCAB));
        CHECK(memcmp(target_output, fresh_output, sizeof(target_output)) == 0);
        CHECK(ds4_qwen4exp_graph_public_digest(target, &target_digest));
        CHECK(ds4_qwen4exp_graph_public_digest(fresh, &fresh_digest));
        CHECK(target_digest == fresh_digest);
        ds4_qwen4exp_graph_destroy(target);
        ds4_qwen4exp_graph_destroy(fresh);
    }
}

static void test_nonfinite_and_multiturn_isolation(void) {
    test_backend bad_backend;
    test_backend first_backend;
    test_backend second_backend;
    ds4_qwen4exp_graph *bad;
    ds4_qwen4exp_graph *first;
    ds4_qwen4exp_graph *second;
    float output[DS4_Q4E_GRAPH_VOCAB];
    float first_output[DS4_Q4E_GRAPH_VOCAB];
    float second_output[DS4_Q4E_GRAPH_VOCAB];
    uint64_t before = 0u;
    uint64_t after = 0u;
    uint64_t first_digest = 0u;
    uint64_t second_digest = 0u;
    test_backend_init(&bad_backend, false);
    test_backend_init(&first_backend, false);
    test_backend_init(&second_backend, false);
    bad = test_graph_create(&bad_backend);
    first = test_graph_create(&first_backend);
    second = test_graph_create(&second_backend);
    if (!bad || !first || !second) goto cleanup;
    CHECK(ds4_qwen4exp_graph_public_digest(bad, &before));
    bad_backend.nonfinite_after = 10u;
    CHECK(!ds4_qwen4exp_graph_run(
        bad, ds4_q4e_golden_tokens, 1u, output, DS4_Q4E_GRAPH_VOCAB));
    CHECK(ds4_qwen4exp_graph_public_digest(bad, &after));
    CHECK(before == after);

    /* Two independent graph objects are the per-sequence axis.  Interleaving
     * turns must not leak PLE history, conv state, or QSA slots between them. */
    CHECK(ds4_qwen4exp_graph_run(first, ds4_q4e_golden_tokens, 2u,
                                  first_output, DS4_Q4E_GRAPH_VOCAB));
    CHECK(ds4_qwen4exp_graph_run(second, ds4_q4e_golden_tokens + 3u, 1u,
                                  second_output, DS4_Q4E_GRAPH_VOCAB));
    CHECK(ds4_qwen4exp_graph_run(first, ds4_q4e_golden_tokens + 2u, 2u,
                                  first_output, DS4_Q4E_GRAPH_VOCAB));
    CHECK(ds4_qwen4exp_graph_run(second, ds4_q4e_golden_tokens + 4u, 2u,
                                  second_output, DS4_Q4E_GRAPH_VOCAB));
    CHECK(ds4_qwen4exp_graph_frontier(first) == 4u);
    CHECK(ds4_qwen4exp_graph_frontier(second) == 3u);
    CHECK(ds4_qwen4exp_graph_public_digest(first, &first_digest));
    CHECK(ds4_qwen4exp_graph_public_digest(second, &second_digest));
    CHECK(first_digest != second_digest);

cleanup:
    ds4_qwen4exp_graph_destroy(bad);
    ds4_qwen4exp_graph_destroy(first);
    ds4_qwen4exp_graph_destroy(second);
}

int main(void) {
    test_golden_and_chunks();
    test_ple_eos_ubatch_transitions();
    test_plan_reset_and_rejections();
    test_failure_transactions();
    test_nonfinite_and_multiturn_isolation();
    if (failures != 0) {
        fprintf(stderr, "%d qwen4exp graph test failure(s)\n", failures);
        return 1;
    }
    puts("qwen4exp graph tests: PASS");
    return 0;
}
