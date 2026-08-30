#include "ds4_qwen4exp_ref.h"

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "qwen4exp/qwen4exp_scalar_golden.inc"

static bool close_f32(float actual, float expected, float tolerance) {
    return fabsf(actual - expected) <=
           tolerance * fmaxf(1.0f, fabsf(expected));
}

static void assert_array_close(const float *actual, const float *expected,
                               size_t count, float tolerance) {
    for (size_t i = 0; i < count; i++) {
        assert(close_f32(actual[i], expected[i], tolerance));
    }
}

static float test_sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    const float e = expf(x);
    return e / (1.0f + e);
}

static float test_silu(float x) {
    return x * test_sigmoid(x);
}

static void test_norm_conventions_and_rejection(void) {
    const float input[] = {3.0f, 4.0f, -2.0f, 1.0f};
    const float zc_weight[] = {0.0f, 0.0f};
    const float conventional_weight[] = {1.0f, 2.0f};
    const float gate[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float output[4] = {91.0f, 92.0f, 93.0f, 94.0f};
    float expected[4];

    assert(ds4_qwen4exp_ref_zero_centered_rmsnorm_f32(
        output, input, zc_weight, 2, 2, 1.0e-6f));
    for (size_t vector = 0; vector < 2; vector++) {
        const float x0 = input[vector * 2];
        const float x1 = input[vector * 2 + 1];
        const float inverse = 1.0f /
            sqrtf((x0 * x0 + x1 * x1) / 2.0f + 1.0e-6f);
        expected[vector * 2] = x0 * inverse;
        expected[vector * 2 + 1] = x1 * inverse;
    }
    assert_array_close(output, expected, 4, 1.0e-6f);
    assert(output[0] != 0.0f); /* Stored zero means scale one. */

    assert(ds4_qwen4exp_ref_sigmoid_gated_rmsnorm_f32(
        output, input, gate, conventional_weight, 2, 2, 1.0e-6f));
    for (size_t vector = 0; vector < 2; vector++) {
        const float x0 = input[vector * 2];
        const float x1 = input[vector * 2 + 1];
        const float inverse = 1.0f /
            sqrtf((x0 * x0 + x1 * x1) / 2.0f + 1.0e-6f);
        expected[vector * 2] = x0 * inverse * 0.5f;
        expected[vector * 2 + 1] = x1 * inverse * 2.0f * 0.5f;
    }
    assert_array_close(output, expected, 4, 1.0e-6f);

    const float zero_weight[] = {0.0f, 0.0f};
    assert(ds4_qwen4exp_ref_sigmoid_gated_rmsnorm_f32(
        output, input, gate, zero_weight, 2, 2, 1.0e-6f));
    for (size_t i = 0; i < 4; i++) assert(output[i] == 0.0f);

    const float bad_input[] = {1.0f, NAN, 2.0f, 3.0f};
    const float sentinel[] = {91.0f, 92.0f, 93.0f, 94.0f};
    memcpy(output, sentinel, sizeof(output));
    assert(!ds4_qwen4exp_ref_zero_centered_rmsnorm_f32(
        output, bad_input, zc_weight, 2, 2, 1.0e-6f));
    assert(memcmp(output, sentinel, sizeof(output)) == 0);
    assert(!ds4_qwen4exp_ref_zero_centered_rmsnorm_f32(
        output, input, zc_weight, SIZE_MAX, 2, 1.0e-6f));
    assert(memcmp(output, sentinel, sizeof(output)) == 0);
    assert(!ds4_qwen4exp_ref_zero_centered_rmsnorm_f32(
        NULL, input, zc_weight, 2, 2, 1.0e-6f));
    assert(!ds4_qwen4exp_ref_sigmoid_gated_rmsnorm_f32(
        output, input, NULL, conventional_weight, 2, 2, 1.0e-6f));
}

static void test_gr_four_stream_equations(void) {
    const float residual[] = {1.0f, -2.0f, 3.0f, -4.0f};
    const float norm_weight[] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float down[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float up[] = {1.0f, -1.0f, 0.5f, -0.5f};
    const float inject_weight[] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f,
    };
    float mixed = -99.0f;
    float injection[4] = {-1.0f, -1.0f, -1.0f, -1.0f};
    float normalized[4];
    const float epsilon = 1.0e-6f;

    for (size_t i = 0; i < 4; i++) {
        normalized[i] = residual[i] /
            sqrtf(residual[i] * residual[i] + epsilon);
    }
    const float hidden = test_silu(
        (normalized[0] + 2.0f * normalized[1] +
         3.0f * normalized[2] + 4.0f * normalized[3]) / 4.0f);
    float expected_mixed = 0.0f;
    for (size_t i = 0; i < 4; i++) {
        expected_mixed += test_sigmoid(up[i] * hidden) * normalized[i];
    }
    expected_mixed /= 4.0f;

    assert(ds4_qwen4exp_ref_gr_prepare_f32(
        &mixed, injection, residual, norm_weight, down, up, inject_weight,
        4, 1, 1, epsilon));
    assert(close_f32(mixed, expected_mixed, 1.0e-6f));
    for (size_t stream = 0; stream < 4; stream++) {
        const float expected_injection =
            2.0f * test_sigmoid(normalized[stream] / 4.0f);
        assert(close_f32(injection[stream], expected_injection, 1.0e-6f));
    }

    float final = 0.0f;
    assert(ds4_qwen4exp_ref_gr_final_mix_f32(
        &final, residual, norm_weight, down, up, 4, 1, 1, epsilon));
    assert(close_f32(final, mixed, 1.0e-6f));

    float updated[4];
    const float block[] = {2.0f};
    memcpy(updated, residual, sizeof(updated));
    assert(ds4_qwen4exp_ref_gr_apply_f32(updated, block, injection, 4, 1));
    for (size_t i = 0; i < 4; i++) {
        assert(close_f32(updated[i], residual[i] + injection[i] * 2.0f,
                         1.0e-6f));
    }

    const float bad_down[] = {1.0f, 2.0f, NAN, 4.0f};
    mixed = -99.0f;
    const float inject_before[] = {-1.0f, -1.0f, -1.0f, -1.0f};
    memcpy(injection, inject_before, sizeof(injection));
    assert(!ds4_qwen4exp_ref_gr_prepare_f32(
        &mixed, injection, residual, norm_weight, bad_down, up, inject_weight,
        4, 1, 1, epsilon));
    assert(mixed == -99.0f);
    assert(memcmp(injection, inject_before, sizeof(injection)) == 0);
    assert(!ds4_qwen4exp_ref_gr_prepare_f32(
        &mixed, injection, residual, norm_weight, down, up, inject_weight,
        5, 1, 1, epsilon));
}

static void test_causal_conv_and_chunk_state(void) {
    const float input[] = {4.0f, 5.0f};
    const float weight[] = {1.0f, 10.0f, 100.0f, 1000.0f};
    float state[] = {1.0f, 2.0f, 3.0f};
    float output[2] = {0.0f, 0.0f};
    const float expected[] = {4321.0f, 5432.0f};
    const float expected_state[] = {3.0f, 4.0f, 5.0f};

    assert(ds4_qwen4exp_ref_causal_conv1d_silu_f32(
        output, state, input, weight, 2, 1, 4));
    assert_array_close(output, expected, 2, 1.0e-6f);
    assert_array_close(state, expected_state, 3, 0.0f);

    const float longer[] = {0.1f, -0.2f, 0.3f, -0.4f, 0.5f};
    const float short_weight[] = {0.2f, -0.1f, 0.4f};
    const float initial_state[] = {0.6f, -0.7f};
    float all_state[2];
    float chunk_state[2];
    float all_output[5];
    float chunk_output[5];
    memcpy(all_state, initial_state, sizeof(all_state));
    memcpy(chunk_state, initial_state, sizeof(chunk_state));
    assert(ds4_qwen4exp_ref_causal_conv1d_silu_f32(
        all_output, all_state, longer, short_weight, 5, 1, 3));
    assert(ds4_qwen4exp_ref_causal_conv1d_silu_f32(
        chunk_output, chunk_state, longer, short_weight, 2, 1, 3));
    assert(ds4_qwen4exp_ref_causal_conv1d_silu_f32(
        chunk_output + 2, chunk_state, longer + 2, short_weight, 3, 1, 3));
    assert_array_close(chunk_output, all_output, 5, 0.0f);
    assert_array_close(chunk_state, all_state, 2, 0.0f);

    float rejected_state[] = {7.0f, 8.0f};
    float rejected_output[] = {9.0f};
    const float bad_input[] = {NAN};
    assert(!ds4_qwen4exp_ref_causal_conv1d_silu_f32(
        rejected_output, rejected_state, bad_input, short_weight, 1, 1, 3));
    assert(rejected_state[0] == 7.0f && rejected_state[1] == 8.0f);
    assert(rejected_output[0] == 9.0f);
    assert(!ds4_qwen4exp_ref_causal_conv1d_silu_f32(
        rejected_output, rejected_state, input, weight, SIZE_MAX, 2, 4));
}

static void test_gdn_controls_and_recurrence(void) {
    const float alpha[] = {0.0f, 1.0f};
    const float beta_logit[] = {0.0f, -2.0f};
    const float a_log[] = {logf(2.0f)};
    const float dt_bias[] = {0.5f};
    float decay[2] = {0.0f, 0.0f};
    float beta[2] = {0.0f, 0.0f};
    assert(ds4_qwen4exp_ref_gdn_controls_f32(
        decay, beta, alpha, beta_logit, a_log, dt_bias, 2, 1));
    assert(close_f32(decay[0], -2.0f * log1pf(expf(0.5f)), 1.0e-6f));
    assert(close_f32(decay[1], -2.0f * log1pf(expf(1.5f)), 1.0e-6f));
    assert(close_f32(beta[0], 0.5f, 1.0e-6f));
    assert(close_f32(beta[1], test_sigmoid(-2.0f), 1.0e-6f));

    /* Division mapping gives heads [0,0,1,1], unlike modulo [0,1,0,1]. */
    const float query[] = {1.0f, -1.0f};
    const float key[] = {1.0f, 1.0f};
    const float value[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float log_decay[] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float unit_beta[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float state[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float output[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float norm_factor = 1.0f / (1.0f + 1.0e-6f);
    const float expected[] = {
        norm_factor, 2.0f * norm_factor,
        -3.0f * norm_factor, -4.0f * norm_factor,
    };
    assert(ds4_qwen4exp_ref_gdn_f32(
        output, state, query, key, value, log_decay, unit_beta,
        1, 2, 4, 1, 1));
    assert_array_close(output, expected, 4, 2.0e-6f);

    /* State is [value_head][key][value], not transposed [value][key]. */
    const float q2[] = {1.0f, 0.0f};
    const float k2[] = {1.0f, 0.0f};
    const float v2[] = {2.0f, 3.0f};
    const float decay2[] = {0.0f};
    const float beta2[] = {1.0f};
    float state2[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float output2[2] = {0.0f, 0.0f};
    assert(ds4_qwen4exp_ref_gdn_f32(
        output2, state2, q2, k2, v2, decay2, beta2, 1, 1, 1, 2, 2));
    assert(state2[0] > 1.9f && state2[1] > 2.9f);
    assert(state2[2] == 0.0f && state2[3] == 0.0f);

    const float chunk_query[] = {0.2f, 0.4f, -0.1f};
    const float chunk_key[] = {0.3f, -0.5f, 0.7f};
    const float chunk_value[] = {1.0f, 2.0f, -1.0f};
    const float chunk_decay[] = {-0.1f, -0.2f, -0.3f};
    const float chunk_beta[] = {0.25f, 0.5f, 0.75f};
    float all_state[] = {0.6f};
    float step_state[] = {0.6f};
    float all_output[3];
    float step_output[3];
    assert(ds4_qwen4exp_ref_gdn_f32(
        all_output, all_state, chunk_query, chunk_key, chunk_value,
        chunk_decay, chunk_beta, 3, 1, 1, 1, 1));
    for (size_t token = 0; token < 3; token++) {
        assert(ds4_qwen4exp_ref_gdn_f32(
            step_output + token, step_state, chunk_query + token,
            chunk_key + token, chunk_value + token, chunk_decay + token,
            chunk_beta + token, 1, 1, 1, 1, 1));
    }
    assert_array_close(step_output, all_output, 3, 0.0f);
    assert_array_close(step_state, all_state, 1, 0.0f);

    float preserve_state[] = {5.0f};
    float preserve_output[] = {6.0f, 7.0f};
    const float bad_query[] = {1.0f, NAN};
    const float two_key[] = {1.0f, 1.0f};
    const float two_value[] = {1.0f, 1.0f};
    const float two_decay[] = {0.0f, 0.0f};
    const float two_beta[] = {1.0f, 1.0f};
    assert(!ds4_qwen4exp_ref_gdn_f32(
        preserve_output, preserve_state, bad_query, two_key, two_value,
        two_decay, two_beta, 2, 1, 1, 1, 1));
    assert(preserve_state[0] == 5.0f);
    assert(preserve_output[0] == 6.0f && preserve_output[1] == 7.0f);
    assert(!ds4_qwen4exp_ref_gdn_f32(
        preserve_output, preserve_state, q2, k2, v2, decay2, beta2,
        SIZE_MAX, 2, 2, 2, 2));

    /* Finite inputs can still overflow the recurrence.  Failure must not
     * publish a partial state or output. */
    const float overflow_query[] = {1.0f, 1.0f};
    const float overflow_key[] = {-1.0f, -1.0f};
    const float overflow_value[] = {FLT_MAX};
    const float overflow_decay[] = {0.0f};
    const float overflow_beta[] = {1.0f};
    float overflow_state[] = {FLT_MAX, FLT_MAX};
    float overflow_output[] = {123.0f};
    const float overflow_state_before[] = {FLT_MAX, FLT_MAX};
    assert(!ds4_qwen4exp_ref_gdn_f32(
        overflow_output, overflow_state, overflow_query, overflow_key,
        overflow_value, overflow_decay, overflow_beta, 1, 1, 1, 2, 1));
    assert(memcmp(overflow_state, overflow_state_before,
                  sizeof(overflow_state)) == 0);
    assert(overflow_output[0] == 123.0f);

    assert(!ds4_qwen4exp_ref_gdn_f32(
        preserve_output, preserve_state, q2, k2, v2, decay2, beta2,
        1, DS4_QWEN4EXP_GDN_KEY_HEADS + 1u,
        DS4_QWEN4EXP_GDN_VALUE_HEADS, 1, 1));
    assert(!ds4_qwen4exp_ref_gdn_f32(
        preserve_output, preserve_state, q2, k2, v2, decay2, beta2,
        1, 1, 1, DS4_QWEN4EXP_GDN_HEAD_DIM + 1u, 1));
}

static void test_router_full_softmax_and_ties(void) {
    const float logits[] = {1000.0f, 1000.0f, -1000.0f, 999.0f};
    uint32_t selected[3] = {99u, 99u, 99u};
    float weight[3] = {-1.0f, -1.0f, -1.0f};
    assert(ds4_qwen4exp_ref_softmax_topk_f32(
        selected, weight, logits, 4, 3));
    assert(selected[0] == 0u && selected[1] == 1u && selected[2] == 3u);
    const float denominator = 2.0f + expf(-1.0f);
    assert(close_f32(weight[0], 1.0f / denominator, 1.0e-6f));
    assert(close_f32(weight[1], 1.0f / denominator, 1.0e-6f));
    assert(close_f32(weight[2], expf(-1.0f) / denominator, 1.0e-6f));
    assert(close_f32(weight[0] + weight[1] + weight[2], 1.0f, 1.0e-6f));

    const float equal[] = {0.0f, 0.0f, 0.0f, 0.0f};
    assert(ds4_qwen4exp_ref_softmax_topk_f32(
        selected, weight, equal, 4, 3));
    assert(selected[0] == 0u && selected[1] == 1u && selected[2] == 2u);

    const float bad[] = {0.0f, INFINITY, 1.0f};
    const uint32_t selected_before[] = {77u, 78u, 79u};
    const float weight_before[] = {7.0f, 8.0f, 9.0f};
    memcpy(selected, selected_before, sizeof(selected));
    memcpy(weight, weight_before, sizeof(weight));
    assert(!ds4_qwen4exp_ref_softmax_topk_f32(
        selected, weight, bad, 3, 2));
    assert(memcmp(selected, selected_before, sizeof(selected)) == 0);
    assert(memcmp(weight, weight_before, sizeof(weight)) == 0);
}

static void test_partial_rope_grouping_scores_and_selection(void) {
    float values[] = {1.0f, 2.0f, 3.0f, 4.0f, 9.0f};
    const uint32_t position[] = {1u};
    const float c = cosf(1.0f);
    const float s = sinf(1.0f);
    const float expected[] = {
        1.0f * c - 3.0f * s,
        2.0f * c - 4.0f * s,
        3.0f * c + 1.0f * s,
        4.0f * c + 2.0f * s,
        9.0f,
    };
    assert(ds4_qwen4exp_ref_partial_rope_f32(
        values, position, 1, 1, 5, 4, 1.0f));
    assert_array_close(values, expected, 5, 1.0e-6f);

    const float raw_key[] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        3.0f, 4.0f, 5.0f, 6.0f,
        -1.0f, 0.0f, 1.0f, 2.0f,
        1.0f, 2.0f, 3.0f, 4.0f,
    };
    const float norm_weight[] = {0.0f, 0.0f, 0.0f, 0.0f};
    float group_key[8];
    assert(ds4_qwen4exp_ref_qsa_group_keys_f32(
        group_key, raw_key, norm_weight, 2, 2, 4, 2, 1.0f, 1.0e-6f));
    for (size_t group = 0; group < 2; group++) {
        float pooled[4];
        float sum_square = 0.0f;
        for (size_t i = 0; i < 4; i++) {
            pooled[i] = (raw_key[(group * 2) * 4 + i] +
                         raw_key[(group * 2 + 1) * 4 + i]) / 2.0f;
            sum_square += pooled[i] * pooled[i];
        }
        const float inverse = 1.0f / sqrtf(sum_square / 4.0f + 1.0e-6f);
        for (size_t i = 0; i < 4; i++) pooled[i] *= inverse;
        const float angle = (float)(group * 2);
        const float expected_first = pooled[0] * cosf(angle) -
                                     pooled[1] * sinf(angle);
        const float expected_second = pooled[1] * cosf(angle) +
                                      pooled[0] * sinf(angle);
        assert(close_f32(group_key[group * 4], expected_first, 1.0e-6f));
        assert(close_f32(group_key[group * 4 + 1], expected_second, 1.0e-6f));
        assert(close_f32(group_key[group * 4 + 2], pooled[2], 1.0e-6f));
        assert(close_f32(group_key[group * 4 + 3], pooled[3], 1.0e-6f));
    }

    const float score_query[] = {1.0f, 0.0f, -1.0f, 0.0f};
    const float score_keys[] = {1.0f, 0.0f, -2.0f, 0.0f};
    float scores[2] = {0.0f, 0.0f};
    assert(ds4_qwen4exp_ref_qsa_scores_f32(
        scores, score_query, score_keys, 2, 2, 2));
    assert(close_f32(scores[0], 1.0f / sqrtf(2.0f), 1.0e-6f));
    assert(close_f32(scores[1], 2.0f / sqrtf(2.0f), 1.0e-6f));

    const float selection_scores[] = {1.0f, 3.0f, 3.0f};
    uint32_t selected_position[10] = {0};
    size_t n_position = 99;
    assert(ds4_qwen4exp_ref_qsa_select_positions(
        selected_position, 10, &n_position, selection_scores, 14, 4, 2));
    const uint32_t expected_position[] = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
    assert(n_position == 10);
    assert(memcmp(selected_position, expected_position,
                  sizeof(expected_position)) == 0);

    uint32_t preserved[] = {41u, 42u};
    const uint32_t preserved_before[] = {41u, 42u};
    n_position = 77;
    assert(!ds4_qwen4exp_ref_qsa_select_positions(
        preserved, 2, &n_position, selection_scores, 14, 4, 2));
    assert(n_position == 77);
    assert(memcmp(preserved, preserved_before, sizeof(preserved)) == 0);
    for (size_t visible = 0; visible < 4; visible++) {
        uint32_t tail_position[3] = {99u, 99u, 99u};
        n_position = 99;
        assert(ds4_qwen4exp_ref_qsa_select_positions(
            tail_position, 3, &n_position, selection_scores,
            visible, 4, 2));
        assert(n_position == visible);
        for (size_t i = 0; i < visible; i++) {
            assert(tail_position[i] == i);
        }
    }
    n_position = 77;
    assert(!ds4_qwen4exp_ref_qsa_select_positions(
        preserved, 2, &n_position, selection_scores, 4, 5, 1));
    assert(n_position == 77);
    assert(memcmp(preserved, preserved_before, sizeof(preserved)) == 0);
    assert(!ds4_qwen4exp_ref_partial_rope_f32(
        values, position, SIZE_MAX, 2, 5, 4, 1.0f));
}

static void manual_ple_rows(
        uint32_t row[DS4_QWEN4EXP_PLE_HEADS],
        uint32_t current,
        uint32_t previous1,
        uint32_t previous2,
        const ds4_qwen4exp_profile *profile) {
    const uint64_t bigram =
        (uint64_t)current * profile->ple_multiplier[0] ^
        (uint64_t)previous1 * profile->ple_multiplier[1];
    const uint64_t trigram = bigram ^
        (uint64_t)previous2 * profile->ple_multiplier[2];
    for (size_t head = 0; head < DS4_QWEN4EXP_PLE_HEADS; head++) {
        const uint64_t hash = head < DS4_QWEN4EXP_PLE_HEADS_PER_NGRAM
            ? bigram : trigram;
        row[head] = profile->ple_head_offset[head] +
                    (uint32_t)(hash % profile->ple_head_prime[head]);
    }
}

static void test_ple_history_hash_gate_and_dilation(void) {
    const ds4_qwen4exp_profile *profile = ds4_qwen4exp_profile_get();
    ds4_qwen4exp_ple_history history;
    ds4_qwen4exp_ple_history next;
    uint32_t row[DS4_QWEN4EXP_PLE_HEADS];
    uint32_t expected_row[DS4_QWEN4EXP_PLE_HEADS];

    ds4_qwen4exp_ref_ple_history_reset(&history);
    assert(history.count == 0u);
    assert(history.token[0] == DS4_QWEN4EXP_PLE_PAD_TOKEN);
    assert(history.token[1] == DS4_QWEN4EXP_PLE_PAD_TOKEN);
    assert(ds4_qwen4exp_ref_ple_rows(row, UINT32_MAX, &history, profile));
    manual_ple_rows(expected_row, UINT32_MAX, DS4_QWEN4EXP_PLE_PAD_TOKEN,
                    DS4_QWEN4EXP_PLE_PAD_TOKEN, profile);
    assert(memcmp(row, expected_row, sizeof(row)) == 0);

    assert(ds4_qwen4exp_ref_ple_history_advance(&next, &history, 10u));
    assert(next.count == 1u && next.token[0] == 10u);
    history = next;
    assert(ds4_qwen4exp_ref_ple_history_advance(&next, &history, 11u));
    assert(next.count == 2u && next.token[0] == 11u && next.token[1] == 10u);
    history = next;

    /* Current EOS still sees predecessors; only its successor is reset. */
    assert(ds4_qwen4exp_ref_ple_rows(
        row, DS4_QWEN4EXP_PLE_PAD_TOKEN, &history, profile));
    manual_ple_rows(expected_row, DS4_QWEN4EXP_PLE_PAD_TOKEN, 11u, 10u,
                    profile);
    assert(memcmp(row, expected_row, sizeof(row)) == 0);
    assert(ds4_qwen4exp_ref_ple_history_advance(
        &next, &history, DS4_QWEN4EXP_PLE_PAD_TOKEN));
    assert(next.count == 0u);
    assert(next.token[0] == DS4_QWEN4EXP_PLE_PAD_TOKEN);
    assert(next.token[1] == DS4_QWEN4EXP_PLE_PAD_TOKEN);

    ds4_qwen4exp_ple_history invalid = history;
    invalid.count = 3u;
    next.count = 99u;
    assert(!ds4_qwen4exp_ref_ple_history_advance(&next, &invalid, 12u));
    assert(next.count == 99u);
    memset(row, 0xa5, sizeof(row));
    uint32_t preserved_row[DS4_QWEN4EXP_PLE_HEADS];
    memcpy(preserved_row, row, sizeof(row));
    assert(!ds4_qwen4exp_ref_ple_rows(row, 12u, &invalid, profile));
    assert(memcmp(row, preserved_row, sizeof(row)) == 0);

    const float query[] = {4.0f, -4.0f, 0.0f, 1.0e-8f};
    const float key[] = {1.0f, 1.0f, 1.0f, 1.0f};
    const float value[] = {2.0f};
    float gated[4];
    assert(ds4_qwen4exp_ref_ple_gate_f32(gated, query, key, value, 4, 1));
    assert(close_f32(gated[0], 2.0f * test_sigmoid(2.0f), 1.0e-6f));
    assert(close_f32(gated[1], 2.0f * test_sigmoid(-2.0f), 1.0e-6f));
    assert(close_f32(gated[2], 1.0f, 1.0e-6f));
    assert(close_f32(gated[3], 2.0f * test_sigmoid(0.001f), 1.0e-6f));

    const float conv_input[] = {5.0f, 6.0f, 7.0f};
    const float conv_weight[] = {1.0f, 10.0f, 100.0f};
    float conv_state[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float conv_output[3];
    const float conv_expected[] = {531.0f, 642.0f, 753.0f};
    const float state_expected[] = {4.0f, 5.0f, 6.0f, 7.0f};
    assert(ds4_qwen4exp_ref_dilated_conv1d_silu_f32(
        conv_output, conv_state, conv_input, conv_weight, 3, 1, 3, 2));
    assert_array_close(conv_output, conv_expected, 3, 1.0e-6f);
    assert_array_close(conv_state, state_expected, 4, 0.0f);

    const float longer[] = {0.5f, -0.2f, 0.8f, -0.1f};
    const float small_weight[] = {0.3f, -0.2f, 0.7f};
    const float initial[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float all_state[4];
    float chunk_state[4];
    float all_output[4];
    float chunk_output[4];
    memcpy(all_state, initial, sizeof(all_state));
    memcpy(chunk_state, initial, sizeof(chunk_state));
    assert(ds4_qwen4exp_ref_dilated_conv1d_silu_f32(
        all_output, all_state, longer, small_weight, 4, 1, 3, 2));
    assert(ds4_qwen4exp_ref_dilated_conv1d_silu_f32(
        chunk_output, chunk_state, longer, small_weight, 1, 1, 3, 2));
    assert(ds4_qwen4exp_ref_dilated_conv1d_silu_f32(
        chunk_output + 1, chunk_state, longer + 1, small_weight, 3, 1, 3, 2));
    assert_array_close(chunk_output, all_output, 4, 0.0f);
    assert_array_close(chunk_state, all_state, 4, 0.0f);
}

static void test_null_and_overflow_shapes(void) {
    float output[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float state[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    const float input[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    const float weight[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    const float before_output[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float before_state[4] = {5.0f, 6.0f, 7.0f, 8.0f};

    assert(!ds4_qwen4exp_ref_dilated_conv1d_silu_f32(
        output, state, input, weight, SIZE_MAX, 2, 3, 2));
    assert(!ds4_qwen4exp_ref_dilated_conv1d_silu_f32(
        output, state, input, weight, SIZE_MAX, 1, 3, 2));
    assert(memcmp(output, before_output, sizeof(output)) == 0);
    assert(memcmp(state, before_state, sizeof(state)) == 0);
    assert(!ds4_qwen4exp_ref_dilated_conv1d_silu_f32(
        output, NULL, input, weight, 1, 1, 3, 2));
    assert(!ds4_qwen4exp_ref_gdn_controls_f32(
        output, output + 2, input, input, input, input, SIZE_MAX, 2));
    assert(!ds4_qwen4exp_ref_qsa_scores_f32(
        output, input, input, SIZE_MAX, 2, 2));
    assert(!ds4_qwen4exp_ref_ple_gate_f32(
        output, input, input, input, SIZE_MAX, 2));
}

static void test_state_reset_copy_and_rewind(void) {
    float state[] = {1.0f, -2.0f, 3.0f, -4.0f};
    const float checkpoint[] = {0.25f, 0.5f, 0.75f, 1.0f};
    const float replacement[] = {-1.0f, -2.0f, -3.0f, -4.0f};

    assert(ds4_qwen4exp_ref_state_reset_f32(state, 4));
    for (size_t i = 0; i < 4; i++) assert(state[i] == 0.0f);
    assert(ds4_qwen4exp_ref_state_copy_f32(state, checkpoint, 4));
    assert_array_close(state, checkpoint, 4, 0.0f);
    assert(ds4_qwen4exp_ref_state_copy_f32(state, state, 4));
    assert(ds4_qwen4exp_ref_state_copy_f32(state, replacement, 4));
    assert_array_close(state, replacement, 4, 0.0f);
    assert(ds4_qwen4exp_ref_state_rewind_f32(state, checkpoint, 4));
    assert_array_close(state, checkpoint, 4, 0.0f);

    float overlap[] = {10.0f, 11.0f, 12.0f, 13.0f, 14.0f};
    const float overlap_before[] = {10.0f, 11.0f, 12.0f, 13.0f, 14.0f};
    assert(!ds4_qwen4exp_ref_state_copy_f32(overlap + 1, overlap, 4));
    assert(memcmp(overlap, overlap_before, sizeof(overlap)) == 0);
    assert(!ds4_qwen4exp_ref_state_rewind_f32(overlap, overlap + 1, 4));
    assert(memcmp(overlap, overlap_before, sizeof(overlap)) == 0);

    const float nonfinite[] = {1.0f, NAN, 3.0f, 4.0f};
    assert(!ds4_qwen4exp_ref_state_copy_f32(state, nonfinite, 4));
    assert_array_close(state, checkpoint, 4, 0.0f);
    assert(!ds4_qwen4exp_ref_state_reset_f32(NULL, 4));
    assert(!ds4_qwen4exp_ref_state_reset_f32(state, 0));
    assert(!ds4_qwen4exp_ref_state_copy_f32(state, checkpoint, SIZE_MAX));
    assert_array_close(state, checkpoint, 4, 0.0f);
}

static void test_pinned_scalar_golden_vectors(void) {
    const ds4_qwen4exp_profile *profile = ds4_qwen4exp_profile_get();
    assert(strcmp(Q4E_SCALAR_HF_REPOSITORY,
                  "Qwen/Qwen3.8-Flash-Next") == 0);
    assert(strcmp(Q4E_SCALAR_HF_REVISION, profile->hf_revision) == 0);
    assert(strcmp(Q4E_SCALAR_TRANSFORMERS_COMMIT,
                  profile->transformers_commit) == 0);
    assert(strcmp(Q4E_SCALAR_TRANSFORMERS_SOURCE_SHA256,
                  "91e9b1e9c74efe373cd989fe1974a8fa305f4aad43628dbcbd03dac20437814f") == 0);
    assert(strcmp(Q4E_SCALAR_CONFIG_SHA256,
                  "889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b") == 0);
    assert(strcmp(Q4E_SCALAR_INVENTORY_SHA256,
                  "a639efc7a5147b04200e870d7e320335527f4361a8327b137feca2683b1dc434") == 0);
    assert(strcmp(Q4E_SCALAR_TOKENIZER_SHA256,
                  "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3") == 0);
    assert(strcmp(Q4E_SCALAR_TOKENIZER_CONFIG_SHA256,
                  "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27") == 0);
    assert(strcmp(Q4E_SCALAR_CHAT_TEMPLATE_SHA256,
                  "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041") == 0);
    assert(strcmp(Q4E_SCALAR_ARRAY_SHA256,
                  "9564a15b4fff26cc1db7c2e9872c0b033bbd4e8a9d1e644c50e649bd00122406") == 0);
    assert(strcmp(Q4E_SCALAR_PYTHON_VERSION, "3.13.13") == 0);
    assert(strcmp(Q4E_SCALAR_NUMPY_VERSION, "2.4.6") == 0);
    assert(strcmp(Q4E_SCALAR_TORCH_VERSION, "2.9.1") == 0);
    assert(strcmp(Q4E_SCALAR_TRANSFORMERS_VERSION, "5.16.0.dev0") == 0);
    assert(strcmp(Q4E_SCALAR_DTYPE, "float32") == 0);
    assert(strcmp(Q4E_SCALAR_DEVICE, "cpu") == 0);
    assert(Q4E_SCALAR_SEED == UINT32_C(1313167444));

    float norm_output[Q4E_NORM_N_VECTOR * Q4E_NORM_DIM];
    assert(ds4_qwen4exp_ref_zero_centered_rmsnorm_f32(
        norm_output, q4e_norm_input, q4e_norm_zero_weight,
        Q4E_NORM_N_VECTOR, Q4E_NORM_DIM, Q4E_SCALAR_EPSILON));
    assert_array_close(norm_output, q4e_norm_zero_output,
                       Q4E_NORM_N_VECTOR * Q4E_NORM_DIM, 3.0e-6f);
    assert(ds4_qwen4exp_ref_zero_centered_rmsnorm_f32(
        norm_output, q4e_norm_input, q4e_norm_weight,
        Q4E_NORM_N_VECTOR, Q4E_NORM_DIM, Q4E_SCALAR_EPSILON));
    assert_array_close(norm_output, q4e_norm_output,
                       Q4E_NORM_N_VECTOR * Q4E_NORM_DIM, 3.0e-6f);
    assert(ds4_qwen4exp_ref_sigmoid_gated_rmsnorm_f32(
        norm_output, q4e_gated_norm_input, q4e_gated_norm_gate,
        q4e_gated_norm_weight, Q4E_NORM_N_VECTOR, Q4E_NORM_DIM,
        Q4E_SCALAR_EPSILON));
    assert_array_close(norm_output, q4e_gated_norm_output,
                       Q4E_NORM_N_VECTOR * Q4E_NORM_DIM, 3.0e-6f);

    float mixed[Q4E_GR_DIM];
    float injection[Q4E_GR_N_STREAM];
    float applied[Q4E_GR_N_STREAM * Q4E_GR_DIM];
    float final[Q4E_GR_DIM];
    assert(ds4_qwen4exp_ref_gr_prepare_f32(
        mixed, injection, q4e_gr_residual, q4e_gr_norm_weight,
        q4e_gr_down, q4e_gr_up, q4e_gr_inject,
        Q4E_GR_N_STREAM, Q4E_GR_DIM, Q4E_GR_RANK, Q4E_SCALAR_EPSILON));
    assert_array_close(mixed, q4e_gr_mixed, Q4E_GR_DIM, 4.0e-6f);
    assert_array_close(injection, q4e_gr_injection, Q4E_GR_N_STREAM,
                       4.0e-6f);
    memcpy(applied, q4e_gr_residual, sizeof(applied));
    assert(ds4_qwen4exp_ref_gr_apply_f32(
        applied, q4e_gr_block_output, injection,
        Q4E_GR_N_STREAM, Q4E_GR_DIM));
    assert_array_close(applied, q4e_gr_applied,
                       Q4E_GR_N_STREAM * Q4E_GR_DIM, 4.0e-6f);
    assert(ds4_qwen4exp_ref_gr_final_mix_f32(
        final, applied, q4e_gr_norm_weight, q4e_gr_down, q4e_gr_up,
        Q4E_GR_N_STREAM, Q4E_GR_DIM, Q4E_GR_RANK, Q4E_SCALAR_EPSILON));
    assert_array_close(final, q4e_gr_final, Q4E_GR_DIM, 4.0e-6f);

    float gdn_conv_state[Q4E_GDN_CONV_N_CHANNEL *
                         (Q4E_GDN_CONV_KERNEL - 1)];
    float gdn_conv_output[Q4E_GDN_CONV_N_TOKEN * Q4E_GDN_CONV_N_CHANNEL];
    memcpy(gdn_conv_state, q4e_gdn_conv_state_initial,
           sizeof(gdn_conv_state));
    assert(ds4_qwen4exp_ref_causal_conv1d_silu_f32(
        gdn_conv_output, gdn_conv_state, q4e_gdn_conv_input,
        q4e_gdn_conv_weight, Q4E_GDN_CONV_N_TOKEN,
        Q4E_GDN_CONV_N_CHANNEL, Q4E_GDN_CONV_KERNEL));
    assert_array_close(gdn_conv_output, q4e_gdn_conv_output,
                       Q4E_GDN_CONV_N_TOKEN * Q4E_GDN_CONV_N_CHANNEL,
                       5.0e-6f);
    assert_array_close(gdn_conv_state, q4e_gdn_conv_state_final,
                       Q4E_GDN_CONV_N_CHANNEL *
                       (Q4E_GDN_CONV_KERNEL - 1), 0.0f);

    float log_decay[Q4E_GDN_N_TOKEN * Q4E_GDN_N_VALUE_HEAD];
    float beta[Q4E_GDN_N_TOKEN * Q4E_GDN_N_VALUE_HEAD];
    assert(ds4_qwen4exp_ref_gdn_controls_f32(
        log_decay, beta, q4e_gdn_alpha_logit, q4e_gdn_beta_logit,
        q4e_gdn_a_log, q4e_gdn_dt_bias,
        Q4E_GDN_N_TOKEN, Q4E_GDN_N_VALUE_HEAD));
    assert_array_close(log_decay, q4e_gdn_log_decay,
                       Q4E_GDN_N_TOKEN * Q4E_GDN_N_VALUE_HEAD, 5.0e-6f);
    assert_array_close(beta, q4e_gdn_beta,
                       Q4E_GDN_N_TOKEN * Q4E_GDN_N_VALUE_HEAD, 3.0e-6f);
    for (size_t head = 0; head < Q4E_GDN_N_VALUE_HEAD; head++) {
        assert(q4e_gdn_head_map[head] == head / Q4E_GDN_REPEAT_RATIO);
    }
    float gdn_state[Q4E_GDN_N_VALUE_HEAD * Q4E_GDN_KEY_DIM *
                    Q4E_GDN_VALUE_DIM];
    float gdn_output[Q4E_GDN_N_TOKEN * Q4E_GDN_N_VALUE_HEAD *
                     Q4E_GDN_VALUE_DIM];
    memcpy(gdn_state, q4e_gdn_state_initial, sizeof(gdn_state));
    assert(ds4_qwen4exp_ref_gdn_f32(
        gdn_output, gdn_state, q4e_gdn_query, q4e_gdn_key, q4e_gdn_value,
        log_decay, beta, Q4E_GDN_N_TOKEN, Q4E_GDN_N_KEY_HEAD,
        Q4E_GDN_N_VALUE_HEAD, Q4E_GDN_KEY_DIM, Q4E_GDN_VALUE_DIM));
    assert_array_close(gdn_output, q4e_gdn_output,
                       Q4E_GDN_N_TOKEN * Q4E_GDN_N_VALUE_HEAD *
                       Q4E_GDN_VALUE_DIM, 1.2e-5f);
    assert_array_close(gdn_state, q4e_gdn_state_final,
                       Q4E_GDN_N_VALUE_HEAD * Q4E_GDN_KEY_DIM *
                       Q4E_GDN_VALUE_DIM, 1.2e-5f);

    {
        uint32_t ids[Q4E_ROUTER_N_SELECTED];
        float weights[Q4E_ROUTER_N_SELECTED];
        assert(ds4_qwen4exp_ref_softmax_topk_f32(
            ids, weights, q4e_router_equal_logits,
            Q4E_ROUTER_N_EXPERT, Q4E_ROUTER_N_SELECTED));
        assert(memcmp(ids, q4e_router_equal_id, sizeof(ids)) == 0);
        assert_array_close(weights, q4e_router_equal_weight,
                           Q4E_ROUTER_N_SELECTED, 3.0e-6f);
    }
    for (size_t test_case = 0;
         test_case < Q4E_ROUTER_N_UPSTREAM_CASE; test_case++) {
        uint32_t ids[Q4E_ROUTER_N_SELECTED];
        float weights[Q4E_ROUTER_N_SELECTED];
        assert(ds4_qwen4exp_ref_softmax_topk_f32(
            ids, weights,
            q4e_router_upstream_logits + test_case * Q4E_ROUTER_N_EXPERT,
            Q4E_ROUTER_N_EXPERT, Q4E_ROUTER_N_SELECTED));
        assert(memcmp(ids, q4e_router_upstream_id +
                      test_case * Q4E_ROUTER_N_SELECTED, sizeof(ids)) == 0);
        assert_array_close(weights, q4e_router_upstream_weight +
                           test_case * Q4E_ROUTER_N_SELECTED,
                           Q4E_ROUTER_N_SELECTED, 5.0e-6f);
    }

    float group_key[Q4E_QSA_N_GROUP * Q4E_QSA_HEAD_DIM];
    assert(ds4_qwen4exp_ref_qsa_group_keys_f32(
        group_key, q4e_qsa_raw_key, q4e_qsa_norm_weight,
        Q4E_QSA_N_GROUP, Q4E_QSA_COMPRESSION, Q4E_QSA_HEAD_DIM,
        Q4E_QSA_N_ROT, Q4E_QSA_THETA, Q4E_SCALAR_EPSILON));
    assert_array_close(group_key, q4e_qsa_group_key,
                       Q4E_QSA_N_GROUP * Q4E_QSA_HEAD_DIM, 5.0e-6f);
    float pooled_key[Q4E_QSA_N_GROUP * Q4E_QSA_HEAD_DIM];
    float normalized_key[Q4E_QSA_N_GROUP * Q4E_QSA_HEAD_DIM];
    for (size_t group = 0; group < Q4E_QSA_N_GROUP; group++) {
        for (size_t dim = 0; dim < Q4E_QSA_HEAD_DIM; dim++) {
            float sum = 0.0f;
            for (size_t token = 0; token < Q4E_QSA_COMPRESSION; token++) {
                sum += q4e_qsa_raw_key[
                    (group * Q4E_QSA_COMPRESSION + token) *
                    Q4E_QSA_HEAD_DIM + dim];
            }
            pooled_key[group * Q4E_QSA_HEAD_DIM + dim] =
                sum / (float)Q4E_QSA_COMPRESSION;
        }
    }
    assert(ds4_qwen4exp_ref_zero_centered_rmsnorm_f32(
        normalized_key, pooled_key, q4e_qsa_norm_weight, Q4E_QSA_N_GROUP,
        Q4E_QSA_HEAD_DIM, Q4E_SCALAR_EPSILON));
    for (size_t group = 0; group < Q4E_QSA_N_GROUP; group++) {
        for (size_t dim = Q4E_QSA_N_ROT; dim < Q4E_QSA_HEAD_DIM; dim++) {
            assert(close_f32(group_key[group * Q4E_QSA_HEAD_DIM + dim],
                             normalized_key[group * Q4E_QSA_HEAD_DIM + dim],
                             5.0e-6f));
        }
    }
    for (size_t group = 0; group < Q4E_QSA_N_GROUP; group++) {
        assert(q4e_qsa_group_position[group] ==
               group * Q4E_QSA_COMPRESSION);
        for (size_t head = 0; head < Q4E_QSA_N_QUERY_HEAD; head++) {
            float dot = 0.0f;
            for (size_t i = 0; i < Q4E_QSA_HEAD_DIM; i++) {
                dot += q4e_qsa_query[head * Q4E_QSA_HEAD_DIM + i] *
                       group_key[group * Q4E_QSA_HEAD_DIM + i];
            }
            assert(close_f32(
                dot, q4e_qsa_head_dot[group * Q4E_QSA_N_QUERY_HEAD + head],
                5.0e-6f));
        }
    }
    float qsa_score[Q4E_QSA_N_GROUP];
    assert(ds4_qwen4exp_ref_qsa_scores_f32(
        qsa_score, q4e_qsa_query, group_key, Q4E_QSA_N_GROUP,
        Q4E_QSA_N_QUERY_HEAD, Q4E_QSA_HEAD_DIM));
    assert_array_close(qsa_score, q4e_qsa_score, Q4E_QSA_N_GROUP, 5.0e-6f);
    bool differs_from_wrong = false;
    for (size_t group = 0; group < Q4E_QSA_N_GROUP; group++) {
        if (!close_f32(qsa_score[group], q4e_qsa_wrong_relu_after_sum[group],
                       1.0e-5f)) {
            differs_from_wrong = true;
        }
    }
    assert(differs_from_wrong);
    uint32_t selected[Q4E_QSA_N_SELECTED];
    size_t n_selected = 0;
    assert(ds4_qwen4exp_ref_qsa_select_positions(
        selected, Q4E_QSA_N_SELECTED, &n_selected, qsa_score,
        Q4E_QSA_VISIBLE_TOKEN, Q4E_QSA_COMPRESSION,
        Q4E_QSA_GROUP_BUDGET));
    assert(n_selected == Q4E_QSA_N_SELECTED);
    assert(memcmp(selected, q4e_qsa_selected, sizeof(selected)) == 0);
    assert(memcmp(selected + Q4E_QSA_N_SELECTED - 2,
                  q4e_qsa_tail_after, sizeof(q4e_qsa_tail_after)) == 0);
    assert(memcmp(q4e_qsa_tail_before, q4e_qsa_tail_after,
                  sizeof(q4e_qsa_tail_before)) == 0);
    assert(ds4_qwen4exp_ref_qsa_select_positions(
        selected, Q4E_QSA_N_SELECTED, &n_selected, q4e_qsa_tie_score,
        Q4E_QSA_VISIBLE_TOKEN, Q4E_QSA_COMPRESSION,
        Q4E_QSA_GROUP_BUDGET));
    assert(memcmp(selected, q4e_qsa_tie_selected, sizeof(selected)) == 0);

    assert(memcmp(profile->ple_multiplier, q4e_ple_multiplier,
                  sizeof(q4e_ple_multiplier)) == 0);
    assert(memcmp(profile->ple_head_prime, q4e_ple_head_prime,
                  sizeof(q4e_ple_head_prime)) == 0);
    assert(memcmp(profile->ple_head_offset, q4e_ple_head_offset,
                  sizeof(q4e_ple_head_offset)) == 0);
    assert(Q4E_PLE_PAD_TOKEN == DS4_QWEN4EXP_PLE_PAD_TOKEN);
    ds4_qwen4exp_ple_history history;
    ds4_qwen4exp_ref_ple_history_reset(&history);
    for (size_t token = 0; token < Q4E_PLE_N_TOKEN; token++) {
        assert(history.count == q4e_ple_history_count_before[token]);
        assert(history.token[0] == q4e_ple_history_before[token * 2]);
        assert(history.token[1] == q4e_ple_history_before[token * 2 + 1]);
        uint32_t rows[Q4E_PLE_N_HEAD];
        assert(ds4_qwen4exp_ref_ple_rows(
            rows, q4e_ple_token[token], &history, profile));
        assert(memcmp(rows, q4e_ple_row + token * Q4E_PLE_N_HEAD,
                      sizeof(rows)) == 0);
        ds4_qwen4exp_ple_history after;
        assert(ds4_qwen4exp_ref_ple_history_advance(
            &after, &history, q4e_ple_token[token]));
        assert(after.count == q4e_ple_history_count_after[token]);
        assert(after.token[0] == q4e_ple_history_after[token * 2]);
        assert(after.token[1] == q4e_ple_history_after[token * 2 + 1]);
        history = after;
    }
    const uint64_t product0 =
        q4e_ple_overflow_token[0] * q4e_ple_multiplier[0];
    const uint64_t product1 =
        q4e_ple_overflow_token[1] * q4e_ple_multiplier[1];
    const uint64_t product2 =
        q4e_ple_overflow_token[2] * q4e_ple_multiplier[2];
    assert(product0 == q4e_ple_overflow_product[0]);
    assert(product1 == q4e_ple_overflow_product[1]);
    assert(product2 == q4e_ple_overflow_product[2]);
    assert((product0 ^ product1) == q4e_ple_overflow_fold[0]);
    assert((product0 ^ product1 ^ product2) == q4e_ple_overflow_fold[1]);

    float ple_gate[Q4E_PLE_GATE_N_STREAM * Q4E_PLE_GATE_DIM];
    assert(ds4_qwen4exp_ref_ple_gate_f32(
        ple_gate, q4e_ple_gate_query, q4e_ple_gate_key,
        q4e_ple_gate_value, Q4E_PLE_GATE_N_STREAM, Q4E_PLE_GATE_DIM));
    assert_array_close(ple_gate, q4e_ple_gate_output,
                       Q4E_PLE_GATE_N_STREAM * Q4E_PLE_GATE_DIM, 5.0e-6f);
    for (size_t stream = 0; stream < Q4E_PLE_GATE_N_STREAM; stream++) {
        float dot = 0.0f;
        for (size_t i = 0; i < Q4E_PLE_GATE_DIM; i++) {
            dot += q4e_ple_gate_query[stream * Q4E_PLE_GATE_DIM + i] *
                   q4e_ple_gate_key[stream * Q4E_PLE_GATE_DIM + i];
        }
        const float scaled = dot / sqrtf((float)Q4E_PLE_GATE_DIM);
        const float signed_root = scaled > 0.0f
            ? sqrtf(fmaxf(scaled, 1.0e-6f))
            : scaled < 0.0f ? -sqrtf(fmaxf(-scaled, 1.0e-6f)) : 0.0f;
        assert(close_f32(signed_root, q4e_ple_gate_signed_root[stream],
                         5.0e-6f));
        assert(close_f32(test_sigmoid(signed_root),
                         q4e_ple_gate_sigmoid[stream], 5.0e-6f));
    }

    float conv_state[Q4E_PLE_CONV_N_CHANNEL * Q4E_PLE_CONV_STATE];
    float conv_output[Q4E_PLE_CONV_N_TOKEN * Q4E_PLE_CONV_N_CHANNEL];
    memcpy(conv_state, q4e_ple_conv_state_initial, sizeof(conv_state));
    assert(ds4_qwen4exp_ref_dilated_conv1d_silu_f32(
        conv_output, conv_state, q4e_ple_conv_input, q4e_ple_conv_weight,
        Q4E_PLE_CONV_N_TOKEN, Q4E_PLE_CONV_N_CHANNEL,
        Q4E_PLE_CONV_KERNEL, Q4E_PLE_CONV_DILATION));
    assert_array_close(conv_output, q4e_ple_conv_output,
                       Q4E_PLE_CONV_N_TOKEN * Q4E_PLE_CONV_N_CHANNEL,
                       5.0e-6f);
    assert_array_close(conv_state, q4e_ple_conv_state_final,
                       Q4E_PLE_CONV_N_CHANNEL * Q4E_PLE_CONV_STATE,
                       0.0f);
    for (size_t i = 0;
         i < Q4E_PLE_CONV_N_TOKEN * Q4E_PLE_CONV_N_CHANNEL; i++) {
        assert(close_f32(test_silu(q4e_ple_conv_preact[i]),
                         q4e_ple_conv_output[i], 5.0e-6f));
    }

    float state_control[Q4E_STATE_CONTROL_VALUES];
    memcpy(state_control, q4e_state_control_initial, sizeof(state_control));
    assert(ds4_qwen4exp_ref_state_copy_f32(
        state_control, q4e_state_control_copied, Q4E_STATE_CONTROL_VALUES));
    assert_array_close(state_control, q4e_state_control_copied,
                       Q4E_STATE_CONTROL_VALUES, 0.0f);
    memcpy(state_control, q4e_state_control_advanced, sizeof(state_control));
    assert(ds4_qwen4exp_ref_state_rewind_f32(
        state_control, q4e_state_control_rewound, Q4E_STATE_CONTROL_VALUES));
    assert_array_close(state_control, q4e_state_control_rewound,
                       Q4E_STATE_CONTROL_VALUES, 0.0f);
    assert(ds4_qwen4exp_ref_state_reset_f32(
        state_control, Q4E_STATE_CONTROL_VALUES));
    assert_array_close(state_control, q4e_state_control_reset,
                       Q4E_STATE_CONTROL_VALUES, 0.0f);
}

int main(void) {
    test_norm_conventions_and_rejection();
    test_gr_four_stream_equations();
    test_causal_conv_and_chunk_state();
    test_gdn_controls_and_recurrence();
    test_router_full_softmax_and_ties();
    test_partial_rope_grouping_scores_and_selection();
    test_ple_history_hash_gate_and_dilation();
    test_null_and_overflow_shapes();
    test_state_reset_copy_and_rewind();
    test_pinned_scalar_golden_vectors();
    puts("qwen4exp scalar reference tests passed");
    return 0;
}
