#include "ds4_qwen_ref.h"
#include "ds4_qwen.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "qwen/qwen36_attention_golden.inc"

static int expect_close(
        const char  *name,
        const float *got,
        const float *want,
        size_t       n,
        float        tolerance) {
    for (size_t i = 0; i < n; i++) {
        const float difference = fabsf(got[i] - want[i]);
        const float scale = fmaxf(1.0f, fabsf(want[i]));
        if (!isfinite(got[i]) || difference > tolerance * scale) {
            fprintf(stderr,
                    "%s[%zu]: got %.9g, want %.9g, difference %.9g\n",
                    name, i, (double)got[i], (double)want[i],
                    (double)difference);
            return 1;
        }
    }
    return 0;
}

static int test_q_gate_layout(void) {
    enum {
        N_VALUE = QWEN_ATTN_N_TOKEN * QWEN_ATTN_N_QUERY_HEAD *
                  QWEN_ATTN_HEAD_DIM,
    };
    float query[N_VALUE];
    float gate[N_VALUE];
    if (!ds4_qwen_ref_split_q_gate_f32(
            query, gate, qwen_attn_projection,
            QWEN_ATTN_N_TOKEN, QWEN_ATTN_N_QUERY_HEAD,
            QWEN_ATTN_HEAD_DIM)) {
        fprintf(stderr, "Q/gate split rejected valid geometry\n");
        return 1;
    }
    if (expect_close("query split", query, qwen_attn_query,
                     N_VALUE, 0.0f) ||
        expect_close("gate split", gate, qwen_attn_gate,
                     N_VALUE, 0.0f)) {
        return 1;
    }
    for (size_t token = 0; token < QWEN_ATTN_N_TOKEN; token++) {
        const size_t output_stride =
            QWEN_ATTN_N_QUERY_HEAD * QWEN_ATTN_HEAD_DIM;
        const size_t projection_stride = 2u * output_stride;
        if (!ds4_qwen35_cpu_split_q_gate_f32(
                query + token * output_stride,
                gate + token * output_stride,
                qwen_attn_projection + token * projection_stride,
                QWEN_ATTN_N_QUERY_HEAD,
                QWEN_ATTN_HEAD_DIM)) {
            return 1;
        }
    }
    return expect_close("production query split", query, qwen_attn_query,
                        N_VALUE, 0.0f) ||
           expect_close("production gate split", gate, qwen_attn_gate,
                        N_VALUE, 0.0f);
}

static int test_norm_rope_attention(void) {
    enum {
        N_QUERY_VALUE = QWEN_ATTN_N_TOKEN * QWEN_ATTN_N_QUERY_HEAD *
                        QWEN_ATTN_HEAD_DIM,
        N_KV_VALUE = QWEN_ATTN_N_TOKEN * QWEN_ATTN_N_KV_HEAD *
                     QWEN_ATTN_HEAD_DIM,
    };
    float query[N_QUERY_VALUE];
    float key[N_KV_VALUE];
    float attention[N_QUERY_VALUE];
    float gated[N_QUERY_VALUE];

    if (!ds4_qwen_ref_head_rms_norm_f32(
            query, qwen_attn_query, qwen_attn_q_weight,
            QWEN_ATTN_N_TOKEN, QWEN_ATTN_N_QUERY_HEAD,
            QWEN_ATTN_HEAD_DIM, QWEN_ATTN_EPSILON) ||
        !ds4_qwen_ref_head_rms_norm_f32(
            key, qwen_attn_key, qwen_attn_k_weight,
            QWEN_ATTN_N_TOKEN, QWEN_ATTN_N_KV_HEAD,
            QWEN_ATTN_HEAD_DIM, QWEN_ATTN_EPSILON)) {
        fprintf(stderr, "head RMSNorm rejected valid geometry\n");
        return 1;
    }
    if (expect_close("query norm", query, qwen_attn_query_norm,
                     N_QUERY_VALUE, 3.0e-6f) ||
        expect_close("key norm", key, qwen_attn_key_norm,
                     N_KV_VALUE, 3.0e-6f)) {
        return 1;
    }

    for (size_t token = 0; token < QWEN_ATTN_N_TOKEN; token++) {
        const size_t query_stride =
            QWEN_ATTN_N_QUERY_HEAD * QWEN_ATTN_HEAD_DIM;
        const size_t key_stride =
            QWEN_ATTN_N_KV_HEAD * QWEN_ATTN_HEAD_DIM;
        if (!ds4_qwen35_cpu_head_rms_norm_f32(
                query + token * query_stride,
                qwen_attn_query + token * query_stride,
                qwen_attn_q_weight,
                QWEN_ATTN_N_QUERY_HEAD,
                QWEN_ATTN_HEAD_DIM,
                QWEN_ATTN_EPSILON) ||
            !ds4_qwen35_cpu_head_rms_norm_f32(
                key + token * key_stride,
                qwen_attn_key + token * key_stride,
                qwen_attn_k_weight,
                QWEN_ATTN_N_KV_HEAD,
                QWEN_ATTN_HEAD_DIM,
                QWEN_ATTN_EPSILON)) {
            return 1;
        }
    }
    if (expect_close("production query norm", query,
                     qwen_attn_query_norm, N_QUERY_VALUE, 3.0e-6f) ||
        expect_close("production key norm", key,
                     qwen_attn_key_norm, N_KV_VALUE, 3.0e-6f)) {
        return 1;
    }

    if (!ds4_qwen_ref_text_rope_f32(
            query, qwen_attn_position, QWEN_ATTN_N_TOKEN,
            QWEN_ATTN_N_QUERY_HEAD, QWEN_ATTN_HEAD_DIM,
            QWEN_ATTN_N_ROT, QWEN_ATTN_ROPE_THETA) ||
        !ds4_qwen_ref_text_rope_f32(
            key, qwen_attn_position, QWEN_ATTN_N_TOKEN,
            QWEN_ATTN_N_KV_HEAD, QWEN_ATTN_HEAD_DIM,
            QWEN_ATTN_N_ROT, QWEN_ATTN_ROPE_THETA)) {
        fprintf(stderr, "text RoPE rejected valid geometry\n");
        return 1;
    }
    if (expect_close("query RoPE", query, qwen_attn_query_rope,
                     N_QUERY_VALUE, 8.0e-6f) ||
        expect_close("key RoPE", key, qwen_attn_key_rope,
                     N_KV_VALUE, 8.0e-6f)) {
        return 1;
    }

    memcpy(query, qwen_attn_query_norm, sizeof(query));
    memcpy(key, qwen_attn_key_norm, sizeof(key));
    for (size_t token = 0; token < QWEN_ATTN_N_TOKEN; token++) {
        if (!ds4_qwen35_cpu_text_rope_f32(
                query + token * QWEN_ATTN_N_QUERY_HEAD * QWEN_ATTN_HEAD_DIM,
                qwen_attn_position[token],
                QWEN_ATTN_N_QUERY_HEAD,
                QWEN_ATTN_HEAD_DIM,
                QWEN_ATTN_N_ROT,
                QWEN_ATTN_ROPE_THETA) ||
            !ds4_qwen35_cpu_text_rope_f32(
                key + token * QWEN_ATTN_N_KV_HEAD * QWEN_ATTN_HEAD_DIM,
                qwen_attn_position[token],
                QWEN_ATTN_N_KV_HEAD,
                QWEN_ATTN_HEAD_DIM,
                QWEN_ATTN_N_ROT,
                QWEN_ATTN_ROPE_THETA)) {
            return 1;
        }
    }
    if (expect_close("production query RoPE", query,
                     qwen_attn_query_rope, N_QUERY_VALUE, 8.0e-6f) ||
        expect_close("production key RoPE", key,
                     qwen_attn_key_rope, N_KV_VALUE, 8.0e-6f)) {
        return 1;
    }

    if (!ds4_qwen_ref_causal_gqa_f32(
            attention, query, key, qwen_attn_value,
            QWEN_ATTN_N_TOKEN, QWEN_ATTN_N_QUERY_HEAD,
            QWEN_ATTN_N_KV_HEAD, QWEN_ATTN_HEAD_DIM)) {
        fprintf(stderr, "causal GQA rejected valid geometry\n");
        return 1;
    }
    if (expect_close("causal GQA", attention, qwen_attn_output,
                     N_QUERY_VALUE, 1.5e-5f)) {
        return 1;
    }

    float score[QWEN_ATTN_N_TOKEN];
    for (size_t token = 0; token < QWEN_ATTN_N_TOKEN; token++) {
        if (!ds4_qwen35_cpu_gqa_decode_f32(
                attention +
                    token * QWEN_ATTN_N_QUERY_HEAD * QWEN_ATTN_HEAD_DIM,
                score,
                QWEN_ATTN_N_TOKEN,
                qwen_attn_query_rope +
                    token * QWEN_ATTN_N_QUERY_HEAD * QWEN_ATTN_HEAD_DIM,
                qwen_attn_key_rope,
                qwen_attn_value,
                token + 1u,
                QWEN_ATTN_N_QUERY_HEAD,
                QWEN_ATTN_N_KV_HEAD,
                QWEN_ATTN_HEAD_DIM)) {
            return 1;
        }
    }
    if (expect_close("production causal GQA", attention,
                     qwen_attn_output, N_QUERY_VALUE, 1.5e-5f)) {
        return 1;
    }

    ds4_qwen_ref_sigmoid_gate_elements_f32(
        gated, attention, qwen_attn_gate, N_QUERY_VALUE);
    if (expect_close("attention gate", gated, qwen_attn_gated,
                     N_QUERY_VALUE, 1.5e-5f)) {
        return 1;
    }
    if (!ds4_qwen35_cpu_sigmoid_gate_elements_f32(
            gated, attention, qwen_attn_gate, N_QUERY_VALUE)) {
        return 1;
    }
    return expect_close("production attention gate", gated,
                        qwen_attn_gated, N_QUERY_VALUE, 1.5e-5f);
}

static int test_causal_prefix_parity(void) {
    enum {
        N_QUERY_VALUE = QWEN_ATTN_N_TOKEN * QWEN_ATTN_N_QUERY_HEAD *
                        QWEN_ATTN_HEAD_DIM,
    };
    float output[N_QUERY_VALUE];
    for (size_t prefix = 1; prefix <= QWEN_ATTN_N_TOKEN; prefix++) {
        if (!ds4_qwen_ref_causal_gqa_f32(
                output, qwen_attn_query_rope, qwen_attn_key_rope,
                qwen_attn_value, prefix, QWEN_ATTN_N_QUERY_HEAD,
                QWEN_ATTN_N_KV_HEAD, QWEN_ATTN_HEAD_DIM)) {
            fprintf(stderr, "causal GQA rejected prefix %zu\n", prefix);
            return 1;
        }
        const size_t n = prefix * QWEN_ATTN_N_QUERY_HEAD *
                         QWEN_ATTN_HEAD_DIM;
        if (expect_close("prefix parity", output, qwen_attn_output,
                         n, 1.5e-5f)) {
            return 1;
        }
    }
    return 0;
}

static int test_contiguous_gqa_mapping(void) {
    const float query[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const float key[2] = {0.0f, 0.0f};
    const float value[2] = {3.0f, -7.0f};
    const float expected[4] = {3.0f, 3.0f, -7.0f, -7.0f};
    float output[4] = {0};
    if (!ds4_qwen_ref_causal_gqa_f32(
            output, query, key, value, 1, 4, 2, 1)) {
        fprintf(stderr, "single-token GQA rejected valid geometry\n");
        return 1;
    }
    if (expect_close("GQA head mapping", output, expected, 4, 0.0f)) {
        return 1;
    }
    float score[1];
    if (!ds4_qwen35_cpu_gqa_decode_f32(
            output, score, 1, query, key, value, 1, 4, 2, 1)) {
        return 1;
    }
    return expect_close("production GQA head mapping", output,
                        expected, 4, 0.0f);
}

static int test_invalid_geometry(void) {
    float value[8] = {0};
    const uint32_t position[1] = {0};
    if (ds4_qwen_ref_split_q_gate_f32(value, value, value, 0, 1, 1) ||
        ds4_qwen_ref_head_rms_norm_f32(
            value, value, value, 1, 1, 1, 0.0f) ||
        ds4_qwen_ref_text_rope_f32(
            value, position, 1, 1, 4, 3, 10000.0f) ||
        ds4_qwen_ref_causal_gqa_f32(
            value, value, value, value, 1, 3, 2, 1)) {
        fprintf(stderr, "Qwen attention reference accepted invalid geometry\n");
        return 1;
    }
    if (ds4_qwen35_cpu_split_q_gate_f32(value, value, value, 0, 1) ||
        ds4_qwen35_cpu_head_rms_norm_f32(
            value, value, value, 1, 1, 0.0f) ||
        ds4_qwen35_cpu_text_rope_f32(
            value, 0, 1, 4, 3, 10000.0f) ||
        ds4_qwen35_cpu_gqa_decode_f32(
            value, value, 0, value, value, value, 1, 3, 2, 1)) {
        fprintf(stderr, "Qwen attention production path accepted invalid geometry\n");
        return 1;
    }
    return 0;
}

int main(void) {
    if (test_q_gate_layout() ||
        test_norm_rope_attention() ||
        test_causal_prefix_parity() ||
        test_contiguous_gqa_mapping() ||
        test_invalid_geometry()) {
        return 1;
    }
    puts("qwen full-attention reference/production tests: OK");
    return 0;
}
