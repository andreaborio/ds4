#include "ds4_qwen_ref.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "qwen/qwen36_gdn_golden.inc"

static int check_close(const char *name,
                       const float *got,
                       const float *want,
                       size_t n,
                       float tolerance) {
    for (size_t i = 0; i < n; i++) {
        const float scale = fmaxf(1.0f, fabsf(want[i]));
        if (!isfinite(got[i]) || fabsf(got[i] - want[i]) > tolerance * scale) {
            fprintf(stderr,
                    "qwen-ref: %s[%zu] got %.9g, expected %.9g\n",
                    name, i, (double)got[i], (double)want[i]);
            return 1;
        }
    }
    return 0;
}

static int test_causal_conv(void) {
    enum {
        N_OUT = QWEN_REF_N_TOKEN * QWEN_REF_N_CHANNEL,
        N_STATE = QWEN_REF_N_CHANNEL * (QWEN_REF_KERNEL - 1),
    };
    float output[N_OUT] = {0};
    float state[N_STATE] = {0};
    ds4_qwen_ref_causal_conv1d_silu_f32(
        output, state, qwen_ref_conv_input, qwen_ref_conv_weight,
        QWEN_REF_N_TOKEN, QWEN_REF_N_CHANNEL, QWEN_REF_KERNEL);
    if (check_close("conv output", output, qwen_ref_conv_output,
                    N_OUT, 2.0e-6f) ||
        check_close("conv state", state, qwen_ref_conv_state,
                    N_STATE, 2.0e-6f)) {
        return 1;
    }

    for (size_t split = 1; split < QWEN_REF_N_TOKEN; split++) {
        float chunk_output[N_OUT] = {0};
        float chunk_state[N_STATE] = {0};
        ds4_qwen_ref_causal_conv1d_silu_f32(
            chunk_output, chunk_state, qwen_ref_conv_input,
            qwen_ref_conv_weight, split, QWEN_REF_N_CHANNEL, QWEN_REF_KERNEL);
        ds4_qwen_ref_causal_conv1d_silu_f32(
            chunk_output + split * QWEN_REF_N_CHANNEL, chunk_state,
            qwen_ref_conv_input + split * QWEN_REF_N_CHANNEL,
            qwen_ref_conv_weight, QWEN_REF_N_TOKEN - split,
            QWEN_REF_N_CHANNEL, QWEN_REF_KERNEL);
        if (check_close("chunk conv output", chunk_output, output,
                        N_OUT, 1.0e-7f) ||
            check_close("chunk conv state", chunk_state, state,
                        N_STATE, 1.0e-7f)) {
            return 1;
        }
    }
    return 0;
}

static int run_delta(float *output, float *state,
                     size_t token_offset, size_t n_token) {
    return ds4_qwen_ref_gated_delta_rule_f32(
        output,
        state,
        qwen_ref_query + token_offset * QWEN_REF_N_KEY_HEAD * QWEN_REF_KEY_DIM,
        qwen_ref_key + token_offset * QWEN_REF_N_KEY_HEAD * QWEN_REF_KEY_DIM,
        qwen_ref_value + token_offset * QWEN_REF_N_VALUE_HEAD * QWEN_REF_VALUE_DIM,
        qwen_ref_log_decay + token_offset * QWEN_REF_N_VALUE_HEAD,
        qwen_ref_beta + token_offset * QWEN_REF_N_VALUE_HEAD,
        n_token,
        QWEN_REF_N_KEY_HEAD,
        QWEN_REF_N_VALUE_HEAD,
        QWEN_REF_KEY_DIM,
        QWEN_REF_VALUE_DIM) ? 0 : 1;
}

static int test_gated_delta(void) {
    enum {
        N_OUT = QWEN_REF_N_TOKEN * QWEN_REF_N_VALUE_HEAD * QWEN_REF_VALUE_DIM,
        N_STATE = QWEN_REF_N_VALUE_HEAD * QWEN_REF_KEY_DIM * QWEN_REF_VALUE_DIM,
    };
    float output[N_OUT] = {0};
    float state[N_STATE];
    memcpy(state, qwen_ref_initial_state, sizeof(state));
    if (run_delta(output, state, 0, QWEN_REF_N_TOKEN) != 0) return 1;
    if (check_close("delta output", output, qwen_ref_delta_output,
                    N_OUT, 3.0e-6f) ||
        check_close("delta state", state, qwen_ref_delta_state,
                    N_STATE, 3.0e-6f)) {
        return 1;
    }

    /* Decode and arbitrary prefill chunks must preserve the same recurrent
     * boundary.  This is the invariant used by prefix reuse and checkpoints. */
    for (size_t first = 1; first < QWEN_REF_N_TOKEN; first++) {
        float chunk_output[N_OUT] = {0};
        float chunk_state[N_STATE];
        memcpy(chunk_state, qwen_ref_initial_state, sizeof(chunk_state));
        if (run_delta(chunk_output, chunk_state, 0, first) != 0 ||
            run_delta(chunk_output + first * QWEN_REF_N_VALUE_HEAD * QWEN_REF_VALUE_DIM,
                      chunk_state, first, QWEN_REF_N_TOKEN - first) != 0 ||
            check_close("chunk output", chunk_output, output, N_OUT, 1.0e-7f) ||
            check_close("chunk state", chunk_state, state, N_STATE, 1.0e-7f)) {
            return 1;
        }
    }
    return 0;
}

static int test_gated_delta_controls(void) {
    enum { N = QWEN_REF_N_TOKEN * QWEN_REF_N_VALUE_HEAD };
    float log_decay[N];
    float beta[N];
    ds4_qwen_ref_gated_delta_controls_f32(
        log_decay, beta, qwen_ref_alpha_logit, qwen_ref_beta_logit,
        qwen_ref_ssm_a, qwen_ref_dt_bias,
        QWEN_REF_N_TOKEN, QWEN_REF_N_VALUE_HEAD);
    return check_close("log decay", log_decay, qwen_ref_log_decay, N, 2.0e-6f) ||
           check_close("beta", beta, qwen_ref_beta, N, 2.0e-6f);
}

static int test_rmsnorm_gate(void) {
    float output[QWEN_REF_N_TOKEN * QWEN_REF_N_VALUE_HEAD * QWEN_REF_VALUE_DIM];
    ds4_qwen_ref_rmsnorm_gated_f32(
        output,
        qwen_ref_delta_output,
        qwen_ref_gate,
        qwen_ref_norm_weight,
        QWEN_REF_N_TOKEN * QWEN_REF_N_VALUE_HEAD,
        QWEN_REF_VALUE_DIM,
        1.0e-6f);
    return check_close("gated RMSNorm", output, qwen_ref_gated_output,
                       sizeof(output) / sizeof(output[0]), 3.0e-6f);
}

static int test_invalid_geometry(void) {
    float output[1] = {0};
    float state[1] = {0};
    float input[1] = {0};
    return ds4_qwen_ref_gated_delta_rule_f32(
        output, state, input, input, input, input, input,
        1, 2, 3, 1, 1) ? 1 : 0;
}

static void fill_router_logits(float logits[256]) {
    for (int i = 0; i < 256; i++) logits[i] = -20.0f - (float)i * 0.001f;
    logits[201] = 3.0f;
    logits[7] = 2.5f;
    logits[88] = 2.0f;
    logits[42] = 1.5f;
    logits[111] = 1.0f;
    logits[3] = 0.5f;
    logits[17] = 0.1f;
    logits[19] = 0.0f;
    logits[23] = -0.1f;
}

static int test_router(void) {
    float logits[256];
    int32_t selected[8];
    float weight[8];
    fill_router_logits(logits);
    if (!ds4_qwen_ref_softmax_topk_f32(selected, weight, logits, 256, 8)) return 1;
    for (size_t i = 0; i < 8; i++) {
        if (selected[i] != qwen_ref_router_id[i]) {
            fprintf(stderr, "qwen-ref: router id[%zu] got %d, expected %d\n",
                    i, selected[i], qwen_ref_router_id[i]);
            return 1;
        }
    }
    if (check_close("router weight", weight, qwen_ref_router_weight, 8, 2.0e-6f)) {
        return 1;
    }
    for (int i = 0; i < 256; i++) logits[i] = -100.0f;
    for (int i = 0; i < 8; i++) logits[i] = 100.0f - (float)i;
    if (!ds4_qwen_ref_softmax_topk_f32(selected, weight, logits, 256, 8)) return 1;
    float total = 0.0f;
    for (int i = 0; i < 8; i++) {
        if (selected[i] != i || !isfinite(weight[i])) return 1;
        total += weight[i];
    }
    if (fabsf(total - 1.0f) > 1.0e-6f) return 1;
    if (ds4_qwen_ref_softmax_topk_f32(
            selected, weight, logits, 7, 8)) {
        return 1;
    }

    /* torch.topk does not define tie order.  ds4 does: lower expert ID wins,
     * so CPU and GPU backends remain deterministic at the kth boundary. */
    for (int i = 0; i < 256; i++) logits[i] = -100.0f;
    for (int i = 0; i < 6; i++) logits[100 + i] = 10.0f - (float)i;
    logits[17] = logits[19] = logits[23] = 0.0f;
    if (!ds4_qwen_ref_softmax_topk_f32(selected, weight, logits, 256, 8) ||
        selected[6] != 17 || selected[7] != 19) {
        fprintf(stderr, "qwen-ref: router tie policy is not deterministic\n");
        return 1;
    }
    return 0;
}

static int test_shared_expert_gate(void) {
    enum { N_VECTOR = 2, DIM = 3 };
    float output[N_VECTOR * DIM];
    ds4_qwen_ref_sigmoid_gate_f32(
        output, qwen_ref_shared_input, qwen_ref_shared_gate_logit,
        N_VECTOR, DIM);
    return check_close("shared expert gate", output, qwen_ref_shared_output,
                       N_VECTOR * DIM, 2.0e-6f);
}

int main(void) {
    int failed = 0;
    failed += test_causal_conv();
    failed += test_gated_delta();
    failed += test_gated_delta_controls();
    failed += test_rmsnorm_gate();
    failed += test_invalid_geometry();
    failed += test_router();
    failed += test_shared_expert_gate();
    if (failed != 0) {
        fprintf(stderr, "qwen reference tests: FAIL (%d)\n", failed);
        return 1;
    }
    printf("qwen reference tests: OK\n");
    return 0;
}
