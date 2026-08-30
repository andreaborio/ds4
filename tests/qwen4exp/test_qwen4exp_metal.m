#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <math.h>
#include <float.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../ds4_qwen4exp_ref.h"
#include "../../runtime/ds4_metal_qwen4exp.inc"

/* Standalone Phase-5 CPU-oracle versus actual-device Metal qualification.
 *
 * clang -std=c11 -fobjc-arc -Wall -Wextra -Werror -Wpedantic \
 *   -framework Foundation -framework Metal \
 *   tests/qwen4exp/test_qwen4exp_metal.m \
 *   ds4_qwen4exp.c ds4_qwen4exp_ref.c -lm \
 *   -o /tmp/test-qwen4exp-metal
 * /tmp/test-qwen4exp-metal
 */

#define REQUIRE(condition, ...) do {                                      \
    if (!(condition)) {                                                   \
        fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);             \
        fprintf(stderr, __VA_ARGS__);                                     \
        fputc('\n', stderr);                                               \
        return false;                                                     \
    }                                                                    \
} while (0)

static const float k_atol = 2.0e-5f;
static const float k_rtol = 2.0e-5f;

static id<MTLBuffer> q4e_buffer(id<MTLDevice> device,
                                const void *bytes,
                                NSUInteger length) {
    if (bytes) {
        return [device newBufferWithBytes:bytes
                                   length:length
                                  options:MTLResourceStorageModeShared];
    }
    id<MTLBuffer> result =
        [device newBufferWithLength:length
                            options:MTLResourceStorageModeShared];
    if (result) memset(result.contents, 0, length);
    return result;
}

static bool q4e_close(float actual, float expected) {
    if (!isfinite(actual) || !isfinite(expected)) return false;
    return fabsf(actual - expected) <= k_atol + k_rtol * fabsf(expected);
}

static bool q4e_check_f32(const char *label,
                          const float *actual,
                          const float *expected,
                          size_t count) {
    for (size_t index = 0u; index < count; index++) {
        if (!q4e_close(actual[index], expected[index])) {
            fprintf(stderr, "%s[%zu]: got %.9g expected %.9g\n",
                    label, index, actual[index], expected[index]);
            return false;
        }
    }
    return true;
}

static bool q4e_status_ok(const uint32_t *status, size_t count) {
    for (size_t index = 0u; index < count; index++)
        if (status[index] != 0u) return false;
    return true;
}

static bool q4e_dispatch(
        id<MTLCommandQueue> queue,
        const ds4_metal_qwen4exp_cache *cache,
        ds4_metal_qwen4exp_kernel kernel,
        const void *arguments,
        NSUInteger argument_size,
        NSArray<id<MTLBuffer>> *buffers,
        NSArray<NSNumber *> *offsets,
        NSUInteger grid_width) {
    REQUIRE(buffers.count <= 12u, "too many test bindings");
    REQUIRE(offsets.count == 0u || offsets.count == buffers.count,
            "offset/buffer mismatch");
    ds4_metal_qwen4exp_binding bindings[12];
    for (NSUInteger index = 0u; index < buffers.count; index++) {
        const NSUInteger offset = offsets.count == 0u
            ? 0u : offsets[index].unsignedIntegerValue;
        REQUIRE(offset <= buffers[index].length, "test offset overflow");
        bindings[index] = (ds4_metal_qwen4exp_binding){
            .buffer = buffers[index],
            .offset = offset,
            .minimum_length = buffers[index].length - offset,
        };
    }
    id<MTLCommandBuffer> command = [queue commandBuffer];
    NSError *error = nil;
    REQUIRE(ds4_metal_qwen4exp_encode(
                cache, kernel, command, arguments, argument_size,
                bindings, buffers.count, MTLSizeMake(grid_width, 1u, 1u),
                MTLSizeMake(1u, 1u, 1u), &error),
            "encode %s: %s",
            ds4_metal_qwen4exp_kernel_name(kernel).UTF8String,
            error.localizedDescription.UTF8String);
    [command commit];
    [command waitUntilCompleted];
    REQUIRE(command.status == MTLCommandBufferStatusCompleted,
            "command %s status %ld: %s",
            ds4_metal_qwen4exp_kernel_name(kernel).UTF8String,
            (long)command.status,
            command.error.localizedDescription.UTF8String);
    return true;
}

static bool test_layout_and_bounds(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        const ds4_metal_qwen4exp_cache *cache) {
    ds4_metal_qwen4exp_head_args args = {
        .n_token = 1u, .input_dim = 1u, .output_dim = 1u,
    };
    float one = 1.0f;
    id<MTLBuffer> input = q4e_buffer(device, &one, sizeof(one));
    id<MTLBuffer> weight = q4e_buffer(device, &one, sizeof(one));
    id<MTLBuffer> output = q4e_buffer(device, NULL, sizeof(one));
    id<MTLBuffer> status = q4e_buffer(device, NULL, sizeof(uint32_t));
    REQUIRE(input && weight && output && status, "bounds buffers");
    ds4_metal_qwen4exp_binding bindings[] = {
        { input, 0u, input.length + 1u },
        { weight, 0u, weight.length },
        { output, 0u, output.length },
        { status, 0u, status.length },
    };
    id<MTLCommandBuffer> command = [queue commandBuffer];
    NSError *error = nil;
    REQUIRE(!ds4_metal_qwen4exp_encode(
                cache, DS4_METAL_QWEN4EXP_HEAD, command, &args, sizeof(args),
                bindings, sizeof(bindings) / sizeof(bindings[0]),
                MTLSizeMake(1u, 1u, 1u), MTLSizeMake(1u, 1u, 1u), &error),
            "undersized binding accepted");
    REQUIRE(error != nil, "missing bounds diagnostic");
    return true;
}

static bool test_embedding_gr(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        const ds4_metal_qwen4exp_cache *cache) {
    enum { TOKEN = 2, STREAM = 4, DIM = 5, RANK = 3, ROW = 3 };
    const uint32_t token_id[TOKEN] = { 2u, 0u };
    float table[ROW * DIM];
    for (size_t i = 0u; i < ROW * DIM; i++)
        table[i] = ((float)(int)(i % 7u) - 3.0f) * 0.13f;
    id<MTLBuffer> b_token = q4e_buffer(device, token_id, sizeof(token_id));
    id<MTLBuffer> b_table = q4e_buffer(device, table, sizeof(table));
    id<MTLBuffer> b_residual =
        q4e_buffer(device, NULL, TOKEN * STREAM * DIM * sizeof(float));
    id<MTLBuffer> b_status =
        q4e_buffer(device, NULL, TOKEN * sizeof(uint32_t));
    REQUIRE(b_token && b_table && b_residual && b_status, "embedding buffers");
    ds4_metal_qwen4exp_embedding_args embedding = {
        .n_token = TOKEN, .n_row = ROW, .n_stream = STREAM, .dim = DIM,
    };
    REQUIRE(q4e_dispatch(queue, cache, DS4_METAL_QWEN4EXP_EMBEDDING,
                         &embedding, sizeof(embedding),
                         @[b_token, b_table, b_residual, b_status], @[], TOKEN),
            "embedding dispatch");
    REQUIRE(q4e_status_ok(b_status.contents, TOKEN), "embedding status");
    const float *residual = b_residual.contents;
    for (size_t token = 0u; token < TOKEN; token++)
        for (size_t stream = 0u; stream < STREAM; stream++)
            for (size_t i = 0u; i < DIM; i++)
                REQUIRE(residual[(token * STREAM + stream) * DIM + i] ==
                            table[token_id[token] * DIM + i],
                        "embedding broadcast mismatch");

    float norm[STREAM * DIM];
    float down[RANK * STREAM * DIM];
    float up[STREAM * DIM * RANK];
    float inject[STREAM * STREAM * DIM];
    for (size_t i = 0u; i < STREAM * DIM; i++)
        norm[i] = ((float)(int)(i % 5u) - 2.0f) * 0.025f;
    for (size_t i = 0u; i < RANK * STREAM * DIM; i++)
        down[i] = ((float)(int)(i % 11u) - 5.0f) * 0.017f;
    for (size_t i = 0u; i < STREAM * DIM * RANK; i++)
        up[i] = ((float)(int)(i % 13u) - 6.0f) * 0.019f;
    for (size_t i = 0u; i < STREAM * STREAM * DIM; i++)
        inject[i] = ((float)(int)(i % 9u) - 4.0f) * 0.021f;

    float expected_mixed[TOKEN * DIM];
    float expected_injection[TOKEN * STREAM];
    for (size_t token = 0u; token < TOKEN; token++) {
        REQUIRE(ds4_qwen4exp_ref_gr_prepare_f32(
                    expected_mixed + token * DIM,
                    expected_injection + token * STREAM,
                    residual + token * STREAM * DIM, norm, down, up, inject,
                    STREAM, DIM, RANK, 1.0e-5f),
                "GR prepare oracle");
    }
    id<MTLBuffer> b_norm = q4e_buffer(device, norm, sizeof(norm));
    id<MTLBuffer> b_down = q4e_buffer(device, down, sizeof(down));
    id<MTLBuffer> b_up = q4e_buffer(device, up, sizeof(up));
    id<MTLBuffer> b_inject = q4e_buffer(device, inject, sizeof(inject));
    id<MTLBuffer> b_mixed = q4e_buffer(device, NULL, sizeof(expected_mixed));
    id<MTLBuffer> b_injection =
        q4e_buffer(device, NULL, sizeof(expected_injection));
    memset(b_status.contents, 0, b_status.length);
    ds4_metal_qwen4exp_gr_args gr = {
        .n_token = TOKEN, .n_stream = STREAM, .dim = DIM, .rank = RANK,
        .epsilon = 1.0e-5f,
    };
    REQUIRE(q4e_dispatch(queue, cache, DS4_METAL_QWEN4EXP_GR_PREPARE,
                         &gr, sizeof(gr),
                         @[b_residual, b_norm, b_down, b_up, b_inject,
                           b_mixed, b_injection, b_status], @[], TOKEN),
            "GR prepare dispatch");
    REQUIRE(q4e_status_ok(b_status.contents, TOKEN), "GR prepare status");
    REQUIRE(q4e_check_f32("GR mixed", b_mixed.contents, expected_mixed,
                          TOKEN * DIM), "GR mixed");
    REQUIRE(q4e_check_f32("GR injection", b_injection.contents,
                          expected_injection, TOKEN * STREAM), "GR injection");

    float block[TOKEN * DIM];
    float expected_residual[TOKEN * STREAM * DIM];
    memcpy(expected_residual, residual, sizeof(expected_residual));
    for (size_t i = 0u; i < TOKEN * DIM; i++)
        block[i] = ((float)(int)(i % 6u) - 2.0f) * 0.071f;
    for (size_t token = 0u; token < TOKEN; token++)
        REQUIRE(ds4_qwen4exp_ref_gr_apply_f32(
                    expected_residual + token * STREAM * DIM,
                    block + token * DIM,
                    expected_injection + token * STREAM, STREAM, DIM),
                "GR apply oracle");
    id<MTLBuffer> b_block = q4e_buffer(device, block, sizeof(block));
    memset(b_status.contents, 0, b_status.length);
    REQUIRE(q4e_dispatch(queue, cache, DS4_METAL_QWEN4EXP_GR_APPLY,
                         &gr, sizeof(gr),
                         @[b_residual, b_block, b_injection, b_status], @[], TOKEN),
            "GR apply dispatch");
    REQUIRE(q4e_status_ok(b_status.contents, TOKEN), "GR apply status");
    REQUIRE(q4e_check_f32("GR apply", b_residual.contents, expected_residual,
                          TOKEN * STREAM * DIM), "GR apply");

    float expected_final[TOKEN * DIM];
    for (size_t token = 0u; token < TOKEN; token++)
        REQUIRE(ds4_qwen4exp_ref_gr_final_mix_f32(
                    expected_final + token * DIM,
                    expected_residual + token * STREAM * DIM,
                    norm, down, up, STREAM, DIM, RANK, 1.0e-5f),
                "GR final oracle");
    id<MTLBuffer> b_final = q4e_buffer(device, NULL, sizeof(expected_final));
    memset(b_status.contents, 0, b_status.length);
    REQUIRE(q4e_dispatch(queue, cache, DS4_METAL_QWEN4EXP_GR_FINAL,
                         &gr, sizeof(gr),
                         @[b_residual, b_norm, b_down, b_up, b_final, b_status],
                         @[], TOKEN), "GR final dispatch");
    REQUIRE(q4e_status_ok(b_status.contents, TOKEN), "GR final status");
    REQUIRE(q4e_check_f32("GR final", b_final.contents, expected_final,
                          TOKEN * DIM), "GR final");
    return true;
}

static bool test_conv_controls_gdn(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        const ds4_metal_qwen4exp_cache *cache) {
    enum { N_TOKEN = 11, N_CHANNEL = 3, KERNEL = 4 };
    float conv_input[N_TOKEN * N_CHANNEL];
    float conv_weight[N_CHANNEL * KERNEL];
    float conv_initial[N_CHANNEL * (KERNEL - 1)];
    for (size_t i = 0u; i < N_TOKEN * N_CHANNEL; i++)
        conv_input[i] = ((float)(int)(i % 10u) - 4.0f) * 0.037f;
    for (size_t i = 0u; i < N_CHANNEL * KERNEL; i++)
        conv_weight[i] = ((float)(int)(i % 7u) - 3.0f) * 0.061f;
    for (size_t i = 0u; i < N_CHANNEL * (KERNEL - 1); i++)
        conv_initial[i] = ((float)(int)i - 3.0f) * 0.029f;
    float conv_expected[N_TOKEN * N_CHANNEL];
    float conv_state_expected[N_CHANNEL * (KERNEL - 1)];
    memcpy(conv_state_expected, conv_initial, sizeof(conv_initial));
    REQUIRE(ds4_qwen4exp_ref_causal_conv1d_silu_f32(
                conv_expected, conv_state_expected, conv_input, conv_weight,
                N_TOKEN, N_CHANNEL, KERNEL), "conv oracle");

    id<MTLBuffer> b_conv_input =
        q4e_buffer(device, conv_input, sizeof(conv_input));
    id<MTLBuffer> b_conv_weight =
        q4e_buffer(device, conv_weight, sizeof(conv_weight));
    for (uint32_t chunk_size = 1u; chunk_size <= 5u; chunk_size++) {
        id<MTLBuffer> b_state =
            q4e_buffer(device, conv_initial, sizeof(conv_initial));
        id<MTLBuffer> b_output =
            q4e_buffer(device, NULL, sizeof(conv_expected));
        id<MTLBuffer> b_status =
            q4e_buffer(device, NULL, N_CHANNEL * sizeof(uint32_t));
        float transition_state[N_CHANNEL * (KERNEL - 1)];
        memcpy(transition_state, conv_initial, sizeof(transition_state));
        REQUIRE(b_state && b_output && b_status, "conv buffers");
        for (uint32_t start = 0u; start < N_TOKEN; start += chunk_size) {
            const uint32_t count =
                chunk_size < N_TOKEN - start ? chunk_size : N_TOKEN - start;
            ds4_metal_qwen4exp_conv_args args = {
                .n_token = count, .n_channel = N_CHANNEL,
                .kernel_size = KERNEL, .dilation = 1u, .n_sequence = 1u,
            };
            memset(b_status.contents, 0, b_status.length);
            NSArray<NSNumber *> *offsets = @[
                @(start * N_CHANNEL * sizeof(float)), @0u, @0u,
                @(start * N_CHANNEL * sizeof(float)), @0u,
            ];
            REQUIRE(q4e_dispatch(
                        queue, cache, DS4_METAL_QWEN4EXP_CONV,
                        &args, sizeof(args),
                        @[b_conv_input, b_conv_weight, b_state, b_output,
                          b_status], offsets, N_CHANNEL),
                    "conv chunk %u", chunk_size);
            REQUIRE(q4e_status_ok(b_status.contents, N_CHANNEL),
                    "conv status chunk %u", chunk_size);
            if (chunk_size == 1u) {
                float transition_output[N_CHANNEL];
                REQUIRE(ds4_qwen4exp_ref_causal_conv1d_silu_f32(
                            transition_output, transition_state,
                            conv_input + start * N_CHANNEL, conv_weight,
                            1u, N_CHANNEL, KERNEL),
                        "conv transition oracle");
                REQUIRE(q4e_check_f32(
                            "conv transition", b_state.contents,
                            transition_state, N_CHANNEL * (KERNEL - 1)),
                        "conv state transition %u", start);
            }
        }
        REQUIRE(q4e_check_f32("conv output", b_output.contents, conv_expected,
                              N_TOKEN * N_CHANNEL), "conv output chunk %u",
                chunk_size);
        REQUIRE(q4e_check_f32("conv state", b_state.contents,
                              conv_state_expected,
                              N_CHANNEL * (KERNEL - 1)),
                "conv state chunk %u", chunk_size);
    }

    enum { KEY_HEAD = 16, VALUE_HEAD = 48, KEY_DIM = 3, VALUE_DIM = 2 };
    float alpha[N_TOKEN * VALUE_HEAD];
    float beta_logit[N_TOKEN * VALUE_HEAD];
    float a_log[VALUE_HEAD];
    float dt_bias[VALUE_HEAD];
    for (size_t i = 0u; i < N_TOKEN * VALUE_HEAD; i++) {
        alpha[i] = ((float)(int)(i % 9u) - 4.0f) * 0.07f;
        beta_logit[i] = ((float)(int)(i % 11u) - 5.0f) * 0.09f;
    }
    for (size_t i = 0u; i < VALUE_HEAD; i++) {
        a_log[i] = -1.3f + (float)(i % 7u) * 0.013f;
        dt_bias[i] = -0.4f + (float)(i % 5u) * 0.021f;
    }
    float decay_expected[N_TOKEN * VALUE_HEAD];
    float beta_expected[N_TOKEN * VALUE_HEAD];
    REQUIRE(ds4_qwen4exp_ref_gdn_controls_f32(
                decay_expected, beta_expected, alpha, beta_logit, a_log,
                dt_bias, N_TOKEN, VALUE_HEAD), "controls oracle");
    id<MTLBuffer> b_alpha = q4e_buffer(device, alpha, sizeof(alpha));
    id<MTLBuffer> b_beta_logit =
        q4e_buffer(device, beta_logit, sizeof(beta_logit));
    id<MTLBuffer> b_a_log = q4e_buffer(device, a_log, sizeof(a_log));
    id<MTLBuffer> b_dt_bias = q4e_buffer(device, dt_bias, sizeof(dt_bias));
    id<MTLBuffer> b_decay = q4e_buffer(device, NULL, sizeof(decay_expected));
    id<MTLBuffer> b_beta = q4e_buffer(device, NULL, sizeof(beta_expected));
    id<MTLBuffer> b_control_status =
        q4e_buffer(device, NULL, N_TOKEN * VALUE_HEAD * sizeof(uint32_t));
    ds4_metal_qwen4exp_controls_args control_args = {
        .n_token = N_TOKEN, .n_value_head = VALUE_HEAD,
    };
    REQUIRE(q4e_dispatch(
                queue, cache, DS4_METAL_QWEN4EXP_CONTROLS,
                &control_args, sizeof(control_args),
                @[b_alpha, b_beta_logit, b_a_log, b_dt_bias, b_decay, b_beta,
                  b_control_status], @[], N_TOKEN * VALUE_HEAD),
            "controls dispatch");
    REQUIRE(q4e_status_ok(b_control_status.contents, N_TOKEN * VALUE_HEAD),
            "controls status");
    REQUIRE(q4e_check_f32("log decay", b_decay.contents, decay_expected,
                          N_TOKEN * VALUE_HEAD), "log decay");
    REQUIRE(q4e_check_f32("beta", b_beta.contents, beta_expected,
                          N_TOKEN * VALUE_HEAD), "beta");

    float query[N_TOKEN * KEY_HEAD * KEY_DIM];
    float key[N_TOKEN * KEY_HEAD * KEY_DIM];
    float value[N_TOKEN * VALUE_HEAD * VALUE_DIM];
    float state_initial[VALUE_HEAD * KEY_DIM * VALUE_DIM];
    for (size_t token = 0u; token < N_TOKEN; token++) {
        for (size_t head = 0u; head < KEY_HEAD; head++) {
            for (size_t dim = 0u; dim < KEY_DIM; dim++) {
                const size_t index = (token * KEY_HEAD + head) * KEY_DIM + dim;
                /* Head sentinel makes every 16->48 mapping observable. */
                query[index] = 0.03f * (float)(head + 1u) +
                               0.007f * (float)(token + dim);
                key[index] = 0.019f * (float)(KEY_HEAD - head) +
                             0.005f * (float)(2u * token + dim);
            }
        }
    }
    for (size_t i = 0u; i < N_TOKEN * VALUE_HEAD * VALUE_DIM; i++)
        value[i] = ((float)(int)(i % 13u) - 6.0f) * 0.023f;
    for (size_t i = 0u; i < VALUE_HEAD * KEY_DIM * VALUE_DIM; i++)
        state_initial[i] = ((float)(int)(i % 17u) - 8.0f) * 0.004f;
    float expected_output[N_TOKEN * VALUE_HEAD * VALUE_DIM];
    float expected_state[VALUE_HEAD * KEY_DIM * VALUE_DIM];
    memcpy(expected_state, state_initial, sizeof(expected_state));
    REQUIRE(ds4_qwen4exp_ref_gdn_f32(
                expected_output, expected_state, query, key, value,
                decay_expected, beta_expected, N_TOKEN, KEY_HEAD, VALUE_HEAD,
                KEY_DIM, VALUE_DIM), "GDN oracle");
    id<MTLBuffer> b_query = q4e_buffer(device, query, sizeof(query));
    id<MTLBuffer> b_key = q4e_buffer(device, key, sizeof(key));
    id<MTLBuffer> b_value = q4e_buffer(device, value, sizeof(value));
    REQUIRE(b_query && b_key && b_value, "GDN input buffers");
    for (uint32_t chunk_size = 1u; chunk_size <= 5u; chunk_size++) {
        id<MTLBuffer> b_state =
            q4e_buffer(device, state_initial, sizeof(state_initial));
        id<MTLBuffer> b_output =
            q4e_buffer(device, NULL, sizeof(expected_output));
        id<MTLBuffer> b_status = q4e_buffer(
            device, NULL, VALUE_HEAD * VALUE_DIM * sizeof(uint32_t));
        float transition_state[VALUE_HEAD * KEY_DIM * VALUE_DIM];
        memcpy(transition_state, state_initial, sizeof(transition_state));
        REQUIRE(b_state && b_output && b_status, "GDN output buffers");
        for (uint32_t start = 0u; start < N_TOKEN; start += chunk_size) {
            const uint32_t count =
                chunk_size < N_TOKEN - start ? chunk_size : N_TOKEN - start;
            ds4_metal_qwen4exp_gdn_args args = {
                .n_token = count, .n_key_head = KEY_HEAD,
                .n_value_head = VALUE_HEAD, .key_dim = KEY_DIM,
                .value_dim = VALUE_DIM,
            };
            memset(b_status.contents, 0, b_status.length);
            NSArray<NSNumber *> *offsets = @[
                @(start * KEY_HEAD * KEY_DIM * sizeof(float)),
                @(start * KEY_HEAD * KEY_DIM * sizeof(float)),
                @(start * VALUE_HEAD * VALUE_DIM * sizeof(float)),
                @(start * VALUE_HEAD * sizeof(float)),
                @(start * VALUE_HEAD * sizeof(float)), @0u,
                @(start * VALUE_HEAD * VALUE_DIM * sizeof(float)), @0u,
            ];
            REQUIRE(q4e_dispatch(
                        queue, cache, DS4_METAL_QWEN4EXP_GDN,
                        &args, sizeof(args),
                        @[b_query, b_key, b_value, b_decay, b_beta, b_state,
                          b_output, b_status], offsets,
                        VALUE_HEAD * VALUE_DIM),
                    "GDN chunk %u", chunk_size);
            REQUIRE(q4e_status_ok(b_status.contents, VALUE_HEAD * VALUE_DIM),
                    "GDN status chunk %u", chunk_size);
            if (chunk_size == 1u) {
                float transition_output[VALUE_HEAD * VALUE_DIM];
                REQUIRE(ds4_qwen4exp_ref_gdn_f32(
                            transition_output, transition_state,
                            query + start * KEY_HEAD * KEY_DIM,
                            key + start * KEY_HEAD * KEY_DIM,
                            value + start * VALUE_HEAD * VALUE_DIM,
                            decay_expected + start * VALUE_HEAD,
                            beta_expected + start * VALUE_HEAD,
                            1u, KEY_HEAD, VALUE_HEAD, KEY_DIM, VALUE_DIM),
                        "GDN transition oracle");
                REQUIRE(q4e_check_f32(
                            "GDN transition", b_state.contents,
                            transition_state,
                            VALUE_HEAD * KEY_DIM * VALUE_DIM),
                        "GDN state transition %u", start);
            }
        }
        REQUIRE(q4e_check_f32("GDN output", b_output.contents,
                              expected_output,
                              N_TOKEN * VALUE_HEAD * VALUE_DIM),
                "GDN output chunk %u", chunk_size);
        REQUIRE(q4e_check_f32("GDN state", b_state.contents, expected_state,
                              VALUE_HEAD * KEY_DIM * VALUE_DIM),
                "GDN state chunk %u", chunk_size);
    }

    /* Explicit private-buffer transaction: an injected completion failure must
     * not publish the successfully-computed private state. */
    id<MTLBuffer> public_state =
        q4e_buffer(device, state_initial, sizeof(state_initial));
    ds4_metal_qwen4exp_state_transaction oom_transaction = {
        .public_state = public_state,
    };
    NSError *allocation_error = nil;
    REQUIRE(!ds4_metal_qwen4exp_prepare_state_transaction(
                device, public_state, sizeof(state_initial),
                sizeof(state_initial) - 1u, &oom_transaction,
                &allocation_error),
            "forced state OOM accepted");
    REQUIRE(allocation_error != nil, "forced state OOM missing diagnostic");
    REQUIRE(oom_transaction.public_state == public_state &&
                oom_transaction.private_state == nil,
            "forced state OOM changed transaction ownership");
    REQUIRE(memcmp(public_state.contents, state_initial,
                   sizeof(state_initial)) == 0,
            "forced state OOM changed public state");
    ds4_metal_qwen4exp_state_transaction transaction = {0};
    REQUIRE(ds4_metal_qwen4exp_prepare_state_transaction(
                device, public_state, sizeof(state_initial),
                sizeof(state_initial), &transaction, &allocation_error),
            "private state preparation: %s",
            allocation_error.localizedDescription.UTF8String);
    id<MTLBuffer> private_state = transaction.private_state;
    id<MTLBuffer> rollback_output = q4e_buffer(
        device, NULL, VALUE_HEAD * VALUE_DIM * sizeof(float));
    id<MTLBuffer> rollback_status = q4e_buffer(
        device, NULL, VALUE_HEAD * VALUE_DIM * sizeof(uint32_t));
    ds4_metal_qwen4exp_gdn_args one = {
        .n_token = 1u, .n_key_head = KEY_HEAD, .n_value_head = VALUE_HEAD,
        .key_dim = KEY_DIM, .value_dim = VALUE_DIM,
    };
    REQUIRE(q4e_dispatch(
                queue, cache, DS4_METAL_QWEN4EXP_GDN, &one, sizeof(one),
                @[b_query, b_key, b_value, b_decay, b_beta, private_state,
                  rollback_output, rollback_status], @[],
                VALUE_HEAD * VALUE_DIM), "rollback private GDN");
    REQUIRE(q4e_status_ok(rollback_status.contents, VALUE_HEAD * VALUE_DIM),
            "rollback status");
    REQUIRE(memcmp(private_state.contents, state_initial,
                   sizeof(state_initial)) != 0,
            "private state did not advance");
    REQUIRE(!ds4_metal_qwen4exp_publish_state(
                &transaction, false, rollback_status.contents,
                VALUE_HEAD * VALUE_DIM),
            "forced command failure published state");
    REQUIRE(transaction.public_state == public_state,
            "forced failure changed public buffer identity");
    REQUIRE(memcmp(transaction.public_state.contents, state_initial,
                   sizeof(state_initial)) == 0,
            "forced failure changed public state bytes");
    id<MTLCommandBuffer> not_completed = [queue commandBuffer];
    REQUIRE(!ds4_metal_qwen4exp_publish_state_after_command(
                &transaction, not_completed, rollback_status.contents,
                VALUE_HEAD * VALUE_DIM),
            "non-completed command published state");
    REQUIRE(transaction.public_state == public_state,
            "non-completed command swapped public state");
    id<MTLCommandBuffer> completed = [queue commandBuffer];
    [completed commit];
    [completed waitUntilCompleted];
    REQUIRE(completed.status == MTLCommandBufferStatusCompleted,
            "rollback completion control failed");
    REQUIRE(ds4_metal_qwen4exp_publish_state_after_command(
                &transaction, completed, rollback_status.contents,
                VALUE_HEAD * VALUE_DIM),
            "successful command did not publish state");
    REQUIRE(transaction.public_state == private_state,
            "success did not swap private state");
    return true;
}

static bool test_shared_conv_row(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        const ds4_metal_qwen4exp_cache *cache) {
    /* The GDN projection/conv channel count comes from the actual Q/K/V
     * segments, not a generic SSM field: 2*key_heads*key_dim +
     * value_heads*value_dim. Keep every segment non-zero and distinguishable. */
    enum {
        G_KEY_HEAD = 2, G_KEY_DIM = 3, G_VALUE_HEAD = 6, G_VALUE_DIM = 2,
        G_CHANNEL = 2 * G_KEY_HEAD * G_KEY_DIM + G_VALUE_HEAD * G_VALUE_DIM,
        G_TOKEN = 2, G_KERNEL = 4, G_HISTORY = G_KERNEL - 1,
        P_CHANNEL = 4, P_TOKEN = 2, P_KERNEL = 4, P_DILATION = 3,
        P_HISTORY = (P_KERNEL - 1) * P_DILATION,
        G_STATE = G_CHANNEL * G_HISTORY,
        P_STATE = P_CHANNEL * P_HISTORY,
        ROW_STATE = G_STATE + P_STATE,
    };
    _Static_assert(G_CHANNEL == 24, "GDN projection segment fixture drift");
    float g_input[G_TOKEN * G_CHANNEL];
    float g_weight[G_CHANNEL * G_KERNEL];
    float p_input[P_TOKEN * P_CHANNEL];
    float p_weight[P_CHANNEL * P_KERNEL];
    float initial[ROW_STATE];
    for (size_t i = 0u; i < sizeof(g_input) / sizeof(float); i++)
        g_input[i] = 0.01f * (float)(i + 1u);
    for (size_t i = 0u; i < sizeof(g_weight) / sizeof(float); i++)
        g_weight[i] = ((float)(int)(i % 7u) - 3.0f) * 0.023f;
    for (size_t i = 0u; i < sizeof(p_input) / sizeof(float); i++)
        p_input[i] = ((float)(int)(i % 5u) - 2.0f) * 0.047f;
    for (size_t i = 0u; i < sizeof(p_weight) / sizeof(float); i++)
        p_weight[i] = ((float)(int)(i % 6u) - 2.0f) * 0.031f;
    for (size_t i = 0u; i < ROW_STATE; i++)
        initial[i] = ((float)(int)(i % 13u) - 6.0f) * 0.009f;
    float expected_state[ROW_STATE];
    float expected_g_output[G_TOKEN * G_CHANNEL];
    float expected_p_output[P_TOKEN * P_CHANNEL];
    memcpy(expected_state, initial, sizeof(initial));
    REQUIRE(ds4_qwen4exp_ref_causal_conv1d_silu_f32(
                expected_g_output, expected_state, g_input, g_weight,
                G_TOKEN, G_CHANNEL, G_KERNEL),
            "shared-row GDN conv oracle");
    REQUIRE(ds4_qwen4exp_ref_dilated_conv1d_silu_f32(
                expected_p_output, expected_state + G_STATE,
                p_input, p_weight, P_TOKEN, P_CHANNEL, P_KERNEL, P_DILATION),
            "shared-row PLE conv oracle");

    id<MTLBuffer> public_row = q4e_buffer(device, initial, sizeof(initial));
    ds4_metal_qwen4exp_state_transaction transaction = {0};
    NSError *error = nil;
    REQUIRE(ds4_metal_qwen4exp_prepare_state_transaction(
                device, public_row, sizeof(initial), sizeof(initial),
                &transaction, &error),
            "shared-row snapshot: %s", error.localizedDescription.UTF8String);
    id<MTLBuffer> b_g_input = q4e_buffer(device, g_input, sizeof(g_input));
    id<MTLBuffer> b_g_weight = q4e_buffer(device, g_weight, sizeof(g_weight));
    id<MTLBuffer> b_g_output =
        q4e_buffer(device, NULL, sizeof(expected_g_output));
    id<MTLBuffer> b_g_status =
        q4e_buffer(device, NULL, G_CHANNEL * sizeof(uint32_t));
    ds4_metal_qwen4exp_conv_args g_args = {
        .n_token = G_TOKEN, .n_channel = G_CHANNEL,
        .kernel_size = G_KERNEL, .dilation = 1u, .n_sequence = 1u,
    };
    REQUIRE(q4e_dispatch(
                queue, cache, DS4_METAL_QWEN4EXP_CONV, &g_args, sizeof(g_args),
                @[b_g_input, b_g_weight, transaction.private_state,
                  b_g_output, b_g_status], @[@0u, @0u, @0u, @0u, @0u],
                G_CHANNEL), "shared-row GDN conv");

    id<MTLBuffer> b_p_input = q4e_buffer(device, p_input, sizeof(p_input));
    id<MTLBuffer> b_p_weight = q4e_buffer(device, p_weight, sizeof(p_weight));
    id<MTLBuffer> b_p_output =
        q4e_buffer(device, NULL, sizeof(expected_p_output));
    id<MTLBuffer> b_p_status =
        q4e_buffer(device, NULL, P_CHANNEL * sizeof(uint32_t));
    ds4_metal_qwen4exp_conv_args p_args = {
        .n_token = P_TOKEN, .n_channel = P_CHANNEL,
        .kernel_size = P_KERNEL, .dilation = P_DILATION, .n_sequence = 1u,
    };
    REQUIRE(q4e_dispatch(
                queue, cache, DS4_METAL_QWEN4EXP_CONV, &p_args, sizeof(p_args),
                @[b_p_input, b_p_weight, transaction.private_state,
                  b_p_output, b_p_status],
                @[@0u, @0u, @(G_STATE * sizeof(float)), @0u, @0u],
                P_CHANNEL), "shared-row PLE conv");
    REQUIRE(q4e_status_ok(b_g_status.contents, G_CHANNEL) &&
                q4e_status_ok(b_p_status.contents, P_CHANNEL),
            "shared-row conv status");
    REQUIRE(q4e_check_f32("shared-row GDN output", b_g_output.contents,
                          expected_g_output, G_TOKEN * G_CHANNEL),
            "shared-row GDN output");
    REQUIRE(q4e_check_f32("shared-row PLE output", b_p_output.contents,
                          expected_p_output, P_TOKEN * P_CHANNEL),
            "shared-row PLE output");
    REQUIRE(q4e_check_f32("shared-row private state",
                          transaction.private_state.contents,
                          expected_state, ROW_STATE),
            "shared-row private state");
    REQUIRE(memcmp(transaction.public_state.contents, initial,
                   sizeof(initial)) == 0,
            "shared-row kernels mutated public snapshot");
    uint32_t combined_status[G_CHANNEL + P_CHANNEL];
    memcpy(combined_status, b_g_status.contents,
           G_CHANNEL * sizeof(uint32_t));
    memcpy(combined_status + G_CHANNEL, b_p_status.contents,
           P_CHANNEL * sizeof(uint32_t));
    REQUIRE(ds4_metal_qwen4exp_publish_state(
                &transaction, true, combined_status,
                G_CHANNEL + P_CHANNEL),
            "shared-row publish");
    REQUIRE(q4e_check_f32("shared-row published state",
                          transaction.public_state.contents,
                          expected_state, ROW_STATE),
            "shared-row published state");

    /* PLE history is per sequence. The tensor is contiguous by sequence before
     * the conv reshape, so two sequences with different histories cannot bleed. */
    enum {
        M_SEQUENCE = 2, M_TOKEN = 3, M_CHANNEL = 2, M_KERNEL = 4,
        M_DILATION = 3, M_HISTORY = (M_KERNEL - 1) * M_DILATION,
    };
    float m_input[M_SEQUENCE * M_TOKEN * M_CHANNEL];
    float m_weight[M_CHANNEL * M_KERNEL];
    float m_initial[M_SEQUENCE * M_CHANNEL * M_HISTORY];
    float m_expected[M_SEQUENCE * M_TOKEN * M_CHANNEL];
    float m_state_expected[M_SEQUENCE * M_CHANNEL * M_HISTORY];
    for (size_t i = 0u; i < sizeof(m_input) / sizeof(float); i++)
        m_input[i] = ((float)(int)(i % 9u) - 4.0f) * 0.041f;
    for (size_t i = 0u; i < sizeof(m_weight) / sizeof(float); i++)
        m_weight[i] = ((float)(int)(i % 7u) - 3.0f) * 0.029f;
    for (size_t i = 0u; i < sizeof(m_initial) / sizeof(float); i++)
        m_initial[i] = ((float)(int)i - 12.0f) * 0.006f;
    memcpy(m_state_expected, m_initial, sizeof(m_initial));
    for (size_t sequence = 0u; sequence < M_SEQUENCE; sequence++)
        REQUIRE(ds4_qwen4exp_ref_dilated_conv1d_silu_f32(
                    m_expected + sequence * M_TOKEN * M_CHANNEL,
                    m_state_expected + sequence * M_CHANNEL * M_HISTORY,
                    m_input + sequence * M_TOKEN * M_CHANNEL, m_weight,
                    M_TOKEN, M_CHANNEL, M_KERNEL, M_DILATION),
                "multi-sequence PLE conv oracle");
    id<MTLBuffer> b_m_input = q4e_buffer(device, m_input, sizeof(m_input));
    id<MTLBuffer> b_m_weight = q4e_buffer(device, m_weight, sizeof(m_weight));
    id<MTLBuffer> b_m_state =
        q4e_buffer(device, m_initial, sizeof(m_initial));
    id<MTLBuffer> b_m_output = q4e_buffer(device, NULL, sizeof(m_expected));
    id<MTLBuffer> b_m_status = q4e_buffer(
        device, NULL, M_SEQUENCE * M_CHANNEL * sizeof(uint32_t));
    ds4_metal_qwen4exp_conv_args m_args = {
        .n_token = M_TOKEN, .n_channel = M_CHANNEL,
        .kernel_size = M_KERNEL, .dilation = M_DILATION,
        .n_sequence = M_SEQUENCE,
    };
    REQUIRE(q4e_dispatch(
                queue, cache, DS4_METAL_QWEN4EXP_CONV, &m_args, sizeof(m_args),
                @[b_m_input, b_m_weight, b_m_state, b_m_output, b_m_status],
                @[], M_SEQUENCE * M_CHANNEL),
            "multi-sequence PLE conv");
    REQUIRE(q4e_status_ok(b_m_status.contents, M_SEQUENCE * M_CHANNEL),
            "multi-sequence PLE status");
    REQUIRE(q4e_check_f32("multi-sequence PLE output", b_m_output.contents,
                          m_expected, M_SEQUENCE * M_TOKEN * M_CHANNEL),
            "multi-sequence PLE output");
    REQUIRE(q4e_check_f32("multi-sequence PLE state", b_m_state.contents,
                          m_state_expected,
                          M_SEQUENCE * M_CHANNEL * M_HISTORY),
            "multi-sequence PLE state");
    return true;
}

static float cpu_sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    const float e = expf(x);
    return e / (1.0f + e);
}

static float cpu_silu(float x) {
    return x * cpu_sigmoid(x);
}

static bool run_router_case(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        const ds4_metal_qwen4exp_cache *cache,
        const float logits[512],
        bool expect_success) {
    uint32_t expected_id[10];
    float expected_weight[10];
    const bool oracle = ds4_qwen4exp_ref_softmax_topk_f32(
        expected_id, expected_weight, logits, 512u, 10u);
    REQUIRE(oracle == expect_success, "router oracle expectation");
    uint32_t id_sentinel[10];
    float weight_sentinel[10];
    for (size_t i = 0u; i < 10u; i++) {
        id_sentinel[i] = UINT32_C(0xdeadbeef);
        weight_sentinel[i] = -77.0f;
    }
    id<MTLBuffer> b_logits = q4e_buffer(device, logits, 512u * sizeof(float));
    id<MTLBuffer> b_id = q4e_buffer(device, id_sentinel, sizeof(id_sentinel));
    id<MTLBuffer> b_weight =
        q4e_buffer(device, weight_sentinel, sizeof(weight_sentinel));
    id<MTLBuffer> b_status = q4e_buffer(device, NULL, sizeof(uint32_t));
    ds4_metal_qwen4exp_router_args args = {
        .n_token = 1u, .n_expert = 512u, .n_selected = 10u,
    };
    REQUIRE(q4e_dispatch(queue, cache, DS4_METAL_QWEN4EXP_ROUTER,
                         &args, sizeof(args),
                         @[b_logits, b_id, b_weight, b_status], @[], 1u),
            "router dispatch");
    const uint32_t status = *(const uint32_t *)b_status.contents;
    if (!expect_success) {
        REQUIRE(status != 0u, "router accepted nonfinite input");
        REQUIRE(memcmp(b_id.contents, id_sentinel, sizeof(id_sentinel)) == 0,
                "router changed IDs on failure");
        REQUIRE(memcmp(b_weight.contents, weight_sentinel,
                       sizeof(weight_sentinel)) == 0,
                "router changed weights on failure");
        return true;
    }
    REQUIRE(status == 0u, "router status %u", status);
    REQUIRE(memcmp(b_id.contents, expected_id, sizeof(expected_id)) == 0,
            "router IDs differ");
    REQUIRE(q4e_check_f32("router weights", b_weight.contents,
                          expected_weight, 10u), "router weights");
    float sum = 0.0f;
    for (size_t i = 0u; i < 10u; i++) sum += ((float *)b_weight.contents)[i];
    REQUIRE(q4e_close(sum, 1.0f), "router renormalization %.9g", sum);
    return true;
}

static void cpu_moe(
        float *output,
        const float *input,
        const uint32_t *selected,
        const float *selected_weight,
        const float *expert_gate,
        const float *expert_up,
        const float *expert_down,
        const float *shared_gate,
        const float *shared_up,
        const float *shared_down,
        const float *shared_router,
        size_t n_selected,
        size_t input_dim,
        size_t expert_dim,
        size_t output_dim) {
    float shared_logit = 0.0f;
    for (size_t i = 0u; i < input_dim; i++)
        shared_logit += input[i] * shared_router[i];
    const float shared_scale = cpu_sigmoid(shared_logit);
    for (size_t out = 0u; out < output_dim; out++) {
        float routed = 0.0f;
        for (size_t route = 0u; route < n_selected; route++) {
            const size_t expert = selected[route];
            float expert_result = 0.0f;
            for (size_t inner = 0u; inner < expert_dim; inner++) {
                float gate = 0.0f;
                float up = 0.0f;
                const size_t base =
                    (expert * expert_dim + inner) * input_dim;
                for (size_t i = 0u; i < input_dim; i++) {
                    gate += expert_gate[base + i] * input[i];
                    up += expert_up[base + i] * input[i];
                }
                expert_result += expert_down[
                    (expert * output_dim + out) * expert_dim + inner] *
                    cpu_silu(gate) * up;
            }
            routed += selected_weight[route] * expert_result;
        }
        float shared = 0.0f;
        for (size_t inner = 0u; inner < expert_dim; inner++) {
            float gate = 0.0f;
            float up = 0.0f;
            for (size_t i = 0u; i < input_dim; i++) {
                gate += shared_gate[inner * input_dim + i] * input[i];
                up += shared_up[inner * input_dim + i] * input[i];
            }
            shared += shared_down[out * expert_dim + inner] *
                cpu_silu(gate) * up;
        }
        output[out] = routed + shared_scale * shared;
    }
}

static bool test_router_moe(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        const ds4_metal_qwen4exp_cache *cache) {
    float logits[512];
    for (size_t i = 0u; i < 512u; i++) logits[i] = 0.0f;
    REQUIRE(run_router_case(device, queue, cache, logits, true),
            "equal router");
    for (size_t i = 0u; i < 512u; i++)
        logits[i] = ((float)(int)((i * 37u) % 101u) - 50.0f) * 3.0f;
    logits[511] = 1000.0f;
    logits[17] = 999.0f;
    REQUIRE(run_router_case(device, queue, cache, logits, true),
            "extreme router");
    logits[73] = NAN;
    REQUIRE(!ds4_metal_qwen4exp_finite_f32(logits, 512u),
            "host finite preflight accepted NaN");
    REQUIRE(run_router_case(device, queue, cache, logits, false),
            "NaN router");
    logits[73] = INFINITY;
    REQUIRE(!ds4_metal_qwen4exp_finite_f32(logits, 512u),
            "host finite preflight accepted Inf");
    REQUIRE(run_router_case(device, queue, cache, logits, false),
            "Inf router");

    enum { EXPERT = 4, SELECTED = 3, INPUT = 3, INNER = 4, OUTPUT = 3 };
    const float input[INPUT] = { -0.4f, 0.7f, 0.2f };
    const uint32_t selected[SELECTED] = { 3u, 0u, 2u };
    const float selected_weight[SELECTED] = { 0.2f, 0.5f, 0.3f };
    float expert_gate[EXPERT * INNER * INPUT];
    float expert_up[EXPERT * INNER * INPUT];
    float expert_down[EXPERT * OUTPUT * INNER];
    float shared_gate[INNER * INPUT];
    float shared_up[INNER * INPUT];
    float shared_down[OUTPUT * INNER];
    const float shared_router[INPUT] = { 0.31f, -0.17f, 0.09f };
    for (size_t i = 0u; i < EXPERT * INNER * INPUT; i++) {
        expert_gate[i] = ((float)(int)(i % 9u) - 4.0f) * 0.043f;
        expert_up[i] = ((float)(int)(i % 7u) - 3.0f) * 0.052f;
    }
    for (size_t i = 0u; i < EXPERT * OUTPUT * INNER; i++)
        expert_down[i] = ((float)(int)(i % 11u) - 5.0f) * 0.027f;
    for (size_t i = 0u; i < INNER * INPUT; i++) {
        shared_gate[i] = ((float)(int)(i % 5u) - 2.0f) * 0.071f;
        shared_up[i] = ((float)(int)(i % 8u) - 3.0f) * 0.039f;
    }
    for (size_t i = 0u; i < OUTPUT * INNER; i++)
        shared_down[i] = ((float)(int)(i % 6u) - 2.0f) * 0.046f;
    float expected[OUTPUT];
    cpu_moe(expected, input, selected, selected_weight, expert_gate, expert_up,
            expert_down, shared_gate, shared_up, shared_down, shared_router,
            SELECTED, INPUT, INNER, OUTPUT);
    id<MTLBuffer> b_input = q4e_buffer(device, input, sizeof(input));
    id<MTLBuffer> b_selected =
        q4e_buffer(device, selected, sizeof(selected));
    id<MTLBuffer> b_selected_weight =
        q4e_buffer(device, selected_weight, sizeof(selected_weight));
    id<MTLBuffer> b_expert_gate =
        q4e_buffer(device, expert_gate, sizeof(expert_gate));
    id<MTLBuffer> b_expert_up =
        q4e_buffer(device, expert_up, sizeof(expert_up));
    id<MTLBuffer> b_expert_down =
        q4e_buffer(device, expert_down, sizeof(expert_down));
    id<MTLBuffer> b_shared_gate =
        q4e_buffer(device, shared_gate, sizeof(shared_gate));
    id<MTLBuffer> b_shared_up =
        q4e_buffer(device, shared_up, sizeof(shared_up));
    id<MTLBuffer> b_shared_down =
        q4e_buffer(device, shared_down, sizeof(shared_down));
    id<MTLBuffer> b_shared_router =
        q4e_buffer(device, shared_router, sizeof(shared_router));
    id<MTLBuffer> b_output = q4e_buffer(device, NULL, sizeof(expected));
    id<MTLBuffer> b_status = q4e_buffer(device, NULL, sizeof(uint32_t));
    ds4_metal_qwen4exp_moe_args args = {
        .n_token = 1u, .n_expert = EXPERT, .n_selected = SELECTED,
        .input_dim = INPUT, .expert_dim = INNER, .output_dim = OUTPUT,
    };
    REQUIRE(q4e_dispatch(
                queue, cache, DS4_METAL_QWEN4EXP_MOE, &args, sizeof(args),
                @[b_input, b_selected, b_selected_weight, b_expert_gate,
                  b_expert_up, b_expert_down, b_shared_gate, b_shared_up,
                  b_shared_down, b_shared_router, b_output, b_status],
                @[], 1u), "MoE dispatch");
    REQUIRE(q4e_status_ok(b_status.contents, 1u), "MoE status");
    REQUIRE(q4e_check_f32("MoE", b_output.contents, expected, OUTPUT),
            "MoE output");
    return true;
}

static void cpu_attention(
        float *output,
        const float *query,
        const float *key,
        const float *value,
        const uint32_t *position,
        size_t n_position,
        size_t query_head,
        size_t n_query_head,
        size_t n_kv_head,
        size_t head_dim) {
    const size_t kv_head = query_head / (n_query_head / n_kv_head);
    float maximum = -INFINITY;
    for (size_t slot = 0u; slot < n_position; slot++) {
        float dot = 0.0f;
        const size_t base =
            (position[slot] * n_kv_head + kv_head) * head_dim;
        for (size_t i = 0u; i < head_dim; i++)
            dot += query[query_head * head_dim + i] * key[base + i];
        const float score = dot / sqrtf((float)head_dim);
        if (score > maximum) maximum = score;
    }
    float denominator = 0.0f;
    for (size_t slot = 0u; slot < n_position; slot++) {
        float dot = 0.0f;
        const size_t base =
            (position[slot] * n_kv_head + kv_head) * head_dim;
        for (size_t i = 0u; i < head_dim; i++)
            dot += query[query_head * head_dim + i] * key[base + i];
        denominator += expf(dot / sqrtf((float)head_dim) - maximum);
    }
    for (size_t i = 0u; i < head_dim; i++) {
        float total = 0.0f;
        for (size_t slot = 0u; slot < n_position; slot++) {
            float dot = 0.0f;
            const size_t base =
                (position[slot] * n_kv_head + kv_head) * head_dim;
            for (size_t j = 0u; j < head_dim; j++)
                dot += query[query_head * head_dim + j] * key[base + j];
            total += expf(dot / sqrtf((float)head_dim) - maximum) /
                denominator * value[base + i];
        }
        output[query_head * head_dim + i] = total;
    }
}

static bool test_qsa(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        const ds4_metal_qwen4exp_cache *cache) {
    enum {
        GROUP = 3, COMPRESSION = 4, INDEX_DIM = 8, N_ROT = 4,
        N_QUERY = 2, INDEX_HEAD = 2, MAX_WIDTH = 12,
    };
    float raw_key[GROUP * COMPRESSION * INDEX_DIM];
    float norm[INDEX_DIM];
    for (size_t i = 0u; i < sizeof(raw_key) / sizeof(raw_key[0]); i++)
        raw_key[i] = ((float)(int)(i % 17u) - 8.0f) * 0.031f;
    for (size_t i = 0u; i < INDEX_DIM; i++)
        norm[i] = ((float)(int)i - 3.0f) * 0.018f;
    float group_expected[GROUP * INDEX_DIM];
    REQUIRE(ds4_qwen4exp_ref_qsa_group_keys_f32(
                group_expected, raw_key, norm, GROUP, COMPRESSION, INDEX_DIM,
                N_ROT, 10000.0f, 1.0e-5f), "QSA group oracle");
    id<MTLBuffer> b_raw = q4e_buffer(device, raw_key, sizeof(raw_key));
    uint32_t logical_slot[GROUP * COMPRESSION];
    uint32_t logical_position[GROUP * COMPRESSION];
    for (uint32_t i = 0u; i < GROUP * COMPRESSION; i++) {
        logical_slot[i] = i;
        logical_position[i] = i;
    }
    id<MTLBuffer> b_logical_slot =
        q4e_buffer(device, logical_slot, sizeof(logical_slot));
    id<MTLBuffer> b_logical_position =
        q4e_buffer(device, logical_position, sizeof(logical_position));
    id<MTLBuffer> b_norm = q4e_buffer(device, norm, sizeof(norm));
    id<MTLBuffer> b_group =
        q4e_buffer(device, NULL, sizeof(group_expected));
    id<MTLBuffer> b_group_status =
        q4e_buffer(device, NULL, GROUP * sizeof(uint32_t));
    ds4_metal_qwen4exp_qsa_group_args group_args = {
        .n_group = GROUP, .compression = COMPRESSION, .head_dim = INDEX_DIM,
        .n_rot = N_ROT, .theta = 10000.0f, .epsilon = 1.0e-5f,
        .position0 = 0u, .n_slot = GROUP * COMPRESSION,
    };
    REQUIRE(q4e_dispatch(queue, cache, DS4_METAL_QWEN4EXP_QSA_GROUP,
                         &group_args, sizeof(group_args),
                         @[b_raw, b_logical_slot, b_logical_position, b_norm,
                           b_group, b_group_status], @[], GROUP),
            "QSA group dispatch");
    REQUIRE(q4e_status_ok(b_group_status.contents, GROUP), "QSA group status");
    REQUIRE(q4e_check_f32("QSA groups", b_group.contents, group_expected,
                          GROUP * INDEX_DIM), "QSA groups");

    /* Finite inputs may still overflow a reduction.  The group output is a
     * transaction: a non-finite intermediate must reject without replacing
     * even one element. */
    {
        float overflow_raw[COMPRESSION * INDEX_DIM];
        float overflow_norm[INDEX_DIM] = {0.0f};
        float sentinel[INDEX_DIM];
        uint32_t overflow_slot[COMPRESSION];
        uint32_t overflow_position[COMPRESSION];
        for (size_t i = 0u; i < COMPRESSION * INDEX_DIM; i++)
            overflow_raw[i] = FLT_MAX;
        for (size_t i = 0u; i < INDEX_DIM; i++) sentinel[i] = 19.25f;
        for (uint32_t i = 0u; i < COMPRESSION; i++) {
            overflow_slot[i] = i;
            overflow_position[i] = i;
        }
        id<MTLBuffer> b_overflow_raw =
            q4e_buffer(device, overflow_raw, sizeof(overflow_raw));
        id<MTLBuffer> b_overflow_norm =
            q4e_buffer(device, overflow_norm, sizeof(overflow_norm));
        id<MTLBuffer> b_overflow_slot =
            q4e_buffer(device, overflow_slot, sizeof(overflow_slot));
        id<MTLBuffer> b_overflow_position =
            q4e_buffer(device, overflow_position, sizeof(overflow_position));
        id<MTLBuffer> b_overflow_group =
            q4e_buffer(device, sentinel, sizeof(sentinel));
        id<MTLBuffer> b_overflow_status =
            q4e_buffer(device, NULL, sizeof(uint32_t));
        ds4_metal_qwen4exp_qsa_group_args overflow_args = group_args;
        overflow_args.n_group = 1u;
        overflow_args.n_slot = COMPRESSION;
        REQUIRE(q4e_dispatch(
                    queue, cache, DS4_METAL_QWEN4EXP_QSA_GROUP,
                    &overflow_args, sizeof(overflow_args),
                    @[b_overflow_raw, b_overflow_slot, b_overflow_position,
                      b_overflow_norm, b_overflow_group, b_overflow_status],
                    @[], 1u), "QSA finite-overflow rejection dispatch");
        REQUIRE(*(uint32_t *)b_overflow_status.contents != 0u,
                "QSA accepted a non-finite finite-input reduction");
        REQUIRE(memcmp(b_overflow_group.contents, sentinel,
                       sizeof(sentinel)) == 0,
                "QSA finite-overflow rejection changed group output");
    }

    /* Upstream's hybrid-index bug only surfaced after attention-cache cells
     * moved. Prove that grouping follows the main cache's logical slot map,
     * never physical adjacency or an independently chosen index-cache layout. */
    enum { PHYSICAL_SLOT = 16 };
    const uint32_t permuted_slot[GROUP * COMPRESSION] = {
        7u, 2u, 15u, 0u, 8u, 3u, 14u, 1u, 9u, 4u, 13u, 5u,
    };
    float physical_raw[PHYSICAL_SLOT * INDEX_DIM];
    memset(physical_raw, 0, sizeof(physical_raw));
    for (size_t logical = 0u; logical < GROUP * COMPRESSION; logical++)
        memcpy(physical_raw + permuted_slot[logical] * INDEX_DIM,
               raw_key + logical * INDEX_DIM, INDEX_DIM * sizeof(float));
    id<MTLBuffer> b_physical_raw =
        q4e_buffer(device, physical_raw, sizeof(physical_raw));
    id<MTLBuffer> b_permuted_slot =
        q4e_buffer(device, permuted_slot, sizeof(permuted_slot));
    memset(b_group.contents, 0, b_group.length);
    memset(b_group_status.contents, 0, b_group_status.length);
    group_args.n_slot = PHYSICAL_SLOT;
    REQUIRE(q4e_dispatch(
                queue, cache, DS4_METAL_QWEN4EXP_QSA_GROUP,
                &group_args, sizeof(group_args),
                @[b_physical_raw, b_permuted_slot, b_logical_position, b_norm,
                  b_group, b_group_status], @[], GROUP),
            "QSA permuted-slot group dispatch");
    REQUIRE(q4e_status_ok(b_group_status.contents, GROUP),
            "QSA permuted-slot status");
    REQUIRE(q4e_check_f32("QSA permuted-slot groups", b_group.contents,
                          group_expected, GROUP * INDEX_DIM),
            "QSA permuted-slot grouping");
    logical_position[5] = 99u;
    memcpy(b_logical_position.contents, logical_position,
           sizeof(logical_position));
    memset(b_group_status.contents, 0, b_group_status.length);
    REQUIRE(q4e_dispatch(
                queue, cache, DS4_METAL_QWEN4EXP_QSA_GROUP,
                &group_args, sizeof(group_args),
                @[b_physical_raw, b_permuted_slot, b_logical_position, b_norm,
                  b_group, b_group_status], @[], GROUP),
            "QSA nonconsecutive-position rejection dispatch");
    REQUIRE(((uint32_t *)b_group_status.contents)[1] != 0u,
            "QSA accepted a group crossing a position hole");
    logical_position[5] = 5u;
    memcpy(b_logical_position.contents, logical_position,
           sizeof(logical_position));

    float index_query[N_QUERY * INDEX_HEAD * INDEX_DIM];
    for (size_t query = 0u; query < N_QUERY; query++) {
        for (size_t head = 0u; head < INDEX_HEAD; head++) {
            for (size_t i = 0u; i < INDEX_DIM; i++) {
                const float sign = head == 0u ? 1.0f : -1.0f;
                index_query[(query * INDEX_HEAD + head) * INDEX_DIM + i] =
                    sign * (((float)(int)((query + i) % 7u) - 3.0f) * 0.11f);
            }
        }
    }
    const uint32_t visible[N_QUERY] = { 11u, 12u };
    uint32_t expected_position[N_QUERY * MAX_WIDTH];
    uint32_t expected_count[N_QUERY];
    memset(expected_position, 0, sizeof(expected_position));
    for (size_t query = 0u; query < N_QUERY; query++) {
        const size_t n_group = visible[query] / COMPRESSION;
        float score[GROUP];
        REQUIRE(ds4_qwen4exp_ref_qsa_scores_f32(
                    score, index_query + query * INDEX_HEAD * INDEX_DIM,
                    group_expected, n_group, INDEX_HEAD, INDEX_DIM),
                "QSA score oracle");
        size_t count = 0u;
        REQUIRE(ds4_qwen4exp_ref_qsa_select_positions(
                    expected_position + query * MAX_WIDTH, MAX_WIDTH, &count,
                    score, visible[query], COMPRESSION, 512u),
                "QSA select oracle");
        expected_count[query] = (uint32_t)count;
    }
    {
        float head_dot[INDEX_HEAD] = {0.0f, 0.0f};
        for (size_t head = 0u; head < INDEX_HEAD; head++)
            for (size_t i = 0u; i < INDEX_DIM; i++)
                head_dot[head] += index_query[head * INDEX_DIM + i] *
                    group_expected[i];
        const float relu_per_head = fmaxf(head_dot[0], 0.0f) +
                                    fmaxf(head_dot[1], 0.0f);
        const float relu_after_sum =
            fmaxf(head_dot[0] + head_dot[1], 0.0f);
        REQUIRE(fabsf(relu_per_head - relu_after_sum) > 1.0e-6f,
                "QSA fixture does not distinguish ReLU-per-head semantics");
    }
    id<MTLBuffer> b_query =
        q4e_buffer(device, index_query, sizeof(index_query));
    id<MTLBuffer> b_visible = q4e_buffer(device, visible, sizeof(visible));
    id<MTLBuffer> b_position =
        q4e_buffer(device, NULL, sizeof(expected_position));
    id<MTLBuffer> b_count =
        q4e_buffer(device, NULL, sizeof(expected_count));
    id<MTLBuffer> b_select_status =
        q4e_buffer(device, NULL, N_QUERY * sizeof(uint32_t));
    ds4_metal_qwen4exp_qsa_select_args select_args = {
        .n_query = N_QUERY, .n_query_head = INDEX_HEAD,
        .head_dim = INDEX_DIM, .n_visible_max = MAX_WIDTH,
        .compression = COMPRESSION, .group_budget = 512u,
        .max_width = MAX_WIDTH,
    };
    REQUIRE(q4e_dispatch(
                queue, cache, DS4_METAL_QWEN4EXP_QSA_SELECT,
                &select_args, sizeof(select_args),
                @[b_query, b_group, b_visible, b_position, b_count,
                  b_select_status], @[], N_QUERY), "QSA select dispatch");
    REQUIRE(q4e_status_ok(b_select_status.contents, N_QUERY),
            "QSA select status");
    REQUIRE(memcmp(b_count.contents, expected_count, sizeof(expected_count)) == 0,
            "QSA selected counts");
    for (size_t query = 0u; query < N_QUERY; query++)
        REQUIRE(memcmp((uint32_t *)b_position.contents + query * MAX_WIDTH,
                       expected_position + query * MAX_WIDTH,
                       expected_count[query] * sizeof(uint32_t)) == 0,
                "QSA selected positions query %zu", query);

    /* The dense Phase-5 ceiling is inclusive at 2051 and fail-closed at 2052.
     * Equal group scores also exercise ascending group-ID ties across all 512
     * complete groups, with the three-token raw tail appended last. */
    {
        float boundary_query = 0.0f;
        float boundary_groups[512] = {0.0f};
        uint32_t boundary_visible = 2051u;
        id<MTLBuffer> b_boundary_query =
            q4e_buffer(device, &boundary_query, sizeof(boundary_query));
        id<MTLBuffer> b_boundary_groups =
            q4e_buffer(device, boundary_groups, sizeof(boundary_groups));
        id<MTLBuffer> b_boundary_visible =
            q4e_buffer(device, &boundary_visible, sizeof(boundary_visible));
        id<MTLBuffer> b_boundary_position =
            q4e_buffer(device, NULL, 4096u * sizeof(uint32_t));
        id<MTLBuffer> b_boundary_count =
            q4e_buffer(device, NULL, sizeof(uint32_t));
        id<MTLBuffer> b_boundary_status =
            q4e_buffer(device, NULL, sizeof(uint32_t));
        ds4_metal_qwen4exp_qsa_select_args boundary_args = {
            .n_query = 1u, .n_query_head = 1u, .head_dim = 1u,
            .n_visible_max = 2051u, .compression = 4u,
            .group_budget = 512u, .max_width = 2051u,
        };
        REQUIRE(q4e_dispatch(
                    queue, cache, DS4_METAL_QWEN4EXP_QSA_SELECT,
                    &boundary_args, sizeof(boundary_args),
                    @[b_boundary_query, b_boundary_groups, b_boundary_visible,
                      b_boundary_position, b_boundary_count, b_boundary_status],
                    @[], 1u), "QSA 2051 dispatch");
        REQUIRE(*(uint32_t *)b_boundary_status.contents == 0u,
                "QSA rejected 2051");
        REQUIRE(*(uint32_t *)b_boundary_count.contents == 2051u,
                "QSA 2051 width");
        for (uint32_t i = 0u; i < 2051u; i++)
            REQUIRE(((uint32_t *)b_boundary_position.contents)[i] == i,
                    "QSA 2051 tie/tail at %u", i);
        const uint32_t rejected_width[] = {
            2052u, 4096u, 262143u, 262144u, 262145u,
        };
        for (size_t case_index = 0u;
             case_index < sizeof(rejected_width) / sizeof(rejected_width[0]);
             case_index++) {
            boundary_visible = rejected_width[case_index];
            memcpy(b_boundary_visible.contents, &boundary_visible,
                   sizeof(boundary_visible));
            memset(b_boundary_status.contents, 0, b_boundary_status.length);
            boundary_args.n_visible_max = boundary_visible;
            boundary_args.max_width = boundary_visible;
            REQUIRE(q4e_dispatch(
                        queue, cache, DS4_METAL_QWEN4EXP_QSA_SELECT,
                        &boundary_args, sizeof(boundary_args),
                        @[b_boundary_query, b_boundary_groups,
                          b_boundary_visible, b_boundary_position,
                          b_boundary_count, b_boundary_status],
                        @[], 1u), "QSA rejected-width dispatch");
            REQUIRE(*(uint32_t *)b_boundary_status.contents != 0u,
                    "QSA accepted width %u", boundary_visible);
        }
    }

    /* Dense attention over the selected width, still bounded by 2051. */
    enum { Q_HEAD = 4, KV_HEAD = 2, ATTN_DIM = 4, N_KEY = 12 };
    float attention_query[N_QUERY * Q_HEAD * ATTN_DIM];
    float attention_key[N_KEY * KV_HEAD * ATTN_DIM];
    float attention_value[N_KEY * KV_HEAD * ATTN_DIM];
    for (size_t i = 0u; i < sizeof(attention_query) / sizeof(float); i++)
        attention_query[i] = ((float)(int)(i % 9u) - 4.0f) * 0.083f;
    for (size_t i = 0u; i < sizeof(attention_key) / sizeof(float); i++) {
        attention_key[i] = ((float)(int)(i % 13u) - 6.0f) * 0.047f;
        attention_value[i] = ((float)(int)(i % 11u) - 5.0f) * 0.059f;
    }
    float attention_expected[N_QUERY * Q_HEAD * ATTN_DIM];
    for (size_t query = 0u; query < N_QUERY; query++) {
        for (size_t head = 0u; head < Q_HEAD; head++) {
            cpu_attention(
                attention_expected + query * Q_HEAD * ATTN_DIM,
                attention_query + query * Q_HEAD * ATTN_DIM,
                attention_key, attention_value,
                expected_position + query * MAX_WIDTH,
                expected_count[query], head, Q_HEAD, KV_HEAD, ATTN_DIM);
        }
    }
    id<MTLBuffer> b_attention_query =
        q4e_buffer(device, attention_query, sizeof(attention_query));
    id<MTLBuffer> b_attention_key =
        q4e_buffer(device, attention_key, sizeof(attention_key));
    id<MTLBuffer> b_attention_value =
        q4e_buffer(device, attention_value, sizeof(attention_value));
    id<MTLBuffer> b_attention_output =
        q4e_buffer(device, NULL, sizeof(attention_expected));
    id<MTLBuffer> b_attention_status = q4e_buffer(
        device, NULL, N_QUERY * Q_HEAD * sizeof(uint32_t));
    ds4_metal_qwen4exp_qsa_attention_args attention_args = {
        .n_query = N_QUERY, .n_query_head = Q_HEAD, .n_kv_head = KV_HEAD,
        .head_dim = ATTN_DIM, .n_key = N_KEY, .max_selected = MAX_WIDTH,
    };
    REQUIRE(q4e_dispatch(
                queue, cache, DS4_METAL_QWEN4EXP_QSA_ATTENTION,
                &attention_args, sizeof(attention_args),
                @[b_attention_query, b_attention_key, b_attention_value,
                  b_position, b_count, b_attention_output,
                  b_attention_status], @[], N_QUERY * Q_HEAD),
            "QSA attention dispatch");
    REQUIRE(q4e_status_ok(b_attention_status.contents, N_QUERY * Q_HEAD),
            "QSA attention status");
    REQUIRE(q4e_check_f32("QSA attention", b_attention_output.contents,
                          attention_expected, N_QUERY * Q_HEAD * ATTN_DIM),
            "QSA attention");
    return true;
}

static bool test_ple_head(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        const ds4_metal_qwen4exp_cache *cache) {
    enum { TOKEN = 2, PLE_HEAD = 4, ROW_DIM = 5, N_ROW = 7 };
    const uint32_t row_id[TOKEN * PLE_HEAD] = { 6u, 0u, 3u, 1u, 2u, 5u, 4u, 0u };
    float rows[N_ROW * ROW_DIM];
    float gathered_expected[TOKEN * PLE_HEAD * ROW_DIM];
    for (size_t i = 0u; i < N_ROW * ROW_DIM; i++)
        rows[i] = ((float)(int)(i % 12u) - 5.0f) * 0.041f;
    for (size_t item = 0u; item < TOKEN * PLE_HEAD; item++)
        memcpy(gathered_expected + item * ROW_DIM,
               rows + row_id[item] * ROW_DIM, ROW_DIM * sizeof(float));
    id<MTLBuffer> b_id = q4e_buffer(device, row_id, sizeof(row_id));
    id<MTLBuffer> b_rows = q4e_buffer(device, rows, sizeof(rows));
    id<MTLBuffer> b_gathered =
        q4e_buffer(device, NULL, sizeof(gathered_expected));
    id<MTLBuffer> b_gather_status = q4e_buffer(
        device, NULL, TOKEN * PLE_HEAD * sizeof(uint32_t));
    ds4_metal_qwen4exp_ple_gather_args gather_args = {
        .n_token = TOKEN, .n_head = PLE_HEAD, .row_dim = ROW_DIM,
        .n_row = N_ROW,
    };
    REQUIRE(q4e_dispatch(
                queue, cache, DS4_METAL_QWEN4EXP_PLE_GATHER,
                &gather_args, sizeof(gather_args),
                @[b_id, b_rows, b_gathered, b_gather_status], @[],
                TOKEN * PLE_HEAD), "PLE gather dispatch");
    REQUIRE(q4e_status_ok(b_gather_status.contents, TOKEN * PLE_HEAD),
            "PLE gather status");
    REQUIRE(q4e_check_f32("PLE gather", b_gathered.contents,
                          gathered_expected, TOKEN * PLE_HEAD * ROW_DIM),
            "PLE gather");

    enum { STREAM = 4, DIM = 5 };
    float query[TOKEN * STREAM * DIM];
    float key[TOKEN * STREAM * DIM];
    float value[TOKEN * DIM];
    float gate_expected[TOKEN * STREAM * DIM];
    for (size_t i = 0u; i < TOKEN * STREAM * DIM; i++) {
        query[i] = ((float)(int)(i % 9u) - 4.0f) * 0.067f;
        key[i] = ((float)(int)(i % 7u) - 3.0f) * -0.073f;
    }
    for (size_t i = 0u; i < TOKEN * DIM; i++)
        value[i] = ((float)(int)(i % 6u) - 2.0f) * 0.089f;
    for (size_t token = 0u; token < TOKEN; token++)
        REQUIRE(ds4_qwen4exp_ref_ple_gate_f32(
                    gate_expected + token * STREAM * DIM,
                    query + token * STREAM * DIM,
                    key + token * STREAM * DIM, value + token * DIM,
                    STREAM, DIM), "PLE gate oracle");
    id<MTLBuffer> b_query = q4e_buffer(device, query, sizeof(query));
    id<MTLBuffer> b_key = q4e_buffer(device, key, sizeof(key));
    id<MTLBuffer> b_value = q4e_buffer(device, value, sizeof(value));
    id<MTLBuffer> b_gate_output =
        q4e_buffer(device, NULL, sizeof(gate_expected));
    id<MTLBuffer> b_gate_status = q4e_buffer(
        device, NULL, TOKEN * STREAM * sizeof(uint32_t));
    ds4_metal_qwen4exp_ple_gate_args gate_args = {
        .n_token = TOKEN, .n_stream = STREAM, .dim = DIM,
    };
    REQUIRE(q4e_dispatch(
                queue, cache, DS4_METAL_QWEN4EXP_PLE_GATE,
                &gate_args, sizeof(gate_args),
                @[b_query, b_key, b_value, b_gate_output, b_gate_status], @[],
                TOKEN * STREAM), "PLE gate dispatch");
    REQUIRE(q4e_status_ok(b_gate_status.contents, TOKEN * STREAM),
            "PLE gate status");
    REQUIRE(q4e_check_f32("PLE gate", b_gate_output.contents, gate_expected,
                          TOKEN * STREAM * DIM), "PLE gate");

    enum { CONV_TOKEN = 5, CONV_CHANNEL = 2, CONV_KERNEL = 4, DILATION = 3 };
    float conv_input[CONV_TOKEN * CONV_CHANNEL];
    float conv_weight[CONV_CHANNEL * CONV_KERNEL];
    float conv_initial[CONV_CHANNEL * DILATION * (CONV_KERNEL - 1)];
    for (size_t i = 0u; i < sizeof(conv_input) / sizeof(float); i++)
        conv_input[i] = ((float)(int)(i % 7u) - 3.0f) * 0.051f;
    for (size_t i = 0u; i < sizeof(conv_weight) / sizeof(float); i++)
        conv_weight[i] = ((float)(int)(i % 5u) - 2.0f) * 0.063f;
    for (size_t i = 0u; i < sizeof(conv_initial) / sizeof(float); i++)
        conv_initial[i] = ((float)(int)(i % 11u) - 5.0f) * 0.017f;
    float conv_expected[CONV_TOKEN * CONV_CHANNEL];
    float conv_state_expected[
        CONV_CHANNEL * DILATION * (CONV_KERNEL - 1)];
    memcpy(conv_state_expected, conv_initial, sizeof(conv_initial));
    REQUIRE(ds4_qwen4exp_ref_dilated_conv1d_silu_f32(
                conv_expected, conv_state_expected, conv_input, conv_weight,
                CONV_TOKEN, CONV_CHANNEL, CONV_KERNEL, DILATION),
            "PLE conv oracle");
    id<MTLBuffer> b_conv_input =
        q4e_buffer(device, conv_input, sizeof(conv_input));
    id<MTLBuffer> b_conv_weight =
        q4e_buffer(device, conv_weight, sizeof(conv_weight));
    const uint32_t irregular_ubatch[] = { 2u, 1u, 2u };
    for (uint32_t plan = 0u; plan < 6u; plan++) {
        id<MTLBuffer> b_conv_state =
            q4e_buffer(device, conv_initial, sizeof(conv_initial));
        id<MTLBuffer> b_conv_output =
            q4e_buffer(device, NULL, sizeof(conv_expected));
        id<MTLBuffer> b_conv_status = q4e_buffer(
            device, NULL, CONV_CHANNEL * sizeof(uint32_t));
        float transition_state[
            CONV_CHANNEL * DILATION * (CONV_KERNEL - 1)];
        memcpy(transition_state, conv_initial, sizeof(transition_state));
        uint32_t start = 0u;
        uint32_t dispatch_index = 0u;
        while (start < CONV_TOKEN) {
            uint32_t count = plan < 5u ? plan + 1u
                : irregular_ubatch[dispatch_index];
            if (count > CONV_TOKEN - start) count = CONV_TOKEN - start;
            ds4_metal_qwen4exp_conv_args conv_args = {
                .n_token = count, .n_channel = CONV_CHANNEL,
                .kernel_size = CONV_KERNEL, .dilation = DILATION,
                .n_sequence = 1u,
            };
            memset(b_conv_status.contents, 0, b_conv_status.length);
            NSArray<NSNumber *> *offsets = @[
                @(start * CONV_CHANNEL * sizeof(float)), @0u, @0u,
                @(start * CONV_CHANNEL * sizeof(float)), @0u,
            ];
            REQUIRE(q4e_dispatch(
                        queue, cache, DS4_METAL_QWEN4EXP_CONV,
                        &conv_args, sizeof(conv_args),
                        @[b_conv_input, b_conv_weight, b_conv_state,
                          b_conv_output, b_conv_status], offsets, CONV_CHANNEL),
                    "PLE conv plan %u dispatch %u", plan, dispatch_index);
            REQUIRE(q4e_status_ok(b_conv_status.contents, CONV_CHANNEL),
                    "PLE conv status plan %u", plan);
            if (plan == 0u) {
                float transition_output[CONV_CHANNEL];
                REQUIRE(ds4_qwen4exp_ref_dilated_conv1d_silu_f32(
                            transition_output, transition_state,
                            conv_input + start * CONV_CHANNEL, conv_weight,
                            1u, CONV_CHANNEL, CONV_KERNEL, DILATION),
                        "PLE conv transition oracle");
                REQUIRE(q4e_check_f32(
                            "PLE conv transition", b_conv_state.contents,
                            transition_state,
                            CONV_CHANNEL * DILATION * (CONV_KERNEL - 1)),
                        "PLE conv state transition %u", start);
            }
            start += count;
            dispatch_index++;
        }
        REQUIRE(q4e_check_f32("PLE conv output", b_conv_output.contents,
                              conv_expected, CONV_TOKEN * CONV_CHANNEL),
                "PLE conv output plan %u", plan);
        REQUIRE(q4e_check_f32(
                    "PLE conv state", b_conv_state.contents,
                    conv_state_expected,
                    CONV_CHANNEL * DILATION * (CONV_KERNEL - 1)),
                "PLE conv state plan %u", plan);
    }

    enum { HEAD_TOKEN = 2, HEAD_IN = 5, HEAD_OUT = 7 };
    float head_input[HEAD_TOKEN * HEAD_IN];
    float head_weight[HEAD_OUT * HEAD_IN];
    float head_expected[HEAD_TOKEN * HEAD_OUT];
    for (size_t i = 0u; i < sizeof(head_input) / sizeof(float); i++)
        head_input[i] = ((float)(int)(i % 8u) - 3.0f) * 0.077f;
    for (size_t i = 0u; i < sizeof(head_weight) / sizeof(float); i++)
        head_weight[i] = ((float)(int)(i % 10u) - 4.0f) * 0.033f;
    for (size_t token = 0u; token < HEAD_TOKEN; token++)
        for (size_t out = 0u; out < HEAD_OUT; out++) {
            float sum = 0.0f;
            for (size_t i = 0u; i < HEAD_IN; i++)
                sum += head_input[token * HEAD_IN + i] *
                    head_weight[out * HEAD_IN + i];
            head_expected[token * HEAD_OUT + out] = sum;
        }
    id<MTLBuffer> b_head_input =
        q4e_buffer(device, head_input, sizeof(head_input));
    id<MTLBuffer> b_head_weight =
        q4e_buffer(device, head_weight, sizeof(head_weight));
    id<MTLBuffer> b_head_output =
        q4e_buffer(device, NULL, sizeof(head_expected));
    id<MTLBuffer> b_head_status = q4e_buffer(
        device, NULL, HEAD_TOKEN * HEAD_OUT * sizeof(uint32_t));
    ds4_metal_qwen4exp_head_args head_args = {
        .n_token = HEAD_TOKEN, .input_dim = HEAD_IN, .output_dim = HEAD_OUT,
    };
    REQUIRE(q4e_dispatch(queue, cache, DS4_METAL_QWEN4EXP_HEAD,
                         &head_args, sizeof(head_args),
                         @[b_head_input, b_head_weight, b_head_output,
                           b_head_status], @[], HEAD_TOKEN * HEAD_OUT),
            "head dispatch");
    REQUIRE(q4e_status_ok(b_head_status.contents, HEAD_TOKEN * HEAD_OUT),
            "head status");
    REQUIRE(q4e_check_f32("head", b_head_output.contents, head_expected,
                          HEAD_TOKEN * HEAD_OUT), "head");
    return true;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        NSString *source_path = argc > 1
            ? [NSString stringWithUTF8String:argv[1]]
            : @"metal/qwen4exp.metal";
        NSError *error = nil;
        NSString *source = [NSString stringWithContentsOfFile:source_path
                                                     encoding:NSUTF8StringEncoding
                                                        error:&error];
        if (!source) {
            fprintf(stderr, "read %s: %s\n", source_path.UTF8String,
                    error.localizedDescription.UTF8String);
            return 2;
        }
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "no Metal device\n");
            return 2;
        }
        ds4_metal_qwen4exp_cache cache = {0};
        if (!ds4_metal_qwen4exp_compile_cache(device, source, &cache, &error)) {
            fprintf(stderr, "compile %s: %s\n", source_path.UTF8String,
                    error.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            fprintf(stderr, "create Metal command queue\n");
            return 2;
        }
        printf("Qwen4Exp Phase-5 Metal fixture on %s\n",
               device.name.UTF8String);
        const bool ok =
            test_layout_and_bounds(device, queue, &cache) &&
            test_embedding_gr(device, queue, &cache) &&
            test_conv_controls_gdn(device, queue, &cache) &&
            test_shared_conv_row(device, queue, &cache) &&
            test_router_moe(device, queue, &cache) &&
            test_qsa(device, queue, &cache) &&
            test_ple_head(device, queue, &cache);
        if (!ok) return 1;
        puts("all Qwen4Exp Phase-5 Metal correctness fixtures passed");
        return 0;
    }
}
