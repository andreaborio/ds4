#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "qwen36_attention_golden.inc"
#include "qwen36_gdn_golden.inc"

/*
 * Standalone model-free CPU-oracle versus Metal test for metal/qwen35.metal.
 * It deliberately compiles the reviewed .metal source at runtime, matching
 * DS4's normal newLibraryWithSource path without requiring a full Xcode install.
 *
 * Build and run from the repository root:
 *
 *   clang -fobjc-arc -framework Foundation -framework Metal \
 *     tests/qwen/test_qwen35_metal.m -o /tmp/test-qwen35-metal
 *   /tmp/test-qwen35-metal
 */

typedef struct {
    uint32_t n_token;
    uint32_t n_query_head;
    uint32_t head_dim;
    uint32_t reserved;
    uint64_t projection_token_stride;
    uint64_t projection_head_stride;
    uint64_t projection_dim_stride;
    uint64_t query_token_stride;
    uint64_t query_head_stride;
    uint64_t query_dim_stride;
    uint64_t gate_token_stride;
    uint64_t gate_head_stride;
    uint64_t gate_dim_stride;
} qwen35_split_args;

typedef struct {
    uint64_t n_value;
    uint64_t input_stride;
    uint64_t gate_stride;
    uint64_t output_stride;
} qwen35_sigmoid_mul_args;

typedef struct {
    uint32_t n_token;
    uint32_t n_head;
    uint32_t head_dim;
    uint32_t n_rot;
    float theta;
    uint32_t reserved;
    uint64_t source_token_stride;
    uint64_t source_head_stride;
    uint64_t source_dim_stride;
    uint64_t output_token_stride;
    uint64_t output_head_stride;
    uint64_t output_dim_stride;
    uint64_t position_stride;
} qwen35_rope_args;

typedef struct {
    uint32_t n_channel;
    uint32_t kernel_size;
    uint64_t input_channel_stride;
    uint64_t weight_channel_stride;
    uint64_t weight_tap_stride;
    uint64_t state_channel_stride;
    uint64_t state_tap_stride;
    uint64_t output_channel_stride;
} qwen35_conv_args;

typedef struct {
    uint32_t n_key_head;
    uint32_t n_value_head;
    uint32_t key_dim;
    uint32_t value_dim;
    uint64_t query_head_stride;
    uint64_t query_dim_stride;
    uint64_t key_head_stride;
    uint64_t key_dim_stride;
    uint64_t value_head_stride;
    uint64_t value_dim_stride;
    uint64_t log_decay_head_stride;
    uint64_t beta_head_stride;
    uint64_t state_head_stride;
    uint64_t state_value_stride;
    uint64_t state_key_stride;
    uint64_t output_head_stride;
    uint64_t output_dim_stride;
} qwen35_delta_args;

typedef struct {
    uint32_t n_vector;
    uint32_t dim;
    float epsilon;
    uint32_t reserved;
    uint64_t input_vector_stride;
    uint64_t input_dim_stride;
    uint64_t gate_vector_stride;
    uint64_t gate_dim_stride;
    uint64_t weight_dim_stride;
    uint64_t output_vector_stride;
    uint64_t output_dim_stride;
} qwen35_norm_args;

typedef struct {
    uint32_t row_index;
    uint32_t n_embd;
    uint32_t block_size;
    uint32_t reserved;
    uint64_t source_row_stride;
    uint64_t source_block_stride;
    uint64_t source_scale_offset;
    uint64_t source_quant_offset;
    uint64_t source_quant_stride;
    uint64_t output_dim_stride;
} qwen35_embedding_args;

typedef struct {
    uint32_t n_value_head;
    uint32_t reserved;
    uint64_t alpha_logit_head_stride;
    uint64_t beta_logit_head_stride;
    uint64_t ssm_a_head_stride;
    uint64_t dt_bias_head_stride;
    uint64_t log_decay_head_stride;
    uint64_t beta_head_stride;
} qwen35_controls_args;

typedef struct {
    uint32_t n_kv;
    uint32_t n_query_head;
    uint32_t n_kv_head;
    uint32_t head_dim;
    uint64_t query_head_stride;
    uint64_t query_dim_stride;
    uint64_t key_token_stride;
    uint64_t key_head_stride;
    uint64_t key_dim_stride;
    uint64_t value_token_stride;
    uint64_t value_head_stride;
    uint64_t value_dim_stride;
    uint64_t output_head_stride;
    uint64_t output_dim_stride;
} qwen35_gqa_args;

_Static_assert(sizeof(qwen35_split_args) == 88, "split ABI drift");
_Static_assert(sizeof(qwen35_sigmoid_mul_args) == 32, "sigmoid ABI drift");
_Static_assert(sizeof(qwen35_rope_args) == 80, "RoPE ABI drift");
_Static_assert(sizeof(qwen35_conv_args) == 56, "conv ABI drift");
_Static_assert(sizeof(qwen35_delta_args) == 120, "DeltaNet ABI drift");
_Static_assert(sizeof(qwen35_norm_args) == 72, "RMSNorm ABI drift");
_Static_assert(sizeof(qwen35_embedding_args) == 64, "embedding ABI drift");
_Static_assert(sizeof(qwen35_controls_args) == 56, "controls ABI drift");
_Static_assert(sizeof(qwen35_gqa_args) == 96, "GQA ABI drift");

static id<MTLBuffer> buffer_with_bytes(
        id<MTLDevice> device,
        const void   *bytes,
        NSUInteger    length) {
    return [device newBufferWithBytes:bytes
                               length:length
                              options:MTLResourceStorageModeShared];
}

static id<MTLBuffer> zero_buffer(
        id<MTLDevice> device,
        NSUInteger    length) {
    id<MTLBuffer> buffer = [device newBufferWithLength:length
                                               options:MTLResourceStorageModeShared];
    if (buffer) memset(buffer.contents, 0, length);
    return buffer;
}

static bool dispatch_kernel(
        id<MTLDevice>                    device,
        id<MTLCommandQueue>              queue,
        id<MTLLibrary>                   library,
        NSString                        *name,
        const void                      *arguments,
        NSUInteger                       argument_size,
        NSArray<id<MTLBuffer>>           *buffers,
        NSArray<NSNumber *>              *offsets,
        NSUInteger                       grid_or_groups,
        NSUInteger                       requested_threads,
        bool                             dispatch_groups,
        NSUInteger                       scratch_planes,
        NSUInteger                       scratch_extra_floats) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (!function) {
        fprintf(stderr, "missing Metal function %s\n", name.UTF8String);
        return false;
    }
    NSError *error = nil;
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) {
        fprintf(stderr, "pipeline %s: %s\n", name.UTF8String,
                error.localizedDescription.UTF8String);
        return false;
    }

    NSUInteger threads = requested_threads != 0
        ? requested_threads
        : pipeline.threadExecutionWidth;
    if (threads == 0 || threads > pipeline.maxTotalThreadsPerThreadgroup) {
        fprintf(stderr, "invalid threadgroup size for %s\n", name.UTF8String);
        return false;
    }
    if (offsets.count != 0 && offsets.count != buffers.count) {
        fprintf(stderr, "buffer/offset count mismatch for %s\n",
                name.UTF8String);
        return false;
    }

    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBytes:arguments length:argument_size atIndex:0];
    for (NSUInteger i = 0; i < buffers.count; i++) {
        const NSUInteger offset = offsets.count != 0
            ? offsets[i].unsignedIntegerValue
            : 0;
        [encoder setBuffer:buffers[i] offset:offset atIndex:i + 1];
    }
    if (scratch_planes != 0 || scratch_extra_floats != 0) {
        const NSUInteger n_simdgroup =
            (threads + pipeline.threadExecutionWidth - 1) /
            pipeline.threadExecutionWidth;
        NSUInteger bytes =
            (scratch_planes * n_simdgroup + scratch_extra_floats) *
            sizeof(float);
        bytes = (bytes + 15u) & ~(NSUInteger)15u;
        [encoder setThreadgroupMemoryLength:bytes atIndex:0];
    }

    const MTLSize threadgroup = MTLSizeMake(threads, 1, 1);
    if (dispatch_groups) {
        [encoder dispatchThreadgroups:MTLSizeMake(grid_or_groups, 1, 1)
                threadsPerThreadgroup:threadgroup];
    } else {
        [encoder dispatchThreads:MTLSizeMake(grid_or_groups, 1, 1)
           threadsPerThreadgroup:threadgroup];
    }
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "dispatch %s: %s\n", name.UTF8String,
                command.error.localizedDescription.UTF8String);
        return false;
    }
    return true;
}

static bool check_f32(
        const char  *name,
        const float *actual,
        const float *expected,
        size_t       count,
        float        absolute_tolerance,
        float        relative_tolerance) {
    float maximum_error = 0.0f;
    for (size_t i = 0; i < count; i++) {
        const float error = fabsf(actual[i] - expected[i]);
        const float limit = absolute_tolerance +
                            relative_tolerance * fabsf(expected[i]);
        if (error > maximum_error) maximum_error = error;
        if (!isfinite(actual[i]) || error > limit) {
            fprintf(stderr,
                    "%s[%zu]: actual %.9g expected %.9g error %.9g limit %.9g\n",
                    name, i, actual[i], expected[i], error, limit);
            return false;
        }
    }
    printf("ok %-28s max_abs_error=%.3g\n", name, maximum_error);
    return true;
}

static uint16_t f32_to_f16_bits(float value) {
    const _Float16 half_value = (_Float16)value;
    uint16_t bits = 0;
    memcpy(&bits, &half_value, sizeof(bits));
    return bits;
}

static float f16_bits_to_f32(uint16_t bits) {
    _Float16 half_value = 0;
    memcpy(&half_value, &bits, sizeof(bits));
    return (float)half_value;
}

static void host_store_f32(uint8_t *base, size_t offset, float value) {
    memcpy(base + offset, &value, sizeof(value));
}

static float host_load_f32(const uint8_t *base, size_t offset) {
    float value = 0.0f;
    memcpy(&value, base + offset, sizeof(value));
    return value;
}

static bool test_embedding_q8_0(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLLibrary> library) {
    enum {
        N_EMBD = 2048,
        Q8_BLOCK = 32,
        BLOCK_COUNT = N_EMBD / Q8_BLOCK,
        BLOCK_STRIDE = 40,
        ROW_STRIDE = BLOCK_COUNT * BLOCK_STRIDE + 16,
        ROW_COUNT = 2,
        OUTPUT_STRIDE = 8,
    };
    uint8_t *encoded = calloc(ROW_COUNT, ROW_STRIDE);
    float *expected = malloc(N_EMBD * sizeof(expected[0]));
    float *actual = malloc(N_EMBD * sizeof(actual[0]));
    if (!encoded || !expected || !actual) {
        free(encoded);
        free(expected);
        free(actual);
        return false;
    }

    static const float scale_pattern[] = {
        0.125f, -0.25f, 0.5f, -0.75f, 1.0f, -1.5f, 2.0f,
    };
    for (uint32_t row = 0; row < ROW_COUNT; row++) {
        for (uint32_t block = 0; block < BLOCK_COUNT; block++) {
            const float requested_scale =
                scale_pattern[(block + 2u * row) %
                              (sizeof(scale_pattern) / sizeof(scale_pattern[0]))];
            const uint16_t scale_bits = f32_to_f16_bits(requested_scale);
            const size_t block_offset =
                (size_t)row * ROW_STRIDE + (size_t)block * BLOCK_STRIDE;
            memcpy(encoded + block_offset, &scale_bits, sizeof(scale_bits));
            const float stored_scale = f16_bits_to_f32(scale_bits);
            for (uint32_t item = 0; item < Q8_BLOCK; item++) {
                const uint32_t dim = block * Q8_BLOCK + item;
                const int32_t quant_value =
                    (int32_t)((dim * 37u + row * 19u) % 255u) - 127;
                const int8_t quant = (int8_t)quant_value;
                memcpy(encoded + block_offset + 2u + item,
                       &quant, sizeof(quant));
                if (row == 1u) expected[dim] = stored_scale * (float)quant;
            }
        }
    }

    id<MTLBuffer> embedding = buffer_with_bytes(
        device, encoded, ROW_COUNT * ROW_STRIDE);
    const NSUInteger output_bytes =
        (N_EMBD - 1u) * OUTPUT_STRIDE + sizeof(float);
    id<MTLBuffer> output = [device newBufferWithLength:output_bytes
                                               options:MTLResourceStorageModeShared];
    if (!embedding || !output) {
        free(encoded);
        free(expected);
        free(actual);
        return false;
    }
    memset(output.contents, 0xa5, output_bytes);
    qwen35_embedding_args args = {
        .row_index = 1,
        .n_embd = N_EMBD,
        .block_size = Q8_BLOCK,
        .source_row_stride = ROW_STRIDE,
        .source_block_stride = BLOCK_STRIDE,
        .source_scale_offset = 0,
        .source_quant_offset = 2,
        .source_quant_stride = 1,
        .output_dim_stride = OUTPUT_STRIDE,
    };
    bool ok = dispatch_kernel(
        device, queue, library,
        @"kernel_qwen35_dequant_embedding_q8_0_f32",
        &args, sizeof(args), @[embedding, output], @[],
        N_EMBD, 64, false, 0, 0);
    if (ok) {
        const uint8_t *bytes = output.contents;
        for (size_t dim = 0; dim < N_EMBD; dim++) {
            actual[dim] = host_load_f32(bytes, dim * OUTPUT_STRIDE);
            for (size_t padding = sizeof(float); padding < OUTPUT_STRIDE;
                 padding++) {
                if (dim + 1u == N_EMBD) break;
                if (bytes[dim * OUTPUT_STRIDE + padding] != 0xa5u) {
                    fprintf(stderr, "embedding output padding overwritten\n");
                    ok = false;
                    break;
                }
            }
            if (!ok) break;
        }
    }
    if (ok) {
        ok = check_f32("Q8_0 embedding row", actual, expected,
                       N_EMBD, 0.0f, 0.0f);
    }
    if (ok && memcmp(embedding.contents, encoded,
                     ROW_COUNT * ROW_STRIDE) != 0) {
        fprintf(stderr, "embedding dequant mutated its source row\n");
        ok = false;
    }
    free(encoded);
    free(expected);
    free(actual);
    return ok;
}

static bool test_gated_delta_controls(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLLibrary> library) {
    id<MTLBuffer> alpha = buffer_with_bytes(
        device, qwen_ref_alpha_logit, sizeof(qwen_ref_alpha_logit));
    id<MTLBuffer> beta_logit = buffer_with_bytes(
        device, qwen_ref_beta_logit, sizeof(qwen_ref_beta_logit));
    id<MTLBuffer> ssm_a = buffer_with_bytes(
        device, qwen_ref_ssm_a, sizeof(qwen_ref_ssm_a));
    id<MTLBuffer> dt_bias = buffer_with_bytes(
        device, qwen_ref_dt_bias, sizeof(qwen_ref_dt_bias));
    id<MTLBuffer> log_decay = zero_buffer(
        device, sizeof(qwen_ref_log_decay));
    id<MTLBuffer> beta = zero_buffer(device, sizeof(qwen_ref_beta));
    if (!alpha || !beta_logit || !ssm_a || !dt_bias || !log_decay || !beta) {
        return false;
    }
    qwen35_controls_args args = {
        .n_value_head = QWEN_REF_N_VALUE_HEAD,
        .alpha_logit_head_stride = sizeof(float),
        .beta_logit_head_stride = sizeof(float),
        .ssm_a_head_stride = sizeof(float),
        .dt_bias_head_stride = sizeof(float),
        .log_decay_head_stride = sizeof(float),
        .beta_head_stride = sizeof(float),
    };
    const NSUInteger row_bytes = QWEN_REF_N_VALUE_HEAD * sizeof(float);
    for (NSUInteger token = 0; token < QWEN_REF_N_TOKEN; token++) {
        NSArray<NSNumber *> *offsets = @[
            @(token * row_bytes), @(token * row_bytes), @0, @0,
            @(token * row_bytes), @(token * row_bytes)
        ];
        if (!dispatch_kernel(
                device, queue, library,
                @"kernel_qwen35_gated_delta_controls_f32",
                &args, sizeof(args),
                @[alpha, beta_logit, ssm_a, dt_bias, log_decay, beta],
                offsets, QWEN_REF_N_VALUE_HEAD, 32, false, 0, 0)) {
            return false;
        }
    }
    return check_f32("DeltaNet log-decay", log_decay.contents,
                     qwen_ref_log_decay,
                     sizeof(qwen_ref_log_decay) / sizeof(float),
                     3.0e-6f, 3.0e-6f) &&
           check_f32("DeltaNet beta", beta.contents, qwen_ref_beta,
                     sizeof(qwen_ref_beta) / sizeof(float),
                     2.0e-6f, 2.0e-6f);
}

static bool cpu_gqa_reference(
        float                  *output,
        const uint8_t          *query,
        const uint8_t          *key_cache,
        const uint8_t          *value_cache,
        const qwen35_gqa_args  *args) {
    if (!output || !query || !key_cache || !value_cache || !args ||
        args->n_kv == 0 || args->n_query_head == 0 ||
        args->n_kv_head == 0 || args->head_dim == 0 ||
        args->n_query_head % args->n_kv_head != 0) {
        return false;
    }
    float *score = malloc(args->n_kv * sizeof(score[0]));
    if (!score) return false;
    const uint32_t query_per_kv =
        args->n_query_head / args->n_kv_head;
    const float scale = 1.0f / sqrtf((float)args->head_dim);

    for (uint32_t query_head = 0; query_head < args->n_query_head;
         query_head++) {
        const uint32_t kv_head = query_head / query_per_kv;
        const size_t query_base =
            (size_t)query_head * args->query_head_stride;
        float maximum = -INFINITY;
        for (uint32_t token = 0; token < args->n_kv; token++) {
            const size_t key_base =
                (size_t)token * args->key_token_stride +
                (size_t)kv_head * args->key_head_stride;
            float dot = 0.0f;
            for (uint32_t dim = 0; dim < args->head_dim; dim++) {
                dot += host_load_f32(
                           query,
                           query_base +
                               (size_t)dim * args->query_dim_stride) *
                       host_load_f32(
                           key_cache,
                           key_base +
                               (size_t)dim * args->key_dim_stride);
            }
            score[token] = dot * scale;
            if (score[token] > maximum) maximum = score[token];
        }

        float denominator = 0.0f;
        for (uint32_t token = 0; token < args->n_kv; token++) {
            score[token] = expf(score[token] - maximum);
            denominator += score[token];
        }
        if (!(denominator > 0.0f) || !isfinite(denominator)) {
            free(score);
            return false;
        }
        for (uint32_t dim = 0; dim < args->head_dim; dim++) {
            float value = 0.0f;
            for (uint32_t token = 0; token < args->n_kv; token++) {
                const size_t value_offset =
                    (size_t)token * args->value_token_stride +
                    (size_t)kv_head * args->value_head_stride +
                    (size_t)dim * args->value_dim_stride;
                value += (score[token] / denominator) *
                         host_load_f32(value_cache, value_offset);
            }
            output[(size_t)query_head * args->head_dim + dim] = value;
        }
    }
    free(score);
    return true;
}

static bool test_gqa_decode(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLLibrary> library) {
    enum {
        N_QUERY_HEAD = 16,
        N_KV_HEAD = 2,
        HEAD_DIM = 256,
        MAX_KV = 7,
        QUERY_DIM_STRIDE = 4,
        QUERY_HEAD_STRIDE = (HEAD_DIM + 3) * QUERY_DIM_STRIDE,
        KEY_DIM_STRIDE = 4,
        KEY_HEAD_STRIDE = (HEAD_DIM + 5) * KEY_DIM_STRIDE,
        KEY_TOKEN_STRIDE = N_KV_HEAD * KEY_HEAD_STRIDE + 16,
        VALUE_DIM_STRIDE = 4,
        VALUE_HEAD_STRIDE = (HEAD_DIM + 7) * VALUE_DIM_STRIDE,
        VALUE_TOKEN_STRIDE = N_KV_HEAD * VALUE_HEAD_STRIDE + 32,
        OUTPUT_DIM_STRIDE = 8,
        OUTPUT_HEAD_STRIDE = HEAD_DIM * OUTPUT_DIM_STRIDE + 16,
    };
    const size_t query_bytes = N_QUERY_HEAD * QUERY_HEAD_STRIDE;
    const size_t key_bytes = MAX_KV * KEY_TOKEN_STRIDE;
    const size_t value_bytes = MAX_KV * VALUE_TOKEN_STRIDE;
    const size_t output_bytes = N_QUERY_HEAD * OUTPUT_HEAD_STRIDE;
    uint8_t *query_host = calloc(1, query_bytes);
    uint8_t *key_host = calloc(1, key_bytes);
    uint8_t *value_host = calloc(1, value_bytes);
    uint8_t *key_snapshot = malloc(key_bytes);
    uint8_t *value_snapshot = malloc(value_bytes);
    float *expected = malloc(
        N_QUERY_HEAD * HEAD_DIM * sizeof(expected[0]));
    float *actual = malloc(
        N_QUERY_HEAD * HEAD_DIM * sizeof(actual[0]));
    if (!query_host || !key_host || !value_host || !key_snapshot ||
        !value_snapshot || !expected || !actual) {
        free(query_host);
        free(key_host);
        free(value_host);
        free(key_snapshot);
        free(value_snapshot);
        free(expected);
        free(actual);
        return false;
    }

    for (uint32_t head = 0; head < N_QUERY_HEAD; head++) {
        for (uint32_t dim = 0; dim < HEAD_DIM; dim++) {
            const float value =
                0.31f * sinf((float)((head + 1u) * (dim + 3u)) * 0.0031f) +
                0.09f * cosf((float)(dim + 5u) * 0.017f) -
                0.015f * (float)head;
            host_store_f32(
                query_host,
                (size_t)head * QUERY_HEAD_STRIDE +
                    (size_t)dim * QUERY_DIM_STRIDE,
                value);
        }
    }
    for (uint32_t token = 0; token < MAX_KV; token++) {
        for (uint32_t head = 0; head < N_KV_HEAD; head++) {
            for (uint32_t dim = 0; dim < HEAD_DIM; dim++) {
                const float key =
                    0.27f * cosf((float)((token + 2u) * (dim + 1u)) *
                                0.0043f) +
                    0.08f * sinf((float)((head + 1u) * (dim + 7u)) *
                                0.011f) +
                    0.02f * (float)token;
                const float value =
                    0.73f * sinf((float)((token + 1u) * (dim + 2u)) *
                                0.0067f) +
                    0.11f * cosf((float)((head + 3u) * (dim + 1u)) *
                                0.009f) -
                    0.03f * (float)token;
                host_store_f32(
                    key_host,
                    (size_t)token * KEY_TOKEN_STRIDE +
                        (size_t)head * KEY_HEAD_STRIDE +
                        (size_t)dim * KEY_DIM_STRIDE,
                    key);
                host_store_f32(
                    value_host,
                    (size_t)token * VALUE_TOKEN_STRIDE +
                        (size_t)head * VALUE_HEAD_STRIDE +
                        (size_t)dim * VALUE_DIM_STRIDE,
                    value);
            }
        }
    }
    memcpy(key_snapshot, key_host, key_bytes);
    memcpy(value_snapshot, value_host, value_bytes);

    id<MTLBuffer> query = buffer_with_bytes(device, query_host, query_bytes);
    id<MTLBuffer> key_cache = buffer_with_bytes(device, key_host, key_bytes);
    id<MTLBuffer> value_cache = buffer_with_bytes(
        device, value_host, value_bytes);
    id<MTLBuffer> output = [device newBufferWithLength:output_bytes
                                               options:MTLResourceStorageModeShared];
    if (!query || !key_cache || !value_cache || !output) {
        free(query_host);
        free(key_host);
        free(value_host);
        free(key_snapshot);
        free(value_snapshot);
        free(expected);
        free(actual);
        return false;
    }

    qwen35_gqa_args args = {
        .n_query_head = N_QUERY_HEAD,
        .n_kv_head = N_KV_HEAD,
        .head_dim = HEAD_DIM,
        .query_head_stride = QUERY_HEAD_STRIDE,
        .query_dim_stride = QUERY_DIM_STRIDE,
        .key_token_stride = KEY_TOKEN_STRIDE,
        .key_head_stride = KEY_HEAD_STRIDE,
        .key_dim_stride = KEY_DIM_STRIDE,
        .value_token_stride = VALUE_TOKEN_STRIDE,
        .value_head_stride = VALUE_HEAD_STRIDE,
        .value_dim_stride = VALUE_DIM_STRIDE,
        .output_head_stride = OUTPUT_HEAD_STRIDE,
        .output_dim_stride = OUTPUT_DIM_STRIDE,
    };
    static const uint32_t frontiers[] = {1, 3, MAX_KV};
    bool ok = true;
    for (size_t run = 0; run < sizeof(frontiers) / sizeof(frontiers[0]); run++) {
        args.n_kv = frontiers[run];
        memset(output.contents, 0x5a, output_bytes);
        if (!cpu_gqa_reference(
                expected, query_host, key_host, value_host, &args) ||
            !dispatch_kernel(
                device, queue, library, @"kernel_qwen35_gqa_decode_f32",
                &args, sizeof(args),
                @[query, key_cache, value_cache, output], @[],
                N_QUERY_HEAD, HEAD_DIM, true, 1, 4)) {
            ok = false;
            break;
        }

        const uint8_t *output_data = output.contents;
        for (uint32_t head = 0; head < N_QUERY_HEAD; head++) {
            for (uint32_t dim = 0; dim < HEAD_DIM; dim++) {
                const size_t offset =
                    (size_t)head * OUTPUT_HEAD_STRIDE +
                    (size_t)dim * OUTPUT_DIM_STRIDE;
                actual[(size_t)head * HEAD_DIM + dim] =
                    host_load_f32(output_data, offset);
                for (size_t padding = sizeof(float);
                     padding < OUTPUT_DIM_STRIDE; padding++) {
                    if (output_data[offset + padding] != 0x5au) {
                        fprintf(stderr, "GQA output padding overwritten\n");
                        ok = false;
                        break;
                    }
                }
                if (!ok) break;
            }
            if (!ok) break;
            const size_t tail =
                (size_t)head * OUTPUT_HEAD_STRIDE +
                (size_t)HEAD_DIM * OUTPUT_DIM_STRIDE;
            for (size_t padding = 0; padding < 16u; padding++) {
                if (output_data[tail + padding] != 0x5au) {
                    fprintf(stderr, "GQA output head padding overwritten\n");
                    ok = false;
                    break;
                }
            }
            if (!ok) break;
        }
        if (!ok) break;
        char label[64];
        snprintf(label, sizeof(label), "GQA cache frontier %u",
                 frontiers[run]);
        if (!check_f32(label, actual, expected,
                       N_QUERY_HEAD * HEAD_DIM, 2.0e-4f, 2.0e-4f)) {
            ok = false;
            break;
        }
        if (memcmp(query.contents, query_host, query_bytes) != 0 ||
            memcmp(key_cache.contents, key_snapshot, key_bytes) != 0 ||
            memcmp(value_cache.contents, value_snapshot, value_bytes) != 0) {
            fprintf(stderr, "GQA mutated a read-only input/cache\n");
            ok = false;
            break;
        }
    }

    free(query_host);
    free(key_host);
    free(value_host);
    free(key_snapshot);
    free(value_snapshot);
    free(expected);
    free(actual);
    return ok;
}

static bool test_split_q_gate(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLLibrary> library) {
    id<MTLBuffer> projection = buffer_with_bytes(
        device, qwen_attn_projection, sizeof(qwen_attn_projection));
    id<MTLBuffer> query = zero_buffer(device, sizeof(qwen_attn_query));
    id<MTLBuffer> gate = zero_buffer(device, sizeof(qwen_attn_gate));
    if (!projection || !query || !gate) return false;

    qwen35_split_args args = {
        .n_token = QWEN_ATTN_N_TOKEN,
        .n_query_head = QWEN_ATTN_N_QUERY_HEAD,
        .head_dim = QWEN_ATTN_HEAD_DIM,
        .projection_token_stride =
            QWEN_ATTN_N_QUERY_HEAD * 2u * QWEN_ATTN_HEAD_DIM * sizeof(float),
        .projection_head_stride =
            2u * QWEN_ATTN_HEAD_DIM * sizeof(float),
        .projection_dim_stride = sizeof(float),
        .query_token_stride =
            QWEN_ATTN_N_QUERY_HEAD * QWEN_ATTN_HEAD_DIM * sizeof(float),
        .query_head_stride = QWEN_ATTN_HEAD_DIM * sizeof(float),
        .query_dim_stride = sizeof(float),
        .gate_token_stride =
            QWEN_ATTN_N_QUERY_HEAD * QWEN_ATTN_HEAD_DIM * sizeof(float),
        .gate_head_stride = QWEN_ATTN_HEAD_DIM * sizeof(float),
        .gate_dim_stride = sizeof(float),
    };
    const NSUInteger count =
        QWEN_ATTN_N_TOKEN * QWEN_ATTN_N_QUERY_HEAD * QWEN_ATTN_HEAD_DIM;
    if (!dispatch_kernel(device, queue, library,
                         @"kernel_qwen35_split_q_gate_f32",
                         &args, sizeof(args), @[projection, query, gate], @[],
                         count, 64, false, 0, 0)) {
        return false;
    }
    return check_f32("split query", query.contents, qwen_attn_query,
                     count, 0.0f, 0.0f) &&
           check_f32("split gate", gate.contents, qwen_attn_gate,
                     count, 0.0f, 0.0f);
}

static bool test_sigmoid_mul(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLLibrary> library) {
    const NSUInteger count =
        sizeof(qwen_attn_output) / sizeof(qwen_attn_output[0]);
    id<MTLBuffer> input = buffer_with_bytes(
        device, qwen_attn_output, sizeof(qwen_attn_output));
    id<MTLBuffer> gate = buffer_with_bytes(
        device, qwen_attn_gate, sizeof(qwen_attn_gate));
    id<MTLBuffer> output = zero_buffer(device, sizeof(qwen_attn_gated));
    if (!input || !gate || !output) return false;
    qwen35_sigmoid_mul_args args = {
        .n_value = count,
        .input_stride = sizeof(float),
        .gate_stride = sizeof(float),
        .output_stride = sizeof(float),
    };
    if (!dispatch_kernel(device, queue, library,
                         @"kernel_qwen35_sigmoid_mul_f32",
                         &args, sizeof(args), @[input, gate, output], @[],
                         count, 64, false, 0, 0)) {
        return false;
    }
    return check_f32("elementwise sigmoid gate", output.contents,
                     qwen_attn_gated, count, 2.0e-6f, 2.0e-6f);
}

static bool test_rope(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLLibrary> library) {
    const NSUInteger count =
        sizeof(qwen_attn_query_norm) / sizeof(qwen_attn_query_norm[0]);
    id<MTLBuffer> source = buffer_with_bytes(
        device, qwen_attn_query_norm, sizeof(qwen_attn_query_norm));
    id<MTLBuffer> position = buffer_with_bytes(
        device, qwen_attn_position, sizeof(qwen_attn_position));
    id<MTLBuffer> output = zero_buffer(device, sizeof(qwen_attn_query_rope));
    if (!source || !position || !output) return false;
    qwen35_rope_args args = {
        .n_token = QWEN_ATTN_N_TOKEN,
        .n_head = QWEN_ATTN_N_QUERY_HEAD,
        .head_dim = QWEN_ATTN_HEAD_DIM,
        .n_rot = QWEN_ATTN_N_ROT,
        .theta = QWEN_ATTN_ROPE_THETA,
        .source_token_stride =
            QWEN_ATTN_N_QUERY_HEAD * QWEN_ATTN_HEAD_DIM * sizeof(float),
        .source_head_stride = QWEN_ATTN_HEAD_DIM * sizeof(float),
        .source_dim_stride = sizeof(float),
        .output_token_stride =
            QWEN_ATTN_N_QUERY_HEAD * QWEN_ATTN_HEAD_DIM * sizeof(float),
        .output_head_stride = QWEN_ATTN_HEAD_DIM * sizeof(float),
        .output_dim_stride = sizeof(float),
        .position_stride = sizeof(uint32_t),
    };
    if (!dispatch_kernel(device, queue, library,
                         @"kernel_qwen35_rope_prefix_f32",
                         &args, sizeof(args), @[source, position, output], @[],
                         count, 64, false, 0, 0)) {
        return false;
    }
    if (!check_f32("prefix RoPE out-of-place", output.contents,
                   qwen_attn_query_rope, count, 3.0e-5f, 3.0e-5f)) {
        return false;
    }

    id<MTLBuffer> in_place = buffer_with_bytes(
        device, qwen_attn_query_norm, sizeof(qwen_attn_query_norm));
    if (!in_place ||
        !dispatch_kernel(device, queue, library,
                         @"kernel_qwen35_rope_prefix_f32",
                         &args, sizeof(args), @[in_place, position, in_place], @[],
                         count, 64, false, 0, 0)) {
        return false;
    }
    return check_f32("prefix RoPE in-place", in_place.contents,
                     qwen_attn_query_rope, count, 3.0e-5f, 3.0e-5f);
}

static bool test_conv(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLLibrary> library) {
    id<MTLBuffer> input = buffer_with_bytes(
        device, qwen_ref_conv_input, sizeof(qwen_ref_conv_input));
    id<MTLBuffer> weight = buffer_with_bytes(
        device, qwen_ref_conv_weight, sizeof(qwen_ref_conv_weight));
    id<MTLBuffer> state = zero_buffer(device, sizeof(qwen_ref_conv_state));
    id<MTLBuffer> output = zero_buffer(device, sizeof(qwen_ref_conv_output));
    if (!input || !weight || !state || !output) return false;
    qwen35_conv_args args = {
        .n_channel = QWEN_REF_N_CHANNEL,
        .kernel_size = QWEN_REF_KERNEL,
        .input_channel_stride = sizeof(float),
        .weight_channel_stride = QWEN_REF_KERNEL * sizeof(float),
        .weight_tap_stride = sizeof(float),
        .state_channel_stride = (QWEN_REF_KERNEL - 1u) * sizeof(float),
        .state_tap_stride = sizeof(float),
        .output_channel_stride = sizeof(float),
    };
    const NSUInteger row_bytes = QWEN_REF_N_CHANNEL * sizeof(float);
    for (NSUInteger token = 0; token < QWEN_REF_N_TOKEN; token++) {
        NSArray<NSNumber *> *offsets = @[
            @(token * row_bytes), @0, @0, @(token * row_bytes)
        ];
        if (!dispatch_kernel(device, queue, library,
                             @"kernel_qwen35_causal_conv_step_f32",
                             &args, sizeof(args),
                             @[input, weight, state, output], offsets,
                             QWEN_REF_N_CHANNEL, 32, false, 0, 0)) {
            return false;
        }
    }
    return check_f32("causal conv output", output.contents,
                     qwen_ref_conv_output,
                     sizeof(qwen_ref_conv_output) / sizeof(float),
                     3.0e-6f, 3.0e-6f) &&
           check_f32("causal conv state", state.contents,
                     qwen_ref_conv_state,
                     sizeof(qwen_ref_conv_state) / sizeof(float),
                     0.0f, 0.0f);
}

static bool test_gated_delta(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLLibrary> library) {
    id<MTLBuffer> query = buffer_with_bytes(
        device, qwen_ref_query, sizeof(qwen_ref_query));
    id<MTLBuffer> key = buffer_with_bytes(
        device, qwen_ref_key, sizeof(qwen_ref_key));
    id<MTLBuffer> value = buffer_with_bytes(
        device, qwen_ref_value, sizeof(qwen_ref_value));
    id<MTLBuffer> log_decay = buffer_with_bytes(
        device, qwen_ref_log_decay, sizeof(qwen_ref_log_decay));
    id<MTLBuffer> beta = buffer_with_bytes(
        device, qwen_ref_beta, sizeof(qwen_ref_beta));
    id<MTLBuffer> state = buffer_with_bytes(
        device, qwen_ref_initial_state, sizeof(qwen_ref_initial_state));
    id<MTLBuffer> output = zero_buffer(device, sizeof(qwen_ref_delta_output));
    if (!query || !key || !value || !log_decay || !beta || !state || !output) {
        return false;
    }
    qwen35_delta_args args = {
        .n_key_head = QWEN_REF_N_KEY_HEAD,
        .n_value_head = QWEN_REF_N_VALUE_HEAD,
        .key_dim = QWEN_REF_KEY_DIM,
        .value_dim = QWEN_REF_VALUE_DIM,
        .query_head_stride = QWEN_REF_KEY_DIM * sizeof(float),
        .query_dim_stride = sizeof(float),
        .key_head_stride = QWEN_REF_KEY_DIM * sizeof(float),
        .key_dim_stride = sizeof(float),
        .value_head_stride = QWEN_REF_VALUE_DIM * sizeof(float),
        .value_dim_stride = sizeof(float),
        .log_decay_head_stride = sizeof(float),
        .beta_head_stride = sizeof(float),
        .state_head_stride =
            QWEN_REF_KEY_DIM * QWEN_REF_VALUE_DIM * sizeof(float),
        .state_value_stride = QWEN_REF_KEY_DIM * sizeof(float),
        .state_key_stride = sizeof(float),
        .output_head_stride = QWEN_REF_VALUE_DIM * sizeof(float),
        .output_dim_stride = sizeof(float),
    };
    const NSUInteger q_row =
        QWEN_REF_N_KEY_HEAD * QWEN_REF_KEY_DIM * sizeof(float);
    const NSUInteger v_row =
        QWEN_REF_N_VALUE_HEAD * QWEN_REF_VALUE_DIM * sizeof(float);
    const NSUInteger control_row =
        QWEN_REF_N_VALUE_HEAD * sizeof(float);
    for (NSUInteger token = 0; token < QWEN_REF_N_TOKEN; token++) {
        NSArray<NSNumber *> *offsets = @[
            @(token * q_row), @(token * q_row), @(token * v_row),
            @(token * control_row), @(token * control_row), @0,
            @(token * v_row)
        ];
        if (!dispatch_kernel(device, queue, library,
                             @"kernel_qwen35_gated_delta_step_f32",
                             &args, sizeof(args),
                             @[query, key, value, log_decay, beta, state, output],
                             offsets, QWEN_REF_N_VALUE_HEAD, 0, true, 2, 0)) {
            return false;
        }
    }
    return check_f32("Gated DeltaNet output", output.contents,
                     qwen_ref_delta_output,
                     sizeof(qwen_ref_delta_output) / sizeof(float),
                     4.0e-5f, 4.0e-5f) &&
           check_f32("Gated DeltaNet state", state.contents,
                     qwen_ref_delta_state,
                     sizeof(qwen_ref_delta_state) / sizeof(float),
                     4.0e-5f, 4.0e-5f);
}

static bool test_rmsnorm_gated(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLLibrary> library) {
    id<MTLBuffer> input = buffer_with_bytes(
        device, qwen_ref_delta_output, sizeof(qwen_ref_delta_output));
    id<MTLBuffer> gate = buffer_with_bytes(
        device, qwen_ref_gate, sizeof(qwen_ref_gate));
    id<MTLBuffer> weight = buffer_with_bytes(
        device, qwen_ref_norm_weight, sizeof(qwen_ref_norm_weight));
    id<MTLBuffer> output = zero_buffer(device, sizeof(qwen_ref_gated_output));
    if (!input || !gate || !weight || !output) return false;
    qwen35_norm_args args = {
        .n_vector = QWEN_REF_N_TOKEN * QWEN_REF_N_VALUE_HEAD,
        .dim = QWEN_REF_VALUE_DIM,
        .epsilon = 1.0e-6f,
        .input_vector_stride = QWEN_REF_VALUE_DIM * sizeof(float),
        .input_dim_stride = sizeof(float),
        .gate_vector_stride = QWEN_REF_VALUE_DIM * sizeof(float),
        .gate_dim_stride = sizeof(float),
        .weight_dim_stride = sizeof(float),
        .output_vector_stride = QWEN_REF_VALUE_DIM * sizeof(float),
        .output_dim_stride = sizeof(float),
    };
    if (!dispatch_kernel(device, queue, library,
                         @"kernel_qwen35_rmsnorm_gated_f32",
                         &args, sizeof(args),
                         @[input, gate, weight, output], @[],
                         args.n_vector, 0, true, 1, 0)) {
        return false;
    }
    return check_f32("per-head gated RMSNorm", output.contents,
                     qwen_ref_gated_output,
                     sizeof(qwen_ref_gated_output) / sizeof(float),
                     4.0e-5f, 4.0e-5f);
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        NSString *source_path = argc > 1
            ? [NSString stringWithUTF8String:argv[1]]
            : @"metal/qwen35.metal";
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
        MTLCompileOptions *options = [MTLCompileOptions new];
        /* fastMathEnabled is the compatibility spelling before macOS 15;
         * keeping it here lets the standalone test build with older SDKs. */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        options.fastMathEnabled = NO;
#pragma clang diagnostic pop
        id<MTLLibrary> library =
            [device newLibraryWithSource:source options:options error:&error];
        if (!library) {
            fprintf(stderr, "compile %s: %s\n", source_path.UTF8String,
                    error.localizedDescription.UTF8String);
            return 1;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        if (!queue) {
            fprintf(stderr, "create Metal command queue\n");
            return 2;
        }

        printf("Qwen Metal fixture on %s\n", device.name.UTF8String);
        const bool ok =
            test_embedding_q8_0(device, queue, library) &&
            test_gated_delta_controls(device, queue, library) &&
            test_split_q_gate(device, queue, library) &&
            test_sigmoid_mul(device, queue, library) &&
            test_rope(device, queue, library) &&
            test_conv(device, queue, library) &&
            test_gated_delta(device, queue, library) &&
            test_rmsnorm_gated(device, queue, library) &&
            test_gqa_decode(device, queue, library);
        if (!ok) return 1;
        puts("all Qwen Metal primitive fixtures passed");
        return 0;
    }
}
