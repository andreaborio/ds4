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

_Static_assert(sizeof(qwen35_split_args) == 88, "split ABI drift");
_Static_assert(sizeof(qwen35_sigmoid_mul_args) == 32, "sigmoid ABI drift");
_Static_assert(sizeof(qwen35_rope_args) == 80, "RoPE ABI drift");
_Static_assert(sizeof(qwen35_conv_args) == 56, "conv ABI drift");
_Static_assert(sizeof(qwen35_delta_args) == 120, "DeltaNet ABI drift");
_Static_assert(sizeof(qwen35_norm_args) == 72, "RMSNorm ABI drift");

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
        NSUInteger                       scratch_planes) {
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
    if (scratch_planes != 0) {
        const NSUInteger n_simdgroup =
            (threads + pipeline.threadExecutionWidth - 1) /
            pipeline.threadExecutionWidth;
        NSUInteger bytes =
            scratch_planes * n_simdgroup * sizeof(float);
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
                         count, 64, false, 0)) {
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
                         count, 64, false, 0)) {
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
                         count, 64, false, 0)) {
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
                         count, 64, false, 0)) {
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
                             QWEN_REF_N_CHANNEL, 32, false, 0)) {
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
                             offsets, QWEN_REF_N_VALUE_HEAD, 0, true, 2)) {
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
                         args.n_vector, 0, true, 1)) {
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
            test_split_q_gate(device, queue, library) &&
            test_sigmoid_mul(device, queue, library) &&
            test_rope(device, queue, library) &&
            test_conv(device, queue, library) &&
            test_gated_delta(device, queue, library) &&
            test_rmsnorm_gated(device, queue, library);
        if (!ok) return 1;
        puts("all Qwen Metal primitive fixtures passed");
        return 0;
    }
}
