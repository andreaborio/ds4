#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../ds4_qwen.h"
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
 *     tests/qwen/test_qwen35_metal.m ds4_qwen.c \
 *     -o /tmp/test-qwen35-metal
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
    float    eps;
    uint32_t reserved_tail;
} qwen35_split_rms_norm_args;

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
    uint32_t n_token;
    uint32_t n_key_head;
    uint32_t n_value_head;
    uint32_t key_dim;
    uint32_t value_dim;
    uint32_t reserved;
    uint64_t projection_token_stride;
    uint64_t query_offset;
    uint64_t key_offset;
    uint64_t value_offset;
    uint64_t query_head_stride;
    uint64_t query_dim_stride;
    uint64_t key_head_stride;
    uint64_t key_dim_stride;
    uint64_t value_head_stride;
    uint64_t value_dim_stride;
    uint64_t log_decay_token_stride;
    uint64_t log_decay_head_stride;
    uint64_t beta_token_stride;
    uint64_t beta_head_stride;
    uint64_t state_head_stride;
    uint64_t state_value_stride;
    uint64_t state_key_stride;
    uint64_t output_token_stride;
    uint64_t output_head_stride;
    uint64_t output_dim_stride;
} qwen35_delta_sequence_args;

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
    uint32_t n_token;
    uint32_t n_row;
    uint32_t n_embd;
    uint32_t block_size;
    uint64_t source_row_stride;
    uint64_t source_block_stride;
    uint64_t source_scale_offset;
    uint64_t source_quant_offset;
    uint64_t source_quant_stride;
    uint64_t token_id_stride;
    uint64_t output_token_stride;
    uint64_t output_dim_stride;
} qwen35_embedding_batch_args;

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
} qwen35_block_q5_k;

typedef struct {
    uint32_t n_token;
    uint32_t n_value_head;
    uint64_t alpha_logit_token_stride;
    uint64_t alpha_logit_head_stride;
    uint64_t beta_logit_token_stride;
    uint64_t beta_logit_head_stride;
    uint64_t ssm_a_head_stride;
    uint64_t dt_bias_head_stride;
    uint64_t log_decay_token_stride;
    uint64_t log_decay_head_stride;
    uint64_t beta_token_stride;
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

typedef struct {
    uint64_t logits_stride;
    uint64_t selected_stride;
    uint64_t selected_weight_stride;
} qwen35_router_top8_args;

_Static_assert(sizeof(qwen35_split_args) == 88, "split ABI drift");
_Static_assert(sizeof(qwen35_sigmoid_mul_args) == 32, "sigmoid ABI drift");
_Static_assert(sizeof(qwen35_rope_args) == 80, "RoPE ABI drift");
_Static_assert(sizeof(qwen35_conv_args) == 56, "conv ABI drift");
_Static_assert(sizeof(qwen35_delta_args) == 120, "DeltaNet ABI drift");
_Static_assert(sizeof(qwen35_delta_sequence_args) == 184,
               "DeltaNet sequence ABI drift");
_Static_assert(sizeof(qwen35_norm_args) == 72, "RMSNorm ABI drift");
_Static_assert(sizeof(qwen35_embedding_args) == 64, "embedding ABI drift");
_Static_assert(sizeof(qwen35_embedding_batch_args) == 80,
               "batched embedding ABI drift");
_Static_assert(sizeof(qwen35_block_q5_k) == 176, "Q5_K block ABI drift");
_Static_assert(sizeof(qwen35_controls_args) == 88, "controls ABI drift");
_Static_assert(sizeof(qwen35_gqa_args) == 96, "GQA ABI drift");
_Static_assert(sizeof(qwen35_router_top8_args) == 24, "router top-8 ABI drift");

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

static bool dispatch_gated_delta_sequence(
        id<MTLDevice>                    device,
        id<MTLCommandQueue>              queue,
        id<MTLLibrary>                   library,
        NSString                        *sequence_name,
        const qwen35_delta_sequence_args *args,
        id<MTLBuffer>                    projection,
        id<MTLBuffer>                    decay,
        id<MTLBuffer>                    beta,
        id<MTLBuffer>                    state,
        id<MTLBuffer>                    output) {
    id<MTLFunction> normalize_function = [library
        newFunctionWithName:@"kernel_qwen35_normalize_qk_sequence_128_f32"];
    id<MTLFunction> sequence_function =
        [library newFunctionWithName:sequence_name];
    if (!normalize_function || !sequence_function) {
        fprintf(stderr, "missing Metal GDN sequence function %s\n",
                sequence_name.UTF8String);
        return false;
    }
    NSError *error = nil;
    id<MTLComputePipelineState> normalize_pipeline =
        [device newComputePipelineStateWithFunction:normalize_function
                                              error:&error];
    if (!normalize_pipeline) {
        fprintf(stderr, "normalize GDN pipeline: %s\n",
                error.localizedDescription.UTF8String);
        return false;
    }
    error = nil;
    id<MTLComputePipelineState> sequence_pipeline =
        [device newComputePipelineStateWithFunction:sequence_function
                                              error:&error];
    if (!sequence_pipeline) {
        fprintf(stderr, "GDN sequence pipeline %s: %s\n",
                sequence_name.UTF8String,
                error.localizedDescription.UTF8String);
        return false;
    }
    if (normalize_pipeline.maxTotalThreadsPerThreadgroup < 32u ||
        sequence_pipeline.maxTotalThreadsPerThreadgroup < 128u) {
        fprintf(stderr, "unsupported GDN sequence threadgroup geometry\n");
        return false;
    }

    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:normalize_pipeline];
    [encoder setBytes:args length:sizeof(*args) atIndex:0];
    [encoder setBuffer:projection offset:0 atIndex:1];
    [encoder dispatchThreadgroups:
         MTLSizeMake(args->n_token, args->n_key_head, 1u)
         threadsPerThreadgroup:MTLSizeMake(32u, 1u, 1u)];

    [encoder setComputePipelineState:sequence_pipeline];
    [encoder setBytes:args length:sizeof(*args) atIndex:0];
    [encoder setBuffer:projection offset:0 atIndex:1];
    [encoder setBuffer:decay offset:0 atIndex:2];
    [encoder setBuffer:beta offset:0 atIndex:3];
    [encoder setBuffer:state offset:0 atIndex:4];
    [encoder setBuffer:output offset:0 atIndex:5];
    [encoder dispatchThreadgroups:
         MTLSizeMake((args->value_dim + 3u) / 4u,
                     args->n_value_head, 1u)
         threadsPerThreadgroup:MTLSizeMake(32u, 4u, 1u)];
    [encoder endEncoding];
    [command commit];
    [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "GDN sequence %s: %s\n",
                sequence_name.UTF8String,
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

static void q5_k_scale_min(
        uint32_t group,
        const uint8_t scales[12],
        uint8_t *scale,
        uint8_t *minimum) {
    if (group < 4u) {
        *scale = scales[group] & 63u;
        *minimum = scales[group + 4u] & 63u;
    } else {
        *scale = (scales[group + 4u] & 0x0fu) |
                 ((scales[group - 4u] & 0xc0u) >> 2);
        *minimum = (scales[group + 4u] >> 4) |
                   ((scales[group] & 0xc0u) >> 2);
    }
}

static float q5_k_value(
        const qwen35_block_q5_k *block,
        uint32_t within_block) {
    const uint32_t group = within_block / 32u;
    const uint32_t lane = within_block - group * 32u;
    uint8_t scale = 0;
    uint8_t minimum = 0;
    q5_k_scale_min(group, block->scales, &scale, &minimum);
    const uint32_t ql_base = (group >> 1u) * 32u + lane;
    const uint32_t shift = (group & 1u) * 4u;
    uint32_t quant = (block->qs[ql_base] >> shift) & 0x0fu;
    if ((block->qh[lane] & (uint8_t)(1u << group)) != 0u) quant += 16u;
    return f16_bits_to_f32(block->d) * (float)scale * (float)quant -
           f16_bits_to_f32(block->dmin) * (float)minimum;
}

static void host_store_f32(uint8_t *base, size_t offset, float value) {
    memcpy(base + offset, &value, sizeof(value));
}

static float host_load_f32(const uint8_t *base, size_t offset) {
    float value = 0.0f;
    memcpy(&value, base + offset, sizeof(value));
    return value;
}

static int32_t host_load_i32(const uint8_t *base, size_t offset) {
    int32_t value = 0;
    memcpy(&value, base + offset, sizeof(value));
    return value;
}

static bool check_strided_guard(
        const char    *name,
        const uint8_t *bytes,
        size_t         byte_count,
        size_t         prefix,
        size_t         stride,
        size_t         value_size,
        size_t         count,
        uint8_t        guard) {
    for (size_t offset = 0; offset < byte_count; offset++) {
        bool writable = false;
        for (size_t slot = 0; slot < count; slot++) {
            const size_t begin = prefix + slot * stride;
            if (offset >= begin && offset < begin + value_size) {
                writable = true;
                break;
            }
        }
        if (!writable && bytes[offset] != guard) {
            fprintf(stderr, "%s guard overwritten at byte %zu\n", name, offset);
            return false;
        }
    }
    return true;
}

static uint32_t router_rng_state = 0x91e0f5d1u;

static uint32_t router_random_u32(void) {
    uint32_t value = router_rng_state;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    router_rng_state = value;
    return value;
}

static float router_random_logit(void) {
    const int32_t centered = (int32_t)(router_random_u32() % 2000001u) - 1000000;
    return (float)centered * (64.0f / 1000000.0f);
}

static bool run_router_top8_case(
        id<MTLDevice>       device,
        id<MTLCommandQueue> queue,
        id<MTLLibrary>      library,
        const char         *name,
        const float         logits[QWEN35_N_EXPERT],
        float              *maximum_weight_error) {
    enum {
        PREFIX = 32,
        SUFFIX = 32,
        LOGITS_STRIDE = 8,
        SELECTED_STRIDE = 12,
        WEIGHT_STRIDE = 16,
        GUARD_LOGITS = 0xc3,
        GUARD_SELECTED = 0xa5,
        GUARD_WEIGHT = 0x5a,
    };
    const size_t logits_bytes =
        PREFIX + (QWEN35_N_EXPERT - 1u) * LOGITS_STRIDE + sizeof(float) + SUFFIX;
    const size_t selected_bytes =
        PREFIX + (QWEN35_N_EXPERT_USED - 1u) * SELECTED_STRIDE +
        sizeof(int32_t) + SUFFIX;
    const size_t weight_bytes =
        PREFIX + (QWEN35_N_EXPERT_USED - 1u) * WEIGHT_STRIDE +
        sizeof(float) + SUFFIX;
    uint8_t *logits_host = malloc(logits_bytes);
    uint8_t *logits_snapshot = malloc(logits_bytes);
    if (!logits_host || !logits_snapshot) {
        free(logits_host);
        free(logits_snapshot);
        return false;
    }
    memset(logits_host, GUARD_LOGITS, logits_bytes);
    for (size_t expert = 0; expert < QWEN35_N_EXPERT; expert++) {
        host_store_f32(
            logits_host,
            PREFIX + expert * LOGITS_STRIDE,
            logits[expert]);
    }
    memcpy(logits_snapshot, logits_host, logits_bytes);

    id<MTLBuffer> logits_buffer = buffer_with_bytes(
        device, logits_host, logits_bytes);
    id<MTLBuffer> selected_buffer = [device newBufferWithLength:selected_bytes
                                                        options:MTLResourceStorageModeShared];
    id<MTLBuffer> weight_buffer = [device newBufferWithLength:weight_bytes
                                                      options:MTLResourceStorageModeShared];
    if (!logits_buffer || !selected_buffer || !weight_buffer) {
        free(logits_host);
        free(logits_snapshot);
        return false;
    }
    memset(selected_buffer.contents, GUARD_SELECTED, selected_bytes);
    memset(weight_buffer.contents, GUARD_WEIGHT, weight_bytes);

    qwen35_router_top8_args args = {
        .logits_stride = LOGITS_STRIDE,
        .selected_stride = SELECTED_STRIDE,
        .selected_weight_stride = WEIGHT_STRIDE,
    };
    bool ok = dispatch_kernel(
        device, queue, library,
        @"kernel_qwen35_router_softmax_top8_f32",
        &args, sizeof(args),
        @[logits_buffer, selected_buffer, weight_buffer],
        @[@(PREFIX), @(PREFIX), @(PREFIX)],
        1, QWEN35_N_EXPERT, true, 0, QWEN35_N_EXPERT);

    int32_t expected_selected[QWEN35_N_EXPERT_USED];
    float expected_weight[QWEN35_N_EXPERT_USED];
    float probability[QWEN35_N_EXPERT];
    if (ok && !ds4_qwen35_cpu_softmax_top8_f32(
            expected_selected, expected_weight, probability, logits)) {
        fprintf(stderr, "%s: production CPU oracle rejected finite logits\n", name);
        ok = false;
    }

    float weight_sum = 0.0f;
    float case_maximum_error = 0.0f;
    const uint8_t *selected_bytes_ptr = selected_buffer.contents;
    const uint8_t *weight_bytes_ptr = weight_buffer.contents;
    if (ok) {
        for (size_t slot = 0; slot < QWEN35_N_EXPERT_USED; slot++) {
            const int32_t actual_selected = host_load_i32(
                selected_bytes_ptr, PREFIX + slot * SELECTED_STRIDE);
            const float actual_weight = host_load_f32(
                weight_bytes_ptr, PREFIX + slot * WEIGHT_STRIDE);
            const float error = fabsf(actual_weight - expected_weight[slot]);
            if (actual_selected != expected_selected[slot]) {
                fprintf(stderr,
                        "%s selected[%zu]: Metal %d CPU %d\n",
                        name, slot, actual_selected, expected_selected[slot]);
                ok = false;
                break;
            }
            if (!isfinite(actual_weight) || error > 3.0e-6f) {
                fprintf(stderr,
                        "%s weight[%zu]: Metal %.9g CPU %.9g error %.9g\n",
                        name, slot, actual_weight, expected_weight[slot], error);
                ok = false;
                break;
            }
            if (error > case_maximum_error) case_maximum_error = error;
            weight_sum += actual_weight;
        }
    }
    if (ok && fabsf(weight_sum - 1.0f) > 2.0e-6f) {
        fprintf(stderr, "%s: selected weights sum to %.9g\n", name, weight_sum);
        ok = false;
    }
    if (ok && memcmp(logits_buffer.contents, logits_snapshot, logits_bytes) != 0) {
        fprintf(stderr, "%s: router mutated logits or input padding\n", name);
        ok = false;
    }
    if (ok && !check_strided_guard(
            "router selected", selected_bytes_ptr, selected_bytes,
            PREFIX, SELECTED_STRIDE, sizeof(int32_t),
            QWEN35_N_EXPERT_USED, GUARD_SELECTED)) {
        ok = false;
    }
    if (ok && !check_strided_guard(
            "router weight", weight_bytes_ptr, weight_bytes,
            PREFIX, WEIGHT_STRIDE, sizeof(float),
            QWEN35_N_EXPERT_USED, GUARD_WEIGHT)) {
        ok = false;
    }
    if (ok && case_maximum_error > *maximum_weight_error) {
        *maximum_weight_error = case_maximum_error;
    }

    free(logits_host);
    free(logits_snapshot);
    return ok;
}

static bool test_router_softmax_top8(
        id<MTLDevice>       device,
        id<MTLCommandQueue> queue,
        id<MTLLibrary>      library) {
    enum { RANDOM_CASE_COUNT = 24 };
    float logits[QWEN35_N_EXPERT];
    float maximum_weight_error = 0.0f;
    int32_t boundary_selected[QWEN35_N_EXPERT_USED];
    float boundary_weight[QWEN35_N_EXPERT_USED];
    float boundary_probability[QWEN35_N_EXPERT];

    for (size_t run = 0; run < RANDOM_CASE_COUNT; run++) {
        for (size_t expert = 0; expert < QWEN35_N_EXPERT; expert++) {
            logits[expert] = router_random_logit();
        }
        char label[48];
        snprintf(label, sizeof(label), "router random %zu", run);
        if (!run_router_top8_case(
                device, queue, library, label, logits,
                &maximum_weight_error)) {
            return false;
        }
    }

    for (size_t expert = 0; expert < QWEN35_N_EXPERT; expert++) {
        logits[expert] = -100000.0f;
    }
    for (size_t slot = 0; slot < QWEN35_N_EXPERT_USED; slot++) {
        logits[31u * slot + 7u] = 100000.0f - 1000.0f * (float)slot;
    }
    if (!run_router_top8_case(
            device, queue, library, "router extreme finite", logits,
            &maximum_weight_error)) {
        return false;
    }

    for (size_t expert = 0; expert < QWEN35_N_EXPERT; expert++) {
        logits[expert] = -100.0f;
    }
    for (size_t slot = 0; slot < 6u; slot++) {
        logits[100u + slot] = 10.0f - (float)slot;
    }
    logits[17] = logits[19] = logits[23] = 0.0f;
    if (!ds4_qwen35_cpu_softmax_top8_f32(
            boundary_selected, boundary_weight, boundary_probability, logits) ||
        boundary_selected[6] != 17 || boundary_selected[7] != 19) {
        fprintf(stderr, "CPU oracle tie policy drifted at the 8th boundary\n");
        return false;
    }
    if (!run_router_top8_case(
            device, queue, library, "router tied 8th boundary", logits,
            &maximum_weight_error)) {
        return false;
    }

    for (size_t expert = 0; expert < QWEN35_N_EXPERT; expert++) {
        logits[expert] = 3.25f;
    }
    if (!run_router_top8_case(
            device, queue, library, "router all tied", logits,
            &maximum_weight_error)) {
        return false;
    }

    for (size_t expert = 0; expert < QWEN35_N_EXPERT; expert++) {
        logits[expert] = -20.0f;
    }
    for (size_t slot = 0; slot < 7u; slot++) {
        logits[240u + slot] = 20.0f - (float)slot;
    }
    logits[201] = 0.001f;
    logits[3] = 0.0f;
    if (!ds4_qwen35_cpu_softmax_top8_f32(
            boundary_selected, boundary_weight, boundary_probability, logits) ||
        boundary_selected[7] != 201) {
        fprintf(stderr, "CPU oracle did not distinguish the 8th/9th boundary\n");
        return false;
    }
    if (!run_router_top8_case(
            device, queue, library, "router distinct 8th/9th", logits,
            &maximum_weight_error)) {
        return false;
    }

    printf("ok %-28s cases=%u ids=exact max_abs_error=%.3g\n",
           "router softmax top-8", RANDOM_CASE_COUNT + 4u,
           maximum_weight_error);
    return true;
}

static bool test_embedding_q5_k(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLLibrary> library) {
    enum {
        N_EMBD = 512,
        Q5_BLOCK = 256,
        BLOCK_COUNT = N_EMBD / Q5_BLOCK,
        ROW_COUNT = 3,
        ROW_PADDING = 16,
        ROW_STRIDE = BLOCK_COUNT * sizeof(qwen35_block_q5_k) + ROW_PADDING,
        OUTPUT_STRIDE = 8,
        BATCH_TOKEN_COUNT = 2,
        TOKEN_ID_STRIDE = 8,
        OUTPUT_TOKEN_PADDING = 16,
        OUTPUT_TOKEN_STRIDE = N_EMBD * OUTPUT_STRIDE + OUTPUT_TOKEN_PADDING,
    };
    uint8_t *encoded = calloc(ROW_COUNT, ROW_STRIDE);
    float *expected = malloc(N_EMBD * sizeof(expected[0]));
    float *actual = malloc(N_EMBD * sizeof(actual[0]));
    float *batch_expected =
        malloc(BATCH_TOKEN_COUNT * N_EMBD * sizeof(batch_expected[0]));
    float *batch_actual =
        malloc(BATCH_TOKEN_COUNT * N_EMBD * sizeof(batch_actual[0]));
    if (!encoded || !expected || !actual || !batch_expected || !batch_actual) {
        free(encoded);
        free(expected);
        free(actual);
        free(batch_expected);
        free(batch_actual);
        return false;
    }

    for (uint32_t row = 0; row < ROW_COUNT; row++) {
        for (uint32_t block = 0; block < BLOCK_COUNT; block++) {
            qwen35_block_q5_k *encoded_block =
                (qwen35_block_q5_k *)(encoded + (size_t)row * ROW_STRIDE) +
                block;
            encoded_block->d =
                f32_to_f16_bits(0.03125f * (float)(1u + row + block));
            encoded_block->dmin =
                f32_to_f16_bits(0.015625f * (float)(1u + 2u * row + block));
            for (uint32_t index = 0; index < 12u; index++) {
                encoded_block->scales[index] =
                    (uint8_t)(11u + row * 41u + block * 23u + index * 17u);
            }
            for (uint32_t index = 0; index < 32u; index++) {
                encoded_block->qh[index] =
                    (uint8_t)(5u + row * 31u + block * 13u + index * 29u);
            }
            for (uint32_t index = 0; index < 128u; index++) {
                encoded_block->qs[index] =
                    (uint8_t)(7u + row * 19u + block * 43u + index * 37u);
            }
        }
    }
    const qwen35_block_q5_k *direct_row =
        (const qwen35_block_q5_k *)(encoded + ROW_STRIDE);
    for (uint32_t dim = 0; dim < N_EMBD; dim++) {
        expected[dim] =
            q5_k_value(direct_row + dim / Q5_BLOCK, dim % Q5_BLOCK);
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
        free(batch_expected);
        free(batch_actual);
        return false;
    }
    memset(output.contents, 0xa5, output_bytes);
    qwen35_embedding_args args = {
        .row_index = 1,
        .n_embd = N_EMBD,
        .block_size = Q5_BLOCK,
        .source_row_stride = ROW_STRIDE,
        .source_block_stride = sizeof(qwen35_block_q5_k),
        .output_dim_stride = OUTPUT_STRIDE,
    };
    bool ok = dispatch_kernel(
        device, queue, library,
        @"kernel_qwen35_dequant_embedding_q5_K_f32",
        &args, sizeof(args), @[embedding, output], @[],
        N_EMBD, 64, false, 0, 0);
    if (ok) {
        const uint8_t *bytes = output.contents;
        for (size_t dim = 0; dim < N_EMBD; dim++) {
            actual[dim] = host_load_f32(bytes, dim * OUTPUT_STRIDE);
        }
        ok = check_strided_guard(
            "Q5_K embedding output", bytes, output_bytes,
            0, OUTPUT_STRIDE, sizeof(float), N_EMBD, 0xa5u);
    }
    if (ok) {
        ok = check_f32(
            "Q5_K embedding row", actual, expected, N_EMBD, 1.0e-6f, 1.0e-6f);
    }

    uint8_t token_ids[BATCH_TOKEN_COUNT * TOKEN_ID_STRIDE];
    memset(token_ids, 0x5a, sizeof(token_ids));
    const int32_t rows[BATCH_TOKEN_COUNT] = {2, 0};
    for (uint32_t token = 0; token < BATCH_TOKEN_COUNT; token++) {
        memcpy(token_ids + token * TOKEN_ID_STRIDE,
               &rows[token], sizeof(rows[token]));
        const qwen35_block_q5_k *row =
            (const qwen35_block_q5_k *)(
                encoded + (size_t)rows[token] * ROW_STRIDE);
        for (uint32_t dim = 0; dim < N_EMBD; dim++) {
            batch_expected[(size_t)token * N_EMBD + dim] =
                q5_k_value(row + dim / Q5_BLOCK, dim % Q5_BLOCK);
        }
    }
    const NSUInteger batch_output_bytes =
        (BATCH_TOKEN_COUNT - 1u) * OUTPUT_TOKEN_STRIDE +
        (N_EMBD - 1u) * OUTPUT_STRIDE + sizeof(float);
    id<MTLBuffer> token_buffer =
        buffer_with_bytes(device, token_ids, sizeof(token_ids));
    id<MTLBuffer> batch_output =
        [device newBufferWithLength:batch_output_bytes
                            options:MTLResourceStorageModeShared];
    if (ok && (!token_buffer || !batch_output)) ok = false;
    if (ok) {
        memset(batch_output.contents, 0xa5, batch_output_bytes);
        qwen35_embedding_batch_args batch_args = {
            .n_token = BATCH_TOKEN_COUNT,
            .n_row = ROW_COUNT,
            .n_embd = N_EMBD,
            .block_size = Q5_BLOCK,
            .source_row_stride = ROW_STRIDE,
            .source_block_stride = sizeof(qwen35_block_q5_k),
            .token_id_stride = TOKEN_ID_STRIDE,
            .output_token_stride = OUTPUT_TOKEN_STRIDE,
            .output_dim_stride = OUTPUT_STRIDE,
        };
        ok = dispatch_kernel(
            device, queue, library,
            @"kernel_qwen35_dequant_embedding_q5_K_batch_f32",
            &batch_args, sizeof(batch_args),
            @[embedding, token_buffer, batch_output], @[],
            BATCH_TOKEN_COUNT * N_EMBD, 64, false, 0, 0);
    }
    if (ok) {
        const uint8_t *bytes = batch_output.contents;
        for (size_t token = 0; token < BATCH_TOKEN_COUNT; token++) {
            for (size_t dim = 0; dim < N_EMBD; dim++) {
                batch_actual[token * N_EMBD + dim] = host_load_f32(
                    bytes, token * OUTPUT_TOKEN_STRIDE + dim * OUTPUT_STRIDE);
            }
        }
        ok = check_f32(
            "batched Q5_K embedding", batch_actual, batch_expected,
            BATCH_TOKEN_COUNT * N_EMBD, 1.0e-6f, 1.0e-6f);
    }
    if (ok && memcmp(embedding.contents, encoded,
                     ROW_COUNT * ROW_STRIDE) != 0) {
        fprintf(stderr, "Q5_K embedding dequant mutated its source rows\n");
        ok = false;
    }
    free(encoded);
    free(expected);
    free(actual);
    free(batch_expected);
    free(batch_actual);
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
    const NSUInteger row_bytes = QWEN_REF_N_VALUE_HEAD * sizeof(float);
    qwen35_controls_args args = {
        .n_token = QWEN_REF_N_TOKEN,
        .n_value_head = QWEN_REF_N_VALUE_HEAD,
        .alpha_logit_token_stride = row_bytes,
        .alpha_logit_head_stride = sizeof(float),
        .beta_logit_token_stride = row_bytes,
        .beta_logit_head_stride = sizeof(float),
        .ssm_a_head_stride = sizeof(float),
        .dt_bias_head_stride = sizeof(float),
        .log_decay_token_stride = row_bytes,
        .log_decay_head_stride = sizeof(float),
        .beta_token_stride = row_bytes,
        .beta_head_stride = sizeof(float),
    };
    const NSUInteger count =
        QWEN_REF_N_TOKEN * QWEN_REF_N_VALUE_HEAD;
    if (!dispatch_kernel(
            device, queue, library,
            @"kernel_qwen35_gated_delta_controls_f32",
            &args, sizeof(args),
            @[alpha, beta_logit, ssm_a, dt_bias, log_decay, beta],
            @[], count, 32, false, 0, 0)) {
        return false;
    }
    return check_f32(
               "DeltaNet log-decay", log_decay.contents,
               qwen_ref_log_decay, count, 3.0e-6f, 3.0e-6f) &&
           check_f32(
               "DeltaNet beta", beta.contents, qwen_ref_beta,
               count, 2.0e-6f, 2.0e-6f);
}

static bool test_gated_delta_sequence(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLLibrary> library) {
    enum {
        N_TOKEN = QWEN_REF_N_TOKEN,
        N_KEY_HEAD = QWEN_REF_N_KEY_HEAD,
        N_VALUE_HEAD = QWEN_REF_N_VALUE_HEAD,
        KEY_DIM = 128,
        VALUE_DIM = 128,
        PROJECTION_ROW =
            2 * N_KEY_HEAD * KEY_DIM + N_VALUE_HEAD * VALUE_DIM,
        PROJECTION_COUNT = N_TOKEN * PROJECTION_ROW,
        STATE_COUNT = N_VALUE_HEAD * VALUE_DIM * KEY_DIM,
        OUTPUT_COUNT = N_TOKEN * N_VALUE_HEAD * VALUE_DIM,
        CONTROL_COUNT = N_TOKEN * N_VALUE_HEAD,
    };
    float *projection_host = malloc(
        (size_t)PROJECTION_COUNT * sizeof(projection_host[0]));
    float *state_host = malloc((size_t)STATE_COUNT * sizeof(state_host[0]));
    if (!projection_host || !state_host) {
        free(projection_host);
        free(state_host);
        return false;
    }
    for (size_t i = 0; i < PROJECTION_COUNT; i++) {
        projection_host[i] =
            ((float)((i * 17u + 11u) % 101u) - 50.0f) / 127.0f;
    }
    for (size_t i = 0; i < STATE_COUNT; i++) {
        state_host[i] =
            ((float)((i * 29u + 7u) % 97u) - 48.0f) / 251.0f;
    }

    id<MTLBuffer> alpha = buffer_with_bytes(
        device, qwen_ref_alpha_logit, sizeof(qwen_ref_alpha_logit));
    id<MTLBuffer> beta_logit = buffer_with_bytes(
        device, qwen_ref_beta_logit, sizeof(qwen_ref_beta_logit));
    id<MTLBuffer> ssm_a = buffer_with_bytes(
        device, qwen_ref_ssm_a, sizeof(qwen_ref_ssm_a));
    id<MTLBuffer> dt_bias = buffer_with_bytes(
        device, qwen_ref_dt_bias, sizeof(qwen_ref_dt_bias));
    id<MTLBuffer> log_decay = zero_buffer(
        device, CONTROL_COUNT * sizeof(float));
    id<MTLBuffer> beta = zero_buffer(
        device, CONTROL_COUNT * sizeof(float));
    id<MTLBuffer> projection0 = buffer_with_bytes(
        device, projection_host, PROJECTION_COUNT * sizeof(float));
    id<MTLBuffer> state0 = buffer_with_bytes(
        device, state_host, STATE_COUNT * sizeof(float));
    id<MTLBuffer> output0 = zero_buffer(
        device, OUTPUT_COUNT * sizeof(float));
    free(projection_host);
    free(state_host);
    if (!alpha || !beta_logit || !ssm_a || !dt_bias || !log_decay ||
        !beta || !projection0 || !state0 || !output0) {
        return false;
    }

    const uint64_t control_row =
        (uint64_t)N_VALUE_HEAD * sizeof(float);
    qwen35_controls_args controls = {
        .n_token = N_TOKEN,
        .n_value_head = N_VALUE_HEAD,
        .alpha_logit_token_stride = control_row,
        .alpha_logit_head_stride = sizeof(float),
        .beta_logit_token_stride = control_row,
        .beta_logit_head_stride = sizeof(float),
        .ssm_a_head_stride = sizeof(float),
        .dt_bias_head_stride = sizeof(float),
        .log_decay_token_stride = control_row,
        .log_decay_head_stride = sizeof(float),
        .beta_token_stride = control_row,
        .beta_head_stride = sizeof(float),
    };
    if (!dispatch_kernel(
            device, queue, library,
            @"kernel_qwen35_gated_delta_controls_f32",
            &controls, sizeof(controls),
            @[alpha, beta_logit, ssm_a, dt_bias, log_decay, beta],
            @[], CONTROL_COUNT, 32, false, 0, 0)) {
        return false;
    }

    const uint64_t query_bytes =
        (uint64_t)N_KEY_HEAD * KEY_DIM * sizeof(float);
    const uint64_t key_bytes = query_bytes;
    qwen35_delta_sequence_args args = {
        .n_token = N_TOKEN,
        .n_key_head = N_KEY_HEAD,
        .n_value_head = N_VALUE_HEAD,
        .key_dim = KEY_DIM,
        .value_dim = VALUE_DIM,
        .projection_token_stride =
            (uint64_t)PROJECTION_ROW * sizeof(float),
        .query_offset = 0,
        .key_offset = query_bytes,
        .value_offset = query_bytes + key_bytes,
        .query_head_stride = KEY_DIM * sizeof(float),
        .query_dim_stride = sizeof(float),
        .key_head_stride = KEY_DIM * sizeof(float),
        .key_dim_stride = sizeof(float),
        .value_head_stride = VALUE_DIM * sizeof(float),
        .value_dim_stride = sizeof(float),
        .log_decay_token_stride = control_row,
        .log_decay_head_stride = sizeof(float),
        .beta_token_stride = control_row,
        .beta_head_stride = sizeof(float),
        .state_head_stride =
            (uint64_t)VALUE_DIM * KEY_DIM * sizeof(float),
        .state_value_stride = KEY_DIM * sizeof(float),
        .state_key_stride = sizeof(float),
        .output_token_stride =
            (uint64_t)N_VALUE_HEAD * VALUE_DIM * sizeof(float),
        .output_head_stride = VALUE_DIM * sizeof(float),
        .output_dim_stride = sizeof(float),
    };
    if (!dispatch_gated_delta_sequence(
            device, queue, library,
            @"kernel_qwen35_gated_delta_sequence_128_normalized_f32",
            &args, projection0, log_decay, beta, state0, output0)) {
        return false;
    }
    puts("ok DeltaNet batched controls-to-sequence path");
    return true;
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
        MAX_KV = 4097,
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
    static const uint32_t serial_frontiers[] = {1, 3, 7};
    static const uint32_t parallel_frontiers[] = {1, 7, 257, 1025, MAX_KV};
    bool ok = true;
    for (int variant = 0; variant < 2 && ok; variant++) {
        const uint32_t *frontiers = variant == 0
            ? serial_frontiers : parallel_frontiers;
        const size_t frontier_count = variant == 0
            ? sizeof(serial_frontiers) / sizeof(serial_frontiers[0])
            : sizeof(parallel_frontiers) / sizeof(parallel_frontiers[0]);
        NSString *kernel = variant == 0
            ? @"kernel_qwen35_gqa_decode_f32"
            : @"kernel_qwen35_gqa_decode_parallel_f32";
        const NSUInteger scratch_planes = variant == 0 ? 1u : HEAD_DIM + 2u;
        const NSUInteger scratch_extra = variant == 0 ? 4u : 1u;
        for (size_t run = 0; run < frontier_count; run++) {
            args.n_kv = frontiers[run];
            memset(output.contents, 0x5a, output_bytes);
            if (!cpu_gqa_reference(
                    expected, query_host, key_host, value_host, &args) ||
                !dispatch_kernel(
                    device, queue, library, kernel,
                    &args, sizeof(args),
                    @[query, key_cache, value_cache, output], @[],
                    N_QUERY_HEAD, HEAD_DIM, true,
                    scratch_planes, scratch_extra)) {
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
            char label[80];
            snprintf(label, sizeof(label), "%s GQA cache frontier %u",
                     variant == 0 ? "serial" : "parallel", frontiers[run]);
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

static bool test_split_q_gate_rms_norm(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLLibrary> library) {
    enum {
        FUSED_N_TOKEN = 2,
        FUSED_N_QUERY_HEAD = 2,
        FUSED_HEAD_DIM = 256,
        FUSED_ROWS = FUSED_N_TOKEN * FUSED_N_QUERY_HEAD,
        FUSED_VALUES = FUSED_ROWS * FUSED_HEAD_DIM,
    };
    float projection_host[2 * FUSED_VALUES];
    float norm_weight[FUSED_HEAD_DIM];
    float expected_query[FUSED_VALUES];
    float expected_gate[FUSED_VALUES];
    for (NSUInteger dim = 0; dim < FUSED_HEAD_DIM; dim++) {
        norm_weight[dim] =
            0.75f + (float)((int)(dim % 17u) - 8) * 0.03125f;
    }
    for (NSUInteger row = 0; row < FUSED_ROWS; row++) {
        const NSUInteger projection_base = row * 2u * FUSED_HEAD_DIM;
        const NSUInteger output_base = row * FUSED_HEAD_DIM;
        for (NSUInteger dim = 0; dim < FUSED_HEAD_DIM; dim++) {
            const float q =
                (float)((int)((row * 29u + dim * 7u) % 53u) - 26) /
                13.0f;
            const float g =
                (float)((int)((row * 11u + dim * 5u) % 47u) - 23) /
                17.0f;
            projection_host[projection_base + dim] = q;
            projection_host[projection_base + FUSED_HEAD_DIM + dim] = g;
            expected_gate[output_base + dim] = g;
        }
    }
    const NSUInteger rows = FUSED_ROWS;
    for (NSUInteger row = 0; row < rows; row++) {
        const NSUInteger projection_base = row * 2u * FUSED_HEAD_DIM;
        const NSUInteger output_base = row * FUSED_HEAD_DIM;
        float sum = 0.0f;
        for (NSUInteger dim = 0; dim < FUSED_HEAD_DIM; dim++) {
            const float value = projection_host[projection_base + dim];
            sum += value * value;
        }
        const float scale =
            1.0f / sqrtf(sum / FUSED_HEAD_DIM + 1.0e-6f);
        for (NSUInteger dim = 0; dim < FUSED_HEAD_DIM; dim++) {
            expected_query[output_base + dim] =
                (projection_host[projection_base + dim] * scale) *
                norm_weight[dim];
        }
    }

    id<MTLBuffer> projection = buffer_with_bytes(
        device, projection_host, sizeof(projection_host));
    id<MTLBuffer> weight = buffer_with_bytes(
        device, norm_weight, sizeof(norm_weight));
    id<MTLBuffer> query = zero_buffer(device, sizeof(expected_query));
    id<MTLBuffer> gate = zero_buffer(device, sizeof(expected_gate));
    if (!projection || !weight || !query || !gate) return false;

    qwen35_split_rms_norm_args args = {
        .n_token = FUSED_N_TOKEN,
        .n_query_head = FUSED_N_QUERY_HEAD,
        .head_dim = FUSED_HEAD_DIM,
        .projection_token_stride =
            FUSED_N_QUERY_HEAD * 2u * FUSED_HEAD_DIM * sizeof(float),
        .projection_head_stride =
            2u * FUSED_HEAD_DIM * sizeof(float),
        .projection_dim_stride = sizeof(float),
        .query_token_stride =
            FUSED_N_QUERY_HEAD * FUSED_HEAD_DIM * sizeof(float),
        .query_head_stride = FUSED_HEAD_DIM * sizeof(float),
        .query_dim_stride = sizeof(float),
        .gate_token_stride =
            FUSED_N_QUERY_HEAD * FUSED_HEAD_DIM * sizeof(float),
        .gate_head_stride = FUSED_HEAD_DIM * sizeof(float),
        .gate_dim_stride = sizeof(float),
        .eps = 1.0e-6f,
    };
    const NSUInteger count = FUSED_VALUES;
    if (!dispatch_kernel(
            device, queue, library,
            @"kernel_qwen35_split_q_gate_rms_norm_f32",
            &args, sizeof(args),
            @[projection, weight, query, gate], @[],
            rows, 64u, true, 0u, 32u)) {
        return false;
    }
    return check_f32("fused split normalized query", query.contents,
                     expected_query, count, 5.0e-6f, 5.0e-6f) &&
           check_f32("fused split gate", gate.contents, expected_gate,
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
        if (getenv("DS4_TEST_QWEN_GQA_ONLY") != NULL) {
            const bool ok = test_gqa_decode(device, queue, library);
            if (!ok) return 1;
            puts("Qwen Metal GQA fixtures passed");
            return 0;
        }
        if (getenv("DS4_TEST_QWEN_EMBEDDING_Q5_ONLY") != NULL) {
            const bool ok = test_embedding_q5_k(device, queue, library);
            if (!ok) return 1;
            puts("Qwen Metal Q5_K embedding fixtures passed");
            return 0;
        }
        if (getenv("DS4_TEST_QWEN_GDN_CONTROLS_ONLY") != NULL) {
            const bool ok =
                test_gated_delta_controls(device, queue, library) &&
                test_gated_delta_sequence(device, queue, library);
            if (!ok) return 1;
            puts("Qwen Metal GDN control fixtures passed");
            return 0;
        }
        if (getenv("DS4_TEST_QWEN_FUSED_SPLIT_Q_NORM_ONLY") != NULL) {
            const bool ok =
                test_split_q_gate_rms_norm(device, queue, library);
            if (!ok) return 1;
            puts("Qwen Metal fused split Q/gate RMSNorm fixture passed");
            return 0;
        }
        const bool ok =
            test_router_softmax_top8(device, queue, library) &&
            test_embedding_q5_k(device, queue, library) &&
            test_gated_delta_controls(device, queue, library) &&
            test_split_q_gate(device, queue, library) &&
            test_split_q_gate_rms_norm(device, queue, library) &&
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
