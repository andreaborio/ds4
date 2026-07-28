#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GGML_TABLE_BEGIN(type, name, size) static const type name[size] = {
#define GGML_TABLE_END() };
#include "../../gguf-tools/vendor/qwen35-iq-ggml-common.h"
#undef GGML_TABLE_BEGIN
#undef GGML_TABLE_END

/*
 * Model-free CPU-oracle coverage for the Qwen3.6 Q2_K_XL routed formats.
 * The source assembly mirrors ds4_gpu_full_source(): dense.metal contributes
 * the shared GGML matvec ABI, then the generated IQ tables and moe.metal
 * contribute the format-specific kernels.
 */

enum {
    QK_K = 256,
    N_EXPERT = 3,
    N_SELECTED = 2,
    N_ROW = 7,
};

typedef struct {
    uint16_t d;
    uint16_t qs[QK_K / 8];
    uint8_t scales[QK_K / 32];
} block_iq2_xs;

typedef struct {
    uint16_t d;
    uint8_t qs[3 * QK_K / 8];
} block_iq3_xxs;

typedef struct {
    uint16_t d;
    uint16_t scales_h;
    uint8_t scales_l[QK_K / 64];
    uint8_t qs[QK_K / 2];
} block_iq4_xs;

typedef struct {
    int32_t nei0;
    int32_t nei1;
    uint64_t nbi1;
    int32_t ne00;
    int32_t ne01;
    int32_t ne02;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    int32_t ne10;
    int32_t ne11;
    int32_t ne12;
    int32_t ne13;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    int32_t ne0;
    int32_t ne1;
    uint64_t nb1;
    int32_t nr0;
} mul_mv_id_args;

_Static_assert(sizeof(block_iq2_xs) == 74, "IQ2_XS block ABI drift");
_Static_assert(sizeof(block_iq3_xxs) == 98, "IQ3_XXS block ABI drift");
_Static_assert(sizeof(block_iq4_xs) == 136, "IQ4_XS block ABI drift");
_Static_assert(sizeof(mul_mv_id_args) == 120, "routed matvec ABI drift");

typedef float (*dequant_value_fn)(const void *block, uint32_t index);

typedef struct {
    const char *label;
    const char *kernel;
    const char *slots_kernel;
    size_t block_bytes;
    uint32_t nr0;
    NSUInteger threadgroup_bytes;
    dequant_value_fn value;
} iq_case;

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

static uint8_t iq_sign_mask(uint32_t index) {
    return (uint8_t)(index |
        (((uint32_t)__builtin_popcount(index) & 1u) << 7u));
}

static float iq2_xs_value(const void *opaque, uint32_t index) {
    const block_iq2_xs *block = opaque;
    const uint32_t group = index / 32u;
    const uint32_t within_group = index - group * 32u;
    const uint32_t subgroup = within_group / 8u;
    const uint32_t lane = within_group - subgroup * 8u;
    const uint16_t code = block->qs[group * 4u + subgroup];
    const uint64_t packed = iq2xs_grid[code & 511u];
    const uint8_t *grid = (const uint8_t *)&packed;
    const uint8_t signs = iq_sign_mask(code >> 9u);
    const uint32_t scale_nibble = subgroup < 2u
        ? (block->scales[group] & 0x0fu)
        : (block->scales[group] >> 4u);
    const float scale =
        f16_bits_to_f32(block->d) * (0.5f + (float)scale_nibble) * 0.25f;
    return scale * (float)grid[lane] *
        ((signs & (uint8_t)(1u << lane)) != 0u ? -1.0f : 1.0f);
}

static float iq3_xxs_value(const void *opaque, uint32_t index) {
    const block_iq3_xxs *block = opaque;
    const uint32_t group = index / 32u;
    const uint32_t within_group = index - group * 32u;
    const uint32_t subgroup = within_group / 8u;
    const uint32_t lane = within_group - subgroup * 8u;
    const uint8_t *q3 = block->qs + group * 8u;
    uint16_t gas[2] = {0, 0};
    memcpy(gas, block->qs + QK_K / 4u + group * 4u, sizeof(gas));
    const uint32_t aux = (uint32_t)gas[0] | ((uint32_t)gas[1] << 16u);
    const uint32_t half = lane / 4u;
    const uint32_t within_grid = lane - half * 4u;
    const uint32_t grid_index = q3[subgroup * 2u + half];
    const uint32_t packed = iq3xxs_grid[grid_index];
    const uint8_t *grid = (const uint8_t *)&packed;
    const uint8_t signs = iq_sign_mask(
        (aux >> (subgroup * 7u)) & 127u);
    const float scale =
        f16_bits_to_f32(block->d) *
        (0.5f + (float)(aux >> 28u)) * 0.5f;
    return scale * (float)grid[within_grid] *
        ((signs & (uint8_t)(1u << lane)) != 0u ? -1.0f : 1.0f);
}

static float iq4_xs_value(const void *opaque, uint32_t index) {
    const block_iq4_xs *block = opaque;
    const uint32_t group = index / 32u;
    const uint32_t lane = index - group * 32u;
    const uint32_t low_scale =
        (block->scales_l[group / 2u] >> (4u * (group % 2u))) & 0x0fu;
    const uint32_t high_scale = (block->scales_h >> (2u * group)) & 3u;
    const int32_t scale = (int32_t)(low_scale | (high_scale << 4u)) - 32;
    const uint32_t q_index =
        group * 16u + (lane % 16u);
    const uint32_t shift = lane < 16u ? 0u : 4u;
    const uint32_t quant = (block->qs[q_index] >> shift) & 0x0fu;
    return f16_bits_to_f32(block->d) * (float)scale *
           (float)kvalues_iq4nl[quant];
}

static uint32_t fixture_rng = 0x7a31c295u;

static uint32_t fixture_random(void) {
    uint32_t value = fixture_rng;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    fixture_rng = value;
    return value;
}

static void fill_fixture(
        uint8_t *weights,
        const iq_case *test_case) {
    const size_t expert_bytes = N_ROW * test_case->block_bytes;
    for (uint32_t expert = 0; expert < N_EXPERT; expert++) {
        for (uint32_t row = 0; row < N_ROW; row++) {
            uint8_t *block = weights +
                (size_t)expert * expert_bytes +
                (size_t)row * test_case->block_bytes;
            for (size_t byte = 0; byte < test_case->block_bytes; byte++) {
                block[byte] = (uint8_t)fixture_random();
            }
            const float scale =
                0.001953125f * (float)(1u + expert + 2u * row);
            const uint16_t half = f32_to_f16_bits(scale);
            memcpy(block, &half, sizeof(half));
        }
    }
}

static NSString *read_source(NSString *path) {
    NSError *error = nil;
    NSString *source = [NSString stringWithContentsOfFile:path
                                                  encoding:NSUTF8StringEncoding
                                                     error:&error];
    if (!source) {
        fprintf(stderr, "read %s: %s\n", path.UTF8String,
                error.localizedDescription.UTF8String);
    }
    return source;
}

static id<MTLLibrary> build_library(id<MTLDevice> device) {
    NSString *dense = read_source(@"metal/dense.metal");
    NSString *tables = read_source(@"metal/qwen35_iq_tables.metal.inc");
    NSString *moe = read_source(@"metal/moe.metal");
    if (!dense || !tables || !moe) return nil;
    NSString *prefix =
        @"#include <metal_stdlib>\n"
         "#ifdef DS4_METAL_HAS_TENSOR\n"
         "#include <metal_tensor>\n"
         "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
         "#endif\n"
         "using namespace metal;\n"
         "#ifdef DS4_METAL_HAS_TENSOR\n"
         "using namespace mpp::tensor_ops;\n"
         "#endif\n"
         "#define MAX(x, y) ((x) > (y) ? (x) : (y))\n"
         "#define MIN(x, y) ((x) < (y) ? (x) : (y))\n"
         "#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }\n"
         "#define QK8_0 32\n"
         "#define N_SIMDWIDTH 32\n"
         "#define N_R0_Q8_0 2\n"
         "#define N_SG_Q8_0 4\n"
         "#define FC_MUL_MV 600\n"
         "#define FC_MUL_MM 700\n"
         "#define FC_BIN 1300\n"
         "#define FOR_UNROLL(x) _Pragma(\"clang loop unroll(full)\") for (x)\n"
         "struct block_q8_0 { half d; int8_t qs[QK8_0]; };\n"
         "void dequantize_f16_t4(device const half4 *src, short il, "
         "thread float4 &reg) { (void)il; reg = float4(*src); }\n";
    NSString *source = [NSString stringWithFormat:
        @"%@\n%@\n%@\n%@\n", prefix, dense, tables, moe];
    MTLCompileOptions *options = [MTLCompileOptions new];
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    options.fastMathEnabled = NO;
#pragma clang diagnostic pop
    NSError *error = nil;
    id<MTLLibrary> library =
        [device newLibraryWithSource:source options:options error:&error];
    if (!library) {
        fprintf(stderr, "compile Qwen IQ Metal source: %s\n",
                error.localizedDescription.UTF8String);
    }
    return library;
}

static id<MTLComputePipelineState> build_pipeline(
        id<MTLDevice> device,
        id<MTLLibrary> library,
        const char *name) {
    MTLFunctionConstantValues *constants = [MTLFunctionConstantValues new];
    int16_t n_simdgroup = 2;
    [constants setConstantValue:&n_simdgroup
                           type:MTLDataTypeShort
                        atIndex:600];
    NSError *error = nil;
    id<MTLFunction> function = [library
        newFunctionWithName:[NSString stringWithUTF8String:name]
             constantValues:constants
                      error:&error];
    if (!function) {
        fprintf(stderr, "function %s: %s\n", name,
                error.localizedDescription.UTF8String);
        return nil;
    }
    id<MTLComputePipelineState> pipeline =
        [device newComputePipelineStateWithFunction:function error:&error];
    if (!pipeline) {
        fprintf(stderr, "pipeline %s: %s\n", name,
                error.localizedDescription.UTF8String);
    }
    return pipeline;
}

static bool finish_command(id<MTLCommandBuffer> command, const char *label) {
    [command commit];
    [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusError) {
        fprintf(stderr, "%s: %s\n", label,
                command.error.localizedDescription.UTF8String);
        return false;
    }
    return true;
}

static bool check_output(
        const iq_case *test_case,
        const uint8_t *weights,
        const float input[QK_K],
        const int32_t selected[N_SELECTED],
        const float *actual,
        const char *path) {
    const size_t expert_bytes = N_ROW * test_case->block_bytes;
    float maximum_error = 0.0f;
    for (uint32_t slot = 0; slot < N_SELECTED; slot++) {
        for (uint32_t row = 0; row < N_ROW; row++) {
            const uint8_t *block = weights +
                (size_t)selected[slot] * expert_bytes +
                (size_t)row * test_case->block_bytes;
            float expected = 0.0f;
            for (uint32_t index = 0; index < QK_K; index++) {
                expected += test_case->value(block, index) * input[index];
            }
            const float observed = actual[(size_t)slot * N_ROW + row];
            const float error = fabsf(observed - expected);
            const float limit = 2.0e-4f + 3.0e-4f * fabsf(expected);
            if (error > maximum_error) maximum_error = error;
            if (!isfinite(observed) || error > limit) {
                fprintf(stderr,
                        "%s %s slot=%u row=%u actual=%.9g expected=%.9g "
                        "error=%.9g limit=%.9g\n",
                        test_case->label, path, slot, row,
                        observed, expected, error, limit);
                return false;
            }
        }
    }
    printf("ok %-8s %-7s max_abs_error=%.3g\n",
           test_case->label, path, maximum_error);
    return true;
}

static bool run_case(
        id<MTLDevice> device,
        id<MTLCommandQueue> queue,
        id<MTLLibrary> library,
        const iq_case *test_case) {
    const size_t row_bytes = test_case->block_bytes;
    const size_t expert_bytes = N_ROW * row_bytes;
    const size_t weight_bytes = N_EXPERT * expert_bytes;
    uint8_t *weights = malloc(weight_bytes);
    if (!weights) return false;
    fill_fixture(weights, test_case);

    float input[QK_K];
    for (uint32_t index = 0; index < QK_K; index++) {
        input[index] =
            ((int32_t)((index * 29u + 7u) % 67u) - 33) * 0.00075f;
    }
    const int32_t selected[N_SELECTED] = {2, 0};
    const mul_mv_id_args args = {
        .nei0 = N_SELECTED,
        .nei1 = 1,
        .nbi1 = N_SELECTED * sizeof(int32_t),
        .ne00 = QK_K,
        .ne01 = N_ROW,
        .ne02 = N_EXPERT,
        .nb00 = row_bytes,
        .nb01 = row_bytes,
        .nb02 = expert_bytes,
        .ne10 = QK_K,
        .ne11 = 1,
        .ne12 = 1,
        .ne13 = 1,
        .nb10 = sizeof(float),
        .nb11 = sizeof(input),
        .nb12 = sizeof(input),
        .ne0 = N_ROW,
        .ne1 = N_SELECTED,
        .nb1 = N_ROW * sizeof(float),
        .nr0 = test_case->nr0,
    };
    id<MTLBuffer> weight_buffer =
        [device newBufferWithBytes:weights
                            length:weight_bytes
                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> input_buffer =
        [device newBufferWithBytes:input
                            length:sizeof(input)
                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> selected_buffer =
        [device newBufferWithBytes:selected
                            length:sizeof(selected)
                           options:MTLResourceStorageModeShared];
    id<MTLBuffer> output_buffer =
        [device newBufferWithLength:N_SELECTED * N_ROW * sizeof(float)
                            options:MTLResourceStorageModeShared];
    id<MTLComputePipelineState> pipeline =
        build_pipeline(device, library, test_case->kernel);
    if (!weight_buffer || !input_buffer || !selected_buffer ||
        !output_buffer || !pipeline) {
        free(weights);
        return false;
    }

    memset(output_buffer.contents, 0, output_buffer.length);
    id<MTLCommandBuffer> command = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder =
        [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBytes:&args length:sizeof(args) atIndex:0];
    [encoder setBuffer:weight_buffer offset:0 atIndex:1];
    [encoder setBuffer:input_buffer offset:0 atIndex:2];
    [encoder setBuffer:output_buffer offset:0 atIndex:3];
    [encoder setBuffer:selected_buffer offset:0 atIndex:4];
    [encoder setThreadgroupMemoryLength:test_case->threadgroup_bytes atIndex:0];
    const NSUInteger rows_per_group = test_case->nr0 * 2u;
    const NSUInteger row_groups =
        (N_ROW + rows_per_group - 1u) / rows_per_group;
    [encoder dispatchThreadgroups:MTLSizeMake(row_groups, 1, N_SELECTED)
             threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
    [encoder endEncoding];
    bool ok = finish_command(command, test_case->kernel);
    if (ok) {
        ok = check_output(
            test_case, weights, input, selected,
            output_buffer.contents, "resident");
    }

    id<MTLComputePipelineState> slots_pipeline =
        build_pipeline(device, library, test_case->slots_kernel);
    NSMutableArray<id<MTLBuffer>> *slot_buffers =
        [NSMutableArray arrayWithCapacity:6];
    for (uint32_t slot = 0; slot < 6u; slot++) {
        const uint32_t expert = slot < N_SELECTED
            ? (uint32_t)selected[slot]
            : 0u;
        id<MTLBuffer> buffer = [device
            newBufferWithBytes:weights + (size_t)expert * expert_bytes
                        length:expert_bytes
                       options:MTLResourceStorageModeShared];
        if (!buffer) ok = false;
        [slot_buffers addObject:buffer];
    }
    if (ok && slots_pipeline) {
        memset(output_buffer.contents, 0, output_buffer.length);
        command = [queue commandBuffer];
        encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:slots_pipeline];
        [encoder setBytes:&args length:sizeof(args) atIndex:0];
        for (NSUInteger slot = 0; slot < slot_buffers.count; slot++) {
            [encoder setBuffer:slot_buffers[slot] offset:0 atIndex:slot + 1u];
        }
        [encoder setBuffer:input_buffer offset:0 atIndex:7];
        [encoder setBuffer:output_buffer offset:0 atIndex:8];
        [encoder setThreadgroupMemoryLength:test_case->threadgroup_bytes atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(row_groups, 1, N_SELECTED)
                 threadsPerThreadgroup:MTLSizeMake(32, 2, 1)];
        [encoder endEncoding];
        ok = finish_command(command, test_case->slots_kernel);
        if (ok) {
            ok = check_output(
                test_case, weights, input, selected,
                output_buffer.contents, "slots6");
        }
    } else {
        ok = false;
    }
    free(weights);
    return ok;
}

int main(void) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "no Metal device\n");
            return 2;
        }
        id<MTLCommandQueue> queue = [device newCommandQueue];
        id<MTLLibrary> library = build_library(device);
        if (!queue || !library) return 1;
        printf("Qwen IQ Metal fixture on %s\n", device.name.UTF8String);
        const iq_case cases[] = {
            {
                .label = "IQ2_XS",
                .kernel = "kernel_mul_mv_id_iq2_xs_f32",
                .slots_kernel = "kernel_mul_mv_slots6_iq2_xs_f32",
                .block_bytes = sizeof(block_iq2_xs),
                .nr0 = 4,
                .threadgroup_bytes =
                    512u * sizeof(uint64_t) + 128u * sizeof(uint8_t),
                .value = iq2_xs_value,
            },
            {
                .label = "IQ3_XXS",
                .kernel = "kernel_mul_mv_id_iq3_xxs_f32",
                .slots_kernel = "kernel_mul_mv_slots6_iq3_xxs_f32",
                .block_bytes = sizeof(block_iq3_xxs),
                .nr0 = 4,
                .threadgroup_bytes =
                    256u * sizeof(uint32_t) + 128u * sizeof(uint8_t),
                .value = iq3_xxs_value,
            },
            {
                .label = "IQ4_XS",
                .kernel = "kernel_mul_mv_id_iq4_xs_f32",
                .slots_kernel = "kernel_mul_mv_slots6_iq4_xs_f32",
                .block_bytes = sizeof(block_iq4_xs),
                .nr0 = 2,
                .threadgroup_bytes = 32u * sizeof(float),
                .value = iq4_xs_value,
            },
        };
        for (size_t index = 0;
             index < sizeof(cases) / sizeof(cases[0]);
             index++) {
            if (!run_case(device, queue, library, &cases[index])) return 1;
        }
        puts("all Qwen Q2_K_XL routed IQ Metal fixtures passed");
        return 0;
    }
}
