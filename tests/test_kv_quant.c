#include "../ds4_kv_quant.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;

#define CHECK(condition) do {                                                \
    if (!(condition)) {                                                      \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        failures++;                                                          \
    }                                                                        \
} while (0)

static void test_surface_planner(void) {
    ds4_kv_surface_plan plan = {0};
    const ds4_kv_surface qwen_key_f32 = {
        .family = DS4_KV_FAMILY_QWEN35,
        .kind = DS4_KV_SURFACE_FULL_KEY,
        .storage = DS4_KV_STORAGE_F32,
        .layer_count = 10,
        .capacity_rows = 32768,
        .vectors_per_row = 2,
        .vector_dim = 256,
    };
    CHECK(ds4_kv_surface_plan_checked(&qwen_key_f32, &plan));
    CHECK(plan.vector_stride_bytes == 1024);
    CHECK(plan.row_stride_bytes == 2048);
    CHECK(plan.total_bytes == UINT64_C(671088640));
    CHECK(plan.metadata_bytes == 0);

    ds4_kv_surface qwen_key_tq4 = qwen_key_f32;
    qwen_key_tq4.storage = DS4_KV_STORAGE_TQ4_KEY;
    CHECK(ds4_kv_surface_plan_checked(&qwen_key_tq4, &plan));
    CHECK(plan.vector_stride_bytes == 130);
    CHECK(plan.row_stride_bytes == 260);
    CHECK(plan.packed_data_bytes == UINT64_C(83886080));
    CHECK(plan.metadata_bytes == UINT64_C(1310720));
    CHECK(plan.total_bytes == UINT64_C(85196800));

    ds4_kv_surface qwen_value_tq4 = qwen_key_f32;
    qwen_value_tq4.kind = DS4_KV_SURFACE_FULL_VALUE;
    qwen_value_tq4.storage = DS4_KV_STORAGE_TQ4_VALUE;
    CHECK(ds4_kv_surface_plan_checked(&qwen_value_tq4, &plan));
    CHECK(plan.vector_stride_bytes == 132);
    CHECK(plan.row_stride_bytes == 264);
    CHECK(plan.total_bytes == UINT64_C(86507520));

    /*
     * These surfaces intentionally use the same checked planner while keeping
     * their model semantics distinct.
     */
    const ds4_kv_surface deepseek_raw = {
        .family = DS4_KV_FAMILY_DEEPSEEK4,
        .kind = DS4_KV_SURFACE_RAW_MLA,
        .storage = DS4_KV_STORAGE_F32,
        .layer_count = 43,
        .capacity_rows = 4224,
        .vectors_per_row = 1,
        .vector_dim = 512,
    };
    CHECK(ds4_kv_surface_plan_checked(&deepseek_raw, &plan));
    CHECK(plan.total_bytes == UINT64_C(371982336));

    const ds4_kv_surface glm_kv_lora = {
        .family = DS4_KV_FAMILY_GLM52,
        .kind = DS4_KV_SURFACE_COMPACT_KV_LORA,
        .storage = DS4_KV_STORAGE_F16,
        .layer_count = 78,
        .capacity_rows = 32768,
        .vectors_per_row = 1,
        .vector_dim = 512,
    };
    CHECK(ds4_kv_surface_plan_checked(&glm_kv_lora, &plan));
    CHECK(plan.total_bytes == UINT64_C(2617245696));

    ds4_kv_surface invalid = qwen_key_f32;
    invalid.vector_dim = 0;
    CHECK(!ds4_kv_surface_plan_checked(&invalid, &plan));

    invalid = qwen_key_f32;
    invalid.family = DS4_KV_FAMILY_GLM52;
    CHECK(!ds4_kv_surface_plan_checked(&invalid, &plan));

    invalid = glm_kv_lora;
    invalid.storage = DS4_KV_STORAGE_TQ4_VALUE;
    CHECK(!ds4_kv_surface_plan_checked(&invalid, &plan));

    invalid = qwen_key_tq4;
    invalid.vector_dim = 192;
    CHECK(!ds4_kv_surface_plan_checked(&invalid, &plan));
}

static void test_plan_overflow(void) {
    ds4_kv_surface_plan plan = {0};
    const ds4_kv_surface enormous = {
        .family = DS4_KV_FAMILY_QWEN35,
        .kind = DS4_KV_SURFACE_FULL_KEY,
        .storage = DS4_KV_STORAGE_F32,
        .layer_count = UINT32_MAX,
        .capacity_rows = UINT32_MAX,
        .vectors_per_row = UINT32_MAX,
        .vector_dim = UINT32_MAX,
    };
    CHECK(!ds4_kv_surface_plan_checked(&enormous, &plan));

    ds4_kv_plan_total total = {
        .packed_data_bytes = UINT64_MAX,
        .metadata_bytes = 0,
        .total_bytes = UINT64_MAX,
    };
    const ds4_kv_surface_plan one = {
        .packed_data_bytes = 1,
        .metadata_bytes = 0,
        .total_bytes = 1,
    };
    CHECK(!ds4_kv_plan_add_checked(&total, &one));
    CHECK(total.packed_data_bytes == UINT64_MAX);
    CHECK(total.total_bytes == UINT64_MAX);
}

static double cosine(const float *a, const float *b, uint32_t n) {
    double dot = 0.0;
    double aa = 0.0;
    double bb = 0.0;
    for (uint32_t i = 0; i < n; i++) {
        dot += (double)a[i] * b[i];
        aa += (double)a[i] * a[i];
        bb += (double)b[i] * b[i];
    }
    return dot / sqrt(aa * bb);
}

static void test_key_reference(void) {
    enum { DIM = 256, PACKED = 130 };
    float input[DIM];
    float output[DIM];
    float scratch[DIM];
    uint8_t packed[PACKED];
    uint8_t packed_again[PACKED];
    for (uint32_t i = 0; i < DIM; i++) {
        input[i] =
            0.7f * sinf((float)(i + 1u) * 0.071f) +
            0.2f * cosf((float)(i + 3u) * 0.193f) +
            0.01f * (float)((int)(i % 11u) - 5);
    }

    CHECK(ds4_kv_tq4_key_bytes(DIM) == PACKED);
    CHECK(ds4_kv_tq4_key_encode_reference(
        packed, sizeof(packed), input, DIM, 0x12345678u,
        scratch, DIM));
    CHECK(ds4_kv_tq4_key_encode_reference(
        packed_again, sizeof(packed_again), input, DIM, 0x12345678u,
        scratch, DIM));
    CHECK(memcmp(packed, packed_again, sizeof(packed)) == 0);
    CHECK(ds4_kv_tq4_key_decode_reference(
        output, DIM, packed, sizeof(packed), 0x12345678u,
        scratch, DIM));
    CHECK(cosine(input, output, DIM) > 0.985);

    memset(input, 0, sizeof(input));
    CHECK(ds4_kv_tq4_key_encode_reference(
        packed, sizeof(packed), input, DIM, 7u, scratch, DIM));
    CHECK(ds4_kv_tq4_key_decode_reference(
        output, DIM, packed, sizeof(packed), 7u, scratch, DIM));
    for (uint32_t i = 0; i < DIM; i++) CHECK(output[i] == 0.0f);

    CHECK(!ds4_kv_tq4_key_encode_reference(
        packed, sizeof(packed), input, 192, 7u, scratch, DIM));
    CHECK(!ds4_kv_tq4_key_encode_reference(
        packed, sizeof(packed) - 1u, input, DIM, 7u, scratch, DIM));
}

static void test_value_reference(void) {
    enum { DIM = 256, PACKED = 132 };
    float input[DIM];
    float output[DIM];
    uint8_t packed[PACKED];
    double squared_error = 0.0;
    double span = 0.0;
    for (uint32_t i = 0; i < DIM; i++) {
        input[i] =
            -1.25f + 2.5f * (float)i / (float)(DIM - 1u) +
            0.03f * sinf((float)i * 0.31f);
    }

    CHECK(ds4_kv_tq4_value_bytes(DIM) == PACKED);
    CHECK(ds4_kv_tq4_value_encode_reference(
        packed, sizeof(packed), input, DIM));
    CHECK(ds4_kv_tq4_value_decode_reference(
        output, DIM, packed, sizeof(packed)));
    for (uint32_t i = 0; i < DIM; i++) {
        const double error = (double)input[i] - output[i];
        squared_error += error * error;
        span = fmax(span, fabs((double)input[i]));
    }
    const double normalized_rmse =
        sqrt(squared_error / DIM) / (span > 0.0 ? span : 1.0);
    CHECK(normalized_rmse < 0.04);
}

int main(void) {
    test_surface_planner();
    test_plan_overflow();
    test_key_reference();
    test_value_reference();
    if (failures != 0) {
        fprintf(stderr, "%d KV quantization test(s) failed\n", failures);
        return 1;
    }
    puts("KV quantization reference tests passed");
    return 0;
}
