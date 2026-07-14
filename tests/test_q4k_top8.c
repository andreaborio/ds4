/* Exercise the real static Q4_K expert kernels with all eight Qwen slots.
 * This translation unit intentionally includes ds4.c so the test cannot drift
 * into a duplicate implementation that misses host-array bound regressions. */

#include "../ds4.c"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

static int failures;

#define CHECK(condition) do {                                                \
    if (!(condition)) {                                                      \
        fprintf(stderr, "q4k top-8 check failed at %s:%d: %s\n",          \
                __FILE__, __LINE__, #condition);                             \
        failures++;                                                          \
    }                                                                        \
} while (0)

typedef struct {
    block_q4_K gate[QWEN35_N_EXPERT_USED];
    block_q4_K up[QWEN35_N_EXPERT_USED];
    block_q4_K down[QWEN35_N_EXPERT_USED];
} q4k_top8_fixture;

static void q4k_fill_constant(block_q4_K *block, uint8_t value) {
    memset(block, 0, sizeof(*block));
    block->d = UINT16_C(0x3c00); /* F16 1.0. */
    for (uint32_t group = 0; group < 4; group++) {
        block->scales[group] = 1;
    }
    for (uint32_t group = 4; group < 8; group++) {
        block->scales[group + 4] = 1;
    }
    memset(block->qs, (int)(value | (uint8_t)(value << 4)),
           sizeof(block->qs));
}

static void q8k_fill_constant(block_q8_K *block, float scale) {
    memset(block, 0, sizeof(*block));
    block->d = scale;
    memset(block->qs, 1, sizeof(block->qs));
    for (uint32_t group = 0; group < QK_K / 16; group++) {
        block->bsums[group] = 16;
    }
}

static ds4_tensor q4k_tensor(uint64_t offset) {
    ds4_tensor tensor;
    memset(&tensor, 0, sizeof(tensor));
    tensor.ndim = 3;
    tensor.dim[0] = QK_K;
    tensor.dim[1] = 1;
    tensor.dim[2] = QWEN35_N_EXPERT_USED;
    tensor.type = DS4_TENSOR_Q4_K;
    tensor.abs_offset = offset;
    tensor.elements =
        (uint64_t)QK_K * QWEN35_N_EXPERT_USED;
    tensor.bytes =
        sizeof(block_q4_K) * QWEN35_N_EXPERT_USED;
    return tensor;
}

static void test_real_q4k_top8_kernels(void) {
    q4k_top8_fixture fixture;
    for (uint32_t expert = 0; expert < QWEN35_N_EXPERT_USED; expert++) {
        q4k_fill_constant(&fixture.gate[expert], 1);
        q4k_fill_constant(&fixture.up[expert], (uint8_t)(expert + 1u));
        q4k_fill_constant(&fixture.down[expert], (uint8_t)(expert + 1u));
    }

    ds4_model model;
    memset(&model, 0, sizeof(model));
    model.map = (const uint8_t *)&fixture;
    model.size = sizeof(fixture);

    const ds4_tensor gate = q4k_tensor(offsetof(q4k_top8_fixture, gate));
    const ds4_tensor up = q4k_tensor(offsetof(q4k_top8_fixture, up));
    const ds4_tensor down = q4k_tensor(offsetof(q4k_top8_fixture, down));
    const int selected[QWEN35_N_EXPERT_USED] = {0, 1, 2, 3, 4, 5, 7, 6};
    const float weight[QWEN35_N_EXPERT_USED] =
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.5f, 0.25f};

    block_q8_K input;
    q8k_fill_constant(&input, 1.0f / (float)QK_K);
    float guarded_mid[QWEN35_N_EXPERT_USED + 2u];
    guarded_mid[0] = 1234.5f;
    guarded_mid[QWEN35_N_EXPERT_USED + 1u] = -987.25f;
    float *mid = guarded_mid + 1;
    memset(mid, 0, QWEN35_N_EXPERT_USED * sizeof(mid[0]));

    matvec_q4_k_experts_mid_prequant(
        mid, &model, &gate, &up, &input,
        selected, weight, QWEN35_N_EXPERT_USED, 0.0f);
    const float silu_one = 1.0f / (1.0f + expf(-1.0f));
    for (uint32_t slot = 0; slot < QWEN35_N_EXPERT_USED; slot++) {
        const float expected =
            silu_one * (float)(selected[slot] + 1) * weight[slot];
        CHECK(fabsf(mid[slot] - expected) < 1.0e-5f);
    }
    CHECK(fabsf(mid[6] - 4.0f * silu_one) < 1.0e-5f);
    CHECK(fabsf(mid[7] - 1.75f * silu_one) < 1.0e-5f);
    CHECK(guarded_mid[0] == 1234.5f);
    CHECK(guarded_mid[QWEN35_N_EXPERT_USED + 1u] == -987.25f);

    block_q8_K down_input[QWEN35_N_EXPERT_USED];
    for (uint32_t slot = 0; slot < QWEN35_N_EXPERT_USED; slot++) {
        q8k_fill_constant(&down_input[slot], 0.0f);
    }
    q8k_fill_constant(&down_input[6], 1.0f / (float)QK_K);
    q8k_fill_constant(&down_input[7], 2.0f / (float)QK_K);
    float guarded_out[3] = {321.0f, 0.0f, -654.0f};
    matvec_q4_k_experts_accum_prequant(
        guarded_out + 1, &model, &down, down_input,
        selected, QWEN35_N_EXPERT_USED);
    CHECK(fabsf(guarded_out[1] - 22.0f) < 1.0e-5f);
    CHECK(guarded_out[0] == 321.0f);
    CHECK(guarded_out[2] == -654.0f);

    const float down_weight[QWEN35_N_EXPERT_USED] =
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.25f};
    guarded_out[1] = 0.0f;
    matvec_q4_k_experts_accum_weighted_prequant(
        guarded_out + 1, &model, &down, down_input,
        selected, down_weight, QWEN35_N_EXPERT_USED);
    CHECK(fabsf(guarded_out[1] - 7.5f) < 1.0e-5f);
    CHECK(guarded_out[0] == 321.0f);
    CHECK(guarded_out[2] == -654.0f);
}

static ds4_tensor dense_tensor(uint32_t type, uint64_t in_dim,
                               uint64_t out_dim, uint64_t offset) {
    ds4_tensor tensor;
    memset(&tensor, 0, sizeof(tensor));
    tensor.ndim = 2;
    tensor.dim[0] = in_dim;
    tensor.dim[1] = out_dim;
    tensor.type = type;
    tensor.abs_offset = offset;
    tensor.elements = in_dim * out_dim;
    return tensor;
}

static void test_qwen_q8_embedding_and_dense_adapter(void) {
    uint8_t embedding[(QWEN35_N_EMBD / 32u) * 34u];
    for (uint32_t block = 0; block < QWEN35_N_EMBD / 32u; block++) {
        const uint16_t scale = f32_to_f16(0.5f);
        memcpy(embedding + (uint64_t)block * 34u, &scale, sizeof(scale));
        int8_t *quant = (int8_t *)(embedding + (uint64_t)block * 34u + 2u);
        for (uint32_t i = 0; i < 32u; i++) {
            quant[i] = (int8_t)((int)((block * 32u + i) % 31u) - 15);
        }
    }

    ds4_model model;
    memset(&model, 0, sizeof(model));
    model.map = embedding;
    model.size = sizeof(embedding);
    ds4_tensor tensor = dense_tensor(
        DS4_TENSOR_Q8_0, QWEN35_N_EMBD, 1u, 0u);
    tensor.bytes = sizeof(embedding);

    float guarded_embedding[QWEN35_N_EMBD + 2u];
    guarded_embedding[0] = 47.25f;
    guarded_embedding[QWEN35_N_EMBD + 1u] = -91.5f;
    CHECK(qwen35_cpu_embed_token(guarded_embedding + 1, &model,
                                 &tensor, 0));
    for (uint32_t i = 0; i < QWEN35_N_EMBD; i++) {
        const float expected = 0.5f * (float)((int)(i % 31u) - 15);
        CHECK(guarded_embedding[i + 1u] == expected);
    }
    CHECK(guarded_embedding[0] == 47.25f);
    CHECK(guarded_embedding[QWEN35_N_EMBD + 1u] == -91.5f);
    CHECK(!qwen35_cpu_embed_token(guarded_embedding + 1, &model,
                                  &tensor, 1));

    uint8_t weights[4u * 34u];
    for (uint32_t row = 0; row < 4u; row++) {
        const uint16_t scale = f32_to_f16(1.0f);
        memcpy(weights + (uint64_t)row * 34u, &scale, sizeof(scale));
        memset(weights + (uint64_t)row * 34u + 2u, (int)(row + 1u), 32u);
    }
    model.map = weights;
    model.size = sizeof(weights);
    ds4_tensor weight0 = dense_tensor(DS4_TENSOR_Q8_0, 32u, 2u, 0u);
    ds4_tensor weight1 = dense_tensor(DS4_TENSOR_Q8_0, 32u, 2u, 68u);
    weight0.bytes = 68u;
    weight1.bytes = 68u;
    float input[32];
    for (uint32_t i = 0; i < 32u; i++) input[i] = 1.0f;

    ds4_qwen35_cpu_scratch scratch = {0};
    CHECK(ds4_qwen35_cpu_scratch_init(&scratch, 1u));
    float guarded0[4] = {111.0f, 0.0f, 0.0f, 222.0f};
    float guarded1[4] = {-333.0f, 0.0f, 0.0f, -444.0f};
    qwen35_cpu_matvec_pair(guarded0 + 1, guarded1 + 1,
                           &model, &weight0, &weight1,
                           input, &scratch);
    CHECK(fabsf(guarded0[1] - 32.0f) < 1.0e-5f);
    CHECK(fabsf(guarded0[2] - 64.0f) < 1.0e-5f);
    CHECK(fabsf(guarded1[1] - 96.0f) < 1.0e-5f);
    CHECK(fabsf(guarded1[2] - 128.0f) < 1.0e-5f);
    CHECK(guarded0[0] == 111.0f && guarded0[3] == 222.0f);
    CHECK(guarded1[0] == -333.0f && guarded1[3] == -444.0f);
    ds4_qwen35_cpu_scratch_free(&scratch);
}

int main(void) {
    test_real_q4k_top8_kernels();
    test_qwen_q8_embedding_and_dense_adapter();
    ds4_threads_shutdown();
    if (failures) {
        fprintf(stderr, "Q4_K top-8 production tests: %d failure(s)\n",
                failures);
        return 1;
    }
    puts("Qwen CPU production kernel tests: OK");
    return 0;
}
