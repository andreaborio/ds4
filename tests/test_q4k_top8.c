/* Exercise the real static Q4_K expert kernels with all eight Qwen slots. */

#include "internal/ds4_qwen_cpu_test_hooks.h"

#include <math.h>
#include <stdio.h>

static int failures;

#define CHECK(condition) do {                                                \
    if (!(condition)) {                                                      \
        fprintf(stderr, "q4k top-8 check failed at %s:%d: %s\n",          \
                __FILE__, __LINE__, #condition);                             \
        failures++;                                                          \
    }                                                                        \
} while (0)

static void test_real_q4k_top8_kernels(void) {
    const int selected[QWEN35_N_EXPERT_USED] = {0, 1, 2, 3, 4, 5, 7, 6};
    const float weight[QWEN35_N_EXPERT_USED] =
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.5f, 0.25f};

    float guarded_mid[QWEN35_N_EXPERT_USED + 2u];
    guarded_mid[0] = 1234.5f;
    guarded_mid[QWEN35_N_EXPERT_USED + 1u] = -987.25f;
    float *mid = guarded_mid + 1;

    CHECK(ds4_test_q4k_top8_mid(mid, selected, weight));
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

    float guarded_out[3] = {321.0f, 0.0f, -654.0f};
    CHECK(ds4_test_q4k_top8_accum(guarded_out + 1, selected, NULL));
    CHECK(fabsf(guarded_out[1] - 22.0f) < 1.0e-5f);
    CHECK(guarded_out[0] == 321.0f);
    CHECK(guarded_out[2] == -654.0f);

    const float down_weight[QWEN35_N_EXPERT_USED] =
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.5f, 0.25f};
    guarded_out[1] = 0.0f;
    CHECK(ds4_test_q4k_top8_accum(
        guarded_out + 1, selected, down_weight));
    CHECK(fabsf(guarded_out[1] - 7.5f) < 1.0e-5f);
    CHECK(guarded_out[0] == 321.0f);
    CHECK(guarded_out[2] == -654.0f);
}

static void test_qwen_q8_embedding_and_dense_adapter(void) {
    float guarded_embedding[QWEN35_N_EMBD + 2u];
    guarded_embedding[0] = 47.25f;
    guarded_embedding[QWEN35_N_EMBD + 1u] = -91.5f;
    CHECK(ds4_test_qwen_q8_embed_token(guarded_embedding + 1, 0));
    for (uint32_t i = 0; i < QWEN35_N_EMBD; i++) {
        const float expected = 0.5f * (float)((int)(i % 31u) - 15);
        CHECK(guarded_embedding[i + 1u] == expected);
    }
    CHECK(guarded_embedding[0] == 47.25f);
    CHECK(guarded_embedding[QWEN35_N_EMBD + 1u] == -91.5f);
    CHECK(!ds4_test_qwen_q8_embed_token(guarded_embedding + 1, 1));

    float guarded0[4] = {111.0f, 0.0f, 0.0f, 222.0f};
    float guarded1[4] = {-333.0f, 0.0f, 0.0f, -444.0f};
    CHECK(ds4_test_qwen_q8_dense_pair(guarded0 + 1, guarded1 + 1));
    CHECK(fabsf(guarded0[1] - 32.0f) < 1.0e-5f);
    CHECK(fabsf(guarded0[2] - 64.0f) < 1.0e-5f);
    CHECK(fabsf(guarded1[1] - 96.0f) < 1.0e-5f);
    CHECK(fabsf(guarded1[2] - 128.0f) < 1.0e-5f);
    CHECK(guarded0[0] == 111.0f && guarded0[3] == 222.0f);
    CHECK(guarded1[0] == -333.0f && guarded1[3] == -444.0f);
}

int main(void) {
    test_real_q4k_top8_kernels();
    test_qwen_q8_embedding_and_dense_adapter();
    ds4_test_threads_shutdown();
    if (failures) {
        fprintf(stderr, "Q4_K top-8 production tests: %d failure(s)\n",
                failures);
        return 1;
    }
    puts("Qwen CPU production kernel tests: OK");
    return 0;
}
