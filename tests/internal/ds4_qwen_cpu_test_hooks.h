#ifndef DS4_QWEN_CPU_TEST_HOOKS_H
#define DS4_QWEN_CPU_TEST_HOOKS_H

#ifndef DS4_TEST_HOOKS
#error "ds4_qwen_cpu_test_hooks.h is private to DS4_TEST_HOOKS builds"
#endif

#include <stdbool.h>

#include "../../ds4_qwen.h"

bool ds4_test_q4k_top8_mid(
        float       mid[QWEN35_N_EXPERT_USED],
        const int   selected[QWEN35_N_EXPERT_USED],
        const float weight[QWEN35_N_EXPERT_USED]);

bool ds4_test_q4k_top8_accum(
        float       out[1],
        const int   selected[QWEN35_N_EXPERT_USED],
        const float weight[QWEN35_N_EXPERT_USED]);

bool ds4_test_qwen_q8_embed_token(
        float out[QWEN35_N_EMBD],
        int   token);

bool ds4_test_qwen_q8_dense_pair(
        float out0[2],
        float out1[2]);

void ds4_test_threads_shutdown(void);

#endif
