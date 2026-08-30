#ifndef DS4_QWEN_CPU_TEST_HOOKS_H
#define DS4_QWEN_CPU_TEST_HOOKS_H

#ifndef DS4_TEST_HOOKS
#error "ds4_qwen_cpu_test_hooks.h is private to DS4_TEST_HOOKS builds"
#endif

#include <stdbool.h>
#include <stddef.h>

#include "../../ds4.h"
#include "../../ds4_qwen.h"

typedef struct {
    const char *text;
    size_t len;
    int id;
} ds4_test_qwen4exp_tokenizer_token;

typedef struct {
    const char *text;
    size_t len;
    int rank;
} ds4_test_qwen4exp_tokenizer_merge;

typedef struct {
    size_t special_count;
    int special_ids[33];
    int bos_id;
    int eos_id;
    int pad_id;
    int im_start_id;
    int im_end_id;
    bool add_bos;
} ds4_test_qwen4exp_tokenizer_contract;

bool ds4_test_qwen4exp_tokenizer_engine_create(
        ds4_engine **engine_out,
        const ds4_test_qwen4exp_tokenizer_token *tokens,
        size_t token_count,
        const ds4_test_qwen4exp_tokenizer_merge *merges,
        size_t merge_count,
        int omitted_token_id);
void ds4_test_qwen4exp_tokenizer_engine_destroy(ds4_engine *engine);
bool ds4_test_qwen4exp_tokenizer_contract_get(
        const ds4_engine *engine,
        ds4_test_qwen4exp_tokenizer_contract *contract_out);
bool ds4_test_qwen4exp_tokenizer_set_family(
        ds4_engine *engine,
        int family);
bool ds4_test_qwen4exp_sampling_boundary(
        ds4_engine *engine,
        int *argmax_out,
        int *sample_out);

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
