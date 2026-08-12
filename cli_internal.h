#ifndef HEBRUS_CLI_INTERNAL_H
#define HEBRUS_CLI_INTERNAL_H

#include "ds4.h"

#include <signal.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef struct {
    const char *prompt;
    const char *system;
    bool raw_prompt;
    int n_predict;
    int ctx_size;
    float temperature;
    float top_p;
    float min_p;
    bool temperature_set;
    bool top_p_set;
    bool min_p_set;
    uint64_t seed;
    bool dump_tokens;
    const char *dump_logits_path;
    const char *dump_generation_evidence_path;
    const char *dump_logprobs_path;
    int dump_logprobs_top_k;
    int decode_consistency_tokens;
    const char *perplexity_file_path;
    const char *imatrix_dataset_path;
    const char *imatrix_output_path;
    int imatrix_max_prompts;
    int imatrix_max_tokens;
    ds4_think_mode think_mode;
    bool head_test;
    bool first_token_test;
    bool metal_graph_test;
    bool metal_graph_full_test;
    bool metal_graph_prompt_test;
} cli_generation_options;

typedef struct {
    ds4_engine_options engine;
    cli_generation_options gen;
    char *prompt_owned;
    bool inspect;
} cli_config;

typedef struct {
    ds4_engine *engine;
    FILE *fp;
    bool format_thinking;
    bool in_think;
    bool color_open;
    bool use_color;
    bool last_output_newline;
    bool failed;
    char pending[16];
    size_t pending_len;
} token_printer;

typedef enum {
    TRANSACTIONAL_FINISH_OUTPUT_LIMIT = 0,
    TRANSACTIONAL_FINISH_TERMINAL,
    TRANSACTIONAL_FINISH_CONTEXT_LIMIT,
    TRANSACTIONAL_FINISH_INTERRUPTED,
    TRANSACTIONAL_FINISH_ERROR,
} transactional_finish_reason;

typedef struct {
    int generated;
    transactional_finish_reason reason;
} transactional_decode_result;

enum {
    CLI_CHAT_TURN_OK = 0,
    CLI_CHAT_TURN_RECOVERABLE = 1,
    CLI_CHAT_TURN_FATAL = 2,
};

typedef struct {
    ds4_session *session;
    ds4_tokens transcript;
    int ctx_size;
    int think_prefix_pos;
    int think_prefix_tokens;
} repl_chat;

#ifdef DS4_CLI_MODEL_FREE_TEST
extern volatile sig_atomic_t cli_interrupted;

bool token_printer_write_text(token_printer *p, const char *text, size_t len);
void generation_done(void *ud);
void print_generated_token(void *ud, int token);
int run_transactional_decode(
        ds4_engine *engine,
        ds4_session *session,
        const cli_generation_options *gen,
        ds4_think_mode think_mode,
        token_printer *printer,
        ds4_tokens *transcript,
        int max_tokens,
        transactional_decode_result *result,
        char *err,
        size_t errlen);
bool cli_transactional_finish_repl_decode(
        ds4_session *session,
        ds4_tokens *transcript,
        int rollback_len,
        bool append_synthetic_eos,
        int eos_token,
        const transactional_decode_result *decode);
bool cli_repl_continue_after_turn(int turn_rc, int *repl_rc);
int run_chat_turn(
        ds4_engine *engine,
        cli_config *cfg,
        repl_chat *chat,
        const char *user_text);
int run_repl(ds4_engine *engine, cli_config *cfg);
#endif

#endif
