#define _GNU_SOURCE

#include <errno.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define DS4_CLI_MODEL_FREE_TEST 1
#include "../../ds4_cli.c"

enum { FAKE_MAX_BLOCKS = 16 };

struct ds4_engine {
    int unused;
};
struct ds4_session {
    ds4_generation_block script[FAKE_MAX_BLOCKS];
    int script_count;
    int script_index;
    ds4_generation_block current;
    ds4_generation_block_request current_request;
    bool active;
    bool pending;
    int pending_token;
    ds4_generation_rng pending_rng;
    bool invalidated;
    int materialized;
    int sync_materialized;
    int sync_count;
    int pos;
    int ctx;
    int evaluated[256];
    bool interrupt_on_begin;
    ds4_generation_block_request requests[FAKE_MAX_BLOCKS];
    int request_count;
    ds4_generation_block_commit commits[FAKE_MAX_BLOCKS];
    int commit_count;
};

static int fake_text_calls[128];
static ds4_session fake_session_pool[4];
static int fake_session_pool_used;
static int fake_session_free_count;
static int fake_power_set_count;
static const char *const *fake_repl_lines;
static int fake_repl_line_count;
static int fake_repl_line_index;

#define CHECK(condition) do {                                                \
    if (!(condition)) {                                                       \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        exit(1);                                                              \
    }                                                                         \
} while (0)

static uint64_t fake_draw(uint64_t state) {
    return state * UINT64_C(6364136223846793005) + UINT64_C(1442695040888963407);
}

static void fake_session_add_block(
        ds4_session *session, uint64_t cookie, const int *tokens, uint32_t count) {
    CHECK(session->script_count < FAKE_MAX_BLOCKS);
    ds4_generation_block *block = &session->script[session->script_count++];
    memset(block, 0, sizeof(*block));
    block->cookie = cookie;
    block->count = count;
    for (uint32_t i = 0; i < count; i++) block->tokens[i] = tokens[i];
}

int ds4_session_generation_block_begin(
        ds4_session *session,
        const ds4_generation_block_request *request,
        ds4_generation_block *block,
        char *err,
        size_t errlen) {
    CHECK(session && request && block);
    CHECK(!session->active);
    if (session->pending) {
        if (request->rng.state != session->pending_rng.state ||
            request->rng.position != session->pending_rng.position) {
            if (err && errlen) snprintf(err, errlen, "stale fake RNG boundary");
            return 1;
        }
        session->pending = false;
        session->materialized++;
    }
    CHECK(session->request_count < FAKE_MAX_BLOCKS);
    session->requests[session->request_count++] = *request;
    if (session->script_index >= session->script_count) {
        memset(block, 0, sizeof(*block));
        return 0;
    }
    *block = session->script[session->script_index++];
    session->current = *block;
    session->current_request = *request;
    session->active = block->cookie != 0;
    if (session->interrupt_on_begin) {
        session->interrupt_on_begin = false;
        cli_interrupted = 1;
    }
    return 0;
}

int ds4_session_generation_block_commit(
        ds4_session *session,
        const ds4_generation_block_commit *commit,
        ds4_generation_rng *rng,
        char *err,
        size_t errlen) {
    (void)err;
    (void)errlen;
    CHECK(session && commit && rng);
    CHECK(session->active);
    CHECK(commit->cookie == session->current.cookie);
    CHECK(commit->adopted_count <= commit->observed_count);
    CHECK(commit->observed_count <= session->current.count);
    CHECK(session->commit_count < FAKE_MAX_BLOCKS);
    session->commits[session->commit_count++] = *commit;

    uint64_t state = session->current_request.rng.state;
    if (session->current_request.temperature > 0.0f) {
        for (uint32_t i = 0; i < commit->observed_count; i++) {
            state = fake_draw(state);
        }
    }
    rng->state = state;
    rng->position = session->current_request.rng.position +
                    commit->observed_count;
    session->active = false;
    session->pending = false;
    if (commit->mode == DS4_GENERATION_COMMIT_INVALIDATE) {
        session->invalidated = true;
    } else if (commit->adopted_count > 0) {
        session->pending = true;
        session->pending_token =
            session->current.tokens[commit->adopted_count - 1];
        session->pending_rng = *rng;
    }
    return 0;
}

void ds4_session_invalidate(ds4_session *session) {
    CHECK(session);
    session->invalidated = true;
    session->active = false;
    session->pending = false;
}

bool ds4_token_is_stop_for_think_mode(
        ds4_engine *engine, int token, ds4_think_mode mode) {
    (void)engine;
    (void)mode;
    return token == 99;
}

char *ds4_token_text(ds4_engine *engine, int token, size_t *len) {
    (void)engine;
    if (token >= 0 && token < (int)(sizeof(fake_text_calls) / sizeof(fake_text_calls[0]))) {
        fake_text_calls[token]++;
    }
    const char *text = "?";
    switch (token) {
    case 1: text = "A"; break;
    case 2: text = "B"; break;
    case 3: text = "<thi"; break;
    case 4: text = "nk>X"; break;
    case 5: text = "</think>"; break;
    case 10: text = "ABCDE"; break;
    case 77:
        text = "I";
        cli_interrupted = 1;
        break;
    default: break;
    }
    *len = strlen(text);
    char *copy = malloc(*len + 1);
    CHECK(copy);
    memcpy(copy, text, *len + 1);
    return copy;
}

void ds4_tokens_push(ds4_tokens *tokens, int token) {
    if (tokens->len == tokens->cap) {
        const int cap = tokens->cap ? tokens->cap * 2 : 8;
        int *next = realloc(tokens->v, (size_t)cap * sizeof(*next));
        CHECK(next);
        tokens->v = next;
        tokens->cap = cap;
    }
    tokens->v[tokens->len++] = token;
}

void ds4_tokens_free(ds4_tokens *tokens) {
    if (!tokens) return;
    free(tokens->v);
    memset(tokens, 0, sizeof(*tokens));
}

bool ds4_engine_is_glm_dsa(ds4_engine *engine) {
    (void)engine;
    return false;
}

ds4_think_mode ds4_think_mode_for_context(
        ds4_think_mode mode, int ctx_size) {
    (void)ctx_size;
    return mode;
}

uint32_t ds4_think_max_min_context(void) {
    return 393216;
}

bool ds4_think_mode_enabled(ds4_think_mode mode) {
    return mode != DS4_THINK_NONE;
}

void ds4_chat_append_message(
        ds4_engine *engine,
        ds4_tokens *tokens,
        const char *role,
        const char *content) {
    (void)engine;
    (void)content;
    ds4_tokens_push(tokens, !strcmp(role, "system") ? 11 : 10);
}

void ds4_chat_append_max_effort_prefix(
        ds4_engine *engine, ds4_tokens *tokens) {
    (void)engine;
    ds4_tokens_push(tokens, 12);
}

bool ds4_chat_append_message_checked(
        ds4_engine *engine,
        ds4_tokens *tokens,
        const char *role,
        const char *content) {
    ds4_chat_append_message(engine, tokens, role, content);
    return true;
}

bool ds4_chat_append_assistant_prefix_checked(
        ds4_engine *engine,
        ds4_tokens *tokens,
        ds4_think_mode think_mode) {
    (void)engine;
    (void)think_mode;
    ds4_tokens_push(tokens, 20);
    return true;
}

int ds4_token_eos(ds4_engine *engine) {
    (void)engine;
    return 88;
}

int ds4_session_common_prefix(
        ds4_session *session, const ds4_tokens *prompt) {
    int common = 0;
    while (common < session->pos && common < prompt->len &&
           session->evaluated[common] == prompt->v[common]) {
        common++;
    }
    return common;
}

int ds4_session_sync(
        ds4_session *session,
        const ds4_tokens *prompt,
        char *err,
        size_t errlen) {
    CHECK(session && prompt);
    session->sync_count++;
    if (prompt->len > session->ctx || prompt->len > 256) {
        if (err && errlen) snprintf(err, errlen, "fake context full");
        return 1;
    }
    const int common = ds4_session_common_prefix(session, prompt);
    if (session->pending && common == session->pos &&
        prompt->len > session->pos &&
        prompt->v[session->pos] == session->pending_token) {
        session->evaluated[session->pos++] = session->pending_token;
        session->pending = false;
        session->materialized++;
        session->sync_materialized++;
    }
    memcpy(session->evaluated, prompt->v,
           (size_t)prompt->len * sizeof(prompt->v[0]));
    session->pos = prompt->len;
    session->pending = false;
    return 0;
}

int ds4_session_pos(ds4_session *session) {
    return session->pos;
}

int ds4_session_ctx(ds4_session *session) {
    return session->ctx;
}

void ds4_session_set_progress(
        ds4_session *session, ds4_session_progress_fn fn, void *ud) {
    (void)session;
    (void)fn;
    (void)ud;
}

void ds4_session_set_display_progress(
        ds4_session *session, ds4_session_progress_fn fn, void *ud) {
    (void)session;
    (void)fn;
    (void)ud;
}

bool ds4_log_is_tty(FILE *fp) {
    (void)fp;
    return false;
}

void ds4_log(FILE *fp, ds4_log_type type, const char *fmt, ...) {
    (void)fp;
    (void)type;
    (void)fmt;
}

void ds4_chat_begin(ds4_engine *engine, ds4_tokens *tokens) {
    (void)engine;
    (void)tokens;
}

int ds4_session_create(
        ds4_session **out, ds4_engine *engine, int ctx_size) {
    (void)engine;
    CHECK(out && fake_session_pool_used < 4);
    ds4_session *session = &fake_session_pool[fake_session_pool_used++];
    memset(session, 0, sizeof(*session));
    session->ctx = ctx_size;
    if (fake_session_pool_used == 1) {
        const int visible[] = {1};
        const int terminal[] = {99};
        fake_session_add_block(session, 80, visible, 1);
        fake_session_add_block(session, 81, terminal, 1);
    }
    *out = session;
    return 0;
}

void ds4_session_free(ds4_session *session) {
    if (!session) return;
    session->active = false;
    session->pending = false;
    fake_session_free_count++;
}

int ds4_session_power(ds4_session *session) {
    (void)session;
    return 100;
}

int ds4_session_set_power(ds4_session *session, int power_percent) {
    CHECK(session && power_percent >= 1 && power_percent <= 100);
    if (session->pending || session->active) return 1;
    fake_power_set_count++;
    return 0;
}

bool ds4_engine_context_memory_estimate_with_prefill(
        const ds4_engine *engine,
        int ctx_size,
        uint32_t prefill_chunk,
        ds4_context_memory *out) {
    (void)engine;
    (void)prefill_chunk;
    if (!out || ctx_size <= 0) return false;
    memset(out, 0, sizeof(*out));
    out->prefill_cap = (uint32_t)ctx_size;
    return true;
}

const char *ds4_backend_name(ds4_backend backend) {
    (void)backend;
    return "fake";
}

char *linenoise(const char *prompt) {
    (void)prompt;
    if (fake_repl_line_index >= fake_repl_line_count) return NULL;
    const char *line = fake_repl_lines[fake_repl_line_index++];
    const size_t len = strlen(line);
    char *copy = malloc(len + 1);
    CHECK(copy);
    memcpy(copy, line, len + 1);
    return copy;
}

void linenoiseFree(void *ptr) {
    free(ptr);
}

void linenoiseSetMultiLine(int enabled) {
    (void)enabled;
}

int linenoiseHistorySetMaxLen(int len) {
    (void)len;
    return 1;
}

int linenoiseHistoryLoad(const char *filename) {
    (void)filename;
    return 0;
}

int linenoiseHistoryAdd(const char *line) {
    (void)line;
    return 1;
}

int linenoiseHistorySave(const char *filename) {
    (void)filename;
    return 0;
}

static token_printer fake_printer(FILE *fp, bool thinking) {
    token_printer printer = {
        .engine = (ds4_engine *)(uintptr_t)1,
        .fp = fp,
        .format_thinking = thinking,
        .in_think = thinking,
        .last_output_newline = true,
    };
    return printer;
}

static cli_generation_options fake_options(float temperature, uint64_t seed) {
    cli_generation_options options = {
        .temperature = temperature,
        .top_p = 0.95f,
        .min_p = 0.05f,
        .seed = seed,
    };
    return options;
}

static void expect_file(FILE *fp, const char *expected) {
    CHECK(fflush(fp) == 0);
    CHECK(fseek(fp, 0, SEEK_SET) == 0);
    char actual[128] = {0};
    const size_t n = fread(actual, 1, sizeof(actual) - 1, fp);
    actual[n] = '\0';
    CHECK(strcmp(actual, expected) == 0);
}

static int run_fake_decode(
        ds4_session *session,
        const cli_generation_options *options,
        token_printer *printer,
        ds4_tokens *transcript,
        int max_tokens,
        transactional_decode_result *result) {
    char err[128] = {0};
    return run_transactional_decode(
        (ds4_engine *)(uintptr_t)1,
        session,
        options,
        DS4_THINK_NONE,
        printer,
        transcript,
        max_tokens,
        result,
        err,
        sizeof(err));
}

static void test_greedy_output_limit_and_pending(void) {
    ds4_session session = {0};
    const int a[] = {1};
    const int b[] = {2};
    fake_session_add_block(&session, 1, a, 1);
    fake_session_add_block(&session, 2, b, 1);
    FILE *fp = tmpfile();
    CHECK(fp);
    token_printer printer = fake_printer(fp, false);
    ds4_tokens transcript = {0};
    transactional_decode_result result = {0};
    const cli_generation_options options = fake_options(0.0f, 123);

    CHECK(run_fake_decode(
        &session, &options, &printer, &transcript, 2, &result) == 0);
    CHECK(result.reason == TRANSACTIONAL_FINISH_OUTPUT_LIMIT);
    CHECK(result.generated == 2);
    CHECK(session.commit_count == 2);
    CHECK(session.commits[0].adopted_count == 1);
    CHECK(session.commits[0].observed_count == 1);
    CHECK(session.commits[1].adopted_count == 1);
    CHECK(session.requests[0].rng.state == 123);
    CHECK(session.requests[1].rng.state == 123);
    CHECK(session.requests[1].rng.position == 1);
    CHECK(session.materialized == 1);
    CHECK(session.pending && session.pending_token == 2);
    CHECK(transcript.len == 2 && transcript.v[0] == 1 && transcript.v[1] == 2);
    expect_file(fp, "AB\n");
    fclose(fp);
    free(transcript.v);
}

static void test_sampled_rng_and_terminal(void) {
    ds4_session session = {0};
    const int a[] = {1};
    const int stop[] = {99};
    fake_session_add_block(&session, 10, a, 1);
    fake_session_add_block(&session, 11, stop, 1);
    FILE *fp = tmpfile();
    CHECK(fp);
    token_printer printer = fake_printer(fp, false);
    transactional_decode_result result = {0};
    const cli_generation_options options = fake_options(0.7f, 9);

    CHECK(run_fake_decode(&session, &options, &printer, NULL, 4, &result) == 0);
    CHECK(result.reason == TRANSACTIONAL_FINISH_TERMINAL);
    CHECK(result.generated == 1);
    CHECK(session.commit_count == 2);
    CHECK(session.commits[1].adopted_count == 0);
    CHECK(session.commits[1].observed_count == 1);
    CHECK(session.requests[1].rng.state == fake_draw(9));
    CHECK(session.requests[1].rng.position == 1);
    CHECK(!session.pending);
    expect_file(fp, "A\n");
    fclose(fp);
}

static void test_multitoken_terminal_does_not_inspect_suffix(void) {
    memset(fake_text_calls, 0, sizeof(fake_text_calls));
    ds4_session session = {0};
    const int block[] = {1, 99, 2};
    fake_session_add_block(&session, 12, block, 3);
    FILE *fp = tmpfile();
    CHECK(fp);
    token_printer printer = fake_printer(fp, false);
    transactional_decode_result result = {0};
    const cli_generation_options options = fake_options(0.0f, 10);

    CHECK(run_fake_decode(&session, &options, &printer, NULL, 3, &result) == 0);
    CHECK(result.reason == TRANSACTIONAL_FINISH_TERMINAL);
    CHECK(result.generated == 1);
    CHECK(session.commit_count == 1);
    CHECK(session.commits[0].adopted_count == 1);
    CHECK(session.commits[0].observed_count == 2);
    CHECK(session.commits[0].mode == DS4_GENERATION_COMMIT_RETAIN);
    CHECK(session.pending && session.pending_token == 1);
    CHECK(fake_text_calls[1] == 1);
    CHECK(fake_text_calls[2] == 0);
    expect_file(fp, "A\n");
    fclose(fp);
}

static void test_interrupt_before_and_after_observation(void) {
    const cli_generation_options options = fake_options(0.0f, 7);
    ds4_session before = {0};
    FILE *fp0 = tmpfile();
    CHECK(fp0);
    token_printer printer0 = fake_printer(fp0, false);
    transactional_decode_result result0 = {0};
    cli_interrupted = 1;
    CHECK(run_fake_decode(&before, &options, &printer0, NULL, 2, &result0) == 0);
    CHECK(result0.reason == TRANSACTIONAL_FINISH_INTERRUPTED);
    CHECK(result0.generated == 0 && before.request_count == 0);
    fclose(fp0);

    ds4_session after = {0};
    const int interrupting[] = {77};
    fake_session_add_block(&after, 20, interrupting, 1);
    FILE *fp1 = tmpfile();
    CHECK(fp1);
    token_printer printer1 = fake_printer(fp1, false);
    ds4_tokens transcript = {0};
    transactional_decode_result result1 = {0};
    cli_interrupted = 0;
    CHECK(run_fake_decode(
        &after, &options, &printer1, &transcript, 2, &result1) == 0);
    CHECK(result1.reason == TRANSACTIONAL_FINISH_INTERRUPTED);
    CHECK(result1.generated == 1);
    CHECK(after.commits[0].adopted_count == 1);
    CHECK(after.commits[0].observed_count == 1);
    CHECK(cli_transactional_finish_repl_decode(
        &after, &transcript, 0, true, 88, &result1));
    CHECK(transcript.len == 2 && transcript.v[0] == 77 && transcript.v[1] == 88);
    fclose(fp1);
    free(transcript.v);
    cli_interrupted = 0;
}

static void test_multitoken_interrupt_does_not_inspect_suffix(void) {
    memset(fake_text_calls, 0, sizeof(fake_text_calls));
    ds4_session session = {0};
    const int block[] = {77, 2};
    fake_session_add_block(&session, 21, block, 2);
    FILE *fp = tmpfile();
    CHECK(fp);
    token_printer printer = fake_printer(fp, false);
    transactional_decode_result result = {0};
    const cli_generation_options options = fake_options(0.0f, 8);
    cli_interrupted = 0;

    CHECK(run_fake_decode(&session, &options, &printer, NULL, 2, &result) == 0);
    CHECK(result.reason == TRANSACTIONAL_FINISH_INTERRUPTED);
    CHECK(result.generated == 1);
    CHECK(session.commits[0].adopted_count == 1);
    CHECK(session.commits[0].observed_count == 1);
    CHECK(fake_text_calls[77] == 1);
    CHECK(fake_text_calls[2] == 0);
    fclose(fp);
    cli_interrupted = 0;
}

static void test_interrupt_zero_rolls_back_repl(void) {
    ds4_session session = {0};
    ds4_tokens transcript = {0};
    ds4_tokens_push(&transcript, 41);
    ds4_tokens_push(&transcript, 42);
    transactional_decode_result result = {
        .generated = 0,
        .reason = TRANSACTIONAL_FINISH_INTERRUPTED,
    };
    CHECK(cli_transactional_finish_repl_decode(
        &session, &transcript, 1, true, 88, &result));
    CHECK(session.invalidated);
    CHECK(transcript.len == 1 && transcript.v[0] == 41);
    free(transcript.v);
}

static cli_config fake_cli_config(int ctx, int max_tokens) {
    cli_config cfg = {0};
    cfg.gen.ctx_size = ctx;
    cfg.gen.n_predict = max_tokens;
    cfg.gen.temperature = 0.0f;
    cfg.gen.top_p = 0.95f;
    cfg.gen.min_p = 0.05f;
    cfg.gen.seed = 55;
    cfg.gen.think_mode = DS4_THINK_NONE;
    return cfg;
}

static repl_chat fake_repl_chat(ds4_session *session, int ctx) {
    repl_chat chat = {
        .session = session,
        .ctx_size = ctx,
    };
    session->ctx = ctx;
    return chat;
}

static void test_real_chat_turn_two_turn_pending_sync(void) {
    ds4_session session = {0};
    repl_chat chat = fake_repl_chat(&session, 64);
    cli_config cfg = fake_cli_config(64, 1);
    const int a[] = {1};
    const int b[] = {2};
    fake_session_add_block(&session, 70, a, 1);
    fake_session_add_block(&session, 71, b, 1);

    CHECK(run_chat_turn(
        (ds4_engine *)(uintptr_t)1, &cfg, &chat, "first") ==
        CLI_CHAT_TURN_OK);
    CHECK(chat.transcript.len == 4);
    CHECK(chat.transcript.v[0] == 10 && chat.transcript.v[1] == 20);
    CHECK(chat.transcript.v[2] == 1 && chat.transcript.v[3] == 88);
    CHECK(session.pending && session.pending_token == 1);
    CHECK(session.pos == 2);

    CHECK(run_chat_turn(
        (ds4_engine *)(uintptr_t)1, &cfg, &chat, "second") ==
        CLI_CHAT_TURN_OK);
    CHECK(session.sync_count == 2);
    CHECK(session.sync_materialized == 1);
    CHECK(session.materialized == 1);
    CHECK(session.pos == 6);
    CHECK(!memcmp(session.evaluated, chat.transcript.v,
                  (size_t)session.pos * sizeof(chat.transcript.v[0])));
    CHECK(chat.transcript.len == 8);
    CHECK(chat.transcript.v[6] == 2 && chat.transcript.v[7] == 88);
    CHECK(session.pending && session.pending_token == 2);
    ds4_tokens_free(&chat.transcript);
}

static void test_real_chat_turn_frontend_eos(void) {
    ds4_session session = {0};
    repl_chat chat = fake_repl_chat(&session, 16);
    cli_config cfg = fake_cli_config(16, 2);
    const int stop[] = {99};
    fake_session_add_block(&session, 72, stop, 1);

    CHECK(run_chat_turn(
        (ds4_engine *)(uintptr_t)1, &cfg, &chat, "stop") ==
        CLI_CHAT_TURN_OK);
    CHECK(session.commit_count == 1);
    CHECK(session.commits[0].adopted_count == 0);
    CHECK(session.commits[0].observed_count == 1);
    CHECK(!session.pending);
    CHECK(chat.transcript.len == 3);
    CHECK(chat.transcript.v[2] == 88);
    ds4_tokens_free(&chat.transcript);
}

static void test_real_chat_turn_interrupt_after_begin(void) {
    ds4_session session = {0};
    repl_chat chat = fake_repl_chat(&session, 16);
    cli_config cfg = fake_cli_config(16, 2);
    const int block[] = {1, 2};
    fake_session_add_block(&session, 73, block, 2);
    ds4_tokens_push(&chat.transcript, 42);
    session.evaluated[0] = 42;
    session.pos = 1;
    session.interrupt_on_begin = true;
    cli_interrupted = 0;

    CHECK(run_chat_turn(
        (ds4_engine *)(uintptr_t)1, &cfg, &chat, "interrupt") ==
        CLI_CHAT_TURN_OK);
    CHECK(session.commit_count == 1);
    CHECK(session.commits[0].adopted_count == 0);
    CHECK(session.commits[0].observed_count == 0);
    CHECK(session.invalidated);
    CHECK(!session.active && !session.pending);
    CHECK(chat.transcript.len == 1 && chat.transcript.v[0] == 42);
    CHECK(cli_interrupted == 0);
    ds4_tokens_free(&chat.transcript);
}

static void test_real_chat_turn_partial_interrupt(void) {
    memset(fake_text_calls, 0, sizeof(fake_text_calls));
    ds4_session session = {0};
    repl_chat chat = fake_repl_chat(&session, 16);
    cli_config cfg = fake_cli_config(16, 2);
    const int block[] = {77, 2};
    fake_session_add_block(&session, 74, block, 2);
    cli_interrupted = 0;

    CHECK(run_chat_turn(
        (ds4_engine *)(uintptr_t)1, &cfg, &chat, "partial") ==
        CLI_CHAT_TURN_OK);
    CHECK(session.commit_count == 1);
    CHECK(session.commits[0].adopted_count == 1);
    CHECK(session.commits[0].observed_count == 1);
    CHECK(!session.invalidated);
    CHECK(session.pending && session.pending_token == 77);
    CHECK(chat.transcript.len == 4);
    CHECK(chat.transcript.v[2] == 77 && chat.transcript.v[3] == 88);
    CHECK(fake_text_calls[77] == 1 && fake_text_calls[2] == 0);
    CHECK(cli_interrupted == 0);
    ds4_tokens_free(&chat.transcript);
}

static void test_real_chat_turn_context_full(void) {
    ds4_session session = {0};
    repl_chat chat = fake_repl_chat(&session, 2);
    cli_config cfg = fake_cli_config(2, 4);

    CHECK(run_chat_turn(
        (ds4_engine *)(uintptr_t)1, &cfg, &chat, "full") ==
        CLI_CHAT_TURN_OK);
    CHECK(session.pos == 2);
    CHECK(session.request_count == 0 && session.commit_count == 0);
    CHECK(chat.transcript.len == 3 && chat.transcript.v[2] == 88);

    CHECK(run_chat_turn(
        (ds4_engine *)(uintptr_t)1, &cfg, &chat, "still-full") ==
        CLI_CHAT_TURN_RECOVERABLE);
    CHECK(chat.transcript.len == 3 && chat.transcript.v[2] == 88);
    CHECK(session.request_count == 0 && !session.active);
    ds4_tokens_free(&chat.transcript);
}

static void test_real_repl_two_turn_and_commands(void) {
    static const char *const lines[] = {
        "first",
        "second",
        "/power 55",
        "/think",
        "/ctx 32",
        "/quit",
    };
    memset(fake_session_pool, 0, sizeof(fake_session_pool));
    fake_session_pool_used = 0;
    fake_session_free_count = 0;
    fake_power_set_count = 0;
    fake_repl_lines = lines;
    fake_repl_line_count = (int)(sizeof(lines) / sizeof(lines[0]));
    fake_repl_line_index = 0;
    cli_interrupted = 0;
    cli_config cfg = fake_cli_config(64, 1);

    CHECK(run_repl((ds4_engine *)(uintptr_t)1, &cfg) == 0);
    CHECK(fake_repl_line_index == fake_repl_line_count);
    CHECK(fake_session_pool_used == 2);
    CHECK(fake_session_free_count == 2);
    CHECK(fake_session_pool[0].sync_count == 2);
    CHECK(fake_session_pool[0].sync_materialized == 1);
    CHECK(fake_session_pool[0].commit_count == 2);
    CHECK(fake_session_pool[0].commits[0].adopted_count == 1);
    CHECK(fake_session_pool[0].commits[1].adopted_count == 0);
    CHECK(fake_session_pool[0].commits[1].observed_count == 1);
    CHECK(fake_power_set_count == 1);
    CHECK(cfg.engine.power_percent == 55);
    CHECK(cfg.gen.think_mode == DS4_THINK_HIGH);
    CHECK(cfg.gen.ctx_size == 32);
    CHECK(fake_session_pool[1].ctx == 32);
}

static void test_split_thinking_delimiter(void) {
    FILE *fp = tmpfile();
    CHECK(fp);
    token_printer printer = fake_printer(fp, true);
    CHECK(token_printer_write_text(&printer, "<thi", 4));
    CHECK(printer.pending_len == 4);
    CHECK(token_printer_write_text(&printer, "nk>X", 4));
    CHECK(printer.pending_len == 0);
    CHECK(token_printer_write_text(&printer, "</think>", 8));
    generation_done(&printer);
    CHECK(!printer.failed);
    expect_file(fp, "X\n");
    fclose(fp);

    FILE *diagnostic_fp = tmpfile();
    CHECK(diagnostic_fp);
    token_printer diagnostic = fake_printer(diagnostic_fp, false);
    print_generated_token(&diagnostic, 1);
    generation_done(&diagnostic);
    expect_file(diagnostic_fp, "A\n");
    fclose(diagnostic_fp);
}

typedef struct {
    int calls;
    bool short_write;
    int successful_writes;
} failing_cookie;

#if defined(__APPLE__)
static int failing_write(void *ud, const char *data, int len) {
    (void)data;
    failing_cookie *cookie = ud;
    cookie->calls++;
    if (cookie->successful_writes > 0) {
        cookie->successful_writes--;
        return len;
    }
    if (cookie->short_write && len > 1) return len - 1;
    errno = EIO;
    return -1;
}

static FILE *failing_stream(failing_cookie *cookie) {
    return funopen(cookie, NULL, failing_write, NULL, NULL);
}
#else
static ssize_t failing_write(void *ud, const char *data, size_t len) {
    (void)data;
    failing_cookie *cookie = ud;
    cookie->calls++;
    if (cookie->successful_writes > 0) {
        cookie->successful_writes--;
        return (ssize_t)len;
    }
    if (cookie->short_write && len > 1) return (ssize_t)len - 1;
    errno = EIO;
    return -1;
}

static FILE *failing_stream(failing_cookie *cookie) {
    cookie_io_functions_t io = {
        .write = failing_write,
    };
    return fopencookie(cookie, "w", io);
}
#endif

static void expect_active_delivery_failure(bool short_write) {
    ds4_session session = {0};
    const int token[] = {10};
    fake_session_add_block(&session, 40, token, 1);
    failing_cookie cookie = {.short_write = short_write};
    FILE *fp = failing_stream(&cookie);
    CHECK(fp);
    char buffered_output[32];
    CHECK(setvbuf(
        fp,
        short_write ? NULL : buffered_output,
        short_write ? _IONBF : _IOFBF,
        short_write ? 0 : sizeof(buffered_output)) == 0);
    token_printer printer = fake_printer(fp, false);
    transactional_decode_result result = {0};
    const cli_generation_options options = fake_options(0.0f, 1);
    cli_interrupted = 0;
    CHECK(run_fake_decode(&session, &options, &printer, NULL, 1, &result) == 1);
    CHECK(session.commit_count == 1);
    CHECK(session.commits[0].adopted_count == 0);
    CHECK(session.commits[0].observed_count == 1);
    CHECK(session.commits[0].mode == DS4_GENERATION_COMMIT_INVALIDATE);
    CHECK(session.invalidated);
    CHECK(cookie.calls > 0);
    int repl_rc = CLI_CHAT_TURN_FATAL;
    CHECK(!cli_repl_continue_after_turn(CLI_CHAT_TURN_FATAL, &repl_rc));
    CHECK(repl_rc == 1);
    fclose(fp);
}

static void test_short_write_and_flush_failure(void) {
    expect_active_delivery_failure(true);
    expect_active_delivery_failure(false);
}

static void test_multitoken_delivery_failure_preserves_exact_prefix(void) {
    memset(fake_text_calls, 0, sizeof(fake_text_calls));
    ds4_session session = {0};
    const int block[] = {1, 10, 2};
    fake_session_add_block(&session, 41, block, 3);
    failing_cookie cookie = {
        .successful_writes = 1,
    };
    FILE *fp = failing_stream(&cookie);
    CHECK(fp);
    CHECK(setvbuf(fp, NULL, _IONBF, 0) == 0);
    token_printer printer = fake_printer(fp, false);
    transactional_decode_result result = {0};
    const cli_generation_options options = fake_options(0.0f, 4);

    CHECK(run_fake_decode(&session, &options, &printer, NULL, 3, &result) == 1);
    CHECK(session.commit_count == 1);
    CHECK(session.commits[0].adopted_count == 1);
    CHECK(session.commits[0].observed_count == 2);
    CHECK(session.commits[0].mode == DS4_GENERATION_COMMIT_INVALIDATE);
    CHECK(session.invalidated);
    CHECK(fake_text_calls[1] == 1);
    CHECK(fake_text_calls[10] == 1);
    CHECK(fake_text_calls[2] == 0);
    fclose(fp);
}

static void test_finalizer_failure_invalidates_after_retain(void) {
    ds4_session session = {0};
    const int partial[] = {3};
    fake_session_add_block(&session, 50, partial, 1);
    failing_cookie cookie = {0};
    FILE *fp = failing_stream(&cookie);
    CHECK(fp);
    CHECK(setvbuf(fp, NULL, _IONBF, 0) == 0);
    token_printer printer = fake_printer(fp, true);
    transactional_decode_result result = {0};
    const cli_generation_options options = fake_options(0.0f, 2);

    CHECK(run_fake_decode(&session, &options, &printer, NULL, 1, &result) == 1);
    CHECK(session.commit_count == 1);
    CHECK(session.commits[0].mode == DS4_GENERATION_COMMIT_RETAIN);
    CHECK(session.commits[0].adopted_count == 1);
    CHECK(session.commits[0].observed_count == 1);
    CHECK(session.invalidated);
    CHECK(cookie.calls > 0);
    fclose(fp);
}

static void test_sigpipe_is_reported_not_fatal(void) {
    int fds[2];
    CHECK(pipe(fds) == 0);
    CHECK(close(fds[0]) == 0);
    FILE *fp = fdopen(fds[1], "w");
    CHECK(fp);
    CHECK(setvbuf(fp, NULL, _IONBF, 0) == 0);
    void (*old_pipe)(int) = signal(SIGPIPE, SIG_IGN);
    CHECK(old_pipe != SIG_ERR);

    ds4_session session = {0};
    const int token[] = {1};
    fake_session_add_block(&session, 60, token, 1);
    token_printer printer = fake_printer(fp, false);
    transactional_decode_result result = {0};
    const cli_generation_options options = fake_options(0.0f, 3);
    CHECK(run_fake_decode(&session, &options, &printer, NULL, 1, &result) == 1);
    CHECK(session.commits[0].mode == DS4_GENERATION_COMMIT_INVALIDATE);
    CHECK(session.invalidated);
    fclose(fp);
    CHECK(signal(SIGPIPE, old_pipe) != SIG_ERR);
}

int main(void) {
    test_greedy_output_limit_and_pending();
    test_sampled_rng_and_terminal();
    test_multitoken_terminal_does_not_inspect_suffix();
    test_interrupt_before_and_after_observation();
    test_multitoken_interrupt_does_not_inspect_suffix();
    test_interrupt_zero_rolls_back_repl();
    test_real_chat_turn_two_turn_pending_sync();
    test_real_chat_turn_frontend_eos();
    test_real_chat_turn_interrupt_after_begin();
    test_real_chat_turn_partial_interrupt();
    test_real_chat_turn_context_full();
    test_real_repl_two_turn_and_commands();
    test_split_thinking_delimiter();
    test_short_write_and_flush_failure();
    test_multitoken_delivery_failure_preserves_exact_prefix();
    test_finalizer_failure_invalidates_after_retain();
    test_sigpipe_is_reported_not_fatal();
    fprintf(stderr, "cli transactional consumer: 17/17 PASS\n");
    return 0;
}
