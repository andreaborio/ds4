/* White-box Qwen session tests.  The fake forward keeps this gate model-free
 * while exercising the production cache/checkpoint transaction and public
 * logits APIs at the real 248320-token vocabulary size. */

#include "../ds4.c"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

#define CHECK(condition) do {                                                \
    if (!(condition)) {                                                      \
        fprintf(stderr, "qwen session check failed at %s:%d: %s\n",       \
                __FILE__, __LINE__, #condition);                             \
        failures++;                                                          \
    }                                                                        \
} while (0)

static int stub_calls;
static int stub_logits_calls;
static int stub_fail_on_call;
static uint32_t stub_position[16];
static int stub_token[16];
static uint32_t stub_logits_position[16];
static int stub_logits_token[16];

static void stub_reset(void) {
    stub_calls = 0;
    stub_logits_calls = 0;
    stub_fail_on_call = 0;
    memset(stub_position, 0, sizeof(stub_position));
    memset(stub_token, 0, sizeof(stub_token));
    memset(stub_logits_position, 0, sizeof(stub_logits_position));
    memset(stub_logits_token, 0, sizeof(stub_logits_token));
}

static bool stub_forward(
        float                         *logits,
        const ds4_model              *model,
        const ds4_qwen35_weights     *weights,
        ds4_qwen35_cpu_cache         *cache,
        int                           token,
        uint32_t                      position,
        ds4_qwen35_cpu_scratch       *scratch) {
    (void)model;
    (void)weights;
    CHECK(cache != NULL);
    CHECK(scratch != NULL && scratch->arena != NULL);
    CHECK(cache->n_tokens == position);
    CHECK(cache->kv_capacity > position);
    if (stub_calls < (int)(sizeof(stub_position) / sizeof(stub_position[0]))) {
        stub_position[stub_calls] = position;
        stub_token[stub_calls] = token;
    }
    stub_calls++;
    if (logits) {
        if (stub_logits_calls <
            (int)(sizeof(stub_logits_position) /
                  sizeof(stub_logits_position[0]))) {
            stub_logits_position[stub_logits_calls] = position;
            stub_logits_token[stub_logits_calls] = token;
        }
        stub_logits_calls++;
        logits[QWEN35_N_VOCAB - 1u] = (float)token;
    }

    if (stub_fail_on_call != 0 && stub_calls == stub_fail_on_call) {
        cache->layer[0].conv[0] = 17.0f;
        cache->layer[0].recurrent[0] = -19.0f;
        const uint64_t row =
            (uint64_t)position * QWEN35_N_HEAD_KV * QWEN35_N_HEAD_DIM;
        cache->layer[3].key[row] = 23.0f;
        cache->layer[3].value[row] = -29.0f;
        CHECK(ds4_qwen35_cpu_cache_advance(cache, 1u));
        return false;
    }
    return ds4_qwen35_cpu_cache_advance(cache, 1u);
}

static bool stub_true_without_advance(
        float                         *logits,
        const ds4_model              *model,
        const ds4_qwen35_weights     *weights,
        ds4_qwen35_cpu_cache         *cache,
        int                           token,
        uint32_t                      position,
        ds4_qwen35_cpu_scratch       *scratch) {
    (void)logits;
    (void)model;
    (void)weights;
    (void)cache;
    (void)token;
    (void)position;
    (void)scratch;
    stub_calls++;
    return true;
}

static bool cancel_after_first_forward(void *ud) {
    (void)ud;
    return stub_calls >= 1;
}

static void fake_qwen_engine(ds4_engine *engine, bool raw_runtime) {
    static ds4_tensor routed_gate = {.type = DS4_TENSOR_Q4_K};
    static ds4_tensor output = {.type = DS4_TENSOR_Q8_0};
    memset(engine, 0, sizeof(*engine));
    engine->model.fd = -1;
    engine->mtp_model.fd = -1;
    engine->model.family = DS4_MODEL_FAMILY_QWEN35_MOE;
    engine->backend = DS4_BACKEND_CPU;
    engine->qwen_raw_runtime = raw_runtime;
    engine->qwen35_weights.layer[0].ffn_gate_exps = &routed_gate;
    engine->qwen35_weights.output = &output;
}

static void test_session_creation_boundary(void) {
    ds4_engine engine;
    ds4_session *session = NULL;
    fake_qwen_engine(&engine, false);
    CHECK(ds4_session_create(&session, &engine, 5) != 0);
    CHECK(session == NULL);

    fake_qwen_engine(&engine, true);
    CHECK(ds4_session_create(&session, &engine,
                             (int)QWEN35_CONTEXT_LENGTH + 1) != 0);
    CHECK(session == NULL);
}

static void test_dynamic_logits(ds4_session *session) {
    const uint32_t n_vocab = QWEN35_N_VOCAB;
    CHECK(ds4_engine_vocab_size(session->engine) == QWEN35_N_VOCAB);
    CHECK(ds4_engine_effective_vocab_size(session->engine) ==
          QWEN35_N_VALID_TOKEN);
    CHECK(ds4_engine_routed_quant_bits(session->engine) == 4);
    CHECK(ds4_engine_has_output_head(session->engine));
    float *input = malloc((size_t)n_vocab * sizeof(input[0]));
    float *copy = malloc(((size_t)n_vocab + 1u) * sizeof(copy[0]));
    CHECK(input != NULL && copy != NULL);
    if (!input || !copy) {
        free(input);
        free(copy);
        return;
    }

    for (uint32_t i = 0; i < n_vocab; i++) input[i] = -INFINITY;
    input[200000] = 3.0f;
    input[QWEN35_N_VALID_TOKEN - 1u] = 4.0f;
    input[n_vocab - 1u] = 5.0f;
    CHECK(ds4_session_set_logits(session, input, (int)n_vocab) == 0);
    CHECK(ds4_session_set_logits(session, input, (int)DS4_N_VOCAB) != 0);
    CHECK(ds4_session_set_logits(session, input, (int)n_vocab - 1) != 0);
    CHECK(ds4_session_argmax(session) ==
          (int)QWEN35_N_VALID_TOKEN - 1);
    CHECK(ds4_session_argmax_excluding(
              session, (int)QWEN35_N_VALID_TOKEN - 1) == 200000);

    uint64_t rng = UINT64_C(0x123456789abcdef0);
    CHECK(ds4_session_sample(session, 0.0f, 0, 1.0f, 0.0f, &rng) ==
          (int)QWEN35_N_VALID_TOKEN - 1);

    ds4_token_score top[2];
    CHECK(ds4_session_top_logprobs(session, top, 2) == 2);
    CHECK(top[0].id == (int)QWEN35_N_VALID_TOKEN - 1);
    CHECK(top[1].id == 200000);
    CHECK(top[0].logprob > top[1].logprob);
    CHECK(top[0].logprob > -1.0f); /* UNUSED logit is not normalized. */
    ds4_token_score one;
    CHECK(ds4_session_token_logprob(
              session, (int)QWEN35_N_VALID_TOKEN - 1, &one) == 1);
    CHECK(one.id == (int)QWEN35_N_VALID_TOKEN - 1 && isfinite(one.logprob));
    CHECK(ds4_session_token_logprob(
              session, (int)QWEN35_N_VALID_TOKEN, &one) == 0);
    CHECK(ds4_session_token_logprob(session, (int)n_vocab, &one) == 0);

    copy[0] = 91.0f;
    CHECK(ds4_session_copy_logits(session, copy, (int)n_vocab - 1) == 0);
    CHECK(copy[0] == 91.0f);
    copy[n_vocab] = -1234.5f;
    CHECK(ds4_session_copy_logits(session, copy, (int)n_vocab) ==
          (int)n_vocab);
    CHECK(copy[200000] == 3.0f);
    CHECK(copy[n_vocab - 1u] == 5.0f);
    CHECK(copy[n_vocab] == -1234.5f);

    free(copy);
    free(input);
}

static void test_eval_transaction(ds4_session *session) {
    char err[160] = {0};
    ds4_session_invalidate(session);
    stub_reset();

    CHECK(ds4_session_eval_qwen35_with_forward(
              session, 7, err, sizeof(err), stub_forward) == 0);
    CHECK(stub_calls == 1 && stub_position[0] == 0 && stub_token[0] == 7);
    CHECK(session->checkpoint_valid && session->checkpoint.len == 1);
    CHECK(session->checkpoint.v[0] == 7);
    CHECK(session->qwen35_cpu_cache.n_tokens == 1);

    CHECK(ds4_session_eval_qwen35_with_forward(
              session, -1, err, sizeof(err), stub_forward) != 0);
    CHECK(ds4_session_eval_qwen35_with_forward(
              session, QWEN35_N_VALID_TOKEN,
              err, sizeof(err), stub_forward) != 0);
    memset(err, 0, sizeof(err));
    session->distributed = (ds4_dist_session *)(uintptr_t)1;
    CHECK(ds4_session_eval(session, QWEN35_N_VALID_TOKEN,
                           err, sizeof(err)) != 0);
    session->distributed = NULL;
    CHECK(strstr(err, "Qwen token id") != NULL);
    CHECK(stub_calls == 1);
    CHECK(session->checkpoint.len == 1 &&
          session->qwen35_cpu_cache.n_tokens == 1);

    CHECK(ds4_session_eval_qwen35_with_forward(
              session, 8, err, sizeof(err),
              stub_true_without_advance) != 0);
    CHECK(session->checkpoint.len == 0 && !session->checkpoint_valid);
    CHECK(session->qwen35_cpu_cache.n_tokens == 0);

    stub_reset();
    stub_fail_on_call = 1;
    CHECK(ds4_session_eval_qwen35_with_forward(
              session, 9, err, sizeof(err), stub_forward) != 0);
    CHECK(session->checkpoint.len == 0 && !session->checkpoint_valid);
    CHECK(session->qwen35_cpu_cache.n_tokens == 0);
    CHECK(session->qwen35_cpu_cache.layer[0].conv[0] == 0.0f);
    CHECK(session->qwen35_cpu_cache.layer[0].recurrent[0] == 0.0f);
    /* Reset is O(fixed GDN state); stale K/V is deliberately invisible. */
    CHECK(session->qwen35_cpu_cache.layer[3].key[0] == 23.0f);

    stub_reset();
    CHECK(ds4_session_eval_qwen35_with_forward(
              session, 10, err, sizeof(err), stub_forward) == 0);
    CHECK(stub_position[0] == 0);
    session->qwen35_cpu_cache.layer[0].conv[0] = 31.0f;
    session->qwen35_cpu_cache.layer[0].recurrent[0] = -37.0f;
    ds4_session_invalidate(session);
    CHECK(session->checkpoint.len == 0 &&
          session->qwen35_cpu_cache.n_tokens == 0);
    CHECK(session->qwen35_cpu_cache.layer[0].conv[0] == 0.0f);
    CHECK(session->qwen35_cpu_cache.layer[0].recurrent[0] == 0.0f);

    stub_reset();
    CHECK(ds4_session_eval_qwen35_with_forward(
              session, 1, err, sizeof(err), stub_forward) == 0);
    CHECK(ds4_session_eval_qwen35_with_forward(
              session, 2, err, sizeof(err), stub_forward) == 0);
    const int calls_before_mismatch = stub_calls;
    session->qwen35_cpu_cache.n_tokens = 1;
    CHECK(ds4_session_eval_qwen35_with_forward(
              session, 3, err, sizeof(err), stub_forward) != 0);
    CHECK(stub_calls == calls_before_mismatch);
    CHECK(session->checkpoint.len == 0 &&
          session->qwen35_cpu_cache.n_tokens == 0);

    stub_reset();
    for (int token = 0; token < session->ctx_size; token++) {
        CHECK(ds4_session_eval_qwen35_with_forward(
                  session, token, err, sizeof(err), stub_forward) == 0);
    }
    CHECK(session->checkpoint.len == session->ctx_size);
    CHECK(session->qwen35_cpu_cache.n_tokens == (uint32_t)session->ctx_size);
    const int calls_at_full = stub_calls;
    CHECK(ds4_session_eval_qwen35_with_forward(
              session, 99, err, sizeof(err), stub_forward) != 0);
    CHECK(stub_calls == calls_at_full);
    CHECK(session->checkpoint.len == session->ctx_size);
    ds4_session_rewind(session, session->checkpoint.len);
    CHECK(session->checkpoint.len == session->ctx_size);
    ds4_session_rewind(session, 2);
    CHECK(session->checkpoint.len == 0 && !session->checkpoint_valid);
    CHECK(session->qwen35_cpu_cache.n_tokens == 0);
}

static void test_sync_transaction(ds4_session *session) {
    char err[160] = {0};
    int first_values[] = {11, 12};
    ds4_tokens first = {.v = first_values, .len = 2, .cap = 2};
    int extended_values[] = {11, 12, 13};
    ds4_tokens extended = {.v = extended_values, .len = 3, .cap = 3};

    ds4_session_invalidate(session);
    stub_reset();
    CHECK(ds4_session_sync_qwen35_with_forward(
              session, &first, err, sizeof(err), stub_forward) == 0);
    CHECK(stub_calls == 2 && stub_logits_calls == 1);
    CHECK(stub_logits_position[0] == 1 && stub_logits_token[0] == 12);
    CHECK(session->logits[QWEN35_N_VOCAB - 1u] == 12.0f);
    CHECK(stub_position[0] == 0 && stub_position[1] == 1);
    CHECK(session->checkpoint.len == 2 &&
          session->qwen35_cpu_cache.n_tokens == 2);

    CHECK(ds4_session_sync_qwen35_with_forward(
              session, &first, err, sizeof(err), stub_forward) == 0);
    CHECK(stub_calls == 2);
    CHECK(ds4_session_sync_qwen35_with_forward(
              session, &extended, err, sizeof(err), stub_forward) == 0);
    CHECK(stub_calls == 3 && stub_logits_calls == 2);
    CHECK(stub_logits_position[1] == 2 && stub_logits_token[1] == 13);
    CHECK(session->logits[QWEN35_N_VOCAB - 1u] == 13.0f);
    CHECK(stub_position[2] == 2 && stub_token[2] == 13);

    int invalid_values[] = {11, QWEN35_N_VALID_TOKEN};
    ds4_tokens invalid = {.v = invalid_values, .len = 2, .cap = 2};
    CHECK(ds4_session_sync_qwen35_with_forward(
              session, &invalid, err, sizeof(err), stub_forward) != 0);
    memset(err, 0, sizeof(err));
    session->distributed = (ds4_dist_session *)(uintptr_t)1;
    CHECK(ds4_session_sync(session, &invalid, err, sizeof(err)) != 0);
    session->distributed = NULL;
    CHECK(strstr(err, "Qwen token id") != NULL);
    CHECK(stub_calls == 3);
    CHECK(session->checkpoint.len == 3 &&
          session->qwen35_cpu_cache.n_tokens == 3);

    int divergent_values[] = {21, 22};
    ds4_tokens divergent = {.v = divergent_values, .len = 2, .cap = 2};
    CHECK(ds4_session_sync_qwen35_with_forward(
              session, &divergent, err, sizeof(err), stub_forward) == 0);
    CHECK(stub_calls == 5);
    CHECK(stub_position[3] == 0 && stub_position[4] == 1);
    CHECK(session->checkpoint.v[0] == 21 && session->checkpoint.v[1] == 22);

    ds4_session_invalidate(session);
    stub_reset();
    session->cancel = cancel_after_first_forward;
    CHECK(ds4_session_sync_qwen35_with_forward(
              session, &extended, err, sizeof(err), stub_forward) ==
          DS4_SESSION_SYNC_INTERRUPTED);
    session->cancel = NULL;
    CHECK(!session->checkpoint_valid && session->checkpoint.len == 0);
    CHECK(session->qwen35_cpu_cache.n_tokens == 0);

    ds4_session_invalidate(session);
    stub_reset();
    stub_fail_on_call = 2;
    CHECK(ds4_session_sync_qwen35_with_forward(
              session, &extended, err, sizeof(err), stub_forward) != 0);
    CHECK(stub_calls == 2);
    CHECK(session->checkpoint.len == 0 && !session->checkpoint_valid);
    CHECK(session->qwen35_cpu_cache.n_tokens == 0);
    CHECK(session->qwen35_cpu_cache.layer[0].conv[0] == 0.0f);
    CHECK(session->qwen35_cpu_cache.layer[0].recurrent[0] == 0.0f);
}

static void test_fail_closed_surfaces(ds4_session *session) {
    char err[160] = {0};
    stub_reset();
    ds4_session_invalidate(session);
    CHECK(ds4_session_eval_qwen35_with_forward(
              session, 42, err, sizeof(err), stub_forward) == 0);
    CHECK(ds4_session_payload_bytes(session) == 0);

    FILE *fp = tmpfile();
    CHECK(fp != NULL);
    if (fp) {
        CHECK(ds4_session_save_payload(session, fp, err, sizeof(err)) != 0);
        rewind(fp);
        CHECK(ds4_session_load_payload(session, fp, 0, err, sizeof(err)) != 0);
        fclose(fp);
    }
    ds4_session_payload_file staged = {0};
    memset(err, 0, sizeof(err));
    CHECK(ds4_session_stage_payload(session, &staged, err, sizeof(err)) != 0);
    CHECK(strcmp(err, "Qwen session payloads are not supported yet") == 0);
    CHECK(staged.path == NULL && staged.bytes == 0);

    ds4_session_snapshot snapshot = {0};
    CHECK(ds4_session_save_snapshot(session, &snapshot, err, sizeof(err)) != 0);
    ds4_session_snapshot_free(&snapshot);
    CHECK(ds4_session_layer_slice_reset(session, err, sizeof(err)) != 0);

    float hidden[QWEN35_N_EMBD] = {0};
    CHECK(ds4_session_eval_output_head_from_hc(
              session, hidden, 1u, session->logits, err, sizeof(err)) != 0);
    const int token = 42;
    CHECK(ds4_session_eval_layer_slice(
              session, &token, 1u, 0u, 0u, 0u, NULL, NULL,
              false, NULL, err, sizeof(err)) != 0);

    ds4_tokens text_tokens = {0};
    ds4_tokenize_text(session->engine, "unsafe fallback", &text_tokens);
    ds4_tokenize_rendered_chat(session->engine, "unsafe fallback", &text_tokens);
    ds4_chat_begin(session->engine, &text_tokens);
    ds4_chat_append_message(session->engine, &text_tokens,
                            "user", "unsafe fallback");
    ds4_chat_append_assistant_prefix(session->engine, &text_tokens,
                                     DS4_THINK_NONE);
    CHECK(text_tokens.len == 0);
    CHECK(ds4_token_eos(session->engine) == -1);
    CHECK(ds4_token_user(session->engine) == -1);
    CHECK(ds4_token_assistant(session->engine) == -1);
    size_t text_len = 99;
    char *text = ds4_token_text(session->engine, 0, &text_len);
    CHECK(text != NULL && text_len == 0 && text[0] == '\0');
    free(text);
    CHECK(ds4_engine_generate_argmax(
              session->engine, &text_tokens, 1, session->ctx_size,
              NULL, NULL, NULL, NULL, NULL) != 0);
    ds4_tokens_free(&text_tokens);
}

int main(void) {
    test_session_creation_boundary();

    ds4_engine engine;
    ds4_session *session = NULL;
    fake_qwen_engine(&engine, true);
    CHECK(ds4_session_create(&session, &engine, 5) == 0);
    if (session) {
        CHECK(ds4_session_ctx(session) == 5);
        CHECK(ds4_session_pos(session) == 0);
        CHECK(ds4_session_prefill_cap(session) == 1);
        CHECK(session->qwen35_cpu_cache.ctx_capacity == 5);
        CHECK(session->qwen35_cpu_scratch.ctx_capacity == 5);
        CHECK(session->cpu_cache.layer[0].raw_kv == NULL);
        test_dynamic_logits(session);
        test_eval_transaction(session);
        test_sync_transaction(session);
        test_fail_closed_surfaces(session);
        ds4_session_free(session);
    }

    ds4_threads_shutdown();
    if (failures) {
        fprintf(stderr, "Qwen session tests: %d failure(s)\n", failures);
        return 1;
    }
    puts("Qwen session tests: OK");
    return 0;
}
