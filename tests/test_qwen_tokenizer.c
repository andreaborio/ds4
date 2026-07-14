/* White-box Qwen3.6 tokenizer and core chat-renderer tests.
 *
 * The fixture is a compact closure of the official tokenizer: it contains
 * only the tokens and merge decisions reached by the pinned golden cases.
 * Keeping the production tokenizer in this translation unit lets this test
 * exercise the exact dispatch, rollback, special-token, and public API paths
 * without downloading model weights or a multi-hundred-megabyte tokenizer.
 */

#define DS4_BPE_TEST_HOOKS 1
#include "../ds4.c"
#include "qwen/qwen36_tokenizer_fixture.inc"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))
#define TEST_SENTINEL 123456789

static int failures;

#define CHECK(condition) do {                                                \
    if (!(condition)) {                                                      \
        fprintf(stderr, "qwen tokenizer check failed at %s:%d: %s\n",     \
                __FILE__, __LINE__, #condition);                             \
        failures++;                                                          \
    }                                                                        \
} while (0)

static void fixture_engine_init(ds4_engine *engine, int omitted_token_id) {
    memset(engine, 0, sizeof(*engine));
    engine->model.fd = -1;
    engine->mtp_model.fd = -1;
    engine->model.family = DS4_MODEL_FAMILY_QWEN35_MOE;

    ds4_vocab *vocab = &engine->vocab;
    vocab->family = DS4_MODEL_FAMILY_QWEN35_MOE;
    vocab->n_vocab = QWEN35_N_VOCAB;
    vocab->bos_id = -1;
    vocab->eos_id = -1;
    vocab->pad_id = -1;
    vocab->user_id = -1;
    vocab->assistant_id = -1;
    vocab->im_start_id = -1;
    vocab->im_end_id = -1;
    vocab->think_start_id = -1;
    vocab->think_end_id = -1;
    vocab->dsml_id = -1;
    vocab->token = xcalloc(
        (size_t)vocab->n_vocab, sizeof(vocab->token[0]));

    table_init(&vocab->token_to_id, QWEN36_TOKENIZER_FIXTURE_TOKEN_COUNT);
    for (size_t i = 0; i < TEST_ARRAY_LEN(qwen36_fixture_tokens); i++) {
        const qwen36_fixture_token *token = &qwen36_fixture_tokens[i];
        CHECK(token->id >= 0 && token->id < vocab->n_vocab);
        CHECK(token->len == strlen(token->text));
        if (token->id < 0 || token->id >= vocab->n_vocab) continue;
        vocab->token[token->id] = (ds4_str){token->text, token->len};
        if (token->id != omitted_token_id) {
            table_put(&vocab->token_to_id, vocab->token[token->id], token->id);
        }
    }

    table_init(&vocab->merge_rank, QWEN36_TOKENIZER_FIXTURE_MERGE_COUNT);
    for (size_t i = 0; i < TEST_ARRAY_LEN(qwen36_fixture_merges); i++) {
        const qwen36_fixture_merge *merge = &qwen36_fixture_merges[i];
        CHECK(merge->len == strlen(merge->text));
        table_put(&vocab->merge_rank,
                  (ds4_str){merge->text, merge->len}, merge->rank);
        if (merge->len > vocab->max_merge_len) {
            vocab->max_merge_len = merge->len;
        }
    }

    vocab_configure_qwen35(vocab);
}

static void fixture_engine_free(ds4_engine *engine) {
    vocab_free(&engine->vocab);
}

static void expect_tokens(
        const ds4_tokens *tokens,
        const int        *expected,
        size_t            expected_len,
        bool              has_sentinel) {
    const int prefix = has_sentinel ? 1 : 0;
    CHECK(tokens->len == prefix + (int)expected_len);
    if (has_sentinel && tokens->len > 0) {
        CHECK(tokens->v[0] == TEST_SENTINEL);
    }
    const int available = tokens->len - prefix;
    const int comparable = available < (int)expected_len ?
        available : (int)expected_len;
    for (int i = 0; i < comparable; i++) {
        if (tokens->v[prefix + i] != expected[i]) {
            fprintf(stderr,
                    "qwen tokenizer mismatch at token %d: got %d, expected %d\n",
                    i, tokens->v[prefix + i], expected[i]);
            failures++;
        }
    }
}

static const qwen36_fixture_case *fixture_case_named(const char *name) {
    for (size_t i = 0; i < TEST_ARRAY_LEN(qwen36_fixture_cases); i++) {
        if (strcmp(qwen36_fixture_cases[i].name, name) == 0) {
            return &qwen36_fixture_cases[i];
        }
    }
    CHECK(false);
    return NULL;
}

static void test_fixture_contract(void) {
    size_t expected_ids = 0;
    CHECK(TEST_ARRAY_LEN(qwen36_fixture_tokens) ==
          QWEN36_TOKENIZER_FIXTURE_TOKEN_COUNT);
    CHECK(TEST_ARRAY_LEN(qwen36_fixture_merges) ==
          QWEN36_TOKENIZER_FIXTURE_MERGE_COUNT);
    CHECK(TEST_ARRAY_LEN(qwen36_fixture_cases) ==
          QWEN36_TOKENIZER_FIXTURE_CASE_COUNT);
    CHECK(QWEN36_TOKENIZER_FIXTURE_CASE_COUNT == 23u);
    for (size_t i = 0; i < TEST_ARRAY_LEN(qwen36_fixture_cases); i++) {
        expected_ids += qwen36_fixture_cases[i].expected_len;
    }
    CHECK(expected_ids == QWEN36_TOKENIZER_FIXTURE_EXPECTED_ID_COUNT);
}

static void test_all_golden_cases(ds4_engine *engine) {
    ds4_tokens tokens = {0};
    for (size_t i = 0; i < TEST_ARRAY_LEN(qwen36_fixture_cases); i++) {
        const qwen36_fixture_case *golden = &qwen36_fixture_cases[i];
        CHECK(golden->text_len == strlen(golden->text));

        tokens.len = 0;
        ds4_tokens_push(&tokens, TEST_SENTINEL);
        if (golden->kind == QWEN36_FIXTURE_TEXT) {
            ds4_tokenize_text(engine, golden->text, &tokens);
        } else {
            ds4_tokenize_rendered_chat(engine, golden->text, &tokens);
        }
        expect_tokens(&tokens, golden->expected, golden->expected_len, true);
    }
    ds4_tokens_free(&tokens);
}

static void expect_public_failure_preserves(
        ds4_engine *engine,
        const char *text,
        bool        rendered) {
    ds4_tokens tokens = {0};
    ds4_tokens_push(&tokens, TEST_SENTINEL);
    if (rendered) {
        ds4_tokenize_rendered_chat(engine, text, &tokens);
    } else {
        ds4_tokenize_text(engine, text, &tokens);
    }
    CHECK(tokens.len == 1);
    CHECK(tokens.v[0] == TEST_SENTINEL);
    ds4_tokens_free(&tokens);
}

static void test_malformed_utf8(ds4_engine *engine) {
    static const char lone_continuation[] = {(char)0x80, '\0'};
    static const char overlong[] = {(char)0xc0, (char)0x80, '\0'};
    static const char truncated[] = {(char)0xe2, (char)0x82, '\0'};
    static const char surrogate[] = {
        (char)0xed, (char)0xa0, (char)0x80, '\0'
    };
    static const char too_large[] = {
        (char)0xf4, (char)0x90, (char)0x80, (char)0x80, '\0'
    };
    static const char five_byte[] = {
        (char)0xf8, (char)0x88, (char)0x80, (char)0x80, (char)0x80, '\0'
    };
    static const char *const malformed[] = {
        lone_continuation, overlong, truncated,
        surrogate, too_large, five_byte,
    };

    for (size_t i = 0; i < TEST_ARRAY_LEN(malformed); i++) {
        expect_public_failure_preserves(engine, malformed[i], false);
        expect_public_failure_preserves(engine, malformed[i], true);
    }
}

static void test_fail_closed_paths(void) {
    ds4_engine missing;
    fixture_engine_init(&missing, 9419); /* final symbol for "Hello" */
    expect_public_failure_preserves(&missing, "Hello", false);
    /* The recognized control must roll back along with the failing tail. */
    expect_public_failure_preserves(
        &missing, "<|im_start|>Hello", true);
    fixture_engine_free(&missing);

    ds4_engine unknown;
    fixture_engine_init(&unknown, -1);
    unknown.model.family = DS4_MODEL_FAMILY_UNKNOWN;
    unknown.vocab.family = DS4_MODEL_FAMILY_UNKNOWN;
    expect_public_failure_preserves(&unknown, "Hello", false);
    fixture_engine_free(&unknown);
}

static void expect_encoded_prompt(
        ds4_engine    *engine,
        const char    *case_name,
        const char    *system,
        const char    *prompt,
        ds4_think_mode think_mode) {
    const qwen36_fixture_case *golden = fixture_case_named(case_name);
    if (!golden) return;

    ds4_tokens tokens = {0};
    ds4_tokens_push(&tokens, TEST_SENTINEL);
    ds4_encode_chat_prompt(engine, system, prompt, think_mode, &tokens);
    expect_tokens(&tokens, golden->expected, golden->expected_len, true);
    ds4_tokens_free(&tokens);
}

static void test_chat_goldens(ds4_engine *engine) {
    expect_encoded_prompt(
        engine, "plain_thinking", NULL,
        "Quanto fa 17 * 23?", DS4_THINK_HIGH);
    expect_encoded_prompt(
        engine, "plain_no_thinking", NULL,
        "Rispondi solo: s\303\254", DS4_THINK_NONE);
    expect_encoded_prompt(
        engine, "system_and_user", "Sei un assistente conciso.",
        "Saluta in italiano.", DS4_THINK_HIGH);
}

static bool tokens_equal(const ds4_tokens *a, const ds4_tokens *b) {
    if (a->len != b->len) return false;
    for (int i = 0; i < a->len; i++) {
        if (a->v[i] != b->v[i]) return false;
    }
    return true;
}

static void test_public_incremental_chat(ds4_engine *engine) {
    const qwen36_fixture_case *golden = fixture_case_named("system_and_user");
    ds4_tokens tokens = {0};
    ds4_chat_begin(engine, &tokens);
    CHECK(tokens.len == 0); /* Qwen declares BOS metadata but adds no BOS. */
    ds4_chat_append_message(
        engine, &tokens, "system", "Sei un assistente conciso.");
    ds4_chat_append_message(
        engine, &tokens, "user", "Saluta in italiano.");
    ds4_chat_append_assistant_prefix(engine, &tokens, DS4_THINK_HIGH);
    if (golden) {
        expect_tokens(&tokens, golden->expected, golden->expected_len, false);
    }
    ds4_tokens_free(&tokens);

    ds4_tokens high = {0};
    ds4_tokens maximum = {0};
    ds4_chat_append_assistant_prefix(engine, &high, DS4_THINK_HIGH);
    ds4_chat_append_assistant_prefix(engine, &maximum, DS4_THINK_MAX);
    CHECK(tokens_equal(&high, &maximum));
    ds4_tokens_free(&high);
    ds4_tokens_free(&maximum);

    ds4_tokens no_op = {0};
    ds4_tokens_push(&no_op, TEST_SENTINEL);
    ds4_chat_append_max_effort_prefix(engine, &no_op);
    CHECK(no_op.len == 1 && no_op.v[0] == TEST_SENTINEL);
    ds4_tokens_free(&no_op);

    /* A generated turn normally finishes directly on <|im_end|>.  The next
     * incremental block must insert the template's newline before im_start. */
    static const int boundary_expected[] = {
        248046, 198, 248045, 846, 198, 87, 248046, 198,
    };
    ds4_tokens boundary = {0};
    ds4_tokens_push(&boundary, 248046);
    ds4_chat_append_message(engine, &boundary, "user", "x");
    expect_tokens(&boundary, boundary_expected,
                  TEST_ARRAY_LEN(boundary_expected), false);
    ds4_tokens_free(&boundary);

    /* Jinja/Python trim includes Unicode White_Space plus U+001C..U+001F. */
    static const char padded[] =
        " \302\240\034\035x\036\037\302\240 ";
    ds4_tokens plain = {0};
    ds4_tokens trimmed = {0};
    ds4_chat_append_message(engine, &plain, "user", "x");
    ds4_chat_append_message(engine, &trimmed, "user", padded);
    CHECK(tokens_equal(&plain, &trimmed));
    ds4_tokens_free(&plain);
    ds4_tokens_free(&trimmed);

    CHECK(ds4_token_eos(engine) == 248046);
    CHECK(ds4_token_user(engine) == -1);
    CHECK(ds4_token_assistant(engine) == -1);
}

static void expect_decoded(
        ds4_engine *engine,
        int         token,
        const char *expected) {
    size_t len = SIZE_MAX;
    char *text = ds4_token_text(engine, token, &len);
    CHECK(text != NULL);
    if (text) {
        CHECK(len == strlen(expected));
        CHECK(memcmp(text, expected, len + 1u) == 0);
    }
    free(text);
}

static void test_decode(ds4_engine *engine) {
    /* U+0120 is GPT-2's reversible spelling of an input space. */
    expect_decoded(engine, 279, " the");
    expect_decoded(engine, 248045, "<|im_start|>");
    expect_decoded(engine, 248068, "<think>");
}

static void bpe_test_vocab_init(
        ds4_vocab *vocab,
        uint64_t   token_count,
        uint64_t   merge_count) {
    memset(vocab, 0, sizeof(*vocab));
    table_init(&vocab->token_to_id, token_count);
    table_init(&vocab->merge_rank, merge_count);
}

static void bpe_test_vocab_free(ds4_vocab *vocab) {
    table_free(&vocab->token_to_id);
    table_free(&vocab->merge_rank);
}

static void bpe_test_put_token(
        ds4_vocab *vocab,
        const char *text,
        int id) {
    table_put(&vocab->token_to_id,
              (ds4_str){text, (uint64_t)strlen(text)}, id);
}

static void bpe_test_put_merge(
        ds4_vocab *vocab,
        const char *text,
        int rank) {
    const uint64_t len = (uint64_t)strlen(text);
    table_put(&vocab->merge_rank,
              (ds4_str){text, len}, rank);
    if (len > vocab->max_merge_len) vocab->max_merge_len = len;
}

static void test_bpe_leftmost_overlap(void) {
    ds4_vocab vocab;
    bpe_test_vocab_init(&vocab, 2, 1);
    bpe_test_put_token(&vocab, "a", 10);
    bpe_test_put_token(&vocab, "aa", 11);
    bpe_test_put_merge(&vocab, "a a", 0);

    token_vec out = {0};
    static const int expected[] = {11, 10};
    CHECK(bpe_emit_piece(
        &vocab, (ds4_str){"aaa", 3}, &out, true));
    expect_tokens(&out, expected, TEST_ARRAY_LEN(expected), false);

    token_vec_free(&out);
    bpe_test_vocab_free(&vocab);
}

static void test_bpe_stale_candidate(void) {
    ds4_vocab vocab;
    bpe_test_vocab_init(&vocab, 2, 2);
    bpe_test_put_token(&vocab, "a", 20);
    bpe_test_put_token(&vocab, "bc", 21);
    bpe_test_put_merge(&vocab, "b c", 0);
    bpe_test_put_merge(&vocab, "a b", 1);

    token_vec out = {0};
    static const int expected[] = {20, 21};
    CHECK(bpe_emit_piece(
        &vocab, (ds4_str){"abc", 3}, &out, true));
    expect_tokens(&out, expected, TEST_ARRAY_LEN(expected), false);

    token_vec_free(&out);
    bpe_test_vocab_free(&vocab);
}

static void test_bpe_long_actual_chain(void) {
    enum { RAW_LEN = 65536 };
    ds4_vocab vocab;
    bpe_test_vocab_init(&vocab, 1, 3);
    bpe_test_put_token(&vocab, "aaaaaaaa", 30);
    bpe_test_put_merge(&vocab, "a a", 0);
    bpe_test_put_merge(&vocab, "aa aa", 1);
    bpe_test_put_merge(&vocab, "aaaa aaaa", 2);

    char *raw = xmalloc(RAW_LEN + 1u);
    memset(raw, 'a', RAW_LEN);
    raw[RAW_LEN] = '\0';

    token_vec out = {0};
    g_bpe_test_rank_lookups = 0;
    CHECK(bpe_emit_piece(
        &vocab, (ds4_str){raw, RAW_LEN}, &out, true));
    CHECK(out.len == RAW_LEN / 8);
    for (int i = 0; i < out.len; i++) CHECK(out.v[i] == 30);
    CHECK(g_bpe_test_rank_lookups <= 3u * RAW_LEN);

    token_vec_free(&out);
    free(raw);
    bpe_test_vocab_free(&vocab);
}

static void test_bpe_heap_allocated_rank_key(void) {
    enum { FINAL_LEN = 512, N_MERGES = 9 };
    ds4_vocab vocab;
    bpe_test_vocab_init(&vocab, 1, N_MERGES);

    char *final_token = xmalloc(FINAL_LEN + 1u);
    memset(final_token, 'a', FINAL_LEN);
    final_token[FINAL_LEN] = '\0';
    bpe_test_put_token(&vocab, final_token, 40);

    char *merge_keys[N_MERGES] = {0};
    size_t symbol_len = 1;
    for (int rank = 0; rank < N_MERGES; rank++) {
        const size_t key_len = symbol_len * 2u + 1u;
        merge_keys[rank] = xmalloc(key_len + 1u);
        memset(merge_keys[rank], 'a', symbol_len);
        merge_keys[rank][symbol_len] = ' ';
        memset(merge_keys[rank] + symbol_len + 1u, 'a', symbol_len);
        merge_keys[rank][key_len] = '\0';
        bpe_test_put_merge(&vocab, merge_keys[rank], rank);
        symbol_len *= 2u;
    }
    CHECK(strlen(merge_keys[N_MERGES - 1]) > 512u);

    token_vec out = {0};
    static const int expected[] = {40};
    CHECK(bpe_emit_piece(
        &vocab, (ds4_str){final_token, FINAL_LEN}, &out, true));
    expect_tokens(&out, expected, TEST_ARRAY_LEN(expected), false);

    token_vec_free(&out);
    bpe_test_vocab_free(&vocab);
    for (int i = 0; i < N_MERGES; i++) free(merge_keys[i]);
    free(final_token);
}

static void test_bpe_deepseek_nonstrict_fallback(void) {
    ds4_vocab vocab;
    bpe_test_vocab_init(&vocab, 2, 1);
    bpe_test_put_token(&vocab, "a", 50);
    bpe_test_put_token(&vocab, "b", 51);
    bpe_test_put_merge(&vocab, "a b", 0);

    token_vec out = {0};
    static const int expected[] = {50, 51};
    CHECK(bpe_emit_piece(
        &vocab, (ds4_str){"ab", 2}, &out, false));
    expect_tokens(&out, expected, TEST_ARRAY_LEN(expected), false);

    token_vec_free(&out);
    bpe_test_vocab_free(&vocab);
}

static void test_heap_bpe(void) {
    test_bpe_leftmost_overlap();
    test_bpe_stale_candidate();
    test_bpe_long_actual_chain();
    test_bpe_heap_allocated_rank_key();
    test_bpe_deepseek_nonstrict_fallback();
}

int main(void) {
    test_fixture_contract();
    test_heap_bpe();

    ds4_engine engine;
    fixture_engine_init(&engine, -1);
    CHECK(engine.vocab.n_special == 33u);
    for (size_t i = 0; i < engine.vocab.n_special; i++) {
        CHECK(engine.vocab.special[i].id == 248044 + (int)i);
    }
    test_all_golden_cases(&engine);
    test_malformed_utf8(&engine);
    test_chat_goldens(&engine);
    test_public_incremental_chat(&engine);
    test_decode(&engine);
    fixture_engine_free(&engine);

    test_fail_closed_paths();
    ds4_threads_shutdown();

    if (failures) {
        fprintf(stderr, "Qwen tokenizer tests: %d failure(s)\n", failures);
        return 1;
    }
    puts("Qwen tokenizer tests: OK");
    return 0;
}
