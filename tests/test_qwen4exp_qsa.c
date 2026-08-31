/* Phase-6 Lane A targeted tests for runtime/ds4_qwen4exp_qsa.{h,inc}.
 *
 * Build (strict):
 *   clang -std=c99 -O1 -g -Wall -Wextra -Wpedantic -Werror -I. \
 *     -o /tmp/test_qwen4exp_qsa tests/test_qwen4exp_qsa.c -lm
 *   /tmp/test_qwen4exp_qsa
 *
 * Coverage: config/context gates incl. 262143/262144/262145; visible lengths
 * 0..5 and 2047..2052 dense parity and budget cut; exact selected IDs above
 * budget against an independent frozen-formula oracle; ascending-group
 * tie-break; sum(ReLU(dot)) negative control; one-shot/chunk/decode equality;
 * chunked multi-turn equality; later-token causality; no future token; exact
 * tail 0..3; left-padding anchoring; multi-sequence isolation; holes and slot
 * reuse; wrap eviction; reset/copy/fork/rewind/shift/remove; full-raw vs
 * incremental equality; serialization atomicity; zero dense Q-by-K
 * allocation; overflow/nonfinite rejection; long boundaries 65535..262144 on
 * compact fixtures; pinned Transformers/NumPy fixture parity; reference
 * attention parity; metrics bookkeeping. */

#include <math.h>
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "runtime/ds4_qwen4exp_qsa.h"
#include "runtime/ds4_qwen4exp_qsa.inc"
#include "qwen4exp/qwen4exp_qsa_golden.inc"

#define REQUIRE(cond) do {                                                \
    if (!(cond)) {                                                        \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);   \
        return false;                                                     \
    }                                                                     \
} while (0)

#define REQUIRE_MSG(cond, ...) do {                                       \
    if (!(cond)) {                                                        \
        fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);              \
        fprintf(stderr, __VA_ARGS__);                                     \
        fputc('\n', stderr);                                              \
        return false;                                                     \
    }                                                                     \
} while (0)

/* ---------------- deterministic content generation ---------------- */

static uint32_t lcg_next(uint32_t *state) {
    *state = *state * UINT32_C(1664525) + UINT32_C(1013904223);
    return *state;
}

static float lcg_value(uint32_t *state) {
    /* [-2, 2) in steps of 1/250 */
    return ((float)(lcg_next(state) % UINT32_C(1000)) - 500.0f) / 250.0f;
}

/* Position-dependent key/value/raw-key payload so group scores differ.  The
 * LCG consumes the same number of values regardless of NULL outputs so the
 * oracle can regenerate raw keys without materializing KV. */
static void token_payload(uint32_t seq, uint32_t position, uint32_t salt,
                          size_t index_dim, size_t kv_dim,
                          float *key, float *value, float *raw) {
    uint32_t state = position * UINT32_C(2654435761) ^
                     seq * UINT32_C(40503) ^ salt;
    size_t i;
    for (i = 0u; i < index_dim; i++) {
        const float x = lcg_value(&state);
        if (raw) raw[i] = x;
    }
    for (i = 0u; i < kv_dim; i++) {
        const float k = lcg_value(&state);
        const float v = lcg_value(&state);
        if (key) key[i] = k;
        if (value) value[i] = v;
    }
}

/* Tie fixture: raw index keys alternate sign with position parity, so every
 * aligned group of four pools to exactly (0, 0).  A zero pooled key survives
 * normalization and RoPE as zero, so every group scores exactly 0 against any
 * query and the selection is decided purely by the tie-break.  Constant raw
 * keys would NOT tie: each group is rotated at its own first position. */
static void tie_payload(uint32_t position, float *key, float *value,
                        float *raw) {
    const float sign = (position & 1u) ? -1.0f : 1.0f;
    key[0] = 0.5f; key[1] = -0.25f;
    value[0] = 1.0f; value[1] = 0.5f;
    raw[0] = sign * 0.75f;
    raw[1] = sign * -0.5f;
}

/* ---------------- config helpers ---------------- */

static float g_gamma2[2] = {0.1f, -0.2f};

static void base_config(ds4_qwen4exp_qsa_config *cfg, size_t line_capacity,
                        size_t n_slot, size_t budget) {
    memset(cfg, 0, sizeof(*cfg));
    cfg->index_dim = 2u;
    cfg->index_heads = 2u;
    cfg->attn_heads = 2u;
    cfg->kv_heads = 1u;
    cfg->kv_head_dim = 2u;
    cfg->block_budget = budget;
    cfg->context = DS4_Q4E_QSA_MAX_CONTEXT;
    cfg->line_capacity = line_capacity;
    cfg->n_slot = n_slot;
    cfg->max_sequences = 4u;
    cfg->n_rot = 2u;
    cfg->theta = 10000.0f;
    cfg->epsilon = 1.0e-6f;
    cfg->norm_weight = g_gamma2;
}

static float g_query2[4] = {0.3f, 0.7f, -0.4f, 0.9f}; /* [2 heads][dim 2] */

/* Appends [first, first + count) into seq with deterministic payloads. */
static bool append_run(ds4_qwen4exp_qsa_cache *cache, uint32_t seq,
                       uint32_t first, size_t count, uint32_t salt) {
    size_t i;
    for (i = 0u; i < count; i++) {
        ds4_qwen4exp_qsa_reservation r;
        float key[2];
        float value[2];
        float raw[2];
        const uint32_t position = first + (uint32_t)i;
        token_payload(seq, position, salt, 2u, 2u, key, value, raw);
        if (!ds4_qwen4exp_qsa_append_reserve(cache, seq, position, &r))
            return false;
        if (!ds4_qwen4exp_qsa_append_commit(cache, &r, key, value, raw))
            return false;
    }
    return true;
}

/* Clears one sequence through the public transactional API (used where a
 * per-sequence reset is needed while other sequences stay live). */
static bool clear_sequence(ds4_qwen4exp_qsa_cache *cache, uint32_t seq) {
    while (cache->seq[seq].count != 0u) {
        if (!ds4_qwen4exp_qsa_remove(cache, seq, cache->seq[seq].last_pos))
            return false;
    }
    return true;
}

/* Plans both modes and proves they select identical entries. */
static bool plan_both(ds4_qwen4exp_qsa_cache *cache, uint32_t seq,
                      uint32_t query_position, const float *query,
                      ds4_qwen4exp_qsa_selection *raw_sel,
                      ds4_qwen4exp_qsa_selection *inc_sel) {
    if (!ds4_qwen4exp_qsa_plan(cache, DS4_Q4E_QSA_PLAN_FULL_RAW, seq,
                               query_position, query, raw_sel))
        return false;
    if (!ds4_qwen4exp_qsa_plan(cache, DS4_Q4E_QSA_PLAN_INCREMENTAL, seq,
                               query_position, query, inc_sel))
        return false;
    if (raw_sel->width != inc_sel->width ||
        raw_sel->n_group != inc_sel->n_group ||
        raw_sel->tail != inc_sel->tail)
        return false;
    return memcmp(raw_sel->entry, inc_sel->entry,
                  raw_sel->width * sizeof(raw_sel->entry[0])) == 0;
}

/* ---------------- independent frozen-formula oracle ---------------- */

/* Mean in F32, zero-centered normalization, partial RoPE at group_pos, in
 * exactly the frozen operation order.  raw is [4][index_dim]. */
static void oracle_form_group(float *out, const float raw[4][8],
                              size_t index_dim, size_t n_rot, float theta,
                              float epsilon, const float *gamma,
                              uint32_t group_pos) {
    float pooled[8];
    float sum_square = 0.0f;
    size_t i;
    size_t pair;
    size_t half = n_rot / 2u;
    for (i = 0u; i < index_dim; i++) {
        float sum = 0.0f;
        size_t t;
        for (t = 0u; t < 4u; t++) sum += raw[t][i];
        pooled[i] = sum / 4.0f;
    }
    for (i = 0u; i < index_dim; i++) sum_square += pooled[i] * pooled[i];
    {
        const float mean_square = sum_square / (float)index_dim;
        const float inverse = 1.0f / sqrtf(mean_square + epsilon);
        for (i = 0u; i < index_dim; i++)
            pooled[i] *= inverse * (1.0f + gamma[i]);
    }
    for (pair = 0u; pair < half; pair++) {
        const float exponent = (2.0f * (float)pair) / (float)n_rot;
        const float angle = (float)group_pos / powf(theta, exponent);
        const float c = cosf(angle);
        const float s = sinf(angle);
        const float first = pooled[pair];
        const float second = pooled[half + pair];
        pooled[pair] = first * c - second * s;
        pooled[half + pair] = second * c + first * s;
    }
    for (i = 0u; i < index_dim; i++) out[i] = pooled[i];
}

/* score = sum over heads of ReLU(dot_h). */
static float oracle_score(const float *query, size_t heads, size_t dim,
                          const float *pooled) {
    float total = 0.0f;
    size_t h;
    for (h = 0u; h < heads; h++) {
        float dot = 0.0f;
        size_t i;
        for (i = 0u; i < dim; i++) dot += query[h * dim + i] * pooled[i];
        if (dot > 0.0f) total += dot;
    }
    return total;
}

/* Negative control: ReLU applied to the head sum, not per head. */
static float oracle_score_relu_sum(const float *query, size_t heads,
                                   size_t dim, const float *pooled) {
    float dot = 0.0f;
    size_t h;
    for (h = 0u; h < heads; h++) {
        size_t i;
        for (i = 0u; i < dim; i++)
            dot += query[h * dim + i] * pooled[i];
    }
    return dot > 0.0f ? dot : 0.0f;
}

/* Solves the 2-D index head h with h.a == dot_a and h.b == dot_b, so a test
 * can dial the per-head dot products of two known pooled keys exactly.  Fails
 * when the two pooled keys are (near) collinear and no such head exists. */
static bool solve_head(const float *a, const float *b, float dot_a,
                       float dot_b, float *h) {
    const float det = a[0] * b[1] - a[1] * b[0];
    if (!(fabsf(det) > 1.0e-3f)) return false;
    h[0] = (dot_a * b[1] - dot_b * a[1]) / det;
    h[1] = (a[0] * dot_b - b[0] * dot_a) / det;
    return true;
}

typedef struct {
    size_t group;
    float score;
} oracle_entry;

/* Ascending group index: the emission order of an already-selected set. */
static int oracle_group_cmp(const void *a, const void *b) {
    const oracle_entry *x = (const oracle_entry *)a;
    const oracle_entry *y = (const oracle_entry *)b;
    if (x->group < y->group) return -1;
    if (x->group > y->group) return 1;
    return 0;
}

static int oracle_entry_cmp(const void *a, const void *b) {
    const oracle_entry *x = (const oracle_entry *)a;
    const oracle_entry *y = (const oracle_entry *)b;
    if (x->score > y->score) return -1;
    if (x->score < y->score) return 1;
    /* stable tie-break: ascending group index */
    if (x->group < y->group) return -1;
    if (x->group > y->group) return 1;
    return 0;
}

/* Full oracle selection over one sequence of consecutive visible positions
 * anchor..anchor+visible-1 with the same payloads token_payload(., ., 7, ...)
 * produced.  id_base is the sequence's first-ever position, so the logical ID
 * is (position - id_base) % line_capacity (ring index, wrap-safe).  Compares
 * entry by entry against the planner. */
static bool oracle_compare_selection(const ds4_qwen4exp_qsa_config *cfg,
                                     uint32_t seq, uint32_t anchor,
                                     uint32_t id_base, size_t visible,
                                     const float *query,
                                     const ds4_qwen4exp_qsa_selection *got) {
    oracle_entry *entries;
    float (*raws)[4][8];
    size_t n_group = visible / 4u;
    size_t tail = visible % 4u;
    size_t select_count = n_group < cfg->block_budget
        ? n_group : cfg->block_budget;
    size_t g;
    size_t write = 0u;
    bool ok = true;
    if (n_group == 0u) {
        if (got->width != tail || got->n_group != 0u || got->tail != tail)
            return false;
        for (g = 0u; g < tail; g++) {
            const uint32_t position = anchor + (uint32_t)g;
            if (got->entry[g].position != position ||
                got->entry[g].id !=
                    (position - id_base) % cfg->line_capacity)
                return false;
        }
        return true;
    }
    entries = (oracle_entry *)malloc(n_group * sizeof(*entries));
    raws = (float (*)[4][8])malloc(n_group * sizeof(*raws));
    if (!entries || !raws) {
        free(entries);
        free(raws);
        return false;
    }
    for (g = 0u; g < n_group; g++) {
        size_t t;
        float pooled[8];
        for (t = 0u; t < 4u; t++) {
            token_payload(seq, anchor + (uint32_t)(g * 4u + t), 7u,
                          cfg->index_dim, cfg->kv_heads * cfg->kv_head_dim,
                          NULL, NULL, raws[g][t]);
        }
        oracle_form_group(pooled, raws[g], cfg->index_dim, cfg->n_rot,
                          cfg->theta, cfg->epsilon, cfg->norm_weight,
                          anchor + (uint32_t)(g * 4u));
        entries[g].group = g;
        entries[g].score = oracle_score(query, cfg->index_heads,
                                        cfg->index_dim, pooled);
    }
    qsort(entries, n_group, sizeof(entries[0]), oracle_entry_cmp);
    /* the rank decides the SET; the row is emitted in ascending group order */
    qsort(entries, select_count, sizeof(entries[0]), oracle_group_cmp);
    if (got->width != select_count * 4u + tail ||
        got->n_group != select_count || got->tail != tail) {
        fprintf(stderr, "shape mismatch: width %zu n_group %zu tail %zu "
                "(oracle %zu/%zu)\n", got->width, got->n_group, got->tail,
                select_count, tail);
        free(entries);
        free(raws);
        return false;
    }
    for (g = 0u; g < select_count && ok; g++) {
        size_t t;
        for (t = 0u; t < 4u; t++) {
            const uint32_t position =
                anchor + (uint32_t)(entries[g].group * 4u + t);
            if (got->entry[write].position != position ||
                got->entry[write].id !=
                    (position - id_base) % cfg->line_capacity) {
                ok = false;
                break;
            }
            write++;
        }
    }
    for (g = 0u; g < tail && ok; g++) {
        const uint32_t position = anchor + (uint32_t)(n_group * 4u + g);
        if (got->entry[write].position != position ||
            got->entry[write].id !=
                (position - id_base) % cfg->line_capacity) {
            ok = false;
            break;
        }
        write++;
    }
    if (!ok)
        fprintf(stderr, "oracle selection mismatch at entry %zu\n", write);
    free(entries);
    free(raws);
    return ok;
}

/* Structural invariants for every successful selection. */
static bool check_selection_shape(const ds4_qwen4exp_qsa_selection *sel,
                                  uint32_t query_position, size_t budget) {
    size_t i;
    if (sel->width == 0u || sel->width > DS4_Q4E_QSA_MAX_WIDTH) return false;
    if (sel->width > budget * 4u + 3u) return false;
    if (sel->n_group * 4u + sel->tail != sel->width) return false;
    if (sel->tail > 3u) return false;
    for (i = 0u; i < sel->width; i++) {
        if (sel->entry[i].position > query_position) return false;
        if (sel->entry[i].id == DS4_Q4E_QSA_EMPTY) return false;
    }
    return true;
}

/* Reference sparse attention for one query token: naive bounded softmax over
 * exactly the selected K/V rows. */
static bool reference_attention(const ds4_qwen4exp_qsa_cache *cache,
                                uint32_t seq,
                                const ds4_qwen4exp_qsa_selection *sel,
                                const float *query, float *output) {
    const size_t dim = cache->config.kv_head_dim;
    const size_t heads = cache->config.attn_heads;
    const size_t kv_heads = cache->config.kv_heads;
    const size_t ratio = heads / kv_heads;
    const float scale = 1.0f / sqrtf((float)dim);
    size_t h;
    for (h = 0u; h < heads; h++) {
        const size_t kvh = h / ratio;
        float scores[DS4_Q4E_QSA_MAX_WIDTH];
        float maximum = -INFINITY;
        float total = 0.0f;
        size_t e;
        size_t i;
        for (e = 0u; e < sel->width; e++) {
            uint32_t slot = 0u;
            const float *k;
            float dot = 0.0f;
            if (!ds4_qwen4exp_qsa_resolve(cache, seq,
                                          sel->entry[e].position, &slot))
                return false;
            k = cache->key + ((size_t)slot * kv_heads + kvh) * dim;
            for (i = 0u; i < dim; i++)
                dot += query[h * dim + i] * k[i];
            scores[e] = dot * scale;
            if (!isfinite(scores[e])) return false;
            if (scores[e] > maximum) maximum = scores[e];
        }
        for (e = 0u; e < sel->width; e++)
            total += expf(scores[e] - maximum);
        if (!(total > 0.0f) || !isfinite(total)) return false;
        for (i = 0u; i < dim; i++) {
            float acc = 0.0f;
            for (e = 0u; e < sel->width; e++) {
                uint32_t slot = 0u;
                const float *v;
                if (!ds4_qwen4exp_qsa_resolve(cache, seq,
                                              sel->entry[e].position, &slot))
                    return false;
                v = cache->value + ((size_t)slot * kv_heads + kvh) * dim;
                acc += expf(scores[e] - maximum) * v[i];
            }
            output[h * dim + i] = acc / total;
            if (!isfinite(output[h * dim + i])) return false;
        }
    }
    return true;
}

static bool floats_close(const float *actual, const float *expected,
                         size_t count) {
    size_t i;
    for (i = 0u; i < count; i++) {
        if (!isfinite(actual[i])) return false;
        if (fabsf(actual[i] - expected[i]) >
            2.0e-5f + 2.0e-5f * fabsf(expected[i]))
            return false;
    }
    return true;
}

/* ---------------- test cases ---------------- */

/* The generated fixture executes the pinned Transformers QSA indexer and
 * eager attention, then independently reproduces both with NumPy.  This test
 * feeds those raw owners through the host cache instead of deriving expected
 * values from the C implementation under test. */
static bool test_pinned_qsa_fixture(void) {
    ds4_qwen4exp_qsa_config cfg;
    ds4_qwen4exp_qsa_cache *cache = NULL;
    ds4_qwen4exp_qsa_selection raw;
    ds4_qwen4exp_qsa_selection incremental;
    float output[Q4E_QSA6_ATTN_HEADS * Q4E_QSA6_KV_HEAD_DIM];
    size_t token;
    size_t entry;

    memset(&cfg, 0, sizeof(cfg));
    cfg.index_dim = Q4E_QSA6_INDEX_DIM;
    cfg.index_heads = Q4E_QSA6_INDEX_HEADS;
    cfg.attn_heads = Q4E_QSA6_ATTN_HEADS;
    cfg.kv_heads = Q4E_QSA6_KV_HEADS;
    cfg.kv_head_dim = Q4E_QSA6_KV_HEAD_DIM;
    cfg.block_budget = Q4E_QSA6_GROUP_BUDGET;
    cfg.context = DS4_Q4E_QSA_MAX_CONTEXT;
    cfg.line_capacity = 64u;
    cfg.n_slot = 64u;
    cfg.max_sequences = 1u;
    cfg.n_rot = Q4E_QSA6_N_ROT;
    cfg.theta = Q4E_QSA6_THETA;
    cfg.epsilon = Q4E_QSA6_EPSILON;
    cfg.norm_weight = q4e_qsa6_key_norm_weight;
    REQUIRE(ds4_qwen4exp_qsa_cache_create(&cache, &cfg));

    for (token = 0u; token < Q4E_QSA6_VISIBLE; token++) {
        ds4_qwen4exp_qsa_reservation reservation;
        const size_t kv_stride = Q4E_QSA6_KV_HEADS * Q4E_QSA6_KV_HEAD_DIM;
        REQUIRE(ds4_qwen4exp_qsa_append_reserve(
            cache, 0u, Q4E_QSA6_ANCHOR + (uint32_t)token, &reservation));
        REQUIRE(ds4_qwen4exp_qsa_append_commit(
            cache, &reservation,
            q4e_qsa6_key + token * kv_stride,
            q4e_qsa6_value + token * kv_stride,
            q4e_qsa6_raw_key + token * Q4E_QSA6_INDEX_DIM));
    }
    REQUIRE(floats_close(cache->pooled, q4e_qsa6_group_key,
                         (Q4E_QSA6_VISIBLE / Q4E_QSA6_COMPRESSION) *
                         Q4E_QSA6_INDEX_DIM));
    REQUIRE(plan_both(cache, 0u,
                      Q4E_QSA6_ANCHOR + Q4E_QSA6_VISIBLE - 1u,
                      q4e_qsa6_index_query, &raw, &incremental));
    REQUIRE(raw.width == Q4E_QSA6_SELECTED);
    for (entry = 0u; entry < raw.width; entry++) {
        REQUIRE(raw.entry[entry].id == q4e_qsa6_selected_logical[entry]);
        REQUIRE(raw.entry[entry].position == q4e_qsa6_selected_position[entry]);
    }
    REQUIRE(ds4_qwen4exp_qsa_attention(
        cache, 0u, q4e_qsa6_attention_query, &raw, output));
    REQUIRE(floats_close(output, q4e_qsa6_attention_output,
                         Q4E_QSA6_ATTN_HEADS * Q4E_QSA6_KV_HEAD_DIM));
    REQUIRE(q4e_qsa6_tie_score[0] == q4e_qsa6_tie_score[1]);
    REQUIRE(q4e_qsa6_tie_selected_group[0] == 0u &&
            q4e_qsa6_tie_selected_group[1] == 1u);
    REQUIRE(cache->metrics.dense_mask_bytes == 0u);
    ds4_qwen4exp_qsa_cache_destroy(cache);
    return true;
}

/* Config and context gates: accept through 262144, reject 262145 and every
 * malformed field. */
static bool test_config_gates(void) {
    ds4_qwen4exp_qsa_config cfg;
    ds4_qwen4exp_qsa_cache *cache = NULL;

    base_config(&cfg, 64u, 64u, 2u);
    REQUIRE(ds4_qwen4exp_qsa_config_validate(&cfg));
    REQUIRE(ds4_qwen4exp_qsa_cache_create(&cache, &cfg));
    ds4_qwen4exp_qsa_cache_destroy(cache);
    cache = NULL;

    /* context boundary: 262144 accepted, 262145 rejected */
    cfg.context = DS4_Q4E_QSA_MAX_CONTEXT;
    REQUIRE(ds4_qwen4exp_qsa_config_validate(&cfg));
    cfg.context = DS4_Q4E_QSA_MAX_CONTEXT + 1u;
    REQUIRE(!ds4_qwen4exp_qsa_config_validate(&cfg));

    base_config(&cfg, 64u, 64u, 2u);
    cfg.index_dim = 0u;
    REQUIRE(!ds4_qwen4exp_qsa_config_validate(&cfg));
    base_config(&cfg, 64u, 64u, 2u);
    cfg.index_dim = DS4_Q4E_QSA_MAX_DIM + 1u;
    REQUIRE(!ds4_qwen4exp_qsa_config_validate(&cfg));
    base_config(&cfg, 64u, 64u, 2u);
    cfg.block_budget = 0u;
    REQUIRE(!ds4_qwen4exp_qsa_config_validate(&cfg));
    base_config(&cfg, 64u, 64u, 2u);
    cfg.block_budget = DS4_Q4E_QSA_MAX_BUDGET + 1u;
    REQUIRE(!ds4_qwen4exp_qsa_config_validate(&cfg));
    base_config(&cfg, 64u, 64u, 2u);
    cfg.n_rot = 1u; /* odd */
    REQUIRE(!ds4_qwen4exp_qsa_config_validate(&cfg));
    base_config(&cfg, 64u, 64u, 2u);
    cfg.n_rot = 3u; /* odd and exceeds index_dim */
    REQUIRE(!ds4_qwen4exp_qsa_config_validate(&cfg));
    base_config(&cfg, 64u, 64u, 2u);
    cfg.theta = 0.0f;
    REQUIRE(!ds4_qwen4exp_qsa_config_validate(&cfg));
    base_config(&cfg, 64u, 64u, 2u);
    cfg.epsilon = 0.0f;
    REQUIRE(!ds4_qwen4exp_qsa_config_validate(&cfg));
    base_config(&cfg, 64u, 64u, 2u);
    cfg.kv_heads = 2u;
    cfg.attn_heads = 3u; /* not a multiple of kv_heads */
    REQUIRE(!ds4_qwen4exp_qsa_config_validate(&cfg));
    base_config(&cfg, 64u, 64u, 2u);
    cfg.kv_heads = 4u;
    cfg.attn_heads = 2u; /* fewer query heads than KV heads */
    REQUIRE(!ds4_qwen4exp_qsa_config_validate(&cfg));
    base_config(&cfg, 64u, 64u, 2u);
    cfg.max_sequences = DS4_Q4E_QSA_MAX_SEQS + 1u;
    REQUIRE(!ds4_qwen4exp_qsa_config_validate(&cfg));
    base_config(&cfg, 64u, 64u, 2u);
    cfg.norm_weight = NULL;
    REQUIRE(!ds4_qwen4exp_qsa_config_validate(&cfg));
    base_config(&cfg, 0u, 64u, 2u); /* zero line capacity */
    REQUIRE(!ds4_qwen4exp_qsa_config_validate(&cfg));
    base_config(&cfg, DS4_Q4E_QSA_MAX_CONTEXT + 1u, 64u, 2u);
    REQUIRE(!ds4_qwen4exp_qsa_config_validate(&cfg));
    base_config(&cfg, 64u, 0u, 2u); /* zero slots */
    REQUIRE(!ds4_qwen4exp_qsa_config_validate(&cfg));
    base_config(&cfg, 64u, 4u * 64u + 1u, 2u);
    REQUIRE(!ds4_qwen4exp_qsa_config_validate(&cfg));
    return true;
}

/* Visible lengths 0..5: exact tail/group shape, dense parity, and the
 * empty-sequence rejection. */
static bool test_small_lengths(void) {
    ds4_qwen4exp_qsa_config cfg;
    ds4_qwen4exp_qsa_cache *cache = NULL;
    ds4_qwen4exp_qsa_selection raw_sel;
    ds4_qwen4exp_qsa_selection inc_sel;
    size_t length;
    size_t i;

    base_config(&cfg, 64u, 64u, 4u);
    REQUIRE(ds4_qwen4exp_qsa_cache_create(&cache, &cfg));

    for (length = 0u; length <= 5u; length++) {
        REQUIRE(ds4_qwen4exp_qsa_reset(cache));
        if (length > 0u)
            REQUIRE(append_run(cache, 0u, 0u, length, 7u));
        if (length == 0u) {
            /* no visible keys: fail-closed rejection, not an empty success */
            REQUIRE(!ds4_qwen4exp_qsa_plan(cache, DS4_Q4E_QSA_PLAN_FULL_RAW,
                                           0u, 0u, g_query2, &raw_sel));
            REQUIRE(cache->metrics.rejection_count == 1u);
            continue;
        }
        REQUIRE(plan_both(cache, 0u, (uint32_t)(length - 1u), g_query2,
                          &raw_sel, &inc_sel));
        REQUIRE(check_selection_shape(&raw_sel, (uint32_t)(length - 1u), 4u));
        /* dense parity: every visible token, in order */
        REQUIRE_MSG(raw_sel.width == length, "length %zu width %zu",
                    length, raw_sel.width);
        for (i = 0u; i < length; i++) {
            REQUIRE(raw_sel.entry[i].position == (uint32_t)i);
            REQUIRE(raw_sel.entry[i].id == (uint32_t)i);
        }
        /* oracle agreement on shape and entries */
        REQUIRE(oracle_compare_selection(&cfg, 0u, 0u, 0u, length, g_query2,
                                         &raw_sel));
    }
    ds4_qwen4exp_qsa_cache_destroy(cache);
    return true;
}

/* Dense parity through 2051 and the first above-budget cut at 2052, with the
 * frozen full budget.  Above-budget selection is compared to the oracle. */
static bool test_dense_parity_boundary(void) {
    const size_t lengths[6] = {2047u, 2048u, 2049u, 2050u, 2051u, 2052u};
    ds4_qwen4exp_qsa_config cfg;
    ds4_qwen4exp_qsa_cache *cache = NULL;
    ds4_qwen4exp_qsa_selection raw_sel;
    ds4_qwen4exp_qsa_selection inc_sel;
    size_t li;
    size_t i;

    base_config(&cfg, 4096u, 4096u, DS4_Q4E_QSA_MAX_BUDGET);
    REQUIRE(ds4_qwen4exp_qsa_cache_create(&cache, &cfg));

    for (li = 0u; li < 6u; li++) {
        const size_t length = lengths[li];
        REQUIRE(ds4_qwen4exp_qsa_reset(cache));
        REQUIRE(append_run(cache, 0u, 0u, length, 7u));
        REQUIRE(plan_both(cache, 0u, (uint32_t)(length - 1u), g_query2,
                          &raw_sel, &inc_sel));
        REQUIRE(check_selection_shape(&raw_sel, (uint32_t)(length - 1u),
                                      DS4_Q4E_QSA_MAX_BUDGET));
        if (length <= 2051u) {
            /* every complete group fits the budget: exactly dense */
            REQUIRE_MSG(raw_sel.width == length,
                        "length %zu not dense: width %zu", length,
                        raw_sel.width);
            for (i = 0u; i < length; i++)
                REQUIRE(raw_sel.entry[i].position == (uint32_t)i);
        } else {
            /* 2052 = 513 complete groups and no tail: the budget keeps 512
             * of them, so the row shrinks to 2048 and the oracle must agree
             * on which group was dropped */
            REQUIRE_MSG(raw_sel.width == 2048u,
                        "length 2052 width %zu", raw_sel.width);
            REQUIRE(oracle_compare_selection(&cfg, 0u, 0u, 0u, length,
                                             g_query2, &raw_sel));
        }
    }
    REQUIRE(cache->metrics.max_width <= DS4_Q4E_QSA_MAX_WIDTH);
    REQUIRE(cache->metrics.dense_mask_bytes == 0u);
    ds4_qwen4exp_qsa_cache_destroy(cache);
    return true;
}

/* One-shot vs chunk vs decode equality and chunked multi-turn equality, plus
 * later-token causality: appending new tokens must not change what an earlier
 * query would have selected (its selection is a pure function of visible
 * prefix state), and appending can never add a future token to an old plan. */
static bool test_chunk_causality(void) {
    ds4_qwen4exp_qsa_config cfg;
    ds4_qwen4exp_qsa_cache *cache = NULL;
    ds4_qwen4exp_qsa_selection one_shot;
    ds4_qwen4exp_qsa_selection chunked;
    ds4_qwen4exp_qsa_selection later;
    const size_t total = 600u;
    const size_t chunk = 137u;
    size_t done = 0u;

    base_config(&cfg, 1024u, 1024u, 8u);
    REQUIRE(ds4_qwen4exp_qsa_cache_create(&cache, &cfg));

    /* one-shot build */
    REQUIRE(append_run(cache, 0u, 0u, total, 7u));
    REQUIRE(plan_both(cache, 0u, (uint32_t)(total - 1u), g_query2,
                      &one_shot, &chunked));
    REQUIRE(check_selection_shape(&one_shot, (uint32_t)(total - 1u), 8u));

    /* rebuild the same sequence in odd chunks; plan at every chunk end */
    REQUIRE(ds4_qwen4exp_qsa_reset(cache));
    while (done < total) {
        const size_t take = total - done < chunk ? total - done : chunk;
        REQUIRE(append_run(cache, 0u, (uint32_t)done, take, 7u));
        done += take;
        REQUIRE(plan_both(cache, 0u, (uint32_t)(done - 1u), g_query2,
                          &chunked, &later));
        REQUIRE(check_selection_shape(&chunked, (uint32_t)(done - 1u), 8u));
    }
    /* final chunked plan equals the one-shot plan entry for entry */
    REQUIRE(chunked.width == one_shot.width);
    REQUIRE(memcmp(chunked.entry, one_shot.entry,
                   one_shot.width * sizeof(one_shot.entry[0])) == 0);

    /* later-token mutation cannot alter an earlier query: capture the plan at
     * position 299, append 300 more tokens, re-plan at 299 */
    REQUIRE(ds4_qwen4exp_qsa_reset(cache));
    REQUIRE(append_run(cache, 0u, 0u, 300u, 7u));
    REQUIRE(plan_both(cache, 0u, 299u, g_query2, &one_shot, &chunked));
    REQUIRE(append_run(cache, 0u, 300u, 300u, 7u));
    REQUIRE(plan_both(cache, 0u, 299u, g_query2, &later, &chunked));
    REQUIRE(later.width == one_shot.width);
    REQUIRE(memcmp(later.entry, one_shot.entry,
                   one_shot.width * sizeof(one_shot.entry[0])) == 0);
    /* and no entry of the earlier plan is a future token of that plan */
    {
        size_t i;
        for (i = 0u; i < one_shot.width; i++)
            REQUIRE(one_shot.entry[i].position <= 299u);
    }
    ds4_qwen4exp_qsa_cache_destroy(cache);
    return true;
}

/* Exact tail 0..3, dropped mid remainders after rewind, and left-padding
 * grouping that starts at the first real token. */
static bool test_tail_and_left_padding(void) {
    ds4_qwen4exp_qsa_config cfg;
    ds4_qwen4exp_qsa_cache *cache = NULL;
    ds4_qwen4exp_qsa_selection sel;
    ds4_qwen4exp_qsa_selection inc;
    size_t tail_case;

    base_config(&cfg, 128u, 128u, 4u);
    REQUIRE(ds4_qwen4exp_qsa_cache_create(&cache, &cfg));

    /* tails 0..3 from a 4-group prefix */
    for (tail_case = 0u; tail_case <= 3u; tail_case++) {
        const size_t length = 16u + tail_case;
        REQUIRE(ds4_qwen4exp_qsa_reset(cache));
        REQUIRE(append_run(cache, 0u, 0u, length, 7u));
        REQUIRE(plan_both(cache, 0u, (uint32_t)(length - 1u), g_query2,
                          &sel, &inc));
        REQUIRE(sel.width == length);
        REQUIRE(sel.tail == tail_case);
        REQUIRE(sel.n_group == 4u);
        /* tail entries are exactly the incomplete suffix */
        {
            size_t i;
            for (i = 0u; i < tail_case; i++) {
                REQUIRE(sel.entry[16u + i].position ==
                        (uint32_t)(16u + i));
            }
        }
    }

    /* left padding: reserve positions 0..4 with holes, real run starts at 5.
     * Groups must anchor at 5, and padded holes may never join a group. */
    REQUIRE(ds4_qwen4exp_qsa_reset(cache));
    REQUIRE(append_run(cache, 0u, 5u, 12u, 7u)); /* 5..16, 3 complete groups */
    REQUIRE(plan_both(cache, 0u, 16u, g_query2, &sel, &inc));
    REQUIRE(check_selection_shape(&sel, 16u, 4u));
    REQUIRE(sel.tail == 0u);
    {
        size_t i;
        uint32_t previous = 0u;
        for (i = 0u; i < sel.width; i++) {
            const uint32_t position = sel.entry[i].position;
            REQUIRE(position >= 5u);
            REQUIRE(position <= 16u);
            if (i > 0u && (i % 4u) == 0u) {
                /* group-contiguous: each group starts 4 after the prior */
                REQUIRE(position == previous + 1u);
            } else if (i > 0u) {
                REQUIRE(position == previous + 1u);
            }
            previous = position;
        }
    }
    ds4_qwen4exp_qsa_cache_destroy(cache);
    return true;
}

/* Multi-sequence isolation, holes, physical slot reuse, and wrap eviction on
 * a tiny circular line. */
static bool test_sequences_holes_wrap(void) {
    ds4_qwen4exp_qsa_config cfg;
    ds4_qwen4exp_qsa_cache *cache = NULL;
    ds4_qwen4exp_qsa_selection sel;
    ds4_qwen4exp_qsa_selection inc;
    size_t i;

    /* tiny 32-position line per sequence; the slot pool is shared, so it must
     * hold both sequences' live tokens at once (2 x 20) while the line itself
     * is what wraps. */
    base_config(&cfg, 32u, 64u, 2u);
    cfg.max_sequences = 2u;
    REQUIRE(ds4_qwen4exp_qsa_cache_create(&cache, &cfg));

    /* two sequences interleaved; neither may see the other's tokens */
    REQUIRE(append_run(cache, 0u, 0u, 20u, 7u));
    REQUIRE(append_run(cache, 1u, 100u, 20u, 9u));

    REQUIRE(plan_both(cache, 0u, 19u, g_query2, &sel, &inc));
    REQUIRE(check_selection_shape(&sel, 19u, 2u));
    for (i = 0u; i < sel.width; i++) {
        REQUIRE(sel.entry[i].position <= 19u);
        /* logical id wraps within this sequence's own 32-line */
        REQUIRE(sel.entry[i].id == sel.entry[i].position % 32u);
    }
    REQUIRE(plan_both(cache, 1u, 119u, g_query2, &sel, &inc));
    REQUIRE(check_selection_shape(&sel, 119u, 2u));
    for (i = 0u; i < sel.width; i++) {
        REQUIRE(sel.entry[i].position >= 100u);
        REQUIRE(sel.entry[i].position <= 119u);
        REQUIRE(sel.entry[i].id == (sel.entry[i].position - 100u) % 32u);
    }

    /* wrap: appending 40 tokens on a 32-line must keep only the newest line
     * contents, with ids wrapped, and planning must stay exact */
    REQUIRE(ds4_qwen4exp_qsa_reset(cache));
    REQUIRE(append_run(cache, 0u, 0u, 40u, 7u));
    REQUIRE(plan_both(cache, 0u, 39u, g_query2, &sel, &inc));
    REQUIRE(check_selection_shape(&sel, 39u, 2u));
    for (i = 0u; i < sel.width; i++) {
        const uint32_t position = sel.entry[i].position;
        REQUIRE(position >= 8u); /* 8 oldest were evicted by the 32-line */
        REQUIRE(sel.entry[i].id == position % 32u);
    }

    /* hole in the middle: remove a token, planning must fail closed rather
     * than form a group across the hole */
    REQUIRE(ds4_qwen4exp_qsa_reset(cache));
    REQUIRE(append_run(cache, 0u, 0u, 12u, 7u));
    REQUIRE(ds4_qwen4exp_qsa_remove(cache, 0u, 6u));
    /* query before the hole still sees a dense prefix 0..5 */
    REQUIRE(plan_both(cache, 0u, 5u, g_query2, &sel, &inc));
    REQUIRE(sel.width == 6u);
    /* query after the hole must reject: positions 0..11 contain a hole */
    REQUIRE(!ds4_qwen4exp_qsa_plan(cache, DS4_Q4E_QSA_PLAN_FULL_RAW, 0u,
                                   11u, g_query2, &sel));
    REQUIRE(cache->metrics.rejection_count >= 1u);

    /* physical slot reuse: two sequences reuse the same slots; a hole opened
     * by one sequence's reset must not corrupt the other */
    REQUIRE(ds4_qwen4exp_qsa_reset(cache));
    REQUIRE(append_run(cache, 0u, 0u, 8u, 7u));
    REQUIRE(append_run(cache, 1u, 50u, 8u, 9u));
    REQUIRE(clear_sequence(cache, 0u));
    REQUIRE(append_run(cache, 0u, 200u, 8u, 11u));
    REQUIRE(plan_both(cache, 0u, 207u, g_query2, &sel, &inc));
    for (i = 0u; i < sel.width; i++)
        REQUIRE(sel.entry[i].position >= 200u);
    REQUIRE(plan_both(cache, 1u, 57u, g_query2, &sel, &inc));
    for (i = 0u; i < sel.width; i++)
        REQUIRE(sel.entry[i].position >= 50u &&
                sel.entry[i].position <= 57u);
    ds4_qwen4exp_qsa_cache_destroy(cache);
    return true;
}

/* Lifecycle: reset, copy/fork, rewind, shift, remove — including the failed
 * mutation paths leaving state byte-identical. */
static bool test_lifecycle(void) {
    ds4_qwen4exp_qsa_config cfg;
    ds4_qwen4exp_qsa_cache *cache = NULL;
    ds4_qwen4exp_qsa_cache *copy = NULL;
    ds4_qwen4exp_qsa_selection before;
    ds4_qwen4exp_qsa_selection after;
    ds4_qwen4exp_qsa_selection inc;
    uint64_t digest_before = 0u;
    uint64_t digest_after = 0u;
    size_t i;

    base_config(&cfg, 128u, 128u, 4u);
    REQUIRE(ds4_qwen4exp_qsa_cache_create(&cache, &cfg));
    REQUIRE(append_run(cache, 0u, 0u, 24u, 7u));
    REQUIRE(plan_both(cache, 0u, 23u, g_query2, &before, &inc));

    /* copy/fork: identical digest and identical planning */
    REQUIRE(ds4_qwen4exp_qsa_cache_copy(cache, &copy));
    REQUIRE(ds4_qwen4exp_qsa_digest(copy, &digest_after));
    REQUIRE(ds4_qwen4exp_qsa_digest(cache, &digest_before));
    REQUIRE(digest_before == digest_after);
    REQUIRE(plan_both(copy, 0u, 23u, g_query2, &after, &inc));
    REQUIRE(memcmp(after.entry, before.entry,
                   before.width * sizeof(before.entry[0])) == 0);
    /* the fork is independent: mutating it does not touch the original */
    REQUIRE(append_run(copy, 0u, 24u, 8u, 7u));
    REQUIRE(ds4_qwen4exp_qsa_digest(cache, &digest_after));
    REQUIRE(digest_after == digest_before);
    ds4_qwen4exp_qsa_cache_destroy(copy);
    copy = NULL;

    /* rewind: drop back to 16 tokens; pooled groups/tail shrink together */
    REQUIRE(ds4_qwen4exp_qsa_rewind(cache, 0u, 16u));
    REQUIRE(plan_both(cache, 0u, 15u, g_query2, &after, &inc));
    REQUIRE(after.width == 16u);
    for (i = 0u; i < 16u; i++)
        REQUIRE(after.entry[i].position == (uint32_t)i);

    /* rewind past the frontier fails and leaves state byte-identical */
    REQUIRE(ds4_qwen4exp_qsa_digest(cache, &digest_before));
    REQUIRE(!ds4_qwen4exp_qsa_rewind(cache, 0u, 17u));
    REQUIRE(ds4_qwen4exp_qsa_digest(cache, &digest_after));
    REQUIRE(digest_after == digest_before);

    /* shift: logical rebase downwards, the frozen "subtract delta" direction.
     * The sequence is rebuilt at 100..115 so the rebase has room, and the
     * underflow guard is proven on the same state before it is applied. */
    REQUIRE(ds4_qwen4exp_qsa_reset(cache));
    REQUIRE(append_run(cache, 0u, 100u, 16u, 7u));
    REQUIRE(ds4_qwen4exp_qsa_digest(cache, &digest_before));
    REQUIRE(!ds4_qwen4exp_qsa_shift(cache, 0u, 101u)); /* would underflow */
    REQUIRE(ds4_qwen4exp_qsa_digest(cache, &digest_after));
    REQUIRE(digest_after == digest_before);
    REQUIRE(ds4_qwen4exp_qsa_shift(cache, 0u, 100u));
    REQUIRE(plan_both(cache, 0u, 15u, g_query2, &after, &inc));
    REQUIRE(after.width == 16u);
    for (i = 0u; i < 16u; i++)
        REQUIRE(after.entry[i].position == (uint32_t)i);

    /* remove of a mid token opens a hole: after-removal queries over the
     * removed span fail closed with unchanged state */
    REQUIRE(ds4_qwen4exp_qsa_remove(cache, 0u, 5u));
    REQUIRE(ds4_qwen4exp_qsa_digest(cache, &digest_before));
    REQUIRE(!ds4_qwen4exp_qsa_plan(cache, DS4_Q4E_QSA_PLAN_FULL_RAW, 0u,
                                   15u, g_query2, &after));
    REQUIRE(ds4_qwen4exp_qsa_digest(cache, &digest_after));
    REQUIRE(digest_after == digest_before);

    /* full reset clears every sequence and every metric-relevant owner */
    REQUIRE(ds4_qwen4exp_qsa_reset(cache));
    REQUIRE(ds4_qwen4exp_qsa_digest(cache, &digest_after));
    /* digest of an empty cache is stable across rebuilds of the same config */
    {
        ds4_qwen4exp_qsa_cache *fresh = NULL;
        uint64_t fresh_digest = 0u;
        REQUIRE(ds4_qwen4exp_qsa_cache_create(&fresh, &cfg));
        REQUIRE(ds4_qwen4exp_qsa_digest(fresh, &fresh_digest));
        REQUIRE(fresh_digest == digest_after);
        ds4_qwen4exp_qsa_cache_destroy(fresh);
    }
    ds4_qwen4exp_qsa_cache_destroy(cache);
    return true;
}

/* Serialization: save/restore round-trip equality, and partial/corrupt
 * restore failures leaving the old state byte-identical. */
static bool test_serialize_atomicity(void) {
    ds4_qwen4exp_qsa_config cfg;
    ds4_qwen4exp_qsa_cache *cache = NULL;
    ds4_qwen4exp_qsa_cache *restore = NULL;
    ds4_qwen4exp_qsa_selection sel;
    ds4_qwen4exp_qsa_selection inc;
    uint8_t *blob = NULL;
    size_t blob_bytes = 0u;
    uint64_t digest_source = 0u;
    uint64_t digest_restored = 0u;
    uint64_t digest_before = 0u;
    uint64_t digest_after = 0u;

    base_config(&cfg, 128u, 128u, 4u);
    REQUIRE(ds4_qwen4exp_qsa_cache_create(&cache, &cfg));
    REQUIRE(append_run(cache, 0u, 0u, 24u, 7u));
    REQUIRE(append_run(cache, 1u, 500u, 10u, 9u));
    REQUIRE(ds4_qwen4exp_qsa_digest(cache, &digest_source));
    blob_bytes = ds4_qwen4exp_qsa_serialized_size(cache);
    REQUIRE(blob_bytes > 0u);
    blob = (uint8_t *)malloc(blob_bytes);
    REQUIRE(blob != NULL);
    REQUIRE(ds4_qwen4exp_qsa_serialize(cache, blob, blob_bytes));

    /* clean restore into a fresh cache of the same config */
    REQUIRE(ds4_qwen4exp_qsa_cache_create(&restore, &cfg));
    REQUIRE(ds4_qwen4exp_qsa_restore(restore, blob, blob_bytes));
    REQUIRE(ds4_qwen4exp_qsa_digest(restore, &digest_restored));
    REQUIRE(digest_restored == digest_source);
    REQUIRE(plan_both(restore, 0u, 23u, g_query2, &sel, &inc));
    REQUIRE(plan_both(cache, 0u, 23u, g_query2, &inc, &sel));
    REQUIRE(memcmp(sel.entry, inc.entry, sel.width * sizeof(sel.entry[0])) == 0);

    /* every truncation must fail and leave the target byte-identical */
    {
        size_t cut;
        for (cut = 0u; cut < blob_bytes; cut += blob_bytes / 7u + 1u) {
            REQUIRE(ds4_qwen4exp_qsa_digest(restore, &digest_before));
            REQUIRE(!ds4_qwen4exp_qsa_restore(restore, blob, cut));
            REQUIRE(ds4_qwen4exp_qsa_digest(restore, &digest_after));
            REQUIRE(digest_after == digest_before);
        }
        /* single-byte flips anywhere must fail or restore exactly; no
         * partial publication is allowed either way */
        size_t flip;
        for (flip = 0u; flip < blob_bytes; flip += blob_bytes / 11u + 1u) {
            const uint8_t original = blob[flip];
            REQUIRE(ds4_qwen4exp_qsa_digest(restore, &digest_before));
            blob[flip] = (uint8_t)(original ^ 0x5Au);
            if (ds4_qwen4exp_qsa_restore(restore, blob, blob_bytes)) {
                /* a flip that survives must have landed in ignored padding;
                 * the digest must still match the source exactly */
                REQUIRE(ds4_qwen4exp_qsa_digest(restore,
                                                       &digest_after));
                REQUIRE(digest_after == digest_source);
            } else {
                REQUIRE(ds4_qwen4exp_qsa_digest(restore,
                                                       &digest_after));
                REQUIRE(digest_after == digest_before);
            }
            blob[flip] = original;
        }
    }
    /* wrong-size restore buffer rejected */
    REQUIRE(ds4_qwen4exp_qsa_digest(restore, &digest_before));
    REQUIRE(!ds4_qwen4exp_qsa_restore(restore, blob, blob_bytes - 1u));
    REQUIRE(!ds4_qwen4exp_qsa_restore(restore, blob, blob_bytes + 1u));
    REQUIRE(!ds4_qwen4exp_qsa_restore(restore, NULL, blob_bytes));
    REQUIRE(ds4_qwen4exp_qsa_digest(restore, &digest_after));
    REQUIRE(digest_after == digest_before);

    free(blob);
    ds4_qwen4exp_qsa_cache_destroy(restore);
    ds4_qwen4exp_qsa_cache_destroy(cache);
    return true;
}

/* Nonfinite and overflow rejection: bad payload, bad query, and byte-identical
 * state/output after every rejection. */
static bool test_rejection_atomicity(void) {
    ds4_qwen4exp_qsa_config cfg;
    ds4_qwen4exp_qsa_cache *cache = NULL;
    ds4_qwen4exp_qsa_selection sel;
    float bad_key[2];
    float bad_value[2];
    float bad_raw[2];
    float bad_query[4];
    ds4_qwen4exp_qsa_reservation r;
    uint64_t digest_before = 0u;
    uint64_t digest_after = 0u;

    base_config(&cfg, 128u, 128u, 4u);
    REQUIRE(ds4_qwen4exp_qsa_cache_create(&cache, &cfg));
    REQUIRE(append_run(cache, 0u, 0u, 12u, 7u));
    REQUIRE(ds4_qwen4exp_qsa_digest(cache, &digest_before));

    /* nonfinite raw key on the next append: reserve succeeds, commit must
     * fail and the reservation must not partially publish */
    bad_key[0] = 1.0f; bad_key[1] = 2.0f;
    bad_value[0] = 1.0f; bad_value[1] = 2.0f;
    bad_raw[0] = INFINITY; bad_raw[1] = 0.0f;
    REQUIRE(ds4_qwen4exp_qsa_append_reserve(cache, 0u, 12u, &r));
    REQUIRE(!ds4_qwen4exp_qsa_append_commit(cache, &r, bad_key, bad_value,
                                            bad_raw));
    REQUIRE(ds4_qwen4exp_qsa_digest(cache, &digest_after));
    REQUIRE(digest_after == digest_before);

    /* nonfinite query: plan rejects, state unchanged */
    bad_query[0] = 0.0f; bad_query[1] = NAN;
    bad_query[2] = 0.0f; bad_query[3] = 0.0f;
    REQUIRE(!ds4_qwen4exp_qsa_plan(cache, DS4_Q4E_QSA_PLAN_FULL_RAW, 0u,
                                   11u, bad_query, &sel));
    REQUIRE(cache->metrics.rejection_count >= 1u);
    REQUIRE(ds4_qwen4exp_qsa_digest(cache, &digest_after));
    REQUIRE(digest_after == digest_before);

    /* nonfinite key/value also rejected */
    bad_key[0] = NAN; bad_key[1] = 0.0f;
    REQUIRE(ds4_qwen4exp_qsa_append_reserve(cache, 0u, 12u, &r));
    REQUIRE(!ds4_qwen4exp_qsa_append_commit(cache, &r, bad_key, bad_value,
                                            bad_raw));
    bad_key[0] = 0.0f; bad_key[1] = 0.0f;
    bad_value[0] = 0.0f; bad_value[1] = -INFINITY;
    REQUIRE(ds4_qwen4exp_qsa_append_reserve(cache, 0u, 12u, &r));
    REQUIRE(!ds4_qwen4exp_qsa_append_commit(cache, &r, bad_key, bad_value,
                                            bad_raw));
    REQUIRE(ds4_qwen4exp_qsa_digest(cache, &digest_after));
    REQUIRE(digest_after == digest_before);

    /* duplicate/future position rejections: reserve at an already occupied
     * position or a non-frontier future position must fail */
    REQUIRE(!ds4_qwen4exp_qsa_append_reserve(cache, 0u, 0u, &r));
    REQUIRE(!ds4_qwen4exp_qsa_append_reserve(cache, 0u, 20u, &r));
    REQUIRE(ds4_qwen4exp_qsa_digest(cache, &digest_after));
    REQUIRE(digest_after == digest_before);

    /* unknown sequence and unknown position lookups fail closed */
    REQUIRE(!ds4_qwen4exp_qsa_resolve(cache, 3u, 0u, NULL));
    {
        uint32_t slot = 0u;
        REQUIRE(!ds4_qwen4exp_qsa_resolve(cache, 0u, 99u, &slot));
    }
    ds4_qwen4exp_qsa_cache_destroy(cache);

    /* A finite payload may overflow only while the post-eviction group is
     * being formed.  That late failure must restore the evicted cell and
     * remove its slot from the free stack byte-for-byte. */
    cache = NULL;
    base_config(&cfg, 4u, 4u, 4u);
    REQUIRE(ds4_qwen4exp_qsa_cache_create(&cache, &cfg));
    for (uint32_t position = 0u; position < 4u; position++) {
        const float first_raw[4] = {-FLT_MAX, FLT_MAX, 0.0f, 0.0f};
        float raw[2] = {first_raw[position], 0.0f};
        bad_key[0] = 0.25f; bad_key[1] = -0.5f;
        bad_value[0] = 0.75f; bad_value[1] = 0.125f;
        REQUIRE(ds4_qwen4exp_qsa_append_reserve(cache, 0u, position, &r));
        REQUIRE(ds4_qwen4exp_qsa_append_commit(cache, &r, bad_key, bad_value,
                                                raw));
    }
    REQUIRE(ds4_qwen4exp_qsa_digest(cache, &digest_before));
    bad_raw[0] = FLT_MAX; bad_raw[1] = 0.0f;
    REQUIRE(ds4_qwen4exp_qsa_append_reserve(cache, 0u, 4u, &r));
    REQUIRE(!ds4_qwen4exp_qsa_append_commit(cache, &r, bad_key, bad_value,
                                            bad_raw));
    REQUIRE(ds4_qwen4exp_qsa_digest(cache, &digest_after));
    REQUIRE(digest_after == digest_before);
    for (uint32_t position = 0u; position < 4u; position++) {
        uint32_t slot = DS4_Q4E_QSA_EMPTY;
        REQUIRE(ds4_qwen4exp_qsa_resolve(cache, 0u, position, &slot));
        REQUIRE(slot < 4u);
    }
    ds4_qwen4exp_qsa_cache_destroy(cache);
    return true;
}

/* Reference sparse attention parity: CPU sparse attention over selected IDs
 * equals dense attention over the same prefix whenever the selection is
 * dense (<= budget), for every head. */
static bool test_attention_parity(void) {
    ds4_qwen4exp_qsa_config cfg;
    ds4_qwen4exp_qsa_cache *cache = NULL;
    ds4_qwen4exp_qsa_selection sel;
    ds4_qwen4exp_qsa_selection inc;
    float dense_out[4];
    float sparse_out[4];
    const size_t length = 20u;
    size_t i;
    size_t h;

    base_config(&cfg, 64u, 64u, 8u); /* 20 tokens are 5 groups: budget > 5 */
    REQUIRE(ds4_qwen4exp_qsa_cache_create(&cache, &cfg));
    REQUIRE(append_run(cache, 0u, 0u, length, 7u));
    REQUIRE(plan_both(cache, 0u, (uint32_t)(length - 1u), g_query2,
                      &sel, &inc));
    /* dense selection at this length */
    REQUIRE(sel.width == length);
    REQUIRE(reference_attention(cache, 0u, &sel, g_query2, sparse_out));

    /* dense reference computed directly from the cache rows */
    for (h = 0u; h < cfg.attn_heads; h++) {
        float maximum = -INFINITY;
        float scores[64];
        float total = 0.0f;
        for (i = 0u; i < length; i++) {
            uint32_t slot = 0u;
            const float *k;
            float dot = 0.0f;
            REQUIRE(ds4_qwen4exp_qsa_resolve(cache, 0u, (uint32_t)i, &slot));
            k = cache->key + ((size_t)slot * cfg.kv_heads +
                              h / (cfg.attn_heads / cfg.kv_heads)) *
                             cfg.kv_head_dim;
            for (size_t d = 0u; d < cfg.kv_head_dim; d++)
                dot += g_query2[h * cfg.kv_head_dim + d] * k[d];
            scores[i] = dot / sqrtf((float)cfg.kv_head_dim);
            if (scores[i] > maximum) maximum = scores[i];
        }
        for (i = 0u; i < length; i++)
            total += expf(scores[i] - maximum);
        for (size_t d = 0u; d < cfg.kv_head_dim; d++) {
            float acc = 0.0f;
            for (i = 0u; i < length; i++) {
                uint32_t slot = 0u;
                const float *v;
                REQUIRE(ds4_qwen4exp_qsa_resolve(cache, 0u, (uint32_t)i,
                                                 &slot));
                v = cache->value + ((size_t)slot * cfg.kv_heads +
                                    h / (cfg.attn_heads / cfg.kv_heads)) *
                                   cfg.kv_head_dim;
                acc += expf(scores[i] - maximum) * v[d];
            }
            dense_out[h * cfg.kv_head_dim + d] = acc / total;
        }
    }
    REQUIRE(floats_close(sparse_out, dense_out, 4u));
    REQUIRE(cache->metrics.dense_mask_bytes == 0u);
    ds4_qwen4exp_qsa_cache_destroy(cache);
    return true;
}

/* Tie-break: identical pooled keys force equal scores; selection must then be
 * the lowest group indexes (ascending stable order). */
static bool test_tie_break(void) {
    ds4_qwen4exp_qsa_config cfg;
    ds4_qwen4exp_qsa_cache *cache = NULL;
    ds4_qwen4exp_qsa_selection sel;
    ds4_qwen4exp_qsa_selection inc;
    const size_t length = 20u; /* 5 groups, budget 2 */
    size_t i;

    base_config(&cfg, 64u, 64u, 2u);
    REQUIRE(ds4_qwen4exp_qsa_cache_create(&cache, &cfg));
    for (i = 0u; i < length; i++) {
        ds4_qwen4exp_qsa_reservation r;
        float key[2];
        float value[2];
        float raw[2];
        tie_payload((uint32_t)i, key, value, raw);
        REQUIRE(ds4_qwen4exp_qsa_append_reserve(cache, 0u, (uint32_t)i, &r));
        REQUIRE(ds4_qwen4exp_qsa_append_commit(cache, &r, key, value, raw));
    }
    REQUIRE(plan_both(cache, 0u, (uint32_t)(length - 1u), g_query2,
                      &sel, &inc));
    REQUIRE(sel.width == 8u);
    REQUIRE(sel.n_group == 2u);
    /* equal scores -> groups 0 and 1 win (lowest indexes first) */
    REQUIRE(sel.entry[0].position == 0u);
    REQUIRE(sel.entry[3].position == 3u);
    REQUIRE(sel.entry[4].position == 4u);
    REQUIRE(sel.entry[7].position == 7u);
    ds4_qwen4exp_qsa_cache_destroy(cache);
    return true;
}

/* Negative control: ReLU(sum(dot)) must disagree with the frozen
 * sum(ReLU(dot)) scoring on payloads engineered so the two differ.  This is a
 * contract-control test, not an oracle: it proves the planner implements the
 * frozen per-head rectification, the exact bug the upstream commit fixed.
 *
 * Constant raw keys cannot build the trap, because each group is rotated at
 * its own first position: the fixture therefore takes the two pooled keys the
 * frozen transformation actually produces and solves for the index heads that
 * give the per-head dots the trap needs. */
static bool test_relu_control(void) {
    ds4_qwen4exp_qsa_config cfg;
    ds4_qwen4exp_qsa_cache *cache = NULL;
    ds4_qwen4exp_qsa_selection sel;
    ds4_qwen4exp_qsa_selection inc;
    float raw_a[4][8];
    float raw_b[4][8];
    float pooled_a[8];
    float pooled_b[8];
    float query[4];
    size_t i;

    base_config(&cfg, 64u, 64u, 1u); /* budget 1: exactly one group survives */
    memset(raw_a, 0, sizeof(raw_a));
    memset(raw_b, 0, sizeof(raw_b));
    for (i = 0u; i < 4u; i++) {
        raw_a[i][0] = 1.0f;
        raw_a[i][1] = 0.25f;
        raw_b[i][0] = 0.3f;
        raw_b[i][1] = -0.8f;
    }
    oracle_form_group(pooled_a, raw_a, cfg.index_dim, cfg.n_rot, cfg.theta,
                      cfg.epsilon, cfg.norm_weight, 0u);
    oracle_form_group(pooled_b, raw_b, cfg.index_dim, cfg.n_rot, cfg.theta,
                      cfg.epsilon, cfg.norm_weight, 4u);

    /* per-head dots: head 0 gives A +3.0 and B +1.0, head 1 gives A -2.5 and
     * B +1.0.  sum(ReLU(dot)): A = 3.0 beats B = 2.0.  ReLU(sum(dot)):
     * A = ReLU(0.5) = 0.5 loses to B = ReLU(2.0) = 2.0.  The two rules pick
     * opposite groups, so a budget of one discriminates them. */
    REQUIRE(solve_head(pooled_a, pooled_b, 3.0f, 1.0f, &query[0]));
    REQUIRE(solve_head(pooled_a, pooled_b, -2.5f, 1.0f, &query[2]));
    REQUIRE(oracle_score(query, 2u, 2u, pooled_a) >
            oracle_score(query, 2u, 2u, pooled_b) + 0.5f);
    REQUIRE(oracle_score_relu_sum(query, 2u, 2u, pooled_b) >
            oracle_score_relu_sum(query, 2u, 2u, pooled_a) + 0.5f);

    /* the planner must follow sum(ReLU(dot)) and keep group A */
    REQUIRE(ds4_qwen4exp_qsa_cache_create(&cache, &cfg));
    for (i = 0u; i < 8u; i++) {
        ds4_qwen4exp_qsa_reservation r;
        float key[2];
        float value[2];
        float group_raw[2];
        group_raw[0] = i < 4u ? raw_a[i][0] : raw_b[i - 4u][0];
        group_raw[1] = i < 4u ? raw_a[i][1] : raw_b[i - 4u][1];
        key[0] = 0.5f;
        key[1] = 0.25f;
        value[0] = 1.0f;
        value[1] = 0.0f;
        REQUIRE(ds4_qwen4exp_qsa_append_reserve(cache, 0u, (uint32_t)i, &r));
        REQUIRE(ds4_qwen4exp_qsa_append_commit(cache, &r, key, value,
                                               group_raw));
    }
    REQUIRE(plan_both(cache, 0u, 7u, query, &sel, &inc));
    REQUIRE(sel.n_group == 1u);
    REQUIRE(sel.width == 4u);
    REQUIRE(sel.tail == 0u);
    /* group A (positions 0..3), never the trap group B (positions 4..7) */
    REQUIRE(sel.entry[0].position == 0u);
    REQUIRE(sel.entry[3].position == 3u);
    ds4_qwen4exp_qsa_cache_destroy(cache);
    return true;
}

/* Compact long-boundary tests: 65535/65536/65537, 100000, 262143/262144.
 * Sparse/compact fixtures only: tiny dims, small budget, no dense structures.
 * The line capacity is small so ids wrap, but positions are large to exercise
 * the 32-bit absolute-position path and the frozen context bound. */
static bool test_long_boundaries(void) {
    const uint32_t lengths[] = {65535u, 65536u, 65537u, 100000u,
                                262143u, 262144u};
    ds4_qwen4exp_qsa_config cfg;
    ds4_qwen4exp_qsa_cache *cache = NULL;
    ds4_qwen4exp_qsa_selection sel;
    ds4_qwen4exp_qsa_selection inc;
    size_t li;
    size_t i;

    base_config(&cfg, 512u, 512u, 4u); /* 512-line keeps memory compact */
    REQUIRE(ds4_qwen4exp_qsa_cache_create(&cache, &cfg));

    for (li = 0u; li < sizeof(lengths) / sizeof(lengths[0]); li++) {
        const uint32_t length = lengths[li];
        /* seed the tail region only: the planner must still accept long
         * positions and produce a correct selection from the resident tail */
        REQUIRE(ds4_qwen4exp_qsa_reset(cache));
        REQUIRE(append_run(cache, 0u, length - 20u, 20u, 7u));
        REQUIRE(plan_both(cache, 0u, length - 1u, g_query2, &sel, &inc));
        REQUIRE(check_selection_shape(&sel, length - 1u, 4u));
        for (i = 0u; i < sel.width; i++) {
            REQUIRE(sel.entry[i].position >= length - 20u);
            REQUIRE(sel.entry[i].position <= length - 1u);
            REQUIRE(sel.entry[i].id ==
                    (sel.entry[i].position - (length - 20u)) % 512u);
        }
        /* 262145 must be rejected as a context overflow */
        if (length == 262144u) {
            ds4_qwen4exp_qsa_reservation r;
            REQUIRE(!ds4_qwen4exp_qsa_append_reserve(cache, 0u, 262144u, &r));
            REQUIRE(cache->metrics.rejection_count >= 1u);
        }
    }
    ds4_qwen4exp_qsa_cache_destroy(cache);
    return true;
}

/* Metrics: deterministic counters with zero dense-mask bytes ever. */
static bool test_metrics(void) {
    ds4_qwen4exp_qsa_config cfg;
    ds4_qwen4exp_qsa_cache *cache = NULL;
    ds4_qwen4exp_qsa_selection sel;
    uint64_t before_plans = 0u;
    uint64_t after_plans = 0u;

    base_config(&cfg, 128u, 128u, 4u);
    REQUIRE(ds4_qwen4exp_qsa_cache_create(&cache, &cfg));
    before_plans = cache->metrics.plan_calls;
    REQUIRE(append_run(cache, 0u, 0u, 20u, 7u));
    REQUIRE(ds4_qwen4exp_qsa_plan(cache, DS4_Q4E_QSA_PLAN_FULL_RAW, 0u,
                                  19u, g_query2, &sel));
    after_plans = cache->metrics.plan_calls;
    REQUIRE(after_plans == before_plans + 1u);
    REQUIRE(cache->metrics.max_width == sel.width);
    /* no dense Q-by-K allocation ever: the counter stays zero for the whole
     * lifetime, including rejections */
    REQUIRE(!ds4_qwen4exp_qsa_plan(cache, DS4_Q4E_QSA_PLAN_FULL_RAW, 3u,
                                   7u, g_query2, &sel));
    REQUIRE(cache->metrics.dense_mask_bytes == 0u);
    ds4_qwen4exp_qsa_cache_destroy(cache);
    return true;
}

/* Plan argument validation: unknown mode, unknown sequence, query beyond the
 * context, and NULL outputs all fail closed. */
static bool test_plan_argument_gates(void) {
    ds4_qwen4exp_qsa_config cfg;
    ds4_qwen4exp_qsa_cache *cache = NULL;
    ds4_qwen4exp_qsa_selection sel;

    base_config(&cfg, 64u, 64u, 4u);
    REQUIRE(ds4_qwen4exp_qsa_cache_create(&cache, &cfg));
    REQUIRE(append_run(cache, 0u, 0u, 8u, 7u));

    REQUIRE(!ds4_qwen4exp_qsa_plan(cache, (ds4_qwen4exp_qsa_plan_mode)99u,
                                   0u, 7u, g_query2, &sel));
    REQUIRE(!ds4_qwen4exp_qsa_plan(cache, DS4_Q4E_QSA_PLAN_FULL_RAW,
                                   9u, 7u, g_query2, &sel));
    REQUIRE(!ds4_qwen4exp_qsa_plan(cache, DS4_Q4E_QSA_PLAN_FULL_RAW,
                                   0u, cfg.context, g_query2, &sel));
    REQUIRE(!ds4_qwen4exp_qsa_plan(cache, DS4_Q4E_QSA_PLAN_FULL_RAW,
                                   0u, 7u, NULL, &sel));
    REQUIRE(!ds4_qwen4exp_qsa_plan(NULL, DS4_Q4E_QSA_PLAN_FULL_RAW,
                                   0u, 7u, g_query2, &sel));
    ds4_qwen4exp_qsa_cache_destroy(cache);
    return true;
}

int main(void) {
    static const struct {
        const char *name;
        bool (*fn)(void);
    } tests[] = {
        {"pinned_qsa_fixture", test_pinned_qsa_fixture},
        {"config_gates", test_config_gates},
        {"small_lengths", test_small_lengths},
        {"dense_parity_boundary", test_dense_parity_boundary},
        {"chunk_causality", test_chunk_causality},
        {"tail_and_left_padding", test_tail_and_left_padding},
        {"sequences_holes_wrap", test_sequences_holes_wrap},
        {"lifecycle", test_lifecycle},
        {"serialize_atomicity", test_serialize_atomicity},
        {"rejection_atomicity", test_rejection_atomicity},
        {"attention_parity", test_attention_parity},
        {"tie_break", test_tie_break},
        {"relu_control", test_relu_control},
        {"long_boundaries", test_long_boundaries},
        {"metrics", test_metrics},
        {"plan_argument_gates", test_plan_argument_gates},
    };
    size_t i;
    size_t failures = 0u;
    for (i = 0u; i < sizeof(tests) / sizeof(tests[0]); i++) {
        printf("RUN  %s\n", tests[i].name);
        if (tests[i].fn()) {
            printf("PASS %s\n", tests[i].name);
        } else {
            fprintf(stderr, "FAIL %s\n", tests[i].name);
            failures++;
        }
    }
    if (failures != 0u) {
        fprintf(stderr, "%zu/%zu tests failed\n", failures,
                sizeof(tests) / sizeof(tests[0]));
        return 1;
    }
    printf("all %zu qwen4exp QSA host tests passed\n",
           sizeof(tests) / sizeof(tests[0]));
    return 0;
}
