/* Phase 6 native sparse QSA host implementation.  Included by ds4.c and the
 * standalone test translation unit; see ds4_qwen4exp_qsa.h for the frozen
 * contract.  Strict C99, no VLAs; allocation happens only at creation and
 * inside transactional structural operations that publish atomically. */

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#define Q4E_QSA_MAGIC 0x51534131u /* "QSA1" */
#define Q4E_QSA_VERSION 1u

typedef struct {
    uint32_t position; /* DS4_Q4E_QSA_EMPTY = hole */
    uint32_t slot;     /* DS4_Q4E_QSA_EMPTY = hole */
} ds4_q4e_qsa_cell;

typedef struct {
    uint32_t count;      /* live cells */
    uint32_t occupied;   /* ring span from head, live cells plus holes */
    uint32_t head;       /* ring index of the oldest cell */
    uint32_t last_pos;   /* newest live position; EMPTY when count == 0 */
    uint32_t n_group;    /* formed complete groups */
    uint32_t tail_start; /* absolute position of the first tail token */
    uint32_t tail_len;   /* 0..3 */
} ds4_q4e_qsa_seq_state;

struct ds4_qwen4exp_qsa_cache {
    ds4_qwen4exp_qsa_config config;
    size_t groups_cap; /* per-sequence group capacity: line_capacity/4 + 1 */

    float *norm_weight; /* [index_dim] */
    ds4_q4e_qsa_cell *cell;     /* [max_sequences][line_capacity] */
    uint32_t *live_idx;         /* [max_sequences][line_capacity] ring idx */
    ds4_q4e_qsa_seq_state *seq; /* [max_sequences] */
    uint32_t *free_slot;        /* [n_slot] stack */
    size_t free_count;
    float *raw_key; /* [n_slot][index_dim] */
    float *key;     /* [n_slot][kv_heads][kv_head_dim] */
    float *value;   /* [n_slot][kv_heads][kv_head_dim] */
    float *pooled;    /* [max_sequences][groups_cap][index_dim] */
    uint32_t *group_pos; /* [max_sequences][groups_cap] */

    /* planner + rebuild workspace, all owned here, never grown */
    float *score;        /* [groups_cap] */
    uint32_t *member;    /* [groups_cap][4] ring indices */
    uint32_t *chosen;    /* [block_budget] */
    float *pool_scratch; /* [compression][index_dim] */
    float *rb_pooled;    /* [groups_cap][index_dim] rebuild scratch */
    uint32_t *rb_pos;    /* [groups_cap] rebuild scratch */
    float *save_pooled;  /* [groups_cap][index_dim] rollback scratch */
    uint32_t *save_pos;  /* [groups_cap] rollback scratch */

    ds4_qwen4exp_qsa_metrics metrics;
    ds4_qwen4exp_qsa_byte_report report;
    int has_pending;
    ds4_qwen4exp_qsa_reservation pending;
};

static bool q4e_qsa_mul2(size_t a, size_t b, size_t *out) {
    if (a == 0u || b == 0u) {
        *out = 0u;
        return true;
    }
    if (a > SIZE_MAX / b) return false;
    *out = a * b;
    return true;
}

static bool q4e_qsa_mul3(size_t a, size_t b, size_t c, size_t *out) {
    size_t ab = 0u;
    return q4e_qsa_mul2(a, b, &ab) && q4e_qsa_mul2(ab, c, out);
}

static bool q4e_qsa_finite(const float *values, size_t count) {
    size_t i;
    if (!values && count != 0u) return false;
    for (i = 0u; i < count; i++) {
        if (!isfinite(values[i])) return false;
    }
    return true;
}

static uint64_t q4e_qsa_digest_bytes(uint64_t hash, const void *bytes,
                                     size_t size) {
    const uint8_t *p = (const uint8_t *)bytes;
    size_t i;
    for (i = 0u; i < size; i++) {
        hash ^= (uint64_t)p[i];
        hash *= UINT64_C(0x100000001b3);
    }
    return hash;
}

static uint64_t q4e_qsa_digest_u32(uint64_t hash, uint32_t value) {
    return q4e_qsa_digest_bytes(hash, &value, sizeof(value));
}

static uint64_t q4e_qsa_digest_size(uint64_t hash, size_t value) {
    return q4e_qsa_digest_bytes(hash, &value, sizeof(value));
}

bool ds4_qwen4exp_qsa_config_validate(const ds4_qwen4exp_qsa_config *config) {
    if (!config) return false;
    if (config->index_dim == 0u || config->index_dim > DS4_Q4E_QSA_MAX_DIM ||
        config->index_heads == 0u || config->index_heads > DS4_Q4E_QSA_MAX_DIM ||
        config->kv_heads == 0u || config->kv_heads > DS4_Q4E_QSA_MAX_DIM ||
        config->attn_heads < config->kv_heads ||
        config->attn_heads > DS4_Q4E_QSA_MAX_DIM ||
        config->attn_heads % config->kv_heads != 0u ||
        config->kv_head_dim == 0u ||
        config->kv_head_dim > DS4_Q4E_QSA_MAX_DIM ||
        config->block_budget == 0u ||
        config->block_budget > DS4_Q4E_QSA_MAX_BUDGET ||
        config->context == 0u || config->context > DS4_Q4E_QSA_MAX_CONTEXT ||
        config->line_capacity == 0u ||
        config->line_capacity > config->context ||
        config->max_sequences == 0u ||
        config->max_sequences > DS4_Q4E_QSA_MAX_SEQS ||
        config->n_slot == 0u ||
        config->n_slot > config->max_sequences * config->line_capacity ||
        config->n_rot < 2u || config->n_rot > config->index_dim ||
        (config->n_rot & 1u) != 0u ||
        !(config->theta > 0.0f) || !isfinite(config->theta) ||
        !(config->epsilon > 0.0f) || !isfinite(config->epsilon) ||
        !config->norm_weight ||
        !q4e_qsa_finite(config->norm_weight, config->index_dim)) {
        return false;
    }
    return true;
}

static size_t q4e_qsa_groups_cap(size_t line_capacity) {
    return line_capacity / DS4_Q4E_QSA_COMPRESSION + 1u;
}

bool ds4_qwen4exp_qsa_cache_create(ds4_qwen4exp_qsa_cache **cache,
                                   const ds4_qwen4exp_qsa_config *config) {
    ds4_qwen4exp_qsa_cache *c = NULL;
    size_t groups_cap = 0u;
    size_t cell_count = 0u;
    size_t raw_count = 0u;
    size_t kv_count = 0u;
    size_t pooled_count = 0u;
    if (!cache || !config || !ds4_qwen4exp_qsa_config_validate(config)) {
        return false;
    }
    groups_cap = q4e_qsa_groups_cap(config->line_capacity);
    if (!q4e_qsa_mul2(config->max_sequences, config->line_capacity,
                      &cell_count) ||
        !q4e_qsa_mul2(config->n_slot, config->index_dim, &raw_count) ||
        !q4e_qsa_mul3(config->n_slot, config->kv_heads, config->kv_head_dim,
                      &kv_count) ||
        !q4e_qsa_mul3(config->max_sequences, groups_cap, config->index_dim,
                      &pooled_count)) {
        return false;
    }
    c = (ds4_qwen4exp_qsa_cache *)calloc(1u, sizeof(*c));
    if (!c) return false;
    c->config = *config;
    c->groups_cap = groups_cap;
    c->norm_weight = (float *)malloc(config->index_dim * sizeof(float));
    c->cell = (ds4_q4e_qsa_cell *)calloc(cell_count, sizeof(ds4_q4e_qsa_cell));
    c->live_idx = (uint32_t *)calloc(cell_count, sizeof(uint32_t));
    c->seq = (ds4_q4e_qsa_seq_state *)calloc(config->max_sequences,
                                             sizeof(ds4_q4e_qsa_seq_state));
    c->free_slot = (uint32_t *)malloc(config->n_slot * sizeof(uint32_t));
    c->raw_key = (float *)calloc(raw_count, sizeof(float));
    c->key = (float *)calloc(kv_count, sizeof(float));
    c->value = (float *)calloc(kv_count, sizeof(float));
    c->pooled = (float *)calloc(pooled_count, sizeof(float));
    c->group_pos = (uint32_t *)calloc(config->max_sequences * groups_cap,
                                      sizeof(uint32_t));
    c->score = (float *)calloc(groups_cap, sizeof(float));
    c->member = (uint32_t *)calloc(groups_cap * 4u, sizeof(uint32_t));
    c->chosen = (uint32_t *)calloc(config->block_budget, sizeof(uint32_t));
    c->pool_scratch = (float *)calloc(DS4_Q4E_QSA_COMPRESSION *
                                      config->index_dim, sizeof(float));
    c->rb_pooled = (float *)calloc(groups_cap * config->index_dim,
                                   sizeof(float));
    c->rb_pos = (uint32_t *)calloc(groups_cap, sizeof(uint32_t));
    c->save_pooled = (float *)calloc(groups_cap * config->index_dim,
                                     sizeof(float));
    c->save_pos = (uint32_t *)calloc(groups_cap, sizeof(uint32_t));
    if (!c->norm_weight || !c->cell || !c->live_idx || !c->seq ||
        !c->free_slot || !c->raw_key || !c->key || !c->value ||
        !c->pooled || !c->group_pos || !c->score || !c->member ||
        !c->chosen || !c->pool_scratch || !c->rb_pooled || !c->rb_pos ||
        !c->save_pooled || !c->save_pos) {
        ds4_qwen4exp_qsa_cache_destroy(c);
        return false;
    }
    memcpy(c->norm_weight, config->norm_weight,
           config->index_dim * sizeof(float));
    c->free_count = config->n_slot;
    for (uint32_t s = 0u; s < config->max_sequences; s++) {
        c->seq[s].last_pos = DS4_Q4E_QSA_EMPTY;
    }
    for (size_t i = 0u; i < config->n_slot; i++) {
        c->free_slot[i] = (uint32_t)(config->n_slot - 1u - i);
    }
    c->report.line_bytes = cell_count * (sizeof(ds4_q4e_qsa_cell) +
                                         sizeof(uint32_t));
    c->report.raw_key_bytes = raw_count * sizeof(float);
    c->report.kv_bytes = kv_count * sizeof(float) * 2u;
    c->report.pooled_bytes = pooled_count * sizeof(float) +
                             config->max_sequences * groups_cap *
                             sizeof(uint32_t);
    c->report.workspace_bytes =
        groups_cap * (sizeof(float) + 4u * sizeof(uint32_t)) +
        config->block_budget * sizeof(uint32_t) +
        DS4_Q4E_QSA_COMPRESSION * config->index_dim * sizeof(float) +
        2u * groups_cap * (config->index_dim * sizeof(float) +
                           sizeof(uint32_t));
    c->report.allocated_bytes = c->report.line_bytes +
                                c->report.raw_key_bytes + c->report.kv_bytes +
                                c->report.pooled_bytes +
                                c->report.workspace_bytes +
                                config->index_dim * sizeof(float) +
                                config->n_slot * sizeof(uint32_t) +
                                config->max_sequences *
                                sizeof(ds4_q4e_qsa_seq_state) +
                                sizeof(*c);
    *cache = c;
    return true;
}

void ds4_qwen4exp_qsa_cache_destroy(ds4_qwen4exp_qsa_cache *cache) {
    if (!cache) return;
    free(cache->norm_weight);
    free(cache->cell);
    free(cache->live_idx);
    free(cache->seq);
    free(cache->free_slot);
    free(cache->raw_key);
    free(cache->key);
    free(cache->value);
    free(cache->pooled);
    free(cache->group_pos);
    free(cache->score);
    free(cache->member);
    free(cache->chosen);
    free(cache->pool_scratch);
    free(cache->rb_pooled);
    free(cache->rb_pos);
    free(cache->save_pooled);
    free(cache->save_pos);
    free(cache);
}

static const ds4_q4e_qsa_seq_state *q4e_qsa_seq_const(
        const ds4_qwen4exp_qsa_cache *cache, uint32_t seq_id) {
    if (!cache || seq_id >= cache->config.max_sequences) return NULL;
    return &cache->seq[seq_id];
}

static ds4_q4e_qsa_seq_state *q4e_qsa_seq(ds4_qwen4exp_qsa_cache *cache,
                                          uint32_t seq_id) {
    if (!cache || seq_id >= cache->config.max_sequences) return NULL;
    return &cache->seq[seq_id];
}

static uint32_t q4e_qsa_slot_pop(ds4_qwen4exp_qsa_cache *cache) {
    if (cache->free_count == 0u) return DS4_Q4E_QSA_EMPTY;
    cache->free_count--;
    return cache->free_slot[cache->free_count];
}

static void q4e_qsa_slot_push(ds4_qwen4exp_qsa_cache *cache, uint32_t slot) {
    if (slot == DS4_Q4E_QSA_EMPTY ||
        cache->free_count >= cache->config.n_slot) {
        return;
    }
    cache->free_slot[cache->free_count++] = slot;
}

static void q4e_qsa_clear_seq_derived(ds4_qwen4exp_qsa_cache *cache,
                                      uint32_t seq_id) {
    cache->seq[seq_id].n_group = 0u;
    cache->seq[seq_id].tail_start = 0u;
    cache->seq[seq_id].tail_len = 0u;
    memset(cache->pooled + (size_t)seq_id * cache->groups_cap *
           cache->config.index_dim, 0,
           cache->groups_cap * cache->config.index_dim * sizeof(float));
    memset(cache->group_pos + (size_t)seq_id * cache->groups_cap, 0,
           cache->groups_cap * sizeof(uint32_t));
}

bool ds4_qwen4exp_qsa_reset(ds4_qwen4exp_qsa_cache *cache) {
    size_t cell_count = 0u;
    if (!cache) return false;
    if (!q4e_qsa_mul2(cache->config.max_sequences,
                      cache->config.line_capacity, &cell_count)) {
        return false;
    }
    memset(cache->cell, 0, cell_count * sizeof(ds4_q4e_qsa_cell));
    memset(cache->live_idx, 0, cell_count * sizeof(uint32_t));
    memset(cache->seq, 0,
           cache->config.max_sequences * sizeof(ds4_q4e_qsa_seq_state));
    memset(cache->pooled, 0,
           cache->config.max_sequences * cache->groups_cap *
           cache->config.index_dim * sizeof(float));
    memset(cache->group_pos, 0,
           cache->config.max_sequences * cache->groups_cap *
           sizeof(uint32_t));
    /* Slot payloads are owners too: a reset returns the cache to its creation
     * state exactly, so a reset cache and a fresh cache of the same config
     * digest identically no matter what was appended before. */
    memset(cache->raw_key, 0,
           cache->config.n_slot * cache->config.index_dim * sizeof(float));
    memset(cache->key, 0,
           cache->config.n_slot * cache->config.kv_heads *
           cache->config.kv_head_dim * sizeof(float));
    memset(cache->value, 0,
           cache->config.n_slot * cache->config.kv_heads *
           cache->config.kv_head_dim * sizeof(float));
    cache->free_count = cache->config.n_slot;
    for (uint32_t s = 0u; s < cache->config.max_sequences; s++) {
        cache->seq[s].last_pos = DS4_Q4E_QSA_EMPTY;
    }
    for (size_t i = 0u; i < cache->config.n_slot; i++) {
        cache->free_slot[i] = (uint32_t)(cache->config.n_slot - 1u - i);
    }
    cache->has_pending = 0;
    memset(&cache->pending, 0, sizeof(cache->pending));
    return true;
}

/* Frozen group transformation: mean in F32, zero-centered RMS norm, partial
 * RoPE at the group's first position.  Bit-identical to the Phase-5 oracle
 * q4e_group_key_one with an explicit group position. */
static bool q4e_qsa_form_group(float *out, const float *const raw[4],
                               size_t index_dim, size_t n_rot, float theta,
                               float epsilon, const float *norm_weight,
                               uint32_t group_pos) {
    float pooled[DS4_Q4E_QSA_MAX_DIM];
    float sum_square = 0.0f;
    size_t i;
    size_t half;
    if (!out || !raw || !raw[0] || !raw[1] || !raw[2] || !raw[3] ||
        index_dim == 0u || index_dim > DS4_Q4E_QSA_MAX_DIM) {
        return false;
    }
    for (i = 0u; i < index_dim; i++) {
        float sum = 0.0f;
        for (size_t t = 0u; t < DS4_Q4E_QSA_COMPRESSION; t++) {
            sum += raw[t][i];
        }
        pooled[i] = sum / (float)DS4_Q4E_QSA_COMPRESSION;
        if (!isfinite(pooled[i])) return false;
    }
    for (i = 0u; i < index_dim; i++) {
        sum_square += pooled[i] * pooled[i];
    }
    {
        const float mean_square = sum_square / (float)index_dim;
        const float denominator = mean_square + epsilon;
        float inverse = 0.0f;
        if (!isfinite(sum_square) || !isfinite(denominator) ||
            !(denominator > 0.0f)) {
            return false;
        }
        inverse = 1.0f / sqrtf(denominator);
        if (!isfinite(inverse)) return false;
        for (i = 0u; i < index_dim; i++) {
            pooled[i] *= inverse * (1.0f + norm_weight[i]);
            if (!isfinite(pooled[i])) return false;
        }
    }
    half = n_rot / 2u;
    for (size_t pair = 0u; pair < half; pair++) {
        const float exponent = (2.0f * (float)pair) / (float)n_rot;
        const float frequency = 1.0f / powf(theta, exponent);
        const float angle = (float)group_pos * frequency;
        const float cosine = cosf(angle);
        const float sine = sinf(angle);
        const float first = pooled[pair];
        const float second = pooled[half + pair];
        const float rotated_first = first * cosine - second * sine;
        const float rotated_second = second * cosine + first * sine;
        if (!isfinite(frequency) || !isfinite(angle) || !isfinite(cosine) ||
            !isfinite(sine) || !isfinite(rotated_first) ||
            !isfinite(rotated_second)) {
            return false;
        }
        pooled[pair] = rotated_first;
        pooled[half + pair] = rotated_second;
    }
    for (i = 0u; i < index_dim; i++) out[i] = pooled[i];
    return true;
}

/* score = sum over index heads of ReLU(dot).  The constant 1/sqrt(dim) scale
 * of the Phase-5 dense kernel cannot reorder anything and is left out, as
 * upstream does; only selected IDs are contractual. */
static bool q4e_qsa_score_group(const float *query, size_t index_heads,
                                size_t index_dim, const float *pooled,
                                float *score) {
    float total = 0.0f;
    if (!query || !pooled || index_heads == 0u || index_dim == 0u) {
        return false;
    }
    for (size_t h = 0u; h < index_heads; h++) {
        float dot = 0.0f;
        for (size_t i = 0u; i < index_dim; i++) {
            dot += query[h * index_dim + i] * pooled[i];
        }
        if (!isfinite(dot)) return false;
        if (dot > 0.0f) total += dot;
    }
    if (!isfinite(total)) return false;
    *score = total;
    return true;
}

/* Rebuilds one sequence's derived groups and tail from its published cells.
 * Deterministic, so the result is bit-identical to the incrementally formed
 * state whenever the base state is unchanged. */
static bool q4e_qsa_rebuild_seq(ds4_qwen4exp_qsa_cache *cache,
                                uint32_t seq_id) {
    const ds4_q4e_qsa_seq_state *meta = q4e_qsa_seq_const(cache, seq_id);
    float *pooled = cache->pooled +
                    (size_t)seq_id * cache->groups_cap *
                    cache->config.index_dim;
    uint32_t *group_pos = cache->group_pos +
                          (size_t)seq_id * cache->groups_cap;
    uint32_t run_start = 0u;
    uint32_t run_len = 0u;
    uint32_t run_cell[4];
    uint32_t tail_start = 0u;
    uint32_t tail_len = 0u;
    uint32_t last_pos = DS4_Q4E_QSA_EMPTY;
    uint32_t n_group = 0u;
    size_t seen = 0u;
    if (!meta) return false;
    memset(cache->rb_pooled, 0,
           cache->groups_cap * cache->config.index_dim * sizeof(float));
    memset(cache->rb_pos, 0, cache->groups_cap * sizeof(uint32_t));
    for (uint32_t j = 0u; j < meta->count; j++) {
        const uint32_t ring = cache->live_idx[(size_t)seq_id *
                                              cache->config.line_capacity + j];
        const ds4_q4e_qsa_cell *c =
            &cache->cell[(size_t)seq_id * cache->config.line_capacity + ring];
        const float *raw[4];
        if (c->position == DS4_Q4E_QSA_EMPTY ||
            c->slot == DS4_Q4E_QSA_EMPTY) {
            return false;
        }
        if (seen == 0u || c->position != last_pos + 1u) {
            /* a new run begins here; any previous remainder is dropped and
             * becomes invisible */
            run_start = c->position;
            run_len = 0u;
        }
        last_pos = c->position;
        run_cell[run_len] = ring;
        run_len++;
        seen++;
        if (run_len == DS4_Q4E_QSA_COMPRESSION) {
            for (size_t t = 0u; t < DS4_Q4E_QSA_COMPRESSION; t++) {
                raw[t] = cache->raw_key +
                         (size_t)cache->cell[(size_t)seq_id *
                                             cache->config.line_capacity +
                                             run_cell[t]].slot *
                         cache->config.index_dim;
            }
            if (n_group >= cache->groups_cap) return false;
            if (!q4e_qsa_form_group(
                        &cache->rb_pooled[(size_t)n_group *
                                          cache->config.index_dim],
                        raw, cache->config.index_dim, cache->config.n_rot,
                        cache->config.theta, cache->config.epsilon,
                        cache->norm_weight, run_start)) {
                return false;
            }
            cache->rb_pos[n_group] = run_start;
            n_group++;
            run_start += DS4_Q4E_QSA_COMPRESSION;
            run_len = 0u;
        }
    }
    tail_start = run_start;
    tail_len = run_len;
    memcpy(pooled, cache->rb_pooled,
           cache->groups_cap * cache->config.index_dim * sizeof(float));
    memcpy(group_pos, cache->rb_pos, cache->groups_cap * sizeof(uint32_t));
    cache->seq[seq_id].n_group = n_group;
    cache->seq[seq_id].tail_start = tail_start;
    cache->seq[seq_id].tail_len = tail_len;
    return true;
}

/* Rolls a completed eviction sub-step back byte for byte. */
static void q4e_qsa_evict_rollback(ds4_qwen4exp_qsa_cache *cache,
                                   uint32_t seq_id, uint32_t evict_ring,
                                   uint32_t evict_pos, uint32_t evict_slot,
                                   uint32_t old_head, uint32_t old_occupied,
                                   uint32_t old_count, uint32_t old_last,
                                   uint32_t old_n_group,
                                   uint32_t old_tail_start,
                                   uint32_t old_tail_len) {
    const size_t dim = cache->config.index_dim;
    q4e_qsa_slot_pop(cache); /* removes the just-pushed evict slot */
    cache->cell[(size_t)seq_id * cache->config.line_capacity +
                evict_ring].position = evict_pos;
    cache->cell[(size_t)seq_id * cache->config.line_capacity +
                evict_ring].slot = evict_slot;
    memmove(cache->live_idx + (size_t)seq_id * cache->config.line_capacity + 1u,
            cache->live_idx + (size_t)seq_id * cache->config.line_capacity,
            (size_t)(old_count - 1u) * sizeof(uint32_t));
    cache->live_idx[(size_t)seq_id * cache->config.line_capacity] = evict_ring;
    cache->seq[seq_id].head = old_head;
    cache->seq[seq_id].occupied = old_occupied;
    cache->seq[seq_id].count = old_count;
    cache->seq[seq_id].last_pos = old_last;
    cache->seq[seq_id].n_group = old_n_group;
    cache->seq[seq_id].tail_start = old_tail_start;
    cache->seq[seq_id].tail_len = old_tail_len;
    memcpy(cache->pooled + (size_t)seq_id * cache->groups_cap * dim,
           cache->save_pooled, cache->groups_cap * dim * sizeof(float));
    memcpy(cache->group_pos + (size_t)seq_id * cache->groups_cap,
           cache->save_pos, cache->groups_cap * sizeof(uint32_t));
}

/* A reservation that reaches commit and then fails validation is void: the
 * transaction aborted, nothing was published, and the caller may reserve the
 * same position again without an explicit cancel.  Only a reservation that
 * does not match the pending one leaves the pending state alone. */
static void q4e_qsa_pending_clear(ds4_qwen4exp_qsa_cache *cache) {
    cache->has_pending = 0;
    memset(&cache->pending, 0, sizeof(cache->pending));
}

bool ds4_qwen4exp_qsa_append_reserve(ds4_qwen4exp_qsa_cache *cache,
                                     uint32_t seq_id, uint32_t position,
                                     ds4_qwen4exp_qsa_reservation *reservation) {
    const ds4_q4e_qsa_seq_state *meta = q4e_qsa_seq_const(cache, seq_id);
    if (!cache || !reservation || !meta || cache->has_pending ||
        position >= cache->config.context) {
        if (cache) cache->metrics.rejection_count++;
        return false;
    }
    /* The first live position of a sequence is arbitrary (left padding); every
     * later append must be exactly the frontier's successor.  A gap would
     * build a sequence the planner has to reject anyway, so it is refused at
     * the point where the caller can still react. */
    if (meta->count != 0u && position != meta->last_pos + 1u) {
        cache->metrics.rejection_count++;
        return false;
    }
    if (cache->free_count == 0u &&
        !(meta->occupied == cache->config.line_capacity &&
          meta->count != 0u)) {
        /* no free slot and no evictable oldest cell */
        cache->metrics.rejection_count++;
        return false;
    }
    reservation->seq = seq_id;
    reservation->position = position;
    reservation->slot = DS4_Q4E_QSA_EMPTY; /* chosen at commit */
    reservation->line_index = 0u;
    cache->pending = *reservation;
    cache->has_pending = 1;
    return true;
}

void ds4_qwen4exp_qsa_append_cancel(ds4_qwen4exp_qsa_cache *cache) {
    if (!cache) return;
    cache->has_pending = 0;
    memset(&cache->pending, 0, sizeof(cache->pending));
}

bool ds4_qwen4exp_qsa_append_commit(
        ds4_qwen4exp_qsa_cache *cache,
        const ds4_qwen4exp_qsa_reservation *reservation,
        const float *key, const float *value, const float *raw_key) {
    ds4_q4e_qsa_seq_state *meta = NULL;
    const size_t dim = cache ? cache->config.index_dim : 0u;
    const size_t kv_dim = cache ? cache->config.kv_heads *
                                cache->config.kv_head_dim : 0u;
    uint32_t slot = DS4_Q4E_QSA_EMPTY;
    uint32_t evict_ring = 0u;
    uint32_t evict_pos = 0u;
    uint32_t evict_slot = 0u;
    uint32_t old_head = 0u;
    uint32_t old_occupied = 0u;
    uint32_t old_count = 0u;
    uint32_t old_last = 0u;
    uint32_t old_tail_start = 0u;
    uint32_t old_tail_len = 0u;
    uint32_t old_n_group = 0u;
    int evict = 0;
    int cont = 0;
    int formed = 0;
    float formed_key[DS4_Q4E_QSA_MAX_DIM];
    uint32_t new_tail_start = 0u;
    uint32_t new_tail_len = 0u;
    uint32_t new_n_group = 0u;
    if (!cache || !reservation || !cache->has_pending ||
        reservation->seq != cache->pending.seq ||
        reservation->position != cache->pending.position) {
        if (cache) cache->metrics.rejection_count++;
        return false;
    }
    meta = q4e_qsa_seq(cache, reservation->seq);
    if (!meta || reservation->position >= cache->config.context ||
        !q4e_qsa_finite(key, kv_dim) || !q4e_qsa_finite(value, kv_dim) ||
        !q4e_qsa_finite(raw_key, dim)) {
        q4e_qsa_pending_clear(cache);
        cache->metrics.rejection_count++;
        return false;
    }
    old_head = meta->head;
    old_occupied = meta->occupied;
    old_count = meta->count;
    old_last = meta->last_pos;
    old_tail_start = meta->tail_start;
    old_tail_len = meta->tail_len;
    old_n_group = meta->n_group;
    evict = meta->occupied == cache->config.line_capacity;
    if (evict) {
        /* transactional eviction sub-step: save the derived region, drop the
         * oldest cell, rebuild; roll everything back on any failure */
        evict_ring = meta->head;
        {
            const ds4_q4e_qsa_cell *oldest =
                &cache->cell[(size_t)reservation->seq *
                             cache->config.line_capacity + evict_ring];
            if (oldest->position == DS4_Q4E_QSA_EMPTY ||
                oldest->slot == DS4_Q4E_QSA_EMPTY) {
                q4e_qsa_pending_clear(cache);
                cache->metrics.rejection_count++;
                return false;
            }
            evict_pos = oldest->position;
            evict_slot = oldest->slot;
        }
        memcpy(cache->save_pooled,
               cache->pooled + (size_t)reservation->seq * cache->groups_cap *
               dim, cache->groups_cap * dim * sizeof(float));
        memcpy(cache->save_pos,
               cache->group_pos + (size_t)reservation->seq * cache->groups_cap,
               cache->groups_cap * sizeof(uint32_t));
        cache->cell[(size_t)reservation->seq * cache->config.line_capacity +
                    evict_ring].position = DS4_Q4E_QSA_EMPTY;
        cache->cell[(size_t)reservation->seq * cache->config.line_capacity +
                    evict_ring].slot = DS4_Q4E_QSA_EMPTY;
        meta->head = (meta->head + 1u) % cache->config.line_capacity;
        meta->occupied--;
        /* the evicted cell was the oldest, i.e. live_idx[0]; the remaining
         * live entries shift left so live_idx[0..count) stays position-ordered */
        meta->count--;
        memmove(cache->live_idx +
                (size_t)reservation->seq * cache->config.line_capacity,
                cache->live_idx +
                (size_t)reservation->seq * cache->config.line_capacity + 1u,
                (size_t)meta->count * sizeof(uint32_t));
        q4e_qsa_slot_push(cache, evict_slot);
        if (!q4e_qsa_rebuild_seq(cache, reservation->seq)) {
            /* roll the eviction back byte for byte */
            q4e_qsa_evict_rollback(cache, reservation->seq, evict_ring,
                                   evict_pos, evict_slot, old_head,
                                   old_occupied, old_count, old_last,
                                   old_n_group, old_tail_start, old_tail_len);
            q4e_qsa_pending_clear(cache);
            cache->metrics.rejection_count++;
            return false;
        }
    }
    /* continuation decision against the post-eviction tail */
    cont = meta->count != 0u &&
           reservation->position == meta->tail_start + meta->tail_len;
    new_tail_start = meta->tail_start;
    new_tail_len = meta->tail_len;
    new_n_group = meta->n_group;
    formed = 0;
    if (cont && new_tail_len + 1u == DS4_Q4E_QSA_COMPRESSION) {
        /* the group's members are the last three live cells plus this one */
        const float *raw[4];
        if (meta->count < DS4_Q4E_QSA_COMPRESSION - 1u) {
            if (evict) {
                q4e_qsa_evict_rollback(cache, reservation->seq, evict_ring,
                                       evict_pos, evict_slot, old_head,
                                       old_occupied, old_count, old_last,
                                       old_n_group, old_tail_start,
                                       old_tail_len);
            }
            q4e_qsa_pending_clear(cache);
            cache->metrics.rejection_count++;
            return false;
        }
        for (size_t t = 0u; t < DS4_Q4E_QSA_COMPRESSION - 1u; t++) {
            const size_t idx = (size_t)meta->count -
                               (DS4_Q4E_QSA_COMPRESSION - 1u) + t;
            const uint32_t ring = cache->live_idx[
                (size_t)reservation->seq * cache->config.line_capacity + idx];
            raw[t] = cache->raw_key +
                     (size_t)cache->cell[(size_t)reservation->seq *
                                         cache->config.line_capacity +
                                         ring].slot * dim;
        }
        raw[DS4_Q4E_QSA_COMPRESSION - 1u] = raw_key;
        if (new_n_group >= cache->groups_cap ||
            !q4e_qsa_form_group(formed_key, raw, dim, cache->config.n_rot,
                                cache->config.theta, cache->config.epsilon,
                                cache->norm_weight, new_tail_start)) {
            if (evict) {
                /* roll the eviction sub-step back */
                q4e_qsa_evict_rollback(cache, reservation->seq, evict_ring,
                                       evict_pos, evict_slot, old_head,
                                       old_occupied, old_count, old_last,
                                       old_n_group, old_tail_start,
                                       old_tail_len);
            }
            q4e_qsa_pending_clear(cache);
            cache->metrics.rejection_count++;
            return false;
        }
        formed = 1;
        new_n_group++;
        new_tail_start += DS4_Q4E_QSA_COMPRESSION;
        new_tail_len = 0u;
    } else if (cont) {
        new_tail_len++;
    } else {
        /* a gap starts a new run; the previous remainder becomes invisible */
        new_tail_start = reservation->position;
        new_tail_len = 1u;
    }
    slot = q4e_qsa_slot_pop(cache);
    if (slot == DS4_Q4E_QSA_EMPTY) {
        if (evict) {
            q4e_qsa_evict_rollback(cache, reservation->seq, evict_ring,
                                   evict_pos, evict_slot, old_head,
                                   old_occupied, old_count, old_last,
                                   old_n_group, old_tail_start,
                                   old_tail_len);
        }
        q4e_qsa_pending_clear(cache);
        cache->metrics.rejection_count++;
        return false;
    }
    /* publish: every store below happens only after all validation */
    memcpy(&cache->raw_key[(size_t)slot * dim], raw_key, dim * sizeof(float));
    memcpy(&cache->key[(size_t)slot * kv_dim], key, kv_dim * sizeof(float));
    memcpy(&cache->value[(size_t)slot * kv_dim], value,
           kv_dim * sizeof(float));
    if (formed) {
        memcpy(cache->pooled +
               ((size_t)reservation->seq * cache->groups_cap +
                (size_t)meta->n_group) * dim,
               formed_key, dim * sizeof(float));
        cache->group_pos[(size_t)reservation->seq * cache->groups_cap +
                         meta->n_group] = new_tail_start -
            DS4_Q4E_QSA_COMPRESSION;
    }
    {
        const uint32_t ring = (meta->head + meta->occupied) %
                              cache->config.line_capacity;
        cache->cell[(size_t)reservation->seq * cache->config.line_capacity +
                    ring].position = reservation->position;
        cache->cell[(size_t)reservation->seq * cache->config.line_capacity +
                    ring].slot = slot;
        meta->occupied++;
        meta->count++;
        meta->last_pos = reservation->position;
        cache->live_idx[(size_t)reservation->seq *
                        cache->config.line_capacity + meta->count - 1u] = ring;
    }
    meta->n_group = new_n_group;
    meta->tail_start = new_tail_start;
    meta->tail_len = new_tail_len;
    cache->metrics.append_calls++;
    q4e_qsa_pending_clear(cache);
    return true;
}

bool ds4_qwen4exp_qsa_rewind(ds4_qwen4exp_qsa_cache *cache, uint32_t seq_id,
                             uint32_t keep_through) {
    ds4_q4e_qsa_seq_state *meta = q4e_qsa_seq(cache, seq_id);
    if (!cache || !meta || cache->has_pending) return false;
    /* Rewinding to a position the sequence never reached is a caller error,
     * not a no-op: an empty sequence and a keep_through beyond the frontier
     * both fail closed rather than silently succeeding. */
    if (meta->count == 0u || meta->last_pos == DS4_Q4E_QSA_EMPTY ||
        keep_through > meta->last_pos) {
        cache->metrics.rejection_count++;
        return false;
    }
    while (meta->occupied > 0u) {
        const uint32_t ring = (meta->head + meta->occupied - 1u) %
                              cache->config.line_capacity;
        ds4_q4e_qsa_cell *c = &cache->cell[(size_t)seq_id *
                                           cache->config.line_capacity + ring];
        if (c->position == DS4_Q4E_QSA_EMPTY) {
            meta->occupied--;
            continue;
        }
        if (c->position <= keep_through) break;
        q4e_qsa_slot_push(cache, c->slot);
        c->position = DS4_Q4E_QSA_EMPTY;
        c->slot = DS4_Q4E_QSA_EMPTY;
        meta->count--;
        meta->occupied--;
    }
    if (meta->count == 0u) {
        meta->head = 0u;
        meta->occupied = 0u;
        meta->last_pos = DS4_Q4E_QSA_EMPTY;
        q4e_qsa_clear_seq_derived(cache, seq_id);
        return true;
    }
    meta->last_pos =
        cache->cell[(size_t)seq_id * cache->config.line_capacity +
                    cache->live_idx[(size_t)seq_id *
                                    cache->config.line_capacity +
                                    meta->count - 1u]].position;
    return q4e_qsa_rebuild_seq(cache, seq_id);
}

bool ds4_qwen4exp_qsa_remove(ds4_qwen4exp_qsa_cache *cache, uint32_t seq_id,
                             uint32_t position) {
    ds4_q4e_qsa_seq_state *meta = q4e_qsa_seq(cache, seq_id);
    size_t low = 0u;
    size_t high = 0u;
    size_t found = SIZE_MAX;
    if (!cache || !meta || cache->has_pending || meta->count == 0u) {
        return false;
    }
    low = 0u;
    high = meta->count;
    while (low < high) {
        const size_t mid = low + (high - low) / 2u;
        const uint32_t ring = cache->live_idx[(size_t)seq_id *
                                              cache->config.line_capacity +
                                              mid];
        const uint32_t p = cache->cell[(size_t)seq_id *
                                       cache->config.line_capacity + ring]
                           .position;
        if (p == position) {
            found = mid;
            break;
        }
        if (p < position) {
            low = mid + 1u;
        } else {
            high = mid;
        }
    }
    if (found == SIZE_MAX) return false;
    {
        const uint32_t ring = cache->live_idx[(size_t)seq_id *
                                              cache->config.line_capacity +
                                              found];
        ds4_q4e_qsa_cell *c = &cache->cell[(size_t)seq_id *
                                           cache->config.line_capacity +
                                           ring];
        q4e_qsa_slot_push(cache, c->slot);
        c->position = DS4_Q4E_QSA_EMPTY;
        c->slot = DS4_Q4E_QSA_EMPTY;
    }
    for (size_t i = found; i + 1u < meta->count; i++) {
        cache->live_idx[(size_t)seq_id * cache->config.line_capacity + i] =
            cache->live_idx[(size_t)seq_id * cache->config.line_capacity +
                            i + 1u];
    }
    meta->count--;
    if (meta->count == 0u) {
        meta->last_pos = DS4_Q4E_QSA_EMPTY;
        q4e_qsa_clear_seq_derived(cache, seq_id);
        return true;
    }
    meta->last_pos =
        cache->cell[(size_t)seq_id * cache->config.line_capacity +
                    cache->live_idx[(size_t)seq_id *
                                    cache->config.line_capacity +
                                    meta->count - 1u]].position;
    return q4e_qsa_rebuild_seq(cache, seq_id);
}

bool ds4_qwen4exp_qsa_shift(ds4_qwen4exp_qsa_cache *cache, uint32_t seq_id,
                            uint32_t delta) {
    ds4_q4e_qsa_seq_state *meta = q4e_qsa_seq(cache, seq_id);
    if (!cache || !meta || cache->has_pending) return false;
    /* validate before any mutation */
    for (uint32_t j = 0u; j < meta->count; j++) {
        const uint32_t ring = cache->live_idx[(size_t)seq_id *
                                              cache->config.line_capacity + j];
        const uint32_t p = cache->cell[(size_t)seq_id *
                                       cache->config.line_capacity + ring]
                           .position;
        if (p == DS4_Q4E_QSA_EMPTY || p < delta) return false;
    }
    for (uint32_t j = 0u; j < meta->count; j++) {
        const uint32_t ring = cache->live_idx[(size_t)seq_id *
                                              cache->config.line_capacity + j];
        cache->cell[(size_t)seq_id * cache->config.line_capacity + ring]
            .position -= delta;
    }
    if (meta->count == 0u) return true;
    meta->last_pos -= delta;
    if (!q4e_qsa_rebuild_seq(cache, seq_id)) {
        /* undo the rebase byte for byte; the group count cannot have grown, so
         * the pre-shift rebuild is guaranteed to succeed again */
        for (uint32_t j = 0u; j < meta->count; j++) {
            const uint32_t ring = cache->live_idx[
                (size_t)seq_id * cache->config.line_capacity + j];
            cache->cell[(size_t)seq_id * cache->config.line_capacity + ring]
                .position += delta;
        }
        meta->last_pos += delta;
        (void)q4e_qsa_rebuild_seq(cache, seq_id);
        cache->metrics.rejection_count++;
        return false;
    }
    return true;
}

bool ds4_qwen4exp_qsa_fork(ds4_qwen4exp_qsa_cache *cache, uint32_t src,
                           uint32_t dst) {
    ds4_q4e_qsa_seq_state *dst_meta = NULL;
    const ds4_q4e_qsa_seq_state *src_meta = NULL;
    uint32_t *popped = NULL;
    size_t popped_count = 0u;
    const size_t dim = cache ? cache->config.index_dim : 0u;
    const size_t kv_dim = cache ? cache->config.kv_heads *
                                cache->config.kv_head_dim : 0u;
    if (!cache || src >= cache->config.max_sequences ||
        dst >= cache->config.max_sequences || src == dst ||
        cache->has_pending) {
        return false;
    }
    src_meta = &cache->seq[src];
    dst_meta = &cache->seq[dst];
    if (src_meta->count == 0u || dst_meta->count != 0u ||
        cache->free_count < src_meta->count) {
        return false;
    }
    /* The free-count precondition makes a mid-loop pop failure impossible;
     * the scratch only exists to unwind transactionally if that invariant
     * were ever violated. */
    popped = (uint32_t *)malloc(src_meta->count * sizeof(uint32_t));
    if (!popped) return false;
    for (uint32_t j = 0u; j < src_meta->count; j++) {
        const uint32_t src_ring = cache->live_idx[(size_t)src *
                                                  cache->config.line_capacity +
                                                  j];
        const ds4_q4e_qsa_cell *sc = &cache->cell[(size_t)src *
                                                  cache->config.line_capacity +
                                                  src_ring];
        const uint32_t new_slot = q4e_qsa_slot_pop(cache);
        if (new_slot == DS4_Q4E_QSA_EMPTY) {
            for (size_t i = 0u; i < popped_count; i++) {
                q4e_qsa_slot_push(cache, popped[i]);
            }
            free(popped);
            return false;
        }
        popped[popped_count++] = new_slot;
        memcpy(&cache->raw_key[(size_t)new_slot * dim],
               &cache->raw_key[(size_t)sc->slot * dim], dim * sizeof(float));
        memcpy(&cache->key[(size_t)new_slot * kv_dim],
               &cache->key[(size_t)sc->slot * kv_dim], kv_dim * sizeof(float));
        memcpy(&cache->value[(size_t)new_slot * kv_dim],
               &cache->value[(size_t)sc->slot * kv_dim],
               kv_dim * sizeof(float));
        cache->cell[(size_t)dst * cache->config.line_capacity + src_ring]
            .position = sc->position;
        cache->cell[(size_t)dst * cache->config.line_capacity + src_ring]
            .slot = new_slot;
        cache->live_idx[(size_t)dst * cache->config.line_capacity + j] =
            src_ring;
    }
    dst_meta->count = src_meta->count;
    dst_meta->occupied = src_meta->occupied;
    dst_meta->head = src_meta->head;
    dst_meta->last_pos = src_meta->last_pos;
    /* derived groups are a deterministic function of the copied raw keys */
    memcpy(&cache->pooled[(size_t)dst * cache->groups_cap * dim],
           &cache->pooled[(size_t)src * cache->groups_cap * dim],
           cache->groups_cap * dim * sizeof(float));
    memcpy(&cache->group_pos[(size_t)dst * cache->groups_cap],
           &cache->group_pos[(size_t)src * cache->groups_cap],
           cache->groups_cap * sizeof(uint32_t));
    dst_meta->n_group = src_meta->n_group;
    dst_meta->tail_start = src_meta->tail_start;
    dst_meta->tail_len = src_meta->tail_len;
    free(popped);
    return true;
}

bool ds4_qwen4exp_qsa_cache_copy(const ds4_qwen4exp_qsa_cache *cache,
                                 ds4_qwen4exp_qsa_cache **copy) {
    ds4_qwen4exp_qsa_cache *dst = NULL;
    const size_t dim = cache ? cache->config.index_dim : 0u;
    const size_t kv_dim = cache ? cache->config.kv_heads *
                                cache->config.kv_head_dim : 0u;
    if (!cache || !copy ||
        !ds4_qwen4exp_qsa_cache_create(&dst, &cache->config)) {
        return false;
    }
    for (uint32_t s = 0u; s < cache->config.max_sequences; s++) {
        const ds4_q4e_qsa_seq_state *src = &cache->seq[s];
        if (src->count == 0u) continue;
        for (uint32_t j = 0u; j < src->count; j++) {
            const uint32_t ring = cache->live_idx[(size_t)s *
                                                  cache->config.line_capacity +
                                                  j];
            const ds4_q4e_qsa_cell *sc = &cache->cell[(size_t)s *
                                                      cache->config.
                                                      line_capacity + ring];
            const uint32_t new_slot = q4e_qsa_slot_pop(dst);
            if (new_slot == DS4_Q4E_QSA_EMPTY) {
                ds4_qwen4exp_qsa_cache_destroy(dst);
                return false;
            }
            memcpy(&dst->raw_key[(size_t)new_slot * dim],
                   &cache->raw_key[(size_t)sc->slot * dim],
                   dim * sizeof(float));
            memcpy(&dst->key[(size_t)new_slot * kv_dim],
                   &cache->key[(size_t)sc->slot * kv_dim],
                   kv_dim * sizeof(float));
            memcpy(&dst->value[(size_t)new_slot * kv_dim],
                   &cache->value[(size_t)sc->slot * kv_dim],
                   kv_dim * sizeof(float));
            dst->cell[(size_t)s * dst->config.line_capacity + ring] = *sc;
            dst->cell[(size_t)s * dst->config.line_capacity + ring].slot =
                new_slot;
            dst->live_idx[(size_t)s * dst->config.line_capacity + j] = ring;
        }
        dst->seq[s] = *src;
        memcpy(&dst->pooled[(size_t)s * dst->groups_cap * dim],
               &cache->pooled[(size_t)s * cache->groups_cap * dim],
               dst->groups_cap * dim * sizeof(float));
        memcpy(&dst->group_pos[(size_t)s * dst->groups_cap],
               &cache->group_pos[(size_t)s * cache->groups_cap],
               dst->groups_cap * sizeof(uint32_t));
    }
    dst->metrics = cache->metrics;
    *copy = dst;
    return true;
}

bool ds4_qwen4exp_qsa_resolve(const ds4_qwen4exp_qsa_cache *cache,
                              uint32_t seq_id, uint32_t position,
                              uint32_t *slot) {
    const ds4_q4e_qsa_seq_state *meta = q4e_qsa_seq_const(cache, seq_id);
    size_t low = 0u;
    size_t high = 0u;
    if (!cache || !meta || !slot || meta->count == 0u) return false;
    low = 0u;
    high = meta->count;
    while (low < high) {
        const size_t mid = low + (high - low) / 2u;
        const uint32_t ring = cache->live_idx[(size_t)seq_id *
                                              cache->config.line_capacity +
                                              mid];
        const ds4_q4e_qsa_cell *c = &cache->cell[(size_t)seq_id *
                                                 cache->config.line_capacity +
                                                 ring];
        if (c->position == position) {
            if (c->slot == DS4_Q4E_QSA_EMPTY) return false;
            *slot = c->slot;
            return true;
        }
        if (c->position < position) {
            low = mid + 1u;
        } else {
            high = mid;
        }
    }
    return false;
}

bool ds4_qwen4exp_qsa_digest(const ds4_qwen4exp_qsa_cache *cache,
                             uint64_t *digest) {
    uint64_t hash = UINT64_C(0xcbf29ce484222325);
    if (!cache || !digest) return false;
    hash = q4e_qsa_digest_size(hash, cache->config.index_dim);
    hash = q4e_qsa_digest_size(hash, cache->config.index_heads);
    hash = q4e_qsa_digest_size(hash, cache->config.attn_heads);
    hash = q4e_qsa_digest_size(hash, cache->config.kv_heads);
    hash = q4e_qsa_digest_size(hash, cache->config.kv_head_dim);
    hash = q4e_qsa_digest_size(hash, cache->config.block_budget);
    hash = q4e_qsa_digest_size(hash, cache->config.context);
    hash = q4e_qsa_digest_size(hash, cache->config.line_capacity);
    hash = q4e_qsa_digest_size(hash, cache->config.n_slot);
    hash = q4e_qsa_digest_size(hash, cache->config.max_sequences);
    hash = q4e_qsa_digest_size(hash, cache->config.n_rot);
    hash = q4e_qsa_digest_bytes(hash, &cache->config.theta, sizeof(float));
    hash = q4e_qsa_digest_bytes(hash, &cache->config.epsilon, sizeof(float));
    hash = q4e_qsa_digest_bytes(hash, cache->norm_weight,
                                cache->config.index_dim * sizeof(float));
    hash = q4e_qsa_digest_bytes(hash, cache->seq,
                                cache->config.max_sequences *
                                sizeof(ds4_q4e_qsa_seq_state));
    hash = q4e_qsa_digest_bytes(
            hash, cache->cell,
            cache->config.max_sequences * cache->config.line_capacity *
            sizeof(ds4_q4e_qsa_cell));
    hash = q4e_qsa_digest_bytes(
            hash, cache->live_idx,
            cache->config.max_sequences * cache->config.line_capacity *
            sizeof(uint32_t));
    hash = q4e_qsa_digest_u32(hash, (uint32_t)cache->free_count);
    /* only the live prefix of the free stack is state; entries beyond
     * free_count are stale scratch and are deliberately excluded so a
     * serialize/restore round trip is digest-identical */
    hash = q4e_qsa_digest_bytes(hash, cache->free_slot,
                                cache->free_count * sizeof(uint32_t));
    hash = q4e_qsa_digest_bytes(hash, cache->raw_key,
                                cache->config.n_slot *
                                cache->config.index_dim * sizeof(float));
    hash = q4e_qsa_digest_bytes(
            hash, cache->key,
            cache->config.n_slot * cache->config.kv_heads *
            cache->config.kv_head_dim * sizeof(float));
    hash = q4e_qsa_digest_bytes(
            hash, cache->value,
            cache->config.n_slot * cache->config.kv_heads *
            cache->config.kv_head_dim * sizeof(float));
    hash = q4e_qsa_digest_bytes(
            hash, cache->pooled,
            cache->config.max_sequences * cache->groups_cap *
            cache->config.index_dim * sizeof(float));
    hash = q4e_qsa_digest_bytes(
            hash, cache->group_pos,
            cache->config.max_sequences * cache->groups_cap *
            sizeof(uint32_t));
    *digest = hash;
    return true;
}

/* Walks the sequence's visible live cells and enumerates complete groups and
 * the visible tail.  Shared by both planner modes; only the pooled-key source
 * differs. */
static bool q4e_qsa_walk(const ds4_qwen4exp_qsa_cache *cache, uint32_t seq_id,
                         uint32_t query_position, uint32_t *group_pos,
                         uint32_t *member, size_t *n_group, uint32_t *tail_cell,
                         size_t *tail_len, size_t *visible) {
    const ds4_q4e_qsa_seq_state *meta = q4e_qsa_seq_const(cache, seq_id);
    uint32_t run_start = 0u;
    uint32_t run_len = 0u;
    uint32_t run_cell[4];
    uint32_t last_pos = DS4_Q4E_QSA_EMPTY;
    size_t groups = 0u;
    size_t seen = 0u;
    if (!cache || !meta || !group_pos || !member || !n_group || !tail_cell ||
        !tail_len || !visible) {
        return false;
    }
    *n_group = 0u;
    *tail_len = 0u;
    *visible = 0u;
    for (uint32_t j = 0u; j < meta->count; j++) {
        const uint32_t ring = cache->live_idx[(size_t)seq_id *
                                              cache->config.line_capacity + j];
        const ds4_q4e_qsa_cell *c = &cache->cell[(size_t)seq_id *
                                                 cache->config.line_capacity +
                                                 ring];
        if (c->position == DS4_Q4E_QSA_EMPTY || c->position > query_position) {
            continue;
        }
        if (seen == 0u) {
            /* the first visible cell anchors the run; a sequence may start at
             * an arbitrary position (left padding) or at the oldest surviving
             * position after eviction */
            run_start = c->position;
            run_len = 0u;
        } else if (c->position != last_pos + 1u) {
            /* A hole inside the visible range has no representation: the
             * remainder before it would be neither a group nor the tail, so it
             * would silently vanish from attention.  Fail closed instead. */
            return false;
        }
        last_pos = c->position;
        run_cell[run_len] = ring;
        run_len++;
        seen++;
        (*visible)++;
        if (run_len == DS4_Q4E_QSA_COMPRESSION) {
            if (groups >= cache->groups_cap) return false;
            group_pos[groups] = run_start;
            for (size_t t = 0u; t < DS4_Q4E_QSA_COMPRESSION; t++) {
                member[groups * 4u + t] = run_cell[t];
            }
            groups++;
            run_start += DS4_Q4E_QSA_COMPRESSION;
            run_len = 0u;
        }
    }
    /* the visible range is one contiguous run, so its remainder is the tail */
    *tail_len = run_len;
    for (uint32_t t = 0u; t < run_len; t++) {
        tail_cell[t] = run_cell[t];
    }
    *n_group = groups;
    return true;
}

bool ds4_qwen4exp_qsa_plan(ds4_qwen4exp_qsa_cache *cache,
                           ds4_qwen4exp_qsa_plan_mode mode, uint32_t seq_id,
                           uint32_t query_position, const float *query,
                           ds4_qwen4exp_qsa_selection *selection) {
    const ds4_q4e_qsa_seq_state *meta = q4e_qsa_seq_const(cache, seq_id);
    uint32_t tail_cell[DS4_Q4E_QSA_MAX_TAIL];
    size_t n_group = 0u;
    size_t tail_len = 0u;
    size_t visible = 0u;
    size_t select_count = 0u;
    size_t width = 0u;
    size_t write = 0u;
    if (!cache || !meta || !query || !selection ||
        (mode != DS4_Q4E_QSA_PLAN_FULL_RAW &&
         mode != DS4_Q4E_QSA_PLAN_INCREMENTAL) ||
        seq_id >= cache->config.max_sequences ||
        query_position >= cache->config.context ||
        !q4e_qsa_finite(query, cache->config.index_heads *
                               cache->config.index_dim)) {
        if (cache) cache->metrics.rejection_count++;
        return false;
    }
    if (!q4e_qsa_walk(cache, seq_id, query_position, cache->rb_pos,
                      cache->member, &n_group, tail_cell, &tail_len,
                      &visible)) {
        cache->metrics.rejection_count++;
        return false;
    }
    if (visible == 0u) {
        cache->metrics.rejection_count++;
        return false;
    }
    /* score every candidate group */
    for (size_t g = 0u; g < n_group; g++) {
        const float *pooled = NULL;
        if (mode == DS4_Q4E_QSA_PLAN_INCREMENTAL) {
            /* map the walk group to the stored group by first position */
            size_t low = 0u;
            size_t high = meta->n_group;
            size_t found = SIZE_MAX;
            const uint32_t *stored = cache->group_pos +
                                     (size_t)seq_id * cache->groups_cap;
            while (low < high) {
                const size_t mid = low + (high - low) / 2u;
                if (stored[mid] == cache->rb_pos[g]) {
                    found = mid;
                    break;
                }
                if (stored[mid] < cache->rb_pos[g]) {
                    low = mid + 1u;
                } else {
                    high = mid;
                }
            }
            if (found == SIZE_MAX) {
                cache->metrics.rejection_count++;
                return false; /* malformed mapping */
            }
            pooled = cache->pooled +
                     ((size_t)seq_id * cache->groups_cap + found) *
                     cache->config.index_dim;
        } else {
            const float *raw[4];
            for (size_t t = 0u; t < DS4_Q4E_QSA_COMPRESSION; t++) {
                const uint32_t ring = cache->member[g * 4u + t];
                const uint32_t slot = cache->cell[(size_t)seq_id *
                    cache->config.line_capacity + ring].slot;
                if (slot == DS4_Q4E_QSA_EMPTY) {
                    cache->metrics.rejection_count++;
                    return false;
                }
                raw[t] = cache->raw_key + (size_t)slot *
                         cache->config.index_dim;
            }
            if (!q4e_qsa_form_group(cache->pool_scratch, raw,
                                    cache->config.index_dim,
                                    cache->config.n_rot, cache->config.theta,
                                    cache->config.epsilon,
                                    cache->norm_weight, cache->rb_pos[g])) {
                cache->metrics.rejection_count++;
                return false;
            }
            pooled = cache->pool_scratch;
        }
        if (!q4e_qsa_score_group(query, cache->config.index_heads,
                                 cache->config.index_dim, pooled,
                                 &cache->score[g])) {
            cache->metrics.rejection_count++;
            return false;
        }
    }
    /* Deterministic bounded selection: the final ranking is the stable order
     * (score descending, group index ascending) truncated to select_count.
     * Groups arrive in ascending index, so a sorted insertion of at most
     * block_budget entries is O(groups * budget) and produces exactly the
     * ranking the exclusion argmax would, without scanning priors per
     * candidate.  Ties keep the earlier (smaller) group first: the walk-back
     * stops at the first entry whose score is not strictly smaller, so an
     * equal score never overtakes the group that arrived earlier. */
    select_count = n_group < cache->config.block_budget
        ? n_group : cache->config.block_budget;
    {
        size_t kept = 0u;
        for (size_t g = 0u; g < n_group; g++) {
            size_t insert;
            size_t shift_from;
            if (kept == cache->config.block_budget) {
                /* full: qualifies only by strictly beating the last entry
                 * (equal score would rank after the earlier group) */
                if (!(cache->score[g] > cache->score[cache->chosen[
                        cache->config.block_budget - 1u]])) {
                    continue;
                }
                /* the last entry is evicted; g still walks back to its rank */
                insert = cache->config.block_budget - 1u;
                shift_from = cache->config.block_budget - 1u;
            } else {
                insert = kept;
                shift_from = kept;
            }
            while (insert > 0u &&
                   cache->score[cache->chosen[insert - 1u]] <
                       cache->score[g]) {
                insert--;
            }
            for (size_t i = shift_from; i > insert; i--) {
                cache->chosen[i] = cache->chosen[i - 1u];
            }
            cache->chosen[insert] = (uint32_t)g;
            if (kept < cache->config.block_budget) kept++;
        }
    }
    /* Emission order is ascending group position, not descending score: the
     * selected set is rank-determined but the sparse row must stay ordered in
     * the sequence's own logical order, so that a fully fitting context is
     * bit-identical to the dense row.  Groups carry ascending positions in
     * ascending index, so sorting the chosen indices ascending is enough. */
    for (size_t i = 1u; i < select_count; i++) {
        const uint32_t value = cache->chosen[i];
        size_t j = i;
        while (j > 0u && cache->chosen[j - 1u] > value) {
            cache->chosen[j] = cache->chosen[j - 1u];
            j--;
        }
        cache->chosen[j] = value;
    }
    width = select_count * DS4_Q4E_QSA_COMPRESSION + tail_len;
    if (width > cache->config.block_budget * DS4_Q4E_QSA_COMPRESSION +
        DS4_Q4E_QSA_MAX_TAIL ||
        width > DS4_Q4E_QSA_MAX_WIDTH) {
        cache->metrics.rejection_count++;
        return false;
    }
    for (size_t slot = 0u; slot < select_count; slot++) {
        const size_t g = cache->chosen[slot];
        for (size_t t = 0u; t < DS4_Q4E_QSA_COMPRESSION; t++) {
            const uint32_t ring = cache->member[g * 4u + t];
            const ds4_q4e_qsa_cell *c = &cache->cell[
                (size_t)seq_id * cache->config.line_capacity + ring];
            selection->entry[write].id = ring;
            selection->entry[write].position = c->position;
            write++;
        }
    }
    for (size_t t = 0u; t < tail_len; t++) {
        const ds4_q4e_qsa_cell *c = &cache->cell[
            (size_t)seq_id * cache->config.line_capacity + tail_cell[t]];
        selection->entry[write].id = tail_cell[t];
        selection->entry[write].position = c->position;
        write++;
    }
    selection->width = write;
    selection->n_group = select_count;
    selection->tail = tail_len;
    cache->metrics.candidate_groups += (uint64_t)n_group;
    cache->metrics.selected_groups += (uint64_t)select_count;
    cache->metrics.tail_tokens += (uint64_t)tail_len;
    if (width > cache->metrics.max_width) {
        cache->metrics.max_width = (uint64_t)width;
    }
    cache->metrics.plan_calls++;
    return true;
}

bool ds4_qwen4exp_qsa_attention(const ds4_qwen4exp_qsa_cache *cache,
                                uint32_t seq_id, const float *query,
                                const ds4_qwen4exp_qsa_selection *selection,
                                float *output) {
    const size_t dim = cache ? cache->config.kv_head_dim : 0u;
    const size_t heads = cache ? cache->config.attn_heads : 0u;
    const size_t kv_heads = cache ? cache->config.kv_heads : 0u;
    const size_t ratio = kv_heads != 0u ? heads / kv_heads : 0u;
    const float scale = dim != 0u ? 1.0f / sqrtf((float)dim) : 0.0f;
    if (!cache || !query || !selection || !output || dim == 0u ||
        heads == 0u || kv_heads == 0u || ratio == 0u ||
        selection->width == 0u || selection->width > DS4_Q4E_QSA_MAX_WIDTH ||
        !q4e_qsa_finite(query, heads * dim)) {
        return false;
    }
    for (size_t h = 0u; h < heads; h++) {
        const size_t kvh = h / ratio;
        const float *q = query + h * dim;
        float maximum = -INFINITY;
        float denominator = 0.0f;
        float acc[DS4_Q4E_QSA_MAX_DIM];
        for (size_t i = 0u; i < dim; i++) acc[i] = 0.0f;
        for (size_t e = 0u; e < selection->width; e++) {
            uint32_t slot = DS4_Q4E_QSA_EMPTY;
            const float *k;
            const float *v;
            float dot = 0.0f;
            float s;
            float weight;
            if (!ds4_qwen4exp_qsa_resolve(cache, seq_id,
                                          selection->entry[e].position,
                                          &slot)) {
                return false;
            }
            if (cache->cell[(size_t)seq_id * cache->config.line_capacity +
                            selection->entry[e].id].slot != slot) {
                return false; /* malformed mapping */
            }
            k = cache->key + ((size_t)slot * kv_heads + kvh) * dim;
            v = cache->value + ((size_t)slot * kv_heads + kvh) * dim;
            if (!q4e_qsa_finite(k, dim) || !q4e_qsa_finite(v, dim)) {
                return false;
            }
            for (size_t i = 0u; i < dim; i++) dot += q[i] * k[i];
            s = dot * scale;
            if (!isfinite(s)) return false;
            if (s > maximum) {
                const float rescale = expf(maximum - s);
                if (!isfinite(rescale)) return false;
                denominator *= rescale;
                for (size_t i = 0u; i < dim; i++) {
                    acc[i] *= rescale;
                    if (!isfinite(acc[i])) return false;
                }
                maximum = s;
            }
            weight = expf(s - maximum);
            if (!isfinite(weight)) return false;
            denominator += weight;
            for (size_t i = 0u; i < dim; i++) {
                acc[i] += weight * v[i];
                if (!isfinite(acc[i])) return false;
            }
            if (!isfinite(denominator)) return false;
        }
        if (!(denominator > 0.0f) || !isfinite(denominator)) return false;
        for (size_t i = 0u; i < dim; i++) {
            output[h * dim + i] = acc[i] / denominator;
            if (!isfinite(output[h * dim + i])) return false;
        }
    }
    return true;
}

const ds4_qwen4exp_qsa_metrics *ds4_qwen4exp_qsa_get_metrics(
        const ds4_qwen4exp_qsa_cache *cache) {
    return cache ? &cache->metrics : NULL;
}

const ds4_qwen4exp_qsa_byte_report *ds4_qwen4exp_qsa_get_byte_report(
        const ds4_qwen4exp_qsa_cache *cache) {
    return cache ? &cache->report : NULL;
}

/* Serialization: fixed layout given the config; the derived arrays are kept
 * zero-filled beyond their live prefix, so the image is deterministic. */
typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t index_dim;
    uint32_t index_heads;
    uint32_t attn_heads;
    uint32_t kv_heads;
    uint32_t kv_head_dim;
    uint32_t block_budget;
    uint32_t context;
    uint32_t line_capacity;
    uint32_t n_slot;
    uint32_t max_sequences;
    uint32_t n_rot;
    uint32_t free_count;
    float theta;
    float epsilon;
    uint32_t checksum; /* FNV-1a over the image with this field read as zero */
    uint32_t reserved0;
} ds4_q4e_qsa_image_header;

/* Every byte of the image is covered, so no single-byte flip can restore into
 * published state.  The checksum field itself is skipped, which makes the
 * value independent of the order in which serialize fills the header. */
static uint32_t q4e_qsa_image_checksum(const uint8_t *image, size_t size) {
    const size_t skip_start = offsetof(ds4_q4e_qsa_image_header, checksum);
    const size_t skip_end = skip_start + sizeof(uint32_t);
    uint32_t hash = UINT32_C(0x811c9dc5);
    size_t i;
    for (i = 0u; i < size; i++) {
        if (i >= skip_start && i < skip_end) continue;
        hash ^= (uint32_t)image[i];
        hash *= UINT32_C(0x01000193);
    }
    return hash;
}

size_t ds4_qwen4exp_qsa_serialized_size(const ds4_qwen4exp_qsa_cache *cache) {
    size_t total = 0u;
    size_t part = 0u;
    if (!cache) return 0u;
    total = sizeof(ds4_q4e_qsa_image_header);
    if (!q4e_qsa_mul2(cache->config.index_dim, sizeof(float), &part)) {
        return 0u;
    }
    total += part; /* norm_weight */
    total += cache->config.max_sequences * sizeof(ds4_q4e_qsa_seq_state);
    total += cache->config.max_sequences * cache->config.line_capacity *
             (sizeof(ds4_q4e_qsa_cell) + sizeof(uint32_t));
    total += cache->config.n_slot * sizeof(uint32_t); /* free list */
    total += cache->config.n_slot * cache->config.index_dim * sizeof(float);
    total += cache->config.n_slot * cache->config.kv_heads *
             cache->config.kv_head_dim * sizeof(float) * 2u;
    total += cache->config.max_sequences * cache->groups_cap *
             cache->config.index_dim * sizeof(float);
    total += cache->config.max_sequences * cache->groups_cap *
             sizeof(uint32_t);
    return total;
}

bool ds4_qwen4exp_qsa_serialize(const ds4_qwen4exp_qsa_cache *cache,
                                void *image, size_t capacity) {
    ds4_q4e_qsa_image_header header;
    uint8_t *cursor = NULL;
    size_t need = 0u;
    if (!cache || !image ||
        !ds4_qwen4exp_qsa_config_validate(&cache->config)) {
        return false;
    }
    need = ds4_qwen4exp_qsa_serialized_size(cache);
    if (need == 0u || capacity < need) return false;
    memset(&header, 0, sizeof(header));
    header.magic = Q4E_QSA_MAGIC;
    header.version = Q4E_QSA_VERSION;
    header.index_dim = (uint32_t)cache->config.index_dim;
    header.index_heads = (uint32_t)cache->config.index_heads;
    header.attn_heads = (uint32_t)cache->config.attn_heads;
    header.kv_heads = (uint32_t)cache->config.kv_heads;
    header.kv_head_dim = (uint32_t)cache->config.kv_head_dim;
    header.block_budget = (uint32_t)cache->config.block_budget;
    header.context = (uint32_t)cache->config.context;
    header.line_capacity = (uint32_t)cache->config.line_capacity;
    header.n_slot = (uint32_t)cache->config.n_slot;
    header.max_sequences = (uint32_t)cache->config.max_sequences;
    header.n_rot = (uint32_t)cache->config.n_rot;
    header.free_count = (uint32_t)cache->free_count;
    header.theta = cache->config.theta;
    header.epsilon = cache->config.epsilon;
    cursor = (uint8_t *)image;
    memcpy(cursor, &header, sizeof(header));
    cursor += sizeof(header);
    memcpy(cursor, cache->norm_weight,
           cache->config.index_dim * sizeof(float));
    cursor += cache->config.index_dim * sizeof(float);
    memcpy(cursor, cache->seq,
           cache->config.max_sequences * sizeof(ds4_q4e_qsa_seq_state));
    cursor += cache->config.max_sequences * sizeof(ds4_q4e_qsa_seq_state);
    memcpy(cursor, cache->cell,
           cache->config.max_sequences * cache->config.line_capacity *
           sizeof(ds4_q4e_qsa_cell));
    cursor += cache->config.max_sequences * cache->config.line_capacity *
              sizeof(ds4_q4e_qsa_cell);
    memcpy(cursor, cache->live_idx,
           cache->config.max_sequences * cache->config.line_capacity *
           sizeof(uint32_t));
    cursor += cache->config.max_sequences * cache->config.line_capacity *
              sizeof(uint32_t);
    memcpy(cursor, cache->free_slot, cache->free_count * sizeof(uint32_t));
    memset(cursor + cache->free_count * sizeof(uint32_t), 0,
           (cache->config.n_slot - cache->free_count) * sizeof(uint32_t));
    cursor += cache->config.n_slot * sizeof(uint32_t);
    memcpy(cursor, cache->raw_key,
           cache->config.n_slot * cache->config.index_dim * sizeof(float));
    cursor += cache->config.n_slot * cache->config.index_dim * sizeof(float);
    memcpy(cursor, cache->key,
           cache->config.n_slot * cache->config.kv_heads *
           cache->config.kv_head_dim * sizeof(float));
    cursor += cache->config.n_slot * cache->config.kv_heads *
              cache->config.kv_head_dim * sizeof(float);
    memcpy(cursor, cache->value,
           cache->config.n_slot * cache->config.kv_heads *
           cache->config.kv_head_dim * sizeof(float));
    cursor += cache->config.n_slot * cache->config.kv_heads *
              cache->config.kv_head_dim * sizeof(float);
    memcpy(cursor, cache->pooled,
           cache->config.max_sequences * cache->groups_cap *
           cache->config.index_dim * sizeof(float));
    cursor += cache->config.max_sequences * cache->groups_cap *
              cache->config.index_dim * sizeof(float);
    memcpy(cursor, cache->group_pos,
           cache->config.max_sequences * cache->groups_cap *
           sizeof(uint32_t));
    ((ds4_q4e_qsa_image_header *)image)->checksum =
        q4e_qsa_image_checksum((const uint8_t *)image, need);
    return true;
}

/* Validates a candidate image fully before it may be published: base-state
 * consistency, slot ownership, finiteness, and a derived-state proof that
 * rebuilds every sequence from the base and compares byte for byte. */
static bool q4e_qsa_image_valid(const ds4_qwen4exp_qsa_cache *c) {
    const size_t dim = c->config.index_dim;
    const size_t kv_dim = c->config.kv_heads * c->config.kv_head_dim;
    uint8_t *used = NULL;
    uint8_t *ring_used = NULL;
    size_t used_count = 0u;
    if (!c) return false;
    used = (uint8_t *)calloc(c->config.n_slot, 1u);
    ring_used = (uint8_t *)calloc(c->config.line_capacity, 1u);
    if (!used || !ring_used) {
        free(used);
        free(ring_used);
        return false;
    }
    for (uint32_t s = 0u; s < c->config.max_sequences; s++) {
        const ds4_q4e_qsa_seq_state *meta = &c->seq[s];
        uint32_t prev = 0u;
        memset(ring_used, 0, c->config.line_capacity);
        if (meta->count > meta->occupied ||
            meta->occupied > c->config.line_capacity ||
            meta->head >= c->config.line_capacity ||
            (meta->count == 0u) != (meta->last_pos == DS4_Q4E_QSA_EMPTY) ||
            meta->n_group > c->groups_cap || meta->tail_len > 3u) {
            free(used);
            free(ring_used);
            return false;
        }
        for (uint32_t j = 0u; j < meta->count; j++) {
            const uint32_t ring = c->live_idx[(size_t)s *
                                              c->config.line_capacity + j];
            const ds4_q4e_qsa_cell *cell;
            const uint32_t ring_delta =
                ring >= meta->head ? ring - meta->head
                                   : ring + (uint32_t)c->config.line_capacity -
                                     meta->head;
            if (ring >= c->config.line_capacity || ring_used[ring] ||
                ring_delta >= meta->occupied) {
                free(used);
                free(ring_used);
                return false;
            }
            ring_used[ring] = 1u;
            cell = &c->cell[(size_t)s * c->config.line_capacity + ring];
            if (cell->position >= c->config.context ||
                cell->position == DS4_Q4E_QSA_EMPTY ||
                cell->slot == DS4_Q4E_QSA_EMPTY ||
                cell->slot >= c->config.n_slot ||
                used[cell->slot]) {
                free(used);
                free(ring_used);
                return false;
            }
            if (j != 0u && cell->position <= prev) {
                free(used);
                free(ring_used);
                return false; /* duplicates or unordered positions */
            }
            prev = cell->position;
            used[cell->slot] = 1u;
            used_count++;
            if (!q4e_qsa_finite(c->raw_key + (size_t)cell->slot * dim, dim) ||
                !q4e_qsa_finite(c->key + (size_t)cell->slot * kv_dim, kv_dim) ||
                !q4e_qsa_finite(c->value + (size_t)cell->slot * kv_dim,
                                kv_dim)) {
                free(used);
                free(ring_used);
                return false;
            }
        }
        if (meta->count != 0u && prev != meta->last_pos) {
            free(used);
            free(ring_used);
            return false;
        }
        for (uint32_t offset = 0u; offset < meta->occupied; offset++) {
            const uint32_t ring = (meta->head + offset) %
                                  (uint32_t)c->config.line_capacity;
            const ds4_q4e_qsa_cell *cell =
                &c->cell[(size_t)s * c->config.line_capacity + ring];
            if (!ring_used[ring] &&
                (cell->position != DS4_Q4E_QSA_EMPTY ||
                 cell->slot != DS4_Q4E_QSA_EMPTY)) {
                free(used);
                free(ring_used);
                return false;
            }
        }
    }
    if (used_count + c->free_count != c->config.n_slot) {
        free(used);
        free(ring_used);
        return false;
    }
    for (size_t i = 0u; i < c->free_count; i++) {
        const uint32_t slot = c->free_slot[i];
        if (slot >= c->config.n_slot || used[slot]) {
            free(used);
            free(ring_used);
            return false;
        }
        used[slot] = 1u;
    }
    free(used);
    free(ring_used);
    /* derived-state proof: rebuild from the base and compare */
    {
        ds4_qwen4exp_qsa_cache *probe = NULL;
        bool ok = true;
        if (!ds4_qwen4exp_qsa_cache_create(&probe, &c->config)) return false;
        memcpy(probe->cell, c->cell,
               c->config.max_sequences * c->config.line_capacity *
               sizeof(ds4_q4e_qsa_cell));
        memcpy(probe->live_idx, c->live_idx,
               c->config.max_sequences * c->config.line_capacity *
               sizeof(uint32_t));
        memcpy(probe->seq, c->seq,
               c->config.max_sequences * sizeof(ds4_q4e_qsa_seq_state));
        memcpy(probe->raw_key, c->raw_key,
               c->config.n_slot * dim * sizeof(float));
        for (uint32_t s = 0u; s < c->config.max_sequences && ok; s++) {
            /* every sequence is proven, including empty ones: a rebuild of an
             * empty sequence yields all-zero derived arrays, so any nonzero
             * corruption there is rejected too */
            probe->seq[s].n_group = 0u;
            probe->seq[s].tail_start = 0u;
            probe->seq[s].tail_len = 0u;
            if (!q4e_qsa_rebuild_seq(probe, s)) {
                ok = false;
                break;
            }
            if (memcmp(probe->pooled + (size_t)s * probe->groups_cap * dim,
                       c->pooled + (size_t)s * probe->groups_cap * dim,
                       probe->groups_cap * dim * sizeof(float)) != 0 ||
                memcmp(probe->group_pos + (size_t)s * probe->groups_cap,
                       c->group_pos + (size_t)s * probe->groups_cap,
                       probe->groups_cap * sizeof(uint32_t)) != 0 ||
                probe->seq[s].n_group != c->seq[s].n_group ||
                probe->seq[s].tail_start != c->seq[s].tail_start ||
                probe->seq[s].tail_len != c->seq[s].tail_len) {
                ok = false;
            }
        }
        ds4_qwen4exp_qsa_cache_destroy(probe);
        return ok;
    }
}

bool ds4_qwen4exp_qsa_restore(ds4_qwen4exp_qsa_cache *cache,
                              const void *image, size_t size) {
    ds4_qwen4exp_qsa_cache *candidate = NULL;
    const ds4_q4e_qsa_image_header *header = NULL;
    const uint8_t *cursor = NULL;
    size_t need = 0u;
    if (!cache || !image || cache->has_pending) return false;
    if (size < sizeof(ds4_q4e_qsa_image_header)) return false;
    header = (const ds4_q4e_qsa_image_header *)image;
    if (header->magic != Q4E_QSA_MAGIC || header->version != Q4E_QSA_VERSION ||
        (size_t)header->index_dim != cache->config.index_dim ||
        (size_t)header->index_heads != cache->config.index_heads ||
        (size_t)header->attn_heads != cache->config.attn_heads ||
        (size_t)header->kv_heads != cache->config.kv_heads ||
        (size_t)header->kv_head_dim != cache->config.kv_head_dim ||
        (size_t)header->block_budget != cache->config.block_budget ||
        (size_t)header->context != cache->config.context ||
        (size_t)header->line_capacity != cache->config.line_capacity ||
        (size_t)header->n_slot != cache->config.n_slot ||
        (size_t)header->max_sequences != cache->config.max_sequences ||
        (size_t)header->n_rot != cache->config.n_rot ||
        header->theta != cache->config.theta ||
        header->epsilon != cache->config.epsilon) {
        return false;
    }
    need = ds4_qwen4exp_qsa_serialized_size(cache);
    if (need == 0u || size != need) return false;
    if (header->checksum !=
        q4e_qsa_image_checksum((const uint8_t *)image, size)) {
        return false;
    }
    if (!ds4_qwen4exp_qsa_cache_create(&candidate, &cache->config)) {
        return false;
    }
    cursor = (const uint8_t *)image + sizeof(ds4_q4e_qsa_image_header);
    memcpy(candidate->norm_weight, cursor,
           cache->config.index_dim * sizeof(float));
    if (!q4e_qsa_finite(candidate->norm_weight, cache->config.index_dim) ||
        memcmp(candidate->norm_weight, cache->norm_weight,
               cache->config.index_dim * sizeof(float)) != 0) {
        ds4_qwen4exp_qsa_cache_destroy(candidate);
        return false;
    }
    cursor += cache->config.index_dim * sizeof(float);
    memcpy(candidate->seq, cursor,
           cache->config.max_sequences * sizeof(ds4_q4e_qsa_seq_state));
    cursor += cache->config.max_sequences * sizeof(ds4_q4e_qsa_seq_state);
    memcpy(candidate->cell, cursor,
           cache->config.max_sequences * cache->config.line_capacity *
           sizeof(ds4_q4e_qsa_cell));
    cursor += cache->config.max_sequences * cache->config.line_capacity *
              sizeof(ds4_q4e_qsa_cell);
    memcpy(candidate->live_idx, cursor,
           cache->config.max_sequences * cache->config.line_capacity *
           sizeof(uint32_t));
    cursor += cache->config.max_sequences * cache->config.line_capacity *
              sizeof(uint32_t);
    candidate->free_count = header->free_count;
    if (candidate->free_count > cache->config.n_slot) {
        ds4_qwen4exp_qsa_cache_destroy(candidate);
        return false;
    }
    memcpy(candidate->free_slot, cursor,
           cache->config.n_slot * sizeof(uint32_t));
    cursor += cache->config.n_slot * sizeof(uint32_t);
    memcpy(candidate->raw_key, cursor,
           cache->config.n_slot * cache->config.index_dim * sizeof(float));
    cursor += cache->config.n_slot * cache->config.index_dim * sizeof(float);
    memcpy(candidate->key, cursor,
           cache->config.n_slot * cache->config.kv_heads *
           cache->config.kv_head_dim * sizeof(float));
    cursor += cache->config.n_slot * cache->config.kv_heads *
              cache->config.kv_head_dim * sizeof(float);
    memcpy(candidate->value, cursor,
           cache->config.n_slot * cache->config.kv_heads *
           cache->config.kv_head_dim * sizeof(float));
    cursor += cache->config.n_slot * cache->config.kv_heads *
              cache->config.kv_head_dim * sizeof(float);
    memcpy(candidate->pooled, cursor,
           cache->config.max_sequences * cache->groups_cap *
           cache->config.index_dim * sizeof(float));
    cursor += cache->config.max_sequences * cache->groups_cap *
              cache->config.index_dim * sizeof(float);
    memcpy(candidate->group_pos, cursor,
           cache->config.max_sequences * cache->groups_cap *
           sizeof(uint32_t));
    if (!q4e_qsa_image_valid(candidate)) {
        ds4_qwen4exp_qsa_cache_destroy(candidate);
        return false;
    }
    /* publish by swapping owners; the old state is destroyed only after the
     * candidate has been fully validated */
    {
        float *tf = candidate->norm_weight;
        ds4_q4e_qsa_cell *tc = candidate->cell;
        uint32_t *tl = candidate->live_idx;
        ds4_q4e_qsa_seq_state *ts = candidate->seq;
        uint32_t *tfs = candidate->free_slot;
        size_t tfc = candidate->free_count;
        float *trk = candidate->raw_key;
        float *tk = candidate->key;
        float *tv = candidate->value;
        float *tp = candidate->pooled;
        uint32_t *tg = candidate->group_pos;
        candidate->norm_weight = cache->norm_weight;
        candidate->cell = cache->cell;
        candidate->live_idx = cache->live_idx;
        candidate->seq = cache->seq;
        candidate->free_slot = cache->free_slot;
        candidate->free_count = cache->free_count;
        candidate->raw_key = cache->raw_key;
        candidate->key = cache->key;
        candidate->value = cache->value;
        candidate->pooled = cache->pooled;
        candidate->group_pos = cache->group_pos;
        cache->norm_weight = tf;
        cache->cell = tc;
        cache->live_idx = tl;
        cache->seq = ts;
        cache->free_slot = tfs;
        cache->free_count = tfc;
        cache->raw_key = trk;
        cache->key = tk;
        cache->value = tv;
        cache->pooled = tp;
        cache->group_pos = tg;
    }
    ds4_qwen4exp_qsa_cache_destroy(candidate);
    return true;
}
