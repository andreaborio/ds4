#ifndef DS4_QWEN4EXP_QSA_H
#define DS4_QWEN4EXP_QSA_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Phase 6: native sparse QSA host state, planners and oracles.
 *
 * This module owns the hybrid raw-index/KV state that mirrors the main KV
 * cache lifecycle, the full-raw and incremental-pooled selection planners,
 * and the reference sparse attention used as the Metal oracle.  It is a
 * model-free correctness module: production admission, runtime selection and
 * public exposure remain fail-closed elsewhere.
 *
 * Frozen semantics (Phase-6 card):
 *   - raw index keys are cached before pooling, normalization or RoPE;
 *   - complete groups are four consecutive valid positions of one sequence,
 *     anchored at the sequence's first real token (left padding begins
 *     grouping there); a group's position is its first token's absolute
 *     position; holes, cross-sequence membership, future positions,
 *     duplicates and malformed mappings are rejected;
 *   - the incomplete visible suffix is a raw tail of zero to three tokens;
 *   - complete groups are mean-pooled in F32, zero-centered-normalized,
 *     partially rotated at the group's first position, and retained exactly
 *     once formed (immutable);
 *   - score is sum over index heads of ReLU(dot); ReLU(sum(dot)) is an
 *     explicit negative control;
 *   - selection is deterministic top-`block_budget` by descending score with
 *     ascending group position as the stable tie-break; the rank decides the
 *     SET, and the row is then emitted in ascending group position, so a
 *     context that fits the budget is byte-identical to the dense row;
 *     each selected group expands to its four group-contiguous logical token
 *     IDs, and the visible raw tail follows;
 *   - total sparse width never exceeds 2051 (512*4 + 3);
 *   - below block_budget + compression - 1 visible tokens every complete
 *     group fits in the budget, so selection is exactly dense;
 *   - the planner never allocates and never forms a dense Q-by-K structure;
 *     all workspace is owned by the cache and sized at creation.
 *
 *   - the visible range of a query is one contiguous run: the first live
 *     position anchors it (left padding, or the oldest survivor after
 *     eviction) and every later live position is its predecessor's successor.
 *     A hole inside that range has no representation - the remainder before it
 *     is neither a group nor the tail - so the planner rejects instead of
 *     silently dropping tokens from attention.  Appends enforce the same
 *     invariant at the point where the caller can still react.
 *
 * Lifecycle sharing: the cache mirrors the main KV cache's logical slot,
 * sequence identity, absolute position, append reservation, removal, reset,
 * copy/fork, rewind, shift and serialization/restore.  Every mutation is
 * transactional: a failed operation leaves the published state byte-identical.
 * Structural operations (remove/rewind/shift/fork/restore/eviction) rebuild
 * the derived pooled state inside the transaction; append-only commits update
 * it incrementally.  Because pooled keys are a deterministic function of the
 * raw keys and their group positions, the full-raw and incremental planners
 * select identical IDs by construction; the tests prove it. */

#define DS4_Q4E_QSA_COMPRESSION 4u
#define DS4_Q4E_QSA_MAX_BUDGET 512u
#define DS4_Q4E_QSA_MAX_TAIL 3u
#define DS4_Q4E_QSA_MAX_WIDTH 2051u /* block_budget*4 + 3 at the frozen budget */
#define DS4_Q4E_QSA_MAX_CONTEXT 262144u
#define DS4_Q4E_QSA_MAX_DIM 256u
#define DS4_Q4E_QSA_MAX_SEQS 64u
#define DS4_Q4E_QSA_EMPTY UINT32_MAX

typedef struct {
    size_t index_dim;      /* raw/pooled index key width, 1..256 */
    size_t index_heads;    /* indexer query heads, 1..256 */
    size_t attn_heads;     /* attention query heads, >= kv_heads */
    size_t kv_heads;       /* attention KV heads (GQA divisor of attn_heads) */
    size_t kv_head_dim;    /* attention head width, 1..256 */
    size_t block_budget;   /* complete groups selected, 1..512 */
    size_t context;        /* absolute position space, 1..262144 */
    size_t line_capacity;  /* per-sequence ring length, 1..context */
    size_t n_slot;         /* shared physical slots, <= lines across sequences */
    size_t max_sequences;  /* sequence identity space, 1..64 */
    size_t n_rot;          /* partial RoPE width, even, <= index_dim */
    float theta;           /* RoPE base, finite > 0 */
    float epsilon;         /* zero-centered norm epsilon, finite > 0 */
    const float *norm_weight; /* [index_dim] zero-centered gammas, copied */
} ds4_qwen4exp_qsa_config;

/* Byte report.  workspace_bytes is the planner scratch owned by the cache;
 * it is O(groups + budget + width) and never O(queries x visible). */
typedef struct {
    size_t line_bytes;
    size_t raw_key_bytes;
    size_t kv_bytes;
    size_t pooled_bytes;
    size_t workspace_bytes;
    size_t allocated_bytes;
} ds4_qwen4exp_qsa_byte_report;

typedef struct {
    uint64_t append_calls;
    uint64_t candidate_groups;
    uint64_t selected_groups;
    uint64_t tail_tokens;
    uint64_t max_width;
    uint64_t plan_calls;
    uint64_t dense_mask_bytes; /* frozen gate: pinned to zero by construction */
    uint64_t rejection_count;
} ds4_qwen4exp_qsa_metrics;

/* One selected token: per-sequence logical ID plus its absolute position. */
typedef struct {
    uint32_t id;       /* per-sequence logical line index */
    uint32_t position; /* absolute position */
} ds4_qwen4exp_qsa_token;

typedef struct {
    ds4_qwen4exp_qsa_token entry[DS4_Q4E_QSA_MAX_WIDTH];
    size_t width;
    size_t n_group; /* complete groups selected */
    size_t tail;    /* visible raw tail tokens, 0..3 */
} ds4_qwen4exp_qsa_selection;

typedef struct ds4_qwen4exp_qsa_cache ds4_qwen4exp_qsa_cache;

/* Reservation handed from append_reserve to append_commit.  A reservation
 * that is never committed publishes nothing; the slot returns to the pool. */
typedef struct {
    uint32_t seq;
    uint32_t position;
    uint32_t slot;
    uint32_t line_index;
} ds4_qwen4exp_qsa_reservation;

typedef enum {
    DS4_Q4E_QSA_PLAN_FULL_RAW = 0,
    DS4_Q4E_QSA_PLAN_INCREMENTAL = 1
} ds4_qwen4exp_qsa_plan_mode;

bool ds4_qwen4exp_qsa_config_validate(const ds4_qwen4exp_qsa_config *config);

bool ds4_qwen4exp_qsa_cache_create(ds4_qwen4exp_qsa_cache **cache,
                                   const ds4_qwen4exp_qsa_config *config);
void ds4_qwen4exp_qsa_cache_destroy(ds4_qwen4exp_qsa_cache *cache);

/* Discards every sequence, line, slot and derived group atomically, including
 * the slot payload arrays: a reset cache and a fresh cache of the same config
 * are digest-identical whatever was appended before. */
bool ds4_qwen4exp_qsa_reset(ds4_qwen4exp_qsa_cache *cache);

/* The first live position of a sequence may be arbitrary (left padding);
 * every later append must be exactly the frontier's successor, so a sequence
 * can never be built into a shape the planner has to reject.  The reservation
 * pins the physical slot; commit validates finiteness, forms the pooled group
 * when the append completes a run of four, and publishes once.  A commit that
 * matches the pending reservation and then fails validation voids it: nothing
 * was published and the same position may be reserved again without an
 * explicit cancel.  A commit whose reservation does not match the pending one
 * leaves the pending reservation alone. */
bool ds4_qwen4exp_qsa_append_reserve(ds4_qwen4exp_qsa_cache *cache,
                                     uint32_t seq, uint32_t position,
                                     ds4_qwen4exp_qsa_reservation *reservation);
bool ds4_qwen4exp_qsa_append_commit(
        ds4_qwen4exp_qsa_cache *cache,
        const ds4_qwen4exp_qsa_reservation *reservation,
        const float *key,      /* [kv_heads][kv_head_dim] */
        const float *value,    /* [kv_heads][kv_head_dim] */
        const float *raw_key); /* [index_dim] */

/* Abandons a pending reservation; nothing was published and no slot was
 * consumed, so the cache is unchanged. */
void ds4_qwen4exp_qsa_append_cancel(ds4_qwen4exp_qsa_cache *cache);

/* Removes one live (seq, position) entry, leaving a hole; the physical slot
 * returns to the pool and is reused by the next append. */
bool ds4_qwen4exp_qsa_remove(ds4_qwen4exp_qsa_cache *cache, uint32_t seq,
                             uint32_t position);

/* Drops every live token of seq strictly after keep_through.  Rewinding an
 * empty sequence, or to a position beyond the frontier, is a caller error and
 * fails closed rather than succeeding as a no-op. */
bool ds4_qwen4exp_qsa_rewind(ds4_qwen4exp_qsa_cache *cache, uint32_t seq,
                             uint32_t keep_through);

/* Subtracts delta from every live position of seq; each must be >= delta.
 * Relative order and slots are preserved; derived groups are rebuilt. */
bool ds4_qwen4exp_qsa_shift(ds4_qwen4exp_qsa_cache *cache, uint32_t seq,
                            uint32_t delta);

/* Copies src's whole live state (line, KV, raw keys, derived groups) into
 * dst, which must currently be empty.  Transactional. */
bool ds4_qwen4exp_qsa_fork(ds4_qwen4exp_qsa_cache *cache, uint32_t src,
                           uint32_t dst);

/* Deep-copies the entire cache.  On failure *copy is untouched. */
bool ds4_qwen4exp_qsa_cache_copy(const ds4_qwen4exp_qsa_cache *cache,
                                 ds4_qwen4exp_qsa_cache **copy);

/* Serializes the full state into caller-provided storage.  The image carries
 * a checksum over every one of its bytes, so no single-byte corruption can
 * reach published state through restore. */
size_t ds4_qwen4exp_qsa_serialized_size(const ds4_qwen4exp_qsa_cache *cache);
bool ds4_qwen4exp_qsa_serialize(const ds4_qwen4exp_qsa_cache *cache,
                                void *image, size_t capacity);
/* Restores atomically: a rejected image leaves the cache byte-identical. */
bool ds4_qwen4exp_qsa_restore(ds4_qwen4exp_qsa_cache *cache,
                              const void *image, size_t size);

/* Digest of every published owner; equal digests imply byte-identical state. */
bool ds4_qwen4exp_qsa_digest(const ds4_qwen4exp_qsa_cache *cache,
                             uint64_t *digest);

/* Slot/position resolution shared with the Metal path. */
bool ds4_qwen4exp_qsa_resolve(const ds4_qwen4exp_qsa_cache *cache, uint32_t seq,
                              uint32_t position, uint32_t *slot);

/* Planners.  query is [index_heads][index_dim], already normalized and
 * rotated by the caller at the query's own position.  Both modes return
 * identical selections; mode only chooses the pooled-key source. */
bool ds4_qwen4exp_qsa_plan(ds4_qwen4exp_qsa_cache *cache,
                           ds4_qwen4exp_qsa_plan_mode mode, uint32_t seq,
                           uint32_t query_position, const float *query,
                           ds4_qwen4exp_qsa_selection *selection);

/* Reference sparse attention over one query token's selected IDs.  query is
 * [attn_heads][kv_head_dim] for a single query token; output is
 * [attn_heads][kv_head_dim].  Gathers only selected K/V through the slot
 * mapping and never forms a dense Q-by-K structure.  Multi-query prefill
 * loops this per query token, exactly like the Metal grid. */
bool ds4_qwen4exp_qsa_attention(const ds4_qwen4exp_qsa_cache *cache,
                                uint32_t seq, const float *query,
                                const ds4_qwen4exp_qsa_selection *selection,
                                float *output);

const ds4_qwen4exp_qsa_metrics *ds4_qwen4exp_qsa_get_metrics(
        const ds4_qwen4exp_qsa_cache *cache);

const ds4_qwen4exp_qsa_byte_report *ds4_qwen4exp_qsa_get_byte_report(
        const ds4_qwen4exp_qsa_cache *cache);

#endif
