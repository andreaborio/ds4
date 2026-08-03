/* DeepSeek V4 Flash DSpark graph/cache contract.
 *
 * This textual partition is owned by ds4.c.  It contains no execution toggle:
 * the engine remains fail-closed until the three-stage drafter and exact
 * verifier are complete.  The contract fixes the seams that those stages must
 * consume without adding allocations or dispatches to target-only sessions.
 */

enum {
    DS4_DSPARK_CAPTURE_STAGE_COUNT = 3,
    DS4_DSPARK_CAPTURE_WIDTH = 4096,
    DS4_DSPARK_HISTORY_ROWS = 128,
    DS4_DSPARK_TARGET_LAYER_COUNT = 43,
    DS4_DSPARK_TARGET_TOP_K = 6,
    DS4_DSPARK_SUPPORT_TOP_K = 6,
    DS4_DSPARK_SUPPORT_CANDIDATE_ROWS = 5,
    /* The target verifier sees pending + five draft rows.  Row five is
     * verifier-only and never expands the five-row DSpark candidate block. */
    DS4_DSPARK_RAW_FINALIZER_MAX_ROWS =
        DS4_DSPARK_SUPPORT_CANDIDATE_ROWS + 1,
    DS4_DSPARK_VERIFY_CAPTURE_MAX_ROWS = 16,
    DS4_DSPARK_RAW_WIDTH = 512,
    DS4_DSPARK_METAL_PAGE_BYTES = 16384,
    /* One token can retain top-k records from every target layer, plus the
     * record currently being filled by asynchronous I/O. */
    DS4_DSPARK_TARGET_CACHE_FLOOR =
        DS4_DSPARK_TARGET_LAYER_COUNT * DS4_DSPARK_TARGET_TOP_K + 1,
    /* Stages execute serially, but one stage can route five candidate rows to
     * disjoint top-6 sets. Keep all 30 records pinned until that stage's GPU
     * work completes, plus one record being filled by asynchronous I/O. */
    DS4_DSPARK_SUPPORT_CACHE_FLOOR =
        DS4_DSPARK_SUPPORT_CANDIDATE_ROWS * DS4_DSPARK_SUPPORT_TOP_K + 1,
    DS4_DSPARK_COMBINED_CACHE_FLOOR =
        DS4_DSPARK_TARGET_CACHE_FLOOR + DS4_DSPARK_SUPPORT_CACHE_FLOOR,
};

typedef enum {
    /* Official DeepSeek inference captures h.mean(dim=2) immediately after
     * the selected decoder block.  The four HC lanes are not the exposed
     * DSpark feature and the learned final HC head is a different operation. */
    DS4_DSPARK_CAPTURE_POST_LAYER_HC_MEAN = 1,
} ds4_dspark_capture_kind;

typedef struct {
    uint32_t target_layer[DS4_DSPARK_CAPTURE_STAGE_COUNT];
    uint32_t stage_count;
    uint32_t hidden_width;
    uint32_t hc_lanes;
    ds4_dspark_capture_kind kind;
} ds4_dspark_capture_contract;

typedef struct {
    uint32_t total_records;
    uint32_t target_records;
    uint32_t support_records;
    uint32_t target_floor;
    uint32_t support_floor;
} ds4_dspark_cache_plan;

static bool ds4_dspark_cache_plan_make(
        uint32_t               total_records,
        bool                   support_enabled,
        ds4_dspark_cache_plan *out);

typedef struct {
    uint32_t target_only_policy_records;
    uint32_t combined_raw_capacity_records;
    uint32_t combined_policy_records;
    uint32_t fixed_equivalent_records;
    uint32_t equal_memory_ceiling_records;
    uint32_t cycle_aligned_target_records;
    uint32_t selected_parent_records;
} ds4_dspark_equal_memory_cache_plan;

/* The combined adaptive plan already sees the added bytes, but a rounded
 * route-cycle or measured DeepSeek ceiling can hide that reduction. Cap the
 * result against the target-only policy less one fixed-byte equivalent. Use
 * the combined plan's raw byte capacity here, not its already cycle-rounded
 * target-only policy count: SUPPORT-aware rounding happens exactly once below.
 * Then reserve SUPPORT and round the remaining TARGET share down to a complete
 * 258-record route cycle; a partial TARGET cycle is not a qualified tier. */
static bool ds4_dspark_equal_memory_cache_plan_make(
        uint32_t                            target_only_policy_records,
        uint32_t                            combined_raw_capacity_records,
        uint32_t                            combined_policy_records,
        uint32_t                            fixed_equivalent_records,
        ds4_dspark_equal_memory_cache_plan *out) {
    if (!out) return false;
    memset(out, 0, sizeof(*out));
    if (target_only_policy_records <= fixed_equivalent_records ||
        combined_raw_capacity_records == 0 || combined_policy_records == 0 ||
        fixed_equivalent_records == 0) {
        return false;
    }
    const uint32_t equal_memory_ceiling =
        target_only_policy_records - fixed_equivalent_records;
    const uint32_t selected =
        combined_raw_capacity_records < equal_memory_ceiling ?
            combined_raw_capacity_records : equal_memory_ceiling;
    if (selected < DS4_DSPARK_COMBINED_CACHE_FLOOR) return false;
    const uint32_t target_allowance =
        selected - DS4_DSPARK_SUPPORT_CACHE_FLOOR;
    const uint32_t target_working_set =
        DS4_DSPARK_TARGET_LAYER_COUNT * DS4_DSPARK_TARGET_TOP_K;
    if (target_allowance < DS4_DSPARK_TARGET_CACHE_FLOOR ||
        target_working_set == 0) {
        return false;
    }
    const uint32_t cycle_aligned_target =
        1u + target_working_set *
                 ((target_allowance - 1u) / target_working_set);
    if (cycle_aligned_target >
        UINT32_MAX - DS4_DSPARK_SUPPORT_CACHE_FLOOR) {
        return false;
    }
    const uint32_t cycle_aligned_parent =
        cycle_aligned_target + DS4_DSPARK_SUPPORT_CACHE_FLOOR;
    ds4_dspark_cache_plan split = {0};
    if (!ds4_dspark_cache_plan_make(
            cycle_aligned_parent, true, &split) ||
        split.target_records != cycle_aligned_target) {
        return false;
    }
    *out = (ds4_dspark_equal_memory_cache_plan){
        .target_only_policy_records = target_only_policy_records,
        .combined_raw_capacity_records = combined_raw_capacity_records,
        .combined_policy_records = combined_policy_records,
        .fixed_equivalent_records = fixed_equivalent_records,
        .equal_memory_ceiling_records = equal_memory_ceiling,
        .cycle_aligned_target_records = cycle_aligned_target,
        .selected_parent_records = cycle_aligned_parent,
    };
    return true;
}

/* Long-context policy helpers return a TARGET cycle count. Combined sessions
 * store parent budgets, so append SUPPORT only after the TARGET result has
 * been chosen and reject partial/over-decode tiers. Zero remains "disabled". */
static bool ds4_dspark_phase_parent_for_target(
        uint32_t  target_records,
        uint32_t  decode_parent_records,
        uint32_t *parent_records_out) {
    if (parent_records_out) *parent_records_out = 0;
    if (!parent_records_out || decode_parent_records == 0) return false;
    if (target_records == 0) return true;
    const uint32_t target_working_set =
        DS4_DSPARK_TARGET_LAYER_COUNT * DS4_DSPARK_TARGET_TOP_K;
    if (target_records < DS4_DSPARK_TARGET_CACHE_FLOOR ||
        (target_records - 1u) % target_working_set != 0 ||
        target_records > UINT32_MAX - DS4_DSPARK_SUPPORT_CACHE_FLOOR) {
        return false;
    }
    const uint32_t parent =
        target_records + DS4_DSPARK_SUPPORT_CACHE_FLOOR;
    ds4_dspark_cache_plan split = {0};
    if (parent > decode_parent_records ||
        !ds4_dspark_cache_plan_make(parent, true, &split) ||
        split.target_records != target_records) {
        return false;
    }
    *parent_records_out = parent;
    return true;
}

static uint32_t ds4_dspark_prefill_cap_floor(uint32_t normal_cap) {
    const uint32_t required =
        DS4_DSPARK_SUPPORT_CANDIDATE_ROWS + 1u;
    return normal_cap < required ? required : normal_cap;
}

/* Conservative sampled-production-path allocation inventory. core_bytes is
 * the 9,764,864-byte drafter core; the default temperature-1 product path
 * additionally owns greedy-head intermediates, all five q distributions, a
 * right-sized six-row target-verifier logits buffer (pending + five drafts),
 * and one frontier snapshot for exact restore-and-replay verification. This
 * does not assume the legacy MTP-only 16-row logits buffer or add a second
 * prefix-one frontier. Metal allocations are page charged individually: tiny
 * token/id buffers still consume one 16 KiB allocation page. Keeping the
 * ingredients named makes later graph changes update admission at the same
 * time instead of growing an invisible reserve. */
typedef struct {
    uint64_t capture_current_bytes;
    uint64_t capture_verifier_bytes;
    uint64_t capture_history_bytes;
    uint64_t capture_bytes;
    uint64_t raw_pack_bytes;
    uint64_t main_x_bytes;
    uint64_t raw_projected_bytes;
    uint64_t raw_normalized_bytes;
    uint64_t raw_ring_bytes;
    uint64_t two_source_attention_staging_bytes;
    uint64_t proposal_scratch_page_bytes;
    uint64_t core_bytes;
    uint64_t greedy_head_scratch_page_bytes;
    uint64_t sampled_q_distribution_page_bytes;
    uint64_t target_verifier_logits_page_bytes;
    uint64_t verifier_ratio4_snapshot_page_bytes;
    uint64_t verifier_ratio128_snapshot_page_bytes;
    uint64_t verifier_frontier_snapshot_page_bytes;
    uint64_t sampled_host_vocab_rows_page_bytes;
    uint64_t sampled_host_confidence_feature_page_bytes;
    uint64_t sampled_host_delta_page_bytes;
    uint64_t total_bytes;
} ds4_dspark_runtime_memory;

static uint64_t ds4_dspark_metal_page_charge(uint64_t bytes) {
    const uint64_t page = DS4_DSPARK_METAL_PAGE_BYTES;
    if (bytes == 0 || bytes > UINT64_MAX - (page - 1u)) return UINT64_MAX;
    return (bytes + page - 1u) & ~(page - 1u);
}

static bool ds4_dspark_runtime_memory_make(
        ds4_dspark_runtime_memory *out) {
    if (!out) return false;
    const uint64_t row_bytes =
        (uint64_t)DS4_DSPARK_CAPTURE_WIDTH * sizeof(float);
    const uint64_t hc_row_bytes = 4u * row_bytes;
    const uint64_t block_rows = DS4_DSPARK_SUPPORT_CANDIDATE_ROWS;
    const uint64_t verifier_rows = DS4_DSPARK_RAW_FINALIZER_MAX_ROWS;
    const uint64_t raw_pack =
        verifier_rows * DS4_DSPARK_CAPTURE_STAGE_COUNT * row_bytes;
    const uint64_t main_x = verifier_rows * row_bytes;
    const uint64_t raw_stage =
        verifier_rows * DS4_DSPARK_RAW_WIDTH * sizeof(float);
    const uint64_t draft_tokens = block_rows * sizeof(int32_t);
    const uint64_t draft_hc = block_rows * hc_row_bytes;
    const uint64_t target_hc = hc_row_bytes;
    const uint64_t stage_input_hc = verifier_rows * hc_row_bytes;
    const uint64_t stage_output_hc = block_rows * hc_row_bytes;
    const uint64_t position_ids = verifier_rows * sizeof(int32_t);
    const uint64_t proposal_scratch =
        ds4_dspark_metal_page_charge(draft_tokens) +
        ds4_dspark_metal_page_charge(draft_hc) +
        ds4_dspark_metal_page_charge(target_hc) +
        ds4_dspark_metal_page_charge(stage_input_hc) +
        ds4_dspark_metal_page_charge(stage_output_hc) +
        ds4_dspark_metal_page_charge(position_ids);
    const uint64_t greedy_head_scratch =
        ds4_dspark_metal_page_charge(
            block_rows * UINT64_C(256) * sizeof(float)) +
        ds4_dspark_metal_page_charge(
            UINT64_C(129280) * sizeof(float)) +
        ds4_dspark_metal_page_charge(block_rows * sizeof(float));
    const uint64_t sampled_q_distribution =
        ds4_dspark_metal_page_charge(
            block_rows * UINT64_C(129280) * sizeof(float));
    const uint64_t target_verifier_logits =
        ds4_dspark_metal_page_charge(
            verifier_rows * UINT64_C(129280) * sizeof(float));
    /* Exact verifier rollback owns one saved frontier. Ratio-4 layers retain
     * K/V plus two independently allocated index buffers; ratio-128 layers
     * retain one K/V pair. Each component is charged at Metal page size. */
    const uint64_t verifier_ratio4_snapshot =
        UINT64_C(21) * (
            UINT64_C(2) *
                ds4_dspark_metal_page_charge(UINT64_C(32768)) +
            UINT64_C(2) *
                ds4_dspark_metal_page_charge(UINT64_C(8192)));
    const uint64_t verifier_ratio128_snapshot =
        UINT64_C(20) * UINT64_C(2) *
            ds4_dspark_metal_page_charge(UINT64_C(262144));
    /* Sampling reuses the baseline full-vocabulary arena serially. DSpark
     * adds exactly two host probability rows (q-selected and the mutable p
     * residual/bonus row) plus one 4352-value confidence feature vector. */
    const uint64_t sampled_host_vocab_rows =
        UINT64_C(2) * ds4_dspark_metal_page_charge(
            UINT64_C(129280) * sizeof(float));
    const uint64_t sampled_host_confidence_feature =
        ds4_dspark_metal_page_charge(UINT64_C(4352) * sizeof(float));
    const ds4_dspark_runtime_memory memory = {
        .capture_current_bytes =
            DS4_DSPARK_CAPTURE_STAGE_COUNT * row_bytes,
        .capture_verifier_bytes =
            DS4_DSPARK_CAPTURE_STAGE_COUNT *
            DS4_DSPARK_VERIFY_CAPTURE_MAX_ROWS * row_bytes,
        .capture_history_bytes =
            DS4_DSPARK_CAPTURE_STAGE_COUNT *
            DS4_DSPARK_HISTORY_ROWS * row_bytes,
        .raw_pack_bytes = raw_pack,
        .main_x_bytes = main_x,
        .raw_projected_bytes = raw_stage,
        .raw_normalized_bytes = raw_stage,
        .raw_ring_bytes =
            (uint64_t)DS4_DSPARK_CAPTURE_STAGE_COUNT *
            DS4_DSPARK_HISTORY_ROWS * DS4_DSPARK_RAW_WIDTH * sizeof(float),
        /* The full-ring attention kernel reads physical slots 0..127 and
         * appends at most five transient draft rows in one persistent/lazy
         * staging allocation.  Threadgroup scratch is static and sub-page. */
        .two_source_attention_staging_bytes =
            (DS4_DSPARK_HISTORY_ROWS +
             DS4_DSPARK_SUPPORT_CANDIDATE_ROWS) *
            DS4_DSPARK_RAW_WIDTH * sizeof(float),
        .proposal_scratch_page_bytes = proposal_scratch,
        .greedy_head_scratch_page_bytes = greedy_head_scratch,
        .sampled_q_distribution_page_bytes = sampled_q_distribution,
        .target_verifier_logits_page_bytes = target_verifier_logits,
        .verifier_ratio4_snapshot_page_bytes = verifier_ratio4_snapshot,
        .verifier_ratio128_snapshot_page_bytes =
            verifier_ratio128_snapshot,
        .sampled_host_vocab_rows_page_bytes = sampled_host_vocab_rows,
        .sampled_host_confidence_feature_page_bytes =
            sampled_host_confidence_feature,
    };
    ds4_dspark_runtime_memory result = memory;
    result.capture_bytes = memory.capture_current_bytes +
                           memory.capture_verifier_bytes +
                           memory.capture_history_bytes;
    result.core_bytes = result.capture_bytes + memory.raw_pack_bytes +
                        memory.main_x_bytes +
                        ds4_dspark_metal_page_charge(
                            memory.raw_projected_bytes) +
                        ds4_dspark_metal_page_charge(
                            memory.raw_normalized_bytes) +
                        memory.raw_ring_bytes +
                        ds4_dspark_metal_page_charge(
                            memory.two_source_attention_staging_bytes) +
                        proposal_scratch;
    result.verifier_frontier_snapshot_page_bytes =
        verifier_ratio4_snapshot + verifier_ratio128_snapshot;
    result.sampled_host_delta_page_bytes =
        sampled_host_vocab_rows + sampled_host_confidence_feature;
    result.total_bytes = result.core_bytes + greedy_head_scratch +
                         sampled_q_distribution + target_verifier_logits +
                         result.verifier_frontier_snapshot_page_bytes +
                         result.sampled_host_delta_page_bytes;
    if (result.capture_current_bytes != UINT64_C(49152) ||
        result.capture_verifier_bytes != UINT64_C(786432) ||
        result.capture_history_bytes != UINT64_C(6291456) ||
        result.capture_bytes != UINT64_C(7127040) ||
        result.raw_pack_bytes != UINT64_C(294912) ||
        result.main_x_bytes != UINT64_C(98304) ||
        result.raw_projected_bytes != UINT64_C(12288) ||
        result.raw_normalized_bytes != UINT64_C(12288) ||
        result.raw_ring_bytes != UINT64_C(786432) ||
        result.two_source_attention_staging_bytes != UINT64_C(272384) ||
        result.proposal_scratch_page_bytes != UINT64_C(1146880) ||
        result.core_bytes != UINT64_C(9764864) ||
        result.greedy_head_scratch_page_bytes != UINT64_C(557056) ||
        result.sampled_q_distribution_page_bytes != UINT64_C(2588672) ||
        result.target_verifier_logits_page_bytes != UINT64_C(3112960) ||
        result.verifier_ratio4_snapshot_page_bytes != UINT64_C(2064384) ||
        result.verifier_ratio128_snapshot_page_bytes != UINT64_C(10485760) ||
        result.verifier_frontier_snapshot_page_bytes != UINT64_C(12550144) ||
        result.sampled_host_vocab_rows_page_bytes != UINT64_C(1048576) ||
        result.sampled_host_confidence_feature_page_bytes != UINT64_C(32768) ||
        result.sampled_host_delta_page_bytes != UINT64_C(1081344) ||
        result.total_bytes != UINT64_C(29655040)) {
        return false;
    }
    *out = result;
    return true;
}

/* Host-side ownership for the three device capture rows.  This structure does
 * not contain activations: it only prevents a partially encoded token, a
 * duplicate layer, or an out-of-order layer from being presented to the
 * drafter as a complete capture generation. */
typedef struct {
    bool enabled;
    bool in_progress;
    uint8_t completed_mask;
    uint64_t generation;
    uint64_t ready_generation;
    uint64_t pending_frontier;
    uint64_t ready_frontier;
} ds4_dspark_capture_state;

/* Logical ownership for the three device history rings.  All stages share
 * one absolute token interval; a generation is visible only after layers
 * 40/41/42 have written that same interval and the enclosing Metal batch has
 * completed.  count may be below 128 after a checkpoint load invalidates the
 * old history; seed_ready_at() stays false until the missing prefix is either
 * rebuilt from token zero or displaced by 128 newly captured rows. */
typedef struct {
    bool enabled;
    bool valid;
    bool in_progress;
    uint8_t completed_mask;
    uint32_t count;
    uint32_t pending_rows;
    uint64_t frontier;
    uint64_t pending_start;
} ds4_dspark_history_state;

/* At most two contiguous device copies are needed to append a logical range
 * to a 128-row physical ring.  input_skip discards an older prefix when one
 * layer-major chunk contains more than the complete retained history. */
typedef struct {
    uint32_t input_skip;
    uint32_t retained_rows;
    uint32_t first_physical_row;
    uint32_t first_rows;
    uint32_t second_rows;
    uint64_t frontier;
} ds4_dspark_history_append_plan;

/* Transactional ownership for the three final-0731 raw-context rings.  The
 * rings share one committed absolute interval, but each stage publishes its
 * writes independently.  generation identifies the one active range and
 * ready_mask is deliberately transaction-local: committed metadata changes
 * only after all three stage bits belong to that generation.
 *
 * A failed transaction invalidates this raw state because a completed stage
 * may already have overwritten physical wrap slots that belonged to the old
 * committed window.  It does not invalidate target KV or the 4096-wide
 * capture/history state, which have separate owners above. */
typedef struct {
    bool enabled;
    bool valid;
    bool in_progress;
    uint8_t ready_mask;
    uint32_t length;
    uint32_t pending_rows;
    uint64_t token_start;
    uint64_t frontier;
    uint64_t generation;
    uint64_t committed_generation;
    uint64_t pending_start;
    uint64_t pending_token_start;
    uint64_t pending_frontier;
    uint32_t pending_length;
} ds4_dspark_raw_context_state;

/* A seed or append writes at most two contiguous regions per stage.  The
 * physical row is derived from the absolute retained_start modulo 128; it is
 * never derived from the current logical length. */
typedef struct {
    uint32_t input_skip;
    uint32_t retained_rows;
    uint32_t first_physical_row;
    uint32_t first_rows;
    uint32_t second_rows;
    uint32_t length;
    uint64_t generation;
    uint64_t range_start;
    uint64_t token_start;
    uint64_t frontier;
} ds4_dspark_raw_context_plan;

/* Proposal rows are not ring ownership.  The official drafter exposes the
 * committed context plus exactly five temporary candidate rows.  The
 * verifier's six-row input/logits scratch is pending + five drafts; it is not
 * a sixth candidate row and remains covered by the separate memory inventory
 * above. */
typedef struct {
    uint32_t committed_rows;
    uint32_t transient_candidate_rows;
    uint64_t committed_token_start;
    uint64_t candidate_start;
} ds4_dspark_raw_context_view;

/* Pure scheduler/verifier contract for one DSpark round.  This is deliberately
 * disconnected from ds4_session and Metal execution: it fixes the token and
 * frontier arithmetic before either path is allowed to mutate production
 * state.
 *
 * At target frontier F, P is the sampled-but-unevaluated token at position F.
 * A depth-D proposal owns q[0..D-1].  The exact target verifier evaluates
 * [P, q0, ..., q(D-1)], hence N=D+1 rows.  If A drafts are accepted, exactly
 * K=A+1 evaluated rows can become visible target state: P plus the accepted
 * draft prefix.  The last emitted token is a replacement on A<D or a bonus on
 * A=D; either way it is the next pending token and is never evaluated by this
 * round.
 *
 * A full accept may retain the verifier state.  A rejection must restore the
 * sole frontier snapshot and replay the K accepted evaluated rows exactly.
 * Consumer-side stop/EOS/interruption can shorten either case later through
 * ds4_dspark_verifier_consumption_make(). */
typedef struct {
    uint64_t frontier;
    uint64_t verifier_frontier;
    uint64_t committed_frontier;
    uint32_t draft_depth;
    uint32_t accepted_drafts;
    uint32_t verifier_rows;
    uint32_t committed_evaluated_rows;
    uint32_t emitted_rows;
    uint32_t final_pending_index;
    uint32_t replay_rows;
    bool full_accept;
    bool keep_verifier_state;
    bool restore_then_replay;
} ds4_dspark_verifier_plan;

typedef struct {
    uint64_t committed_frontier;
    uint32_t consumed_rows;
    uint32_t committed_evaluated_rows;
    uint32_t replay_rows;
    bool final_pending_consumed;
    bool keep_verifier_state;
    bool restore_then_replay;
} ds4_dspark_verifier_consumption;

typedef enum {
    DS4_DSPARK_TAIL_NO_OUTPUT = 0,
    DS4_DSPARK_TAIL_PENDING_ONLY = 1,
    DS4_DSPARK_TAIL_VERIFY = 2,
} ds4_dspark_tail_kind;

typedef struct {
    ds4_dspark_tail_kind kind;
    uint32_t available_rows;
    uint32_t draft_depth;
    uint32_t verifier_rows;
    uint32_t maximum_emitted_rows;
} ds4_dspark_tail_plan;

/* A frontier snapshot does not copy the target raw-SWA ring.  Future verifier
 * rows may therefore remain as invisible garbage after restore, but they must
 * not wrap over any live prefix row.  live=min(F, raw_window), so reserving N
 * distinct future slots is safe exactly when raw_cap-live >= N. */
static bool ds4_dspark_verifier_raw_preflight(
        uint64_t frontier,
        uint32_t verifier_rows,
        uint32_t raw_cap,
        uint32_t raw_window) {
    if (verifier_rows == 0 ||
        verifier_rows > DS4_DSPARK_RAW_FINALIZER_MAX_ROWS ||
        raw_cap == 0 || raw_window == 0 || raw_window > raw_cap) {
        return false;
    }
    const uint32_t live = frontier < raw_window ?
        (uint32_t)frontier : raw_window;
    return live <= raw_cap && verifier_rows <= raw_cap - live;
}

static bool ds4_dspark_verifier_plan_make(
        uint64_t                    frontier,
        uint32_t                    draft_depth,
        uint32_t                    accepted_drafts,
        uint32_t                    raw_cap,
        uint32_t                    raw_window,
        ds4_dspark_verifier_plan   *out) {
    if (!out || draft_depth > DS4_DSPARK_SUPPORT_CANDIDATE_ROWS ||
        accepted_drafts > draft_depth) {
        return false;
    }
    const uint32_t verifier_rows = draft_depth + 1u;
    const uint32_t committed_rows = accepted_drafts + 1u;
    if (frontier > UINT64_MAX - verifier_rows ||
        frontier > UINT64_MAX - committed_rows ||
        !ds4_dspark_verifier_raw_preflight(
            frontier, verifier_rows, raw_cap, raw_window)) {
        return false;
    }
    const bool full = accepted_drafts == draft_depth;
    *out = (ds4_dspark_verifier_plan){
        .frontier = frontier,
        .verifier_frontier = frontier + verifier_rows,
        .committed_frontier = frontier + committed_rows,
        .draft_depth = draft_depth,
        .accepted_drafts = accepted_drafts,
        .verifier_rows = verifier_rows,
        .committed_evaluated_rows = committed_rows,
        .emitted_rows = committed_rows + 1u,
        .final_pending_index = committed_rows,
        .replay_rows = full ? 0u : committed_rows,
        .full_accept = full,
        .keep_verifier_state = full,
        .restore_then_replay = !full,
    };
    return true;
}

/* Convert the prefix actually consumed by CLI/server/agent into target state.
 * The final bonus/replacement is pending, so consuming it never adds another
 * evaluated row.  Any shorter prefix forces restore-and-replay even when the
 * model verifier originally accepted the complete draft.  consumed_rows=0 is
 * the rollback form used when a stop token is not part of the caller's live
 * transcript. */
static bool ds4_dspark_verifier_consumption_make(
        const ds4_dspark_verifier_plan *plan,
        uint32_t                        consumed_rows,
        ds4_dspark_verifier_consumption *out) {
    if (!plan || !out || consumed_rows > plan->emitted_rows ||
        plan->committed_evaluated_rows == 0 ||
        plan->emitted_rows != plan->committed_evaluated_rows + 1u ||
        plan->final_pending_index != plan->committed_evaluated_rows) {
        return false;
    }
    const uint32_t committed =
        consumed_rows < plan->committed_evaluated_rows ?
            consumed_rows : plan->committed_evaluated_rows;
    if (plan->frontier > UINT64_MAX - committed) return false;
    const bool keep = plan->full_accept &&
                      committed == plan->verifier_rows;
    *out = (ds4_dspark_verifier_consumption){
        .committed_frontier = plan->frontier + committed,
        .consumed_rows = consumed_rows,
        .committed_evaluated_rows = committed,
        .replay_rows = keep ? 0u : committed,
        .final_pending_consumed =
            consumed_rows == plan->emitted_rows,
        .keep_verifier_state = keep,
        .restore_then_replay = !keep,
    };
    return true;
}

/* Tail selection reserves one output for P and one for the unevaluated final
 * pending token.  With only one slot, P is emitted without advancing target
 * state.  pending_terminal covers EOS or a caller-defined stop recognized
 * before verification; stops discovered inside a returned block use the
 * consumed-prefix contract above. */
static bool ds4_dspark_tail_plan_make(
        uint32_t              maximum_rows,
        uint32_t              context_room,
        uint32_t              desired_depth,
        bool                  pending_terminal,
        ds4_dspark_tail_plan *out) {
    if (!out || desired_depth > DS4_DSPARK_SUPPORT_CANDIDATE_ROWS) {
        return false;
    }
    const uint32_t available = maximum_rows < context_room ?
        maximum_rows : context_room;
    ds4_dspark_tail_plan plan = {
        .kind = DS4_DSPARK_TAIL_NO_OUTPUT,
        .available_rows = available,
    };
    if (available == 0) {
        *out = plan;
        return true;
    }
    if (available == 1u || pending_terminal) {
        plan.kind = DS4_DSPARK_TAIL_PENDING_ONLY;
        plan.maximum_emitted_rows = 1u;
        *out = plan;
        return true;
    }
    const uint32_t tail_depth = available - 2u;
    plan.kind = DS4_DSPARK_TAIL_VERIFY;
    plan.draft_depth = desired_depth < tail_depth ?
        desired_depth : tail_depth;
    plan.verifier_rows = plan.draft_depth + 1u;
    plan.maximum_emitted_rows = plan.draft_depth + 2u;
    *out = plan;
    return true;
}

/* Exhaust the complete D=0..5, A=0..D state space and every consumer prefix.
 * This runs inside the existing model-free runtime contract gate, so the pure
 * plan cannot drift while the product DSpark entry remains disconnected. */
static bool ds4_dspark_verifier_contract_self_check(void) {
    const uint64_t frontier = UINT64_C(1000);
    const uint32_t raw_window = DS4_DSPARK_HISTORY_ROWS;
    const uint32_t raw_cap =
        DS4_DSPARK_HISTORY_ROWS + DS4_DSPARK_RAW_FINALIZER_MAX_ROWS;
    ds4_dspark_verifier_plan plan;
    ds4_dspark_verifier_consumption consumption;
    ds4_dspark_tail_plan tail;

    for (uint32_t depth = 0;
         depth <= DS4_DSPARK_SUPPORT_CANDIDATE_ROWS; depth++) {
        const uint32_t verifier_rows = depth + 1u;
        for (uint32_t accepted = 0; accepted <= depth; accepted++) {
            const uint32_t committed = accepted + 1u;
            const bool full = accepted == depth;
            if (!ds4_dspark_verifier_plan_make(
                    frontier, depth, accepted,
                    raw_cap, raw_window, &plan) ||
                plan.frontier != frontier ||
                plan.draft_depth != depth ||
                plan.accepted_drafts != accepted ||
                plan.verifier_rows != verifier_rows ||
                plan.verifier_frontier != frontier + verifier_rows ||
                plan.committed_evaluated_rows != committed ||
                plan.committed_frontier != frontier + committed ||
                plan.emitted_rows != committed + 1u ||
                plan.final_pending_index != committed ||
                plan.full_accept != full ||
                plan.keep_verifier_state != full ||
                plan.restore_then_replay == full ||
                plan.replay_rows != (full ? 0u : committed)) {
                return false;
            }
            for (uint32_t consumed = 0;
                 consumed <= plan.emitted_rows; consumed++) {
                const uint32_t consumed_evaluated =
                    consumed < committed ? consumed : committed;
                const bool keep =
                    full && consumed_evaluated == verifier_rows;
                if (!ds4_dspark_verifier_consumption_make(
                        &plan, consumed, &consumption) ||
                    consumption.consumed_rows != consumed ||
                    consumption.committed_evaluated_rows !=
                        consumed_evaluated ||
                    consumption.committed_frontier !=
                        frontier + consumed_evaluated ||
                    consumption.final_pending_consumed !=
                        (consumed == plan.emitted_rows) ||
                    consumption.keep_verifier_state != keep ||
                    consumption.restore_then_replay == keep ||
                    consumption.replay_rows !=
                        (keep ? 0u : consumed_evaluated)) {
                    return false;
                }
            }
            if (ds4_dspark_verifier_consumption_make(
                    &plan, plan.emitted_rows + 1u, &consumption)) {
                return false;
            }
        }
    }

    /* Tail 0..7 maps to no output, pending-only, then depths 0..5. */
    for (uint32_t available = 0; available <= 7u; available++) {
        if (!ds4_dspark_tail_plan_make(
                available, available,
                DS4_DSPARK_SUPPORT_CANDIDATE_ROWS, false, &tail) ||
            tail.available_rows != available) {
            return false;
        }
        if (available == 0u) {
            if (tail.kind != DS4_DSPARK_TAIL_NO_OUTPUT ||
                tail.draft_depth != 0u || tail.verifier_rows != 0u ||
                tail.maximum_emitted_rows != 0u) {
                return false;
            }
        } else if (available == 1u) {
            if (tail.kind != DS4_DSPARK_TAIL_PENDING_ONLY ||
                tail.draft_depth != 0u || tail.verifier_rows != 0u ||
                tail.maximum_emitted_rows != 1u) {
                return false;
            }
        } else {
            const uint32_t expected_depth = available - 2u;
            if (tail.kind != DS4_DSPARK_TAIL_VERIFY ||
                tail.draft_depth != expected_depth ||
                tail.verifier_rows != expected_depth + 1u ||
                tail.maximum_emitted_rows != available) {
                return false;
            }
        }
    }

    /* Context room, not only the caller's generation limit, caps the tail. */
    if (!ds4_dspark_tail_plan_make(7u, 3u, 5u, false, &tail) ||
        tail.kind != DS4_DSPARK_TAIL_VERIFY ||
        tail.available_rows != 3u || tail.draft_depth != 1u ||
        tail.verifier_rows != 2u || tail.maximum_emitted_rows != 3u) {
        return false;
    }
    /* EOS/known stop at P is emitted alone and never starts a verifier. */
    for (uint32_t available = 1u; available <= 7u; available++) {
        if (!ds4_dspark_tail_plan_make(
                available, available, 5u, true, &tail) ||
            tail.kind != DS4_DSPARK_TAIL_PENDING_ONLY ||
            tail.maximum_emitted_rows != 1u ||
            tail.draft_depth != 0u || tail.verifier_rows != 0u) {
            return false;
        }
    }

    /* Exact raw boundary plus one-negative controls. */
    if (!ds4_dspark_verifier_raw_preflight(
            frontier, DS4_DSPARK_RAW_FINALIZER_MAX_ROWS,
            raw_cap, raw_window) ||
        ds4_dspark_verifier_raw_preflight(
            frontier, DS4_DSPARK_RAW_FINALIZER_MAX_ROWS,
            raw_cap - 1u, raw_window) ||
        !ds4_dspark_verifier_raw_preflight(3u, 3u, 6u, 6u) ||
        ds4_dspark_verifier_raw_preflight(3u, 3u, 5u, 5u)) {
        return false;
    }

    /* Malformed geometry, overflow, and missing outputs all fail closed. */
    return !ds4_dspark_verifier_plan_make(
               frontier, 0u, 0u, raw_cap, raw_window, NULL) &&
           !ds4_dspark_verifier_plan_make(
               frontier, 6u, 0u, raw_cap, raw_window, &plan) &&
           !ds4_dspark_verifier_plan_make(
               frontier, 2u, 3u, raw_cap, raw_window, &plan) &&
           !ds4_dspark_verifier_plan_make(
               frontier, 0u, 0u, 0u, raw_window, &plan) &&
           !ds4_dspark_verifier_plan_make(
               frontier, 0u, 0u, raw_cap, 0u, &plan) &&
           !ds4_dspark_verifier_plan_make(
               frontier, 0u, 0u, raw_window - 1u, raw_window, &plan) &&
           !ds4_dspark_verifier_plan_make(
               UINT64_MAX, 0u, 0u, raw_cap, raw_window, &plan) &&
           !ds4_dspark_verifier_consumption_make(NULL, 0u, &consumption) &&
           !ds4_dspark_verifier_consumption_make(&plan, 0u, NULL) &&
           !ds4_dspark_tail_plan_make(1u, 1u, 0u, false, NULL) &&
           !ds4_dspark_tail_plan_make(7u, 7u, 6u, false, &tail);
}

static const ds4_dspark_capture_contract DS4_DSPARK_CAPTURE_CONTRACT = {
    .target_layer = {40, 41, 42},
    .stage_count = DS4_DSPARK_CAPTURE_STAGE_COUNT,
    .hidden_width = DS4_DSPARK_CAPTURE_WIDTH,
    .hc_lanes = 4,
    .kind = DS4_DSPARK_CAPTURE_POST_LAYER_HC_MEAN,
};

/* Return the capture slot for a completed zero-based target layer. */
static bool ds4_dspark_capture_slot_after_layer(
        uint32_t layer,
        uint32_t *slot) {
    for (uint32_t index = 0;
         index < DS4_DSPARK_CAPTURE_CONTRACT.stage_count;
         index++) {
        if (DS4_DSPARK_CAPTURE_CONTRACT.target_layer[index] == layer) {
            if (slot) *slot = index;
            return true;
        }
    }
    return false;
}

static void ds4_dspark_capture_state_init(
        ds4_dspark_capture_state *state,
        bool                      enabled) {
    if (!state) return;
    memset(state, 0, sizeof(*state));
    state->enabled = enabled;
}

/* Starting a target forward invalidates the previous generation immediately:
 * the same three device rows are about to be overwritten in place. */
static bool ds4_dspark_capture_state_begin(
        ds4_dspark_capture_state *state,
        uint64_t                  frontier) {
    if (!state || frontier == 0) return false;
    if (!state->enabled) return true;
    if (state->in_progress) return false;
    state->generation = state->generation == UINT64_MAX ? 1u :
                        state->generation + 1u;
    state->ready_generation = 0;
    state->ready_frontier = 0;
    state->completed_mask = 0;
    state->pending_frontier = frontier;
    state->in_progress = true;
    return true;
}

static void ds4_dspark_capture_state_invalidate(
        ds4_dspark_capture_state *state) {
    if (!state) return;
    state->in_progress = false;
    state->completed_mask = 0;
    state->ready_generation = 0;
    state->pending_frontier = 0;
    state->ready_frontier = 0;
}

/* Validate a capture point before enqueueing its Metal reduction.  Non-target
 * layers are explicit no-ops.  Target layers must arrive once and in the
 * official 40 -> 41 -> 42 order. */
static bool ds4_dspark_capture_state_reserve_after_layer(
        const ds4_dspark_capture_state *state,
        uint32_t                        layer,
        bool                           *capture,
        uint32_t                       *slot) {
    if (!state || !capture || !slot) return false;
    *capture = false;
    *slot = UINT32_MAX;
    if (!state->enabled) return true;

    uint32_t capture_slot = UINT32_MAX;
    if (!ds4_dspark_capture_slot_after_layer(layer, &capture_slot)) return true;
    if (!state->in_progress || capture_slot >= DS4_DSPARK_CAPTURE_STAGE_COUNT)
        return false;
    const uint8_t expected = (uint8_t)((1u << capture_slot) - 1u);
    if (state->completed_mask != expected) return false;
    *capture = true;
    *slot = capture_slot;
    return true;
}

static bool ds4_dspark_capture_state_commit_slot(
        ds4_dspark_capture_state *state,
        uint32_t                  slot) {
    if (!state || !state->enabled || !state->in_progress ||
        slot >= DS4_DSPARK_CAPTURE_STAGE_COUNT) {
        return false;
    }
    const uint8_t expected = (uint8_t)((1u << slot) - 1u);
    if (state->completed_mask != expected) return false;
    state->completed_mask |= (uint8_t)(1u << slot);
    return true;
}

static bool ds4_dspark_capture_state_finish(
        ds4_dspark_capture_state *state,
        bool                      forward_succeeded) {
    if (!state) return false;
    if (!state->enabled) return forward_succeeded;
    const uint8_t complete =
        (uint8_t)((1u << DS4_DSPARK_CAPTURE_STAGE_COUNT) - 1u);
    const bool ready = state->in_progress && forward_succeeded &&
                       state->completed_mask == complete;
    state->in_progress = false;
    if (ready) {
        state->ready_generation = state->generation;
        state->ready_frontier = state->pending_frontier;
    } else {
        ds4_dspark_capture_state_invalidate(state);
    }
    state->pending_frontier = 0;
    return ready;
}

static bool ds4_dspark_capture_state_ready_at(
        const ds4_dspark_capture_state *state,
        uint64_t                        frontier) {
    return state && state->enabled && !state->in_progress &&
           state->generation != 0 &&
           state->ready_generation == state->generation &&
           frontier != 0 && state->ready_frontier == frontier;
}

static void ds4_dspark_history_state_init(
        ds4_dspark_history_state *state,
        bool                      enabled) {
    if (!state) return;
    memset(state, 0, sizeof(*state));
    state->enabled = enabled;
}

static void ds4_dspark_history_state_invalidate(
        ds4_dspark_history_state *state) {
    if (!state) return;
    const bool enabled = state->enabled;
    memset(state, 0, sizeof(*state));
    state->enabled = enabled;
}

static bool ds4_dspark_history_append_plan_make(
        uint64_t                           start,
        uint32_t                           rows,
        ds4_dspark_history_append_plan    *out) {
    if (!out || rows == 0 || start > UINT64_MAX - rows) return false;
    const uint32_t retained = rows < DS4_DSPARK_HISTORY_ROWS ?
        rows : DS4_DSPARK_HISTORY_ROWS;
    const uint32_t skip = rows - retained;
    const uint64_t retained_start = start + skip;
    const uint32_t physical =
        (uint32_t)(retained_start % DS4_DSPARK_HISTORY_ROWS);
    uint32_t first = DS4_DSPARK_HISTORY_ROWS - physical;
    if (first > retained) first = retained;
    const ds4_dspark_history_append_plan plan = {
        .input_skip = skip,
        .retained_rows = retained,
        .first_physical_row = physical,
        .first_rows = first,
        .second_rows = retained - first,
        .frontier = start + rows,
    };
    *out = plan;
    return true;
}

static bool ds4_dspark_history_state_begin(
        ds4_dspark_history_state *state,
        uint64_t                  start,
        uint32_t                  rows) {
    ds4_dspark_history_append_plan plan;
    if (!state || !ds4_dspark_history_append_plan_make(
                      start, rows, &plan)) {
        return false;
    }
    if (!state->enabled) return true;
    if (state->in_progress) return false;
    if (start == 0) ds4_dspark_history_state_invalidate(state);
    if (state->valid && state->frontier != start) return false;
    state->in_progress = true;
    state->completed_mask = 0;
    state->pending_start = start;
    state->pending_rows = rows;
    return true;
}

static bool ds4_dspark_history_state_reserve_after_layer(
        const ds4_dspark_history_state *state,
        uint32_t                        layer,
        bool                           *capture,
        uint32_t                       *slot) {
    if (!state || !capture || !slot) return false;
    *capture = false;
    *slot = UINT32_MAX;
    if (!state->enabled) return true;
    uint32_t capture_slot = UINT32_MAX;
    if (!ds4_dspark_capture_slot_after_layer(layer, &capture_slot)) return true;
    if (!state->in_progress || capture_slot >= DS4_DSPARK_CAPTURE_STAGE_COUNT)
        return false;
    const uint8_t expected = (uint8_t)((1u << capture_slot) - 1u);
    if (state->completed_mask != expected) return false;
    *capture = true;
    *slot = capture_slot;
    return true;
}

static bool ds4_dspark_history_state_commit_slot(
        ds4_dspark_history_state *state,
        uint32_t                  slot) {
    if (!state || !state->enabled || !state->in_progress ||
        slot >= DS4_DSPARK_CAPTURE_STAGE_COUNT) {
        return false;
    }
    const uint8_t expected = (uint8_t)((1u << slot) - 1u);
    if (state->completed_mask != expected) return false;
    state->completed_mask |= (uint8_t)(1u << slot);
    return true;
}

static bool ds4_dspark_history_state_finish(
        ds4_dspark_history_state *state,
        bool                      forward_succeeded) {
    if (!state) return false;
    if (!state->enabled) return forward_succeeded;
    const uint8_t complete =
        (uint8_t)((1u << DS4_DSPARK_CAPTURE_STAGE_COUNT) - 1u);
    const bool ready = state->in_progress && forward_succeeded &&
                       state->pending_rows != 0 &&
                       state->pending_start <=
                           UINT64_MAX - state->pending_rows &&
                       state->completed_mask == complete;
    if (!ready) {
        ds4_dspark_history_state_invalidate(state);
        return false;
    }
    const uint32_t previous = state->valid ? state->count : 0u;
    uint64_t count = (uint64_t)previous + state->pending_rows;
    if (count > DS4_DSPARK_HISTORY_ROWS) count = DS4_DSPARK_HISTORY_ROWS;
    state->count = (uint32_t)count;
    state->frontier = state->pending_start + state->pending_rows;
    state->valid = true;
    state->in_progress = false;
    state->completed_mask = 0;
    state->pending_start = 0;
    state->pending_rows = 0;
    return true;
}

static bool ds4_dspark_history_state_current_ready_at(
        const ds4_dspark_history_state *state,
        uint64_t                        frontier) {
    return state && state->enabled && state->valid && !state->in_progress &&
           state->count != 0 && frontier != 0 && state->frontier == frontier;
}

static bool ds4_dspark_history_state_seed_ready_at(
        const ds4_dspark_history_state *state,
        uint64_t                        frontier) {
    return ds4_dspark_history_state_current_ready_at(state, frontier) &&
           (state->count == DS4_DSPARK_HISTORY_ROWS ||
            state->frontier == state->count);
}

static bool ds4_dspark_history_self_check_commit(
        ds4_dspark_history_state       *state,
        uint64_t                        start,
        uint32_t                        rows,
        ds4_dspark_history_append_plan *plan,
        uint64_t                        ring[DS4_DSPARK_HISTORY_ROWS]) {
    if (!ds4_dspark_history_append_plan_make(start, rows, plan) ||
        !ds4_dspark_history_state_begin(state, start, rows)) {
        return false;
    }
    for (uint32_t stage = 0;
         stage < DS4_DSPARK_CAPTURE_STAGE_COUNT; stage++) {
        bool capture = false;
        uint32_t slot = UINT32_MAX;
        if (!ds4_dspark_history_state_reserve_after_layer(
                state, DS4_DSPARK_CAPTURE_CONTRACT.target_layer[stage],
                &capture, &slot) || !capture || slot != stage ||
            !ds4_dspark_history_state_commit_slot(state, slot)) {
            return false;
        }
    }
    if (!ds4_dspark_history_state_finish(state, true)) return false;

    /* Mirror the two device blits with absolute row ids.  This validates that
     * input_skip and physical wrap preserve the exact logical suffix, not only
     * that the host metadata reaches the expected frontier. */
    for (uint32_t i = 0; i < plan->retained_rows; i++) {
        const uint64_t absolute = start + plan->input_skip + i;
        ring[absolute % DS4_DSPARK_HISTORY_ROWS] = absolute;
    }
    const uint64_t logical_start = state->frontier - state->count;
    for (uint64_t absolute = logical_start;
         absolute < state->frontier; absolute++) {
        if (ring[absolute % DS4_DSPARK_HISTORY_ROWS] != absolute) return false;
    }
    return true;
}

static bool ds4_dspark_history_state_self_check(void) {
    ds4_dspark_history_state state;
    ds4_dspark_history_append_plan plan;
    uint64_t ring[DS4_DSPARK_HISTORY_ROWS];
    memset(ring, 0xff, sizeof(ring));
    ds4_dspark_history_state_init(&state, true);
    if (!ds4_dspark_history_self_check_commit(
            &state, 0u, 200u, &plan, ring) ||
        plan.input_skip != 72u || plan.retained_rows != 128u ||
        plan.first_physical_row != 72u || plan.first_rows != 56u ||
        plan.second_rows != 72u || state.count != 128u ||
        !ds4_dspark_history_state_seed_ready_at(&state, 200u)) {
        return false;
    }
    if (!ds4_dspark_history_self_check_commit(
            &state, 200u, 40u, &plan, ring) ||
        plan.first_physical_row != 72u || plan.first_rows != 40u ||
        plan.second_rows != 0u || state.frontier != 240u) {
        return false;
    }
    if (!ds4_dspark_history_self_check_commit(
            &state, 240u, 20u, &plan, ring) ||
        plan.first_physical_row != 112u || plan.first_rows != 16u ||
        plan.second_rows != 4u || state.frontier != 260u) {
        return false;
    }
    if (!ds4_dspark_history_self_check_commit(
            &state, 260u, 1u, &plan, ring) ||
        plan.first_physical_row != 4u || plan.first_rows != 1u ||
        state.frontier != 261u || state.count != 128u) {
        return false;
    }
    /* A verifier accepting k rows is the same contiguous ring append as a
     * target forward, while publication of the current tap selects row k-1. */
    if (!ds4_dspark_history_self_check_commit(
            &state, 261u, 5u, &plan, ring) ||
        plan.first_physical_row != 5u || plan.first_rows != 5u ||
        plan.second_rows != 0u || state.frontier != 266u ||
        ds4_dspark_history_state_begin(&state, 265u, 1u)) {
        return false;
    }
    ds4_dspark_history_state_invalidate(&state);
    if (!ds4_dspark_history_self_check_commit(
            &state, 1000u, 1u, &plan, ring) ||
        state.count != 1u ||
        ds4_dspark_history_state_seed_ready_at(&state, 1001u) ||
        !ds4_dspark_history_self_check_commit(
            &state, 1001u, 127u, &plan, ring) ||
        !ds4_dspark_history_state_seed_ready_at(&state, 1128u)) {
        return false;
    }
    if (!ds4_dspark_history_state_begin(&state, 0u, 5u)) return false;
    ds4_dspark_history_state_invalidate(&state);
    return !state.valid && !state.in_progress && state.count == 0u &&
           state.frontier == 0u;
}

static uint64_t ds4_dspark_raw_context_next_generation(uint64_t generation) {
    return generation == UINT64_MAX ? 1u : generation + 1u;
}

static uint32_t ds4_dspark_raw_context_physical_row(uint64_t absolute) {
    return (uint32_t)(absolute % DS4_DSPARK_HISTORY_ROWS);
}

static bool ds4_dspark_raw_context_state_coherent(
        const ds4_dspark_raw_context_state *state) {
    if (!state || state->length > DS4_DSPARK_HISTORY_ROWS ||
        state->pending_length > DS4_DSPARK_HISTORY_ROWS ||
        (state->ready_mask &
         (uint8_t)~((1u << DS4_DSPARK_CAPTURE_STAGE_COUNT) - 1u)) != 0) {
        return false;
    }
    if (state->valid) {
        if (state->length == 0 || state->committed_generation == 0 ||
            state->token_start > UINT64_MAX - state->length ||
            state->token_start + state->length != state->frontier) {
            return false;
        }
    } else if (state->length != 0 || state->token_start != 0 ||
               state->frontier != 0 || state->committed_generation != 0) {
        return false;
    }
    if (state->in_progress) {
        if (state->generation == 0 || state->pending_rows == 0 ||
            state->pending_length == 0 ||
            state->pending_start > UINT64_MAX - state->pending_rows ||
            state->pending_start + state->pending_rows !=
                state->pending_frontier ||
            state->pending_token_start >
                UINT64_MAX - state->pending_length ||
            state->pending_token_start + state->pending_length !=
                state->pending_frontier) {
            return false;
        }
    } else if (state->ready_mask != 0 || state->pending_rows != 0 ||
               state->pending_start != 0 ||
               state->pending_token_start != 0 ||
               state->pending_frontier != 0 ||
               state->pending_length != 0 ||
               (state->valid &&
                state->committed_generation != state->generation)) {
        return false;
    }
    return true;
}

static void ds4_dspark_raw_context_state_init(
        ds4_dspark_raw_context_state *state,
        bool                          enabled) {
    if (!state) return;
    memset(state, 0, sizeof(*state));
    state->enabled = enabled;
}

static void ds4_dspark_raw_context_state_clear_pending(
        ds4_dspark_raw_context_state *state) {
    state->in_progress = false;
    state->ready_mask = 0;
    state->pending_rows = 0;
    state->pending_start = 0;
    state->pending_token_start = 0;
    state->pending_frontier = 0;
    state->pending_length = 0;
}

/* Clear only DSpark raw ownership.  generation survives so a late completion
 * from the failed transaction cannot become current again. */
static void ds4_dspark_raw_context_state_invalidate(
        ds4_dspark_raw_context_state *state) {
    if (!state) return;
    state->valid = false;
    state->length = 0;
    state->token_start = 0;
    state->frontier = 0;
    state->committed_generation = 0;
    ds4_dspark_raw_context_state_clear_pending(state);
}

/* Reset also advances the epoch.  This is the fail-closed hook for target KV
 * resets or rewinds that cannot retain a coherent raw prefix. */
static void ds4_dspark_raw_context_state_reset(
        ds4_dspark_raw_context_state *state) {
    if (!state) return;
    state->generation =
        ds4_dspark_raw_context_next_generation(state->generation);
    ds4_dspark_raw_context_state_invalidate(state);
}

static bool ds4_dspark_raw_context_state_begin(
        ds4_dspark_raw_context_state *state,
        bool                          seed,
        uint64_t                      start,
        uint32_t                      rows,
        ds4_dspark_raw_context_plan  *out) {
    if (!state || !out || !state->enabled || state->in_progress || rows == 0 ||
        start > UINT64_MAX - rows ||
        !ds4_dspark_raw_context_state_coherent(state) ||
        (seed ? state->valid :
                (!state->valid || start != state->frontier))) {
        return false;
    }
    const uint32_t retained = rows < DS4_DSPARK_HISTORY_ROWS ?
        rows : DS4_DSPARK_HISTORY_ROWS;
    const uint32_t input_skip = rows - retained;
    const uint64_t retained_start = start + input_skip;
    const uint64_t frontier = start + rows;
    const uint64_t logical_rows =
        (seed ? 0u : state->length) + (uint64_t)rows;
    const uint32_t length = logical_rows < DS4_DSPARK_HISTORY_ROWS ?
        (uint32_t)logical_rows : DS4_DSPARK_HISTORY_ROWS;
    const uint64_t token_start = frontier - length;
    const uint32_t physical =
        ds4_dspark_raw_context_physical_row(retained_start);
    uint32_t first = DS4_DSPARK_HISTORY_ROWS - physical;
    if (first > retained) first = retained;
    const uint64_t generation =
        ds4_dspark_raw_context_next_generation(state->generation);
    const ds4_dspark_raw_context_plan plan = {
        .input_skip = input_skip,
        .retained_rows = retained,
        .first_physical_row = physical,
        .first_rows = first,
        .second_rows = retained - first,
        .length = length,
        .generation = generation,
        .range_start = start,
        .token_start = token_start,
        .frontier = frontier,
    };
    state->generation = generation;
    state->in_progress = true;
    state->ready_mask = 0;
    state->pending_rows = rows;
    state->pending_start = start;
    state->pending_token_start = token_start;
    state->pending_frontier = frontier;
    state->pending_length = length;
    *out = plan;
    return true;
}

static bool ds4_dspark_raw_context_state_begin_seed(
        ds4_dspark_raw_context_state *state,
        uint64_t                      start,
        uint32_t                      rows,
        ds4_dspark_raw_context_plan  *out) {
    return ds4_dspark_raw_context_state_begin(
        state, true, start, rows, out);
}

static bool ds4_dspark_raw_context_state_begin_append(
        ds4_dspark_raw_context_state *state,
        uint64_t                      start,
        uint32_t                      rows,
        ds4_dspark_raw_context_plan  *out) {
    return ds4_dspark_raw_context_state_begin(
        state, false, start, rows, out);
}

static bool ds4_dspark_raw_context_state_mark_stage_ready(
        ds4_dspark_raw_context_state *state,
        uint64_t                      generation,
        uint32_t                      stage) {
    if (!state || !state->enabled || !state->in_progress ||
        generation == 0 || generation != state->generation ||
        stage >= DS4_DSPARK_CAPTURE_STAGE_COUNT ||
        !ds4_dspark_raw_context_state_coherent(state)) {
        return false;
    }
    const uint8_t bit = (uint8_t)(1u << stage);
    if ((state->ready_mask & bit) != 0) return false;
    state->ready_mask |= bit;
    return true;
}

static bool ds4_dspark_raw_context_state_commit(
        ds4_dspark_raw_context_state *state,
        uint64_t                      generation) {
    const uint8_t complete =
        (uint8_t)((1u << DS4_DSPARK_CAPTURE_STAGE_COUNT) - 1u);
    if (!state || !state->enabled || !state->in_progress ||
        generation == 0 || generation != state->generation ||
        state->ready_mask != complete ||
        !ds4_dspark_raw_context_state_coherent(state)) {
        return false;
    }
    state->valid = true;
    state->length = state->pending_length;
    state->token_start = state->pending_token_start;
    state->frontier = state->pending_frontier;
    state->committed_generation = generation;
    ds4_dspark_raw_context_state_clear_pending(state);
    return ds4_dspark_raw_context_state_coherent(state);
}

static bool ds4_dspark_raw_context_state_abort(
        ds4_dspark_raw_context_state *state,
        uint64_t                      generation) {
    if (!state || !state->enabled || !state->in_progress ||
        generation == 0 || generation != state->generation) {
        return false;
    }
    ds4_dspark_raw_context_state_invalidate(state);
    return ds4_dspark_raw_context_state_coherent(state);
}

static bool ds4_dspark_raw_context_state_ready_at(
        const ds4_dspark_raw_context_state *state,
        uint64_t                            frontier) {
    return state && state->enabled && state->valid && !state->in_progress &&
           state->length != 0 && state->committed_generation != 0 &&
           state->committed_generation == state->generation &&
           state->frontier == frontier &&
           ds4_dspark_raw_context_state_coherent(state);
}

/* A rewind inside the retained absolute interval can publish its prefix by
 * shortening metadata.  Rewinding outside that interval, or while device
 * writes are in flight, invalidates raw ownership instead of guessing. */
static bool ds4_dspark_raw_context_state_rewind(
        ds4_dspark_raw_context_state *state,
        uint64_t                      frontier) {
    if (!state || !state->enabled) return false;
    if (!state->valid || state->in_progress ||
        !ds4_dspark_raw_context_state_coherent(state) ||
        frontier < state->token_start || frontier > state->frontier) {
        ds4_dspark_raw_context_state_reset(state);
        return false;
    }
    if (frontier == state->frontier) return true;
    if (frontier == state->token_start) {
        ds4_dspark_raw_context_state_reset(state);
        return true;
    }
    state->generation =
        ds4_dspark_raw_context_next_generation(state->generation);
    state->length = (uint32_t)(frontier - state->token_start);
    state->frontier = frontier;
    state->committed_generation = state->generation;
    return ds4_dspark_raw_context_state_coherent(state);
}

static bool ds4_dspark_raw_context_view_make(
        const ds4_dspark_raw_context_state *state,
        uint64_t                            candidate_start,
        ds4_dspark_raw_context_view        *out) {
    if (!state || !out ||
        candidate_start >
            UINT64_MAX - DS4_DSPARK_SUPPORT_CANDIDATE_ROWS ||
        !ds4_dspark_raw_context_state_ready_at(state, candidate_start)) {
        return false;
    }
    *out = (ds4_dspark_raw_context_view){
        .committed_rows = state->length,
        .transient_candidate_rows = DS4_DSPARK_SUPPORT_CANDIDATE_ROWS,
        .committed_token_start = state->token_start,
        .candidate_start = candidate_start,
    };
    return true;
}

static void ds4_dspark_raw_context_self_check_write(
        const ds4_dspark_raw_context_plan *plan,
        uint32_t                           stage,
        uint64_t ring[DS4_DSPARK_CAPTURE_STAGE_COUNT]
                     [DS4_DSPARK_HISTORY_ROWS]) {
    for (uint32_t row = 0; row < plan->retained_rows; row++) {
        const uint64_t absolute =
            plan->range_start + plan->input_skip + row;
        ring[stage][ds4_dspark_raw_context_physical_row(absolute)] = absolute;
    }
}

static bool ds4_dspark_raw_context_self_check_commit(
        ds4_dspark_raw_context_state      *state,
        const ds4_dspark_raw_context_plan *plan,
        uint64_t ring[DS4_DSPARK_CAPTURE_STAGE_COUNT]
                     [DS4_DSPARK_HISTORY_ROWS]) {
    for (uint32_t stage = 0; stage < DS4_DSPARK_CAPTURE_STAGE_COUNT; stage++) {
        ds4_dspark_raw_context_self_check_write(plan, stage, ring);
        if (!ds4_dspark_raw_context_state_mark_stage_ready(
                state, plan->generation, stage)) {
            return false;
        }
    }
    if (!ds4_dspark_raw_context_state_commit(state, plan->generation)) {
        return false;
    }
    for (uint32_t stage = 0; stage < DS4_DSPARK_CAPTURE_STAGE_COUNT; stage++) {
        for (uint64_t absolute = state->token_start;
             absolute < state->frontier; absolute++) {
            if (ring[stage][ds4_dspark_raw_context_physical_row(absolute)] !=
                absolute) {
                return false;
            }
        }
    }
    return true;
}

static bool ds4_dspark_raw_context_state_self_check(void) {
    ds4_dspark_raw_context_state state;
    ds4_dspark_raw_context_plan plan;
    ds4_dspark_raw_context_view view;
    uint64_t ring[DS4_DSPARK_CAPTURE_STAGE_COUNT]
                 [DS4_DSPARK_HISTORY_ROWS];
    memset(ring, 0xff, sizeof(ring));

    ds4_dspark_raw_context_state_init(&state, false);
    if (ds4_dspark_raw_context_state_begin_seed(
            &state, 0u, DS4_DSPARK_HISTORY_ROWS, &plan)) {
        return false;
    }

    /* A current target row is first appended to committed main_kv ownership;
     * the five draft rows remain a separate transient attention source.  The
     * largest six-row finalizer batch must likewise commit every row before a
     * view is exposed, including a physical 127 -> 0 wrap. */
    ds4_dspark_raw_context_state_init(&state, true);
    if (!ds4_dspark_raw_context_state_begin_seed(
            &state, 127u, DS4_DSPARK_RAW_FINALIZER_MAX_ROWS, &plan) ||
        plan.input_skip != 0u || plan.retained_rows != 6u ||
        plan.first_physical_row != 127u || plan.first_rows != 1u ||
        plan.second_rows != 5u || plan.length != 6u ||
        plan.token_start != 127u || plan.frontier != 133u ||
        ds4_dspark_raw_context_view_make(&state, 133u, &view) ||
        !ds4_dspark_raw_context_self_check_commit(&state, &plan, ring) ||
        !ds4_dspark_raw_context_view_make(&state, 133u, &view) ||
        view.committed_rows != 6u ||
        view.committed_token_start != 127u ||
        view.candidate_start != 133u ||
        view.transient_candidate_rows !=
            DS4_DSPARK_SUPPORT_CANDIDATE_ROWS) {
        return false;
    }

    ds4_dspark_raw_context_state_init(&state, true);
    if (!ds4_dspark_raw_context_state_begin_seed(
            &state, 0u, DS4_DSPARK_HISTORY_ROWS, &plan) ||
        plan.input_skip != 0u || plan.retained_rows != 128u ||
        plan.first_physical_row != 0u || plan.first_rows != 128u ||
        plan.second_rows != 0u || plan.length != 128u ||
        state.valid || state.length != 0u || state.frontier != 0u ||
        !ds4_dspark_raw_context_self_check_commit(&state, &plan, ring) ||
        state.token_start != 0u || state.length != 128u ||
        state.frontier != 128u ||
        !ds4_dspark_raw_context_state_ready_at(&state, 128u) ||
        ds4_dspark_raw_context_state_ready_at(&state, 127u) ||
        !ds4_dspark_raw_context_view_make(&state, 128u, &view) ||
        view.committed_rows != 128u ||
        view.transient_candidate_rows != 5u ||
        view.committed_token_start != 0u || view.candidate_start != 128u ||
        DS4_DSPARK_SUPPORT_CANDIDATE_ROWS != 5u ||
        ds4_dspark_raw_context_view_make(&state, 129u, &view)) {
        return false;
    }

    const uint64_t seed_generation = state.generation;
    if (!ds4_dspark_raw_context_state_begin_append(
            &state, 128u, 1u, &plan) ||
        plan.first_physical_row != 0u || plan.first_rows != 1u ||
        plan.second_rows != 0u || plan.token_start != 1u ||
        plan.frontier != 129u || state.frontier != 128u ||
        ds4_dspark_raw_context_state_mark_stage_ready(
            &state, seed_generation, 0u) ||
        ds4_dspark_raw_context_state_commit(&state, plan.generation)) {
        return false;
    }
    if (!ds4_dspark_raw_context_self_check_commit(&state, &plan, ring) ||
        state.token_start != 1u || state.length != 128u ||
        state.frontier != 129u || ring[0][0] != 128u ||
        ds4_dspark_raw_context_state_begin_append(
            &state, 130u, 1u, &plan)) {
        return false;
    }

    const uint64_t committed_generation = state.generation;
    if (!ds4_dspark_raw_context_state_begin_append(
            &state, 129u, 2u, &plan)) {
        return false;
    }
    const uint64_t partial_generation = plan.generation;
    ds4_dspark_raw_context_self_check_write(&plan, 0u, ring);
    if (ds4_dspark_raw_context_state_mark_stage_ready(
            &state, committed_generation, 0u) ||
        !ds4_dspark_raw_context_state_mark_stage_ready(
            &state, partial_generation, 0u) ||
        ds4_dspark_raw_context_state_mark_stage_ready(
            &state, partial_generation, 0u) ||
        state.ready_mask != 1u ||
        ds4_dspark_raw_context_state_abort(
            &state, committed_generation) ||
        !state.in_progress || state.ready_mask != 1u ||
        !ds4_dspark_raw_context_state_abort(&state, partial_generation) ||
        state.valid || state.in_progress || state.length != 0u ||
        state.frontier != 0u ||
        ds4_dspark_raw_context_state_ready_at(&state, 129u)) {
        return false;
    }

    if (ds4_dspark_raw_context_next_generation(UINT64_MAX) != 1u ||
        !ds4_dspark_raw_context_state_begin_seed(
            &state, 500u, 130u, &plan) ||
        plan.input_skip != 2u || plan.retained_rows != 128u ||
        plan.first_physical_row != 118u || plan.first_rows != 10u ||
        plan.second_rows != 118u || plan.token_start != 502u ||
        plan.frontier != 630u ||
        !ds4_dspark_raw_context_self_check_commit(&state, &plan, ring) ||
        state.token_start != 502u || state.length != 128u ||
        state.frontier != 630u ||
        ds4_dspark_raw_context_state_rewind(&state, 631u) ||
        state.valid || state.in_progress) {
        return false;
    }

    if (!ds4_dspark_raw_context_state_begin_seed(
            &state, 700u, 1u, &plan) ||
        !ds4_dspark_raw_context_self_check_commit(&state, &plan, ring) ||
        !ds4_dspark_raw_context_state_begin_append(
            &state, 701u, 1u, &plan) ||
        ds4_dspark_raw_context_state_rewind(&state, 701u) ||
        state.valid || state.in_progress) {
        return false;
    }

    if (!ds4_dspark_raw_context_state_begin_seed(
            &state, 1000u, 1u, &plan)) {
        return false;
    }
    const uint64_t reset_stale_generation = plan.generation;
    ds4_dspark_raw_context_state_reset(&state);
    if (state.valid || state.in_progress ||
        ds4_dspark_raw_context_state_mark_stage_ready(
            &state, reset_stale_generation, 0u) ||
        !ds4_dspark_raw_context_state_begin_seed(
            &state, 1000u, 1u, &plan) ||
        !ds4_dspark_raw_context_self_check_commit(&state, &plan, ring) ||
        state.token_start != 1000u || state.length != 1u ||
        state.frontier != 1001u ||
        !ds4_dspark_raw_context_state_begin_append(
            &state, 1001u, 127u, &plan) ||
        plan.first_physical_row != 105u || plan.first_rows != 23u ||
        plan.second_rows != 104u ||
        !ds4_dspark_raw_context_self_check_commit(&state, &plan, ring) ||
        state.token_start != 1000u || state.length != 128u ||
        state.frontier != 1128u) {
        return false;
    }

    const uint64_t before_rewind_generation = state.generation;
    if (!ds4_dspark_raw_context_state_rewind(&state, 1100u) ||
        state.generation == before_rewind_generation ||
        state.token_start != 1000u || state.length != 100u ||
        state.frontier != 1100u ||
        !ds4_dspark_raw_context_state_ready_at(&state, 1100u) ||
        ds4_dspark_raw_context_physical_row(1100u) != 76u ||
        ds4_dspark_raw_context_state_rewind(&state, 999u) ||
        state.valid || state.length != 0u || state.frontier != 0u) {
        return false;
    }

    if (!ds4_dspark_raw_context_state_begin_seed(
            &state, 2000u, 2u, &plan) ||
        !ds4_dspark_raw_context_self_check_commit(&state, &plan, ring)) {
        return false;
    }

    ds4_dspark_raw_context_state_reset(&state);
    if (!ds4_dspark_raw_context_state_begin_seed(
            &state,
            UINT64_MAX - DS4_DSPARK_SUPPORT_CANDIDATE_ROWS - 1u,
            1u,
            &plan) ||
        !ds4_dspark_raw_context_self_check_commit(&state, &plan, ring) ||
        !ds4_dspark_raw_context_view_make(
            &state,
            UINT64_MAX - DS4_DSPARK_SUPPORT_CANDIDATE_ROWS,
            &view)) {
        return false;
    }
    ds4_dspark_raw_context_state_reset(&state);
    if (!ds4_dspark_raw_context_state_begin_seed(
            &state, UINT64_MAX - 1u, 1u, &plan) ||
        !ds4_dspark_raw_context_self_check_commit(&state, &plan, ring) ||
        ds4_dspark_raw_context_view_make(&state, UINT64_MAX, &view)) {
        return false;
    }
    const uint64_t before_reset_generation = state.generation;
    ds4_dspark_raw_context_state_reset(&state);
    return state.generation != before_reset_generation && !state.valid &&
           !state.in_progress && state.length == 0u &&
           state.token_start == 0u && state.frontier == 0u &&
           ds4_dspark_raw_context_state_coherent(&state);
}

/* Device append/publication commits a stage bit only after every blit for that
 * stage succeeds.  A failure injected after the first valid blit therefore
 * leaves both transactions incomplete; finish(false) must make the partially
 * overwritten device bytes unreachable through ready metadata. */
static bool ds4_dspark_failed_blit_state_self_check(void) {
    ds4_dspark_capture_state current;
    ds4_dspark_history_state history;
    bool capture_current = false;
    bool capture_history = false;
    uint32_t current_slot = UINT32_MAX;
    uint32_t history_slot = UINT32_MAX;
    ds4_dspark_capture_state_init(&current, true);
    ds4_dspark_history_state_init(&history, true);
    if (!ds4_dspark_history_state_begin(&history, 0u, 3u) ||
        !ds4_dspark_capture_state_begin(&current, 3u) ||
        !ds4_dspark_capture_state_reserve_after_layer(
            &current, DS4_DSPARK_CAPTURE_CONTRACT.target_layer[0],
            &capture_current, &current_slot) ||
        !ds4_dspark_history_state_reserve_after_layer(
            &history, DS4_DSPARK_CAPTURE_CONTRACT.target_layer[0],
            &capture_history, &history_slot) ||
        !capture_current || !capture_history ||
        current_slot != 0u || history_slot != 0u) {
        return false;
    }
    /* No commit_slot(): the shared Metal helper reported failure after blit 1. */
    if (ds4_dspark_capture_state_finish(&current, false) ||
        ds4_dspark_history_state_finish(&history, false)) {
        return false;
    }
    return !ds4_dspark_capture_state_ready_at(&current, 3u) &&
           !ds4_dspark_history_state_current_ready_at(&history, 3u) &&
           !ds4_dspark_history_state_seed_ready_at(&history, 3u) &&
           !current.in_progress && !history.in_progress && !history.valid;
}

/* Reserve SUPPORT inside one explicit record envelope.  The support quota is
 * intentionally fixed at its safety floor in this tranche; evidence may later
 * justify a larger adaptive share, but it may never consume TARGET implicitly.
 * With support disabled, the plan is an identity operation and reserves zero
 * bytes/tensors/records beyond the existing target path. */
static bool ds4_dspark_cache_plan_make(
        uint32_t total_records,
        bool support_enabled,
        ds4_dspark_cache_plan *out) {
    if (!out || total_records < DS4_DSPARK_TARGET_CACHE_FLOOR) return false;
    ds4_dspark_cache_plan plan = {
        .total_records = total_records,
        .target_records = total_records,
        .support_records = 0,
        .target_floor = DS4_DSPARK_TARGET_CACHE_FLOOR,
        .support_floor = support_enabled ? DS4_DSPARK_SUPPORT_CACHE_FLOOR : 0,
    };
    if (support_enabled) {
        if (total_records < DS4_DSPARK_COMBINED_CACHE_FLOOR) return false;
        plan.support_records = DS4_DSPARK_SUPPORT_CACHE_FLOOR;
        plan.target_records = total_records - plan.support_records;
    }
    *out = plan;
    return true;
}

static bool ds4_dspark_runtime_contract_self_check(void) {
    uint32_t slot = UINT32_MAX;
    ds4_dspark_cache_plan target_only;
    ds4_dspark_cache_plan combined;
    ds4_dspark_equal_memory_cache_plan equal_envelope;
    ds4_dspark_equal_memory_cache_plan equal_safety;
    uint32_t long_parent = 0;
    uint32_t extended_parent = 0;
    ds4_dspark_runtime_memory memory;
    ds4_dspark_capture_state capture_state;
    bool capture = false;
    uint32_t capture_slot = UINT32_MAX;
    bool raw_finalizer_present = true;
#ifndef DS4_NO_GPU
    raw_finalizer_present =
        metal_graph_dspark_raw_finalizer_code_present();
#endif
    ds4_dspark_capture_state_init(&capture_state, true);
    return DS4_DSPARK_CAPTURE_CONTRACT.stage_count == 3u &&
           DS4_DSPARK_CAPTURE_CONTRACT.hidden_width == 4096u &&
           DS4_DSPARK_CAPTURE_CONTRACT.hc_lanes == 4u &&
           !ds4_dspark_capture_slot_after_layer(39u, &slot) &&
           ds4_dspark_capture_slot_after_layer(40u, &slot) && slot == 0u &&
           ds4_dspark_capture_slot_after_layer(41u, &slot) && slot == 1u &&
           ds4_dspark_capture_slot_after_layer(42u, &slot) && slot == 2u &&
           !ds4_dspark_capture_slot_after_layer(43u, &slot) &&
           ds4_dspark_capture_state_begin(&capture_state, 43u) &&
           ds4_dspark_capture_state_reserve_after_layer(
               &capture_state, 39u, &capture, &capture_slot) &&
           !capture && capture_slot == UINT32_MAX &&
           !ds4_dspark_capture_state_reserve_after_layer(
               &capture_state, 41u, &capture, &capture_slot) &&
           ds4_dspark_capture_state_reserve_after_layer(
               &capture_state, 40u, &capture, &capture_slot) &&
           capture && capture_slot == 0u &&
           ds4_dspark_capture_state_commit_slot(&capture_state, 0u) &&
           !ds4_dspark_capture_state_commit_slot(&capture_state, 0u) &&
           ds4_dspark_capture_state_reserve_after_layer(
               &capture_state, 41u, &capture, &capture_slot) &&
           capture && capture_slot == 1u &&
           ds4_dspark_capture_state_commit_slot(&capture_state, 1u) &&
           ds4_dspark_capture_state_reserve_after_layer(
               &capture_state, 42u, &capture, &capture_slot) &&
           capture && capture_slot == 2u &&
           ds4_dspark_capture_state_commit_slot(&capture_state, 2u) &&
           ds4_dspark_capture_state_finish(&capture_state, true) &&
           ds4_dspark_capture_state_ready_at(&capture_state, 43u) &&
           !ds4_dspark_capture_state_ready_at(&capture_state, 42u) &&
           ds4_dspark_verifier_contract_self_check() &&
           ds4_dspark_history_state_self_check() &&
           ds4_dspark_raw_context_state_self_check() &&
           ds4_dspark_failed_blit_state_self_check() &&
           raw_finalizer_present &&
           ds4_dspark_runtime_memory_make(&memory) &&
           memory.capture_bytes == UINT64_C(7127040) &&
           memory.raw_pack_bytes == UINT64_C(294912) &&
           memory.main_x_bytes == UINT64_C(98304) &&
           memory.raw_projected_bytes == UINT64_C(12288) &&
           memory.raw_normalized_bytes == UINT64_C(12288) &&
           memory.two_source_attention_staging_bytes == UINT64_C(272384) &&
           memory.core_bytes == UINT64_C(9764864) &&
           memory.verifier_frontier_snapshot_page_bytes ==
               UINT64_C(12550144) &&
           memory.sampled_host_delta_page_bytes == UINT64_C(1081344) &&
           memory.total_bytes == UINT64_C(29655040) &&
           ds4_dspark_equal_memory_cache_plan_make(
               4387u, 5000u, 4387u, 83u, &equal_envelope) &&
           equal_envelope.equal_memory_ceiling_records == 4304u &&
           equal_envelope.cycle_aligned_target_records == 4129u &&
           equal_envelope.selected_parent_records == 4160u &&
           ds4_dspark_equal_memory_cache_plan_make(
               4387u, 4100u, 3871u, 83u, &equal_safety) &&
           equal_safety.equal_memory_ceiling_records == 4304u &&
           equal_safety.cycle_aligned_target_records == 3871u &&
           equal_safety.selected_parent_records == 3902u &&
           ds4_dspark_phase_parent_for_target(
               4129u, 4160u, &long_parent) && long_parent == 4160u &&
           ds4_dspark_phase_parent_for_target(
               2065u, 4160u, &extended_parent) &&
           extended_parent == 2096u &&
           !ds4_dspark_phase_parent_for_target(
               4098u, 4160u, &long_parent) &&
           ds4_dspark_prefill_cap_floor(1u) == 6u &&
           ds4_dspark_prefill_cap_floor(5u) == 6u &&
           ds4_dspark_prefill_cap_floor(6u) == 6u &&
           ds4_dspark_prefill_cap_floor(4096u) == 4096u &&
           ds4_dspark_cache_plan_make(
               DS4_DSPARK_TARGET_CACHE_FLOOR, false, &target_only) &&
           target_only.target_records == DS4_DSPARK_TARGET_CACHE_FLOOR &&
           target_only.support_records == 0u &&
           !ds4_dspark_cache_plan_make(
               DS4_DSPARK_COMBINED_CACHE_FLOOR - 1u, true, &combined) &&
           ds4_dspark_cache_plan_make(
               DS4_DSPARK_COMBINED_CACHE_FLOOR, true, &combined) &&
           combined.target_records == DS4_DSPARK_TARGET_CACHE_FLOOR &&
           combined.support_records == DS4_DSPARK_SUPPORT_CACHE_FLOOR;
}
