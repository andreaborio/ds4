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
    DS4_DSPARK_SUPPORT_EXPERT_COUNT = 256,
    DS4_DSPARK_SUPPORT_CANDIDATE_ROWS = 5,
    /* Pinned final-0731 head geometry.  Admission rejects any support model
     * whose declared rank differs, so the graph may size head storage from
     * this constant before a model pointer is available. */
    DS4_DSPARK_MARKOV_RANK = 256,
    DS4_DSPARK_CONFIDENCE_FEATURE_WIDTH =
        DS4_DSPARK_CAPTURE_WIDTH + DS4_DSPARK_MARKOV_RANK,
    /* The target verifier sees pending + five draft rows.  Row five is
     * verifier-only and never expands the five-row DSpark candidate block. */
    DS4_DSPARK_RAW_FINALIZER_MAX_ROWS =
        DS4_DSPARK_SUPPORT_CANDIDATE_ROWS + 1,
    DS4_DSPARK_VERIFY_CAPTURE_MAX_ROWS = 16,
    DS4_DSPARK_RAW_WIDTH = 512,
    DS4_DSPARK_METAL_PAGE_BYTES = 16384,
    /* Three 256-entry uint64 address tables followed by the five-by-six
     * int32 route list occupy 6,264 bytes.  The reusable graph owner reserves
     * one complete Metal allocation page so the future stage executor need
     * not allocate this view on the decode hot path. */
    DS4_DSPARK_SELECTED_ADDRESS_VIEW_BYTES =
        3 * 256 * (int)sizeof(uint64_t) +
        DS4_DSPARK_SUPPORT_CANDIDATE_ROWS *
            DS4_DSPARK_SUPPORT_TOP_K * (int)sizeof(int32_t),
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
 * the 9,781,248-byte drafter core, including its reusable selected-address
 * page; the default temperature-1 product path
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
    uint64_t selected_address_page_bytes;
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
        .selected_address_page_bytes =
            DS4_DSPARK_METAL_PAGE_BYTES,
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
                        memory.selected_address_page_bytes +
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
        result.selected_address_page_bytes != UINT64_C(16384) ||
        result.proposal_scratch_page_bytes != UINT64_C(1146880) ||
        result.core_bytes != UINT64_C(9781248) ||
        result.greedy_head_scratch_page_bytes != UINT64_C(557056) ||
        result.sampled_q_distribution_page_bytes != UINT64_C(2588672) ||
        result.target_verifier_logits_page_bytes != UINT64_C(3112960) ||
        result.verifier_ratio4_snapshot_page_bytes != UINT64_C(2064384) ||
        result.verifier_ratio128_snapshot_page_bytes != UINT64_C(10485760) ||
        result.verifier_frontier_snapshot_page_bytes != UINT64_C(12550144) ||
        result.sampled_host_vocab_rows_page_bytes != UINT64_C(1048576) ||
        result.sampled_host_confidence_feature_page_bytes != UINT64_C(32768) ||
        result.sampled_host_delta_page_bytes != UINT64_C(1081344) ||
        result.total_bytes != UINT64_C(29671424)) {
        return false;
    }
    *out = result;
    return true;
}

#if !defined(DS4_NO_GPU) || defined(DS4_TEST_HOOKS)
typedef struct {
    bool frontier_enabled;
    bool prefix1_enabled;
} ds4_dspark_verifier_snapshot_plan;

static ds4_dspark_verifier_snapshot_plan
ds4_dspark_verifier_snapshot_plan_make(
        bool enable_mtp,
        bool enable_dspark) {
    return (ds4_dspark_verifier_snapshot_plan){
        .frontier_enabled = enable_mtp || enable_dspark,
        .prefix1_enabled = enable_mtp,
    };
}

typedef enum {
    DS4_DSPARK_FRONTIER_ATTN_KV = 0,
    DS4_DSPARK_FRONTIER_ATTN_SCORE,
    DS4_DSPARK_FRONTIER_INDEX_KV,
    DS4_DSPARK_FRONTIER_INDEX_SCORE,
    DS4_DSPARK_FRONTIER_PREFIX1_ATTN_KV,
    DS4_DSPARK_FRONTIER_PREFIX1_ATTN_SCORE,
    DS4_DSPARK_FRONTIER_PREFIX1_INDEX_KV,
    DS4_DSPARK_FRONTIER_PREFIX1_INDEX_SCORE,
    DS4_DSPARK_FRONTIER_KIND_COUNT,
} ds4_dspark_verifier_frontier_kind;

typedef uint32_t (*ds4_dspark_verifier_frontier_ratio_fn)(
    uint32_t layer, void *opaque);
typedef void *(*ds4_dspark_verifier_frontier_get_fn)(
    uint32_t layer, ds4_dspark_verifier_frontier_kind kind, void *opaque);
typedef bool (*ds4_dspark_verifier_frontier_set_fn)(
    uint32_t layer, ds4_dspark_verifier_frontier_kind kind,
    void *allocation, void *opaque);
typedef void *(*ds4_dspark_verifier_frontier_allocate_fn)(
    uint64_t bytes, void *opaque);
typedef uint64_t (*ds4_dspark_verifier_frontier_bytes_fn)(
    const void *allocation, void *opaque);
typedef void (*ds4_dspark_verifier_frontier_release_fn)(
    void *allocation, void *opaque);

/* Model-free ownership core for the verifier rollback tensors.  Production
 * supplies a thin Metal adapter; the contract self-check supplies an opaque
 * fake owner and can therefore force allocation failures without a model or
 * device.  Geometry lives here so allocation and validation cannot silently
 * diverge. */
typedef struct {
    ds4_dspark_verifier_snapshot_plan plan;
    uint32_t layer_count;
    uint32_t attn_head_dim;
    uint32_t index_head_dim;
    ds4_dspark_verifier_frontier_ratio_fn ratio;
    ds4_dspark_verifier_frontier_get_fn get;
    ds4_dspark_verifier_frontier_set_fn set;
    ds4_dspark_verifier_frontier_allocate_fn allocate;
    ds4_dspark_verifier_frontier_bytes_fn bytes;
    ds4_dspark_verifier_frontier_release_fn release;
    void *opaque;
} ds4_dspark_verifier_frontier_ops;

static bool ds4_dspark_verifier_frontier_mul(
        uint64_t left,
        uint64_t right,
        uint64_t *out) {
    if (!out || (right != 0u && left > UINT64_MAX / right)) return false;
    *out = left * right;
    return true;
}

static bool ds4_dspark_verifier_frontier_ops_valid(
        const ds4_dspark_verifier_frontier_ops *ops,
        bool                                    allocating) {
    return ops && ops->layer_count != 0u &&
           ops->attn_head_dim != 0u && ops->index_head_dim != 0u &&
           ops->ratio && ops->get && ops->set && ops->bytes &&
           ops->release && (!allocating || ops->allocate) &&
           (!ops->plan.prefix1_enabled || ops->plan.frontier_enabled);
}

static bool ds4_dspark_verifier_frontier_expected(
        const ds4_dspark_verifier_frontier_ops *ops,
        uint32_t                                layer,
        ds4_dspark_verifier_frontier_kind       kind,
        bool                                   *wanted,
        uint64_t                               *bytes) {
    if (!ops || !wanted || !bytes || layer >= ops->layer_count ||
        kind >= DS4_DSPARK_FRONTIER_KIND_COUNT) {
        return false;
    }
    *wanted = false;
    *bytes = 0u;
    const uint32_t ratio = ops->ratio(layer, ops->opaque);
    if (ratio != 0u && ratio != 4u && ratio != 128u) return false;
    if (!ops->plan.frontier_enabled || ratio == 0u) return true;

    const bool prefix1 =
        kind >= DS4_DSPARK_FRONTIER_PREFIX1_ATTN_KV;
    if (prefix1 && !ops->plan.prefix1_enabled) return true;
    const ds4_dspark_verifier_frontier_kind base = prefix1 ?
        (ds4_dspark_verifier_frontier_kind)(
            kind - DS4_DSPARK_FRONTIER_PREFIX1_ATTN_KV) : kind;
    const bool index = base == DS4_DSPARK_FRONTIER_INDEX_KV ||
                       base == DS4_DSPARK_FRONTIER_INDEX_SCORE;
    if (index && ratio != 4u) return true;

    const uint32_t coff = ratio == 4u ? 2u : 1u;
    uint64_t result = coff;
    if (!ds4_dspark_verifier_frontier_mul(
            result, index ? ops->index_head_dim : ops->attn_head_dim,
            &result) ||
        !ds4_dspark_verifier_frontier_mul(result, coff, &result) ||
        !ds4_dspark_verifier_frontier_mul(result, ratio, &result) ||
        !ds4_dspark_verifier_frontier_mul(
            result, sizeof(float), &result)) {
        return false;
    }
    *wanted = true;
    *bytes = result;
    return true;
}

static bool ds4_dspark_verifier_frontier_free(
        const ds4_dspark_verifier_frontier_ops *ops) {
    if (!ds4_dspark_verifier_frontier_ops_valid(ops, false)) return false;
    bool ok = true;
    for (uint32_t layer = 0; layer < ops->layer_count; layer++) {
        for (uint32_t raw_kind = 0;
             raw_kind < DS4_DSPARK_FRONTIER_KIND_COUNT; raw_kind++) {
            const ds4_dspark_verifier_frontier_kind kind =
                (ds4_dspark_verifier_frontier_kind)raw_kind;
            void *allocation = ops->get(layer, kind, ops->opaque);
            const bool cleared =
                ops->set(layer, kind, NULL, ops->opaque);
            if (cleared && allocation) {
                ops->release(allocation, ops->opaque);
            }
            ok = cleared && ok;
        }
    }
    return ok;
}

static bool ds4_dspark_verifier_frontier_valid(
        const ds4_dspark_verifier_frontier_ops *ops,
        uint64_t                               *page_bytes_out) {
    if (!ds4_dspark_verifier_frontier_ops_valid(ops, false)) return false;
    uint64_t page_bytes = 0u;
    for (uint32_t layer = 0; layer < ops->layer_count; layer++) {
        for (uint32_t raw_kind = 0;
             raw_kind < DS4_DSPARK_FRONTIER_KIND_COUNT; raw_kind++) {
            const ds4_dspark_verifier_frontier_kind kind =
                (ds4_dspark_verifier_frontier_kind)raw_kind;
            bool wanted = false;
            uint64_t expected_bytes = 0u;
            if (!ds4_dspark_verifier_frontier_expected(
                    ops, layer, kind, &wanted, &expected_bytes)) {
                return false;
            }
            const void *allocation = ops->get(layer, kind, ops->opaque);
            if ((!wanted && allocation) ||
                (wanted && (!allocation ||
                            ops->bytes(allocation, ops->opaque) !=
                                expected_bytes))) {
                return false;
            }
            if (wanted) {
                const uint64_t charge =
                    ds4_dspark_metal_page_charge(expected_bytes);
                if (charge == UINT64_MAX ||
                    page_bytes > UINT64_MAX - charge) {
                    return false;
                }
                page_bytes += charge;
            }
        }
    }
    if (page_bytes_out) *page_bytes_out = page_bytes;
    return true;
}

static bool ds4_dspark_verifier_frontier_allocate(
        const ds4_dspark_verifier_frontier_ops *ops) {
    if (!ds4_dspark_verifier_frontier_ops_valid(
            ops, ops && ops->plan.frontier_enabled)) {
        return false;
    }
    if (!ds4_dspark_verifier_frontier_free(ops)) return false;
    if (!ops->plan.frontier_enabled) {
        return ds4_dspark_verifier_frontier_valid(ops, NULL);
    }
    for (uint32_t layer = 0; layer < ops->layer_count; layer++) {
        for (uint32_t raw_kind = 0;
             raw_kind < DS4_DSPARK_FRONTIER_KIND_COUNT; raw_kind++) {
            const ds4_dspark_verifier_frontier_kind kind =
                (ds4_dspark_verifier_frontier_kind)raw_kind;
            bool wanted = false;
            uint64_t bytes = 0u;
            if (!ds4_dspark_verifier_frontier_expected(
                    ops, layer, kind, &wanted, &bytes)) {
                goto fail;
            }
            if (!wanted) continue;
            void *allocation = ops->allocate(bytes, ops->opaque);
            if (!allocation) goto fail;
            if (!ops->set(layer, kind, allocation, ops->opaque)) {
                ops->release(allocation, ops->opaque);
                goto fail;
            }
        }
    }
    if (!ds4_dspark_verifier_frontier_valid(ops, NULL)) goto fail;
    return true;

fail:
    ds4_dspark_verifier_frontier_free(ops);
    return false;
}
#endif

/* Opaque, private owner for the one reusable selected-address Metal page.
 * Keeping the ownership protocol independent from Metal lets the model-free
 * contract force allocator failure and verify exact unwind. */
typedef struct {
    void *selected_address_page;
    uint64_t selected_address_page_bytes;
} ds4_dspark_workspace_owner;

typedef void *(*ds4_dspark_workspace_allocate_fn)(
    uint64_t bytes, void *opaque);
typedef void (*ds4_dspark_workspace_release_fn)(
    void *allocation, void *opaque);

#if !defined(DS4_NO_GPU) || defined(DS4_TEST_HOOKS)
static bool ds4_dspark_workspace_owner_valid(
        const ds4_dspark_workspace_owner *owner,
        bool                              enabled) {
    if (!owner) return false;
    if (!enabled) {
        return owner->selected_address_page == NULL &&
               owner->selected_address_page_bytes == 0u;
    }
    return owner->selected_address_page != NULL &&
           owner->selected_address_page_bytes ==
               DS4_DSPARK_METAL_PAGE_BYTES;
}

static void ds4_dspark_workspace_owner_release(
        ds4_dspark_workspace_owner    *owner,
        ds4_dspark_workspace_release_fn release,
        void                          *opaque) {
    if (!owner) return;
    if (owner->selected_address_page && release) {
        release(owner->selected_address_page, opaque);
    }
    memset(owner, 0, sizeof(*owner));
}

static bool ds4_dspark_workspace_owner_allocate(
        ds4_dspark_workspace_owner     *owner,
        bool                            enabled,
        ds4_dspark_workspace_allocate_fn allocate,
        void                           *opaque) {
    if (!owner || !ds4_dspark_workspace_owner_valid(owner, false)) {
        return false;
    }
    if (!enabled) return true;
    if (!allocate) return false;
    void *page = allocate(DS4_DSPARK_METAL_PAGE_BYTES, opaque);
    if (!page) return false;
    owner->selected_address_page = page;
    owner->selected_address_page_bytes = DS4_DSPARK_METAL_PAGE_BYTES;
    return true;
}
#endif

#ifdef DS4_TEST_HOOKS
enum {
    /* Keep one wider target-only control so the DSpark 43-layer fixture can
     * never narrow the generic DeepSeek graph owner by accident. */
    DS4_DSPARK_VERIFIER_SELF_CHECK_LAYER_CAPACITY = 79,
};

typedef struct {
    uint32_t alloc_calls;
    uint32_t release_calls;
    bool fail;
    uint8_t page_sentinel;
} ds4_dspark_workspace_self_check_state;

static void *ds4_dspark_workspace_self_check_allocate(
        uint64_t bytes,
        void    *opaque) {
    ds4_dspark_workspace_self_check_state *state =
        (ds4_dspark_workspace_self_check_state *)opaque;
    if (!state || bytes != DS4_DSPARK_METAL_PAGE_BYTES) return NULL;
    state->alloc_calls++;
    return state->fail ? NULL : &state->page_sentinel;
}

static void ds4_dspark_workspace_self_check_release(
        void *allocation,
        void *opaque) {
    ds4_dspark_workspace_self_check_state *state =
        (ds4_dspark_workspace_self_check_state *)opaque;
    if (state && allocation == &state->page_sentinel) {
        state->release_calls++;
    }
}

static bool ds4_dspark_workspace_owner_self_check(void) {
    const ds4_dspark_verifier_snapshot_plan target =
        ds4_dspark_verifier_snapshot_plan_make(false, false);
    const ds4_dspark_verifier_snapshot_plan dspark =
        ds4_dspark_verifier_snapshot_plan_make(false, true);
    const ds4_dspark_verifier_snapshot_plan mtp =
        ds4_dspark_verifier_snapshot_plan_make(true, false);
    ds4_dspark_workspace_owner owner = {0};
    ds4_dspark_workspace_self_check_state state = {0};
    if (target.frontier_enabled || target.prefix1_enabled ||
        !dspark.frontier_enabled || dspark.prefix1_enabled ||
        !mtp.frontier_enabled || !mtp.prefix1_enabled ||
        !ds4_dspark_workspace_owner_allocate(
            &owner, false, ds4_dspark_workspace_self_check_allocate,
            &state) ||
        state.alloc_calls != 0u ||
        !ds4_dspark_workspace_owner_valid(&owner, false)) {
        return false;
    }
    state.fail = true;
    if (ds4_dspark_workspace_owner_allocate(
            &owner, true, ds4_dspark_workspace_self_check_allocate,
            &state) ||
        state.alloc_calls != 1u || state.release_calls != 0u ||
        !ds4_dspark_workspace_owner_valid(&owner, false)) {
        return false;
    }
    state.fail = false;
    if (!ds4_dspark_workspace_owner_allocate(
            &owner, true, ds4_dspark_workspace_self_check_allocate,
            &state) ||
        state.alloc_calls != 2u ||
        !ds4_dspark_workspace_owner_valid(&owner, true) ||
        ds4_dspark_workspace_owner_allocate(
            &owner, true, ds4_dspark_workspace_self_check_allocate,
            &state)) {
        return false;
    }
    ds4_dspark_workspace_owner_release(
        &owner, ds4_dspark_workspace_self_check_release, &state);
    return state.alloc_calls == 2u && state.release_calls == 1u &&
           ds4_dspark_workspace_owner_valid(&owner, false);
}

typedef struct {
    uint64_t bytes;
    bool live;
} ds4_dspark_verifier_frontier_self_check_allocation;

typedef struct {
    void *slot[DS4_DSPARK_VERIFIER_SELF_CHECK_LAYER_CAPACITY]
              [DS4_DSPARK_FRONTIER_KIND_COUNT];
    ds4_dspark_verifier_frontier_self_check_allocation allocation[
        DS4_DSPARK_VERIFIER_SELF_CHECK_LAYER_CAPACITY *
        DS4_DSPARK_FRONTIER_KIND_COUNT];
    uint32_t layer_count;
    uint32_t alloc_calls;
    uint32_t release_calls;
    uint32_t live_count;
    int32_t fail_after;
    bool invalid_release;
} ds4_dspark_verifier_frontier_self_check_state;

static void ds4_dspark_verifier_frontier_self_check_reset(
        ds4_dspark_verifier_frontier_self_check_state *state,
        int32_t                                         fail_after) {
    memset(state, 0, sizeof(*state));
    state->layer_count = DS4_DSPARK_TARGET_LAYER_COUNT;
    state->fail_after = fail_after;
}

static uint32_t ds4_dspark_verifier_frontier_self_check_ratio(
        uint32_t layer,
        void    *opaque) {
    (void)opaque;
    if (layer < 2u) return 0u;
    return (layer & 1u) == 0u ? 4u : 128u;
}

static void *ds4_dspark_verifier_frontier_self_check_get(
        uint32_t                                layer,
        ds4_dspark_verifier_frontier_kind       kind,
        void                                   *opaque) {
    ds4_dspark_verifier_frontier_self_check_state *state = opaque;
    if (!state || layer >= state->layer_count ||
        kind >= DS4_DSPARK_FRONTIER_KIND_COUNT) {
        return NULL;
    }
    return state->slot[layer][kind];
}

static bool ds4_dspark_verifier_frontier_self_check_set(
        uint32_t                                layer,
        ds4_dspark_verifier_frontier_kind       kind,
        void                                   *allocation,
        void                                   *opaque) {
    ds4_dspark_verifier_frontier_self_check_state *state = opaque;
    if (!state || layer >= state->layer_count ||
        kind >= DS4_DSPARK_FRONTIER_KIND_COUNT ||
        (allocation && state->slot[layer][kind])) {
        return false;
    }
    state->slot[layer][kind] = allocation;
    return true;
}

static void *ds4_dspark_verifier_frontier_self_check_allocate(
        uint64_t bytes,
        void    *opaque) {
    ds4_dspark_verifier_frontier_self_check_state *state = opaque;
    if (!state) return NULL;
    const uint32_t call = state->alloc_calls++;
    if ((int32_t)call == state->fail_after ||
        call >= sizeof(state->allocation) / sizeof(state->allocation[0])) {
        return NULL;
    }
    state->allocation[call].bytes = bytes;
    state->allocation[call].live = true;
    state->live_count++;
    return &state->allocation[call];
}

static uint64_t ds4_dspark_verifier_frontier_self_check_bytes(
        const void *allocation,
        void       *opaque) {
    ds4_dspark_verifier_frontier_self_check_state *state = opaque;
    if (!state || !allocation) return 0u;
    for (uint32_t index = 0;
         index < sizeof(state->allocation) / sizeof(state->allocation[0]);
         index++) {
        if (allocation == &state->allocation[index] &&
            state->allocation[index].live) {
            return state->allocation[index].bytes;
        }
    }
    return 0u;
}

static void ds4_dspark_verifier_frontier_self_check_release(
        void *allocation,
        void *opaque) {
    ds4_dspark_verifier_frontier_self_check_state *state = opaque;
    if (!state || !allocation) return;
    for (uint32_t index = 0;
         index < sizeof(state->allocation) / sizeof(state->allocation[0]);
         index++) {
        if (allocation != &state->allocation[index]) continue;
        if (!state->allocation[index].live || state->live_count == 0u) {
            state->invalid_release = true;
            return;
        }
        state->allocation[index].live = false;
        state->release_calls++;
        state->live_count--;
        return;
    }
    state->invalid_release = true;
}

static ds4_dspark_verifier_frontier_ops
ds4_dspark_verifier_frontier_self_check_ops(
        ds4_dspark_verifier_frontier_self_check_state *state,
        ds4_dspark_verifier_snapshot_plan              plan) {
    return (ds4_dspark_verifier_frontier_ops){
        .plan = plan,
        .layer_count = state ? state->layer_count : 0u,
        .attn_head_dim = 512u,
        .index_head_dim = 128u,
        .ratio = ds4_dspark_verifier_frontier_self_check_ratio,
        .get = ds4_dspark_verifier_frontier_self_check_get,
        .set = ds4_dspark_verifier_frontier_self_check_set,
        .allocate = ds4_dspark_verifier_frontier_self_check_allocate,
        .bytes = ds4_dspark_verifier_frontier_self_check_bytes,
        .release = ds4_dspark_verifier_frontier_self_check_release,
        .opaque = state,
    };
}

static bool ds4_dspark_verifier_frontier_self_check_clear(
        const ds4_dspark_verifier_frontier_self_check_state *state) {
    if (!state || state->live_count != 0u || state->invalid_release) {
        return false;
    }
    for (uint32_t layer = 0; layer < state->layer_count; layer++) {
        for (uint32_t kind = 0;
             kind < DS4_DSPARK_FRONTIER_KIND_COUNT; kind++) {
            if (state->slot[layer][kind]) return false;
        }
    }
    return true;
}

static bool ds4_dspark_verifier_frontier_self_check(void) {
    const ds4_dspark_verifier_snapshot_plan target =
        ds4_dspark_verifier_snapshot_plan_make(false, false);
    const ds4_dspark_verifier_snapshot_plan dspark =
        ds4_dspark_verifier_snapshot_plan_make(false, true);
    const ds4_dspark_verifier_snapshot_plan mtp =
        ds4_dspark_verifier_snapshot_plan_make(true, false);
    ds4_dspark_verifier_frontier_self_check_state state;
    uint64_t page_bytes = UINT64_MAX;

    ds4_dspark_verifier_frontier_self_check_reset(&state, 0);
    ds4_dspark_verifier_frontier_ops ops =
        ds4_dspark_verifier_frontier_self_check_ops(&state, target);
    if (!ds4_dspark_verifier_frontier_allocate(&ops) ||
        state.alloc_calls != 0u ||
        !ds4_dspark_verifier_frontier_valid(&ops, &page_bytes) ||
        page_bytes != 0u ||
        !ds4_dspark_verifier_frontier_self_check_clear(&state)) {
        return false;
    }

    ds4_dspark_verifier_frontier_self_check_reset(&state, 0);
    state.layer_count = DS4_DSPARK_VERIFIER_SELF_CHECK_LAYER_CAPACITY;
    ops = ds4_dspark_verifier_frontier_self_check_ops(&state, target);
    if (!ds4_dspark_verifier_frontier_allocate(&ops) ||
        state.alloc_calls != 0u ||
        !ds4_dspark_verifier_frontier_valid(&ops, &page_bytes) ||
        page_bytes != 0u ||
        !ds4_dspark_verifier_frontier_self_check_clear(&state)) {
        return false;
    }

    ds4_dspark_verifier_frontier_self_check_reset(&state, -1);
    state.layer_count = DS4_DSPARK_VERIFIER_SELF_CHECK_LAYER_CAPACITY;
    ops = ds4_dspark_verifier_frontier_self_check_ops(&state, mtp);
    if (!ds4_dspark_verifier_frontier_allocate(&ops) ||
        state.alloc_calls != 464u || state.live_count != 464u ||
        !ds4_dspark_verifier_frontier_valid(&ops, &page_bytes) ||
        page_bytes != UINT64_C(47513600) ||
        !state.slot[78][DS4_DSPARK_FRONTIER_ATTN_KV] ||
        !state.slot[78][DS4_DSPARK_FRONTIER_INDEX_KV] ||
        !state.slot[78][DS4_DSPARK_FRONTIER_PREFIX1_ATTN_KV] ||
        !state.slot[78][DS4_DSPARK_FRONTIER_PREFIX1_INDEX_KV] ||
        !state.slot[77][DS4_DSPARK_FRONTIER_ATTN_KV] ||
        state.slot[77][DS4_DSPARK_FRONTIER_INDEX_KV] ||
        !state.slot[77][DS4_DSPARK_FRONTIER_PREFIX1_ATTN_KV] ||
        state.slot[77][DS4_DSPARK_FRONTIER_PREFIX1_INDEX_KV] ||
        !ds4_dspark_verifier_frontier_free(&ops) ||
        state.release_calls != 464u ||
        !ds4_dspark_verifier_frontier_self_check_clear(&state)) {
        return false;
    }

    ds4_dspark_verifier_frontier_self_check_reset(&state, -1);
    ops = ds4_dspark_verifier_frontier_self_check_ops(&state, dspark);
    if (!ds4_dspark_verifier_frontier_allocate(&ops) ||
        state.alloc_calls != 124u || state.live_count != 124u ||
        !ds4_dspark_verifier_frontier_valid(&ops, &page_bytes) ||
        page_bytes != UINT64_C(12550144) ||
        !state.slot[2][DS4_DSPARK_FRONTIER_ATTN_KV] ||
        !state.slot[2][DS4_DSPARK_FRONTIER_INDEX_KV] ||
        state.slot[2][DS4_DSPARK_FRONTIER_PREFIX1_ATTN_KV] ||
        !state.slot[3][DS4_DSPARK_FRONTIER_ATTN_KV] ||
        state.slot[3][DS4_DSPARK_FRONTIER_INDEX_KV] ||
        !ds4_dspark_verifier_frontier_free(&ops) ||
        state.release_calls != 124u ||
        !ds4_dspark_verifier_frontier_self_check_clear(&state)) {
        return false;
    }

    ds4_dspark_verifier_frontier_self_check_reset(&state, -1);
    ops = ds4_dspark_verifier_frontier_self_check_ops(&state, mtp);
    if (!ds4_dspark_verifier_frontier_allocate(&ops) ||
        state.alloc_calls != 248u || state.live_count != 248u ||
        !ds4_dspark_verifier_frontier_valid(&ops, &page_bytes) ||
        page_bytes != UINT64_C(2) * UINT64_C(12550144) ||
        !state.slot[2][DS4_DSPARK_FRONTIER_PREFIX1_ATTN_KV] ||
        !state.slot[2][DS4_DSPARK_FRONTIER_PREFIX1_INDEX_KV] ||
        !state.slot[3][DS4_DSPARK_FRONTIER_PREFIX1_ATTN_KV] ||
        state.slot[3][DS4_DSPARK_FRONTIER_PREFIX1_INDEX_KV] ||
        !ds4_dspark_verifier_frontier_free(&ops) ||
        state.release_calls != 248u ||
        !ds4_dspark_verifier_frontier_self_check_clear(&state)) {
        return false;
    }

    const int32_t failpoint[] = {0, 62, 123};
    for (uint32_t index = 0;
         index < sizeof(failpoint) / sizeof(failpoint[0]); index++) {
        ds4_dspark_verifier_frontier_self_check_reset(
            &state, failpoint[index]);
        ops = ds4_dspark_verifier_frontier_self_check_ops(&state, dspark);
        if (ds4_dspark_verifier_frontier_allocate(&ops) ||
            state.alloc_calls != (uint32_t)failpoint[index] + 1u ||
            state.release_calls != (uint32_t)failpoint[index] ||
            !ds4_dspark_verifier_frontier_self_check_clear(&state)) {
            return false;
        }
    }
    return true;
}
#endif

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

#if !defined(DS4_NO_GPU) || defined(DS4_TEST_HOOKS)
/* Raw ownership can represent a wrapped partial ring, but the official
 * final-0731 proposal accepts partial startup only at physical slot zero.
 * Full non-causal attention keeps all 128 physical rows and may carry any
 * cursor. Keep the narrower executor conversion explicit and testable. */
static bool ds4_dspark_stage_committed_start_make(
        uint32_t  length,
        uint64_t  token_start,
        uint32_t *out) {
    if (!out || length < 2u || length > DS4_DSPARK_HISTORY_ROWS) {
        return false;
    }
    const uint32_t physical =
        (uint32_t)(token_start % DS4_DSPARK_HISTORY_ROWS);
    if (length < DS4_DSPARK_HISTORY_ROWS && physical != 0u) {
        return false;
    }
    *out = length == DS4_DSPARK_HISTORY_ROWS ? physical : 0u;
    return true;
}

/* The router publishes six rank-ordered selections per candidate row, while
 * the official routed reduction walks those selections in ascending expert-id
 * order.  Keep the permutation host-side and pair each weight with its expert:
 * sorting only the ids would silently apply a route probability to the wrong
 * FFN result.  Rows remain independent because the same expert may legitimately
 * serve more than one candidate.
 *
 * This plan is production-private and owns no device state.  Building through
 * a local shadow means malformed router output cannot partially overwrite the
 * caller's previously valid plan. */
typedef struct {
    int32_t expert_id[DS4_DSPARK_SUPPORT_CANDIDATE_ROWS]
                     [DS4_DSPARK_SUPPORT_TOP_K];
    float weight[DS4_DSPARK_SUPPORT_CANDIDATE_ROWS]
                [DS4_DSPARK_SUPPORT_TOP_K];
} ds4_dspark_stage_route_plan;

/* The project compiles this translation unit with -ffast-math, under which a
 * source-level isfinite() test may be folded away.  Authenticate the IEEE-754
 * exponent bits instead so NaN/Inf router output still fails closed in the
 * actual release compile mode. */
static bool ds4_dspark_stage_route_f32_finite(float value) {
    uint32_t bits = 0u;
    memcpy(&bits, &value, sizeof(bits));
    return (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}

static bool ds4_dspark_stage_route_row_valid(
        const int32_t *expert_id,
        const float   *weight) {
    if (!expert_id || !weight) return false;
    float sum = 0.0f;
    for (uint32_t slot = 0; slot < DS4_DSPARK_SUPPORT_TOP_K; slot++) {
        if (expert_id[slot] < 0 ||
            expert_id[slot] >= DS4_DSPARK_SUPPORT_EXPERT_COUNT ||
            !ds4_dspark_stage_route_f32_finite(weight[slot]) ||
            weight[slot] < 0.0f) {
            return false;
        }
        for (uint32_t prior = 0; prior < slot; prior++) {
            if (expert_id[prior] == expert_id[slot]) return false;
        }
        sum += weight[slot];
        if (!ds4_dspark_stage_route_f32_finite(sum)) return false;
    }
    /* The router normalizes the six unbiased probabilities and then applies
     * the exact final-0731 scale 1.5.  Six F32 divides/multiplies and a serial
     * validation sum need a small rounding envelope, but a merely positive or
     * arbitrary caller-provided weight vector is not authenticated output. */
    const float expected = 1.5f;
    const float tolerance = 32.0f * FLT_EPSILON * expected;
    return sum > 0.0f && fabsf(sum - expected) <= tolerance;
}

static bool ds4_dspark_stage_route_plan_coherent(
        const ds4_dspark_stage_route_plan *plan) {
    if (!plan) return false;
    for (uint32_t row = 0;
         row < DS4_DSPARK_SUPPORT_CANDIDATE_ROWS; row++) {
        if (!ds4_dspark_stage_route_row_valid(
                plan->expert_id[row], plan->weight[row])) {
            return false;
        }
        for (uint32_t slot = 1; slot < DS4_DSPARK_SUPPORT_TOP_K; slot++) {
            if (plan->expert_id[row][slot - 1u] >=
                plan->expert_id[row][slot]) {
                return false;
            }
        }
    }
    return true;
}

static bool ds4_dspark_stage_route_plan_make(
        const int32_t                 *expert_id,
        const float                   *weight,
        ds4_dspark_stage_route_plan   *out) {
    if (!expert_id || !weight || !out) return false;
    ds4_dspark_stage_route_plan shadow;
    for (uint32_t row = 0;
         row < DS4_DSPARK_SUPPORT_CANDIDATE_ROWS; row++) {
        const uint32_t base = row * DS4_DSPARK_SUPPORT_TOP_K;
        if (!ds4_dspark_stage_route_row_valid(
                expert_id + base, weight + base)) {
            return false;
        }
        for (uint32_t slot = 0; slot < DS4_DSPARK_SUPPORT_TOP_K; slot++) {
            const int32_t id = expert_id[base + slot];
            const float route_weight = weight[base + slot];
            uint32_t insert = slot;
            while (insert != 0u &&
                   shadow.expert_id[row][insert - 1u] > id) {
                shadow.expert_id[row][insert] =
                    shadow.expert_id[row][insert - 1u];
                shadow.weight[row][insert] = shadow.weight[row][insert - 1u];
                insert--;
            }
            shadow.expert_id[row][insert] = id;
            shadow.weight[row][insert] = route_weight;
        }
    }
    if (!ds4_dspark_stage_route_plan_coherent(&shadow)) return false;
    *out = shadow;
    return true;
}

/* One graph-owned stage publication.  The state is single-owner host state,
 * not a cross-thread atomic primitive: "publication" means that READY is
 * written in one final structure assignment only after the caller has drained
 * and validated its device shadow.  EXECUTING data is never exposed through
 * ready(), and abort/invalidate erase both route and cookie-bearing shadows. */
typedef enum {
    DS4_DSPARK_STAGE_CANDIDATE_EMPTY = 0,
    DS4_DSPARK_STAGE_CANDIDATE_EXECUTING = 1,
    DS4_DSPARK_STAGE_CANDIDATE_READY = 2,
} ds4_dspark_stage_candidate_phase;

typedef struct {
    uint64_t cookie;
    uint32_t stage;
    ds4_dspark_stage_route_plan route;
} ds4_dspark_stage_candidate_publication;

typedef struct {
    bool initialized;
    bool shadow_valid;
    ds4_dspark_stage_candidate_phase phase;
    uint64_t last_cookie;
    uint64_t active_cookie;
    uint32_t active_stage;
    ds4_dspark_stage_candidate_publication shadow;
    ds4_dspark_stage_candidate_publication ready;
} ds4_dspark_stage_candidate_state;

static bool ds4_dspark_stage_candidate_publication_is_zero(
        const ds4_dspark_stage_candidate_publication *publication) {
    if (!publication) return false;
    if (publication->cookie != 0u || publication->stage != 0u) return false;
    for (uint32_t row = 0;
         row < DS4_DSPARK_SUPPORT_CANDIDATE_ROWS; row++) {
        for (uint32_t slot = 0; slot < DS4_DSPARK_SUPPORT_TOP_K; slot++) {
            if (publication->route.expert_id[row][slot] != 0 ||
                publication->route.weight[row][slot] != 0.0f) {
                return false;
            }
        }
    }
    return true;
}

static bool ds4_dspark_stage_candidate_publication_coherent(
        const ds4_dspark_stage_candidate_publication *publication) {
    return publication && publication->cookie != 0u &&
           publication->stage < DS4_DSPARK_CAPTURE_STAGE_COUNT &&
           ds4_dspark_stage_route_plan_coherent(&publication->route);
}

static bool ds4_dspark_stage_candidate_state_coherent(
        const ds4_dspark_stage_candidate_state *state) {
    if (!state || !state->initialized) return false;
    switch (state->phase) {
    case DS4_DSPARK_STAGE_CANDIDATE_EMPTY:
        return !state->shadow_valid && state->active_cookie == 0u &&
               state->active_stage == UINT32_MAX &&
               ds4_dspark_stage_candidate_publication_is_zero(
                   &state->shadow) &&
               ds4_dspark_stage_candidate_publication_is_zero(&state->ready);
    case DS4_DSPARK_STAGE_CANDIDATE_EXECUTING:
        if (state->active_cookie == 0u ||
            state->active_cookie != state->last_cookie ||
            state->active_stage >= DS4_DSPARK_CAPTURE_STAGE_COUNT ||
            !ds4_dspark_stage_candidate_publication_is_zero(&state->ready)) {
            return false;
        }
        if (!state->shadow_valid) {
            return ds4_dspark_stage_candidate_publication_is_zero(
                &state->shadow);
        }
        return ds4_dspark_stage_candidate_publication_coherent(
                   &state->shadow) &&
               state->shadow.cookie == state->active_cookie &&
               state->shadow.stage == state->active_stage;
    case DS4_DSPARK_STAGE_CANDIDATE_READY:
        return !state->shadow_valid && state->active_cookie == 0u &&
               state->active_stage == UINT32_MAX &&
               ds4_dspark_stage_candidate_publication_is_zero(
                   &state->shadow) &&
               ds4_dspark_stage_candidate_publication_coherent(
                   &state->ready) &&
               state->ready.cookie == state->last_cookie;
    default:
        return false;
    }
}

static void ds4_dspark_stage_candidate_state_clear_publication(
        ds4_dspark_stage_candidate_state *state) {
    if (!state) return;
    state->shadow_valid = false;
    state->phase = DS4_DSPARK_STAGE_CANDIDATE_EMPTY;
    state->active_cookie = 0u;
    state->active_stage = UINT32_MAX;
    memset(&state->shadow, 0, sizeof(state->shadow));
    memset(&state->ready, 0, sizeof(state->ready));
}

static void ds4_dspark_stage_candidate_state_init(
        ds4_dspark_stage_candidate_state *state) {
    if (!state) return;
    memset(state, 0, sizeof(*state));
    state->initialized = true;
    state->active_stage = UINT32_MAX;
}

static void ds4_dspark_stage_candidate_state_invalidate(
        ds4_dspark_stage_candidate_state *state) {
    if (!state) return;
    if (!state->initialized) {
        memset(state, 0, sizeof(*state));
        return;
    }
    ds4_dspark_stage_candidate_state_clear_publication(state);
}

static void ds4_dspark_stage_candidate_state_free(
        ds4_dspark_stage_candidate_state *state) {
    if (state) memset(state, 0, sizeof(*state));
}

static bool ds4_dspark_stage_candidate_state_begin(
        ds4_dspark_stage_candidate_state *state,
        uint32_t                          stage,
        uint64_t                         *cookie_out) {
    if (!state || !cookie_out ||
        !ds4_dspark_stage_candidate_state_coherent(state) ||
        state->phase != DS4_DSPARK_STAGE_CANDIDATE_EMPTY ||
        state->last_cookie == UINT64_MAX ||
        stage >= DS4_DSPARK_CAPTURE_STAGE_COUNT) {
        return false;
    }
    ds4_dspark_stage_candidate_state next = *state;
    const uint64_t cookie = state->last_cookie + 1u;
    next.phase = DS4_DSPARK_STAGE_CANDIDATE_EXECUTING;
    next.last_cookie = cookie;
    next.active_cookie = cookie;
    next.active_stage = stage;
    *state = next;
    *cookie_out = cookie;
    return true;
}

static bool ds4_dspark_stage_candidate_state_set_shadow(
        ds4_dspark_stage_candidate_state *state,
        uint64_t                          cookie,
        const ds4_dspark_stage_route_plan *route) {
    if (!state || !route ||
        !ds4_dspark_stage_candidate_state_coherent(state) ||
        state->phase != DS4_DSPARK_STAGE_CANDIDATE_EXECUTING ||
        state->shadow_valid || cookie == 0u ||
        cookie != state->active_cookie ||
        !ds4_dspark_stage_route_plan_coherent(route)) {
        return false;
    }
    const ds4_dspark_stage_candidate_publication shadow = {
        .cookie = cookie,
        .stage = state->active_stage,
        .route = *route,
    };
    state->shadow = shadow;
    state->shadow_valid = true;
    return true;
}

static bool ds4_dspark_stage_candidate_state_publish(
        ds4_dspark_stage_candidate_state *state,
        uint64_t                          cookie) {
    if (!state || !ds4_dspark_stage_candidate_state_coherent(state) ||
        state->phase != DS4_DSPARK_STAGE_CANDIDATE_EXECUTING ||
        !state->shadow_valid || cookie == 0u ||
        cookie != state->active_cookie) {
        return false;
    }
    ds4_dspark_stage_candidate_state next = *state;
    next.ready = next.shadow;
    memset(&next.shadow, 0, sizeof(next.shadow));
    next.shadow_valid = false;
    next.phase = DS4_DSPARK_STAGE_CANDIDATE_READY;
    next.active_cookie = 0u;
    next.active_stage = UINT32_MAX;
    if (!ds4_dspark_stage_candidate_state_coherent(&next)) return false;
    *state = next;
    return true;
}

static bool ds4_dspark_stage_candidate_state_ready(
        const ds4_dspark_stage_candidate_state       *state,
        uint64_t                                      cookie,
        ds4_dspark_stage_candidate_publication       *out) {
    if (!state || !out ||
        !ds4_dspark_stage_candidate_state_coherent(state) ||
        state->phase != DS4_DSPARK_STAGE_CANDIDATE_READY ||
        cookie == 0u || cookie != state->ready.cookie) {
        return false;
    }
    *out = state->ready;
    return true;
}

static bool ds4_dspark_stage_candidate_state_consume(
        ds4_dspark_stage_candidate_state *state,
        uint64_t                          cookie) {
    if (!state || !ds4_dspark_stage_candidate_state_coherent(state) ||
        state->phase != DS4_DSPARK_STAGE_CANDIDATE_READY ||
        cookie == 0u || cookie != state->ready.cookie) {
        return false;
    }
    ds4_dspark_stage_candidate_state next = *state;
    ds4_dspark_stage_candidate_state_clear_publication(&next);
    if (!ds4_dspark_stage_candidate_state_coherent(&next)) return false;
    *state = next;
    return true;
}

static bool ds4_dspark_stage_candidate_state_abort(
        ds4_dspark_stage_candidate_state *state,
        uint64_t                          cookie) {
    if (!state || !ds4_dspark_stage_candidate_state_coherent(state) ||
        state->phase != DS4_DSPARK_STAGE_CANDIDATE_EXECUTING ||
        cookie == 0u || cookie != state->active_cookie) {
        return false;
    }
    ds4_dspark_stage_candidate_state next = *state;
    ds4_dspark_stage_candidate_state_clear_publication(&next);
    if (!ds4_dspark_stage_candidate_state_coherent(&next)) return false;
    *state = next;
    return true;
}

#ifdef DS4_TEST_HOOKS
static float ds4_dspark_stage_candidate_self_check_float(uint32_t bits) {
    float value = 0.0f;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static bool ds4_dspark_stage_candidate_state_self_check(void) {
    uint32_t committed_start = UINT32_MAX;
    if (ds4_dspark_stage_committed_start_make(
            1u, 0u, &committed_start) ||
        ds4_dspark_stage_committed_start_make(
            6u, 127u, &committed_start) ||
        !ds4_dspark_stage_committed_start_make(
            6u, 128u, &committed_start) ||
        committed_start != 0u ||
        !ds4_dspark_stage_committed_start_make(
            DS4_DSPARK_HISTORY_ROWS, 130u, &committed_start) ||
        committed_start != 2u) {
        return false;
    }
    int32_t expert_id[DS4_DSPARK_SUPPORT_CANDIDATE_ROWS]
                     [DS4_DSPARK_SUPPORT_TOP_K];
    float weight[DS4_DSPARK_SUPPORT_CANDIDATE_ROWS]
                [DS4_DSPARK_SUPPORT_TOP_K];
    for (uint32_t row = 0;
         row < DS4_DSPARK_SUPPORT_CANDIDATE_ROWS; row++) {
        for (uint32_t slot = 0; slot < DS4_DSPARK_SUPPORT_TOP_K; slot++) {
            expert_id[row][slot] =
                (int32_t)(row * 32u + (DS4_DSPARK_SUPPORT_TOP_K - slot));
            weight[row][slot] = (float)slot * 0.1f;
        }
    }
    /* Duplicates across rows are valid; only a duplicate within one six-slot
     * selection would execute the same expert twice for that candidate. */
    expert_id[4][5] = expert_id[0][0];

    ds4_dspark_stage_route_plan route;
    if (!ds4_dspark_stage_route_plan_make(
            &expert_id[0][0], &weight[0][0], &route) ||
        !ds4_dspark_stage_route_plan_coherent(&route)) {
        return false;
    }
    for (uint32_t row = 0;
         row < DS4_DSPARK_SUPPORT_CANDIDATE_ROWS; row++) {
        for (uint32_t slot = 1; slot < DS4_DSPARK_SUPPORT_TOP_K; slot++) {
            if (route.expert_id[row][slot - 1u] >=
                    route.expert_id[row][slot]) {
                return false;
            }
        }
        for (uint32_t slot = 0; slot < DS4_DSPARK_SUPPORT_TOP_K; slot++) {
            const uint32_t source_slot =
                DS4_DSPARK_SUPPORT_TOP_K - slot - 1u;
            if (route.expert_id[row][slot] != expert_id[row][source_slot] ||
                route.weight[row][slot] != weight[row][source_slot]) {
                return false;
            }
        }
    }

    ds4_dspark_stage_route_plan sentinel = route;
    ds4_dspark_stage_route_plan unchanged;
    const int32_t original_id = expert_id[2][3];
    const float original_weight = weight[2][3];
#define DS4_DSPARK_ROUTE_REJECTS_WITHOUT_PUBLISH(ID_VALUE, WEIGHT_VALUE) do { \
        memcpy(&unchanged, &sentinel, sizeof(unchanged));                    \
        expert_id[2][3] = (ID_VALUE);                                       \
        weight[2][3] = (WEIGHT_VALUE);                                      \
        if (ds4_dspark_stage_route_plan_make(                               \
                &expert_id[0][0], &weight[0][0], &unchanged) ||             \
            memcmp(&unchanged, &sentinel, sizeof(unchanged)) != 0) {         \
            return false;                                                   \
        }                                                                   \
        expert_id[2][3] = original_id;                                      \
        weight[2][3] = original_weight;                                     \
    } while (0)
    DS4_DSPARK_ROUTE_REJECTS_WITHOUT_PUBLISH(-1, original_weight);
    DS4_DSPARK_ROUTE_REJECTS_WITHOUT_PUBLISH(
        DS4_DSPARK_SUPPORT_EXPERT_COUNT, original_weight);
    DS4_DSPARK_ROUTE_REJECTS_WITHOUT_PUBLISH(
        original_id,
        ds4_dspark_stage_candidate_self_check_float(UINT32_C(0x7fc12345)));
    DS4_DSPARK_ROUTE_REJECTS_WITHOUT_PUBLISH(
        original_id,
        ds4_dspark_stage_candidate_self_check_float(UINT32_C(0x7f800000)));
    DS4_DSPARK_ROUTE_REJECTS_WITHOUT_PUBLISH(original_id, -0.25f);
    DS4_DSPARK_ROUTE_REJECTS_WITHOUT_PUBLISH(original_id, 0.0f);
#undef DS4_DSPARK_ROUTE_REJECTS_WITHOUT_PUBLISH

    memcpy(&unchanged, &sentinel, sizeof(unchanged));
    expert_id[2][3] = expert_id[2][2];
    if (ds4_dspark_stage_route_plan_make(
            &expert_id[0][0], &weight[0][0], &unchanged) ||
        memcmp(&unchanged, &sentinel, sizeof(unchanged)) != 0) {
        return false;
    }
    expert_id[2][3] = original_id;

    ds4_dspark_stage_candidate_state state;
    ds4_dspark_stage_candidate_state snapshot;
    ds4_dspark_stage_candidate_publication publication;
    uint64_t cookie = 0u;
    ds4_dspark_stage_candidate_state_init(&state);
    memcpy(&snapshot, &state, sizeof(snapshot));
    if (!ds4_dspark_stage_candidate_state_coherent(&state) ||
        ds4_dspark_stage_candidate_state_begin(
            &state, DS4_DSPARK_CAPTURE_STAGE_COUNT, &cookie) ||
        memcmp(&state, &snapshot, sizeof(state)) != 0 ||
        !ds4_dspark_stage_candidate_state_begin(&state, 1u, &cookie) ||
        cookie != 1u ||
        state.phase != DS4_DSPARK_STAGE_CANDIDATE_EXECUTING ||
        ds4_dspark_stage_candidate_state_ready(&state, cookie, &publication)) {
        return false;
    }

    memcpy(&snapshot, &state, sizeof(snapshot));
    if (ds4_dspark_stage_candidate_state_publish(&state, cookie) ||
        memcmp(&state, &snapshot, sizeof(state)) != 0 ||
        ds4_dspark_stage_candidate_state_set_shadow(
            &state, cookie + 1u, &route) ||
        memcmp(&state, &snapshot, sizeof(state)) != 0 ||
        !ds4_dspark_stage_candidate_state_set_shadow(
            &state, cookie, &route) ||
        !state.shadow_valid ||
        ds4_dspark_stage_candidate_state_ready(
            &state, cookie, &publication)) {
        return false;
    }
    memcpy(&snapshot, &state, sizeof(snapshot));
    if (ds4_dspark_stage_candidate_state_set_shadow(
            &state, cookie, &route) ||
        memcmp(&state, &snapshot, sizeof(state)) != 0) {
        return false;
    }

    memcpy(&snapshot, &state, sizeof(snapshot));
    if (ds4_dspark_stage_candidate_state_publish(&state, cookie + 1u) ||
        memcmp(&state, &snapshot, sizeof(state)) != 0 ||
        !ds4_dspark_stage_candidate_state_publish(&state, cookie) ||
        state.phase != DS4_DSPARK_STAGE_CANDIDATE_READY ||
        state.shadow_valid ||
        !ds4_dspark_stage_candidate_state_ready(
            &state, cookie, &publication) ||
        publication.cookie != cookie || publication.stage != 1u ||
        memcmp(&publication.route, &route, sizeof(route)) != 0) {
        return false;
    }
    memcpy(&snapshot, &state, sizeof(snapshot));
    if (ds4_dspark_stage_candidate_state_begin(&state, 0u, &cookie) ||
        memcmp(&state, &snapshot, sizeof(state)) != 0) {
        return false;
    }

    memcpy(&snapshot, &state, sizeof(snapshot));
    if (ds4_dspark_stage_candidate_state_consume(&state, cookie + 1u) ||
        memcmp(&state, &snapshot, sizeof(state)) != 0 ||
        !ds4_dspark_stage_candidate_state_consume(&state, cookie) ||
        !ds4_dspark_stage_candidate_state_coherent(&state) ||
        state.phase != DS4_DSPARK_STAGE_CANDIDATE_EMPTY ||
        state.last_cookie != cookie ||
        !ds4_dspark_stage_candidate_state_begin(&state, 2u, &cookie) ||
        cookie != 2u ||
        !ds4_dspark_stage_candidate_state_set_shadow(
            &state, cookie, &route)) {
        return false;
    }

    memcpy(&snapshot, &state, sizeof(snapshot));
    if (ds4_dspark_stage_candidate_state_abort(&state, cookie - 1u) ||
        memcmp(&state, &snapshot, sizeof(state)) != 0 ||
        !ds4_dspark_stage_candidate_state_abort(&state, cookie) ||
        state.phase != DS4_DSPARK_STAGE_CANDIDATE_EMPTY ||
        state.last_cookie != cookie ||
        !ds4_dspark_stage_candidate_state_coherent(&state)) {
        return false;
    }

    if (!ds4_dspark_stage_candidate_state_begin(&state, 0u, &cookie) ||
        cookie != 3u) {
        return false;
    }
    ds4_dspark_stage_candidate_state_invalidate(&state);
    if (!ds4_dspark_stage_candidate_state_coherent(&state) ||
        state.phase != DS4_DSPARK_STAGE_CANDIDATE_EMPTY ||
        state.last_cookie != cookie ||
        !ds4_dspark_stage_candidate_state_begin(&state, 0u, &cookie) ||
        cookie != 4u ||
        !ds4_dspark_stage_candidate_state_set_shadow(
            &state, cookie, &route) ||
        !ds4_dspark_stage_candidate_state_publish(&state, cookie)) {
        return false;
    }
    ds4_dspark_stage_candidate_state_invalidate(&state);
    if (!ds4_dspark_stage_candidate_state_coherent(&state) ||
        state.phase != DS4_DSPARK_STAGE_CANDIDATE_EMPTY ||
        state.last_cookie != cookie) {
        return false;
    }

    state.last_cookie = UINT64_MAX;
    memcpy(&snapshot, &state, sizeof(snapshot));
    if (ds4_dspark_stage_candidate_state_begin(&state, 0u, &cookie) ||
        memcmp(&state, &snapshot, sizeof(state)) != 0) {
        return false;
    }
    ds4_dspark_stage_candidate_state_invalidate(&state);
    if (!ds4_dspark_stage_candidate_state_coherent(&state) ||
        state.last_cookie != UINT64_MAX) {
        return false;
    }
    ds4_dspark_stage_candidate_state_free(&state);
    return !state.initialized;
}
#endif
#endif

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

/* Session-owned lifecycle for one proposed output block.  This contract is
 * deliberately pure: it owns only block geometry and a monotonic cookie.
 * It does not own or advance tokens, RNG, snapshots, KV/cache state, logits,
 * device work, or caller output.  Those effects remain disconnected until a
 * later tranche wires every CLI/server/agent consumer to this lifecycle. */
typedef struct {
    uint64_t cookie;
    ds4_dspark_tail_kind kind;
    uint32_t block_rows;
    uint32_t evaluated_rows;
} ds4_dspark_generation_block;

typedef struct {
    bool initialized;
    bool active;
    uint64_t last_cookie;
    uint64_t active_cookie;
    uint64_t frontier;
    ds4_dspark_tail_kind kind;
    uint32_t block_rows;
    uint32_t evaluated_rows;
    ds4_dspark_verifier_plan verifier_plan;
} ds4_dspark_generation_block_state;

/* Session operations that must not observe or mutate state halfway through a
 * proposed block.  SYNC means a caller-visible session synchronization, not
 * an internal device synchronization used to execute a future verifier. */
typedef enum {
    DS4_DSPARK_SESSION_OPERATION_SAMPLE = 0,
    DS4_DSPARK_SESSION_OPERATION_EVAL = 1,
    DS4_DSPARK_SESSION_OPERATION_SYNC = 2,
    DS4_DSPARK_SESSION_OPERATION_REWIND = 3,
    DS4_DSPARK_SESSION_OPERATION_SAVE = 4,
    DS4_DSPARK_SESSION_OPERATION_COUNT = 5,
} ds4_dspark_session_operation;

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

static bool ds4_dspark_verifier_plan_coherent(
        const ds4_dspark_verifier_plan *plan) {
    if (!plan ||
        plan->draft_depth > DS4_DSPARK_SUPPORT_CANDIDATE_ROWS ||
        plan->accepted_drafts > plan->draft_depth) {
        return false;
    }
    const uint32_t verifier_rows = plan->draft_depth + 1u;
    const uint32_t committed_rows = plan->accepted_drafts + 1u;
    const bool full = plan->accepted_drafts == plan->draft_depth;
    return plan->frontier <= UINT64_MAX - verifier_rows &&
           plan->frontier <= UINT64_MAX - committed_rows &&
           plan->verifier_rows == verifier_rows &&
           plan->verifier_frontier == plan->frontier + verifier_rows &&
           plan->committed_evaluated_rows == committed_rows &&
           plan->committed_frontier == plan->frontier + committed_rows &&
           plan->emitted_rows == committed_rows + 1u &&
           plan->final_pending_index == committed_rows &&
           plan->replay_rows == (full ? 0u : committed_rows) &&
           plan->full_accept == full &&
           plan->keep_verifier_state == full &&
           plan->restore_then_replay != full;
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
    if (!out || !ds4_dspark_verifier_plan_coherent(plan) ||
        consumed_rows > plan->emitted_rows) {
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

static bool ds4_dspark_verifier_plan_is_zero(
        const ds4_dspark_verifier_plan *plan) {
    return plan && plan->frontier == 0 && plan->verifier_frontier == 0 &&
           plan->committed_frontier == 0 && plan->draft_depth == 0 &&
           plan->accepted_drafts == 0 && plan->verifier_rows == 0 &&
           plan->committed_evaluated_rows == 0 && plan->emitted_rows == 0 &&
           plan->final_pending_index == 0 && plan->replay_rows == 0 &&
           !plan->full_accept && !plan->keep_verifier_state &&
           !plan->restore_then_replay;
}

static bool ds4_dspark_tail_plan_coherent(
        const ds4_dspark_tail_plan *plan) {
    if (!plan ||
        plan->kind < DS4_DSPARK_TAIL_NO_OUTPUT ||
        plan->kind > DS4_DSPARK_TAIL_VERIFY ||
        plan->draft_depth > DS4_DSPARK_SUPPORT_CANDIDATE_ROWS) {
        return false;
    }
    if (plan->kind == DS4_DSPARK_TAIL_NO_OUTPUT) {
        return plan->available_rows == 0 && plan->draft_depth == 0 &&
               plan->verifier_rows == 0 &&
               plan->maximum_emitted_rows == 0;
    }
    if (plan->kind == DS4_DSPARK_TAIL_PENDING_ONLY) {
        return plan->available_rows != 0 && plan->draft_depth == 0 &&
               plan->verifier_rows == 0 &&
               plan->maximum_emitted_rows == 1u;
    }
    return plan->available_rows >= 2u &&
           plan->verifier_rows == plan->draft_depth + 1u &&
           plan->maximum_emitted_rows == plan->draft_depth + 2u &&
           plan->maximum_emitted_rows <= plan->available_rows;
}

static bool ds4_dspark_generation_block_state_coherent(
        const ds4_dspark_generation_block_state *state) {
    if (!state || !state->initialized) return false;
    if (!state->active) {
        return state->active_cookie == 0 && state->frontier == 0 &&
               state->kind == DS4_DSPARK_TAIL_NO_OUTPUT &&
               state->block_rows == 0 &&
               state->evaluated_rows == 0 &&
               ds4_dspark_verifier_plan_is_zero(&state->verifier_plan);
    }
    if (state->active_cookie == 0 ||
        state->active_cookie != state->last_cookie) {
        return false;
    }
    if (state->kind == DS4_DSPARK_TAIL_PENDING_ONLY) {
        return state->frontier != UINT64_MAX && state->block_rows == 1u &&
               state->evaluated_rows == 0 &&
               ds4_dspark_verifier_plan_is_zero(&state->verifier_plan);
    }
    return state->kind == DS4_DSPARK_TAIL_VERIFY &&
           ds4_dspark_verifier_plan_coherent(&state->verifier_plan) &&
           state->frontier == state->verifier_plan.frontier &&
           state->block_rows == state->verifier_plan.emitted_rows &&
           state->evaluated_rows ==
               state->verifier_plan.committed_evaluated_rows;
}

static void ds4_dspark_generation_block_state_clear_active(
        ds4_dspark_generation_block_state *state) {
    if (!state) return;
    state->active = false;
    state->active_cookie = 0;
    state->frontier = 0;
    state->kind = DS4_DSPARK_TAIL_NO_OUTPUT;
    state->block_rows = 0;
    state->evaluated_rows = 0;
    memset(&state->verifier_plan, 0, sizeof(state->verifier_plan));
}

static void ds4_dspark_generation_block_state_init(
        ds4_dspark_generation_block_state *state) {
    if (!state) return;
    memset(state, 0, sizeof(*state));
    state->initialized = true;
}

/* Invalidation aborts the active block but preserves the session's cookie
 * generation, so a stale consumer can never alias the next transaction. */
static void ds4_dspark_generation_block_state_invalidate(
        ds4_dspark_generation_block_state *state) {
    if (!state) return;
    if (!state->initialized) {
        memset(state, 0, sizeof(*state));
        return;
    }
    ds4_dspark_generation_block_state_clear_active(state);
}

/* Free ends the owning session lifetime.  There are no resources in this pure
 * checkpoint; zeroing nevertheless makes every later operation fail closed
 * until a new owner explicitly initializes the state. */
static void ds4_dspark_generation_block_state_free(
        ds4_dspark_generation_block_state *state) {
    if (!state) return;
    memset(state, 0, sizeof(*state));
}

static bool ds4_dspark_generation_block_state_begin(
        ds4_dspark_generation_block_state *state,
        const ds4_dspark_verifier_plan     *plan,
        ds4_dspark_generation_block        *out) {
    if (!state || !out || !ds4_dspark_generation_block_state_coherent(state) ||
        state->active || state->last_cookie == UINT64_MAX ||
        !ds4_dspark_verifier_plan_coherent(plan)) {
        return false;
    }
    const uint64_t cookie = state->last_cookie + 1u;
    const ds4_dspark_generation_block block = {
        .cookie = cookie,
        .kind = DS4_DSPARK_TAIL_VERIFY,
        .block_rows = plan->emitted_rows,
        .evaluated_rows = plan->committed_evaluated_rows,
    };
    state->active = true;
    state->last_cookie = cookie;
    state->active_cookie = cookie;
    state->frontier = plan->frontier;
    state->kind = block.kind;
    state->block_rows = block.block_rows;
    state->evaluated_rows = block.evaluated_rows;
    state->verifier_plan = *plan;
    *out = block;
    return true;
}

/* A one-slot or known-terminal tail exposes the already sampled pending token
 * without constructing a depth-zero verifier plan.  No target row has been
 * evaluated, so adopting either prefix C=0 or C=1 leaves the target frontier
 * unchanged. */
static bool ds4_dspark_generation_block_state_begin_pending_only(
        ds4_dspark_generation_block_state *state,
        uint64_t                            frontier,
        const ds4_dspark_tail_plan         *tail,
        ds4_dspark_generation_block        *out) {
    if (!state || !out || !ds4_dspark_generation_block_state_coherent(state) ||
        state->active || state->last_cookie == UINT64_MAX ||
        frontier == UINT64_MAX || !ds4_dspark_tail_plan_coherent(tail) ||
        tail->kind != DS4_DSPARK_TAIL_PENDING_ONLY) {
        return false;
    }
    const uint64_t cookie = state->last_cookie + 1u;
    const ds4_dspark_generation_block block = {
        .cookie = cookie,
        .kind = DS4_DSPARK_TAIL_PENDING_ONLY,
        .block_rows = 1u,
        .evaluated_rows = 0,
    };
    state->active = true;
    state->last_cookie = cookie;
    state->active_cookie = cookie;
    state->frontier = frontier;
    state->kind = block.kind;
    state->block_rows = block.block_rows;
    state->evaluated_rows = block.evaluated_rows;
    *out = block;
    return true;
}

/* Successful commit always closes the transaction.  consumed_rows=0 is the
 * abort form.  A verifier block derives the exact restore/replay result; a
 * pending-only block has no target mutation to restore and accepts only C=0
 * or C=1. */
static bool ds4_dspark_generation_block_state_commit(
        ds4_dspark_generation_block_state *state,
        uint64_t                            cookie,
        uint32_t                            consumed_rows,
        ds4_dspark_verifier_consumption    *out) {
    if (!state || !out ||
        !ds4_dspark_generation_block_state_coherent(state) ||
        !state->active || cookie == 0 || cookie != state->active_cookie ||
        consumed_rows > state->block_rows) {
        return false;
    }
    ds4_dspark_verifier_consumption consumption;
    if (state->kind == DS4_DSPARK_TAIL_PENDING_ONLY) {
        consumption = (ds4_dspark_verifier_consumption){
            .committed_frontier = state->frontier,
            .consumed_rows = consumed_rows,
            .final_pending_consumed = consumed_rows == 1u,
        };
    } else {
        if (state->kind != DS4_DSPARK_TAIL_VERIFY ||
            !ds4_dspark_verifier_consumption_make(
                &state->verifier_plan, consumed_rows, &consumption)) {
            return false;
        }
    }
    ds4_dspark_generation_block_state_clear_active(state);
    *out = consumption;
    return true;
}

static bool ds4_dspark_generation_block_state_allows(
        const ds4_dspark_generation_block_state *state,
        ds4_dspark_session_operation              operation) {
    return operation >= DS4_DSPARK_SESSION_OPERATION_SAMPLE &&
           operation < DS4_DSPARK_SESSION_OPERATION_COUNT &&
           ds4_dspark_generation_block_state_coherent(state) &&
           !state->active;
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
        if (!ds4_dspark_tail_plan_coherent(&plan)) return false;
        *out = plan;
        return true;
    }
    if (available == 1u || pending_terminal) {
        plan.kind = DS4_DSPARK_TAIL_PENDING_ONLY;
        plan.maximum_emitted_rows = 1u;
        if (!ds4_dspark_tail_plan_coherent(&plan)) return false;
        *out = plan;
        return true;
    }
    const uint32_t tail_depth = available - 2u;
    plan.kind = DS4_DSPARK_TAIL_VERIFY;
    plan.draft_depth = desired_depth < tail_depth ?
        desired_depth : tail_depth;
    plan.verifier_rows = plan.draft_depth + 1u;
    plan.maximum_emitted_rows = plan.draft_depth + 2u;
    if (!ds4_dspark_tail_plan_coherent(&plan)) return false;
    *out = plan;
    return true;
}

/* Exhaust the transaction independently for every verifier geometry and every
 * consumer prefix.  Failed controls compare the complete state object so a
 * rejected cookie/count/begin cannot even advance hidden ownership bytes. */
static bool ds4_dspark_generation_block_state_self_check(void) {
    const uint64_t frontier = UINT64_C(1000);
    const uint32_t raw_window = DS4_DSPARK_HISTORY_ROWS;
    const uint32_t raw_cap =
        DS4_DSPARK_HISTORY_ROWS + DS4_DSPARK_RAW_FINALIZER_MAX_ROWS;
    const ds4_dspark_session_operation guarded_operations[] = {
        DS4_DSPARK_SESSION_OPERATION_SAMPLE,
        DS4_DSPARK_SESSION_OPERATION_EVAL,
        DS4_DSPARK_SESSION_OPERATION_SYNC,
        DS4_DSPARK_SESSION_OPERATION_REWIND,
        DS4_DSPARK_SESSION_OPERATION_SAVE,
    };

    for (uint32_t depth = 0;
         depth <= DS4_DSPARK_SUPPORT_CANDIDATE_ROWS; depth++) {
        for (uint32_t accepted = 0; accepted <= depth; accepted++) {
            ds4_dspark_verifier_plan plan;
            if (!ds4_dspark_verifier_plan_make(
                    frontier, depth, accepted,
                    raw_cap, raw_window, &plan)) {
                return false;
            }
            const uint32_t maximum_consumed = plan.emitted_rows;
            for (uint32_t consumed = 0;
                 consumed <= maximum_consumed; consumed++) {
                ds4_dspark_generation_block_state state;
                ds4_dspark_generation_block block;
                ds4_dspark_generation_block second_block;
                ds4_dspark_verifier_consumption expected;
                ds4_dspark_verifier_consumption actual;
                ds4_dspark_generation_block_state snapshot;
                ds4_dspark_generation_block_state_init(&state);
                if (!ds4_dspark_generation_block_state_coherent(&state)) {
                    return false;
                }
                for (uint32_t index = 0;
                     index < sizeof(guarded_operations) /
                                 sizeof(guarded_operations[0]); index++) {
                    if (!ds4_dspark_generation_block_state_allows(
                            &state, guarded_operations[index])) {
                        return false;
                    }
                }
                if (!ds4_dspark_generation_block_state_begin(
                        &state, &plan, &block) || !state.active ||
                    block.cookie == 0 || block.cookie != state.last_cookie ||
                    block.cookie != state.active_cookie ||
                    block.kind != DS4_DSPARK_TAIL_VERIFY ||
                    block.block_rows != plan.emitted_rows ||
                    block.evaluated_rows !=
                        plan.committed_evaluated_rows ||
                    state.kind != block.kind ||
                    state.frontier != plan.frontier ||
                    state.block_rows != block.block_rows ||
                    state.evaluated_rows != block.evaluated_rows ||
                    !ds4_dspark_generation_block_state_coherent(&state)) {
                    return false;
                }
                for (uint32_t index = 0;
                     index < sizeof(guarded_operations) /
                                 sizeof(guarded_operations[0]); index++) {
                    if (ds4_dspark_generation_block_state_allows(
                            &state, guarded_operations[index])) {
                        return false;
                    }
                }

                memcpy(&snapshot, &state, sizeof(snapshot));
                if (ds4_dspark_generation_block_state_begin(
                        &state, &plan, &second_block) ||
                    memcmp(&state, &snapshot, sizeof(state)) != 0) {
                    return false;
                }
                if (ds4_dspark_generation_block_state_commit(
                        &state, 0u, consumed, &actual) ||
                    memcmp(&state, &snapshot, sizeof(state)) != 0) {
                    return false;
                }
                if (ds4_dspark_generation_block_state_commit(
                        &state, block.cookie + 1u, consumed, &actual) ||
                    memcmp(&state, &snapshot, sizeof(state)) != 0) {
                    return false;
                }
                if (ds4_dspark_generation_block_state_commit(
                        &state, block.cookie, maximum_consumed + 1u, &actual) ||
                    memcmp(&state, &snapshot, sizeof(state)) != 0) {
                    return false;
                }
                if (ds4_dspark_generation_block_state_commit(
                        &state, block.cookie, consumed, NULL) ||
                    memcmp(&state, &snapshot, sizeof(state)) != 0) {
                    return false;
                }
                if (!ds4_dspark_verifier_consumption_make(
                        &plan, consumed, &expected) ||
                    !ds4_dspark_generation_block_state_commit(
                        &state, block.cookie, consumed, &actual) ||
                    actual.committed_frontier != expected.committed_frontier ||
                    actual.consumed_rows != expected.consumed_rows ||
                    actual.committed_evaluated_rows !=
                        expected.committed_evaluated_rows ||
                    actual.replay_rows != expected.replay_rows ||
                    actual.final_pending_consumed !=
                        expected.final_pending_consumed ||
                    actual.keep_verifier_state !=
                        expected.keep_verifier_state ||
                    actual.restore_then_replay !=
                        expected.restore_then_replay ||
                    state.active || state.active_cookie != 0 ||
                    state.kind != DS4_DSPARK_TAIL_NO_OUTPUT ||
                    state.frontier != 0 ||
                    state.last_cookie != block.cookie ||
                    !ds4_dspark_generation_block_state_coherent(&state)) {
                    return false;
                }
                /* C=0 is the explicit abort: no evaluated row advances and
                 * the verifier state cannot be retained. */
                if (consumed == 0u &&
                    (actual.committed_frontier != plan.frontier ||
                     actual.committed_evaluated_rows != 0u ||
                     actual.replay_rows != 0u ||
                     actual.final_pending_consumed ||
                     actual.keep_verifier_state ||
                     !actual.restore_then_replay)) {
                    return false;
                }
                for (uint32_t index = 0;
                     index < sizeof(guarded_operations) /
                                 sizeof(guarded_operations[0]); index++) {
                    if (!ds4_dspark_generation_block_state_allows(
                            &state, guarded_operations[index])) {
                        return false;
                    }
                }

                memcpy(&snapshot, &state, sizeof(snapshot));
                if (ds4_dspark_generation_block_state_commit(
                        &state, block.cookie, consumed, &actual) ||
                    memcmp(&state, &snapshot, sizeof(state)) != 0) {
                    return false;
                }
                if (!ds4_dspark_generation_block_state_begin(
                        &state, &plan, &second_block) ||
                    second_block.cookie != block.cookie + 1u) {
                    return false;
                }
                memcpy(&snapshot, &state, sizeof(snapshot));
                if (ds4_dspark_generation_block_state_commit(
                        &state, block.cookie, consumed, &actual) ||
                    memcmp(&state, &snapshot, sizeof(state)) != 0) {
                    return false;
                }

                const uint64_t invalidated_cookie = state.last_cookie;
                ds4_dspark_generation_block_state_invalidate(&state);
                if (state.active || state.active_cookie != 0 ||
                    state.last_cookie != invalidated_cookie ||
                    !ds4_dspark_generation_block_state_coherent(&state) ||
                    !ds4_dspark_generation_block_state_begin(
                        &state, &plan, &second_block) ||
                    second_block.cookie != invalidated_cookie + 1u) {
                    return false;
                }
                memcpy(&snapshot, &state, sizeof(snapshot));
                if (ds4_dspark_generation_block_state_commit(
                        &state, invalidated_cookie, consumed, &actual) ||
                    memcmp(&state, &snapshot, sizeof(state)) != 0) {
                    return false;
                }

                ds4_dspark_generation_block_state_free(&state);
                memcpy(&snapshot, &state, sizeof(snapshot));
                if (state.initialized || state.active ||
                    ds4_dspark_generation_block_state_coherent(&state) ||
                    ds4_dspark_generation_block_state_begin(
                        &state, &plan, &second_block) ||
                    ds4_dspark_generation_block_state_allows(
                        &state, DS4_DSPARK_SESSION_OPERATION_SAMPLE) ||
                    memcmp(&state, &snapshot, sizeof(state)) != 0) {
                    return false;
                }
            }
        }
    }

    ds4_dspark_verifier_plan plan;
    ds4_dspark_generation_block block;
    ds4_dspark_generation_block_state state;
    ds4_dspark_generation_block_state snapshot;
    ds4_dspark_verifier_consumption consumption;
    if (!ds4_dspark_verifier_plan_make(
            frontier, 0u, 0u, raw_cap, raw_window, &plan)) {
        return false;
    }

    ds4_dspark_generation_block_state_init(&state);
    state.last_cookie = UINT64_MAX - 1u;
    if (!ds4_dspark_generation_block_state_begin(&state, &plan, &block) ||
        block.cookie != UINT64_MAX ||
        !ds4_dspark_generation_block_state_commit(
            &state, block.cookie, 0u, &consumption)) {
        return false;
    }
    memcpy(&snapshot, &state, sizeof(snapshot));
    if (ds4_dspark_generation_block_state_begin(&state, &plan, &block) ||
        memcmp(&state, &snapshot, sizeof(state)) != 0) {
        return false;
    }

    ds4_dspark_generation_block_state_init(&state);
    state.last_cookie = UINT64_MAX;
    memcpy(&snapshot, &state, sizeof(snapshot));
    if (ds4_dspark_generation_block_state_begin(&state, &plan, &block) ||
        memcmp(&state, &snapshot, sizeof(state)) != 0) {
        return false;
    }

    ds4_dspark_generation_block_state_init(&state);
    ds4_dspark_verifier_plan malformed = plan;
    malformed.emitted_rows++;
    memcpy(&snapshot, &state, sizeof(snapshot));
    if (ds4_dspark_generation_block_state_begin(
            &state, &malformed, &block) ||
        memcmp(&state, &snapshot, sizeof(state)) != 0) {
        return false;
    }

    ds4_dspark_generation_block_state_init(&state);
    malformed = plan;
    malformed.replay_rows++;
    memcpy(&snapshot, &state, sizeof(snapshot));
    if (ds4_dspark_generation_block_state_begin(
            &state, &malformed, &block) ||
        memcmp(&state, &snapshot, sizeof(state)) != 0) {
        return false;
    }

    ds4_dspark_generation_block_state_init(&state);
    malformed = plan;
    malformed.keep_verifier_state = !malformed.keep_verifier_state;
    memcpy(&snapshot, &state, sizeof(snapshot));
    if (ds4_dspark_generation_block_state_begin(
            &state, &malformed, &block) ||
        memcmp(&state, &snapshot, sizeof(state)) != 0) {
        return false;
    }

    ds4_dspark_generation_block_state_init(&state);
    memcpy(&snapshot, &state, sizeof(snapshot));
    if (ds4_dspark_generation_block_state_begin(&state, NULL, &block) ||
        ds4_dspark_generation_block_state_begin(&state, &plan, NULL) ||
        ds4_dspark_generation_block_state_commit(
            &state, 1u, 0u, &consumption) ||
        memcmp(&state, &snapshot, sizeof(state)) != 0 ||
        ds4_dspark_generation_block_state_allows(
            &state, (ds4_dspark_session_operation)-1) ||
        ds4_dspark_generation_block_state_allows(
            &state, DS4_DSPARK_SESSION_OPERATION_COUNT) ||
        ds4_dspark_generation_block_state_allows(
            NULL, DS4_DSPARK_SESSION_OPERATION_SAMPLE) ||
        ds4_dspark_generation_block_state_begin(NULL, &plan, &block) ||
        ds4_dspark_generation_block_state_commit(
            NULL, 1u, 0u, &consumption)) {
        return false;
    }

    ds4_dspark_generation_block_state_free(&state);
    memcpy(&snapshot, &state, sizeof(snapshot));
    if (ds4_dspark_generation_block_state_begin(&state, &plan, &block) ||
        memcmp(&state, &snapshot, sizeof(state)) != 0) {
        return false;
    }
    ds4_dspark_generation_block_state_init(&state);
    if (!ds4_dspark_generation_block_state_begin(&state, &plan, &block) ||
        block.cookie != 1u ||
        !ds4_dspark_generation_block_state_commit(
            &state, block.cookie, 0u, &consumption)) {
        return false;
    }

    ds4_dspark_generation_block_state_init(NULL);
    ds4_dspark_generation_block_state_invalidate(NULL);
    ds4_dspark_generation_block_state_free(NULL);
    return true;
}

/* PENDING_ONLY is a real generation block, but not a verifier round.  Exhaust
 * both adopted prefixes under the three ways the tail can select it: caller
 * maximum, context room, or a terminal pending token. */
static bool ds4_dspark_pending_only_state_self_check(void) {
    const uint64_t frontier = UINT64_C(1000);
    const struct {
        uint32_t maximum_rows;
        uint32_t context_room;
        uint32_t desired_depth;
        bool pending_terminal;
        uint32_t available_rows;
    } cases[] = {
        {1u, 7u, DS4_DSPARK_SUPPORT_CANDIDATE_ROWS, false, 1u},
        {7u, 1u, DS4_DSPARK_SUPPORT_CANDIDATE_ROWS, false, 1u},
        {7u, 7u, DS4_DSPARK_SUPPORT_CANDIDATE_ROWS, true, 7u},
    };
    const ds4_dspark_session_operation guarded_operations[] = {
        DS4_DSPARK_SESSION_OPERATION_SAMPLE,
        DS4_DSPARK_SESSION_OPERATION_EVAL,
        DS4_DSPARK_SESSION_OPERATION_SYNC,
        DS4_DSPARK_SESSION_OPERATION_REWIND,
        DS4_DSPARK_SESSION_OPERATION_SAVE,
    };
    ds4_dspark_verifier_plan verifier_plan;
    const uint32_t raw_cap =
        DS4_DSPARK_HISTORY_ROWS + DS4_DSPARK_RAW_FINALIZER_MAX_ROWS;
    if (!ds4_dspark_verifier_plan_make(
            frontier, 0u, 0u, raw_cap, DS4_DSPARK_HISTORY_ROWS,
            &verifier_plan)) {
        return false;
    }

    for (uint32_t case_index = 0;
         case_index < sizeof(cases) / sizeof(cases[0]); case_index++) {
        ds4_dspark_tail_plan tail;
        if (!ds4_dspark_tail_plan_make(
                cases[case_index].maximum_rows,
                cases[case_index].context_room,
                cases[case_index].desired_depth,
                cases[case_index].pending_terminal, &tail) ||
            !ds4_dspark_tail_plan_coherent(&tail) ||
            tail.kind != DS4_DSPARK_TAIL_PENDING_ONLY ||
            tail.available_rows != cases[case_index].available_rows ||
            tail.draft_depth != 0 || tail.verifier_rows != 0 ||
            tail.maximum_emitted_rows != 1u) {
            return false;
        }
        for (uint32_t consumed = 0; consumed <= 1u; consumed++) {
            ds4_dspark_generation_block_state state;
            ds4_dspark_generation_block_state snapshot;
            ds4_dspark_generation_block block;
            ds4_dspark_generation_block second_block;
            ds4_dspark_verifier_consumption consumption;
            ds4_dspark_generation_block_state_init(&state);
            if (!ds4_dspark_generation_block_state_begin_pending_only(
                    &state, frontier, &tail, &block) ||
                block.cookie == 0 ||
                block.kind != DS4_DSPARK_TAIL_PENDING_ONLY ||
                block.block_rows != 1u || block.evaluated_rows != 0 ||
                !state.active || state.active_cookie != block.cookie ||
                state.last_cookie != block.cookie ||
                state.kind != DS4_DSPARK_TAIL_PENDING_ONLY ||
                state.frontier != frontier || state.block_rows != 1u ||
                state.evaluated_rows != 0 ||
                !ds4_dspark_verifier_plan_is_zero(&state.verifier_plan) ||
                !ds4_dspark_generation_block_state_coherent(&state)) {
                return false;
            }
            for (uint32_t index = 0;
                 index < sizeof(guarded_operations) /
                             sizeof(guarded_operations[0]); index++) {
                if (ds4_dspark_generation_block_state_allows(
                        &state, guarded_operations[index])) {
                    return false;
                }
            }

            memcpy(&snapshot, &state, sizeof(snapshot));
            if (ds4_dspark_generation_block_state_begin_pending_only(
                    &state, frontier, &tail, &second_block) ||
                memcmp(&state, &snapshot, sizeof(state)) != 0 ||
                ds4_dspark_generation_block_state_begin(
                    &state, &verifier_plan, &second_block) ||
                memcmp(&state, &snapshot, sizeof(state)) != 0 ||
                ds4_dspark_generation_block_state_commit(
                    &state, 0u, consumed, &consumption) ||
                memcmp(&state, &snapshot, sizeof(state)) != 0 ||
                ds4_dspark_generation_block_state_commit(
                    &state, block.cookie + 1u, consumed, &consumption) ||
                memcmp(&state, &snapshot, sizeof(state)) != 0 ||
                ds4_dspark_generation_block_state_commit(
                    &state, block.cookie, 2u, &consumption) ||
                memcmp(&state, &snapshot, sizeof(state)) != 0 ||
                ds4_dspark_generation_block_state_commit(
                    &state, block.cookie, consumed, NULL) ||
                memcmp(&state, &snapshot, sizeof(state)) != 0) {
                return false;
            }

            if (!ds4_dspark_generation_block_state_commit(
                    &state, block.cookie, consumed, &consumption) ||
                consumption.committed_frontier != frontier ||
                consumption.consumed_rows != consumed ||
                consumption.committed_evaluated_rows != 0 ||
                consumption.replay_rows != 0 ||
                consumption.final_pending_consumed != (consumed == 1u) ||
                consumption.keep_verifier_state ||
                consumption.restore_then_replay || state.active ||
                state.active_cookie != 0 || state.frontier != 0 ||
                state.kind != DS4_DSPARK_TAIL_NO_OUTPUT ||
                state.block_rows != 0 || state.evaluated_rows != 0 ||
                state.last_cookie != block.cookie ||
                !ds4_dspark_verifier_plan_is_zero(&state.verifier_plan) ||
                !ds4_dspark_generation_block_state_coherent(&state)) {
                return false;
            }
            for (uint32_t index = 0;
                 index < sizeof(guarded_operations) /
                             sizeof(guarded_operations[0]); index++) {
                if (!ds4_dspark_generation_block_state_allows(
                        &state, guarded_operations[index])) {
                    return false;
                }
            }

            memcpy(&snapshot, &state, sizeof(snapshot));
            if (ds4_dspark_generation_block_state_commit(
                    &state, block.cookie, consumed, &consumption) ||
                memcmp(&state, &snapshot, sizeof(state)) != 0 ||
                !ds4_dspark_generation_block_state_begin_pending_only(
                    &state, frontier, &tail, &second_block) ||
                second_block.cookie != block.cookie + 1u) {
                return false;
            }
            memcpy(&snapshot, &state, sizeof(snapshot));
            if (ds4_dspark_generation_block_state_commit(
                    &state, block.cookie, consumed, &consumption) ||
                memcmp(&state, &snapshot, sizeof(state)) != 0) {
                return false;
            }
            const uint64_t invalidated_cookie = state.last_cookie;
            ds4_dspark_generation_block_state_invalidate(&state);
            if (state.last_cookie != invalidated_cookie || state.active ||
                !ds4_dspark_generation_block_state_coherent(&state) ||
                !ds4_dspark_generation_block_state_begin_pending_only(
                    &state, frontier, &tail, &second_block) ||
                second_block.cookie != invalidated_cookie + 1u) {
                return false;
            }
            memcpy(&snapshot, &state, sizeof(snapshot));
            if (ds4_dspark_generation_block_state_commit(
                    &state, invalidated_cookie, consumed, &consumption) ||
                memcmp(&state, &snapshot, sizeof(state)) != 0) {
                return false;
            }
            ds4_dspark_generation_block_state_free(&state);
            memcpy(&snapshot, &state, sizeof(snapshot));
            if (ds4_dspark_generation_block_state_coherent(&state) ||
                ds4_dspark_generation_block_state_begin_pending_only(
                    &state, frontier, &tail, &second_block) ||
                memcmp(&state, &snapshot, sizeof(state)) != 0) {
                return false;
            }
        }
    }

    ds4_dspark_tail_plan tail;
    ds4_dspark_generation_block_state state;
    ds4_dspark_generation_block_state snapshot;
    ds4_dspark_generation_block block;
    ds4_dspark_verifier_consumption consumption;
    if (!ds4_dspark_tail_plan_make(1u, 1u, 0u, false, &tail)) return false;

    /* Cookie and frontier overflow fail without changing ownership bytes. */
    ds4_dspark_generation_block_state_init(&state);
    memcpy(&snapshot, &state, sizeof(snapshot));
    if (ds4_dspark_generation_block_state_begin_pending_only(
            &state, UINT64_MAX, &tail, &block) ||
        memcmp(&state, &snapshot, sizeof(state)) != 0) {
        return false;
    }
    state.last_cookie = UINT64_MAX - 1u;
    if (!ds4_dspark_generation_block_state_begin_pending_only(
            &state, UINT64_MAX - 1u, &tail, &block) ||
        block.cookie != UINT64_MAX ||
        !ds4_dspark_generation_block_state_commit(
            &state, block.cookie, 1u, &consumption) ||
        consumption.committed_frontier != UINT64_MAX - 1u ||
        consumption.committed_evaluated_rows != 0 ||
        !consumption.final_pending_consumed) {
        return false;
    }
    memcpy(&snapshot, &state, sizeof(snapshot));
    if (ds4_dspark_generation_block_state_begin_pending_only(
            &state, frontier, &tail, &block) ||
        memcmp(&state, &snapshot, sizeof(state)) != 0) {
        return false;
    }

    /* Every malformed tail or missing input is rejected byte-identically. */
    const ds4_dspark_tail_plan malformed[] = {
        {
            .kind = DS4_DSPARK_TAIL_NO_OUTPUT,
            .available_rows = 1u,
            .maximum_emitted_rows = 1u,
        },
        {
            .kind = DS4_DSPARK_TAIL_PENDING_ONLY,
            .available_rows = 1u,
            .draft_depth = 1u,
            .maximum_emitted_rows = 1u,
        },
        {
            .kind = DS4_DSPARK_TAIL_PENDING_ONLY,
            .available_rows = 1u,
            .verifier_rows = 1u,
            .maximum_emitted_rows = 1u,
        },
        {
            .kind = DS4_DSPARK_TAIL_PENDING_ONLY,
            .available_rows = 1u,
            .maximum_emitted_rows = 2u,
        },
        {
            .kind = DS4_DSPARK_TAIL_VERIFY,
            .available_rows = 2u,
            .verifier_rows = 1u,
            .maximum_emitted_rows = 2u,
        },
        {
            .kind = (ds4_dspark_tail_kind)3,
            .available_rows = 1u,
            .maximum_emitted_rows = 1u,
        },
    };
    ds4_dspark_generation_block_state_init(&state);
    for (uint32_t index = 0;
         index < sizeof(malformed) / sizeof(malformed[0]); index++) {
        memcpy(&snapshot, &state, sizeof(snapshot));
        if (ds4_dspark_generation_block_state_begin_pending_only(
                &state, frontier, &malformed[index], &block) ||
            memcmp(&state, &snapshot, sizeof(state)) != 0) {
            return false;
        }
    }
    memcpy(&snapshot, &state, sizeof(snapshot));
    return !ds4_dspark_generation_block_state_begin_pending_only(
               &state, frontier, NULL, &block) &&
           memcmp(&state, &snapshot, sizeof(state)) == 0 &&
           !ds4_dspark_generation_block_state_begin_pending_only(
               &state, frontier, &tail, NULL) &&
           memcmp(&state, &snapshot, sizeof(state)) == 0 &&
           !ds4_dspark_generation_block_state_begin_pending_only(
               NULL, frontier, &tail, &block);
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
    return ds4_dspark_generation_block_state_self_check() &&
           ds4_dspark_pending_only_state_self_check() &&
           !ds4_dspark_verifier_plan_make(
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

/* Freeze the exact publication sequence a three-stage proposal must follow.
 *
 * The candidate state owns a single slot and the three stages share one
 * candidate buffer, so a truthful proposal consumes each stage's publication
 * before starting the next.  This check proves both halves: the sequence the
 * proposal uses succeeds end to end, and the alternative that keeps earlier
 * cookies alive is rejected at the second begin rather than silently
 * producing a stale publication. */
static bool ds4_dspark_stage_chain_sequence_self_check(void) {
    ds4_dspark_stage_candidate_state state;
    ds4_dspark_stage_route_plan route;
    ds4_dspark_stage_candidate_publication publication;
    int32_t expert_id[DS4_DSPARK_SUPPORT_CANDIDATE_ROWS]
                     [DS4_DSPARK_SUPPORT_TOP_K];
    float weight[DS4_DSPARK_SUPPORT_CANDIDATE_ROWS]
                [DS4_DSPARK_SUPPORT_TOP_K];
    /* Powers of two summing to exactly the pinned route scale 1.5, so the
     * synthetic plan satisfies the same authentication a real router output
     * must satisfy without relying on any rounding envelope. */
    static const float kRouteWeights[DS4_DSPARK_SUPPORT_TOP_K] = {
        0.5f, 0.25f, 0.25f, 0.25f, 0.125f, 0.125f
    };
    for (uint32_t row = 0;
         row < DS4_DSPARK_SUPPORT_CANDIDATE_ROWS; row++) {
        for (uint32_t slot = 0; slot < DS4_DSPARK_SUPPORT_TOP_K; slot++) {
            expert_id[row][slot] = (int32_t)(row * 8u + slot);
            weight[row][slot] = kRouteWeights[slot];
        }
    }
    if (!ds4_dspark_stage_route_plan_make(
            &expert_id[0][0], &weight[0][0], &route)) {
        return false;
    }

    /* The sequence the proposal performs. */
    ds4_dspark_stage_candidate_state_init(&state);
    uint64_t previous_cookie = 0u;
    for (uint32_t stage = 0;
         stage < DS4_DSPARK_CAPTURE_STAGE_COUNT; stage++) {
        uint64_t cookie = 0u;
        if (!ds4_dspark_stage_candidate_state_begin(&state, stage, &cookie) ||
            cookie == 0u || cookie == previous_cookie ||
            !ds4_dspark_stage_candidate_state_set_shadow(
                &state, cookie, &route) ||
            !ds4_dspark_stage_candidate_state_publish(&state, cookie) ||
            !ds4_dspark_stage_candidate_state_ready(
                &state, cookie, &publication) ||
            publication.stage != stage ||
            publication.cookie != cookie) {
            return false;
        }
        /* A stale cookie must never resolve, even while the slot is ready. */
        if (previous_cookie != 0u &&
            ds4_dspark_stage_candidate_state_ready(
                &state, previous_cookie, &publication)) {
            return false;
        }
        if (!ds4_dspark_stage_candidate_state_consume(&state, cookie) ||
            ds4_dspark_stage_candidate_state_consume(&state, cookie)) {
            return false;
        }
        previous_cookie = cookie;
    }

    /* Holding a publication across the next begin must fail closed: this is
     * the defect the single-slot contract exists to prevent. */
    ds4_dspark_stage_candidate_state_init(&state);
    uint64_t held = 0u;
    uint64_t second = 0u;
    if (!ds4_dspark_stage_candidate_state_begin(&state, 0u, &held) ||
        !ds4_dspark_stage_candidate_state_set_shadow(&state, held, &route) ||
        !ds4_dspark_stage_candidate_state_publish(&state, held) ||
        ds4_dspark_stage_candidate_state_begin(&state, 1u, &second) ||
        second != 0u) {
        return false;
    }
    /* After the held publication is consumed the next stage may begin. */
    if (!ds4_dspark_stage_candidate_state_consume(&state, held) ||
        !ds4_dspark_stage_candidate_state_begin(&state, 1u, &second) ||
        second == 0u || second == held) {
        return false;
    }
    if (!ds4_dspark_stage_candidate_state_abort(&state, second) ||
        state.phase != DS4_DSPARK_STAGE_CANDIDATE_EMPTY) {
        return false;
    }
    return true;
}

/* Every HC head allocation must already be charged by the runtime inventory.
 *
 * The head owns exactly four real allocations.  Three are the greedy head
 * scratch the inventory reserves (Markov embeddings, one vocabulary row, the
 * confidence column) and the fourth reuses the accounted sampled q
 * distribution, which is what the corrected block becomes for the sampled
 * lane.  Everything else the head touches is a view into storage that is
 * already accounted, so it adds no bytes.  Admission charges before it
 * executes, so a head that allocated outside this set would spend a budget
 * the plan never reserved: this check makes that fail closed instead. */
static bool ds4_dspark_head_storage_contract_check(uint64_t vocab) {
    ds4_dspark_runtime_memory memory;
    if (vocab == 0u || !ds4_dspark_runtime_memory_make(&memory)) return false;
    const uint64_t rows = DS4_DSPARK_SUPPORT_CANDIDATE_ROWS;
    const uint64_t markov_bytes =
        rows * DS4_DSPARK_MARKOV_RANK * sizeof(float);
    const uint64_t logits_row_bytes = vocab * sizeof(float);
    const uint64_t confidence_bytes = rows * sizeof(float);
    const uint64_t corrected_bytes = rows * vocab * sizeof(float);
    const uint64_t greedy_charged =
        ds4_dspark_metal_page_charge(markov_bytes) +
        ds4_dspark_metal_page_charge(logits_row_bytes) +
        ds4_dspark_metal_page_charge(confidence_bytes);
    /* This equivalence is page granular, so a small-tensor drift that stays
     * inside one 16 KiB page (for example a Markov rank change) does not trip
     * it.  That drift is caught byte exactly elsewhere: metadata admission
     * pins the model rank against DS4_DSPARK_MARKOV_RANK, and the graph-time
     * validator pins every head tensor to its exact byte length. */
    return greedy_charged == memory.greedy_head_scratch_page_bytes &&
           ds4_dspark_metal_page_charge(corrected_bytes) ==
               memory.sampled_q_distribution_page_bytes;
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
    bool workspace_owner_ok = true;
    bool verifier_frontier_ok = true;
    bool stage_candidate_ok = true;
#ifndef DS4_NO_GPU
    raw_finalizer_present =
        metal_graph_dspark_raw_finalizer_code_present();
#endif
#ifdef DS4_TEST_HOOKS
    workspace_owner_ok = ds4_dspark_workspace_owner_self_check();
    verifier_frontier_ok =
        ds4_dspark_verifier_frontier_self_check();
    stage_candidate_ok = ds4_dspark_stage_candidate_state_self_check();
#endif
    ds4_dspark_capture_state_init(&capture_state, true);
    return DS4_DSPARK_CAPTURE_CONTRACT.stage_count == 3u &&
           DS4_DSPARK_SELECTED_ADDRESS_VIEW_BYTES == 6264u &&
           DS4_DSPARK_SELECTED_ADDRESS_VIEW_BYTES <=
               DS4_DSPARK_METAL_PAGE_BYTES &&
           DS4_DSPARK_CAPTURE_CONTRACT.hidden_width == 4096u &&
           DS4_DSPARK_CAPTURE_CONTRACT.hc_lanes == 4u &&
           verifier_frontier_ok &&
           stage_candidate_ok &&
           ds4_dspark_stage_chain_sequence_self_check() &&
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
           workspace_owner_ok &&
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
           memory.selected_address_page_bytes == UINT64_C(16384) &&
           memory.core_bytes == UINT64_C(9781248) &&
           memory.verifier_frontier_snapshot_page_bytes ==
               UINT64_C(12550144) &&
           memory.sampled_host_delta_page_bytes == UINT64_C(1081344) &&
           memory.total_bytes == UINT64_C(29671424) &&
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
