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
    DS4_DSPARK_TARGET_LAYER_COUNT = 43,
    DS4_DSPARK_TARGET_TOP_K = 6,
    DS4_DSPARK_SUPPORT_TOP_K = 6,
    /* One token can retain top-k records from every target layer, plus the
     * record currently being filled by asynchronous I/O. */
    DS4_DSPARK_TARGET_CACHE_FLOOR =
        DS4_DSPARK_TARGET_LAYER_COUNT * DS4_DSPARK_TARGET_TOP_K + 1,
    /* Three support stages have independent ownership and the same in-flight
     * allowance. */
    DS4_DSPARK_SUPPORT_CACHE_FLOOR =
        DS4_DSPARK_CAPTURE_STAGE_COUNT * DS4_DSPARK_SUPPORT_TOP_K + 1,
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
    return DS4_DSPARK_CAPTURE_CONTRACT.stage_count == 3u &&
           DS4_DSPARK_CAPTURE_CONTRACT.hidden_width == 4096u &&
           DS4_DSPARK_CAPTURE_CONTRACT.hc_lanes == 4u &&
           !ds4_dspark_capture_slot_after_layer(39u, &slot) &&
           ds4_dspark_capture_slot_after_layer(40u, &slot) && slot == 0u &&
           ds4_dspark_capture_slot_after_layer(41u, &slot) && slot == 1u &&
           ds4_dspark_capture_slot_after_layer(42u, &slot) && slot == 2u &&
           !ds4_dspark_capture_slot_after_layer(43u, &slot) &&
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
