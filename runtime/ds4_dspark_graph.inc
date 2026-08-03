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
    ds4_dspark_capture_state capture_state;
    bool capture = false;
    uint32_t capture_slot = UINT32_MAX;
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
           ds4_dspark_history_state_self_check() &&
           ds4_dspark_failed_blit_state_self_check() &&
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
