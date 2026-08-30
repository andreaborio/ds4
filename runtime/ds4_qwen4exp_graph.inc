#include "ds4_qwen4exp_graph.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define DS4_Q4E_GRAPH_MAGIC UINT64_C(0x5134454752415048)

typedef struct {
    size_t frontier;
    size_t qsa_count;
    size_t ple_history_count;
    uint32_t ple_history[2];
} ds4_q4e_graph_control;

typedef struct {
    ds4_q4e_graph_control *control;
    float *wide;
    float *activation;
    float *block_output;
    float *injection;
    float *router_logits;
    uint32_t *route_id;
    float *route_weight;
    float *gdn_conv;
    float *gdn_recurrent;
    float *qsa_key;
    float *qsa_value;
    float *qsa_raw_index;
    uint32_t *qsa_position;
    uint32_t *ple_row;
    float *ple_conv;
    float *logits;
} ds4_q4e_graph_view;

struct ds4_qwen4exp_graph {
    uint64_t magic;
    ds4_qwen4exp_graph_geometry geometry;
    ds4_qwen4exp_graph_plan plan;
    ds4_qwen4exp_graph_backend backend;
    unsigned char *bank[2];
    unsigned int active_bank;
};

static bool ds4_q4e_graph_size_add(size_t a, size_t b, size_t *result) {
    if (!result || a > SIZE_MAX - b) return false;
    *result = a + b;
    return true;
}

static bool ds4_q4e_graph_size_mul(size_t a, size_t b, size_t *result) {
    if (!result || (a != 0u && b > SIZE_MAX / a)) return false;
    *result = a * b;
    return true;
}

static bool ds4_q4e_graph_size_mul3(
        size_t a, size_t b, size_t c, size_t *result) {
    size_t ab = 0u;
    return ds4_q4e_graph_size_mul(a, b, &ab) &&
           ds4_q4e_graph_size_mul(ab, c, result);
}

static bool ds4_q4e_graph_align(size_t value, size_t *result) {
    const size_t alignment = sizeof(uint64_t);
    const size_t remainder = value % alignment;
    if (remainder == 0u) {
        *result = value;
        return true;
    }
    return ds4_q4e_graph_size_add(value, alignment - remainder, result);
}

static bool ds4_q4e_graph_add_owner(
        size_t count, size_t element_size, size_t *owner, size_t *total) {
    size_t bytes = 0u;
    size_t next = 0u;
    if (!ds4_q4e_graph_size_mul(count, element_size, &bytes) ||
        !ds4_q4e_graph_align(*total, &next) ||
        !ds4_q4e_graph_size_add(next, bytes, total)) {
        return false;
    }
    *owner = bytes;
    return true;
}

void ds4_qwen4exp_graph_geometry_frozen(
        ds4_qwen4exp_graph_geometry *geometry) {
    if (!geometry) return;
    *geometry = (ds4_qwen4exp_graph_geometry) {
        DS4_Q4E_GRAPH_LAYERS,
        DS4_Q4E_GRAPH_GDN_LAYERS,
        DS4_Q4E_GRAPH_QSA_LAYERS,
        DS4_Q4E_GRAPH_HIDDEN,
        DS4_Q4E_GRAPH_STREAMS,
        DS4_Q4E_GRAPH_GR_RANK,
        DS4_Q4E_GRAPH_CONTEXT,
        DS4_Q4E_GRAPH_VOCAB,
        DS4_Q4E_GRAPH_EXPERTS,
        DS4_Q4E_GRAPH_EXPERTS_USED,
        DS4_Q4E_GRAPH_EXPERT_DIM,
        DS4_Q4E_GRAPH_GDN_KEY_HEADS,
        DS4_Q4E_GRAPH_GDN_VALUE_HEADS,
        DS4_Q4E_GRAPH_GDN_KEY_DIM,
        DS4_Q4E_GRAPH_GDN_VALUE_DIM,
        DS4_Q4E_GRAPH_GDN_CONV_KERNEL,
        DS4_Q4E_GRAPH_QSA_QUERY_HEADS,
        DS4_Q4E_GRAPH_QSA_KV_HEADS,
        DS4_Q4E_GRAPH_QSA_HEAD_DIM,
        DS4_Q4E_GRAPH_QSA_INDEX_HEADS,
        DS4_Q4E_GRAPH_QSA_INDEX_DIM,
        DS4_Q4E_GRAPH_QSA_COMPRESSION,
        DS4_Q4E_GRAPH_QSA_BLOCK_BUDGET,
        DS4_Q4E_GRAPH_PLE_HEADS,
        DS4_Q4E_GRAPH_PLE_HEAD_DIM,
        DS4_Q4E_GRAPH_PLE_CONV_KERNEL,
        DS4_Q4E_GRAPH_PLE_CONV_DILATION,
        DS4_Q4E_GRAPH_PLE_PAD_TOKEN,
    };
}

bool ds4_qwen4exp_graph_geometry_validate(
        const ds4_qwen4exp_graph_geometry *geometry) {
    ds4_qwen4exp_graph_geometry frozen;
    if (!geometry) return false;
    ds4_qwen4exp_graph_geometry_frozen(&frozen);
#define DS4_Q4E_GEOMETRY_FIELD(field) geometry->field == frozen.field
    return DS4_Q4E_GEOMETRY_FIELD(layers) &&
           DS4_Q4E_GEOMETRY_FIELD(gdn_layers) &&
           DS4_Q4E_GEOMETRY_FIELD(qsa_layers) &&
           DS4_Q4E_GEOMETRY_FIELD(hidden) &&
           DS4_Q4E_GEOMETRY_FIELD(streams) &&
           DS4_Q4E_GEOMETRY_FIELD(gr_rank) &&
           DS4_Q4E_GEOMETRY_FIELD(context) &&
           DS4_Q4E_GEOMETRY_FIELD(vocab) &&
           DS4_Q4E_GEOMETRY_FIELD(experts) &&
           DS4_Q4E_GEOMETRY_FIELD(experts_used) &&
           DS4_Q4E_GEOMETRY_FIELD(expert_dim) &&
           DS4_Q4E_GEOMETRY_FIELD(gdn_key_heads) &&
           DS4_Q4E_GEOMETRY_FIELD(gdn_value_heads) &&
           DS4_Q4E_GEOMETRY_FIELD(gdn_key_dim) &&
           DS4_Q4E_GEOMETRY_FIELD(gdn_value_dim) &&
           DS4_Q4E_GEOMETRY_FIELD(gdn_conv_kernel) &&
           DS4_Q4E_GEOMETRY_FIELD(qsa_query_heads) &&
           DS4_Q4E_GEOMETRY_FIELD(qsa_kv_heads) &&
           DS4_Q4E_GEOMETRY_FIELD(qsa_head_dim) &&
           DS4_Q4E_GEOMETRY_FIELD(qsa_index_heads) &&
           DS4_Q4E_GEOMETRY_FIELD(qsa_index_dim) &&
           DS4_Q4E_GEOMETRY_FIELD(qsa_compression) &&
           DS4_Q4E_GEOMETRY_FIELD(qsa_block_budget) &&
           DS4_Q4E_GEOMETRY_FIELD(ple_heads) &&
           DS4_Q4E_GEOMETRY_FIELD(ple_head_dim) &&
           DS4_Q4E_GEOMETRY_FIELD(ple_conv_kernel) &&
           DS4_Q4E_GEOMETRY_FIELD(ple_conv_dilation) &&
           DS4_Q4E_GEOMETRY_FIELD(ple_pad_token);
#undef DS4_Q4E_GEOMETRY_FIELD
}

bool ds4_qwen4exp_graph_plan_make(
        const ds4_qwen4exp_graph_geometry *geometry,
        ds4_qwen4exp_graph_plan           *plan) {
    ds4_qwen4exp_graph_plan next;
    size_t total = sizeof(ds4_q4e_graph_control);
    size_t count = 0u;
    size_t value = 0u;
    size_t ignored = 0u;
    if (!geometry || !plan) return false;
    memset(&next, 0, sizeof(next));

    if (!ds4_q4e_graph_size_mul(geometry->streams, geometry->hidden, &count) ||
        !ds4_q4e_graph_add_owner(count, sizeof(float),
                                  &next.wide_residual_bytes, &total) ||
        !ds4_q4e_graph_size_add(geometry->hidden, geometry->hidden, &count) ||
        !ds4_q4e_graph_size_add(count, geometry->streams, &count) ||
        !ds4_q4e_graph_add_owner(count, sizeof(float),
                                  &next.private_activation_bytes, &total)) {
        return false;
    }

    if (geometry->gdn_conv_kernel == 0u ||
        !ds4_q4e_graph_size_mul(2u, geometry->gdn_key_heads, &count) ||
        !ds4_q4e_graph_size_mul(count, geometry->gdn_key_dim, &count) ||
        !ds4_q4e_graph_size_mul(geometry->gdn_value_heads,
                                geometry->gdn_value_dim, &value) ||
        !ds4_q4e_graph_size_add(count, value, &count) ||
        !ds4_q4e_graph_size_mul3(count, geometry->gdn_conv_kernel - 1u,
                                 geometry->gdn_layers, &count) ||
        !ds4_q4e_graph_add_owner(count, sizeof(float),
                                  &next.gdn_conv_bytes, &total) ||
        !ds4_q4e_graph_size_mul3(geometry->gdn_value_heads,
                                 geometry->gdn_key_dim,
                                 geometry->gdn_value_dim, &count) ||
        !ds4_q4e_graph_size_mul(count, geometry->gdn_layers, &count) ||
        !ds4_q4e_graph_add_owner(count, sizeof(float),
                                  &next.gdn_recurrent_bytes, &total)) {
        return false;
    }

    if (!ds4_q4e_graph_size_mul3(geometry->context, geometry->qsa_kv_heads,
                                  geometry->qsa_head_dim, &count) ||
        !ds4_q4e_graph_size_mul(count, 2u, &count) ||
        !ds4_q4e_graph_add_owner(count, sizeof(float),
                                  &next.qsa_kv_bytes, &total) ||
        !ds4_q4e_graph_size_mul(geometry->context,
                                 geometry->qsa_index_dim, &count) ||
        !ds4_q4e_graph_add_owner(count, sizeof(float),
                                  &next.qsa_raw_index_bytes, &total) ||
        !ds4_q4e_graph_add_owner(geometry->context, sizeof(uint32_t),
                                  &next.qsa_position_bytes, &total)) {
        return false;
    }

    next.ple_history_bytes = sizeof(((ds4_q4e_graph_control *)0)->ple_history) +
                             sizeof(((ds4_q4e_graph_control *)0)->ple_history_count);
    if (geometry->ple_conv_kernel == 0u ||
        !ds4_q4e_graph_size_mul(geometry->streams, geometry->hidden, &count) ||
        !ds4_q4e_graph_size_mul(count, geometry->ple_conv_dilation, &count) ||
        !ds4_q4e_graph_size_mul(count, geometry->ple_conv_kernel - 1u, &count) ||
        !ds4_q4e_graph_add_owner(count, sizeof(float),
                                  &next.ple_conv_bytes, &total) ||
        !ds4_q4e_graph_add_owner(geometry->ple_heads, sizeof(uint32_t),
                                  &ignored, &total)) {
        return false;
    }

    if (!ds4_q4e_graph_size_add(geometry->experts,
                                 geometry->experts_used, &count) ||
        !ds4_q4e_graph_size_mul(count, sizeof(float), &value) ||
        !ds4_q4e_graph_size_mul(geometry->experts_used,
                                sizeof(uint32_t), &count) ||
        !ds4_q4e_graph_size_add(value, count, &next.route_bytes) ||
        !ds4_q4e_graph_add_owner(geometry->experts, sizeof(float),
                                  &ignored, &total) ||
        !ds4_q4e_graph_add_owner(geometry->experts_used, sizeof(uint32_t),
                                  &ignored, &total) ||
        !ds4_q4e_graph_add_owner(geometry->experts_used, sizeof(float),
                                  &ignored, &total) ||
        !ds4_q4e_graph_add_owner(geometry->vocab, sizeof(float),
                                  &next.logits_bytes, &total) ||
        !ds4_q4e_graph_align(total, &next.bank_bytes)) {
        return false;
    }
    next.control_bytes = sizeof(ds4_q4e_graph_control);
    if (!ds4_q4e_graph_size_mul(next.bank_bytes, 2u, &value) ||
        !ds4_q4e_graph_size_add(sizeof(ds4_qwen4exp_graph), value,
                                &next.allocated_bytes) ||
        !ds4_qwen4exp_graph_geometry_validate(geometry)) {
        return false;
    }
    *plan = next;
    return true;
}

static bool ds4_q4e_graph_take(
        unsigned char *bank, size_t bank_bytes, size_t bytes,
        size_t *offset, void **pointer) {
    size_t aligned = 0u;
    size_t end = 0u;
    if (!ds4_q4e_graph_align(*offset, &aligned) ||
        !ds4_q4e_graph_size_add(aligned, bytes, &end) || end > bank_bytes) {
        return false;
    }
    *pointer = bank + aligned;
    *offset = end;
    return true;
}

static bool ds4_q4e_graph_bind(
        const ds4_qwen4exp_graph *graph, unsigned int index,
        ds4_q4e_graph_view *view) {
    unsigned char *bank;
    size_t offset = sizeof(ds4_q4e_graph_control);
    size_t qsa_half;
    void *pointer = NULL;
    if (!graph || !view || index > 1u || !graph->bank[index]) return false;
    bank = graph->bank[index];
    memset(view, 0, sizeof(*view));
    view->control = (ds4_q4e_graph_control *)bank;
#define DS4_Q4E_TAKE(field, bytes, type)                                      \
    do {                                                                       \
        if (!ds4_q4e_graph_take(bank, graph->plan.bank_bytes, (bytes),         \
                                 &offset, &pointer)) return false;              \
        view->field = (type *)pointer;                                          \
    } while (0)
    DS4_Q4E_TAKE(wide, graph->plan.wide_residual_bytes, float);
    DS4_Q4E_TAKE(activation, graph->geometry.hidden * sizeof(float), float);
    DS4_Q4E_TAKE(block_output, graph->geometry.hidden * sizeof(float), float);
    DS4_Q4E_TAKE(injection, graph->geometry.streams * sizeof(float), float);
    DS4_Q4E_TAKE(gdn_conv, graph->plan.gdn_conv_bytes, float);
    DS4_Q4E_TAKE(gdn_recurrent, graph->plan.gdn_recurrent_bytes, float);
    qsa_half = graph->plan.qsa_kv_bytes / 2u;
    DS4_Q4E_TAKE(qsa_key, qsa_half, float);
    DS4_Q4E_TAKE(qsa_value, qsa_half, float);
    DS4_Q4E_TAKE(qsa_raw_index, graph->plan.qsa_raw_index_bytes, float);
    DS4_Q4E_TAKE(qsa_position, graph->plan.qsa_position_bytes, uint32_t);
    DS4_Q4E_TAKE(ple_conv, graph->plan.ple_conv_bytes, float);
    DS4_Q4E_TAKE(ple_row, graph->geometry.ple_heads * sizeof(uint32_t), uint32_t);
    DS4_Q4E_TAKE(router_logits, graph->geometry.experts * sizeof(float), float);
    DS4_Q4E_TAKE(route_id, graph->geometry.experts_used * sizeof(uint32_t), uint32_t);
    DS4_Q4E_TAKE(route_weight, graph->geometry.experts_used * sizeof(float), float);
    DS4_Q4E_TAKE(logits, graph->plan.logits_bytes, float);
#undef DS4_Q4E_TAKE
    return offset <= graph->plan.bank_bytes;
}

static bool ds4_q4e_graph_valid(const ds4_qwen4exp_graph *graph) {
    return graph && graph->magic == DS4_Q4E_GRAPH_MAGIC &&
           graph->active_bank <= 1u && graph->bank[0] && graph->bank[1] &&
           graph->backend.stage &&
           ds4_qwen4exp_graph_geometry_validate(&graph->geometry);
}

static void ds4_q4e_graph_bank_reset(
        const ds4_qwen4exp_graph *graph, unsigned int index) {
    ds4_q4e_graph_view view;
    memset(graph->bank[index], 0, graph->plan.bank_bytes);
    if (ds4_q4e_graph_bind(graph, index, &view)) {
        view.control->ple_history[0] = graph->geometry.ple_pad_token;
        view.control->ple_history[1] = graph->geometry.ple_pad_token;
    }
}

bool ds4_qwen4exp_graph_create(
        ds4_qwen4exp_graph                **graph,
        const ds4_qwen4exp_graph_geometry *geometry,
        ds4_qwen4exp_graph_backend          backend) {
    ds4_qwen4exp_graph_plan plan;
    ds4_qwen4exp_graph *next;
    if (!graph || !geometry || !backend.stage ||
        !ds4_qwen4exp_graph_plan_make(geometry, &plan)) {
        return false;
    }
    next = (ds4_qwen4exp_graph *)calloc(1u, sizeof(*next));
    if (!next) return false;
    next->bank[0] = (unsigned char *)malloc(plan.bank_bytes);
    next->bank[1] = (unsigned char *)malloc(plan.bank_bytes);
    if (!next->bank[0] || !next->bank[1]) {
        free(next->bank[0]);
        free(next->bank[1]);
        free(next);
        return false;
    }
    next->magic = DS4_Q4E_GRAPH_MAGIC;
    next->geometry = *geometry;
    next->plan = plan;
    next->backend = backend;
    next->active_bank = 0u;
    ds4_q4e_graph_bank_reset(next, 0u);
    ds4_q4e_graph_bank_reset(next, 1u);
    *graph = next;
    return true;
}

void ds4_qwen4exp_graph_destroy(ds4_qwen4exp_graph *graph) {
    if (!graph) return;
    if (graph->magic == DS4_Q4E_GRAPH_MAGIC) {
        if (graph->bank[0]) {
            memset(graph->bank[0], 0, graph->plan.bank_bytes);
            free(graph->bank[0]);
        }
        if (graph->bank[1]) {
            memset(graph->bank[1], 0, graph->plan.bank_bytes);
            free(graph->bank[1]);
        }
    }
    memset(graph, 0, sizeof(*graph));
    free(graph);
}

bool ds4_qwen4exp_graph_reset(ds4_qwen4exp_graph *graph) {
    if (!ds4_q4e_graph_valid(graph)) return false;
    ds4_q4e_graph_bank_reset(graph, 0u);
    ds4_q4e_graph_bank_reset(graph, 1u);
    graph->active_bank = 0u;
    return true;
}

static bool ds4_q4e_graph_finite(const float *values, size_t count) {
    size_t i;
    if (!values) return false;
    for (i = 0u; i < count; ++i) {
        if (!isfinite(values[i])) return false;
    }
    return true;
}

static bool ds4_q4e_graph_state_finite(
        const ds4_qwen4exp_graph *graph, const ds4_q4e_graph_view *view) {
    return ds4_q4e_graph_finite(view->wide,
             graph->plan.wide_residual_bytes / sizeof(float)) &&
           ds4_q4e_graph_finite(view->activation, graph->geometry.hidden) &&
           ds4_q4e_graph_finite(view->block_output, graph->geometry.hidden) &&
           ds4_q4e_graph_finite(view->injection, graph->geometry.streams) &&
           ds4_q4e_graph_finite(view->gdn_conv,
             graph->plan.gdn_conv_bytes / sizeof(float)) &&
           ds4_q4e_graph_finite(view->gdn_recurrent,
             graph->plan.gdn_recurrent_bytes / sizeof(float)) &&
           ds4_q4e_graph_finite(view->qsa_key,
             graph->plan.qsa_kv_bytes / (2u * sizeof(float))) &&
           ds4_q4e_graph_finite(view->qsa_value,
             graph->plan.qsa_kv_bytes / (2u * sizeof(float))) &&
           ds4_q4e_graph_finite(view->qsa_raw_index,
             graph->plan.qsa_raw_index_bytes / sizeof(float)) &&
           ds4_q4e_graph_finite(view->ple_conv,
             graph->plan.ple_conv_bytes / sizeof(float)) &&
           ds4_q4e_graph_finite(view->router_logits, graph->geometry.experts) &&
           ds4_q4e_graph_finite(view->route_weight,
             graph->geometry.experts_used) &&
           ds4_q4e_graph_finite(view->logits, graph->geometry.vocab);
}

static bool ds4_q4e_graph_route_validate(
        const ds4_qwen4exp_graph *graph, const ds4_q4e_graph_view *view) {
    uint32_t expected_id[DS4_Q4E_GRAPH_EXPERTS_USED];
    float expected_weight[DS4_Q4E_GRAPH_EXPERTS_USED];
    float maximum;
    float total = 0.0f;
    float selected_total = 0.0f;
    size_t slot;
    size_t expert;
    if (!ds4_q4e_graph_finite(view->router_logits, graph->geometry.experts) ||
        !ds4_q4e_graph_finite(view->route_weight,
                              graph->geometry.experts_used)) {
        return false;
    }
    maximum = view->router_logits[0];
    for (expert = 1u; expert < graph->geometry.experts; ++expert) {
        if (view->router_logits[expert] > maximum) {
            maximum = view->router_logits[expert];
        }
    }
    for (expert = 0u; expert < graph->geometry.experts; ++expert) {
        total += expf(view->router_logits[expert] - maximum);
    }
    if (!(total > 0.0f) || !isfinite(total)) return false;
    for (slot = 0u; slot < graph->geometry.experts_used; ++slot) {
        size_t best = graph->geometry.experts;
        float best_probability = 0.0f;
        for (expert = 0u; expert < graph->geometry.experts; ++expert) {
            bool seen = false;
            size_t prior;
            for (prior = 0u; prior < slot; ++prior) {
                if (expected_id[prior] == (uint32_t)expert) {
                    seen = true;
                    break;
                }
            }
            if (!seen) {
                const float probability =
                    expf(view->router_logits[expert] - maximum) / total;
                if (best == graph->geometry.experts ||
                    probability > best_probability ||
                    (probability == best_probability && expert < best)) {
                    best = expert;
                    best_probability = probability;
                }
            }
        }
        expected_id[slot] = (uint32_t)best;
        expected_weight[slot] = best_probability;
        selected_total += best_probability;
    }
    if (!(selected_total > 0.0f) || !isfinite(selected_total)) return false;
    total = 0.0f;
    for (slot = 0u; slot < graph->geometry.experts_used; ++slot) {
        expected_weight[slot] /= selected_total;
        if (view->route_id[slot] != expected_id[slot] ||
            view->route_weight[slot] < 0.0f ||
            fabsf(view->route_weight[slot] - expected_weight[slot]) > 2.0e-5f) {
            return false;
        }
        total += view->route_weight[slot];
    }
    return isfinite(total) && fabsf(total - 1.0f) <= 2.0e-5f;
}

static void ds4_q4e_graph_ple_expected(
        const ds4_q4e_graph_control *control, uint32_t token,
        uint32_t row[DS4_Q4E_GRAPH_PLE_HEADS]) {
    static const uint32_t prime[DS4_Q4E_GRAPH_PLE_HEADS] = {
        17u, 19u, 23u, 29u, 31u, 37u, 41u, 43u,
        47u, 53u, 59u, 61u, 67u, 71u, 73u, 79u,
    };
    static const uint64_t multiplier[3] = {
        UINT64_C(23703573157769), UINT64_C(20109073645365),
        UINT64_C(8052911324071),
    };
    const uint64_t previous1 = control->ple_history_count >= 1u
        ? control->ple_history[0] : DS4_Q4E_GRAPH_PLE_PAD_TOKEN;
    const uint64_t previous2 = control->ple_history_count >= 2u
        ? control->ple_history[1] : DS4_Q4E_GRAPH_PLE_PAD_TOKEN;
    const uint64_t bigram = (uint64_t)token * multiplier[0] ^
                             previous1 * multiplier[1];
    const uint64_t trigram = bigram ^ previous2 * multiplier[2];
    uint32_t offset = 0u;
    size_t head;
    for (head = 0u; head < DS4_Q4E_GRAPH_PLE_HEADS; ++head) {
        const uint64_t hash = head < 8u ? bigram : trigram;
        row[head] = offset + (uint32_t)(hash % prime[head]);
        offset += prime[head];
    }
}

static bool ds4_q4e_graph_dispatch(
        ds4_qwen4exp_graph *graph, ds4_q4e_graph_view *view,
        ds4_qwen4exp_graph_stage stage, size_t ordinal, size_t layer,
        size_t gdn_layer, size_t position, uint32_t token) {
    ds4_qwen4exp_graph_stage_io io;
    const size_t conv_per_layer = graph->plan.gdn_conv_bytes /
        (graph->geometry.gdn_layers * sizeof(float));
    const size_t recurrent_per_layer = graph->plan.gdn_recurrent_bytes /
        (graph->geometry.gdn_layers * sizeof(float));
    const size_t state_layer = gdn_layer < graph->geometry.gdn_layers
        ? gdn_layer : graph->geometry.gdn_layers - 1u;
    memset(&io, 0, sizeof(io));
    io.geometry = &graph->geometry;
    io.stage = stage;
    io.stage_ordinal = ordinal;
    io.layer = layer;
    io.gdn_layer = gdn_layer;
    io.position = position;
    io.token_id = token;
    io.wide = view->wide;
    io.activation = view->activation;
    io.block_output = view->block_output;
    io.injection = view->injection;
    io.router_logits = view->router_logits;
    io.route_id = view->route_id;
    io.route_weight = view->route_weight;
    io.gdn_conv = view->gdn_conv + state_layer * conv_per_layer;
    io.gdn_recurrent = view->gdn_recurrent +
                       state_layer * recurrent_per_layer;
    io.qsa_key = view->qsa_key;
    io.qsa_value = view->qsa_value;
    io.qsa_raw_index = view->qsa_raw_index;
    io.qsa_position = view->qsa_position;
    io.qsa_count = &view->control->qsa_count;
    io.ple_row = view->ple_row;
    io.ple_history = view->control->ple_history;
    io.ple_history_count = &view->control->ple_history_count;
    io.ple_conv = view->ple_conv;
    io.logits = view->logits;
    if (!graph->backend.stage(graph->backend.context, &io) ||
        !ds4_q4e_graph_state_finite(graph, view)) {
        return false;
    }
    if (stage == DS4_Q4E_STAGE_ROUTER &&
        !ds4_q4e_graph_route_validate(graph, view)) {
        return false;
    }
    if (stage == DS4_Q4E_STAGE_QSA &&
        (view->control->qsa_count != position + 1u ||
         view->qsa_position[position] != (uint32_t)position)) {
        return false;
    }
    return true;
}

static bool ds4_q4e_graph_range(
        const void *pointer, size_t bytes, uintptr_t *begin, uintptr_t *end) {
    const uintptr_t first = (uintptr_t)pointer;
    if (!pointer || bytes == 0u || first > UINTPTR_MAX - bytes) return false;
    *begin = first;
    *end = first + bytes;
    return true;
}

static bool ds4_q4e_graph_overlap(
        const void *a, size_t a_bytes, const void *b, size_t b_bytes) {
    uintptr_t a_begin;
    uintptr_t a_end;
    uintptr_t b_begin;
    uintptr_t b_end;
    if (!ds4_q4e_graph_range(a, a_bytes, &a_begin, &a_end) ||
        !ds4_q4e_graph_range(b, b_bytes, &b_begin, &b_end)) {
        return true;
    }
    return a_begin < b_end && b_begin < a_end;
}

bool ds4_qwen4exp_graph_run(
        ds4_qwen4exp_graph *graph,
        const uint32_t     *token,
        size_t              n_token,
        float              *output_logits,
        size_t              output_count) {
    ds4_q4e_graph_view view;
    unsigned int private_bank;
    size_t token_bytes = 0u;
    size_t output_bytes = 0u;
    size_t start;
    size_t token_index;
    size_t ordinal = 0u;
    if (!ds4_q4e_graph_valid(graph) || !token || !output_logits ||
        n_token == 0u || output_count != graph->geometry.vocab ||
        !ds4_q4e_graph_size_mul(n_token, sizeof(token[0]), &token_bytes) ||
        !ds4_q4e_graph_size_mul(output_count, sizeof(output_logits[0]),
                                &output_bytes) ||
        ds4_q4e_graph_overlap(token, token_bytes, output_logits, output_bytes) ||
        ds4_q4e_graph_overlap(token, token_bytes, graph, sizeof(*graph)) ||
        ds4_q4e_graph_overlap(output_logits, output_bytes, graph, sizeof(*graph)) ||
        ds4_q4e_graph_overlap(token, token_bytes, graph->bank[0],
                              graph->plan.bank_bytes) ||
        ds4_q4e_graph_overlap(token, token_bytes, graph->bank[1],
                              graph->plan.bank_bytes) ||
        ds4_q4e_graph_overlap(output_logits, output_bytes, graph->bank[0],
                              graph->plan.bank_bytes) ||
        ds4_q4e_graph_overlap(output_logits, output_bytes, graph->bank[1],
                              graph->plan.bank_bytes)) {
        return false;
    }
    if (!ds4_q4e_graph_bind(graph, graph->active_bank, &view)) return false;
    start = view.control->frontier;
    if (start > graph->geometry.context ||
        n_token > graph->geometry.context - start) {
        return false;
    }
    for (token_index = 0u; token_index < n_token; ++token_index) {
        if (token[token_index] >= graph->geometry.vocab) return false;
    }

    private_bank = graph->active_bank ^ 1u;
    memcpy(graph->bank[private_bank], graph->bank[graph->active_bank],
           graph->plan.bank_bytes);
    if (!ds4_q4e_graph_bind(graph, private_bank, &view)) return false;

    for (token_index = 0u; token_index < n_token; ++token_index) {
        const size_t position = start + token_index;
        size_t layer;
        size_t gdn_layer = 0u;
        memset(view.wide, 0, graph->plan.wide_residual_bytes);
        memset(view.activation, 0, graph->geometry.hidden * sizeof(float));
        memset(view.block_output, 0, graph->geometry.hidden * sizeof(float));
        memset(view.injection, 0, graph->geometry.streams * sizeof(float));
        memset(view.router_logits, 0,
               graph->geometry.experts * sizeof(float));
        memset(view.route_id, 0,
               graph->geometry.experts_used * sizeof(uint32_t));
        memset(view.route_weight, 0,
               graph->geometry.experts_used * sizeof(float));
        memset(view.logits, 0, graph->plan.logits_bytes);

#define DS4_Q4E_DISPATCH(stage_value, layer_value, gdn_value)                 \
        do {                                                                  \
            if (!ds4_q4e_graph_dispatch(graph, &view, (stage_value),          \
                    ordinal++, (layer_value), (gdn_value), position,           \
                    token[token_index])) return false;                         \
        } while (0)
        DS4_Q4E_DISPATCH(DS4_Q4E_STAGE_EMBEDDING, 0u, 0u);
        DS4_Q4E_DISPATCH(DS4_Q4E_STAGE_FOUR_STREAM_RESIDUAL, 0u, 0u);
        for (layer = 0u; layer < graph->geometry.layers; ++layer) {
            if (layer == 1u) {
                uint32_t expected_row[DS4_Q4E_GRAPH_PLE_HEADS];
                uint32_t previous_history[2];
                size_t previous_count = view.control->ple_history_count;
                size_t head;
                previous_history[0] = view.control->ple_history[0];
                previous_history[1] = view.control->ple_history[1];
                ds4_q4e_graph_ple_expected(view.control, token[token_index],
                                            expected_row);
                memcpy(view.ple_row, expected_row, sizeof(expected_row));
                DS4_Q4E_DISPATCH(DS4_Q4E_STAGE_PLE, layer, gdn_layer);
                for (head = 0u; head < graph->geometry.ple_heads; ++head) {
                    if (view.ple_row[head] != expected_row[head]) return false;
                }
                if (token[token_index] == graph->geometry.ple_pad_token) {
                    if (view.control->ple_history_count != 0u ||
                        view.control->ple_history[0] !=
                            graph->geometry.ple_pad_token ||
                        view.control->ple_history[1] !=
                            graph->geometry.ple_pad_token) return false;
                } else if (view.control->ple_history_count !=
                              (previous_count < 2u ? previous_count + 1u : 2u) ||
                           view.control->ple_history[0] != token[token_index] ||
                           view.control->ple_history[1] !=
                              (previous_count > 0u ? previous_history[0]
                                                   : graph->geometry.ple_pad_token)) {
                    return false;
                }
            }
            DS4_Q4E_DISPATCH(DS4_Q4E_STAGE_ATTN_GR_PREPARE, layer, gdn_layer);
            if (layer < graph->geometry.gdn_layers) {
                DS4_Q4E_DISPATCH(DS4_Q4E_STAGE_GDN, layer, gdn_layer);
            } else {
                DS4_Q4E_DISPATCH(DS4_Q4E_STAGE_QSA, layer, gdn_layer);
            }
            DS4_Q4E_DISPATCH(DS4_Q4E_STAGE_ATTN_GR_APPLY, layer, gdn_layer);
            DS4_Q4E_DISPATCH(DS4_Q4E_STAGE_MOE_GR_PREPARE, layer, gdn_layer);
            DS4_Q4E_DISPATCH(DS4_Q4E_STAGE_ROUTER, layer, gdn_layer);
            DS4_Q4E_DISPATCH(DS4_Q4E_STAGE_ROUTED_SHARED_MOE, layer, gdn_layer);
            DS4_Q4E_DISPATCH(DS4_Q4E_STAGE_MOE_GR_APPLY, layer, gdn_layer);
            if (layer < graph->geometry.gdn_layers) ++gdn_layer;
        }
        DS4_Q4E_DISPATCH(DS4_Q4E_STAGE_FINAL_GR_MIXER,
                          graph->geometry.layers, gdn_layer);
        DS4_Q4E_DISPATCH(DS4_Q4E_STAGE_OUTPUT_HEAD,
                          graph->geometry.layers, gdn_layer);
#undef DS4_Q4E_DISPATCH
        view.control->frontier = position + 1u;
    }

    /* This index flip is the sole publication point.  No callback, allocation,
     * validation or fallible operation follows it. */
    graph->active_bank = private_bank;
    memcpy(output_logits, view.logits, output_bytes);
    return true;
}

size_t ds4_qwen4exp_graph_frontier(const ds4_qwen4exp_graph *graph) {
    ds4_q4e_graph_view view;
    if (!ds4_q4e_graph_valid(graph) ||
        !ds4_q4e_graph_bind(graph, graph->active_bank, &view)) {
        return 0u;
    }
    return view.control->frontier;
}

bool ds4_qwen4exp_graph_public_logits(
        const ds4_qwen4exp_graph *graph,
        float                    *output,
        size_t                    output_count) {
    ds4_q4e_graph_view view;
    size_t bytes = 0u;
    if (!ds4_q4e_graph_valid(graph) || !output ||
        output_count != graph->geometry.vocab ||
        !ds4_q4e_graph_size_mul(output_count, sizeof(output[0]), &bytes) ||
        !ds4_q4e_graph_bind(graph, graph->active_bank, &view) ||
        ds4_q4e_graph_overlap(output, bytes, graph, sizeof(*graph)) ||
        ds4_q4e_graph_overlap(output, bytes, graph->bank[0],
                              graph->plan.bank_bytes) ||
        ds4_q4e_graph_overlap(output, bytes, graph->bank[1],
                              graph->plan.bank_bytes)) {
        return false;
    }
    memcpy(output, view.logits, bytes);
    return true;
}

const ds4_qwen4exp_graph_plan *ds4_qwen4exp_graph_byte_report(
        const ds4_qwen4exp_graph *graph) {
    return ds4_q4e_graph_valid(graph) ? &graph->plan : NULL;
}

bool ds4_qwen4exp_graph_public_digest(
        const ds4_qwen4exp_graph *graph,
        uint64_t                 *digest) {
    const unsigned char *bytes;
    uint64_t value = UINT64_C(1469598103934665603);
    size_t i;
    if (!ds4_q4e_graph_valid(graph) || !digest ||
        ds4_q4e_graph_overlap(digest, sizeof(*digest), graph,
                              sizeof(*graph)) ||
        ds4_q4e_graph_overlap(digest, sizeof(*digest), graph->bank[0],
                              graph->plan.bank_bytes) ||
        ds4_q4e_graph_overlap(digest, sizeof(*digest), graph->bank[1],
                              graph->plan.bank_bytes)) {
        return false;
    }
    bytes = graph->bank[graph->active_bank];
    for (i = 0u; i < graph->plan.bank_bytes; ++i) {
        value ^= bytes[i];
        value *= UINT64_C(1099511628211);
    }
    *digest = value;
    return true;
}

size_t ds4_qwen4exp_graph_stages_per_token(void) {
    return 2u + 1u + 7u * DS4_Q4E_GRAPH_LAYERS + 2u;
}
