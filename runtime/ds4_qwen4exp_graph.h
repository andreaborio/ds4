#ifndef DS4_QWEN4EXP_GRAPH_H
#define DS4_QWEN4EXP_GRAPH_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Phase-5 freezes a deliberately small geometry while retaining the production
 * equations: four residual streams, GDN/GDN/GDN/QSA, 512-way/top-10 MoE and
 * sixteen PLE n-gram heads.  It is a correctness graph, not product admission. */
enum {
    DS4_Q4E_GRAPH_LAYERS = 4,
    DS4_Q4E_GRAPH_GDN_LAYERS = 3,
    DS4_Q4E_GRAPH_QSA_LAYERS = 1,
    DS4_Q4E_GRAPH_HIDDEN = 4,
    DS4_Q4E_GRAPH_STREAMS = 4,
    DS4_Q4E_GRAPH_WIDE = 16,
    DS4_Q4E_GRAPH_GR_RANK = 2,
    DS4_Q4E_GRAPH_CONTEXT = 8,
    DS4_Q4E_GRAPH_VOCAB = 13,
    DS4_Q4E_GRAPH_EXPERTS = 512,
    DS4_Q4E_GRAPH_EXPERTS_USED = 10,
    DS4_Q4E_GRAPH_EXPERT_DIM = 3,
    DS4_Q4E_GRAPH_GDN_KEY_HEADS = 1,
    DS4_Q4E_GRAPH_GDN_VALUE_HEADS = 3,
    DS4_Q4E_GRAPH_GDN_KEY_DIM = 2,
    DS4_Q4E_GRAPH_GDN_VALUE_DIM = 2,
    DS4_Q4E_GRAPH_GDN_CONV_KERNEL = 4,
    DS4_Q4E_GRAPH_GDN_CONV_CHANNELS = 10,
    DS4_Q4E_GRAPH_QSA_QUERY_HEADS = 2,
    DS4_Q4E_GRAPH_QSA_KV_HEADS = 1,
    DS4_Q4E_GRAPH_QSA_HEAD_DIM = 2,
    DS4_Q4E_GRAPH_QSA_INDEX_HEADS = 2,
    DS4_Q4E_GRAPH_QSA_INDEX_DIM = 2,
    DS4_Q4E_GRAPH_QSA_COMPRESSION = 4,
    DS4_Q4E_GRAPH_QSA_BLOCK_BUDGET = 2,
    DS4_Q4E_GRAPH_DENSE_QSA_LIMIT = 2051,
    DS4_Q4E_GRAPH_PLE_HEADS = 16,
    DS4_Q4E_GRAPH_PLE_HEAD_DIM = 1,
    DS4_Q4E_GRAPH_PLE_CONV_KERNEL = 4,
    DS4_Q4E_GRAPH_PLE_CONV_DILATION = 3,
    DS4_Q4E_GRAPH_PLE_CONV_STATE = 9,
    DS4_Q4E_GRAPH_PLE_PAD_TOKEN = 12,
};

typedef struct {
    size_t layers;
    size_t gdn_layers;
    size_t qsa_layers;
    size_t hidden;
    size_t streams;
    size_t gr_rank;
    size_t context;
    size_t vocab;
    size_t experts;
    size_t experts_used;
    size_t expert_dim;
    size_t gdn_key_heads;
    size_t gdn_value_heads;
    size_t gdn_key_dim;
    size_t gdn_value_dim;
    size_t gdn_conv_kernel;
    size_t qsa_query_heads;
    size_t qsa_kv_heads;
    size_t qsa_head_dim;
    size_t qsa_index_heads;
    size_t qsa_index_dim;
    size_t qsa_compression;
    size_t qsa_block_budget;
    size_t ple_heads;
    size_t ple_head_dim;
    size_t ple_conv_kernel;
    size_t ple_conv_dilation;
    uint32_t ple_pad_token;
} ds4_qwen4exp_graph_geometry;

/* Every owner is reported separately.  bank_bytes is one complete public or
 * private state image; allocated_bytes includes both banks and coordinator
 * bookkeeping.  Offsets are intentionally private to the implementation. */
typedef struct {
    size_t wide_residual_bytes;
    size_t private_activation_bytes;
    size_t gdn_conv_bytes;
    size_t gdn_recurrent_bytes;
    size_t qsa_kv_bytes;
    size_t qsa_raw_index_bytes;
    size_t qsa_position_bytes;
    size_t ple_history_bytes;
    size_t ple_conv_bytes;
    size_t route_bytes;
    size_t logits_bytes;
    size_t control_bytes;
    size_t bank_bytes;
    size_t allocated_bytes;
} ds4_qwen4exp_graph_plan;

typedef enum {
    DS4_Q4E_STAGE_EMBEDDING = 0,
    DS4_Q4E_STAGE_FOUR_STREAM_RESIDUAL,
    DS4_Q4E_STAGE_PLE,
    DS4_Q4E_STAGE_ATTN_GR_PREPARE,
    DS4_Q4E_STAGE_GDN,
    DS4_Q4E_STAGE_QSA,
    DS4_Q4E_STAGE_ATTN_GR_APPLY,
    DS4_Q4E_STAGE_MOE_GR_PREPARE,
    DS4_Q4E_STAGE_ROUTER,
    DS4_Q4E_STAGE_ROUTED_SHARED_MOE,
    DS4_Q4E_STAGE_MOE_GR_APPLY,
    DS4_Q4E_STAGE_FINAL_GR_MIXER,
    DS4_Q4E_STAGE_OUTPUT_HEAD,
    DS4_Q4E_STAGE_COUNT
} ds4_qwen4exp_graph_stage;

/* All pointers are transaction-private.  A backend may mutate only the fields
 * appropriate to the requested stage.  Layouts are row-major and sizes are
 * fixed by geometry: wide [stream][hidden], activation/block [hidden], GDN
 * state [value_head][key_dim][value_dim], QSA caches [slot][head][dim], PLE
 * conv [wide][9], router [512]/[10], logits [vocab]. */
typedef struct {
    const ds4_qwen4exp_graph_geometry *geometry;
    ds4_qwen4exp_graph_stage stage;
    size_t stage_ordinal;
    size_t layer;
    size_t gdn_layer;
    size_t position;
    uint32_t token_id;

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
    size_t *qsa_count;

    uint32_t *ple_row;
    uint32_t *ple_history;
    size_t *ple_history_count;
    float *ple_conv;
    float *logits;
} ds4_qwen4exp_graph_stage_io;

typedef bool (*ds4_qwen4exp_graph_stage_fn)(
        void                           *context,
        ds4_qwen4exp_graph_stage_io    *io);

typedef struct {
    ds4_qwen4exp_graph_stage_fn stage;
    void *context;
} ds4_qwen4exp_graph_backend;

typedef struct ds4_qwen4exp_graph ds4_qwen4exp_graph;

void ds4_qwen4exp_graph_geometry_frozen(
        ds4_qwen4exp_graph_geometry *geometry);

bool ds4_qwen4exp_graph_geometry_validate(
        const ds4_qwen4exp_graph_geometry *geometry);

bool ds4_qwen4exp_graph_plan_make(
        const ds4_qwen4exp_graph_geometry *geometry,
        ds4_qwen4exp_graph_plan           *plan);

bool ds4_qwen4exp_graph_create(
        ds4_qwen4exp_graph               **graph,
        const ds4_qwen4exp_graph_geometry *geometry,
        ds4_qwen4exp_graph_backend         backend);

void ds4_qwen4exp_graph_destroy(ds4_qwen4exp_graph *graph);

bool ds4_qwen4exp_graph_reset(ds4_qwen4exp_graph *graph);

/* The whole supplied chunk is one transaction.  On success output receives
 * the last token's logits and one bank-index flip publishes all state. */
bool ds4_qwen4exp_graph_run(
        ds4_qwen4exp_graph *graph,
        const uint32_t     *token,
        size_t              n_token,
        float              *output_logits,
        size_t              output_count);

size_t ds4_qwen4exp_graph_frontier(const ds4_qwen4exp_graph *graph);

bool ds4_qwen4exp_graph_public_logits(
        const ds4_qwen4exp_graph *graph,
        float                    *output,
        size_t                    output_count);

const ds4_qwen4exp_graph_plan *ds4_qwen4exp_graph_byte_report(
        const ds4_qwen4exp_graph *graph);

/* Test/integration audit helpers.  The digest covers public frontier, logits,
 * wide/private activation and every persistent GDN/QSA/PLE owner. */
bool ds4_qwen4exp_graph_public_digest(
        const ds4_qwen4exp_graph *graph,
        uint64_t                 *digest);

size_t ds4_qwen4exp_graph_stages_per_token(void);

#endif
