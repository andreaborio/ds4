#ifndef DS4_QWEN4EXP_H
#define DS4_QWEN4EXP_H

#include <stdbool.h>
#include <stdint.h>

#define DS4_QWEN4EXP_PROFILE_ID "qwen4exp-base-v1"
#define DS4_QWEN4EXP_ARCHITECTURE "qwen4exp"

enum {
    DS4_QWEN4EXP_N_LAYER = 48,
    DS4_QWEN4EXP_N_GDN_LAYER = 36,
    DS4_QWEN4EXP_N_QSA_LAYER = 12,
    DS4_QWEN4EXP_HIDDEN_SIZE = 2560,
    DS4_QWEN4EXP_RESIDUAL_STREAMS = 4,
    DS4_QWEN4EXP_WIDE_SIZE = 10240,
    DS4_QWEN4EXP_GR_LOW_RANK = 320,
    DS4_QWEN4EXP_VOCAB_SIZE = 248320,
    DS4_QWEN4EXP_CONTEXT_LENGTH = 262144,

    DS4_QWEN4EXP_QSA_QUERY_HEADS = 24,
    DS4_QWEN4EXP_QSA_KV_HEADS = 2,
    DS4_QWEN4EXP_QSA_HEAD_DIM = 256,
    DS4_QWEN4EXP_QSA_ROTARY_DIM = 64,
    DS4_QWEN4EXP_QSA_INDEX_QUERY_HEADS = 4,
    DS4_QWEN4EXP_QSA_INDEX_KEY_HEADS = 1,
    DS4_QWEN4EXP_QSA_INDEX_HEAD_DIM = 128,
    DS4_QWEN4EXP_QSA_COMPRESSION = 4,
    DS4_QWEN4EXP_QSA_TOKEN_BUDGET = 2048,
    DS4_QWEN4EXP_QSA_BLOCK_BUDGET = 512,
    DS4_QWEN4EXP_QSA_MAX_SELECTED_WIDTH = 2051,

    DS4_QWEN4EXP_GDN_KEY_HEADS = 16,
    DS4_QWEN4EXP_GDN_VALUE_HEADS = 48,
    DS4_QWEN4EXP_GDN_HEAD_DIM = 128,
    DS4_QWEN4EXP_GDN_REPEAT_RATIO = 3,
    DS4_QWEN4EXP_GDN_CONV_KERNEL = 4,

    DS4_QWEN4EXP_EXPERTS = 512,
    DS4_QWEN4EXP_EXPERTS_USED = 10,
    DS4_QWEN4EXP_EXPERT_DIM = 640,

    DS4_QWEN4EXP_PLE_RUNTIME_LAYER = 1,
    DS4_QWEN4EXP_PLE_LAYER_ORDINAL = 0,
    DS4_QWEN4EXP_PLE_NGRAM_SIZE = 3,
    DS4_QWEN4EXP_PLE_HEADS_PER_NGRAM = 8,
    DS4_QWEN4EXP_PLE_HEADS = 16,
    DS4_QWEN4EXP_PLE_HEAD_DIM = 160,
    DS4_QWEN4EXP_PLE_FLAT_SIZE = 2560,
    DS4_QWEN4EXP_PLE_CONV_KERNEL = 4,
    DS4_QWEN4EXP_PLE_CONV_DILATION = 3,
    DS4_QWEN4EXP_PLE_CONV_STATE = 9,
    DS4_QWEN4EXP_PLE_HASH_MULTIPLIERS = 3,
    DS4_QWEN4EXP_PLE_PAD_TOKEN = 248044,
};

typedef enum {
    DS4_QWEN4EXP_LAYER_GDN = 0,
    DS4_QWEN4EXP_LAYER_QSA = 1,
} ds4_qwen4exp_layer_type;

/* Immutable semantic descriptor for the one pinned base-text profile.  It is
 * model metadata, not a support or artifact-codec claim. */
typedef struct {
    const char *profile_id;
    const char *architecture;
    const char *hf_revision;
    const char *transformers_commit;

    uint32_t n_layer;
    uint32_t hidden_size;
    uint32_t residual_streams;
    uint32_t gr_low_rank;
    uint32_t vocab_size;
    uint32_t context_length;

    uint32_t qsa_query_heads;
    uint32_t qsa_kv_heads;
    uint32_t qsa_head_dim;
    uint32_t qsa_rotary_dim;
    uint32_t qsa_index_query_heads;
    uint32_t qsa_index_key_heads;
    uint32_t qsa_index_head_dim;
    uint32_t qsa_compression;
    uint32_t qsa_token_budget;

    uint32_t gdn_key_heads;
    uint32_t gdn_value_heads;
    uint32_t gdn_head_dim;
    uint32_t gdn_conv_kernel;

    uint32_t experts;
    uint32_t experts_used;
    uint32_t expert_dim;

    uint32_t ple_runtime_layer;
    uint32_t ple_layer_ordinal;
    uint32_t ple_ngram_size;
    uint32_t ple_heads_per_ngram;
    uint32_t ple_head_dim;
    uint32_t ple_rows;
    uint32_t ple_conv_kernel;
    uint32_t ple_conv_dilation;
    uint32_t ple_pad_token;

    float rms_epsilon;
    float rope_theta;
    uint32_t mrope_section[3];
    ds4_qwen4exp_layer_type layer_type[DS4_QWEN4EXP_N_LAYER];
    uint64_t ple_multiplier[DS4_QWEN4EXP_PLE_HASH_MULTIPLIERS];
    uint32_t ple_head_prime[DS4_QWEN4EXP_PLE_HEADS];
    uint32_t ple_head_offset[DS4_QWEN4EXP_PLE_HEADS];
} ds4_qwen4exp_profile;

/* Per-sequence logical state.  Byte sizes are explicit so later physical
 * codecs can be planned without silently assuming BF16 or F32. */
typedef struct {
    uint32_t context;
    uint32_t qsa_kv_element_bytes;
    uint32_t qsa_index_element_bytes;
    uint32_t gdn_element_bytes;
    uint32_t ple_element_bytes;

    uint64_t gdn_conv_values_per_layer;
    uint64_t gdn_recurrent_values_per_layer;
    uint64_t gdn_conv_bytes;
    uint64_t gdn_recurrent_bytes;
    uint64_t qsa_kv_bytes_per_token;
    uint64_t qsa_kv_bytes;
    uint64_t qsa_raw_index_bytes_per_token;
    uint64_t qsa_raw_index_bytes;
    uint64_t ple_history_bytes;
    uint64_t ple_conv_bytes;
    uint64_t fixed_bytes;
    uint64_t context_bytes;
    uint64_t total_bytes;
} ds4_qwen4exp_state_plan;

const ds4_qwen4exp_profile *ds4_qwen4exp_profile_get(void);

bool ds4_qwen4exp_profile_validate(const ds4_qwen4exp_profile *profile);

bool ds4_qwen4exp_layer_type_get(
        uint32_t                   layer,
        ds4_qwen4exp_layer_type  *type);

bool ds4_qwen4exp_layer_is_gdn(uint32_t layer);
bool ds4_qwen4exp_layer_is_qsa(uint32_t layer);
bool ds4_qwen4exp_layer_has_ple(uint32_t layer);

bool ds4_qwen4exp_state_plan_make(
        uint32_t                     context,
        uint32_t                     qsa_kv_element_bytes,
        uint32_t                     qsa_index_element_bytes,
        uint32_t                     gdn_element_bytes,
        uint32_t                     ple_element_bytes,
        ds4_qwen4exp_state_plan     *plan);

bool ds4_qwen4exp_expert_parameter_count(uint64_t *count);
bool ds4_qwen4exp_routed_parameter_count(uint64_t *count);
bool ds4_qwen4exp_active_routed_parameter_count(uint64_t *count);
bool ds4_qwen4exp_ple_parameter_count(uint64_t *count);

#endif
