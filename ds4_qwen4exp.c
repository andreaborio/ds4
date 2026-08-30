#include "ds4_qwen4exp.h"

#include <limits.h>
#include <stddef.h>
#include <string.h>

static const ds4_qwen4exp_profile qwen4exp_profile = {
    .profile_id = DS4_QWEN4EXP_PROFILE_ID,
    .source_architecture = DS4_QWEN4EXP_SOURCE_ARCHITECTURE,
    .gguf_architecture = DS4_QWEN4EXP_GGUF_ARCHITECTURE,
    .hf_revision = "de4b8e4d43b917e7706784d8bb445c9af86a3540",
    .transformers_commit = "42ca97014c85d71a88ad60d55f08cb9fb4d26e2c",
    .text_only = true,
    .source_mtp_layers = DS4_QWEN4EXP_SOURCE_MTP_LAYERS,
    .base_profile_includes_mtp = false,

    .n_layer = DS4_QWEN4EXP_N_LAYER,
    .hidden_size = DS4_QWEN4EXP_HIDDEN_SIZE,
    .residual_streams = DS4_QWEN4EXP_RESIDUAL_STREAMS,
    .gr_low_rank = DS4_QWEN4EXP_GR_LOW_RANK,
    .vocab_size = DS4_QWEN4EXP_VOCAB_SIZE,
    .context_length = DS4_QWEN4EXP_CONTEXT_LENGTH,

    .qsa_query_heads = DS4_QWEN4EXP_QSA_QUERY_HEADS,
    .qsa_kv_heads = DS4_QWEN4EXP_QSA_KV_HEADS,
    .qsa_head_dim = DS4_QWEN4EXP_QSA_HEAD_DIM,
    .qsa_rotary_dim = DS4_QWEN4EXP_QSA_ROTARY_DIM,
    .qsa_index_query_heads = DS4_QWEN4EXP_QSA_INDEX_QUERY_HEADS,
    .qsa_index_key_heads = DS4_QWEN4EXP_QSA_INDEX_KEY_HEADS,
    .qsa_index_head_dim = DS4_QWEN4EXP_QSA_INDEX_HEAD_DIM,
    .qsa_compression = DS4_QWEN4EXP_QSA_COMPRESSION,
    .qsa_token_budget = DS4_QWEN4EXP_QSA_TOKEN_BUDGET,
    .qsa_mrope_interleaved = true,
    .qsa_output_gate = DS4_QWEN4EXP_GATE_SIGMOID,

    .gdn_key_heads = DS4_QWEN4EXP_GDN_KEY_HEADS,
    .gdn_value_heads = DS4_QWEN4EXP_GDN_VALUE_HEADS,
    .gdn_head_dim = DS4_QWEN4EXP_GDN_HEAD_DIM,
    .gdn_conv_kernel = DS4_QWEN4EXP_GDN_CONV_KERNEL,
    .gdn_recurrent_element_bytes = 4u,
    .gdn_output_gate = DS4_QWEN4EXP_GATE_SIGMOID,

    .experts = DS4_QWEN4EXP_EXPERTS,
    .experts_used = DS4_QWEN4EXP_EXPERTS_USED,
    .expert_dim = DS4_QWEN4EXP_EXPERT_DIM,
    .shared_expert_dim = DS4_QWEN4EXP_SHARED_EXPERT_DIM,
    .router_softmax_element_bytes = 4u,
    .router_full_softmax = true,
    .router_normalize_selected = true,
    .router_tie_policy = DS4_QWEN4EXP_ROUTER_TIE_ASCENDING_EXPERT_ID,

    .ple_source_layer_id = DS4_QWEN4EXP_PLE_SOURCE_LAYER_ID,
    .ple_runtime_layer = DS4_QWEN4EXP_PLE_RUNTIME_LAYER,
    .ple_layer_ordinal = DS4_QWEN4EXP_PLE_LAYER_ORDINAL,
    .ple_seed = DS4_QWEN4EXP_PLE_SEED,
    .ple_vocab_base = DS4_QWEN4EXP_PLE_VOCAB_BASE,
    .ple_split_parts = DS4_QWEN4EXP_PLE_SPLIT_PARTS,
    .ple_row_alignment = DS4_QWEN4EXP_PLE_ROW_ALIGNMENT,
    .ple_ngram_size = DS4_QWEN4EXP_PLE_NGRAM_SIZE,
    .ple_heads_per_ngram = DS4_QWEN4EXP_PLE_HEADS_PER_NGRAM,
    .ple_head_dim = DS4_QWEN4EXP_PLE_HEAD_DIM,
    .ple_rows = 320001536u,
    .ple_conv_kernel = DS4_QWEN4EXP_PLE_CONV_KERNEL,
    .ple_conv_dilation = DS4_QWEN4EXP_PLE_CONV_DILATION,
    .ple_pad_token = DS4_QWEN4EXP_PLE_PAD_TOKEN,

    .rms_epsilon = 1.0e-6f,
    .rope_theta = 10000000.0f,
    .mrope_section = {11u, 11u, 10u},
    .layer_type = {
        DS4_QWEN4EXP_LAYER_GDN, DS4_QWEN4EXP_LAYER_GDN,
        DS4_QWEN4EXP_LAYER_GDN, DS4_QWEN4EXP_LAYER_QSA,
        DS4_QWEN4EXP_LAYER_GDN, DS4_QWEN4EXP_LAYER_GDN,
        DS4_QWEN4EXP_LAYER_GDN, DS4_QWEN4EXP_LAYER_QSA,
        DS4_QWEN4EXP_LAYER_GDN, DS4_QWEN4EXP_LAYER_GDN,
        DS4_QWEN4EXP_LAYER_GDN, DS4_QWEN4EXP_LAYER_QSA,
        DS4_QWEN4EXP_LAYER_GDN, DS4_QWEN4EXP_LAYER_GDN,
        DS4_QWEN4EXP_LAYER_GDN, DS4_QWEN4EXP_LAYER_QSA,
        DS4_QWEN4EXP_LAYER_GDN, DS4_QWEN4EXP_LAYER_GDN,
        DS4_QWEN4EXP_LAYER_GDN, DS4_QWEN4EXP_LAYER_QSA,
        DS4_QWEN4EXP_LAYER_GDN, DS4_QWEN4EXP_LAYER_GDN,
        DS4_QWEN4EXP_LAYER_GDN, DS4_QWEN4EXP_LAYER_QSA,
        DS4_QWEN4EXP_LAYER_GDN, DS4_QWEN4EXP_LAYER_GDN,
        DS4_QWEN4EXP_LAYER_GDN, DS4_QWEN4EXP_LAYER_QSA,
        DS4_QWEN4EXP_LAYER_GDN, DS4_QWEN4EXP_LAYER_GDN,
        DS4_QWEN4EXP_LAYER_GDN, DS4_QWEN4EXP_LAYER_QSA,
        DS4_QWEN4EXP_LAYER_GDN, DS4_QWEN4EXP_LAYER_GDN,
        DS4_QWEN4EXP_LAYER_GDN, DS4_QWEN4EXP_LAYER_QSA,
        DS4_QWEN4EXP_LAYER_GDN, DS4_QWEN4EXP_LAYER_GDN,
        DS4_QWEN4EXP_LAYER_GDN, DS4_QWEN4EXP_LAYER_QSA,
        DS4_QWEN4EXP_LAYER_GDN, DS4_QWEN4EXP_LAYER_GDN,
        DS4_QWEN4EXP_LAYER_GDN, DS4_QWEN4EXP_LAYER_QSA,
        DS4_QWEN4EXP_LAYER_GDN, DS4_QWEN4EXP_LAYER_GDN,
        DS4_QWEN4EXP_LAYER_GDN, DS4_QWEN4EXP_LAYER_QSA,
    },
    .ple_multiplier = {
        UINT64_C(23703573157769),
        UINT64_C(20109073645365),
        UINT64_C(8052911324071),
    },
    .ple_head_prime = {
        20000003u, 20000023u, 20000033u, 20000047u,
        20000059u, 20000063u, 20000069u, 20000077u,
        20000081u, 20000093u, 20000107u, 20000147u,
        20000153u, 20000159u, 20000161u, 20000171u,
    },
    .ple_head_offset = {
        0u, 20000003u, 40000026u, 60000059u,
        80000106u, 100000165u, 120000228u, 140000297u,
        160000374u, 180000455u, 200000548u, 220000655u,
        240000802u, 260000955u, 280001114u, 300001275u,
    },
};

static bool checked_add_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out || a > UINT64_MAX - b) return false;
    *out = a + b;
    return true;
}

static bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out || (a != 0u && b > UINT64_MAX / a)) return false;
    *out = a * b;
    return true;
}

static bool same_string(const char *a, const char *b) {
    return a && b && strcmp(a, b) == 0;
}

static bool state_element_bytes_valid(uint32_t bytes) {
    return bytes == 1u || bytes == 2u || bytes == 4u;
}

static bool profile_formulas_validate(const ds4_qwen4exp_profile *profile) {
    uint64_t value;
    uint64_t total;
    uint32_t i;

    if (profile->hidden_size == 0u || profile->residual_streams == 0u ||
        profile->vocab_size == 0u || profile->n_layer == 0u ||
        profile->expert_dim == 0u || profile->experts == 0u ||
        profile->experts_used == 0u || profile->ple_head_dim == 0u ||
        profile->ple_rows == 0u || profile->ple_row_alignment == 0u) {
        return false;
    }

    if (!checked_mul_u64(profile->hidden_size,
                         profile->residual_streams, &value) ||
        value != DS4_QWEN4EXP_WIDE_SIZE) {
        return false;
    }
    if (!checked_mul_u64(profile->qsa_query_heads,
                         profile->qsa_head_dim, &value) ||
        value != 6144u) {
        return false;
    }
    if (!checked_mul_u64(profile->qsa_kv_heads,
                         profile->qsa_head_dim, &value) ||
        value != 512u) {
        return false;
    }
    if (profile->qsa_compression == 0u ||
        profile->qsa_token_budget / profile->qsa_compression !=
                DS4_QWEN4EXP_QSA_BLOCK_BUDGET ||
        profile->qsa_token_budget + profile->qsa_compression - 1u !=
                DS4_QWEN4EXP_QSA_MAX_SELECTED_WIDTH) {
        return false;
    }
    if (profile->gdn_key_heads == 0u ||
        profile->gdn_value_heads / profile->gdn_key_heads !=
                DS4_QWEN4EXP_GDN_REPEAT_RATIO ||
        profile->gdn_value_heads % profile->gdn_key_heads != 0u) {
        return false;
    }
    if (profile->ple_ngram_size < 2u ||
        !checked_mul_u64(profile->ple_ngram_size - 1u,
                         profile->ple_heads_per_ngram, &value) ||
        value != DS4_QWEN4EXP_PLE_HEADS) {
        return false;
    }
    if (!checked_mul_u64(DS4_QWEN4EXP_PLE_HEADS,
                         profile->ple_head_dim, &value) ||
        value != DS4_QWEN4EXP_PLE_FLAT_SIZE) {
        return false;
    }
    if (profile->ple_conv_kernel == 0u ||
        !checked_mul_u64(profile->ple_conv_kernel - 1u,
                         profile->ple_conv_dilation, &value) ||
        value != DS4_QWEN4EXP_PLE_CONV_STATE) {
        return false;
    }
    if (!checked_mul_u64(profile->hidden_size, profile->expert_dim, &value) ||
        !checked_mul_u64(value, 3u, &value) || value != UINT64_C(4915200)) {
        return false;
    }
    if (!checked_mul_u64(value, profile->experts, &total) ||
        !checked_mul_u64(total, profile->n_layer, &total) ||
        total != UINT64_C(120795955200)) {
        return false;
    }
    if (!checked_mul_u64(value, profile->experts_used, &total) ||
        !checked_mul_u64(total, profile->n_layer, &total) ||
        total != UINT64_C(2359296000)) {
        return false;
    }
    if (!checked_mul_u64(profile->ple_rows, profile->ple_head_dim, &value) ||
        value != UINT64_C(51200245760)) {
        return false;
    }
    total = 0u;
    for (i = 0u; i < DS4_QWEN4EXP_PLE_HEADS; ++i) {
        if (profile->ple_head_offset[i] != total ||
            !checked_add_u64(total, profile->ple_head_prime[i], &total)) {
            return false;
        }
    }
    if (total != UINT64_C(320001446) || profile->ple_rows < total ||
        profile->ple_rows - total != 90u ||
        profile->ple_rows % profile->ple_row_alignment != 0u) {
        return false;
    }
    for (i = 0u; i < DS4_QWEN4EXP_PLE_HASH_MULTIPLIERS; ++i) {
        if ((profile->ple_multiplier[i] & UINT64_C(1)) == 0u ||
            !checked_mul_u64(profile->vocab_size - 1u,
                             profile->ple_multiplier[i], &value) ||
            value > INT64_MAX) {
            return false;
        }
    }
    return true;
}

const ds4_qwen4exp_profile *ds4_qwen4exp_profile_get(void) {
    return &qwen4exp_profile;
}

bool ds4_qwen4exp_profile_validate(const ds4_qwen4exp_profile *profile) {
    static const uint32_t expected_mrope[3] = {11u, 11u, 10u};
    uint32_t i;
    uint32_t gdn_layers = 0u;
    uint32_t qsa_layers = 0u;

    if (!profile ||
        !same_string(profile->profile_id, DS4_QWEN4EXP_PROFILE_ID) ||
        !same_string(profile->source_architecture,
                     DS4_QWEN4EXP_SOURCE_ARCHITECTURE) ||
        !same_string(profile->gguf_architecture,
                     DS4_QWEN4EXP_GGUF_ARCHITECTURE) ||
        !same_string(profile->hf_revision,
                     "de4b8e4d43b917e7706784d8bb445c9af86a3540") ||
        !same_string(profile->transformers_commit,
                     "42ca97014c85d71a88ad60d55f08cb9fb4d26e2c") ||
        !profile_formulas_validate(profile) ||
        !profile->text_only ||
        profile->source_mtp_layers != DS4_QWEN4EXP_SOURCE_MTP_LAYERS ||
        profile->base_profile_includes_mtp ||
        profile->n_layer != DS4_QWEN4EXP_N_LAYER ||
        profile->hidden_size != DS4_QWEN4EXP_HIDDEN_SIZE ||
        profile->residual_streams != DS4_QWEN4EXP_RESIDUAL_STREAMS ||
        profile->gr_low_rank != DS4_QWEN4EXP_GR_LOW_RANK ||
        profile->vocab_size != DS4_QWEN4EXP_VOCAB_SIZE ||
        profile->context_length != DS4_QWEN4EXP_CONTEXT_LENGTH ||
        profile->qsa_query_heads != DS4_QWEN4EXP_QSA_QUERY_HEADS ||
        profile->qsa_kv_heads != DS4_QWEN4EXP_QSA_KV_HEADS ||
        profile->qsa_head_dim != DS4_QWEN4EXP_QSA_HEAD_DIM ||
        profile->qsa_rotary_dim != DS4_QWEN4EXP_QSA_ROTARY_DIM ||
        profile->qsa_index_query_heads != DS4_QWEN4EXP_QSA_INDEX_QUERY_HEADS ||
        profile->qsa_index_key_heads != DS4_QWEN4EXP_QSA_INDEX_KEY_HEADS ||
        profile->qsa_index_head_dim != DS4_QWEN4EXP_QSA_INDEX_HEAD_DIM ||
        profile->qsa_compression != DS4_QWEN4EXP_QSA_COMPRESSION ||
        profile->qsa_token_budget != DS4_QWEN4EXP_QSA_TOKEN_BUDGET ||
        !profile->qsa_mrope_interleaved ||
        profile->qsa_output_gate != DS4_QWEN4EXP_GATE_SIGMOID ||
        profile->gdn_key_heads != DS4_QWEN4EXP_GDN_KEY_HEADS ||
        profile->gdn_value_heads != DS4_QWEN4EXP_GDN_VALUE_HEADS ||
        profile->gdn_head_dim != DS4_QWEN4EXP_GDN_HEAD_DIM ||
        profile->gdn_conv_kernel != DS4_QWEN4EXP_GDN_CONV_KERNEL ||
        profile->gdn_recurrent_element_bytes != 4u ||
        profile->gdn_output_gate != DS4_QWEN4EXP_GATE_SIGMOID ||
        profile->experts != DS4_QWEN4EXP_EXPERTS ||
        profile->experts_used != DS4_QWEN4EXP_EXPERTS_USED ||
        profile->expert_dim != DS4_QWEN4EXP_EXPERT_DIM ||
        profile->shared_expert_dim != DS4_QWEN4EXP_SHARED_EXPERT_DIM ||
        profile->router_softmax_element_bytes != 4u ||
        !profile->router_full_softmax ||
        !profile->router_normalize_selected ||
        profile->router_tie_policy !=
                DS4_QWEN4EXP_ROUTER_TIE_ASCENDING_EXPERT_ID ||
        profile->ple_source_layer_id != DS4_QWEN4EXP_PLE_SOURCE_LAYER_ID ||
        profile->ple_runtime_layer != DS4_QWEN4EXP_PLE_RUNTIME_LAYER ||
        profile->ple_layer_ordinal != DS4_QWEN4EXP_PLE_LAYER_ORDINAL ||
        profile->ple_seed != DS4_QWEN4EXP_PLE_SEED ||
        profile->ple_vocab_base != DS4_QWEN4EXP_PLE_VOCAB_BASE ||
        profile->ple_split_parts != DS4_QWEN4EXP_PLE_SPLIT_PARTS ||
        profile->ple_row_alignment != DS4_QWEN4EXP_PLE_ROW_ALIGNMENT ||
        profile->ple_ngram_size != DS4_QWEN4EXP_PLE_NGRAM_SIZE ||
        profile->ple_heads_per_ngram != DS4_QWEN4EXP_PLE_HEADS_PER_NGRAM ||
        profile->ple_head_dim != DS4_QWEN4EXP_PLE_HEAD_DIM ||
        profile->ple_rows != 320001536u ||
        profile->ple_conv_kernel != DS4_QWEN4EXP_PLE_CONV_KERNEL ||
        profile->ple_conv_dilation != DS4_QWEN4EXP_PLE_CONV_DILATION ||
        profile->ple_pad_token != DS4_QWEN4EXP_PLE_PAD_TOKEN ||
        profile->rms_epsilon != 1.0e-6f ||
        profile->rope_theta != 10000000.0f) {
        return false;
    }
    for (i = 0u; i < 3u; ++i) {
        if (profile->mrope_section[i] != expected_mrope[i]) return false;
    }
    for (i = 0u; i < DS4_QWEN4EXP_N_LAYER; ++i) {
        const ds4_qwen4exp_layer_type expected =
                i % 4u == 3u ? DS4_QWEN4EXP_LAYER_QSA
                             : DS4_QWEN4EXP_LAYER_GDN;
        if (profile->layer_type[i] != expected) return false;
        if (profile->layer_type[i] == DS4_QWEN4EXP_LAYER_GDN) ++gdn_layers;
        else ++qsa_layers;
    }
    if (gdn_layers != DS4_QWEN4EXP_N_GDN_LAYER ||
        qsa_layers != DS4_QWEN4EXP_N_QSA_LAYER) {
        return false;
    }
    for (i = 0u; i < DS4_QWEN4EXP_PLE_HASH_MULTIPLIERS; ++i) {
        if (profile->ple_multiplier[i] != qwen4exp_profile.ple_multiplier[i]) {
            return false;
        }
    }
    for (i = 0u; i < DS4_QWEN4EXP_PLE_HEADS; ++i) {
        if (profile->ple_head_prime[i] != qwen4exp_profile.ple_head_prime[i] ||
            profile->ple_head_offset[i] != qwen4exp_profile.ple_head_offset[i]) {
            return false;
        }
    }
    return true;
}

bool ds4_qwen4exp_layer_type_get(
        uint32_t                  layer,
        ds4_qwen4exp_layer_type *type) {
    if (!type || layer >= DS4_QWEN4EXP_N_LAYER) return false;
    *type = qwen4exp_profile.layer_type[layer];
    return true;
}

bool ds4_qwen4exp_layer_is_gdn(uint32_t layer) {
    return layer < DS4_QWEN4EXP_N_LAYER &&
           qwen4exp_profile.layer_type[layer] == DS4_QWEN4EXP_LAYER_GDN;
}

bool ds4_qwen4exp_layer_is_qsa(uint32_t layer) {
    return layer < DS4_QWEN4EXP_N_LAYER &&
           qwen4exp_profile.layer_type[layer] == DS4_QWEN4EXP_LAYER_QSA;
}

bool ds4_qwen4exp_layer_has_ple(uint32_t layer) {
    return layer == DS4_QWEN4EXP_PLE_RUNTIME_LAYER;
}

bool ds4_qwen4exp_state_plan_make(
        uint32_t                 context,
        uint32_t                 qsa_kv_element_bytes,
        uint32_t                 qsa_index_element_bytes,
        uint32_t                 gdn_state_element_bytes,
        uint32_t                 ple_element_bytes,
        ds4_qwen4exp_state_plan *plan) {
    ds4_qwen4exp_state_plan next = {0};
    uint64_t gdn_qkv_channels;
    uint64_t value;

    if (!plan || context == 0u || context > DS4_QWEN4EXP_CONTEXT_LENGTH ||
        !state_element_bytes_valid(qsa_kv_element_bytes) ||
        !state_element_bytes_valid(qsa_index_element_bytes) ||
        /* Persistent GDN convolution and recurrence state is always FP32. */
        gdn_state_element_bytes != 4u ||
        !state_element_bytes_valid(ple_element_bytes)) {
        return false;
    }
    next.context = context;
    next.qsa_kv_element_bytes = qsa_kv_element_bytes;
    next.qsa_index_element_bytes = qsa_index_element_bytes;
    next.gdn_state_element_bytes = gdn_state_element_bytes;
    next.ple_element_bytes = ple_element_bytes;

    if (!checked_mul_u64(2u, DS4_QWEN4EXP_GDN_KEY_HEADS,
                         &gdn_qkv_channels) ||
        !checked_add_u64(gdn_qkv_channels, DS4_QWEN4EXP_GDN_VALUE_HEADS,
                         &gdn_qkv_channels) ||
        !checked_mul_u64(gdn_qkv_channels, DS4_QWEN4EXP_GDN_HEAD_DIM,
                         &gdn_qkv_channels) ||
        !checked_mul_u64(gdn_qkv_channels,
                         DS4_QWEN4EXP_GDN_CONV_KERNEL - 1u,
                         &next.gdn_conv_values_per_layer) ||
        !checked_mul_u64(DS4_QWEN4EXP_GDN_VALUE_HEADS,
                         DS4_QWEN4EXP_GDN_HEAD_DIM,
                         &next.gdn_recurrent_values_per_layer) ||
        !checked_mul_u64(next.gdn_recurrent_values_per_layer,
                         DS4_QWEN4EXP_GDN_HEAD_DIM,
                         &next.gdn_recurrent_values_per_layer)) {
        return false;
    }
    if (!checked_mul_u64(next.gdn_conv_values_per_layer,
                         DS4_QWEN4EXP_N_GDN_LAYER, &next.gdn_conv_bytes) ||
        !checked_mul_u64(next.gdn_conv_bytes, gdn_state_element_bytes,
                         &next.gdn_conv_bytes) ||
        !checked_mul_u64(next.gdn_recurrent_values_per_layer,
                         DS4_QWEN4EXP_N_GDN_LAYER,
                         &next.gdn_recurrent_bytes) ||
        !checked_mul_u64(next.gdn_recurrent_bytes, gdn_state_element_bytes,
                         &next.gdn_recurrent_bytes)) {
        return false;
    }
    if (!checked_mul_u64(DS4_QWEN4EXP_N_QSA_LAYER,
                         DS4_QWEN4EXP_QSA_KV_HEADS,
                         &next.qsa_kv_bytes_per_token) ||
        !checked_mul_u64(next.qsa_kv_bytes_per_token,
                         DS4_QWEN4EXP_QSA_HEAD_DIM,
                         &next.qsa_kv_bytes_per_token) ||
        !checked_mul_u64(next.qsa_kv_bytes_per_token, 2u,
                         &next.qsa_kv_bytes_per_token) ||
        !checked_mul_u64(next.qsa_kv_bytes_per_token,
                         qsa_kv_element_bytes,
                         &next.qsa_kv_bytes_per_token) ||
        !checked_mul_u64(next.qsa_kv_bytes_per_token, context,
                         &next.qsa_kv_bytes)) {
        return false;
    }
    if (!checked_mul_u64(DS4_QWEN4EXP_N_QSA_LAYER,
                         DS4_QWEN4EXP_QSA_INDEX_KEY_HEADS,
                         &next.qsa_raw_index_bytes_per_token) ||
        !checked_mul_u64(next.qsa_raw_index_bytes_per_token,
                         DS4_QWEN4EXP_QSA_INDEX_HEAD_DIM,
                         &next.qsa_raw_index_bytes_per_token) ||
        !checked_mul_u64(next.qsa_raw_index_bytes_per_token,
                         qsa_index_element_bytes,
                         &next.qsa_raw_index_bytes_per_token) ||
        !checked_mul_u64(next.qsa_raw_index_bytes_per_token, context,
                         &next.qsa_raw_index_bytes)) {
        return false;
    }
    if (!checked_mul_u64(DS4_QWEN4EXP_PLE_NGRAM_SIZE - 1u,
                         sizeof(uint32_t), &next.ple_history_token_bytes) ||
        !checked_mul_u64(DS4_QWEN4EXP_WIDE_SIZE,
                         DS4_QWEN4EXP_PLE_CONV_STATE,
                         &next.ple_conv_bytes) ||
        !checked_mul_u64(next.ple_conv_bytes, ple_element_bytes,
                         &next.ple_conv_bytes)) {
        return false;
    }
    if (!checked_add_u64(next.gdn_conv_bytes, next.gdn_recurrent_bytes,
                         &next.fixed_tensor_bytes) ||
        !checked_add_u64(next.fixed_tensor_bytes,
                         next.ple_history_token_bytes,
                         &next.fixed_tensor_bytes) ||
        !checked_add_u64(next.fixed_tensor_bytes, next.ple_conv_bytes,
                         &next.fixed_tensor_bytes) ||
        !checked_add_u64(next.qsa_kv_bytes, next.qsa_raw_index_bytes,
                         &next.context_tensor_bytes) ||
        !checked_add_u64(next.fixed_tensor_bytes,
                         next.context_tensor_bytes, &value)) {
        return false;
    }
    next.tensor_payload_bytes = value;
    *plan = next;
    return true;
}

bool ds4_qwen4exp_expert_parameter_count(uint64_t *count) {
    uint64_t next;
    if (!count ||
        !checked_mul_u64(DS4_QWEN4EXP_HIDDEN_SIZE,
                         DS4_QWEN4EXP_EXPERT_DIM, &next) ||
        !checked_mul_u64(next, 3u, &next)) {
        return false;
    }
    *count = next;
    return true;
}

bool ds4_qwen4exp_routed_parameter_count(uint64_t *count) {
    uint64_t next;
    if (!count || !ds4_qwen4exp_expert_parameter_count(&next) ||
        !checked_mul_u64(next, DS4_QWEN4EXP_EXPERTS, &next) ||
        !checked_mul_u64(next, DS4_QWEN4EXP_N_LAYER, &next)) {
        return false;
    }
    *count = next;
    return true;
}

bool ds4_qwen4exp_active_routed_parameter_count(uint64_t *count) {
    uint64_t next;
    if (!count || !ds4_qwen4exp_expert_parameter_count(&next) ||
        !checked_mul_u64(next, DS4_QWEN4EXP_EXPERTS_USED, &next) ||
        !checked_mul_u64(next, DS4_QWEN4EXP_N_LAYER, &next)) {
        return false;
    }
    *count = next;
    return true;
}

bool ds4_qwen4exp_ple_parameter_count(uint64_t *count) {
    uint64_t next;
    if (!count || !checked_mul_u64(qwen4exp_profile.ple_rows,
                                   qwen4exp_profile.ple_head_dim, &next)) {
        return false;
    }
    *count = next;
    return true;
}
