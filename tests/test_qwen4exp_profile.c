#include "ds4_qwen4exp.h"

#include <assert.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define ASSERT_REJECT_U32(field) do {                  \
    candidate = *profile;                              \
    candidate.field ^= UINT32_C(1);                    \
    assert(!ds4_qwen4exp_profile_validate(&candidate));\
} while (0)

static void test_constants_and_descriptor(void) {
    static const uint64_t multipliers[] = {
        UINT64_C(23703573157769),
        UINT64_C(20109073645365),
        UINT64_C(8052911324071),
    };
    static const uint32_t primes[] = {
        20000003u, 20000023u, 20000033u, 20000047u,
        20000059u, 20000063u, 20000069u, 20000077u,
        20000081u, 20000093u, 20000107u, 20000147u,
        20000153u, 20000159u, 20000161u, 20000171u,
    };
    static const uint32_t offsets[] = {
        0u, 20000003u, 40000026u, 60000059u,
        80000106u, 100000165u, 120000228u, 140000297u,
        160000374u, 180000455u, 200000548u, 220000655u,
        240000802u, 260000955u, 280001114u, 300001275u,
    };
    const ds4_qwen4exp_profile *profile = ds4_qwen4exp_profile_get();
    uint32_t i;
    uint32_t gdn_count = 0u;
    uint32_t qsa_count = 0u;

    assert(profile != NULL);
    assert(profile == ds4_qwen4exp_profile_get());
    assert(ds4_qwen4exp_profile_validate(profile));
    assert(strcmp(profile->profile_id, "qwen4exp-base-v1") == 0);
    assert(strcmp(profile->source_architecture,
                  "Qwen4ExpForConditionalGeneration") == 0);
    assert(strcmp(profile->gguf_architecture, "qwen4exp") == 0);
    assert(strcmp(profile->hf_revision,
                  "de4b8e4d43b917e7706784d8bb445c9af86a3540") == 0);
    assert(strcmp(profile->transformers_commit,
                  "42ca97014c85d71a88ad60d55f08cb9fb4d26e2c") == 0);
    assert(profile->text_only);
    assert(profile->source_mtp_layers == 1u);
    assert(!profile->base_profile_includes_mtp);

    assert(profile->n_layer == 48u);
    assert(profile->hidden_size == 2560u);
    assert(profile->residual_streams == 4u);
    assert(profile->gr_low_rank == 320u);
    assert(profile->vocab_size == 248320u);
    assert(profile->context_length == 262144u);
    assert(profile->qsa_query_heads == 24u);
    assert(profile->qsa_kv_heads == 2u);
    assert(profile->qsa_head_dim == 256u);
    assert(profile->qsa_rotary_dim == 64u);
    assert(profile->qsa_index_query_heads == 4u);
    assert(profile->qsa_index_key_heads == 1u);
    assert(profile->qsa_index_head_dim == 128u);
    assert(profile->qsa_compression == 4u);
    assert(profile->qsa_token_budget == 2048u);
    assert(profile->qsa_mrope_interleaved);
    assert(profile->qsa_output_gate == DS4_QWEN4EXP_GATE_SIGMOID);
    assert(profile->gdn_key_heads == 16u);
    assert(profile->gdn_value_heads == 48u);
    assert(profile->gdn_head_dim == 128u);
    assert(profile->gdn_conv_kernel == 4u);
    assert(profile->gdn_recurrent_element_bytes == 4u);
    assert(profile->gdn_output_gate == DS4_QWEN4EXP_GATE_SIGMOID);
    assert(profile->experts == 512u);
    assert(profile->experts_used == 10u);
    assert(profile->expert_dim == 640u);
    assert(profile->shared_expert_dim == 640u);
    assert(profile->router_softmax_element_bytes == 4u);
    assert(profile->router_full_softmax);
    assert(profile->router_normalize_selected);
    assert(profile->router_tie_policy ==
           DS4_QWEN4EXP_ROUTER_TIE_ASCENDING_EXPERT_ID);
    assert(profile->ple_source_layer_id == 2u);
    assert(profile->ple_runtime_layer == 1u);
    assert(profile->ple_layer_ordinal == 0u);
    assert(profile->ple_seed == 1234u);
    assert(profile->ple_vocab_base == 20000000u);
    assert(profile->ple_split_parts == 128u);
    assert(profile->ple_row_alignment == 128u);
    assert(profile->ple_ngram_size == 3u);
    assert(profile->ple_heads_per_ngram == 8u);
    assert(profile->ple_head_dim == 160u);
    assert(profile->ple_rows == 320001536u);
    assert(profile->ple_conv_kernel == 4u);
    assert(profile->ple_conv_dilation == 3u);
    assert(profile->ple_pad_token == 248044u);
    assert(profile->rms_epsilon == 1.0e-6f);
    assert(profile->rope_theta == 10000000.0f);
    assert(profile->mrope_section[0] == 11u);
    assert(profile->mrope_section[1] == 11u);
    assert(profile->mrope_section[2] == 10u);

    assert(DS4_QWEN4EXP_WIDE_SIZE == 4 * 2560);
    assert(DS4_QWEN4EXP_QSA_BLOCK_BUDGET == 2048 / 4);
    assert(DS4_QWEN4EXP_QSA_MAX_SELECTED_WIDTH == 2048 + 4 - 1);
    assert(DS4_QWEN4EXP_GDN_REPEAT_RATIO == 48 / 16);
    assert(DS4_QWEN4EXP_PLE_HEADS == (3 - 1) * 8);
    assert(DS4_QWEN4EXP_PLE_FLAT_SIZE == 16 * 160);
    assert(DS4_QWEN4EXP_PLE_CONV_STATE == (4 - 1) * 3);

    for (i = 0u; i < DS4_QWEN4EXP_N_LAYER; ++i) {
        const ds4_qwen4exp_layer_type expected =
                i % 4u == 3u ? DS4_QWEN4EXP_LAYER_QSA
                             : DS4_QWEN4EXP_LAYER_GDN;
        assert(profile->layer_type[i] == expected);
        if (expected == DS4_QWEN4EXP_LAYER_GDN) ++gdn_count;
        else ++qsa_count;
    }
    assert(gdn_count == 36u);
    assert(qsa_count == 12u);
    for (i = 0u; i < DS4_QWEN4EXP_PLE_HASH_MULTIPLIERS; ++i) {
        assert(profile->ple_multiplier[i] == multipliers[i]);
    }
    for (i = 0u; i < DS4_QWEN4EXP_PLE_HEADS; ++i) {
        assert(profile->ple_head_prime[i] == primes[i]);
        assert(profile->ple_head_offset[i] == offsets[i]);
    }
}

static void test_profile_rejects_every_mutable_field(void) {
    const ds4_qwen4exp_profile *profile = ds4_qwen4exp_profile_get();
    ds4_qwen4exp_profile candidate;
    uint32_t i;

    assert(!ds4_qwen4exp_profile_validate(NULL));

    candidate = *profile;
    candidate.profile_id = NULL;
    assert(!ds4_qwen4exp_profile_validate(&candidate));
    candidate = *profile;
    candidate.profile_id = "qwen4exp-base-v2";
    assert(!ds4_qwen4exp_profile_validate(&candidate));
    candidate = *profile;
    candidate.source_architecture = "Qwen4ForConditionalGeneration";
    assert(!ds4_qwen4exp_profile_validate(&candidate));
    candidate = *profile;
    candidate.gguf_architecture = "qwen4";
    assert(!ds4_qwen4exp_profile_validate(&candidate));
    candidate = *profile;
    candidate.hf_revision = "de4b8e4d";
    assert(!ds4_qwen4exp_profile_validate(&candidate));
    candidate = *profile;
    candidate.transformers_commit = "42ca9701";
    assert(!ds4_qwen4exp_profile_validate(&candidate));

    candidate = *profile;
    candidate.text_only = false;
    assert(!ds4_qwen4exp_profile_validate(&candidate));
    ASSERT_REJECT_U32(source_mtp_layers);
    candidate = *profile;
    candidate.base_profile_includes_mtp = true;
    assert(!ds4_qwen4exp_profile_validate(&candidate));

    ASSERT_REJECT_U32(n_layer);
    ASSERT_REJECT_U32(hidden_size);
    ASSERT_REJECT_U32(residual_streams);
    ASSERT_REJECT_U32(gr_low_rank);
    ASSERT_REJECT_U32(vocab_size);
    ASSERT_REJECT_U32(context_length);
    ASSERT_REJECT_U32(qsa_query_heads);
    ASSERT_REJECT_U32(qsa_kv_heads);
    ASSERT_REJECT_U32(qsa_head_dim);
    ASSERT_REJECT_U32(qsa_rotary_dim);
    ASSERT_REJECT_U32(qsa_index_query_heads);
    ASSERT_REJECT_U32(qsa_index_key_heads);
    ASSERT_REJECT_U32(qsa_index_head_dim);
    ASSERT_REJECT_U32(qsa_compression);
    ASSERT_REJECT_U32(qsa_token_budget);
    candidate = *profile;
    candidate.qsa_mrope_interleaved = false;
    assert(!ds4_qwen4exp_profile_validate(&candidate));
    candidate = *profile;
    candidate.qsa_output_gate = (ds4_qwen4exp_gate_activation)0;
    assert(!ds4_qwen4exp_profile_validate(&candidate));
    ASSERT_REJECT_U32(gdn_key_heads);
    ASSERT_REJECT_U32(gdn_value_heads);
    ASSERT_REJECT_U32(gdn_head_dim);
    ASSERT_REJECT_U32(gdn_conv_kernel);
    ASSERT_REJECT_U32(gdn_recurrent_element_bytes);
    candidate = *profile;
    candidate.gdn_output_gate = (ds4_qwen4exp_gate_activation)0;
    assert(!ds4_qwen4exp_profile_validate(&candidate));
    ASSERT_REJECT_U32(experts);
    ASSERT_REJECT_U32(experts_used);
    ASSERT_REJECT_U32(expert_dim);
    ASSERT_REJECT_U32(shared_expert_dim);
    ASSERT_REJECT_U32(router_softmax_element_bytes);
    candidate = *profile;
    candidate.router_full_softmax = false;
    assert(!ds4_qwen4exp_profile_validate(&candidate));
    candidate = *profile;
    candidate.router_normalize_selected = false;
    assert(!ds4_qwen4exp_profile_validate(&candidate));
    candidate = *profile;
    candidate.router_tie_policy = (ds4_qwen4exp_router_tie_policy)0;
    assert(!ds4_qwen4exp_profile_validate(&candidate));
    ASSERT_REJECT_U32(ple_source_layer_id);
    ASSERT_REJECT_U32(ple_runtime_layer);
    ASSERT_REJECT_U32(ple_layer_ordinal);
    ASSERT_REJECT_U32(ple_seed);
    ASSERT_REJECT_U32(ple_vocab_base);
    ASSERT_REJECT_U32(ple_split_parts);
    ASSERT_REJECT_U32(ple_row_alignment);
    ASSERT_REJECT_U32(ple_ngram_size);
    ASSERT_REJECT_U32(ple_heads_per_ngram);
    ASSERT_REJECT_U32(ple_head_dim);
    ASSERT_REJECT_U32(ple_rows);
    ASSERT_REJECT_U32(ple_conv_kernel);
    ASSERT_REJECT_U32(ple_conv_dilation);
    ASSERT_REJECT_U32(ple_pad_token);

    candidate = *profile;
    candidate.rms_epsilon = 0.0f;
    assert(!ds4_qwen4exp_profile_validate(&candidate));
    candidate = *profile;
    candidate.rope_theta = 0.0f;
    assert(!ds4_qwen4exp_profile_validate(&candidate));

    for (i = 0u; i < 3u; ++i) {
        candidate = *profile;
        candidate.mrope_section[i] ^= 1u;
        assert(!ds4_qwen4exp_profile_validate(&candidate));
    }
    for (i = 0u; i < DS4_QWEN4EXP_N_LAYER; ++i) {
        candidate = *profile;
        candidate.layer_type[i] = candidate.layer_type[i] == DS4_QWEN4EXP_LAYER_GDN
                ? DS4_QWEN4EXP_LAYER_QSA : DS4_QWEN4EXP_LAYER_GDN;
        assert(!ds4_qwen4exp_profile_validate(&candidate));
    }
    for (i = 0u; i < DS4_QWEN4EXP_PLE_HASH_MULTIPLIERS; ++i) {
        candidate = *profile;
        candidate.ple_multiplier[i] ^= UINT64_C(2);
        assert(!ds4_qwen4exp_profile_validate(&candidate));
    }
    for (i = 0u; i < DS4_QWEN4EXP_PLE_HEADS; ++i) {
        candidate = *profile;
        candidate.ple_head_prime[i] ^= 2u;
        assert(!ds4_qwen4exp_profile_validate(&candidate));
        candidate = *profile;
        candidate.ple_head_offset[i] ^= 1u;
        assert(!ds4_qwen4exp_profile_validate(&candidate));
    }

    /* Preserve every earlier formula, then overflow the routed-total product. */
    candidate = *profile;
    candidate.experts = UINT32_MAX;
    candidate.n_layer = UINT32_MAX;
    {
        const uint64_t expert_parameters =
                (uint64_t)candidate.hidden_size * candidate.expert_dim * 3u;
        const uint64_t all_experts_one_layer =
                expert_parameters * candidate.experts;
        assert(expert_parameters == UINT64_C(4915200));
        assert(expert_parameters <= UINT64_MAX / candidate.experts);
        assert(all_experts_one_layer > UINT64_MAX / candidate.n_layer);
    }
    assert(!ds4_qwen4exp_profile_validate(&candidate));
}

static void test_layer_queries(void) {
    ds4_qwen4exp_layer_type type = (ds4_qwen4exp_layer_type)77;
    uint32_t i;

    for (i = 0u; i < DS4_QWEN4EXP_N_LAYER; ++i) {
        const bool qsa = i % 4u == 3u;
        assert(ds4_qwen4exp_layer_type_get(i, &type));
        assert(type == (qsa ? DS4_QWEN4EXP_LAYER_QSA
                            : DS4_QWEN4EXP_LAYER_GDN));
        assert(ds4_qwen4exp_layer_is_gdn(i) == !qsa);
        assert(ds4_qwen4exp_layer_is_qsa(i) == qsa);
        assert(ds4_qwen4exp_layer_has_ple(i) == (i == 1u));
    }
    type = (ds4_qwen4exp_layer_type)77;
    assert(!ds4_qwen4exp_layer_type_get(DS4_QWEN4EXP_N_LAYER, &type));
    assert(type == (ds4_qwen4exp_layer_type)77);
    assert(!ds4_qwen4exp_layer_type_get(UINT32_MAX, &type));
    assert(type == (ds4_qwen4exp_layer_type)77);
    assert(!ds4_qwen4exp_layer_type_get(0u, NULL));
    assert(!ds4_qwen4exp_layer_is_gdn(DS4_QWEN4EXP_N_LAYER));
    assert(!ds4_qwen4exp_layer_is_qsa(UINT32_MAX));
    assert(!ds4_qwen4exp_layer_has_ple(UINT32_MAX));
}

static void assert_plan_consistent(
        const ds4_qwen4exp_state_plan *plan,
        uint32_t                       context,
        uint32_t                       qsa_kv_bytes,
        uint32_t                       qsa_index_bytes,
        uint32_t                       gdn_bytes,
        uint32_t                       ple_bytes) {
    const uint64_t expected_gdn_conv_values = UINT64_C(10240) * 3u;
    const uint64_t expected_gdn_recurrent_values = UINT64_C(48) * 128u * 128u;
    const uint64_t expected_gdn_conv_bytes =
            36u * expected_gdn_conv_values * gdn_bytes;
    const uint64_t expected_gdn_recurrent_bytes =
            36u * expected_gdn_recurrent_values * gdn_bytes;
    const uint64_t expected_qsa_kv_per_token =
            UINT64_C(12) * 2u * 256u * 2u * qsa_kv_bytes;
    const uint64_t expected_qsa_index_per_token =
            UINT64_C(12) * 128u * qsa_index_bytes;
    const uint64_t expected_ple_history_bytes = 2u * sizeof(uint32_t);
    const uint64_t expected_ple_conv_bytes =
            UINT64_C(10240) * 9u * ple_bytes;
    const uint64_t expected_fixed = expected_gdn_conv_bytes +
            expected_gdn_recurrent_bytes + expected_ple_history_bytes +
            expected_ple_conv_bytes;
    const uint64_t expected_context =
            (expected_qsa_kv_per_token + expected_qsa_index_per_token) * context;

    assert(plan->context == context);
    assert(plan->qsa_kv_element_bytes == qsa_kv_bytes);
    assert(plan->qsa_index_element_bytes == qsa_index_bytes);
    assert(plan->gdn_state_element_bytes == gdn_bytes);
    assert(plan->ple_element_bytes == ple_bytes);
    assert(plan->gdn_conv_values_per_layer == expected_gdn_conv_values);
    assert(plan->gdn_recurrent_values_per_layer == expected_gdn_recurrent_values);
    assert(plan->gdn_conv_bytes == expected_gdn_conv_bytes);
    assert(plan->gdn_recurrent_bytes == expected_gdn_recurrent_bytes);
    assert(plan->qsa_kv_bytes_per_token == expected_qsa_kv_per_token);
    assert(plan->qsa_kv_bytes == expected_qsa_kv_per_token * context);
    assert(plan->qsa_raw_index_bytes_per_token == expected_qsa_index_per_token);
    assert(plan->qsa_raw_index_bytes == expected_qsa_index_per_token * context);
    assert(plan->ple_history_token_bytes == expected_ple_history_bytes);
    assert(plan->ple_conv_bytes == expected_ple_conv_bytes);
    assert(plan->fixed_tensor_bytes == expected_fixed);
    assert(plan->context_tensor_bytes == expected_context);
    assert(plan->tensor_payload_bytes == expected_fixed + expected_context);
}

static void test_state_plans(void) {
    static const uint32_t valid_element_bytes[] = {1u, 2u, 4u};
    ds4_qwen4exp_state_plan plan;
    uint32_t i;
    uint32_t j;
    uint32_t k;

    assert(ds4_qwen4exp_state_plan_make(1u, 2u, 2u, 4u, 2u, &plan));
    assert_plan_consistent(&plan, 1u, 2u, 2u, 4u, 2u);
    assert(plan.gdn_conv_values_per_layer == 30720u);
    assert(plan.gdn_recurrent_values_per_layer == 786432u);
    assert(plan.gdn_conv_bytes == UINT64_C(4423680));
    assert(plan.gdn_recurrent_bytes == UINT64_C(113246208));
    assert(plan.qsa_kv_bytes_per_token == 24576u);
    assert(plan.qsa_raw_index_bytes_per_token == 3072u);
    assert(plan.ple_history_token_bytes == 8u);
    assert(plan.ple_conv_bytes == 184320u);
    assert(plan.fixed_tensor_bytes == UINT64_C(117854216));
    assert(plan.context_tensor_bytes == 27648u);
    assert(plan.tensor_payload_bytes == UINT64_C(117881864));

    assert(ds4_qwen4exp_state_plan_make(262144u, 2u, 2u, 4u, 2u,
                                        &plan));
    assert_plan_consistent(&plan, 262144u, 2u, 2u, 4u, 2u);
    assert(plan.qsa_kv_bytes == UINT64_C(6442450944));
    assert(plan.qsa_raw_index_bytes == UINT64_C(805306368));
    assert(plan.context_tensor_bytes == UINT64_C(7247757312));
    assert(plan.tensor_payload_bytes == UINT64_C(7365611528));

    for (i = 0u; i < 3u; ++i) {
        for (j = 0u; j < 3u; ++j) {
            for (k = 0u; k < 3u; ++k) {
                assert(ds4_qwen4exp_state_plan_make(
                        262144u, valid_element_bytes[i],
                        valid_element_bytes[j], 4u, valid_element_bytes[k],
                        &plan));
                assert_plan_consistent(
                        &plan, 262144u, valid_element_bytes[i],
                        valid_element_bytes[j], 4u, valid_element_bytes[k]);
            }
        }
    }
}

static void assert_plan_failure_preserves_output(
        uint32_t context,
        uint32_t qsa_kv_bytes,
        uint32_t qsa_index_bytes,
        uint32_t gdn_bytes,
        uint32_t ple_bytes) {
    ds4_qwen4exp_state_plan plan;
    ds4_qwen4exp_state_plan before;

    memset(&plan, 0xa5, sizeof(plan));
    memcpy(&before, &plan, sizeof(before));
    assert(!ds4_qwen4exp_state_plan_make(
            context, qsa_kv_bytes, qsa_index_bytes, gdn_bytes, ple_bytes,
            &plan));
    assert(memcmp(&plan, &before, sizeof(plan)) == 0);
}

static void test_state_plan_failures_are_transactional(void) {
    static const uint32_t invalid_element_bytes[] = {
        0u, 3u, 5u, 8u, UINT32_MAX,
    };
    uint32_t i;

    assert_plan_failure_preserves_output(0u, 2u, 2u, 4u, 2u);
    assert_plan_failure_preserves_output(262145u, 2u, 2u, 4u, 2u);
    assert_plan_failure_preserves_output(UINT32_MAX, 2u, 2u, 4u, 2u);
    assert_plan_failure_preserves_output(1u, 2u, 2u, 1u, 2u);
    assert_plan_failure_preserves_output(1u, 2u, 2u, 2u, 2u);
    for (i = 0u; i < sizeof(invalid_element_bytes) /
                            sizeof(invalid_element_bytes[0]); ++i) {
        assert_plan_failure_preserves_output(
                1u, invalid_element_bytes[i], 2u, 4u, 2u);
        assert_plan_failure_preserves_output(
                1u, 2u, invalid_element_bytes[i], 4u, 2u);
        assert_plan_failure_preserves_output(
                1u, 2u, 2u, invalid_element_bytes[i], 2u);
        assert_plan_failure_preserves_output(
                1u, 2u, 2u, 4u, invalid_element_bytes[i]);
    }
    assert(!ds4_qwen4exp_state_plan_make(1u, 2u, 2u, 4u, 2u, NULL));
}

static void test_parameter_counts(void) {
    uint64_t count = UINT64_MAX;

    assert(ds4_qwen4exp_expert_parameter_count(&count));
    assert(count == UINT64_C(4915200));
    assert(ds4_qwen4exp_routed_parameter_count(&count));
    assert(count == UINT64_C(120795955200));
    assert(ds4_qwen4exp_active_routed_parameter_count(&count));
    assert(count == UINT64_C(2359296000));
    assert(ds4_qwen4exp_ple_parameter_count(&count));
    assert(count == UINT64_C(51200245760));

    assert(!ds4_qwen4exp_expert_parameter_count(NULL));
    assert(!ds4_qwen4exp_routed_parameter_count(NULL));
    assert(!ds4_qwen4exp_active_routed_parameter_count(NULL));
    assert(!ds4_qwen4exp_ple_parameter_count(NULL));
}

int main(void) {
    test_constants_and_descriptor();
    test_profile_rejects_every_mutable_field();
    test_layer_queries();
    test_state_plans();
    test_state_plan_failures_are_transactional();
    test_parameter_counts();
    puts("qwen4exp profile tests passed");
    return 0;
}
