/* White-box engine/session tests.  The fake forward keeps this gate model-free
 * while exercising production accounting, cache/checkpoint transactions and
 * public logits APIs at the real 248320-token vocabulary size. */

#define DS4_TEST_HOOKS 1
#include "../ds4.c"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

#define CHECK(condition) do {                                                \
    if (!(condition)) {                                                      \
        fprintf(stderr, "qwen session check failed at %s:%d: %s\n",       \
                __FILE__, __LINE__, #condition);                             \
        failures++;                                                          \
    }                                                                        \
} while (0)

static int stub_calls;
static int stub_logits_calls;
static int stub_fail_on_call;
static uint32_t stub_position[16];
static int stub_token[16];
static uint32_t stub_logits_position[16];
static int stub_logits_token[16];

static void stub_reset(void) {
    stub_calls = 0;
    stub_logits_calls = 0;
    stub_fail_on_call = 0;
    memset(stub_position, 0, sizeof(stub_position));
    memset(stub_token, 0, sizeof(stub_token));
    memset(stub_logits_position, 0, sizeof(stub_logits_position));
    memset(stub_logits_token, 0, sizeof(stub_logits_token));
}

static bool stub_forward(
        float                         *logits,
        const ds4_model              *model,
        const ds4_qwen35_weights     *weights,
        ds4_qwen35_cpu_cache         *cache,
        int                           token,
        uint32_t                      position,
        ds4_qwen35_cpu_scratch       *scratch) {
    (void)model;
    (void)weights;
    CHECK(cache != NULL);
    CHECK(scratch != NULL && scratch->arena != NULL);
    CHECK(cache->n_tokens == position);
    CHECK(cache->kv_capacity > position);
    if (stub_calls < (int)(sizeof(stub_position) / sizeof(stub_position[0]))) {
        stub_position[stub_calls] = position;
        stub_token[stub_calls] = token;
    }
    stub_calls++;
    if (logits) {
        if (stub_logits_calls <
            (int)(sizeof(stub_logits_position) /
                  sizeof(stub_logits_position[0]))) {
            stub_logits_position[stub_logits_calls] = position;
            stub_logits_token[stub_logits_calls] = token;
        }
        stub_logits_calls++;
        logits[QWEN35_N_VOCAB - 1u] = (float)token;
    }

    if (stub_fail_on_call != 0 && stub_calls == stub_fail_on_call) {
        cache->layer[0].conv[0] = 17.0f;
        cache->layer[0].recurrent[0] = -19.0f;
        const uint64_t row =
            (uint64_t)position * QWEN35_N_HEAD_KV * QWEN35_N_HEAD_DIM;
        cache->layer[3].key[row] = 23.0f;
        cache->layer[3].value[row] = -29.0f;
        CHECK(ds4_qwen35_cpu_cache_advance(cache, 1u));
        return false;
    }
    return ds4_qwen35_cpu_cache_advance(cache, 1u);
}

static bool stub_true_without_advance(
        float                         *logits,
        const ds4_model              *model,
        const ds4_qwen35_weights     *weights,
        ds4_qwen35_cpu_cache         *cache,
        int                           token,
        uint32_t                      position,
        ds4_qwen35_cpu_scratch       *scratch) {
    (void)logits;
    (void)model;
    (void)weights;
    (void)cache;
    (void)token;
    (void)position;
    (void)scratch;
    stub_calls++;
    return true;
}

static bool cancel_after_first_forward(void *ud) {
    (void)ud;
    return stub_calls >= 1;
}

static bool cancel_always(void *ud) {
    (void)ud;
    return true;
}

static void fake_qwen_engine(ds4_engine *engine, bool raw_runtime) {
    static ds4_tensor routed_gate = {.type = DS4_TENSOR_Q4_K};
    static ds4_tensor output = {.type = DS4_TENSOR_Q8_0};
    memset(engine, 0, sizeof(*engine));
    engine->model.fd = -1;
    engine->mtp_model.fd = -1;
    engine->model.family = DS4_MODEL_FAMILY_QWEN35_MOE;
    engine->backend = DS4_BACKEND_CPU;
    engine->qwen_raw_runtime = raw_runtime;
    engine->qwen35_weights.layer[0].ffn_gate_exps = &routed_gate;
    engine->qwen35_weights.output = &output;
}

static void test_gpu_dense_layout_contract(void) {
    ds4_tensor matrix = {
        .ndim = 2,
        .dim = {QWEN35_N_EMBD, QWEN35_N_EXPERT},
        .type = DS4_TENSOR_F32,
    };
    ds4_tensor scalar_gate = {
        .ndim = 1,
        .dim = {QWEN35_N_EMBD},
        .type = DS4_TENSOR_F32,
    };

    CHECK(qwen35_gpu_dense_weight_layout_matches(
              &matrix, QWEN35_N_EMBD, QWEN35_N_EXPERT));
    CHECK(qwen35_gpu_dense_weight_layout_matches(
              &scalar_gate, QWEN35_N_EMBD, 1u));
    CHECK(!qwen35_gpu_dense_weight_layout_matches(
               &scalar_gate, QWEN35_N_EMBD, 2u));
    scalar_gate.type = DS4_TENSOR_Q8_0;
    CHECK(!qwen35_gpu_dense_weight_layout_matches(
               &scalar_gate, QWEN35_N_EMBD, 1u));
    matrix.dim[1]--;
    CHECK(!qwen35_gpu_dense_weight_layout_matches(
               &matrix, QWEN35_N_EMBD, QWEN35_N_EXPERT));
}

static void test_session_creation_boundary(void) {
    ds4_engine engine;
    ds4_session *session = NULL;
    fake_qwen_engine(&engine, false);
    CHECK(ds4_session_create(&session, &engine, 5) != 0);
    CHECK(session == NULL);

    fake_qwen_engine(&engine, true);
    CHECK(ds4_session_create(&session, &engine,
                             (int)QWEN35_CONTEXT_LENGTH + 1) != 0);
    CHECK(session == NULL);
}

static void test_model_aware_context_memory(void) {
    ds4_engine engine;
    ds4_context_memory memory = {0};
    fake_qwen_engine(&engine, true);

    CHECK(!ds4_engine_context_memory_estimate_with_prefill(
              NULL, 1, 0, &memory));
    CHECK(!ds4_engine_context_memory_estimate_with_prefill(
              &engine, 0, 0, &memory));
    CHECK(!ds4_engine_context_memory_estimate_with_prefill(
              &engine, (int)QWEN35_CONTEXT_LENGTH + 1, 0, &memory));
    CHECK(!ds4_engine_context_memory_estimate_with_prefill(
              &engine, 1, 0, NULL));

    CHECK(ds4_engine_context_memory_estimate_with_prefill(
              &engine, 1, 4096, &memory));
    CHECK(memory.raw_bytes == UINT64_C(40960));
    CHECK(memory.compressed_bytes == UINT64_C(65863680));
    CHECK(memory.scratch_bytes == UINT64_C(1185636));
    CHECK(memory.total_bytes == UINT64_C(67090276));
    CHECK(memory.prefill_cap == 1u);
    CHECK(memory.raw_cap == 1u);
    CHECK(memory.comp_cap == 0u);

    CHECK(ds4_engine_context_memory_estimate_with_prefill(
              &engine, 32768, 0, &memory));
    CHECK(memory.raw_bytes == UINT64_C(1342177280));
    CHECK(memory.compressed_bytes == UINT64_C(65863680));
    CHECK(memory.scratch_bytes == UINT64_C(1316704));
    CHECK(memory.total_bytes == UINT64_C(1409357664));
    CHECK(memory.prefill_cap == 1u);
    CHECK(memory.raw_cap == 32768u);
    CHECK(memory.comp_cap == 0u);

    memset(&engine, 0, sizeof(engine));
    engine.backend = DS4_BACKEND_CPU;
    engine.model.family = DS4_MODEL_FAMILY_DEEPSEEK4;
    const ds4_context_memory legacy =
        ds4_context_memory_estimate_with_prefill(
            engine.backend, 32768, 256);
    CHECK(ds4_engine_context_memory_estimate_with_prefill(
              &engine, 32768, 256, &memory));
    CHECK(memcmp(&memory, &legacy, sizeof(memory)) == 0);
}

static void test_qwen_metal_session_context_budget(void) {
    ds4_engine resident_engine;
    ds4_engine ssd_engine;
    ds4_context_memory resident_8k = {0};
    ds4_context_memory resident_100k = {0};
    ds4_context_memory resident_max = {0};
    ds4_context_memory ssd_explicit_8k = {0};
    ds4_context_memory ssd_resolved_8k = {0};
    ds4_context_memory requested = {0};
    fake_qwen_engine(&resident_engine, true);
    resident_engine.backend = DS4_BACKEND_METAL;
    resident_engine.qwen_metal_runtime = true;

    CHECK(ds4_engine_context_memory_estimate_with_prefill(
              &resident_engine, 8192, 0, &resident_8k));
    CHECK(resident_8k.prefill_cap == QWEN35_RESIDENT_PREFILL_TOKENS);
    resident_engine.residency_plan.runtime_bytes = resident_8k.total_bytes;

    CHECK(ds4_qwen35_metal_session_context_fits_runtime_plan(
              &resident_engine, 8192, &requested));
    CHECK(requested.total_bytes == resident_8k.total_bytes);
    CHECK(ds4_qwen35_metal_session_context_fits_runtime_plan(
              &resident_engine, 4096, &requested));
    CHECK(requested.total_bytes < resident_8k.total_bytes);
    CHECK(!ds4_qwen35_metal_session_context_fits_runtime_plan(
               &resident_engine, 8193, &requested));
    CHECK(requested.total_bytes > resident_8k.total_bytes);

    CHECK(ds4_engine_context_memory_estimate_with_prefill(
              &resident_engine, 100000, 0, &resident_100k));
    CHECK(resident_100k.raw_cap == 100000u);
    CHECK(resident_100k.prefill_cap == QWEN35_RESIDENT_PREFILL_TOKENS);
    CHECK(ds4_engine_context_memory_estimate_with_prefill(
              &resident_engine, QWEN35_CONTEXT_LENGTH, 0, &resident_max));
    CHECK(resident_max.raw_cap == QWEN35_CONTEXT_LENGTH);
    CHECK(resident_max.prefill_cap == QWEN35_RESIDENT_PREFILL_TOKENS);
    CHECK(!ds4_engine_context_memory_estimate_with_prefill(
               &resident_engine, QWEN35_CONTEXT_LENGTH + 1u, 0,
               &requested));

    fake_qwen_engine(&ssd_engine, true);
    ssd_engine.backend = DS4_BACKEND_METAL;
    ssd_engine.qwen_metal_runtime = true;
    ssd_engine.residency_requested = DS4_RESIDENCY_SSD;
    CHECK(ds4_engine_context_memory_estimate_with_prefill(
              &ssd_engine, 8192, 0, &ssd_explicit_8k));
    CHECK(ssd_explicit_8k.prefill_cap == QWEN35_PREFILL_MICRO_TOKENS);

    ssd_engine.residency_requested = DS4_RESIDENCY_AUTO;
    ssd_engine.residency_plan.resolved = DS4_RESIDENCY_SSD;
    CHECK(ds4_engine_context_memory_estimate_with_prefill(
              &ssd_engine, 8192, 0, &ssd_resolved_8k));
    CHECK(ssd_resolved_8k.prefill_cap == QWEN35_PREFILL_MICRO_TOKENS);
    CHECK(ssd_resolved_8k.total_bytes == ssd_explicit_8k.total_bytes);
    CHECK(resident_8k.total_bytes - ssd_resolved_8k.total_bytes ==
          UINT64_C(1582768128));
    CHECK(resident_8k.scratch_bytes - ssd_resolved_8k.scratch_bytes ==
          UINT64_C(1582768128));

    uint64_t resident_runtime_8k = 0;
    uint64_t resident_runtime_32k = 0;
    uint64_t resident_runtime_100k = 0;
    uint64_t resident_runtime_max = 0;
    uint64_t ssd_runtime_8k = 0;
    uint64_t ssd_runtime_100k = 0;
    uint64_t ssd_runtime_max = 0;
    CHECK(qwen35_metal_persistent_runtime_bytes(
              8192, QWEN35_RESIDENT_PREFILL_TOKENS,
              &resident_runtime_8k));
    CHECK(qwen35_metal_persistent_runtime_bytes(
              32768, QWEN35_RESIDENT_PREFILL_TOKENS,
              &resident_runtime_32k));
    CHECK(qwen35_metal_persistent_runtime_bytes(
              100000, QWEN35_RESIDENT_PREFILL_TOKENS,
              &resident_runtime_100k));
    CHECK(qwen35_metal_persistent_runtime_bytes(
              QWEN35_CONTEXT_LENGTH, QWEN35_RESIDENT_PREFILL_TOKENS,
              &resident_runtime_max));
    CHECK(!qwen35_metal_persistent_runtime_bytes(
               QWEN35_CONTEXT_LENGTH + 1u,
               QWEN35_RESIDENT_PREFILL_TOKENS, &resident_runtime_max));
    CHECK(!qwen35_metal_persistent_runtime_bytes(
               8192, 0, &resident_runtime_8k));
    CHECK(!qwen35_metal_persistent_runtime_bytes(
               8192, QWEN35_RESIDENT_PREFILL_TOKENS + 1u,
               &resident_runtime_8k));
    CHECK(qwen35_metal_persistent_runtime_bytes(
              8192, QWEN35_PREFILL_MICRO_TOKENS, &ssd_runtime_8k));
    CHECK(qwen35_metal_persistent_runtime_bytes(
              100000, QWEN35_PREFILL_MICRO_TOKENS,
              &ssd_runtime_100k));
    CHECK(qwen35_metal_persistent_runtime_bytes(
              QWEN35_CONTEXT_LENGTH, QWEN35_PREFILL_MICRO_TOKENS,
              &ssd_runtime_max));
    CHECK(resident_runtime_32k - resident_runtime_8k ==
          (uint64_t)(32768u - 8192u) *
              QWEN35_METAL_GRAPH_CONTEXT_BYTES_PER_TOKEN);
    CHECK(QWEN35_FLASH_PREFILL_PAD_BYTES == UINT64_C(524288));
    CHECK(resident_runtime_100k > resident_runtime_32k);
    CHECK(resident_runtime_max > resident_runtime_100k);
    CHECK(resident_runtime_8k - ssd_runtime_8k ==
          UINT64_C(1582768128));
    CHECK(resident_runtime_100k - ssd_runtime_100k ==
          UINT64_C(1582768128));
    CHECK(resident_runtime_max - ssd_runtime_max ==
          UINT64_C(1582768128));
    CHECK(resident_8k.total_bytes == resident_runtime_8k);
    CHECK(ssd_resolved_8k.total_bytes == ssd_runtime_8k);

    ds4_residency_plan auto_to_ssd = {
        .requested = DS4_RESIDENCY_AUTO,
        .resolved = DS4_RESIDENCY_SSD,
        .model_bytes = UINT64_C(123456789),
        .runtime_bytes = resident_runtime_100k,
        .headroom_bytes = UINT64_C(987654321),
    };
    uint64_t resident_required = 0;
    uint64_t ssd_required = 0;
    CHECK(qwen35_u64_add(auto_to_ssd.model_bytes,
                         resident_runtime_100k, &resident_required));
    CHECK(qwen35_u64_add(resident_required, auto_to_ssd.headroom_bytes,
                         &resident_required));
    auto_to_ssd.required_bytes = resident_required;
    CHECK(qwen35_u64_add(auto_to_ssd.model_bytes,
                         ssd_runtime_100k, &ssd_required));
    CHECK(qwen35_u64_add(ssd_required, auto_to_ssd.headroom_bytes,
                         &ssd_required));
    CHECK(qwen35_metal_residency_plan_rebase_runtime(
              &auto_to_ssd, 100000, QWEN35_PREFILL_MICRO_TOKENS));
    CHECK(auto_to_ssd.runtime_bytes == ssd_runtime_100k);
    CHECK(auto_to_ssd.required_bytes == ssd_required);
    CHECK(resident_required - auto_to_ssd.required_bytes ==
          UINT64_C(1582768128));
}

static void test_qwen_residency_request_normalization(void) {
    ds4_engine_options opt = {0};
    ds4_residency_mode mode = DS4_RESIDENCY_SSD;

    opt.residency = DS4_RESIDENCY_AUTO;
    CHECK(ds4_engine_normalize_residency_request(&opt, &mode));
    CHECK(mode == DS4_RESIDENCY_AUTO);

    opt.residency = DS4_RESIDENCY_RESIDENT;
    CHECK(ds4_engine_normalize_residency_request(&opt, &mode));
    CHECK(mode == DS4_RESIDENCY_RESIDENT);

    opt.residency = DS4_RESIDENCY_SSD;
    CHECK(ds4_engine_normalize_residency_request(&opt, &mode));
    CHECK(mode == DS4_RESIDENCY_SSD);

    opt = (ds4_engine_options){0};
    opt.residency = DS4_RESIDENCY_AUTO;
    opt.ssd_streaming_cache_experts = 321;
    CHECK(ds4_engine_normalize_residency_request(&opt, &mode));
    CHECK(mode == DS4_RESIDENCY_SSD);

    opt.residency = DS4_RESIDENCY_RESIDENT;
    CHECK(!ds4_engine_normalize_residency_request(&opt, &mode));

    opt = (ds4_engine_options){0};
    opt.residency = DS4_RESIDENCY_AUTO;
    opt.ssd_streaming = true;
    CHECK(ds4_engine_normalize_residency_request(&opt, &mode));
    CHECK(mode == DS4_RESIDENCY_SSD);
}

typedef struct {
    ds4_model model;
    ds4_qwen35_weights weights;
    ds4_tensor tensors[QWEN35_N_TENSOR];
    uint32_t next_tensor;
    uint64_t next_offset;
    uint64_t page;
} qwen35_ssd_fixture;

static ds4_tensor *qwen35_ssd_fixture_add(
        qwen35_ssd_fixture *fixture,
        uint32_t            type,
        uint32_t            ndim,
        uint64_t            d0,
        uint64_t            d1,
        uint64_t            d2) {
    if (!fixture || fixture->next_tensor >= QWEN35_N_TENSOR ||
        ndim == 0 || ndim > 3) {
        return NULL;
    }
    const uint64_t dim[3] = {d0, d1, d2};
    uint64_t elements = 1;
    for (uint32_t i = 0; i < ndim; i++) {
        if (dim[i] == 0 || elements > UINT64_MAX / dim[i]) return NULL;
        elements *= dim[i];
    }
    uint64_t bytes = 0;
    if (!tensor_nbytes(type, elements, &bytes) || bytes == 0) return NULL;

    const uint64_t offset = align_up(fixture->next_offset, fixture->page);
    if (offset < fixture->next_offset || offset > UINT64_MAX - bytes ||
        offset < fixture->model.tensor_data_pos) {
        return NULL;
    }
    ds4_tensor *tensor = &fixture->tensors[fixture->next_tensor++];
    *tensor = (ds4_tensor){
        .ndim = ndim,
        .dim = {d0, d1, d2, 0},
        .type = type,
        .rel_offset = offset - fixture->model.tensor_data_pos,
        .abs_offset = offset,
        .elements = elements,
        .bytes = bytes,
    };
    fixture->next_offset = offset + bytes;
    return tensor;
}

static bool qwen35_ssd_fixture_make(
        qwen35_ssd_fixture       *fixture,
        ds4_qwen35_quant_profile  profile) {
    if (!fixture ||
        (profile != QWEN35_QUANT_PROFILE_Q2_K_XL &&
         profile != QWEN35_QUANT_PROFILE_MLX_AFFINE4_G64)) {
        return false;
    }
    memset(fixture, 0, sizeof(*fixture));
    fixture->weights.profile = profile;
    const bool affine =
        profile == QWEN35_QUANT_PROFILE_MLX_AFFINE4_G64;
    const long page_long = sysconf(_SC_PAGESIZE);
    if (page_long <= 0) return false;
    fixture->page = (uint64_t)page_long;
    if ((fixture->page & (fixture->page - 1u)) != 0) return false;
    fixture->model = (ds4_model){
        .fd = -1,
        .map = (const uint8_t *)(uintptr_t)1,
        .alignment = 32,
        .tensor_data_pos = 3u * fixture->page,
        .family = DS4_MODEL_FAMILY_QWEN35_MOE,
        .tensors = fixture->tensors,
    };
    fixture->next_offset = 5u * fixture->page;

#define FIXTURE_ADD(dst_, type_, ndim_, d0_, d1_, d2_) do {          \
    (dst_) = qwen35_ssd_fixture_add(                                 \
        fixture, (type_), (ndim_), (d0_), (d1_), (d2_));            \
    if (!(dst_)) return false;                                       \
} while (0)

    FIXTURE_ADD(fixture->weights.token_embd,
                affine ? DS4_TENSOR_Q8_0 : DS4_TENSOR_Q5_K, 2,
                QWEN35_N_EMBD, QWEN35_N_VOCAB, 0);
    for (uint32_t il = 0; il < QWEN35_N_LAYER; il++) {
        ds4_qwen35_layer_weights *layer = &fixture->weights.layer[il];
        FIXTURE_ADD(layer->attn_norm, DS4_TENSOR_F32, 1,
                    QWEN35_N_EMBD, 0, 0);
        FIXTURE_ADD(layer->post_attention_norm, DS4_TENSOR_F32, 1,
                    QWEN35_N_EMBD, 0, 0);
        if (ds4_qwen35_layer_is_full_attention(il)) {
            FIXTURE_ADD(layer->attn_q,
                        affine ? DS4_TENSOR_Q8_0 : DS4_TENSOR_Q5_K, 2,
                        QWEN35_N_EMBD, 8192, 0);
            FIXTURE_ADD(layer->attn_k,
                        affine ? DS4_TENSOR_Q8_0 : DS4_TENSOR_Q6_K, 2,
                        QWEN35_N_EMBD, 512, 0);
            FIXTURE_ADD(layer->attn_v,
                        affine ? DS4_TENSOR_Q8_0 : DS4_TENSOR_Q6_K, 2,
                        QWEN35_N_EMBD, 512, 0);
            FIXTURE_ADD(layer->attn_output,
                        affine ? DS4_TENSOR_Q8_0 : DS4_TENSOR_Q5_K, 2,
                        4096, QWEN35_N_EMBD, 0);
            FIXTURE_ADD(layer->attn_q_norm, DS4_TENSOR_F32, 1,
                        QWEN35_N_HEAD_DIM, 0, 0);
            FIXTURE_ADD(layer->attn_k_norm, DS4_TENSOR_F32, 1,
                        QWEN35_N_HEAD_DIM, 0, 0);
        } else {
            const uint32_t recurrent_dense_type = affine
                ? DS4_TENSOR_Q8_0
                : (il == 1u ? DS4_TENSOR_Q6_K : DS4_TENSOR_Q5_K);
            FIXTURE_ADD(layer->attn_gate, recurrent_dense_type, 2,
                        QWEN35_N_EMBD, 4096, 0);
            FIXTURE_ADD(layer->attn_qkv, recurrent_dense_type, 2,
                        QWEN35_N_EMBD, 8192, 0);
            FIXTURE_ADD(layer->ssm_a, DS4_TENSOR_F32, 1,
                        QWEN35_SSM_VALUE_HEAD, 0, 0);
            FIXTURE_ADD(layer->ssm_alpha, DS4_TENSOR_F32, 2,
                        QWEN35_N_EMBD, QWEN35_SSM_DT_RANK, 0);
            FIXTURE_ADD(layer->ssm_beta, DS4_TENSOR_F32, 2,
                        QWEN35_N_EMBD, QWEN35_SSM_DT_RANK, 0);
            FIXTURE_ADD(layer->ssm_conv1d, DS4_TENSOR_F32, 2,
                        QWEN35_SSM_CONV_KERNEL,
                        QWEN35_SSM_CONV_CHANNEL, 0);
            FIXTURE_ADD(layer->ssm_dt, DS4_TENSOR_F32, 1,
                        QWEN35_SSM_DT_RANK, 0, 0);
            FIXTURE_ADD(layer->ssm_norm, DS4_TENSOR_F32, 1,
                        QWEN35_SSM_STATE, 0, 0);
            FIXTURE_ADD(layer->ssm_out,
                        affine ? DS4_TENSOR_Q8_0 : DS4_TENSOR_Q6_K, 2,
                        QWEN35_SSM_INNER, QWEN35_N_EMBD, 0);
        }
        FIXTURE_ADD(layer->ffn_gate_inp, DS4_TENSOR_F32, 2,
                    QWEN35_N_EMBD, QWEN35_N_EXPERT, 0);
        uint32_t routed_gate_type =
            affine ? DS4_TENSOR_Q4_K : DS4_TENSOR_IQ2_XS;
        uint32_t routed_down_type =
            affine ? DS4_TENSOR_Q4_K : DS4_TENSOR_IQ3_XXS;
        if (!affine && il == 1u) {
            routed_gate_type = DS4_TENSOR_IQ3_XXS;
            routed_down_type = DS4_TENSOR_IQ4_XS;
        } else if (!affine &&
                   (il == 34u || il == 38u || il == 39u)) {
            routed_down_type = DS4_TENSOR_IQ4_XS;
        }
        FIXTURE_ADD(layer->ffn_gate_exps, routed_gate_type, 3,
                    QWEN35_N_EMBD, QWEN35_N_FF_EXP,
                    QWEN35_N_EXPERT);
        FIXTURE_ADD(layer->ffn_up_exps, routed_gate_type, 3,
                    QWEN35_N_EMBD, QWEN35_N_FF_EXP,
                    QWEN35_N_EXPERT);
        FIXTURE_ADD(layer->ffn_down_exps, routed_down_type, 3,
                    QWEN35_N_FF_EXP, QWEN35_N_EMBD,
                    QWEN35_N_EXPERT);
        FIXTURE_ADD(layer->ffn_gate_inp_shexp, DS4_TENSOR_F32, 1,
                    QWEN35_N_EMBD, 0, 0);
        const uint32_t shared_gate_type = affine
            ? DS4_TENSOR_Q8_0
            : (il == 1u ? DS4_TENSOR_Q6_K : DS4_TENSOR_Q5_K);
        const uint32_t shared_down_type = affine
            ? DS4_TENSOR_Q8_0
            : (il == 1u ? DS4_TENSOR_Q8_0 : DS4_TENSOR_Q6_K);
        FIXTURE_ADD(layer->ffn_gate_shexp, shared_gate_type, 2,
                    QWEN35_N_EMBD, QWEN35_N_FF_SHARED, 0);
        FIXTURE_ADD(layer->ffn_up_shexp, shared_gate_type, 2,
                    QWEN35_N_EMBD, QWEN35_N_FF_SHARED, 0);
        FIXTURE_ADD(layer->ffn_down_shexp, shared_down_type, 2,
                    QWEN35_N_FF_SHARED, QWEN35_N_EMBD, 0);
    }
    FIXTURE_ADD(fixture->weights.output_norm, DS4_TENSOR_F32, 1,
                QWEN35_N_EMBD, 0, 0);
    FIXTURE_ADD(fixture->weights.output,
                affine ? DS4_TENSOR_Q8_0 : DS4_TENSOR_Q4_K, 2,
                QWEN35_N_EMBD, QWEN35_N_VOCAB, 0);
#undef FIXTURE_ADD

    if (fixture->next_tensor != QWEN35_N_TENSOR) return false;
    fixture->model.n_tensors = fixture->next_tensor;
    fixture->model.size = align_up(fixture->next_offset, fixture->page);
    return fixture->model.size >= fixture->next_offset;
}

static bool qwen35_ssd_fixture_is_routed(
        const qwen35_ssd_fixture *fixture,
        const ds4_tensor         *tensor) {
    for (uint32_t il = 0; il < QWEN35_N_LAYER; il++) {
        const ds4_qwen35_layer_weights *layer = &fixture->weights.layer[il];
        if (tensor == layer->ffn_gate_exps ||
            tensor == layer->ffn_up_exps ||
            tensor == layer->ffn_down_exps) {
            return true;
        }
    }
    return false;
}

static bool qwen35_span_covers_tensor(
        const ds4_model_map_span_vec *spans,
        const ds4_tensor             *tensor) {
    if (!spans || !tensor || tensor->abs_offset > UINT64_MAX - tensor->bytes) {
        return false;
    }
    const uint64_t end = tensor->abs_offset + tensor->bytes;
    for (uint32_t i = 0; i < spans->len; i++) {
        if (spans->v[i].off <= tensor->abs_offset &&
            spans->v[i].end >= end) {
            return true;
        }
    }
    return false;
}

static bool qwen35_span_overlaps_tensor(
        const ds4_model_map_span_vec *spans,
        const ds4_tensor             *tensor) {
    if (!spans || !tensor || tensor->abs_offset > UINT64_MAX - tensor->bytes) {
        return true;
    }
    const uint64_t end = tensor->abs_offset + tensor->bytes;
    for (uint32_t i = 0; i < spans->len; i++) {
        if (spans->v[i].off < end && tensor->abs_offset < spans->v[i].end) {
            return true;
        }
    }
    return false;
}

static void test_qwen35_ssd_static_contract(void) {
    qwen35_ssd_fixture *fixture = calloc(1, sizeof(*fixture));
    CHECK(fixture != NULL);
    if (!fixture) return;
    CHECK(qwen35_ssd_fixture_make(
              fixture, QWEN35_QUANT_PROFILE_Q2_K_XL));
    if (fixture->next_tensor != QWEN35_N_TENSOR) {
        free(fixture);
        return;
    }

    ds4_qwen35_streaming_cache_geometry geometry;
    memset(&geometry, 0xa5, sizeof(geometry));
    CHECK(!qwen35_streaming_cache_geometry_make(NULL, &geometry));
    CHECK(geometry.per_expert_bytes == 0);
    CHECK(!qwen35_streaming_cache_geometry_make(&fixture->weights, NULL));
    CHECK(qwen35_streaming_cache_geometry_make(
              &fixture->weights, &geometry));
    CHECK(geometry.gate_expert_bytes == UINT64_C(401408));
    CHECK(geometry.up_expert_bytes == UINT64_C(401408));
    CHECK(geometry.down_expert_bytes == UINT64_C(557056));
    CHECK(geometry.per_expert_bytes == UINT64_C(1359872));
    CHECK(geometry.cacheable_layers == 40);
    CHECK(geometry.experts_per_layer == 256);
    CHECK(geometry.selected_per_layer == 8);
    CHECK(geometry.working_set_experts == 320);
    CHECK(geometry.minimum_cache_experts == 321);
    CHECK(geometry.minimum_cache_bytes == UINT64_C(436518912));
    CHECK(geometry.warning_cache_experts == 640);
    CHECK(geometry.max_cacheable_experts == 10240);

    uint32_t cache_experts = 99;
    CHECK(!qwen35_streaming_cache_budget_from_request(
              NULL, 0, 0, 321, &cache_experts));
    CHECK(cache_experts == 0);
    CHECK(!qwen35_streaming_cache_budget_from_request(
              &geometry, 321, geometry.minimum_cache_bytes, 0,
              &cache_experts));
    CHECK(!qwen35_streaming_cache_budget_from_request(
              &geometry, 320, 0, 0, &cache_experts));
    CHECK(!qwen35_streaming_cache_budget_from_request(
              &geometry, 0, geometry.minimum_cache_bytes - 1u, 0,
              &cache_experts));
    CHECK(qwen35_streaming_cache_budget_from_request(
              &geometry, 0, geometry.minimum_cache_bytes, 0,
              &cache_experts));
    CHECK(cache_experts == 321);
    CHECK(qwen35_streaming_cache_budget_from_request(
              &geometry, 640, 0, 0, &cache_experts));
    CHECK(cache_experts == 640);
    CHECK(qwen35_streaming_cache_budget_from_request(
              &geometry, 0, 0, 321, &cache_experts));
    CHECK(cache_experts == 321);
    CHECK(!qwen35_streaming_cache_budget_from_request(
              &geometry, 10241, 0, 0, &cache_experts));

    ds4_tensor *mutated = fixture->weights.layer[7].ffn_gate_exps;
    const ds4_tensor saved_mutated = *mutated;
    mutated->dim[1]--;
    CHECK(!qwen35_streaming_cache_geometry_make(
              &fixture->weights, &geometry));
    *mutated = saved_mutated;
    mutated->type = DS4_TENSOR_F16;
    CHECK(!qwen35_streaming_cache_geometry_make(
              &fixture->weights, &geometry));
    *mutated = saved_mutated;
    mutated->bytes--;
    CHECK(!qwen35_streaming_cache_geometry_make(
              &fixture->weights, &geometry));
    *mutated = saved_mutated;

    ds4_model_map_span_vec spans = {0};
    uint64_t payload = 0;
    CHECK(qwen35_weights_model_map_non_routed_spans(
              &fixture->model, &fixture->weights, &spans, &payload));
    CHECK(payload == QWEN35_Q2_NON_ROUTED_PAYLOAD_BYTES);
    CHECK(payload == UINT64_C(1751935488));
    CHECK(model_map_span_vec_total_bytes(&spans) == payload);
    CHECK(spans.len != 0);
    CHECK(spans.v[0].off == fixture->weights.token_embd->abs_offset);

    uint32_t routed_count = 0;
    uint32_t static_count = 0;
    uint64_t independent_payload = 0;
    for (uint32_t i = 0; i < QWEN35_N_TENSOR; i++) {
        const ds4_tensor *tensor = &fixture->tensors[i];
        if (qwen35_ssd_fixture_is_routed(fixture, tensor)) {
            routed_count++;
            CHECK(!qwen35_span_overlaps_tensor(&spans, tensor));
        } else {
            static_count++;
            CHECK(qwen35_span_covers_tensor(&spans, tensor));
            CHECK(independent_payload <= UINT64_MAX - tensor->bytes);
            independent_payload += tensor->bytes;
        }
    }
    CHECK(routed_count == QWEN35_ROUTED_EXPERT_TENSOR_COUNT);
    CHECK(static_count == QWEN35_NON_ROUTED_TENSOR_COUNT);
    CHECK(independent_payload == payload);
    free(spans.v);

    uint64_t page_payload = 0;
    uint64_t page_coverage = 0;
    memset(&spans, 0, sizeof(spans));
    CHECK(qwen35_weights_model_map_non_routed_page_spans(
              &fixture->model, &fixture->weights, &spans,
              &page_payload, &page_coverage));
    CHECK(page_payload == payload);
    CHECK(page_coverage == model_map_span_vec_total_bytes(&spans));
    CHECK(page_coverage >= page_payload);
    for (uint32_t i = 0; i < spans.len; i++) {
        CHECK(spans.v[i].off % fixture->page == 0);
        CHECK(spans.v[i].end % fixture->page == 0);
        CHECK(spans.v[i].end > spans.v[i].off);
        if (i != 0) CHECK(spans.v[i].off > spans.v[i - 1u].end);
    }
    for (uint32_t i = 0; i < QWEN35_N_TENSOR; i++) {
        const ds4_tensor *tensor = &fixture->tensors[i];
        if (qwen35_ssd_fixture_is_routed(fixture, tensor)) {
            CHECK(!qwen35_span_overlaps_tensor(&spans, tensor));
        } else {
            CHECK(qwen35_span_covers_tensor(&spans, tensor));
        }
    }
    fprintf(stderr,
            "Qwen SSD contract: static=%" PRIu64 " bytes, "
            "page coverage=%" PRIu64 " bytes across %u spans\n",
            page_payload, page_coverage, spans.len);
    free(spans.v);

    memset(&spans, 0, sizeof(spans));
    ds4_tensor *saved_output = fixture->weights.output;
    fixture->weights.output = fixture->weights.token_embd;
    CHECK(!qwen35_weights_model_map_non_routed_spans(
              &fixture->model, &fixture->weights, &spans, &payload));
    CHECK(spans.v == NULL && payload == 0);
    fixture->weights.output = saved_output;

    const uint64_t saved_size = fixture->model.size;
    const uint64_t saved_output_abs = saved_output->abs_offset;
    const uint64_t saved_output_rel = saved_output->rel_offset;
    saved_output->abs_offset = UINT64_MAX - 7u;
    saved_output->rel_offset = saved_output->abs_offset -
                               fixture->model.tensor_data_pos;
    CHECK(!qwen35_weights_model_map_non_routed_spans(
              &fixture->model, &fixture->weights, &spans, &payload));
    CHECK(spans.v == NULL && payload == 0);
    saved_output->abs_offset = saved_output_abs;
    saved_output->rel_offset = saved_output_rel;

    fixture->model.n_tensors--;
    CHECK(!qwen35_weights_model_map_non_routed_spans(
              &fixture->model, &fixture->weights, &spans, &payload));
    fixture->model.n_tensors++;

    const uint32_t saved_output_type = saved_output->type;
    const uint64_t saved_output_bytes = saved_output->bytes;
    saved_output->type = DS4_TENSOR_F16;
    CHECK(tensor_nbytes(saved_output->type, saved_output->elements,
                        &saved_output->bytes));
    fixture->model.size = align_up(saved_output->abs_offset +
                                   saved_output->bytes, fixture->page);
    CHECK(!qwen35_weights_model_map_non_routed_spans(
              &fixture->model, &fixture->weights, &spans, &payload));
    CHECK(spans.v == NULL && payload == 0);
    saved_output->type = saved_output_type;
    saved_output->bytes = saved_output_bytes;
    fixture->model.size = saved_size;

    ds4_tensor *output_norm = fixture->weights.output_norm;
    saved_output->abs_offset = output_norm->abs_offset;
    saved_output->rel_offset = output_norm->rel_offset;
    CHECK(!qwen35_weights_model_map_non_routed_spans(
              &fixture->model, &fixture->weights, &spans, &payload));
    CHECK(spans.v == NULL && payload == 0);
    saved_output->abs_offset = saved_output_abs;
    saved_output->rel_offset = saved_output_rel;

    const uint8_t *saved_map = fixture->model.map;
    fixture->model.map = NULL;
    page_payload = 123;
    page_coverage = 456;
    CHECK(!qwen35_weights_model_map_non_routed_page_spans(
              &fixture->model, &fixture->weights, &spans,
              &page_payload, &page_coverage));
    CHECK(spans.v == NULL && page_payload == 0 && page_coverage == 0);
    fixture->model.map = saved_map;

    free(fixture);

    /* Affine4 and Q2 share this planner, so replacing the old fixture with
     * the smaller profile would silently drop coverage of the published
     * artifact's larger physical slab and non-routed map. */
    fixture = calloc(1, sizeof(*fixture));
    CHECK(fixture != NULL);
    if (!fixture) return;
    CHECK(qwen35_ssd_fixture_make(
              fixture, QWEN35_QUANT_PROFILE_MLX_AFFINE4_G64));
    memset(&geometry, 0, sizeof(geometry));
    CHECK(qwen35_streaming_cache_geometry_make(
              &fixture->weights, &geometry));
    CHECK(geometry.gate_expert_bytes == UINT64_C(589824));
    CHECK(geometry.up_expert_bytes == UINT64_C(589824));
    CHECK(geometry.down_expert_bytes == UINT64_C(589824));
    CHECK(geometry.per_expert_bytes == UINT64_C(1769472));
    CHECK(geometry.minimum_cache_experts == 321);
    CHECK(geometry.minimum_cache_bytes == UINT64_C(568000512));
    CHECK(geometry.warning_cache_experts == 640);
    CHECK(geometry.max_cacheable_experts == 10240);

    memset(&spans, 0, sizeof(spans));
    payload = 0;
    CHECK(qwen35_weights_model_map_non_routed_spans(
              &fixture->model, &fixture->weights, &spans, &payload));
    CHECK(payload == QWEN35_AFFINE_NON_ROUTED_PAYLOAD_BYTES);
    CHECK(payload == UINT64_C(2678180352));
    CHECK(model_map_span_vec_total_bytes(&spans) == payload);
    free(spans.v);
    free(fixture);
}

static sample_outcome test_sample_outcome(
        const float *logits,
        uint32_t     n_vocab,
        float        temperature,
        int          top_k,
        float        top_p,
        float        min_p,
        uint64_t    *rng) {
    sample_outcome out = {
        .token = -1,
        .drew = false,
        .rng_after = 0,
    };
    const int token = sample_top_p_min_p(
        logits, n_vocab, temperature, top_k, top_p, min_p, rng, &out);
    CHECK(token == out.token);
    return out;
}

static void test_sampler_outcomes(void) {
    const uint64_t seed = UINT64_C(0x0123456789abcdef);

    /* Greedy and no-finite fallbacks must not require an RNG pointer. */
    const float greedy_logits[] = {1.0f, 3.0f, 3.0f, 2.0f};
    sample_outcome out = test_sample_outcome(
        greedy_logits, 4, 0.0f, 0, 1.0f, 0.0f, NULL);
    CHECK(out.token == 1 && out.rng_after == 0 && !out.drew);
    CHECK(ds4_sample_logits(greedy_logits, 4, 0.0f, 0, 1.0f, 0.0f,
                            NULL) == 1);

    const float nonfinite_logits[] = {NAN, INFINITY, -INFINITY};
    out = test_sample_outcome(
        nonfinite_logits, 3, 1.0f, 0, 1.0f, 0.0f, NULL);
    CHECK(out.token == 1 && out.rng_after == 0 && !out.drew);
    out = test_sample_outcome(
        nonfinite_logits, 3, 1.0f, 3, 1.0f, 0.0f, NULL);
    CHECK(out.token == 1 && out.rng_after == 0 && !out.drew);

    /* A non-finite probability sum returns the best token without drawing. */
    const float two_logits[] = {2.0f, 0.0f};
    uint64_t rng = seed;
    out = test_sample_outcome(two_logits, 2, NAN, 0, 1.0f, 0.0f, &rng);
    CHECK(out.token == 0 && out.rng_after == seed && !out.drew);
    CHECK(rng == seed);
    rng = seed;
    out = test_sample_outcome(two_logits, 2, NAN, 2, 1.0f, 0.0f, &rng);
    CHECK(out.token == 0 && out.rng_after == seed && !out.drew);
    CHECK(rng == seed);

    /* min_p > 1 filters even the maximum candidate in the full-vocab path.
     * This is the reachable n == 0 fallback.  Once n is nonzero, each filter
     * loop necessarily retains its first candidate, so filtered == 0 is only
     * a defensive guard. */
    rng = seed;
    out = test_sample_outcome(two_logits, 2, 1.0f, 0, 1.0f, 2.0f, &rng);
    CHECK(out.token == 0 && out.rng_after == seed && !out.drew);
    CHECK(rng == seed);

    /* Exercise the fast helper's <= 0 sum guard with deliberately inconsistent
     * white-box bookkeeping; no public call can produce finite > 0 while all
     * logits are non-finite. */
    sample_outcome guarded = {
        .token = -17,
        .rng_after = UINT64_C(0xdeadbeef),
        .drew = true,
    };
    rng = seed;
    int guarded_token = -17;
    CHECK(sample_fast_top_p(nonfinite_logits, 3, 1, 0.0f, 2,
                            1.0f, 0.5f, 0.0f, &rng,
                            &guarded_token, &guarded));
    CHECK(guarded_token == guarded.token);
    CHECK(guarded.token == 2 && guarded.rng_after == seed && !guarded.drew);
    CHECK(rng == seed);
    guarded = (sample_outcome){
        .token = -19,
        .rng_after = UINT64_C(0xcafef00d),
        .drew = true,
    };
    guarded_token = -19;
    CHECK(!sample_fast_top_p(nonfinite_logits, 3, 0, 0.0f, 2,
                             1.0f, 0.5f, 0.0f, NULL,
                             &guarded_token, &guarded));
    CHECK(guarded_token == -19);
    CHECK(guarded.token == -19 && guarded.rng_after == UINT64_C(0xcafef00d) &&
          guarded.drew);

    /* One retained token still performs a categorical draw.  This deliberately
     * returns the same token as the fallbacks above while distinguishing drew
     * and the exact zero-seed xorshift state. */
    rng = 0;
    out = test_sample_outcome(two_logits, 2, 1.0f, 0, 0.5f, 0.0f, &rng);
    CHECK(out.token == 0 && out.drew);
    CHECK(out.rng_after == UINT64_C(0x03f721dffe39b342));
    CHECK(rng == out.rng_after);

    /* Force fast-top-p to decline its 512-entry heap.  The full fallback owns
     * the one and only draw; the declined helper owns none. */
    float broad_logits[600];
    for (uint32_t i = 0; i < 600; i++) {
        broad_logits[i] = -(float)i / 1000.0f;
    }
    guarded = (sample_outcome){
        .token = -23,
        .rng_after = UINT64_C(0x1234),
        .drew = true,
    };
    rng = 1;
    guarded_token = -23;
    CHECK(!sample_fast_top_p(broad_logits, 600, 600, 0.0f, 0,
                             1.0f, 0.9f, 0.0f, &rng,
                             &guarded_token, &guarded));
    CHECK(guarded_token == -23);
    CHECK(rng == 1 && guarded.token == -23 &&
          guarded.rng_after == UINT64_C(0x1234) && guarded.drew);
    rng = 1;
    out = test_sample_outcome(
        broad_logits, 600, 1.0f, 0, 0.9f, 0.0f, &rng);
    CHECK(out.drew && out.rng_after == UINT64_C(0x0000000002000001));
    CHECK(rng == out.rng_after);
    uint64_t wrapper_rng = 1;
    CHECK(ds4_sample_logits(broad_logits, 600, 1.0f, 0, 0.9f, 0.0f,
                            &wrapper_rng) == out.token);
    CHECK(wrapper_rng == out.rng_after);

    /* top_k keeps its existing earlier-id tie order and reports the draw made
     * after filtering. */
    const float tied_logits[] = {2.0f, 2.0f, 0.0f};
    rng = UINT64_C(0x5830920757d41153);
    out = test_sample_outcome(tied_logits, 3, 1.0f, 2, 1.0f, 0.0f, &rng);
    CHECK(out.token == 1 && out.drew);
    CHECK(out.rng_after == UINT64_C(0x44da53dec8eb16d8));
    CHECK(rng == out.rng_after);

    /* Canonicalized controls must be exactly equivalent, including RNG. */
    uint64_t canonical_rng = 1;
    sample_outcome canonical = test_sample_outcome(
        tied_logits, 3, 1.0f, 0, 1.0f, 0.0f, &canonical_rng);
    rng = 1;
    out = test_sample_outcome(
        tied_logits, 3, 1.0f, -7, 0.0f, -0.5f, &rng);
    CHECK(out.token == canonical.token && out.rng_after == canonical.rng_after &&
          out.drew == canonical.drew && rng == canonical_rng);

    canonical_rng = 1;
    canonical = test_sample_outcome(
        tied_logits, 3, 1.0f, 3, 1.0f, 0.0f, &canonical_rng);
    rng = 1;
    out = test_sample_outcome(
        tied_logits, 3, 1.0f, 4096, 2.0f, 0.0f, &rng);
    CHECK(out.token == canonical.token && out.rng_after == canonical.rng_after &&
          out.drew == canonical.drew && rng == canonical_rng);

    /* Public invalid-input behavior must not acquire RNG ownership. */
    rng = seed;
    CHECK(ds4_sample_logits(NULL, 2, 1.0f, 0, 1.0f, 0.0f, &rng) == 0);
    CHECK(ds4_sample_logits(two_logits, 0, 1.0f, 0, 1.0f, 0.0f, &rng) == 0);
    CHECK(rng == seed);
}

static void test_dynamic_logits(ds4_session *session) {
    const uint32_t n_vocab = QWEN35_N_VOCAB;
    CHECK(ds4_engine_vocab_size(session->engine) == QWEN35_N_VOCAB);
    CHECK(ds4_engine_effective_vocab_size(session->engine) ==
          QWEN35_N_VALID_TOKEN);
    CHECK(ds4_engine_routed_quant_bits(session->engine) == 4);
    CHECK(ds4_engine_has_output_head(session->engine));
    CHECK(ds4_engine_set_power(session->engine, 99) != 0);
    CHECK(ds4_engine_set_power(session->engine, 100) == 0);
    CHECK(ds4_session_set_power(session, 99) != 0);
    CHECK(ds4_session_set_power(session, 100) == 0);
    CHECK(ds4_session_power(session) == 100);
    float *input = malloc((size_t)n_vocab * sizeof(input[0]));
    float *copy = malloc(((size_t)n_vocab + 1u) * sizeof(copy[0]));
    CHECK(input != NULL && copy != NULL);
    if (!input || !copy) {
        free(input);
        free(copy);
        return;
    }

    for (uint32_t i = 0; i < n_vocab; i++) input[i] = -INFINITY;
    input[200000] = 3.0f;
    input[QWEN35_N_VALID_TOKEN - 1u] = 4.0f;
    input[n_vocab - 1u] = 5.0f;
    CHECK(ds4_session_set_logits(session, input, (int)n_vocab) == 0);
    CHECK(ds4_session_set_logits(session, input, (int)DS4_N_VOCAB) != 0);
    CHECK(ds4_session_set_logits(session, input, (int)n_vocab - 1) != 0);
    CHECK(ds4_session_argmax(session) ==
          (int)QWEN35_N_VALID_TOKEN - 1);
    CHECK(ds4_session_argmax_excluding(
              session, (int)QWEN35_N_VALID_TOKEN - 1) == 200000);

    uint64_t rng = UINT64_C(0x123456789abcdef0);
    CHECK(ds4_session_sample(session, 0.0f, 0, 1.0f, 0.0f, &rng) ==
          (int)QWEN35_N_VALID_TOKEN - 1);

    uint64_t direct_rng = 0;
    sample_outcome direct = test_sample_outcome(
        session->logits, ds4_session_selectable_vocab_size(session),
        1.0f, 2, 1.0f, 0.0f, &direct_rng);
    uint64_t session_rng = 0;
    CHECK(ds4_session_sample(session, 1.0f, 2, 1.0f, 0.0f, &session_rng) ==
          direct.token);
    CHECK(direct.drew && session_rng == direct.rng_after &&
          direct_rng == direct.rng_after);

    ds4_token_score top[2];
    CHECK(ds4_session_top_logprobs(session, top, 2) == 2);
    CHECK(top[0].id == (int)QWEN35_N_VALID_TOKEN - 1);
    CHECK(top[1].id == 200000);
    CHECK(top[0].logprob > top[1].logprob);
    CHECK(top[0].logprob > -1.0f); /* UNUSED logit is not normalized. */
    ds4_token_score one;
    CHECK(ds4_session_token_logprob(
              session, (int)QWEN35_N_VALID_TOKEN - 1, &one) == 1);
    CHECK(one.id == (int)QWEN35_N_VALID_TOKEN - 1 && isfinite(one.logprob));
    CHECK(ds4_session_token_logprob(
              session, (int)QWEN35_N_VALID_TOKEN, &one) == 0);
    CHECK(ds4_session_token_logprob(session, (int)n_vocab, &one) == 0);

    copy[0] = 91.0f;
    CHECK(ds4_session_copy_logits(session, copy, (int)n_vocab - 1) == 0);
    CHECK(copy[0] == 91.0f);
    copy[n_vocab] = -1234.5f;
    CHECK(ds4_session_copy_logits(session, copy, (int)n_vocab) ==
          (int)n_vocab);
    CHECK(copy[200000] == 3.0f);
    CHECK(copy[n_vocab - 1u] == 5.0f);
    CHECK(copy[n_vocab] == -1234.5f);

    free(copy);
    free(input);
}

static void test_eval_transaction(ds4_session *session) {
    char err[160] = {0};
    ds4_session_invalidate(session);
    stub_reset();

    CHECK(ds4_session_eval_qwen35_with_forward(
              session, 7, err, sizeof(err), stub_forward) == 0);
    CHECK(stub_calls == 1 && stub_position[0] == 0 && stub_token[0] == 7);
    CHECK(session->checkpoint_valid && session->checkpoint.len == 1);
    CHECK(session->checkpoint.v[0] == 7);
    CHECK(session->qwen35_cpu_cache.n_tokens == 1);

    CHECK(ds4_session_eval_qwen35_with_forward(
              session, -1, err, sizeof(err), stub_forward) != 0);
    CHECK(ds4_session_eval_qwen35_with_forward(
              session, QWEN35_N_VALID_TOKEN,
              err, sizeof(err), stub_forward) != 0);
    CHECK(ds4_session_eval(session, QWEN35_N_VALID_TOKEN,
                           err, sizeof(err)) != 0);
    CHECK(strstr(err, "Qwen token id") != NULL);
    CHECK(stub_calls == 1);
    CHECK(session->checkpoint.len == 1 &&
          session->qwen35_cpu_cache.n_tokens == 1);

    CHECK(ds4_session_eval_qwen35_with_forward(
              session, 8, err, sizeof(err),
              stub_true_without_advance) != 0);
    CHECK(session->checkpoint.len == 0 && !session->checkpoint_valid);
    CHECK(session->qwen35_cpu_cache.n_tokens == 0);

    stub_reset();
    stub_fail_on_call = 1;
    CHECK(ds4_session_eval_qwen35_with_forward(
              session, 9, err, sizeof(err), stub_forward) != 0);
    CHECK(session->checkpoint.len == 0 && !session->checkpoint_valid);
    CHECK(session->qwen35_cpu_cache.n_tokens == 0);
    CHECK(session->qwen35_cpu_cache.layer[0].conv[0] == 0.0f);
    CHECK(session->qwen35_cpu_cache.layer[0].recurrent[0] == 0.0f);
    /* Reset is O(fixed GDN state); stale K/V is deliberately invisible. */
    CHECK(session->qwen35_cpu_cache.layer[3].key[0] == 23.0f);

    stub_reset();
    CHECK(ds4_session_eval_qwen35_with_forward(
              session, 10, err, sizeof(err), stub_forward) == 0);
    CHECK(stub_position[0] == 0);
    session->qwen35_cpu_cache.layer[0].conv[0] = 31.0f;
    session->qwen35_cpu_cache.layer[0].recurrent[0] = -37.0f;
    ds4_session_invalidate(session);
    CHECK(session->checkpoint.len == 0 &&
          session->qwen35_cpu_cache.n_tokens == 0);
    CHECK(session->qwen35_cpu_cache.layer[0].conv[0] == 0.0f);
    CHECK(session->qwen35_cpu_cache.layer[0].recurrent[0] == 0.0f);

    stub_reset();
    CHECK(ds4_session_eval_qwen35_with_forward(
              session, 1, err, sizeof(err), stub_forward) == 0);
    CHECK(ds4_session_eval_qwen35_with_forward(
              session, 2, err, sizeof(err), stub_forward) == 0);
    const int calls_before_mismatch = stub_calls;
    session->qwen35_cpu_cache.n_tokens = 1;
    CHECK(ds4_session_eval_qwen35_with_forward(
              session, 3, err, sizeof(err), stub_forward) != 0);
    CHECK(stub_calls == calls_before_mismatch);
    CHECK(session->checkpoint.len == 0 &&
          session->qwen35_cpu_cache.n_tokens == 0);

    stub_reset();
    for (int token = 0; token < session->ctx_size; token++) {
        CHECK(ds4_session_eval_qwen35_with_forward(
                  session, token, err, sizeof(err), stub_forward) == 0);
    }
    CHECK(session->checkpoint.len == session->ctx_size);
    CHECK(session->qwen35_cpu_cache.n_tokens == (uint32_t)session->ctx_size);
    const int calls_at_full = stub_calls;
    CHECK(ds4_session_eval_qwen35_with_forward(
              session, 99, err, sizeof(err), stub_forward) != 0);
    CHECK(stub_calls == calls_at_full);
    CHECK(session->checkpoint.len == session->ctx_size);
    ds4_session_rewind(session, session->checkpoint.len);
    CHECK(session->checkpoint.len == session->ctx_size);
    ds4_session_rewind(session, 2);
    CHECK(session->checkpoint.len == 0 && !session->checkpoint_valid);
    CHECK(session->qwen35_cpu_cache.n_tokens == 0);
}

static void test_sync_transaction(ds4_session *session) {
    char err[160] = {0};
    int first_values[] = {11, 12};
    ds4_tokens first = {.v = first_values, .len = 2, .cap = 2};
    int extended_values[] = {11, 12, 13};
    ds4_tokens extended = {.v = extended_values, .len = 3, .cap = 3};

    ds4_session_invalidate(session);
    stub_reset();
    CHECK(ds4_session_sync_qwen35_with_forward(
              session, &first, err, sizeof(err), stub_forward) == 0);
    CHECK(stub_calls == 2 && stub_logits_calls == 1);
    CHECK(stub_logits_position[0] == 1 && stub_logits_token[0] == 12);
    CHECK(session->logits[QWEN35_N_VOCAB - 1u] == 12.0f);
    CHECK(stub_position[0] == 0 && stub_position[1] == 1);
    CHECK(session->checkpoint.len == 2 &&
          session->qwen35_cpu_cache.n_tokens == 2);

    CHECK(ds4_session_sync_qwen35_with_forward(
              session, &first, err, sizeof(err), stub_forward) == 0);
    CHECK(stub_calls == 2);
    CHECK(ds4_session_sync_qwen35_with_forward(
              session, &extended, err, sizeof(err), stub_forward) == 0);
    CHECK(stub_calls == 3 && stub_logits_calls == 2);
    CHECK(stub_logits_position[1] == 2 && stub_logits_token[1] == 13);
    CHECK(session->logits[QWEN35_N_VOCAB - 1u] == 13.0f);
    CHECK(stub_position[2] == 2 && stub_token[2] == 13);

    int invalid_values[] = {11, QWEN35_N_VALID_TOKEN};
    ds4_tokens invalid = {.v = invalid_values, .len = 2, .cap = 2};
    CHECK(ds4_session_sync_qwen35_with_forward(
              session, &invalid, err, sizeof(err), stub_forward) != 0);
    CHECK(ds4_session_sync(session, &invalid, err, sizeof(err)) != 0);
    CHECK(strstr(err, "Qwen token id") != NULL);
    CHECK(stub_calls == 3);
    CHECK(session->checkpoint.len == 3 &&
          session->qwen35_cpu_cache.n_tokens == 3);

    int divergent_values[] = {21, 22};
    ds4_tokens divergent = {.v = divergent_values, .len = 2, .cap = 2};
    CHECK(ds4_session_sync_qwen35_with_forward(
              session, &divergent, err, sizeof(err), stub_forward) == 0);
    CHECK(stub_calls == 5);
    CHECK(stub_position[3] == 0 && stub_position[4] == 1);
    CHECK(session->checkpoint.v[0] == 21 && session->checkpoint.v[1] == 22);

    ds4_session_invalidate(session);
    stub_reset();
    session->cancel = cancel_after_first_forward;
    CHECK(ds4_session_sync_qwen35_with_forward(
              session, &extended, err, sizeof(err), stub_forward) ==
          DS4_SESSION_SYNC_INTERRUPTED);
    session->cancel = NULL;
    CHECK(!session->checkpoint_valid && session->checkpoint.len == 0);
    CHECK(session->qwen35_cpu_cache.n_tokens == 0);

    ds4_session_invalidate(session);
    stub_reset();
    stub_fail_on_call = 2;
    CHECK(ds4_session_sync_qwen35_with_forward(
              session, &extended, err, sizeof(err), stub_forward) != 0);
    CHECK(stub_calls == 2);
    CHECK(session->checkpoint.len == 0 && !session->checkpoint_valid);
    CHECK(session->qwen35_cpu_cache.n_tokens == 0);
    CHECK(session->qwen35_cpu_cache.layer[0].conv[0] == 0.0f);
    CHECK(session->qwen35_cpu_cache.layer[0].recurrent[0] == 0.0f);
}

static void test_fail_closed_surfaces(ds4_session *session) {
    char err[160] = {0};
    stub_reset();
    ds4_session_invalidate(session);
    CHECK(ds4_session_eval_qwen35_with_forward(
              session, 42, err, sizeof(err), stub_forward) == 0);
    CHECK(ds4_session_payload_bytes(session) == 0);
    CHECK(ds4_session_imatrix_enable(session) == 2);

    FILE *fp = tmpfile();
    CHECK(fp != NULL);
    if (fp) {
        CHECK(ds4_session_save_payload(session, fp, err, sizeof(err)) != 0);
        rewind(fp);
        CHECK(ds4_session_load_payload(session, fp, 0, err, sizeof(err)) != 0);
        fclose(fp);
    }
    ds4_session_payload_file staged = {0};
    memset(err, 0, sizeof(err));
    CHECK(ds4_session_stage_payload(session, &staged, err, sizeof(err)) != 0);
    CHECK(strcmp(err, "Qwen session payloads are not supported yet") == 0);
    CHECK(staged.path == NULL && staged.bytes == 0);

    ds4_session_snapshot snapshot = {0};
    CHECK(ds4_session_save_snapshot(session, &snapshot, err, sizeof(err)) != 0);
    ds4_session_snapshot_free(&snapshot);
    ds4_tokens text_tokens = {0};
    ds4_tokenize_text(session->engine, "unsafe fallback", &text_tokens);
    ds4_tokenize_rendered_chat(session->engine, "unsafe fallback", &text_tokens);
    ds4_chat_begin(session->engine, &text_tokens);
    ds4_chat_append_message(session->engine, &text_tokens,
                            "user", "unsafe fallback");
    ds4_chat_append_assistant_prefix(session->engine, &text_tokens,
                                     DS4_THINK_NONE);
    CHECK(text_tokens.len == 0);
    CHECK(ds4_token_eos(session->engine) == -1);
    CHECK(ds4_token_user(session->engine) == -1);
    CHECK(ds4_token_assistant(session->engine) == -1);
    size_t text_len = 99;
    char *text = ds4_token_text(session->engine, 0, &text_len);
    CHECK(text != NULL && text_len == 0 && text[0] == '\0');
    free(text);
    CHECK(ds4_engine_generate_argmax(
              session->engine, &text_tokens, 1, session->ctx_size,
              NULL, NULL, NULL, NULL, NULL) != 0);
    ds4_tokens_free(&text_tokens);
}

static void generation_test_set_logits(ds4_session *session, int best) {
    const uint32_t n_vocab = QWEN35_N_VOCAB;
    float *logits = malloc((size_t)n_vocab * sizeof(*logits));
    CHECK(logits != NULL);
    if (!logits) return;
    for (uint32_t i = 0; i < n_vocab; i++) logits[i] = -INFINITY;
    logits[best] = 3.0f;
    CHECK(ds4_session_set_logits(session, logits, (int)n_vocab) == 0);
    free(logits);
}

static void generation_test_seed_checkpoint(ds4_session *session, int token) {
    char err[160] = {0};
    ds4_session_invalidate(session);
    stub_reset();
    CHECK(ds4_session_eval_qwen35_with_forward(
              session, token, err, sizeof(err), stub_forward) == 0);
    CHECK(session->checkpoint_valid && session->checkpoint.len == 1);
    CHECK(session->qwen35_cpu_cache.n_tokens == 1u);
    generation_test_set_logits(session, 42);
    stub_reset();
}

static ds4_generation_block_request generation_test_request(
        float temperature, uint64_t rng, int room) {
    return (ds4_generation_block_request){
        .temperature = temperature,
        .top_k = 0,
        .top_p = 0.5f,
        .min_p = 0.0f,
        .rng = {.state = rng, .position = 17u},
        .max_output_tokens = room,
    };
}

static void test_generation_block_transaction(ds4_session *session) {
    const uint64_t seed = UINT64_C(0x0123456789abcdef);
    char err[160] = {0};
    ds4_generation_block block = {0};
    ds4_generation_block_request greedy =
        generation_test_request(0.0f, seed, 7);

    /* Begin samples only.  RETAIN C=1 keeps the adopted token pending and the
     * next begin evaluates it exactly once before sampling again. */
    generation_test_seed_checkpoint(session, 11);
    CHECK(ds4_session_generation_block_begin(
              session, &greedy, &block, err, sizeof(err)) == 0);
    CHECK(block.cookie != 0 && block.count == 1u && block.tokens[0] == 42);
    CHECK(session->generation_active && !session->pending_valid);
    CHECK(ds4_session_pos(session) == 1 && stub_calls == 0);
    ds4_generation_rng rng = greedy.rng;
    ds4_generation_block_commit commit = {
        .cookie = block.cookie,
        .adopted_count = 1u,
        .observed_count = 1u,
        .mode = DS4_GENERATION_COMMIT_RETAIN,
    };
    CHECK(ds4_session_generation_block_commit(
              session, &commit, &rng, err, sizeof(err)) == 0);
    CHECK(rng.state == seed && rng.position == 18u &&
          !session->generation_active && session->pending_valid);
    CHECK(session->pending_token == 42 && ds4_session_pos(session) == 1);
    CHECK(ds4_session_argmax(session) == -1);

    ds4_session state = *session;
    ds4_generation_block stale_block = {
        .cookie = UINT64_C(0x12345678), .count = 7u,
        .tokens = {7, 6, 5, 4, 3, 2, 1},
    };
    const ds4_generation_block stale_block_before = stale_block;
    CHECK(ds4_session_generation_block_begin_qwen35_with_forward(
              session, &greedy, &stale_block,
              err, sizeof(err), stub_forward) != 0);
    CHECK(memcmp(session, &state, sizeof(state)) == 0 && stub_calls == 0);
    CHECK(memcmp(&stale_block, &stale_block_before, sizeof(stale_block)) == 0);

    ds4_generation_block_request continuation = greedy;
    continuation.rng = rng;
    ds4_generation_block second = {0};
    CHECK(ds4_session_generation_block_begin_qwen35_with_forward(
              session, &continuation, &second,
              err, sizeof(err), stub_forward) == 0);
    CHECK(stub_calls == 1 && stub_position[0] == 1 && stub_token[0] == 42);
    CHECK(ds4_session_pos(session) == 2 && !session->pending_valid);
    CHECK(second.count == 1u && second.tokens[0] == 42);
    CHECK(second.cookie > block.cookie);
    rng = continuation.rng;
    commit = (ds4_generation_block_commit){
        .cookie = second.cookie,
        .adopted_count = 0u,
        .observed_count = 0u,
        .mode = DS4_GENERATION_COMMIT_RETAIN,
    };
    CHECK(ds4_session_generation_block_commit(
              session, &commit, &rng, err, sizeof(err)) == 0);
    CHECK(rng.state == seed && rng.position == 18u &&
          !session->pending_valid && ds4_session_pos(session) == 2);

    /* A sampled terminal can be observed without entering KV. */
    generation_test_set_logits(session, 42);
    ds4_generation_block_request sampled =
        generation_test_request(1.0f, 0, 1);
    stub_reset();
    CHECK(ds4_session_generation_block_begin(
              session, &sampled, &block, err, sizeof(err)) == 0);
    CHECK(block.count == 1u && block.tokens[0] == 42 && stub_calls == 0);
    const int pos_before_terminal = ds4_session_pos(session);
    rng = sampled.rng;
    commit = (ds4_generation_block_commit){
        .cookie = block.cookie,
        .adopted_count = 0u,
        .observed_count = 1u,
        .mode = DS4_GENERATION_COMMIT_RETAIN,
    };
    CHECK(ds4_session_generation_block_commit(
              session, &commit, &rng, err, sizeof(err)) == 0);
    CHECK(rng.state == UINT64_C(0x03f721dffe39b342) &&
          rng.position == 18u);
    CHECK(ds4_session_pos(session) == pos_before_terminal && stub_calls == 0);
    CHECK(!session->pending_valid && !session->generation_active);

    /* C=O=0 is a true abort, including RNG. */
    sampled.rng.state = seed;
    CHECK(ds4_session_generation_block_begin(
              session, &sampled, &block, err, sizeof(err)) == 0);
    rng = sampled.rng;
    commit = (ds4_generation_block_commit){
        .cookie = block.cookie,
        .adopted_count = 0u,
        .observed_count = 0u,
        .mode = DS4_GENERATION_COMMIT_RETAIN,
    };
    CHECK(ds4_session_generation_block_commit(
              session, &commit, &rng, err, sizeof(err)) == 0);
    CHECK(rng.state == seed && rng.position == 17u && !session->pending_valid);

    /* Rejected begin/commit transitions are byte-identical in the session and
     * cannot acquire the caller's RNG or output block. */
    CHECK(ds4_session_generation_block_begin(
              session, &sampled, &block, err, sizeof(err)) == 0);
    state = *session;
    ds4_generation_block untouched = {
        .cookie = UINT64_C(0xa5a5a5a5a5a5a5a5),
        .count = 6u,
        .tokens = {9, 8, 7, 6, 5, 4, 3},
    };
    const ds4_generation_block untouched_before = untouched;
    CHECK(ds4_session_generation_block_begin(
              session, &sampled, &untouched, err, sizeof(err)) != 0);
    CHECK(memcmp(session, &state, sizeof(state)) == 0);
    CHECK(memcmp(&untouched, &untouched_before, sizeof(untouched)) == 0);

    const ds4_generation_block_commit invalid_commits[] = {
        {block.cookie + 1u, 0u, 0u, DS4_GENERATION_COMMIT_RETAIN},
        {block.cookie, 1u, 0u, DS4_GENERATION_COMMIT_RETAIN},
        {block.cookie, 0u, 2u, DS4_GENERATION_COMMIT_RETAIN},
        {block.cookie, 0u, 0u, (ds4_generation_commit_mode)99},
    };
    for (size_t i = 0; i < sizeof(invalid_commits) / sizeof(invalid_commits[0]); i++) {
        rng = sampled.rng;
        CHECK(ds4_session_generation_block_commit(
                  session, &invalid_commits[i], &rng, err, sizeof(err)) != 0);
        CHECK(memcmp(session, &state, sizeof(state)) == 0 &&
              rng.state == seed && rng.position == 17u);
    }
    commit = (ds4_generation_block_commit){
        .cookie = block.cookie,
        .adopted_count = 0u,
        .observed_count = 0u,
        .mode = DS4_GENERATION_COMMIT_RETAIN,
    };
    rng = sampled.rng;
    rng.state++;
    CHECK(ds4_session_generation_block_commit(
              session, &commit, &rng, err, sizeof(err)) != 0);
    CHECK(memcmp(session, &state, sizeof(state)) == 0 &&
          rng.state == seed + 1u && rng.position == 17u);
    rng = sampled.rng;
    rng.position++;
    CHECK(ds4_session_generation_block_commit(
              session, &commit, &rng, err, sizeof(err)) != 0);
    CHECK(memcmp(session, &state, sizeof(state)) == 0 &&
          rng.state == seed && rng.position == 18u);
    rng = sampled.rng;
    CHECK(ds4_session_generation_block_commit(
              session, &commit, &rng, err, sizeof(err)) == 0);
    state = *session;
    CHECK(ds4_session_generation_block_commit(
              session, &commit, &rng, err, sizeof(err)) != 0);
    CHECK(memcmp(session, &state, sizeof(state)) == 0 &&
          rng.state == seed && rng.position == 17u);

    const uint64_t saved_last_cookie = session->generation_last_cookie;
    session->generation_last_cookie = UINT64_MAX;
    state = *session;
    untouched = untouched_before;
    CHECK(ds4_session_generation_block_begin(
              session, &sampled, &untouched, err, sizeof(err)) != 0);
    CHECK(memcmp(session, &state, sizeof(state)) == 0);
    CHECK(memcmp(&untouched, &untouched_before, sizeof(untouched)) == 0);
    session->generation_last_cookie = saved_last_cookie;

    /* Every state/logit mutation surface is closed while a transaction is
     * active.  Error reporting and caller-owned output buffers may change, but
     * the session, backend frontier and RNG do not. */
    CHECK(ds4_session_generation_block_begin(
              session, &greedy, &block, err, sizeof(err)) == 0);
    state = *session;
    const int active_stub_calls = stub_calls;
    uint64_t guarded_rng = seed;
    CHECK(ds4_session_sample(session, 1.0f, 0, 1.0f, 0.0f, &guarded_rng) == -1);
    CHECK(guarded_rng == seed);
    CHECK(ds4_session_argmax(session) == -1);
    CHECK(ds4_session_argmax_excluding(session, 42) == -1);
    ds4_token_score score = {0};
    CHECK(ds4_session_top_logprobs(session, &score, 1) == 0);
    CHECK(ds4_session_token_logprob(session, 42, &score) == 0);
    float one_logit = 0.0f;
    CHECK(ds4_session_copy_logits(session, &one_logit, 1) == 0);
    CHECK(ds4_session_set_logits(session, &one_logit, 1) != 0);
    CHECK(ds4_session_eval(session, 43, err, sizeof(err)) != 0);
    const int active_pos = ds4_session_pos(session);
    ds4_session_rewind(session, 0);
    CHECK(ds4_session_pos(session) == active_pos);
    const int prompt_values[] = {11, 42};
    ds4_tokens prompt = {(int *)prompt_values, 2, 2};
    CHECK(ds4_session_sync_qwen35_with_forward(
              session, &prompt, err, sizeof(err), stub_forward) != 0);
    CHECK(ds4_session_rewrite_from_common(
              session, &prompt, session->checkpoint.len,
              err, sizeof(err)) == DS4_SESSION_REWRITE_ERROR);
    CHECK(ds4_session_payload_bytes(session) == 0);
    FILE *fp = tmpfile();
    CHECK(fp != NULL);
    if (fp) {
        CHECK(ds4_session_save_payload(session, fp, err, sizeof(err)) != 0);
        rewind(fp);
        CHECK(ds4_session_load_payload(session, fp, 0, err, sizeof(err)) != 0);
        fclose(fp);
    }
    ds4_session_payload_file staged = {0};
    CHECK(ds4_session_stage_payload(session, &staged, err, sizeof(err)) != 0);
    CHECK(staged.path == NULL && staged.bytes == 0);
    ds4_session_snapshot snapshot = {0};
    CHECK(ds4_session_save_snapshot(session, &snapshot, err, sizeof(err)) != 0);
    CHECK(ds4_session_load_snapshot(session, &snapshot, err, sizeof(err)) != 0);
    CHECK(ds4_session_set_power(session, 100) != 0);
    int accepted_token = -1;
    CHECK(ds4_session_eval_speculative_argmax(
              session, 42, 1, -1, &accepted_token, 1,
              err, sizeof(err)) == -1);
    CHECK(memcmp(session, &state, sizeof(state)) == 0);
    CHECK(stub_calls == active_stub_calls);
    commit = (ds4_generation_block_commit){
        .cookie = block.cookie,
        .adopted_count = 0u,
        .observed_count = 0u,
        .mode = DS4_GENERATION_COMMIT_RETAIN,
    };
    rng = greedy.rng;
    CHECK(ds4_session_generation_block_commit(
              session, &commit, &rng, err, sizeof(err)) == 0);

    /* Pending also makes logits and persistence stale; rewind is the explicit
     * non-destructive way to discard it. */
    CHECK(ds4_session_generation_block_begin(
              session, &greedy, &block, err, sizeof(err)) == 0);
    commit = (ds4_generation_block_commit){
        .cookie = block.cookie, .adopted_count = 1u, .observed_count = 1u,
        .mode = DS4_GENERATION_COMMIT_RETAIN,
    };
    rng = greedy.rng;
    CHECK(ds4_session_generation_block_commit(
              session, &commit, &rng, err, sizeof(err)) == 0);
    CHECK(session->pending_valid && ds4_session_payload_bytes(session) == 0);
    CHECK(ds4_session_set_power(session, 100) == 0);
    CHECK(session->pending_valid);
    staged = (ds4_session_payload_file){0};
    CHECK(ds4_session_stage_payload(session, &staged, err, sizeof(err)) != 0);
    CHECK(strcmp(err, "generation block has an unevaluated pending token") == 0);
    int rewrite_values[] = {11, 42};
    ds4_tokens rewrite_prompt = {rewrite_values, 2, 2};
    CHECK(ds4_session_tokens(session)->len == ds4_session_pos(session));
    CHECK(ds4_session_common_prefix(session, &rewrite_prompt) ==
          ds4_session_pos(session));
    state = *session;
    CHECK(ds4_session_rewrite_from_common(
              session, &rewrite_prompt, session->checkpoint.len,
              err, sizeof(err)) == DS4_SESSION_REWRITE_ERROR);
    CHECK(memcmp(session, &state, sizeof(state)) == 0);
    ds4_session_rewind(session, ds4_session_pos(session));
    CHECK(!session->pending_valid);

    /* Eval cannot guess whether a newly supplied token follows or replaces the
     * adopted pending token.  It fails byte-identically; begin(0) is the
     * explicit one-token flush. */
    generation_test_set_logits(session, 42);
    CHECK(ds4_session_generation_block_begin(
              session, &greedy, &block, err, sizeof(err)) == 0);
    commit = (ds4_generation_block_commit){
        .cookie = block.cookie, .adopted_count = 1u, .observed_count = 1u,
        .mode = DS4_GENERATION_COMMIT_RETAIN,
    };
    rng = greedy.rng;
    CHECK(ds4_session_generation_block_commit(
              session, &commit, &rng, err, sizeof(err)) == 0);
    stub_reset();
    state = *session;
    CHECK(ds4_session_eval(session, 43, err, sizeof(err)) != 0);
    CHECK(stub_calls == 0 && memcmp(session, &state, sizeof(state)) == 0);
    ds4_generation_block_request flush = greedy;
    flush.rng = rng;
    flush.max_output_tokens = 0;
    CHECK(ds4_session_generation_block_begin_qwen35_with_forward(
              session, &flush, &block, err, sizeof(err), stub_forward) == 0);
    CHECK(block.count == 0u && stub_calls == 1 && stub_token[0] == 42);
    CHECK(!session->pending_valid);

    /* Sync preserves the pending token only for an exact extension. */
    generation_test_seed_checkpoint(session, 11);
    CHECK(ds4_session_generation_block_begin(
              session, &greedy, &block, err, sizeof(err)) == 0);
    commit = (ds4_generation_block_commit){
        .cookie = block.cookie, .adopted_count = 1u, .observed_count = 1u,
        .mode = DS4_GENERATION_COMMIT_RETAIN,
    };
    rng = greedy.rng;
    CHECK(ds4_session_generation_block_commit(
              session, &commit, &rng, err, sizeof(err)) == 0);
    int matching_values[] = {11, 42, 55};
    ds4_tokens matching = {matching_values, 3, 3};
    stub_reset();
    session->cancel = cancel_always;
    state = *session;
    CHECK(ds4_session_sync(
              session, &matching, err, sizeof(err)) ==
          DS4_SESSION_SYNC_INTERRUPTED);
    CHECK(memcmp(session, &state, sizeof(state)) == 0 && stub_calls == 0);
    session->cancel = NULL;
    CHECK(ds4_session_sync_qwen35_with_forward(
              session, &matching, err, sizeof(err), stub_forward) == 0);
    CHECK(stub_calls == 2 && stub_token[0] == 42 && stub_token[1] == 55);
    CHECK(session->checkpoint.len == 3 && !session->pending_valid);

    generation_test_seed_checkpoint(session, 11);
    CHECK(ds4_session_generation_block_begin(
              session, &greedy, &block, err, sizeof(err)) == 0);
    commit.cookie = block.cookie;
    rng = greedy.rng;
    CHECK(ds4_session_generation_block_commit(
              session, &commit, &rng, err, sizeof(err)) == 0);
    int divergent_values[] = {11, 77};
    ds4_tokens divergent = {divergent_values, 2, 2};
    stub_reset();
    CHECK(ds4_session_sync_qwen35_with_forward(
              session, &divergent, err, sizeof(err), stub_forward) == 0);
    CHECK(stub_calls == 1 && stub_token[0] == 77);
    CHECK(session->checkpoint.len == 2 && !session->pending_valid);

    /* Context and caller output room produce an empty successful block without
     * opening or consuming a cookie. */
    generation_test_seed_checkpoint(session, 11);
    const uint64_t cookie_before_empty = session->generation_last_cookie;
    ds4_generation_block_request no_output = greedy;
    no_output.max_output_tokens = 0;
    block = (ds4_generation_block){.cookie = 99, .count = 7u};
    CHECK(ds4_session_generation_block_begin(
              session, &no_output, &block, err, sizeof(err)) == 0);
    CHECK(block.cookie == 0 && block.count == 0u && !session->generation_active);
    CHECK(session->generation_last_cookie == cookie_before_empty);
    session->generation_last_cookie = UINT64_MAX;
    block = (ds4_generation_block){.cookie = 99, .count = 7u};
    CHECK(ds4_session_generation_block_begin(
              session, &no_output, &block, err, sizeof(err)) == 0);
    CHECK(block.cookie == 0 && block.count == 0u &&
          session->generation_last_cookie == UINT64_MAX);
    session->generation_last_cookie = cookie_before_empty;
    ds4_generation_block_request exhausted_position = greedy;
    exhausted_position.rng.position = UINT64_MAX;
    state = *session;
    untouched = untouched_before;
    CHECK(ds4_session_generation_block_begin(
              session, &exhausted_position, &untouched,
              err, sizeof(err)) != 0);
    CHECK(memcmp(session, &state, sizeof(state)) == 0);
    CHECK(memcmp(&untouched, &untouched_before, sizeof(untouched)) == 0);
    ds4_generation_block_request malformed = greedy;
    malformed.temperature = NAN;
    state = *session;
    CHECK(ds4_session_generation_block_begin(
              session, &malformed, &untouched, err, sizeof(err)) != 0);
    CHECK(memcmp(session, &state, sizeof(state)) == 0);

    /* A zero-output begin is also the final pending flush.  It evaluates once,
     * does not consume a cookie, and accepts even an exhausted next position
     * because it opens no new block. */
    ds4_generation_block_request near_exhausted = greedy;
    near_exhausted.rng.position = UINT64_MAX - 1u;
    CHECK(ds4_session_generation_block_begin(
              session, &near_exhausted, &block, err, sizeof(err)) == 0);
    commit = (ds4_generation_block_commit){
        .cookie = block.cookie, .adopted_count = 1u, .observed_count = 1u,
        .mode = DS4_GENERATION_COMMIT_RETAIN,
    };
    rng = near_exhausted.rng;
    CHECK(ds4_session_generation_block_commit(
              session, &commit, &rng, err, sizeof(err)) == 0);
    CHECK(rng.position == UINT64_MAX);
    const uint64_t cookie_before_flush = session->generation_last_cookie;
    no_output.rng = rng;
    stub_reset();
    CHECK(ds4_session_generation_block_begin_qwen35_with_forward(
              session, &no_output, &block, err, sizeof(err), stub_forward) == 0);
    CHECK(block.cookie == 0 && block.count == 0u && !session->generation_active);
    CHECK(session->generation_last_cookie == cookie_before_flush);
    CHECK(stub_calls == 1 && stub_token[0] == 42 && !session->pending_valid);
    while (session->checkpoint.len < session->ctx_size - 1) {
        CHECK(ds4_session_eval_qwen35_with_forward(
                  session, 80 + session->checkpoint.len,
                  err, sizeof(err), stub_forward) == 0);
    }
    CHECK(session->checkpoint.len == session->ctx_size - 1);
    generation_test_set_logits(session, 42);
    CHECK(ds4_session_generation_block_begin(
              session, &greedy, &block, err, sizeof(err)) == 0);
    commit = (ds4_generation_block_commit){
        .cookie = block.cookie, .adopted_count = 1u, .observed_count = 1u,
        .mode = DS4_GENERATION_COMMIT_RETAIN,
    };
    rng = greedy.rng;
    CHECK(ds4_session_generation_block_commit(
              session, &commit, &rng, err, sizeof(err)) == 0);
    CHECK(session->pending_valid && session->checkpoint.len == session->ctx_size - 1);
    flush.rng = rng;
    stub_reset();
    CHECK(ds4_session_generation_block_begin_qwen35_with_forward(
              session, &flush, &block, err, sizeof(err), stub_forward) == 0);
    CHECK(block.cookie == 0 && block.count == 0u &&
          session->checkpoint.len == session->ctx_size &&
          stub_calls == 1 && stub_token[0] == 42);
    block = (ds4_generation_block){.cookie = 99, .count = 7u};
    CHECK(ds4_session_generation_block_begin(
              session, &greedy, &block, err, sizeof(err)) == 0);
    CHECK(block.cookie == 0 && block.count == 0u && !session->generation_active);
    CHECK(ds4_session_payload_bytes(session) == 0);
    staged = (ds4_session_payload_file){0};
    CHECK(ds4_session_stage_payload(session, &staged, err, sizeof(err)) != 0);
    CHECK(strcmp(err, "full-context session cannot be staged") == 0);
    fp = tmpfile();
    CHECK(fp != NULL);
    if (fp) {
        CHECK(ds4_session_save_payload(session, fp, err, sizeof(err)) != 0);
        CHECK(strcmp(err, "full-context session cannot be saved") == 0);
        fclose(fp);
    }

    /* A failed pending materialization invalidates the whole frontier instead
     * of losing ownership of the adopted token or preserving partial KV. */
    generation_test_seed_checkpoint(session, 11);
    generation_test_set_logits(session, 42);
    CHECK(ds4_session_generation_block_begin(
              session, &greedy, &block, err, sizeof(err)) == 0);
    commit = (ds4_generation_block_commit){
        .cookie = block.cookie, .adopted_count = 1u, .observed_count = 1u,
        .mode = DS4_GENERATION_COMMIT_RETAIN,
    };
    rng = greedy.rng;
    CHECK(ds4_session_generation_block_commit(
              session, &commit, &rng, err, sizeof(err)) == 0);
    stub_reset();
    stub_fail_on_call = 1;
    flush.rng = rng;
    CHECK(ds4_session_generation_block_begin_qwen35_with_forward(
              session, &flush, &block, err, sizeof(err), stub_forward) != 0);
    CHECK(!session->checkpoint_valid && session->checkpoint.len == 0);
    CHECK(session->qwen35_cpu_cache.n_tokens == 0u);
    CHECK(!session->generation_active && !session->pending_valid);

    /* Sampled fallbacks and greedy sampling publish no draw. */
    generation_test_seed_checkpoint(session, 11);
    for (uint32_t i = 0; i < QWEN35_N_VOCAB; i++) session->logits[i] = NAN;
    sampled.rng.state = seed;
    CHECK(ds4_session_generation_block_begin(
              session, &sampled, &block, err, sizeof(err)) == 0);
    CHECK(block.tokens[0] == 0);
    commit = (ds4_generation_block_commit){
        .cookie = block.cookie, .adopted_count = 0u, .observed_count = 1u,
        .mode = DS4_GENERATION_COMMIT_RETAIN,
    };
    rng = sampled.rng;
    CHECK(ds4_session_generation_block_commit(
              session, &commit, &rng, err, sizeof(err)) == 0);
    CHECK(rng.state == seed && rng.position == 18u);

    /* INVALIDATE is the stop-string/stream-failure form: observed RNG survives,
     * but target, checkpoint, pending and active state do not. */
    generation_test_seed_checkpoint(session, 11);
    sampled.rng.state = 0;
    CHECK(ds4_session_generation_block_begin(
              session, &sampled, &block, err, sizeof(err)) == 0);
    commit = (ds4_generation_block_commit){
        .cookie = block.cookie, .adopted_count = 0u, .observed_count = 1u,
        .mode = DS4_GENERATION_COMMIT_INVALIDATE,
    };
    rng = sampled.rng;
    CHECK(ds4_session_generation_block_commit(
              session, &commit, &rng, err, sizeof(err)) == 0);
    CHECK(rng.state == UINT64_C(0x03f721dffe39b342) &&
          rng.position == 18u);
    CHECK(!session->checkpoint_valid && session->checkpoint.len == 0);
    CHECK(session->qwen35_cpu_cache.n_tokens == 0u);
    CHECK(!session->generation_active && !session->pending_valid);

    /* Invalidate is also the explicit escape hatch for an abandoned active
     * transaction. */
    generation_test_seed_checkpoint(session, 11);
    CHECK(ds4_session_generation_block_begin(
              session, &greedy, &block, err, sizeof(err)) == 0);
    ds4_session_invalidate(session);
    CHECK(!session->generation_active && !session->pending_valid);
    CHECK(!session->checkpoint_valid && session->checkpoint.len == 0);
}

int main(void) {
    CHECK(ds4_dspark_runtime_contract_self_check());
    CHECK(ds4_test_dspark_memory_accounting());
    test_sampler_outcomes();
#ifndef DS4_NO_GPU
    CHECK(ds4_internal_dspark_raw_finalizer_test());
#endif
    test_session_creation_boundary();
    test_model_aware_context_memory();
    test_qwen_metal_session_context_budget();
    test_qwen_residency_request_normalization();
    test_qwen35_ssd_static_contract();
    test_gpu_dense_layout_contract();

    ds4_engine engine;
    ds4_session *session = NULL;
    fake_qwen_engine(&engine, true);
    CHECK(ds4_session_create(&session, &engine, 5) == 0);
    if (session) {
        CHECK(ds4_session_ctx(session) == 5);
        CHECK(ds4_session_pos(session) == 0);
        CHECK(ds4_session_prefill_cap(session) == 1);
        CHECK(session->qwen35_cpu_cache.ctx_capacity == 5);
        CHECK(session->qwen35_cpu_scratch.ctx_capacity == 5);
        CHECK(session->cpu_cache.layer[0].raw_kv == NULL);
        test_dynamic_logits(session);
        test_eval_transaction(session);
        test_sync_transaction(session);
        test_generation_block_transaction(session);
        test_fail_closed_surfaces(session);
        ds4_session_free(session);
    }

    ds4_threads_shutdown();
    if (failures) {
        fprintf(stderr, "Qwen session tests: %d failure(s)\n", failures);
        return 1;
    }
    puts("Qwen session tests: OK");
    return 0;
}
