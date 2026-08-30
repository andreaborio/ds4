/*
 * Qwen4Exp structural admission partition.
 *
 * This file is included by ds4.c after its GGUF parser helpers.  It admits no
 * execution profile: production builds deliberately have no physical profile,
 * while DS4_TEST_HOOKS exposes one sparse BF16 structural fixture profile.
 */

enum {
    Q4E_PHYSICAL_TENSORS = 1069,
    Q4E_DENSE_TENSORS = 1067,
    Q4E_SOURCE_TENSORS = 1658,
};

#define Q4E_TEST_PHYSICAL_PROFILE_ID "qwen4exp-phase3-fixture-bf16-v1"
#define Q4E_SOURCE_REVISION \
    "de4b8e4d43b917e7706784d8bb445c9af86a3540"
#define Q4E_TRANSFORMERS_COMMIT \
    "42ca97014c85d71a88ad60d55f08cb9fb4d26e2c"
#define Q4E_INVENTORY_SHA256 \
    "a639efc7a5147b04200e870d7e320335527f4361a8327b137feca2683b1dc434"
#define Q4E_TOKENIZER_SHA256 \
    "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
#define Q4E_TOKENIZER_CONFIG_SHA256 \
    "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27"
#define Q4E_TEMPLATE_SHA256 \
    "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
#define Q4E_EXPERT_CODEC_DESCRIPTOR_SHA256 \
    "74ff9e25a49c2ca8f8620f5360308876b163257f12dff146cc56749222583f4b"
#define Q4E_PLE_CODEC_DESCRIPTOR_SHA256 \
    "810f4424febbd36b6659465e555cd781b1cc8ef3a5b3e25126df7cabc8ca8a31"
#define Q4E_PLE_TEST_CODEC_ID "q4exp-fixture-bf16-v1"
#define Q4E_SOURCE_BYTES UINT64_C(359999963128)

typedef enum {
    DS4_QWEN4EXP_STAGE_NONE = 0,
    DS4_QWEN4EXP_STAGE_IDENTITY,
    DS4_QWEN4EXP_STAGE_PHYSICAL_PROFILE,
    DS4_QWEN4EXP_STAGE_METADATA,
    DS4_QWEN4EXP_STAGE_TOKENIZER,
    DS4_QWEN4EXP_STAGE_INVENTORY,
    DS4_QWEN4EXP_STAGE_EXPERT_STORE,
    DS4_QWEN4EXP_STAGE_PLE_STORE,
    DS4_QWEN4EXP_STAGE_POLICY,
    DS4_QWEN4EXP_STAGE_OWNERSHIP,
    DS4_QWEN4EXP_STAGE_ACCEPTED,
} ds4_qwen4exp_stage;

typedef struct {
    ds4_qwen4exp_stage stage;
    char field[128];
    char expected[160];
    char observed[160];
} ds4_qwen4exp_rejection;

typedef struct {
    const char *id;
    uint32_t dense_ggml_type;
    uint32_t expert_storage;
    uint32_t expert_group_size;
    ds4_ple_store_codec ple_codec;
    uint32_t ple_rows_per_page;
    uint32_t ple_page_alignment;
} ds4_qwen4exp_physical_profile;

typedef struct {
    uint64_t off;
    uint64_t end;
} q4e_span;

typedef struct {
    bool admitted;
    bool runtime_supported;
    bool payload_verified;
    bool text_only;
    bool mtp_present;
    bool vision_present;
    bool tokenizer_content_verified;
    bool expert_manifest_verified;
    bool ple_manifest_verified;
    ds4_qwen4exp_stage stage;
    const char *physical_profile_id;
    uint64_t physical_tensors;
    uint64_t dense_tensors;
    uint64_t expert_stores;
    uint64_t ple_stores;
    uint64_t file_bytes;
    uint64_t header_bytes;
    uint64_t dense_bytes;
    uint64_t expert_bytes;
    uint64_t ple_bytes;
    uint64_t padding_bytes;
    uint64_t owned_bytes;
    uint64_t dense_page_bytes;
    uint64_t source_tensor_count;
    uint64_t source_bytes;
} ds4_qwen4exp_admission_report;

typedef struct {
    ds4_qwen4exp_admission_report report;
    ds4_expert_store *expert_store;
    ds4_ple_store *ple_store;
    q4e_span *dense_page_spans;
    size_t dense_page_span_count;
} ds4_qwen4exp_loader;

typedef enum {
    Q4E_OWNER_DENSE = 1,
    Q4E_OWNER_EXPERT = 2,
    Q4E_OWNER_PLE = 3,
} q4e_owner;

typedef struct {
    uint64_t off;
    uint64_t end;
    q4e_owner owner;
    ds4_str name;
} q4e_owned_span;

static const char *qwen4exp_stage_name(ds4_qwen4exp_stage stage) {
    switch (stage) {
    case DS4_QWEN4EXP_STAGE_NONE: return "none";
    case DS4_QWEN4EXP_STAGE_IDENTITY: return "identity";
    case DS4_QWEN4EXP_STAGE_PHYSICAL_PROFILE: return "physical_profile";
    case DS4_QWEN4EXP_STAGE_METADATA: return "metadata";
    case DS4_QWEN4EXP_STAGE_TOKENIZER: return "tokenizer";
    case DS4_QWEN4EXP_STAGE_INVENTORY: return "inventory";
    case DS4_QWEN4EXP_STAGE_EXPERT_STORE: return "expert_store";
    case DS4_QWEN4EXP_STAGE_PLE_STORE: return "ple_store";
    case DS4_QWEN4EXP_STAGE_POLICY: return "policy";
    case DS4_QWEN4EXP_STAGE_OWNERSHIP: return "ownership";
    case DS4_QWEN4EXP_STAGE_ACCEPTED: return "accepted";
    }
    return "invalid";
}

static bool q4e_reject(ds4_qwen4exp_rejection *rejection,
                       ds4_qwen4exp_stage stage,
                       const char *field,
                       const char *expected,
                       const char *observed) {
    if (rejection) {
        memset(rejection, 0, sizeof(*rejection));
        rejection->stage = stage;
        snprintf(rejection->field, sizeof(rejection->field), "%s",
                 field ? field : "unknown");
        snprintf(rejection->expected, sizeof(rejection->expected), "%s",
                 expected ? expected : "");
        snprintf(rejection->observed, sizeof(rejection->observed), "%s",
                 observed ? observed : "");
    }
    return false;
}

static bool q4e_reject_u64(ds4_qwen4exp_rejection *rejection,
                           ds4_qwen4exp_stage stage,
                           const char *field,
                           uint64_t expected,
                           uint64_t observed) {
    char want[32];
    char got[32];
    snprintf(want, sizeof(want), "%" PRIu64, expected);
    snprintf(got, sizeof(got), "%" PRIu64, observed);
    return q4e_reject(rejection, stage, field, want, got);
}

static bool q4e_str_equal(ds4_str value, const char *expected) {
    const size_t n = expected ? strlen(expected) : 0u;
    return expected && value.len == n && memcmp(value.ptr, expected, n) == 0;
}

static bool q4e_ds4_str_equal(ds4_str a, ds4_str b) {
    return a.len == b.len && memcmp(a.ptr, b.ptr, (size_t)a.len) == 0;
}

static int q4e_ds4_str_compare(ds4_str a, ds4_str b) {
    const size_t common = (size_t)(a.len < b.len ? a.len : b.len);
    const int result = memcmp(a.ptr, b.ptr, common);
    if (result != 0) return result;
    return a.len < b.len ? -1 : a.len > b.len ? 1 : 0;
}

static bool q4e_starts_with(ds4_str value, const char *prefix) {
    const size_t n = strlen(prefix);
    return value.len >= n && memcmp(value.ptr, prefix, n) == 0;
}

static bool q4e_contains(ds4_str value, const char *needle) {
    const size_t n = strlen(needle);
    if (n == 0u || value.len < n) return false;
    for (uint64_t i = 0; i <= value.len - n; ++i) {
        if (memcmp(value.ptr + i, needle, n) == 0) return true;
    }
    return false;
}

static const char *q4e_metadata_type_name(uint32_t type) {
    switch (type) {
    case GGUF_VALUE_UINT8: return "UINT8";
    case GGUF_VALUE_INT8: return "INT8";
    case GGUF_VALUE_UINT16: return "UINT16";
    case GGUF_VALUE_INT16: return "INT16";
    case GGUF_VALUE_UINT32: return "UINT32";
    case GGUF_VALUE_INT32: return "INT32";
    case GGUF_VALUE_FLOAT32: return "FLOAT32";
    case GGUF_VALUE_BOOL: return "BOOL";
    case GGUF_VALUE_STRING: return "STRING";
    case GGUF_VALUE_ARRAY: return "ARRAY";
    case GGUF_VALUE_UINT64: return "UINT64";
    case GGUF_VALUE_INT64: return "INT64";
    case GGUF_VALUE_FLOAT64: return "FLOAT64";
    default: return "UNKNOWN";
    }
}

static ds4_kv *q4e_kv_exact(const ds4_model *model,
                            const char *key,
                            uint32_t type,
                            ds4_qwen4exp_stage stage,
                            ds4_qwen4exp_rejection *rejection) {
    ds4_kv *kv = model_find_kv(model, key);
    if (!kv) {
        q4e_reject(rejection, stage, key, q4e_metadata_type_name(type),
                   "missing");
        return NULL;
    }
    if (kv->type != type) {
        q4e_reject(rejection, stage, key, q4e_metadata_type_name(type),
                   q4e_metadata_type_name(kv->type));
        return NULL;
    }
    return kv;
}

static bool q4e_expect_string(const ds4_model *model,
                              const char *key,
                              const char *expected,
                              ds4_qwen4exp_stage stage,
                              ds4_qwen4exp_rejection *rejection) {
    ds4_kv *kv = q4e_kv_exact(model, key, GGUF_VALUE_STRING, stage, rejection);
    if (!kv) return false;
    ds4_cursor cursor = cursor_at(model, kv->value_pos);
    ds4_str value = {0};
    if (!cursor_string(&cursor, &value)) {
        return q4e_reject(rejection, stage, key, expected, "truncated");
    }
    if (q4e_str_equal(value, expected)) return true;
    char observed[128];
    const int n = value.len < 96u ? (int)value.len : 96;
    snprintf(observed, sizeof(observed), "%.*s", n, value.ptr);
    return q4e_reject(rejection, stage, key, expected, observed);
}

static bool q4e_expect_u32(const ds4_model *model,
                           const char *key,
                           uint32_t expected,
                           ds4_qwen4exp_stage stage,
                           ds4_qwen4exp_rejection *rejection) {
    ds4_kv *kv = q4e_kv_exact(model, key, GGUF_VALUE_UINT32, stage, rejection);
    uint32_t value = 0;
    if (!kv) return false;
    ds4_cursor cursor = cursor_at(model, kv->value_pos);
    if (!cursor_u32(&cursor, &value)) {
        return q4e_reject(rejection, stage, key, "UINT32", "truncated");
    }
    return value == expected ? true :
        q4e_reject_u64(rejection, stage, key, expected, value);
}

static bool q4e_expect_u64(const ds4_model *model,
                           const char *key,
                           uint64_t expected,
                           ds4_qwen4exp_stage stage,
                           ds4_qwen4exp_rejection *rejection) {
    ds4_kv *kv = q4e_kv_exact(model, key, GGUF_VALUE_UINT64, stage, rejection);
    uint64_t value = 0;
    if (!kv) return false;
    ds4_cursor cursor = cursor_at(model, kv->value_pos);
    if (!cursor_u64(&cursor, &value)) {
        return q4e_reject(rejection, stage, key, "UINT64", "truncated");
    }
    return value == expected ? true :
        q4e_reject_u64(rejection, stage, key, expected, value);
}

static bool q4e_expect_bool(const ds4_model *model,
                            const char *key,
                            bool expected,
                            ds4_qwen4exp_stage stage,
                            ds4_qwen4exp_rejection *rejection) {
    ds4_kv *kv = q4e_kv_exact(model, key, GGUF_VALUE_BOOL, stage, rejection);
    uint8_t value = 0;
    if (!kv) return false;
    ds4_cursor cursor = cursor_at(model, kv->value_pos);
    if (!cursor_read(&cursor, &value, sizeof(value))) {
        return q4e_reject(rejection, stage, key, "BOOL", "truncated");
    }
    if (value > 1u) {
        return q4e_reject(rejection, stage, key, expected ? "true" : "false",
                          "non-canonical BOOL");
    }
    if ((value != 0u) == expected) return true;
    return q4e_reject(rejection, stage, key, expected ? "true" : "false",
                      value ? "true" : "false");
}

static bool q4e_expect_f32(const ds4_model *model,
                           const char *key,
                           float expected,
                           ds4_qwen4exp_stage stage,
                           ds4_qwen4exp_rejection *rejection) {
    ds4_kv *kv = q4e_kv_exact(model, key, GGUF_VALUE_FLOAT32, stage, rejection);
    float value = 0.0f;
    if (!kv) return false;
    ds4_cursor cursor = cursor_at(model, kv->value_pos);
    if (!cursor_read(&cursor, &value, sizeof(value))) {
        return q4e_reject(rejection, stage, key, "FLOAT32", "truncated");
    }
    if (memcmp(&value, &expected, sizeof(value)) == 0) return true;
    char want[48];
    char got[48];
    snprintf(want, sizeof(want), "%.9g", (double)expected);
    snprintf(got, sizeof(got), "%.9g", (double)value);
    return q4e_reject(rejection, stage, key, want, got);
}

static bool q4e_expect_u32_array(const ds4_model *model,
                                 const char *key,
                                 const uint32_t *expected,
                                 uint64_t count,
                                 ds4_qwen4exp_stage stage,
                                 ds4_qwen4exp_rejection *rejection) {
    ds4_kv *kv = q4e_kv_exact(model, key, GGUF_VALUE_ARRAY, stage, rejection);
    ds4_array_ref array = {0};
    if (!kv) return false;
    if (!model_get_array(model, key, &array) ||
        array.type != GGUF_VALUE_UINT32 || array.len != count) {
        char want[64];
        char got[64];
        snprintf(want, sizeof(want), "ARRAY<UINT32>[%" PRIu64 "]", count);
        if (!model_get_array(model, key, &array)) {
            snprintf(got, sizeof(got), "malformed ARRAY");
        } else {
            snprintf(got, sizeof(got), "ARRAY<%s>[%" PRIu64 "]",
                     q4e_metadata_type_name(array.type), array.len);
        }
        return q4e_reject(rejection, stage, key, want, got);
    }
    ds4_cursor cursor = cursor_at(model, array.data_pos);
    for (uint64_t i = 0; i < count; ++i) {
        uint32_t value = 0;
        if (!cursor_u32(&cursor, &value)) {
            return q4e_reject(rejection, stage, key, "complete array",
                              "truncated");
        }
        if (value != expected[i]) {
            char field[160];
            snprintf(field, sizeof(field), "%s[%" PRIu64 "]", key, i);
            return q4e_reject_u64(rejection, stage, field,
                                  expected[i], value);
        }
    }
    return true;
}

static bool q4e_expect_u64_array(const ds4_model *model,
                                 const char *key,
                                 const uint64_t *expected,
                                 uint64_t count,
                                 ds4_qwen4exp_stage stage,
                                 ds4_qwen4exp_rejection *rejection) {
    ds4_kv *kv = q4e_kv_exact(model, key, GGUF_VALUE_ARRAY, stage, rejection);
    ds4_array_ref array = {0};
    if (!kv) return false;
    if (!model_get_array(model, key, &array) ||
        array.type != GGUF_VALUE_UINT64 || array.len != count) {
        char want[64];
        char got[64];
        snprintf(want, sizeof(want), "ARRAY<UINT64>[%" PRIu64 "]", count);
        if (!model_get_array(model, key, &array)) {
            snprintf(got, sizeof(got), "malformed ARRAY");
        } else {
            snprintf(got, sizeof(got), "ARRAY<%s>[%" PRIu64 "]",
                     q4e_metadata_type_name(array.type), array.len);
        }
        return q4e_reject(rejection, stage, key, want, got);
    }
    ds4_cursor cursor = cursor_at(model, array.data_pos);
    for (uint64_t i = 0; i < count; ++i) {
        uint64_t value = 0;
        if (!cursor_u64(&cursor, &value)) {
            return q4e_reject(rejection, stage, key, "complete array",
                              "truncated");
        }
        if (value != expected[i]) {
            char field[160];
            snprintf(field, sizeof(field), "%s[%" PRIu64 "]", key, i);
            return q4e_reject_u64(rejection, stage, field,
                                  expected[i], value);
        }
    }
    return true;
}

static bool q4e_metadata_unique(const ds4_model *model,
                                ds4_qwen4exp_rejection *rejection) {
    for (uint64_t i = 0; i < model->n_kv; ++i) {
        for (uint64_t j = i + 1u; j < model->n_kv; ++j) {
            if (q4e_ds4_str_equal(model->kv[i].key, model->kv[j].key)) {
                char field[128];
                const int n = model->kv[i].key.len < 96u
                    ? (int)model->kv[i].key.len : 96;
                snprintf(field, sizeof(field), "%.*s", n,
                         model->kv[i].key.ptr);
                return q4e_reject(rejection, DS4_QWEN4EXP_STAGE_IDENTITY,
                                  field, "one metadata entry", "duplicate");
            }
        }
    }
    return true;
}

static bool q4e_expect_absent(const ds4_model *model,
                              const char *key,
                              ds4_qwen4exp_stage stage,
                              ds4_qwen4exp_rejection *rejection) {
    return model_find_kv(model, key) == NULL ? true :
        q4e_reject(rejection, stage, key, "absent", "present");
}

static const ds4_qwen4exp_physical_profile *q4e_physical_profile_select(
        ds4_str id) {
#ifdef DS4_TEST_HOOKS
    static const ds4_qwen4exp_physical_profile fixture = {
        .id = Q4E_TEST_PHYSICAL_PROFILE_ID,
        .dense_ggml_type = 30u,
        .expert_storage = DS4_EXPERT_STORE_STORAGE_MLX_AFFINE4,
        .expert_group_size = 64u,
        .ple_codec = {
            .id = Q4E_PLE_TEST_CODEC_ID,
            .version = 1u,
            .group_size = 1u,
            .encoded_row_bytes = 320u,
        },
        .ple_rows_per_page = 320001536u,
        .ple_page_alignment = 4096u,
    };
    if (q4e_str_equal(id, fixture.id)) return &fixture;
#else
    (void)id;
#endif
    return NULL;
}

static bool q4e_validate_identity(const ds4_model *model,
                                  ds4_qwen4exp_rejection *rejection) {
    return q4e_metadata_unique(model, rejection) &&
        q4e_expect_string(model, "general.architecture",
                          DS4_QWEN4EXP_GGUF_ARCHITECTURE,
                          DS4_QWEN4EXP_STAGE_IDENTITY, rejection) &&
        q4e_expect_string(model, "ds4.model.profile_id",
                          DS4_QWEN4EXP_PROFILE_ID,
                          DS4_QWEN4EXP_STAGE_IDENTITY, rejection) &&
        q4e_expect_string(model, "ds4.model.source_architecture",
                          DS4_QWEN4EXP_SOURCE_ARCHITECTURE,
                          DS4_QWEN4EXP_STAGE_IDENTITY, rejection) &&
        q4e_expect_string(model, "ds4.model.source_revision",
                          Q4E_SOURCE_REVISION,
                          DS4_QWEN4EXP_STAGE_IDENTITY, rejection) &&
        q4e_expect_string(model, "ds4.model.transformers_revision",
                          Q4E_TRANSFORMERS_COMMIT,
                          DS4_QWEN4EXP_STAGE_IDENTITY, rejection) &&
        q4e_expect_string(model, "ds4.model.tensor_inventory_digest",
                          Q4E_INVENTORY_SHA256,
                          DS4_QWEN4EXP_STAGE_IDENTITY, rejection) &&
        q4e_expect_u32(model, "ds4.model.source_tensor_count",
                       Q4E_SOURCE_TENSORS,
                       DS4_QWEN4EXP_STAGE_IDENTITY, rejection) &&
        q4e_expect_u64(model, "ds4.model.source_bytes", Q4E_SOURCE_BYTES,
                       DS4_QWEN4EXP_STAGE_IDENTITY, rejection);
}

static const ds4_qwen4exp_physical_profile *q4e_validate_physical_profile(
        const ds4_model *model,
        ds4_qwen4exp_rejection *rejection) {
    const char *key = "ds4.model.physical_profile_id";
    ds4_kv *kv = q4e_kv_exact(model, key, GGUF_VALUE_STRING,
                              DS4_QWEN4EXP_STAGE_PHYSICAL_PROFILE, rejection);
    ds4_str id = {0};
    if (!kv) return NULL;
    ds4_cursor cursor = cursor_at(model, kv->value_pos);
    if (!cursor_string(&cursor, &id)) {
        q4e_reject(rejection, DS4_QWEN4EXP_STAGE_PHYSICAL_PROFILE, key,
                   "qualified physical profile", "truncated");
        return NULL;
    }
    const ds4_qwen4exp_physical_profile *profile =
        q4e_physical_profile_select(id);
    if (!profile) {
        char observed[128];
        const int n = id.len < 96u ? (int)id.len : 96;
        snprintf(observed, sizeof(observed), "%.*s", n, id.ptr);
        q4e_reject(rejection, DS4_QWEN4EXP_STAGE_PHYSICAL_PROFILE, key,
                   "qualified physical profile", observed);
    }
    return profile;
}

static bool q4e_validate_semantic_metadata(
        const ds4_model *model,
        const ds4_qwen4exp_profile *profile,
        ds4_qwen4exp_rejection *rejection) {
    uint32_t layer_types[DS4_QWEN4EXP_N_LAYER];
    uint32_t ple_layer_ids[1] = {DS4_QWEN4EXP_PLE_SOURCE_LAYER_ID};
    uint64_t ple_primes[DS4_QWEN4EXP_PLE_HEADS];
    uint64_t ple_offsets[DS4_QWEN4EXP_PLE_HEADS];
    for (uint32_t i = 0; i < DS4_QWEN4EXP_N_LAYER; ++i) {
        layer_types[i] = (uint32_t)profile->layer_type[i];
    }
    for (uint32_t i = 0; i < DS4_QWEN4EXP_PLE_HEADS; ++i) {
        ple_primes[i] = profile->ple_head_prime[i];
        ple_offsets[i] = profile->ple_head_offset[i];
    }
    if (model->n_kv != 108u) {
        return q4e_reject_u64(rejection, DS4_QWEN4EXP_STAGE_METADATA,
                              "metadata entry count", 108u, model->n_kv);
    }
    if (!q4e_expect_absent(model, "general.alignment",
                           DS4_QWEN4EXP_STAGE_METADATA, rejection) ||
        !q4e_expect_absent(model,
            "qwen4exp.mtp.mtp_use_hidden_state_from_layer",
            DS4_QWEN4EXP_STAGE_METADATA, rejection) ||
        !q4e_expect_absent(model, "qwen4exp.tokenizer.bos_token_id",
                           DS4_QWEN4EXP_STAGE_METADATA, rejection)) {
        return false;
    }

#define Q4E_U32(key, value) \
    do { if (!q4e_expect_u32(model, (key), (value), \
            DS4_QWEN4EXP_STAGE_METADATA, rejection)) return false; } while (0)
#define Q4E_BOOL(key, value) \
    do { if (!q4e_expect_bool(model, (key), (value), \
            DS4_QWEN4EXP_STAGE_METADATA, rejection)) return false; } while (0)
#define Q4E_F32(key, value) \
    do { if (!q4e_expect_f32(model, (key), (value), \
            DS4_QWEN4EXP_STAGE_METADATA, rejection)) return false; } while (0)
#define Q4E_STRING(key, value) \
    do { if (!q4e_expect_string(model, (key), (value), \
            DS4_QWEN4EXP_STAGE_METADATA, rejection)) return false; } while (0)
#define Q4E_U64(key, value) \
    do { if (!q4e_expect_u64(model, (key), (value), \
            DS4_QWEN4EXP_STAGE_METADATA, rejection)) return false; } while (0)

    Q4E_STRING("qwen4exp.top.model_type", "qwen4_exp");
    Q4E_BOOL("qwen4exp.top.language_model_only", false);
    Q4E_BOOL("qwen4exp.top.tie_word_embeddings", false);
    Q4E_U32("qwen4exp.top.image_token_id", 248056u);
    Q4E_U32("qwen4exp.top.video_token_id", 248057u);
    Q4E_U32("qwen4exp.top.vision_start_token_id", 248053u);
    Q4E_U32("qwen4exp.top.vision_end_token_id", 248054u);

    Q4E_STRING("qwen4exp.text.model_type", "qwen4_exp_text");
    Q4E_BOOL("qwen4exp.text.attention_bias", false);
    Q4E_F32("qwen4exp.text.attention_dropout", 0.0f);
    Q4E_U32("qwen4exp.text.bos_token_id", profile->ple_pad_token);
    Q4E_STRING("qwen4exp.text.dtype", "bfloat16");
    Q4E_U32("qwen4exp.text.eos_token_id", profile->ple_pad_token);
    Q4E_U32("qwen4exp.text.full_attention_interval", 4u);
    Q4E_U32("qwen4exp.text.hc_count", profile->residual_streams);
    Q4E_U32("qwen4exp.text.hc_lowrank", profile->gr_low_rank);
    Q4E_U32("qwen4exp.text.head_dim", profile->qsa_head_dim);
    Q4E_U32("qwen4exp.text.heads_per_ngram",
            profile->ple_heads_per_ngram);
    Q4E_STRING("qwen4exp.text.hidden_act", "silu");
    Q4E_U32("qwen4exp.text.hidden_size", profile->hidden_size);
    Q4E_U32("qwen4exp.text.indexer_budget", profile->qsa_token_budget);
    Q4E_U32("qwen4exp.text.indexer_compress_ratio",
            profile->qsa_compression);
    Q4E_U32("qwen4exp.text.indexer_head_dim",
            profile->qsa_index_head_dim);
    Q4E_U32("qwen4exp.text.indexer_kv_heads",
            profile->qsa_index_key_heads);
    Q4E_U32("qwen4exp.text.indexer_n_heads",
            profile->qsa_index_query_heads);
    Q4E_F32("qwen4exp.text.initializer_range", 0.02f);
    Q4E_U32("qwen4exp.text.linear_conv_kernel_dim",
            profile->gdn_conv_kernel);
    Q4E_U32("qwen4exp.text.linear_key_head_dim", profile->gdn_head_dim);
    Q4E_U32("qwen4exp.text.linear_num_key_heads", profile->gdn_key_heads);
    Q4E_U32("qwen4exp.text.linear_num_value_heads",
            profile->gdn_value_heads);
    Q4E_U32("qwen4exp.text.linear_value_head_dim", profile->gdn_head_dim);
    Q4E_U32("qwen4exp.text.make_ngram_vocab_size_divisible_by",
            profile->ple_row_alignment);
    Q4E_STRING("qwen4exp.text.mamba_ssm_dtype", "float32");
    Q4E_U32("qwen4exp.text.max_position_embeddings",
            profile->context_length);
    Q4E_U32("qwen4exp.text.moe_intermediate_size", profile->expert_dim);
    Q4E_U32("qwen4exp.text.mtp_num_hidden_layers",
            profile->source_mtp_layers);
    Q4E_BOOL("qwen4exp.text.mtp_use_dedicated_embeddings", false);
    Q4E_U32("qwen4exp.text.ngram_size", profile->ple_ngram_size);
    Q4E_U32("qwen4exp.text.ngram_vocab_size_base",
            profile->ple_vocab_base);
    Q4E_U32("qwen4exp.text.num_attention_heads",
            profile->qsa_query_heads);
    Q4E_U32("qwen4exp.text.num_experts", profile->experts);
    Q4E_U32("qwen4exp.text.num_experts_per_tok", profile->experts_used);
    Q4E_U32("qwen4exp.text.num_hidden_layers", profile->n_layer);
    Q4E_U32("qwen4exp.text.num_key_value_heads", profile->qsa_kv_heads);
    Q4E_STRING("qwen4exp.text.output_gate_type", "sigmoid");
    Q4E_BOOL("qwen4exp.text.output_router_logits", false);
    Q4E_F32("qwen4exp.text.partial_rotary_factor", 0.25f);
    Q4E_U32("qwen4exp.text.ple_conv_kernel_size",
            profile->ple_conv_kernel);
    Q4E_U32("qwen4exp.text.ple_embed_dim", DS4_QWEN4EXP_PLE_FLAT_SIZE);
    Q4E_F32("qwen4exp.text.rms_norm_eps", profile->rms_epsilon);
    Q4E_F32("qwen4exp.text.router_aux_loss_coef", 0.001f);
    Q4E_U32("qwen4exp.text.shared_expert_intermediate_size",
            profile->shared_expert_dim);
    Q4E_U32("qwen4exp.text.split_ngram_parts", profile->ple_split_parts);
    Q4E_BOOL("qwen4exp.text.tie_word_embeddings", false);
    Q4E_BOOL("qwen4exp.text.use_cache", true);
    Q4E_U32("qwen4exp.text.vocab_size", profile->vocab_size);
    Q4E_STRING("qwen4exp.text.rope.rope_type", "default");
    Q4E_U32("qwen4exp.text.rope.rope_theta", 10000000u);
    Q4E_F32("qwen4exp.text.rope.partial_rotary_factor", 0.25f);
    Q4E_BOOL("qwen4exp.text.rope.mrope_interleaved",
             profile->qsa_mrope_interleaved);

    Q4E_BOOL("qwen4exp.mtp.hybrid", true);
    Q4E_U32("qwen4exp.mtp.num_hidden_layers", profile->source_mtp_layers);
    Q4E_U32("qwen4exp.mtp.rope_theta", 10000000u);
    Q4E_U32("qwen4exp.default.seed", profile->ple_seed);
    Q4E_BOOL("qwen4exp.text.norm_topk_prob",
             profile->router_normalize_selected);
    Q4E_STRING("qwen4exp.ple.hash_id", DS4_PLE_STORE_V1_HASH_ID);
    Q4E_U32("qwen4exp.ple.hash_seed", profile->ple_seed);
    Q4E_U64("qwen4exp.ple.head_rows", UINT64_C(320001446));
    Q4E_U64("qwen4exp.ple.rows", profile->ple_rows);

#undef Q4E_U32
#undef Q4E_BOOL
#undef Q4E_F32
#undef Q4E_STRING
#undef Q4E_U64

    return q4e_expect_u32_array(model, "qwen4exp.text.layer_pattern",
                                layer_types, DS4_QWEN4EXP_N_LAYER,
                                DS4_QWEN4EXP_STAGE_METADATA, rejection) &&
        q4e_expect_u32_array(model, "qwen4exp.text.ple_layer_ids",
                             ple_layer_ids, 1u,
                             DS4_QWEN4EXP_STAGE_METADATA, rejection) &&
        q4e_expect_u32_array(model, "qwen4exp.text.rope.mrope_section",
                             profile->mrope_section, 3u,
                             DS4_QWEN4EXP_STAGE_METADATA, rejection) &&
        q4e_expect_u64_array(model, "qwen4exp.ple.layer_multipliers",
                             profile->ple_multiplier,
                             DS4_QWEN4EXP_PLE_HASH_MULTIPLIERS,
                             DS4_QWEN4EXP_STAGE_METADATA, rejection) &&
        q4e_expect_u64_array(model, "qwen4exp.ple.head_primes",
                             ple_primes,
                             DS4_QWEN4EXP_PLE_HEADS,
                             DS4_QWEN4EXP_STAGE_METADATA, rejection) &&
        q4e_expect_u64_array(model, "qwen4exp.ple.head_offsets",
                             ple_offsets,
                             DS4_QWEN4EXP_PLE_HEADS,
                             DS4_QWEN4EXP_STAGE_METADATA, rejection);
}

static bool q4e_validate_tokenizer(const ds4_model *model,
                                   const ds4_qwen4exp_profile *profile,
                                   ds4_qwen4exp_rejection *rejection) {
    return q4e_expect_string(model, "ds4.tokenizer.digest",
                             Q4E_TOKENIZER_SHA256,
                             DS4_QWEN4EXP_STAGE_TOKENIZER, rejection) &&
        q4e_expect_string(model, "ds4.tokenizer_config.digest",
                          Q4E_TOKENIZER_CONFIG_SHA256,
                          DS4_QWEN4EXP_STAGE_TOKENIZER, rejection) &&
        q4e_expect_string(model, "ds4.chat_template.digest",
                          Q4E_TEMPLATE_SHA256,
                          DS4_QWEN4EXP_STAGE_TOKENIZER, rejection) &&
        q4e_expect_u32(model, "qwen4exp.tokenizer.end_of_text_token_id",
                       profile->ple_pad_token,
                       DS4_QWEN4EXP_STAGE_TOKENIZER, rejection) &&
        q4e_expect_u32(model, "qwen4exp.tokenizer.pad_token_id",
                       profile->ple_pad_token,
                       DS4_QWEN4EXP_STAGE_TOKENIZER, rejection) &&
        q4e_expect_u32(model, "qwen4exp.tokenizer.im_start_token_id", 248045u,
                       DS4_QWEN4EXP_STAGE_TOKENIZER, rejection) &&
        q4e_expect_u32(model, "qwen4exp.tokenizer.im_end_token_id", 248046u,
                       DS4_QWEN4EXP_STAGE_TOKENIZER, rejection) &&
        q4e_expect_u32(model, "qwen4exp.tokenizer.eos_token_id", 248046u,
                       DS4_QWEN4EXP_STAGE_TOKENIZER, rejection) &&
        q4e_expect_u32(model, "qwen4exp.tokenizer.vision_start_token_id",
                       248053u,
                       DS4_QWEN4EXP_STAGE_TOKENIZER, rejection) &&
        q4e_expect_u32(model, "qwen4exp.tokenizer.vision_end_token_id",
                       248054u,
                       DS4_QWEN4EXP_STAGE_TOKENIZER, rejection) &&
        q4e_expect_u32(model, "qwen4exp.tokenizer.vision_pad_token_id",
                       248055u,
                       DS4_QWEN4EXP_STAGE_TOKENIZER, rejection) &&
        q4e_expect_u32(model, "qwen4exp.tokenizer.image_pad_token_id",
                       248056u,
                       DS4_QWEN4EXP_STAGE_TOKENIZER, rejection) &&
        q4e_expect_u32(model, "qwen4exp.tokenizer.video_pad_token_id",
                       248057u,
                       DS4_QWEN4EXP_STAGE_TOKENIZER, rejection);
}

static int q4e_owned_span_compare(const void *a, const void *b) {
    const q4e_owned_span *sa = a;
    const q4e_owned_span *sb = b;
    if (sa->off < sb->off) return -1;
    if (sa->off > sb->off) return 1;
    if (sa->end < sb->end) return -1;
    if (sa->end > sb->end) return 1;
    return 0;
}

static int q4e_span_compare(const void *a, const void *b) {
    const q4e_span *sa = a;
    const q4e_span *sb = b;
    if (sa->off < sb->off) return -1;
    if (sa->off > sb->off) return 1;
    if (sa->end < sb->end) return -1;
    if (sa->end > sb->end) return 1;
    return 0;
}

static bool q4e_add_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out || a > UINT64_MAX - b) return false;
    *out = a + b;
    return true;
}

static bool q4e_tensor_names_unique(const ds4_model *model,
                                    ds4_qwen4exp_rejection *rejection) {
    for (uint64_t i = 0; i < model->n_tensors; ++i) {
        for (uint64_t j = i + 1u; j < model->n_tensors; ++j) {
            if (q4e_ds4_str_equal(model->tensors[i].name,
                                  model->tensors[j].name)) {
                char field[128];
                const int n = model->tensors[i].name.len < 96u
                    ? (int)model->tensors[i].name.len : 96;
                snprintf(field, sizeof(field), "%.*s", n,
                         model->tensors[i].name.ptr);
                return q4e_reject(rejection, DS4_QWEN4EXP_STAGE_INVENTORY,
                                  field, "one tensor", "duplicate");
            }
        }
    }
    return true;
}

static bool q4e_forbidden_inventory_absent(
        const ds4_model *model,
        ds4_qwen4exp_rejection *rejection) {
    for (uint64_t i = 0; i < model->n_tensors; ++i) {
        const ds4_str name = model->tensors[i].name;
        const bool forbidden =
            q4e_starts_with(name, "model.visual.") ||
            q4e_starts_with(name, "mtp.") ||
            q4e_contains(name, ".mlp.experts.") ||
            q4e_contains(name,
                ".ple.ple_embedding.ngram_embedding.shard_") ||
            q4e_contains(name, ".ple.ple_embedding.layer_multipliers") ||
            q4e_contains(name,
                ".ple.ple_embedding.ngram_heads_offsets") ||
            q4e_contains(name,
                ".ple.ple_embedding.ngram_heads_vocab_sizes") ||
            (q4e_starts_with(name, "blk.") &&
             (q4e_contains(name, ".ffn_gate_exps.weight") ||
              q4e_contains(name, ".ffn_up_exps.weight") ||
              q4e_contains(name, ".ffn_down_exps.weight")));
        if (forbidden) {
            char observed[128];
            const int n = name.len < 96u ? (int)name.len : 96;
            snprintf(observed, sizeof(observed), "%.*s", n, name.ptr);
            return q4e_reject(rejection, DS4_QWEN4EXP_STAGE_INVENTORY,
                              "tensor policy", "no canonical/vision/MTP tensor",
                              observed);
        }
    }
    return true;
}

static bool q4e_expect_tensor(const ds4_model *model,
                              const char *name,
                              uint32_t type,
                              uint32_t ndim,
                              uint64_t d0,
                              uint64_t d1,
                              uint64_t d2,
                              q4e_owner owner,
                              bool *seen,
                              q4e_owned_span *ranges,
                              uint64_t *range_count,
                              ds4_qwen4exp_rejection *rejection) {
    ds4_tensor *tensor = model_find_tensor(model, name);
    if (!tensor) {
        return q4e_reject(rejection, DS4_QWEN4EXP_STAGE_INVENTORY,
                          name, "required tensor", "missing");
    }
    const uint64_t index = (uint64_t)(tensor - model->tensors);
    if (index >= model->n_tensors || seen[index]) {
        return q4e_reject(rejection, DS4_QWEN4EXP_STAGE_INVENTORY,
                          name, "one tensor", "duplicate binding");
    }
    if (tensor->type != type) {
        char expected[48];
        char observed[48];
        snprintf(expected, sizeof(expected), "%s(%u)",
                 tensor_type_name(type), type);
        snprintf(observed, sizeof(observed), "%s(%u)",
                 tensor_type_name(tensor->type), tensor->type);
        return q4e_reject(rejection, DS4_QWEN4EXP_STAGE_INVENTORY,
                          name, expected, observed);
    }
    const uint64_t dims[3] = {d0, d1, d2};
    if (tensor->ndim != ndim) {
        return q4e_reject_u64(rejection, DS4_QWEN4EXP_STAGE_INVENTORY,
                              name, ndim, tensor->ndim);
    }
    for (uint32_t i = 0; i < ndim; ++i) {
        if (tensor->dim[i] != dims[i]) {
            char field[160];
            snprintf(field, sizeof(field), "%s.dim[%u]", name, i);
            return q4e_reject_u64(rejection,
                                  DS4_QWEN4EXP_STAGE_INVENTORY,
                                  field, dims[i], tensor->dim[i]);
        }
    }
    uint64_t end = 0;
    if (tensor->bytes == 0 ||
        !q4e_add_u64(tensor->abs_offset, tensor->bytes, &end) ||
        tensor->abs_offset < model->tensor_data_pos || end > model->size ||
        model->alignment == 0 || tensor->rel_offset % model->alignment != 0 ||
        tensor->abs_offset - model->tensor_data_pos != tensor->rel_offset) {
        return q4e_reject(rejection, DS4_QWEN4EXP_STAGE_INVENTORY,
                          name, "bounded aligned tensor extent", "invalid extent");
    }
    seen[index] = true;
    ranges[*range_count].off = tensor->abs_offset;
    ranges[*range_count].end = end;
    ranges[*range_count].owner = owner;
    ranges[*range_count].name = tensor->name;
    ++*range_count;
    return true;
}

static bool q4e_expect_dense(const ds4_model *model,
                             const ds4_qwen4exp_physical_profile *physical,
                             const char *name,
                             uint32_t ndim,
                             uint64_t d0,
                             uint64_t d1,
                             uint64_t d2,
                             bool *seen,
                             q4e_owned_span *ranges,
                             uint64_t *range_count,
                             ds4_qwen4exp_rejection *rejection) {
    return q4e_expect_tensor(model, name, physical->dense_ggml_type,
                             ndim, d0, d1, d2, Q4E_OWNER_DENSE,
                             seen, ranges, range_count, rejection);
}

static bool q4e_expect_layer_dense(
        const ds4_model *model,
        const ds4_qwen4exp_physical_profile *physical,
        uint32_t layer,
        const char *suffix,
        uint32_t ndim,
        uint64_t d0,
        uint64_t d1,
        uint64_t d2,
        bool *seen,
        q4e_owned_span *ranges,
        uint64_t *range_count,
        ds4_qwen4exp_rejection *rejection) {
    char name[256];
    const int result = snprintf(name, sizeof(name),
        "model.language_model.layers.%u.%s", layer, suffix);
    if (result < 0 || (size_t)result >= sizeof(name)) {
        return q4e_reject(rejection, DS4_QWEN4EXP_STAGE_INVENTORY,
                          "tensor name", "bounded generated name", "overflow");
    }
    return q4e_expect_dense(model, physical, name, ndim, d0, d1, d2,
                            seen, ranges, range_count, rejection);
}

static bool q4e_validate_inventory(
        const ds4_model *model,
        const ds4_qwen4exp_profile *profile,
        const ds4_qwen4exp_physical_profile *physical,
        q4e_owned_span **ranges_out,
        ds4_tensor **expert_tensor_out,
        ds4_tensor **ple_tensor_out,
        ds4_qwen4exp_rejection *rejection) {
    bool *seen = NULL;
    q4e_owned_span *ranges = NULL;
    uint64_t count = 0;
    bool ok = false;

    if (model->n_tensors != Q4E_PHYSICAL_TENSORS) {
        return q4e_reject_u64(rejection, DS4_QWEN4EXP_STAGE_INVENTORY,
                              "physical tensor count",
                              Q4E_PHYSICAL_TENSORS, model->n_tensors);
    }
    if (!q4e_tensor_names_unique(model, rejection) ||
        !q4e_forbidden_inventory_absent(model, rejection)) {
        return false;
    }
    seen = calloc((size_t)model->n_tensors, sizeof(*seen));
    ranges = calloc((size_t)model->n_tensors, sizeof(*ranges));
    if (!seen || !ranges) {
        q4e_reject(rejection, DS4_QWEN4EXP_STAGE_INVENTORY,
                   "inventory scratch", "allocated", "out of memory");
        goto done;
    }

#define Q4E_GLOBAL(name, ndim, d0, d1, d2) \
    do { if (!q4e_expect_dense(model, physical, (name), (ndim), \
            (d0), (d1), (d2), seen, ranges, &count, rejection)) goto done; } while (0)
#define Q4E_LAYER(suffix, ndim, d0, d1, d2) \
    do { if (!q4e_expect_layer_dense(model, physical, layer, (suffix), (ndim), \
            (d0), (d1), (d2), seen, ranges, &count, rejection)) goto done; } while (0)

    Q4E_GLOBAL("lm_head.weight", 2u, 2560u, 248320u, 0u);
    Q4E_GLOBAL("model.language_model.embed_tokens.weight",
               2u, 2560u, 248320u, 0u);
    Q4E_GLOBAL("model.language_model.hyper_connection_mixer.hc_norm.weight",
               1u, 10240u, 0u, 0u);
    Q4E_GLOBAL(
        "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight",
        2u, 10240u, 320u, 0u);
    Q4E_GLOBAL(
        "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight",
        2u, 320u, 10240u, 0u);

    for (uint32_t layer = 0; layer < profile->n_layer; ++layer) {
        Q4E_LAYER("attn_hyper_connection.block_inject_weight.weight",
                  2u, 10240u, 4u, 0u);
        Q4E_LAYER("attn_hyper_connection.hc_norm.weight",
                  1u, 10240u, 0u, 0u);
        Q4E_LAYER("attn_hyper_connection.input_mix_weight_down.weight",
                  2u, 10240u, 320u, 0u);
        Q4E_LAYER("attn_hyper_connection.input_mix_weight_up.weight",
                  2u, 320u, 10240u, 0u);
        Q4E_LAYER("mlp.gate.weight", 2u, 2560u, 512u, 0u);
        Q4E_LAYER("mlp.shared_expert.down_proj.weight",
                  2u, 640u, 2560u, 0u);
        Q4E_LAYER("mlp.shared_expert.gate_proj.weight",
                  2u, 2560u, 640u, 0u);
        Q4E_LAYER("mlp.shared_expert.up_proj.weight",
                  2u, 2560u, 640u, 0u);
        Q4E_LAYER("mlp.shared_expert_gate.weight",
                  2u, 2560u, 1u, 0u);
        Q4E_LAYER("mlp_hyper_connection.block_inject_weight.weight",
                  2u, 10240u, 4u, 0u);
        Q4E_LAYER("mlp_hyper_connection.hc_norm.weight",
                  1u, 10240u, 0u, 0u);
        Q4E_LAYER("mlp_hyper_connection.input_mix_weight_down.weight",
                  2u, 10240u, 320u, 0u);
        Q4E_LAYER("mlp_hyper_connection.input_mix_weight_up.weight",
                  2u, 320u, 10240u, 0u);

        if (profile->layer_type[layer] == DS4_QWEN4EXP_LAYER_GDN) {
            Q4E_LAYER("linear_attn.A_log", 1u, 48u, 0u, 0u);
            Q4E_LAYER("linear_attn.conv1d.weight",
                      3u, 4u, 1u, 10240u);
            Q4E_LAYER("linear_attn.dt_bias", 1u, 48u, 0u, 0u);
            Q4E_LAYER("linear_attn.in_proj_a.weight",
                      2u, 2560u, 48u, 0u);
            Q4E_LAYER("linear_attn.in_proj_b.weight",
                      2u, 2560u, 48u, 0u);
            Q4E_LAYER("linear_attn.in_proj_qkv.weight",
                      2u, 2560u, 10240u, 0u);
            Q4E_LAYER("linear_attn.in_proj_z.weight",
                      2u, 2560u, 6144u, 0u);
            Q4E_LAYER("linear_attn.norm.weight", 1u, 128u, 0u, 0u);
            Q4E_LAYER("linear_attn.out_proj.weight",
                      2u, 6144u, 2560u, 0u);
        } else if (profile->layer_type[layer] == DS4_QWEN4EXP_LAYER_QSA) {
            Q4E_LAYER("self_attn.indexer.index_qk_proj.weight",
                      2u, 2560u, 640u, 0u);
            Q4E_LAYER("self_attn.indexer.k_layernorm.weight",
                      1u, 128u, 0u, 0u);
            Q4E_LAYER("self_attn.indexer.q_layernorm.weight",
                      1u, 128u, 0u, 0u);
            Q4E_LAYER("self_attn.k_norm.weight", 1u, 256u, 0u, 0u);
            Q4E_LAYER("self_attn.k_proj.weight",
                      2u, 2560u, 512u, 0u);
            Q4E_LAYER("self_attn.o_proj.weight",
                      2u, 6144u, 2560u, 0u);
            Q4E_LAYER("self_attn.q_norm.weight", 1u, 256u, 0u, 0u);
            Q4E_LAYER("self_attn.q_proj.weight",
                      2u, 2560u, 12288u, 0u);
            Q4E_LAYER("self_attn.v_proj.weight",
                      2u, 2560u, 512u, 0u);
        } else {
            q4e_reject_u64(rejection, DS4_QWEN4EXP_STAGE_INVENTORY,
                           "layer type", DS4_QWEN4EXP_LAYER_GDN,
                           profile->layer_type[layer]);
            goto done;
        }
    }

    {
        const uint32_t layer = profile->ple_runtime_layer;
        Q4E_LAYER("ple.conv1d.weight", 3u, 4u, 1u, 10240u);
        Q4E_LAYER("ple.key_proj.weight", 2u, 2560u, 10240u, 0u);
        Q4E_LAYER("ple.norm_conv.weight", 1u, 10240u, 0u, 0u);
        Q4E_LAYER("ple.norm_key.weight", 1u, 10240u, 0u, 0u);
        Q4E_LAYER("ple.norm_query.weight", 1u, 10240u, 0u, 0u);
        Q4E_LAYER("ple.value_proj.weight", 2u, 2560u, 2560u, 0u);
    }

#undef Q4E_GLOBAL
#undef Q4E_LAYER

    if (count != Q4E_DENSE_TENSORS) {
        q4e_reject_u64(rejection, DS4_QWEN4EXP_STAGE_INVENTORY,
                       "dense tensor count", Q4E_DENSE_TENSORS, count);
        goto done;
    }
    if (!q4e_expect_tensor(model, DS4_EXPERT_STORE_V2_TENSOR,
                           DS4_TENSOR_I8, 1u,
                           model_find_tensor(model, DS4_EXPERT_STORE_V2_TENSOR)
                               ? model_find_tensor(model,
                                   DS4_EXPERT_STORE_V2_TENSOR)->bytes : 0u,
                           0u, 0u, Q4E_OWNER_EXPERT, seen, ranges, &count,
                           rejection) ||
        !q4e_expect_tensor(model, DS4_PLE_STORE_V1_TENSOR,
                           DS4_TENSOR_I8, 1u,
                           model_find_tensor(model, DS4_PLE_STORE_V1_TENSOR)
                               ? model_find_tensor(model,
                                   DS4_PLE_STORE_V1_TENSOR)->bytes : 0u,
                           0u, 0u, Q4E_OWNER_PLE, seen, ranges, &count,
                           rejection)) {
        goto done;
    }
    if (count != Q4E_PHYSICAL_TENSORS) {
        q4e_reject_u64(rejection, DS4_QWEN4EXP_STAGE_INVENTORY,
                       "bound tensor count", Q4E_PHYSICAL_TENSORS, count);
        goto done;
    }
    for (uint64_t i = 0; i < model->n_tensors; ++i) {
        if (!seen[i]) {
            char observed[128];
            const int n = model->tensors[i].name.len < 96u
                ? (int)model->tensors[i].name.len : 96;
            snprintf(observed, sizeof(observed), "%.*s", n,
                     model->tensors[i].name.ptr);
            q4e_reject(rejection, DS4_QWEN4EXP_STAGE_INVENTORY,
                       "tensor inventory", "no extra tensor", observed);
            goto done;
        }
    }
    *expert_tensor_out = model_find_tensor(model, DS4_EXPERT_STORE_V2_TENSOR);
    *ple_tensor_out = model_find_tensor(model, DS4_PLE_STORE_V1_TENSOR);
    *ranges_out = ranges;
    ranges = NULL;
    ok = true;

done:
    free(seen);
    free(ranges);
    return ok;
}

static bool q4e_hex_digest(const char *hex, uint8_t digest[32]) {
    if (!hex || strlen(hex) != 64u) return false;
    for (size_t i = 0; i < 32u; ++i) {
        const unsigned char a = (unsigned char)hex[i * 2u];
        const unsigned char b = (unsigned char)hex[i * 2u + 1u];
        uint8_t hi;
        uint8_t lo;
        if (a >= '0' && a <= '9') hi = (uint8_t)(a - '0');
        else if (a >= 'a' && a <= 'f') hi = (uint8_t)(a - 'a' + 10u);
        else return false;
        if (b >= '0' && b <= '9') lo = (uint8_t)(b - '0');
        else if (b >= 'a' && b <= 'f') lo = (uint8_t)(b - 'a' + 10u);
        else return false;
        digest[i] = (uint8_t)((hi << 4u) | lo);
    }
    return true;
}

static bool q4e_open_expert_store(
        const ds4_model *model,
        const ds4_qwen4exp_physical_profile *physical,
        const ds4_tensor *tensor,
        ds4_expert_store **store_out,
        ds4_qwen4exp_rejection *rejection) {
    char error[192] = {0};
    ds4_expert_store *store = NULL;
    uint8_t source_sha[32];
    if (physical->expert_storage !=
            DS4_EXPERT_STORE_STORAGE_MLX_AFFINE4 ||
        physical->expert_group_size != 64u) {
        return q4e_reject(rejection, DS4_QWEN4EXP_STAGE_EXPERT_STORE,
                          "physical expert profile",
                          "MLX affine4 group 64", "invalid");
    }
    if (!q4e_expect_u32(model, "ds4.expert_store.family", 4u,
                        DS4_QWEN4EXP_STAGE_EXPERT_STORE, rejection) ||
        !q4e_expect_string(model, "ds4.expert_store.profile",
                           DS4_QWEN4EXP_PROFILE_ID,
                           DS4_QWEN4EXP_STAGE_EXPERT_STORE, rejection) ||
        !q4e_expect_string(model, "ds4.expert_store.codec", "mlx-affine4",
                           DS4_QWEN4EXP_STAGE_EXPERT_STORE, rejection) ||
        !q4e_expect_string(model,
                           "ds4.expert_store.codec_descriptor_digest",
                           Q4E_EXPERT_CODEC_DESCRIPTOR_SHA256,
                           DS4_QWEN4EXP_STAGE_EXPERT_STORE, rejection)) {
        return false;
    }
    if (!ds4_expert_store_open_embedded(
            &store, model->fd, tensor->abs_offset, tensor->bytes,
            DS4_EXPERT_STORE_FAMILY_QWEN4EXP, error, sizeof(error))) {
        return q4e_reject(rejection, DS4_QWEN4EXP_STAGE_EXPERT_STORE,
                          DS4_EXPERT_STORE_V2_TENSOR,
                          "valid Qwen4Exp ExpertMajor v2 manifest", error);
    }
    const ds4_expert_store_manifest *manifest =
        ds4_expert_store_manifest_get(store);
    if (!manifest || !q4e_hex_digest(Q4E_INVENTORY_SHA256, source_sha) ||
        manifest->version != DS4_EXPERT_STORE_V2_VERSION ||
        manifest->family != DS4_EXPERT_STORE_FAMILY_QWEN4EXP ||
        manifest->storage_format != physical->expert_storage ||
        manifest->group_size != physical->expert_group_size ||
        manifest->layer_count != DS4_QWEN4EXP_N_LAYER ||
        manifest->expert_count != DS4_QWEN4EXP_EXPERTS ||
        manifest->expert_used_count != DS4_QWEN4EXP_EXPERTS_USED ||
        manifest->source_tensor_count != Q4E_SOURCE_TENSORS ||
        manifest->source_size != Q4E_SOURCE_BYTES ||
        manifest->store_size != tensor->bytes ||
        memcmp(manifest->source_sha256, source_sha, sizeof(source_sha)) != 0) {
        ds4_expert_store_close(store);
        return q4e_reject(rejection, DS4_QWEN4EXP_STAGE_EXPERT_STORE,
                          "expert manifest cross-check",
                          "exact family/profile/source/geometry",
                          "mismatch");
    }
    for (uint32_t layer = 0; layer < DS4_QWEN4EXP_N_LAYER; ++layer) {
        const ds4_expert_store_layer *entry =
            ds4_expert_store_layer_at(store, layer);
        if (!entry || entry->layer != layer ||
            entry->expert_count != DS4_QWEN4EXP_EXPERTS) {
            ds4_expert_store_close(store);
            return q4e_reject_u64(rejection,
                                  DS4_QWEN4EXP_STAGE_EXPERT_STORE,
                                  "expert layer descriptor", layer,
                                  entry ? entry->layer : UINT64_MAX);
        }
    }
    *store_out = store;
    return true;
}

static bool q4e_open_ple_store(
        const ds4_model *model,
        const ds4_qwen4exp_profile *profile,
        const ds4_qwen4exp_physical_profile *physical,
        const ds4_tensor *tensor,
        ds4_ple_store **store_out,
        ds4_qwen4exp_rejection *rejection) {
    char error[192] = {0};
    ds4_ple_store *store = NULL;
    ds4_ple_store_geometry geometry = {
        .row_count = profile->ple_rows,
        .row_width = profile->ple_head_dim,
        .row_alignment = profile->ple_row_alignment,
        .rows_per_page = physical->ple_rows_per_page,
        .page_alignment = physical->ple_page_alignment,
    };
    memcpy(geometry.head_prime, profile->ple_head_prime,
           sizeof(geometry.head_prime));
    memcpy(geometry.head_offset, profile->ple_head_offset,
           sizeof(geometry.head_offset));

    if (!q4e_expect_u32(model, "ds4.ple_store.family", 4u,
                        DS4_QWEN4EXP_STAGE_PLE_STORE, rejection) ||
        !q4e_expect_string(model, "ds4.ple_store.profile",
                           DS4_QWEN4EXP_PROFILE_ID,
                           DS4_QWEN4EXP_STAGE_PLE_STORE, rejection) ||
        !q4e_expect_string(model, "ds4.ple_store.codec",
                           physical->ple_codec.id,
                           DS4_QWEN4EXP_STAGE_PLE_STORE, rejection) ||
        !q4e_expect_string(model, "ds4.ple_store.codec_descriptor_digest",
                           Q4E_PLE_CODEC_DESCRIPTOR_SHA256,
                           DS4_QWEN4EXP_STAGE_PLE_STORE, rejection)) {
        return false;
    }
    if (!ds4_ple_store_open_embedded(
            &store, model->fd, tensor->abs_offset, tensor->bytes,
            &geometry, &physical->ple_codec, error, sizeof(error))) {
        return q4e_reject(rejection, DS4_QWEN4EXP_STAGE_PLE_STORE,
                          DS4_PLE_STORE_V1_TENSOR,
                          "valid Qwen4Exp PLE v1 manifest", error);
    }
    const ds4_ple_store_manifest *manifest =
        ds4_ple_store_manifest_get(store);
    if (!manifest ||
        manifest->version != DS4_PLE_STORE_V1_VERSION ||
        manifest->family != DS4_PLE_STORE_FAMILY_QWEN4EXP ||
        strcmp(manifest->profile_id, DS4_QWEN4EXP_PROFILE_ID) != 0 ||
        strcmp(manifest->hash_id, DS4_PLE_STORE_V1_HASH_ID) != 0 ||
        strcmp(manifest->codec_id, physical->ple_codec.id) != 0 ||
        manifest->codec_version != physical->ple_codec.version ||
        manifest->codec_group_size != physical->ple_codec.group_size ||
        manifest->encoded_row_bytes != physical->ple_codec.encoded_row_bytes ||
        manifest->row_count != profile->ple_rows ||
        manifest->row_width != profile->ple_head_dim ||
        manifest->row_alignment != profile->ple_row_alignment ||
        manifest->rows_per_page != physical->ple_rows_per_page ||
        manifest->page_alignment != physical->ple_page_alignment ||
        manifest->store_bytes != tensor->bytes ||
        memcmp(manifest->head_prime, profile->ple_head_prime,
               sizeof(manifest->head_prime)) != 0 ||
        memcmp(manifest->head_offset, profile->ple_head_offset,
               sizeof(manifest->head_offset)) != 0) {
        ds4_ple_store_close(store);
        return q4e_reject(rejection, DS4_QWEN4EXP_STAGE_PLE_STORE,
                          "PLE manifest cross-check",
                          "exact profile/hash/codec/geometry", "mismatch");
    }
    *store_out = store;
    return true;
}

static bool q4e_validate_policy(const ds4_model *model,
                                ds4_qwen4exp_rejection *rejection) {
    return q4e_expect_bool(model, "ds4.text_only", true,
                           DS4_QWEN4EXP_STAGE_POLICY, rejection) &&
        q4e_expect_bool(model, "ds4.mtp.present", false,
                        DS4_QWEN4EXP_STAGE_POLICY, rejection) &&
        q4e_expect_bool(model, "ds4.mtp.executed", false,
                        DS4_QWEN4EXP_STAGE_POLICY, rejection);
}

static bool q4e_spans_intersect(q4e_span a, q4e_span b) {
    return a.off < b.end && b.off < a.end;
}

static bool q4e_validate_ownership(
        const ds4_model *model,
        q4e_owned_span *ranges,
        q4e_span **dense_pages_out,
        size_t *dense_page_count_out,
        ds4_qwen4exp_admission_report *report,
        ds4_qwen4exp_rejection *rejection) {
    uint64_t dense_bytes = 0;
    uint64_t expert_bytes = 0;
    uint64_t ple_bytes = 0;
    uint64_t padding_bytes = 0;
    uint64_t previous_end = model->tensor_data_pos;
    q4e_span expert = {0};
    q4e_span ple = {0};
    q4e_span *pages = NULL;
    size_t page_count = 0;

    qsort(ranges, Q4E_PHYSICAL_TENSORS, sizeof(ranges[0]),
          q4e_owned_span_compare);
    for (uint64_t i = 0; i < Q4E_PHYSICAL_TENSORS; ++i) {
        const q4e_owner expected_owner = i < Q4E_DENSE_TENSORS
            ? Q4E_OWNER_DENSE : i == Q4E_DENSE_TENSORS
            ? Q4E_OWNER_EXPERT : Q4E_OWNER_PLE;
        if (ranges[i].owner != expected_owner) {
            q4e_reject(rejection, DS4_QWEN4EXP_STAGE_OWNERSHIP,
                       "physical owner order", "dense then ExpertMajor then PLE",
                       "owner out of order");
            goto fail;
        }
        if (i != 0u && ranges[i].owner == Q4E_OWNER_DENSE &&
            q4e_ds4_str_compare(ranges[i - 1u].name, ranges[i].name) >= 0) {
            q4e_reject(rejection, DS4_QWEN4EXP_STAGE_OWNERSHIP,
                       "dense physical order", "ascending source identity",
                       "out of order");
            goto fail;
        }
        if (ranges[i].off < previous_end || ranges[i].end <= ranges[i].off) {
            q4e_reject(rejection, DS4_QWEN4EXP_STAGE_OWNERSHIP,
                       "raw tensor spans", "sorted and disjoint", "overlap");
            goto fail;
        }
        uint64_t aligned = 0;
        const uint64_t owner_alignment = ranges[i].owner == Q4E_OWNER_DENSE
            ? model->alignment : 4096u;
        if (!checked_align_up_u64(previous_end, owner_alignment, &aligned) ||
            ranges[i].off != aligned) {
            q4e_reject(rejection, DS4_QWEN4EXP_STAGE_OWNERSHIP,
                       "raw tensor coverage", "only minimal GGUF padding",
                       "gap or misalignment");
            goto fail;
        }
        if (!q4e_add_u64(padding_bytes, ranges[i].off - previous_end,
                         &padding_bytes)) {
            q4e_reject(rejection, DS4_QWEN4EXP_STAGE_OWNERSHIP,
                       "padding bytes", "no overflow", "overflow");
            goto fail;
        }
        const uint64_t bytes = ranges[i].end - ranges[i].off;
        uint64_t *owner_bytes = ranges[i].owner == Q4E_OWNER_DENSE
            ? &dense_bytes : ranges[i].owner == Q4E_OWNER_EXPERT
            ? &expert_bytes : &ple_bytes;
        if (!q4e_add_u64(*owner_bytes, bytes, owner_bytes)) {
            q4e_reject(rejection, DS4_QWEN4EXP_STAGE_OWNERSHIP,
                       "owner bytes", "no overflow", "overflow");
            goto fail;
        }
        if (ranges[i].owner == Q4E_OWNER_EXPERT) {
            if (expert.end != 0u || ranges[i].off % 4096u != 0u ||
                ranges[i].end % 4096u != 0u) {
                q4e_reject(rejection, DS4_QWEN4EXP_STAGE_OWNERSHIP,
                           "ExpertMajor owner",
                           "one outer 4096-aligned extent", "invalid");
                goto fail;
            }
            expert = (q4e_span){ranges[i].off, ranges[i].end};
        } else if (ranges[i].owner == Q4E_OWNER_PLE) {
            if (ple.end != 0u || ranges[i].off % 4096u != 0u ||
                ranges[i].end % 4096u != 0u) {
                q4e_reject(rejection, DS4_QWEN4EXP_STAGE_OWNERSHIP,
                           "PLE owner", "one outer 4096-aligned extent",
                           "invalid");
                goto fail;
            }
            ple = (q4e_span){ranges[i].off, ranges[i].end};
        }
        previous_end = ranges[i].end;
    }
    if (previous_end > model->size) {
        q4e_reject(rejection, DS4_QWEN4EXP_STAGE_OWNERSHIP,
                   "final tensor extent", "inside file", "outside file");
        goto fail;
    }
    if (previous_end != model->size) {
        q4e_reject(rejection, DS4_QWEN4EXP_STAGE_OWNERSHIP,
                   "PLE end", "exact file extent", "trailing bytes");
        goto fail;
    }
    if (expert.end == 0u || ple.end == 0u || q4e_spans_intersect(expert, ple)) {
        q4e_reject(rejection, DS4_QWEN4EXP_STAGE_OWNERSHIP,
                   "opaque extents", "one disjoint ExpertMajor and PLE extent",
                   "missing or overlapping");
        goto fail;
    }

    const uint64_t page = 4096u;
    pages = calloc(Q4E_DENSE_TENSORS, sizeof(*pages));
    if (!pages) {
        q4e_reject(rejection, DS4_QWEN4EXP_STAGE_OWNERSHIP,
                   "dense page spans", "allocated", "out of memory");
        goto fail;
    }
    for (uint64_t i = 0; i < Q4E_PHYSICAL_TENSORS; ++i) {
        if (ranges[i].owner != Q4E_OWNER_DENSE) continue;
        const uint64_t lo = ranges[i].off & ~(page - 1u);
        uint64_t hi = 0;
        if (!checked_align_up_u64(ranges[i].end, page, &hi)) {
            q4e_reject(rejection, DS4_QWEN4EXP_STAGE_OWNERSHIP,
                       "dense page extent", "no overflow", "overflow");
            goto fail;
        }
        if (hi > model->size) hi = model->size;
        pages[page_count++] = (q4e_span){lo, hi};
    }
    qsort(pages, page_count, sizeof(pages[0]), q4e_span_compare);
    size_t merged = 0;
    for (size_t i = 0; i < page_count; ++i) {
        if (merged != 0u && pages[i].off <= pages[merged - 1u].end) {
            if (pages[i].end > pages[merged - 1u].end) {
                pages[merged - 1u].end = pages[i].end;
            }
        } else {
            pages[merged++] = pages[i];
        }
    }
    page_count = merged;
    uint64_t dense_page_bytes = 0;
    for (size_t i = 0; i < page_count; ++i) {
        if (q4e_spans_intersect(pages[i], expert) ||
            q4e_spans_intersect(pages[i], ple)) {
            q4e_reject(rejection, DS4_QWEN4EXP_STAGE_OWNERSHIP,
                       "dense page span", "disjoint from opaque stores",
                       "page-rounded intersection");
            goto fail;
        }
        if (!q4e_add_u64(dense_page_bytes,
                         pages[i].end - pages[i].off,
                         &dense_page_bytes)) {
            q4e_reject(rejection, DS4_QWEN4EXP_STAGE_OWNERSHIP,
                       "dense page bytes", "no overflow", "overflow");
            goto fail;
        }
    }

    uint64_t owned = 0;
    uint64_t covered = model->tensor_data_pos;
    if (!q4e_add_u64(owned, dense_bytes, &owned) ||
        !q4e_add_u64(owned, expert_bytes, &owned) ||
        !q4e_add_u64(owned, ple_bytes, &owned) ||
        !q4e_add_u64(covered, owned, &covered) ||
        !q4e_add_u64(covered, padding_bytes, &covered) ||
        covered != model->size) {
        q4e_reject(rejection, DS4_QWEN4EXP_STAGE_OWNERSHIP,
                   "whole-file ownership", "exact file byte coverage",
                   "coverage mismatch");
        goto fail;
    }
    report->file_bytes = model->size;
    report->header_bytes = model->tensor_data_pos;
    report->dense_bytes = dense_bytes;
    report->expert_bytes = expert_bytes;
    report->ple_bytes = ple_bytes;
    report->padding_bytes = padding_bytes;
    report->owned_bytes = owned;
    report->dense_page_bytes = dense_page_bytes;
    *dense_pages_out = pages;
    *dense_page_count_out = page_count;
    return true;

fail:
    free(pages);
    return false;
}

static DS4_MAYBE_UNUSED void qwen4exp_loader_close(
        ds4_qwen4exp_loader *loader) {
    if (!loader) return;
    ds4_expert_store_close(loader->expert_store);
    ds4_ple_store_close(loader->ple_store);
    free(loader->dense_page_spans);
    memset(loader, 0, sizeof(*loader));
}

static DS4_MAYBE_UNUSED bool qwen4exp_loader_admit(
        const ds4_model *model,
        ds4_qwen4exp_loader *out,
        ds4_qwen4exp_rejection *rejection) {
    ds4_qwen4exp_loader pending;
    q4e_owned_span *ranges = NULL;
    ds4_tensor *expert_tensor = NULL;
    ds4_tensor *ple_tensor = NULL;
    const ds4_qwen4exp_physical_profile *physical = NULL;
    const ds4_qwen4exp_profile *semantic = ds4_qwen4exp_profile_get();
    memset(&pending, 0, sizeof(pending));
    if (rejection) memset(rejection, 0, sizeof(*rejection));

    if (!model || !out || !semantic ||
        !ds4_qwen4exp_profile_validate(semantic)) {
        return q4e_reject(rejection, DS4_QWEN4EXP_STAGE_IDENTITY,
                          "loader input/profile", "valid", "invalid");
    }
    pending.report.stage = DS4_QWEN4EXP_STAGE_IDENTITY;
    if (!q4e_validate_identity(model, rejection)) goto fail;

    pending.report.stage = DS4_QWEN4EXP_STAGE_PHYSICAL_PROFILE;
    physical = q4e_validate_physical_profile(model, rejection);
    if (!physical) goto fail;
    pending.report.physical_profile_id = physical->id;

    pending.report.stage = DS4_QWEN4EXP_STAGE_METADATA;
    if (!q4e_validate_semantic_metadata(model, semantic, rejection)) goto fail;

    pending.report.stage = DS4_QWEN4EXP_STAGE_TOKENIZER;
    if (!q4e_validate_tokenizer(model, semantic, rejection)) goto fail;

    pending.report.stage = DS4_QWEN4EXP_STAGE_INVENTORY;
    if (!q4e_validate_inventory(model, semantic, physical, &ranges,
                                &expert_tensor, &ple_tensor, rejection)) {
        goto fail;
    }

    pending.report.stage = DS4_QWEN4EXP_STAGE_EXPERT_STORE;
    if (!q4e_open_expert_store(model, physical, expert_tensor,
                               &pending.expert_store, rejection)) {
        goto fail;
    }
    pending.report.expert_manifest_verified = true;

    pending.report.stage = DS4_QWEN4EXP_STAGE_PLE_STORE;
    if (!q4e_open_ple_store(model, semantic, physical, ple_tensor,
                            &pending.ple_store, rejection)) {
        goto fail;
    }
    pending.report.ple_manifest_verified = true;

    pending.report.stage = DS4_QWEN4EXP_STAGE_POLICY;
    if (!q4e_validate_policy(model, rejection)) goto fail;

    pending.report.stage = DS4_QWEN4EXP_STAGE_OWNERSHIP;
    if (!q4e_validate_ownership(model, ranges,
                                &pending.dense_page_spans,
                                &pending.dense_page_span_count,
                                &pending.report, rejection)) {
        goto fail;
    }
    free(ranges);
    ranges = NULL;

    pending.report.admitted = true;
    pending.report.runtime_supported = false;
    pending.report.payload_verified = false;
    pending.report.text_only = true;
    pending.report.mtp_present = false;
    pending.report.vision_present = false;
    pending.report.tokenizer_content_verified = false;
    pending.report.stage = DS4_QWEN4EXP_STAGE_ACCEPTED;
    pending.report.physical_tensors = Q4E_PHYSICAL_TENSORS;
    pending.report.dense_tensors = Q4E_DENSE_TENSORS;
    pending.report.expert_stores = 1u;
    pending.report.ple_stores = 1u;
    pending.report.source_tensor_count = Q4E_SOURCE_TENSORS;
    pending.report.source_bytes = Q4E_SOURCE_BYTES;
    *out = pending;
    if (rejection) memset(rejection, 0, sizeof(*rejection));
    return true;

fail:
    free(ranges);
    qwen4exp_loader_close(&pending);
    return false;
}

static void q4e_json_string(FILE *stream, const char *value) {
    fputc('"', stream);
    if (value) {
        for (const unsigned char *p = (const unsigned char *)value; *p; ++p) {
            switch (*p) {
            case '"': fputs("\\\"", stream); break;
            case '\\': fputs("\\\\", stream); break;
            case '\b': fputs("\\b", stream); break;
            case '\f': fputs("\\f", stream); break;
            case '\n': fputs("\\n", stream); break;
            case '\r': fputs("\\r", stream); break;
            case '\t': fputs("\\t", stream); break;
            default:
                if (*p < 0x20u) fprintf(stream, "\\u%04x", (unsigned)*p);
                else fputc(*p, stream);
                break;
            }
        }
    }
    fputc('"', stream);
}

static const char *q4e_json_bool(bool value) {
    return value ? "true" : "false";
}

static DS4_MAYBE_UNUSED void qwen4exp_loader_print_json(
        FILE *stream,
        const ds4_qwen4exp_loader *loader,
        const ds4_qwen4exp_rejection *rejection) {
    if (!stream) return;
    ds4_qwen4exp_admission_report empty;
    memset(&empty, 0, sizeof(empty));
    const ds4_qwen4exp_admission_report *r = loader ? &loader->report : &empty;
    const ds4_qwen4exp_stage stage = r->admitted
        ? r->stage : rejection ? rejection->stage : r->stage;
    fputs("{\"schemaVersion\":1,\"family\":\"qwen4exp\",", stream);
    fputs("\"profileId\":\"qwen4exp-base-v1\",\"physicalProfileId\":", stream);
    if (r->physical_profile_id) q4e_json_string(stream, r->physical_profile_id);
    else fputs("null", stream);
    fputs(",\"stage\":", stream);
    q4e_json_string(stream, qwen4exp_stage_name(stage));
    fprintf(stream,
        ",\"admitted\":%s,\"runtimeSupported\":%s,"
        "\"payloadVerified\":%s,\"textOnly\":%s,"
        "\"mtpPresent\":%s,\"visionPresent\":%s,"
        "\"physicalTensors\":%" PRIu64 ",\"denseTensors\":%" PRIu64 ","
        "\"expertStores\":%" PRIu64 ",\"pleStores\":%" PRIu64 ","
        "\"fileBytes\":%" PRIu64 ",\"headerBytes\":%" PRIu64 ","
        "\"denseBytes\":%" PRIu64 ",\"expertBytes\":%" PRIu64 ","
        "\"pleBytes\":%" PRIu64 ",\"paddingBytes\":%" PRIu64 ","
        "\"ownedBytes\":%" PRIu64 ",\"densePageBytes\":%" PRIu64 ","
        "\"expertManifestVerified\":%s,\"pleManifestVerified\":%s,"
        "\"sourceTensorCount\":%" PRIu64 ",\"sourceBytes\":%" PRIu64 ","
        "\"sourceInventorySha256\":\"%s\","
        "\"tokenizerContentVerified\":%s,\"rejection\":",
        q4e_json_bool(r->admitted), q4e_json_bool(r->runtime_supported),
        q4e_json_bool(r->payload_verified), q4e_json_bool(r->text_only),
        q4e_json_bool(r->mtp_present), q4e_json_bool(r->vision_present),
        r->physical_tensors, r->dense_tensors, r->expert_stores,
        r->ple_stores, r->file_bytes, r->header_bytes, r->dense_bytes,
        r->expert_bytes, r->ple_bytes, r->padding_bytes, r->owned_bytes,
        r->dense_page_bytes, q4e_json_bool(r->expert_manifest_verified),
        q4e_json_bool(r->ple_manifest_verified), r->source_tensor_count,
        r->source_bytes, Q4E_INVENTORY_SHA256,
        q4e_json_bool(r->tokenizer_content_verified));
    if (rejection && rejection->field[0] != '\0') {
        fputs("{\"field\":", stream);
        q4e_json_string(stream, rejection->field);
        fputs(",\"expected\":", stream);
        q4e_json_string(stream, rejection->expected);
        fputs(",\"observed\":", stream);
        q4e_json_string(stream, rejection->observed);
        fputc('}', stream);
    } else {
        fputs("null", stream);
    }
    fputs("}\n", stream);
}

static DS4_MAYBE_UNUSED void qwen4exp_loader_print_summary(
        FILE *stream,
        const ds4_qwen4exp_loader *loader) {
    if (!stream || !loader) return;
    const ds4_qwen4exp_admission_report *r = &loader->report;
    fprintf(stream,
        "ds4: Qwen4Exp structural admission: stage=%s tensors=%" PRIu64
        " dense=%" PRIu64 " expertStores=%" PRIu64 " pleStores=%" PRIu64
        " runtimeSupported=%s payloadVerified=%s\n",
        qwen4exp_stage_name(r->stage), r->physical_tensors,
        r->dense_tensors, r->expert_stores, r->ple_stores,
        q4e_json_bool(r->runtime_supported),
        q4e_json_bool(r->payload_verified));
}

#undef Q4E_SOURCE_BYTES
#undef Q4E_PLE_TEST_CODEC_ID
#undef Q4E_PLE_CODEC_DESCRIPTOR_SHA256
#undef Q4E_EXPERT_CODEC_DESCRIPTOR_SHA256
#undef Q4E_TEMPLATE_SHA256
#undef Q4E_TOKENIZER_CONFIG_SHA256
#undef Q4E_TOKENIZER_SHA256
#undef Q4E_INVENTORY_SHA256
#undef Q4E_TRANSFORMERS_COMMIT
#undef Q4E_SOURCE_REVISION
#undef Q4E_TEST_PHYSICAL_PROFILE_ID
