#ifndef DS4_EXPERT_STORE_H
#define DS4_EXPERT_STORE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define DS4_EXPERT_STORE_V2_TENSOR "ds4.expert_major.v2"
#define DS4_DSPARK_EXPERT_STORE_V2_TENSOR "ds4.dspark.expert_major.v2"

enum {
    DS4_EXPERT_STORE_V2_VERSION = 2,
    DS4_EXPERT_STORE_V2_SHA256_BYTES = 32,
    DS4_EXPERT_STORE_V2_COMPONENTS = 3,
    DS4_EXPERT_STORE_V2_MAX_LAYERS = 79,
    DS4_EXPERT_STORE_V2_MAX_MODEL_LAYER = 127,
    DS4_EXPERT_STORE_V2_MAX_EXPERTS = 384,
    DS4_DSPARK_0731_STAGE_COUNT = 3,
    DS4_DSPARK_0731_EXPERT_COUNT = 256,
    DS4_DSPARK_0731_EXPERT_USED_COUNT = 6,
    DS4_DSPARK_0731_STATIC_TENSOR_COUNT = 72,
    DS4_DSPARK_0731_ROUTED_TENSOR_COUNT = 9,
    DS4_DSPARK_0731_SOURCE_TENSOR_COUNT =
        DS4_DSPARK_0731_STATIC_TENSOR_COUNT +
        DS4_DSPARK_0731_ROUTED_TENSOR_COUNT,
    DS4_DSPARK_0731_PHYSICAL_TENSOR_COUNT =
        DS4_DSPARK_0731_STATIC_TENSOR_COUNT + 1,
};

#define DS4_DSPARK_0731_RECORD_BYTES UINT64_C(7077888)
#define DS4_DSPARK_0731_DATA_OFFSET UINT64_C(4096)
#define DS4_DSPARK_0731_STAGE_BYTES UINT64_C(1811939328)
#define DS4_DSPARK_0731_DATA_BYTES UINT64_C(5435817984)
#define DS4_DSPARK_0731_STORE_BYTES UINT64_C(5435822080)
#define DS4_DSPARK_0731_STATIC_PAYLOAD_BYTES UINT64_C(553290668)
#define DS4_DSPARK_0731_SOURCE_BYTES UINT64_C(5989114912)

typedef enum {
    /* GGML block payloads such as Q2_K, IQ2_XS, IQ3_XXS, and IQ4_XS. */
    DS4_EXPERT_STORE_STORAGE_GGML = 0,
    /* MLX affine 4-bit groups, physically interleaved as
     * [32 packed bytes | BF16 scale | BF16 bias] for every 64 weights.
     * The logical component type remains Q4_K because the byte geometry is
     * identical (36 bytes per 64 values / 144 bytes per 256 values). */
    DS4_EXPERT_STORE_STORAGE_MLX_AFFINE4 = 1,
} ds4_expert_store_storage;

typedef enum {
    DS4_EXPERT_STORE_FAMILY_DEEPSEEK4 = 1,
    DS4_EXPERT_STORE_FAMILY_GLM_DSA = 2,
    DS4_EXPERT_STORE_FAMILY_QWEN35_MOE = 3,
} ds4_expert_store_family;

typedef enum {
    DS4_EXPERT_STORE_GATE = 0,
    DS4_EXPERT_STORE_UP = 1,
    DS4_EXPERT_STORE_DOWN = 2,
} ds4_expert_store_role;

typedef struct {
    uint32_t role;
    uint32_t ggml_type;
    uint32_t block_elements;
    uint64_t dim[3];
    uint64_t expert_bytes;
    uint64_t record_offset;
} ds4_expert_store_component;

typedef struct {
    uint32_t layer;
    uint32_t expert_count;
    uint64_t record_bytes;
    uint64_t data_offset;
    uint64_t data_size;
    ds4_expert_store_component component[DS4_EXPERT_STORE_V2_COMPONENTS];
} ds4_expert_store_layer;

typedef struct {
    uint32_t version;
    uint32_t family;
    uint32_t storage_format;
    uint32_t group_size;
    uint32_t layer_count;
    uint32_t expert_count;
    uint32_t expert_used_count;
    uint64_t source_tensor_count;
    uint64_t source_size;
    uint64_t data_offset;
    uint64_t data_size;
    uint64_t store_size;
    uint8_t source_sha256[DS4_EXPERT_STORE_V2_SHA256_BYTES];
    uint8_t payload_sha256[DS4_EXPERT_STORE_V2_SHA256_BYTES];
    uint8_t manifest_sha256[DS4_EXPERT_STORE_V2_SHA256_BYTES];
} ds4_expert_store_manifest;

typedef struct ds4_expert_store ds4_expert_store;

/* Opens the v2 store embedded in one opaque I8 GGUF tensor. The descriptor is
 * duplicated, so the returned reader has an explicit lifetime. Validation is
 * structural and checks the manifest digest; the multi-GiB payload digest is
 * an offline publication gate rather than a model-startup read. */
bool ds4_expert_store_open_embedded(
        ds4_expert_store **out,
        int                fd,
        uint64_t           offset,
        uint64_t           bytes,
        uint32_t           expected_family,
        char              *error,
        size_t             error_size);

void ds4_expert_store_close(ds4_expert_store *store);

const ds4_expert_store_manifest *ds4_expert_store_manifest_get(
        const ds4_expert_store *store);

const ds4_expert_store_layer *ds4_expert_store_layer_get(
        const ds4_expert_store *store,
        uint32_t                layer);

/* Descriptor-order access supports routed inventories with a dense prefix,
 * such as GLM layers 3..78. */
const ds4_expert_store_layer *ds4_expert_store_layer_at(
        const ds4_expert_store *store,
        uint32_t                index);

/* Applies the exact routed-store contract of the final DeepSeek V4 Flash 0731
 * DSpark support checkpoint. This deliberately sits above the generic v2
 * codec validation: another three-layer DeepSeek store is not interchangeable
 * merely because its header and descriptor digest are well formed. */
bool ds4_expert_store_validate_dspark_0731(
        const ds4_expert_store *store,
        char                   *error,
        size_t                  error_size);

/* Returns an absolute file offset for one complete component of one expert. */
bool ds4_expert_store_slice_get(
        const ds4_expert_store *store,
        uint32_t                layer,
        uint32_t                expert,
        uint32_t                role,
        uint64_t               *offset,
        uint64_t               *bytes);

int ds4_expert_store_fd(const ds4_expert_store *store);
uint64_t ds4_expert_store_file_offset(const ds4_expert_store *store);

#endif
