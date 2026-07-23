#ifndef DS4_KV_QUANT_H
#define DS4_KV_QUANT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/*
 * Internal KV-storage contract.
 *
 * This header deliberately describes storage surfaces rather than model
 * tensors. Qwen full K/V rows, DeepSeek raw/compressed rows, and GLM compact
 * MLA/indexer rows have different semantics; callers must name the surface
 * and may not infer one model's layout from another.
 */

typedef enum {
    DS4_KV_FAMILY_QWEN35 = 1,
    DS4_KV_FAMILY_DEEPSEEK4 = 2,
    DS4_KV_FAMILY_GLM52 = 3,
} ds4_kv_family;

typedef enum {
    DS4_KV_SURFACE_FULL_KEY = 1,
    DS4_KV_SURFACE_FULL_VALUE = 2,
    DS4_KV_SURFACE_RAW_MLA = 3,
    DS4_KV_SURFACE_COMPRESSED_ATTN = 4,
    DS4_KV_SURFACE_COMPACT_KV_LORA = 5,
    DS4_KV_SURFACE_COMPACT_K_ROPE = 6,
    DS4_KV_SURFACE_INDEXER_KEY = 7,
} ds4_kv_surface_kind;

typedef enum {
    DS4_KV_STORAGE_F32 = 1,
    DS4_KV_STORAGE_F16 = 2,
    DS4_KV_STORAGE_Q8_SCALE_F16 = 3,
    DS4_KV_STORAGE_TQ4_KEY = 4,
    DS4_KV_STORAGE_TQ4_VALUE = 5,
} ds4_kv_storage;

enum {
    /*
     * This version identifies the row layouts implemented below. It is not a
     * session snapshot version: a production snapshot owner must persist and
     * validate this identity before enabling packed caches.
     */
    DS4_KV_TQ_FORMAT_VERSION = 1,
};

typedef struct {
    ds4_kv_family family;
    ds4_kv_surface_kind kind;
    ds4_kv_storage storage;
    uint32_t layer_count;
    uint32_t capacity_rows;
    uint32_t vectors_per_row;
    uint32_t vector_dim;
} ds4_kv_surface;

typedef struct {
    uint64_t packed_data_bytes;
    uint64_t metadata_bytes;
    uint64_t total_bytes;
    uint64_t vector_stride_bytes;
    uint64_t row_stride_bytes;
} ds4_kv_surface_plan;

typedef struct {
    uint64_t packed_data_bytes;
    uint64_t metadata_bytes;
    uint64_t total_bytes;
} ds4_kv_plan_total;

bool ds4_kv_surface_plan_checked(
        const ds4_kv_surface *surface,
        ds4_kv_surface_plan  *plan);

bool ds4_kv_plan_add_checked(
        ds4_kv_plan_total         *total,
        const ds4_kv_surface_plan *surface);

size_t ds4_kv_tq4_key_bytes(uint32_t vector_dim);
size_t ds4_kv_tq4_value_bytes(uint32_t vector_dim);

/*
 * Deterministic scalar reference for the native Metal implementation.
 *
 * Keys: normalize -> deterministic signed Walsh-Hadamard rotation ->
 * Gaussian Lloyd-Max 4-bit indices, followed by an F16 vector norm.
 *
 * Values: per-vector min/max uniform 4-bit indices, followed by F16 scale and
 * minimum. The packed nibble order is low dimension first.
 *
 * Key dimensions must be powers of two. `scratch` must hold vector_dim floats.
 * The reference is for conformance tests and golden-vector generation, not a
 * CPU production fallback.
 */
bool ds4_kv_tq4_key_encode_reference(
        uint8_t       *packed,
        size_t         packed_bytes,
        const float   *input,
        uint32_t       vector_dim,
        uint32_t       seed,
        float         *scratch,
        size_t         scratch_count);

bool ds4_kv_tq4_key_decode_reference(
        float         *output,
        uint32_t       vector_dim,
        const uint8_t *packed,
        size_t         packed_bytes,
        uint32_t       seed,
        float         *scratch,
        size_t         scratch_count);

bool ds4_kv_tq4_value_encode_reference(
        uint8_t       *packed,
        size_t         packed_bytes,
        const float   *input,
        uint32_t       vector_dim);

bool ds4_kv_tq4_value_decode_reference(
        float         *output,
        uint32_t       vector_dim,
        const uint8_t *packed,
        size_t         packed_bytes);

#endif
