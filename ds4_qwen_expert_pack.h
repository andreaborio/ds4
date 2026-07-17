#ifndef DS4_QWEN_EXPERT_PACK_H
#define DS4_QWEN_EXPERT_PACK_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

enum {
    DS4_QWEN_EXPERT_PACK_SHA256_BYTES = 32,
    DS4_QWEN_EXPERT_PACK_FORMAT_VERSION = 1,
    DS4_QWEN_EXPERT_PACK_Q4_K_TYPE = 12,
};

/* Geometry is explicit at every entry point.  The production tool supplies
 * the fixed Qwen3.6 shape; tests use a small shape without weakening the
 * validator used by the runtime. */
typedef struct {
    uint32_t n_layer;
    uint32_t n_expert;
    uint32_t n_embd;
    uint32_t n_ff_exp;
    uint32_t quant_type;
    uint64_t gguf_tensor_count;
} ds4_qwen_expert_pack_geometry;

typedef struct {
    ds4_qwen_expert_pack_geometry geometry;
    uint64_t source_size;
    uint64_t entry_count;
    uint64_t gate_bytes;
    uint64_t up_bytes;
    uint64_t down_bytes;
    uint64_t data_offset;
    uint64_t data_size;
    uint64_t file_size;
    uint8_t source_sha256[DS4_QWEN_EXPERT_PACK_SHA256_BYTES];
    uint8_t data_sha256[DS4_QWEN_EXPERT_PACK_SHA256_BYTES];
} ds4_qwen_expert_pack_manifest;

typedef struct {
    uint64_t offset;
    uint64_t size;
} ds4_qwen_expert_pack_slice;

/* One lookup yields the same three-span contract a GGUF-backed loader needs.
 * The bytes are not transformed: gate, up, and down are merely adjacent in
 * the sidecar. */
typedef struct {
    ds4_qwen_expert_pack_slice gate;
    ds4_qwen_expert_pack_slice up;
    ds4_qwen_expert_pack_slice down;
} ds4_qwen_expert_pack_span;

typedef enum {
    DS4_QWEN_EXPERT_PACK_OK = 0,
    /* Missing, stale, malformed, or incompatible packs are an optimization
     * miss.  Callers must fall back to the canonical GGUF, never abort model
     * loading or use an unvalidated span. */
    DS4_QWEN_EXPERT_PACK_FALLBACK = 1,
    DS4_QWEN_EXPERT_PACK_ERROR = 2,
} ds4_qwen_expert_pack_result;

typedef struct ds4_qwen_expert_pack ds4_qwen_expert_pack;

typedef enum {
    DS4_QWEN_EXPERT_PACK_HASH_SOURCE = 0,
    DS4_QWEN_EXPERT_PACK_WRITE_DATA = 1,
    DS4_QWEN_EXPERT_PACK_VERIFY_DATA = 2,
    DS4_QWEN_EXPERT_PACK_VERIFY_SOURCE_SPANS = 3,
} ds4_qwen_expert_pack_phase;

typedef void (*ds4_qwen_expert_pack_progress_fn)(
        void                              *context,
        ds4_qwen_expert_pack_phase        phase,
        uint64_t                          completed,
        uint64_t                          total);

typedef struct {
    ds4_qwen_expert_pack_geometry geometry;
    /* Free space must cover the complete temporary pack plus this reserve.
     * Available space already excludes an old destination, so this also
     * models the peak of an atomic regeneration. */
    uint64_t filesystem_reserve_bytes;
    ds4_qwen_expert_pack_progress_fn progress;
    void *progress_context;
} ds4_qwen_expert_pack_build_options;

/* Fixed production geometry accepted by ds4's Qwen backend. */
ds4_qwen_expert_pack_geometry ds4_qwen35_expert_pack_geometry(void);

/* Build into a temporary file in the destination directory, verify every
 * copied span, fsync it, then atomically rename it.  A false return is only
 * possible before the rename, so an existing destination is unchanged.  A
 * true return may carry a non-empty durability warning when the new pack was
 * installed but the parent directory could not be fsynced. */
bool ds4_qwen_expert_pack_build(
        const char                                *gguf_path,
        const char                                *pack_path,
        const ds4_qwen_expert_pack_build_options  *options,
        char                                      *error,
        size_t                                     error_size);

/* Opening validates the format, index checksum, exact monotonic offsets, and
 * requested geometry.  It deliberately does not authorize data access until
 * the caller proves which source GGUF is in use. */
ds4_qwen_expert_pack_result ds4_qwen_expert_pack_open(
        ds4_qwen_expert_pack                 **out,
        const char                            *pack_path,
        const ds4_qwen_expert_pack_geometry   *expected_geometry,
        char                                  *error,
        size_t                                 error_size);

void ds4_qwen_expert_pack_close(ds4_qwen_expert_pack *pack);

const ds4_qwen_expert_pack_manifest *ds4_qwen_expert_pack_manifest_get(
        const ds4_qwen_expert_pack *pack);

/* A trusted hash recorded by an offline campaign avoids rereading the entire
 * GGUF immediately before a cold benchmark.  Passing the wrong size or hash
 * deauthorizes every span and returns FALLBACK. */
ds4_qwen_expert_pack_result ds4_qwen_expert_pack_validate_source_digest(
        ds4_qwen_expert_pack *pack,
        uint64_t               gguf_size,
        const uint8_t          gguf_sha256[DS4_QWEN_EXPERT_PACK_SHA256_BYTES],
        char                  *error,
        size_t                 error_size);

/* Strict convenience path: hash the source file now and compare it with the
 * manifest.  The function also rejects a source modified while hashing. */
ds4_qwen_expert_pack_result ds4_qwen_expert_pack_validate_source_file(
        ds4_qwen_expert_pack *pack,
        const char            *gguf_path,
        char                  *error,
        size_t                 error_size);

/* Rehash the packed payload.  Builders always perform this before rename; a
 * freshly opened reader must also validate it before spans are authorized.
 * Campaigns should do this offline, separately from cold/warm measurement. */
ds4_qwen_expert_pack_result ds4_qwen_expert_pack_verify_payload(
        ds4_qwen_expert_pack *pack,
        char                  *error,
        size_t                 error_size);

/* Authorize a payload from a digest established during the offline pack
 * verification step.  This is the cold-campaign counterpart of
 * verify_payload(): it never reads the packed weight bytes, but it still
 * requires an exact match with the digest protected by the pack header's
 * index checksum. */
ds4_qwen_expert_pack_result ds4_qwen_expert_pack_validate_payload_digest(
        ds4_qwen_expert_pack *pack,
        const uint8_t         payload_sha256[DS4_QWEN_EXPERT_PACK_SHA256_BYTES],
        char                 *error,
        size_t                error_size);

/* Returns false until both source identity and payload validation succeed.
 * This makes a stale or damaged pack a structural fallback rather than a path
 * from which wrong weights can leak. */
bool ds4_qwen_expert_pack_span_get(
        const ds4_qwen_expert_pack *pack,
        uint32_t                    layer,
        uint32_t                    expert,
        ds4_qwen_expert_pack_span  *span);

int ds4_qwen_expert_pack_fd(const ds4_qwen_expert_pack *pack);

#endif
