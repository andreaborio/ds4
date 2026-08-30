#ifndef DS4_PLE_STORE_H
#define DS4_PLE_STORE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define DS4_PLE_STORE_V1_TENSOR "ds4.ple_rows.v1"
#define DS4_PLE_STORE_V1_PROFILE_ID "qwen4exp-base-v1"
#define DS4_PLE_STORE_V1_HASH_ID "SplitMix64-Qwen4Exp-v1"

enum {
    DS4_PLE_STORE_V1_VERSION = 1,
    DS4_PLE_STORE_FAMILY_QWEN4EXP = 4,
    DS4_PLE_STORE_V1_HEADS = 16,
    DS4_PLE_STORE_V1_SHA256_BYTES = 32,
    DS4_PLE_STORE_V1_ID_BYTES = 32,
    DS4_PLE_STORE_V1_HEADER_BYTES = 512,
    DS4_PLE_STORE_V1_PAGE_HEADER_BYTES = 64,
    DS4_PLE_STORE_V1_MIN_PAGE_ALIGNMENT = 4096,
};

/* Codec identity is deliberately supplied by the caller.  PLE v1 freezes the
 * structural page format, not an unqualified release codec. */
typedef struct {
    const char *id;
    uint32_t version;
    uint32_t group_size;
    uint32_t encoded_row_bytes;
} ds4_ple_store_codec;

/* Logical and page geometry is explicit so miniature converter fixtures do
 * not need to pretend they contain the production 320,001,536-row table. */
typedef struct {
    uint64_t row_count;
    uint32_t row_width;
    uint32_t row_alignment;
    uint32_t rows_per_page;
    uint32_t page_alignment;
    uint32_t head_prime[DS4_PLE_STORE_V1_HEADS];
    uint32_t head_offset[DS4_PLE_STORE_V1_HEADS];
} ds4_ple_store_geometry;

typedef struct {
    uint32_t version;
    uint32_t family;
    char profile_id[DS4_PLE_STORE_V1_ID_BYTES + 1];
    char hash_id[DS4_PLE_STORE_V1_ID_BYTES + 1];
    char codec_id[DS4_PLE_STORE_V1_ID_BYTES + 1];
    uint32_t codec_version;
    uint32_t codec_group_size;
    uint32_t encoded_row_bytes;
    uint64_t row_count;
    uint32_t row_width;
    uint32_t row_alignment;
    uint32_t head_count;
    uint32_t head_prime[DS4_PLE_STORE_V1_HEADS];
    uint32_t head_offset[DS4_PLE_STORE_V1_HEADS];
    uint32_t rows_per_page;
    uint32_t page_alignment;
    uint64_t page_count;
    uint64_t page_stride;
    uint64_t page_digest_offset;
    uint64_t page_digest_bytes;
    uint64_t payload_offset;
    uint64_t payload_bytes;
    uint64_t store_bytes;
    uint8_t payload_sha256[DS4_PLE_STORE_V1_SHA256_BYTES];
    uint8_t manifest_sha256[DS4_PLE_STORE_V1_SHA256_BYTES];
} ds4_ple_store_manifest;

typedef struct ds4_ple_store ds4_ple_store;

/* Validates descriptor invariants and all derived fixed-page extent arithmetic
 * without opening or allocating the described payload. */
bool ds4_ple_store_descriptor_validate(
        const ds4_ple_store_geometry *geometry,
        const ds4_ple_store_codec    *codec,
        char                         *error,
        size_t                        error_size);

/* Opens one complete PLE v1 extent embedded at [offset, offset + bytes).
 * Opening verifies all closed identities, caller-provided geometry/codec,
 * checked extents, and the manifest digest without touching the bulk payload.
 * Use verify_page/verify_all before consuming untrusted payload bytes. */
bool ds4_ple_store_open_embedded(
        ds4_ple_store                **out,
        int                            fd,
        uint64_t                       offset,
        uint64_t                       bytes,
        const ds4_ple_store_geometry  *expected_geometry,
        const ds4_ple_store_codec     *expected_codec,
        char                          *error,
        size_t                         error_size);

void ds4_ple_store_close(ds4_ple_store *store);

const ds4_ple_store_manifest *ds4_ple_store_manifest_get(
        const ds4_ple_store *store);

int ds4_ple_store_fd(const ds4_ple_store *store);
uint64_t ds4_ple_store_file_offset(const ds4_ple_store *store);

/* Returns the absolute file offset of the complete fixed page containing row,
 * its slot, and its page number.  Outputs are cleared on failure. */
bool ds4_ple_store_locate_row(
        const ds4_ple_store *store,
        uint64_t             row,
        uint64_t            *page_offset,
        uint32_t            *slot,
        uint64_t            *page);

bool ds4_ple_store_verify_page(
        const ds4_ple_store *store,
        uint64_t             page,
        char                *error,
        size_t               error_size);

/* Publication/offline gate: verifies every page digest and the whole physical
 * payload digest.  It streams fixed-size buffers and allocates no page index. */
bool ds4_ple_store_verify_all(
        const ds4_ple_store *store,
        char                *error,
        size_t               error_size);

/* Reads and verifies one immutable page snapshot, then copies the encoded row.
 * The caller's output is unchanged on every failure. */
bool ds4_ple_store_read_row(
        const ds4_ple_store *store,
        uint64_t             row,
        void                *encoded,
        size_t               encoded_size,
        char                *error,
        size_t               error_size);

typedef bool (*ds4_ple_store_encode_row_fn)(
        void     *context,
        uint64_t  row,
        uint8_t  *encoded,
        uint32_t  encoded_bytes);

typedef enum {
    DS4_PLE_STORE_SYNC_TEMP_FILE = 1,
    DS4_PLE_STORE_SYNC_PARENT_BEFORE_RENAME = 2,
    DS4_PLE_STORE_SYNC_PARENT_AFTER_RENAME = 3,
} ds4_ple_store_sync_phase;

/* Optional narrow I/O seam for deterministic converter failure injection.
 * Production callers use ds4_ple_store_write_atomic and the system fsync. */
typedef int (*ds4_ple_store_sync_fn)(
        void                     *context,
        int                       fd,
        ds4_ple_store_sync_phase  phase);

typedef struct {
    ds4_ple_store_sync_fn sync;
    void *context;
} ds4_ple_store_writer_ops;

/* Builds a complete standalone PLE extent in a sibling temporary file, fsyncs
 * it, reopens it through the runtime parser, verifies payload and boundary
 * pages, then atomically renames it over target_path.  Any failure before the
 * rename removes the temporary file and leaves an existing target untouched.
 * Successful rename is the commit point.  A later directory-fsync error is
 * reported but never attempts an unsafe rollback of the complete new target. */
bool ds4_ple_store_write_atomic(
        const char                    *target_path,
        const ds4_ple_store_geometry  *geometry,
        const ds4_ple_store_codec     *codec,
        ds4_ple_store_encode_row_fn    encode_row,
        void                          *encode_context,
        char                          *error,
        size_t                         error_size);

bool ds4_ple_store_write_atomic_with_ops(
        const char                     *target_path,
        const ds4_ple_store_geometry   *geometry,
        const ds4_ple_store_codec      *codec,
        ds4_ple_store_encode_row_fn     encode_row,
        void                           *encode_context,
        const ds4_ple_store_writer_ops *ops,
        char                           *error,
        size_t                          error_size);

#endif
