#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include "ds4_ple_store.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

enum {
    HEADER_VERSION_OFFSET = 8,
    HEADER_BYTES_OFFSET = 12,
    HEADER_FAMILY_OFFSET = 16,
    HEADER_HEAD_COUNT_OFFSET = 20,
    HEADER_PROFILE_OFFSET = 24,
    HEADER_HASH_OFFSET = 56,
    HEADER_CODEC_OFFSET = 88,
    HEADER_CODEC_VERSION_OFFSET = 120,
    HEADER_CODEC_GROUP_OFFSET = 124,
    HEADER_ROW_BYTES_OFFSET = 128,
    HEADER_ROWS_PER_PAGE_OFFSET = 132,
    HEADER_PAGE_ALIGNMENT_OFFSET = 136,
    HEADER_PAGE_HEADER_BYTES_OFFSET = 140,
    HEADER_ROW_COUNT_OFFSET = 144,
    HEADER_ROW_WIDTH_OFFSET = 152,
    HEADER_ROW_ALIGNMENT_OFFSET = 156,
    HEADER_PAGE_COUNT_OFFSET = 160,
    HEADER_PAGE_STRIDE_OFFSET = 168,
    HEADER_PAGE_DIGEST_OFFSET_OFFSET = 176,
    HEADER_PAGE_DIGEST_BYTES_OFFSET = 184,
    HEADER_PAYLOAD_OFFSET_OFFSET = 192,
    HEADER_PAYLOAD_BYTES_OFFSET = 200,
    HEADER_STORE_BYTES_OFFSET = 208,
    HEADER_HEAD_PRIME_OFFSET = 216,
    HEADER_HEAD_OFFSET_OFFSET = 280,
    HEADER_PAYLOAD_DIGEST_OFFSET = 344,
    HEADER_MANIFEST_DIGEST_OFFSET = 376,
    HEADER_RESERVED_OFFSET = 408,
    PAGE_VERSION_OFFSET = 8,
    PAGE_HEADER_BYTES_OFFSET = 12,
    PAGE_INDEX_OFFSET = 16,
    PAGE_FIRST_ROW_OFFSET = 24,
    PAGE_ROW_COUNT_OFFSET = 32,
    PAGE_ROW_WIDTH_OFFSET = 36,
    PAGE_ROW_BYTES_OFFSET = 40,
    PAGE_ROWS_PER_PAGE_OFFSET = 44,
    PAGE_CODEC_VERSION_OFFSET = 48,
    PAGE_CODEC_GROUP_OFFSET = 52,
    PAGE_FAMILY_OFFSET = 56,
    PAGE_RESERVED_OFFSET = 60,
    IO_CHUNK_BYTES = 64 * 1024,
};

static const uint8_t store_magic[8] = {
    'D', 'S', '4', 'P', 'L', 'E', 'V', '1'
};

static const uint8_t page_magic[8] = {
    'D', 'S', '4', 'P', 'L', 'P', 'G', '1'
};

typedef struct {
    uint64_t page_count;
    uint64_t page_stride;
    uint64_t digest_offset;
    uint64_t digest_bytes;
    uint64_t payload_offset;
    uint64_t payload_bytes;
    uint64_t store_bytes;
} ple_layout;

typedef struct {
    uint32_t state[8];
    uint64_t bytes;
    uint8_t block[64];
    size_t block_len;
} sha256_context;

struct ds4_ple_store {
    int fd;
    uint64_t file_offset;
    uint64_t extent_size;
    ds4_ple_store_manifest manifest;
};

static void set_error(char *error, size_t size, const char *fmt, ...) {
    if (!error || size == 0) return;
    va_list args;
    va_start(args, fmt);
    vsnprintf(error, size, fmt, args);
    va_end(args);
}

static uint32_t load_u32_le(const uint8_t *p) {
    return (uint32_t)p[0] |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

static uint64_t load_u64_le(const uint8_t *p) {
    return (uint64_t)load_u32_le(p) |
           ((uint64_t)load_u32_le(p + 4) << 32);
}

static void store_u32_le(uint8_t *p, uint32_t value) {
    p[0] = (uint8_t)value;
    p[1] = (uint8_t)(value >> 8);
    p[2] = (uint8_t)(value >> 16);
    p[3] = (uint8_t)(value >> 24);
}

static void store_u64_le(uint8_t *p, uint64_t value) {
    store_u32_le(p, (uint32_t)value);
    store_u32_le(p + 4, (uint32_t)(value >> 32));
}

static bool add_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out || a > UINT64_MAX - b) return false;
    *out = a + b;
    return true;
}

static bool mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out || (a != 0 && b > UINT64_MAX / a)) return false;
    *out = a * b;
    return true;
}

static bool align_u64(uint64_t value, uint32_t alignment, uint64_t *out) {
    if (!out || alignment == 0 || (alignment & (alignment - 1u)) != 0) {
        return false;
    }
    const uint64_t mask = (uint64_t)alignment - 1u;
    if (value > UINT64_MAX - mask) return false;
    *out = (value + mask) & ~mask;
    return true;
}

static bool all_zero(const uint8_t *data, size_t size) {
    for (size_t i = 0; i < size; i++) {
        if (data[i] != 0) return false;
    }
    return true;
}

static bool pread_all(int fd, void *dst_pointer, size_t size, uint64_t offset) {
    uint8_t *dst = dst_pointer;
    while (size != 0) {
        const ssize_t got = pread(fd, dst, size, (off_t)offset);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) return false;
        dst += (size_t)got;
        size -= (size_t)got;
        offset += (uint64_t)got;
    }
    return true;
}

static bool pwrite_all(int fd, const void *src_pointer, size_t size,
                       uint64_t offset) {
    const uint8_t *src = src_pointer;
    while (size != 0) {
        const ssize_t put = pwrite(fd, src, size, (off_t)offset);
        if (put < 0 && errno == EINTR) continue;
        if (put <= 0) return false;
        src += (size_t)put;
        size -= (size_t)put;
        offset += (uint64_t)put;
    }
    return true;
}

static uint32_t rotr32(uint32_t value, unsigned shift) {
    return (value >> shift) | (value << (32u - shift));
}

static void sha256_transform(sha256_context *context,
                             const uint8_t block[64]) {
    static const uint32_t k[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
        0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
        0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
        0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
        0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
        0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
        0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
        0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
    };
    uint32_t w[64];
    for (size_t i = 0; i < 16; i++) {
        w[i] = ((uint32_t)block[i * 4] << 24) |
               ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) |
               (uint32_t)block[i * 4 + 3];
    }
    for (size_t i = 16; i < 64; i++) {
        const uint32_t s0 = rotr32(w[i - 15], 7) ^
                            rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = rotr32(w[i - 2], 17) ^
                            rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = context->state[0], b = context->state[1];
    uint32_t c = context->state[2], d = context->state[3];
    uint32_t e = context->state[4], f = context->state[5];
    uint32_t g = context->state[6], h = context->state[7];
    for (size_t i = 0; i < 64; i++) {
        const uint32_t s1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        const uint32_t ch = (e & f) ^ (~e & g);
        const uint32_t t1 = h + s1 + ch + k[i] + w[i];
        const uint32_t s0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t t2 = s0 + maj;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    context->state[0] += a; context->state[1] += b;
    context->state[2] += c; context->state[3] += d;
    context->state[4] += e; context->state[5] += f;
    context->state[6] += g; context->state[7] += h;
}

static void sha256_init(sha256_context *context) {
    *context = (sha256_context){
        .state = {
            0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
            0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
        },
    };
}

static void sha256_update(sha256_context *context, const void *data_pointer,
                          size_t size) {
    const uint8_t *data = data_pointer;
    context->bytes += size;
    while (size != 0) {
        size_t take = sizeof(context->block) - context->block_len;
        if (take > size) take = size;
        memcpy(context->block + context->block_len, data, take);
        context->block_len += take;
        data += take;
        size -= take;
        if (context->block_len == sizeof(context->block)) {
            sha256_transform(context, context->block);
            context->block_len = 0;
        }
    }
}

static void sha256_final(sha256_context *context, uint8_t digest[32]) {
    const uint64_t bits = context->bytes * UINT64_C(8);
    context->block[context->block_len++] = 0x80;
    if (context->block_len > 56) {
        memset(context->block + context->block_len, 0,
               sizeof(context->block) - context->block_len);
        sha256_transform(context, context->block);
        context->block_len = 0;
    }
    memset(context->block + context->block_len, 0, 56 - context->block_len);
    for (size_t i = 0; i < 8; i++) {
        context->block[56 + i] = (uint8_t)(bits >> (56 - i * 8));
    }
    sha256_transform(context, context->block);
    for (size_t i = 0; i < 8; i++) {
        digest[i * 4] = (uint8_t)(context->state[i] >> 24);
        digest[i * 4 + 1] = (uint8_t)(context->state[i] >> 16);
        digest[i * 4 + 2] = (uint8_t)(context->state[i] >> 8);
        digest[i * 4 + 3] = (uint8_t)context->state[i];
    }
}

static bool wire_id_matches(const uint8_t wire[DS4_PLE_STORE_V1_ID_BYTES],
                            const char *expected) {
    if (!expected) return false;
    const size_t length = strlen(expected);
    return length != 0 && length < DS4_PLE_STORE_V1_ID_BYTES &&
           memcmp(wire, expected, length) == 0 && wire[length] == 0 &&
           all_zero(wire + length + 1,
                    DS4_PLE_STORE_V1_ID_BYTES - length - 1);
}

static bool store_wire_id(uint8_t wire[DS4_PLE_STORE_V1_ID_BYTES],
                          const char *value) {
    if (!value) return false;
    const size_t length = strlen(value);
    if (length == 0 || length >= DS4_PLE_STORE_V1_ID_BYTES) return false;
    memset(wire, 0, DS4_PLE_STORE_V1_ID_BYTES);
    memcpy(wire, value, length);
    return true;
}

static bool geometry_valid(const ds4_ple_store_geometry *geometry,
                           const ds4_ple_store_codec *codec,
                           char *error, size_t error_size) {
    if (!geometry || !codec || !codec->id || geometry->row_count == 0 ||
        geometry->row_width == 0 || geometry->rows_per_page == 0 ||
        codec->version == 0 || codec->group_size == 0 ||
        codec->encoded_row_bytes == 0) {
        set_error(error, error_size, "invalid PLE geometry or codec descriptor");
        return false;
    }
    const size_t codec_length = strlen(codec->id);
    if (codec_length == 0 || codec_length >= DS4_PLE_STORE_V1_ID_BYTES) {
        set_error(error, error_size, "invalid PLE codec identifier");
        return false;
    }
    if (geometry->row_alignment == 0 ||
        (geometry->row_alignment & (geometry->row_alignment - 1u)) != 0) {
        set_error(error, error_size,
                  "PLE logical row alignment must be a power of two");
        return false;
    }
    if (geometry->page_alignment < DS4_PLE_STORE_V1_MIN_PAGE_ALIGNMENT ||
        (geometry->page_alignment & (geometry->page_alignment - 1u)) != 0) {
        set_error(error, error_size,
                  "PLE page alignment must be a power of two >= %u",
                  DS4_PLE_STORE_V1_MIN_PAGE_ALIGNMENT);
        return false;
    }
    if (geometry->head_offset[0] != 0) {
        set_error(error, error_size, "PLE head offset 0 must be zero");
        return false;
    }
    for (size_t head = 0; head < DS4_PLE_STORE_V1_HEADS; head++) {
        if (geometry->head_prime[head] == 0) {
            set_error(error, error_size, "PLE head prime %zu is zero", head);
            return false;
        }
        if (head != 0) {
            const uint64_t expected =
                (uint64_t)geometry->head_offset[head - 1] +
                geometry->head_prime[head - 1];
            if (expected > UINT32_MAX ||
                geometry->head_offset[head] != (uint32_t)expected) {
                set_error(error, error_size,
                          "PLE head offset %zu is not cumulative", head);
                return false;
            }
        }
    }
    uint64_t segment_end = 0;
    uint64_t padded_end = 0;
    if (!add_u64(geometry->head_offset[DS4_PLE_STORE_V1_HEADS - 1],
                 geometry->head_prime[DS4_PLE_STORE_V1_HEADS - 1],
                 &segment_end) ||
        segment_end > geometry->row_count ||
        !align_u64(segment_end, geometry->row_alignment, &padded_end) ||
        padded_end != geometry->row_count) {
        set_error(error, error_size,
                  "PLE row count is not the aligned 16-head segment extent");
        return false;
    }
    return true;
}

static bool compute_layout(const ds4_ple_store_geometry *geometry,
                           const ds4_ple_store_codec *codec,
                           ple_layout *layout,
                           char *error, size_t error_size) {
    if (!layout || !geometry_valid(geometry, codec, error, error_size)) {
        return false;
    }
    uint64_t row_payload = 0;
    uint64_t minimum_stride = 0;
    uint64_t digest_end = 0;
    if (!mul_u64(geometry->rows_per_page, codec->encoded_row_bytes,
                 &row_payload) ||
        !add_u64(DS4_PLE_STORE_V1_PAGE_HEADER_BYTES, row_payload,
                 &minimum_stride) ||
        !align_u64(minimum_stride, geometry->page_alignment,
                   &layout->page_stride)) {
        set_error(error, error_size, "PLE page-stride arithmetic overflow");
        return false;
    }
    layout->page_count =
        UINT64_C(1) + (geometry->row_count - UINT64_C(1)) /
                         geometry->rows_per_page;
    layout->digest_offset = DS4_PLE_STORE_V1_HEADER_BYTES;
    if (!mul_u64(layout->page_count, DS4_PLE_STORE_V1_SHA256_BYTES,
                 &layout->digest_bytes) ||
        !add_u64(layout->digest_offset, layout->digest_bytes, &digest_end) ||
        !align_u64(digest_end, geometry->page_alignment,
                   &layout->payload_offset) ||
        !mul_u64(layout->page_count, layout->page_stride,
                 &layout->payload_bytes) ||
        !add_u64(layout->payload_offset, layout->payload_bytes,
                 &layout->store_bytes) ||
        layout->store_bytes > (uint64_t)INT64_MAX) {
        set_error(error, error_size, "PLE store extent arithmetic overflow");
        return false;
    }
    return true;
}

bool ds4_ple_store_descriptor_validate(
        const ds4_ple_store_geometry *geometry,
        const ds4_ple_store_codec *codec,
        char *error, size_t error_size) {
    if (error && error_size != 0) error[0] = '\0';
    ple_layout layout;
    return compute_layout(geometry, codec, &layout, error, error_size);
}

static bool hash_file_region(int fd, uint64_t offset, uint64_t bytes,
                             sha256_context *hash) {
    uint8_t buffer[IO_CHUNK_BYTES];
    while (bytes != 0) {
        size_t take = sizeof(buffer);
        if ((uint64_t)take > bytes) take = (size_t)bytes;
        if (!pread_all(fd, buffer, take, offset)) return false;
        sha256_update(hash, buffer, take);
        offset += take;
        bytes -= take;
    }
    return true;
}

static bool page_relative_offset(const ds4_ple_store_manifest *manifest,
                                 uint64_t page, uint64_t *offset) {
    uint64_t displacement = 0;
    uint64_t end = 0;
    return manifest && offset && page < manifest->page_count &&
           mul_u64(page, manifest->page_stride, &displacement) &&
           add_u64(manifest->payload_offset, displacement, offset) &&
           add_u64(*offset, manifest->page_stride, &end) &&
           end <= manifest->store_bytes;
}

static bool validate_page_header(const ds4_ple_store *store, uint64_t page,
                                 const uint8_t header[64], char *error,
                                 size_t error_size) {
    const ds4_ple_store_manifest *manifest = &store->manifest;
    uint64_t first_row = 0;
    if (!mul_u64(page, manifest->rows_per_page, &first_row)) {
        set_error(error, error_size, "PLE page %" PRIu64 " row overflow", page);
        return false;
    }
    const uint64_t remaining = manifest->row_count - first_row;
    const uint32_t row_count = remaining < manifest->rows_per_page
        ? (uint32_t)remaining : manifest->rows_per_page;
    if (memcmp(header, page_magic, sizeof(page_magic)) != 0 ||
        load_u32_le(header + PAGE_VERSION_OFFSET) != DS4_PLE_STORE_V1_VERSION ||
        load_u32_le(header + PAGE_HEADER_BYTES_OFFSET) !=
            DS4_PLE_STORE_V1_PAGE_HEADER_BYTES ||
        load_u64_le(header + PAGE_INDEX_OFFSET) != page ||
        load_u64_le(header + PAGE_FIRST_ROW_OFFSET) != first_row ||
        load_u32_le(header + PAGE_ROW_COUNT_OFFSET) != row_count ||
        load_u32_le(header + PAGE_ROW_WIDTH_OFFSET) != manifest->row_width ||
        load_u32_le(header + PAGE_ROW_BYTES_OFFSET) !=
            manifest->encoded_row_bytes ||
        load_u32_le(header + PAGE_ROWS_PER_PAGE_OFFSET) !=
            manifest->rows_per_page ||
        load_u32_le(header + PAGE_CODEC_VERSION_OFFSET) !=
            manifest->codec_version ||
        load_u32_le(header + PAGE_CODEC_GROUP_OFFSET) !=
            manifest->codec_group_size ||
        load_u32_le(header + PAGE_FAMILY_OFFSET) != manifest->family ||
        load_u32_le(header + PAGE_RESERVED_OFFSET) != 0) {
        set_error(error, error_size,
                  "PLE page %" PRIu64 " duplicated geometry is invalid", page);
        return false;
    }
    return true;
}

bool ds4_ple_store_open_embedded(
        ds4_ple_store **out, int fd, uint64_t offset, uint64_t bytes,
        const ds4_ple_store_geometry *expected_geometry,
        const ds4_ple_store_codec *expected_codec,
        char *error, size_t error_size) {
    if (out) *out = NULL;
    if (error && error_size != 0) error[0] = '\0';
    if (!out || fd < 0 || bytes < DS4_PLE_STORE_V1_HEADER_BYTES ||
        offset > (uint64_t)INT64_MAX || bytes > (uint64_t)INT64_MAX - offset ||
        !geometry_valid(expected_geometry, expected_codec, error, error_size)) {
        if (!error || error_size == 0 || error[0] == '\0') {
            set_error(error, error_size, "invalid embedded PLE-store extent");
        }
        return false;
    }
    struct stat status;
    if (fstat(fd, &status) != 0) {
        set_error(error, error_size,
                  "cannot inspect the underlying PLE file: %s",
                  strerror(errno));
        return false;
    }
    if (!S_ISREG(status.st_mode)) {
        set_error(error, error_size,
                  "the underlying PLE owner must be a regular file");
        return false;
    }
    if (status.st_size < 0 || offset + bytes > (uint64_t)status.st_size) {
        set_error(error, error_size,
                  "PLE owner extent exceeds the underlying file");
        return false;
    }
    uint8_t header[DS4_PLE_STORE_V1_HEADER_BYTES];
    if (!pread_all(fd, header, sizeof(header), offset)) {
        set_error(error, error_size, "cannot read PLE manifest: %s",
                  strerror(errno));
        return false;
    }
    ple_layout expected_layout;
    if (!compute_layout(expected_geometry, expected_codec, &expected_layout,
                        error, error_size)) {
        return false;
    }
    if (offset % expected_geometry->page_alignment != 0) {
        set_error(error, error_size,
                  "PLE owner extent is not page-aligned");
        return false;
    }
    if (memcmp(header, store_magic, sizeof(store_magic)) != 0) {
        set_error(error, error_size, "PLE manifest magic mismatch");
        return false;
    }
    if (load_u32_le(header + HEADER_VERSION_OFFSET) !=
            DS4_PLE_STORE_V1_VERSION ||
        load_u32_le(header + HEADER_BYTES_OFFSET) !=
            DS4_PLE_STORE_V1_HEADER_BYTES) {
        set_error(error, error_size, "PLE manifest version/header mismatch");
        return false;
    }
    if (load_u32_le(header + HEADER_FAMILY_OFFSET) !=
            DS4_PLE_STORE_FAMILY_QWEN4EXP ||
        load_u32_le(header + HEADER_HEAD_COUNT_OFFSET) !=
            DS4_PLE_STORE_V1_HEADS ||
        !wire_id_matches(header + HEADER_PROFILE_OFFSET,
                         DS4_PLE_STORE_V1_PROFILE_ID) ||
        !wire_id_matches(header + HEADER_HASH_OFFSET,
                         DS4_PLE_STORE_V1_HASH_ID)) {
        set_error(error, error_size,
                  "PLE family/profile/hash identity mismatch");
        return false;
    }
    if (!wire_id_matches(header + HEADER_CODEC_OFFSET, expected_codec->id) ||
        load_u32_le(header + HEADER_CODEC_VERSION_OFFSET) !=
            expected_codec->version ||
        load_u32_le(header + HEADER_CODEC_GROUP_OFFSET) !=
            expected_codec->group_size ||
        load_u32_le(header + HEADER_ROW_BYTES_OFFSET) !=
            expected_codec->encoded_row_bytes) {
        set_error(error, error_size, "PLE codec descriptor mismatch");
        return false;
    }
    if (load_u64_le(header + HEADER_ROW_COUNT_OFFSET) !=
            expected_geometry->row_count ||
        load_u32_le(header + HEADER_ROW_WIDTH_OFFSET) !=
            expected_geometry->row_width ||
        load_u32_le(header + HEADER_ROW_ALIGNMENT_OFFSET) !=
            expected_geometry->row_alignment ||
        load_u32_le(header + HEADER_ROWS_PER_PAGE_OFFSET) !=
            expected_geometry->rows_per_page ||
        load_u32_le(header + HEADER_PAGE_ALIGNMENT_OFFSET) !=
            expected_geometry->page_alignment ||
        load_u32_le(header + HEADER_PAGE_HEADER_BYTES_OFFSET) !=
            DS4_PLE_STORE_V1_PAGE_HEADER_BYTES) {
        set_error(error, error_size, "PLE logical/page geometry mismatch");
        return false;
    }
    for (size_t head = 0; head < DS4_PLE_STORE_V1_HEADS; head++) {
        if (load_u32_le(header + HEADER_HEAD_PRIME_OFFSET + head * 4) !=
                expected_geometry->head_prime[head] ||
            load_u32_le(header + HEADER_HEAD_OFFSET_OFFSET + head * 4) !=
                expected_geometry->head_offset[head]) {
            set_error(error, error_size,
                      "PLE head prime/offset mismatch at head %zu", head);
            return false;
        }
    }
    if (!all_zero(header + HEADER_RESERVED_OFFSET,
                  sizeof(header) - HEADER_RESERVED_OFFSET)) {
        set_error(error, error_size, "PLE manifest reserved fields are nonzero");
        return false;
    }
    if (load_u64_le(header + HEADER_PAGE_COUNT_OFFSET) !=
            expected_layout.page_count ||
        load_u64_le(header + HEADER_PAGE_STRIDE_OFFSET) !=
            expected_layout.page_stride ||
        load_u64_le(header + HEADER_PAGE_DIGEST_OFFSET_OFFSET) !=
            expected_layout.digest_offset ||
        load_u64_le(header + HEADER_PAGE_DIGEST_BYTES_OFFSET) !=
            expected_layout.digest_bytes ||
        load_u64_le(header + HEADER_PAYLOAD_OFFSET_OFFSET) !=
            expected_layout.payload_offset ||
        load_u64_le(header + HEADER_PAYLOAD_BYTES_OFFSET) !=
            expected_layout.payload_bytes ||
        load_u64_le(header + HEADER_STORE_BYTES_OFFSET) !=
            expected_layout.store_bytes ||
        bytes != expected_layout.store_bytes) {
        set_error(error, error_size, "PLE checked physical extent mismatch");
        return false;
    }

    uint8_t digest_header[DS4_PLE_STORE_V1_HEADER_BYTES];
    memcpy(digest_header, header, sizeof(digest_header));
    memset(digest_header + HEADER_MANIFEST_DIGEST_OFFSET, 0,
           DS4_PLE_STORE_V1_SHA256_BYTES);
    sha256_context manifest_hash;
    uint8_t manifest_digest[DS4_PLE_STORE_V1_SHA256_BYTES];
    sha256_init(&manifest_hash);
    sha256_update(&manifest_hash, digest_header, sizeof(digest_header));
    if (!hash_file_region(fd, offset + DS4_PLE_STORE_V1_HEADER_BYTES,
                          expected_layout.payload_offset -
                              DS4_PLE_STORE_V1_HEADER_BYTES,
                          &manifest_hash)) {
        set_error(error, error_size, "cannot hash PLE manifest extent: %s",
                  strerror(errno));
        return false;
    }
    sha256_final(&manifest_hash, manifest_digest);
    if (memcmp(manifest_digest, header + HEADER_MANIFEST_DIGEST_OFFSET,
               sizeof(manifest_digest)) != 0) {
        set_error(error, error_size, "PLE manifest SHA-256 mismatch");
        return false;
    }

    int owned_fd;
#ifdef F_DUPFD_CLOEXEC
    owned_fd = fcntl(fd, F_DUPFD_CLOEXEC, 0);
#else
    owned_fd = dup(fd);
#endif
    if (owned_fd < 0) {
        set_error(error, error_size, "cannot duplicate PLE-store fd: %s",
                  strerror(errno));
        return false;
    }
    ds4_ple_store *store = calloc(1, sizeof(*store));
    if (!store) {
        close(owned_fd);
        set_error(error, error_size, "out of memory opening PLE store");
        return false;
    }
    store->fd = owned_fd;
    store->file_offset = offset;
    store->extent_size = bytes;
    ds4_ple_store_manifest *manifest = &store->manifest;
    manifest->version = DS4_PLE_STORE_V1_VERSION;
    manifest->family = DS4_PLE_STORE_FAMILY_QWEN4EXP;
    memcpy(manifest->profile_id, DS4_PLE_STORE_V1_PROFILE_ID,
           sizeof(DS4_PLE_STORE_V1_PROFILE_ID));
    memcpy(manifest->hash_id, DS4_PLE_STORE_V1_HASH_ID,
           sizeof(DS4_PLE_STORE_V1_HASH_ID));
    memcpy(manifest->codec_id, expected_codec->id,
           strlen(expected_codec->id) + 1);
    manifest->codec_version = expected_codec->version;
    manifest->codec_group_size = expected_codec->group_size;
    manifest->encoded_row_bytes = expected_codec->encoded_row_bytes;
    manifest->row_count = expected_geometry->row_count;
    manifest->row_width = expected_geometry->row_width;
    manifest->row_alignment = expected_geometry->row_alignment;
    manifest->head_count = DS4_PLE_STORE_V1_HEADS;
    memcpy(manifest->head_prime, expected_geometry->head_prime,
           sizeof(manifest->head_prime));
    memcpy(manifest->head_offset, expected_geometry->head_offset,
           sizeof(manifest->head_offset));
    manifest->rows_per_page = expected_geometry->rows_per_page;
    manifest->page_alignment = expected_geometry->page_alignment;
    manifest->page_count = expected_layout.page_count;
    manifest->page_stride = expected_layout.page_stride;
    manifest->page_digest_offset = expected_layout.digest_offset;
    manifest->page_digest_bytes = expected_layout.digest_bytes;
    manifest->payload_offset = expected_layout.payload_offset;
    manifest->payload_bytes = expected_layout.payload_bytes;
    manifest->store_bytes = expected_layout.store_bytes;
    memcpy(manifest->payload_sha256, header + HEADER_PAYLOAD_DIGEST_OFFSET,
           DS4_PLE_STORE_V1_SHA256_BYTES);
    memcpy(manifest->manifest_sha256, header + HEADER_MANIFEST_DIGEST_OFFSET,
           DS4_PLE_STORE_V1_SHA256_BYTES);
    *out = store;
    return true;
}

void ds4_ple_store_close(ds4_ple_store *store) {
    if (!store) return;
    if (store->fd >= 0) close(store->fd);
    free(store);
}

const ds4_ple_store_manifest *ds4_ple_store_manifest_get(
        const ds4_ple_store *store) {
    return store ? &store->manifest : NULL;
}

int ds4_ple_store_fd(const ds4_ple_store *store) {
    return store ? store->fd : -1;
}

uint64_t ds4_ple_store_file_offset(const ds4_ple_store *store) {
    return store ? store->file_offset : 0;
}

bool ds4_ple_store_locate_row(const ds4_ple_store *store, uint64_t row,
                              uint64_t *page_offset, uint32_t *slot,
                              uint64_t *page) {
    if (page_offset) *page_offset = 0;
    if (slot) *slot = 0;
    if (page) *page = 0;
    if (!store || !page_offset || !slot || !page ||
        row >= store->manifest.row_count) {
        return false;
    }
    const uint64_t local_page = row / store->manifest.rows_per_page;
    uint64_t relative = 0;
    uint64_t absolute = 0;
    if (!page_relative_offset(&store->manifest, local_page, &relative) ||
        !add_u64(store->file_offset, relative, &absolute)) {
        return false;
    }
    *page_offset = absolute;
    *slot = (uint32_t)(row % store->manifest.rows_per_page);
    *page = local_page;
    return true;
}

static bool read_page_digest(const ds4_ple_store *store, uint64_t page,
                             uint8_t digest[32]) {
    uint64_t displacement = 0;
    uint64_t relative = 0;
    uint64_t absolute = 0;
    return mul_u64(page, DS4_PLE_STORE_V1_SHA256_BYTES, &displacement) &&
           add_u64(store->manifest.page_digest_offset, displacement,
                   &relative) &&
           add_u64(store->file_offset, relative, &absolute) &&
           pread_all(store->fd, digest, DS4_PLE_STORE_V1_SHA256_BYTES,
                     absolute);
}

static bool read_verified_page(const ds4_ple_store *store, uint64_t page,
                               uint8_t **page_out,
                               char *error, size_t error_size) {
    if (page_out) *page_out = NULL;
    if (!store || page >= store->manifest.page_count) {
        set_error(error, error_size, "PLE page index is out of range");
        return false;
    }
    if (!page_out || store->manifest.page_stride > SIZE_MAX) {
        set_error(error, error_size, "PLE page exceeds the address space");
        return false;
    }
    uint64_t relative = 0;
    uint64_t absolute = 0;
    if (!page_relative_offset(&store->manifest, page, &relative) ||
        !add_u64(store->file_offset, relative, &absolute)) {
        set_error(error, error_size, "PLE page extent overflow");
        return false;
    }
    uint8_t *snapshot = malloc((size_t)store->manifest.page_stride);
    if (!snapshot) {
        set_error(error, error_size,
                  "out of memory reading one verified PLE page");
        return false;
    }
    if (!pread_all(store->fd, snapshot, (size_t)store->manifest.page_stride,
                   absolute)) {
        set_error(error, error_size, "cannot read PLE page %" PRIu64 ": %s",
                  page, strerror(errno));
        free(snapshot);
        return false;
    }
    if (!validate_page_header(store, page, snapshot, error, error_size)) {
        free(snapshot);
        return false;
    }
    sha256_context hash;
    uint8_t observed[DS4_PLE_STORE_V1_SHA256_BYTES];
    uint8_t expected[DS4_PLE_STORE_V1_SHA256_BYTES];
    sha256_init(&hash);
    sha256_update(&hash, snapshot, (size_t)store->manifest.page_stride);
    if (!read_page_digest(store, page, expected)) {
        set_error(error, error_size, "cannot hash PLE page %" PRIu64 ": %s",
                  page, strerror(errno));
        free(snapshot);
        return false;
    }
    sha256_final(&hash, observed);
    if (memcmp(observed, expected, sizeof(observed)) != 0) {
        set_error(error, error_size,
                  "PLE page %" PRIu64 " SHA-256 mismatch", page);
        free(snapshot);
        return false;
    }
    *page_out = snapshot;
    return true;
}

bool ds4_ple_store_verify_page(const ds4_ple_store *store, uint64_t page,
                               char *error, size_t error_size) {
    if (error && error_size != 0) error[0] = '\0';
    uint8_t *snapshot = NULL;
    const bool valid = read_verified_page(
        store, page, &snapshot, error, error_size);
    free(snapshot);
    return valid;
}

bool ds4_ple_store_verify_all(const ds4_ple_store *store,
                              char *error, size_t error_size) {
    if (error && error_size != 0) error[0] = '\0';
    if (!store) {
        set_error(error, error_size, "invalid PLE store");
        return false;
    }
    sha256_context payload_hash;
    sha256_init(&payload_hash);
    uint8_t buffer[IO_CHUNK_BYTES];
    for (uint64_t page = 0; page < store->manifest.page_count; page++) {
        uint64_t relative = 0;
        uint64_t absolute = 0;
        if (!page_relative_offset(&store->manifest, page, &relative) ||
            !add_u64(store->file_offset, relative, &absolute)) {
            set_error(error, error_size, "PLE page extent overflow");
            return false;
        }
        uint8_t page_header[DS4_PLE_STORE_V1_PAGE_HEADER_BYTES];
        if (!pread_all(store->fd, page_header, sizeof(page_header), absolute) ||
            !validate_page_header(store, page, page_header,
                                  error, error_size)) {
            if (error && error_size != 0 && error[0] == '\0') {
                set_error(error, error_size,
                          "cannot read PLE page %" PRIu64, page);
            }
            return false;
        }
        sha256_context page_hash;
        sha256_init(&page_hash);
        uint64_t left = store->manifest.page_stride;
        uint64_t cursor = absolute;
        while (left != 0) {
            size_t take = sizeof(buffer);
            if ((uint64_t)take > left) take = (size_t)left;
            if (!pread_all(store->fd, buffer, take, cursor)) {
                set_error(error, error_size,
                          "cannot read PLE payload page %" PRIu64 ": %s",
                          page, strerror(errno));
                return false;
            }
            sha256_update(&page_hash, buffer, take);
            sha256_update(&payload_hash, buffer, take);
            cursor += take;
            left -= take;
        }
        uint8_t observed[DS4_PLE_STORE_V1_SHA256_BYTES];
        uint8_t expected[DS4_PLE_STORE_V1_SHA256_BYTES];
        sha256_final(&page_hash, observed);
        if (!read_page_digest(store, page, expected)) {
            set_error(error, error_size,
                      "cannot read PLE page digest %" PRIu64, page);
            return false;
        }
        if (memcmp(observed, expected, sizeof(observed)) != 0) {
            set_error(error, error_size,
                      "PLE page %" PRIu64 " SHA-256 mismatch", page);
            return false;
        }
    }
    uint8_t observed_payload[DS4_PLE_STORE_V1_SHA256_BYTES];
    sha256_final(&payload_hash, observed_payload);
    if (memcmp(observed_payload, store->manifest.payload_sha256,
               sizeof(observed_payload)) != 0) {
        set_error(error, error_size, "PLE whole-payload SHA-256 mismatch");
        return false;
    }
    return true;
}

bool ds4_ple_store_read_row(const ds4_ple_store *store, uint64_t row,
                            void *encoded, size_t encoded_size,
                            char *error, size_t error_size) {
    if (error && error_size != 0) error[0] = '\0';
    if (!store || !encoded ||
        encoded_size != store->manifest.encoded_row_bytes) {
        set_error(error, error_size, "invalid PLE encoded-row buffer");
        return false;
    }
    uint64_t page_offset = 0;
    uint64_t page = 0;
    uint32_t slot = 0;
    if (!ds4_ple_store_locate_row(store, row, &page_offset, &slot, &page)) {
        set_error(error, error_size, "PLE row %" PRIu64 " is out of range", row);
        return false;
    }
    uint8_t *snapshot = NULL;
    if (!read_verified_page(store, page, &snapshot, error, error_size)) {
        return false;
    }
    uint64_t row_displacement = 0;
    uint64_t row_end = 0;
    if (!mul_u64(slot, store->manifest.encoded_row_bytes,
                 &row_displacement) ||
        !add_u64(row_displacement, DS4_PLE_STORE_V1_PAGE_HEADER_BYTES,
                 &row_displacement) ||
        !add_u64(row_displacement, store->manifest.encoded_row_bytes,
                 &row_end) ||
        row_end > store->manifest.page_stride) {
        set_error(error, error_size, "PLE row %" PRIu64 " extent overflow",
                  row);
        free(snapshot);
        return false;
    }
    memcpy(encoded, snapshot + (size_t)row_displacement, encoded_size);
    free(snapshot);
    return true;
}

static void encode_page_header(uint8_t header[64],
                               const ds4_ple_store_geometry *geometry,
                               const ds4_ple_store_codec *codec,
                               uint64_t page, uint64_t first_row,
                               uint32_t row_count) {
    memset(header, 0, DS4_PLE_STORE_V1_PAGE_HEADER_BYTES);
    memcpy(header, page_magic, sizeof(page_magic));
    store_u32_le(header + PAGE_VERSION_OFFSET, DS4_PLE_STORE_V1_VERSION);
    store_u32_le(header + PAGE_HEADER_BYTES_OFFSET,
                 DS4_PLE_STORE_V1_PAGE_HEADER_BYTES);
    store_u64_le(header + PAGE_INDEX_OFFSET, page);
    store_u64_le(header + PAGE_FIRST_ROW_OFFSET, first_row);
    store_u32_le(header + PAGE_ROW_COUNT_OFFSET, row_count);
    store_u32_le(header + PAGE_ROW_WIDTH_OFFSET, geometry->row_width);
    store_u32_le(header + PAGE_ROW_BYTES_OFFSET, codec->encoded_row_bytes);
    store_u32_le(header + PAGE_ROWS_PER_PAGE_OFFSET,
                 geometry->rows_per_page);
    store_u32_le(header + PAGE_CODEC_VERSION_OFFSET, codec->version);
    store_u32_le(header + PAGE_CODEC_GROUP_OFFSET, codec->group_size);
    store_u32_le(header + PAGE_FAMILY_OFFSET,
                 DS4_PLE_STORE_FAMILY_QWEN4EXP);
}

static bool encode_manifest_header(
        uint8_t header[DS4_PLE_STORE_V1_HEADER_BYTES],
        const ds4_ple_store_geometry *geometry,
        const ds4_ple_store_codec *codec,
        const ple_layout *layout,
        const uint8_t payload_digest[DS4_PLE_STORE_V1_SHA256_BYTES]) {
    memset(header, 0, DS4_PLE_STORE_V1_HEADER_BYTES);
    memcpy(header, store_magic, sizeof(store_magic));
    store_u32_le(header + HEADER_VERSION_OFFSET, DS4_PLE_STORE_V1_VERSION);
    store_u32_le(header + HEADER_BYTES_OFFSET, DS4_PLE_STORE_V1_HEADER_BYTES);
    store_u32_le(header + HEADER_FAMILY_OFFSET,
                 DS4_PLE_STORE_FAMILY_QWEN4EXP);
    store_u32_le(header + HEADER_HEAD_COUNT_OFFSET,
                 DS4_PLE_STORE_V1_HEADS);
    if (!store_wire_id(header + HEADER_PROFILE_OFFSET,
                       DS4_PLE_STORE_V1_PROFILE_ID) ||
        !store_wire_id(header + HEADER_HASH_OFFSET,
                       DS4_PLE_STORE_V1_HASH_ID) ||
        !store_wire_id(header + HEADER_CODEC_OFFSET, codec->id)) {
        return false;
    }
    store_u32_le(header + HEADER_CODEC_VERSION_OFFSET, codec->version);
    store_u32_le(header + HEADER_CODEC_GROUP_OFFSET, codec->group_size);
    store_u32_le(header + HEADER_ROW_BYTES_OFFSET, codec->encoded_row_bytes);
    store_u32_le(header + HEADER_ROWS_PER_PAGE_OFFSET,
                 geometry->rows_per_page);
    store_u32_le(header + HEADER_PAGE_ALIGNMENT_OFFSET,
                 geometry->page_alignment);
    store_u32_le(header + HEADER_PAGE_HEADER_BYTES_OFFSET,
                 DS4_PLE_STORE_V1_PAGE_HEADER_BYTES);
    store_u64_le(header + HEADER_ROW_COUNT_OFFSET, geometry->row_count);
    store_u32_le(header + HEADER_ROW_WIDTH_OFFSET, geometry->row_width);
    store_u32_le(header + HEADER_ROW_ALIGNMENT_OFFSET,
                 geometry->row_alignment);
    store_u64_le(header + HEADER_PAGE_COUNT_OFFSET, layout->page_count);
    store_u64_le(header + HEADER_PAGE_STRIDE_OFFSET, layout->page_stride);
    store_u64_le(header + HEADER_PAGE_DIGEST_OFFSET_OFFSET,
                 layout->digest_offset);
    store_u64_le(header + HEADER_PAGE_DIGEST_BYTES_OFFSET,
                 layout->digest_bytes);
    store_u64_le(header + HEADER_PAYLOAD_OFFSET_OFFSET,
                 layout->payload_offset);
    store_u64_le(header + HEADER_PAYLOAD_BYTES_OFFSET, layout->payload_bytes);
    store_u64_le(header + HEADER_STORE_BYTES_OFFSET, layout->store_bytes);
    for (size_t head = 0; head < DS4_PLE_STORE_V1_HEADS; head++) {
        store_u32_le(header + HEADER_HEAD_PRIME_OFFSET + head * 4,
                     geometry->head_prime[head]);
        store_u32_le(header + HEADER_HEAD_OFFSET_OFFSET + head * 4,
                     geometry->head_offset[head]);
    }
    memcpy(header + HEADER_PAYLOAD_DIGEST_OFFSET, payload_digest,
           DS4_PLE_STORE_V1_SHA256_BYTES);
    return true;
}

static char *sibling_template(const char *target) {
    static const char suffix[] = ".tmp.XXXXXX";
    if (!target) return NULL;
    const size_t length = strlen(target);
    if (length == 0 || length > SIZE_MAX - sizeof(suffix)) return NULL;
    char *result = malloc(length + sizeof(suffix));
    if (!result) return NULL;
    memcpy(result, target, length);
    memcpy(result + length, suffix, sizeof(suffix));
    return result;
}

static char *parent_directory(const char *path) {
    const char *slash = strrchr(path, '/');
    if (!slash) {
        char *dot = malloc(2);
        if (dot) memcpy(dot, ".", 2);
        return dot;
    }
    size_t length = (size_t)(slash - path);
    if (length == 0) length = 1;
    char *directory = malloc(length + 1);
    if (!directory) return NULL;
    memcpy(directory, path, length);
    directory[length] = '\0';
    return directory;
}

static int writer_sync(const ds4_ple_store_writer_ops *ops, int fd,
                       ds4_ple_store_sync_phase phase) {
    return ops ? ops->sync(ops->context, fd, phase) : fsync(fd);
}

static bool fsync_parent(const char *path,
                         const ds4_ple_store_writer_ops *ops,
                         ds4_ple_store_sync_phase phase,
                         char *error, size_t error_size) {
    char *directory = parent_directory(path);
    if (!directory) {
        set_error(error, error_size, "out of memory resolving target directory");
        return false;
    }
    int flags = O_RDONLY;
#ifdef O_DIRECTORY
    flags |= O_DIRECTORY;
#endif
    const int fd = open(directory, flags);
    const int saved_open = errno;
    free(directory);
    if (fd < 0) {
        set_error(error, error_size, "cannot open target directory: %s",
                  strerror(saved_open));
        return false;
    }
    bool ok = true;
    if (writer_sync(ops, fd, phase) != 0) {
        set_error(error, error_size, "cannot fsync target directory: %s",
                  strerror(errno));
        ok = false;
    }
    if (close(fd) != 0 && ok) {
        set_error(error, error_size, "cannot close target directory: %s",
                  strerror(errno));
        ok = false;
    }
    return ok;
}

bool ds4_ple_store_write_atomic_with_ops(
        const char *target_path, const ds4_ple_store_geometry *geometry,
        const ds4_ple_store_codec *codec,
        ds4_ple_store_encode_row_fn encode_row, void *encode_context,
        const ds4_ple_store_writer_ops *ops,
        char *error, size_t error_size) {
    if (error && error_size != 0) error[0] = '\0';
    ple_layout layout;
    if (!target_path || !target_path[0] || !encode_row ||
        (ops && !ops->sync) ||
        !compute_layout(geometry, codec, &layout, error, error_size)) {
        if (!error || error_size == 0 || error[0] == '\0') {
            set_error(error, error_size, "invalid PLE writer arguments");
        }
        return false;
    }
    if (layout.page_stride > SIZE_MAX) {
        set_error(error, error_size, "PLE page stride exceeds address space");
        return false;
    }
    char *temporary = sibling_template(target_path);
    if (!temporary) {
        set_error(error, error_size, "cannot allocate PLE temporary path");
        return false;
    }
    int fd = mkstemp(temporary);
    if (fd < 0) {
        set_error(error, error_size, "cannot create PLE temporary file: %s",
                  strerror(errno));
        free(temporary);
        return false;
    }
    bool installed = false;
    bool ok = false;
    uint8_t *page_buffer = NULL;
    if (fchmod(fd, 0644) != 0 || ftruncate(fd, (off_t)layout.store_bytes) != 0) {
        set_error(error, error_size, "cannot size PLE temporary file: %s",
                  strerror(errno));
        goto done;
    }
    page_buffer = calloc(1, (size_t)layout.page_stride);
    if (!page_buffer) {
        set_error(error, error_size, "out of memory allocating one PLE page");
        goto done;
    }
    sha256_context payload_hash;
    sha256_init(&payload_hash);
    for (uint64_t page = 0; page < layout.page_count; page++) {
        memset(page_buffer, 0, (size_t)layout.page_stride);
        uint64_t first_row = 0;
        if (!mul_u64(page, geometry->rows_per_page, &first_row)) {
            set_error(error, error_size, "PLE writer row arithmetic overflow");
            goto done;
        }
        const uint64_t remaining = geometry->row_count - first_row;
        const uint32_t page_rows = remaining < geometry->rows_per_page
            ? (uint32_t)remaining : geometry->rows_per_page;
        encode_page_header(page_buffer, geometry, codec, page, first_row,
                           page_rows);
        for (uint32_t slot = 0; slot < page_rows; slot++) {
            uint8_t *encoded = page_buffer +
                DS4_PLE_STORE_V1_PAGE_HEADER_BYTES +
                (size_t)slot * codec->encoded_row_bytes;
            if (!encode_row(encode_context, first_row + slot, encoded,
                            codec->encoded_row_bytes)) {
                set_error(error, error_size,
                          "PLE row encoder failed at row %" PRIu64,
                          first_row + slot);
                goto done;
            }
        }
        uint64_t page_displacement = 0;
        uint64_t page_offset = 0;
        if (!mul_u64(page, layout.page_stride, &page_displacement) ||
            !add_u64(layout.payload_offset, page_displacement, &page_offset) ||
            !pwrite_all(fd, page_buffer, (size_t)layout.page_stride,
                        page_offset)) {
            set_error(error, error_size, "cannot write PLE page %" PRIu64 ": %s",
                      page, strerror(errno));
            goto done;
        }
        sha256_context page_hash;
        uint8_t page_digest[DS4_PLE_STORE_V1_SHA256_BYTES];
        sha256_init(&page_hash);
        sha256_update(&page_hash, page_buffer, (size_t)layout.page_stride);
        sha256_final(&page_hash, page_digest);
        sha256_update(&payload_hash, page_buffer, (size_t)layout.page_stride);
        uint64_t digest_displacement = 0;
        uint64_t digest_offset = 0;
        if (!mul_u64(page, DS4_PLE_STORE_V1_SHA256_BYTES,
                     &digest_displacement) ||
            !add_u64(layout.digest_offset, digest_displacement,
                     &digest_offset) ||
            !pwrite_all(fd, page_digest, sizeof(page_digest), digest_offset)) {
            set_error(error, error_size,
                      "cannot write PLE page digest %" PRIu64 ": %s",
                      page, strerror(errno));
            goto done;
        }
    }
    uint8_t payload_digest[DS4_PLE_STORE_V1_SHA256_BYTES];
    uint8_t header[DS4_PLE_STORE_V1_HEADER_BYTES];
    sha256_final(&payload_hash, payload_digest);
    if (!encode_manifest_header(header, geometry, codec, &layout,
                                payload_digest)) {
        set_error(error, error_size, "cannot encode PLE manifest identifiers");
        goto done;
    }
    sha256_context manifest_hash;
    uint8_t manifest_digest[DS4_PLE_STORE_V1_SHA256_BYTES];
    sha256_init(&manifest_hash);
    sha256_update(&manifest_hash, header, sizeof(header));
    if (!hash_file_region(fd, DS4_PLE_STORE_V1_HEADER_BYTES,
                          layout.payload_offset -
                              DS4_PLE_STORE_V1_HEADER_BYTES,
                          &manifest_hash)) {
        set_error(error, error_size, "cannot hash PLE manifest: %s",
                  strerror(errno));
        goto done;
    }
    sha256_final(&manifest_hash, manifest_digest);
    memcpy(header + HEADER_MANIFEST_DIGEST_OFFSET, manifest_digest,
           sizeof(manifest_digest));
    if (!pwrite_all(fd, header, sizeof(header), 0) ||
        writer_sync(ops, fd, DS4_PLE_STORE_SYNC_TEMP_FILE) != 0) {
        set_error(error, error_size, "cannot publish PLE temporary file: %s",
                  strerror(errno));
        goto done;
    }
    if (close(fd) != 0) {
        fd = -1;
        set_error(error, error_size, "cannot close PLE temporary file: %s",
                  strerror(errno));
        goto done;
    }
    fd = -1;

    const int verify_fd = open(temporary, O_RDONLY);
    if (verify_fd < 0) {
        set_error(error, error_size, "cannot reopen PLE temporary file: %s",
                  strerror(errno));
        goto done;
    }
    ds4_ple_store *store = NULL;
    if (!ds4_ple_store_open_embedded(&store, verify_fd, 0,
                                     layout.store_bytes, geometry, codec,
                                     error, error_size)) {
        close(verify_fd);
        goto done;
    }
    close(verify_fd);
    if (!ds4_ple_store_verify_all(store, error, error_size) ||
        !ds4_ple_store_verify_page(store, 0, error, error_size) ||
        (layout.page_count > 1 &&
         !ds4_ple_store_verify_page(store, 1, error, error_size)) ||
        !ds4_ple_store_verify_page(store, layout.page_count - 1,
                                   error, error_size)) {
        ds4_ple_store_close(store);
        goto done;
    }
    ds4_ple_store_close(store);
    if (!fsync_parent(target_path, ops,
                      DS4_PLE_STORE_SYNC_PARENT_BEFORE_RENAME,
                      error, error_size)) {
        goto done;
    }
    if (rename(temporary, target_path) != 0) {
        set_error(error, error_size, "cannot atomically install PLE store: %s",
                  strerror(errno));
        goto done;
    }
    installed = true;
    if (!fsync_parent(target_path, ops,
                      DS4_PLE_STORE_SYNC_PARENT_AFTER_RENAME,
                      error, error_size)) {
        goto done;
    }
    ok = true;

done:
    free(page_buffer);
    if (fd >= 0) close(fd);
    if (!installed) unlink(temporary);
    free(temporary);
    return ok;
}

bool ds4_ple_store_write_atomic(
        const char *target_path, const ds4_ple_store_geometry *geometry,
        const ds4_ple_store_codec *codec,
        ds4_ple_store_encode_row_fn encode_row, void *encode_context,
        char *error, size_t error_size) {
    return ds4_ple_store_write_atomic_with_ops(
        target_path, geometry, codec, encode_row, encode_context, NULL,
        error, error_size);
}
