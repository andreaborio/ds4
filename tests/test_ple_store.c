#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64
#define _DARWIN_C_SOURCE

#include "ds4_ple_store.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", \
                __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

enum {
    FIXTURE_ROWS = 100,
    FIXTURE_WIDTH = 8,
    FIXTURE_ROWS_PER_PAGE = 8,
    FIXTURE_ROW_BYTES = 8,
    HEADER_FAMILY_OFFSET = 16,
    HEADER_PROFILE_OFFSET = 24,
    HEADER_HASH_OFFSET = 56,
    HEADER_CODEC_OFFSET = 88,
    HEADER_ROW_BYTES_OFFSET = 128,
    HEADER_PAGE_ALIGNMENT_OFFSET = 136,
    HEADER_ROW_COUNT_OFFSET = 144,
    HEADER_ROW_ALIGNMENT_OFFSET = 156,
    HEADER_PAGE_COUNT_OFFSET = 160,
    HEADER_PAGE_STRIDE_OFFSET = 168,
    HEADER_PAYLOAD_OFFSET_OFFSET = 192,
    HEADER_HEAD_PRIME_OFFSET = 216,
    HEADER_MANIFEST_DIGEST_OFFSET = 376,
    HEADER_RESERVED_OFFSET = 408,
    PAGE_ROW_WIDTH_OFFSET = 36,
    PAGE_CODEC_GROUP_OFFSET = 52,
    PAGE_FAMILY_OFFSET = 56,
};

static const uint32_t fixture_prime[DS4_PLE_STORE_V1_HEADS] = {
    5u, 5u, 5u, 5u, 5u, 5u, 5u, 5u,
    5u, 5u, 5u, 5u, 5u, 11u, 11u, 13u,
};

static const uint32_t fixture_offset[DS4_PLE_STORE_V1_HEADS] = {
    0u, 5u, 10u, 15u, 20u, 25u, 30u, 35u,
    40u, 45u, 50u, 55u, 60u, 65u, 76u, 87u,
};

static const uint32_t pinned_prime[DS4_PLE_STORE_V1_HEADS] = {
    20000003u, 20000023u, 20000033u, 20000047u,
    20000059u, 20000063u, 20000069u, 20000077u,
    20000081u, 20000093u, 20000107u, 20000147u,
    20000153u, 20000159u, 20000161u, 20000171u,
};

static const uint32_t pinned_offset[DS4_PLE_STORE_V1_HEADS] = {
    0u, 20000003u, 40000026u, 60000059u,
    80000106u, 100000165u, 120000228u, 140000297u,
    160000374u, 180000455u, 200000548u, 220000655u,
    240000802u, 260000955u, 280001114u, 300001275u,
};

/* Independently calculated with Python hashlib over the canonical fixture. */
static const uint8_t fixture_payload_sha256[32] = {
    0xcdu, 0x3bu, 0x35u, 0x73u, 0xe6u, 0x30u, 0xbfu, 0xa2u,
    0xe3u, 0x91u, 0xe0u, 0xd6u, 0xc3u, 0xf1u, 0xf1u, 0x10u,
    0x30u, 0x86u, 0x26u, 0xa7u, 0xa3u, 0xccu, 0x5eu, 0x91u,
    0x32u, 0x59u, 0xb4u, 0xf4u, 0x6bu, 0x22u, 0xb1u, 0x44u,
};

static const uint8_t fixture_manifest_sha256[32] = {
    0x82u, 0xe7u, 0x7du, 0x85u, 0xcbu, 0x6eu, 0xd9u, 0x54u,
    0x68u, 0x8au, 0xaeu, 0xa5u, 0x0cu, 0x25u, 0x09u, 0x6eu,
    0x51u, 0x0bu, 0xf4u, 0xf3u, 0xecu, 0x3cu, 0x37u, 0xc6u,
    0x3fu, 0x2fu, 0x87u, 0x5au, 0x8fu, 0x2fu, 0x3du, 0x2fu,
};

typedef struct {
    uint64_t fail_at;
    uint8_t bias;
} encoder_context;

typedef struct {
    uint32_t state[8];
    uint64_t bytes;
    uint8_t block[64];
    size_t block_len;
} test_sha256_context;

typedef struct {
    ds4_ple_store_sync_phase fail_phase;
    uint32_t calls[4];
} sync_context;

static uint32_t test_rotr32(uint32_t value, unsigned shift) {
    return (value >> shift) | (value << (32u - shift));
}

static void test_sha256_transform(test_sha256_context *context,
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
        const uint32_t s0 = test_rotr32(w[i - 15], 7) ^
                            test_rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = test_rotr32(w[i - 2], 17) ^
                            test_rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = context->state[0], b = context->state[1];
    uint32_t c = context->state[2], d = context->state[3];
    uint32_t e = context->state[4], f = context->state[5];
    uint32_t g = context->state[6], h = context->state[7];
    for (size_t i = 0; i < 64; i++) {
        const uint32_t s1 = test_rotr32(e, 6) ^ test_rotr32(e, 11) ^
                            test_rotr32(e, 25);
        const uint32_t ch = (e & f) ^ (~e & g);
        const uint32_t t1 = h + s1 + ch + k[i] + w[i];
        const uint32_t s0 = test_rotr32(a, 2) ^ test_rotr32(a, 13) ^
                            test_rotr32(a, 22);
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

static void test_sha256_init(test_sha256_context *context) {
    *context = (test_sha256_context){
        .state = {
            0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
            0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
        },
    };
}

static void test_sha256_update(test_sha256_context *context,
                               const void *data_pointer, size_t size) {
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
            test_sha256_transform(context, context->block);
            context->block_len = 0;
        }
    }
}

static void test_sha256_final(test_sha256_context *context,
                              uint8_t digest[32]) {
    const uint64_t bits = context->bytes * UINT64_C(8);
    context->block[context->block_len++] = 0x80;
    if (context->block_len > 56) {
        memset(context->block + context->block_len, 0,
               sizeof(context->block) - context->block_len);
        test_sha256_transform(context, context->block);
        context->block_len = 0;
    }
    memset(context->block + context->block_len, 0, 56 - context->block_len);
    for (size_t i = 0; i < 8; i++) {
        context->block[56 + i] = (uint8_t)(bits >> (56 - i * 8));
    }
    test_sha256_transform(context, context->block);
    for (size_t i = 0; i < 8; i++) {
        digest[i * 4] = (uint8_t)(context->state[i] >> 24);
        digest[i * 4 + 1] = (uint8_t)(context->state[i] >> 16);
        digest[i * 4 + 2] = (uint8_t)(context->state[i] >> 8);
        digest[i * 4 + 3] = (uint8_t)context->state[i];
    }
}

static int injected_sync(void *context_pointer, int fd,
                         ds4_ple_store_sync_phase phase) {
    sync_context *context = context_pointer;
    if (!context || phase < DS4_PLE_STORE_SYNC_TEMP_FILE ||
        phase > DS4_PLE_STORE_SYNC_PARENT_AFTER_RENAME) {
        errno = EINVAL;
        return -1;
    }
    context->calls[phase]++;
    if (phase == context->fail_phase) {
        errno = EIO;
        return -1;
    }
    return fsync(fd);
}

static void store_u32_le(uint8_t out[4], uint32_t value) {
    out[0] = (uint8_t)value;
    out[1] = (uint8_t)(value >> 8);
    out[2] = (uint8_t)(value >> 16);
    out[3] = (uint8_t)(value >> 24);
}

static void store_u64_le(uint8_t out[8], uint64_t value) {
    store_u32_le(out, (uint32_t)value);
    store_u32_le(out + 4, (uint32_t)(value >> 32));
}

static bool write_all_at(int fd, const void *source_pointer, size_t size,
                         uint64_t offset) {
    const uint8_t *source = source_pointer;
    while (size != 0) {
        const ssize_t written = pwrite(fd, source, size, (off_t)offset);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) return false;
        source += (size_t)written;
        size -= (size_t)written;
        offset += (uint64_t)written;
    }
    return true;
}

static bool read_all_at(int fd, void *destination_pointer, size_t size,
                        uint64_t offset) {
    uint8_t *destination = destination_pointer;
    while (size != 0) {
        const ssize_t got = pread(fd, destination, size, (off_t)offset);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) return false;
        destination += (size_t)got;
        size -= (size_t)got;
        offset += (uint64_t)got;
    }
    return true;
}

static uint64_t file_size(int fd) {
    struct stat status;
    if (fstat(fd, &status) != 0 || status.st_size < 0) return UINT64_MAX;
    return (uint64_t)status.st_size;
}

static ds4_ple_store_geometry fixture_geometry(void) {
    ds4_ple_store_geometry geometry = {
        .row_count = FIXTURE_ROWS,
        .row_width = FIXTURE_WIDTH,
        .row_alignment = 4,
        .rows_per_page = FIXTURE_ROWS_PER_PAGE,
        .page_alignment = DS4_PLE_STORE_V1_MIN_PAGE_ALIGNMENT,
    };
    memcpy(geometry.head_prime, fixture_prime, sizeof(fixture_prime));
    memcpy(geometry.head_offset, fixture_offset, sizeof(fixture_offset));
    return geometry;
}

static ds4_ple_store_codec fixture_codec(void) {
    /* This is intentionally a raw synthetic test codec, not a release codec. */
    const ds4_ple_store_codec codec = {
        .id = "fixture-raw-u8-nonprod",
        .version = 1,
        .group_size = 1,
        .encoded_row_bytes = FIXTURE_ROW_BYTES,
    };
    return codec;
}

static void expected_row(uint64_t row, uint8_t bias,
                         uint8_t out[FIXTURE_ROW_BYTES]) {
    for (uint32_t byte = 0; byte < FIXTURE_ROW_BYTES; byte++) {
        out[byte] = (uint8_t)(bias + row * 17u + byte * 29u);
    }
}

static bool encode_fixture_row(void *context_pointer, uint64_t row,
                               uint8_t *encoded, uint32_t encoded_bytes) {
    encoder_context *context = context_pointer;
    if (!context || !encoded || encoded_bytes != FIXTURE_ROW_BYTES ||
        row >= context->fail_at) {
        return false;
    }
    expected_row(row, context->bias, encoded);
    return true;
}

static bool write_literal(const char *path, const void *data, size_t size) {
    const int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) return false;
    const bool ok = write_all_at(fd, data, size, 0);
    return close(fd) == 0 && ok;
}

static bool file_equals(const char *path, const void *data, size_t size) {
    const int fd = open(path, O_RDONLY);
    if (fd < 0) return false;
    uint8_t buffer[64];
    const bool ok = size <= sizeof(buffer) && file_size(fd) == size &&
                    read_all_at(fd, buffer, size, 0) &&
                    memcmp(buffer, data, size) == 0;
    close(fd);
    return ok;
}

static bool copy_file(const char *source_path, const char *target_path,
                      uint64_t prefix, uint64_t suffix) {
    const int source = open(source_path, O_RDONLY);
    if (source < 0) return false;
    const uint64_t source_size = file_size(source);
    const int target = open(target_path, O_RDWR | O_CREAT | O_TRUNC, 0600);
    if (source_size == UINT64_MAX || target < 0 ||
        prefix > (uint64_t)INT64_MAX ||
        source_size > (uint64_t)INT64_MAX - prefix ||
        suffix > (uint64_t)INT64_MAX - prefix - source_size ||
        ftruncate(target, (off_t)(prefix + source_size + suffix)) != 0) {
        if (target >= 0) close(target);
        close(source);
        return false;
    }
    uint8_t buffer[4096];
    uint64_t cursor = 0;
    bool ok = true;
    while (cursor < source_size) {
        size_t take = sizeof(buffer);
        if ((uint64_t)take > source_size - cursor) {
            take = (size_t)(source_size - cursor);
        }
        if (!read_all_at(source, buffer, take, cursor) ||
            !write_all_at(target, buffer, take, prefix + cursor)) {
            ok = false;
            break;
        }
        cursor += take;
    }
    if (close(target) != 0) ok = false;
    if (close(source) != 0) ok = false;
    return ok;
}

static bool mutate_byte(const char *path, uint64_t offset, uint8_t xor_value) {
    const int fd = open(path, O_RDWR);
    if (fd < 0) return false;
    uint8_t value = 0;
    const bool ok = read_all_at(fd, &value, 1, offset) &&
                    ((value ^= xor_value), true) &&
                    write_all_at(fd, &value, 1, offset) &&
                    close(fd) == 0;
    if (!ok) close(fd);
    return ok;
}

static bool mutate_u32(const char *path, uint64_t offset, uint32_t value) {
    const int fd = open(path, O_RDWR);
    if (fd < 0) return false;
    uint8_t bytes[4];
    store_u32_le(bytes, value);
    const bool ok = write_all_at(fd, bytes, sizeof(bytes), offset);
    return close(fd) == 0 && ok;
}

static bool mutate_u64(const char *path, uint64_t offset, uint64_t value) {
    const int fd = open(path, O_RDWR);
    if (fd < 0) return false;
    uint8_t bytes[8];
    store_u64_le(bytes, value);
    const bool ok = write_all_at(fd, bytes, sizeof(bytes), offset);
    return close(fd) == 0 && ok;
}

static bool test_hash_file_region(int fd, uint64_t offset, uint64_t bytes,
                                  test_sha256_context *hash) {
    uint8_t buffer[4096];
    while (bytes != 0) {
        size_t take = sizeof(buffer);
        if ((uint64_t)take > bytes) take = (size_t)bytes;
        if (!read_all_at(fd, buffer, take, offset)) return false;
        test_sha256_update(hash, buffer, take);
        offset += take;
        bytes -= take;
    }
    return true;
}

/* Re-authenticates page zero and the manifest after a payload mutation while
 * deliberately retaining the old whole-payload digest in the header. */
static bool rewrite_page_and_manifest_digests(
        const char *path, uint64_t payload_offset, uint64_t page_stride) {
    if (page_stride > SIZE_MAX) return false;
    const int fd = open(path, O_RDWR);
    if (fd < 0) return false;
    uint8_t *page = malloc((size_t)page_stride);
    uint8_t header[DS4_PLE_STORE_V1_HEADER_BYTES];
    bool ok = page != NULL &&
              read_all_at(fd, page, (size_t)page_stride, payload_offset) &&
              read_all_at(fd, header, sizeof(header), 0);
    if (ok) {
        test_sha256_context page_hash;
        uint8_t page_digest[DS4_PLE_STORE_V1_SHA256_BYTES];
        test_sha256_init(&page_hash);
        test_sha256_update(&page_hash, page, (size_t)page_stride);
        test_sha256_final(&page_hash, page_digest);
        ok = write_all_at(fd, page_digest, sizeof(page_digest),
                          DS4_PLE_STORE_V1_HEADER_BYTES);
    }
    if (ok) {
        memset(header + HEADER_MANIFEST_DIGEST_OFFSET, 0,
               DS4_PLE_STORE_V1_SHA256_BYTES);
        test_sha256_context manifest_hash;
        uint8_t manifest_digest[DS4_PLE_STORE_V1_SHA256_BYTES];
        test_sha256_init(&manifest_hash);
        test_sha256_update(&manifest_hash, header, sizeof(header));
        ok = test_hash_file_region(
            fd, DS4_PLE_STORE_V1_HEADER_BYTES,
            payload_offset - DS4_PLE_STORE_V1_HEADER_BYTES,
            &manifest_hash);
        if (ok) {
            test_sha256_final(&manifest_hash, manifest_digest);
            ok = write_all_at(
                fd, manifest_digest, sizeof(manifest_digest),
                HEADER_MANIFEST_DIGEST_OFFSET);
        }
    }
    free(page);
    if (close(fd) != 0) ok = false;
    return ok;
}

static bool rejected_at_open(const char *path, uint64_t offset, uint64_t bytes,
                             const ds4_ple_store_geometry *geometry,
                             const ds4_ple_store_codec *codec) {
    const int fd = open(path, O_RDONLY);
    if (fd < 0) return false;
    char error[256] = {0};
    ds4_ple_store *store = NULL;
    const bool rejected = !ds4_ple_store_open_embedded(
        &store, fd, offset, bytes, geometry, codec, error, sizeof(error));
    ds4_ple_store_close(store);
    close(fd);
    return rejected && error[0] != '\0';
}

static bool no_sibling_temporaries(const char *directory,
                                   const char *target_basename) {
    DIR *stream = opendir(directory);
    if (!stream) return false;
    char prefix[256];
    if (snprintf(prefix, sizeof(prefix), "%s.tmp.", target_basename) < 0) {
        closedir(stream);
        return false;
    }
    bool none = true;
    struct dirent *entry;
    while ((entry = readdir(stream)) != NULL) {
        if (strncmp(entry->d_name, prefix, strlen(prefix)) == 0) {
            none = false;
            break;
        }
    }
    closedir(stream);
    return none;
}

int main(void) {
    char directory[] = "/tmp/ds4-ple-store.XXXXXX";
    CHECK(mkdtemp(directory) != NULL);
    char target[1024];
    char corrupt[1024];
    char embedded[1024];
    char blocked_target[1024];
    char blocked_marker[1024];
    CHECK(snprintf(target, sizeof(target), "%s/store.ple", directory) > 0);
    CHECK(snprintf(corrupt, sizeof(corrupt), "%s/corrupt.ple", directory) > 0);
    CHECK(snprintf(embedded, sizeof(embedded), "%s/embedded.bin", directory) > 0);
    CHECK(snprintf(blocked_target, sizeof(blocked_target),
                   "%s/blocked-target", directory) > 0);
    CHECK(snprintf(blocked_marker, sizeof(blocked_marker),
                   "%s/blocked-target/marker", directory) > 0);

    const ds4_ple_store_geometry geometry = fixture_geometry();
    const ds4_ple_store_codec codec = fixture_codec();
    char error[256] = {0};
    CHECK(ds4_ple_store_descriptor_validate(
        &geometry, &codec, error, sizeof(error)));
    ds4_ple_store_geometry pinned_geometry = geometry;
    pinned_geometry.row_count = UINT64_C(320001536);
    pinned_geometry.row_width = 160;
    pinned_geometry.row_alignment = 128;
    memcpy(pinned_geometry.head_prime, pinned_prime, sizeof(pinned_prime));
    memcpy(pinned_geometry.head_offset, pinned_offset, sizeof(pinned_offset));
    CHECK(ds4_ple_store_descriptor_validate(
        &pinned_geometry, &codec, error, sizeof(error)));

    int pipe_fds[2];
    CHECK(pipe(pipe_fds) == 0);
    ds4_ple_store *non_regular_store = NULL;
    error[0] = '\0';
    CHECK(!ds4_ple_store_open_embedded(
        &non_regular_store, pipe_fds[0], 0,
        DS4_PLE_STORE_V1_HEADER_BYTES, &geometry, &codec,
        error, sizeof(error)));
    CHECK(non_regular_store == NULL);
    CHECK(strstr(error, "regular file") != NULL);
    CHECK(close(pipe_fds[0]) == 0);
    CHECK(close(pipe_fds[1]) == 0);

    static const char old_target[] = "previous-target";
    CHECK(write_literal(target, old_target, sizeof(old_target)));

    encoder_context failed_encoder = {.fail_at = 17, .bias = 0};
    CHECK(!ds4_ple_store_write_atomic(
        target, &geometry, &codec, encode_fixture_row, &failed_encoder,
        error, sizeof(error)));
    CHECK(error[0] != '\0');
    CHECK(file_equals(target, old_target, sizeof(old_target)));
    CHECK(no_sibling_temporaries(directory, "store.ple"));

    ds4_ple_store_geometry overflow = geometry;
    overflow.row_count = UINT64_MAX;
    error[0] = '\0';
    CHECK(!ds4_ple_store_write_atomic(
        target, &overflow, &codec, encode_fixture_row, &failed_encoder,
        error, sizeof(error)));
    CHECK(error[0] != '\0');
    CHECK(file_equals(target, old_target, sizeof(old_target)));

    ds4_ple_store_geometry extent_overflow = geometry;
    for (uint32_t head = 0; head < DS4_PLE_STORE_V1_HEADS; head++) {
        extent_overflow.head_prime[head] = 1;
        extent_overflow.head_offset[head] = head;
    }
    extent_overflow.head_prime[DS4_PLE_STORE_V1_HEADS - 1] = UINT32_MAX;
    extent_overflow.row_count = UINT64_C(4294967310);
    extent_overflow.row_alignment = 2;
    extent_overflow.rows_per_page = UINT32_MAX;
    ds4_ple_store_codec huge_row_codec = codec;
    huge_row_codec.encoded_row_bytes = UINT32_MAX;
    error[0] = '\0';
    CHECK(!ds4_ple_store_descriptor_validate(
        &extent_overflow, &huge_row_codec, error, sizeof(error)));
    CHECK(error[0] != '\0');
    CHECK(file_equals(target, old_target, sizeof(old_target)));

    ds4_ple_store_geometry bad_alignment = geometry;
    bad_alignment.page_alignment = 6000;
    CHECK(!ds4_ple_store_write_atomic(
        target, &bad_alignment, &codec, encode_fixture_row, &failed_encoder,
        error, sizeof(error)));
    CHECK(file_equals(target, old_target, sizeof(old_target)));

    ds4_ple_store_geometry short_segments = geometry;
    short_segments.row_count = 99;
    CHECK(!ds4_ple_store_write_atomic(
        target, &short_segments, &codec, encode_fixture_row, &failed_encoder,
        error, sizeof(error)));
    CHECK(file_equals(target, old_target, sizeof(old_target)));
    ds4_ple_store_geometry long_padding = geometry;
    long_padding.row_count = 104;
    CHECK(!ds4_ple_store_write_atomic(
        target, &long_padding, &codec, encode_fixture_row, &failed_encoder,
        error, sizeof(error)));
    CHECK(file_equals(target, old_target, sizeof(old_target)));

    encoder_context encoder = {.fail_at = UINT64_MAX, .bias = 0};
    sync_context failed_sync = {
        .fail_phase = DS4_PLE_STORE_SYNC_PARENT_BEFORE_RENAME,
    };
    const ds4_ple_store_writer_ops writer_ops = {
        .sync = injected_sync,
        .context = &failed_sync,
    };
    error[0] = '\0';
    CHECK(!ds4_ple_store_write_atomic_with_ops(
        target, &geometry, &codec, encode_fixture_row, &encoder, &writer_ops,
        error, sizeof(error)));
    CHECK(error[0] != '\0');
    CHECK(failed_sync.calls[DS4_PLE_STORE_SYNC_TEMP_FILE] == 1);
    CHECK(failed_sync.calls[
              DS4_PLE_STORE_SYNC_PARENT_BEFORE_RENAME] == 1);
    CHECK(failed_sync.calls[
              DS4_PLE_STORE_SYNC_PARENT_AFTER_RENAME] == 0);
    CHECK(file_equals(target, old_target, sizeof(old_target)));
    CHECK(no_sibling_temporaries(directory, "store.ple"));

    CHECK(ds4_ple_store_write_atomic(
        target, &geometry, &codec, encode_fixture_row, &encoder,
        error, sizeof(error)));
    CHECK(no_sibling_temporaries(directory, "store.ple"));

    int fd = open(target, O_RDONLY);
    CHECK(fd >= 0);
    const uint64_t bytes = file_size(fd);
    CHECK(bytes != UINT64_MAX);
    ds4_ple_store *store = NULL;
    CHECK(ds4_ple_store_open_embedded(
        &store, fd, 0, bytes, &geometry, &codec, error, sizeof(error)));
    CHECK(store != NULL);
    const ds4_ple_store_manifest *manifest =
        ds4_ple_store_manifest_get(store);
    CHECK(manifest != NULL);
    CHECK(manifest->version == DS4_PLE_STORE_V1_VERSION);
    CHECK(manifest->family == DS4_PLE_STORE_FAMILY_QWEN4EXP);
    CHECK(strcmp(manifest->profile_id, DS4_PLE_STORE_V1_PROFILE_ID) == 0);
    CHECK(strcmp(manifest->hash_id, DS4_PLE_STORE_V1_HASH_ID) == 0);
    CHECK(strcmp(manifest->codec_id, codec.id) == 0);
    CHECK(manifest->row_count == FIXTURE_ROWS);
    CHECK(manifest->row_width == FIXTURE_WIDTH);
    CHECK(manifest->row_alignment == 4);
    CHECK(manifest->head_count == DS4_PLE_STORE_V1_HEADS);
    CHECK(manifest->page_count == 13);
    CHECK(manifest->page_stride == 4096);
    CHECK(manifest->page_digest_offset == DS4_PLE_STORE_V1_HEADER_BYTES);
    CHECK(manifest->page_digest_bytes ==
          manifest->page_count * DS4_PLE_STORE_V1_SHA256_BYTES);
    CHECK(manifest->payload_offset == 4096);
    CHECK(manifest->payload_bytes == manifest->page_count * 4096);
    CHECK(manifest->store_bytes == bytes);
    CHECK(memcmp(manifest->payload_sha256, fixture_payload_sha256,
                 sizeof(fixture_payload_sha256)) == 0);
    CHECK(memcmp(manifest->manifest_sha256, fixture_manifest_sha256,
                 sizeof(fixture_manifest_sha256)) == 0);
    CHECK(memcmp(manifest->head_prime, fixture_prime,
                 sizeof(fixture_prime)) == 0);
    CHECK(memcmp(manifest->head_offset, fixture_offset,
                 sizeof(fixture_offset)) == 0);
    CHECK(ds4_ple_store_fd(store) >= 0);
    CHECK(ds4_ple_store_file_offset(store) == 0);
    CHECK(close(fd) == 0);

    CHECK(ds4_ple_store_verify_all(store, error, sizeof(error)));
    const uint64_t rows[] = {0, 7, 8, 63, 64, 99};
    for (size_t index = 0; index < sizeof(rows) / sizeof(rows[0]); index++) {
        const uint64_t row = rows[index];
        uint64_t page_offset = UINT64_MAX;
        uint64_t page = UINT64_MAX;
        uint32_t slot = UINT32_MAX;
        CHECK(ds4_ple_store_locate_row(
            store, row, &page_offset, &slot, &page));
        CHECK(page == row / FIXTURE_ROWS_PER_PAGE);
        CHECK(slot == row % FIXTURE_ROWS_PER_PAGE);
        CHECK(page_offset ==
              manifest->payload_offset + page * manifest->page_stride);
        uint8_t observed[FIXTURE_ROW_BYTES];
        uint8_t expected[FIXTURE_ROW_BYTES];
        expected_row(row, 0, expected);
        CHECK(ds4_ple_store_read_row(
            store, row, observed, sizeof(observed), error, sizeof(error)));
        CHECK(memcmp(observed, expected, sizeof(observed)) == 0);
    }
    uint64_t bad_page_offset = UINT64_MAX;
    uint64_t bad_page = UINT64_MAX;
    uint32_t bad_slot = UINT32_MAX;
    CHECK(!ds4_ple_store_locate_row(
        store, FIXTURE_ROWS, &bad_page_offset, &bad_slot, &bad_page));
    CHECK(bad_page_offset == 0 && bad_slot == 0 && bad_page == 0);
    CHECK(!ds4_ple_store_verify_page(
        store, manifest->page_count, error, sizeof(error)));
    const uint64_t payload_offset = manifest->payload_offset;
    const uint64_t page_stride = manifest->page_stride;
    ds4_ple_store_close(store);
    store = NULL;

    CHECK(copy_file(target, embedded, 4096, 173));
    fd = open(embedded, O_RDONLY);
    CHECK(fd >= 0);
    CHECK(ds4_ple_store_open_embedded(
        &store, fd, 4096, bytes, &geometry, &codec, error, sizeof(error)));
    CHECK(ds4_ple_store_file_offset(store) == 4096);
    CHECK(close(fd) == 0);
    CHECK(ds4_ple_store_verify_all(store, error, sizeof(error)));
    ds4_ple_store_close(store);
    store = NULL;
    CHECK(rejected_at_open(
        embedded, 4096, bytes + 1, &geometry, &codec));
    CHECK(copy_file(target, embedded, 4097, 0));
    CHECK(rejected_at_open(
        embedded, 4097, bytes, &geometry, &codec));

    ds4_ple_store_codec wrong_codec = codec;
    wrong_codec.id = "fixture-raw-u8-other";
    CHECK(rejected_at_open(target, 0, bytes, &geometry, &wrong_codec));
    wrong_codec = codec;
    wrong_codec.group_size = 2;
    CHECK(rejected_at_open(target, 0, bytes, &geometry, &wrong_codec));
    wrong_codec = codec;
    wrong_codec.encoded_row_bytes++;
    CHECK(rejected_at_open(target, 0, bytes, &geometry, &wrong_codec));

    ds4_ple_store_geometry wrong_geometry = geometry;
    wrong_geometry.row_count--;
    CHECK(rejected_at_open(target, 0, bytes, &wrong_geometry, &codec));
    wrong_geometry = geometry;
    wrong_geometry.row_width++;
    CHECK(rejected_at_open(target, 0, bytes, &wrong_geometry, &codec));
    wrong_geometry = geometry;
    wrong_geometry.head_prime[5]++;
    CHECK(rejected_at_open(target, 0, bytes, &wrong_geometry, &codec));
    wrong_geometry = geometry;
    wrong_geometry.head_offset[5]++;
    CHECK(rejected_at_open(target, 0, bytes, &wrong_geometry, &codec));
    wrong_geometry = geometry;
    wrong_geometry.row_count = 99;
    CHECK(rejected_at_open(target, 0, bytes, &wrong_geometry, &codec));
    wrong_geometry = geometry;
    wrong_geometry.row_count = 104;
    CHECK(rejected_at_open(target, 0, bytes, &wrong_geometry, &codec));

    CHECK(copy_file(target, corrupt, 0, 0));
    fd = open(corrupt, O_RDWR);
    CHECK(fd >= 0);
    CHECK(ftruncate(fd, (off_t)(bytes - 1)) == 0);
    CHECK(close(fd) == 0);
    CHECK(rejected_at_open(corrupt, 0, bytes, &geometry, &codec));

    CHECK(copy_file(target, corrupt, 0, 0));
    CHECK(mutate_byte(corrupt, HEADER_MANIFEST_DIGEST_OFFSET, 0x80));
    CHECK(rejected_at_open(corrupt, 0, bytes, &geometry, &codec));

    CHECK(copy_file(target, corrupt, 0, 0));
    CHECK(mutate_byte(corrupt, DS4_PLE_STORE_V1_HEADER_BYTES, 0x01));
    CHECK(rejected_at_open(corrupt, 0, bytes, &geometry, &codec));

    CHECK(copy_file(target, corrupt, 0, 0));
    CHECK(mutate_u32(corrupt, HEADER_FAMILY_OFFSET, 3));
    CHECK(rejected_at_open(corrupt, 0, bytes, &geometry, &codec));
    CHECK(copy_file(target, corrupt, 0, 0));
    CHECK(mutate_byte(corrupt, HEADER_PROFILE_OFFSET, 0x01));
    CHECK(rejected_at_open(corrupt, 0, bytes, &geometry, &codec));
    CHECK(copy_file(target, corrupt, 0, 0));
    CHECK(mutate_byte(corrupt, HEADER_HASH_OFFSET, 0x01));
    CHECK(rejected_at_open(corrupt, 0, bytes, &geometry, &codec));
    CHECK(copy_file(target, corrupt, 0, 0));
    CHECK(mutate_byte(corrupt, HEADER_CODEC_OFFSET, 0x01));
    CHECK(rejected_at_open(corrupt, 0, bytes, &geometry, &codec));
    CHECK(copy_file(target, corrupt, 0, 0));
    CHECK(mutate_u32(corrupt, HEADER_ROW_BYTES_OFFSET, UINT32_MAX));
    CHECK(rejected_at_open(corrupt, 0, bytes, &geometry, &codec));
    CHECK(copy_file(target, corrupt, 0, 0));
    CHECK(mutate_u64(corrupt, HEADER_ROW_COUNT_OFFSET, 99));
    CHECK(rejected_at_open(corrupt, 0, bytes, &geometry, &codec));
    CHECK(copy_file(target, corrupt, 0, 0));
    CHECK(mutate_u64(corrupt, HEADER_ROW_COUNT_OFFSET, 104));
    CHECK(rejected_at_open(corrupt, 0, bytes, &geometry, &codec));
    CHECK(copy_file(target, corrupt, 0, 0));
    CHECK(mutate_u32(corrupt, HEADER_PAGE_ALIGNMENT_OFFSET, 2048));
    CHECK(rejected_at_open(corrupt, 0, bytes, &geometry, &codec));
    CHECK(copy_file(target, corrupt, 0, 0));
    CHECK(mutate_u32(corrupt, HEADER_ROW_ALIGNMENT_OFFSET, 8));
    CHECK(rejected_at_open(corrupt, 0, bytes, &geometry, &codec));
    CHECK(copy_file(target, corrupt, 0, 0));
    CHECK(mutate_u64(corrupt, HEADER_PAGE_COUNT_OFFSET, UINT64_MAX));
    CHECK(rejected_at_open(corrupt, 0, bytes, &geometry, &codec));
    CHECK(copy_file(target, corrupt, 0, 0));
    CHECK(mutate_u64(corrupt, HEADER_PAGE_STRIDE_OFFSET, UINT64_MAX));
    CHECK(rejected_at_open(corrupt, 0, bytes, &geometry, &codec));
    CHECK(copy_file(target, corrupt, 0, 0));
    CHECK(mutate_u32(corrupt, HEADER_HEAD_PRIME_OFFSET + 4, 0));
    CHECK(rejected_at_open(corrupt, 0, bytes, &geometry, &codec));
    CHECK(copy_file(target, corrupt, 0, 0));
    CHECK(mutate_byte(corrupt, HEADER_RESERVED_OFFSET, 0x01));
    CHECK(rejected_at_open(corrupt, 0, bytes, &geometry, &codec));

    CHECK(copy_file(target, corrupt, 0, 0));
    CHECK(mutate_byte(corrupt, payload_offset + 64 + 5, 0x40));
    fd = open(corrupt, O_RDONLY);
    CHECK(fd >= 0);
    CHECK(ds4_ple_store_open_embedded(
        &store, fd, 0, bytes, &geometry, &codec, error, sizeof(error)));
    CHECK(close(fd) == 0);
    CHECK(!ds4_ple_store_verify_page(store, 0, error, sizeof(error)));
    CHECK(!ds4_ple_store_verify_all(store, error, sizeof(error)));
    uint8_t transactional_row[FIXTURE_ROW_BYTES];
    memset(transactional_row, 0xa5, sizeof(transactional_row));
    CHECK(!ds4_ple_store_read_row(
        store, 0, transactional_row, sizeof(transactional_row),
        error, sizeof(error)));
    for (size_t byte = 0; byte < sizeof(transactional_row); byte++) {
        CHECK(transactional_row[byte] == 0xa5);
    }
    ds4_ple_store_close(store);
    store = NULL;

    CHECK(copy_file(target, corrupt, 0, 0));
    fd = open(corrupt, O_RDWR);
    CHECK(fd >= 0);
    CHECK(ds4_ple_store_open_embedded(
        &store, fd, 0, bytes, &geometry, &codec, error, sizeof(error)));
    CHECK(ftruncate(fd, (off_t)(payload_offset + 64 + 4)) == 0);
    memset(transactional_row, 0x5a, sizeof(transactional_row));
    CHECK(!ds4_ple_store_read_row(
        store, 0, transactional_row, sizeof(transactional_row),
        error, sizeof(error)));
    for (size_t byte = 0; byte < sizeof(transactional_row); byte++) {
        CHECK(transactional_row[byte] == 0x5a);
    }
    CHECK(close(fd) == 0);
    ds4_ple_store_close(store);
    store = NULL;

    CHECK(copy_file(target, corrupt, 0, 0));
    CHECK(mutate_byte(corrupt, payload_offset + 64 + 5, 0x40));
    CHECK(rewrite_page_and_manifest_digests(
        corrupt, payload_offset, page_stride));
    fd = open(corrupt, O_RDONLY);
    CHECK(fd >= 0);
    CHECK(ds4_ple_store_open_embedded(
        &store, fd, 0, bytes, &geometry, &codec, error, sizeof(error)));
    CHECK(close(fd) == 0);
    CHECK(ds4_ple_store_verify_page(store, 0, error, sizeof(error)));
    error[0] = '\0';
    CHECK(!ds4_ple_store_verify_all(store, error, sizeof(error)));
    CHECK(strstr(error, "whole-payload") != NULL);
    ds4_ple_store_close(store);
    store = NULL;

    CHECK(copy_file(target, corrupt, 0, 0));
    CHECK(mutate_u32(
        corrupt, payload_offset + PAGE_ROW_WIDTH_OFFSET, FIXTURE_WIDTH + 1));
    fd = open(corrupt, O_RDONLY);
    CHECK(fd >= 0);
    CHECK(ds4_ple_store_open_embedded(
        &store, fd, 0, bytes, &geometry, &codec, error, sizeof(error)));
    CHECK(close(fd) == 0);
    CHECK(!ds4_ple_store_verify_page(store, 0, error, sizeof(error)));
    ds4_ple_store_close(store);
    store = NULL;

    CHECK(copy_file(target, corrupt, 0, 0));
    CHECK(mutate_u32(
        corrupt, payload_offset + page_stride + PAGE_CODEC_GROUP_OFFSET, 9));
    fd = open(corrupt, O_RDONLY);
    CHECK(fd >= 0);
    CHECK(ds4_ple_store_open_embedded(
        &store, fd, 0, bytes, &geometry, &codec, error, sizeof(error)));
    CHECK(close(fd) == 0);
    CHECK(!ds4_ple_store_verify_page(store, 1, error, sizeof(error)));
    ds4_ple_store_close(store);
    store = NULL;

    CHECK(copy_file(target, corrupt, 0, 0));
    CHECK(mutate_u32(
        corrupt, payload_offset + 12 * page_stride + PAGE_FAMILY_OFFSET, 3));
    fd = open(corrupt, O_RDONLY);
    CHECK(fd >= 0);
    CHECK(ds4_ple_store_open_embedded(
        &store, fd, 0, bytes, &geometry, &codec, error, sizeof(error)));
    CHECK(close(fd) == 0);
    CHECK(!ds4_ple_store_verify_page(store, 12, error, sizeof(error)));
    ds4_ple_store_close(store);

    CHECK(mkdir(blocked_target, 0700) == 0);
    CHECK(write_literal(blocked_marker, old_target, sizeof(old_target)));
    error[0] = '\0';
    CHECK(!ds4_ple_store_write_atomic(
        blocked_target, &geometry, &codec, encode_fixture_row, &encoder,
        error, sizeof(error)));
    CHECK(error[0] != '\0');
    CHECK(file_equals(blocked_marker, old_target, sizeof(old_target)));
    CHECK(no_sibling_temporaries(directory, "blocked-target"));

    CHECK(unlink(blocked_marker) == 0);
    CHECK(rmdir(blocked_target) == 0);
    CHECK(unlink(corrupt) == 0);
    CHECK(unlink(embedded) == 0);
    CHECK(unlink(target) == 0);
    CHECK(rmdir(directory) == 0);
    puts("PLE store v1 parser/writer tests passed");
    return 0;
}
