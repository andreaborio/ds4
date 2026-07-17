#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include "ds4_qwen_expert_pack.h"
#include "ds4_qwen_native_gguf.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

enum {
    FIXTURE_LAYER = 2,
    FIXTURE_EXPERT = 3,
    FIXTURE_DIM = 256,
    FIXTURE_TENSOR = FIXTURE_LAYER * 3,
    FIXTURE_EXPERT_BYTES = 256 * 144,
    FIXTURE_MATRIX_BYTES = FIXTURE_EXPERT_BYTES * FIXTURE_EXPERT,
};

typedef struct {
    uint64_t source_offset[FIXTURE_LAYER][3][FIXTURE_EXPERT];
    uint64_t source_size;
} fixture_layout;

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", \
                __FILE__, __LINE__, #condition); \
        return false; \
    } \
} while (0)

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

static bool write_exact(FILE *file, const void *data, size_t size) {
    return fwrite(data, 1, size, file) == size;
}

static bool write_u32(FILE *file, uint32_t value) {
    uint8_t bytes[4];
    store_u32_le(bytes, value);
    return write_exact(file, bytes, sizeof(bytes));
}

static bool write_u64(FILE *file, uint64_t value) {
    uint8_t bytes[8];
    store_u64_le(bytes, value);
    return write_exact(file, bytes, sizeof(bytes));
}

static bool write_string(FILE *file, const char *value) {
    const size_t size = strlen(value);
    return write_u64(file, size) && write_exact(file, value, size);
}

static bool write_u32_metadata(FILE *file, const char *key, uint32_t value) {
    return write_string(file, key) && write_u32(file, 4) &&
           write_u32(file, value);
}

static uint8_t fixture_byte(
        uint32_t layer,
        uint32_t kind,
        uint32_t expert,
        uint64_t offset) {
    return (uint8_t)((layer * 71u + kind * 23u + expert * 7u + offset) % 251u);
}

static bool write_fixture_gguf(const char *path, fixture_layout *layout) {
    memset(layout, 0, sizeof(*layout));
    FILE *file = fopen(path, "wb");
    CHECK(file != NULL);
    CHECK(write_u32(file, UINT32_C(0x46554747)));
    CHECK(write_u32(file, 3));
    CHECK(write_u64(file, FIXTURE_TENSOR));
    CHECK(write_u64(file, 6));

    CHECK(write_string(file, "general.architecture"));
    CHECK(write_u32(file, 8));
    CHECK(write_string(file, "qwen35moe"));
    CHECK(write_u32_metadata(file, "general.alignment", 32));
    CHECK(write_u32_metadata(file, "qwen35moe.block_count", FIXTURE_LAYER));
    CHECK(write_u32_metadata(file, "qwen35moe.embedding_length", FIXTURE_DIM));
    CHECK(write_u32_metadata(file, "qwen35moe.expert_count", FIXTURE_EXPERT));
    CHECK(write_u32_metadata(
        file, "qwen35moe.expert_feed_forward_length", FIXTURE_DIM));

    const char *kind_name[3] = {
        "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps",
    };
    uint64_t relative = 0;
    for (uint32_t layer = 0; layer < FIXTURE_LAYER; layer++) {
        for (uint32_t kind = 0; kind < 3; kind++) {
            char name[96];
            const int count = snprintf(name, sizeof(name),
                                       "blk.%u.%s.weight",
                                       layer, kind_name[kind]);
            CHECK(count > 0 && (size_t)count < sizeof(name));
            CHECK(write_string(file, name));
            CHECK(write_u32(file, 3));
            CHECK(write_u64(file, FIXTURE_DIM));
            CHECK(write_u64(file, FIXTURE_DIM));
            CHECK(write_u64(file, FIXTURE_EXPERT));
            CHECK(write_u32(file, DS4_QWEN_EXPERT_PACK_Q4_K_TYPE));
            CHECK(write_u64(file, relative));
            relative += FIXTURE_MATRIX_BYTES;
        }
    }
    const off_t directory_end = ftello(file);
    CHECK(directory_end >= 0);
    const size_t padding = (size_t)((32 - ((uint64_t)directory_end % 32)) % 32);
    const uint8_t zeros[32] = {0};
    CHECK(write_exact(file, zeros, padding));
    const uint64_t data_offset = (uint64_t)directory_end + padding;

    uint8_t *expert_data = malloc(FIXTURE_EXPERT_BYTES);
    CHECK(expert_data != NULL);
    for (uint32_t layer = 0; layer < FIXTURE_LAYER; layer++) {
        for (uint32_t kind = 0; kind < 3; kind++) {
            const uint64_t matrix_relative =
                ((uint64_t)layer * 3 + kind) * FIXTURE_MATRIX_BYTES;
            for (uint32_t expert = 0; expert < FIXTURE_EXPERT; expert++) {
                layout->source_offset[layer][kind][expert] =
                    data_offset + matrix_relative +
                    (uint64_t)expert * FIXTURE_EXPERT_BYTES;
                for (uint64_t i = 0; i < FIXTURE_EXPERT_BYTES; i++) {
                    expert_data[i] = fixture_byte(layer, kind, expert, i);
                }
                CHECK(write_exact(file, expert_data, FIXTURE_EXPERT_BYTES));
            }
        }
    }
    free(expert_data);
    CHECK(fflush(file) == 0);
    const off_t end = ftello(file);
    CHECK(end >= 0);
    layout->source_size = (uint64_t)end;
    CHECK(fclose(file) == 0);
    return true;
}

static ds4_qwen_expert_pack_geometry fixture_geometry(void) {
    return (ds4_qwen_expert_pack_geometry){
        .n_layer = FIXTURE_LAYER,
        .n_expert = FIXTURE_EXPERT,
        .n_embd = FIXTURE_DIM,
        .n_ff_exp = FIXTURE_DIM,
        .quant_type = DS4_QWEN_EXPERT_PACK_Q4_K_TYPE,
        .gguf_tensor_count = FIXTURE_TENSOR,
    };
}

static bool files_equal(const char *a_path, const char *b_path) {
    int a = open(a_path, O_RDONLY);
    int b = open(b_path, O_RDONLY);
    CHECK(a >= 0 && b >= 0);
    struct stat a_stat;
    struct stat b_stat;
    CHECK(fstat(a, &a_stat) == 0 && fstat(b, &b_stat) == 0);
    CHECK(a_stat.st_size == b_stat.st_size);
    uint8_t a_buffer[8192];
    uint8_t b_buffer[8192];
    off_t offset = 0;
    while (offset < a_stat.st_size) {
        size_t take = sizeof(a_buffer);
        if ((off_t)take > a_stat.st_size - offset) {
            take = (size_t)(a_stat.st_size - offset);
        }
        CHECK(pread(a, a_buffer, take, offset) == (ssize_t)take);
        CHECK(pread(b, b_buffer, take, offset) == (ssize_t)take);
        CHECK(memcmp(a_buffer, b_buffer, take) == 0);
        offset += (off_t)take;
    }
    CHECK(close(a) == 0 && close(b) == 0);
    return true;
}

static bool embed_file(
        const char *source_path,
        const char *container_path,
        uint64_t    offset,
        uint64_t   *source_bytes_out) {
    const int source = open(source_path, O_RDONLY);
    const int container = open(container_path, O_CREAT | O_TRUNC | O_RDWR, 0600);
    CHECK(source >= 0 && container >= 0);
    struct stat st;
    CHECK(fstat(source, &st) == 0 && st.st_size >= 0);
    CHECK(offset <= (uint64_t)INT64_MAX - (uint64_t)st.st_size);
    CHECK(ftruncate(container, (off_t)(offset + (uint64_t)st.st_size)) == 0);
    uint8_t buffer[8192];
    uint64_t copied = 0;
    while (copied < (uint64_t)st.st_size) {
        size_t take = sizeof(buffer);
        if ((uint64_t)take > (uint64_t)st.st_size - copied) {
            take = (size_t)((uint64_t)st.st_size - copied);
        }
        CHECK(pread(source, buffer, take, (off_t)copied) == (ssize_t)take);
        CHECK(pwrite(container, buffer, take,
                     (off_t)(offset + copied)) == (ssize_t)take);
        copied += take;
    }
    CHECK(fsync(container) == 0);
    CHECK(close(source) == 0 && close(container) == 0);
    if (source_bytes_out) *source_bytes_out = (uint64_t)st.st_size;
    return true;
}

static bool span_matches_source(
        int source_fd,
        uint64_t source_offset,
        int pack_fd,
        ds4_qwen_expert_pack_slice slice) {
    CHECK(slice.size == FIXTURE_EXPERT_BYTES);
    uint8_t source[4096];
    uint8_t packed[4096];
    uint64_t compared = 0;
    while (compared < slice.size) {
        size_t take = sizeof(source);
        if ((uint64_t)take > slice.size - compared) {
            take = (size_t)(slice.size - compared);
        }
        CHECK(pread(source_fd, source, take,
                    (off_t)(source_offset + compared)) == (ssize_t)take);
        CHECK(pread(pack_fd, packed, take,
                    (off_t)(slice.offset + compared)) == (ssize_t)take);
        CHECK(memcmp(source, packed, take) == 0);
        compared += take;
    }
    return true;
}

static void digest_hex(const uint8_t digest[32], char out[65]) {
    static const char hex[] = "0123456789abcdef";
    for (size_t i = 0; i < 32; i++) {
        out[i * 2] = hex[digest[i] >> 4];
        out[i * 2 + 1] = hex[digest[i] & 15];
    }
    out[64] = '\0';
}

static bool no_temporary_pack(const char *directory, const char *basename) {
    DIR *dir = opendir(directory);
    CHECK(dir != NULL);
    char prefix[256];
    const int count = snprintf(prefix, sizeof(prefix), "%s.tmp.", basename);
    CHECK(count > 0 && (size_t)count < sizeof(prefix));
    bool found = false;
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (strncmp(entry->d_name, prefix, strlen(prefix)) == 0) {
            found = true;
            break;
        }
    }
    closedir(dir);
    CHECK(!found);
    return true;
}

static bool test_pack_format_and_invalidation(void) {
    char directory[] = "/tmp/ds4-qwen-pack-test.XXXXXX";
    const int directory_seed = mkstemp(directory);
    CHECK(directory_seed >= 0);
    CHECK(close(directory_seed) == 0);
    CHECK(unlink(directory) == 0);
    CHECK(mkdir(directory, 0700) == 0);
    char gguf[512];
    char pack_a[512];
    char pack_b[512];
    char embedded_path[512];
    char native_gguf[512];
    CHECK(snprintf(gguf, sizeof(gguf), "%s/model.gguf", directory) > 0);
    CHECK(snprintf(pack_a, sizeof(pack_a), "%s/experts.pack", directory) > 0);
    CHECK(snprintf(pack_b, sizeof(pack_b), "%s/experts-copy.pack", directory) > 0);
    CHECK(snprintf(embedded_path, sizeof(embedded_path),
                   "%s/embedded.bin", directory) > 0);
    CHECK(snprintf(native_gguf, sizeof(native_gguf),
                   "%s/model.ds4.gguf", directory) > 0);

    fixture_layout fixture;
    CHECK(write_fixture_gguf(gguf, &fixture));
    char error[512] = {0};
    const ds4_qwen_expert_pack_build_options options = {
        .geometry = fixture_geometry(),
        .filesystem_reserve_bytes = 0,
    };
    CHECK(ds4_qwen_expert_pack_build(
        gguf, pack_a, &options, error, sizeof(error)));
    CHECK(ds4_qwen_expert_pack_build(
        gguf, pack_b, &options, error, sizeof(error)));
    CHECK(files_equal(pack_a, pack_b));

    const ds4_qwen_native_gguf_options native_options = {
        .geometry = fixture_geometry(),
        .filesystem_reserve_bytes = 0,
    };
    const bool native_built = ds4_qwen_native_gguf_build(
        gguf, pack_a, native_gguf, &native_options,
        error, sizeof(error));
    if (!native_built) fprintf(stderr, "native build: %s\n", error);
    CHECK(native_built);
    CHECK(ds4_qwen_native_gguf_verify(
        gguf, native_gguf, &native_options,
        error, sizeof(error)));
    struct stat native_stat;
    struct stat pack_stat;
    CHECK(stat(native_gguf, &native_stat) == 0);
    CHECK(stat(pack_a, &pack_stat) == 0);
    CHECK(native_stat.st_size > pack_stat.st_size);
    CHECK((uint64_t)(native_stat.st_size - pack_stat.st_size) < 16384);

    ds4_qwen_expert_pack_geometry wrong = fixture_geometry();
    wrong.n_expert++;
    ds4_qwen_expert_pack *pack = NULL;
    CHECK(ds4_qwen_expert_pack_open(
        &pack, pack_a, &wrong, error,
        sizeof(error)) == DS4_QWEN_EXPERT_PACK_FALLBACK);
    CHECK(pack == NULL);

    const ds4_qwen_expert_pack_geometry geometry = fixture_geometry();
    int format_fd = open(pack_b, O_RDWR);
    CHECK(format_fd >= 0);
    uint8_t format_byte = 0;
    CHECK(pread(format_fd, &format_byte, 1, 8) == 1);
    const uint8_t invalid_version = (uint8_t)(format_byte ^ 1);
    CHECK(pwrite(format_fd, &invalid_version, 1, 8) == 1);
    CHECK(fsync(format_fd) == 0);
    CHECK(close(format_fd) == 0);
    CHECK(ds4_qwen_expert_pack_open(
        &pack, pack_b, &geometry, error,
        sizeof(error)) == DS4_QWEN_EXPERT_PACK_FALLBACK);
    CHECK(pack == NULL);
    format_fd = open(pack_b, O_RDWR);
    CHECK(format_fd >= 0);
    CHECK(pwrite(format_fd, &format_byte, 1, 8) == 1);
    CHECK(fsync(format_fd) == 0);
    CHECK(close(format_fd) == 0);
    CHECK(files_equal(pack_a, pack_b));

    CHECK(ds4_qwen_expert_pack_open(
        &pack, pack_a, &geometry, error,
        sizeof(error)) == DS4_QWEN_EXPERT_PACK_OK);
    ds4_qwen_expert_pack_span span;
    CHECK(!ds4_qwen_expert_pack_span_get(pack, 0, 0, &span));
    const ds4_qwen_expert_pack_manifest *manifest =
        ds4_qwen_expert_pack_manifest_get(pack);
    CHECK(manifest != NULL);
    uint8_t wrong_payload_digest[32];
    memcpy(wrong_payload_digest, manifest->data_sha256,
           sizeof(wrong_payload_digest));
    wrong_payload_digest[0] ^= 0x80;
    CHECK(ds4_qwen_expert_pack_validate_payload_digest(
        pack, wrong_payload_digest, error,
        sizeof(error)) == DS4_QWEN_EXPERT_PACK_FALLBACK);
    CHECK(!ds4_qwen_expert_pack_span_get(pack, 0, 0, &span));
    CHECK(ds4_qwen_expert_pack_validate_source_file(
        pack, gguf, error,
        sizeof(error)) == DS4_QWEN_EXPERT_PACK_OK);
    CHECK(!ds4_qwen_expert_pack_span_get(pack, 0, 0, &span));
    CHECK(ds4_qwen_expert_pack_validate_payload_digest(
        pack, manifest->data_sha256, error,
        sizeof(error)) == DS4_QWEN_EXPERT_PACK_OK);
    CHECK(ds4_qwen_expert_pack_span_get(pack, 0, 0, &span));
    /* The offline rehash path remains independently covered; it resets and
     * reauthorizes the same gate after reading the actual payload bytes. */
    CHECK(ds4_qwen_expert_pack_verify_payload(
        pack, error, sizeof(error)) == DS4_QWEN_EXPERT_PACK_OK);
    CHECK(manifest->source_size == fixture.source_size);
    CHECK(manifest->entry_count == FIXTURE_LAYER * FIXTURE_EXPERT);
    CHECK(manifest->gate_bytes == FIXTURE_EXPERT_BYTES);
    CHECK(manifest->up_bytes == FIXTURE_EXPERT_BYTES);
    CHECK(manifest->down_bytes == FIXTURE_EXPERT_BYTES);
    char digest[65];
    digest_hex(manifest->source_sha256, digest);
    CHECK(strcmp(digest,
                 "fef0f83363ededa2c8030407b9ed6e471"
                 "9766afb4ac34a9f73d222ce86f2034b") == 0);
    digest_hex(manifest->data_sha256, digest);
    CHECK(strcmp(digest,
                 "6c56357797c1d55fd9a77278e7152c08"
                 "a8911eacab00f5a053e5ec5ec7fd8cb5") == 0);

    const int source_fd = open(gguf, O_RDONLY);
    CHECK(source_fd >= 0);
    for (uint32_t layer = 0; layer < FIXTURE_LAYER; layer++) {
        for (uint32_t expert = 0; expert < FIXTURE_EXPERT; expert++) {
            CHECK(ds4_qwen_expert_pack_span_get(pack, layer, expert, &span));
            CHECK(span.up.offset == span.gate.offset + span.gate.size);
            CHECK(span.down.offset == span.up.offset + span.up.size);
            CHECK(span_matches_source(
                source_fd, fixture.source_offset[layer][0][expert],
                ds4_qwen_expert_pack_fd(pack), span.gate));
            CHECK(span_matches_source(
                source_fd, fixture.source_offset[layer][1][expert],
                ds4_qwen_expert_pack_fd(pack), span.up));
            CHECK(span_matches_source(
                source_fd, fixture.source_offset[layer][2][expert],
                ds4_qwen_expert_pack_fd(pack), span.down));
        }
    }
    CHECK(close(source_fd) == 0);

    /* The exact same bytes can live inside a larger GGUF tensor. The reader
     * owns a duplicated descriptor and returns physical, container-relative
     * offsets so callers never need a second offset convention. */
    const uint64_t embedded_offset = 12345;
    uint64_t embedded_bytes = 0;
    CHECK(embed_file(pack_a, embedded_path, embedded_offset,
                     &embedded_bytes));
    const int embedded_fd = open(embedded_path, O_RDONLY);
    CHECK(embedded_fd >= 0);
    ds4_qwen_expert_pack *embedded = NULL;
    CHECK(ds4_qwen_expert_pack_open_embedded(
        &embedded, embedded_fd, embedded_offset, embedded_bytes,
        &geometry, error, sizeof(error)) == DS4_QWEN_EXPERT_PACK_OK);
    CHECK(ds4_qwen_expert_pack_file_offset(embedded) == embedded_offset);
    CHECK(ds4_qwen_expert_pack_validate_source_file(
        embedded, gguf, error,
        sizeof(error)) == DS4_QWEN_EXPERT_PACK_OK);
    CHECK(ds4_qwen_expert_pack_validate_payload_digest(
        embedded, manifest->data_sha256, error,
        sizeof(error)) == DS4_QWEN_EXPERT_PACK_OK);
    CHECK(ds4_qwen_expert_pack_span_get(embedded, 1, 2, &span));
    CHECK(span.gate.offset >= embedded_offset);
    const int embedded_source_fd = open(gguf, O_RDONLY);
    CHECK(embedded_source_fd >= 0);
    CHECK(span_matches_source(
        embedded_source_fd, fixture.source_offset[1][0][2],
        ds4_qwen_expert_pack_fd(embedded), span.gate));
    CHECK(close(embedded_source_fd) == 0);
    ds4_qwen_expert_pack_close(embedded);
    embedded = NULL;
    CHECK(ds4_qwen_expert_pack_open_embedded(
        &embedded, embedded_fd, embedded_offset, embedded_bytes + 1,
        &geometry, error, sizeof(error)) == DS4_QWEN_EXPERT_PACK_FALLBACK);
    CHECK(embedded == NULL);
    CHECK(close(embedded_fd) == 0);

    /* Source identity is content-based: same-size edits deauthorize spans, and
     * restoring the exact byte makes the canonical GGUF valid again. */
    int mutable_source = open(gguf, O_RDWR);
    CHECK(mutable_source >= 0);
    uint8_t original = 0;
    CHECK(pread(mutable_source, &original, 1,
                (off_t)fixture.source_offset[1][2][2]) == 1);
    const uint8_t changed = (uint8_t)(original ^ 0x5a);
    CHECK(pwrite(mutable_source, &changed, 1,
                 (off_t)fixture.source_offset[1][2][2]) == 1);
    CHECK(fsync(mutable_source) == 0);
    CHECK(ds4_qwen_expert_pack_validate_source_file(
        pack, gguf, error,
        sizeof(error)) == DS4_QWEN_EXPERT_PACK_FALLBACK);
    CHECK(!ds4_qwen_expert_pack_span_get(pack, 0, 0, &span));
    CHECK(pwrite(mutable_source, &original, 1,
                 (off_t)fixture.source_offset[1][2][2]) == 1);
    CHECK(fsync(mutable_source) == 0);
    CHECK(close(mutable_source) == 0);
    CHECK(ds4_qwen_expert_pack_validate_source_file(
        pack, gguf, error,
        sizeof(error)) == DS4_QWEN_EXPERT_PACK_OK);

    /* The aggregate payload checksum catches corruption without trusting the
     * offset table alone. */
    const uint64_t corrupt_offset = manifest->data_offset + 17;
    ds4_qwen_expert_pack_close(pack);
    pack = NULL;
    int mutable_pack = open(pack_a, O_RDWR);
    CHECK(mutable_pack >= 0);
    CHECK(pread(mutable_pack, &original, 1, (off_t)corrupt_offset) == 1);
    const uint8_t corrupt = (uint8_t)(original ^ 0xa5);
    CHECK(pwrite(mutable_pack, &corrupt, 1, (off_t)corrupt_offset) == 1);
    CHECK(fsync(mutable_pack) == 0);
    CHECK(close(mutable_pack) == 0);
    CHECK(ds4_qwen_expert_pack_open(
        &pack, pack_a, &geometry, error,
        sizeof(error)) == DS4_QWEN_EXPERT_PACK_OK);
    CHECK(ds4_qwen_expert_pack_verify_payload(
        pack, error, sizeof(error)) == DS4_QWEN_EXPERT_PACK_FALLBACK);
    ds4_qwen_expert_pack_close(pack);
    pack = NULL;
    mutable_pack = open(pack_a, O_RDWR);
    CHECK(mutable_pack >= 0);
    CHECK(pwrite(mutable_pack, &original, 1, (off_t)corrupt_offset) == 1);
    CHECK(fsync(mutable_pack) == 0);
    CHECK(close(mutable_pack) == 0);
    CHECK(files_equal(pack_a, pack_b));

    ds4_qwen_expert_pack_build_options no_space = options;
    no_space.filesystem_reserve_bytes = UINT64_MAX;
    CHECK(!ds4_qwen_expert_pack_build(
        gguf, pack_a, &no_space, error, sizeof(error)));
    CHECK(strstr(error, "insufficient free space") != NULL);
    CHECK(files_equal(pack_a, pack_b));

    /* A real mid-build filesystem failure leaves the old destination byte for
     * byte intact and removes its private temporary file. */
    const pid_t child = fork();
    CHECK(child >= 0);
    if (child == 0) {
        signal(SIGXFSZ, SIG_IGN);
        struct rlimit limit;
        if (getrlimit(RLIMIT_FSIZE, &limit) != 0) _exit(10);
        limit.rlim_cur = 1024;
        if (setrlimit(RLIMIT_FSIZE, &limit) != 0) _exit(11);
        char child_error[512] = {0};
        const bool built = ds4_qwen_expert_pack_build(
            gguf, pack_a, &options, child_error, sizeof(child_error));
        _exit(built ? 12 : 0);
    }
    int status = 0;
    CHECK(waitpid(child, &status, 0) == child);
    CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    CHECK(files_equal(pack_a, pack_b));
    CHECK(no_temporary_pack(directory, "experts.pack"));

    CHECK(unlink(pack_b) == 0);
    CHECK(unlink(pack_a) == 0);
    CHECK(unlink(embedded_path) == 0);
    CHECK(unlink(native_gguf) == 0);
    CHECK(unlink(gguf) == 0);
    CHECK(rmdir(directory) == 0);
    return true;
}

int main(void) {
    if (!test_pack_format_and_invalidation()) return 1;
    puts("Qwen expert pack tests: OK");
    return 0;
}
