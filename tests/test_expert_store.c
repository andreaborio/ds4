#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include "ds4_expert_store.h"

#include <fcntl.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", \
                __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static bool parse_u64(const char *text, uint64_t *out) {
    char *end = NULL;
    const unsigned long long value = strtoull(text, &end, 10);
    if (!text[0] || !end || *end) return false;
    *out = (uint64_t)value;
    return true;
}

static bool family_is_supported(uint64_t family) {
    return family == DS4_EXPERT_STORE_FAMILY_DEEPSEEK4 ||
           family == DS4_EXPERT_STORE_FAMILY_GLM_DSA ||
           family == DS4_EXPERT_STORE_FAMILY_QWEN35_MOE;
}

int main(int argc, char **argv) {
    if (argc != 5 && argc != 7 && argc != 8) {
        fprintf(stderr,
                "usage: %s NATIVE.gguf STORE_OFFSET STORE_BYTES FAMILY "
                "[STORAGE_FORMAT GROUP_SIZE "
                "[dspark-0731|dspark-0731-reject|store-reject]]\n",
                argv[0]);
        return 2;
    }
    uint64_t offset = 0, bytes = 0, family64 = 0;
    CHECK(parse_u64(argv[2], &offset));
    CHECK(parse_u64(argv[3], &bytes));
    CHECK(parse_u64(argv[4], &family64));
    CHECK(family_is_supported(family64));
    const uint32_t family = (uint32_t)family64;
    uint64_t storage64 = DS4_EXPERT_STORE_STORAGE_GGML;
    uint64_t group64 = 0;
    if (argc >= 7) {
        CHECK(parse_u64(argv[5], &storage64));
        CHECK(parse_u64(argv[6], &group64));
        CHECK(storage64 <= UINT32_MAX);
        CHECK(group64 <= UINT32_MAX);
    }
    const bool validate_dspark =
        argc == 8 && strcmp(argv[7], "dspark-0731") == 0;
    const bool reject_dspark =
        argc == 8 && strcmp(argv[7], "dspark-0731-reject") == 0;
    const bool reject_store =
        argc == 8 && strcmp(argv[7], "store-reject") == 0;
    if (argc == 8)
        CHECK(validate_dspark || reject_dspark || reject_store);
    const int fd = open(argv[1], O_RDONLY);
    CHECK(fd >= 0);

    char error[256] = {0};
    ds4_expert_store *store = NULL;
    const bool opened = ds4_expert_store_open_embedded(
        &store, fd, offset, bytes, family, error, sizeof(error));
    if (reject_store) {
        CHECK(!opened);
        CHECK(store == NULL);
        CHECK(error[0] != '\0');
        CHECK(close(fd) == 0);
        puts("expert-store v2 C reader rejection: OK");
        return 0;
    }
    CHECK(opened);
    CHECK(store != NULL);
    const ds4_expert_store_manifest *manifest =
        ds4_expert_store_manifest_get(store);
    CHECK(manifest != NULL);
    CHECK(manifest->version == DS4_EXPERT_STORE_V2_VERSION);
    CHECK(manifest->family == family);
    CHECK(manifest->storage_format == (uint32_t)storage64);
    CHECK(manifest->group_size == (uint32_t)group64);
    if (argc == 5) {
        CHECK(manifest->layer_count == 2);
        CHECK(manifest->expert_count == 3);
        CHECK(manifest->expert_used_count == 2);
        CHECK(manifest->source_tensor_count == 7);
    }
    if (validate_dspark) {
        CHECK(ds4_expert_store_validate_dspark_0731(
            store, error, sizeof(error)));
    } else if (reject_dspark) {
        CHECK(!ds4_expert_store_validate_dspark_0731(
            store, error, sizeof(error)));
        CHECK(error[0] != '\0');
    }
    CHECK(ds4_expert_store_fd(store) >= 0);
    CHECK(ds4_expert_store_file_offset(store) == offset);

    for (uint32_t index = 0; index < manifest->layer_count; index++) {
        const ds4_expert_store_layer *entry =
            ds4_expert_store_layer_at(store, index);
        CHECK(entry != NULL);
        const uint32_t expected_layer =
            family == DS4_EXPERT_STORE_FAMILY_GLM_DSA ? index + 3u : index;
        CHECK(entry->layer == expected_layer);
        CHECK(ds4_expert_store_layer_get(store, expected_layer) == entry);
        CHECK(entry->expert_count == manifest->expert_count);
        for (uint32_t expert = 0; expert < manifest->expert_count; expert++) {
            uint64_t previous_end = 0;
            for (uint32_t role = 0; role < 3; role++) {
                uint64_t slice_offset = 0, slice_bytes = 0;
                CHECK(ds4_expert_store_slice_get(
                    store, expected_layer, expert, role,
                    &slice_offset, &slice_bytes));
                CHECK(slice_bytes == entry->component[role].expert_bytes);
                if (role != 0) CHECK(slice_offset == previous_end);
                previous_end = slice_offset + slice_bytes;
            }
        }
    }
    ds4_expert_store_close(store);
    store = NULL;

    const uint32_t families[] = {
        DS4_EXPERT_STORE_FAMILY_DEEPSEEK4,
        DS4_EXPERT_STORE_FAMILY_GLM_DSA,
        DS4_EXPERT_STORE_FAMILY_QWEN35_MOE,
        999,
    };
    for (size_t index = 0; index < sizeof(families) / sizeof(families[0]);
         index++) {
        if (families[index] == family) continue;
        error[0] = '\0';
        CHECK(!ds4_expert_store_open_embedded(
            &store, fd, offset, bytes, families[index],
            error, sizeof(error)));
        CHECK(store == NULL);
        CHECK(error[0] != '\0');
    }
    CHECK(close(fd) == 0);
    puts("expert-store v2 C reader: OK");
    return 0;
}
