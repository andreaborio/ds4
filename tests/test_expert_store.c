#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include "ds4_expert_store.h"

#include <fcntl.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
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
    if (argc != 5 && argc != 7 && argc != 10 && argc != 11) {
        fprintf(stderr,
                "usage: %s FILE STORE_OFFSET STORE_BYTES FAMILY "
                "[STORAGE GROUP] | "
                "[STORAGE LAYERS EXPERTS USED SOURCE_TENSORS] | "
                "[STORAGE GROUP LAYERS EXPERTS USED SOURCE_TENSORS]\n",
                argv[0]);
        return 2;
    }
    uint64_t offset = 0, bytes = 0, family64 = 0;
    uint64_t storage64 = DS4_EXPERT_STORE_STORAGE_GGML;
    uint64_t group64 = 0;
    CHECK(parse_u64(argv[2], &offset));
    CHECK(parse_u64(argv[3], &bytes));
    CHECK(parse_u64(argv[4], &family64));
    uint64_t layers64 = 2, experts64 = 3, used64 = 2, source_tensors = 7;
    if (argc == 7 || argc == 11) {
        CHECK(parse_u64(argv[5], &storage64));
        CHECK(parse_u64(argv[6], &group64));
    } else if (argc == 10) {
        CHECK(parse_u64(argv[5], &storage64));
        group64 = storage64 == DS4_EXPERT_STORE_STORAGE_MLX_AFFINE2 ?
            (family64 == DS4_EXPERT_STORE_FAMILY_DEEPSEEK4 ?
                DS4_EXPERT_STORE_GROUP_PROFILE_AFFINE2_G32_U64_D64 :
                DS4_EXPERT_STORE_GLM_AFFINE2_GROUP_SIZE) :
            (storage64 == DS4_EXPERT_STORE_STORAGE_MLX_AFFINE4 ? 64u : 0u);
    }
    if (argc == 10 || argc == 11) {
        const int base = argc == 11 ? 7 : 6;
        CHECK(parse_u64(argv[base], &layers64));
        CHECK(parse_u64(argv[base + 1], &experts64));
        CHECK(parse_u64(argv[base + 2], &used64));
        CHECK(parse_u64(argv[base + 3], &source_tensors));
    }
    CHECK(family_is_supported(family64));
    const uint32_t family = (uint32_t)family64;
    CHECK(storage64 <= UINT32_MAX);
    CHECK(layers64 <= UINT32_MAX);
    CHECK(experts64 <= UINT32_MAX);
    CHECK(used64 <= UINT32_MAX);
    const int fd = open(argv[1], O_RDONLY);
    CHECK(fd >= 0);

    char error[256] = {0};
    ds4_expert_store *store = NULL;
    CHECK(ds4_expert_store_open_embedded(
        &store, fd, offset, bytes, family,
        error, sizeof(error)));
    CHECK(store != NULL);
    const ds4_expert_store_manifest *manifest =
        ds4_expert_store_manifest_get(store);
    CHECK(manifest != NULL);
    CHECK(manifest->version == DS4_EXPERT_STORE_V2_VERSION);
    CHECK(manifest->family == family);
    CHECK(manifest->storage_format == storage64);
    CHECK(manifest->group_size == group64);
    CHECK(manifest->layer_count == (uint32_t)layers64);
    CHECK(manifest->expert_count == (uint32_t)experts64);
    CHECK(manifest->expert_used_count == (uint32_t)used64);
    CHECK(manifest->source_tensor_count == source_tensors);
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
