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

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s NATIVE.gguf STORE_OFFSET STORE_BYTES\n",
                argv[0]);
        return 2;
    }
    uint64_t offset = 0, bytes = 0;
    CHECK(parse_u64(argv[2], &offset));
    CHECK(parse_u64(argv[3], &bytes));
    const int fd = open(argv[1], O_RDONLY);
    CHECK(fd >= 0);

    char error[256] = {0};
    ds4_expert_store *store = NULL;
    CHECK(ds4_expert_store_open_embedded(
        &store, fd, offset, bytes, DS4_EXPERT_STORE_FAMILY_DEEPSEEK4,
        error, sizeof(error)));
    CHECK(store != NULL);
    const ds4_expert_store_manifest *manifest =
        ds4_expert_store_manifest_get(store);
    CHECK(manifest != NULL);
    CHECK(manifest->version == DS4_EXPERT_STORE_V2_VERSION);
    CHECK(manifest->family == DS4_EXPERT_STORE_FAMILY_DEEPSEEK4);
    CHECK(manifest->layer_count == 2);
    CHECK(manifest->expert_count == 3);
    CHECK(manifest->expert_used_count == 2);
    CHECK(manifest->source_tensor_count == 7);
    CHECK(ds4_expert_store_fd(store) >= 0);
    CHECK(ds4_expert_store_file_offset(store) == offset);

    for (uint32_t layer = 0; layer < manifest->layer_count; layer++) {
        const ds4_expert_store_layer *entry =
            ds4_expert_store_layer_get(store, layer);
        CHECK(entry != NULL);
        CHECK(entry->layer == layer);
        CHECK(entry->expert_count == manifest->expert_count);
        for (uint32_t expert = 0; expert < manifest->expert_count; expert++) {
            uint64_t previous_end = 0;
            for (uint32_t role = 0; role < 3; role++) {
                uint64_t slice_offset = 0, slice_bytes = 0;
                CHECK(ds4_expert_store_slice_get(
                    store, layer, expert, role, &slice_offset, &slice_bytes));
                CHECK(slice_bytes == entry->component[role].expert_bytes);
                if (role != 0) CHECK(slice_offset == previous_end);
                previous_end = slice_offset + slice_bytes;
            }
        }
    }
    ds4_expert_store_close(store);
    store = NULL;

    error[0] = '\0';
    CHECK(!ds4_expert_store_open_embedded(
        &store, fd, offset, bytes, 999, error, sizeof(error)));
    CHECK(store == NULL);
    CHECK(close(fd) == 0);
    puts("expert-store v2 C reader: OK");
    return 0;
}
