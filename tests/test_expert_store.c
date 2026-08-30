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
    TEST_STORE_HEADER_BYTES = 256,
    TEST_STORE_LAYER_BYTES = 224,
    TEST_STORE_COMPONENT_BYTES = 56,
    TEST_STORE_COMPONENT_OFFSET = 32,
    TEST_STORE_ALIGNMENT = 4096,
    TEST_STORE_DIGEST_OFFSET = 168,
    TEST_QWEN4EXP_LAYERS = 48,
    TEST_QWEN4EXP_EXPERTS = 512,
};

typedef struct {
    uint32_t state[8];
    uint64_t bytes;
    uint8_t block[64];
    size_t block_len;
} test_sha256_context;

static uint32_t test_rotr32(uint32_t value, unsigned shift) {
    return (value >> shift) | (value << (32u - shift));
}

static void test_sha256_transform(
        test_sha256_context *context,
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

static void test_sha256_update(
        test_sha256_context *context,
        const void *data_pointer,
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
            test_sha256_transform(context, context->block);
            context->block_len = 0;
        }
    }
}

static void test_sha256_final(
        test_sha256_context *context,
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

static void store_u32_le(uint8_t *p, uint32_t value) {
    for (size_t i = 0; i < 4; i++) p[i] = (uint8_t)(value >> (8u * i));
}

static void store_u64_le(uint8_t *p, uint64_t value) {
    for (size_t i = 0; i < 8; i++) p[i] = (uint8_t)(value >> (8u * i));
}

static bool pwrite_all(int fd, const void *src_pointer, size_t size,
                       uint64_t offset) {
    const uint8_t *src = src_pointer;
    while (size != 0) {
        const ssize_t written = pwrite(fd, src, size, (off_t)offset);
        if (written <= 0) return false;
        src += (size_t)written;
        size -= (size_t)written;
        offset += (uint64_t)written;
    }
    return true;
}

typedef enum {
    TEST_STORE_VALID,
    TEST_STORE_BAD_LAYER_COUNT,
    TEST_STORE_BAD_EXPERT_COUNT,
    TEST_STORE_BAD_EXPERT_USED,
    TEST_STORE_BAD_LAYER_ORDER,
    TEST_STORE_BAD_STORAGE,
    TEST_STORE_BAD_GEOMETRY,
    TEST_STORE_BAD_ROLE_ORDER,
    TEST_STORE_BAD_BLOCK,
    TEST_STORE_BAD_MANIFEST_DIGEST,
    TEST_STORE_COMPONENT_OVERFLOW,
} test_store_variant;

static int make_qwen4exp_sparse_store(
        test_store_variant variant,
        int *out_fd,
        uint64_t *out_offset,
        uint64_t *out_bytes) {
    static const uint8_t magic[8] = {
        'D', 'S', '4', 'E', 'X', 'P', 'V', '2'
    };
    uint8_t header[TEST_STORE_HEADER_BYTES] = {0};
    uint8_t descriptors[
        TEST_QWEN4EXP_LAYERS * TEST_STORE_LAYER_BYTES] = {0};
    const uint64_t hidden =
        variant == TEST_STORE_BAD_GEOMETRY ? 2304u : 2560u;
    const uint64_t expert_width = 640u;
    const uint64_t gate_bytes =
        (hidden / 64u) * 36u * expert_width;
    const uint64_t down_bytes =
        (expert_width / 64u) * 36u * hidden;
    const uint64_t record_bytes = gate_bytes * 2u + down_bytes;
    const uint64_t layer_bytes = record_bytes * TEST_QWEN4EXP_EXPERTS;
    const uint64_t descriptor_bytes = variant == TEST_STORE_BAD_LAYER_COUNT
        ? (TEST_QWEN4EXP_LAYERS - 1u) * TEST_STORE_LAYER_BYTES
        : sizeof(descriptors);
    const uint64_t data_offset = TEST_STORE_ALIGNMENT * 3u;
    const uint64_t data_size = layer_bytes * TEST_QWEN4EXP_LAYERS;
    const uint64_t store_size = data_offset + data_size;
    const uint64_t embedded_offset = TEST_STORE_ALIGNMENT;

    memcpy(header, magic, sizeof(magic));
    store_u32_le(header + 8, DS4_EXPERT_STORE_V2_VERSION);
    store_u32_le(header + 12, TEST_STORE_HEADER_BYTES);
    store_u32_le(header + 16, DS4_EXPERT_STORE_FAMILY_QWEN4EXP);
    store_u32_le(header + 20,
                 variant == TEST_STORE_BAD_EXPERT_USED ? 9u : 10u);
    store_u32_le(header + 24, variant == TEST_STORE_BAD_LAYER_COUNT
                 ? TEST_QWEN4EXP_LAYERS - 1u : TEST_QWEN4EXP_LAYERS);
    store_u32_le(header + 28, variant == TEST_STORE_BAD_EXPERT_COUNT
                 ? TEST_QWEN4EXP_EXPERTS - 1u : TEST_QWEN4EXP_EXPERTS);
    store_u64_le(header + 32, 1294u);
    store_u64_le(header + 40, variant == TEST_STORE_BAD_LAYER_COUNT
                 ? TEST_QWEN4EXP_LAYERS - 1u : TEST_QWEN4EXP_LAYERS);
    store_u64_le(header + 48, descriptor_bytes);
    store_u64_le(header + 56, TEST_STORE_HEADER_BYTES);
    store_u64_le(header + 64, data_offset);
    store_u64_le(header + 72, data_size);
    store_u64_le(header + 80, store_size);
    store_u64_le(header + 88, UINT64_C(359999963128));
    store_u32_le(header + 160, DS4_EXPERT_STORE_STORAGE_MLX_AFFINE4);
    store_u32_le(header + 164, 64u);
    if (variant == TEST_STORE_BAD_STORAGE) {
        store_u32_le(header + 160, DS4_EXPERT_STORE_STORAGE_GGML);
        store_u32_le(header + 164, 0u);
    }

    uint64_t layer_offset = data_offset;
    for (uint32_t layer = 0; layer < TEST_QWEN4EXP_LAYERS; layer++) {
        uint8_t *entry = descriptors + layer * TEST_STORE_LAYER_BYTES;
        store_u32_le(entry, layer);
        store_u32_le(entry + 4, TEST_QWEN4EXP_EXPERTS);
        store_u64_le(entry + 8, record_bytes);
        store_u64_le(entry + 16, layer_offset);
        store_u64_le(entry + 24, layer_bytes);
        uint64_t record_offset = 0;
        for (uint32_t role = 0; role < DS4_EXPERT_STORE_V2_COMPONENTS;
             role++) {
            uint8_t *component = entry + TEST_STORE_COMPONENT_OFFSET +
                role * TEST_STORE_COMPONENT_BYTES;
            const uint64_t dim0 = role == DS4_EXPERT_STORE_DOWN
                ? expert_width : hidden;
            const uint64_t dim1 = role == DS4_EXPERT_STORE_DOWN
                ? hidden : expert_width;
            const uint64_t component_bytes = role == DS4_EXPERT_STORE_DOWN
                ? down_bytes : gate_bytes;
            store_u32_le(component, role);
            store_u32_le(component + 4, 12u);
            store_u32_le(component + 8, 3u);
            store_u32_le(component + 12, 64u);
            store_u64_le(component + 16, dim0);
            store_u64_le(component + 24, dim1);
            store_u64_le(component + 32, TEST_QWEN4EXP_EXPERTS);
            store_u64_le(component + 40, component_bytes);
            store_u64_le(component + 48, record_offset);
            record_offset += component_bytes;
        }
        layer_offset += layer_bytes;
    }

    if (variant == TEST_STORE_BAD_LAYER_ORDER) {
        store_u32_le(descriptors + TEST_STORE_LAYER_BYTES, 2u);
    } else if (variant == TEST_STORE_BAD_ROLE_ORDER) {
        store_u32_le(descriptors + TEST_STORE_COMPONENT_OFFSET, 1u);
    } else if (variant == TEST_STORE_BAD_BLOCK) {
        store_u32_le(descriptors + TEST_STORE_COMPONENT_OFFSET + 12, 256u);
    } else if (variant == TEST_STORE_COMPONENT_OVERFLOW) {
        store_u64_le(descriptors + TEST_STORE_COMPONENT_OFFSET + 16,
                     UINT64_MAX - 63u);
    }

    test_sha256_context hash;
    uint8_t digest[32];
    test_sha256_init(&hash);
    test_sha256_update(&hash, header, sizeof(header));
    test_sha256_update(&hash, descriptors, (size_t)descriptor_bytes);
    test_sha256_final(&hash, digest);
    memcpy(header + TEST_STORE_DIGEST_OFFSET, digest, sizeof(digest));
    if (variant == TEST_STORE_BAD_MANIFEST_DIGEST) descriptors[9] ^= 0x80u;

    char path[] = "/tmp/ds4-qwen4exp-store-XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0) return -1;
    (void)unlink(path);
    if (embedded_offset + store_size > (uint64_t)INT64_MAX ||
        ftruncate(fd, (off_t)(embedded_offset + store_size)) != 0 ||
        !pwrite_all(fd, header, sizeof(header), embedded_offset) ||
        !pwrite_all(fd, descriptors, sizeof(descriptors),
                    embedded_offset + TEST_STORE_HEADER_BYTES)) {
        close(fd);
        return -1;
    }
    *out_fd = fd;
    *out_offset = embedded_offset;
    *out_bytes = store_size;
    return 0;
}

static int expect_qwen4exp_rejection(test_store_variant variant) {
    int fd = -1;
    uint64_t offset = 0, bytes = 0;
    if (make_qwen4exp_sparse_store(
            variant, &fd, &offset, &bytes) != 0) return -1;
    ds4_expert_store *store = (ds4_expert_store *)(uintptr_t)1u;
    char error[256] = {0};
    const bool opened = ds4_expert_store_open_embedded(
        &store, fd, offset, bytes, DS4_EXPERT_STORE_FAMILY_QWEN4EXP,
        error, sizeof(error));
    if (opened || store != NULL || error[0] == '\0') {
        if (opened) ds4_expert_store_close(store);
        close(fd);
        return -1;
    }
    close(fd);
    return 0;
}

static int test_qwen35_affine_legacy_store(void) {
    static const uint8_t magic[8] = {
        'D', 'S', '4', 'E', 'X', 'P', 'V', '2'
    };
    enum { layers = 2, experts = 3 };
    uint8_t header[TEST_STORE_HEADER_BYTES] = {0};
    uint8_t descriptors[layers * TEST_STORE_LAYER_BYTES] = {0};
    const uint64_t component_bytes = 144u * 256u;
    const uint64_t record_bytes = component_bytes * 3u;
    const uint64_t layer_bytes = record_bytes * experts;
    const uint64_t data_offset = TEST_STORE_ALIGNMENT;
    const uint64_t store_size = data_offset + layer_bytes * layers;

    memcpy(header, magic, sizeof(magic));
    store_u32_le(header + 8, DS4_EXPERT_STORE_V2_VERSION);
    store_u32_le(header + 12, TEST_STORE_HEADER_BYTES);
    store_u32_le(header + 16, DS4_EXPERT_STORE_FAMILY_QWEN35_MOE);
    store_u32_le(header + 20, 2u);
    store_u32_le(header + 24, layers);
    store_u32_le(header + 28, experts);
    store_u64_le(header + 32, 7u);
    store_u64_le(header + 40, layers);
    store_u64_le(header + 48, sizeof(descriptors));
    store_u64_le(header + 56, TEST_STORE_HEADER_BYTES);
    store_u64_le(header + 64, data_offset);
    store_u64_le(header + 72, store_size - data_offset);
    store_u64_le(header + 80, store_size);
    store_u64_le(header + 88, 1u);
    store_u32_le(header + 160, DS4_EXPERT_STORE_STORAGE_MLX_AFFINE4);
    store_u32_le(header + 164, 64u);

    uint64_t layer_offset = data_offset;
    for (uint32_t layer = 0; layer < layers; layer++) {
        uint8_t *entry = descriptors + layer * TEST_STORE_LAYER_BYTES;
        store_u32_le(entry, layer);
        store_u32_le(entry + 4, experts);
        store_u64_le(entry + 8, record_bytes);
        store_u64_le(entry + 16, layer_offset);
        store_u64_le(entry + 24, layer_bytes);
        for (uint32_t role = 0; role < DS4_EXPERT_STORE_V2_COMPONENTS;
             role++) {
            uint8_t *component = entry + TEST_STORE_COMPONENT_OFFSET +
                role * TEST_STORE_COMPONENT_BYTES;
            store_u32_le(component, role);
            store_u32_le(component + 4, 12u);
            store_u32_le(component + 8, 3u);
            store_u32_le(component + 12, 256u);
            store_u64_le(component + 16, 256u);
            store_u64_le(component + 24, 256u);
            store_u64_le(component + 32, experts);
            store_u64_le(component + 40, component_bytes);
            store_u64_le(component + 48, role * component_bytes);
        }
        layer_offset += layer_bytes;
    }

    test_sha256_context hash;
    uint8_t digest[32];
    test_sha256_init(&hash);
    test_sha256_update(&hash, header, sizeof(header));
    test_sha256_update(&hash, descriptors, sizeof(descriptors));
    test_sha256_final(&hash, digest);
    memcpy(header + TEST_STORE_DIGEST_OFFSET, digest, sizeof(digest));

    char path[] = "/tmp/ds4-qwen35-affine-store-XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0) return -1;
    (void)unlink(path);
    if (ftruncate(fd, (off_t)store_size) != 0 ||
        !pwrite_all(fd, header, sizeof(header), 0u) ||
        !pwrite_all(fd, descriptors, sizeof(descriptors),
                    TEST_STORE_HEADER_BYTES)) {
        close(fd);
        return -1;
    }
    ds4_expert_store *store = NULL;
    char error[256] = {0};
    if (!ds4_expert_store_open_embedded(
            &store, fd, 0u, store_size,
            DS4_EXPERT_STORE_FAMILY_QWEN35_MOE,
            error, sizeof(error))) {
        close(fd);
        return -1;
    }
    const ds4_expert_store_layer *entry =
        ds4_expert_store_layer_get(store, 0u);
    const ds4_expert_store_manifest *manifest =
        ds4_expert_store_manifest_get(store);
    const bool valid = entry && manifest &&
        manifest->storage_format == DS4_EXPERT_STORE_STORAGE_MLX_AFFINE4 &&
        manifest->group_size == 64u &&
        entry->component[DS4_EXPERT_STORE_GATE].block_elements == 256u;
    ds4_expert_store_close(store);
    close(fd);
    return valid ? 0 : -1;
}

static int test_qwen4exp_sparse_store(void) {
    int fd = -1;
    uint64_t offset = 0, bytes = 0;
    if (make_qwen4exp_sparse_store(
            TEST_STORE_VALID, &fd, &offset, &bytes) != 0) return -1;

    struct stat st;
    if (fstat(fd, &st) != 0 || (uint64_t)st.st_size != offset + bytes ||
        st.st_blocks < 0 || (uint64_t)st.st_blocks * 512u >= bytes / 1024u) {
        close(fd);
        return -1;
    }
    ds4_expert_store *store = NULL;
    char error[256] = {0};
    if (!ds4_expert_store_open_embedded(
            &store, fd, offset, bytes,
            DS4_EXPERT_STORE_FAMILY_QWEN4EXP,
            error, sizeof(error))) {
        close(fd);
        return -1;
    }
    const ds4_expert_store_manifest *manifest =
        ds4_expert_store_manifest_get(store);
    if (!manifest || manifest->layer_count != TEST_QWEN4EXP_LAYERS ||
        manifest->expert_count != TEST_QWEN4EXP_EXPERTS ||
        manifest->expert_used_count != 10u ||
        manifest->storage_format != DS4_EXPERT_STORE_STORAGE_MLX_AFFINE4 ||
        manifest->group_size != 64u) {
        ds4_expert_store_close(store);
        close(fd);
        return -1;
    }
    const ds4_expert_store_layer *first =
        ds4_expert_store_layer_get(store, 0u);
    const ds4_expert_store_layer *last =
        ds4_expert_store_layer_get(store, TEST_QWEN4EXP_LAYERS - 1u);
    if (!first || !last || last->layer != TEST_QWEN4EXP_LAYERS - 1u ||
        ds4_expert_store_layer_at(store, TEST_QWEN4EXP_LAYERS) != NULL ||
        ds4_expert_store_layer_get(store, TEST_QWEN4EXP_LAYERS) != NULL ||
        first->component[DS4_EXPERT_STORE_GATE].dim[0] != 2560u ||
        first->component[DS4_EXPERT_STORE_GATE].dim[1] != 640u ||
        first->component[DS4_EXPERT_STORE_GATE].block_elements != 64u ||
        first->component[DS4_EXPERT_STORE_DOWN].dim[0] != 640u ||
        first->component[DS4_EXPERT_STORE_DOWN].dim[1] != 2560u) {
        ds4_expert_store_close(store);
        close(fd);
        return -1;
    }

    uint64_t gate_offset = 0, gate_bytes = 0;
    uint64_t up_offset = 0, up_bytes = 0;
    uint64_t down_offset = 0, down_bytes = 0;
    if (!ds4_expert_store_slice_get(
            store, 0u, 0u, DS4_EXPERT_STORE_GATE,
            &gate_offset, &gate_bytes) ||
        !ds4_expert_store_slice_get(
            store, 0u, 0u, DS4_EXPERT_STORE_UP,
            &up_offset, &up_bytes) ||
        !ds4_expert_store_slice_get(
            store, TEST_QWEN4EXP_LAYERS - 1u,
            TEST_QWEN4EXP_EXPERTS - 1u, DS4_EXPERT_STORE_DOWN,
            &down_offset, &down_bytes) ||
        gate_offset != offset + manifest->data_offset ||
        up_offset != gate_offset + gate_bytes ||
        down_offset + down_bytes != offset + manifest->store_size) {
        ds4_expert_store_close(store);
        close(fd);
        return -1;
    }
    uint64_t rejected_offset = 123u, rejected_bytes = 456u;
    if (ds4_expert_store_slice_get(
            store, 0u, TEST_QWEN4EXP_EXPERTS,
            DS4_EXPERT_STORE_GATE, &rejected_offset, &rejected_bytes) ||
        rejected_offset != 0u || rejected_bytes != 0u) {
        ds4_expert_store_close(store);
        close(fd);
        return -1;
    }
    rejected_offset = 123u;
    rejected_bytes = 456u;
    if (ds4_expert_store_slice_get(
            store, TEST_QWEN4EXP_LAYERS, 0u,
            DS4_EXPERT_STORE_GATE, &rejected_offset, &rejected_bytes) ||
        rejected_offset != 0u || rejected_bytes != 0u) {
        ds4_expert_store_close(store);
        close(fd);
        return -1;
    }
    rejected_offset = 123u;
    rejected_bytes = 456u;
    if (ds4_expert_store_slice_get(
            store, 0u, 0u, DS4_EXPERT_STORE_V2_COMPONENTS,
            &rejected_offset, &rejected_bytes) ||
        rejected_offset != 0u || rejected_bytes != 0u) {
        ds4_expert_store_close(store);
        close(fd);
        return -1;
    }
    ds4_expert_store *wrong_family =
        (ds4_expert_store *)(uintptr_t)1u;
    error[0] = '\0';
    if (ds4_expert_store_open_embedded(
            &wrong_family, fd, offset, bytes,
            DS4_EXPERT_STORE_FAMILY_QWEN35_MOE,
            error, sizeof(error)) || wrong_family != NULL ||
        error[0] == '\0') {
        ds4_expert_store_close(store);
        close(fd);
        return -1;
    }
    ds4_expert_store_close(store);
    close(fd);

    const test_store_variant rejected[] = {
        TEST_STORE_BAD_LAYER_COUNT,
        TEST_STORE_BAD_EXPERT_COUNT,
        TEST_STORE_BAD_EXPERT_USED,
        TEST_STORE_BAD_LAYER_ORDER,
        TEST_STORE_BAD_STORAGE,
        TEST_STORE_BAD_GEOMETRY,
        TEST_STORE_BAD_ROLE_ORDER,
        TEST_STORE_BAD_BLOCK,
        TEST_STORE_BAD_MANIFEST_DIGEST,
        TEST_STORE_COMPONENT_OVERFLOW,
    };
    for (size_t i = 0; i < sizeof(rejected) / sizeof(rejected[0]); i++) {
        if (expect_qwen4exp_rejection(rejected[i]) != 0) return -1;
    }

    store = (ds4_expert_store *)(uintptr_t)1u;
    error[0] = '\0';
    if (ds4_expert_store_open_embedded(
            &store, 0, UINT64_MAX - 127u, 256u,
            DS4_EXPERT_STORE_FAMILY_QWEN4EXP,
            error, sizeof(error)) ||
        store != NULL || error[0] == '\0') {
        return -1;
    }
    return 0;
}

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
           family == DS4_EXPERT_STORE_FAMILY_QWEN35_MOE ||
           family == DS4_EXPERT_STORE_FAMILY_QWEN4EXP;
}

int main(int argc, char **argv) {
    CHECK(DS4_EXPERT_STORE_V2_MAX_EXPERTS == 512);
    CHECK(test_qwen35_affine_legacy_store() == 0);
    CHECK(test_qwen4exp_sparse_store() == 0);
    if (argc == 1) {
        puts("expert-store v2 Qwen4Exp sparse reader: OK");
        return 0;
    }
    if (argc != 5 && argc != 7) {
        fprintf(stderr,
                "usage: %s NATIVE.gguf STORE_OFFSET STORE_BYTES FAMILY "
                "[STORAGE_FORMAT GROUP_SIZE]\n",
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
    if (argc == 7) {
        CHECK(parse_u64(argv[5], &storage64));
        CHECK(parse_u64(argv[6], &group64));
        CHECK(storage64 <= UINT32_MAX);
        CHECK(group64 <= UINT32_MAX);
    }
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
    CHECK(manifest->storage_format == (uint32_t)storage64);
    CHECK(manifest->group_size == (uint32_t)group64);
    if (argc == 5 && family != DS4_EXPERT_STORE_FAMILY_QWEN4EXP) {
        CHECK(manifest->layer_count == 2);
        CHECK(manifest->expert_count == 3);
        CHECK(manifest->expert_used_count == 2);
        CHECK(manifest->source_tensor_count == 7);
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
        DS4_EXPERT_STORE_FAMILY_QWEN4EXP,
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
    puts("expert-store v2 C reader and Qwen4Exp sparse checks: OK");
    return 0;
}
