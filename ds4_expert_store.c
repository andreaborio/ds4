#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include "ds4_expert_store.h"

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
    STORE_HEADER_BYTES = 256,
    STORE_LAYER_BYTES = 224,
    STORE_COMPONENT_BYTES = 56,
    STORE_COMPONENT_OFFSET = 32,
    STORE_DATA_ALIGNMENT = 4096,
    STORE_MANIFEST_DIGEST_OFFSET = 168,
};

static const uint8_t store_magic[8] = {
    'D', 'S', '4', 'E', 'X', 'P', 'V', '2'
};

struct ds4_expert_store {
    int fd;
    uint64_t file_offset;
    uint64_t extent_size;
    ds4_expert_store_manifest manifest;
    ds4_expert_store_layer *layers;
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

typedef struct {
    uint32_t state[8];
    uint64_t bytes;
    uint8_t block[64];
    size_t block_len;
} sha256_context;

static uint32_t rotr32(uint32_t value, unsigned shift) {
    return (value >> shift) | (value << (32u - shift));
}

static void sha256_transform(sha256_context *context, const uint8_t block[64]) {
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

static bool quant_layout(uint32_t type, uint32_t *elements, uint32_t *bytes) {
    if (!elements || !bytes) return false;
    switch (type) {
    case 10: *elements = 256; *bytes = 84; return true;  /* Q2_K */
    case 12: *elements = 256; *bytes = 144; return true; /* Q4_K */
    case 13: *elements = 256; *bytes = 176; return true; /* Q5_K */
    case 14: *elements = 256; *bytes = 210; return true; /* Q6_K */
    case 16: *elements = 256; *bytes = 66; return true;  /* IQ2_XXS */
    default: return false;
    }
}

static bool all_zero(const uint8_t *data, size_t size) {
    for (size_t i = 0; i < size; i++) if (data[i] != 0) return false;
    return true;
}

static bool family_is_supported(uint32_t family) {
    return family == DS4_EXPERT_STORE_FAMILY_DEEPSEEK4 ||
           family == DS4_EXPERT_STORE_FAMILY_GLM_DSA ||
           family == DS4_EXPERT_STORE_FAMILY_QWEN35_MOE;
}

static bool family_has_dense_routed_prefix(uint32_t family) {
    return family == DS4_EXPERT_STORE_FAMILY_DEEPSEEK4 ||
           family == DS4_EXPERT_STORE_FAMILY_QWEN35_MOE;
}

bool ds4_expert_store_open_embedded(
        ds4_expert_store **out,
        int                fd,
        uint64_t           offset,
        uint64_t           bytes,
        uint32_t           expected_family,
        char              *error,
        size_t             error_size) {
    if (out) *out = NULL;
    if (!out || fd < 0 || bytes < STORE_HEADER_BYTES ||
        offset > (uint64_t)INT64_MAX || bytes > (uint64_t)INT64_MAX - offset) {
        set_error(error, error_size, "invalid embedded expert-store extent");
        return false;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 0 || !S_ISREG(st.st_mode) ||
        offset > (uint64_t)st.st_size || bytes > (uint64_t)st.st_size - offset) {
        set_error(error, error_size, "expert store points outside its file");
        return false;
    }

    uint8_t header[STORE_HEADER_BYTES];
    if (!pread_all(fd, header, sizeof(header), offset)) {
        set_error(error, error_size, "cannot read expert-store header: %s",
                  strerror(errno));
        return false;
    }
    const uint32_t version = load_u32_le(header + 8);
    const uint32_t header_bytes = load_u32_le(header + 12);
    const uint32_t family = load_u32_le(header + 16);
    const uint32_t expert_used = load_u32_le(header + 20);
    const uint32_t layer_count = load_u32_le(header + 24);
    const uint32_t expert_count = load_u32_le(header + 28);
    const uint64_t source_tensors = load_u64_le(header + 32);
    const uint64_t descriptor_count = load_u64_le(header + 40);
    const uint64_t descriptor_bytes = load_u64_le(header + 48);
    const uint64_t descriptor_offset = load_u64_le(header + 56);
    const uint64_t data_offset = load_u64_le(header + 64);
    const uint64_t data_size = load_u64_le(header + 72);
    const uint64_t store_size = load_u64_le(header + 80);
    const uint64_t source_size = load_u64_le(header + 88);
    const uint32_t storage_format = load_u32_le(header + 160);
    const uint32_t group_size = load_u32_le(header + 164);
    const bool storage_valid =
        (storage_format == DS4_EXPERT_STORE_STORAGE_GGML &&
         group_size == 0u) ||
        (storage_format == DS4_EXPERT_STORE_STORAGE_MLX_AFFINE4 &&
         group_size == 64u &&
         family == DS4_EXPERT_STORE_FAMILY_QWEN35_MOE) ||
        (storage_format == DS4_EXPERT_STORE_STORAGE_MLX_AFFINE2 &&
         group_size ==
             DS4_EXPERT_STORE_GROUP_PROFILE_AFFINE2_G32_U64_D64 &&
         family == DS4_EXPERT_STORE_FAMILY_DEEPSEEK4);

    uint64_t want_descriptor_bytes = 0;
    uint64_t descriptors_end = 0;
    if (memcmp(header, store_magic, sizeof(store_magic)) != 0 ||
        version != DS4_EXPERT_STORE_V2_VERSION ||
        header_bytes != STORE_HEADER_BYTES ||
        family != expected_family ||
        !family_is_supported(family) ||
        layer_count == 0 || layer_count > DS4_EXPERT_STORE_V2_MAX_LAYERS ||
        expert_count == 0 || expert_count > DS4_EXPERT_STORE_V2_MAX_EXPERTS ||
        expert_used == 0 || expert_used > expert_count ||
        descriptor_count != layer_count ||
        !mul_u64(descriptor_count, STORE_LAYER_BYTES,
                 &want_descriptor_bytes) ||
        descriptor_bytes != want_descriptor_bytes ||
        descriptor_offset != STORE_HEADER_BYTES ||
        !add_u64(descriptor_offset, descriptor_bytes, &descriptors_end) ||
        descriptors_end > data_offset ||
        data_offset % STORE_DATA_ALIGNMENT != 0 ||
        !add_u64(data_offset, data_size, &descriptors_end) ||
        descriptors_end != store_size || store_size != bytes ||
        source_tensors <= (uint64_t)layer_count * 3u || source_size == 0 ||
        !storage_valid ||
        !all_zero(header + 200, STORE_HEADER_BYTES - 200)) {
        set_error(error, error_size,
                  "expert-store v2 header or family contract is invalid");
        return false;
    }

    uint8_t *raw = malloc((size_t)descriptor_bytes);
    ds4_expert_store_layer *layers =
        calloc(layer_count, sizeof(layers[0]));
    if (!raw || !layers) {
        free(raw);
        free(layers);
        set_error(error, error_size, "out of memory reading expert manifest");
        return false;
    }
    if (!pread_all(fd, raw, (size_t)descriptor_bytes,
                   offset + descriptor_offset)) {
        free(raw);
        free(layers);
        set_error(error, error_size, "cannot read expert manifest: %s",
                  strerror(errno));
        return false;
    }

    uint8_t digest[32];
    uint8_t digest_header[STORE_HEADER_BYTES];
    memcpy(digest_header, header, sizeof(digest_header));
    memset(digest_header + STORE_MANIFEST_DIGEST_OFFSET, 0, 32);
    sha256_context hash;
    sha256_init(&hash);
    sha256_update(&hash, digest_header, sizeof(digest_header));
    sha256_update(&hash, raw, (size_t)descriptor_bytes);
    sha256_final(&hash, digest);
    if (memcmp(digest, header + STORE_MANIFEST_DIGEST_OFFSET, 32) != 0) {
        free(raw);
        free(layers);
        set_error(error, error_size, "expert manifest SHA-256 mismatch");
        return false;
    }

    uint64_t previous_end = data_offset;
    uint32_t previous_layer = 0;
    for (uint32_t il = 0; il < layer_count; il++) {
        const uint8_t *entry = raw + (uint64_t)il * STORE_LAYER_BYTES;
        ds4_expert_store_layer *layer = &layers[il];
        layer->layer = load_u32_le(entry);
        layer->expert_count = load_u32_le(entry + 4);
        layer->record_bytes = load_u64_le(entry + 8);
        layer->data_offset = load_u64_le(entry + 16);
        layer->data_size = load_u64_le(entry + 24);
        uint64_t expected_record_offset = 0;
        for (uint32_t role = 0; role < 3; role++) {
            const uint8_t *component =
                entry + STORE_COMPONENT_OFFSET + role * STORE_COMPONENT_BYTES;
            ds4_expert_store_component *out_component =
                &layer->component[role];
            out_component->role = load_u32_le(component);
            out_component->ggml_type = load_u32_le(component + 4);
            const uint32_t ndim = load_u32_le(component + 8);
            out_component->block_elements = load_u32_le(component + 12);
            out_component->dim[0] = load_u64_le(component + 16);
            out_component->dim[1] = load_u64_le(component + 24);
            out_component->dim[2] = load_u64_le(component + 32);
            out_component->expert_bytes = load_u64_le(component + 40);
            out_component->record_offset = load_u64_le(component + 48);

            uint32_t block_elements = 0, block_bytes = 0;
            if (storage_format == DS4_EXPERT_STORE_STORAGE_MLX_AFFINE2) {
                block_elements = role == DS4_EXPERT_STORE_GATE ? 32u : 64u;
                block_bytes = role == DS4_EXPERT_STORE_GATE ? 12u : 20u;
            }
            uint64_t row_blocks = 0, row_bytes = 0, expert_bytes = 0;
            if (out_component->role != role || ndim != 3 ||
                (storage_format == DS4_EXPERT_STORE_STORAGE_MLX_AFFINE2
                    ? out_component->ggml_type !=
                          DS4_EXPERT_STORE_TYPE_MLX_AFFINE2
                    : !quant_layout(out_component->ggml_type,
                                    &block_elements, &block_bytes)) ||
                (storage_format ==
                     DS4_EXPERT_STORE_STORAGE_MLX_AFFINE4 &&
                 out_component->ggml_type != 12u) ||
                out_component->block_elements != block_elements ||
                out_component->dim[0] == 0 ||
                out_component->dim[0] % block_elements != 0 ||
                out_component->dim[1] == 0 ||
                out_component->dim[2] != expert_count ||
                !mul_u64(out_component->dim[0] / block_elements,
                         block_bytes, &row_blocks) ||
                !mul_u64(row_blocks, out_component->dim[1], &row_bytes) ||
                !mul_u64(row_bytes, 1, &expert_bytes) ||
                out_component->expert_bytes != expert_bytes ||
                out_component->record_offset != expected_record_offset ||
                !add_u64(expected_record_offset, expert_bytes,
                         &expected_record_offset)) {
                free(raw);
                free(layers);
                set_error(error, error_size,
                          "expert manifest component geometry is invalid at layer %u role %u",
                          il, role);
                return false;
            }
        }
        uint64_t expected_layer_bytes = 0;
        uint64_t layer_end = 0;
        if (layer->layer > DS4_EXPERT_STORE_V2_MAX_MODEL_LAYER ||
            (il != 0 && layer->layer <= previous_layer) ||
            (family_has_dense_routed_prefix(family) && layer->layer != il) ||
            layer->expert_count != expert_count ||
            layer->record_bytes != expected_record_offset ||
            !mul_u64(layer->record_bytes, expert_count,
                     &expected_layer_bytes) ||
            layer->data_size != expected_layer_bytes ||
            layer->data_offset < previous_end ||
            layer->data_offset % STORE_DATA_ALIGNMENT != 0 ||
            !add_u64(layer->data_offset, layer->data_size, &layer_end) ||
            layer_end > store_size ||
            !all_zero(entry + 200, STORE_LAYER_BYTES - 200)) {
            free(raw);
            free(layers);
            set_error(error, error_size,
                      "expert manifest layer extent is invalid at layer %u", il);
            return false;
        }
        previous_layer = layer->layer;
        previous_end = layer_end;
    }
    if (previous_end != store_size) {
        free(raw);
        free(layers);
        set_error(error, error_size,
                  "expert manifest does not cover the complete store payload");
        return false;
    }
    free(raw);

    int owned_fd;
#ifdef F_DUPFD_CLOEXEC
    owned_fd = fcntl(fd, F_DUPFD_CLOEXEC, 0);
#else
    owned_fd = dup(fd);
#endif
    if (owned_fd < 0) {
        free(layers);
        set_error(error, error_size, "cannot duplicate expert-store descriptor: %s",
                  strerror(errno));
        return false;
    }
    ds4_expert_store *store = calloc(1, sizeof(*store));
    if (!store) {
        close(owned_fd);
        free(layers);
        set_error(error, error_size, "out of memory opening expert store");
        return false;
    }
    store->fd = owned_fd;
    store->file_offset = offset;
    store->extent_size = bytes;
    store->layers = layers;
    store->manifest = (ds4_expert_store_manifest){
        .version = version,
        .family = family,
        .storage_format = storage_format,
        .group_size = group_size,
        .layer_count = layer_count,
        .expert_count = expert_count,
        .expert_used_count = expert_used,
        .source_tensor_count = source_tensors,
        .source_size = source_size,
        .data_offset = data_offset,
        .data_size = data_size,
        .store_size = store_size,
    };
    memcpy(store->manifest.source_sha256, header + 96, 32);
    memcpy(store->manifest.payload_sha256, header + 128, 32);
    memcpy(store->manifest.manifest_sha256, header + 168, 32);
    *out = store;
    return true;
}

void ds4_expert_store_close(ds4_expert_store *store) {
    if (!store) return;
    if (store->fd >= 0) close(store->fd);
    free(store->layers);
    free(store);
}

const ds4_expert_store_manifest *ds4_expert_store_manifest_get(
        const ds4_expert_store *store) {
    return store ? &store->manifest : NULL;
}

const ds4_expert_store_layer *ds4_expert_store_layer_get(
        const ds4_expert_store *store, uint32_t layer) {
    if (!store) return NULL;
    if (layer < store->manifest.layer_count &&
        store->layers[layer].layer == layer) {
        return &store->layers[layer];
    }
    uint32_t lo = 0;
    uint32_t hi = store->manifest.layer_count;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2u;
        if (store->layers[mid].layer < layer) lo = mid + 1u;
        else hi = mid;
    }
    if (lo >= store->manifest.layer_count ||
        store->layers[lo].layer != layer) {
        return NULL;
    }
    return &store->layers[lo];
}

const ds4_expert_store_layer *ds4_expert_store_layer_at(
        const ds4_expert_store *store, uint32_t index) {
    if (!store || index >= store->manifest.layer_count) return NULL;
    return &store->layers[index];
}

bool ds4_expert_store_slice_get(
        const ds4_expert_store *store,
        uint32_t                layer,
        uint32_t                expert,
        uint32_t                role,
        uint64_t               *offset,
        uint64_t               *bytes) {
    if (offset) *offset = 0;
    if (bytes) *bytes = 0;
    if (!store || !offset || !bytes ||
        expert >= store->manifest.expert_count || role >= 3) {
        return false;
    }
    const ds4_expert_store_layer *entry =
        ds4_expert_store_layer_get(store, layer);
    if (!entry) return false;
    const ds4_expert_store_component *component = &entry->component[role];
    uint64_t relative = 0;
    if (!mul_u64(expert, entry->record_bytes, &relative) ||
        !add_u64(relative, entry->data_offset, &relative) ||
        !add_u64(relative, component->record_offset, &relative) ||
        !add_u64(relative, store->file_offset, offset)) {
        return false;
    }
    *bytes = component->expert_bytes;
    return true;
}

int ds4_expert_store_fd(const ds4_expert_store *store) {
    return store ? store->fd : -1;
}

uint64_t ds4_expert_store_file_offset(const ds4_expert_store *store) {
    return store ? store->file_offset : 0;
}
