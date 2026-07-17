#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include "ds4_qwen_expert_pack.h"

#include "ds4_qwen.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

enum {
    PACK_HEADER_BYTES = 256,
    PACK_ENTRY_BYTES = 40,
    PACK_DATA_ALIGNMENT = 4096,
    PACK_INDEX_HASH_OFFSET = 192,
    PACK_IO_BYTES = 1024 * 1024,
    GGUF_Q4_K_BLOCK_ELEMS = 256,
    GGUF_Q4_K_BLOCK_BYTES = 144,
    GGUF_VALUE_UINT8 = 0,
    GGUF_VALUE_INT8 = 1,
    GGUF_VALUE_UINT16 = 2,
    GGUF_VALUE_INT16 = 3,
    GGUF_VALUE_UINT32 = 4,
    GGUF_VALUE_INT32 = 5,
    GGUF_VALUE_FLOAT32 = 6,
    GGUF_VALUE_BOOL = 7,
    GGUF_VALUE_STRING = 8,
    GGUF_VALUE_ARRAY = 9,
    GGUF_VALUE_UINT64 = 10,
    GGUF_VALUE_INT64 = 11,
    GGUF_VALUE_FLOAT64 = 12,
};

static const uint8_t pack_magic[8] = {
    'D', 'S', '4', 'Q', 'X', 'P', 'K', '1'
};

typedef struct {
    uint64_t offset;
    uint64_t total_bytes;
    uint64_t gate_bytes;
    uint64_t up_bytes;
    uint64_t down_bytes;
} pack_entry;

struct ds4_qwen_expert_pack {
    int fd;
    uint64_t file_offset;
    uint64_t extent_size;
    ds4_qwen_expert_pack_manifest manifest;
    pack_entry *entry;
    bool source_validated;
    bool payload_validated;
};

typedef struct {
    bool present;
    uint64_t rel_offset;
    uint64_t abs_offset;
    uint64_t expert_bytes;
    uint64_t tensor_bytes;
} gguf_expert_matrix;

typedef struct {
    int fd;
    uint64_t size;
    ds4_qwen_expert_pack_geometry geometry;
    gguf_expert_matrix *matrix; /* [layer][gate, up, down] */
} gguf_expert_layout;

typedef struct {
    FILE *file;
    uint64_t size;
    char *error;
    size_t error_size;
} gguf_reader;

static void set_error(char *error, size_t error_size, const char *fmt, ...) {
    if (!error || error_size == 0) return;
    va_list args;
    va_start(args, fmt);
    vsnprintf(error, error_size, fmt, args);
    va_end(args);
}

static bool checked_add_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out || a > UINT64_MAX - b) return false;
    *out = a + b;
    return true;
}

static bool checked_mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out || (a != 0 && b > UINT64_MAX / a)) return false;
    *out = a * b;
    return true;
}

static bool align_up_u64(uint64_t value, uint64_t alignment, uint64_t *out) {
    if (!out || alignment == 0) return false;
    const uint64_t remainder = value % alignment;
    if (remainder == 0) {
        *out = value;
        return true;
    }
    return checked_add_u64(value, alignment - remainder, out);
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

/* Compact SHA-256 keeps the pack format self-contained on macOS and Linux;
 * relying on an external executable would make validation non-reusable by the
 * future loader. */
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

    uint32_t a = context->state[0];
    uint32_t b = context->state[1];
    uint32_t c = context->state[2];
    uint32_t d = context->state[3];
    uint32_t e = context->state[4];
    uint32_t f = context->state[5];
    uint32_t g = context->state[6];
    uint32_t h = context->state[7];
    for (size_t i = 0; i < 64; i++) {
        const uint32_t sum1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        const uint32_t choice = (e & f) ^ (~e & g);
        const uint32_t temp1 = h + sum1 + choice + k[i] + w[i];
        const uint32_t sum0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t temp2 = sum0 + majority;
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }
    context->state[0] += a;
    context->state[1] += b;
    context->state[2] += c;
    context->state[3] += d;
    context->state[4] += e;
    context->state[5] += f;
    context->state[6] += g;
    context->state[7] += h;
}

static void sha256_init(sha256_context *context) {
    *context = (sha256_context){
        .state = {
            0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
            0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
        },
    };
}

static void sha256_update(
        sha256_context *context,
        const void     *data_pointer,
        size_t          size) {
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

static void sha256_final(
        sha256_context *context,
        uint8_t         digest[DS4_QWEN_EXPERT_PACK_SHA256_BYTES]) {
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

static bool pread_all(
        int fd, void *buffer_pointer, size_t size, uint64_t offset,
        char *error, size_t error_size) {
    uint8_t *buffer = buffer_pointer;
    while (size != 0) {
        const ssize_t got = pread(fd, buffer, size, (off_t)offset);
        if (got < 0 && errno == EINTR) continue;
        if (got < 0) {
            set_error(error, error_size, "pread at byte %" PRIu64 ": %s",
                      offset, strerror(errno));
            return false;
        }
        if (got == 0) {
            set_error(error, error_size,
                      "unexpected end of file at byte %" PRIu64, offset);
            return false;
        }
        buffer += (size_t)got;
        size -= (size_t)got;
        offset += (uint64_t)got;
    }
    return true;
}

static bool pwrite_all(
        int fd, const void *buffer_pointer, size_t size, uint64_t offset,
        char *error, size_t error_size) {
    const uint8_t *buffer = buffer_pointer;
    while (size != 0) {
        const ssize_t wrote = pwrite(fd, buffer, size, (off_t)offset);
        if (wrote < 0 && errno == EINTR) continue;
        if (wrote < 0) {
            set_error(error, error_size, "pwrite at byte %" PRIu64 ": %s",
                      offset, strerror(errno));
            return false;
        }
        if (wrote == 0) {
            set_error(error, error_size,
                      "pwrite made no progress at byte %" PRIu64, offset);
            return false;
        }
        buffer += (size_t)wrote;
        size -= (size_t)wrote;
        offset += (uint64_t)wrote;
    }
    return true;
}

static bool hash_fd_range(
        int                                  fd,
        uint64_t                             offset,
        uint64_t                             size,
        uint8_t                              digest[32],
        ds4_qwen_expert_pack_progress_fn     progress,
        void                                *progress_context,
        ds4_qwen_expert_pack_phase           phase,
        char                                *error,
        size_t                               error_size) {
#if defined(__APPLE__) && defined(F_NOCACHE)
    /* Builders and verifiers intentionally stream multi-GiB artifacts. Keep
     * those one-shot reads from evicting a live inference model's hot pages. */
    (void)fcntl(fd, F_NOCACHE, 1);
#endif
    uint8_t *buffer = malloc(PACK_IO_BYTES);
    if (!buffer) {
        set_error(error, error_size, "out of memory while hashing file");
        return false;
    }
    sha256_context hash;
    sha256_init(&hash);
    uint64_t completed = 0;
    if (progress) progress(progress_context, phase, 0, size);
    while (completed < size) {
        size_t take = PACK_IO_BYTES;
        if ((uint64_t)take > size - completed) take = (size_t)(size - completed);
        if (!pread_all(fd, buffer, take, offset + completed,
                       error, error_size)) {
            free(buffer);
            return false;
        }
        sha256_update(&hash, buffer, take);
        completed += take;
        if (progress) progress(progress_context, phase, completed, size);
    }
    sha256_final(&hash, digest);
    free(buffer);
    return true;
}

static bool gguf_read(gguf_reader *reader, void *out, size_t size) {
    if (size == 0) return true;
    if (fread(out, 1, size, reader->file) == size) return true;
    set_error(reader->error, reader->error_size,
              "truncated GGUF metadata or tensor directory");
    return false;
}

static bool gguf_u32(gguf_reader *reader, uint32_t *out) {
    uint8_t bytes[4];
    if (!gguf_read(reader, bytes, sizeof(bytes))) return false;
    *out = load_u32_le(bytes);
    return true;
}

static bool gguf_u64(gguf_reader *reader, uint64_t *out) {
    uint8_t bytes[8];
    if (!gguf_read(reader, bytes, sizeof(bytes))) return false;
    *out = load_u64_le(bytes);
    return true;
}

static bool gguf_skip(gguf_reader *reader, uint64_t size) {
    /* Tokenizer metadata contains hundreds of thousands of short strings.
     * Consume those through stdio's sequential buffer; seeking once per token
     * would turn directory discovery into a syscall-heavy pre-pass. */
    if (size <= 4096) {
        uint8_t scratch[256];
        while (size != 0) {
            size_t take = sizeof(scratch);
            if ((uint64_t)take > size) take = (size_t)size;
            if (!gguf_read(reader, scratch, take)) return false;
            size -= take;
        }
        return true;
    }
    const off_t current = ftello(reader->file);
    if (current < 0 || (uint64_t)current > reader->size ||
        size > reader->size - (uint64_t)current || size > (uint64_t)INT64_MAX) {
        set_error(reader->error, reader->error_size,
                  "GGUF value points outside the source file");
        return false;
    }
    if (fseeko(reader->file, (off_t)size, SEEK_CUR) != 0) {
        set_error(reader->error, reader->error_size,
                  "cannot seek across GGUF value: %s", strerror(errno));
        return false;
    }
    return true;
}

static bool gguf_string(
        gguf_reader *reader,
        char       **out,
        uint64_t     maximum_size) {
    uint64_t size = 0;
    if (!gguf_u64(reader, &size)) return false;
    if (size > maximum_size || size > SIZE_MAX - 1) {
        set_error(reader->error, reader->error_size,
                  "GGUF string is too large (%" PRIu64 " bytes)", size);
        return false;
    }
    char *value = malloc((size_t)size + 1);
    if (!value) {
        set_error(reader->error, reader->error_size,
                  "out of memory while reading GGUF string");
        return false;
    }
    if (!gguf_read(reader, value, (size_t)size)) {
        free(value);
        return false;
    }
    value[size] = '\0';
    *out = value;
    return true;
}

static uint64_t gguf_scalar_size(uint32_t type) {
    switch (type) {
    case GGUF_VALUE_UINT8:
    case GGUF_VALUE_INT8:
    case GGUF_VALUE_BOOL:
        return 1;
    case GGUF_VALUE_UINT16:
    case GGUF_VALUE_INT16:
        return 2;
    case GGUF_VALUE_UINT32:
    case GGUF_VALUE_INT32:
    case GGUF_VALUE_FLOAT32:
        return 4;
    case GGUF_VALUE_UINT64:
    case GGUF_VALUE_INT64:
    case GGUF_VALUE_FLOAT64:
        return 8;
    default:
        return 0;
    }
}

static bool gguf_skip_value(gguf_reader *reader, uint32_t type, unsigned depth) {
    if (depth > 8) {
        set_error(reader->error, reader->error_size,
                  "GGUF metadata array nesting is too deep");
        return false;
    }
    const uint64_t scalar = gguf_scalar_size(type);
    if (scalar != 0) return gguf_skip(reader, scalar);
    if (type == GGUF_VALUE_STRING) {
        uint64_t size = 0;
        return gguf_u64(reader, &size) && gguf_skip(reader, size);
    }
    if (type != GGUF_VALUE_ARRAY) {
        set_error(reader->error, reader->error_size,
                  "unknown GGUF metadata type %u", type);
        return false;
    }

    uint32_t item_type = 0;
    uint64_t count = 0;
    if (!gguf_u32(reader, &item_type) || !gguf_u64(reader, &count)) return false;
    const uint64_t item_size = gguf_scalar_size(item_type);
    if (item_size != 0) {
        uint64_t bytes = 0;
        if (!checked_mul_u64(count, item_size, &bytes)) {
            set_error(reader->error, reader->error_size,
                      "GGUF metadata array size overflows");
            return false;
        }
        return gguf_skip(reader, bytes);
    }
    for (uint64_t i = 0; i < count; i++) {
        if (!gguf_skip_value(reader, item_type, depth + 1)) return false;
    }
    return true;
}

static bool geometry_equal(
        const ds4_qwen_expert_pack_geometry *a,
        const ds4_qwen_expert_pack_geometry *b) {
    return a && b &&
           a->n_layer == b->n_layer &&
           a->n_expert == b->n_expert &&
           a->n_embd == b->n_embd &&
           a->n_ff_exp == b->n_ff_exp &&
           a->quant_type == b->quant_type &&
           a->gguf_tensor_count == b->gguf_tensor_count;
}

static bool geometry_valid(const ds4_qwen_expert_pack_geometry *geometry) {
    return geometry && geometry->n_layer != 0 && geometry->n_expert != 0 &&
           geometry->n_embd != 0 && geometry->n_ff_exp != 0 &&
           geometry->quant_type == DS4_QWEN_EXPERT_PACK_Q4_K_TYPE &&
           geometry->gguf_tensor_count != 0 &&
           geometry->n_embd % GGUF_Q4_K_BLOCK_ELEMS == 0 &&
           geometry->n_ff_exp % GGUF_Q4_K_BLOCK_ELEMS == 0;
}

static bool gguf_read_required_u32(
        gguf_reader *reader,
        uint32_t     type,
        uint32_t    *out,
        const char  *key) {
    if (type != GGUF_VALUE_UINT32 || !gguf_u32(reader, out)) {
        if (type != GGUF_VALUE_UINT32) {
            set_error(reader->error, reader->error_size,
                      "GGUF key %s must be uint32", key);
        }
        return false;
    }
    return true;
}

static int routed_tensor_kind(const char *name, uint32_t *layer_out) {
    static const char *suffix[3] = {
        ".ffn_gate_exps.weight",
        ".ffn_up_exps.weight",
        ".ffn_down_exps.weight",
    };
    if (strncmp(name, "blk.", 4) != 0) return -1;
    char *end = NULL;
    errno = 0;
    const unsigned long layer = strtoul(name + 4, &end, 10);
    if (errno != 0 || end == name + 4 || layer > UINT32_MAX) return -1;
    for (int kind = 0; kind < 3; kind++) {
        if (strcmp(end, suffix[kind]) == 0) {
            *layer_out = (uint32_t)layer;
            return kind;
        }
    }
    return -1;
}

static bool q4_matrix_expert_bytes(
        uint64_t input_dim,
        uint64_t output_dim,
        uint64_t *out) {
    uint64_t row_bytes = 0;
    return input_dim != 0 && output_dim != 0 &&
           input_dim % GGUF_Q4_K_BLOCK_ELEMS == 0 &&
           checked_mul_u64(input_dim / GGUF_Q4_K_BLOCK_ELEMS,
                           GGUF_Q4_K_BLOCK_BYTES, &row_bytes) &&
           checked_mul_u64(row_bytes, output_dim, out);
}

static void gguf_layout_close(gguf_expert_layout *layout) {
    if (!layout) return;
    free(layout->matrix);
    if (layout->fd >= 0) close(layout->fd);
    memset(layout, 0, sizeof(*layout));
    layout->fd = -1;
}

static bool gguf_layout_open(
        gguf_expert_layout                    *layout,
        const char                            *path,
        const ds4_qwen_expert_pack_geometry   *expected,
        char                                  *error,
        size_t                                 error_size) {
    memset(layout, 0, sizeof(*layout));
    layout->fd = -1;
    if (!path || !geometry_valid(expected)) {
        set_error(error, error_size, "invalid GGUF path or expected geometry");
        return false;
    }

    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        set_error(error, error_size, "cannot open GGUF %s: %s",
                  path, strerror(errno));
        return false;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 32) {
        set_error(error, error_size, "GGUF source is not a regular GGUF file");
        close(fd);
        return false;
    }
    const int parser_fd = dup(fd);
    if (parser_fd < 0) {
        set_error(error, error_size, "cannot duplicate GGUF descriptor: %s",
                  strerror(errno));
        close(fd);
        return false;
    }
    FILE *file = fdopen(parser_fd, "rb");
    if (!file) {
        set_error(error, error_size, "cannot create GGUF parser: %s",
                  strerror(errno));
        close(parser_fd);
        close(fd);
        return false;
    }

    gguf_reader reader = {
        .file = file,
        .size = (uint64_t)st.st_size,
        .error = error,
        .error_size = error_size,
    };
    uint32_t magic = 0;
    uint32_t version = 0;
    uint64_t tensor_count = 0;
    uint64_t kv_count = 0;
    if (!gguf_u32(&reader, &magic) || !gguf_u32(&reader, &version) ||
        !gguf_u64(&reader, &tensor_count) || !gguf_u64(&reader, &kv_count)) {
        fclose(file);
        close(fd);
        return false;
    }
    if (magic != UINT32_C(0x46554747) || version != 3) {
        set_error(error, error_size, "source must be a GGUF v3 file");
        fclose(file);
        close(fd);
        return false;
    }
    if (tensor_count != expected->gguf_tensor_count) {
        set_error(error, error_size,
                  "GGUF tensor count mismatch: got %" PRIu64 ", expected %" PRIu64,
                  tensor_count, expected->gguf_tensor_count);
        fclose(file);
        close(fd);
        return false;
    }

    ds4_qwen_expert_pack_geometry actual = {
        .quant_type = expected->quant_type,
        .gguf_tensor_count = tensor_count,
    };
    uint32_t alignment = 32;
    bool have_arch = false;
    bool have_layer = false;
    bool have_expert = false;
    bool have_embd = false;
    bool have_ff = false;
    for (uint64_t i = 0; i < kv_count; i++) {
        char *key = NULL;
        uint32_t type = 0;
        if (!gguf_string(&reader, &key, 1024 * 1024) ||
            !gguf_u32(&reader, &type)) {
            free(key);
            fclose(file);
            close(fd);
            return false;
        }
        bool ok = true;
        if (strcmp(key, "general.architecture") == 0) {
            char *architecture = NULL;
            ok = type == GGUF_VALUE_STRING &&
                 gguf_string(&reader, &architecture, 1024);
            if (!ok && type != GGUF_VALUE_STRING) {
                set_error(error, error_size,
                          "GGUF general.architecture must be a string");
            }
            if (ok && strcmp(architecture, "qwen35moe") != 0) {
                set_error(error, error_size,
                          "GGUF architecture is %s, expected qwen35moe",
                          architecture);
                ok = false;
            }
            have_arch = ok;
            free(architecture);
        } else if (strcmp(key, "general.alignment") == 0) {
            ok = gguf_read_required_u32(&reader, type, &alignment, key) &&
                 alignment != 0;
            if (!ok && alignment == 0) {
                set_error(error, error_size, "GGUF alignment cannot be zero");
            }
        } else if (strcmp(key, "qwen35moe.block_count") == 0) {
            ok = gguf_read_required_u32(&reader, type, &actual.n_layer, key);
            have_layer = ok;
        } else if (strcmp(key, "qwen35moe.expert_count") == 0) {
            ok = gguf_read_required_u32(&reader, type, &actual.n_expert, key);
            have_expert = ok;
        } else if (strcmp(key, "qwen35moe.embedding_length") == 0) {
            ok = gguf_read_required_u32(&reader, type, &actual.n_embd, key);
            have_embd = ok;
        } else if (strcmp(key,
                          "qwen35moe.expert_feed_forward_length") == 0) {
            ok = gguf_read_required_u32(&reader, type, &actual.n_ff_exp, key);
            have_ff = ok;
        } else {
            ok = gguf_skip_value(&reader, type, 0);
        }
        free(key);
        if (!ok) {
            fclose(file);
            close(fd);
            return false;
        }
    }
    if (!have_arch || !have_layer || !have_expert || !have_embd || !have_ff) {
        set_error(error, error_size,
                  "GGUF is missing required qwen35moe geometry metadata");
        fclose(file);
        close(fd);
        return false;
    }
    if (!geometry_equal(&actual, expected)) {
        set_error(error, error_size,
                  "GGUF Qwen geometry does not match the requested pack geometry");
        fclose(file);
        close(fd);
        return false;
    }

    uint64_t matrix_count = 0;
    if (!checked_mul_u64(expected->n_layer, 3, &matrix_count) ||
        matrix_count > SIZE_MAX / sizeof(gguf_expert_matrix)) {
        set_error(error, error_size, "Qwen matrix table size overflows");
        fclose(file);
        close(fd);
        return false;
    }
    gguf_expert_matrix *matrix = calloc(
        (size_t)matrix_count, sizeof(*matrix));
    if (!matrix) {
        set_error(error, error_size,
                  "out of memory while indexing routed tensors");
        fclose(file);
        close(fd);
        return false;
    }

    bool parsed = true;
    for (uint64_t i = 0; i < tensor_count && parsed; i++) {
        char *name = NULL;
        uint32_t ndim = 0;
        uint64_t dim[4] = {0};
        uint32_t type = 0;
        uint64_t rel_offset = 0;
        if (!gguf_string(&reader, &name, 1024 * 1024) ||
            !gguf_u32(&reader, &ndim)) {
            free(name);
            parsed = false;
            break;
        }
        if (ndim == 0 || ndim > 4) {
            set_error(error, error_size,
                      "tensor %s has unsupported rank %u", name, ndim);
            free(name);
            parsed = false;
            break;
        }
        for (uint32_t d = 0; d < ndim; d++) {
            if (!gguf_u64(&reader, &dim[d])) {
                parsed = false;
                break;
            }
        }
        if (!parsed || !gguf_u32(&reader, &type) ||
            !gguf_u64(&reader, &rel_offset)) {
            free(name);
            parsed = false;
            break;
        }

        uint32_t layer = 0;
        const int kind = routed_tensor_kind(name, &layer);
        if (kind >= 0) {
            if (layer >= expected->n_layer || ndim != 3 ||
                type != expected->quant_type) {
                set_error(error, error_size,
                          "routed tensor %s has incompatible layer, rank, or type",
                          name);
                free(name);
                parsed = false;
                break;
            }
            const uint64_t want0 = kind == 2 ? expected->n_ff_exp
                                              : expected->n_embd;
            const uint64_t want1 = kind == 2 ? expected->n_embd
                                              : expected->n_ff_exp;
            if (dim[0] != want0 || dim[1] != want1 ||
                dim[2] != expected->n_expert) {
                set_error(error, error_size,
                          "routed tensor %s has incompatible dimensions", name);
                free(name);
                parsed = false;
                break;
            }
            gguf_expert_matrix *slot = &matrix[(uint64_t)layer * 3 + kind];
            if (slot->present ||
                !q4_matrix_expert_bytes(dim[0], dim[1],
                                        &slot->expert_bytes) ||
                !checked_mul_u64(slot->expert_bytes, expected->n_expert,
                                 &slot->tensor_bytes)) {
                set_error(error, error_size,
                          "duplicate or overflowing routed tensor %s", name);
                free(name);
                parsed = false;
                break;
            }
            slot->present = true;
            slot->rel_offset = rel_offset;
        }
        free(name);
    }

    uint64_t tensor_data_offset = 0;
    const off_t directory_end = ftello(file);
    if (!parsed || directory_end < 0 ||
        !align_up_u64((uint64_t)directory_end, alignment,
                      &tensor_data_offset)) {
        if (parsed) set_error(error, error_size, "GGUF data offset overflows");
        free(matrix);
        fclose(file);
        close(fd);
        return false;
    }
    for (uint64_t i = 0; i < matrix_count; i++) {
        if (!matrix[i].present ||
            !checked_add_u64(tensor_data_offset, matrix[i].rel_offset,
                             &matrix[i].abs_offset) ||
            matrix[i].abs_offset > (uint64_t)st.st_size ||
            matrix[i].tensor_bytes >
                (uint64_t)st.st_size - matrix[i].abs_offset) {
            set_error(error, error_size,
                      "required routed tensor is missing or outside the GGUF");
            free(matrix);
            fclose(file);
            close(fd);
            return false;
        }
    }

    fclose(file);
    layout->fd = fd;
    layout->size = (uint64_t)st.st_size;
    layout->geometry = *expected;
    layout->matrix = matrix;
    return true;
}

static void header_encode(
        uint8_t                                      header[PACK_HEADER_BYTES],
        const ds4_qwen_expert_pack_manifest         *manifest,
        const uint8_t                                index_sha256[32]) {
    /* Never serialize a native struct: fixed little-endian fields and zeroed
     * reserved bytes make identical GGUF content produce identical sidecars
     * across compiler padding and host ABI changes. */
    memset(header, 0, PACK_HEADER_BYTES);
    memcpy(header, pack_magic, sizeof(pack_magic));
    store_u32_le(header + 8, DS4_QWEN_EXPERT_PACK_FORMAT_VERSION);
    store_u32_le(header + 12, PACK_HEADER_BYTES);
    store_u32_le(header + 16, PACK_ENTRY_BYTES);
    store_u32_le(header + 20, 0); /* flags */
    store_u32_le(header + 24, manifest->geometry.n_layer);
    store_u32_le(header + 28, manifest->geometry.n_expert);
    store_u32_le(header + 32, manifest->geometry.n_embd);
    store_u32_le(header + 36, manifest->geometry.n_ff_exp);
    store_u32_le(header + 40, manifest->geometry.quant_type);
    store_u32_le(header + 44, 0);
    store_u64_le(header + 48, manifest->entry_count);
    store_u64_le(header + 56, PACK_HEADER_BYTES);
    store_u64_le(header + 64, manifest->data_offset);
    store_u64_le(header + 72, manifest->data_size);
    store_u64_le(header + 80, manifest->file_size);
    store_u64_le(header + 88, manifest->source_size);
    store_u64_le(header + 96, manifest->gate_bytes);
    store_u64_le(header + 104, manifest->up_bytes);
    store_u64_le(header + 112, manifest->down_bytes);
    store_u64_le(header + 120, manifest->geometry.gguf_tensor_count);
    memcpy(header + 128, manifest->source_sha256, 32);
    memcpy(header + 160, manifest->data_sha256, 32);
    if (index_sha256) memcpy(header + PACK_INDEX_HASH_OFFSET, index_sha256, 32);
}

static void table_encode(
        uint8_t          *table,
        const pack_entry *entry,
        uint64_t          count) {
    for (uint64_t i = 0; i < count; i++) {
        uint8_t *row = table + i * PACK_ENTRY_BYTES;
        store_u64_le(row, entry[i].offset);
        store_u64_le(row + 8, entry[i].total_bytes);
        store_u64_le(row + 16, entry[i].gate_bytes);
        store_u64_le(row + 24, entry[i].up_bytes);
        store_u64_le(row + 32, entry[i].down_bytes);
    }
}

static void index_digest(
        const uint8_t header_with_zero_digest[PACK_HEADER_BYTES],
        const uint8_t *table,
        size_t table_bytes,
        uint8_t digest[32]) {
    sha256_context hash;
    sha256_init(&hash);
    sha256_update(&hash, header_with_zero_digest, PACK_HEADER_BYTES);
    sha256_update(&hash, table, table_bytes);
    sha256_final(&hash, digest);
}

static bool make_manifest_layout(
        const gguf_expert_layout             *source,
        ds4_qwen_expert_pack_manifest        *manifest,
        pack_entry                          **entry_out,
        char                                 *error,
        size_t                                error_size) {
    memset(manifest, 0, sizeof(*manifest));
    *entry_out = NULL;
    uint64_t entry_count = 0;
    uint64_t table_bytes = 0;
    uint64_t table_end = 0;
    if (!checked_mul_u64(source->geometry.n_layer,
                         source->geometry.n_expert, &entry_count) ||
        entry_count == 0 ||
        entry_count > SIZE_MAX / sizeof(pack_entry) ||
        !checked_mul_u64(entry_count, PACK_ENTRY_BYTES, &table_bytes) ||
        !checked_add_u64(PACK_HEADER_BYTES, table_bytes, &table_end)) {
        set_error(error, error_size, "expert pack index size overflows");
        return false;
    }
    uint64_t data_offset = 0;
    if (!align_up_u64(table_end, PACK_DATA_ALIGNMENT, &data_offset)) {
        set_error(error, error_size, "expert pack data offset overflows");
        return false;
    }

    const uint64_t gate_bytes = source->matrix[0].expert_bytes;
    const uint64_t up_bytes = source->matrix[1].expert_bytes;
    const uint64_t down_bytes = source->matrix[2].expert_bytes;
    uint64_t per_expert = 0;
    uint64_t data_size = 0;
    uint64_t file_size = 0;
    if (!checked_add_u64(gate_bytes, up_bytes, &per_expert) ||
        !checked_add_u64(per_expert, down_bytes, &per_expert) ||
        !checked_mul_u64(entry_count, per_expert, &data_size) ||
        !checked_add_u64(data_offset, data_size, &file_size) ||
        file_size > INT64_MAX) {
        set_error(error, error_size, "expert pack payload size overflows");
        return false;
    }
    for (uint32_t layer = 0; layer < source->geometry.n_layer; layer++) {
        const gguf_expert_matrix *matrix = &source->matrix[(uint64_t)layer * 3];
        if (matrix[0].expert_bytes != gate_bytes ||
            matrix[1].expert_bytes != up_bytes ||
            matrix[2].expert_bytes != down_bytes) {
            set_error(error, error_size,
                      "routed expert byte geometry changes between layers");
            return false;
        }
    }

    pack_entry *entry = calloc((size_t)entry_count, sizeof(*entry));
    if (!entry) {
        set_error(error, error_size,
                  "out of memory while creating expert pack index");
        return false;
    }
    uint64_t offset = data_offset;
    for (uint64_t i = 0; i < entry_count; i++) {
        /* The implicit table key is (layer, expert).  Every entry is one
         * contiguous gate/up/down record, so one SSD read can populate the
         * future cache slot without changing any quantized byte. */
        entry[i] = (pack_entry){
            .offset = offset,
            .total_bytes = per_expert,
            .gate_bytes = gate_bytes,
            .up_bytes = up_bytes,
            .down_bytes = down_bytes,
        };
        offset += per_expert;
    }
    *manifest = (ds4_qwen_expert_pack_manifest){
        .geometry = source->geometry,
        .source_size = source->size,
        .entry_count = entry_count,
        .gate_bytes = gate_bytes,
        .up_bytes = up_bytes,
        .down_bytes = down_bytes,
        .data_offset = data_offset,
        .data_size = data_size,
        .file_size = file_size,
    };
    *entry_out = entry;
    return true;
}

ds4_qwen_expert_pack_geometry ds4_qwen35_expert_pack_geometry(void) {
    return (ds4_qwen_expert_pack_geometry){
        .n_layer = QWEN35_N_LAYER,
        .n_expert = QWEN35_N_EXPERT,
        .n_embd = QWEN35_N_EMBD,
        .n_ff_exp = QWEN35_N_FF_EXP,
        .quant_type = DS4_QWEN_EXPERT_PACK_Q4_K_TYPE,
        .gguf_tensor_count = QWEN35_N_TENSOR,
    };
}

void ds4_qwen_expert_pack_close(ds4_qwen_expert_pack *pack) {
    if (!pack) return;
    free(pack->entry);
    if (pack->fd >= 0) close(pack->fd);
    free(pack);
}

static ds4_qwen_expert_pack_result pack_open_fallback(
        ds4_qwen_expert_pack *pack,
        char *error,
        size_t error_size,
        const char *fmt,
        ...) {
    if (error && error_size != 0) {
        va_list args;
        va_start(args, fmt);
        vsnprintf(error, error_size, fmt, args);
        va_end(args);
    }
    ds4_qwen_expert_pack_close(pack);
    return DS4_QWEN_EXPERT_PACK_FALLBACK;
}

static ds4_qwen_expert_pack_result expert_pack_open_owned_fd(
        ds4_qwen_expert_pack                 **out,
        int                                    owned_fd,
        uint64_t                               file_offset,
        uint64_t                               extent_size,
        const ds4_qwen_expert_pack_geometry   *expected_geometry,
        const char                            *container_name,
        char                                  *error,
        size_t                                 error_size) {
    if (out) *out = NULL;
    ds4_qwen_expert_pack *pack = calloc(1, sizeof(*pack));
    if (!pack) {
        if (owned_fd >= 0) close(owned_fd);
        set_error(error, error_size, "out of memory while opening expert pack");
        return DS4_QWEN_EXPERT_PACK_ERROR;
    }
    pack->fd = owned_fd;
    pack->file_offset = file_offset;
    pack->extent_size = extent_size;
    struct stat st;
    if (pack->fd < 0 || fstat(pack->fd, &st) != 0 ||
        !S_ISREG(st.st_mode) || st.st_size < 0 ||
        extent_size < PACK_HEADER_BYTES ||
        file_offset > (uint64_t)st.st_size ||
        extent_size > (uint64_t)st.st_size - file_offset) {
        return pack_open_fallback(pack, error, error_size,
                                  "%s is not a bounded regular format-v1 file",
                                  container_name);
    }
    uint8_t header[PACK_HEADER_BYTES];
    char io_error[256] = {0};
    if (!pread_all(pack->fd, header, sizeof(header), file_offset,
                   io_error, sizeof(io_error))) {
        return pack_open_fallback(pack, error, error_size,
                                  "cannot read %s header: %s",
                                  container_name, io_error);
    }
    if (memcmp(header, pack_magic, sizeof(pack_magic)) != 0 ||
        load_u32_le(header + 8) != DS4_QWEN_EXPERT_PACK_FORMAT_VERSION ||
        load_u32_le(header + 12) != PACK_HEADER_BYTES ||
        load_u32_le(header + 16) != PACK_ENTRY_BYTES ||
        load_u32_le(header + 20) != 0 ||
        load_u32_le(header + 44) != 0) {
        return pack_open_fallback(pack, error, error_size,
                                  "expert pack header/version mismatch");
    }

    ds4_qwen_expert_pack_manifest manifest = {
        .geometry = {
            .n_layer = load_u32_le(header + 24),
            .n_expert = load_u32_le(header + 28),
            .n_embd = load_u32_le(header + 32),
            .n_ff_exp = load_u32_le(header + 36),
            .quant_type = load_u32_le(header + 40),
            .gguf_tensor_count = load_u64_le(header + 120),
        },
        .entry_count = load_u64_le(header + 48),
        .data_offset = load_u64_le(header + 64),
        .data_size = load_u64_le(header + 72),
        .file_size = load_u64_le(header + 80),
        .source_size = load_u64_le(header + 88),
        .gate_bytes = load_u64_le(header + 96),
        .up_bytes = load_u64_le(header + 104),
        .down_bytes = load_u64_le(header + 112),
    };
    memcpy(manifest.source_sha256, header + 128, 32);
    memcpy(manifest.data_sha256, header + 160, 32);
    if (!geometry_equal(&manifest.geometry, expected_geometry)) {
        return pack_open_fallback(pack, error, error_size,
                                  "expert pack geometry mismatch");
    }

    uint64_t expected_entry_count = 0;
    uint64_t table_bytes_u64 = 0;
    uint64_t table_end = 0;
    uint64_t expected_data_offset = 0;
    uint64_t per_expert = 0;
    uint64_t expected_data_size = 0;
    uint64_t expected_file_size = 0;
    const uint64_t table_offset = load_u64_le(header + 56);
    if (!checked_mul_u64(manifest.geometry.n_layer,
                         manifest.geometry.n_expert, &expected_entry_count) ||
        !checked_mul_u64(expected_entry_count, PACK_ENTRY_BYTES,
                         &table_bytes_u64) ||
        table_bytes_u64 > SIZE_MAX ||
        !checked_add_u64(PACK_HEADER_BYTES, table_bytes_u64, &table_end) ||
        !align_up_u64(table_end, PACK_DATA_ALIGNMENT, &expected_data_offset) ||
        !checked_add_u64(manifest.gate_bytes, manifest.up_bytes, &per_expert) ||
        !checked_add_u64(per_expert, manifest.down_bytes, &per_expert) ||
        !checked_mul_u64(expected_entry_count, per_expert,
                         &expected_data_size) ||
        !checked_add_u64(expected_data_offset, expected_data_size,
                         &expected_file_size) ||
        manifest.entry_count != expected_entry_count ||
        table_offset != PACK_HEADER_BYTES ||
        manifest.data_offset != expected_data_offset ||
        manifest.data_size != expected_data_size ||
        manifest.file_size != expected_file_size ||
        manifest.file_size != extent_size ||
        manifest.source_size == 0 || per_expert == 0) {
        return pack_open_fallback(pack, error, error_size,
                                  "expert pack sizes or offsets are invalid");
    }
    for (size_t i = 224; i < sizeof(header); i++) {
        if (header[i] != 0) {
            return pack_open_fallback(pack, error, error_size,
                                      "expert pack reserved header bytes are nonzero");
        }
    }

    const size_t table_bytes = (size_t)table_bytes_u64;
    uint8_t *table = malloc(table_bytes);
    if (!table) {
        ds4_qwen_expert_pack_close(pack);
        set_error(error, error_size,
                  "out of memory while reading expert pack index");
        return DS4_QWEN_EXPERT_PACK_ERROR;
    }
    if (!pread_all(pack->fd, table, table_bytes,
                   file_offset + PACK_HEADER_BYTES,
                   io_error, sizeof(io_error))) {
        free(table);
        return pack_open_fallback(pack, error, error_size,
                                  "cannot read expert pack index: %s", io_error);
    }
    uint8_t wanted_index_sha[32];
    memcpy(wanted_index_sha, header + PACK_INDEX_HASH_OFFSET, 32);
    memset(header + PACK_INDEX_HASH_OFFSET, 0, 32);
    uint8_t got_index_sha[32];
    index_digest(header, table, table_bytes, got_index_sha);
    if (memcmp(wanted_index_sha, got_index_sha, 32) != 0) {
        free(table);
        return pack_open_fallback(pack, error, error_size,
                                  "expert pack manifest/index checksum mismatch");
    }

    pack->entry = calloc((size_t)manifest.entry_count,
                         sizeof(pack->entry[0]));
    if (!pack->entry) {
        free(table);
        ds4_qwen_expert_pack_close(pack);
        set_error(error, error_size,
                  "out of memory while decoding expert pack index");
        return DS4_QWEN_EXPERT_PACK_ERROR;
    }
    uint64_t next_offset = manifest.data_offset;
    bool valid = true;
    for (uint64_t i = 0; i < manifest.entry_count; i++) {
        const uint8_t *row = table + i * PACK_ENTRY_BYTES;
        pack_entry *entry = &pack->entry[i];
        entry->offset = load_u64_le(row);
        entry->total_bytes = load_u64_le(row + 8);
        entry->gate_bytes = load_u64_le(row + 16);
        entry->up_bytes = load_u64_le(row + 24);
        entry->down_bytes = load_u64_le(row + 32);
        if (entry->offset != next_offset ||
            entry->total_bytes != per_expert ||
            entry->gate_bytes != manifest.gate_bytes ||
            entry->up_bytes != manifest.up_bytes ||
            entry->down_bytes != manifest.down_bytes ||
            !checked_add_u64(next_offset, per_expert, &next_offset)) {
            valid = false;
            break;
        }
    }
    free(table);
    if (!valid || next_offset != manifest.file_size) {
        return pack_open_fallback(pack, error, error_size,
                                  "expert pack index is non-contiguous or malformed");
    }
    pack->manifest = manifest;
    *out = pack;
    if (error && error_size) error[0] = '\0';
    return DS4_QWEN_EXPERT_PACK_OK;
}

ds4_qwen_expert_pack_result ds4_qwen_expert_pack_open(
        ds4_qwen_expert_pack                 **out,
        const char                            *pack_path,
        const ds4_qwen_expert_pack_geometry   *expected_geometry,
        char                                  *error,
        size_t                                 error_size) {
    if (out) *out = NULL;
    if (!out || !pack_path || !geometry_valid(expected_geometry)) {
        set_error(error, error_size,
                  "invalid expert pack path or expected geometry");
        return DS4_QWEN_EXPERT_PACK_ERROR;
    }
    const int fd = open(pack_path, O_RDONLY);
    if (fd < 0) {
        set_error(error, error_size, "expert pack unavailable: %s",
                  strerror(errno));
        return DS4_QWEN_EXPERT_PACK_FALLBACK;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size < 0) {
        close(fd);
        set_error(error, error_size,
                  "expert pack is not a regular format-v1 file");
        return DS4_QWEN_EXPERT_PACK_FALLBACK;
    }
    return expert_pack_open_owned_fd(
        out, fd, 0, (uint64_t)st.st_size, expected_geometry,
        "expert pack", error, error_size);
}

ds4_qwen_expert_pack_result ds4_qwen_expert_pack_open_embedded(
        ds4_qwen_expert_pack                 **out,
        int                                    fd,
        uint64_t                               offset,
        uint64_t                               bytes,
        const ds4_qwen_expert_pack_geometry   *expected_geometry,
        char                                  *error,
        size_t                                 error_size) {
    if (out) *out = NULL;
    if (!out || fd < 0 || bytes < PACK_HEADER_BYTES ||
        !geometry_valid(expected_geometry)) {
        set_error(error, error_size,
                  "invalid embedded expert pack descriptor or geometry");
        return DS4_QWEN_EXPERT_PACK_ERROR;
    }
    const int owned_fd = dup(fd);
    if (owned_fd < 0) {
        set_error(error, error_size,
                  "cannot duplicate embedded expert pack descriptor: %s",
                  strerror(errno));
        return DS4_QWEN_EXPERT_PACK_ERROR;
    }
    return expert_pack_open_owned_fd(
        out, owned_fd, offset, bytes, expected_geometry,
        "embedded expert pack", error, error_size);
}

const ds4_qwen_expert_pack_manifest *ds4_qwen_expert_pack_manifest_get(
        const ds4_qwen_expert_pack *pack) {
    return pack ? &pack->manifest : NULL;
}

ds4_qwen_expert_pack_result ds4_qwen_expert_pack_validate_source_digest(
        ds4_qwen_expert_pack *pack,
        uint64_t               gguf_size,
        const uint8_t          gguf_sha256[32],
        char                  *error,
        size_t                 error_size) {
    if (!pack || !gguf_sha256) {
        set_error(error, error_size, "invalid source digest validation request");
        return DS4_QWEN_EXPERT_PACK_ERROR;
    }
    pack->source_validated = false;
    if (gguf_size != pack->manifest.source_size ||
        memcmp(gguf_sha256, pack->manifest.source_sha256, 32) != 0) {
        set_error(error, error_size,
                  "expert pack belongs to a different GGUF; use GGUF fallback");
        return DS4_QWEN_EXPERT_PACK_FALLBACK;
    }
    pack->source_validated = true;
    if (error && error_size) error[0] = '\0';
    return DS4_QWEN_EXPERT_PACK_OK;
}

static bool stat_stable(const struct stat *a, const struct stat *b) {
#if defined(__APPLE__)
    const bool time_equal =
        a->st_mtime == b->st_mtime &&
        a->st_mtimensec == b->st_mtimensec &&
        a->st_ctime == b->st_ctime &&
        a->st_ctimensec == b->st_ctimensec;
#else
    const bool time_equal =
        a->st_mtim.tv_sec == b->st_mtim.tv_sec &&
        a->st_mtim.tv_nsec == b->st_mtim.tv_nsec &&
        a->st_ctim.tv_sec == b->st_ctim.tv_sec &&
        a->st_ctim.tv_nsec == b->st_ctim.tv_nsec;
#endif
    return a->st_dev == b->st_dev && a->st_ino == b->st_ino &&
           a->st_size == b->st_size && time_equal;
}

ds4_qwen_expert_pack_result ds4_qwen_expert_pack_validate_source_file(
        ds4_qwen_expert_pack *pack,
        const char            *gguf_path,
        char                  *error,
        size_t                 error_size) {
    if (error && error_size) error[0] = '\0';
    if (!pack || !gguf_path) {
        set_error(error, error_size, "invalid source-file validation request");
        return DS4_QWEN_EXPERT_PACK_ERROR;
    }
    pack->source_validated = false;
    const int fd = open(gguf_path, O_RDONLY);
    if (fd < 0) {
        set_error(error, error_size, "cannot open source GGUF: %s",
                  strerror(errno));
        return DS4_QWEN_EXPERT_PACK_FALLBACK;
    }
    struct stat before;
    struct stat after;
    uint8_t digest[32];
    bool ok = fstat(fd, &before) == 0 && S_ISREG(before.st_mode) &&
              before.st_size >= 0 &&
              hash_fd_range(fd, 0, (uint64_t)before.st_size, digest,
                            NULL, NULL, DS4_QWEN_EXPERT_PACK_HASH_SOURCE,
                            error, error_size) &&
              fstat(fd, &after) == 0 && stat_stable(&before, &after);
    close(fd);
    if (!ok) {
        if (error && error_size && error[0] == '\0') {
            set_error(error, error_size,
                      "source GGUF changed while it was being hashed");
        }
        return DS4_QWEN_EXPERT_PACK_FALLBACK;
    }
    return ds4_qwen_expert_pack_validate_source_digest(
        pack, (uint64_t)before.st_size, digest, error, error_size);
}

ds4_qwen_expert_pack_result ds4_qwen_expert_pack_verify_payload(
        ds4_qwen_expert_pack *pack,
        char                  *error,
        size_t                 error_size) {
    if (!pack) {
        set_error(error, error_size, "invalid expert pack payload request");
        return DS4_QWEN_EXPERT_PACK_ERROR;
    }
    pack->payload_validated = false;
    uint8_t digest[32];
    if (!hash_fd_range(pack->fd,
                       pack->file_offset + pack->manifest.data_offset,
                       pack->manifest.data_size, digest,
                       NULL, NULL, DS4_QWEN_EXPERT_PACK_VERIFY_DATA,
                       error, error_size)) {
        return DS4_QWEN_EXPERT_PACK_FALLBACK;
    }
    if (memcmp(digest, pack->manifest.data_sha256, 32) != 0) {
        set_error(error, error_size,
                  "expert pack payload checksum mismatch; use GGUF fallback");
        return DS4_QWEN_EXPERT_PACK_FALLBACK;
    }
    pack->payload_validated = true;
    if (error && error_size) error[0] = '\0';
    return DS4_QWEN_EXPERT_PACK_OK;
}

ds4_qwen_expert_pack_result ds4_qwen_expert_pack_validate_payload_digest(
        ds4_qwen_expert_pack *pack,
        const uint8_t         payload_sha256[32],
        char                 *error,
        size_t                error_size) {
    if (!pack || !payload_sha256) {
        set_error(error, error_size,
                  "invalid expert pack payload digest request");
        return DS4_QWEN_EXPERT_PACK_ERROR;
    }
    pack->payload_validated = false;
    if (memcmp(payload_sha256, pack->manifest.data_sha256, 32) != 0) {
        set_error(error, error_size,
                  "expert pack payload identity mismatch; use GGUF fallback");
        return DS4_QWEN_EXPERT_PACK_FALLBACK;
    }
    pack->payload_validated = true;
    if (error && error_size) error[0] = '\0';
    return DS4_QWEN_EXPERT_PACK_OK;
}

bool ds4_qwen_expert_pack_span_get(
        const ds4_qwen_expert_pack *pack,
        uint32_t                    layer,
        uint32_t                    expert,
        ds4_qwen_expert_pack_span  *span) {
    if (!pack || !span || !pack->source_validated ||
        !pack->payload_validated ||
        layer >= pack->manifest.geometry.n_layer ||
        expert >= pack->manifest.geometry.n_expert) {
        return false;
    }
    const uint64_t index =
        (uint64_t)layer * pack->manifest.geometry.n_expert + expert;
    const pack_entry *entry = &pack->entry[index];
    *span = (ds4_qwen_expert_pack_span){
        .gate = {
            .offset = pack->file_offset + entry->offset,
            .size = entry->gate_bytes,
        },
        .up = {
            .offset = pack->file_offset + entry->offset + entry->gate_bytes,
            .size = entry->up_bytes,
        },
        .down = {
            .offset = pack->file_offset + entry->offset +
                      entry->gate_bytes + entry->up_bytes,
            .size = entry->down_bytes,
        },
    };
    return true;
}

int ds4_qwen_expert_pack_fd(const ds4_qwen_expert_pack *pack) {
    return pack ? pack->fd : -1;
}

uint64_t ds4_qwen_expert_pack_file_offset(
        const ds4_qwen_expert_pack *pack) {
    return pack ? pack->file_offset : 0;
}

static char *parent_directory(const char *path) {
    const char *slash = strrchr(path, '/');
    if (!slash) {
        char *dot = malloc(2);
        if (dot) memcpy(dot, ".", 2);
        return dot;
    }
    const size_t size = slash == path ? 1 : (size_t)(slash - path);
    char *directory = malloc(size + 1);
    if (!directory) return NULL;
    memcpy(directory, path, size);
    directory[size] = '\0';
    return directory;
}

static bool preflight_free_space(
        const char *pack_path,
        uint64_t file_size,
        uint64_t reserve_bytes,
        char *error,
        size_t error_size) {
    char *directory = parent_directory(pack_path);
    if (!directory) {
        set_error(error, error_size,
                  "out of memory while resolving destination directory");
        return false;
    }
    struct statvfs space;
    if (statvfs(directory, &space) != 0) {
        set_error(error, error_size, "cannot inspect free space in %s: %s",
                  directory, strerror(errno));
        free(directory);
        return false;
    }
    free(directory);
    const uint64_t unit = space.f_frsize != 0 ? space.f_frsize : space.f_bsize;
    if (unit == 0) {
        set_error(error, error_size,
                  "filesystem reported a zero allocation unit");
        return false;
    }
    const uint64_t available =
        space.f_bavail > UINT64_MAX / unit
            ? UINT64_MAX
            : (uint64_t)space.f_bavail * unit;
    uint64_t required = 0;
    if (!checked_add_u64(file_size, reserve_bytes, &required) ||
        available < required) {
        set_error(error, error_size,
                  "insufficient free space: need temporary pack %" PRIu64
                  " bytes plus reserve %" PRIu64 ", have %" PRIu64,
                  file_size, reserve_bytes, available);
        return false;
    }
    return true;
}

static bool source_is_destination(
        int source_fd,
        const char *pack_path,
        char *error,
        size_t error_size) {
    struct stat source;
    struct stat destination;
    if (fstat(source_fd, &source) != 0) {
        set_error(error, error_size, "cannot stat source GGUF: %s",
                  strerror(errno));
        return true;
    }
    if (stat(pack_path, &destination) != 0) {
        if (errno == ENOENT) return false;
        set_error(error, error_size, "cannot stat pack destination: %s",
                  strerror(errno));
        return true;
    }
    if (source.st_dev == destination.st_dev &&
        source.st_ino == destination.st_ino) {
        set_error(error, error_size,
                  "pack destination must not replace the source GGUF");
        return true;
    }
    return false;
}

static bool copy_source_range(
        int source_fd,
        uint64_t source_offset,
        int destination_fd,
        uint64_t destination_offset,
        uint64_t size,
        uint8_t *buffer,
        sha256_context *hash,
        char *error,
        size_t error_size) {
    uint64_t copied = 0;
    while (copied < size) {
        size_t take = PACK_IO_BYTES;
        if ((uint64_t)take > size - copied) take = (size_t)(size - copied);
        if (!pread_all(source_fd, buffer, take, source_offset + copied,
                       error, error_size) ||
            !pwrite_all(destination_fd, buffer, take,
                        destination_offset + copied, error, error_size)) {
            return false;
        }
        sha256_update(hash, buffer, take);
        copied += take;
    }
    return true;
}

static bool write_payload(
        const gguf_expert_layout                 *source,
        int                                       destination_fd,
        const pack_entry                        *entry,
        uint8_t                                   digest[32],
        const ds4_qwen_expert_pack_build_options *options,
        char                                     *error,
        size_t                                    error_size) {
    if (!entry || source->geometry.n_layer == 0 ||
        source->geometry.n_expert == 0) {
        set_error(error, error_size, "invalid expert payload layout");
        return false;
    }
    uint8_t *buffer = malloc(PACK_IO_BYTES);
    if (!buffer) {
        set_error(error, error_size,
                  "out of memory while copying expert payload");
        return false;
    }
    sha256_context hash;
    sha256_init(&hash);
    uint64_t completed = 0;
    uint64_t total = 0;
    uint64_t entry_count = 0;
    uint64_t per_expert = 0;
    if (!checked_add_u64(entry[0].gate_bytes, entry[0].up_bytes,
                         &per_expert) ||
        !checked_add_u64(per_expert, entry[0].down_bytes, &per_expert) ||
        !checked_mul_u64(source->geometry.n_layer,
                         source->geometry.n_expert, &entry_count) ||
        !checked_mul_u64(entry_count, per_expert, &total)) {
        set_error(error, error_size, "expert payload size overflows");
        free(buffer);
        return false;
    }
    if (options->progress) {
        options->progress(options->progress_context,
                          DS4_QWEN_EXPERT_PACK_WRITE_DATA, 0, total);
    }
    for (uint32_t layer = 0; layer < source->geometry.n_layer; layer++) {
        const gguf_expert_matrix *matrix =
            &source->matrix[(uint64_t)layer * 3];
        for (uint32_t expert = 0; expert < source->geometry.n_expert; expert++) {
            const uint64_t index =
                (uint64_t)layer * source->geometry.n_expert + expert;
            uint64_t destination = entry[index].offset;
            for (uint32_t kind = 0; kind < 3; kind++) {
                /* GGUF stores each whole gate/up/down tensor contiguously by
                 * expert.  The pack changes only physical ordering, copying
                 * each expert slice verbatim into its combined record. */
                const uint64_t source_offset =
                    matrix[kind].abs_offset +
                    (uint64_t)expert * matrix[kind].expert_bytes;
                if (!copy_source_range(
                        source->fd, source_offset,
                        destination_fd, destination,
                        matrix[kind].expert_bytes, buffer, &hash,
                        error, error_size)) {
                    free(buffer);
                    return false;
                }
                destination += matrix[kind].expert_bytes;
                completed += matrix[kind].expert_bytes;
                if (options->progress) {
                    options->progress(options->progress_context,
                                      DS4_QWEN_EXPERT_PACK_WRITE_DATA,
                                      completed, total);
                }
            }
        }
    }
    sha256_final(&hash, digest);
    free(buffer);
    return completed == total;
}

static bool compare_ranges(
        int source_fd,
        uint64_t source_offset,
        int pack_fd,
        uint64_t pack_offset,
        uint64_t size,
        uint8_t *source_buffer,
        uint8_t *pack_buffer,
        char *error,
        size_t error_size) {
    uint64_t compared = 0;
    while (compared < size) {
        size_t take = PACK_IO_BYTES;
        if ((uint64_t)take > size - compared) take = (size_t)(size - compared);
        if (!pread_all(source_fd, source_buffer, take,
                       source_offset + compared, error, error_size) ||
            !pread_all(pack_fd, pack_buffer, take,
                       pack_offset + compared, error, error_size)) {
            return false;
        }
        if (memcmp(source_buffer, pack_buffer, take) != 0) {
            set_error(error, error_size,
                      "packed bytes differ from GGUF at packed offset %" PRIu64,
                      pack_offset + compared);
            return false;
        }
        compared += take;
    }
    return true;
}

static bool verify_source_spans(
        const gguf_expert_layout                 *source,
        ds4_qwen_expert_pack                     *pack,
        const ds4_qwen_expert_pack_build_options *options,
        char                                     *error,
        size_t                                    error_size) {
    uint8_t *source_buffer = malloc(PACK_IO_BYTES);
    uint8_t *pack_buffer = malloc(PACK_IO_BYTES);
    if (!source_buffer || !pack_buffer) {
        free(source_buffer);
        free(pack_buffer);
        set_error(error, error_size,
                  "out of memory while verifying expert payload");
        return false;
    }
    uint64_t completed = 0;
    const uint64_t total = pack->manifest.data_size;
    if (options->progress) {
        options->progress(options->progress_context,
                          DS4_QWEN_EXPERT_PACK_VERIFY_SOURCE_SPANS, 0, total);
    }
    for (uint32_t layer = 0; layer < source->geometry.n_layer; layer++) {
        const gguf_expert_matrix *matrix =
            &source->matrix[(uint64_t)layer * 3];
        for (uint32_t expert = 0; expert < source->geometry.n_expert; expert++) {
            ds4_qwen_expert_pack_span span;
            if (!ds4_qwen_expert_pack_span_get(pack, layer, expert, &span)) {
                set_error(error, error_size,
                          "validated pack did not expose expert span");
                free(source_buffer);
                free(pack_buffer);
                return false;
            }
            const ds4_qwen_expert_pack_slice slice[3] = {
                span.gate, span.up, span.down,
            };
            for (uint32_t kind = 0; kind < 3; kind++) {
                const uint64_t source_offset =
                    matrix[kind].abs_offset +
                    (uint64_t)expert * matrix[kind].expert_bytes;
                if (slice[kind].size != matrix[kind].expert_bytes ||
                    !compare_ranges(source->fd, source_offset,
                                    ds4_qwen_expert_pack_fd(pack),
                                    slice[kind].offset, slice[kind].size,
                                    source_buffer, pack_buffer,
                                    error, error_size)) {
                    free(source_buffer);
                    free(pack_buffer);
                    return false;
                }
                completed += slice[kind].size;
                if (options->progress) {
                    options->progress(
                        options->progress_context,
                        DS4_QWEN_EXPERT_PACK_VERIFY_SOURCE_SPANS,
                        completed, total);
                }
            }
        }
    }
    free(source_buffer);
    free(pack_buffer);
    return completed == total;
}

static bool fsync_parent_directory(
        const char *path,
        char *error,
        size_t error_size) {
    char *directory = parent_directory(path);
    if (!directory) {
        set_error(error, error_size,
                  "out of memory while syncing destination directory");
        return false;
    }
#ifdef O_DIRECTORY
    const int fd = open(directory, O_RDONLY | O_DIRECTORY);
#else
    const int fd = open(directory, O_RDONLY);
#endif
    if (fd < 0) {
        set_error(error, error_size, "cannot open destination directory: %s",
                  strerror(errno));
        free(directory);
        return false;
    }
    const bool ok = fsync(fd) == 0;
    if (!ok) {
        set_error(error, error_size, "cannot fsync destination directory: %s",
                  strerror(errno));
    }
    close(fd);
    free(directory);
    return ok;
}

bool ds4_qwen_expert_pack_build(
        const char                                *gguf_path,
        const char                                *pack_path,
        const ds4_qwen_expert_pack_build_options  *options,
        char                                      *error,
        size_t                                     error_size) {
    if (error && error_size) error[0] = '\0';
    if (!gguf_path || !pack_path || !options ||
        !geometry_valid(&options->geometry)) {
        set_error(error, error_size, "invalid expert pack build request");
        return false;
    }

    bool success = false;
    bool temp_exists = false;
    int temp_fd = -1;
    char *temp_path = NULL;
    uint8_t *table = NULL;
    pack_entry *entry = NULL;
    ds4_qwen_expert_pack *verification_pack = NULL;
    gguf_expert_layout source = { .fd = -1 };
    ds4_qwen_expert_pack_manifest manifest;
    struct stat source_before;
    struct stat source_after;

    if (!gguf_layout_open(&source, gguf_path, &options->geometry,
                          error, error_size) ||
        source_is_destination(source.fd, pack_path, error, error_size) ||
        !make_manifest_layout(&source, &manifest, &entry,
                              error, error_size) ||
        !preflight_free_space(pack_path, manifest.file_size,
                              options->filesystem_reserve_bytes,
                              error, error_size)) {
        goto cleanup;
    }
    if (fstat(source.fd, &source_before) != 0) {
        set_error(error, error_size, "cannot stat source GGUF: %s",
                  strerror(errno));
        goto cleanup;
    }
    if (!hash_fd_range(source.fd, 0, source.size, manifest.source_sha256,
                       options->progress, options->progress_context,
                       DS4_QWEN_EXPERT_PACK_HASH_SOURCE,
                       error, error_size)) {
        goto cleanup;
    }

    const size_t pack_path_size = strlen(pack_path);
    if (pack_path_size > SIZE_MAX - sizeof(".tmp.XXXXXX")) {
        set_error(error, error_size, "expert pack path is too long");
        goto cleanup;
    }
    temp_path = malloc(pack_path_size + sizeof(".tmp.XXXXXX"));
    if (!temp_path) {
        set_error(error, error_size,
                  "out of memory while creating temporary path");
        goto cleanup;
    }
    snprintf(temp_path, pack_path_size + sizeof(".tmp.XXXXXX"),
             "%s.tmp.XXXXXX", pack_path);
    temp_fd = mkstemp(temp_path);
    if (temp_fd < 0) {
        set_error(error, error_size, "cannot create temporary pack: %s",
                  strerror(errno));
        goto cleanup;
    }
    temp_exists = true;
    mode_t pack_mode = source_before.st_mode & 0666;
    if (pack_mode == 0) pack_mode = 0600;
    /* A derived weight file must never become more permissive than a private
     * source model merely because the builder used a conventional 0644. */
    if (fchmod(temp_fd, pack_mode) != 0 ||
        ftruncate(temp_fd, (off_t)manifest.file_size) != 0) {
        set_error(error, error_size,
                  "cannot size temporary pack to %" PRIu64 " bytes: %s",
                  manifest.file_size, strerror(errno));
        goto cleanup;
    }

    if (!write_payload(&source, temp_fd, entry, manifest.data_sha256,
                       options, error, error_size)) {
        goto cleanup;
    }

    uint64_t table_bytes_u64 = 0;
    if (!checked_mul_u64(manifest.entry_count, PACK_ENTRY_BYTES,
                         &table_bytes_u64) || table_bytes_u64 > SIZE_MAX) {
        set_error(error, error_size, "expert pack table size overflows");
        goto cleanup;
    }
    const size_t table_bytes = (size_t)table_bytes_u64;
    table = calloc(1, table_bytes);
    if (!table) {
        set_error(error, error_size,
                  "out of memory while encoding expert pack table");
        goto cleanup;
    }
    table_encode(table, entry, manifest.entry_count);
    uint8_t header[PACK_HEADER_BYTES];
    uint8_t index_sha[32];
    header_encode(header, &manifest, NULL);
    index_digest(header, table, table_bytes, index_sha);
    header_encode(header, &manifest, index_sha);
    if (!pwrite_all(temp_fd, header, sizeof(header), 0,
                    error, error_size) ||
        !pwrite_all(temp_fd, table, table_bytes, PACK_HEADER_BYTES,
                    error, error_size) ||
        fsync(temp_fd) != 0) {
        if (error && error_size && error[0] == '\0') {
            set_error(error, error_size, "cannot fsync temporary pack: %s",
                      strerror(errno));
        }
        goto cleanup;
    }
    if (close(temp_fd) != 0) {
        temp_fd = -1;
        set_error(error, error_size, "cannot close temporary pack: %s",
                  strerror(errno));
        goto cleanup;
    }
    temp_fd = -1;

    /* Reopen through the public reader before installation.  This catches a
     * writer/reader format disagreement, while the following two checks prove
     * both the aggregate checksum and every GGUF component offset. */
    if (ds4_qwen_expert_pack_open(
            &verification_pack, temp_path, &options->geometry,
            error, error_size) != DS4_QWEN_EXPERT_PACK_OK ||
        ds4_qwen_expert_pack_validate_source_digest(
            verification_pack, source.size, manifest.source_sha256,
            error, error_size) != DS4_QWEN_EXPERT_PACK_OK) {
        goto cleanup;
    }
    if (options->progress) {
        options->progress(options->progress_context,
                          DS4_QWEN_EXPERT_PACK_VERIFY_DATA,
                          0, manifest.data_size);
    }
    if (ds4_qwen_expert_pack_verify_payload(
            verification_pack, error,
            error_size) != DS4_QWEN_EXPERT_PACK_OK) {
        goto cleanup;
    }
    if (options->progress) {
        options->progress(options->progress_context,
                          DS4_QWEN_EXPERT_PACK_VERIFY_DATA,
                          manifest.data_size, manifest.data_size);
    }
    if (!verify_source_spans(&source, verification_pack, options,
                             error, error_size)) {
        goto cleanup;
    }
    ds4_qwen_expert_pack_close(verification_pack);
    verification_pack = NULL;

    if (fstat(source.fd, &source_after) != 0 ||
        !stat_stable(&source_before, &source_after)) {
        set_error(error, error_size,
                  "source GGUF changed while the sidecar was being built");
        goto cleanup;
    }
    if (rename(temp_path, pack_path) != 0) {
        set_error(error, error_size, "cannot atomically install expert pack: %s",
                  strerror(errno));
        goto cleanup;
    }
    temp_exists = false;
    success = true;
    /* Once rename succeeds there is no honest rollback: reporting failure
     * would violate the API guarantee that false leaves the destination
     * untouched.  Keep the valid installed pack and surface the weaker
     * crash-durability guarantee as a success warning instead. */
    (void)fsync_parent_directory(pack_path, error, error_size);

cleanup:
    ds4_qwen_expert_pack_close(verification_pack);
    if (temp_fd >= 0) close(temp_fd);
    if (temp_exists && temp_path) unlink(temp_path);
    free(temp_path);
    free(table);
    free(entry);
    gguf_layout_close(&source);
    return success;
}
