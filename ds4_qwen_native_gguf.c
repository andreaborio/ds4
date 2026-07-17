#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include "ds4_qwen_native_gguf.h"

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
    NATIVE_GGUF_HEADER_BYTES = 24,
    NATIVE_COPY_BYTES = 1024 * 1024,
    NATIVE_MAX_STRING_BYTES = 1024 * 1024,
    NATIVE_MAX_DIMS = 4,
    NATIVE_GGUF_MAGIC = 0x46554747,
    NATIVE_GGUF_VERSION = 3,
    NATIVE_GGUF_I8_TYPE = 24,
    NATIVE_PACK_ALIGNMENT = 4096,
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

typedef struct {
    uint32_t block_elems;
    uint32_t block_bytes;
} native_type_info;

static const native_type_info native_types[] = {
    [0] = {1, 4}, [1] = {1, 2}, [2] = {32, 18}, [3] = {32, 20},
    [6] = {32, 22}, [7] = {32, 24}, [8] = {32, 34}, [9] = {32, 40},
    [10] = {256, 84}, [11] = {256, 110}, [12] = {256, 144},
    [13] = {256, 176}, [14] = {256, 210}, [15] = {256, 292},
    [16] = {256, 66}, [17] = {256, 74}, [18] = {256, 98},
    [19] = {256, 110}, [20] = {256, 50}, [21] = {256, 110},
    [22] = {256, 82}, [23] = {256, 136}, [24] = {1, 1},
    [25] = {1, 2}, [26] = {1, 4}, [27] = {1, 8}, [28] = {1, 8},
    [29] = {256, 56}, [30] = {1, 2},
};

typedef struct {
    char *name;
    uint32_t ndim;
    uint64_t dim[NATIVE_MAX_DIMS];
    uint32_t type;
    uint64_t rel_offset;
    uint64_t abs_offset;
    uint64_t elements;
    uint64_t bytes;
    uint64_t output_rel_offset;
    int routed_kind;
    uint32_t routed_layer;
} native_tensor;

typedef struct {
    int fd;
    uint64_t size;
    uint32_t version;
    uint64_t n_tensors;
    uint64_t n_kv;
    uint64_t alignment;
    uint64_t metadata_end;
    uint64_t tensor_data_offset;
    native_tensor *tensor;
} native_model;

typedef struct {
    int fd;
    uint64_t size;
    uint64_t pos;
    char *error;
    size_t error_size;
} native_reader;

static void set_error(char *error, size_t error_size, const char *fmt, ...) {
    if (!error || error_size == 0) return;
    va_list args;
    va_start(args, fmt);
    vsnprintf(error, error_size, fmt, args);
    va_end(args);
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

static bool align_up_u64(uint64_t value, uint64_t alignment, uint64_t *out) {
    if (!out || alignment == 0) return false;
    const uint64_t rem = value % alignment;
    return rem == 0 ? (*out = value, true) :
        add_u64(value, alignment - rem, out);
}

static uint32_t load_u32_le(const uint8_t p[4]) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static uint64_t load_u64_le(const uint8_t p[8]) {
    return (uint64_t)load_u32_le(p) |
           ((uint64_t)load_u32_le(p + 4) << 32);
}

static void store_u32_le(uint8_t p[4], uint32_t value) {
    p[0] = (uint8_t)value;
    p[1] = (uint8_t)(value >> 8);
    p[2] = (uint8_t)(value >> 16);
    p[3] = (uint8_t)(value >> 24);
}

static void store_u64_le(uint8_t p[8], uint64_t value) {
    store_u32_le(p, (uint32_t)value);
    store_u32_le(p + 4, (uint32_t)(value >> 32));
}

static bool pread_all(
        int fd, void *buffer_pointer, size_t size, uint64_t offset,
        char *error, size_t error_size) {
    uint8_t *buffer = buffer_pointer;
    while (size != 0) {
        const ssize_t got = pread(fd, buffer, size, (off_t)offset);
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) {
            set_error(error, error_size,
                      "cannot read byte %" PRIu64 ": %s", offset,
                      got < 0 ? strerror(errno) : "unexpected end of file");
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
        if (wrote <= 0) {
            set_error(error, error_size,
                      "cannot write byte %" PRIu64 ": %s", offset,
                      wrote < 0 ? strerror(errno) : "write made no progress");
            return false;
        }
        buffer += (size_t)wrote;
        size -= (size_t)wrote;
        offset += (uint64_t)wrote;
    }
    return true;
}

static bool reader_bytes(native_reader *reader, void *out, size_t bytes) {
    if (reader->pos > reader->size ||
        (uint64_t)bytes > reader->size - reader->pos) {
        set_error(reader->error, reader->error_size,
                  "GGUF metadata or directory is truncated");
        return false;
    }
    if (!pread_all(reader->fd, out, bytes, reader->pos,
                   reader->error, reader->error_size)) {
        return false;
    }
    reader->pos += bytes;
    return true;
}

static bool reader_skip(native_reader *reader, uint64_t bytes) {
    if (reader->pos > reader->size || bytes > reader->size - reader->pos) {
        set_error(reader->error, reader->error_size,
                  "GGUF metadata value is truncated");
        return false;
    }
    reader->pos += bytes;
    return true;
}

static bool reader_u32(native_reader *reader, uint32_t *out) {
    uint8_t bytes[4];
    if (!reader_bytes(reader, bytes, sizeof(bytes))) return false;
    *out = load_u32_le(bytes);
    return true;
}

static bool reader_u64(native_reader *reader, uint64_t *out) {
    uint8_t bytes[8];
    if (!reader_bytes(reader, bytes, sizeof(bytes))) return false;
    *out = load_u64_le(bytes);
    return true;
}

static bool reader_string(native_reader *reader, char **out) {
    *out = NULL;
    uint64_t bytes = 0;
    if (!reader_u64(reader, &bytes) || bytes > NATIVE_MAX_STRING_BYTES ||
        bytes > SIZE_MAX - 1 || reader->pos > reader->size ||
        bytes > reader->size - reader->pos) {
        set_error(reader->error, reader->error_size,
                  "GGUF string is invalid or too large");
        return false;
    }
    char *text = malloc((size_t)bytes + 1);
    if (!text) {
        set_error(reader->error, reader->error_size,
                  "out of memory while reading GGUF string");
        return false;
    }
    if (!reader_bytes(reader, text, (size_t)bytes)) {
        free(text);
        return false;
    }
    text[bytes] = '\0';
    *out = text;
    return true;
}

static uint64_t scalar_bytes(uint32_t type) {
    switch (type) {
    case GGUF_VALUE_UINT8:
    case GGUF_VALUE_INT8:
    case GGUF_VALUE_BOOL: return 1;
    case GGUF_VALUE_UINT16:
    case GGUF_VALUE_INT16: return 2;
    case GGUF_VALUE_UINT32:
    case GGUF_VALUE_INT32:
    case GGUF_VALUE_FLOAT32: return 4;
    case GGUF_VALUE_UINT64:
    case GGUF_VALUE_INT64:
    case GGUF_VALUE_FLOAT64: return 8;
    default: return 0;
    }
}

static bool reader_skip_value(native_reader *reader, uint32_t type, int depth) {
    if (depth > 8) {
        set_error(reader->error, reader->error_size,
                  "GGUF metadata array nesting is too deep");
        return false;
    }
    const uint64_t scalar = scalar_bytes(type);
    if (scalar != 0) return reader_skip(reader, scalar);
    if (type == GGUF_VALUE_STRING) {
        uint64_t bytes = 0;
        return reader_u64(reader, &bytes) && reader_skip(reader, bytes);
    }
    if (type != GGUF_VALUE_ARRAY) {
        set_error(reader->error, reader->error_size,
                  "unknown GGUF metadata type %u", type);
        return false;
    }
    uint32_t item_type = 0;
    uint64_t count = 0;
    if (!reader_u32(reader, &item_type) || !reader_u64(reader, &count)) {
        return false;
    }
    const uint64_t item_bytes = scalar_bytes(item_type);
    uint64_t total = 0;
    if (item_bytes != 0) {
        return mul_u64(count, item_bytes, &total) && reader_skip(reader, total);
    }
    for (uint64_t i = 0; i < count; i++) {
        if (!reader_skip_value(reader, item_type, depth + 1)) return false;
    }
    return true;
}

static bool tensor_bytes(
        uint32_t type, uint64_t elements, uint64_t *bytes) {
    if (type >= sizeof(native_types) / sizeof(native_types[0]) ||
        native_types[type].block_elems == 0) {
        return false;
    }
    const native_type_info info = native_types[type];
    uint64_t blocks = elements / info.block_elems;
    if (elements % info.block_elems != 0) blocks++;
    return mul_u64(blocks, info.block_bytes, bytes);
}

static int routed_tensor_kind(const char *name, uint32_t *layer_out) {
    unsigned layer = 0;
    const char *patterns[3] = {
        "blk.%u.ffn_gate_exps.weight%n",
        "blk.%u.ffn_up_exps.weight%n",
        "blk.%u.ffn_down_exps.weight%n",
    };
    for (int kind = 0; kind < 3; kind++) {
        int consumed = -1;
        const int count = sscanf(name, patterns[kind], &layer, &consumed);
        if (count == 1 && consumed >= 0 && name[consumed] == '\0' &&
            layer <= UINT32_MAX) {
            *layer_out = (uint32_t)layer;
            return kind;
        }
    }
    return -1;
}

static void native_model_close(native_model *model) {
    if (!model) return;
    if (model->tensor) {
        for (uint64_t i = 0; i < model->n_tensors; i++) {
            free(model->tensor[i].name);
        }
    }
    free(model->tensor);
    if (model->fd >= 0) close(model->fd);
    memset(model, 0, sizeof(*model));
    model->fd = -1;
}

static bool native_model_open(
        native_model *model,
        const char   *path,
        char         *error,
        size_t        error_size) {
    memset(model, 0, sizeof(*model));
    model->fd = -1;
    model->fd = open(path, O_RDONLY);
    if (model->fd < 0) {
        set_error(error, error_size, "cannot open GGUF %s: %s",
                  path, strerror(errno));
        return false;
    }
#if defined(__APPLE__) && defined(F_NOCACHE)
    /* Conversion and publication verification are sequential one-shot I/O;
     * bypassing the unified cache avoids displacing a concurrently loaded
     * model. The hint is descriptor-local and failure is harmless. */
    (void)fcntl(model->fd, F_NOCACHE, 1);
#endif
    struct stat st;
    if (fstat(model->fd, &st) != 0 || !S_ISREG(st.st_mode) ||
        st.st_size < NATIVE_GGUF_HEADER_BYTES) {
        set_error(error, error_size, "GGUF is not a regular v3 file");
        native_model_close(model);
        return false;
    }
    model->size = (uint64_t)st.st_size;
    native_reader reader = {
        .fd = model->fd,
        .size = model->size,
        .error = error,
        .error_size = error_size,
    };
    uint32_t magic = 0;
    if (!reader_u32(&reader, &magic) ||
        !reader_u32(&reader, &model->version) ||
        !reader_u64(&reader, &model->n_tensors) ||
        !reader_u64(&reader, &model->n_kv) ||
        magic != NATIVE_GGUF_MAGIC || model->version != NATIVE_GGUF_VERSION ||
        model->n_tensors == 0 ||
        model->n_tensors > SIZE_MAX / sizeof(native_tensor)) {
        set_error(error, error_size, "unsupported or malformed GGUF header");
        native_model_close(model);
        return false;
    }
    model->alignment = 32;
    for (uint64_t i = 0; i < model->n_kv; i++) {
        char *key = NULL;
        uint32_t type = 0;
        if (!reader_string(&reader, &key) || !reader_u32(&reader, &type)) {
            free(key);
            native_model_close(model);
            return false;
        }
        if (strcmp(key, "general.alignment") == 0 &&
            type == GGUF_VALUE_UINT32) {
            uint32_t alignment = 0;
            if (!reader_u32(&reader, &alignment) || alignment == 0) {
                free(key);
                set_error(error, error_size, "invalid GGUF alignment");
                native_model_close(model);
                return false;
            }
            model->alignment = alignment;
        } else if (!reader_skip_value(&reader, type, 0)) {
            free(key);
            native_model_close(model);
            return false;
        }
        free(key);
    }
    model->metadata_end = reader.pos;
    model->tensor = calloc((size_t)model->n_tensors, sizeof(model->tensor[0]));
    if (!model->tensor) {
        set_error(error, error_size,
                  "out of memory while reading GGUF tensor directory");
        native_model_close(model);
        return false;
    }
    for (uint64_t i = 0; i < model->n_tensors; i++) {
        native_tensor *tensor = &model->tensor[i];
        tensor->routed_kind = -1;
        if (!reader_string(&reader, &tensor->name) ||
            !reader_u32(&reader, &tensor->ndim) ||
            tensor->ndim == 0 || tensor->ndim > NATIVE_MAX_DIMS) {
            set_error(error, error_size, "invalid GGUF tensor descriptor");
            native_model_close(model);
            return false;
        }
        tensor->elements = 1;
        for (uint32_t d = 0; d < tensor->ndim; d++) {
            if (!reader_u64(&reader, &tensor->dim[d]) ||
                !mul_u64(tensor->elements, tensor->dim[d],
                         &tensor->elements)) {
                set_error(error, error_size, "GGUF tensor dimensions overflow");
                native_model_close(model);
                return false;
            }
        }
        if (!reader_u32(&reader, &tensor->type) ||
            !reader_u64(&reader, &tensor->rel_offset) ||
            !tensor_bytes(tensor->type, tensor->elements, &tensor->bytes)) {
            set_error(error, error_size,
                      "GGUF tensor %s has unsupported type or size",
                      tensor->name);
            native_model_close(model);
            return false;
        }
        tensor->routed_kind = routed_tensor_kind(
            tensor->name, &tensor->routed_layer);
    }
    if (!align_up_u64(reader.pos, model->alignment,
                      &model->tensor_data_offset)) {
        set_error(error, error_size, "GGUF tensor data offset overflows");
        native_model_close(model);
        return false;
    }
    for (uint64_t i = 0; i < model->n_tensors; i++) {
        native_tensor *tensor = &model->tensor[i];
        if (!add_u64(model->tensor_data_offset, tensor->rel_offset,
                     &tensor->abs_offset) ||
            tensor->abs_offset > model->size ||
            tensor->bytes > model->size - tensor->abs_offset) {
            set_error(error, error_size,
                      "GGUF tensor %s points outside the file", tensor->name);
            native_model_close(model);
            return false;
        }
    }
    return true;
}

static bool source_layout_valid(
        const native_model                       *source,
        const ds4_qwen_expert_pack_geometry      *geometry,
        char                                     *error,
        size_t                                    error_size) {
    if (!source || !geometry ||
        source->n_tensors != geometry->gguf_tensor_count ||
        source->alignment == 0 ||
        source->alignment > NATIVE_PACK_ALIGNMENT ||
        NATIVE_PACK_ALIGNMENT % source->alignment != 0) {
        set_error(error, error_size,
                  "source GGUF inventory or alignment is incompatible");
        return false;
    }
    uint8_t *seen = calloc((size_t)geometry->n_layer * 3, 1);
    if (!seen) {
        set_error(error, error_size,
                  "out of memory while validating routed tensor inventory");
        return false;
    }
    uint64_t routed = 0;
    bool valid = true;
    for (uint64_t i = 0; i < source->n_tensors && valid; i++) {
        const native_tensor *tensor = &source->tensor[i];
        if (tensor->routed_kind < 0) continue;
        const uint32_t layer = tensor->routed_layer;
        const uint32_t kind = (uint32_t)tensor->routed_kind;
        const uint64_t d0 =
            kind == 2 ? geometry->n_ff_exp : geometry->n_embd;
        const uint64_t d1 =
            kind == 2 ? geometry->n_embd : geometry->n_ff_exp;
        if (layer >= geometry->n_layer ||
            seen[(uint64_t)layer * 3 + kind] ||
            tensor->ndim != 3 ||
            tensor->type != geometry->quant_type ||
            tensor->dim[0] != d0 || tensor->dim[1] != d1 ||
            tensor->dim[2] != geometry->n_expert) {
            valid = false;
            break;
        }
        seen[(uint64_t)layer * 3 + kind] = 1;
        routed++;
    }
    if (valid) {
        const uint64_t expected = (uint64_t)geometry->n_layer * 3;
        valid = routed == expected;
        for (uint64_t i = 0; i < expected && valid; i++) {
            valid = seen[i] != 0;
        }
    }
    free(seen);
    if (!valid) {
        set_error(error, error_size,
                  "source GGUF routed tensor set does not match Qwen geometry");
    }
    return valid;
}

static uint64_t tensor_descriptor_bytes(const native_tensor *tensor) {
    return 8 + strlen(tensor->name) + 4 +
           (uint64_t)tensor->ndim * 8 + 4 + 8;
}

static bool write_u32_at(
        int fd, uint64_t *position, uint32_t value,
        char *error, size_t error_size) {
    uint8_t bytes[4];
    store_u32_le(bytes, value);
    if (!pwrite_all(fd, bytes, sizeof(bytes), *position, error, error_size)) {
        return false;
    }
    *position += sizeof(bytes);
    return true;
}

static bool write_u64_at(
        int fd, uint64_t *position, uint64_t value,
        char *error, size_t error_size) {
    uint8_t bytes[8];
    store_u64_le(bytes, value);
    if (!pwrite_all(fd, bytes, sizeof(bytes), *position, error, error_size)) {
        return false;
    }
    *position += sizeof(bytes);
    return true;
}

static bool write_string_at(
        int fd, uint64_t *position, const char *text,
        char *error, size_t error_size) {
    const size_t bytes = strlen(text);
    if (!write_u64_at(fd, position, bytes, error, error_size) ||
        !pwrite_all(fd, text, bytes, *position, error, error_size)) {
        return false;
    }
    *position += bytes;
    return true;
}

static bool write_tensor_descriptor(
        int fd, uint64_t *position, const native_tensor *tensor,
        char *error, size_t error_size) {
    if (!write_string_at(fd, position, tensor->name, error, error_size) ||
        !write_u32_at(fd, position, tensor->ndim, error, error_size)) {
        return false;
    }
    for (uint32_t d = 0; d < tensor->ndim; d++) {
        if (!write_u64_at(fd, position, tensor->dim[d], error, error_size)) {
            return false;
        }
    }
    return write_u32_at(fd, position, tensor->type, error, error_size) &&
           write_u64_at(fd, position, tensor->output_rel_offset,
                        error, error_size);
}

static bool copy_range(
        int source_fd, uint64_t source_offset,
        int destination_fd, uint64_t destination_offset,
        uint64_t bytes, uint8_t *buffer,
        uint64_t *completed, uint64_t total,
        const ds4_qwen_native_gguf_options *options,
        ds4_qwen_expert_pack_phase phase,
        char *error, size_t error_size) {
    uint64_t copied = 0;
    while (copied < bytes) {
        size_t take = NATIVE_COPY_BYTES;
        if ((uint64_t)take > bytes - copied) {
            take = (size_t)(bytes - copied);
        }
        if (!pread_all(source_fd, buffer, take, source_offset + copied,
                       error, error_size) ||
            !pwrite_all(destination_fd, buffer, take,
                        destination_offset + copied, error, error_size)) {
            return false;
        }
        copied += take;
        *completed += take;
        if (options->progress) {
            options->progress(options->progress_context, phase,
                              *completed, total);
        }
    }
    return true;
}

static bool compare_range(
        int a_fd, uint64_t a_offset,
        int b_fd, uint64_t b_offset,
        uint64_t bytes, uint8_t *a, uint8_t *b,
        uint64_t *completed, uint64_t total,
        const ds4_qwen_native_gguf_options *options,
        char *error, size_t error_size) {
    uint64_t compared = 0;
    while (compared < bytes) {
        size_t take = NATIVE_COPY_BYTES;
        if ((uint64_t)take > bytes - compared) {
            take = (size_t)(bytes - compared);
        }
        if (!pread_all(a_fd, a, take, a_offset + compared,
                       error, error_size) ||
            !pread_all(b_fd, b, take, b_offset + compared,
                       error, error_size)) {
            return false;
        }
        if (memcmp(a, b, take) != 0) {
            set_error(error, error_size,
                      "native GGUF comparison failed at source byte %" PRIu64,
                      a_offset + compared);
            return false;
        }
        compared += take;
        *completed += take;
        if (options->progress) {
            options->progress(options->progress_context,
                              DS4_QWEN_EXPERT_PACK_VERIFY_NATIVE_GGUF,
                              *completed, total);
        }
    }
    return true;
}

static native_tensor *find_tensor(native_model *model, const char *name) {
    for (uint64_t i = 0; i < model->n_tensors; i++) {
        if (strcmp(model->tensor[i].name, name) == 0) {
            return &model->tensor[i];
        }
    }
    return NULL;
}

static bool descriptor_equal(
        const native_tensor *a,
        const native_tensor *b) {
    if (!a || !b || a->ndim != b->ndim || a->type != b->type ||
        a->bytes != b->bytes) {
        return false;
    }
    for (uint32_t d = 0; d < a->ndim; d++) {
        if (a->dim[d] != b->dim[d]) return false;
    }
    return true;
}

static bool paths_are_same_file(const char *a, const char *b) {
    struct stat sa;
    struct stat sb;
    return stat(a, &sa) == 0 && stat(b, &sb) == 0 &&
           sa.st_dev == sb.st_dev && sa.st_ino == sb.st_ino;
}

static char *parent_directory(const char *path) {
    const char *slash = strrchr(path, '/');
    if (!slash) return strdup(".");
    const size_t bytes = slash == path ? 1 : (size_t)(slash - path);
    char *directory = malloc(bytes + 1);
    if (!directory) return NULL;
    memcpy(directory, path, bytes);
    directory[bytes] = '\0';
    return directory;
}

static bool free_space_available(
        const char *destination,
        uint64_t output_bytes,
        uint64_t reserve,
        char *error,
        size_t error_size) {
    char *directory = parent_directory(destination);
    if (!directory) {
        set_error(error, error_size,
                  "out of memory while checking destination space");
        return false;
    }
    struct statvfs vfs;
    const bool stat_ok = statvfs(directory, &vfs) == 0;
    free(directory);
    if (!stat_ok) {
        set_error(error, error_size,
                  "cannot inspect destination free space: %s",
                  strerror(errno));
        return false;
    }
    uint64_t available = 0;
    if (!mul_u64((uint64_t)vfs.f_bavail, (uint64_t)vfs.f_frsize,
                 &available)) {
        available = UINT64_MAX;
    }
    /* Atomic regeneration keeps an old destination alive until rename, so
     * its blocks are intentionally not counted as reclaimable free space. */
    uint64_t required = 0;
    if (!add_u64(output_bytes, reserve, &required) ||
        required > available) {
        set_error(error, error_size,
                  "insufficient free space: need %" PRIu64
                  " bytes including reserve, have %" PRIu64,
                  required, available);
        return false;
    }
    return true;
}

static bool fsync_parent(
        const char *path,
        char *error,
        size_t error_size) {
    char *directory = parent_directory(path);
    if (!directory) return false;
    const int fd = open(directory, O_RDONLY);
    free(directory);
    if (fd < 0) {
        set_error(error, error_size,
                  "cannot open destination directory for fsync: %s",
                  strerror(errno));
        return false;
    }
    const bool ok = fsync(fd) == 0;
    if (!ok) {
        set_error(error, error_size,
                  "cannot fsync destination directory: %s",
                  strerror(errno));
    }
    close(fd);
    return ok;
}

bool ds4_qwen_native_gguf_verify(
        const char                           *source_gguf_path,
        const char                           *native_gguf_path,
        const ds4_qwen_native_gguf_options   *options,
        char                                 *error,
        size_t                                error_size) {
    if (error && error_size) error[0] = '\0';
    if (!source_gguf_path || !native_gguf_path || !options) {
        set_error(error, error_size, "invalid native GGUF verify request");
        return false;
    }
    native_model source;
    native_model native;
    memset(&source, 0, sizeof(source));
    memset(&native, 0, sizeof(native));
    source.fd = native.fd = -1;
    ds4_qwen_expert_pack *embedded = NULL;
    uint8_t *a = NULL;
    uint8_t *b = NULL;
    bool ok = false;

    if (!native_model_open(&source, source_gguf_path, error, error_size) ||
        !source_layout_valid(&source, &options->geometry,
                             error, error_size) ||
        !native_model_open(&native, native_gguf_path, error, error_size)) {
        goto cleanup;
    }
    const uint64_t routed_count =
        (uint64_t)options->geometry.n_layer * 3;
    const uint64_t expected_tensors =
        source.n_tensors - routed_count + 1;
    native_tensor *store =
        find_tensor(&native, DS4_QWEN_NATIVE_EXPERT_TENSOR);
    if (native.n_kv != source.n_kv ||
        native.n_tensors != expected_tensors ||
        native.metadata_end != source.metadata_end ||
        native.alignment != source.alignment ||
        !store || store->ndim != 1 ||
        store->type != NATIVE_GGUF_I8_TYPE ||
        store->dim[0] != store->bytes) {
        set_error(error, error_size,
                  "native GGUF header, metadata, or store descriptor mismatch");
        goto cleanup;
    }

    uint64_t total =
        source.metadata_end - NATIVE_GGUF_HEADER_BYTES;
    for (uint64_t i = 0; i < source.n_tensors; i++) {
        if (source.tensor[i].routed_kind < 0 &&
            !add_u64(total, source.tensor[i].bytes, &total)) {
            set_error(error, error_size,
                      "native GGUF verification size overflows");
            goto cleanup;
        }
    }
    a = malloc(NATIVE_COPY_BYTES);
    b = malloc(NATIVE_COPY_BYTES);
    if (!a || !b) {
        set_error(error, error_size,
                  "out of memory while verifying native GGUF");
        goto cleanup;
    }
    uint64_t completed = 0;
    if (options->progress) {
        options->progress(options->progress_context,
                          DS4_QWEN_EXPERT_PACK_VERIFY_NATIVE_GGUF,
                          0, total);
    }
    if (!compare_range(
            source.fd, NATIVE_GGUF_HEADER_BYTES,
            native.fd, NATIVE_GGUF_HEADER_BYTES,
            source.metadata_end - NATIVE_GGUF_HEADER_BYTES,
            a, b, &completed, total, options, error, error_size)) {
        goto cleanup;
    }
    for (uint64_t i = 0; i < source.n_tensors; i++) {
        const native_tensor *source_tensor = &source.tensor[i];
        if (source_tensor->routed_kind >= 0) continue;
        native_tensor *native_tensor_value =
            find_tensor(&native, source_tensor->name);
        if (!descriptor_equal(source_tensor, native_tensor_value) ||
            !compare_range(
                source.fd, source_tensor->abs_offset,
                native.fd, native_tensor_value->abs_offset,
                source_tensor->bytes, a, b, &completed, total,
                options, error, error_size)) {
            if (error && error_size && error[0] == '\0') {
                set_error(error, error_size,
                          "native tensor %s differs from source",
                          source_tensor->name);
            }
            goto cleanup;
        }
    }
    if (ds4_qwen_expert_pack_open_embedded(
            &embedded, native.fd, store->abs_offset, store->bytes,
            &options->geometry, error,
            error_size) != DS4_QWEN_EXPERT_PACK_OK ||
        ds4_qwen_expert_pack_validate_source_file(
            embedded, source_gguf_path, error,
            error_size) != DS4_QWEN_EXPERT_PACK_OK ||
        ds4_qwen_expert_pack_verify_payload(
            embedded, error,
            error_size) != DS4_QWEN_EXPERT_PACK_OK) {
        goto cleanup;
    }
    if (options->progress) {
        options->progress(options->progress_context,
                          DS4_QWEN_EXPERT_PACK_VERIFY_NATIVE_GGUF,
                          total, total);
    }
    ok = true;

cleanup:
    free(a);
    free(b);
    ds4_qwen_expert_pack_close(embedded);
    native_model_close(&source);
    native_model_close(&native);
    return ok;
}

bool ds4_qwen_native_gguf_build(
        const char                           *source_gguf_path,
        const char                           *expert_pack_path,
        const char                           *destination_gguf_path,
        const ds4_qwen_native_gguf_options   *options,
        char                                 *error,
        size_t                                error_size) {
    if (error && error_size) error[0] = '\0';
    if (!source_gguf_path || !expert_pack_path ||
        !destination_gguf_path || !options ||
        paths_are_same_file(source_gguf_path, destination_gguf_path) ||
        paths_are_same_file(expert_pack_path, destination_gguf_path)) {
        set_error(error, error_size,
                  "invalid native GGUF paths or in-place destination");
        return false;
    }
    native_model source;
    memset(&source, 0, sizeof(source));
    source.fd = -1;
    ds4_qwen_expert_pack *pack = NULL;
    int temporary_fd = -1;
    char *temporary_path = NULL;
    bool temporary_exists = false;
    uint8_t *buffer = NULL;
    bool success = false;

    if (!native_model_open(&source, source_gguf_path, error, error_size) ||
        !source_layout_valid(&source, &options->geometry,
                             error, error_size) ||
        ds4_qwen_expert_pack_open(
            &pack, expert_pack_path, &options->geometry,
            error, error_size) != DS4_QWEN_EXPERT_PACK_OK ||
        ds4_qwen_expert_pack_validate_source_file(
            pack, source_gguf_path,
            error, error_size) != DS4_QWEN_EXPERT_PACK_OK ||
        ds4_qwen_expert_pack_verify_payload(
            pack, error, error_size) != DS4_QWEN_EXPERT_PACK_OK) {
        goto cleanup;
    }
    const ds4_qwen_expert_pack_manifest *manifest =
        ds4_qwen_expert_pack_manifest_get(pack);
    const uint64_t routed_count =
        (uint64_t)options->geometry.n_layer * 3;
    const uint64_t output_tensor_count =
        source.n_tensors - routed_count + 1;
    uint64_t directory_bytes = 0;
    uint64_t copy_bytes = manifest->file_size;
    for (uint64_t i = 0; i < source.n_tensors; i++) {
        native_tensor *tensor = &source.tensor[i];
        if (tensor->routed_kind >= 0) continue;
        if (!add_u64(directory_bytes, tensor_descriptor_bytes(tensor),
                     &directory_bytes) ||
            !add_u64(copy_bytes, tensor->bytes, &copy_bytes)) {
            set_error(error, error_size,
                      "native GGUF layout size overflows");
            goto cleanup;
        }
    }
    native_tensor store = {
        .name = (char *)DS4_QWEN_NATIVE_EXPERT_TENSOR,
        .ndim = 1,
        .dim = {manifest->file_size},
        .type = NATIVE_GGUF_I8_TYPE,
        .elements = manifest->file_size,
        .bytes = manifest->file_size,
        .routed_kind = -1,
    };
    if (!add_u64(directory_bytes, tensor_descriptor_bytes(&store),
                 &directory_bytes)) {
        set_error(error, error_size,
                  "native GGUF directory size overflows");
        goto cleanup;
    }
    uint64_t directory_end = 0;
    uint64_t tensor_data_offset = 0;
    if (!add_u64(source.metadata_end, directory_bytes, &directory_end) ||
        !align_up_u64(directory_end, source.alignment,
                      &tensor_data_offset)) {
        set_error(error, error_size,
                  "native GGUF data offset overflows");
        goto cleanup;
    }
    uint64_t relative = 0;
    for (uint64_t i = 0; i < source.n_tensors; i++) {
        native_tensor *tensor = &source.tensor[i];
        if (tensor->routed_kind >= 0) continue;
        if (!align_up_u64(relative, source.alignment, &relative)) {
            set_error(error, error_size,
                      "native tensor offset overflows");
            goto cleanup;
        }
        tensor->output_rel_offset = relative;
        if (!add_u64(relative, tensor->bytes, &relative)) {
            set_error(error, error_size,
                      "native tensor payload overflows");
            goto cleanup;
        }
    }
    uint64_t store_absolute = 0;
    if (!add_u64(tensor_data_offset, relative, &store_absolute) ||
        !align_up_u64(store_absolute, NATIVE_PACK_ALIGNMENT,
                      &store_absolute) ||
        store_absolute < tensor_data_offset) {
        set_error(error, error_size,
                  "embedded store alignment overflows");
        goto cleanup;
    }
    store.output_rel_offset = store_absolute - tensor_data_offset;
    uint64_t output_size = 0;
    if (!add_u64(store_absolute, manifest->file_size, &output_size) ||
        output_size > INT64_MAX ||
        !free_space_available(
            destination_gguf_path, output_size,
            options->filesystem_reserve_bytes,
            error, error_size)) {
        goto cleanup;
    }

    const size_t temporary_bytes =
        strlen(destination_gguf_path) + 64;
    temporary_path = malloc(temporary_bytes);
    if (!temporary_path) {
        set_error(error, error_size,
                  "out of memory while creating native GGUF temporary path");
        goto cleanup;
    }
    snprintf(temporary_path, temporary_bytes, "%s.tmp.%ld",
             destination_gguf_path, (long)getpid());
    temporary_fd =
        open(temporary_path, O_CREAT | O_EXCL | O_RDWR, 0600);
    if (temporary_fd < 0) {
        set_error(error, error_size,
                  "cannot create native GGUF temporary file: %s",
                  strerror(errno));
        goto cleanup;
    }
#if defined(__APPLE__) && defined(F_NOCACHE)
    (void)fcntl(temporary_fd, F_NOCACHE, 1);
#endif
    temporary_exists = true;
    if (ftruncate(temporary_fd, (off_t)output_size) != 0) {
        set_error(error, error_size,
                  "cannot size native GGUF temporary file: %s",
                  strerror(errno));
        goto cleanup;
    }

    uint64_t position = 0;
    if (!write_u32_at(
            temporary_fd, &position, NATIVE_GGUF_MAGIC,
            error, error_size) ||
        !write_u32_at(
            temporary_fd, &position, source.version,
            error, error_size) ||
        !write_u64_at(
            temporary_fd, &position, output_tensor_count,
            error, error_size) ||
        !write_u64_at(
            temporary_fd, &position, source.n_kv,
            error, error_size)) {
        goto cleanup;
    }
    buffer = malloc(NATIVE_COPY_BYTES);
    if (!buffer) {
        set_error(error, error_size,
                  "out of memory while writing native GGUF");
        goto cleanup;
    }
    const uint64_t metadata_bytes =
        source.metadata_end - NATIVE_GGUF_HEADER_BYTES;
    uint64_t progress_total = 0;
    if (!add_u64(copy_bytes, metadata_bytes, &progress_total)) {
        set_error(error, error_size,
                  "native GGUF progress size overflows");
        goto cleanup;
    }
    uint64_t completed = 0;
    if (options->progress) {
        options->progress(options->progress_context,
                          DS4_QWEN_EXPERT_PACK_WRITE_NATIVE_GGUF,
                          0, progress_total);
    }
    if (!copy_range(
            source.fd, NATIVE_GGUF_HEADER_BYTES,
            temporary_fd, NATIVE_GGUF_HEADER_BYTES,
            metadata_bytes, buffer, &completed,
            progress_total, options,
            DS4_QWEN_EXPERT_PACK_WRITE_NATIVE_GGUF,
            error, error_size)) {
        goto cleanup;
    }
    position = source.metadata_end;
    for (uint64_t i = 0; i < source.n_tensors; i++) {
        if (source.tensor[i].routed_kind < 0 &&
            !write_tensor_descriptor(
                temporary_fd, &position, &source.tensor[i],
                error, error_size)) {
            goto cleanup;
        }
    }
    if (!write_tensor_descriptor(
            temporary_fd, &position, &store,
            error, error_size) ||
        position != directory_end) {
        set_error(error, error_size,
                  "native GGUF directory writer disagrees with layout");
        goto cleanup;
    }
    for (uint64_t i = 0; i < source.n_tensors; i++) {
        const native_tensor *tensor = &source.tensor[i];
        if (tensor->routed_kind >= 0) continue;
        if (!copy_range(
                source.fd, tensor->abs_offset,
                temporary_fd,
                tensor_data_offset + tensor->output_rel_offset,
                tensor->bytes, buffer, &completed, progress_total,
                options, DS4_QWEN_EXPERT_PACK_WRITE_NATIVE_GGUF,
                error, error_size)) {
            goto cleanup;
        }
    }
    if (!copy_range(
            ds4_qwen_expert_pack_fd(pack),
            ds4_qwen_expert_pack_file_offset(pack),
            temporary_fd, store_absolute, manifest->file_size,
            buffer, &completed, progress_total, options,
            DS4_QWEN_EXPERT_PACK_WRITE_NATIVE_GGUF,
            error, error_size) ||
        fsync(temporary_fd) != 0) {
        if (error && error_size && error[0] == '\0') {
            set_error(error, error_size,
                      "cannot fsync native GGUF temporary file: %s",
                      strerror(errno));
        }
        goto cleanup;
    }
    if (close(temporary_fd) != 0) {
        temporary_fd = -1;
        set_error(error, error_size,
                  "cannot close native GGUF temporary file: %s",
                  strerror(errno));
        goto cleanup;
    }
    temporary_fd = -1;
    if (!ds4_qwen_native_gguf_verify(
            source_gguf_path, temporary_path, options,
            error, error_size)) {
        goto cleanup;
    }
    if (rename(temporary_path, destination_gguf_path) != 0) {
        set_error(error, error_size,
                  "cannot atomically install native GGUF: %s",
                  strerror(errno));
        goto cleanup;
    }
    temporary_exists = false;
    success = true;
    (void)fsync_parent(destination_gguf_path, error, error_size);

cleanup:
    free(buffer);
    if (temporary_fd >= 0) close(temporary_fd);
    if (temporary_exists && temporary_path) unlink(temporary_path);
    free(temporary_path);
    ds4_qwen_expert_pack_close(pack);
    native_model_close(&source);
    return success;
}
