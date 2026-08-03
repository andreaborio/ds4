/*
 * DeepSeek V4 Flash/Pro HF -> GGUF quantizer.
 *
 * This is a plain C, model-specific version of the DS4 quantization pipeline.
 * It deliberately keeps only the pieces needed by the DeepSeek V4 Flash and
 * Pro GGUF recipes used by this repository:
 *
 * - safetensors index/header loading;
 * - FP8 E4M3 + E8M0 dequantization for dense tensors;
 * - packed FP4 + E8M0 dequantization for routed experts;
 * - local Q8_0, Q4_K, Q2_K, and IQ2_XXS quantization;
 * - GGUF metadata/tensor-order reuse from an existing template GGUF.
 * - authenticated, template-free final-0731 DSpark support generation from
 *   the official config, index, and shards 46-48.
 *
 * The optional imatrix is the legacy llama.cpp binary .dat format emitted by
 * ds4's collector.  DS4 stores one packed vector per routed tensor, laid out as
 * n_experts consecutive per-expert importance vectors.  When no external
 * imatrix is supplied and IQ2_XXS requires one, this tool falls back to the
 * same synthetic weight-energy heuristic used by the old generator:
 * each column importance is sum(row[column]^2) over the dequantized weight.
 */

#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L

#include "quants.h"

#include <assert.h>
#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef __APPLE__
#include <sys/clonefile.h>
#endif

#if defined(_WIN32)
#error "deepseek4-quantize.c currently targets POSIX systems"
#endif

#define DS4_KV_QUANTIZE_IMATRIX_FILE      "quantize.imatrix.file"
#define DS4_KV_QUANTIZE_IMATRIX_DATASET   "quantize.imatrix.dataset"
#define DS4_KV_QUANTIZE_IMATRIX_N_ENTRIES "quantize.imatrix.entries_count"
#define DS4_KV_QUANTIZE_IMATRIX_N_CHUNKS  "quantize.imatrix.chunks_count"
#define DS4_GGUF_DEFAULT_ALIGNMENT 32
#define DS4_REUSE_KEY_HEXLEN 16          /* fnv1a64 hex digits in quantize.reuse_key */

#define DS4_DSPARK_STAGE_COUNT 3
#define DS4_DSPARK_EXPERT_COUNT 256
#define DS4_DSPARK_OUTPUT_TENSORS 81
#define DS4_DSPARK_SOURCE_TENSORS 4705
#define DS4_DSPARK_SOURCE_FILES 5

/*
 * Publication builds have no runtime provenance override. The test contract
 * exists only in the separately compiled synthetic test executable; the
 * normal Makefile never defines DS4_DSPARK_QUANTIZER_TEST_CONTRACT.
 */
#ifdef DS4_DSPARK_QUANTIZER_TEST_CONTRACT
#ifndef DS4_DSPARK_TEST_CONFIG_SHA256
#error "synthetic DSpark contract requires config SHA-256"
#endif
#ifndef DS4_DSPARK_TEST_INDEX_SHA256
#error "synthetic DSpark contract requires index SHA-256"
#endif
#ifndef DS4_DSPARK_TEST_SHARD46_SHA256
#error "synthetic DSpark contract requires shard 46 SHA-256"
#endif
#ifndef DS4_DSPARK_TEST_SHARD47_SHA256
#error "synthetic DSpark contract requires shard 47 SHA-256"
#endif
#ifndef DS4_DSPARK_TEST_SHARD48_SHA256
#error "synthetic DSpark contract requires shard 48 SHA-256"
#endif
#define DS4_DSPARK_SOURCE_REVISION "synthetic-test-contract"
#define DS4_DSPARK_CONFIG_SHA256 DS4_DSPARK_TEST_CONFIG_SHA256
#define DS4_DSPARK_INDEX_SHA256 DS4_DSPARK_TEST_INDEX_SHA256
#define DS4_DSPARK_CONFIG_BYTES DS4_DSPARK_TEST_CONFIG_BYTES
#define DS4_DSPARK_INDEX_BYTES DS4_DSPARK_TEST_INDEX_BYTES
#define DS4_DSPARK_INDEX_TENSORS DS4_DSPARK_TEST_INDEX_TENSORS
#define DS4_DSPARK_SHARD46_BYTES DS4_DSPARK_TEST_SHARD46_BYTES
#define DS4_DSPARK_SHARD47_BYTES DS4_DSPARK_TEST_SHARD47_BYTES
#define DS4_DSPARK_SHARD48_BYTES DS4_DSPARK_TEST_SHARD48_BYTES
#define DS4_DSPARK_SHARD46_SHA256 DS4_DSPARK_TEST_SHARD46_SHA256
#define DS4_DSPARK_SHARD47_SHA256 DS4_DSPARK_TEST_SHARD47_SHA256
#define DS4_DSPARK_SHARD48_SHA256 DS4_DSPARK_TEST_SHARD48_SHA256
#else
#define DS4_DSPARK_SOURCE_REVISION \
    "7872f01b1d1fe23eabc4c98b48bffcef5a386062"
#define DS4_DSPARK_CONFIG_SHA256 \
    "6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023"
#define DS4_DSPARK_INDEX_SHA256 \
    "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"
#define DS4_DSPARK_CONFIG_BYTES UINT64_C(1888)
#define DS4_DSPARK_INDEX_BYTES UINT64_C(5602871)
#define DS4_DSPARK_INDEX_TENSORS 72317
#define DS4_DSPARK_SHARD46_BYTES UINT64_C(3610455184)
#define DS4_DSPARK_SHARD47_BYTES UINT64_C(3560111960)
#define DS4_DSPARK_SHARD48_BYTES UINT64_C(3692775244)
#define DS4_DSPARK_SHARD46_SHA256 \
    "5db924ca907e0d93acd975bd5079c3662717f9ac709f23d079bd8f816d29d9dd"
#define DS4_DSPARK_SHARD47_SHA256 \
    "62816173f9f6e136b20b48e3b6f16613ac9ea02b5603f636928b253244a548bd"
#define DS4_DSPARK_SHARD48_SHA256 \
    "cc43742bd24ae6bcdea343a91442f6f66aed2cfebcc6b235470204851ce2f8a9"
#endif

#if defined(DS4_DSPARK_QUANTIZER_TEST_TINY_GEOMETRY) && \
    !defined(DS4_DSPARK_QUANTIZER_TEST_CONTRACT)
#error "tiny DSpark geometry is restricted to the synthetic test contract"
#endif

typedef enum {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
} gguf_value_type;

static void die(const char *msg) {
    fprintf(stderr, "error: %s\n", msg);
    exit(1);
}

static void die_errno(const char *what, const char *path) {
    fprintf(stderr, "error: %s %s: %s\n", what, path ? path : "", strerror(errno));
    exit(1);
}

static void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) die("out of memory");
    return p;
}

static void *xcalloc(size_t n, size_t sz) {
    void *p = calloc(n ? n : 1, sz ? sz : 1);
    if (!p) die("out of memory");
    return p;
}

static void *xrealloc(void *p, size_t n) {
    void *q = realloc(p, n ? n : 1);
    if (!q) die("out of memory");
    return q;
}

static char *xstrdup(const char *s) {
    size_t n = strlen(s);
    char *p = xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}

static char *xstrndup(const char *s, size_t n) {
    char *p = xmalloc(n + 1);
    memcpy(p, s, n);
    p[n] = '\0';
    return p;
}

static char *path_join(const char *a, const char *b) {
    const size_t na = strlen(a);
    const size_t nb = strlen(b);
    const bool slash = na && a[na - 1] == '/';
    char *out = xmalloc(na + (slash ? 0 : 1) + nb + 1);
    memcpy(out, a, na);
    size_t pos = na;
    if (!slash) out[pos++] = '/';
    memcpy(out + pos, b, nb + 1);
    return out;
}

static bool str_starts(const char *s, const char *prefix) {
    return strncmp(s, prefix, strlen(prefix)) == 0;
}

static bool str_ends(const char *s, const char *suffix) {
    const size_t ns = strlen(s);
    const size_t nf = strlen(suffix);
    return ns >= nf && memcmp(s + ns - nf, suffix, nf) == 0;
}

static char *read_file(const char *path, size_t *len_out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) die_errno("open", path);
    if (fseeko(fp, 0, SEEK_END) != 0) die_errno("seek", path);
    off_t n = ftello(fp);
    if (n < 0) die_errno("tell", path);
    if (fseeko(fp, 0, SEEK_SET) != 0) die_errno("seek", path);
    char *buf = xmalloc((size_t)n + 1);
    if (n && fread(buf, 1, (size_t)n, fp) != (size_t)n) die_errno("read", path);
    buf[n] = '\0';
    fclose(fp);
    if (len_out) *len_out = (size_t)n;
    return buf;
}

typedef struct {
    uint32_t state[8];
    uint64_t bytes;
    uint8_t block[64];
    size_t block_len;
} ds4q_sha256;

static uint32_t sha256_rotr(uint32_t value, unsigned shift) {
    return (value >> shift) | (value << (32u - shift));
}

static void sha256_transform(ds4q_sha256 *context,
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
        const uint32_t s0 = sha256_rotr(w[i - 15], 7) ^
                            sha256_rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = sha256_rotr(w[i - 2], 17) ^
                            sha256_rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = context->state[0], b = context->state[1];
    uint32_t c = context->state[2], d = context->state[3];
    uint32_t e = context->state[4], f = context->state[5];
    uint32_t g = context->state[6], h = context->state[7];
    for (size_t i = 0; i < 64; i++) {
        const uint32_t s1 = sha256_rotr(e, 6) ^ sha256_rotr(e, 11) ^
                            sha256_rotr(e, 25);
        const uint32_t ch = (e & f) ^ (~e & g);
        const uint32_t t1 = h + s1 + ch + k[i] + w[i];
        const uint32_t s0 = sha256_rotr(a, 2) ^ sha256_rotr(a, 13) ^
                            sha256_rotr(a, 22);
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

static void sha256_init(ds4q_sha256 *context) {
    *context = (ds4q_sha256){
        .state = {
            0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
            0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
        },
    };
}

static void sha256_update(ds4q_sha256 *context, const void *data_pointer,
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

static void sha256_final(ds4q_sha256 *context, uint8_t digest[32]) {
    const uint64_t bits = context->bytes * UINT64_C(8);
    context->block[context->block_len++] = 0x80;
    if (context->block_len > 56) {
        memset(context->block + context->block_len, 0,
               sizeof(context->block) - context->block_len);
        sha256_transform(context, context->block);
        context->block_len = 0;
    }
    memset(context->block + context->block_len, 0, 56 - context->block_len);
    for (size_t i = 0; i < 8; i++)
        context->block[56 + i] = (uint8_t)(bits >> (56 - i * 8));
    sha256_transform(context, context->block);
    for (size_t i = 0; i < 8; i++) {
        digest[i * 4] = (uint8_t)(context->state[i] >> 24);
        digest[i * 4 + 1] = (uint8_t)(context->state[i] >> 16);
        digest[i * 4 + 2] = (uint8_t)(context->state[i] >> 8);
        digest[i * 4 + 3] = (uint8_t)context->state[i];
    }
}

static void sha256_fd_hex(int fd, char hex[65], const char *label) {
    ds4q_sha256 context;
    sha256_init(&context);
    uint8_t *buffer = xmalloc(1u << 20);
    off_t offset = 0;
    for (;;) {
        ssize_t got = pread(fd, buffer, 1u << 20, offset);
        if (got < 0) die_errno("hash", label);
        if (got == 0) break;
        sha256_update(&context, buffer, (size_t)got);
        offset += got;
    }
    free(buffer);
    uint8_t digest[32];
    sha256_final(&context, digest);
    for (size_t i = 0; i < sizeof(digest); i++)
        snprintf(hex + i * 2, 3, "%02x", digest[i]);
    hex[64] = '\0';
}

typedef struct {
    dev_t device;
    ino_t inode;
    uint64_t size;
    char sha256[65];
} dspark_file_identity;

static dspark_file_identity require_fd_identity(
        int fd, const char *path, uint64_t expected_size,
        const char *expected_sha256, const char *label) {
    struct stat st;
    if (fstat(fd, &st) != 0) die_errno("fstat", path);
    if (!S_ISREG(st.st_mode)) {
        fprintf(stderr, "error: %s is not a regular file (%s)\n", label, path);
        exit(1);
    }
    if (st.st_size < 0 || (uint64_t)st.st_size != expected_size) {
        fprintf(stderr,
                "error: %s size mismatch: got %" PRIu64 " expected %" PRIu64
                " (%s)\n", label,
                st.st_size < 0 ? UINT64_C(0) : (uint64_t)st.st_size,
                expected_size, path);
        exit(1);
    }
    char actual[65];
    sha256_fd_hex(fd, actual, path);
    if (strcmp(actual, expected_sha256) != 0) {
        fprintf(stderr,
                "error: %s SHA-256 mismatch: got %s expected %s (%s)\n",
                label, actual, expected_sha256, path);
        exit(1);
    }
    dspark_file_identity result = {
        .device = st.st_dev,
        .inode = st.st_ino,
        .size = (uint64_t)st.st_size,
    };
    memcpy(result.sha256, actual, sizeof(result.sha256));
    return result;
}

static uint64_t read_u64_le_fp(FILE *fp, const char *what) {
    uint8_t b[8];
    if (fread(b, 1, sizeof(b), fp) != sizeof(b)) {
        fprintf(stderr, "error: short read while reading %s\n", what);
        exit(1);
    }
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)b[i] << (8 * i);
    return v;
}

static void pread_exact_fd(int fd, void *buffer, size_t size,
                           off_t offset, const char *what) {
    uint8_t *cursor = buffer;
    size_t done = 0;
    while (done < size) {
        ssize_t got = pread(fd, cursor + done, size - done,
                            offset + (off_t)done);
        if (got < 0) die_errno("pread", what);
        if (got == 0) {
            fprintf(stderr, "error: short read while reading %s\n", what);
            exit(1);
        }
        done += (size_t)got;
    }
}

static uint64_t pread_u64_le_fd(int fd, off_t offset, const char *what) {
    uint8_t bytes[8];
    pread_exact_fd(fd, bytes, sizeof(bytes), offset, what);
    uint64_t value = 0;
    for (int i = 0; i < 8; i++)
        value |= (uint64_t)bytes[i] << (8 * i);
    return value;
}

static uint32_t read_u32_le_fp(FILE *fp, const char *what) {
    uint32_t v;
    if (fread(&v, 1, sizeof(v), fp) != sizeof(v)) {
        fprintf(stderr, "error: short read while reading %s\n", what);
        exit(1);
    }
    return v;
}

static int32_t read_i32_fp(FILE *fp, const char *what) {
    int32_t v;
    if (fread(&v, 1, sizeof(v), fp) != sizeof(v)) {
        fprintf(stderr, "error: short read while reading %s\n", what);
        exit(1);
    }
    return v;
}

static uint16_t load_u16_le(const uint8_t *p) {
    return (uint16_t)p[0] | ((uint16_t)p[1] << 8);
}

static int64_t load_i64_le(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; i++) v |= (uint64_t)p[i] << (8 * i);
    return (int64_t)v;
}

/* =====
 * Minimal JSON tokenizer
 *
 * Safetensors uses ordinary JSON for the model index and per-shard headers.
 * We only need objects, arrays, strings, and primitive numbers; escaped tensor
 * names do not occur in the files produced by Hugging Face, so strings are
 * copied as raw UTF-8 slices after locating the closing quote.
 */

typedef enum {
    JT_OBJECT,
    JT_ARRAY,
    JT_STRING,
    JT_PRIMITIVE,
} json_type;

typedef struct {
    json_type type;
    int start;
    int end;
    int parent;
    int size;
} json_tok;

typedef struct {
    json_tok *v;
    int len;
    int cap;
    const char *js;
    int js_len;
} json_doc;

static int json_add(json_doc *d, json_type type, int start, int end, int parent) {
    if (d->len == d->cap) {
        d->cap = d->cap ? d->cap * 2 : 4096;
        d->v = xrealloc(d->v, (size_t)d->cap * sizeof(d->v[0]));
    }
    int id = d->len++;
    d->v[id] = (json_tok){ .type = type, .start = start, .end = end, .parent = parent, .size = 0 };
    if (parent >= 0) d->v[parent].size++;
    return id;
}

static json_doc json_parse_text(const char *js, size_t len) {
    json_doc d = { .js = js, .js_len = (int)len };
    int parent = -1;
    for (int i = 0; i < (int)len; i++) {
        unsigned char c = (unsigned char)js[i];
        if (isspace(c) || c == ':' || c == ',') continue;
        if (c == '{' || c == '[') {
            parent = json_add(&d, c == '{' ? JT_OBJECT : JT_ARRAY, i, -1, parent);
            continue;
        }
        if (c == '}' || c == ']') {
            if (parent < 0) die("bad JSON: unmatched close");
            d.v[parent].end = i + 1;
            parent = d.v[parent].parent;
            continue;
        }
        if (c == '"') {
            int start = i + 1;
            i++;
            bool esc = false;
            for (; i < (int)len; i++) {
                if (esc) {
                    esc = false;
                } else if (js[i] == '\\') {
                    esc = true;
                } else if (js[i] == '"') {
                    break;
                }
            }
            if (i >= (int)len) die("bad JSON: unterminated string");
            json_add(&d, JT_STRING, start, i, parent);
            continue;
        }
        int start = i;
        while (i < (int)len && !isspace((unsigned char)js[i]) &&
               js[i] != ',' && js[i] != ']' && js[i] != '}') {
            i++;
        }
        json_add(&d, JT_PRIMITIVE, start, i, parent);
        i--;
    }
    if (parent != -1) die("bad JSON: unterminated object/array");
    return d;
}

static void json_free(json_doc *d) {
    free(d->v);
    memset(d, 0, sizeof(*d));
}

static bool json_tok_eq(const json_doc *d, int tok, const char *s) {
    const json_tok *t = &d->v[tok];
    const int n = t->end - t->start;
    return t->type == JT_STRING && (int)strlen(s) == n && memcmp(d->js + t->start, s, (size_t)n) == 0;
}

static char *json_strdup_tok(const json_doc *d, int tok) {
    const json_tok *t = &d->v[tok];
    return xstrndup(d->js + t->start, (size_t)(t->end - t->start));
}

static bool json_is_descendant(const json_doc *d, int tok, int parent) {
    for (int p = d->v[tok].parent; p >= 0; p = d->v[p].parent) {
        if (p == parent) return true;
    }
    return false;
}

static int json_skip(const json_doc *d, int tok) {
    int i = tok + 1;
    while (i < d->len && json_is_descendant(d, i, tok)) i++;
    return i;
}

static int json_obj_get(const json_doc *d, int obj, const char *key) {
    if (obj < 0 || d->v[obj].type != JT_OBJECT) return -1;
    for (int i = obj + 1; i < d->len && d->v[i].parent == obj;) {
        int k = i;
        int v = i + 1;
        if (v >= d->len || d->v[v].parent != obj) return -1;
        if (json_tok_eq(d, k, key)) return v;
        i = json_skip(d, v);
    }
    return -1;
}

static int64_t json_i64(const json_doc *d, int tok) {
    char tmp[64];
    const int n = d->v[tok].end - d->v[tok].start;
    if (n <= 0 || n >= (int)sizeof(tmp)) die("bad JSON integer");
    memcpy(tmp, d->js + d->v[tok].start, (size_t)n);
    tmp[n] = '\0';
    return strtoll(tmp, NULL, 10);
}

/* =====
 * Small string hash map
 */

typedef struct {
    char *key;
    int value;
} hslot;

typedef struct {
    hslot *slots;
    int cap;
} hmap;

static uint64_t fnv1a_str(const char *s) {
    uint64_t h = 1469598103934665603ull;
    while (*s) {
        h ^= (uint8_t)*s++;
        h *= 1099511628211ull;
    }
    return h;
}

static void hmap_build(hmap *m, char **keys, int n) {
    int cap = 1;
    while (cap < n * 3) cap <<= 1;
    m->cap = cap ? cap : 2;
    m->slots = xcalloc((size_t)m->cap, sizeof(m->slots[0]));
    for (int i = 0; i < n; i++) {
        uint64_t h = fnv1a_str(keys[i]);
        int p = (int)(h & (uint64_t)(m->cap - 1));
        while (m->slots[p].key) p = (p + 1) & (m->cap - 1);
        m->slots[p].key = keys[i];
        m->slots[p].value = i;
    }
}

static int hmap_get(const hmap *m, const char *key) {
    if (!m->slots) return -1;
    uint64_t h = fnv1a_str(key);
    int p = (int)(h & (uint64_t)(m->cap - 1));
    while (m->slots[p].key) {
        if (strcmp(m->slots[p].key, key) == 0) return m->slots[p].value;
        p = (p + 1) & (m->cap - 1);
    }
    return -1;
}

static void hmap_free(hmap *m) {
    free(m->slots);
    memset(m, 0, sizeof(*m));
}

/* =====
 * safetensors database
 */

#define MAX_DIMS 8

typedef struct {
    char *dtype;
    int n_dims;
    int64_t shape[MAX_DIMS];
    uint64_t begin;
    uint64_t end;
} st_info;

typedef struct {
    char *name;
    char *file;
} weight_map_entry;

typedef struct {
    char *name;
    st_info info;
} tensor_entry;

typedef struct {
    char *file;
    char *path;
    uint64_t data_base;
    tensor_entry *tensors;
    int n_tensors;
    int cap_tensors;
    hmap tensor_map;
    int fd;
    bool owns_fd;
    pthread_mutex_t lock;
    bool loaded;
} shard;

typedef struct {
    const char *file;
    int fd;
} st_borrowed_file;

typedef struct {
    char *hf_dir;
    weight_map_entry *weights;
    int n_weights;
    hmap weight_map;
    shard *shards;
    int n_shards;
    int cap_shards;
    int index_fd;
    const st_borrowed_file *borrowed_files;
    int n_borrowed_files;
    pthread_mutex_t lock;
} st_db;

typedef struct {
    char *dtype;
    int n_dims;
    int64_t shape[MAX_DIMS];
    uint8_t *data;
    size_t nbytes;
} st_value;

static void st_value_free(st_value *v) {
    free(v->dtype);
    free(v->data);
    memset(v, 0, sizeof(*v));
}

static void parse_shape(const json_doc *d, int arr_tok, st_info *info, const char *name) {
    if (d->v[arr_tok].type != JT_ARRAY) {
        fprintf(stderr, "error: bad shape for %s\n", name);
        exit(1);
    }
    int nd = 0;
    for (int i = arr_tok + 1; i < d->len && d->v[i].parent == arr_tok; i = json_skip(d, i)) {
        if (nd >= MAX_DIMS) die("too many safetensors dimensions");
        info->shape[nd++] = json_i64(d, i);
    }
    info->n_dims = nd;
}

static int db_find_shard(st_db *db, const char *file) {
    for (int i = 0; i < db->n_shards; i++) {
        if (strcmp(db->shards[i].file, file) == 0) return i;
    }
    if (db->n_shards == db->cap_shards) {
        db->cap_shards = db->cap_shards ? db->cap_shards * 2 : 32;
        db->shards = xrealloc(db->shards, (size_t)db->cap_shards * sizeof(db->shards[0]));
    }
    shard *s = &db->shards[db->n_shards];
    memset(s, 0, sizeof(*s));
    s->file = xstrdup(file);
    s->path = path_join(db->hf_dir, file);
    s->fd = -1;
    for (int i = 0; i < db->n_borrowed_files; i++) {
        if (strcmp(file, db->borrowed_files[i].file) == 0) {
            s->fd = db->borrowed_files[i].fd;
            break;
        }
    }
    pthread_mutex_init(&s->lock, NULL);
    return db->n_shards++;
}

static void shard_add_tensor(shard *s, char *name, st_info info) {
    if (s->n_tensors == s->cap_tensors) {
        s->cap_tensors = s->cap_tensors ? s->cap_tensors * 2 : 256;
        s->tensors = xrealloc(s->tensors, (size_t)s->cap_tensors * sizeof(s->tensors[0]));
    }
    s->tensors[s->n_tensors++] = (tensor_entry){ .name = name, .info = info };
}

static void shard_load(shard *s) {
    if (s->loaded) return;
    if (s->fd < 0) {
        s->fd = open(s->path, O_RDONLY | O_CLOEXEC);
        if (s->fd < 0) die_errno("open", s->path);
        s->owns_fd = true;
    }
    uint64_t header_len = pread_u64_le_fd(
        s->fd, 0, "safetensors header length");
    char *header = xmalloc((size_t)header_len + 1);
    pread_exact_fd(s->fd, header, (size_t)header_len, 8,
                   "safetensors header");
    header[header_len] = '\0';
    s->data_base = 8 + header_len;

    json_doc d = json_parse_text(header, (size_t)header_len);
    if (d.len < 1 || d.v[0].type != JT_OBJECT) die("bad safetensors header");
    for (int i = 1; i < d.len && d.v[i].parent == 0;) {
        int k = i;
        int v = i + 1;
        if (v >= d.len || d.v[v].parent != 0) die("bad safetensors header object");
        if (!json_tok_eq(&d, k, "__metadata__")) {
            char *name = json_strdup_tok(&d, k);
            st_info info = {0};
            int dtype = json_obj_get(&d, v, "dtype");
            int shape = json_obj_get(&d, v, "shape");
            int offsets = json_obj_get(&d, v, "data_offsets");
            if (dtype < 0 || shape < 0 || offsets < 0) die("bad safetensors tensor entry");
            info.dtype = json_strdup_tok(&d, dtype);
            parse_shape(&d, shape, &info, name);
            int n_off = 0;
            for (int j = offsets + 1; j < d.len && d.v[j].parent == offsets; j = json_skip(&d, j)) {
                int64_t x = json_i64(&d, j);
                if (n_off == 0) info.begin = (uint64_t)x;
                else if (n_off == 1) info.end = (uint64_t)x;
                n_off++;
            }
            if (n_off != 2) die("bad safetensors data_offsets");
            shard_add_tensor(s, name, info);
        }
        i = json_skip(&d, v);
    }
    char **keys = xmalloc((size_t)s->n_tensors * sizeof(keys[0]));
    for (int i = 0; i < s->n_tensors; i++) keys[i] = s->tensors[i].name;
    hmap_build(&s->tensor_map, keys, s->n_tensors);
    free(keys);
    json_free(&d);
    free(header);
    s->loaded = true;
}

static char *read_fd_all(int fd, size_t *len_out, const char *what) {
    struct stat st;
    if (fstat(fd, &st) != 0) die_errno("fstat", what);
    if (st.st_size < 0 || (uint64_t)st.st_size > SIZE_MAX - 1)
        die("input is too large to parse");
    size_t size = (size_t)st.st_size;
    char *text = xmalloc(size + 1);
    if (size) pread_exact_fd(fd, text, size, 0, what);
    text[size] = '\0';
    if (len_out) *len_out = size;
    return text;
}

static void db_open_with_files(st_db *db, const char *hf_dir, int index_fd,
                               const st_borrowed_file *borrowed_files,
                               int n_borrowed_files) {
    memset(db, 0, sizeof(*db));
    pthread_mutex_init(&db->lock, NULL);
    db->hf_dir = xstrdup(hf_dir);
    db->index_fd = index_fd;
    db->borrowed_files = borrowed_files;
    db->n_borrowed_files = n_borrowed_files;
    char *index_path = path_join(hf_dir, "model.safetensors.index.json");
    size_t len = 0;
    char *text = index_fd >= 0
        ? read_fd_all(index_fd, &len, "safetensors index")
        : read_file(index_path, &len);
    json_doc d = json_parse_text(text, len);
    int weight_map = json_obj_get(&d, 0, "weight_map");
    if (weight_map < 0 || d.v[weight_map].type != JT_OBJECT) die("safetensors index has no weight_map");

    int cap = 4096;
    db->weights = xmalloc((size_t)cap * sizeof(db->weights[0]));
    for (int i = weight_map + 1; i < d.len && d.v[i].parent == weight_map;) {
        int k = i;
        int v = i + 1;
        if (db->n_weights == cap) {
            cap *= 2;
            db->weights = xrealloc(db->weights, (size_t)cap * sizeof(db->weights[0]));
        }
        db->weights[db->n_weights].name = json_strdup_tok(&d, k);
        db->weights[db->n_weights].file = json_strdup_tok(&d, v);
        db->n_weights++;
        i = json_skip(&d, v);
    }
    char **keys = xmalloc((size_t)db->n_weights * sizeof(keys[0]));
    for (int i = 0; i < db->n_weights; i++) {
        keys[i] = db->weights[i].name;
        db_find_shard(db, db->weights[i].file);
    }
    hmap_build(&db->weight_map, keys, db->n_weights);
    free(keys);
    json_free(&d);
    free(text);
    free(index_path);
}

static void db_open(st_db *db, const char *hf_dir) {
    db_open_with_files(db, hf_dir, -1, NULL, 0);
}

static void db_close(st_db *db) {
    for (int i = 0; i < db->n_weights; i++) {
        free(db->weights[i].name);
        free(db->weights[i].file);
    }
    for (int i = 0; i < db->n_shards; i++) {
        shard *s = &db->shards[i];
        if (s->owns_fd && s->fd >= 0) close(s->fd);
        for (int j = 0; j < s->n_tensors; j++) {
            free(s->tensors[j].name);
            free(s->tensors[j].info.dtype);
        }
        free(s->tensors);
        hmap_free(&s->tensor_map);
        pthread_mutex_destroy(&s->lock);
        free(s->file);
        free(s->path);
    }
    hmap_free(&db->weight_map);
    pthread_mutex_destroy(&db->lock);
    free(db->weights);
    free(db->shards);
    free(db->hf_dir);
    memset(db, 0, sizeof(*db));
}

static bool db_has(const st_db *db, const char *name) {
    return hmap_get(&db->weight_map, name) >= 0;
}

static const char *const dspark_shards[DS4_DSPARK_STAGE_COUNT] = {
    "model-00046-of-00048.safetensors",
    "model-00047-of-00048.safetensors",
    "model-00048-of-00048.safetensors",
};

static const char *const dspark_common_source_suffixes[] = {
    "attn.attn_sink",
    "attn.kv_norm.weight",
    "attn.q_norm.weight",
    "attn.wkv.scale",
    "attn.wkv.weight",
    "attn.wo_a.scale",
    "attn.wo_a.weight",
    "attn.wo_b.scale",
    "attn.wo_b.weight",
    "attn.wq_a.scale",
    "attn.wq_a.weight",
    "attn.wq_b.scale",
    "attn.wq_b.weight",
    "attn_norm.weight",
    "ffn.gate.bias",
    "ffn.gate.weight",
    "ffn.shared_experts.w1.scale",
    "ffn.shared_experts.w1.weight",
    "ffn.shared_experts.w2.scale",
    "ffn.shared_experts.w2.weight",
    "ffn.shared_experts.w3.scale",
    "ffn.shared_experts.w3.weight",
    "ffn_norm.weight",
    "hc_attn_base",
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_ffn_base",
    "hc_ffn_fn",
    "hc_ffn_scale",
};

static const char *const dspark_stage0_source_suffixes[] = {
    "main_norm.weight", "main_proj.scale", "main_proj.weight",
};

static const char *const dspark_stage2_source_suffixes[] = {
    "confidence_head.proj.weight",
    "hc_head_base",
    "hc_head_fn",
    "hc_head_scale",
    "markov_head.markov_w1.weight",
    "markov_head.markov_w2.weight",
    "norm.weight",
};

static bool dspark_is_shard(const char *file) {
    for (int stage = 0; stage < DS4_DSPARK_STAGE_COUNT; stage++)
        if (strcmp(file, dspark_shards[stage]) == 0) return true;
    return false;
}

static void dspark_require_mapping(const st_db *db, const char *name,
                                   int stage) {
    int wi = hmap_get(&db->weight_map, name);
    if (wi < 0) {
        fprintf(stderr, "error: final 0731 index is missing DSpark tensor %s\n",
                name);
        exit(1);
    }
    if (strcmp(db->weights[wi].file, dspark_shards[stage]) != 0) {
        fprintf(stderr,
                "error: final 0731 index maps %s to %s, expected %s\n",
                name, db->weights[wi].file, dspark_shards[stage]);
        exit(1);
    }
}

static void validate_dspark_source_index(const st_db *db) {
    if (db->n_weights != DS4_DSPARK_INDEX_TENSORS) {
        fprintf(stderr,
                "error: final 0731 index tensor count mismatch: got %d expected %d\n",
                db->n_weights, DS4_DSPARK_INDEX_TENSORS);
        exit(1);
    }
    int mtp_count = 0;
    int stage_count[DS4_DSPARK_STAGE_COUNT] = {0};
    for (int i = 0; i < db->n_weights; i++) {
        const char *name = db->weights[i].name;
        const char *file = db->weights[i].file;
        if (str_starts(name, "mtp.")) {
            int stage = -1;
            int consumed = 0;
            if (sscanf(name, "mtp.%d.%n", &stage, &consumed) != 1 ||
                consumed <= 0 || stage < 0 || stage >= DS4_DSPARK_STAGE_COUNT ||
                strcmp(file, dspark_shards[stage]) != 0) {
                fprintf(stderr,
                        "error: invalid final 0731 DSpark index entry: %s -> %s\n",
                        name, file);
                exit(1);
            }
            mtp_count++;
            stage_count[stage]++;
        } else if (dspark_is_shard(file)) {
            fprintf(stderr,
                    "error: final DSpark shard contains non-mtp index entry: %s\n",
                    name);
            exit(1);
        }
    }
    static const int expected_stage_count[DS4_DSPARK_STAGE_COUNT] = {
        1568, 1565, 1572,
    };
    if (mtp_count != DS4_DSPARK_SOURCE_TENSORS) {
        fprintf(stderr,
                "error: final 0731 mtp tensor count mismatch: got %d expected %d\n",
                mtp_count, DS4_DSPARK_SOURCE_TENSORS);
        exit(1);
    }
    for (int stage = 0; stage < DS4_DSPARK_STAGE_COUNT; stage++) {
        if (stage_count[stage] != expected_stage_count[stage]) {
            fprintf(stderr,
                    "error: final 0731 mtp.%d source count mismatch: got %d expected %d\n",
                    stage, stage_count[stage], expected_stage_count[stage]);
            exit(1);
        }
        char name[512];
        for (size_t i = 0;
             i < sizeof(dspark_common_source_suffixes) /
                     sizeof(dspark_common_source_suffixes[0]); i++) {
            snprintf(name, sizeof(name), "mtp.%d.%s", stage,
                     dspark_common_source_suffixes[i]);
            dspark_require_mapping(db, name, stage);
        }
        for (int expert = 0; expert < DS4_DSPARK_EXPERT_COUNT; expert++) {
            for (int part = 1; part <= 3; part++) {
                snprintf(name, sizeof(name),
                         "mtp.%d.ffn.experts.%d.w%d.weight",
                         stage, expert, part);
                dspark_require_mapping(db, name, stage);
                snprintf(name, sizeof(name),
                         "mtp.%d.ffn.experts.%d.w%d.scale",
                         stage, expert, part);
                dspark_require_mapping(db, name, stage);
            }
        }
    }
    char name[512];
    for (size_t i = 0;
         i < sizeof(dspark_stage0_source_suffixes) /
                 sizeof(dspark_stage0_source_suffixes[0]); i++) {
        snprintf(name, sizeof(name), "mtp.0.%s",
                 dspark_stage0_source_suffixes[i]);
        dspark_require_mapping(db, name, 0);
    }
    for (size_t i = 0;
         i < sizeof(dspark_stage2_source_suffixes) /
                 sizeof(dspark_stage2_source_suffixes[0]); i++) {
        snprintf(name, sizeof(name), "mtp.2.%s",
                 dspark_stage2_source_suffixes[i]);
        dspark_require_mapping(db, name, 2);
    }
}

static tensor_entry *db_tensor(st_db *db, const char *name, shard **shard_out) {
    pthread_mutex_lock(&db->lock);
    int wi = hmap_get(&db->weight_map, name);
    if (wi < 0) {
        fprintf(stderr, "error: HF tensor not found: %s\n", name);
        exit(1);
    }
    const char *file = db->weights[wi].file;
    int si = db_find_shard(db, file);
    shard *s = &db->shards[si];
    shard_load(s);
    int ti = hmap_get(&s->tensor_map, name);
    if (ti < 0) {
        fprintf(stderr, "error: HF tensor %s missing from shard %s\n", name, file);
        exit(1);
    }
    if (shard_out) *shard_out = s;
    tensor_entry *te = &s->tensors[ti];
    pthread_mutex_unlock(&db->lock);
    return te;
}

static st_value db_read(st_db *db, const char *name) {
    shard *s = NULL;
    tensor_entry *te = db_tensor(db, name, &s);
    const size_t nbytes = (size_t)(te->info.end - te->info.begin);
    st_value v = {0};
    v.dtype = xstrdup(te->info.dtype);
    v.n_dims = te->info.n_dims;
    memcpy(v.shape, te->info.shape, sizeof(v.shape));
    v.nbytes = nbytes;
    v.data = xmalloc(nbytes);
    pthread_mutex_lock(&s->lock);
    if (nbytes) {
        pread_exact_fd(s->fd, v.data, nbytes,
                       (off_t)(s->data_base + te->info.begin), s->path);
    }
    pthread_mutex_unlock(&s->lock);
    return v;
}

/* =====
 * DeepSeek V4 data conversion
 */

static float e8m0_to_f32(uint8_t e) {
    const uint32_t bits = e == 0 ? 0x00400000u : ((uint32_t)e << 23);
    float result;
    memcpy(&result, &bits, sizeof(result));
    return result;
}

static float e4m3fn_to_f32(uint8_t x) {
    const uint8_t abs = x & 0x7f;
    const bool sign = (x & 0x80) != 0;
    if (abs == 0) return sign ? -0.0f : 0.0f;
    if (abs == 0x7f) return 0.0f;
    const int exp = (x >> 3) & 0x0f;
    const int man = x & 0x07;
    float value = exp == 0 ? ldexpf((float)man, -9)
                           : ldexpf(1.0f + (float)man / 8.0f, exp - 7);
    return sign ? -value : value;
}

static float bf16_to_f32_bits(uint16_t bits) {
    return ds4q_bf16_to_f32(bits);
}

static int64_t value_nelements(const st_value *v) {
    int64_t n = 1;
    for (int i = 0; i < v->n_dims; i++) n *= v->shape[i];
    return n;
}

static float *tensor_to_f32(const st_value *t, int64_t *n_out) {
    const int64_t n = value_nelements(t);
    float *out = xmalloc((size_t)n * sizeof(float));
    if (strcmp(t->dtype, "F32") == 0) {
        if (t->nbytes != (size_t)n * sizeof(float)) die("bad F32 byte size");
        memcpy(out, t->data, t->nbytes);
    } else if (strcmp(t->dtype, "BF16") == 0) {
        if (t->nbytes != (size_t)n * sizeof(uint16_t)) die("bad BF16 byte size");
        for (int64_t i = 0; i < n; i++) out[i] = bf16_to_f32_bits(load_u16_le(t->data + (size_t)i * 2));
    } else if (strcmp(t->dtype, "F16") == 0) {
        if (t->nbytes != (size_t)n * sizeof(uint16_t)) die("bad F16 byte size");
        for (int64_t i = 0; i < n; i++) out[i] = ds4q_f16_to_f32(load_u16_le(t->data + (size_t)i * 2));
    } else if (strcmp(t->dtype, "F8_E4M3") == 0) {
        if (t->nbytes != (size_t)n) die("bad F8_E4M3 byte size");
        for (int64_t i = 0; i < n; i++) out[i] = e4m3fn_to_f32(t->data[i]);
    } else {
        fprintf(stderr, "error: cannot convert HF dtype directly: %s\n", t->dtype);
        exit(1);
    }
    if (n_out) *n_out = n;
    return out;
}

static float *dequant_fp8_weight(const st_value *w, const st_value *scale, int64_t *n_out) {
    if (strcmp(w->dtype, "F8_E4M3") != 0 || strcmp(scale->dtype, "F8_E8M0") != 0) die("bad FP8 weight/scale dtype");
    if (w->n_dims != 2 || scale->n_dims != 2) die("FP8 tensor must be 2D");
    const int64_t out_dim = w->shape[0];
    const int64_t in_dim = w->shape[1];
    const int64_t block_out = 128;
    const int64_t block_in = 128;
    if (out_dim % block_out || in_dim % block_in) die("FP8 dims are not divisible by 128");
    const int64_t scale_rows = out_dim / block_out;
    const int64_t scale_cols = in_dim / block_in;
    if (scale->shape[0] != scale_rows || scale->shape[1] != scale_cols) die("FP8 scale shape mismatch");
    float *out = xmalloc((size_t)out_dim * (size_t)in_dim * sizeof(float));
    for (int64_t ob = 0; ob < scale_rows; ob++) {
        for (int64_t ib = 0; ib < scale_cols; ib++) {
            const float s = e8m0_to_f32(scale->data[(size_t)ob * (size_t)scale_cols + (size_t)ib]);
            for (int64_t r = 0; r < block_out; r++) {
                const int64_t row = ob * block_out + r;
                const size_t base = (size_t)row * (size_t)in_dim + (size_t)ib * (size_t)block_in;
                for (int64_t c = 0; c < block_in; c++) {
                    out[base + (size_t)c] = e4m3fn_to_f32(w->data[base + (size_t)c]) * s;
                }
            }
        }
    }
    if (n_out) *n_out = out_dim * in_dim;
    return out;
}

static float *dequant_fp4_weight(const st_value *w, const st_value *scale, int64_t *n_out) {
    static const float fp4_table[16] = {
        0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
        0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    if (strcmp(w->dtype, "I8") != 0 || strcmp(scale->dtype, "F8_E8M0") != 0) die("bad FP4 weight/scale dtype");
    if (w->n_dims != 2 || scale->n_dims != 2) die("FP4 tensor must be 2D");
    const int64_t out_dim = w->shape[0];
    const int64_t packed_in = w->shape[1];
    const int64_t in_dim = packed_in * 2;
    if (in_dim % 32) die("FP4 in_dim is not divisible by 32");
    const int64_t n_blocks = in_dim / 32;
    if (scale->shape[0] != out_dim || scale->shape[1] != n_blocks) die("FP4 scale shape mismatch");
    float *out = xmalloc((size_t)out_dim * (size_t)in_dim * sizeof(float));
    for (int64_t r = 0; r < out_dim; r++) {
        for (int64_t b = 0; b < n_blocks; b++) {
            const float s = e8m0_to_f32(scale->data[(size_t)r * (size_t)n_blocks + (size_t)b]);
            const size_t wbase = ((size_t)r * (size_t)n_blocks + (size_t)b) * 16;
            const size_t obase = (size_t)r * (size_t)in_dim + (size_t)b * 32;
            for (int64_t j = 0; j < 16; j++) {
                const uint8_t q = w->data[wbase + (size_t)j];
                out[obase + (size_t)(2*j + 0)] = fp4_table[q & 0x0f] * s;
                out[obase + (size_t)(2*j + 1)] = fp4_table[(q >> 4) & 0x0f] * s;
            }
        }
    }
    if (n_out) *n_out = out_dim * in_dim;
    return out;
}

/* =====
 * Imatrix
 */

typedef struct {
    char *name;
    float *values;
    int n_values;
} imatrix_entry;

typedef struct {
    char *file;
    char *dataset;
    imatrix_entry *entries;
    int n_entries;
    hmap map;
    int chunks;
    bool strict;
} imatrix_store;

static void imatrix_load(imatrix_store *im, const char *path, bool strict) {
    memset(im, 0, sizeof(*im));
    im->file = xstrdup(path);
    im->strict = strict;
    im->chunks = -1;
    FILE *fp = fopen(path, "rb");
    if (!fp) die_errno("open imatrix", path);
    int32_t n_entries = read_i32_fp(fp, "imatrix entry count");
    if (n_entries < 1) die("imatrix has no entries");
    im->entries = xcalloc((size_t)n_entries, sizeof(im->entries[0]));
    im->n_entries = n_entries;
    for (int i = 0; i < n_entries; i++) {
        int32_t len = read_i32_fp(fp, "imatrix name length");
        if (len <= 0 || len > 4096) die("bad imatrix name length");
        char *name = xmalloc((size_t)len + 1);
        if (fread(name, 1, (size_t)len, fp) != (size_t)len) die("short imatrix name read");
        name[len] = '\0';
        int32_t ncall = read_i32_fp(fp, "imatrix calls");
        int32_t nval = read_i32_fp(fp, "imatrix values");
        if (nval < 1) die("bad imatrix value count");
        float *values = xmalloc((size_t)nval * sizeof(float));
        if (fread(values, sizeof(float), (size_t)nval, fp) != (size_t)nval) die("short imatrix value read");
        if (ncall > 0) {
            for (int j = 0; j < nval; j++) values[j] /= (float)ncall;
        }
        for (int j = 0; j < nval; j++) {
            if (!isfinite(values[j])) die("non-finite imatrix value");
        }
        im->entries[i] = (imatrix_entry){ .name = name, .values = values, .n_values = nval };
    }
    if (fgetc(fp) != EOF) {
        if (fseeko(fp, -1, SEEK_CUR) == 0) {
            im->chunks = read_i32_fp(fp, "imatrix chunks");
            int32_t dlen = read_i32_fp(fp, "imatrix dataset length");
            if (dlen > 0 && dlen < (1 << 20)) {
                im->dataset = xmalloc((size_t)dlen + 1);
                if (fread(im->dataset, 1, (size_t)dlen, fp) == (size_t)dlen) {
                    im->dataset[dlen] = '\0';
                } else {
                    free(im->dataset);
                    im->dataset = NULL;
                }
            }
        }
    }
    fclose(fp);
    char **keys = xmalloc((size_t)n_entries * sizeof(keys[0]));
    for (int i = 0; i < n_entries; i++) keys[i] = im->entries[i].name;
    hmap_build(&im->map, keys, n_entries);
    free(keys);
    fprintf(stderr, "loaded imatrix %s: %d entries%s%s\n",
            path, n_entries, im->dataset ? ", dataset=" : "", im->dataset ? im->dataset : "");
}

static bool imatrix_enabled(const imatrix_store *im) {
    return im && im->n_entries > 0;
}

static const float *imatrix_find(
        const imatrix_store *im,
        const char **names,
        int n_names,
        int64_t ncols,
        int expert_id,
        int n_experts) {
    if (!imatrix_enabled(im)) return NULL;
    char tmp[4096];
    for (int pass = 0; pass < 3; pass++) {
        for (int i = 0; i < n_names; i++) {
            if (!names[i]) continue;
            const char *candidate = names[i];
            if (expert_id >= 0 && pass < 2) {
                snprintf(tmp, sizeof(tmp), "%s.expert%s%d", names[i], pass == 0 ? "." : "_", expert_id);
                candidate = tmp;
            } else if (pass < 2) {
                continue;
            }
            int idx = hmap_get(&im->map, candidate);
            if (idx < 0) continue;
            const imatrix_entry *e = &im->entries[idx];
            if ((int64_t)e->n_values == ncols) return e->values;
            if (expert_id >= 0 && n_experts > 0 && (int64_t)e->n_values == ncols * (int64_t)n_experts) {
                return e->values + (size_t)expert_id * (size_t)ncols;
            }
            fprintf(stderr, "error: imatrix size mismatch for %s: got %d expected %" PRId64 "\n",
                    candidate, e->n_values, ncols);
            exit(1);
        }
    }
    if (im->strict) {
        fprintf(stderr, "error: missing imatrix entry for %s\n", names[0] ? names[0] : "(unnamed)");
        exit(1);
    }
    return NULL;
}

static void imatrix_free(imatrix_store *im) {
    for (int i = 0; i < im->n_entries; i++) {
        free(im->entries[i].name);
        free(im->entries[i].values);
    }
    free(im->entries);
    free(im->file);
    free(im->dataset);
    hmap_free(&im->map);
    memset(im, 0, sizeof(*im));
}

/* =====
 * GGUF tensor mapping and quantization policy
 */

typedef enum { EXP_NONE, EXP_W1, EXP_W2, EXP_W3 } expert_part;

typedef struct {
    bool is_expert;
    bool is_dspark;
    int layer;
    expert_part part;
} expert_tensor;

static expert_tensor parse_expert_tensor(const char *name) {
    expert_tensor e = {0};
    int layer = -1;
    char kind[16];
    int rest = 0;
    bool dspark = false;
    int matched = sscanf(name, "blk.%d.ffn_%15[^_]_exps.weight%n",
                         &layer, kind, &rest);
    if (matched != 2 || rest != (int)strlen(name)) {
        rest = 0;
        matched = sscanf(name, "mtp.%d.ffn_%15[^_]_exps.weight%n",
                         &layer, kind, &rest);
        dspark = true;
    }
    if (matched == 2 && rest == (int)strlen(name))
    {
        if (strcmp(kind, "gate") == 0 || strcmp(kind, "down") == 0 || strcmp(kind, "up") == 0) {
            e.is_expert = true;
            e.is_dspark = dspark;
            e.layer = layer;
            e.part = strcmp(kind, "gate") == 0 ? EXP_W1 : strcmp(kind, "down") == 0 ? EXP_W2 : EXP_W3;
        }
    }
    return e;
}

static const char *expert_part_name(expert_part p) {
    switch (p) {
        case EXP_W1: return "w1";
        case EXP_W2: return "w2";
        case EXP_W3: return "w3";
        default: die("bad expert part");
    }
    return "";
}

typedef struct {
    const char *gguf;
    const char *hf;
} name_map;

static const name_map top_map[] = {
    { "token_embd.weight",      "embed.weight" },
    { "output_norm.weight",     "norm.weight" },
    { "output.weight",          "head.weight" },
    { "output_hc_base.weight",  "hc_head_base" },
    { "output_hc_fn.weight",    "hc_head_fn" },
    { "output_hc_scale.weight", "hc_head_scale" },
};

static const name_map layer_map[] = {
    { "hc_attn_base.weight",              "hc_attn_base" },
    { "hc_attn_fn.weight",                "hc_attn_fn" },
    { "hc_attn_scale.weight",             "hc_attn_scale" },
    { "hc_ffn_base.weight",               "hc_ffn_base" },
    { "hc_ffn_fn.weight",                 "hc_ffn_fn" },
    { "hc_ffn_scale.weight",              "hc_ffn_scale" },
    { "attn_sinks.weight",                "attn.attn_sink" },
    { "attn_q_a.weight",                  "attn.wq_a.weight" },
    { "attn_q_b.weight",                  "attn.wq_b.weight" },
    { "attn_q_a_norm.weight",             "attn.q_norm.weight" },
    { "attn_kv.weight",                   "attn.wkv.weight" },
    { "attn_kv_a_norm.weight",            "attn.kv_norm.weight" },
    { "attn_output_a.weight",             "attn.wo_a.weight" },
    { "attn_output_b.weight",             "attn.wo_b.weight" },
    { "attn_compressor_ape.weight",       "attn.compressor.ape" },
    { "attn_compressor_kv.weight",        "attn.compressor.wkv.weight" },
    { "attn_compressor_gate.weight",      "attn.compressor.wgate.weight" },
    { "attn_compressor_norm.weight",      "attn.compressor.norm.weight" },
    { "indexer.attn_q_b.weight",          "attn.indexer.wq_b.weight" },
    { "indexer.proj.weight",              "attn.indexer.weights_proj.weight" },
    { "indexer_compressor_ape.weight",    "attn.indexer.compressor.ape" },
    { "indexer_compressor_kv.weight",     "attn.indexer.compressor.wkv.weight" },
    { "indexer_compressor_gate.weight",   "attn.indexer.compressor.wgate.weight" },
    { "indexer_compressor_norm.weight",   "attn.indexer.compressor.norm.weight" },
    { "attn_norm.weight",                 "attn_norm.weight" },
    { "ffn_norm.weight",                  "ffn_norm.weight" },
    { "ffn_gate_shexp.weight",            "ffn.shared_experts.w1.weight" },
    { "ffn_up_shexp.weight",              "ffn.shared_experts.w3.weight" },
    { "ffn_down_shexp.weight",            "ffn.shared_experts.w2.weight" },
    { "ffn_gate_inp.weight",              "ffn.gate.weight" },
    { "exp_probs_b.bias",                 "ffn.gate.bias" },
    { "ffn_gate_tid2eid.weight",          "ffn.gate.tid2eid" },
};

static const name_map dspark_special_map[] = {
    { "main_proj.weight",                    "main_proj.weight" },
    { "main_norm.weight",                    "main_norm.weight" },
    { "norm.weight",                         "norm.weight" },
    { "hc_head_base.weight",                 "hc_head_base" },
    { "hc_head_fn.weight",                   "hc_head_fn" },
    { "hc_head_scale.weight",                "hc_head_scale" },
    { "markov_head.markov_w1.weight",        "markov_head.markov_w1.weight" },
    { "markov_head.markov_w2.weight",        "markov_head.markov_w2.weight" },
    { "confidence_head.proj.weight",          "confidence_head.proj.weight" },
};

static char *hf_name_for_regular(const char *gguf_name) {
    for (size_t i = 0; i < sizeof(top_map) / sizeof(top_map[0]); i++) {
        if (strcmp(gguf_name, top_map[i].gguf) == 0) return xstrdup(top_map[i].hf);
    }
    int layer = -1;
    bool dspark = false;
    const char *p = gguf_name;
    if (sscanf(p, "blk.%d.", &layer) != 1) {
        if (sscanf(p, "mtp.%d.", &layer) != 1) {
            fprintf(stderr, "error: cannot map GGUF tensor to HF tensor: %s\n", gguf_name);
            exit(1);
        }
        dspark = true;
    }
    const char *rest = strchr(p + 4, '.');
    if (!rest) die("bad layer tensor name");
    rest++;
    for (size_t i = 0; i < sizeof(layer_map) / sizeof(layer_map[0]); i++) {
        if (strcmp(rest, layer_map[i].gguf) == 0) {
            char buf[512];
            snprintf(buf, sizeof(buf), dspark ? "mtp.%d.%s" : "layers.%d.%s",
                     layer, layer_map[i].hf);
            return xstrdup(buf);
        }
    }
    if (dspark) {
        for (size_t i = 0;
             i < sizeof(dspark_special_map) / sizeof(dspark_special_map[0]);
             i++) {
            if (strcmp(rest, dspark_special_map[i].gguf) == 0) {
                char buf[512];
                snprintf(buf, sizeof(buf), "mtp.%d.%s", layer,
                         dspark_special_map[i].hf);
                return xstrdup(buf);
            }
        }
    }
    fprintf(stderr, "error: cannot map GGUF tensor to HF tensor: %s\n", gguf_name);
    exit(1);
}

typedef struct {
    char *prefix;
    ds4q_type type;
} type_override;

typedef struct {
    ds4q_type routed_w1, routed_w2, routed_w3;
    ds4q_type attention_proj, attention, shared, embedding, output, dense;
    type_override *overrides;
    int n_overrides;
} quant_policy;

static bool is_attention_projection(const char *name) {
    return strstr(name, ".attn_kv.weight") || strstr(name, ".attn_q_a.weight") ||
           strstr(name, ".attn_q_b.weight") || strstr(name, ".attn_output_a.weight") ||
           strstr(name, ".attn_output_b.weight");
}

static bool is_attention_tensor(const char *name) {
    return strstr(name, ".attn") || strstr(name, "attn_") || strstr(name, ".indexer") || strstr(name, "indexer_");
}

static bool is_shared_expert(const char *name) {
    return strstr(name, "_shexp.") != NULL;
}

static bool is_output_tensor(const char *name) {
    return str_starts(name, "output.");
}

typedef struct {
    char *name;
    int n_dims;
    int64_t ne[DS4Q_MAX_DIMS];
    ds4q_type type;
    uint64_t old_offset;
    uint64_t new_offset;
    size_t size;
} tensor_meta;

static int tensor_n_dims(const tensor_meta *t) {
    int n = t->n_dims;
    while (n > 1 && t->ne[n - 1] == 1) n--;
    return n;
}

static ds4q_type policy_type(const quant_policy *p, const char *name, const tensor_meta *tmpl) {
    for (int i = 0; i < p->n_overrides; i++) {
        if (strcmp(name, p->overrides[i].prefix) == 0 || str_starts(name, p->overrides[i].prefix)) {
            return p->overrides[i].type;
        }
    }
    expert_tensor e = parse_expert_tensor(name);
    if (e.is_expert) {
        if (e.part == EXP_W1 && p->routed_w1 != DS4Q_TYPE_COUNT) return p->routed_w1;
        if (e.part == EXP_W2 && p->routed_w2 != DS4Q_TYPE_COUNT) return p->routed_w2;
        if (e.part == EXP_W3 && p->routed_w3 != DS4Q_TYPE_COUNT) return p->routed_w3;
        return tmpl->type;
    }
    if (tmpl->type != DS4Q_TYPE_F32 && tmpl->type != DS4Q_TYPE_F16 &&
        tmpl->type != DS4Q_TYPE_BF16 && !ds4q_can_quantize(tmpl->type)) {
        return tmpl->type;
    }
    if (tensor_n_dims(tmpl) <= 1) return tmpl->type;
    if (strcmp(name, "token_embd.weight") == 0 && p->embedding != DS4Q_TYPE_COUNT) return p->embedding;
    if (is_output_tensor(name) && p->output != DS4Q_TYPE_COUNT) return p->output;
    if (is_shared_expert(name) && p->shared != DS4Q_TYPE_COUNT) return p->shared;
    if (is_attention_projection(name) && p->attention_proj != DS4Q_TYPE_COUNT) return p->attention_proj;
    if (is_attention_tensor(name) && p->attention != DS4Q_TYPE_COUNT) return p->attention;
    if (p->dense != DS4Q_TYPE_COUNT) return p->dense;
    return tmpl->type;
}

static ds4q_type parse_type(const char *raw) {
    char wanted[64];
    size_t n = 0;
    for (const char *p = raw; *p && n + 1 < sizeof(wanted); p++) {
        if (*p != '-' && *p != '_') wanted[n++] = (char)tolower((unsigned char)*p);
    }
    wanted[n] = '\0';
    if (strcmp(wanted, "copy") == 0 || strcmp(wanted, "template") == 0) return DS4Q_TYPE_COUNT;
    for (int i = 0; i < DS4Q_TYPE_COUNT; i++) {
        char name[64];
        size_t m = 0;
        const char *tn = ds4q_type_name((ds4q_type)i);
        if (!tn) continue;
        for (const char *p = tn; *p && m + 1 < sizeof(name); p++) {
            if (*p != '-' && *p != '_') name[m++] = (char)tolower((unsigned char)*p);
        }
        name[m] = '\0';
        if (strcmp(name, wanted) == 0) return (ds4q_type)i;
    }
    fprintf(stderr, "error: unknown quant type: %s\n", raw);
    exit(1);
}

static bool is_quantizable_target(ds4q_type type) {
    return type == DS4Q_TYPE_F32 || type == DS4Q_TYPE_F16 || type == DS4Q_TYPE_BF16 || ds4q_can_quantize(type);
}

/* =====
 * Tensor generation
 */

typedef struct {
    uint8_t *data;
    size_t size;
} byte_buf;

static byte_buf f32_to_type(const float *src, int64_t n, ds4q_type type, int64_t ncols, const float *imat) {
    if (ncols <= 0 || n % ncols != 0) die("bad ncols for tensor conversion");
    byte_buf out = {0};
    if (type == DS4Q_TYPE_F32) {
        out.size = (size_t)n * sizeof(float);
        out.data = xmalloc(out.size);
        memcpy(out.data, src, out.size);
        return out;
    }
    if (type == DS4Q_TYPE_F16) {
        out.size = (size_t)n * sizeof(uint16_t);
        out.data = xmalloc(out.size);
        ds4q_f32_to_f16_row(src, (uint16_t *)out.data, n);
        return out;
    }
    if (type == DS4Q_TYPE_BF16) {
        out.size = (size_t)n * sizeof(uint16_t);
        out.data = xmalloc(out.size);
        ds4q_f32_to_bf16_row(src, (uint16_t *)out.data, n);
        return out;
    }
    if (!ds4q_can_quantize(type)) die("unsupported quant target type");
    if (ncols % ds4q_block_size(type) != 0) die("ncols is not divisible by quant block size");
    const int64_t nrows = n / ncols;
    out.size = (size_t)nrows * ds4q_row_size(type, ncols);
    out.data = xmalloc(out.size);

    float *synthetic = NULL;
    const float *im_ptr = imat;
    if (!im_ptr && ds4q_requires_imatrix(type)) {
        synthetic = xcalloc((size_t)ncols, sizeof(float));
        for (int64_t r = 0; r < nrows; r++) {
            const float *row = src + (size_t)r * (size_t)ncols;
            for (int64_t c = 0; c < ncols; c++) synthetic[c] += row[c] * row[c];
        }
        im_ptr = synthetic;
    }
    size_t written = ds4q_quantize_chunk(type, src, out.data, 0, nrows, ncols, im_ptr);
    free(synthetic);
    if (written != out.size) die("ds4q_quantize_chunk wrote unexpected byte count");
    return out;
}

static byte_buf i64_to_i32(const st_value *src) {
    if (strcmp(src->dtype, "I64") != 0) die("expected I64 source for I32 tensor");
    const int64_t n = value_nelements(src);
    if (src->nbytes != (size_t)n * sizeof(int64_t)) die("bad I64 byte size");
    byte_buf out = { .size = (size_t)n * sizeof(int32_t), .data = xmalloc((size_t)n * sizeof(int32_t)) };
    int32_t *dst = (int32_t *)out.data;
    for (int64_t i = 0; i < n; i++) {
        int64_t v = load_i64_le(src->data + (size_t)i * 8);
        if (v < INT32_MIN || v > INT32_MAX) die("I64 value out of I32 range");
        dst[i] = (int32_t)v;
    }
    return out;
}

static size_t tensor_nbytes(ds4q_type type, const int64_t *ne, int n_dims) {
    size_t nbytes = ds4q_row_size(type, ne[0]);
    for (int i = 1; i < n_dims; i++) nbytes *= (size_t)ne[i];
    return nbytes;
}

typedef struct {
    const char *suffix;
    int n_dims;
    int64_t ne[3];
    ds4q_type type;
} dspark_tensor_spec;

#ifdef DS4_DSPARK_QUANTIZER_TEST_TINY_GEOMETRY
/*
 * The synthetic lane exercises the complete writer and quantizers without a
 * multi-gigabyte fixture. Types, rank, tensor count, expert count, and source
 * inventory remain production-identical; only dimensions are reduced.
 */
static const dspark_tensor_spec dspark_block_specs[] = {
    { "hc_attn_base.weight",       1, {2},                   DS4Q_TYPE_F32 },
    { "hc_attn_fn.weight",         2, {8, 2},                DS4Q_TYPE_F16 },
    { "hc_attn_scale.weight",      1, {2},                   DS4Q_TYPE_F32 },
    { "attn_sinks.weight",         1, {2},                   DS4Q_TYPE_F32 },
    { "attn_q_a.weight",           2, {128, 128},            DS4Q_TYPE_Q8_0 },
    { "attn_q_a_norm.weight",      1, {128},                 DS4Q_TYPE_F32 },
    { "attn_q_b.weight",           2, {128, 128},            DS4Q_TYPE_Q8_0 },
    { "attn_kv.weight",            2, {128, 128},            DS4Q_TYPE_Q8_0 },
    { "attn_kv_a_norm.weight",     1, {128},                 DS4Q_TYPE_F32 },
    { "attn_output_a.weight",      2, {128, 128},            DS4Q_TYPE_Q8_0 },
    { "attn_output_b.weight",      2, {128, 128},            DS4Q_TYPE_Q8_0 },
    { "attn_norm.weight",          1, {128},                 DS4Q_TYPE_F32 },
    { "hc_ffn_base.weight",        1, {2},                   DS4Q_TYPE_F32 },
    { "hc_ffn_fn.weight",          2, {8, 2},                DS4Q_TYPE_F16 },
    { "hc_ffn_scale.weight",       1, {2},                   DS4Q_TYPE_F32 },
    { "ffn_gate_inp.weight",       2, {128, 256},            DS4Q_TYPE_Q8_0 },
    { "exp_probs_b.bias",          1, {256},                 DS4Q_TYPE_F32 },
    { "ffn_norm.weight",           1, {128},                 DS4Q_TYPE_F32 },
    { "ffn_gate_exps.weight",      3, {256, 1, 256},         DS4Q_TYPE_IQ2_XXS },
    { "ffn_up_exps.weight",        3, {256, 1, 256},         DS4Q_TYPE_IQ2_XXS },
    { "ffn_down_exps.weight",      3, {256, 1, 256},         DS4Q_TYPE_Q2_K },
    { "ffn_gate_shexp.weight",     2, {128, 128},            DS4Q_TYPE_Q8_0 },
    { "ffn_up_shexp.weight",       2, {128, 128},            DS4Q_TYPE_Q8_0 },
    { "ffn_down_shexp.weight",     2, {128, 128},            DS4Q_TYPE_Q8_0 },
};

static const dspark_tensor_spec dspark_stage0_specs[] = {
    { "main_proj.weight",          2, {128, 128},            DS4Q_TYPE_Q8_0 },
    { "main_norm.weight",          1, {128},                 DS4Q_TYPE_F32 },
};

static const dspark_tensor_spec dspark_stage2_specs[] = {
    { "norm.weight",                        1, {128},         DS4Q_TYPE_F32 },
    { "hc_head_base.weight",                1, {2},           DS4Q_TYPE_F32 },
    { "hc_head_fn.weight",                  2, {8, 2},        DS4Q_TYPE_F16 },
    { "hc_head_scale.weight",               1, {1},           DS4Q_TYPE_F32 },
    { "markov_head.markov_w1.weight",       2, {32, 4},       DS4Q_TYPE_Q8_0 },
    { "markov_head.markov_w2.weight",       2, {32, 4},       DS4Q_TYPE_Q8_0 },
    { "confidence_head.proj.weight",        2, {32, 1},       DS4Q_TYPE_Q8_0 },
};
#else
static const dspark_tensor_spec dspark_block_specs[] = {
    { "hc_attn_base.weight",       1, {24},                  DS4Q_TYPE_F32 },
    { "hc_attn_fn.weight",         2, {16384, 24},           DS4Q_TYPE_F16 },
    { "hc_attn_scale.weight",      1, {3},                   DS4Q_TYPE_F32 },
    { "attn_sinks.weight",         1, {64},                  DS4Q_TYPE_F32 },
    { "attn_q_a.weight",           2, {4096, 1024},          DS4Q_TYPE_Q8_0 },
    { "attn_q_a_norm.weight",      1, {1024},                DS4Q_TYPE_F32 },
    { "attn_q_b.weight",           2, {1024, 32768},         DS4Q_TYPE_Q8_0 },
    { "attn_kv.weight",            2, {4096, 512},           DS4Q_TYPE_Q8_0 },
    { "attn_kv_a_norm.weight",     1, {512},                 DS4Q_TYPE_F32 },
    { "attn_output_a.weight",      2, {4096, 8192},          DS4Q_TYPE_Q8_0 },
    { "attn_output_b.weight",      2, {8192, 4096},          DS4Q_TYPE_Q8_0 },
    { "attn_norm.weight",          1, {4096},                DS4Q_TYPE_F32 },
    { "hc_ffn_base.weight",        1, {24},                  DS4Q_TYPE_F32 },
    { "hc_ffn_fn.weight",          2, {16384, 24},           DS4Q_TYPE_F16 },
    { "hc_ffn_scale.weight",       1, {3},                   DS4Q_TYPE_F32 },
    { "ffn_gate_inp.weight",       2, {4096, 256},           DS4Q_TYPE_Q8_0 },
    { "exp_probs_b.bias",          1, {256},                 DS4Q_TYPE_F32 },
    { "ffn_norm.weight",           1, {4096},                DS4Q_TYPE_F32 },
    { "ffn_gate_exps.weight",      3, {4096, 2048, 256},     DS4Q_TYPE_IQ2_XXS },
    { "ffn_up_exps.weight",        3, {4096, 2048, 256},     DS4Q_TYPE_IQ2_XXS },
    { "ffn_down_exps.weight",      3, {2048, 4096, 256},     DS4Q_TYPE_Q2_K },
    { "ffn_gate_shexp.weight",     2, {4096, 2048},          DS4Q_TYPE_Q8_0 },
    { "ffn_up_shexp.weight",       2, {4096, 2048},          DS4Q_TYPE_Q8_0 },
    { "ffn_down_shexp.weight",     2, {2048, 4096},          DS4Q_TYPE_Q8_0 },
};

static const dspark_tensor_spec dspark_stage0_specs[] = {
    { "main_proj.weight",          2, {12288, 4096},         DS4Q_TYPE_Q8_0 },
    { "main_norm.weight",          1, {4096},                DS4Q_TYPE_F32 },
};

static const dspark_tensor_spec dspark_stage2_specs[] = {
    { "norm.weight",                        1, {4096},        DS4Q_TYPE_F32 },
    { "hc_head_base.weight",                1, {4},           DS4Q_TYPE_F32 },
    { "hc_head_fn.weight",                  2, {16384, 4},    DS4Q_TYPE_F16 },
    { "hc_head_scale.weight",               1, {1},           DS4Q_TYPE_F32 },
    { "markov_head.markov_w1.weight",       2, {256, 129280}, DS4Q_TYPE_Q8_0 },
    { "markov_head.markov_w2.weight",       2, {256, 129280}, DS4Q_TYPE_Q8_0 },
    { "confidence_head.proj.weight",        2, {4352, 1},     DS4Q_TYPE_Q8_0 },
};
#endif

static void check_reversed_shape(const char *gguf_name, const st_info *info, const tensor_meta *tmpl) {
    int nd = str_starts(gguf_name, "mtp.") ? tmpl->n_dims
                                            : tensor_n_dims(tmpl);
    if (info->n_dims != nd) {
        fprintf(stderr, "error: rank mismatch for %s\n", gguf_name);
        exit(1);
    }
    for (int i = 0; i < nd; i++) {
        if (tmpl->ne[i] != info->shape[nd - 1 - i]) {
            fprintf(stderr, "error: shape mismatch for %s\n", gguf_name);
            exit(1);
        }
    }
}

static byte_buf generate_regular(st_db *db, const char *gguf_name, const tensor_meta *tmpl,
                                 ds4q_type target, const imatrix_store *imatrix) {
    char *hf_name = hf_name_for_regular(gguf_name);
    tensor_entry *te = db_tensor(db, hf_name, NULL);
    check_reversed_shape(gguf_name, &te->info, tmpl);
    if (target == DS4Q_TYPE_I32) {
        st_value sv = db_read(db, hf_name);
        byte_buf b = i64_to_i32(&sv);
        st_value_free(&sv);
        free(hf_name);
        return b;
    }
    if (!is_quantizable_target(target)) die("unsupported regular target type");
    int64_t n = 0;
    float *f32 = NULL;
    if (strcmp(te->info.dtype, "F8_E4M3") == 0) {
        if (!str_ends(hf_name, ".weight")) die("FP8 tensor without .weight suffix");
        char *scale_name = xstrdup(hf_name);
        strcpy(scale_name + strlen(scale_name) - strlen(".weight"), ".scale");
        if (!db_has(db, scale_name)) die("missing FP8 scale tensor");
        st_value w = db_read(db, hf_name);
        st_value s = db_read(db, scale_name);
        f32 = dequant_fp8_weight(&w, &s, &n);
        st_value_free(&w);
        st_value_free(&s);
        free(scale_name);
    } else {
        st_value w = db_read(db, hf_name);
        f32 = tensor_to_f32(&w, &n);
        st_value_free(&w);
    }
    const char *names[2] = { gguf_name, hf_name };
    const float *imat = imatrix_find(imatrix, names, 2, tmpl->ne[0], -1, 0);
    byte_buf b = f32_to_type(f32, n, target, tmpl->ne[0], imat);
    free(f32);
    free(hf_name);
    return b;
}

typedef struct {
    st_db *db;
    const char *gguf_name;
    const tensor_meta *tmpl;
    ds4q_type target;
    int n_experts;
    const imatrix_store *imatrix;
    expert_tensor expert;
    const char *wid;
    int64_t ncols;
    int64_t nrows;
    size_t per_expert;
    byte_buf *out;
    int next;
    int done;
    pthread_mutex_t lock;
} expert_job;

static void generate_one_expert(expert_job *j, int xid) {
    char prefix[256];
    snprintf(prefix, sizeof(prefix),
             j->expert.is_dspark ? "mtp.%d.ffn.experts.%d.%s"
                                 : "layers.%d.ffn.experts.%d.%s",
             j->expert.layer, xid, j->wid);
    char weight_name[320];
    char scale_name[320];
    snprintf(weight_name, sizeof(weight_name), "%s.weight", prefix);
    snprintf(scale_name, sizeof(scale_name), "%s.scale", prefix);
    st_value w = db_read(j->db, weight_name);
    st_value s = db_read(j->db, scale_name);
    if (w.n_dims != 2 || w.shape[0] != j->nrows || w.shape[1] * 2 != j->ncols) die("expert shape mismatch");
    int64_t n = 0;
    float *f32 = dequant_fp4_weight(&w, &s, &n);
    const char *names[3] = { j->gguf_name, weight_name, NULL };
    const float *imat = imatrix_find(j->imatrix, names, 2, j->ncols, xid, j->n_experts);
    byte_buf q = f32_to_type(f32, n, j->target, j->ncols, imat);
    if (q.size != j->per_expert) die("expert quantized size mismatch");
    memcpy(j->out->data + (size_t)xid * j->per_expert, q.data, q.size);
    free(q.data);
    free(f32);
    st_value_free(&w);
    st_value_free(&s);
}

static void *expert_worker(void *arg) {
    expert_job *j = arg;
    for (;;) {
        pthread_mutex_lock(&j->lock);
        int xid = j->next++;
        pthread_mutex_unlock(&j->lock);
        if (xid >= j->n_experts) break;
        generate_one_expert(j, xid);
        pthread_mutex_lock(&j->lock);
        int done = ++j->done;
        if (done % 32 == 0 || done == j->n_experts) {
            fprintf(stderr, "generate_expert_tensor: layer %d %s %d/%d experts\n",
                    j->expert.layer, j->wid, done, j->n_experts);
        }
        pthread_mutex_unlock(&j->lock);
    }
    return NULL;
}

static byte_buf generate_expert(st_db *db, const char *gguf_name, const tensor_meta *tmpl,
                                ds4q_type target, int n_experts, int n_threads,
                                const imatrix_store *imatrix) {
    expert_tensor e = parse_expert_tensor(gguf_name);
    if (!e.is_expert) die("not an expert tensor");
    if (!is_quantizable_target(target)) die("unsupported expert target type");
    const char *wid = expert_part_name(e.part);
    const int64_t ncols = tmpl->ne[0];
    const int64_t nrows = tmpl->ne[1];
    const size_t per_expert = (size_t)nrows * ds4q_row_size(target, ncols);
    byte_buf out = { .size = per_expert * (size_t)n_experts, .data = xmalloc(per_expert * (size_t)n_experts) };
    ds4q_quantize_init(target);
    int worker_count = n_threads > 0 ? n_threads : 8;
    if (worker_count < 1) worker_count = 1;
    if (worker_count > n_experts) worker_count = n_experts;
    fprintf(stderr, "generate_expert_tensor: layer %d %s using %d worker%s\n",
            e.layer, wid, worker_count, worker_count == 1 ? "" : "s");
    expert_job job = {
        .db = db, .gguf_name = gguf_name, .tmpl = tmpl, .target = target,
        .n_experts = n_experts, .imatrix = imatrix, .expert = e, .wid = wid,
        .ncols = ncols, .nrows = nrows, .per_expert = per_expert, .out = &out,
    };
    pthread_mutex_init(&job.lock, NULL);
    pthread_t *threads = xcalloc((size_t)worker_count, sizeof(threads[0]));
    for (int i = 1; i < worker_count; i++) pthread_create(&threads[i], NULL, expert_worker, &job);
    expert_worker(&job);
    for (int i = 1; i < worker_count; i++) pthread_join(threads[i], NULL);
    pthread_mutex_destroy(&job.lock);
    free(threads);
    return out;
}

static byte_buf generate_tensor(st_db *db, const char *name, const tensor_meta *tmpl,
                                ds4q_type target, int n_experts, int n_threads,
                                const imatrix_store *imatrix) {
    if (parse_expert_tensor(name).is_expert) {
        return generate_expert(db, name, tmpl, target, n_experts, n_threads, imatrix);
    }
    return generate_regular(db, name, tmpl, target, imatrix);
}

/* =====
 * Minimal GGUF reader/writer
 *
 * GGUF metadata is copied as raw KV records from the template.  Tensor infos
 * are rewritten with the new target types and offsets.  This keeps the tool C
 * only and independent from general-purpose GGUF libraries.
 */

typedef struct {
    size_t start;
    size_t end;
} byte_span;

typedef struct {
    char *path;
    uint32_t version;
    uint64_t n_kv;
    uint64_t n_tensors;
    uint8_t *kv_raw;
    size_t kv_raw_len;
    size_t alignment;
    int n_experts;
    size_t data_offset;
    char *reuse_key;          /* quantize.reuse_key KV, if present (for --reuse) */
    char *reuse_key_weights;  /* quantize.reuse_key_weights KV (imatrix-independent half) */
    char *reuse_imatrix_coverage; /* quantize.reuse_imatrix_coverage KV (steered regular-tensor set) */
    tensor_meta *tensors;
    hmap tensor_map;
} gguf_file;

typedef struct {
    tensor_meta *tensors;
    uint64_t n_tensors;
    uint64_t n_kv_extra;
    size_t meta_size;
    size_t data_offset;
    size_t tensor_bytes;
    size_t alignment;
} output_context;

static size_t gguf_scalar_size(uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL: return 1;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16: return 2;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32: return 4;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64: return 8;
        default: return 0;
    }
}

static char *read_gguf_string_fp(FILE *fp) {
    uint64_t n = read_u64_le_fp(fp, "GGUF string length");
    char *s = xmalloc((size_t)n + 1);
    if (n && fread(s, 1, (size_t)n, fp) != (size_t)n) die("short GGUF string read");
    s[n] = '\0';
    return s;
}

static void skip_bytes_fp(FILE *fp, uint64_t n) {
    if (fseeko(fp, (off_t)n, SEEK_CUR) != 0) die("GGUF seek failed");
}

static void skip_gguf_value_fp(FILE *fp, uint32_t type) {
    if (type == GGUF_TYPE_STRING) {
        uint64_t n = read_u64_le_fp(fp, "GGUF string length");
        skip_bytes_fp(fp, n);
        return;
    }
    if (type == GGUF_TYPE_ARRAY) {
        uint32_t elem_type = read_u32_le_fp(fp, "GGUF array type");
        uint64_t n = read_u64_le_fp(fp, "GGUF array count");
        if (elem_type == GGUF_TYPE_STRING) {
            for (uint64_t i = 0; i < n; i++) {
                uint64_t len = read_u64_le_fp(fp, "GGUF array string length");
                skip_bytes_fp(fp, len);
            }
        } else {
            size_t sz = gguf_scalar_size(elem_type);
            if (!sz) die("unsupported GGUF array type");
            skip_bytes_fp(fp, n * sz);
        }
        return;
    }
    size_t sz = gguf_scalar_size(type);
    if (!sz) die("unsupported GGUF value type");
    skip_bytes_fp(fp, sz);
}

static size_t gguf_string_size(const char *s) {
    return sizeof(uint64_t) + strlen(s);
}

static void write_u32(FILE *fp, uint32_t v) {
    if (fwrite(&v, sizeof(v), 1, fp) != 1) die("write u32 failed");
}

static void write_u64(FILE *fp, uint64_t v) {
    if (fwrite(&v, sizeof(v), 1, fp) != 1) die("write u64 failed");
}

static void write_gguf_string(FILE *fp, const char *s) {
    uint64_t n = strlen(s);
    write_u64(fp, n);
    if (n && fwrite(s, 1, (size_t)n, fp) != (size_t)n) die("write string failed");
}

static bool is_imatrix_kv_key(const char *key) {
    return str_starts(key, "quantize.imatrix.");
}

static size_t extra_imatrix_kv_size(const imatrix_store *im) {
    if (!imatrix_enabled(im)) return 0;
    size_t n = 0;
    n += gguf_string_size(DS4_KV_QUANTIZE_IMATRIX_FILE) + 4 + gguf_string_size(im->file);
    n += gguf_string_size(DS4_KV_QUANTIZE_IMATRIX_N_ENTRIES) + 4 + 8;
    if (im->dataset) n += gguf_string_size(DS4_KV_QUANTIZE_IMATRIX_DATASET) + 4 + gguf_string_size(im->dataset);
    if (im->chunks > 0) n += gguf_string_size(DS4_KV_QUANTIZE_IMATRIX_N_CHUNKS) + 4 + 8;
    return n;
}

static uint64_t extra_imatrix_kv_count(const imatrix_store *im) {
    if (!imatrix_enabled(im)) return 0;
    return 2 + (im->dataset ? 1 : 0) + (im->chunks > 0 ? 1 : 0);
}

static void write_imatrix_kvs(FILE *fp, const imatrix_store *im) {
    if (!imatrix_enabled(im)) return;
    write_gguf_string(fp, DS4_KV_QUANTIZE_IMATRIX_FILE);
    write_u32(fp, GGUF_TYPE_STRING);
    write_gguf_string(fp, im->file);

    write_gguf_string(fp, DS4_KV_QUANTIZE_IMATRIX_N_ENTRIES);
    write_u32(fp, GGUF_TYPE_UINT64);
    write_u64(fp, (uint64_t)im->n_entries);

    if (im->dataset) {
        write_gguf_string(fp, DS4_KV_QUANTIZE_IMATRIX_DATASET);
        write_u32(fp, GGUF_TYPE_STRING);
        write_gguf_string(fp, im->dataset);
    }
    if (im->chunks > 0) {
        write_gguf_string(fp, DS4_KV_QUANTIZE_IMATRIX_N_CHUNKS);
        write_u32(fp, GGUF_TYPE_UINT64);
        write_u64(fp, (uint64_t)im->chunks);
    }
}

static size_t reuse_key_kv_size(void) {
    return gguf_string_size("quantize.reuse_key") + 4 + sizeof(uint64_t) + DS4_REUSE_KEY_HEXLEN;
}

static void write_reuse_key_kv(FILE *fp, const char *reuse_key) {
    write_gguf_string(fp, "quantize.reuse_key");
    write_u32(fp, GGUF_TYPE_STRING);
    write_gguf_string(fp, reuse_key);
}

/*
 * quantize.reuse_key_weights: the imatrix-INDEPENDENT half of the reuse identity
 * (safetensors index + shard stats + template salt, no imatrix). Tensors whose
 * quantization never reads the imatrix are byte-identical across builds that share
 * this key even when the imatrix differs — so a re-calibration can still copy them
 * from a prior build instead of regenerating.
 */
static size_t reuse_key_weights_kv_size(void) {
    return gguf_string_size("quantize.reuse_key_weights") + 4 + sizeof(uint64_t) + DS4_REUSE_KEY_HEXLEN;
}

static void write_reuse_key_weights_kv(FILE *fp, const char *key) {
    write_gguf_string(fp, "quantize.reuse_key_weights");
    write_u32(fp, GGUF_TYPE_STRING);
    write_gguf_string(fp, key);
}

static size_t reuse_imatrix_coverage_kv_size(void) {
    return gguf_string_size("quantize.reuse_imatrix_coverage") + 4 + sizeof(uint64_t) + DS4_REUSE_KEY_HEXLEN;
}

static void write_reuse_imatrix_coverage_kv(FILE *fp, const char *key) {
    write_gguf_string(fp, "quantize.reuse_imatrix_coverage");
    write_u32(fp, GGUF_TYPE_STRING);
    write_gguf_string(fp, key);
}

static gguf_file load_gguf_metadata(const char *path) {
    gguf_file g = {0};
    g.path = xstrdup(path);
    FILE *fp = fopen(path, "rb");
    if (!fp) die_errno("open GGUF", path);
    char magic[4];
    if (fread(magic, 1, sizeof(magic), fp) != sizeof(magic) || memcmp(magic, "GGUF", 4) != 0) {
        die("bad GGUF template");
    }
    g.version = read_u32_le_fp(fp, "GGUF version");
    g.n_tensors = read_u64_le_fp(fp, "GGUF tensor count");
    g.n_kv = read_u64_le_fp(fp, "GGUF KV count");
    g.alignment = DS4_GGUF_DEFAULT_ALIGNMENT;
    byte_span *kv_keep = xcalloc((size_t)g.n_kv, sizeof(kv_keep[0]));
    uint64_t n_kv_keep = 0;

    off_t kv_start = ftello(fp);
    if (kv_start < 0) die("GGUF ftell failed");
    for (uint64_t i = 0; i < g.n_kv; i++) {
        off_t rec_start = ftello(fp);
        if (rec_start < 0 || rec_start < kv_start) die("GGUF ftell failed");
        char *key = read_gguf_string_fp(fp);
        uint32_t type = read_u32_le_fp(fp, "GGUF KV type");
        if (strcmp(key, "general.alignment") == 0 && type == GGUF_TYPE_UINT32) {
            uint32_t a = read_u32_le_fp(fp, "GGUF alignment");
            if (a) g.alignment = a;
        } else if (strcmp(key, "deepseek4.expert_count") == 0 && type == GGUF_TYPE_UINT32) {
            uint32_t n = read_u32_le_fp(fp, "GGUF expert count");
            if (n <= (uint32_t)INT_MAX) g.n_experts = (int)n;
        } else if (strcmp(key, "deepseek4.expert_count") == 0 && type == GGUF_TYPE_UINT64) {
            uint64_t n = read_u64_le_fp(fp, "GGUF expert count");
            if (n <= (uint64_t)INT_MAX) g.n_experts = (int)n;
        } else if (strcmp(key, "quantize.reuse_key") == 0 && type == GGUF_TYPE_STRING) {
            free(g.reuse_key);
            g.reuse_key = read_gguf_string_fp(fp);
        } else if (strcmp(key, "quantize.reuse_key_weights") == 0 && type == GGUF_TYPE_STRING) {
            free(g.reuse_key_weights);
            g.reuse_key_weights = read_gguf_string_fp(fp);
        } else if (strcmp(key, "quantize.reuse_imatrix_coverage") == 0 && type == GGUF_TYPE_STRING) {
            free(g.reuse_imatrix_coverage);
            g.reuse_imatrix_coverage = read_gguf_string_fp(fp);
        } else {
            skip_gguf_value_fp(fp, type);
        }
        off_t rec_end = ftello(fp);
        if (rec_end < 0 || rec_end < rec_start) die("GGUF ftell failed");

        /*
         * Template GGUFs may already carry imatrix provenance from a previous
         * quantization.  Drop those keys and write the current run's keys later,
         * otherwise the output can contain duplicate GGUF metadata with stale
         * and new values.
         */
        if (!is_imatrix_kv_key(key) && strcmp(key, "quantize.reuse_key") != 0 &&
            strcmp(key, "quantize.reuse_key_weights") != 0 &&
            strcmp(key, "quantize.reuse_imatrix_coverage") != 0) {
            kv_keep[n_kv_keep++] = (byte_span){
                .start = (size_t)(rec_start - kv_start),
                .end = (size_t)(rec_end - kv_start),
            };
        }
        free(key);
    }
    off_t tensor_start = ftello(fp);
    if (tensor_start < 0 || tensor_start < kv_start) die("GGUF ftell failed");
    size_t kv_full_len = (size_t)(tensor_start - kv_start);
    uint8_t *kv_full = xmalloc(kv_full_len);
    if (fseeko(fp, kv_start, SEEK_SET) != 0) die("GGUF seek failed");
    if (kv_full_len && fread(kv_full, 1, kv_full_len, fp) != kv_full_len) die("GGUF KV read failed");

    for (uint64_t i = 0; i < n_kv_keep; i++) g.kv_raw_len += kv_keep[i].end - kv_keep[i].start;
    g.kv_raw = xmalloc(g.kv_raw_len);
    size_t kv_pos = 0;
    for (uint64_t i = 0; i < n_kv_keep; i++) {
        size_t n = kv_keep[i].end - kv_keep[i].start;
        memcpy(g.kv_raw + kv_pos, kv_full + kv_keep[i].start, n);
        kv_pos += n;
    }
    g.n_kv = n_kv_keep;
    free(kv_full);
    free(kv_keep);
    if (fseeko(fp, tensor_start, SEEK_SET) != 0) die("GGUF seek failed");

    g.tensors = xcalloc((size_t)g.n_tensors, sizeof(g.tensors[0]));
    for (uint64_t i = 0; i < g.n_tensors; i++) {
        tensor_meta *t = &g.tensors[i];
        t->name = read_gguf_string_fp(fp);
        t->n_dims = (int)read_u32_le_fp(fp, "GGUF tensor rank");
        if (t->n_dims < 1 || t->n_dims > DS4Q_MAX_DIMS) die("bad GGUF tensor rank");
        for (int j = 0; j < t->n_dims; j++) t->ne[j] = (int64_t)read_u64_le_fp(fp, "GGUF tensor dim");
        t->type = (ds4q_type)read_u32_le_fp(fp, "GGUF tensor type");
        t->old_offset = read_u64_le_fp(fp, "GGUF tensor offset");
        t->size = tensor_nbytes(t->type, t->ne, t->n_dims);
    }
    off_t meta_end = ftello(fp);
    if (meta_end < 0) die("GGUF ftell failed");
    g.data_offset = ds4q_pad((size_t)meta_end, g.alignment);
    char **keys = xmalloc((size_t)g.n_tensors * sizeof(keys[0]));
    for (uint64_t i = 0; i < g.n_tensors; i++) keys[i] = g.tensors[i].name;
    hmap_build(&g.tensor_map, keys, (int)g.n_tensors);
    free(keys);
    fclose(fp);
    return g;
}

static byte_buf read_gguf_tensor_data(const gguf_file *g, const char *path, const char *name) {
    int idx = hmap_get(&g->tensor_map, name);
    if (idx < 0) {
        fprintf(stderr, "error: tensor not found in GGUF: %s\n", name);
        exit(1);
    }
    const tensor_meta *t = &g->tensors[idx];
    byte_buf b = { .size = t->size, .data = xmalloc(t->size) };
    FILE *fp = fopen(path, "rb");
    if (!fp) die_errno("open GGUF", path);
    if (fseeko(fp, (off_t)(g->data_offset + t->old_offset), SEEK_SET) != 0) die_errno("seek GGUF", path);
    if (b.size && fread(b.data, 1, b.size, fp) != b.size) die_errno("read GGUF tensor", path);
    fclose(fp);
    return b;
}

static uint64_t fnv1a64_update(uint64_t h, const uint8_t *data, size_t n) {
    for (size_t i = 0; i < n; i++) {
        h ^= data[i];
        h *= 1099511628211ull;
    }
    return h;
}

static uint64_t fnv1a64_bytes(const uint8_t *data, size_t n) {
    return fnv1a64_update(1469598103934665603ull, data, n);
}

/* Stream a file through the rolling hash without holding it all in memory (the imatrix
 * .dat can be hundreds of MB). */
static uint64_t fnv1a64_file(uint64_t h, const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) die_errno("open", path);
    static uint8_t buf[1 << 20];
    size_t r;
    while ((r = fread(buf, 1, sizeof(buf), fp)) > 0) h = fnv1a64_update(h, buf, r);
    fclose(fp);
    return h;
}

/* Fold each *.safetensors shard's (name, size, mtime) into a salt — order-independent (XOR
 * of per-file hashes). Cheap (stat only, no reads), and an in-place weight change (which bumps
 * mtime/size) invalidates the reuse key, so --reuse won't copy stale tensors from a build made
 * against different weights at the same --hf path. */
static uint64_t shard_stat_salt(const char *hf_dir) {
    DIR *d = opendir(hf_dir);
    if (!d) return 0;
    uint64_t salt = 0;
    struct dirent *de;
    while ((de = readdir(d))) {
        if (!str_ends(de->d_name, ".safetensors")) continue;
        char *full = path_join(hf_dir, de->d_name);
        struct stat sb;
        if (stat(full, &sb) == 0) {
            uint64_t fh = fnv1a64_bytes((const uint8_t *)de->d_name, strlen(de->d_name));
            int64_t sz = (int64_t)sb.st_size, mt = (int64_t)sb.st_mtime;
            fh = fnv1a64_update(fh, (const uint8_t *)&sz, sizeof(sz));
            fh = fnv1a64_update(fh, (const uint8_t *)&mt, sizeof(mt));
            salt ^= fh;
        }
        free(full);
    }
    closedir(d);
    return salt;
}

/* A reuse key identifies the (model weights, imatrix, template structure) a build came from.
 * Two builds with the same key produce byte-identical tensors for any tensor of the same
 * target type+shape (quantization is deterministic), so --reuse can copy them instead of
 * regenerating. Cheap: hashes the safetensors index (structure) + each shard's size/mtime +
 * the imatrix content + a structural salt — NOT the multi-GB weight bytes. The key is 16 hex
 * chars. (If you modify shards in place without touching mtime/size, see the caveat.) */
static char *compute_reuse_key(const char *hf_dir, const char *imatrix_file, const gguf_file *tmpl) {
    uint64_t h = 1469598103934665603ull;
    char *idx = path_join(hf_dir, "model.safetensors.index.json");
    size_t n = 0;
    char *buf = read_file(idx, &n);
    h = fnv1a64_update(h, (const uint8_t *)buf, n);
    free(buf);
    free(idx);
    uint64_t shards = shard_stat_salt(hf_dir);
    h = fnv1a64_update(h, (const uint8_t *)&shards, sizeof(shards));
    if (imatrix_file) h = fnv1a64_file(h, imatrix_file);
    uint64_t salt[2] = { (uint64_t)tmpl->n_tensors, (uint64_t)tmpl->alignment };
    h = fnv1a64_update(h, (const uint8_t *)salt, sizeof(salt));
    char *s = xmalloc(DS4_REUSE_KEY_HEXLEN + 1);
    snprintf(s, DS4_REUSE_KEY_HEXLEN + 1, "%016llx", (unsigned long long)h);
    return s;
}

static output_context build_output_context(const gguf_file *tmpl, const quant_policy *policy, const imatrix_store *im) {
    output_context out = {0};
    out.n_tensors = tmpl->n_tensors;
    out.n_kv_extra = extra_imatrix_kv_count(im) + 3;   /* quantize.reuse_key + _weights + _imatrix_coverage */
    out.alignment = tmpl->alignment;
    out.tensors = xcalloc((size_t)out.n_tensors, sizeof(out.tensors[0]));
    size_t tensor_info = 0;
    size_t off = 0;
    for (uint64_t i = 0; i < out.n_tensors; i++) {
        const tensor_meta *src = &tmpl->tensors[i];
        tensor_meta *dst = &out.tensors[i];
        *dst = *src;
        dst->name = src->name;
        ds4q_type type = policy_type(policy, src->name, src);
        if (type == DS4Q_TYPE_COUNT) type = src->type;
        if (type != DS4Q_TYPE_I32 && !is_quantizable_target(type)) die("unsupported planned tensor type");
        if (ds4q_can_quantize(type) && src->ne[0] % ds4q_block_size(type) != 0) die("ne[0] not divisible by block size");
        dst->type = type;
        dst->size = tensor_nbytes(type, src->ne, src->n_dims);
        dst->new_offset = off;
        off += ds4q_pad(dst->size, tmpl->alignment);
        tensor_info += gguf_string_size(dst->name) + 4 + (size_t)dst->n_dims * 8 + 4 + 8;
    }
    out.tensor_bytes = off;
    out.meta_size = 4 + 4 + 8 + 8 + tmpl->kv_raw_len + extra_imatrix_kv_size(im) + reuse_key_kv_size()
                    + reuse_key_weights_kv_size() + reuse_imatrix_coverage_kv_size() + tensor_info;
    out.data_offset = ds4q_pad(out.meta_size, tmpl->alignment);
    return out;
}

static void dspark_add_tensor(gguf_file *tmpl, uint64_t *position, int stage,
                              const dspark_tensor_spec *spec) {
    if (*position >= DS4_DSPARK_OUTPUT_TENSORS)
        die("DSpark output tensor plan overflow");
    tensor_meta *tensor = &tmpl->tensors[(*position)++];
    char name[512];
    snprintf(name, sizeof(name), "mtp.%d.%s", stage, spec->suffix);
    tensor->name = xstrdup(name);
    tensor->n_dims = spec->n_dims;
    for (int dim = 0; dim < spec->n_dims; dim++)
        tensor->ne[dim] = spec->ne[dim];
    tensor->type = spec->type;
    tensor->size = tensor_nbytes(tensor->type, tensor->ne, tensor->n_dims);
}

static gguf_file make_dspark_support_template(void) {
    gguf_file tmpl = {0};
    tmpl.path = xstrdup("final-0731-dspark-support-plan");
    tmpl.version = 3;
    tmpl.n_tensors = DS4_DSPARK_OUTPUT_TENSORS;
    tmpl.n_kv = 15;
    tmpl.alignment = DS4_GGUF_DEFAULT_ALIGNMENT;
    tmpl.n_experts = DS4_DSPARK_EXPERT_COUNT;
    tmpl.tensors = xcalloc((size_t)tmpl.n_tensors, sizeof(tmpl.tensors[0]));
    uint64_t position = 0;
    for (int stage = 0; stage < DS4_DSPARK_STAGE_COUNT; stage++) {
        for (size_t i = 0;
             i < sizeof(dspark_block_specs) / sizeof(dspark_block_specs[0]);
             i++)
            dspark_add_tensor(&tmpl, &position, stage, &dspark_block_specs[i]);
        if (stage == 0) {
            for (size_t i = 0;
                 i < sizeof(dspark_stage0_specs) /
                         sizeof(dspark_stage0_specs[0]); i++)
                dspark_add_tensor(&tmpl, &position, stage,
                                  &dspark_stage0_specs[i]);
        }
        if (stage == DS4_DSPARK_STAGE_COUNT - 1) {
            for (size_t i = 0;
                 i < sizeof(dspark_stage2_specs) /
                         sizeof(dspark_stage2_specs[0]); i++)
                dspark_add_tensor(&tmpl, &position, stage,
                                  &dspark_stage2_specs[i]);
        }
    }
    if (position != DS4_DSPARK_OUTPUT_TENSORS)
        die("DSpark output tensor plan is not exactly 81 tensors");
    char **keys = xmalloc((size_t)tmpl.n_tensors * sizeof(keys[0]));
    for (uint64_t i = 0; i < tmpl.n_tensors; i++) keys[i] = tmpl.tensors[i].name;
    hmap_build(&tmpl.tensor_map, keys, (int)tmpl.n_tensors);
    free(keys);
    return tmpl;
}

static size_t dspark_metadata_size(void) {
    size_t size = 0;
#define DSPARK_STRING_KV_SIZE(key, value) \
    (gguf_string_size(key) + 4 + gguf_string_size(value))
#define DSPARK_U32_KV_SIZE(key) (gguf_string_size(key) + 4 + 4)
    size += DSPARK_STRING_KV_SIZE("general.architecture", "deepseek4-dspark");
    size += DSPARK_STRING_KV_SIZE("general.name",
                                  "DeepSeek V4 Flash DSpark support");
    size += DSPARK_U32_KV_SIZE("general.alignment");
    size += DSPARK_U32_KV_SIZE("dspark.block_size");
    size += DSPARK_U32_KV_SIZE("dspark.markov_rank");
    size += DSPARK_U32_KV_SIZE("dspark.noise_token_id");
    size += gguf_string_size("dspark.target_layer_ids") + 4 + 4 + 8 + 3 * 4;
    size += DSPARK_U32_KV_SIZE("dspark.stage_count");
    size += DSPARK_U32_KV_SIZE("dspark.n_layers");
    size += DSPARK_STRING_KV_SIZE("dspark.source.revision",
                                  DS4_DSPARK_SOURCE_REVISION);
    size += DSPARK_STRING_KV_SIZE("dspark.source.config_sha256",
                                  DS4_DSPARK_CONFIG_SHA256);
    size += DSPARK_STRING_KV_SIZE("dspark.source.index_sha256",
                                  DS4_DSPARK_INDEX_SHA256);
    size += DSPARK_STRING_KV_SIZE("dspark.source.shard46_sha256",
                                  DS4_DSPARK_SHARD46_SHA256);
    size += DSPARK_STRING_KV_SIZE("dspark.source.shard47_sha256",
                                  DS4_DSPARK_SHARD47_SHA256);
    size += DSPARK_STRING_KV_SIZE("dspark.source.shard48_sha256",
                                  DS4_DSPARK_SHARD48_SHA256);
#undef DSPARK_U32_KV_SIZE
#undef DSPARK_STRING_KV_SIZE
    return size;
}

static output_context build_dspark_output_context(const gguf_file *tmpl) {
    output_context out = {0};
    out.n_tensors = tmpl->n_tensors;
    out.n_kv_extra = 0;
    out.alignment = tmpl->alignment;
    out.tensors = xcalloc((size_t)out.n_tensors, sizeof(out.tensors[0]));
    size_t tensor_info = 0;
    size_t offset = 0;
    for (uint64_t i = 0; i < out.n_tensors; i++) {
        out.tensors[i] = tmpl->tensors[i];
        out.tensors[i].new_offset = offset;
        offset += ds4q_pad(out.tensors[i].size, out.alignment);
        tensor_info += gguf_string_size(out.tensors[i].name) + 4 +
                       (size_t)out.tensors[i].n_dims * 8 + 4 + 8;
    }
    out.tensor_bytes = offset;
    out.meta_size = 4 + 4 + 8 + 8 + dspark_metadata_size() + tensor_info;
    out.data_offset = ds4q_pad(out.meta_size, out.alignment);
    return out;
}

static void write_string_kv(FILE *fp, const char *key, const char *value) {
    write_gguf_string(fp, key);
    write_u32(fp, GGUF_TYPE_STRING);
    write_gguf_string(fp, value);
}

static void write_u32_kv(FILE *fp, const char *key, uint32_t value) {
    write_gguf_string(fp, key);
    write_u32(fp, GGUF_TYPE_UINT32);
    write_u32(fp, value);
}

static void write_dspark_metadata(FILE *fp) {
    write_string_kv(fp, "general.architecture", "deepseek4-dspark");
    write_string_kv(fp, "general.name", "DeepSeek V4 Flash DSpark support");
    write_u32_kv(fp, "general.alignment", DS4_GGUF_DEFAULT_ALIGNMENT);
    write_u32_kv(fp, "dspark.block_size", 5);
    write_u32_kv(fp, "dspark.markov_rank", 256);
    write_u32_kv(fp, "dspark.noise_token_id", 128799);
    write_gguf_string(fp, "dspark.target_layer_ids");
    write_u32(fp, GGUF_TYPE_ARRAY);
    write_u32(fp, GGUF_TYPE_UINT32);
    write_u64(fp, 3);
    write_u32(fp, 40);
    write_u32(fp, 41);
    write_u32(fp, 42);
    write_u32_kv(fp, "dspark.stage_count", DS4_DSPARK_STAGE_COUNT);
    write_u32_kv(fp, "dspark.n_layers", DS4_DSPARK_STAGE_COUNT);
    write_string_kv(fp, "dspark.source.revision",
                    DS4_DSPARK_SOURCE_REVISION);
    write_string_kv(fp, "dspark.source.config_sha256",
                    DS4_DSPARK_CONFIG_SHA256);
    write_string_kv(fp, "dspark.source.index_sha256",
                    DS4_DSPARK_INDEX_SHA256);
    write_string_kv(fp, "dspark.source.shard46_sha256",
                    DS4_DSPARK_SHARD46_SHA256);
    write_string_kv(fp, "dspark.source.shard47_sha256",
                    DS4_DSPARK_SHARD47_SHA256);
    write_string_kv(fp, "dspark.source.shard48_sha256",
                    DS4_DSPARK_SHARD48_SHA256);
}

typedef struct {
    bool include_shards;
    int fds[DS4_DSPARK_SOURCE_FILES];
    char *paths[DS4_DSPARK_SOURCE_FILES];
    dspark_file_identity initial[DS4_DSPARK_SOURCE_FILES];
    st_borrowed_file borrowed_shards[DS4_DSPARK_STAGE_COUNT];
} dspark_source_files;

static const char *const dspark_source_names[DS4_DSPARK_SOURCE_FILES] = {
    "config.json",
    "model.safetensors.index.json",
    "model-00046-of-00048.safetensors",
    "model-00047-of-00048.safetensors",
    "model-00048-of-00048.safetensors",
};

static const uint64_t dspark_source_sizes[DS4_DSPARK_SOURCE_FILES] = {
    DS4_DSPARK_CONFIG_BYTES,
    DS4_DSPARK_INDEX_BYTES,
    DS4_DSPARK_SHARD46_BYTES,
    DS4_DSPARK_SHARD47_BYTES,
    DS4_DSPARK_SHARD48_BYTES,
};

static const char *const dspark_source_hashes[DS4_DSPARK_SOURCE_FILES] = {
    DS4_DSPARK_CONFIG_SHA256,
    DS4_DSPARK_INDEX_SHA256,
    DS4_DSPARK_SHARD46_SHA256,
    DS4_DSPARK_SHARD47_SHA256,
    DS4_DSPARK_SHARD48_SHA256,
};

static dspark_source_files open_dspark_source_files(
        const char *hf_dir, bool include_shards) {
    dspark_source_files source = { .include_shards = include_shards };
    for (int i = 0; i < DS4_DSPARK_SOURCE_FILES; i++) source.fds[i] = -1;
    const int count = include_shards ? DS4_DSPARK_SOURCE_FILES : 2;
    for (int i = 0; i < count; i++) {
        source.paths[i] = path_join(hf_dir, dspark_source_names[i]);
        source.fds[i] = open(source.paths[i],
                             O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
        if (source.fds[i] < 0) die_errno("open authenticated input",
                                         source.paths[i]);
        char label[64];
        if (i < 2) {
            snprintf(label, sizeof(label), "final 0731 %s",
                     i == 0 ? "config" : "index");
        } else {
            snprintf(label, sizeof(label), "final 0731 DSpark shard %d",
                     i + 44);
        }
        source.initial[i] = require_fd_identity(
            source.fds[i], source.paths[i], dspark_source_sizes[i],
            dspark_source_hashes[i], label);
    }
    if (include_shards) {
        for (int stage = 0; stage < DS4_DSPARK_STAGE_COUNT; stage++) {
            source.borrowed_shards[stage] = (st_borrowed_file){
                .file = dspark_shards[stage],
                .fd = source.fds[stage + 2],
            };
        }
    }
    return source;
}

static void require_dspark_source_fds_unchanged(
        const dspark_source_files *source) {
    const int count = source->include_shards ? DS4_DSPARK_SOURCE_FILES : 2;
    for (int i = 0; i < count; i++) {
        char label[64];
        snprintf(label, sizeof(label), "authenticated source slot %d", i);
        dspark_file_identity final = require_fd_identity(
            source->fds[i], source->paths[i], dspark_source_sizes[i],
            dspark_source_hashes[i], label);
        const dspark_file_identity *initial = &source->initial[i];
        if (initial->device != final.device ||
            initial->inode != final.inode || initial->size != final.size ||
            strcmp(initial->sha256, final.sha256) != 0)
            die("authenticated DSpark descriptor changed while in use");
    }
}

static void close_dspark_source_files(dspark_source_files *source) {
    for (int i = 0; i < DS4_DSPARK_SOURCE_FILES; i++) {
        if (source->fds[i] >= 0) close(source->fds[i]);
        free(source->paths[i]);
        source->fds[i] = -1;
        source->paths[i] = NULL;
    }
}

static void write_padding(FILE *fp, size_t n) {
    static const uint8_t zeros[4096] = {0};
    while (n) {
        size_t chunk = n < sizeof(zeros) ? n : sizeof(zeros);
        if (fwrite(zeros, 1, chunk, fp) != chunk) die("write padding failed");
        n -= chunk;
    }
}

static char *dspark_active_temp_path;
static dev_t dspark_active_temp_device;
static ino_t dspark_active_temp_inode;
static bool dspark_active_temp_identity;
static char *dspark_active_destination_path;
static dev_t dspark_active_destination_device;
static ino_t dspark_active_destination_inode;
static bool dspark_active_destination_identity;

static bool unlink_owned_path(const char *path, dev_t device, ino_t inode) {
    struct stat st;
    if (lstat(path, &st) != 0) return errno == ENOENT;
    if (st.st_dev != device || st.st_ino != inode) return false;
    return unlink(path) == 0 || errno == ENOENT;
}

static void cleanup_dspark_temporary_output(void) {
    if (dspark_active_destination_path && dspark_active_destination_identity)
        unlink_owned_path(dspark_active_destination_path,
                          dspark_active_destination_device,
                          dspark_active_destination_inode);
    if (dspark_active_temp_path && dspark_active_temp_identity)
        unlink_owned_path(dspark_active_temp_path,
                          dspark_active_temp_device,
                          dspark_active_temp_inode);
}

#ifdef DS4_DSPARK_QUANTIZER_TEST_CONTRACT
/* Deterministic test-only rendezvous for the swap/restore TOCTOU regression. */
static void dspark_test_gate_for(const char *environment, const char *phase) {
    const char *directory = getenv(environment);
    if (!directory || !directory[0]) return;
    char ready_name[128], continue_name[128];
    snprintf(ready_name, sizeof(ready_name), "%s.ready", phase);
    snprintf(continue_name, sizeof(continue_name), "%s.continue", phase);
    char *ready = path_join(directory, ready_name);
    char *proceed = path_join(directory, continue_name);
    int fd = open(ready, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC,
                  0600);
    if (fd < 0) die_errno("create DSpark test gate", ready);
    close(fd);
    for (int attempt = 0; attempt < 30000; attempt++) {
        struct stat st;
        if (lstat(proceed, &st) == 0) {
            free(ready);
            free(proceed);
            return;
        }
        if (errno != ENOENT) die_errno("inspect DSpark test gate", proceed);
        usleep(1000);
    }
    die("timed out waiting for DSpark swap test gate");
}

static void dspark_test_gate(const char *phase) {
    dspark_test_gate_for("DS4_DSPARK_TEST_SWAP_GATE", phase);
}
#endif

static char *parent_directory(const char *path) {
    char *parent = xstrdup(path);
    char *slash = strrchr(parent, '/');
    if (!slash) {
        free(parent);
        return xstrdup(".");
    }
    if (slash == parent) {
        slash[1] = '\0';
    } else {
        *slash = '\0';
    }
    return parent;
}

static void copy_fd_exact(int source_fd, int destination_fd, uint64_t size,
                          const char *label) {
    uint8_t *buffer = xmalloc(1u << 20);
    uint64_t offset = 0;
    while (offset < size) {
        size_t take = size - offset < (1u << 20)
            ? (size_t)(size - offset) : (1u << 20);
        pread_exact_fd(source_fd, buffer, take, (off_t)offset, label);
        size_t written = 0;
        while (written < take) {
            ssize_t count = pwrite(destination_fd, buffer + written,
                                   take - written,
                                   (off_t)(offset + written));
            if (count < 0) die_errno("copy installed output", label);
            if (count == 0) die("short write while installing output");
            written += (size_t)count;
        }
        offset += take;
    }
    free(buffer);
}

#ifdef __APPLE__
static bool clone_fallback_errno(int error) {
    return error == ENOTSUP || error == EOPNOTSUPP ||
           error == ENOSYS || error == EXDEV;
}
#endif

/* A prior tensor is reusable iff it has the same name, target type and shape as the planned
 * output tensor. The matching reuse key (checked by the caller) already guarantees the same
 * weights+imatrix, and quantization is deterministic, so the bytes are identical. */
static bool reuse_eligible(const gguf_file *prior, const tensor_meta *dst) {
    if (!prior) return false;
    int idx = hmap_get(&prior->tensor_map, dst->name);
    if (idx < 0) return false;
    const tensor_meta *pt = &prior->tensors[idx];
    if (pt->type != dst->type || pt->n_dims != dst->n_dims || pt->size != dst->size) return false;
    for (int j = 0; j < dst->n_dims; j++) if (pt->ne[j] != dst->ne[j]) return false;
    return true;
}

/*
 * Conservative imatrix-dependence test for the per-tensor reuse gate. Mirrors the
 * lookups the generators perform: routed expert families (".._exps.") are always
 * imatrix-steered when an imatrix is loaded; for regular tensors we probe the same
 * name candidates generate_regular() would. Over-approximation is safe (a tensor
 * wrongly marked dependent is merely regenerated); under-approximation is not.
 */
static bool tensor_uses_imatrix(const imatrix_store *im, const tensor_meta *dst) {
    if (!imatrix_enabled(im)) return false;
    if (dst->type == DS4Q_TYPE_I32) return false;   /* generate_regular returns before any lookup */
    if (strstr(dst->name, "_exps.") != NULL) return true;
    char *hf = hf_name_for_regular(dst->name);
    const char *names[2] = { dst->name, hf };
    const float *hit = imatrix_find(im, names, 2, dst->ne[0], -1, 0);
    free(hf);
    return hit != NULL;
}

/*
 * Coverage identity of (imatrix, plan): an fnv1a64 over the names of the REGULAR
 * (non-expert) output tensors this imatrix actually steers, in plan order. Two builds
 * may copy each other's imatrix-independent tensors only when this set agrees: a
 * regular tensor steered by the PRIOR's imatrix but not by ours would otherwise be
 * copied with stale bytes (the prior GGUF records no per-tensor coverage, so equality
 * of the two coverage keys is the only sound gate). Routed expert tensors are excluded:
 * weights-only mode always regenerates them.
 */
static char *compute_imatrix_coverage_key(const imatrix_store *im, const output_context *out_ctx) {
    uint64_t h = 1469598103934665603ull;
    for (uint64_t i = 0; i < out_ctx->n_tensors; i++) {
        const tensor_meta *dst = &out_ctx->tensors[i];
        if (strstr(dst->name, "_exps.") != NULL) continue;
        if (!tensor_uses_imatrix(im, dst)) continue;
        h = fnv1a64_update(h, (const uint8_t *)dst->name, strlen(dst->name));
        h = fnv1a64_update(h, (const uint8_t *)"\n", 1);
    }
    char *s = xmalloc(DS4_REUSE_KEY_HEXLEN + 1);
    snprintf(s, DS4_REUSE_KEY_HEXLEN + 1, "%016llx", (unsigned long long)h);
    return s;
}

static void write_full_gguf(st_db *db, const gguf_file *tmpl, const output_context *out_ctx,
                            const char *out_path, int n_experts, int n_threads,
                            const imatrix_store *imatrix, const char *reuse_key,
                            const char *reuse_key_weights, const char *reuse_imatrix_coverage,
                            bool reuse_weights_only, const gguf_file *prior) {
    int output_fd = open(out_path,
                         O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC,
                         0644);
    if (output_fd < 0) die_errno("create exclusive output", out_path);
    FILE *fp = fdopen(output_fd, "wb");
    if (!fp) {
        close(output_fd);
        die_errno("open output stream", out_path);
    }
    if (fwrite("GGUF", 1, 4, fp) != 4) die("write GGUF magic failed");
    write_u32(fp, tmpl->version);
    write_u64(fp, tmpl->n_tensors);
    write_u64(fp, tmpl->n_kv + out_ctx->n_kv_extra);
    if (fwrite(tmpl->kv_raw, 1, tmpl->kv_raw_len, fp) != tmpl->kv_raw_len) die("write GGUF KV failed");
    write_imatrix_kvs(fp, imatrix);
    write_reuse_key_kv(fp, reuse_key);
    write_reuse_key_weights_kv(fp, reuse_key_weights);
    write_reuse_imatrix_coverage_kv(fp, reuse_imatrix_coverage);
    for (uint64_t i = 0; i < out_ctx->n_tensors; i++) {
        const tensor_meta *t = &out_ctx->tensors[i];
        write_gguf_string(fp, t->name);
        write_u32(fp, (uint32_t)t->n_dims);
        for (int j = 0; j < t->n_dims; j++) write_u64(fp, (uint64_t)t->ne[j]);
        write_u32(fp, (uint32_t)t->type);
        write_u64(fp, t->new_offset);
    }
    long pos = ftell(fp);
    if (pos < 0) die("ftell failed");
    if ((size_t)pos > out_ctx->data_offset) die("GGUF metadata larger than planned");
    write_padding(fp, out_ctx->data_offset - (size_t)pos);

    uint64_t n_reused = 0;
    for (uint64_t i = 0; i < out_ctx->n_tensors; i++) {
        const tensor_meta *src = &tmpl->tensors[i];
        const tensor_meta *dst = &out_ctx->tensors[i];
        byte_buf data;
        bool reused = reuse_eligible(prior, dst) &&
                      (!reuse_weights_only || !tensor_uses_imatrix(imatrix, dst));
        if (reused) {
            data = read_gguf_tensor_data(prior, prior->path, dst->name);
            n_reused++;
        }
        fprintf(stderr, "[%4" PRIu64 "/%4" PRIu64 "] %s -> %s%s\n", i + 1, out_ctx->n_tensors,
                dst->name, ds4q_type_name(dst->type), reused ? "  (reused)" : "");
        if (!reused) data = generate_tensor(db, dst->name, src, dst->type, n_experts, n_threads, imatrix);
        size_t expected = dst->size;
        if (data.size != expected) {
            fprintf(stderr, "error: %s size mismatch for %s: got %zu expected %zu\n",
                    reused ? "reused" : "generated", dst->name, data.size, expected);
            exit(1);
        }
        if (fwrite(data.data, 1, data.size, fp) != data.size) die_errno("write tensor", out_path);
        size_t padded = ds4q_pad(data.size, out_ctx->alignment);
        write_padding(fp, padded - data.size);
        fprintf(stderr, "       %s %.2f MiB\n", reused ? "copied" : "generated", (double)data.size / 1048576.0);
        free(data.data);
    }
    if (prior) fprintf(stderr, "reuse: copied %" PRIu64 " / %" PRIu64 " tensors from %s\n",
                       n_reused, out_ctx->n_tensors, prior->path);
    fclose(fp);
}

static void write_dspark_support_gguf(st_db *db, const gguf_file *tmpl,
                                      const output_context *out_ctx,
                                      const dspark_source_files *source_files,
                                      const char *out_path, int n_threads) {
    size_t temp_size = strlen(out_path) + 64;
    char *temp_path = xmalloc(temp_size);
    snprintf(temp_path, temp_size, "%s.tmp.%ld", out_path, (long)getpid());
    static bool cleanup_registered;
    if (!cleanup_registered) {
        if (atexit(cleanup_dspark_temporary_output) != 0)
            die("cannot register DSpark temporary-output cleanup");
        cleanup_registered = true;
    }
    int temp_fd = open(temp_path,
                       O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC,
                       0644);
    if (temp_fd < 0) die_errno("create exclusive output", temp_path);
    dspark_active_temp_path = temp_path;
    struct stat owned_temp_stat;
    if (fstat(temp_fd, &owned_temp_stat) != 0)
        die_errno("stat owned temporary output", temp_path);
    dspark_active_temp_device = owned_temp_stat.st_dev;
    dspark_active_temp_inode = owned_temp_stat.st_ino;
    dspark_active_temp_identity = true;
    int stream_fd = fcntl(temp_fd, F_DUPFD_CLOEXEC, 0);
    if (stream_fd < 0) die_errno("duplicate output descriptor", temp_path);
    FILE *fp = fdopen(stream_fd, "wb");
    if (!fp) {
        close(stream_fd);
        die_errno("open output stream", temp_path);
    }
#ifdef DS4_DSPARK_QUANTIZER_TEST_CONTRACT
    dspark_test_gate("after-open");
#endif
    if (fwrite("GGUF", 1, 4, fp) != 4) die("write GGUF magic failed");
    write_u32(fp, tmpl->version);
    write_u64(fp, tmpl->n_tensors);
    write_u64(fp, tmpl->n_kv);
    write_dspark_metadata(fp);
    for (uint64_t i = 0; i < out_ctx->n_tensors; i++) {
        const tensor_meta *tensor = &out_ctx->tensors[i];
        write_gguf_string(fp, tensor->name);
        write_u32(fp, (uint32_t)tensor->n_dims);
        for (int dim = 0; dim < tensor->n_dims; dim++)
            write_u64(fp, (uint64_t)tensor->ne[dim]);
        write_u32(fp, (uint32_t)tensor->type);
        write_u64(fp, tensor->new_offset);
    }
    off_t position = ftello(fp);
    if (position < 0 || (size_t)position > out_ctx->data_offset)
        die("DSpark GGUF metadata larger than planned");
    write_padding(fp, out_ctx->data_offset - (size_t)position);

    for (uint64_t i = 0; i < out_ctx->n_tensors; i++) {
        const tensor_meta *source = &tmpl->tensors[i];
        const tensor_meta *output = &out_ctx->tensors[i];
        fprintf(stderr, "[%2" PRIu64 "/%2" PRIu64 "] %s -> %s\n",
                i + 1, out_ctx->n_tensors, output->name,
                ds4q_type_name(output->type));
        byte_buf data = generate_tensor(
            db, output->name, source, output->type,
            DS4_DSPARK_EXPERT_COUNT, n_threads, NULL
        );
        if (data.size != output->size) {
            fprintf(stderr,
                    "error: generated size mismatch for %s: got %zu expected %zu\n",
                    output->name, data.size, output->size);
            exit(1);
        }
        if (data.size && fwrite(data.data, 1, data.size, fp) != data.size)
            die_errno("write tensor", temp_path);
        write_padding(fp, ds4q_pad(data.size, out_ctx->alignment) - data.size);
        free(data.data);
    }
    if (fflush(fp) != 0 || fsync(temp_fd) != 0)
        die_errno("sync output", temp_path);
    if (fclose(fp) != 0) die_errno("close output", temp_path);
#ifdef DS4_DSPARK_QUANTIZER_TEST_CONTRACT
    dspark_test_gate("before-final-auth");
#endif
    require_dspark_source_fds_unchanged(source_files);

    const uint64_t expected_size =
        (uint64_t)out_ctx->data_offset + (uint64_t)out_ctx->tensor_bytes;
    struct stat output_fd_stat;
    if (fstat(temp_fd, &output_fd_stat) != 0)
        die_errno("stat output descriptor", temp_path);
    if (!S_ISREG(output_fd_stat.st_mode) || output_fd_stat.st_size < 0 ||
        (uint64_t)output_fd_stat.st_size != expected_size) {
        fprintf(stderr,
                "error: generated DSpark support size mismatch: got %" PRIu64
                " expected %" PRIu64 "\n",
                output_fd_stat.st_size < 0 ? UINT64_C(0) :
                    (uint64_t)output_fd_stat.st_size,
                expected_size);
        exit(1);
    }
    char output_sha256[65];
    sha256_fd_hex(temp_fd, output_sha256, "DSpark output descriptor");

#ifdef DS4_DSPARK_QUANTIZER_TEST_CONTRACT
    dspark_test_gate("before-install");
#endif

    char *parent = parent_directory(out_path);
    const char *name = strrchr(out_path, '/');
    name = name ? name + 1 : out_path;
    if (!name[0]) die("output filename is empty");
    int directory_fd = open(parent, O_RDONLY | O_DIRECTORY | O_CLOEXEC);
    if (directory_fd < 0) die_errno("open output directory", parent);
    int installed_fd = -1;
    bool copy_fallback = false;
    bool cloned_output = false;
    struct stat cloned_path_stat = {0};
#ifdef __APPLE__
    int clone_result;
#ifdef DS4_DSPARK_QUANTIZER_TEST_CONTRACT
    if (getenv("DS4_DSPARK_TEST_FORCE_FCLONE_UNSUPPORTED")) {
        errno = ENOTSUP;
        clone_result = -1;
    } else {
        clone_result = fclonefileat(temp_fd, directory_fd, name, 0);
    }
#else
    clone_result = fclonefileat(temp_fd, directory_fd, name, 0);
#endif
    if (clone_result != 0) {
        int clone_error = errno;
        if (clone_error == EEXIST) die("output appeared during build");
        if (!clone_fallback_errno(clone_error))
            die_errno("clone verified output descriptor", out_path);
        copy_fallback = true;
    } else {
        if (fstatat(directory_fd, name, &cloned_path_stat,
                    AT_SYMLINK_NOFOLLOW) != 0)
            die_errno("stat cloned output", out_path);
        if (!S_ISREG(cloned_path_stat.st_mode))
            die("cloned output is not a regular file");
        dspark_active_destination_path = (char *)out_path;
        dspark_active_destination_device = cloned_path_stat.st_dev;
        dspark_active_destination_inode = cloned_path_stat.st_ino;
        dspark_active_destination_identity = true;
        cloned_output = true;
#ifdef DS4_DSPARK_QUANTIZER_TEST_CONTRACT
        dspark_test_gate_for("DS4_DSPARK_TEST_CLONE_REOPEN_GATE",
                             "after-clone-stat");
        if (getenv("DS4_DSPARK_TEST_FORCE_CLONE_REOPEN_FAILURE")) {
            errno = EIO;
            installed_fd = -1;
        } else {
            installed_fd = openat(directory_fd, name,
                                  O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
        }
#else
        installed_fd = openat(directory_fd, name,
                              O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
#endif
        if (installed_fd < 0)
            die_errno("open cloned output", out_path);
    }
#else
    copy_fallback = true;
#endif
    struct stat installed_fd_stat;
    if (copy_fallback) {
        installed_fd = openat(
            directory_fd, name,
            O_RDWR | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0644
        );
        if (installed_fd < 0) {
            if (errno == EEXIST) die("output appeared during build");
            die_errno("create installed output", out_path);
        }
        if (fstat(installed_fd, &installed_fd_stat) != 0)
            die_errno("stat installed output", out_path);
        dspark_active_destination_path = (char *)out_path;
        dspark_active_destination_device = installed_fd_stat.st_dev;
        dspark_active_destination_inode = installed_fd_stat.st_ino;
        dspark_active_destination_identity = true;
        if (ftruncate(installed_fd, (off_t)expected_size) != 0)
            die_errno("size installed output", out_path);
        copy_fd_exact(temp_fd, installed_fd, expected_size, out_path);
        if (fsync(installed_fd) != 0)
            die_errno("sync installed output", out_path);
    }

    if (fstat(installed_fd, &installed_fd_stat) != 0)
        die_errno("stat installed output descriptor", out_path);
    if (cloned_output &&
        (installed_fd_stat.st_dev != cloned_path_stat.st_dev ||
         installed_fd_stat.st_ino != cloned_path_stat.st_ino))
        die("cloned output identity changed before reopen");
    if (!cloned_output) {
        dspark_active_destination_path = (char *)out_path;
        dspark_active_destination_device = installed_fd_stat.st_dev;
        dspark_active_destination_inode = installed_fd_stat.st_ino;
        dspark_active_destination_identity = true;
    }
#ifdef DS4_DSPARK_QUANTIZER_TEST_CONTRACT
    dspark_test_gate("after-installed-open");
#endif
    char installed_sha256[65];
    sha256_fd_hex(installed_fd, installed_sha256,
                  "installed DSpark output descriptor");
    if (!S_ISREG(installed_fd_stat.st_mode) || installed_fd_stat.st_size < 0 ||
        (uint64_t)installed_fd_stat.st_size != expected_size ||
        strcmp(installed_sha256, output_sha256) != 0)
        die("installed DSpark support does not match verified output descriptor");
    struct stat output_stat;
    if (fstatat(directory_fd, name, &output_stat, AT_SYMLINK_NOFOLLOW) != 0)
        die_errno("stat installed output path", out_path);
    if (!S_ISREG(output_stat.st_mode) ||
        output_stat.st_dev != installed_fd_stat.st_dev ||
        output_stat.st_ino != installed_fd_stat.st_ino ||
        output_stat.st_size != installed_fd_stat.st_size)
        die("installed DSpark support identity changed unexpectedly");
    if (fsync(directory_fd) != 0)
        die_errno("sync output directory", parent);
    if (close(installed_fd) != 0)
        die_errno("close installed output", out_path);
    if (close(directory_fd) != 0)
        die_errno("close output directory", parent);
    dspark_active_destination_path = NULL;
    dspark_active_destination_identity = false;
    free(parent);

    if (!unlink_owned_path(temp_path, output_fd_stat.st_dev,
                           output_fd_stat.st_ino))
        fprintf(stderr,
                "warning: temporary pathname was replaced; left intact: %s\n",
                temp_path);
    dspark_active_temp_path = NULL;
    dspark_active_temp_identity = false;
    if (close(temp_fd) != 0) die_errno("close output descriptor", out_path);
    free(temp_path);
    printf("dspark_support_revision: %s\n", DS4_DSPARK_SOURCE_REVISION);
    printf("dspark_support_bytes: %" PRIu64 "\n",
           (uint64_t)output_stat.st_size);
    printf("dspark_support_sha256: %s\n", output_sha256);
    printf("dspark_support_status: unqualified-authenticated-composer-input\n");
}

static void print_plan(const gguf_file *tmpl, const output_context *out_ctx) {
    size_t tensor_bytes = 0;
    size_t changed = 0;
    for (uint64_t i = 0; i < out_ctx->n_tensors; i++) {
        tensor_bytes += out_ctx->tensors[i].size;
        const tensor_meta *src = &tmpl->tensors[i];
        const tensor_meta *dst = &out_ctx->tensors[i];
        if (src->type != dst->type) {
            changed++;
            printf("type_change: %s %s -> %s\n", dst->name, ds4q_type_name(src->type), ds4q_type_name(dst->type));
        }
    }
    printf("n_tensors: %" PRIu64 "\n", out_ctx->n_tensors);
    printf("meta_bytes: %zu\n", out_ctx->data_offset);
    printf("tensor_bytes_unpadded: %zu\n", tensor_bytes);
    printf("approx_file_bytes: %zu\n", out_ctx->data_offset + out_ctx->tensor_bytes);
    printf("type_changes: %zu\n", changed);
}

static void print_dspark_inventory(const output_context *out_ctx) {
    for (uint64_t i = 0; i < out_ctx->n_tensors; i++) {
        const tensor_meta *tensor = &out_ctx->tensors[i];
        printf("dspark_tensor: %s type=%u dims=", tensor->name,
               (unsigned)tensor->type);
        for (int dim = 0; dim < tensor->n_dims; dim++)
            printf("%s%" PRId64, dim ? "x" : "", tensor->ne[dim]);
        printf("\n");
    }
}

/* =====
 * CLI
 */

typedef struct {
    char *hf_dir;
    char *template_gguf;
    char *out_gguf;
    char *compare_gguf;
    char *compare_tensor;
    char *imatrix_file;
    char *reuse_gguf;
    quant_policy policy;
    int n_experts;
    int n_threads;
    bool dspark_support_only;
    bool check_only;
    bool dry_run;
    bool imatrix_strict;
} params;

static void usage(const char *argv0) {
    printf("usage: %s --hf DIR --template MODEL.gguf --out OUT.gguf [options]\n", argv0);
    printf("       %s --dspark-support-only --hf DIR "
           "[--out SUPPORT.gguf|--check|--dry-run]\n", argv0);
    printf("\nDeepSeek V4 Flash/Pro safetensors -> GGUF quantizer in plain C.\n\n");
    printf("options:\n");
    printf("  --hf DIR               Hugging Face model directory with model.safetensors.index.json\n");
    printf("  --dspark-support-only  build the authenticated final-0731 81-tensor\n");
    printf("                         DSpark composer input from shards 46-48 only\n");
    printf("  --check                authenticate final DSpark config/index/shards and exit\n");
    printf("  --template FILE        existing DS4 GGUF used for metadata, tensor order, shapes\n");
    printf("  --out FILE             output GGUF path\n");
    printf("  --compare-gguf FILE    reference GGUF for --compare-tensor, default template\n");
    printf("  --compare-tensor NAME  regenerate one tensor, byte-compare, and exit\n");
    printf("  --dry-run              print output plan without reading HF tensor data\n");
    printf("  --imatrix FILE         legacy .dat imatrix from ds4 --imatrix-out\n");
    printf("  --imatrix-strict       fail if a quantized tensor has no matching imatrix vector\n");
    printf("  --reuse PRIOR.gguf     copy byte-identical tensors from a prior build (same --hf +\n");
    printf("                         --imatrix) instead of regenerating; only changed tensors\n");
    printf("                         (e.g. boosted layers at a new type) are quantized\n");
    printf("  --experts TYPE         set routed w1/w2/w3 expert tensors to TYPE\n");
    printf("  --routed-w1 TYPE       routed gate expert tensor type\n");
    printf("  --routed-w2 TYPE       routed down expert tensor type\n");
    printf("  --routed-w3 TYPE       routed up expert tensor type\n");
    printf("  --attention-proj TYPE  attn_q/kv/output projection type\n");
    printf("  --attention TYPE       other 2D attention/indexer/compressor type\n");
    printf("  --shared TYPE          shared expert tensor type\n");
    printf("  --embedding TYPE       token embedding type\n");
    printf("  --output TYPE          output.* tensor type\n");
    printf("  --dense TYPE           remaining 2D+ non-routed tensor type\n");
    printf("  --tensor-type PFX=TYPE exact tensor-name or prefix override; may repeat\n");
    printf("  --n-experts N          routed expert count, default template metadata\n");
    printf("  --threads N            expert worker count, default 8\n");
    printf("\nTYPE examples: f16, f32, bf16, q8_0, q4_k, q2_k, iq2_xxs\n");
}

static char *need_value(int argc, char **argv, int *i, const char *arg) {
    if (++*i >= argc) {
        fprintf(stderr, "error: missing value for %s\n", arg);
        exit(1);
    }
    return argv[*i];
}

static bool file_exists(const char *path) {
    struct stat st;
    if (lstat(path, &st) == 0) return true;
    if (errno == ENOENT) return false;
    die_errno("inspect output", path);
    return false;
}

static void reject_dspark_output_alias(const char *hf_dir,
                                       const char *out_path) {
    const char *const inputs[] = {
        "config.json",
        "model.safetensors.index.json",
        "model-00046-of-00048.safetensors",
        "model-00047-of-00048.safetensors",
        "model-00048-of-00048.safetensors",
    };
    struct stat output_stat;
    const bool output_exists = stat(out_path, &output_stat) == 0;
    for (size_t i = 0; i < sizeof(inputs) / sizeof(inputs[0]); i++) {
        char *input_path = path_join(hf_dir, inputs[i]);
        struct stat input_stat;
        const bool same_inode = output_exists && stat(input_path, &input_stat) == 0 &&
            input_stat.st_dev == output_stat.st_dev &&
            input_stat.st_ino == output_stat.st_ino;
        if (strcmp(input_path, out_path) == 0 || same_inode) {
            fprintf(stderr, "error: DSpark output aliases authenticated input: %s\n",
                    input_path);
            free(input_path);
            exit(1);
        }
        free(input_path);
    }
}

static params parse_args(int argc, char **argv) {
    params p = {0};
    p.policy.routed_w1 = p.policy.routed_w2 = p.policy.routed_w3 = DS4Q_TYPE_COUNT;
    p.policy.attention_proj = p.policy.attention = p.policy.shared = DS4Q_TYPE_COUNT;
    p.policy.embedding = p.policy.output = p.policy.dense = DS4Q_TYPE_COUNT;
    p.n_experts = 0;
    p.n_threads = 8;

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            usage(argv[0]);
            exit(0);
        } else if (strcmp(arg, "--hf") == 0) {
            p.hf_dir = need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--dspark-support-only") == 0) {
            p.dspark_support_only = true;
        } else if (strcmp(arg, "--check") == 0) {
            p.check_only = true;
        } else if (strcmp(arg, "--template") == 0) {
            p.template_gguf = need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--out") == 0) {
            p.out_gguf = need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--compare-gguf") == 0) {
            p.compare_gguf = need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--compare-tensor") == 0) {
            p.compare_tensor = need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--dry-run") == 0) {
            p.dry_run = true;
        } else if (strcmp(arg, "--imatrix") == 0) {
            p.imatrix_file = need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--reuse") == 0) {
            p.reuse_gguf = need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--imatrix-strict") == 0) {
            p.imatrix_strict = true;
        } else if (strcmp(arg, "--experts") == 0 || strcmp(arg, "--routed") == 0) {
            ds4q_type t = parse_type(need_value(argc, argv, &i, arg));
            p.policy.routed_w1 = p.policy.routed_w2 = p.policy.routed_w3 = t;
        } else if (strcmp(arg, "--routed-w1") == 0 || strcmp(arg, "--routed-gate") == 0) {
            p.policy.routed_w1 = parse_type(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--routed-w2") == 0 || strcmp(arg, "--routed-down") == 0) {
            p.policy.routed_w2 = parse_type(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--routed-w3") == 0 || strcmp(arg, "--routed-up") == 0) {
            p.policy.routed_w3 = parse_type(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--attention-proj") == 0 || strcmp(arg, "--attn-proj") == 0) {
            p.policy.attention_proj = parse_type(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--attention") == 0) {
            p.policy.attention = parse_type(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--shared") == 0) {
            p.policy.shared = parse_type(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--embedding") == 0) {
            p.policy.embedding = parse_type(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--output") == 0) {
            p.policy.output = parse_type(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--dense") == 0) {
            p.policy.dense = parse_type(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--tensor-type") == 0) {
            char *spec = need_value(argc, argv, &i, arg);
            char *eq = strchr(spec, '=');
            if (!eq || eq == spec || !eq[1]) die("bad --tensor-type, expected NAME=TYPE");
            *eq = '\0';
            p.policy.overrides = xrealloc(p.policy.overrides, (size_t)(p.policy.n_overrides + 1) * sizeof(p.policy.overrides[0]));
            p.policy.overrides[p.policy.n_overrides++] = (type_override){ xstrdup(spec), parse_type(eq + 1) };
        } else if (strcmp(arg, "--n-experts") == 0) {
            p.n_experts = atoi(need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--threads") == 0) {
            p.n_threads = atoi(need_value(argc, argv, &i, arg));
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg);
            exit(1);
        }
    }
    if (!p.hf_dir) die("--hf is required");
    if (p.dspark_support_only) {
        if (p.template_gguf || p.compare_gguf || p.compare_tensor ||
            p.imatrix_file || p.reuse_gguf || p.n_experts != 0 ||
            p.policy.n_overrides != 0 ||
            p.policy.routed_w1 != DS4Q_TYPE_COUNT ||
            p.policy.routed_w2 != DS4Q_TYPE_COUNT ||
            p.policy.routed_w3 != DS4Q_TYPE_COUNT ||
            p.policy.attention_proj != DS4Q_TYPE_COUNT ||
            p.policy.attention != DS4Q_TYPE_COUNT ||
            p.policy.shared != DS4Q_TYPE_COUNT ||
            p.policy.embedding != DS4Q_TYPE_COUNT ||
            p.policy.output != DS4Q_TYPE_COUNT ||
            p.policy.dense != DS4Q_TYPE_COUNT)
            die("--dspark-support-only has a fixed inventory and quantization recipe");
        if (p.dry_run && p.check_only)
            die("--dry-run and --check are mutually exclusive");
        if ((p.dry_run || p.check_only) && p.out_gguf)
            die("--out is not used with DSpark --dry-run or --check");
        if (!p.dry_run && !p.check_only && !p.out_gguf)
            die("--out is required for DSpark support generation");
    } else {
        if (p.check_only) die("--check requires --dspark-support-only");
        if (!p.template_gguf) die("--template is required");
        if (!p.dry_run && !p.compare_tensor && !p.out_gguf)
            die("--out is required unless --dry-run or --compare-tensor is used");
    }
    if (p.compare_tensor && !p.compare_gguf) p.compare_gguf = p.template_gguf;
    if (p.out_gguf && file_exists(p.out_gguf)) die("output exists");
    return p;
}

static void free_gguf_file(gguf_file *g) {
    free(g->path);
    free(g->kv_raw);
    free(g->reuse_key);
    free(g->reuse_key_weights);
    free(g->reuse_imatrix_coverage);
    for (uint64_t i = 0; i < g->n_tensors; i++) free(g->tensors[i].name);
    free(g->tensors);
    hmap_free(&g->tensor_map);
    memset(g, 0, sizeof(*g));
}

static void compare_one_tensor(st_db *db, const gguf_file *tmpl, const output_context *out_ctx,
                               const params *p, const imatrix_store *imatrix) {
    int idx = hmap_get(&tmpl->tensor_map, p->compare_tensor);
    if (idx < 0) {
        fprintf(stderr, "error: tensor not found in template: %s\n", p->compare_tensor);
        exit(1);
    }
    fprintf(stderr, "regenerating %s as %s\n",
            p->compare_tensor, ds4q_type_name(out_ctx->tensors[idx].type));
    byte_buf generated = generate_tensor(db, p->compare_tensor, &tmpl->tensors[idx],
                                         out_ctx->tensors[idx].type, p->n_experts, p->n_threads, imatrix);
    gguf_file ref = load_gguf_metadata(p->compare_gguf);
    byte_buf reference = read_gguf_tensor_data(&ref, p->compare_gguf, p->compare_tensor);
    printf("tensor: %s\n", p->compare_tensor);
    printf("type: %s\n", ds4q_type_name(out_ctx->tensors[idx].type));
    printf("generated_bytes: %zu\n", generated.size);
    printf("reference_bytes: %zu\n", reference.size);
    printf("generated_fnv1a64: %016" PRIx64 "\n", fnv1a64_bytes(generated.data, generated.size));
    printf("reference_fnv1a64: %016" PRIx64 "\n", fnv1a64_bytes(reference.data, reference.size));
    size_t mismatches = 0;
    size_t first = SIZE_MAX;
    const size_t n = generated.size < reference.size ? generated.size : reference.size;
    for (size_t i = 0; i < n; i++) {
        if (generated.data[i] != reference.data[i]) {
            if (first == SIZE_MAX) first = i;
            mismatches++;
        }
    }
    if (generated.size != reference.size) {
        if (first == SIZE_MAX) first = n;
        mismatches += generated.size > reference.size ? generated.size - reference.size : reference.size - generated.size;
    }
    if (!mismatches) {
        printf("byte_compare: OK\n");
    } else {
        printf("byte_compare: FAIL mismatches=%zu first=%zu\n", mismatches, first);
    }
    free(generated.data);
    free(reference.data);
    free_gguf_file(&ref);
}

int main(int argc, char **argv) {
    params p = parse_args(argc, argv);
    if (p.dspark_support_only) {
        gguf_file support = make_dspark_support_template();
        output_context support_out = build_dspark_output_context(&support);
        print_plan(&support, &support_out);
        printf("dspark_source_revision: %s\n", DS4_DSPARK_SOURCE_REVISION);
        printf("dspark_source_tensors: %d\n", DS4_DSPARK_SOURCE_TENSORS);
        printf("dspark_output_tensors: %d\n", DS4_DSPARK_OUTPUT_TENSORS);
        printf("dspark_source_shards: 46,47,48\n");
        if (p.dry_run) print_dspark_inventory(&support_out);
        if (p.out_gguf) reject_dspark_output_alias(p.hf_dir, p.out_gguf);
        dspark_source_files source_files =
            open_dspark_source_files(p.hf_dir, !p.dry_run);
        st_db support_db;
        db_open_with_files(
            &support_db, p.hf_dir, source_files.fds[1],
            source_files.include_shards ? source_files.borrowed_shards : NULL,
            source_files.include_shards ? DS4_DSPARK_STAGE_COUNT : 0
        );
        validate_dspark_source_index(&support_db);
        if (p.dry_run) {
            require_dspark_source_fds_unchanged(&source_files);
            printf("dspark_dry_run: OK (config/index authenticated; shards unread)\n");
        } else if (p.check_only) {
            require_dspark_source_fds_unchanged(&source_files);
            printf("dspark_source_check: OK\n");
        } else {
            write_dspark_support_gguf(
                &support_db, &support, &support_out,
                &source_files,
                p.out_gguf, p.n_threads
            );
        }
        db_close(&support_db);
        close_dspark_source_files(&source_files);
        free(support_out.tensors);
        free_gguf_file(&support);
        return 0;
    }
    imatrix_store imatrix = {0};
    if (p.imatrix_file) imatrix_load(&imatrix, p.imatrix_file, p.imatrix_strict);

    gguf_file tmpl = load_gguf_metadata(p.template_gguf);
    if (p.n_experts <= 0) {
        if (tmpl.n_experts > 0) {
            p.n_experts = tmpl.n_experts;
            fprintf(stderr, "using %d routed experts from template metadata\n", p.n_experts);
        } else {
            p.n_experts = 256;
            fprintf(stderr, "warning: template has no deepseek4.expert_count; using Flash default %d routed experts\n", p.n_experts);
        }
    } else {
        fprintf(stderr, "using %d routed experts from --n-experts\n", p.n_experts);
    }
    output_context out_ctx = build_output_context(&tmpl, &p.policy, &imatrix);
    print_plan(&tmpl, &out_ctx);
    if (p.dry_run) return 0;

    /* --reuse must not alias --out: the output is opened "wb" (truncated) before the prior's
     * tensors are read, so reusing from the output file would read a zeroed file. Fail fast. */
    if (p.reuse_gguf && p.out_gguf) {
        struct stat ra, rb;
        bool same = (stat(p.reuse_gguf, &ra) == 0 && stat(p.out_gguf, &rb) == 0 &&
                     ra.st_dev == rb.st_dev && ra.st_ino == rb.st_ino);
        if (same || strcmp(p.reuse_gguf, p.out_gguf) == 0) die("--reuse and --out must not be the same file");
    }

    /* This build's reuse identity (stamped into the output as quantize.reuse_key). Resolved
     * before the slow db_open so a missing/mismatched --reuse prior is reported fast. Does
     * not need the FP weights — only the safetensors index, the imatrix, and the template. */
    char *reuse_key = compute_reuse_key(p.hf_dir, p.imatrix_file, &tmpl);
    /* the imatrix-independent half: same hash with no imatrix folded in */
    char *reuse_key_weights = compute_reuse_key(p.hf_dir, NULL, &tmpl);
    char *reuse_imatrix_coverage = compute_imatrix_coverage_key(&imatrix, &out_ctx);
    gguf_file prior = {0};
    const gguf_file *prior_use = NULL;
    bool reuse_weights_only = false;
    if (p.reuse_gguf) {
        prior = load_gguf_metadata(p.reuse_gguf);
        if (prior.reuse_key && strcmp(prior.reuse_key, reuse_key) == 0) {
            prior_use = &prior;
            fprintf(stderr, "reuse: %s matches this build (key %s) — copying unchanged tensors\n",
                    p.reuse_gguf, reuse_key);
        } else if (imatrix_enabled(&imatrix) && prior.reuse_key_weights &&
                   strcmp(prior.reuse_key_weights, reuse_key_weights) == 0 &&
                   prior.reuse_imatrix_coverage &&
                   strcmp(prior.reuse_imatrix_coverage, reuse_imatrix_coverage) == 0) {
            /* Same FP weights + template, different imatrix, and the two imatrices steer
             * the SAME regular-tensor set (coverage keys equal): only steered tensors
             * change; everything else is byte-identical and copyable. The current build
             * must itself have a live imatrix — without one this build's reuse_key equals
             * its weights key and copying a steered prior would poison that clean key. */
            prior_use = &prior;
            reuse_weights_only = true;
            fprintf(stderr, "reuse: %s shares the weights key (%s) and imatrix coverage (%s) "
                            "but not the imatrix — copying imatrix-independent tensors, "
                            "regenerating the steered ones\n",
                    p.reuse_gguf, reuse_key_weights, reuse_imatrix_coverage);
        } else {
            fprintf(stderr, "reuse: %s has key %s but this build is %s — regenerating all tensors\n",
                    p.reuse_gguf, prior.reuse_key ? prior.reuse_key : "(none)", reuse_key);
        }
    }

    st_db db;
    db_open(&db, p.hf_dir);
    if (p.compare_tensor) {
        compare_one_tensor(&db, &tmpl, &out_ctx, &p, &imatrix);
        if (p.reuse_gguf) free_gguf_file(&prior);
        free(reuse_key);
        free(reuse_key_weights);
        free(reuse_imatrix_coverage);
        db_close(&db);
        imatrix_free(&imatrix);
        free_gguf_file(&tmpl);
        free(out_ctx.tensors);
        return 0;
    }

    write_full_gguf(&db, &tmpl, &out_ctx, p.out_gguf, p.n_experts, p.n_threads, &imatrix,
                    reuse_key, reuse_key_weights, reuse_imatrix_coverage, reuse_weights_only,
                    prior_use);
    fprintf(stderr, "wrote %s\n", p.out_gguf);

    if (p.reuse_gguf) free_gguf_file(&prior);
    free(reuse_key);
    free(reuse_key_weights);
    free(reuse_imatrix_coverage);
    db_close(&db);
    imatrix_free(&imatrix);
    free_gguf_file(&tmpl);
    free(out_ctx.tensors);
    for (int i = 0; i < p.policy.n_overrides; i++) free(p.policy.overrides[i].prefix);
    free(p.policy.overrides);
    return 0;
}
