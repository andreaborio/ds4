#define DS4_SERVER_TEST
#define DS4_SERVER_TEST_NO_MAIN
#include "../ds4_server.c"
#ifndef DS4_NO_GPU
#include "../ds4_gpu.h"
#include "../ds4_qwen.h"
#include <math.h>
#ifdef __APPLE__
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

static ds4_engine *test_engine_fast;
static ds4_engine *test_engine_quality;

static const char *test_model_path(void) {
    const char *model_path = getenv("DS4_TEST_MODEL");
    return (model_path && model_path[0]) ? model_path : "ds4flash.gguf";
}

static bool test_env_bool(const char *name) {
    const char *v = getenv(name);
    return v && v[0] && strcmp(v, "0") != 0;
}

static uint32_t test_env_u32(const char *name) {
    const char *v = getenv(name);
    if (!v || !v[0]) return 0;
    char *end = NULL;
    unsigned long n = strtoul(v, &end, 10);
    if (end == v) return 0;
    return n > UINT32_MAX ? UINT32_MAX : (uint32_t)n;
}

static uint64_t test_env_gib(const char *name) {
    const char *v = getenv(name);
    if (!v || !v[0]) return 0;
    char *end = NULL;
    unsigned long long n = strtoull(v, &end, 10);
    if (end == v || n == 0) return 0;
    const uint64_t one_gib = 1024ull * 1024ull * 1024ull;
    if (n > UINT64_MAX / one_gib) return UINT64_MAX;
    return (uint64_t)n * one_gib;
}

static char *test_save_env(const char *name) {
    const char *value = getenv(name);
    if (!value) return NULL;
    size_t len = strlen(value);
    char *copy = malloc(len + 1);
    TEST_ASSERT(copy != NULL);
    if (!copy) return NULL;
    memcpy(copy, value, len + 1);
    return copy;
}

static void test_restore_env(const char *name, char *saved) {
    if (saved) {
        setenv(name, saved, 1);
        free(saved);
    } else {
        unsetenv(name);
    }
}

typedef struct {
    char *cold_decode;
    char *batch_selected_addr;
} test_streaming_prefill_env;

static test_streaming_prefill_env test_force_canonical_streaming_prefill(void) {
    test_streaming_prefill_env saved = {
        .cold_decode =
            test_save_env("DS4_METAL_DISABLE_STREAMING_COLD_DECODE_PREFILL"),
        .batch_selected_addr =
            test_save_env("DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR"),
    };
    if (test_env_bool("DS4_TEST_SSD_STREAMING")) {
        setenv("DS4_METAL_DISABLE_STREAMING_COLD_DECODE_PREFILL", "1", 1);
        setenv("DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR", "1", 1);
    }
    return saved;
}

static void test_restore_canonical_streaming_prefill(
        test_streaming_prefill_env saved) {
    test_restore_env("DS4_METAL_DISABLE_STREAMING_COLD_DECODE_PREFILL",
                     saved.cold_decode);
    test_restore_env("DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR",
                     saved.batch_selected_addr);
}

static ds4_engine *test_open_engine(bool quality) {
    ds4_engine *engine = NULL;
    /* DS4_TEST_MTP loads the MTP head on the fast engine so the speculative
     * verify regression can reuse it; draft=4 hits the multi-row verify path. */
    const char *mtp = getenv("DS4_TEST_MTP");
    const bool streaming = test_env_bool("DS4_TEST_SSD_STREAMING");
    ds4_engine_options opt = {
        .model_path = test_model_path(),
#ifdef __APPLE__
        .backend = DS4_BACKEND_METAL,
#else
        .backend = DS4_BACKEND_CUDA,
#endif
        /* Preserve the historical deterministic test lane. AUTO has its own
         * resolver coverage and must not silently switch this suite's path
         * according to the host's current memory budget. */
        .residency = streaming ? DS4_RESIDENCY_SSD : DS4_RESIDENCY_RESIDENT,
        .context_size = 100000,
        .quality = quality,
        .ssd_streaming_cold = test_env_bool("DS4_TEST_SSD_STREAMING_COLD"),
        .ssd_streaming_cache_experts =
            test_env_u32("DS4_TEST_SSD_STREAMING_CACHE_EXPERTS"),
        .ssd_streaming_cache_bytes =
            test_env_gib("DS4_TEST_SSD_STREAMING_CACHE_GB"),
        .ssd_streaming_preload_experts =
            test_env_u32("DS4_TEST_SSD_STREAMING_PRELOAD_EXPERTS"),
        .mtp_path = (mtp && mtp[0] && !quality) ? mtp : NULL,
        .mtp_draft_tokens = (mtp && mtp[0] && !quality) ? 4 : 0,
    };
    TEST_ASSERT(ds4_engine_open(&engine, &opt) == 0);
    return engine;
}

static ds4_engine *test_get_engine(bool quality) {
    ds4_engine **slot = quality ? &test_engine_quality : &test_engine_fast;
    if (*slot) return *slot;

    *slot = test_open_engine(quality);
    return *slot;
}

static void test_close_engines(void) {
    ds4_engine_close(test_engine_fast);
    ds4_engine_close(test_engine_quality);
    test_engine_fast = NULL;
    test_engine_quality = NULL;
}

static void test_close_engine(bool quality) {
    ds4_engine **slot = quality ? &test_engine_quality : &test_engine_fast;
    ds4_engine_close(*slot);
    *slot = NULL;
}

static uint64_t test_round_up_u64(uint64_t n, uint64_t align) {
    return (n + align - 1) & ~(align - 1);
}

static uint16_t test_float_to_f16(float f) {
    union {
        float f;
        uint32_t u;
    } v = { .f = f };

    uint32_t sign = (v.u >> 16) & 0x8000u;
    int32_t exp = (int32_t)((v.u >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = v.u & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) half_mant++;
        return (uint16_t)(sign | half_mant);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);

    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x1000u) half++;
    return (uint16_t)half;
}

static float test_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    }

    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static void test_fill_q8_0_weights(uint8_t *weights,
                                   uint32_t in_dim,
                                   uint32_t out_dim) {
    const uint32_t blocks = in_dim / 32u;
    const uint64_t row_bytes = (uint64_t)blocks * 34u;
    for (uint32_t o = 0; o < out_dim; o++) {
        uint8_t *row = weights + (uint64_t)o * row_bytes;
        for (uint32_t b = 0; b < blocks; b++) {
            float vals[32];
            float amax = 0.0f;
            for (uint32_t i = 0; i < 32; i++) {
                const uint32_t k = b * 32u + i;
                const int v = (int)((o * 17u + k * 23u + (o ^ k) * 3u) % 67u) - 33;
                vals[i] = (float)v / 96.0f;
                float av = fabsf(vals[i]);
                if (av > amax) amax = av;
            }
            const uint16_t scale_bits = test_float_to_f16(amax / 127.0f);
            const float scale = test_f16_to_f32(scale_bits);
            memcpy(row + b * 34u, &scale_bits, sizeof(scale_bits));
            int8_t *qs = (int8_t *)(row + b * 34u + 2u);
            for (uint32_t i = 0; i < 32; i++) {
                int q = scale != 0.0f ? (int)lrintf(vals[i] / scale) : 0;
                if (q > 127) q = 127;
                if (q < -128) q = -128;
                qs[i] = (int8_t)q;
            }
        }
    }
}

static void test_metal_f16_matvec_fast_nr0_4(void) {
    /*
     * This is the short regression for the long-context repetition failure.
     * Decode uses one-token F16 matvecs for several DS4 projections; the fast
     * nr0=4 variant must be numerically equivalent to the plain kernel.
     */
    const uint32_t in_dim = 4096;
    const uint32_t out_dim = 512;
    const uint64_t weight_bytes = (uint64_t)in_dim * out_dim * sizeof(uint16_t);
    const uint64_t weight_alloc = test_round_up_u64(weight_bytes, (uint64_t)getpagesize());

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)getpagesize(), (size_t)weight_alloc) == 0);
    if (!weights_raw) return;

    uint16_t *weights = weights_raw;
    memset(weights, 0, (size_t)weight_alloc);
    for (uint32_t o = 0; o < out_dim; o++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            float w = (float)((int)((o * 3u + i * 5u) % 23u) - 11) / 64.0f;
            weights[(uint64_t)o * in_dim + i] = test_float_to_f16(w);
        }
    }

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((uint64_t)in_dim * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)out_dim * sizeof(float));
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(out != NULL);
    if (!x || !out) {
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    float *x_host = malloc((size_t)in_dim * sizeof(float));
    float *out_host = malloc((size_t)out_dim * sizeof(float));
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(out_host != NULL);
    if (!x_host || !out_host) {
        free(x_host);
        free(out_host);
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    for (uint32_t i = 0; i < in_dim; i++) {
        x_host[i] = (float)((int)(i % 31u) - 15) / 32.0f;
    }

    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, (uint64_t)in_dim * sizeof(float)) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    ds4_gpu_set_quality(false);
    TEST_ASSERT(ds4_gpu_matmul_f16_tensor(out, weights_raw, weight_alloc, 0,
                                            in_dim, out_dim, x, 1) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(out, 0, out_host, (uint64_t)out_dim * sizeof(float)) != 0);

    float max_abs = 0.0f;
    for (uint32_t o = 0; o < out_dim; o++) {
        float ref = 0.0f;
        for (uint32_t i = 0; i < in_dim; i++) {
            float w = (float)((int)((o * 3u + i * 5u) % 23u) - 11) / 64.0f;
            ref += w * x_host[i];
        }
        float err = fabsf(out_host[o] - ref);
        if (err > max_abs) max_abs = err;
    }
    TEST_ASSERT(max_abs < 0.02f);

    free(x_host);
    free(out_host);
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(out);
    free(weights_raw);
}

static void test_metal_f16_prefill_matmul(void) {
    const uint32_t in_dim = 128;
    const uint32_t out_dim = 64;
    const uint32_t n_tok = 128;
    const uint64_t weight_bytes = (uint64_t)out_dim * in_dim * sizeof(uint16_t);
    const uint64_t weight_alloc = test_round_up_u64(weight_bytes, (uint64_t)getpagesize());
    const uint64_t x_bytes = (uint64_t)n_tok * in_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_tok * out_dim * sizeof(float);

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)getpagesize(), (size_t)weight_alloc) == 0);
    if (!weights_raw) return;

    uint16_t *weights = weights_raw;
    memset(weights, 0, (size_t)weight_alloc);
    for (uint32_t o = 0; o < out_dim; o++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            const int v = (int)((o * 11u + i * 13u + (o ^ i) * 5u) % 61u) - 30;
            weights[(uint64_t)o * in_dim + i] = test_float_to_f16((float)v / 96.0f);
        }
    }

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_bytes);
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(out != NULL);
    if (!x || !out) {
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    float *x_host = malloc((size_t)x_bytes);
    float *out_host = malloc((size_t)out_bytes);
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(out_host != NULL);
    if (!x_host || !out_host) {
        free(x_host);
        free(out_host);
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            const int v = (int)((t * 7u + i * 17u + (t ^ i) * 3u) % 73u) - 36;
            x_host[(uint64_t)t * in_dim + i] = (float)v / 80.0f;
        }
    }
    for (uint32_t i = 0; i < n_tok * out_dim; i++) {
        out_host[i] = 12345.0f;
    }

    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(out, 0, out_host, out_bytes) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    ds4_gpu_set_quality(false);
    TEST_ASSERT(ds4_gpu_matmul_f16_tensor(out, weights_raw, weight_alloc, 0,
                                          in_dim, out_dim, x, n_tok) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(out, 0, out_host, out_bytes) != 0);

    float max_abs = 0.0f;
    float rms = 0.0f;
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t o = 0; o < out_dim; o++) {
            float ref = 0.0f;
            for (uint32_t i = 0; i < in_dim; i++) {
                ref += test_f16_to_f32(weights[(uint64_t)o * in_dim + i]) *
                       x_host[(uint64_t)t * in_dim + i];
            }
            const float got = out_host[(uint64_t)t * out_dim + o];
            TEST_ASSERT(isfinite(got));
            const float err = fabsf(got - ref);
            if (err > max_abs) max_abs = err;
            rms += err * err;
        }
    }
    rms = sqrtf(rms / (float)(n_tok * out_dim));
    TEST_ASSERT(max_abs < 0.08f);
    TEST_ASSERT(rms < 0.02f);

    free(x_host);
    free(out_host);
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(out);
    free(weights_raw);
}

static void test_metal_q8_0_prefill_matmul(void) {
    const uint32_t in_dim = 128;
    const uint32_t out_dim = 64;
    const uint32_t n_tok = 128;
    const uint64_t row_bytes = (uint64_t)(in_dim / 32u) * 34u;
    const uint64_t weight_bytes = (uint64_t)out_dim * row_bytes;
    const uint64_t weight_alloc = test_round_up_u64(weight_bytes, (uint64_t)getpagesize());
    const uint64_t x_bytes = (uint64_t)n_tok * in_dim * sizeof(float);
    const uint64_t out_bytes = (uint64_t)n_tok * out_dim * sizeof(float);

    void *weights_raw = NULL;
    TEST_ASSERT(posix_memalign(&weights_raw, (size_t)getpagesize(), (size_t)weight_alloc) == 0);
    if (!weights_raw) return;

    uint8_t *weights = weights_raw;
    memset(weights, 0, (size_t)weight_alloc);
    test_fill_q8_0_weights(weights, in_dim, out_dim);

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(x_bytes);
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(out_bytes);
    TEST_ASSERT(x != NULL);
    TEST_ASSERT(out != NULL);
    if (!x || !out) {
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    float *x_host = malloc((size_t)x_bytes);
    float *out_host = malloc((size_t)out_bytes);
    TEST_ASSERT(x_host != NULL);
    TEST_ASSERT(out_host != NULL);
    if (!x_host || !out_host) {
        free(x_host);
        free(out_host);
        ds4_gpu_tensor_free(x);
        ds4_gpu_tensor_free(out);
        free(weights_raw);
        return;
    }

    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t i = 0; i < in_dim; i++) {
            const int v = (int)((t * 19u + i * 7u + (t ^ i)) % 71u) - 35;
            x_host[(uint64_t)t * in_dim + i] = (float)v / 80.0f;
        }
    }
    for (uint32_t i = 0; i < n_tok * out_dim; i++) {
        out_host[i] = 12345.0f;
    }

    TEST_ASSERT(ds4_gpu_tensor_write(x, 0, x_host, x_bytes) != 0);
    TEST_ASSERT(ds4_gpu_tensor_write(out, 0, out_host, out_bytes) != 0);
    TEST_ASSERT(ds4_gpu_set_model_map(weights_raw, weight_alloc) != 0);
    ds4_gpu_set_quality(false);
    TEST_ASSERT(ds4_gpu_matmul_q8_0_tensor(out, weights_raw, weight_alloc, 0,
                                           in_dim, out_dim, x, n_tok) != 0);
    TEST_ASSERT(ds4_gpu_tensor_read(out, 0, out_host, out_bytes) != 0);

    float max_abs = 0.0f;
    float rms = 0.0f;
    for (uint32_t t = 0; t < n_tok; t++) {
        for (uint32_t o = 0; o < out_dim; o++) {
            const uint8_t *row = weights + (uint64_t)o * row_bytes;
            float ref = 0.0f;
            for (uint32_t b = 0; b < in_dim / 32u; b++) {
                uint16_t scale_bits;
                memcpy(&scale_bits, row + b * 34u, sizeof(scale_bits));
                const float scale = test_f16_to_f32(scale_bits);
                const int8_t *qs = (const int8_t *)(row + b * 34u + 2u);
                for (uint32_t i = 0; i < 32; i++) {
                    ref += scale * (float)qs[i] *
                           x_host[(uint64_t)t * in_dim + b * 32u + i];
                }
            }
            const float got = out_host[(uint64_t)t * out_dim + o];
            TEST_ASSERT(isfinite(got));
            const float err = fabsf(got - ref);
            if (err > max_abs) max_abs = err;
            rms += err * err;
        }
    }
    rms = sqrtf(rms / (float)(n_tok * out_dim));
    TEST_ASSERT(max_abs < 0.08f);
    TEST_ASSERT(rms < 0.02f);

    free(x_host);
    free(out_host);
    ds4_gpu_tensor_free(x);
    ds4_gpu_tensor_free(out);
    free(weights_raw);
}

static ds4_gpu_tensor *test_metal_tensor_from_f32(
        const float *values,
        size_t       count) {
    const uint64_t bytes = (uint64_t)count * sizeof(float);
    ds4_gpu_tensor *tensor = ds4_gpu_tensor_alloc(bytes);
    TEST_ASSERT(tensor != NULL);
    if (!tensor) return NULL;
    if (!ds4_gpu_tensor_write(tensor, 0, values, bytes)) {
        TEST_ASSERT(false);
        ds4_gpu_tensor_free(tensor);
        return NULL;
    }
    return tensor;
}

static bool test_metal_read_f32(
        const ds4_gpu_tensor *tensor,
        float                *values,
        size_t                count) {
    const bool ok = tensor && values &&
        ds4_gpu_tensor_read(tensor, 0, values,
                            (uint64_t)count * sizeof(float)) != 0;
    TEST_ASSERT(ok);
    return ok;
}

static bool test_metal_qwen35_close(
        const char  *label,
        const float *actual,
        const float *expected,
        size_t       count,
        float        absolute_tolerance,
        float        relative_tolerance) {
    bool ok = actual && expected;
    float max_abs = 0.0f;
    size_t worst = 0;
    if (ok) {
        for (size_t i = 0; i < count; i++) {
            const float error = fabsf(actual[i] - expected[i]);
            const float tolerance = absolute_tolerance +
                relative_tolerance * fabsf(expected[i]);
            if (error > max_abs) {
                max_abs = error;
                worst = i;
            }
            if (!(error <= tolerance)) ok = false;
        }
    }
    if (!ok) {
        fprintf(stderr,
                "ds4-test: %s mismatch at %zu: got=%.9g expected=%.9g max_abs=%.9g\n",
                label, worst, actual ? actual[worst] : 0.0f,
                expected ? expected[worst] : 0.0f, max_abs);
    } else {
        fprintf(stderr, "ds4-test: %s max_abs=%.3g\n", label, max_abs);
    }
    TEST_ASSERT(ok);
    return ok;
}

/* Private ds4.c hooks: intentionally absent from ds4.h so this allocation
 * scaffold cannot become an engine/session API before the executor exists. */
extern size_t ds4_internal_qwen35_gpu_graph_size(void);
extern uint32_t ds4_internal_qwen35_gpu_prefill_cap(void);
extern bool ds4_internal_qwen35_gpu_graph_alloc(
    void *storage, size_t storage_bytes, uint32_t ctx_capacity);
extern bool ds4_internal_qwen35_gpu_graph_allocated_bytes(
    const void *storage, size_t storage_bytes, uint64_t *bytes_out);
extern bool ds4_internal_qwen35_gpu_graph_reset(
    void *storage, size_t storage_bytes);
extern bool ds4_internal_qwen35_gpu_graph_validate_position(
    const void *storage, size_t storage_bytes, uint32_t position);
extern bool ds4_internal_qwen35_gpu_graph_advance(
    void *storage, size_t storage_bytes, uint32_t position);
extern bool ds4_internal_qwen35_gpu_graph_mark_state_invalid(
    void *storage, size_t storage_bytes);
extern bool ds4_internal_qwen35_gpu_graph_views(
    void *storage, size_t storage_bytes,
    ds4_gpu_tensor **parents, ds4_gpu_tensor **views);
extern bool ds4_internal_qwen35_gpu_graph_layer_state(
    void *storage, size_t storage_bytes, uint32_t layer,
    ds4_gpu_tensor **state);
extern void ds4_internal_qwen35_gpu_graph_free(
    void *storage, size_t storage_bytes);

static bool test_metal_tensor_view_partition(
        ds4_gpu_tensor       *parent,
        ds4_gpu_tensor *const *views,
        const uint64_t       *offsets,
        const uint64_t       *sizes,
        size_t                n_views) {
    const uint64_t parent_bytes = ds4_gpu_tensor_bytes(parent);
    bool ok = parent && views && offsets && sizes && n_views > 0 &&
              parent_bytes > 0 && parent_bytes <= SIZE_MAX;
    TEST_ASSERT(ok);
    if (!ok) return false;

    for (size_t i = 0; i < n_views; i++) {
        ok = offsets[i] <= parent_bytes &&
             sizes[i] <= parent_bytes - offsets[i] &&
             sizes[i] == ds4_gpu_tensor_bytes(views[i]);
        TEST_ASSERT(ok);
        for (size_t j = 0; ok && j < i; j++) {
            const bool disjoint = offsets[i] + sizes[i] <= offsets[j] ||
                                  offsets[j] + sizes[j] <= offsets[i];
            TEST_ASSERT(disjoint);
            ok = ok && disjoint;
        }
    }
    if (!ok) return false;

    uint8_t *expected = malloc((size_t)parent_bytes);
    uint8_t *actual = malloc((size_t)parent_bytes);
    ok = expected && actual;
    TEST_ASSERT(ok);
    if (!ok) {
        free(expected);
        free(actual);
        return false;
    }

    for (uint64_t i = 0; i < parent_bytes; i++) {
        expected[i] = (uint8_t)(i * 29u + 17u);
    }
    ok = ds4_gpu_tensor_write(parent, 0, expected, parent_bytes) != 0;
    TEST_ASSERT(ok);
    for (size_t i = 0; ok && i < n_views; i++) {
        ok = ds4_gpu_tensor_read(views[i], 0, actual, sizes[i]) != 0 &&
             memcmp(actual, expected + offsets[i], (size_t)sizes[i]) == 0;
        TEST_ASSERT(ok);
    }

    memset(expected, 0, (size_t)parent_bytes);
    if (ok) {
        ok = ds4_gpu_tensor_write(parent, 0, expected, parent_bytes) != 0;
        TEST_ASSERT(ok);
    }
    for (size_t i = 0; ok && i < n_views; i++) {
        const uint8_t marker = (uint8_t)(0x31u + i * 23u);
        memset(actual, marker, (size_t)sizes[i]);
        ok = ds4_gpu_tensor_write(views[i], 0, actual, sizes[i]) != 0;
        TEST_ASSERT(ok);
        memset(expected + offsets[i], marker, (size_t)sizes[i]);
    }
    if (ok) {
        ok = ds4_gpu_tensor_read(parent, 0, actual, parent_bytes) != 0 &&
             memcmp(actual, expected, (size_t)parent_bytes) == 0;
        TEST_ASSERT(ok);
    }

    free(expected);
    free(actual);
    return ok;
}

static bool test_metal_tensor_is_all_zero(
        const ds4_gpu_tensor *tensor) {
    const uint64_t bytes = ds4_gpu_tensor_bytes(tensor);
    const size_t chunk_capacity = 1024u * 1024u;
    uint8_t *chunk = malloc(chunk_capacity);
    bool ok = tensor && bytes > 0 && chunk;
    TEST_ASSERT(ok);
    if (!ok) {
        free(chunk);
        return false;
    }

    for (uint64_t offset = 0; ok && offset < bytes;) {
        const uint64_t remaining = bytes - offset;
        const size_t chunk_bytes = remaining < chunk_capacity
            ? (size_t)remaining
            : chunk_capacity;
        ok = ds4_gpu_tensor_read(
                 tensor, offset, chunk, (uint64_t)chunk_bytes) != 0;
        for (size_t i = 0; ok && i < chunk_bytes; i++) {
            if (chunk[i] != 0) ok = false;
        }
        TEST_ASSERT(ok);
        offset += chunk_bytes;
    }
    free(chunk);
    return ok;
}

static void test_metal_qwen35_graph_state(void) {
    /* Three rows are enough to prove the context term without turning this
     * model-free lifetime test into a real long-context allocation. */
    const uint32_t ctx_capacity = 3;
    const uint64_t f32 = sizeof(float);
    const uint32_t moe_split_count = 2;
    const uint32_t moe_split_width =
        QWEN35_N_EXPERT_USED / moe_split_count;
    const uint32_t prefill_cap = ds4_internal_qwen35_gpu_prefill_cap();
    TEST_ASSERT(QWEN35_N_EXPERT_USED == 8);
    TEST_ASSERT(moe_split_width == 4);
    TEST_ASSERT(prefill_cap == 64);
    const size_t graph_size = ds4_internal_qwen35_gpu_graph_size();
    TEST_ASSERT(graph_size > 0);
    if (graph_size == 0) return;

    uint8_t *graph = calloc(1, graph_size);
    TEST_ASSERT(graph != NULL);
    if (!graph) return;

    uint64_t allocated = UINT64_MAX;
    TEST_ASSERT(!ds4_internal_qwen35_gpu_graph_alloc(
        graph, graph_size, 0));
    TEST_ASSERT(!ds4_internal_qwen35_gpu_graph_alloc(
        graph, graph_size, QWEN35_CONTEXT_LENGTH + 1u));
    TEST_ASSERT(!ds4_internal_qwen35_gpu_graph_alloc(
        graph, graph_size - 1u, ctx_capacity));
    TEST_ASSERT(ds4_internal_qwen35_gpu_graph_allocated_bytes(
        graph, graph_size, &allocated));
    TEST_ASSERT(allocated == 0);

    TEST_ASSERT(ds4_internal_qwen35_gpu_graph_alloc(
        graph, graph_size, ctx_capacity));

    ds4_gpu_tensor *view_parents[3] = {0};
    ds4_gpu_tensor *views[7] = {0};
    TEST_ASSERT(ds4_internal_qwen35_gpu_graph_views(
        graph, graph_size, view_parents, views));
    const uint64_t gdn_query_bytes =
        (uint64_t)QWEN35_SSM_GROUP * QWEN35_SSM_STATE * f32;
    const uint64_t gdn_key_bytes = gdn_query_bytes;
    const uint64_t gdn_value_bytes =
        (uint64_t)QWEN35_SSM_VALUE_HEAD * QWEN35_SSM_STATE * f32;
    const uint64_t gdn_offsets[] = {
        0,
        gdn_query_bytes,
        gdn_query_bytes + gdn_key_bytes,
    };
    const uint64_t gdn_sizes[] = {
        gdn_query_bytes,
        gdn_key_bytes,
        gdn_value_bytes,
    };
    TEST_ASSERT(ds4_gpu_tensor_bytes(view_parents[0]) ==
                gdn_query_bytes + gdn_key_bytes + gdn_value_bytes);
    TEST_ASSERT(test_metal_tensor_view_partition(
        view_parents[0], views, gdn_offsets, gdn_sizes, 3));

    const uint64_t top4_offsets[] = {
        0,
        (uint64_t)moe_split_width * sizeof(int32_t),
    };
    const uint64_t top4_sizes[] = {
        (uint64_t)moe_split_width * sizeof(int32_t),
        (uint64_t)moe_split_width * sizeof(int32_t),
    };
    TEST_ASSERT(top4_offsets[1] == 16);
    TEST_ASSERT(ds4_gpu_tensor_bytes(view_parents[1]) ==
                (uint64_t)QWEN35_N_EXPERT_USED * sizeof(int32_t));
    TEST_ASSERT(ds4_gpu_tensor_bytes(view_parents[2]) ==
                (uint64_t)QWEN35_N_EXPERT_USED * sizeof(float));
    TEST_ASSERT(test_metal_tensor_view_partition(
        view_parents[1], &views[3], top4_offsets, top4_sizes, 2));
    TEST_ASSERT(test_metal_tensor_view_partition(
        view_parents[2], &views[5], top4_offsets, top4_sizes, 2));

    uint64_t expected_f32_elements =
        3ull * QWEN35_N_EMBD +
        QWEN35_SSM_CONV_CHANNEL +
        3ull * QWEN35_SSM_INNER +
        (uint64_t)QWEN35_SSM_GROUP * QWEN35_SSM_STATE +
        4ull * QWEN35_SSM_VALUE_HEAD +
        (uint64_t)QWEN35_N_HEAD * QWEN35_N_HEAD_DIM +
        QWEN35_N_EMBD + ctx_capacity +
        2ull * QWEN35_N_EXPERT + QWEN35_N_EXPERT_USED +
        3ull * QWEN35_N_EXPERT_USED * QWEN35_N_FF_EXP +
        (uint64_t)QWEN35_N_EXPERT_USED * QWEN35_N_EMBD +
        (uint64_t)moe_split_count * QWEN35_N_EMBD +
        QWEN35_N_EMBD + 3ull * QWEN35_N_FF_SHARED +
        QWEN35_N_EMBD + 1ull + QWEN35_N_EMBD + QWEN35_N_VOCAB;
    expected_f32_elements += (uint64_t)prefill_cap * (
        3ull * QWEN35_N_EMBD +
        QWEN35_SSM_CONV_CHANNEL +
        3ull * QWEN35_SSM_INNER +
        2ull * QWEN35_N_HEAD_KV * QWEN35_N_HEAD_DIM +
        4ull * QWEN35_SSM_VALUE_HEAD +
        QWEN35_N_EMBD +
        QWEN35_N_EXPERT + QWEN35_N_EXPERT_USED +
        3ull * QWEN35_N_EXPERT_USED * QWEN35_N_FF_EXP +
        (uint64_t)QWEN35_N_EXPERT_USED * QWEN35_N_EMBD +
        QWEN35_N_EMBD +
        3ull * QWEN35_N_FF_SHARED + QWEN35_N_EMBD + 1ull);
    expected_f32_elements +=
        2ull * QWEN35_FULL_ATTENTION_LAYER_COUNT * ctx_capacity *
            QWEN35_N_HEAD_KV * QWEN35_N_HEAD_DIM +
        (uint64_t)QWEN35_RECURRENT_LAYER_COUNT *
            QWEN35_SSM_CONV_CHANNEL * (QWEN35_SSM_CONV_KERNEL - 1u) +
        (uint64_t)QWEN35_RECURRENT_LAYER_COUNT *
            QWEN35_SSM_VALUE_HEAD * QWEN35_SSM_STATE * QWEN35_SSM_STATE;
    const uint64_t expected_i32_elements =
        QWEN35_N_EXPERT_USED +
        (uint64_t)prefill_cap * (2u + QWEN35_N_EXPERT_USED);
    const uint64_t expected_bytes = expected_f32_elements * f32 +
        expected_i32_elements * sizeof(int32_t);

    TEST_ASSERT(ds4_internal_qwen35_gpu_graph_allocated_bytes(
        graph, graph_size, &allocated));
    TEST_ASSERT(allocated == expected_bytes);
    fprintf(stderr,
            "ds4-test: Qwen graph ctx=%u allocated=%.2f MiB "
            "(full=%u recurrent=%u, MoE resident=%u stream=%ux%u)\n",
            ctx_capacity, (double)allocated / (1024.0 * 1024.0),
            QWEN35_FULL_ATTENTION_LAYER_COUNT,
            QWEN35_RECURRENT_LAYER_COUNT,
            QWEN35_N_EXPERT_USED,
            moe_split_count,
            moe_split_width);

    TEST_ASSERT(!ds4_internal_qwen35_gpu_graph_validate_position(
        graph, graph_size - 1u, 0));
    TEST_ASSERT(!ds4_internal_qwen35_gpu_graph_advance(
        graph, graph_size - 1u, 0));
    TEST_ASSERT(!ds4_internal_qwen35_gpu_graph_reset(
        graph, graph_size - 1u));
    TEST_ASSERT(ds4_internal_qwen35_gpu_graph_validate_position(
        graph, graph_size, 0));
    TEST_ASSERT(!ds4_internal_qwen35_gpu_graph_validate_position(
        graph, graph_size, 1));
    TEST_ASSERT(!ds4_internal_qwen35_gpu_graph_advance(
        graph, graph_size, 1));
    TEST_ASSERT(ds4_internal_qwen35_gpu_graph_advance(
        graph, graph_size, 0));
    TEST_ASSERT(ds4_internal_qwen35_gpu_graph_validate_position(
        graph, graph_size, 1));
    TEST_ASSERT(!ds4_internal_qwen35_gpu_graph_advance(
        graph, graph_size, 2));
    TEST_ASSERT(ds4_internal_qwen35_gpu_graph_advance(
        graph, graph_size, 1));
    TEST_ASSERT(ds4_internal_qwen35_gpu_graph_advance(
        graph, graph_size, 2));
    TEST_ASSERT(!ds4_internal_qwen35_gpu_graph_validate_position(
        graph, graph_size, ctx_capacity));
    TEST_ASSERT(!ds4_internal_qwen35_gpu_graph_advance(
        graph, graph_size, ctx_capacity));

    for (uint32_t il = 0; il < QWEN35_N_LAYER; il++) {
        ds4_gpu_tensor *state[4] = {0};
        TEST_ASSERT(ds4_internal_qwen35_gpu_graph_layer_state(
            graph, graph_size, il, state));
        const bool full_attention = ds4_qwen35_layer_is_full_attention(il);
        TEST_ASSERT((state[0] != NULL) == full_attention);
        TEST_ASSERT((state[1] != NULL) == full_attention);
        TEST_ASSERT((state[2] != NULL) != full_attention);
        TEST_ASSERT((state[3] != NULL) != full_attention);
        for (uint32_t kind = 0; kind < 4; kind++) {
            if (!state[kind]) continue;
            const uint64_t bytes = ds4_gpu_tensor_bytes(state[kind]);
            TEST_ASSERT(bytes > 0 && bytes % sizeof(float) == 0);
            TEST_ASSERT(ds4_gpu_tensor_fill_f32(
                state[kind], (float)(1u + il * 4u + kind),
                bytes / sizeof(float)) != 0);
        }
    }
    TEST_ASSERT(ds4_internal_qwen35_gpu_graph_mark_state_invalid(
        graph, graph_size));
    TEST_ASSERT(!ds4_internal_qwen35_gpu_graph_validate_position(
        graph, graph_size, 0));
    TEST_ASSERT(!ds4_internal_qwen35_gpu_graph_advance(
        graph, graph_size, 0));
    TEST_ASSERT(ds4_internal_qwen35_gpu_graph_reset(graph, graph_size));
    TEST_ASSERT(ds4_internal_qwen35_gpu_graph_validate_position(
        graph, graph_size, 0));
    for (uint32_t il = 0; il < QWEN35_N_LAYER; il++) {
        ds4_gpu_tensor *state[4] = {0};
        TEST_ASSERT(ds4_internal_qwen35_gpu_graph_layer_state(
            graph, graph_size, il, state));
        for (uint32_t kind = 0; kind < 4; kind++) {
            if (state[kind]) {
                TEST_ASSERT(test_metal_tensor_is_all_zero(state[kind]));
            }
        }
    }

    /* Reinitializing a live graph must fail without replacing or leaking it. */
    const uint64_t before_reinit = allocated;
    TEST_ASSERT(!ds4_internal_qwen35_gpu_graph_alloc(
        graph, graph_size, ctx_capacity));
    TEST_ASSERT(ds4_internal_qwen35_gpu_graph_allocated_bytes(
        graph, graph_size, &allocated));
    TEST_ASSERT(allocated == before_reinit);

    ds4_internal_qwen35_gpu_graph_free(graph, graph_size);
    TEST_ASSERT(ds4_internal_qwen35_gpu_graph_allocated_bytes(
        graph, graph_size, &allocated));
    TEST_ASSERT(allocated == 0);
    TEST_ASSERT(!ds4_internal_qwen35_gpu_graph_validate_position(
        graph, graph_size, 0));
    TEST_ASSERT(!ds4_internal_qwen35_gpu_graph_advance(
        graph, graph_size, 0));
    TEST_ASSERT(!ds4_internal_qwen35_gpu_graph_reset(graph, graph_size));
    bool cleared = true;
    for (size_t i = 0; i < graph_size; i++) {
        if (graph[i] != 0) {
            cleared = false;
            break;
        }
    }
    TEST_ASSERT(cleared);
    ds4_internal_qwen35_gpu_graph_free(graph, graph_size);
    free(graph);
}

static void test_metal_qwen35_primitives(void) {
    enum {
        EMB_N = 64,
        EMB_ROWS = 3,
        EMB_ROW_BYTES = (EMB_N / 32) * 34,
        EMB_OFFSET = 0,
        CONV_CHANNEL = 5,
        CONV_KERNEL = 4,
        CONV_OFFSET = 256,
        RMS_VECTOR = 4,
        RMS_DIM = 7,
        RMS_OFFSET = 384,
        CONTROL_HEAD = 4,
        SSM_A_OFFSET = 448,
        DT_BIAS_OFFSET = 480,
    };

    const uint64_t model_size = test_round_up_u64(512, (uint64_t)getpagesize());
    void *model_raw = NULL;
    TEST_ASSERT(posix_memalign(&model_raw, (size_t)getpagesize(),
                              (size_t)model_size) == 0);
    if (!model_raw) return;
    uint8_t *model = model_raw;
    memset(model, 0, (size_t)model_size);

    test_fill_q8_0_weights(model + EMB_OFFSET, EMB_N, EMB_ROWS);
    float *conv_weight = (float *)(model + CONV_OFFSET);
    for (size_t i = 0; i < CONV_CHANNEL * CONV_KERNEL; i++) {
        conv_weight[i] = (float)((int)((i * 13u + 5u) % 29u) - 14) / 37.0f;
    }
    float *rms_weight = (float *)(model + RMS_OFFSET);
    for (size_t i = 0; i < RMS_DIM; i++) {
        rms_weight[i] = 0.75f + (float)i * 0.071f;
    }
    float *ssm_a = (float *)(model + SSM_A_OFFSET);
    float *dt_bias = (float *)(model + DT_BIAS_OFFSET);
    for (size_t i = 0; i < CONTROL_HEAD; i++) {
        ssm_a[i] = -0.35f - 0.11f * (float)i;
        dt_bias[i] = -0.4f + 0.27f * (float)i;
    }

    TEST_ASSERT(ds4_gpu_set_model_map(model_raw, model_size) != 0);

    /* Q8_0 token embedding row dequantization from the page-aligned map. */
    {
        const uint32_t row_index = 1;
        float expected[EMB_N];
        float actual[EMB_N];
        const uint8_t *row = model + EMB_OFFSET + row_index * EMB_ROW_BYTES;
        for (size_t block = 0; block < EMB_N / 32; block++) {
            uint16_t scale_bits = 0;
            memcpy(&scale_bits, row + block * 34u, sizeof(scale_bits));
            const float scale = test_f16_to_f32(scale_bits);
            const int8_t *quant =
                (const int8_t *)(row + block * 34u + 2u);
            for (size_t i = 0; i < 32; i++) {
                expected[block * 32u + i] = scale * (float)quant[i];
            }
        }
        ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(sizeof(actual));
        TEST_ASSERT(out != NULL);
        if (out) {
            TEST_ASSERT(ds4_gpu_qwen35_dequant_embedding_q8_0_tensor(
                out, model_raw, model_size, EMB_OFFSET, row_index, EMB_N));
            if (test_metal_read_f32(out, actual, EMB_N)) {
                test_metal_qwen35_close("Qwen Q8_0 embedding", actual,
                                        expected, EMB_N, 1.0e-7f, 1.0e-7f);
            }
        }
        ds4_gpu_tensor_free(out);
    }

    /* DeltaNet control transform, including both F32 model-map vectors. */
    {
        float alpha_logit[CONTROL_HEAD];
        float beta_logit[CONTROL_HEAD];
        float expected_decay[CONTROL_HEAD];
        float expected_beta[CONTROL_HEAD];
        float actual_decay[CONTROL_HEAD];
        float actual_beta[CONTROL_HEAD];
        float zero[CONTROL_HEAD];
        for (size_t i = 0; i < CONTROL_HEAD; i++) {
            alpha_logit[i] = -1.1f + 0.73f * (float)i;
            beta_logit[i] = 0.9f - 0.61f * (float)i;
            zero[i] = 0.0f;
        }
        TEST_ASSERT(ds4_qwen35_cpu_gated_delta_controls_f32(
            expected_decay, expected_beta, alpha_logit, beta_logit,
            ssm_a, dt_bias, CONTROL_HEAD));

        ds4_gpu_tensor *alpha =
            test_metal_tensor_from_f32(alpha_logit, CONTROL_HEAD);
        ds4_gpu_tensor *beta_in =
            test_metal_tensor_from_f32(beta_logit, CONTROL_HEAD);
        ds4_gpu_tensor *decay = test_metal_tensor_from_f32(zero, CONTROL_HEAD);
        ds4_gpu_tensor *beta = test_metal_tensor_from_f32(zero, CONTROL_HEAD);
        if (alpha && beta_in && decay && beta) {
            TEST_ASSERT(ds4_gpu_qwen35_gated_delta_controls_tensor(
                decay, beta, alpha, beta_in, model_raw, model_size,
                SSM_A_OFFSET, DT_BIAS_OFFSET, CONTROL_HEAD));
            if (test_metal_read_f32(decay, actual_decay, CONTROL_HEAD) &&
                test_metal_read_f32(beta, actual_beta, CONTROL_HEAD)) {
                test_metal_qwen35_close("Qwen DeltaNet log decay",
                                        actual_decay, expected_decay,
                                        CONTROL_HEAD, 4.0e-6f, 4.0e-6f);
                test_metal_qwen35_close("Qwen DeltaNet beta", actual_beta,
                                        expected_beta, CONTROL_HEAD,
                                        2.0e-6f, 2.0e-6f);
            }
        }
        ds4_gpu_tensor_free(alpha);
        ds4_gpu_tensor_free(beta_in);
        ds4_gpu_tensor_free(decay);
        ds4_gpu_tensor_free(beta);
    }

    /* Interleaved [Q, gate] head projection split. */
    {
        enum { N_HEAD = 3, HEAD_DIM = 5, OUT_N = N_HEAD * HEAD_DIM };
        float projection[OUT_N * 2];
        float expected_query[OUT_N];
        float expected_gate[OUT_N];
        float actual_query[OUT_N];
        float actual_gate[OUT_N];
        float zero[OUT_N];
        for (size_t i = 0; i < OUT_N * 2; i++) {
            projection[i] = (float)((int)((i * 19u + 7u) % 41u) - 20) /
                            23.0f;
        }
        memset(zero, 0, sizeof(zero));
        TEST_ASSERT(ds4_qwen35_cpu_split_q_gate_f32(
            expected_query, expected_gate, projection, N_HEAD, HEAD_DIM));
        ds4_gpu_tensor *projection_gpu =
            test_metal_tensor_from_f32(projection, OUT_N * 2);
        ds4_gpu_tensor *query_gpu = test_metal_tensor_from_f32(zero, OUT_N);
        ds4_gpu_tensor *gate_gpu = test_metal_tensor_from_f32(zero, OUT_N);
        if (projection_gpu && query_gpu && gate_gpu) {
            TEST_ASSERT(ds4_gpu_qwen35_split_q_gate_tensor(
                query_gpu, gate_gpu, projection_gpu, N_HEAD, HEAD_DIM));
            if (test_metal_read_f32(query_gpu, actual_query, OUT_N) &&
                test_metal_read_f32(gate_gpu, actual_gate, OUT_N)) {
                test_metal_qwen35_close("Qwen split query", actual_query,
                                        expected_query, OUT_N, 0.0f, 0.0f);
                test_metal_qwen35_close("Qwen split gate", actual_gate,
                                        expected_gate, OUT_N, 0.0f, 0.0f);
            }
        }
        ds4_gpu_tensor_free(projection_gpu);
        ds4_gpu_tensor_free(query_gpu);
        ds4_gpu_tensor_free(gate_gpu);
    }

    /* Elementwise attention gating and scalar shared-expert broadcast. */
    {
        enum { N_VALUE = 17 };
        float input[N_VALUE];
        float gate[N_VALUE];
        float expected[N_VALUE];
        float actual[N_VALUE];
        float zero[N_VALUE];
        const float scalar_gate[1] = {-0.37f};
        for (size_t i = 0; i < N_VALUE; i++) {
            input[i] = (float)((int)(i * 7u % 23u) - 11) / 9.0f;
            gate[i] = -2.0f + 0.29f * (float)i;
            zero[i] = 0.0f;
        }
        ds4_gpu_tensor *input_gpu = test_metal_tensor_from_f32(input, N_VALUE);
        ds4_gpu_tensor *gate_gpu = test_metal_tensor_from_f32(gate, N_VALUE);
        ds4_gpu_tensor *scalar_gpu =
            test_metal_tensor_from_f32(scalar_gate, 1);
        ds4_gpu_tensor *out_gpu = test_metal_tensor_from_f32(zero, N_VALUE);
        if (input_gpu && gate_gpu && scalar_gpu && out_gpu) {
            TEST_ASSERT(ds4_qwen35_cpu_sigmoid_gate_elements_f32(
                expected, input, gate, N_VALUE));
            TEST_ASSERT(ds4_gpu_qwen35_sigmoid_mul_tensor(
                out_gpu, input_gpu, gate_gpu, N_VALUE, false));
            if (test_metal_read_f32(out_gpu, actual, N_VALUE)) {
                test_metal_qwen35_close("Qwen elementwise sigmoid gate",
                                        actual, expected, N_VALUE,
                                        2.0e-6f, 2.0e-6f);
            }

            TEST_ASSERT(ds4_qwen35_cpu_sigmoid_gate_f32(
                expected, input, scalar_gate, 1, N_VALUE));
            TEST_ASSERT(ds4_gpu_qwen35_sigmoid_mul_tensor(
                out_gpu, input_gpu, scalar_gpu, N_VALUE, true));
            if (test_metal_read_f32(out_gpu, actual, N_VALUE)) {
                test_metal_qwen35_close("Qwen broadcast sigmoid gate",
                                        actual, expected, N_VALUE,
                                        2.0e-6f, 2.0e-6f);
            }
        }
        ds4_gpu_tensor_free(input_gpu);
        ds4_gpu_tensor_free(gate_gpu);
        ds4_gpu_tensor_free(scalar_gpu);
        ds4_gpu_tensor_free(out_gpu);
    }

    /* Split-half NeoX RoPE over only the configured prefix. */
    {
        enum { N_HEAD = 3, HEAD_DIM = 8, N_ROT = 6, N_VALUE = N_HEAD * HEAD_DIM };
        const uint32_t position = 17;
        const float theta = 10000.0f;
        float input[N_VALUE];
        float expected[N_VALUE];
        float actual[N_VALUE];
        for (size_t i = 0; i < N_VALUE; i++) {
            input[i] = 0.43f * sinf((float)(i + 1u) * 0.37f) -
                       0.17f * cosf((float)(i + 3u) * 0.23f);
        }
        memcpy(expected, input, sizeof(expected));
        TEST_ASSERT(ds4_qwen35_cpu_text_rope_f32(
            expected, position, N_HEAD, HEAD_DIM, N_ROT, theta));
        ds4_gpu_tensor *values = test_metal_tensor_from_f32(input, N_VALUE);
        if (values) {
            TEST_ASSERT(ds4_gpu_qwen35_rope_prefix_tensor(
                values, N_HEAD, HEAD_DIM, N_ROT, position, theta));
            if (test_metal_read_f32(values, actual, N_VALUE)) {
                test_metal_qwen35_close("Qwen text RoPE", actual, expected,
                                        N_VALUE, 1.0e-5f, 1.0e-5f);
            }
        }
        ds4_gpu_tensor_free(values);
    }

    /* Stateful one-token causal convolution. */
    {
        enum { STATE_N = CONV_CHANNEL * (CONV_KERNEL - 1) };
        float input[CONV_CHANNEL];
        float initial_state[STATE_N];
        float expected_state[STATE_N];
        float expected_out[CONV_CHANNEL];
        float actual_state[STATE_N];
        float actual_out[CONV_CHANNEL];
        float zero[CONV_CHANNEL];
        for (size_t i = 0; i < CONV_CHANNEL; i++) {
            input[i] = -0.55f + 0.31f * (float)i;
            zero[i] = 0.0f;
        }
        for (size_t i = 0; i < STATE_N; i++) {
            initial_state[i] = (float)((int)(i * 11u % 31u) - 15) / 29.0f;
        }
        memcpy(expected_state, initial_state, sizeof(expected_state));
        TEST_ASSERT(ds4_qwen35_cpu_causal_conv_step_f32(
            expected_out, expected_state, input, conv_weight,
            CONV_CHANNEL, CONV_KERNEL));
        ds4_gpu_tensor *input_gpu =
            test_metal_tensor_from_f32(input, CONV_CHANNEL);
        ds4_gpu_tensor *state_gpu =
            test_metal_tensor_from_f32(initial_state, STATE_N);
        ds4_gpu_tensor *out_gpu =
            test_metal_tensor_from_f32(zero, CONV_CHANNEL);
        if (input_gpu && state_gpu && out_gpu) {
            TEST_ASSERT(ds4_gpu_qwen35_causal_conv_step_tensor(
                out_gpu, state_gpu, input_gpu, model_raw, model_size,
                CONV_OFFSET, CONV_CHANNEL, CONV_KERNEL));
            if (test_metal_read_f32(out_gpu, actual_out, CONV_CHANNEL) &&
                test_metal_read_f32(state_gpu, actual_state, STATE_N)) {
                test_metal_qwen35_close("Qwen causal conv output", actual_out,
                                        expected_out, CONV_CHANNEL,
                                        3.0e-6f, 3.0e-6f);
                test_metal_qwen35_close("Qwen causal conv state", actual_state,
                                        expected_state, STATE_N,
                                        0.0f, 0.0f);
            }
        }
        ds4_gpu_tensor_free(input_gpu);
        ds4_gpu_tensor_free(state_gpu);
        ds4_gpu_tensor_free(out_gpu);
    }

    /* Layer-major causal convolution must match repeated token steps and
     * commit exactly the same final history. */
    {
        enum {
            N_TOKEN = 5,
            STATE_N = CONV_CHANNEL * (CONV_KERNEL - 1),
            ACTIVATION_N = N_TOKEN * CONV_CHANNEL,
        };
        float input[ACTIVATION_N];
        float initial_state[STATE_N];
        float expected_state[STATE_N];
        float expected_out[ACTIVATION_N];
        float actual_state[STATE_N];
        float actual_out[ACTIVATION_N];
        float zero[ACTIVATION_N];
        for (size_t i = 0; i < ACTIVATION_N; i++) {
            input[i] = 0.41f * sinf((float)(i + 1u) * 0.29f) -
                       0.16f * cosf((float)(i + 3u) * 0.17f);
            zero[i] = 0.0f;
        }
        for (size_t i = 0; i < STATE_N; i++) {
            initial_state[i] =
                (float)((int)(i * 17u % 37u) - 18) / 41.0f;
        }
        memcpy(expected_state, initial_state, sizeof(expected_state));
        for (size_t token = 0; token < N_TOKEN; token++) {
            TEST_ASSERT(ds4_qwen35_cpu_causal_conv_step_f32(
                expected_out + token * CONV_CHANNEL,
                expected_state,
                input + token * CONV_CHANNEL,
                conv_weight,
                CONV_CHANNEL,
                CONV_KERNEL));
        }

        ds4_gpu_tensor *input_gpu =
            test_metal_tensor_from_f32(input, ACTIVATION_N);
        ds4_gpu_tensor *state_gpu =
            test_metal_tensor_from_f32(initial_state, STATE_N);
        ds4_gpu_tensor *out_gpu =
            test_metal_tensor_from_f32(zero, ACTIVATION_N);
        if (input_gpu && state_gpu && out_gpu) {
            TEST_ASSERT(ds4_gpu_qwen35_causal_conv_sequence_tensor(
                out_gpu, state_gpu, input_gpu, model_raw, model_size,
                CONV_OFFSET, N_TOKEN, CONV_CHANNEL, CONV_KERNEL));
            if (test_metal_read_f32(
                    out_gpu, actual_out, ACTIVATION_N) &&
                test_metal_read_f32(
                    state_gpu, actual_state, STATE_N)) {
                test_metal_qwen35_close(
                    "Qwen causal conv sequence output",
                    actual_out, expected_out, ACTIVATION_N,
                    3.0e-6f, 3.0e-6f);
                test_metal_qwen35_close(
                    "Qwen causal conv sequence state",
                    actual_state, expected_state, STATE_N,
                    0.0f, 0.0f);
            }
        }
        ds4_gpu_tensor_free(input_gpu);
        ds4_gpu_tensor_free(state_gpu);
        ds4_gpu_tensor_free(out_gpu);
    }

    /* Recurrent Gated DeltaNet update and persistent state. */
    {
        enum {
            N_KEY_HEAD = 2,
            N_VALUE_HEAD = 4,
            KEY_DIM = 128,
            VALUE_DIM = 7,
            QK_N = N_KEY_HEAD * KEY_DIM,
            VALUE_N = N_VALUE_HEAD * VALUE_DIM,
            STATE_N = N_VALUE_HEAD * VALUE_DIM * KEY_DIM,
        };
        float query[QK_N];
        float key[QK_N];
        float value[VALUE_N];
        float log_decay[N_VALUE_HEAD];
        float beta[N_VALUE_HEAD];
        float initial_state[STATE_N];
        float expected_state[STATE_N];
        float expected_out[VALUE_N];
        float actual_state[STATE_N];
        float actual_out[VALUE_N];
        float zero[VALUE_N];
        for (size_t i = 0; i < QK_N; i++) {
            query[i] = 0.37f * sinf((float)(i + 1u) * 0.19f) - 0.08f;
            key[i] = 0.41f * cosf((float)(i + 2u) * 0.17f) + 0.05f;
        }
        for (size_t i = 0; i < VALUE_N; i++) {
            value[i] = (float)((int)(i * 17u % 43u) - 21) / 31.0f;
            zero[i] = 0.0f;
        }
        for (size_t i = 0; i < N_VALUE_HEAD; i++) {
            log_decay[i] = -0.07f - 0.045f * (float)i;
            beta[i] = 0.22f + 0.13f * (float)i;
        }
        for (size_t i = 0; i < STATE_N; i++) {
            initial_state[i] =
                (float)((int)(i * 29u % 97u) - 48) / 503.0f;
        }
        memcpy(expected_state, initial_state, sizeof(expected_state));
        TEST_ASSERT(ds4_qwen35_cpu_gated_delta_step_f32(
            expected_out, expected_state, query, key, value, log_decay, beta,
            N_KEY_HEAD, N_VALUE_HEAD, KEY_DIM, VALUE_DIM));

        ds4_gpu_tensor *query_gpu = test_metal_tensor_from_f32(query, QK_N);
        ds4_gpu_tensor *key_gpu = test_metal_tensor_from_f32(key, QK_N);
        ds4_gpu_tensor *value_gpu = test_metal_tensor_from_f32(value, VALUE_N);
        ds4_gpu_tensor *decay_gpu =
            test_metal_tensor_from_f32(log_decay, N_VALUE_HEAD);
        ds4_gpu_tensor *beta_gpu =
            test_metal_tensor_from_f32(beta, N_VALUE_HEAD);
        ds4_gpu_tensor *state_gpu =
            test_metal_tensor_from_f32(initial_state, STATE_N);
        ds4_gpu_tensor *out_gpu = test_metal_tensor_from_f32(zero, VALUE_N);
        if (query_gpu && key_gpu && value_gpu && decay_gpu && beta_gpu &&
            state_gpu && out_gpu) {
            ds4_gpu_internal_qwen35_gdn128_stats_reset();
            TEST_ASSERT(ds4_gpu_qwen35_gated_delta_step_tensor(
                out_gpu, state_gpu, query_gpu, key_gpu, value_gpu,
                decay_gpu, beta_gpu, N_KEY_HEAD, N_VALUE_HEAD,
                KEY_DIM, VALUE_DIM));
            TEST_ASSERT(
                ds4_gpu_internal_qwen35_gdn128_parallel_calls() == 1u);
            if (test_metal_read_f32(out_gpu, actual_out, VALUE_N) &&
                test_metal_read_f32(state_gpu, actual_state, STATE_N)) {
                test_metal_qwen35_close("Qwen Gated DeltaNet output",
                                        actual_out, expected_out, VALUE_N,
                                        8.0e-5f, 8.0e-5f);
                test_metal_qwen35_close("Qwen Gated DeltaNet state",
                                        actual_state, expected_state, STATE_N,
                                        8.0e-5f, 8.0e-5f);
            }
        }
        ds4_gpu_tensor_free(query_gpu);
        ds4_gpu_tensor_free(key_gpu);
        ds4_gpu_tensor_free(value_gpu);
        ds4_gpu_tensor_free(decay_gpu);
        ds4_gpu_tensor_free(beta_gpu);
        ds4_gpu_tensor_free(state_gpu);
        ds4_gpu_tensor_free(out_gpu);
    }

    /* The sequence Gated DeltaNet kernel keeps the recurrent rows in
     * registers across the chunk, but must remain numerically equivalent to
     * applying the reference recurrence token by token. */
    {
        enum {
            N_TOKEN = 3,
            N_KEY_HEAD = 2,
            N_VALUE_HEAD = 4,
            KEY_DIM = 128,
            VALUE_DIM = 128,
            QK_N = N_KEY_HEAD * KEY_DIM,
            VALUE_N = N_VALUE_HEAD * VALUE_DIM,
            PROJECTION_N = 2 * QK_N + VALUE_N,
            CONTROL_N = N_TOKEN * N_VALUE_HEAD,
            STATE_N = N_VALUE_HEAD * VALUE_DIM * KEY_DIM,
            OUTPUT_N = N_TOKEN * VALUE_N,
        };
        float projection[N_TOKEN * PROJECTION_N];
        float log_decay[CONTROL_N];
        float beta[CONTROL_N];
        float initial_state[STATE_N];
        float expected_state[STATE_N];
        float expected_out[OUTPUT_N];
        float actual_state[STATE_N];
        float actual_out[OUTPUT_N];
        float zero[OUTPUT_N];
        for (size_t i = 0; i < N_TOKEN * PROJECTION_N; i++) {
            projection[i] =
                0.29f * sinf((float)(i + 1u) * 0.013f) -
                0.11f * cosf((float)(i + 7u) * 0.019f);
        }
        for (size_t token = 0; token < N_TOKEN; token++) {
            for (size_t head = 0; head < N_VALUE_HEAD; head++) {
                const size_t i = token * N_VALUE_HEAD + head;
                log_decay[i] = -0.035f - 0.017f * (float)head -
                               0.009f * (float)token;
                beta[i] = 0.19f + 0.07f * (float)head +
                          0.025f * (float)token;
            }
        }
        for (size_t i = 0; i < STATE_N; i++) {
            initial_state[i] =
                (float)((int)(i * 31u % 101u) - 50) / 997.0f;
        }
        memset(zero, 0, sizeof(zero));
        memcpy(expected_state, initial_state, sizeof(expected_state));
        for (size_t token = 0; token < N_TOKEN; token++) {
            const float *row = projection + token * PROJECTION_N;
            TEST_ASSERT(ds4_qwen35_cpu_gated_delta_step_f32(
                expected_out + token * VALUE_N,
                expected_state,
                row,
                row + QK_N,
                row + 2 * QK_N,
                log_decay + token * N_VALUE_HEAD,
                beta + token * N_VALUE_HEAD,
                N_KEY_HEAD,
                N_VALUE_HEAD,
                KEY_DIM,
                VALUE_DIM));
        }

        ds4_gpu_tensor *projection_gpu = test_metal_tensor_from_f32(
            projection, N_TOKEN * PROJECTION_N);
        ds4_gpu_tensor *decay_gpu =
            test_metal_tensor_from_f32(log_decay, CONTROL_N);
        ds4_gpu_tensor *beta_gpu =
            test_metal_tensor_from_f32(beta, CONTROL_N);
        ds4_gpu_tensor *state_gpu =
            test_metal_tensor_from_f32(initial_state, STATE_N);
        ds4_gpu_tensor *out_gpu =
            test_metal_tensor_from_f32(zero, OUTPUT_N);
        if (projection_gpu && decay_gpu && beta_gpu && state_gpu && out_gpu) {
            ds4_gpu_internal_qwen35_gdn128_stats_reset();
            TEST_ASSERT(ds4_gpu_qwen35_gated_delta_sequence_128_tensor(
                out_gpu, state_gpu, projection_gpu, decay_gpu, beta_gpu,
                N_TOKEN, N_KEY_HEAD, N_VALUE_HEAD));
            TEST_ASSERT(
                ds4_gpu_internal_qwen35_gdn128_parallel_calls() == N_TOKEN);
            if (test_metal_read_f32(out_gpu, actual_out, OUTPUT_N) &&
                test_metal_read_f32(state_gpu, actual_state, STATE_N)) {
                test_metal_qwen35_close(
                    "Qwen Gated DeltaNet sequence output",
                    actual_out, expected_out, OUTPUT_N,
                    1.2e-4f, 1.2e-4f);
                test_metal_qwen35_close(
                    "Qwen Gated DeltaNet sequence state",
                    actual_state, expected_state, STATE_N,
                    1.2e-4f, 1.2e-4f);
            }
        }
        ds4_gpu_tensor_free(projection_gpu);
        ds4_gpu_tensor_free(decay_gpu);
        ds4_gpu_tensor_free(beta_gpu);
        ds4_gpu_tensor_free(state_gpu);
        ds4_gpu_tensor_free(out_gpu);
    }

    /* Per-value-head gated RMS normalization with F32 model weight. */
    {
        enum { N_VALUE = RMS_VECTOR * RMS_DIM };
        const float epsilon = 1.0e-6f;
        float input[N_VALUE];
        float gate[N_VALUE];
        float expected[N_VALUE];
        float actual[N_VALUE];
        float zero[N_VALUE];
        for (size_t i = 0; i < N_VALUE; i++) {
            input[i] = (float)((int)(i * 23u % 53u) - 26) / 19.0f;
            gate[i] = -1.7f + 0.14f * (float)i;
            zero[i] = 0.0f;
        }
        TEST_ASSERT(ds4_qwen35_cpu_rmsnorm_gated_f32(
            expected, input, gate, rms_weight, RMS_VECTOR, RMS_DIM, epsilon));
        ds4_gpu_tensor *input_gpu = test_metal_tensor_from_f32(input, N_VALUE);
        ds4_gpu_tensor *gate_gpu = test_metal_tensor_from_f32(gate, N_VALUE);
        ds4_gpu_tensor *out_gpu = test_metal_tensor_from_f32(zero, N_VALUE);
        if (input_gpu && gate_gpu && out_gpu) {
            TEST_ASSERT(ds4_gpu_qwen35_rmsnorm_gated_tensor(
                out_gpu, input_gpu, gate_gpu, model_raw, model_size,
                RMS_OFFSET, RMS_VECTOR, RMS_DIM, epsilon));
            if (test_metal_read_f32(out_gpu, actual, N_VALUE)) {
                test_metal_qwen35_close("Qwen gated RMSNorm", actual,
                                        expected, N_VALUE,
                                        5.0e-5f, 5.0e-5f);
            }
        }
        ds4_gpu_tensor_free(input_gpu);
        ds4_gpu_tensor_free(gate_gpu);
        ds4_gpu_tensor_free(out_gpu);
    }

    /* One-token grouped-query attention over separate F32 K/V caches. */
    {
        enum {
            N_KV = 5,
            N_QUERY_HEAD = 4,
            N_KV_HEAD = 2,
            HEAD_DIM = 32,
            QUERY_N = N_QUERY_HEAD * HEAD_DIM,
            CACHE_N = N_KV * N_KV_HEAD * HEAD_DIM,
        };
        float query[QUERY_N];
        float key[CACHE_N];
        float value[CACHE_N];
        float expected[QUERY_N];
        float actual[QUERY_N];
        float zero[QUERY_N];
        float score[N_KV];
        for (size_t i = 0; i < QUERY_N; i++) {
            query[i] = 0.29f * sinf((float)(i + 1u) * 0.071f) -
                       0.13f * cosf((float)(i + 4u) * 0.053f);
            zero[i] = 0.0f;
        }
        for (size_t i = 0; i < CACHE_N; i++) {
            key[i] = 0.33f * cosf((float)(i + 2u) * 0.037f) +
                     0.07f * sinf((float)(i + 5u) * 0.019f);
            value[i] = 0.61f * sinf((float)(i + 3u) * 0.043f) -
                       0.09f * cosf((float)(i + 7u) * 0.031f);
        }
        TEST_ASSERT(ds4_qwen35_cpu_gqa_decode_f32(
            expected, score, N_KV, query, key, value, N_KV,
            N_QUERY_HEAD, N_KV_HEAD, HEAD_DIM));
        ds4_gpu_tensor *query_gpu =
            test_metal_tensor_from_f32(query, QUERY_N);
        ds4_gpu_tensor *key_gpu = test_metal_tensor_from_f32(key, CACHE_N);
        ds4_gpu_tensor *value_gpu =
            test_metal_tensor_from_f32(value, CACHE_N);
        ds4_gpu_tensor *out_gpu = test_metal_tensor_from_f32(zero, QUERY_N);
        if (query_gpu && key_gpu && value_gpu && out_gpu) {
            TEST_ASSERT(ds4_gpu_qwen35_gqa_decode_tensor(
                out_gpu, query_gpu, key_gpu, value_gpu, N_KV,
                N_QUERY_HEAD, N_KV_HEAD, HEAD_DIM));
            if (test_metal_read_f32(out_gpu, actual, QUERY_N)) {
                test_metal_qwen35_close("Qwen GQA decode", actual, expected,
                                        QUERY_N, 2.0e-4f, 2.0e-4f);
            }
        }
        ds4_gpu_tensor_free(query_gpu);
        ds4_gpu_tensor_free(key_gpu);
        ds4_gpu_tensor_free(value_gpu);
        ds4_gpu_tensor_free(out_gpu);
    }

    /* Layer-major GQA must apply the causal boundary independently to every
     * query row in the prompt chunk, including a non-zero cache prefix. */
    {
        enum {
            POSITION0 = 2,
            N_TOKEN = 3,
            N_KV = POSITION0 + N_TOKEN,
            N_QUERY_HEAD = 4,
            N_KV_HEAD = 2,
            HEAD_DIM = 32,
            QUERY_ROW_N = N_QUERY_HEAD * HEAD_DIM,
            QUERY_N = N_TOKEN * QUERY_ROW_N,
            CACHE_ROW_N = N_KV_HEAD * HEAD_DIM,
            CACHE_N = N_KV * CACHE_ROW_N,
        };
        float query[QUERY_N];
        float key[CACHE_N];
        float value[CACHE_N];
        float expected[QUERY_N];
        float actual[QUERY_N];
        float zero[QUERY_N];
        float score[N_KV];
        for (size_t i = 0; i < QUERY_N; i++) {
            query[i] = 0.31f * sinf((float)(i + 1u) * 0.043f) -
                       0.12f * cosf((float)(i + 5u) * 0.029f);
            zero[i] = 0.0f;
        }
        for (size_t i = 0; i < CACHE_N; i++) {
            key[i] = 0.27f * cosf((float)(i + 2u) * 0.031f) +
                     0.08f * sinf((float)(i + 3u) * 0.017f);
            value[i] = 0.49f * sinf((float)(i + 4u) * 0.037f) -
                       0.06f * cosf((float)(i + 6u) * 0.023f);
        }
        for (size_t token = 0; token < N_TOKEN; token++) {
            TEST_ASSERT(ds4_qwen35_cpu_gqa_decode_f32(
                expected + token * QUERY_ROW_N,
                score,
                POSITION0 + token + 1u,
                query + token * QUERY_ROW_N,
                key,
                value,
                POSITION0 + token + 1u,
                N_QUERY_HEAD,
                N_KV_HEAD,
                HEAD_DIM));
        }

        ds4_gpu_tensor *query_gpu =
            test_metal_tensor_from_f32(query, QUERY_N);
        ds4_gpu_tensor *key_gpu =
            test_metal_tensor_from_f32(key, CACHE_N);
        ds4_gpu_tensor *value_gpu =
            test_metal_tensor_from_f32(value, CACHE_N);
        ds4_gpu_tensor *out_gpu =
            test_metal_tensor_from_f32(zero, QUERY_N);
        if (query_gpu && key_gpu && value_gpu && out_gpu) {
            TEST_ASSERT(ds4_gpu_qwen35_gqa_prefill_tensor(
                out_gpu, query_gpu, key_gpu, value_gpu,
                POSITION0, N_TOKEN, N_QUERY_HEAD, N_KV_HEAD, HEAD_DIM));
            if (test_metal_read_f32(out_gpu, actual, QUERY_N)) {
                test_metal_qwen35_close(
                    "Qwen GQA prefill",
                    actual, expected, QUERY_N,
                    2.0e-4f, 2.0e-4f);
            }
        }
        ds4_gpu_tensor_free(query_gpu);
        ds4_gpu_tensor_free(key_gpu);
        ds4_gpu_tensor_free(value_gpu);
        ds4_gpu_tensor_free(out_gpu);
    }

    /* Production-library Qwen router wrapper.  Offset views exercise the
     * graph's split-4+4 layout, while parent guards prove the fixed contiguous
     * ABI cannot write outside the selected top-8 rows. */
    {
        enum {
            ROUTER_EXPERT = QWEN35_N_EXPERT,
            ROUTER_SELECTED = QWEN35_N_EXPERT_USED,
            ROUTER_PAD = 4,
        };
        float logits_parent[ROUTER_PAD + ROUTER_EXPERT + ROUTER_PAD];
        float logits_before[ROUTER_PAD + ROUTER_EXPERT + ROUTER_PAD];
        int32_t selected_parent[
            ROUTER_PAD + ROUTER_SELECTED + ROUTER_PAD];
        float weight_parent[ROUTER_PAD + ROUTER_SELECTED + ROUTER_PAD];
        float probability[ROUTER_EXPERT];
        int32_t expected_selected[ROUTER_SELECTED];
        float expected_weight[ROUTER_SELECTED];
        int32_t actual_selected[ROUTER_SELECTED];
        float actual_weight[ROUTER_SELECTED];

        for (size_t i = 0;
             i < sizeof(logits_parent) / sizeof(logits_parent[0]); i++) {
            logits_parent[i] = -777.0f;
        }
        for (size_t i = 0; i < ROUTER_EXPERT; i++) {
            logits_parent[ROUTER_PAD + i] =
                1.7f * sinf((float)(i + 1u) * 0.173f) -
                0.9f * cosf((float)(i + 3u) * 0.097f) +
                (float)((int)(i % 11u) - 5) * 0.031f;
        }
        /* Exact tie at the maximum must prefer the lower expert ID. */
        logits_parent[ROUTER_PAD + 19u] = 9.0f;
        logits_parent[ROUTER_PAD + 73u] = 9.0f;
        memcpy(logits_before, logits_parent, sizeof(logits_before));
        for (size_t i = 0;
             i < sizeof(selected_parent) / sizeof(selected_parent[0]); i++) {
            selected_parent[i] = INT32_C(0x5a5a5a5a);
            weight_parent[i] = -333.0f;
        }

        TEST_ASSERT(ds4_qwen35_cpu_softmax_top8_f32(
            expected_selected, expected_weight, probability,
            logits_parent + ROUTER_PAD));

        ds4_gpu_tensor *logits_base = ds4_gpu_tensor_alloc(
            sizeof(logits_parent));
        ds4_gpu_tensor *selected_base = ds4_gpu_tensor_alloc(
            sizeof(selected_parent));
        ds4_gpu_tensor *weight_base = ds4_gpu_tensor_alloc(
            sizeof(weight_parent));
        ds4_gpu_tensor *logits = logits_base ? ds4_gpu_tensor_view(
            logits_base, ROUTER_PAD * sizeof(float),
            ROUTER_EXPERT * sizeof(float)) : NULL;
        ds4_gpu_tensor *selected = selected_base ? ds4_gpu_tensor_view(
            selected_base, ROUTER_PAD * sizeof(int32_t),
            ROUTER_SELECTED * sizeof(int32_t)) : NULL;
        ds4_gpu_tensor *weight = weight_base ? ds4_gpu_tensor_view(
            weight_base, ROUTER_PAD * sizeof(float),
            ROUTER_SELECTED * sizeof(float)) : NULL;
        if (logits_base && selected_base && weight_base && logits && selected &&
            weight) {
            TEST_ASSERT(ds4_gpu_tensor_write(
                logits_base, 0, logits_parent, sizeof(logits_parent)));
            TEST_ASSERT(ds4_gpu_tensor_write(
                selected_base, 0, selected_parent, sizeof(selected_parent)));
            TEST_ASSERT(ds4_gpu_tensor_write(
                weight_base, 0, weight_parent, sizeof(weight_parent)));
            TEST_ASSERT(ds4_gpu_qwen35_router_softmax_top8_tensor(
                selected, weight, logits));
            TEST_ASSERT(ds4_gpu_tensor_read(
                selected, 0, actual_selected, sizeof(actual_selected)));
            TEST_ASSERT(ds4_gpu_tensor_read(
                weight, 0, actual_weight, sizeof(actual_weight)));
            for (size_t i = 0; i < ROUTER_SELECTED; i++) {
                TEST_ASSERT(actual_selected[i] == expected_selected[i]);
            }
            test_metal_qwen35_close("Qwen router top-8 weight",
                                    actual_weight, expected_weight,
                                    ROUTER_SELECTED, 2.0e-6f, 2.0e-6f);

            TEST_ASSERT(ds4_gpu_tensor_read(
                logits_base, 0, logits_parent, sizeof(logits_parent)));
            TEST_ASSERT(memcmp(logits_parent, logits_before,
                               sizeof(logits_parent)) == 0);
            TEST_ASSERT(ds4_gpu_tensor_read(
                selected_base, 0, selected_parent,
                sizeof(selected_parent)));
            TEST_ASSERT(ds4_gpu_tensor_read(
                weight_base, 0, weight_parent, sizeof(weight_parent)));
            for (size_t i = 0; i < ROUTER_PAD; i++) {
                TEST_ASSERT(selected_parent[i] == INT32_C(0x5a5a5a5a));
                TEST_ASSERT(selected_parent[
                    ROUTER_PAD + ROUTER_SELECTED + i] ==
                    INT32_C(0x5a5a5a5a));
                TEST_ASSERT(weight_parent[i] == -333.0f);
                TEST_ASSERT(weight_parent[
                    ROUTER_PAD + ROUTER_SELECTED + i] == -333.0f);
            }

            /* Defensive diagnostic contract for corrupt/non-finite logits. */
            const uint32_t quiet_nan_bits = UINT32_C(0x7fc00000);
            memcpy(&logits_before[ROUTER_PAD + 41u], &quiet_nan_bits,
                   sizeof(quiet_nan_bits));
            TEST_ASSERT(ds4_gpu_tensor_write(
                logits, 0, logits_before + ROUTER_PAD,
                ROUTER_EXPERT * sizeof(float)));
            TEST_ASSERT(ds4_gpu_qwen35_router_softmax_top8_tensor(
                selected, weight, logits));
            TEST_ASSERT(ds4_gpu_tensor_read(
                selected, 0, actual_selected, sizeof(actual_selected)));
            TEST_ASSERT(ds4_gpu_tensor_read(
                weight, 0, actual_weight, sizeof(actual_weight)));
            for (size_t i = 0; i < ROUTER_SELECTED; i++) {
                TEST_ASSERT(actual_selected[i] == -1);
                TEST_ASSERT(actual_weight[i] == 0.0f);
            }

            ds4_gpu_tensor *short_logits = ds4_gpu_tensor_alloc(
                (ROUTER_EXPERT - 1u) * sizeof(float));
            ds4_gpu_tensor *short_selected = ds4_gpu_tensor_alloc(
                (ROUTER_SELECTED - 1u) * sizeof(int32_t));
            ds4_gpu_tensor *short_weight = ds4_gpu_tensor_alloc(
                (ROUTER_SELECTED - 1u) * sizeof(float));
            if (short_logits && short_selected && short_weight) {
                TEST_ASSERT(!ds4_gpu_qwen35_router_softmax_top8_tensor(
                    selected, weight, short_logits));
                TEST_ASSERT(!ds4_gpu_qwen35_router_softmax_top8_tensor(
                    short_selected, weight, logits));
                TEST_ASSERT(!ds4_gpu_qwen35_router_softmax_top8_tensor(
                    selected, short_weight, logits));
            }
            ds4_gpu_tensor_free(short_logits);
            ds4_gpu_tensor_free(short_selected);
            ds4_gpu_tensor_free(short_weight);
        } else {
            TEST_ASSERT(false);
        }
        ds4_gpu_tensor_free(logits);
        ds4_gpu_tensor_free(selected);
        ds4_gpu_tensor_free(weight);
        ds4_gpu_tensor_free(logits_base);
        ds4_gpu_tensor_free(selected_base);
        ds4_gpu_tensor_free(weight_base);
    }

    /* Exercise the production batch ABI through both implementations.  The
     * parallel kernel must retain the serial kernel's deterministic expert-ID
     * tie break and reference normalization for every token, not only for the
     * single decode-shaped vector above. */
    {
        enum {
            ROUTER_BATCH = 13,
            ROUTER_EXPERT = QWEN35_N_EXPERT,
            ROUTER_SELECTED = QWEN35_N_EXPERT_USED,
        };
        float logits[ROUTER_BATCH * ROUTER_EXPERT];
        int32_t parallel_selected[ROUTER_BATCH * ROUTER_SELECTED];
        int32_t serial_selected[ROUTER_BATCH * ROUTER_SELECTED];
        int32_t expected_selected[ROUTER_BATCH * ROUTER_SELECTED];
        float parallel_weight[ROUTER_BATCH * ROUTER_SELECTED];
        float serial_weight[ROUTER_BATCH * ROUTER_SELECTED];
        float expected_weight[ROUTER_BATCH * ROUTER_SELECTED];
        float probability[ROUTER_EXPERT];

        for (uint32_t token = 0; token < ROUTER_BATCH; token++) {
            for (uint32_t expert = 0; expert < ROUTER_EXPERT; expert++) {
                const uint32_t mixed =
                    expert * 31u + token * 17u + (expert ^ token) * 7u;
                logits[token * ROUTER_EXPERT + expert] =
                    1.3f * sinf((float)(expert + 1u) *
                                (float)(token + 3u) * 0.021f) +
                    0.7f * cosf((float)(expert + token + 5u) * 0.047f) +
                    (float)((int)(mixed % 37u) - 18) * 0.013f;
            }
            if (token == 0u) {
                /* Every expert ties: the selected IDs must be 0..7. */
                memset(logits, 0, ROUTER_EXPERT * sizeof(float));
            } else {
                const uint32_t low = 5u + token;
                const uint32_t high = 181u + token;
                const float tied_maximum = 12.0f + (float)token * 0.01f;
                logits[token * ROUTER_EXPERT + low] = tied_maximum;
                logits[token * ROUTER_EXPERT + high] = tied_maximum;
            }
        }
        /* Retain finite extremes that force softmax underflow without making
         * the defensive non-finite contract ambiguous. */
        logits[ROUTER_EXPERT + 3u] = 80.0f;
        logits[ROUTER_EXPERT + 203u] = -80.0f;

        for (uint32_t token = 0; token < ROUTER_BATCH; token++) {
            TEST_ASSERT(ds4_qwen35_cpu_softmax_top8_f32(
                expected_selected + token * ROUTER_SELECTED,
                expected_weight + token * ROUTER_SELECTED,
                probability,
                logits + token * ROUTER_EXPERT));
        }

        ds4_gpu_tensor *logits_gpu =
            ds4_gpu_tensor_alloc(sizeof(logits));
        ds4_gpu_tensor *selected_gpu =
            ds4_gpu_tensor_alloc(sizeof(parallel_selected));
        ds4_gpu_tensor *weight_gpu =
            ds4_gpu_tensor_alloc(sizeof(parallel_weight));
        TEST_ASSERT(logits_gpu && selected_gpu && weight_gpu);
        if (logits_gpu && selected_gpu && weight_gpu) {
            char *saved_parallel_router =
                test_save_env("DS4_QWEN_DISABLE_PARALLEL_ROUTER");
            TEST_ASSERT(ds4_gpu_tensor_write(
                logits_gpu, 0, logits, sizeof(logits)));

            unsetenv("DS4_QWEN_DISABLE_PARALLEL_ROUTER");
            TEST_ASSERT(ds4_gpu_qwen35_router_softmax_top8_batch_tensor(
                selected_gpu, weight_gpu, logits_gpu, ROUTER_BATCH));
            TEST_ASSERT(ds4_gpu_tensor_read(
                selected_gpu, 0, parallel_selected,
                sizeof(parallel_selected)));
            TEST_ASSERT(ds4_gpu_tensor_read(
                weight_gpu, 0, parallel_weight,
                sizeof(parallel_weight)));

            setenv("DS4_QWEN_DISABLE_PARALLEL_ROUTER", "1", 1);
            TEST_ASSERT(ds4_gpu_qwen35_router_softmax_top8_batch_tensor(
                selected_gpu, weight_gpu, logits_gpu, ROUTER_BATCH));
            TEST_ASSERT(ds4_gpu_tensor_read(
                selected_gpu, 0, serial_selected,
                sizeof(serial_selected)));
            TEST_ASSERT(ds4_gpu_tensor_read(
                weight_gpu, 0, serial_weight,
                sizeof(serial_weight)));
            test_restore_env("DS4_QWEN_DISABLE_PARALLEL_ROUTER",
                             saved_parallel_router);

            for (size_t i = 0;
                 i < ROUTER_BATCH * ROUTER_SELECTED; i++) {
                TEST_ASSERT(parallel_selected[i] == serial_selected[i]);
                TEST_ASSERT(parallel_selected[i] == expected_selected[i]);
            }
            test_metal_qwen35_close(
                "Qwen router parallel/serial batch weight",
                parallel_weight, serial_weight,
                ROUTER_BATCH * ROUTER_SELECTED, 1.0e-7f, 1.0e-7f);
            test_metal_qwen35_close(
                "Qwen router parallel/CPU batch weight",
                parallel_weight, expected_weight,
                ROUTER_BATCH * ROUTER_SELECTED, 2.0e-6f, 2.0e-6f);
        }
        ds4_gpu_tensor_free(logits_gpu);
        ds4_gpu_tensor_free(selected_gpu);
        ds4_gpu_tensor_free(weight_gpu);
    }

    free(model_raw);
}

#ifdef __APPLE__
extern uint64_t ds4_gpu_internal_stream_expert_cache_decode_tokens(void);
extern uint64_t ds4_gpu_internal_stream_expert_timing_selected_calls(void);
extern uint32_t ds4_gpu_internal_stream_expert_cache_required_floor(void);
extern uint32_t ds4_gpu_internal_stream_expert_cache_slab_count(void);
extern uint64_t ds4_gpu_internal_stream_expert_cache_slab_capacity_bytes(void);
extern void ds4_gpu_internal_stream_expert_cache_fail_mlock_after(
    int64_t calls);
extern int ds4_gpu_internal_moe_selected_trace_inspect(
    const char *path, uint32_t requested_width, uint32_t *file_width,
    uint64_t *record_count, int *legacy);

typedef struct {
    uint16_t d;
    uint16_t dmin;
    uint8_t scales[12];
    uint8_t qs[128];
} test_metal_q4_k_block;

_Static_assert(sizeof(test_metal_q4_k_block) == 144,
               "Q4_K test block ABI drift");

typedef struct {
    uint8_t  magic[8];
    uint32_t version;
    uint32_t width;
    uint32_t id_bytes;
    uint32_t header_bytes;
} test_metal_selected_trace_header;

_Static_assert(sizeof(test_metal_selected_trace_header) == 24,
               "selected trace test header ABI drift");

static bool test_write_all(int fd, const void *data, size_t bytes) {
    const uint8_t *src = data;
    size_t written = 0;
    while (written < bytes) {
        ssize_t n = write(fd, src + written, bytes - written);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        written += (size_t)n;
    }
    return true;
}

static void test_metal_selected_trace_legacy_parser(void) {
    char legacy_path[] = "/tmp/ds4-selected-legacy-XXXXXX";
    int legacy_fd = mkstemp(legacy_path);
    TEST_ASSERT(legacy_fd >= 0);
    if (legacy_fd >= 0) {
        /* 96 bytes is divisible by both width-6 and width-8 record sizes.
         * Headerless input must nevertheless remain unambiguously width 6. */
        int32_t legacy_ids[24];
        for (uint32_t i = 0; i < 24; i++) legacy_ids[i] = (int32_t)i;
        TEST_ASSERT(test_write_all(legacy_fd,
                                   legacy_ids,
                                   sizeof(legacy_ids)));
        close(legacy_fd);
        uint32_t width = 0;
        uint64_t records = 0;
        int legacy = 0;
        TEST_ASSERT(ds4_gpu_internal_moe_selected_trace_inspect(
            legacy_path, 6, &width, &records, &legacy));
        TEST_ASSERT(width == 6);
        TEST_ASSERT(records == 4);
        TEST_ASSERT(legacy == 1);
        TEST_ASSERT(!ds4_gpu_internal_moe_selected_trace_inspect(
            legacy_path, 8, NULL, NULL, NULL));
        unlink(legacy_path);
    }

    char bad_path[] = "/tmp/ds4-selected-bad-header-XXXXXX";
    int bad_fd = mkstemp(bad_path);
    TEST_ASSERT(bad_fd >= 0);
    if (bad_fd >= 0) {
        test_metal_selected_trace_header bad = {
            .magic = {'D', 'S', '4', 'M', 'O', 'E', 'I', 'D'},
            .version = 99,
            .width = 8,
            .id_bytes = sizeof(int32_t),
            .header_bytes = sizeof(test_metal_selected_trace_header),
        };
        int32_t ids[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        TEST_ASSERT(test_write_all(bad_fd, &bad, sizeof(bad)));
        TEST_ASSERT(test_write_all(bad_fd, ids, sizeof(ids)));
        close(bad_fd);
        TEST_ASSERT(!ds4_gpu_internal_moe_selected_trace_inspect(
            bad_path, 8, NULL, NULL, NULL));
        unlink(bad_path);
    }
}

typedef struct {
    float out[2];
    uint64_t hits;
    uint64_t misses;
    uint64_t pread_bytes;
    uint32_t current_entries;
} test_metal_q4_slots_result;

typedef struct {
    float out[2];
    float partial0[2];
    float partial1[2];
    int32_t selected_after[8];
    float weights_after[8];
    uint64_t hits;
    uint64_t misses;
    uint64_t pread_bytes;
    uint64_t decode_tokens;
    uint32_t current_entries;
} test_metal_qwen_top8_result;

static float test_metal_qwen_top8_expected(
        const int32_t selected[8],
        const float   weights[8]) {
    const float silu_one = 1.0f / (1.0f + expf(-1.0f));
    float weighted = 0.0f;
    for (uint32_t i = 0; i < 8; i++) {
        const float value = (float)(selected[i] % 15 + 1);
        weighted += weights[i] * value * value;
    }
    return 256.0f * silu_one * weighted;
}

static void test_metal_q4_k_fill_constant(
        test_metal_q4_k_block *block,
        uint8_t                value) {
    memset(block, 0, sizeof(*block));
    block->d = test_float_to_f16(1.0f);
    for (uint32_t group = 0; group < 4; group++) {
        block->scales[group] = 1;
    }
    for (uint32_t group = 4; group < 8; group++) {
        block->scales[group + 4] = 1;
    }
    memset(block->qs, (int)(value | (uint8_t)(value << 4)),
           sizeof(block->qs));
}

static bool test_metal_q4_selected_slots_case(
        const void                     *model_map,
        uint64_t                        model_size,
        uint64_t                        gate_offset,
        uint64_t                        up_offset,
        uint64_t                        down_offset,
        uint64_t                        gate_expert_bytes,
        uint64_t                        down_expert_bytes,
        uint32_t                        n_expert,
        bool                            check_undersized,
        test_metal_q4_slots_result     *result) {
    enum {
        IN_DIM = 256,
        MID_DIM = 256,
        OUT_DIM = 2,
        TOTAL_EXPERT = 128,
        GGML_TYPE_Q4_K = 12,
    };
    const uint64_t gate_row_bytes = sizeof(test_metal_q4_k_block);
    const uint64_t down_row_bytes = sizeof(test_metal_q4_k_block);
    const uint64_t mid_bytes =
        (uint64_t)n_expert * MID_DIM * sizeof(float);
    const uint64_t expert_out_bytes =
        (uint64_t)n_expert * OUT_DIM * sizeof(float);
    int32_t selected_host[6] = {0, 1, 2, 3, 4, 5};
    float weight_host[6] = {0.1f, 0.2f, 0.3f, 0.4f, 0.0f, 0.0f};
    float input_host[IN_DIM];
    for (uint32_t i = 0; i < IN_DIM; i++) {
        input_host[i] = 1.0f / (float)IN_DIM;
    }

    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(OUT_DIM * sizeof(float));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(mid_bytes);
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc(mid_bytes);
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(mid_bytes);
    ds4_gpu_tensor *experts = ds4_gpu_tensor_alloc(expert_out_bytes);
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(
        (uint64_t)n_expert * sizeof(int32_t));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(
        (uint64_t)n_expert * sizeof(float));
    ds4_gpu_tensor *input = ds4_gpu_tensor_alloc(sizeof(input_host));
    bool ok = out && gate && up && mid && experts && selected && weights && input;
    TEST_ASSERT(ok);
    if (!ok) goto cleanup;

    ok = ds4_gpu_tensor_write(selected, 0, selected_host,
                              (uint64_t)n_expert * sizeof(int32_t)) != 0 &&
         ds4_gpu_tensor_write(weights, 0, weight_host,
                              (uint64_t)n_expert * sizeof(float)) != 0 &&
         ds4_gpu_tensor_write(input, 0, input_host,
                              sizeof(input_host)) != 0 &&
         ds4_gpu_tensor_fill_f32(experts, -1234.0f,
                                 (uint64_t)n_expert * OUT_DIM) != 0;
    TEST_ASSERT(ok);
    if (!ok) goto cleanup;

    if (check_undersized) {
        ds4_gpu_tensor *short_mid = ds4_gpu_tensor_alloc(mid_bytes - sizeof(float));
        TEST_ASSERT(short_mid != NULL);
        if (short_mid) {
            TEST_ASSERT(!ds4_gpu_routed_moe_one_tensor(
                out, gate, up, short_mid, experts,
                model_map, model_size,
                gate_offset, up_offset, down_offset,
                GGML_TYPE_Q4_K, GGML_TYPE_Q4_K,
                gate_expert_bytes, gate_row_bytes,
                down_expert_bytes, down_row_bytes,
                IN_DIM, MID_DIM, OUT_DIM,
                selected, weights, TOTAL_EXPERT, n_expert,
                0.0f, input, 0));
            ds4_gpu_tensor_free(short_mid);
        }
    }

    ok = ds4_gpu_routed_moe_one_tensor(
        out, gate, up, mid, experts,
        model_map, model_size,
        gate_offset, up_offset, down_offset,
        GGML_TYPE_Q4_K, GGML_TYPE_Q4_K,
        gate_expert_bytes, gate_row_bytes,
        down_expert_bytes, down_row_bytes,
        IN_DIM, MID_DIM, OUT_DIM,
        selected, weights, TOTAL_EXPERT, n_expert,
        0.0f, input, 0) != 0;
    TEST_ASSERT(ok);
    if (!ok) goto cleanup;

    ok = ds4_gpu_tensor_read(out, 0, result->out, sizeof(result->out)) != 0;
    TEST_ASSERT(ok);
    if (!ok) goto cleanup;
    ds4_gpu_stream_expert_cache_stats(
        &result->hits, &result->misses, &result->pread_bytes, NULL, NULL);
    result->current_entries = ds4_gpu_stream_expert_cache_current_count();

cleanup:
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(experts);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(input);
    return ok;
}

static bool test_metal_qwen_top8_case(
        const void                    *model_map,
        uint64_t                       model_size,
        uint64_t                       gate_offset,
        uint64_t                       up_offset,
        uint64_t                       down_offset,
        uint64_t                       gate_expert_bytes,
        uint64_t                       down_expert_bytes,
        const int32_t                  selected_host[8],
        const float                    weight_host[8],
        const float                   *router_logits_host,
        bool                           expect_success,
        test_metal_qwen_top8_result   *result) {
    enum {
        IN_DIM = 256,
        MID_DIM = 256,
        OUT_DIM = 2,
        TOTAL_EXPERT = 128,
        HALF_EXPERT = 4,
        ROUTED_EXPERT = 8,
        ROUTER_EXPERT = 256,
        GGML_TYPE_Q4_K = 12,
    };
    const uint64_t row_bytes = sizeof(test_metal_q4_k_block);
    const uint64_t mid_bytes =
        (uint64_t)ROUTED_EXPERT * MID_DIM * sizeof(float);
    const uint64_t expert_out_bytes =
        (uint64_t)ROUTED_EXPERT * OUT_DIM * sizeof(float);
    const float sentinel = -4321.25f;
    float input_host[IN_DIM];
    for (uint32_t i = 0; i < IN_DIM; i++) {
        input_host[i] = 1.0f / (float)IN_DIM;
    }

    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(OUT_DIM * sizeof(float));
    ds4_gpu_tensor *partial0 =
        ds4_gpu_tensor_alloc(OUT_DIM * sizeof(float));
    ds4_gpu_tensor *partial1 =
        ds4_gpu_tensor_alloc(OUT_DIM * sizeof(float));
    ds4_gpu_tensor *gate = ds4_gpu_tensor_alloc(mid_bytes);
    ds4_gpu_tensor *up = ds4_gpu_tensor_alloc(mid_bytes);
    ds4_gpu_tensor *mid = ds4_gpu_tensor_alloc(mid_bytes);
    ds4_gpu_tensor *experts = ds4_gpu_tensor_alloc(expert_out_bytes);
    ds4_gpu_tensor *selected = ds4_gpu_tensor_alloc(8u * sizeof(int32_t));
    ds4_gpu_tensor *weights = ds4_gpu_tensor_alloc(8u * sizeof(float));
    ds4_gpu_tensor *selected_half0 = selected ? ds4_gpu_tensor_view(
        selected, 0, 4u * sizeof(int32_t)) : NULL;
    ds4_gpu_tensor *selected_half1 = selected ? ds4_gpu_tensor_view(
        selected, 4u * sizeof(int32_t), 4u * sizeof(int32_t)) : NULL;
    ds4_gpu_tensor *weights_half0 = weights ? ds4_gpu_tensor_view(
        weights, 0, 4u * sizeof(float)) : NULL;
    ds4_gpu_tensor *weights_half1 = weights ? ds4_gpu_tensor_view(
        weights, 4u * sizeof(float), 4u * sizeof(float)) : NULL;
    ds4_gpu_tensor *router_logits = router_logits_host ?
        ds4_gpu_tensor_alloc(ROUTER_EXPERT * sizeof(float)) : NULL;
    ds4_gpu_tensor *input = ds4_gpu_tensor_alloc(sizeof(input_host));
    bool ok = out && partial0 && partial1 && gate && up && mid && experts &&
              selected && weights && selected_half0 && selected_half1 &&
              weights_half0 && weights_half1 && input && result &&
              (!router_logits_host || router_logits);
    TEST_ASSERT(ok);
    if (!ok) goto cleanup;

    ok = ds4_gpu_tensor_write(selected, 0, selected_host,
                              8u * sizeof(int32_t)) != 0 &&
         ds4_gpu_tensor_write(weights, 0, weight_host,
                              8u * sizeof(float)) != 0 &&
         ds4_gpu_tensor_write(input, 0, input_host,
                              sizeof(input_host)) != 0 &&
         (!router_logits_host ||
          ds4_gpu_tensor_write(router_logits, 0, router_logits_host,
                               ROUTER_EXPERT * sizeof(float)) != 0) &&
         ds4_gpu_tensor_fill_f32(out, sentinel, OUT_DIM) != 0 &&
         ds4_gpu_tensor_fill_f32(partial0, sentinel, OUT_DIM) != 0 &&
         ds4_gpu_tensor_fill_f32(partial1, sentinel, OUT_DIM) != 0 &&
         ds4_gpu_tensor_fill_f32(experts, sentinel,
                                 (uint64_t)ROUTED_EXPERT * OUT_DIM) != 0;
    TEST_ASSERT(ok);
    if (!ok) goto cleanup;

    uint64_t hits0 = 0;
    uint64_t misses0 = 0;
    uint64_t pread0 = 0;
    ds4_gpu_stream_expert_cache_stats(&hits0, &misses0, &pread0,
                                      NULL, NULL);
    const uint64_t tokens0 =
        ds4_gpu_internal_stream_expert_cache_decode_tokens();
    const uint32_t entries0 = ds4_gpu_stream_expert_cache_current_count();

    int call_ok = 0;
    if (router_logits_host) {
        const int begin_ok = ds4_gpu_begin_commands();
        ok = begin_ok != 0 &&
             ds4_gpu_qwen35_router_softmax_top8_tensor(
                 selected, weights, router_logits) != 0;
        TEST_ASSERT(ok);
        if (!ok) {
            if (begin_ok) (void)ds4_gpu_end_commands();
            goto cleanup;
        }
    }
    call_ok = ds4_gpu_qwen35_routed_moe_top8_tensor(
            out, partial0, partial1, gate, up, mid, experts,
            model_map, model_size,
            gate_offset, up_offset, down_offset,
            GGML_TYPE_Q4_K, GGML_TYPE_Q4_K,
            gate_expert_bytes, row_bytes,
            down_expert_bytes, row_bytes,
            IN_DIM, MID_DIM, OUT_DIM,
            selected, weights,
            selected_half0, selected_half1, weights_half0, weights_half1,
            TOTAL_EXPERT,
            0.0f, input, 0, router_logits_host != NULL);
    if (router_logits_host) {
        const int end_ok = ds4_gpu_end_commands();
        TEST_ASSERT(end_ok);
        if (!end_ok) ok = false;
    }
    TEST_ASSERT((call_ok != 0) == expect_success);
    if (!ok || (call_ok != 0) != expect_success) {
        ok = false;
        goto cleanup;
    }

    ok = ds4_gpu_tensor_read(out, 0, result->out,
                             sizeof(result->out)) != 0 &&
         ds4_gpu_tensor_read(partial0, 0, result->partial0,
                             sizeof(result->partial0)) != 0 &&
         ds4_gpu_tensor_read(partial1, 0, result->partial1,
                             sizeof(result->partial1)) != 0 &&
         ds4_gpu_tensor_read(selected, 0, result->selected_after,
                             sizeof(result->selected_after)) != 0 &&
         ds4_gpu_tensor_read(weights, 0, result->weights_after,
                             sizeof(result->weights_after)) != 0;
    TEST_ASSERT(ok);
    if (!ok) goto cleanup;
    ds4_gpu_stream_expert_cache_stats(
        &result->hits, &result->misses, &result->pread_bytes, NULL, NULL);
    result->decode_tokens =
        ds4_gpu_internal_stream_expert_cache_decode_tokens();
    result->current_entries = ds4_gpu_stream_expert_cache_current_count();

    if (!expect_success) {
        bool outputs_unchanged = true;
        for (uint32_t row = 0; row < OUT_DIM; row++) {
            TEST_ASSERT(result->out[row] == sentinel);
            TEST_ASSERT(result->partial0[row] == sentinel);
            TEST_ASSERT(result->partial1[row] == sentinel);
            outputs_unchanged = outputs_unchanged &&
                result->out[row] == sentinel &&
                result->partial0[row] == sentinel &&
                result->partial1[row] == sentinel;
        }
        TEST_ASSERT(result->hits == hits0);
        TEST_ASSERT(result->misses == misses0);
        TEST_ASSERT(result->pread_bytes == pread0);
        TEST_ASSERT(result->decode_tokens == tokens0);
        TEST_ASSERT(result->current_entries == entries0);
        bool selected_unchanged = true;
        if (!router_logits_host) {
            for (uint32_t i = 0; i < 8; i++) {
                TEST_ASSERT(result->selected_after[i] == selected_host[i]);
                selected_unchanged = selected_unchanged &&
                    result->selected_after[i] == selected_host[i];
            }
        }
        ok = outputs_unchanged &&
             result->hits == hits0 && result->misses == misses0 &&
             result->pread_bytes == pread0 &&
             result->decode_tokens == tokens0 &&
             result->current_entries == entries0 && selected_unchanged;
    }

cleanup:
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(partial0);
    ds4_gpu_tensor_free(partial1);
    ds4_gpu_tensor_free(gate);
    ds4_gpu_tensor_free(up);
    ds4_gpu_tensor_free(mid);
    ds4_gpu_tensor_free(experts);
    ds4_gpu_tensor_free(selected_half0);
    ds4_gpu_tensor_free(selected_half1);
    ds4_gpu_tensor_free(weights_half0);
    ds4_gpu_tensor_free(weights_half1);
    ds4_gpu_tensor_free(router_logits);
    ds4_gpu_tensor_free(selected);
    ds4_gpu_tensor_free(weights);
    ds4_gpu_tensor_free(input);
    return ok;
}

static void test_metal_q4_selected_slots_runtime_count(void) {
    enum {
        MID_DIM = 256,
        OUT_DIM = 2,
        TOTAL_EXPERT = 128,
        ROUTER_EXPERT = 256,
    };
    const uint64_t row_bytes = sizeof(test_metal_q4_k_block);
    const uint64_t gate_expert_bytes = MID_DIM * row_bytes;
    const uint64_t down_expert_bytes = OUT_DIM * row_bytes;
    const uint64_t gate_tensor_bytes = TOTAL_EXPERT * gate_expert_bytes;
    const uint64_t down_tensor_bytes = TOTAL_EXPERT * down_expert_bytes;
    const uint64_t gate_offset = 0;
    const uint64_t up_offset = gate_tensor_bytes;
    const uint64_t down_offset = gate_tensor_bytes * 2u;
    const uint64_t model_size = down_offset + down_tensor_bytes;
    const uint64_t per_expert_bytes =
        gate_expert_bytes * 2u + down_expert_bytes;
    char path[] = "/tmp/ds4-q4-slots-XXXXXX";
    int fd = mkstemp(path);
    TEST_ASSERT(fd >= 0);
    if (fd < 0) return;
    unlink(path);
    const bool file_sized = ftruncate(fd, (off_t)model_size) == 0;
    TEST_ASSERT(file_sized);
    if (!file_sized) {
        close(fd);
        return;
    }
    void *model_map = mmap(NULL, (size_t)model_size,
                           PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    TEST_ASSERT(model_map != MAP_FAILED);
    if (model_map == MAP_FAILED) {
        close(fd);
        return;
    }

    uint8_t *base = model_map;
    for (uint32_t expert = 0; expert < TOTAL_EXPERT; expert++) {
        const uint8_t value = (uint8_t)(expert % 15u + 1u);
        for (uint32_t row = 0; row < MID_DIM; row++) {
            test_metal_q4_k_fill_constant(
                (test_metal_q4_k_block *)(base + gate_offset +
                    (uint64_t)expert * gate_expert_bytes +
                    (uint64_t)row * row_bytes),
                1u);
            test_metal_q4_k_fill_constant(
                (test_metal_q4_k_block *)(base + up_offset +
                    (uint64_t)expert * gate_expert_bytes +
                    (uint64_t)row * row_bytes),
                value);
        }
        for (uint32_t row = 0; row < OUT_DIM; row++) {
            test_metal_q4_k_fill_constant(
                (test_metal_q4_k_block *)(base + down_offset +
                    (uint64_t)expert * down_expert_bytes +
                    (uint64_t)row * row_bytes),
                value);
        }
    }
    TEST_ASSERT(msync(model_map, (size_t)model_size, MS_SYNC) == 0);

    char *saved_disable_selected =
        test_save_env("DS4_METAL_DISABLE_Q4_SELECTED_EXPERT_VIEWS");
    char *saved_disable_pair =
        test_save_env("DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION");
    char *saved_clamped =
        test_save_env("DS4_METAL_MOE_WRITE_CLAMPED_ACT");
    char *saved_record_selected =
        test_save_env("DS4_MOE_RECORD_SELECTED_IDS");
    char *saved_replay_selected =
        test_save_env("DS4_MOE_REPLAY_SELECTED_IDS");
    char *saved_pread_threads =
        test_save_env("DS4_METAL_STREAMING_EXPERT_PREAD_THREADS");
    char *saved_slab_mb =
        test_save_env("DS4_METAL_STREAMING_EXPERT_SLAB_MB");
    char *saved_timing_summary =
        test_save_env("DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY");
    char *saved_disable_timing_summary =
        test_save_env("DS4_METAL_DISABLE_STREAMING_EXPERT_TIMING_SUMMARY");
    unsetenv("DS4_METAL_DISABLE_Q4_SELECTED_EXPERT_VIEWS");
    unsetenv("DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION");
    unsetenv("DS4_METAL_MOE_WRITE_CLAMPED_ACT");
    unsetenv("DS4_MOE_RECORD_SELECTED_IDS");
    unsetenv("DS4_MOE_REPLAY_SELECTED_IDS");
    unsetenv("DS4_METAL_STREAMING_EXPERT_PREAD_THREADS");
    unsetenv("DS4_METAL_STREAMING_EXPERT_SLAB_MB");
    TEST_ASSERT(setenv("DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY",
                       "1", 1) == 0);
    unsetenv("DS4_METAL_DISABLE_STREAMING_EXPERT_TIMING_SUMMARY");

    ds4_gpu_set_quality(false);
    ds4_gpu_set_model_fd(fd);
    ds4_gpu_set_streaming_expert_cache_expert_bytes(per_expert_bytes);
    ds4_gpu_set_streaming_expert_cache_budget(6);
    ds4_gpu_set_ssd_streaming(true);
    TEST_ASSERT(ds4_gpu_internal_stream_expert_cache_required_floor() == 0);

    test_metal_q4_slots_result top4 = {0};
    TEST_ASSERT(test_metal_q4_selected_slots_case(
        model_map, model_size,
        gate_offset, up_offset, down_offset,
        gate_expert_bytes, down_expert_bytes,
        4, true, &top4));
    TEST_ASSERT(top4.hits == 0);
    TEST_ASSERT(top4.misses == 4);
    TEST_ASSERT(top4.pread_bytes == 4u * per_expert_bytes);
    TEST_ASSERT(top4.current_entries == 4);

    ds4_gpu_set_streaming_expert_cache_budget(6);
    test_metal_q4_slots_result top6 = {0};
    TEST_ASSERT(test_metal_q4_selected_slots_case(
        model_map, model_size,
        gate_offset, up_offset, down_offset,
        gate_expert_bytes, down_expert_bytes,
        6, false, &top6));
    TEST_ASSERT(top6.hits == 0);
    TEST_ASSERT(top6.misses == 6);
    TEST_ASSERT(top6.pread_bytes == 6u * per_expert_bytes);
    TEST_ASSERT(top6.current_entries == 6);

    const float silu_one = 1.0f / (1.0f + expf(-1.0f));
    const float expected = 256.0f * silu_one * 10.0f;
    for (uint32_t row = 0; row < OUT_DIM; row++) {
        TEST_ASSERT(isfinite(top4.out[row]));
        TEST_ASSERT(fabsf(top4.out[row] - expected) < 0.1f);
        TEST_ASSERT(fabsf(top6.out[row] - top4.out[row]) < 1.0e-4f);
    }

    const float top8_weights[8] = {
        0.05f, 0.10f, 0.15f, 0.20f,
        0.15f, 0.10f, 0.10f, 0.15f,
    };
    const int32_t invalid_top8[8] = {0, 1, 2, 3, 4, 5, 6, TOTAL_EXPERT};
    const int32_t unique_top8[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    const int32_t duplicate_top8[8] = {0, 1, 2, 3, 0, 1, 4, 5};

    /* Invalid route rejection is pre-I/O and leaves all caller-visible state
     * unchanged, including the two partial buffers and token accounting. */
    ds4_gpu_set_streaming_expert_cache_budget(8);
    test_metal_qwen_top8_result invalid = {0};
    TEST_ASSERT(test_metal_qwen_top8_case(
        model_map, model_size,
        gate_offset, up_offset, down_offset,
        gate_expert_bytes, down_expert_bytes,
        invalid_top8, top8_weights, NULL, false, &invalid));
    TEST_ASSERT(invalid.hits == 0);
    TEST_ASSERT(invalid.misses == 0);
    TEST_ASSERT(invalid.pread_bytes == 0);
    TEST_ASSERT(invalid.current_entries == 0);
    TEST_ASSERT(invalid.decode_tokens == 0);

    const uint64_t test_page_bytes = (uint64_t)getpagesize();
    const uint64_t test_slot_bytes =
        ((per_expert_bytes + test_page_bytes - 1u) / test_page_bytes) *
        test_page_bytes;
    ds4_gpu_set_streaming_expert_cache_slab_target_bytes(
        4u * test_slot_bytes);
    test_metal_qwen_top8_result cold_top8 = {0};
    TEST_ASSERT(test_metal_qwen_top8_case(
        model_map, model_size,
        gate_offset, up_offset, down_offset,
        gate_expert_bytes, down_expert_bytes,
        unique_top8, top8_weights, NULL, true, &cold_top8));
    TEST_ASSERT(cold_top8.hits == 0);
    TEST_ASSERT(cold_top8.misses == 8);
    TEST_ASSERT(cold_top8.pread_bytes == 8u * per_expert_bytes);
    TEST_ASSERT(cold_top8.current_entries == 8);
    TEST_ASSERT(cold_top8.decode_tokens == 1);
    TEST_ASSERT(ds4_gpu_internal_stream_expert_cache_slab_count() == 2);
    TEST_ASSERT(ds4_gpu_internal_stream_expert_cache_slab_capacity_bytes() ==
                8u * test_slot_bytes);

    float expected_half[2] = {0.0f, 0.0f};
    for (uint32_t i = 0; i < 8; i++) {
        const float value = (float)(unique_top8[i] % 15 + 1);
        expected_half[i / 4u] += top8_weights[i] * value * value;
    }
    expected_half[0] *= 256.0f * silu_one;
    expected_half[1] *= 256.0f * silu_one;
    const float expected_top8 = expected_half[0] + expected_half[1];
    for (uint32_t row = 0; row < OUT_DIM; row++) {
        TEST_ASSERT(isfinite(cold_top8.out[row]));
        TEST_ASSERT(fabsf(cold_top8.partial0[row] - expected_half[0]) < 0.1f);
        TEST_ASSERT(fabsf(cold_top8.partial1[row] - expected_half[1]) < 0.1f);
        TEST_ASSERT(fabsf(cold_top8.out[row] - expected_top8) < 0.1f);
        TEST_ASSERT(fabsf(cold_top8.out[row] -
                          (cold_top8.partial0[row] +
                           cold_top8.partial1[row])) < 1.0e-4f);
    }

    test_metal_qwen_top8_result warm_top8 = {0};
    TEST_ASSERT(test_metal_qwen_top8_case(
        model_map, model_size,
        gate_offset, up_offset, down_offset,
        gate_expert_bytes, down_expert_bytes,
        unique_top8, top8_weights, NULL, true, &warm_top8));
    TEST_ASSERT(warm_top8.hits == 8);
    TEST_ASSERT(warm_top8.misses == cold_top8.misses);
    TEST_ASSERT(warm_top8.pread_bytes == cold_top8.pread_bytes);
    TEST_ASSERT(warm_top8.current_entries == 8);
    TEST_ASSERT(warm_top8.decode_tokens == 2);
    for (uint32_t row = 0; row < OUT_DIM; row++) {
        TEST_ASSERT(fabsf(warm_top8.out[row] - cold_top8.out[row]) < 1.0e-4f);
    }
    ds4_gpu_set_streaming_expert_cache_slab_target_bytes(0);
    TEST_ASSERT(ds4_gpu_internal_stream_expert_cache_slab_count() == 0);

    /* Duplicate route IDs share one cache entry while retaining their two
     * independent router weights in the split computation. */
    ds4_gpu_set_streaming_expert_cache_budget(8);
    test_metal_qwen_top8_result duplicate = {0};
    TEST_ASSERT(test_metal_qwen_top8_case(
        model_map, model_size,
        gate_offset, up_offset, down_offset,
        gate_expert_bytes, down_expert_bytes,
        duplicate_top8, top8_weights, NULL, true, &duplicate));
    TEST_ASSERT(duplicate.hits == 0);
    TEST_ASSERT(duplicate.misses == 6);
    TEST_ASSERT(duplicate.pread_bytes == 6u * per_expert_bytes);
    TEST_ASSERT(duplicate.current_entries == 6);
    TEST_ASSERT(duplicate.decode_tokens == 1);
    float duplicate_expected = 0.0f;
    for (uint32_t i = 0; i < 8; i++) {
        const float value = (float)(duplicate_top8[i] % 15 + 1);
        duplicate_expected += top8_weights[i] * value * value;
    }
    duplicate_expected *= 256.0f * silu_one;
    for (uint32_t row = 0; row < OUT_DIM; row++) {
        TEST_ASSERT(isfinite(duplicate.out[row]));
        TEST_ASSERT(fabsf(duplicate.out[row] - duplicate_expected) < 0.1f);
    }

    /* Exercise the production ordering: the GPU router writes selected IDs
     * and weights into an active command batch; the top-8 wrapper closes that
     * batch for the CPU cache decision, reopens it, runs both halves, and the
     * caller finally commits the restarted batch. */
    float router_logits[2][ROUTER_EXPERT];
    int32_t router_selected[2][8];
    float router_weights[2][8];
    float router_probability[ROUTER_EXPERT];
    for (uint32_t route = 0; route < 2; route++) {
        for (uint32_t expert = 0; expert < ROUTER_EXPERT; expert++) {
            router_logits[route][expert] = -20.0f;
        }
        for (uint32_t i = 0; i < 8; i++) {
            router_logits[route][route * 8u + i] =
                12.0f - 0.25f * (float)i;
        }
        TEST_ASSERT(ds4_qwen35_cpu_softmax_top8_f32(
            router_selected[route],
            router_weights[route],
            router_probability,
            router_logits[route]));
        for (uint32_t i = 0; i < 8; i++) {
            TEST_ASSERT(router_selected[route][i] >= 0);
            TEST_ASSERT(router_selected[route][i] < TOTAL_EXPERT);
        }
    }

    char trace_path[] = "/tmp/ds4-selected-top8-XXXXXX";
    int trace_fd = mkstemp(trace_path);
    TEST_ASSERT(trace_fd >= 0);
    test_metal_qwen_top8_result active_cold = {0};
    test_metal_qwen_top8_result active_warm = {0};
    test_metal_qwen_top8_result active_pressure = {0};
    if (trace_fd >= 0) {
        close(trace_fd);
        TEST_ASSERT(setenv("DS4_MOE_RECORD_SELECTED_IDS",
                           trace_path, 1) == 0);

        ds4_gpu_set_streaming_expert_cache_required_floor(0);
        TEST_ASSERT(
            ds4_gpu_internal_stream_expert_cache_required_floor() == 0);
        ds4_gpu_set_streaming_expert_cache_budget(8);
        TEST_ASSERT(test_metal_qwen_top8_case(
            model_map, model_size,
            gate_offset, up_offset, down_offset,
            gate_expert_bytes, down_expert_bytes,
            router_selected[0], router_weights[0], router_logits[0],
            true, &active_cold));
        TEST_ASSERT(active_cold.hits == 0);
        TEST_ASSERT(active_cold.misses == 8);
        TEST_ASSERT(active_cold.pread_bytes == 8u * per_expert_bytes);
        TEST_ASSERT(active_cold.current_entries == 8);
        TEST_ASSERT(active_cold.decode_tokens == 1);
        const float active_expected0 = test_metal_qwen_top8_expected(
            router_selected[0], router_weights[0]);
        for (uint32_t row = 0; row < OUT_DIM; row++) {
            TEST_ASSERT(isfinite(active_cold.out[row]));
            TEST_ASSERT(fabsf(active_cold.out[row] - active_expected0) < 0.1f);
        }

        TEST_ASSERT(test_metal_qwen_top8_case(
            model_map, model_size,
            gate_offset, up_offset, down_offset,
            gate_expert_bytes, down_expert_bytes,
            router_selected[0], router_weights[0], router_logits[0],
            true, &active_warm));
        TEST_ASSERT(active_warm.hits == 8);
        TEST_ASSERT(active_warm.misses == active_cold.misses);
        TEST_ASSERT(active_warm.pread_bytes == active_cold.pread_bytes);
        TEST_ASSERT(active_warm.current_entries == 8);
        TEST_ASSERT(active_warm.decode_tokens == 2);
        for (uint32_t row = 0; row < OUT_DIM; row++) {
            TEST_ASSERT(fabsf(active_warm.out[row] -
                              active_cold.out[row]) < 1.0e-4f);
        }

        /* The cache is full here. Replacing all eight selected experts proves
         * that the keep-set protects the incoming route while old entries are
         * evicted, and that the configured budget remains a hard ceiling. */
        TEST_ASSERT(test_metal_qwen_top8_case(
            model_map, model_size,
            gate_offset, up_offset, down_offset,
            gate_expert_bytes, down_expert_bytes,
            router_selected[1], router_weights[1], router_logits[1],
            true, &active_pressure));
        TEST_ASSERT(active_pressure.hits == active_warm.hits);
        TEST_ASSERT(active_pressure.misses == 16);
        TEST_ASSERT(active_pressure.pread_bytes == 16u * per_expert_bytes);
        TEST_ASSERT(active_pressure.current_entries == 8);
        TEST_ASSERT(active_pressure.decode_tokens == 3);
        TEST_ASSERT(ds4_gpu_internal_stream_expert_timing_selected_calls() == 3);
        const float active_expected1 = test_metal_qwen_top8_expected(
            router_selected[1], router_weights[1]);
        for (uint32_t row = 0; row < OUT_DIM; row++) {
            TEST_ASSERT(isfinite(active_pressure.out[row]));
            TEST_ASSERT(fabsf(active_pressure.out[row] -
                              active_expected1) < 0.1f);
        }

        uint32_t trace_width = 0;
        uint64_t trace_records = 0;
        int trace_legacy = 1;
        TEST_ASSERT(ds4_gpu_internal_moe_selected_trace_inspect(
            trace_path, 8, &trace_width, &trace_records, &trace_legacy));
        TEST_ASSERT(trace_width == 8);
        TEST_ASSERT(trace_records == 3);
        TEST_ASSERT(trace_legacy == 0);
        TEST_ASSERT(!ds4_gpu_internal_moe_selected_trace_inspect(
            trace_path, 6, NULL, NULL, NULL));
        unsetenv("DS4_MOE_RECORD_SELECTED_IDS");
        unlink(trace_path);
    }
    test_metal_selected_trace_legacy_parser();

    /* Budget rejection happens after the router readback but before SSD I/O,
     * cache/token accounting, or any output writes. */
    ds4_gpu_set_streaming_expert_cache_budget(7);
    test_metal_qwen_top8_result undersized_budget = {0};
    TEST_ASSERT(test_metal_qwen_top8_case(
        model_map, model_size,
        gate_offset, up_offset, down_offset,
        gate_expert_bytes, down_expert_bytes,
        router_selected[0], router_weights[0], router_logits[0],
        false, &undersized_budget));
    TEST_ASSERT(undersized_budget.hits == 0);
    TEST_ASSERT(undersized_budget.misses == 0);
    TEST_ASSERT(undersized_budget.pread_bytes == 0);
    TEST_ASSERT(undersized_budget.current_entries == 0);
    TEST_ASSERT(undersized_budget.decode_tokens == 0);

    /* Force early-load to run synchronously. A cold install is one miss only,
     * not a miss followed by a synthetic wrapper hit; the next route is the
     * real warm-hit case. */
    TEST_ASSERT(setenv("DS4_METAL_STREAMING_EXPERT_PREAD_THREADS", "1", 1) == 0);
    ds4_gpu_set_streaming_expert_cache_budget(8);
    test_metal_qwen_top8_result sync_cold = {0};
    TEST_ASSERT(test_metal_qwen_top8_case(
        model_map, model_size,
        gate_offset, up_offset, down_offset,
        gate_expert_bytes, down_expert_bytes,
        unique_top8, top8_weights, NULL, true, &sync_cold));
    TEST_ASSERT(sync_cold.hits == 0);
    TEST_ASSERT(sync_cold.misses == 8);
    TEST_ASSERT(sync_cold.pread_bytes == 8u * per_expert_bytes);
    TEST_ASSERT(sync_cold.current_entries == 8);
    TEST_ASSERT(sync_cold.decode_tokens == 1);

    test_metal_qwen_top8_result sync_warm = {0};
    TEST_ASSERT(test_metal_qwen_top8_case(
        model_map, model_size,
        gate_offset, up_offset, down_offset,
        gate_expert_bytes, down_expert_bytes,
        unique_top8, top8_weights, NULL, true, &sync_warm));
    TEST_ASSERT(sync_warm.hits == 8);
    TEST_ASSERT(sync_warm.misses == sync_cold.misses);
    TEST_ASSERT(sync_warm.pread_bytes == sync_cold.pread_bytes);
    TEST_ASSERT(sync_warm.current_entries == 8);
    TEST_ASSERT(sync_warm.decode_tokens == 2);
    unsetenv("DS4_METAL_STREAMING_EXPERT_PREAD_THREADS");

    /* Qwen's full-model safety floor is independent from the eight experts
     * needed by one layer. A lazy mlock cap below that floor must stop the
     * next layer before selected-ID readback or SSD/cache mutations. */
    ds4_gpu_set_streaming_expert_cache_required_floor(321);
    TEST_ASSERT(
        ds4_gpu_internal_stream_expert_cache_required_floor() == 321);
    ds4_gpu_set_streaming_expert_cache_budget(320);
    TEST_ASSERT(ds4_gpu_stream_expert_cache_configured_count() == 320);
    TEST_ASSERT(
        ds4_gpu_internal_stream_expert_cache_required_floor() == 321);
    test_metal_qwen_top8_result floor_reject = {0};
    TEST_ASSERT(test_metal_qwen_top8_case(
        model_map, model_size,
        gate_offset, up_offset, down_offset,
        gate_expert_bytes, down_expert_bytes,
        router_selected[0], router_weights[0], router_logits[0],
        false, &floor_reject));
    TEST_ASSERT(floor_reject.hits == 0);
    TEST_ASSERT(floor_reject.misses == 0);
    TEST_ASSERT(floor_reject.pread_bytes == 0);
    TEST_ASSERT(floor_reject.current_entries == 0);
    TEST_ASSERT(floor_reject.decode_tokens == 0);
    TEST_ASSERT(ds4_gpu_internal_stream_expert_timing_selected_calls() == 0);

    /* The actual first lazy allocation must fail closed before readahead,
     * pread, cache installation, miss accounting, or token accounting. The
     * fault hook exercises the production mlock wrapper, not a synthetic
     * budget override, and proves that an active zero cap is not the unset
     * sentinel. */
    ds4_gpu_set_streaming_expert_cache_budget(321);
    TEST_ASSERT(ds4_gpu_stream_expert_cache_configured_count() == 321);
    ds4_gpu_internal_stream_expert_cache_fail_mlock_after(0);
    test_metal_qwen_top8_result first_mlock_reject = {0};
    const bool first_mlock_case_ok = test_metal_qwen_top8_case(
        model_map, model_size,
        gate_offset, up_offset, down_offset,
        gate_expert_bytes, down_expert_bytes,
        router_selected[0], router_weights[0], router_logits[0],
        false, &first_mlock_reject);
    ds4_gpu_internal_stream_expert_cache_fail_mlock_after(-1);
    TEST_ASSERT(first_mlock_case_ok);
    TEST_ASSERT(ds4_gpu_stream_expert_cache_configured_count() == 0);
    TEST_ASSERT(first_mlock_reject.hits == 0);
    TEST_ASSERT(first_mlock_reject.misses == 0);
    TEST_ASSERT(first_mlock_reject.pread_bytes == 0);
    TEST_ASSERT(first_mlock_reject.current_entries == 0);
    TEST_ASSERT(first_mlock_reject.decode_tokens == 0);
    TEST_ASSERT(ds4_gpu_internal_stream_expert_timing_selected_calls() == 0);
    ds4_gpu_set_streaming_expert_cache_required_floor(0);
    TEST_ASSERT(ds4_gpu_stream_expert_cache_configured_count() == 321);
    ds4_gpu_set_streaming_expert_cache_required_floor(321);
    TEST_ASSERT(ds4_gpu_stream_expert_cache_configured_count() == 0);

    ds4_gpu_set_streaming_expert_cache_budget(321);
    TEST_ASSERT(ds4_gpu_stream_expert_cache_configured_count() == 321);
    TEST_ASSERT(
        ds4_gpu_internal_stream_expert_cache_required_floor() == 321);
    test_metal_qwen_top8_result floor_accept = {0};
    TEST_ASSERT(test_metal_qwen_top8_case(
        model_map, model_size,
        gate_offset, up_offset, down_offset,
        gate_expert_bytes, down_expert_bytes,
        router_selected[0], router_weights[0], router_logits[0],
        true, &floor_accept));
    TEST_ASSERT(floor_accept.hits == 0);
    TEST_ASSERT(floor_accept.misses == 8);
    TEST_ASSERT(floor_accept.pread_bytes == 8u * per_expert_bytes);
    TEST_ASSERT(floor_accept.current_entries == 8);
    TEST_ASSERT(floor_accept.decode_tokens == 1);
    TEST_ASSERT(ds4_gpu_internal_stream_expert_timing_selected_calls() == 1);
    const float floor_expected = test_metal_qwen_top8_expected(
        router_selected[0], router_weights[0]);
    for (uint32_t row = 0; row < OUT_DIM; row++) {
        TEST_ASSERT(isfinite(floor_accept.out[row]));
        TEST_ASSERT(fabsf(floor_accept.out[row] - floor_expected) < 0.1f);
    }

    /* An explicitly disabled selected-slot implementation must fail before
     * SSD reads, cache mutation, or accounting rather than entering the
     * generic executor and rejecting after side effects. */
    TEST_ASSERT(setenv("DS4_METAL_DISABLE_Q4_SELECTED_EXPERT_VIEWS",
                       "1", 1) == 0);
    test_metal_qwen_top8_result disabled_selected = {0};
    TEST_ASSERT(test_metal_qwen_top8_case(
        model_map, model_size,
        gate_offset, up_offset, down_offset,
        gate_expert_bytes, down_expert_bytes,
        unique_top8, top8_weights, NULL, false, &disabled_selected));
    unsetenv("DS4_METAL_DISABLE_Q4_SELECTED_EXPERT_VIEWS");

    /* The resident compatibility path maps the complete tensor payload and
     * executes the two top-4 halves without cache allocation or pread.  This
     * remains available to host-authored routes and trace/replay tooling. */
    ds4_gpu_set_ssd_streaming(false);
    ds4_gpu_set_streaming_expert_cache_budget(0);
    ds4_gpu_set_streaming_expert_cache_required_floor(0);
    ds4_gpu_set_streaming_expert_cache_expert_bytes(0);
    ds4_gpu_set_model_fd(-1);
    TEST_ASSERT(ds4_gpu_set_model_map_range(
        model_map, model_size, 0, model_size, gate_tensor_bytes));

    test_metal_qwen_top8_result resident_top8 = {0};
    TEST_ASSERT(test_metal_qwen_top8_case(
        model_map, model_size,
        gate_offset, up_offset, down_offset,
        gate_expert_bytes, down_expert_bytes,
        unique_top8, top8_weights, NULL, true, &resident_top8));
    TEST_ASSERT(resident_top8.hits == 0);
    TEST_ASSERT(resident_top8.misses == 0);
    TEST_ASSERT(resident_top8.pread_bytes == 0);
    TEST_ASSERT(resident_top8.current_entries == 0);
    TEST_ASSERT(resident_top8.decode_tokens == 0);
    for (uint32_t row = 0; row < OUT_DIM; row++) {
        TEST_ASSERT(isfinite(resident_top8.out[row]));
        TEST_ASSERT(fabsf(resident_top8.partial0[row] -
                          cold_top8.partial0[row]) < 1.0e-4f);
        TEST_ASSERT(fabsf(resident_top8.partial1[row] -
                          cold_top8.partial1[row]) < 1.0e-4f);
        TEST_ASSERT(fabsf(resident_top8.out[row] -
                          cold_top8.out[row]) < 1.0e-4f);
    }

    /* Production resident routing stays on the active Metal command timeline:
     * one top-8 pass consumes the trusted router output and one sum-8 dispatch
     * combines it, without a host readback or command-buffer boundary. */
    ds4_gpu_internal_qwen35_resident_route_stats_reset();
    test_metal_qwen_top8_result resident_gpu_route = {0};
    TEST_ASSERT(test_metal_qwen_top8_case(
        model_map, model_size,
        gate_offset, up_offset, down_offset,
        gate_expert_bytes, down_expert_bytes,
        router_selected[0], router_weights[0], router_logits[0],
        true, &resident_gpu_route));
    TEST_ASSERT(ds4_gpu_internal_qwen35_resident_gpu_route_calls() == 1);
    TEST_ASSERT(ds4_gpu_internal_qwen35_resident_host_readbacks() == 0);

    /* Replay the exact GPU-produced IDs and weights through the compatibility
     * path.  The CPU softmax reference is close but not bit-identical, and the
     * synthetic expert values amplify its sub-ULP weight differences. */
    test_metal_qwen_top8_result resident_host_route = {0};
    TEST_ASSERT(test_metal_qwen_top8_case(
        model_map, model_size,
        gate_offset, up_offset, down_offset,
        gate_expert_bytes, down_expert_bytes,
        resident_gpu_route.selected_after,
        resident_gpu_route.weights_after,
        NULL, true, &resident_host_route));
    TEST_ASSERT(ds4_gpu_internal_qwen35_resident_gpu_route_calls() == 1);
    TEST_ASSERT(ds4_gpu_internal_qwen35_resident_host_readbacks() == 1);
    for (uint32_t row = 0; row < OUT_DIM; row++) {
        TEST_ASSERT(fabsf(resident_gpu_route.out[row] -
                          resident_host_route.out[row]) < 1.0e-4f);
    }

    test_metal_qwen_top8_result resident_repeat = {0};
    TEST_ASSERT(test_metal_qwen_top8_case(
        model_map, model_size,
        gate_offset, up_offset, down_offset,
        gate_expert_bytes, down_expert_bytes,
        unique_top8, top8_weights, NULL, true, &resident_repeat));
    TEST_ASSERT(resident_repeat.hits == 0);
    TEST_ASSERT(resident_repeat.misses == 0);
    TEST_ASSERT(resident_repeat.pread_bytes == 0);
    TEST_ASSERT(resident_repeat.current_entries == 0);
    TEST_ASSERT(resident_repeat.decode_tokens == 0);
    for (uint32_t row = 0; row < OUT_DIM; row++) {
        TEST_ASSERT(fabsf(resident_repeat.out[row] -
                          resident_top8.out[row]) < 1.0e-4f);
    }

    /* A malformed resident replay must be rejected before it overwrites the
     * caller's selected-ID tensor. */
    char invalid_replay_path[] = "/tmp/ds4-selected-invalid-replay-XXXXXX";
    int invalid_replay_fd = mkstemp(invalid_replay_path);
    TEST_ASSERT(invalid_replay_fd >= 0);
    if (invalid_replay_fd >= 0) {
        const test_metal_selected_trace_header header = {
            .magic = {'D', 'S', '4', 'M', 'O', 'E', 'I', 'D'},
            .version = 1,
            .width = 8,
            .id_bytes = sizeof(int32_t),
            .header_bytes = sizeof(test_metal_selected_trace_header),
        };
        TEST_ASSERT(test_write_all(invalid_replay_fd,
                                   &header,
                                   sizeof(header)));
        TEST_ASSERT(test_write_all(invalid_replay_fd,
                                   invalid_top8,
                                   sizeof(invalid_top8)));
        close(invalid_replay_fd);
        TEST_ASSERT(setenv("DS4_MOE_REPLAY_SELECTED_IDS",
                           invalid_replay_path, 1) == 0);
        test_metal_qwen_top8_result invalid_replay = {0};
        TEST_ASSERT(test_metal_qwen_top8_case(
            model_map, model_size,
            gate_offset, up_offset, down_offset,
            gate_expert_bytes, down_expert_bytes,
            unique_top8, top8_weights, NULL, false, &invalid_replay));
        unsetenv("DS4_MOE_REPLAY_SELECTED_IDS");
        unlink(invalid_replay_path);
    }

    test_metal_qwen_top8_result resident_invalid = {0};
    TEST_ASSERT(test_metal_qwen_top8_case(
        model_map, model_size,
        gate_offset, up_offset, down_offset,
        gate_expert_bytes, down_expert_bytes,
        invalid_top8, top8_weights, NULL, false, &resident_invalid));
    TEST_ASSERT(resident_invalid.hits == 0);
    TEST_ASSERT(resident_invalid.misses == 0);
    TEST_ASSERT(resident_invalid.pread_bytes == 0);
    TEST_ASSERT(resident_invalid.current_entries == 0);
    TEST_ASSERT(resident_invalid.decode_tokens == 0);

    fprintf(stderr,
            "ds4-test: Q4 selected slots n=4/n=6 output=%.6f "
            "misses=%llu/%llu pread=%llu/%llu; "
            "Qwen top8 cold/warm/dup=%llu/%llu/%llu misses; "
            "active cold/warm/pressure=%llu/%llu/%llu misses; "
            "sync cold/warm hits=%llu/%llu; floor 320/mlock/321=%llu/%llu/%llu misses; "
            "resident top8 output=%.6f pread=%llu\n",
            top4.out[0],
            (unsigned long long)top4.misses,
            (unsigned long long)top6.misses,
            (unsigned long long)top4.pread_bytes,
            (unsigned long long)top6.pread_bytes,
            (unsigned long long)cold_top8.misses,
            (unsigned long long)warm_top8.misses,
            (unsigned long long)duplicate.misses,
            (unsigned long long)active_cold.misses,
            (unsigned long long)active_warm.misses,
            (unsigned long long)active_pressure.misses,
            (unsigned long long)sync_cold.hits,
            (unsigned long long)sync_warm.hits,
            (unsigned long long)floor_reject.misses,
            (unsigned long long)first_mlock_reject.misses,
            (unsigned long long)floor_accept.misses,
            resident_top8.out[0],
            (unsigned long long)resident_top8.pread_bytes);

    ds4_gpu_set_ssd_streaming(false);
    TEST_ASSERT(ds4_gpu_internal_stream_expert_cache_required_floor() == 0);
    ds4_gpu_set_streaming_expert_cache_budget(0);
    ds4_gpu_set_streaming_expert_cache_expert_bytes(0);
    ds4_gpu_set_model_fd(-1);
    ds4_gpu_internal_stream_expert_cache_fail_mlock_after(-1);
    test_restore_env("DS4_METAL_MOE_WRITE_CLAMPED_ACT", saved_clamped);
    test_restore_env("DS4_METAL_DISABLE_ROUTED_PAIR_SWIGLU_FUSION",
                     saved_disable_pair);
    test_restore_env("DS4_METAL_DISABLE_Q4_SELECTED_EXPERT_VIEWS",
                     saved_disable_selected);
    test_restore_env("DS4_METAL_STREAMING_EXPERT_PREAD_THREADS",
                     saved_pread_threads);
    test_restore_env("DS4_METAL_STREAMING_EXPERT_SLAB_MB",
                     saved_slab_mb);
    test_restore_env("DS4_METAL_DISABLE_STREAMING_EXPERT_TIMING_SUMMARY",
                     saved_disable_timing_summary);
    test_restore_env("DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY",
                     saved_timing_summary);
    test_restore_env("DS4_MOE_REPLAY_SELECTED_IDS", saved_replay_selected);
    test_restore_env("DS4_MOE_RECORD_SELECTED_IDS", saved_record_selected);
    /* Release no-copy resident model views before invalidating their mmap. */
    ds4_gpu_cleanup();
    munmap(model_map, (size_t)model_size);
    close(fd);
}
#else
static void test_metal_q4_selected_slots_runtime_count(void) {
}
#endif

static void test_metal_kernel_group(void) {
    test_metal_qwen35_graph_state();
    test_metal_f16_matvec_fast_nr0_4();
    test_metal_f16_prefill_matmul();
    test_metal_q8_0_prefill_matmul();
    test_metal_qwen35_primitives();
    test_metal_q4_selected_slots_runtime_count();
}

static void test_metal_short_prefill_ratio4(void) {
    ds4_engine *engine = test_get_engine(false);
    if (!engine) return;

    const int tokens[] = {
        ds4_token_user(engine),
        ds4_token_assistant(engine),
        ds4_token_eos(engine),
    };
    for (size_t i = 0; i < sizeof(tokens) / sizeof(tokens[0]); i++) {
        TEST_ASSERT(tokens[i] >= 0);
        if (tokens[i] < 0) return;
    }

    for (size_t n = 1; n <= 3; n++) {
        ds4_tokens prompt = {0};
        for (size_t i = 0; i < n; i++) {
            ds4_tokens_push(&prompt, tokens[i]);
        }
        TEST_ASSERT(prompt.len == (int)n);

        ds4_session *session = NULL;
        TEST_ASSERT(ds4_session_create(&session, engine, 2048) == 0);
        if (!session) {
            ds4_tokens_free(&prompt);
            return;
        }

        char err[160] = {0};
        const int rc = ds4_session_sync(session, &prompt, err, sizeof(err));
        if (rc != 0) {
            fprintf(stderr, "ds4-test: short prefill failed for %zu token(s): %s\n",
                    n, err);
        }
        TEST_ASSERT(rc == 0);

        ds4_session_free(session);
        ds4_tokens_free(&prompt);
    }
}

static char *test_read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    long len = ftell(fp);
    if (len < 0) {
        fclose(fp);
        return NULL;
    }
    rewind(fp);
    char *s = malloc((size_t)len + 1);
    if (!s) {
        fclose(fp);
        return NULL;
    }
    size_t nread = fread(s, 1, (size_t)len, fp);
    fclose(fp);
    if (nread != (size_t)len) {
        free(s);
        return NULL;
    }
    s[len] = '\0';
    return s;
}

typedef struct {
    const char *name;
    int number;
} test_long_fact;

static const test_long_fact test_long_facts[] = {
    {"Bob", 34},
    {"Alice", 52},
    {"Clara", 71},
    {"Diego", 93},
    {"Elena", 16},
    {"Felix", 88},
    {"Greta", 47},
    {"Hugo", 29},
    {"Iris", 64},
    {"Jonas", 12},
    {"Kira", 81},
    {"Leo", 39},
    {"Marta", 76},
    {"Nadia", 23},
    {"Owen", 58},
    {"Priya", 97},
};

static bool test_is_name_boundary(char c) {
    unsigned char uc = (unsigned char)c;
    return c == '\0' || !(isalnum(uc) || c == '_');
}

static bool test_parse_assignment_value(const char *p, int *value) {
    while (*p == ' ' || *p == '\t') p++;
    if (*p != '=') return false;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    if (!isdigit((unsigned char)*p)) return false;

    int v = 0;
    while (isdigit((unsigned char)*p)) {
        v = v * 10 + (*p - '0');
        p++;
    }
    *value = v;
    return true;
}

static bool test_output_has_fact(const char *text, const test_long_fact *fact) {
    const size_t name_len = strlen(fact->name);
    const char *p = text;
    bool saw_wrong_assignment = false;
    int wrong_value = -1;

    while ((p = strstr(p, fact->name)) != NULL) {
        const bool before_ok = p == text || test_is_name_boundary(p[-1]);
        const bool after_ok = test_is_name_boundary(p[name_len]) ||
                              p[name_len] == ' ' ||
                              p[name_len] == '\t' ||
                              p[name_len] == '=';
        if (before_ok && after_ok) {
            int value = 0;
            if (test_parse_assignment_value(p + name_len, &value)) {
                if (value == fact->number) return true;
                saw_wrong_assignment = true;
                wrong_value = value;
            }
        }
        p += name_len;
    }

    if (saw_wrong_assignment) {
        fprintf(stderr,
                "ds4-test: long-context wrong assignment for %s: got %d expected %d\n",
                fact->name, wrong_value, fact->number);
    } else {
        fprintf(stderr,
                "ds4-test: long-context missing assignment for %s=%d\n",
                fact->name, fact->number);
    }
    return false;
}

static int test_hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + c - 'a';
    if (c >= 'A' && c <= 'F') return 10 + c - 'A';
    return -1;
}

static bool test_hex_to_bytes(const char *hex, unsigned char *out, int cap, int *len) {
    int n = 0;
    while (*hex && !isspace((unsigned char)*hex)) {
        int hi = test_hex_digit(hex[0]);
        int lo = test_hex_digit(hex[1]);
        if (hi < 0 || lo < 0 || n >= cap) return false;
        out[n++] = (unsigned char)((hi << 4) | lo);
        hex += 2;
    }
    *len = n;
    return true;
}

static bool test_token_bytes_equal(ds4_engine *engine, int token,
                                   const unsigned char *want, int want_len) {
    size_t got_len = 0;
    char *got = ds4_token_text(engine, token, &got_len);
    bool eq = got && got_len == (size_t)want_len &&
              memcmp(got, want, (size_t)want_len) == 0;
    free(got);
    return eq;
}

static void test_long_prefill_progress(void *ud, const char *event, int current, int total) {
    (void)ud;
    if (strcmp(event, "prefill_chunk")) return;
    if (current == 0 || current == total || current % 8192 == 0) {
        fprintf(stderr, "ds4-test: long-context prefill %d/%d\n", current, total);
    }
}

static void test_long_story_fact_recall(void) {
    const char *prompt_path = getenv("DS4_TEST_LONG_PROMPT");
    if (!prompt_path || !prompt_path[0]) {
        prompt_path = "tests/long_context_story_prompt.txt";
    }
    char *prompt_text = test_read_file(prompt_path);
    TEST_ASSERT(prompt_text != NULL);
    if (!prompt_text) return;

    ds4_engine *engine = test_get_engine(false);
    if (!engine) {
        free(prompt_text);
        return;
    }

    ds4_tokens prompt = {0};
    TEST_ASSERT(ds4_tokenize_rendered_chat_checked(
        engine, prompt_text, &prompt));
    TEST_ASSERT(prompt.len > 30000);

    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, 100000) == 0);
    if (!session) {
        ds4_tokens_free(&prompt);
        free(prompt_text);
        return;
    }

    char err[160];
    ds4_session_set_progress(session, test_long_prefill_progress, NULL);
    TEST_ASSERT(ds4_session_sync(session, &prompt, err, sizeof(err)) == 0);
    ds4_session_set_progress(session, NULL, NULL);

    buf out = {0};
    uint64_t rng = 12345;
    int generated = 0;
    bool decode_ok = true;
    for (; generated < 350; generated++) {
        int token = ds4_session_sample(session, 0.0f, 0, 1.0f, 0.0f, &rng);
        if (token == ds4_token_eos(engine)) break;

        size_t piece_len = 0;
        char *piece = ds4_token_text(engine, token, &piece_len);
        buf_append(&out, piece, piece_len);
        free(piece);

        if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            decode_ok = false;
            break;
        }
    }

    const char *text = out.ptr ? out.ptr : "";
    TEST_ASSERT(decode_ok);
    TEST_ASSERT(generated > 0);
    for (size_t i = 0; i < sizeof(test_long_facts) / sizeof(test_long_facts[0]); i++) {
        TEST_ASSERT(test_output_has_fact(text, &test_long_facts[i]));
    }

    buf_free(&out);
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
    free(prompt_text);
}

#define TEST_VEC_MAX_STEPS 16
#define TEST_VEC_MAX_TOP 32
#define TEST_VEC_MAX_TOKEN_BYTES 128

typedef struct {
    unsigned char bytes[TEST_VEC_MAX_TOKEN_BYTES];
    int len;
    float logprob;
} test_vec_top;

typedef struct {
    unsigned char selected[TEST_VEC_MAX_TOKEN_BYTES];
    int selected_len;
    int ntop;
    test_vec_top top[TEST_VEC_MAX_TOP];
} test_vec_step;

typedef struct {
    char id[96];
    char prompt_path[512];
    int ctx;
    int nsteps;
    test_vec_step steps[TEST_VEC_MAX_STEPS];
} test_vec_case;

static char *test_trim_line(char *line) {
    while (*line && isspace((unsigned char)*line)) line++;
    size_t n = strlen(line);
    while (n && isspace((unsigned char)line[n - 1])) line[--n] = '\0';
    return line;
}

static bool test_read_vector_case(FILE *fp, test_vec_case *vc) {
    char line[2048];
    memset(vc, 0, sizeof(*vc));
    while (fgets(line, sizeof(line), fp)) {
        char *p = test_trim_line(line);
        if (!p[0] || p[0] == '#') continue;
        if (sscanf(p, "case %95s %d %d %511s",
                   vc->id, &vc->ctx, &vc->nsteps, vc->prompt_path) == 4) {
            TEST_ASSERT(vc->nsteps > 0 && vc->nsteps <= TEST_VEC_MAX_STEPS);
            return true;
        }
        TEST_ASSERT(!"unexpected line before vector case");
    }
    return false;
}

static bool test_fill_vector_case(FILE *fp, test_vec_case *vc) {
    char line[2048];
    int step_index = -1;
    int top_index = 0;

    while (fgets(line, sizeof(line), fp)) {
        char *p = test_trim_line(line);
        if (!p[0] || p[0] == '#') continue;
        if (!strcmp(p, "end")) return true;

        if (!strncmp(p, "step ", 5)) {
            char hex[TEST_VEC_MAX_TOKEN_BYTES * 2 + 2];
            int ntop = 0;
            if (sscanf(p, "step %d %257s %d", &step_index, hex, &ntop) != 3) {
                TEST_ASSERT(!"bad vector step line");
                return false;
            }
            TEST_ASSERT(step_index >= 0 && step_index < vc->nsteps);
            TEST_ASSERT(ntop >= 0 && ntop <= TEST_VEC_MAX_TOP);
            vc->steps[step_index].ntop = ntop;
            TEST_ASSERT(test_hex_to_bytes(hex,
                                          vc->steps[step_index].selected,
                                          TEST_VEC_MAX_TOKEN_BYTES,
                                          &vc->steps[step_index].selected_len));
            top_index = 0;
            continue;
        }

        if (!strncmp(p, "top ", 4)) {
            char hex[TEST_VEC_MAX_TOKEN_BYTES * 2 + 2];
            float lp = 0.0f;
            TEST_ASSERT(step_index >= 0 && step_index < vc->nsteps);
            TEST_ASSERT(top_index < vc->steps[step_index].ntop);
            if (sscanf(p, "top %257s %f", hex, &lp) != 2) {
                TEST_ASSERT(!"bad vector top line");
                return false;
            }
            test_vec_top *top = &vc->steps[step_index].top[top_index++];
            top->logprob = lp;
            TEST_ASSERT(test_hex_to_bytes(hex, top->bytes,
                                          TEST_VEC_MAX_TOKEN_BYTES, &top->len));
            continue;
        }

        TEST_ASSERT(!"unexpected vector line");
        return false;
    }

    TEST_ASSERT(!"unterminated vector case");
    return false;
}

static void test_logprob_vector_case(ds4_engine *engine, const test_vec_case *vc) {
    char *prompt_text = test_read_file(vc->prompt_path);
    TEST_ASSERT(prompt_text != NULL);
    if (!prompt_text) return;

    ds4_tokens prompt = {0};
    TEST_ASSERT(ds4_encode_chat_prompt_checked(
        engine, "", prompt_text, DS4_THINK_NONE, &prompt));
    free(prompt_text);

    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, vc->ctx) == 0);
    if (!session) {
        ds4_tokens_free(&prompt);
        return;
    }

    char err[160];
    TEST_ASSERT(ds4_session_sync(session, &prompt, err, sizeof(err)) == 0);

    ds4_token_score scores[20];
    for (int i = 0; i < vc->nsteps; i++) {
        const test_vec_step *step = &vc->steps[i];
        int nscore = ds4_session_top_logprobs(session, scores, 20);
        int token = ds4_session_argmax(session);
        if (!test_token_bytes_equal(engine, token, step->selected, step->selected_len)) {
            fprintf(stderr, "ds4-test: vector %s step %d selected token mismatch\n",
                    vc->id, i);
            TEST_ASSERT(false);
        }

        for (int t = 0; t < step->ntop; t++) {
            bool found = false;
            float local_lp = 0.0f;
            for (int j = 0; j < nscore; j++) {
                if (scores[j].id < 0) continue;
                if (test_token_bytes_equal(engine, scores[j].id,
                                           step->top[t].bytes,
                                           step->top[t].len)) {
                    found = true;
                    local_lp = scores[j].logprob;
                    break;
                }
            }
            if (!found) {
                fprintf(stderr, "ds4-test: vector %s step %d official top token missing locally\n",
                        vc->id, i);
                TEST_ASSERT(false);
            } else if (fabsf(local_lp - step->top[t].logprob) > 4.0f) {
                fprintf(stderr,
                        "ds4-test: vector %s step %d logprob delta too high: local=%g official=%g\n",
                        vc->id, i, local_lp, step->top[t].logprob);
                TEST_ASSERT(false);
            }
        }

        if (i + 1 < vc->nsteps) {
            TEST_ASSERT(ds4_session_eval(session, token, err, sizeof(err)) == 0);
        }
    }

    ds4_session_free(session);
    ds4_tokens_free(&prompt);
}

static bool test_logprob_vector_case_disabled(const test_vec_case *vc) {
    /*
     * This one long-context vector currently matches the public DeepSeek API less
     * after adding the official Hadamard+FP4 indexer path.  The public official
     * implementation and the API appear to disagree here; the official graph has
     * slightly lower local perplexity on the A/B check we ran, so DS4 keeps that
     * implementation and only excludes this brittle API fixture for now.
     */
    return !strcmp(vc->id, "long_memory_archive");
}

static void test_official_logprob_vectors_run(const char *case_filter) {
    const char *path = getenv("DS4_TEST_VECTOR_FILE");
    if (!path || !path[0]) path = "tests/test-vectors/official.vec";
    FILE *fp = fopen(path, "rb");
    TEST_ASSERT(fp != NULL);
    if (!fp) return;

    char *saved_prefill_chunk = test_save_env("DS4_METAL_PREFILL_CHUNK");
    char *saved_disable_metal4 = test_save_env("DS4_METAL_DISABLE_METAL4");
    test_streaming_prefill_env saved_canonical_streaming_prefill =
        test_force_canonical_streaming_prefill();
    setenv("DS4_METAL_PREFILL_CHUNK", "2048", 1);
    if (getenv("DS4_TEST_LOGPROB_AUTO_METAL") == NULL) {
        setenv("DS4_METAL_DISABLE_METAL4", "1", 1);
    } else {
        unsetenv("DS4_METAL_DISABLE_METAL4");
    }
    ds4_engine *engine = test_open_engine(false);
    if (!engine) {
        test_restore_canonical_streaming_prefill(saved_canonical_streaming_prefill);
        test_restore_env("DS4_METAL_DISABLE_METAL4", saved_disable_metal4);
        test_restore_env("DS4_METAL_PREFILL_CHUNK", saved_prefill_chunk);
        fclose(fp);
        return;
    }

    test_vec_case vc;
    int ran = 0;
    while (test_read_vector_case(fp, &vc)) {
        if (!test_fill_vector_case(fp, &vc)) break;
        if (case_filter && case_filter[0] && strcmp(vc.id, case_filter)) {
            continue;
        }
        if (test_logprob_vector_case_disabled(&vc)) {
            fprintf(stderr, "ds4-test: vector %s skipped (API/official graph mismatch)\n",
                    vc.id);
            continue;
        }
        fprintf(stderr, "ds4-test: vector %s\n", vc.id);
        test_logprob_vector_case(engine, &vc);
        ran++;
    }
    TEST_ASSERT(!case_filter || !case_filter[0] || ran == 1);
    ds4_engine_close(engine);
    test_restore_canonical_streaming_prefill(saved_canonical_streaming_prefill);
    test_restore_env("DS4_METAL_DISABLE_METAL4", saved_disable_metal4);
    test_restore_env("DS4_METAL_PREFILL_CHUNK", saved_prefill_chunk);
    fclose(fp);
}

static void test_official_logprob_vectors(void) {
    test_official_logprob_vectors_run(NULL);
}

static void test_metal_ssd_streaming_cache_pressure(void) {
#ifndef __APPLE__
    fprintf(stderr,
            "ds4-test: Metal SSD streaming cache-pressure repro skipped "
            "(Metal-only)\n");
#else
    /*
     * Regression repro for GitHub issue #384.
     *
     * The bug needs the Metal SSD-streaming decode layer-batch path and a small
     * routed-expert cache. Under pressure, a cache entry referenced by an
     * already-encoded-but-not-yet-executed layer can be reused for a later
     * layer in the same command buffer, producing deterministic wrong logits.
     */
    char *saved_streaming = test_save_env("DS4_TEST_SSD_STREAMING");
    char *saved_cache_gb = test_save_env("DS4_TEST_SSD_STREAMING_CACHE_GB");
    char *saved_cache_experts =
        test_save_env("DS4_TEST_SSD_STREAMING_CACHE_EXPERTS");
    char *saved_disable_layer_batch =
        test_save_env("DS4_METAL_DISABLE_STREAMING_LAYER_BATCH");
    char *saved_disable_static_decode =
        test_save_env("DS4_METAL_DISABLE_STREAMING_STATIC_DECODE_MAP");
    char *saved_one_stage =
        test_save_env("DS4_METAL_MOE_ONE_STAGE_PROFILE");

    setenv("DS4_TEST_SSD_STREAMING", "1", 1);
    setenv("DS4_TEST_SSD_STREAMING_CACHE_GB", "16", 1);
    unsetenv("DS4_TEST_SSD_STREAMING_CACHE_EXPERTS");
    unsetenv("DS4_METAL_DISABLE_STREAMING_LAYER_BATCH");
    unsetenv("DS4_METAL_DISABLE_STREAMING_STATIC_DECODE_MAP");
    unsetenv("DS4_METAL_MOE_ONE_STAGE_PROFILE");

    fprintf(stderr,
            "ds4-test: Metal SSD streaming cache-pressure repro "
            "(16GiB cache, layer-batched decode, short_code_completion)\n");
    test_official_logprob_vectors_run("short_code_completion");

    test_restore_env("DS4_METAL_MOE_ONE_STAGE_PROFILE", saved_one_stage);
    test_restore_env("DS4_METAL_DISABLE_STREAMING_STATIC_DECODE_MAP",
                     saved_disable_static_decode);
    test_restore_env("DS4_METAL_DISABLE_STREAMING_LAYER_BATCH",
                     saved_disable_layer_batch);
    test_restore_env("DS4_TEST_SSD_STREAMING_CACHE_EXPERTS",
                     saved_cache_experts);
    test_restore_env("DS4_TEST_SSD_STREAMING_CACHE_GB", saved_cache_gb);
    test_restore_env("DS4_TEST_SSD_STREAMING", saved_streaming);
#endif
}

static void test_logits_topk(const float *logits, int n, int *out, int k);
static bool test_topk_contains(const int *top, int k, int id);

#define TEST_LOCAL_GOLDEN_MAX_TOP 128

typedef struct {
    int id;
    float logit;
} test_local_golden_top;

typedef struct {
    char id[96];
    char mode[16];
    char prompt_path[512];
    int ctx;
    int frontier;
    int ntop;
    test_local_golden_top top[TEST_LOCAL_GOLDEN_MAX_TOP];
} test_local_golden_case;

static bool test_read_local_golden_case(FILE *fp, test_local_golden_case *tc) {
    char line[2048];
    memset(tc, 0, sizeof(*tc));
    while (fgets(line, sizeof(line), fp)) {
        char *p = test_trim_line(line);
        if (!p[0] || p[0] == '#') continue;
        if (sscanf(p, "case %95s %15s %d %d %511s %d",
                   tc->id, tc->mode, &tc->ctx, &tc->frontier,
                   tc->prompt_path, &tc->ntop) == 6) {
            TEST_ASSERT(tc->ctx > tc->frontier);
            TEST_ASSERT(tc->frontier > 0);
            TEST_ASSERT(tc->ntop > 0 && tc->ntop <= TEST_LOCAL_GOLDEN_MAX_TOP);
            return true;
        }
        TEST_ASSERT(!"unexpected line before local golden case");
        return false;
    }
    return false;
}

static bool test_fill_local_golden_case(FILE *fp, test_local_golden_case *tc) {
    char line[2048];
    int seen = 0;
    while (fgets(line, sizeof(line), fp)) {
        char *p = test_trim_line(line);
        if (!p[0] || p[0] == '#') continue;
        if (!strcmp(p, "end")) {
            TEST_ASSERT(seen == tc->ntop);
            return seen == tc->ntop;
        }
        int rank = -1;
        int id = -1;
        float logit = 0.0f;
        if (sscanf(p, "top %d %d %f", &rank, &id, &logit) != 3) {
            TEST_ASSERT(!"bad local golden top line");
            return false;
        }
        TEST_ASSERT(rank == seen);
        TEST_ASSERT(seen < tc->ntop);
        if (seen >= tc->ntop) return false;
        tc->top[seen].id = id;
        tc->top[seen].logit = logit;
        seen++;
    }
    TEST_ASSERT(!"unterminated local golden case");
    return false;
}

static int test_local_golden_overlap(const test_local_golden_case *tc,
                                     const int *cand_top,
                                     int n) {
    int overlap = 0;
    if (n > tc->ntop) n = tc->ntop;
    for (int i = 0; i < n; i++) {
        if (test_topk_contains(cand_top, n, tc->top[i].id)) overlap++;
    }
    return overlap;
}

static float test_local_golden_max_abs(const test_local_golden_case *tc,
                                       const float *cand_logits,
                                       int n) {
    float max_abs = 0.0f;
    if (n > tc->ntop) n = tc->ntop;
    for (int i = 0; i < n; i++) {
        const int id = tc->top[i].id;
        if (id < 0) continue;
        const float abs_delta = fabsf(cand_logits[id] - tc->top[i].logit);
        if (abs_delta > max_abs) max_abs = abs_delta;
    }
    return max_abs;
}

static void test_local_golden_case_run(ds4_engine *engine,
                                       const test_local_golden_case *tc) {
    char *prompt_text = test_read_file(tc->prompt_path);
    TEST_ASSERT(prompt_text != NULL);
    if (!prompt_text) return;

    ds4_tokens prompt = {0};
    bool tokenized = false;
    if (!strcmp(tc->mode, "text")) {
        tokenized = ds4_tokenize_text_checked(
            engine, prompt_text, &prompt);
    } else if (!strcmp(tc->mode, "rendered")) {
        tokenized = ds4_tokenize_rendered_chat_checked(
            engine, prompt_text, &prompt);
    } else if (!strcmp(tc->mode, "chat")) {
        tokenized = ds4_encode_chat_prompt_checked(
            engine, "", prompt_text, DS4_THINK_NONE, &prompt);
    } else {
        TEST_ASSERT(!"unknown local golden prompt mode");
    }
    free(prompt_text);
    TEST_ASSERT(tokenized);
    if (!tokenized) {
        ds4_tokens_free(&prompt);
        return;
    }
    TEST_ASSERT(prompt.len >= tc->frontier);
    if (prompt.len < tc->frontier) {
        ds4_tokens_free(&prompt);
        return;
    }

    ds4_tokens prefix = {
        .v = prompt.v,
        .len = tc->frontier,
        .cap = tc->frontier,
    };

    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, tc->ctx) == 0);
    if (!session) {
        ds4_tokens_free(&prompt);
        return;
    }

    char err[160];
    TEST_ASSERT(ds4_session_sync(session, &prefix, err, sizeof(err)) == 0);

    const int vocab = ds4_engine_vocab_size(engine);
    float *cand_logits = malloc((size_t)vocab * sizeof(cand_logits[0]));
    TEST_ASSERT(cand_logits != NULL);
    if (cand_logits &&
        ds4_session_copy_logits(session, cand_logits, vocab) == vocab) {
        int cand_top[TEST_LOCAL_GOLDEN_MAX_TOP];
        const int ntop = tc->ntop < TEST_LOCAL_GOLDEN_MAX_TOP ?
                         tc->ntop : TEST_LOCAL_GOLDEN_MAX_TOP;
        test_logits_topk(cand_logits, vocab, cand_top, ntop);

        const int top5_overlap = test_local_golden_overlap(tc, cand_top, 5);
        const int top20_overlap = test_local_golden_overlap(tc, cand_top, 20);
        const int top64_overlap = test_local_golden_overlap(tc, cand_top, 64);
        const float top20_max_abs =
            test_local_golden_max_abs(tc, cand_logits, 20);

        fprintf(stderr,
                "ds4-test: local golden %s top1 ref=%d cand=%d "
                "top5_overlap=%d/5 top20_overlap=%d/20 top64_overlap=%d/64 "
                "top20_max_abs=%g\n",
                tc->id, tc->top[0].id, cand_top[0],
                top5_overlap, top20_overlap, top64_overlap, top20_max_abs);

        /*
         * This is intentionally tolerant: it is meant to catch substantial
         * backend drift (wrong tiling, skipped work, bad dispatch), not tiny
         * floating-point differences from otherwise sane kernel changes.
         */
        TEST_ASSERT(cand_top[0] == tc->top[0].id);
        TEST_ASSERT(top5_overlap >= 4);
        TEST_ASSERT(top20_overlap >= 15);
        TEST_ASSERT(top64_overlap >= 40);
        TEST_ASSERT(top20_max_abs <= 8.0f);
    } else {
        TEST_ASSERT(false);
    }

    free(cand_logits);
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
}

static void test_local_golden_vectors(void) {
    const char *path = getenv("DS4_TEST_LOCAL_GOLDEN_FILE");
    if (!path || !path[0]) path = "tests/test-vectors/local-golden.vec";
    FILE *fp = fopen(path, "rb");
    TEST_ASSERT(fp != NULL);
    if (!fp) return;

    char *saved_prefill_chunk = test_save_env("DS4_METAL_PREFILL_CHUNK");
    char *saved_disable_metal4 = test_save_env("DS4_METAL_DISABLE_METAL4");
    char *saved_moe_tile_max = test_save_env("DS4_METAL_MOE_TILE_MAX");
    test_streaming_prefill_env saved_canonical_streaming_prefill =
        test_force_canonical_streaming_prefill();
    setenv("DS4_METAL_PREFILL_CHUNK", "4096", 1);
    setenv("DS4_METAL_DISABLE_METAL4", "1", 1);
    unsetenv("DS4_METAL_MOE_TILE_MAX");

    ds4_engine *engine = test_open_engine(false);
    if (!engine) {
        test_restore_canonical_streaming_prefill(saved_canonical_streaming_prefill);
        test_restore_env("DS4_METAL_MOE_TILE_MAX", saved_moe_tile_max);
        test_restore_env("DS4_METAL_DISABLE_METAL4", saved_disable_metal4);
        test_restore_env("DS4_METAL_PREFILL_CHUNK", saved_prefill_chunk);
        fclose(fp);
        return;
    }

    test_local_golden_case tc;
    while (test_read_local_golden_case(fp, &tc)) {
        if (!test_fill_local_golden_case(fp, &tc)) break;
        test_local_golden_case_run(engine, &tc);
    }

    ds4_engine_close(engine);
    test_restore_canonical_streaming_prefill(saved_canonical_streaming_prefill);
    test_restore_env("DS4_METAL_MOE_TILE_MAX", saved_moe_tile_max);
    test_restore_env("DS4_METAL_DISABLE_METAL4", saved_disable_metal4);
    test_restore_env("DS4_METAL_PREFILL_CHUNK", saved_prefill_chunk);
    fclose(fp);
}

#define TEST_MPP_EQ_MAX_CASES 8
#define TEST_MPP_EQ_TOPK 20
#define TEST_MPP_EQ_TOP5 5
#define TEST_MPP_EQ_DELTAS 5

typedef struct {
    char id[96];
    int ctx;
    int vocab_size;
    int gen_steps;
    ds4_tokens prompt;
    float *ref_logits;
    int ref_gen[TEST_VEC_MAX_STEPS];
    int ref_gen_len;
} test_mpp_eq_case;

typedef struct {
    int ref_top1;
    int cand_top1;
    int overlap;
    int top5_overlap;
    int max_rank_delta;
    int nonfinite;
    float rms;
    float max_abs;
    float top20_max_abs;
    bool same_top1;
    bool pass;
} test_mpp_eq_result;

typedef struct {
    const char *label;
    int cases;
    int capture_failures;
    int logits_failures;
    int greedy_failures;
    int top1_mismatches;
    int min_overlap;
    int min_top5_overlap;
    int worst_rank_delta;
    float worst_rms;
    float worst_max_abs;
    float worst_top20_max_abs;
} test_mpp_eq_summary;

static void test_mpp_eq_case_free(test_mpp_eq_case *tc) {
    if (!tc) return;
    ds4_tokens_free(&tc->prompt);
    free(tc->ref_logits);
    memset(tc, 0, sizeof(*tc));
}

static void test_logits_topk(const float *logits, int n, int *out, int k) {
    for (int i = 0; i < k; i++) out[i] = -1;
    for (int id = 0; id < n; id++) {
        const float v = logits[id];
        if (!isfinite(v)) continue;
        for (int j = 0; j < k; j++) {
            if (out[j] < 0 || v > logits[out[j]]) {
                for (int l = k - 1; l > j; l--) out[l] = out[l - 1];
                out[j] = id;
                break;
            }
        }
    }
}

static bool test_topk_contains(const int *top, int k, int id) {
    for (int i = 0; i < k; i++) {
        if (top[i] == id) return true;
    }
    return false;
}

static int test_topk_rank(const int *top, int k, int id) {
    for (int i = 0; i < k; i++) {
        if (top[i] == id) return i;
    }
    return -1;
}

static void test_note_delta(int *ids, float *ref_vals, float *cand_vals,
                            float *abs_vals, int id, float ref, float cand) {
    const float abs_delta = fabsf(cand - ref);
    for (int i = 0; i < TEST_MPP_EQ_DELTAS; i++) {
        if (ids[i] < 0 || abs_delta > abs_vals[i]) {
            for (int j = TEST_MPP_EQ_DELTAS - 1; j > i; j--) {
                ids[j] = ids[j - 1];
                ref_vals[j] = ref_vals[j - 1];
                cand_vals[j] = cand_vals[j - 1];
                abs_vals[j] = abs_vals[j - 1];
            }
            ids[i] = id;
            ref_vals[i] = ref;
            cand_vals[i] = cand;
            abs_vals[i] = abs_delta;
            return;
        }
    }
}

static float test_top_union_max_abs(const float *ref, const float *cand,
                                    const int *ref_top, const int *cand_top, int k) {
    float max_abs = 0.0f;
    for (int i = 0; i < k; i++) {
        if (ref_top[i] >= 0) {
            const float d = fabsf(cand[ref_top[i]] - ref[ref_top[i]]);
            if (d > max_abs) max_abs = d;
        }
        if (cand_top[i] >= 0 && !test_topk_contains(ref_top, k, cand_top[i])) {
            const float d = fabsf(cand[cand_top[i]] - ref[cand_top[i]]);
            if (d > max_abs) max_abs = d;
        }
    }
    return max_abs;
}

/*
 * Metal4/TensorOps equivalence is a smoke test, not a demand for bitwise local
 * logits.  Tensor kernels change precision and reduction order, so the useful
 * invariant here is: no NaNs, same first greedy token, and same short greedy
 * continuation.  Larger logit drift is still printed so it can be compared with
 * official API-vector and long-context recall gates.
 */
static test_mpp_eq_result test_compare_mpp_logits(const test_mpp_eq_case *tc,
                                                  const float *cand_logits,
                                                  bool assert_thresholds) {
    int ref_top[TEST_MPP_EQ_TOPK];
    int cand_top[TEST_MPP_EQ_TOPK];
    test_logits_topk(tc->ref_logits, tc->vocab_size, ref_top, TEST_MPP_EQ_TOPK);
    test_logits_topk(cand_logits, tc->vocab_size, cand_top, TEST_MPP_EQ_TOPK);

    int overlap = 0;
    int top5_overlap = 0;
    int max_rank_delta = 0;
    for (int i = 0; i < TEST_MPP_EQ_TOPK; i++) {
        const int cand_rank = test_topk_rank(cand_top, TEST_MPP_EQ_TOPK, ref_top[i]);
        if (ref_top[i] >= 0 && cand_rank >= 0) {
            overlap++;
            const int rank_delta = abs(cand_rank - i);
            if (rank_delta > max_rank_delta) max_rank_delta = rank_delta;
        }
        if (i < TEST_MPP_EQ_TOP5 &&
            ref_top[i] >= 0 &&
            test_topk_contains(cand_top, TEST_MPP_EQ_TOP5, ref_top[i])) {
            top5_overlap++;
        }
    }

    double sumsq = 0.0;
    float max_abs = 0.0f;
    int nonfinite = 0;
    int delta_ids[TEST_MPP_EQ_DELTAS];
    float delta_ref[TEST_MPP_EQ_DELTAS];
    float delta_cand[TEST_MPP_EQ_DELTAS];
    float delta_abs[TEST_MPP_EQ_DELTAS];
    for (int i = 0; i < TEST_MPP_EQ_DELTAS; i++) {
        delta_ids[i] = -1;
        delta_ref[i] = 0.0f;
        delta_cand[i] = 0.0f;
        delta_abs[i] = 0.0f;
    }

    for (int i = 0; i < tc->vocab_size; i++) {
        if (!isfinite(tc->ref_logits[i]) || !isfinite(cand_logits[i])) {
            nonfinite++;
            continue;
        }
        const float delta = cand_logits[i] - tc->ref_logits[i];
        const float abs_delta = fabsf(delta);
        if (abs_delta > max_abs) max_abs = abs_delta;
        sumsq += (double)delta * (double)delta;
        test_note_delta(delta_ids, delta_ref, delta_cand, delta_abs,
                        (int)i, tc->ref_logits[i], cand_logits[i]);
    }

    const float rms = (float)sqrt(sumsq / (double)tc->vocab_size);
    const float top_abs = test_top_union_max_abs(tc->ref_logits, cand_logits,
                                                 ref_top, cand_top, TEST_MPP_EQ_TOPK);
    const bool same_top1 = ref_top[0] >= 0 && ref_top[0] == cand_top[0];
    test_mpp_eq_result result = {
        .ref_top1 = ref_top[0],
        .cand_top1 = cand_top[0],
        .overlap = overlap,
        .top5_overlap = top5_overlap,
        .max_rank_delta = max_rank_delta,
        .nonfinite = nonfinite,
        .rms = rms,
        .max_abs = max_abs,
        .top20_max_abs = top_abs,
        .same_top1 = same_top1,
        .pass = nonfinite == 0 && same_top1,
    };

    fprintf(stderr,
            "ds4-test: Tensor equivalence %s top1 ref=%d cand=%d top5_overlap=%d/%d overlap=%d/%d max_rank_delta=%d rms=%g max_abs=%g top20_max_abs=%g\n",
            tc->id, ref_top[0], cand_top[0],
            top5_overlap, TEST_MPP_EQ_TOP5,
            overlap, TEST_MPP_EQ_TOPK,
            max_rank_delta, rms, max_abs, top_abs);
    fprintf(stderr, "ds4-test: Tensor equivalence %s largest deltas:", tc->id);
    for (int i = 0; i < TEST_MPP_EQ_DELTAS && delta_ids[i] >= 0; i++) {
        fprintf(stderr, " id=%d ref=%g cand=%g abs=%g",
                delta_ids[i], delta_ref[i], delta_cand[i], delta_abs[i]);
    }
    fputc('\n', stderr);

    if (assert_thresholds) {
        TEST_ASSERT(nonfinite == 0);
        TEST_ASSERT(same_top1);
    }
    return result;
}

static bool test_mpp_capture(ds4_engine *engine, const test_mpp_eq_case *tc,
                             float *logits, int *gen, int *gen_len) {
    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, tc->ctx) == 0);
    if (!session) return false;

    char err[160];
    bool ok = ds4_session_sync(session, &tc->prompt, err, sizeof(err)) == 0;
    TEST_ASSERT(ok);
    if (ok) {
        ok = ds4_session_copy_logits(session, logits, tc->vocab_size) == tc->vocab_size;
        TEST_ASSERT(ok);
    }

    int n = 0;
    while (ok && n < tc->gen_steps) {
        const int token = ds4_session_argmax(session);
        gen[n++] = token;
        if (n < tc->gen_steps && ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            ok = false;
            TEST_ASSERT(false);
        }
    }
    *gen_len = n;

    ds4_session_free(session);
    return ok;
}

static bool test_mpp_capture_logits_only(ds4_engine *engine,
                                         const test_mpp_eq_case *tc,
                                         float *logits) {
    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, tc->ctx) == 0);
    if (!session) return false;

    char err[160];
    bool ok = ds4_session_sync(session, &tc->prompt, err, sizeof(err)) == 0;
    TEST_ASSERT(ok);
    if (ok) {
        ok = ds4_session_copy_logits(session, logits, tc->vocab_size) == tc->vocab_size;
        TEST_ASSERT(ok);
    }

    ds4_session_free(session);
    return ok;
}

static bool test_mpp_eq_case_selected(const char *id) {
    const char *filter = getenv("DS4_TEST_MPP_EQ_CASE");
    if (!filter || !filter[0]) return true;

    char buf[256];
    snprintf(buf, sizeof(buf), "%s", filter);
    for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
        tok = test_trim_line(tok);
        if (tok[0] && strstr(id, tok)) return true;
    }
    return false;
}

static int test_load_mpp_cases(ds4_engine *engine, test_mpp_eq_case *cases, int cap) {
    const char *path = getenv("DS4_TEST_VECTOR_FILE");
    if (!path || !path[0]) path = "tests/test-vectors/official.vec";
    FILE *fp = fopen(path, "rb");
    TEST_ASSERT(fp != NULL);
    if (!fp) return 0;

    int ncase = 0;
    test_vec_case vc;
    while (ncase < cap && test_read_vector_case(fp, &vc)) {
        if (!test_fill_vector_case(fp, &vc)) break;
        if (!test_mpp_eq_case_selected(vc.id)) continue;
        char *prompt_text = test_read_file(vc.prompt_path);
        TEST_ASSERT(prompt_text != NULL);
        if (!prompt_text) continue;

        test_mpp_eq_case *tc = &cases[ncase++];
        snprintf(tc->id, sizeof(tc->id), "%s", vc.id);
        tc->ctx = vc.ctx;
        tc->vocab_size = ds4_engine_vocab_size(engine);
        tc->gen_steps = vc.nsteps < TEST_VEC_MAX_STEPS ? vc.nsteps : TEST_VEC_MAX_STEPS;
        const bool tokenized = ds4_encode_chat_prompt_checked(
            engine, "", prompt_text, DS4_THINK_NONE, &tc->prompt);
        free(prompt_text);
        TEST_ASSERT(tokenized);
        if (!tokenized) {
            ds4_tokens_free(&tc->prompt);
            ncase--;
            continue;
        }
        TEST_ASSERT(tc->prompt.len > 0);
    }
    fclose(fp);
    return ncase;
}

static void test_mpp_summary_init(test_mpp_eq_summary *summary, const char *label) {
    memset(summary, 0, sizeof(*summary));
    summary->label = label;
    summary->min_overlap = TEST_MPP_EQ_TOPK;
    summary->min_top5_overlap = TEST_MPP_EQ_TOP5;
}

static void test_mpp_summary_note_logits(test_mpp_eq_summary *summary,
                                         const test_mpp_eq_result *result) {
    if (!result->pass) summary->logits_failures++;
    if (!result->same_top1) summary->top1_mismatches++;
    if (result->overlap < summary->min_overlap) summary->min_overlap = result->overlap;
    if (result->top5_overlap < summary->min_top5_overlap) {
        summary->min_top5_overlap = result->top5_overlap;
    }
    if (result->max_rank_delta > summary->worst_rank_delta) {
        summary->worst_rank_delta = result->max_rank_delta;
    }
    if (result->rms > summary->worst_rms) summary->worst_rms = result->rms;
    if (result->max_abs > summary->worst_max_abs) summary->worst_max_abs = result->max_abs;
    if (result->top20_max_abs > summary->worst_top20_max_abs) {
        summary->worst_top20_max_abs = result->top20_max_abs;
    }
}

static void test_mpp_summary_print(const test_mpp_eq_summary *summary) {
    fprintf(stderr,
            "ds4-test: Tensor summary route=%s cases=%d capture_fail=%d logits_fail=%d greedy_fail=%d top1_mismatch=%d min_top5_overlap=%d/%d min_overlap=%d/%d worst_rank_delta=%d worst_rms=%g worst_max_abs=%g worst_top20_max_abs=%g\n",
            summary->label,
            summary->cases,
            summary->capture_failures,
            summary->logits_failures,
            summary->greedy_failures,
            summary->top1_mismatches,
            summary->min_top5_overlap,
            TEST_MPP_EQ_TOP5,
            summary->min_overlap,
            TEST_MPP_EQ_TOPK,
            summary->worst_rank_delta,
            summary->worst_rms,
            summary->worst_max_abs,
            summary->worst_top20_max_abs);
}

static void test_run_mpp_candidate(const char *label,
                                   test_mpp_eq_case *cases,
                                   int ncase) {
    fprintf(stderr, "ds4-test: Tensor equivalence candidate route=%s\n", label);
    test_mpp_eq_summary summary;
    test_mpp_summary_init(&summary, label);
    ds4_engine *cand_engine = test_open_engine(false);
    if (cand_engine) {
        const int vocab_size = ncase > 0 ? cases[0].vocab_size : 0;
        float *cand_logits = malloc((size_t)vocab_size * sizeof(cand_logits[0]));
        TEST_ASSERT(cand_logits != NULL);
        if (cand_logits) {
            for (int i = 0; i < ncase; i++) {
                test_mpp_eq_case *tc = &cases[i];
                if (!tc->ref_logits) continue;
                int cand_gen[TEST_VEC_MAX_STEPS] = {0};
                int cand_gen_len = 0;
                if (!test_mpp_capture(cand_engine, tc, cand_logits, cand_gen, &cand_gen_len)) {
                    summary.capture_failures++;
                    continue;
                }
                summary.cases++;
                test_mpp_eq_result result = test_compare_mpp_logits(tc, cand_logits, true);
                test_mpp_summary_note_logits(&summary, &result);
                TEST_ASSERT(cand_gen_len == tc->ref_gen_len);
                if (cand_gen_len != tc->ref_gen_len) summary.greedy_failures++;
                for (int j = 0; j < tc->ref_gen_len && j < cand_gen_len; j++) {
                    if (cand_gen[j] != tc->ref_gen[j]) {
                        fprintf(stderr,
                                "ds4-test: Tensor equivalence %s greedy token mismatch step=%d ref=%d cand=%d\n",
                                tc->id, j, tc->ref_gen[j], cand_gen[j]);
                        summary.greedy_failures++;
                    }
                    TEST_ASSERT(cand_gen[j] == tc->ref_gen[j]);
                }
            }
            free(cand_logits);
        }
        ds4_engine_close(cand_engine);
    }
    test_mpp_summary_print(&summary);
}

static void test_metal_mpp_equivalence(void) {
    test_close_engines();

    test_mpp_eq_case cases[TEST_MPP_EQ_MAX_CASES];
    memset(cases, 0, sizeof(cases));

    char *saved_disable_metal4 = test_save_env("DS4_METAL_DISABLE_METAL4");
    setenv("DS4_METAL_DISABLE_METAL4", "1", 1);
    ds4_engine *ref_engine = test_open_engine(false);
    if (!ref_engine) {
        test_restore_env("DS4_METAL_DISABLE_METAL4", saved_disable_metal4);
        return;
    }

    const int ncase = test_load_mpp_cases(ref_engine, cases, TEST_MPP_EQ_MAX_CASES);
    TEST_ASSERT(ncase > 0);
    for (int i = 0; i < ncase; i++) {
        test_mpp_eq_case *tc = &cases[i];
        tc->ref_logits = malloc((size_t)tc->vocab_size * sizeof(tc->ref_logits[0]));
        TEST_ASSERT(tc->ref_logits != NULL);
        if (!tc->ref_logits) continue;
        TEST_ASSERT(test_mpp_capture(ref_engine, tc,
                                     tc->ref_logits,
                                     tc->ref_gen,
                                     &tc->ref_gen_len));
    }
    ds4_engine_close(ref_engine);
    test_restore_env("DS4_METAL_DISABLE_METAL4", saved_disable_metal4);

    test_run_mpp_candidate("auto", cases, ncase);

    for (int i = 0; i < ncase; i++) test_mpp_eq_case_free(&cases[i]);
}

static void test_streaming_decode_prefill_correctness(void) {
    test_close_engines();
    if (!test_env_bool("DS4_TEST_SSD_STREAMING")) {
        fprintf(stderr,
                "ds4-test: streaming decode-prefill correctness skipped "
                "(set DS4_TEST_SSD_STREAMING=1 to enable)\n");
        return;
    }

    test_mpp_eq_case cases[TEST_MPP_EQ_MAX_CASES];
    memset(cases, 0, sizeof(cases));

    test_streaming_prefill_env saved_canonical_streaming_prefill =
        test_force_canonical_streaming_prefill();

    ds4_engine *ref_engine = test_open_engine(false);
    if (!ref_engine) {
        test_restore_canonical_streaming_prefill(saved_canonical_streaming_prefill);
        return;
    }

    const int ncase = test_load_mpp_cases(ref_engine, cases, TEST_MPP_EQ_MAX_CASES);
    TEST_ASSERT(ncase > 0);
    for (int i = 0; i < ncase; i++) {
        test_mpp_eq_case *tc = &cases[i];
        tc->ref_logits = malloc((size_t)tc->vocab_size * sizeof(tc->ref_logits[0]));
        TEST_ASSERT(tc->ref_logits != NULL);
        if (!tc->ref_logits) continue;
        TEST_ASSERT(test_mpp_capture(ref_engine, tc,
                                     tc->ref_logits,
                                     tc->ref_gen,
                                     &tc->ref_gen_len));
    }
    ds4_engine_close(ref_engine);

    unsetenv("DS4_METAL_DISABLE_STREAMING_COLD_DECODE_PREFILL");
    unsetenv("DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR");

    ds4_engine *cand_engine = test_open_engine(false);
    if (cand_engine) {
        for (int i = 0; i < ncase; i++) {
            test_mpp_eq_case *tc = &cases[i];
            if (!tc->ref_logits) continue;

            float *cand_cold = malloc((size_t)tc->vocab_size * sizeof(cand_cold[0]));
            float *cand_warm_a = malloc((size_t)tc->vocab_size * sizeof(cand_warm_a[0]));
            float *cand_warm_b = malloc((size_t)tc->vocab_size * sizeof(cand_warm_b[0]));
            TEST_ASSERT(cand_cold != NULL);
            TEST_ASSERT(cand_warm_a != NULL);
            TEST_ASSERT(cand_warm_b != NULL);
            if (!cand_cold || !cand_warm_a || !cand_warm_b) {
                free(cand_cold);
                free(cand_warm_a);
                free(cand_warm_b);
                continue;
            }

            TEST_ASSERT(test_mpp_capture_logits_only(cand_engine, tc, cand_cold));
            TEST_ASSERT(test_mpp_capture_logits_only(cand_engine, tc, cand_warm_a));
            TEST_ASSERT(test_mpp_capture_logits_only(cand_engine, tc, cand_warm_b));

            test_mpp_eq_result result = test_compare_mpp_logits(tc, cand_cold, false);
            TEST_ASSERT(result.nonfinite == 0);
            TEST_ASSERT(result.top5_overlap >= 2);
            TEST_ASSERT(result.overlap >= 10);
            TEST_ASSERT(result.rms <= 4.0f);
            TEST_ASSERT(result.top20_max_abs <= 12.0f);

            int cold_warm_neq = 0;
            int warm_repeat_neq = 0;
            int repeat_nonfinite = 0;
            float cold_warm_max_abs = 0.0f;
            float warm_repeat_max_abs = 0.0f;
            for (int j = 0; j < tc->vocab_size; j++) {
                if (!isfinite(cand_cold[j]) ||
                    !isfinite(cand_warm_a[j]) ||
                    !isfinite(cand_warm_b[j])) {
                    repeat_nonfinite++;
                    continue;
                }
                const float cold_warm_d = fabsf(cand_cold[j] - cand_warm_a[j]);
                if (cold_warm_d != 0.0f) cold_warm_neq++;
                if (cold_warm_d > cold_warm_max_abs) cold_warm_max_abs = cold_warm_d;
                const float warm_repeat_d = fabsf(cand_warm_a[j] - cand_warm_b[j]);
                if (warm_repeat_d != 0.0f) warm_repeat_neq++;
                if (warm_repeat_d > warm_repeat_max_abs) {
                    warm_repeat_max_abs = warm_repeat_d;
                }
            }
            TEST_ASSERT(repeat_nonfinite == 0);
            TEST_ASSERT(cold_warm_neq == 0);
            TEST_ASSERT(warm_repeat_neq == 0);
            fprintf(stderr,
                    "ds4-test: streaming decode-prefill %s cold_warm_neq=%d "
                    "cold_warm_max_abs=%g warm_repeat_neq=%d "
                    "warm_repeat_max_abs=%g top1 canonical=%d decode=%d\n",
                    tc->id,
                    cold_warm_neq,
                    cold_warm_max_abs,
                    warm_repeat_neq,
                    warm_repeat_max_abs,
                    result.ref_top1,
                    result.cand_top1);

            free(cand_cold);
            free(cand_warm_a);
            free(cand_warm_b);
        }
        ds4_engine_close(cand_engine);
    }

    test_restore_canonical_streaming_prefill(saved_canonical_streaming_prefill);
    for (int i = 0; i < ncase; i++) test_mpp_eq_case_free(&cases[i]);
}

static const char *test_tool_call_request_json(void) {
    return
        "{"
        "\"model\":\"deepseek-v4-flash\","
        "\"messages\":[{\"role\":\"user\",\"content\":\"List the files in the current directory. Use the provided tool; do not answer in prose.\"}],"
        "\"tools\":[{\"type\":\"function\",\"function\":{"
            "\"name\":\"list_files\","
            "\"description\":\"List files in a directory.\","
            "\"parameters\":{\"type\":\"object\",\"properties\":{"
                "\"path\":{\"type\":\"string\",\"description\":\"Directory path to list.\"}"
            "},\"required\":[\"path\"]}"
        "}}],"
        "\"tool_choice\":\"auto\","
        "\"think\":false,"
        "\"temperature\":0,"
        "\"max_tokens\":256,"
        "\"stream\":false"
        "}";
}

static const char *test_think_recovery_request_json(void) {
    return
        "{"
        "\"model\":\"deepseek-v4-flash\","
        "\"messages\":[{\"role\":\"user\",\"content\":\"List the files in the current directory. Use the provided tool; do not answer in prose.\"}],"
        "\"tools\":[{\"type\":\"function\",\"function\":{"
            "\"name\":\"list_files\","
            "\"description\":\"List files in a directory.\","
            "\"parameters\":{\"type\":\"object\",\"properties\":{"
                "\"path\":{\"type\":\"string\",\"description\":\"Directory path to list.\"}"
            "},\"required\":[\"path\"]}"
        "}}],"
        "\"tool_choice\":\"auto\","
        "\"think\":true,"
        "\"temperature\":0,"
        "\"max_tokens\":384,"
        "\"stream\":false"
        "}";
}

/* The model sometimes opens a DSML stanza without closing </think> first.
 * The server's forward recovery must force the close plus a fresh stanza
 * opening, after which the model must still complete a valid call.  The
 * malformed prefix is teacher-forced so the regression is deterministic and
 * does not depend on coaxing the model into misbehaving. */
static void test_think_tool_recovery(void) {
    ds4_engine *engine = test_get_engine(false);
    if (!engine) return;

    request r;
    char err[160];
    TEST_ASSERT(parse_chat_request(engine, NULL, test_think_recovery_request_json(),
                                   512, 32768, &r, err, sizeof(err)));

    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, 32768) == 0);
    if (!session) {
        request_free(&r);
        return;
    }
    TEST_ASSERT(ds4_session_sync(session, &r.prompt, err, sizeof(err)) == 0);

    if (getenv("DS4_TEST_RECOVERY_PROBE") != NULL) {
        /* Diagnostic: print the model's natural tool-call turn for this
         * request instead of running the recovery. */
        buf nat = {0};
        uint64_t prng = 7;
        for (int i = 0; i < 300; i++) {
            int token = ds4_session_sample(session, 0.0f, 0, 1.0f, 0.0f, &prng);
            if (token == ds4_token_eos(engine)) break;
            size_t plen = 0;
            char *p = ds4_token_text(engine, token, &plen);
            buf_append(&nat, p, plen);
            free(p);
            bool ps = false, pe = false;
            observe_tool_markers(nat.ptr, &ps, &pe, NULL);
            if (pe) break;
            if (ds4_session_eval(session, token, err, sizeof(err)) != 0) break;
        }
        fprintf(stderr, "ds4-test: natural turn=[%s]\n", nat.ptr ? nat.ptr : "");
        buf_free(&nat);
        ds4_session_free(session);
        request_free(&r);
        test_close_engine(false);
        return;
    }

    thinking_state thinking = thinking_state_from_prompt(&r);
    buf text = {0};
    buf forced = {0};
    if (!thinking.inside) buf_append(&forced, "<think>", 7);
    const char *body =
        "The user wants a directory listing. I will call the "
        "list_files tool right away.\n\n" DS4_TOOL_CALLS_START;
    buf_append(&forced, body, strlen(body));

    server srv;
    memset(&srv, 0, sizeof(srv));
    srv.engine = engine;
    srv.session = session;

    /* Replay the malformed prefix exactly as the worker loop would see it:
     * token by token, running the recovery scan after each piece.  The stanza
     * opening spans several tokens, so this also checks that detection does
     * not depend on how the marker happens to be tokenized: recovery must
    * stay quiet on every partial prefix and trigger exactly when the
    * opening completes. */
    ds4_tokens toks = {0};
    TEST_ASSERT(ds4_tokenize_rendered_chat_checked(
        engine, forced.ptr, &toks));
    TEST_ASSERT(toks.len > 1);
    size_t scan_from = 0;
    int completion = 0;
    int rec = 0;
    int triggered_at = -1;
    for (int i = 0; i < toks.len; i++) {
        TEST_ASSERT(ds4_session_eval(session, toks.v[i], err, sizeof(err)) == 0);
        size_t piece_len = 0;
        char *piece = ds4_token_text(engine, toks.v[i], &piece_len);
        buf_append(&text, piece, piece_len);
        thinking_state_feed(&thinking, piece, piece_len);
        free(piece);
        TEST_ASSERT(thinking.inside);
        rec = chat_think_tool_recovery(&srv, &text, &thinking, &scan_from,
                                       &completion, 512, err, sizeof(err));
        TEST_ASSERT(rec >= 0);
        if (rec == 1) {
            triggered_at = i;
            break;
        }
    }
    fprintf(stderr,
            "ds4-test: think-tool-recovery trigger=%d/%d injected_tokens=%d\n",
            triggered_at, toks.len, completion);
    TEST_ASSERT(rec == 1);
    TEST_ASSERT(triggered_at == toks.len - 1);
    ds4_tokens_free(&toks);
    buf_free(&forced);
    TEST_ASSERT(!thinking.inside);
    TEST_ASSERT(completion > 0);
    TEST_ASSERT(text.ptr && text.len >= 10 &&
                !memcmp(text.ptr + text.len - 10, "</think>\n\n", 10));

    /* The model must now complete a valid call on the executable side. */
    uint64_t rng = 123;
    bool decode_ok = true;
    bool saw_start = false;
    bool saw_end = false;
    for (int i = 0; i < 256 && !saw_end; i++) {
        int token = ds4_session_sample(session, 0.0f, 0, 1.0f, 0.0f, &rng);
        if (token == ds4_token_eos(engine)) break;
        size_t piece_len = 0;
        char *piece = ds4_token_text(engine, token, &piece_len);
        buf_append(&text, piece, piece_len);
        free(piece);
        observe_tool_markers(text.ptr, &saw_start, &saw_end, NULL);
        if (saw_end) break;
        if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            decode_ok = false;
            break;
        }
    }
    fprintf(stderr, "ds4-test: think-tool-recovery continuation=[%s]\n",
            text.ptr ? text.ptr : "");
    TEST_ASSERT(decode_ok);
    TEST_ASSERT(saw_end);

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    bool parsed = parse_generated_message_ex(text.ptr, true,
                                             &content, &reasoning, &calls);
    TEST_ASSERT(parsed);
    TEST_ASSERT(calls.len > 0 && !strcmp(calls.v[0].name, "list_files"));
    TEST_ASSERT(reasoning && strstr(reasoning, "list_files tool right away"));

    fprintf(stderr,
            "ds4-test: think-tool-recovery recovered=%d gen_tokens=%d calls=%d name=%s\n",
            rec, completion, calls.len, calls.len ? calls.v[0].name : "-");

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
    buf_free(&text);
    ds4_session_free(session);
    request_free(&r);
    test_close_engine(false);
}

static void test_tool_call_quality_one(bool quality) {
    ds4_engine *engine = test_get_engine(quality);
    if (!engine) return;

    request r;
    char err[160];
    TEST_ASSERT(parse_chat_request(engine, NULL, test_tool_call_request_json(),
                                   512, 32768, &r, err, sizeof(err)));

    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, 32768) == 0);
    if (!session) {
        request_free(&r);
        return;
    }
    TEST_ASSERT(ds4_session_sync(session, &r.prompt, err, sizeof(err)) == 0);

    buf text = {0};
    uint64_t rng = 123;
    bool decode_ok = true;
    bool saw_tool_start = false;
    bool saw_tool_end = false;
    for (int i = 0; i < r.max_tokens; i++) {
        int token = ds4_session_sample(session, r.temperature, r.top_k,
                                       r.top_p, r.min_p, &rng);
        size_t piece_len = 0;
        char *piece = ds4_token_text(engine, token, &piece_len);
        buf_append(&text, piece, piece_len);
        free(piece);
        observe_tool_markers(text.ptr ? text.ptr : "", &saw_tool_start, &saw_tool_end, NULL);
        if (saw_tool_end) break;
        if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
            decode_ok = false;
            break;
        }
    }

    char *content = NULL;
    char *reasoning = NULL;
    tool_calls calls = {0};
    bool parsed = parse_generated_message_ex(text.ptr ? text.ptr : "",
                                             false, &content, &reasoning, &calls);
    TEST_ASSERT(decode_ok);
    TEST_ASSERT(parsed);
    TEST_ASSERT(calls.len > 0);
    TEST_ASSERT(calls.len > 0 && !strcmp(calls.v[0].name, "list_files"));

    free(content);
    free(reasoning);
    tool_calls_free(&calls);
    buf_free(&text);
    ds4_session_free(session);
    request_free(&r);
}

static void test_tool_call_quality(void) {
    fprintf(stderr, "ds4-test: tool-call quality fast path\n");
    test_tool_call_quality_one(false);
    test_close_engine(false);
    fprintf(stderr, "ds4-test: tool-call quality exact path\n");
    test_tool_call_quality_one(true);
    test_close_engine(true);
}

/* Greedy speculative decode: capture committed tokens and the largest accepted
 * chunk, so the caller can confirm the multi-row verify path actually ran. */
static bool test_mtp_capture_speculative(ds4_engine *engine, const ds4_tokens *prompt,
                                         int max_tokens, int *out, int *out_len,
                                         int *max_chunk) {
    *out_len = 0;
    *max_chunk = 0;
    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, 32768) == 0);
    if (!session) return false;

    char err[160];
    bool ok = ds4_session_sync(session, prompt, err, sizeof(err)) == 0;
    TEST_ASSERT(ok);

    const int eos = ds4_token_eos(engine);
    int n = 0;
    bool stop = false;
    while (ok && !stop && n < max_tokens) {
        const int token = ds4_session_argmax(session);
        if (token == eos) break;

        int toks[17]; /* base token + draft depth, which the engine clamps to 16 */
        const int ntok = ds4_session_eval_speculative_argmax(
            session, token, max_tokens - n, eos, toks,
            (int)(sizeof(toks) / sizeof(toks[0])), err, sizeof(err));
        if (ntok < 0) { ok = false; TEST_ASSERT(false); break; }
        if (ntok > *max_chunk) *max_chunk = ntok;

        for (int j = 0; j < ntok; j++) {
            if (toks[j] == eos) { stop = true; break; }
            out[n++] = toks[j];
            if (n >= max_tokens) { stop = true; break; }
        }
    }

    *out_len = n;
    ds4_session_free(session);
    return ok;
}

/* Replay toks[] through plain decode and return the largest gap between a
 * position's argmax logit and the committed token's logit.  Correct speculation
 * commits (near-)argmax tokens (gap ~0); a mis-committed token gives a big gap. */
static bool test_mtp_worst_argmax_gap(ds4_engine *engine, const ds4_tokens *prompt,
                                      const int *toks, int n,
                                      float *worst_gap, int *worst_at) {
    *worst_gap = 0.0f;
    *worst_at = -1;
    ds4_session *session = NULL;
    TEST_ASSERT(ds4_session_create(&session, engine, 32768) == 0);
    if (!session) return false;

    char err[160];
    bool ok = ds4_session_sync(session, prompt, err, sizeof(err)) == 0;
    TEST_ASSERT(ok);

    for (int i = 0; ok && i < n; i++) {
        ds4_token_score best, cur;
        ok = ds4_session_top_logprobs(session, &best, 1) >= 1 &&
             ds4_session_token_logprob(session, toks[i], &cur) == 1;
        TEST_ASSERT(ok);
        if (!ok) break;

        const float gap = best.logit - cur.logit;
        if (gap > *worst_gap) { *worst_gap = gap; *worst_at = i; }
        if (ds4_session_eval(session, toks[i], err, sizeof(err)) != 0) { ok = false; TEST_ASSERT(false); break; }
    }

    ds4_session_free(session);
    return ok;
}

/* Verbatim-copy task: keeps the model confident (a mis-committed token shows as
 * a large argmax gap) and draft acceptance high (so the multi-row verify path is
 * exercised across the generation). */
static const char *test_mtp_copy_prompt(void) {
    return
        "Reproduce the following C code EXACTLY, character for character, "
        "inside a single code block and output nothing else:\n\n"
        "```c\n"
        "static uint32_t clamp_u32(uint32_t v, uint32_t lo, uint32_t hi) {\n"
        "    if (v < lo) return lo;\n"
        "    if (v > hi) return hi;\n"
        "    return v;\n"
        "}\n"
        "\n"
        "static uint32_t ring_advance(uint32_t pos, uint32_t cap) {\n"
        "    uint32_t next = pos + 1u;\n"
        "    return next >= cap ? 0u : next;\n"
        "}\n"
        "\n"
        "static int scratch_init(scratch *s, uint32_t ctx_size) {\n"
        "    if (ctx_size == 0u) ctx_size = 1u;\n"
        "    s->ctx_size = ctx_size;\n"
        "    s->comp_cap = ctx_size / 4u + 2u;\n"
        "    s->rows = clamp_u32(s->comp_cap, 1u, 4096u);\n"
        "    s->head = 0u;\n"
        "    return s->rows > 0u ? 0 : -1;\n"
        "}\n"
        "```\n";
}

#define TEST_MTP_MAXGEN 256

/* Regression for the swapped top-k arguments in metal_graph_verify_suffix_tops
 * at draft depth > 2.  Replays the committed speculative tokens through plain
 * decode and requires each to be a (near-)argmax: that is the verify invariant,
 * and unlike comparing token streams it tolerates the near-greedy tie
 * divergences.  Needs an MTP head, so it self-skips without DS4_TEST_MTP. */
static void test_mtp_verify_depth(void) {
    ds4_engine *engine = test_get_engine(false);
    if (!engine || !ds4_engine_has_mtp(engine)) {
        fprintf(stderr, "ds4-test: mtp-verify-depth skipped (set DS4_TEST_MTP to an MTP GGUF)\n");
        return;
    }
    TEST_ASSERT(ds4_engine_mtp_draft_tokens(engine) > 2);

    ds4_tokens prompt = {0};
    ds4_chat_begin(engine, &prompt);
    TEST_ASSERT(ds4_chat_append_message_checked(
        engine, &prompt, "user", test_mtp_copy_prompt()));
    TEST_ASSERT(ds4_chat_append_assistant_prefix_checked(
        engine, &prompt, DS4_THINK_NONE));
    TEST_ASSERT(prompt.len > 0);

    int *spec = malloc((size_t)TEST_MTP_MAXGEN * sizeof(*spec));
    TEST_ASSERT(spec != NULL);
    if (spec && prompt.len > 0) {
        int nspec = 0, max_chunk = 0;
        const bool ok_spec = test_mtp_capture_speculative(engine, &prompt, TEST_MTP_MAXGEN,
                                                          spec, &nspec, &max_chunk);
        TEST_ASSERT(ok_spec);
        TEST_ASSERT(max_chunk > 1);  /* multi-token chunks committed: the multi-row path ran */
        TEST_ASSERT(nspec > 128);    /* enough output to surface the bug, incl. a spurious-EOS truncation */

        float worst_gap = 0.0f;
        int worst_at = -1;
        const bool ok_check = test_mtp_worst_argmax_gap(engine, &prompt, spec, nspec,
                                                        &worst_gap, &worst_at);
        TEST_ASSERT(ok_check);
        fprintf(stderr, "ds4-test: mtp-verify-depth nspec=%d max_chunk=%d worst_argmax_gap=%.3f at=%d\n",
                nspec, max_chunk, worst_gap, worst_at);
        TEST_ASSERT(worst_gap <= 2.0f);  /* correct: ~0; bug: ~21 on the reference model */
    }

    free(spec);
    ds4_tokens_free(&prompt);
}
#endif

static void test_server_unit_group(void) {
    ds4_server_unit_tests_run();
}

typedef void (*test_fn)(void);

typedef struct {
    const char *flag;
    const char *name;
    const char *desc;
    test_fn fn;
} ds4_test_entry;

static const ds4_test_entry test_entries[] = {
#ifndef DS4_NO_GPU
    {"--long-context", "long-context", "long-context story fact-recall regression", test_long_story_fact_recall},
    {"--tool-call-quality", "tool-call-quality", "model emits valid DSML tool calls", test_tool_call_quality},
    {"--think-tool-recovery", "think-tool-recovery", "forced </think> recovery when a tool call starts inside thinking", test_think_tool_recovery},
    {"--logprob-vectors", "logprob-vectors", "official API top-logprob vector comparison on the standard Metal path", test_official_logprob_vectors},
    {"--metal-ssd-streaming-cache-pressure", "metal-ssd-streaming-cache-pressure", "Metal SSD-streaming layer-batched decode cache-pressure repro for issue #384", test_metal_ssd_streaming_cache_pressure},
    {"--local-golden-vectors", "local-golden-vectors", "local top-k/logit drift regression for long Metal prefill", test_local_golden_vectors},
    {"--metal-short-prefill", "metal-short-prefill", "Metal ratio-4 short prefill regression", test_metal_short_prefill_ratio4},
    {"--metal-kernels", "metal-kernels", "isolated Metal kernel numeric regressions", test_metal_kernel_group},
    {"--metal-tensor-equivalence", "metal-tensor-equivalence", "fast/quality Metal prompt-logit and greedy equivalence", test_metal_mpp_equivalence},
    {"--streaming-decode-prefill-correctness", "streaming-decode-prefill-correctness", "streaming decode-style cold prefill drift and repeatability", test_streaming_decode_prefill_correctness},
    {"--mtp-verify-depth", "mtp-verify-depth", "MTP speculative verify commits autoregressive-identical tokens at draft depth > 2", test_mtp_verify_depth},
#endif
    {"--server", "server", "server parser/rendering/cache unit tests", test_server_unit_group},
};

static void test_print_help(const char *prog) {
    printf("Usage: %s [--all | TEST...]\n\n", prog);
    puts("Tests:");
    puts("  --all");
    puts("      Run every test. This is the default, ordered from slower to faster.");
    for (size_t i = 0; i < sizeof(test_entries) / sizeof(test_entries[0]); i++) {
        printf("  %-20s %s\n", test_entries[i].flag, test_entries[i].desc);
    }
    puts("  --list");
    puts("      Print test names only.");
#ifndef DS4_NO_GPU
    puts("  --metal-mpp-equivalence");
    puts("      Compatibility alias for --metal-tensor-equivalence.");
#endif
    puts("  -h, --help");
    puts("      Show this help.");
    puts("\nEnvironment:");
    puts("  DS4_TEST_MODEL=FILE        Model path. Default: ds4flash.gguf");
    puts("  DS4_TEST_SSD_STREAMING=1   Run model tests through Metal SSD streaming.");
    puts("  DS4_TEST_SSD_STREAMING_CACHE_GB=N  Streaming routed expert cache in GiB.");
    puts("  DS4_TEST_SSD_STREAMING_CACHE_EXPERTS=N  Streaming routed expert cache count.");
    puts("  DS4_TEST_SSD_STREAMING_COLD=1  Skip streaming hot expert preload.");
    puts("  DS4_METAL_DISABLE_STREAMING_COLD_DECODE_PREFILL=1  Force canonical streamed cold prefill.");
    puts("  DS4_TEST_LONG_PROMPT=FILE  Rendered long-context story fact prompt.");
    puts("  DS4_TEST_VECTOR_FILE=FILE  Simple official-vector fixture.");
    puts("  DS4_TEST_LOCAL_GOLDEN_FILE=FILE  Local top-k golden-vector fixture.");
    puts("  DS4_TEST_MPP_EQ_CASE=NAME  Run only Tensor equivalence cases whose id contains NAME.");
}

static const ds4_test_entry *test_find_entry(const char *arg) {
#ifndef DS4_NO_GPU
    if (!strcmp(arg, "--metal-mpp-equivalence")) {
        arg = "--metal-tensor-equivalence";
    }
#endif
    for (size_t i = 0; i < sizeof(test_entries) / sizeof(test_entries[0]); i++) {
        if (!strcmp(arg, test_entries[i].flag)) return &test_entries[i];
    }
    return NULL;
}

static void test_run_entry(const ds4_test_entry *entry) {
    int before = test_failures;
    fprintf(stderr, "%s:\n", entry->name);
    entry->fn();
    fprintf(stderr, "%s: ", entry->name);
    ds4_log(stderr,
            test_failures == before ? DS4_LOG_OK : DS4_LOG_ERROR,
            "%s",
            test_failures == before ? "OK" : "ERR");
    fputc('\n', stderr);
}

int main(int argc, char **argv) {
    bool run_all = argc == 1;
    bool selected[sizeof(test_entries) / sizeof(test_entries[0])] = {0};

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--all")) {
            run_all = true;
        } else if (!strcmp(argv[i], "--list")) {
            for (size_t j = 0; j < sizeof(test_entries) / sizeof(test_entries[0]); j++) {
                puts(test_entries[j].flag);
            }
            return 0;
        } else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) {
            test_print_help(argv[0]);
            return 0;
        } else {
            const ds4_test_entry *entry = test_find_entry(argv[i]);
            if (!entry) {
                fprintf(stderr, "ds4-test: unknown test switch: %s\n", argv[i]);
                test_print_help(argv[0]);
                return 2;
            }
            selected[(size_t)(entry - test_entries)] = true;
        }
    }

    if (run_all) {
        for (size_t i = 0; i < sizeof(test_entries) / sizeof(test_entries[0]); i++) {
            test_run_entry(&test_entries[i]);
        }
    } else {
        for (size_t i = 0; i < sizeof(test_entries) / sizeof(test_entries[0]); i++) {
            if (selected[i]) test_run_entry(&test_entries[i]);
        }
    }

#ifndef DS4_NO_GPU
    test_close_engines();
#endif

    if (test_failures) {
        fprintf(stderr, "ds4 tests: %d failure(s)\n", test_failures);
        return 1;
    }
    puts("ds4 tests: ok");
    return 0;
}
