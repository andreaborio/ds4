#include "ds4.h"
#include "ds4_distributed.h"
#include "ds4_help.h"
#ifndef DS4_NO_GPU
#include "ds4_gpu.h"
#endif

/* Purpose-built throughput benchmark.
 *
 * The benchmark walks one fixed token sequence to configurable context
 * frontiers, measuring only the newest prefill interval at each frontier.  It
 * then snapshots the live session in memory when the payload is small enough,
 * performs a fixed greedy decode run without allowing EOS, restores the
 * snapshot or replays the prefix, and continues to the next frontier.  Snapshot
 * save/restore time is intentionally outside both timing windows.
 */

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    const char *model_path;
    const char *prompt_path;
    const char *chat_prompt_path;
    const char *system;
    const char *csv_path;
    const char *expert_profile_path;
    ds4_backend backend;
    int threads;
    int ctx_start;
    int ctx_max;
    int ctx_alloc;
    int step_incr;
    int gen_tokens;
    int power_percent;
    uint32_t prefill_chunk;
    uint32_t ssd_streaming_cache_experts;
    uint64_t ssd_streaming_cache_bytes;
    uint32_t ssd_streaming_full_layers;
    uint32_t ssd_streaming_preload_experts;
    uint64_t simulate_used_memory_bytes;
    double step_mul;
    const char *dump_frontier_logits_dir;
    const char *dump_decode_evidence_dir;
    ds4_dist_options dist;
    bool warm_weights;
    bool quality;
    ds4_residency_mode residency;
    bool ssd_streaming_cold;
    bool ssd_streaming_full_layers_set;
} bench_config;

static double bench_now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}

static int compare_double_ascending(const void *a, const void *b) {
    const double lhs = *(const double *)a;
    const double rhs = *(const double *)b;
    return (lhs > rhs) - (lhs < rhs);
}

static double nearest_rank_percentile(const double *sorted,
                                      size_t count,
                                      double percentile) {
    if (!sorted || count == 0) return 0.0;
    size_t rank = (size_t)ceil(percentile * (double)count);
    if (rank == 0) rank = 1;
    if (rank > count) rank = count;
    return sorted[rank - 1];
}

static void usage(FILE *fp, const char *topic) {
    ds4_help_print(fp, DS4_HELP_BENCH, topic);
}

static int parse_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v <= 0 || v > INT_MAX) {
        fprintf(stderr, "ds4-bench: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static int parse_nonnegative_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v < 0 || v > INT_MAX) {
        fprintf(stderr, "ds4-bench: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static double parse_double_arg(const char *s, const char *opt) {
    char *end = NULL;
    double v = strtod(s, &end);
    if (s[0] == '\0' || *end != '\0' || !isfinite(v)) {
        fprintf(stderr, "ds4-bench: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return v;
}

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4-bench: %s requires an argument\n", opt);
        exit(2);
    }
    return argv[++*i];
}

static ds4_backend parse_backend(const char *s, const char *opt) {
    if (!strcmp(s, "metal")) return DS4_BACKEND_METAL;
#ifdef DS4_ROCM_BUILD
    if (!strcmp(s, "rocm")) return DS4_BACKEND_CUDA;
#else
    if (!strcmp(s, "cuda")) return DS4_BACKEND_CUDA;
#endif
    if (!strcmp(s, "cpu")) return DS4_BACKEND_CPU;
    fprintf(stderr, "ds4-bench: invalid value for %s: %s\n", opt, s);
#ifdef DS4_ROCM_BUILD
    fprintf(stderr, "ds4-bench: valid backends are: metal, rocm, cpu\n");
#else
    fprintf(stderr, "ds4-bench: valid backends are: metal, cuda, cpu\n");
#endif
    exit(2);
}

static ds4_backend default_backend(void) {
#ifdef DS4_NO_GPU
    return DS4_BACKEND_CPU;
#elif defined(__APPLE__)
    return DS4_BACKEND_METAL;
#else
    return DS4_BACKEND_CUDA;
#endif
}

static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "ds4-bench: failed to open %s: %s\n", path, strerror(errno));
        exit(1);
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "ds4-bench: failed to seek %s\n", path);
        fclose(fp);
        exit(1);
    }
    long n = ftell(fp);
    if (n < 0) {
        fprintf(stderr, "ds4-bench: failed to tell %s\n", path);
        fclose(fp);
        exit(1);
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        fprintf(stderr, "ds4-bench: failed to rewind %s\n", path);
        fclose(fp);
        exit(1);
    }
    char *buf = malloc((size_t)n + 1);
    if (!buf) {
        fprintf(stderr, "ds4-bench: out of memory reading %s\n", path);
        fclose(fp);
        exit(1);
    }
    if (fread(buf, 1, (size_t)n, fp) != (size_t)n) {
        fprintf(stderr, "ds4-bench: failed to read %s\n", path);
        free(buf);
        fclose(fp);
        exit(1);
    }
    fclose(fp);
    buf[n] = '\0';
    return buf;
}

static bench_config parse_options(int argc, char **argv) {
    bench_config c = {
        .model_path = "ds4flash.gguf",
        .system = "You are a helpful assistant.",
        .backend = default_backend(),
        .ctx_start = 2048,
        .ctx_max = 32768,
        .step_incr = 2048,
        .gen_tokens = 128,
        .step_mul = 1.0,
    };

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            const char *topic = (i + 1 < argc && argv[i + 1][0] != '-') ?
                argv[i + 1] : NULL;
            usage(stdout, topic);
            exit(0);
        }
        char dist_parse_err[256] = {0};
        ds4_dist_cli_parse_result dist_parse =
            ds4_dist_parse_cli_arg(arg,
                                   &i,
                                   argc,
                                   argv,
                                   &c.dist,
                                   dist_parse_err,
                                   sizeof(dist_parse_err));
        if (dist_parse == DS4_DIST_CLI_ERROR) {
            fprintf(stderr,
                    "ds4-bench: %s\n",
                    dist_parse_err[0] ? dist_parse_err : "invalid distributed option");
            exit(2);
        }
        if (dist_parse == DS4_DIST_CLI_MATCHED) continue;

        if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            c.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--prompt-file")) {
            c.prompt_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--chat-prompt-file")) {
            c.chat_prompt_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-sys") || !strcmp(arg, "--system")) {
            c.system = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--ctx-start")) {
            c.ctx_start = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--ctx-max")) {
            c.ctx_max = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--ctx-alloc")) {
            c.ctx_alloc = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--step-incr")) {
            c.step_incr = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--step-mul")) {
            c.step_mul = parse_double_arg(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--gen-tokens") || !strcmp(arg, "--tokens") || !strcmp(arg, "-n")) {
            c.gen_tokens = parse_nonnegative_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--csv")) {
            c.csv_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--dump-frontier-logits-dir")) {
            c.dump_frontier_logits_dir = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--dump-decode-evidence-dir")) {
            c.dump_decode_evidence_dir = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--expert-profile")) {
            c.expert_profile_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-t") || !strcmp(arg, "--threads")) {
            c.threads = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--backend")) {
            c.backend = parse_backend(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--metal")) {
            c.backend = DS4_BACKEND_METAL;
#ifdef DS4_ROCM_BUILD
        } else if (!strcmp(arg, "--rocm")) {
            c.backend = DS4_BACKEND_CUDA;
#else
        } else if (!strcmp(arg, "--cuda")) {
            c.backend = DS4_BACKEND_CUDA;
#endif
        } else if (!strcmp(arg, "--cpu")) {
            c.backend = DS4_BACKEND_CPU;
        } else if (!strcmp(arg, "--quality")) {
            c.quality = true;
        } else if (!strcmp(arg, "--ssd-streaming")) {
            c.residency = DS4_RESIDENCY_SSD;
        } else if (!strcmp(arg, "--resident") ||
                   !strcmp(arg, "--no-ssd-streaming")) {
            c.residency = DS4_RESIDENCY_RESIDENT;
        } else if (!strcmp(arg, "--ssd-streaming-cold")) {
            c.ssd_streaming_cold = true;
        } else if (!strcmp(arg, "--ssd-streaming-cache-experts")) {
            uint32_t experts = 0;
            uint64_t bytes = 0;
            if (!ds4_parse_streaming_cache_experts_arg(
                    need_arg(&i, argc, argv, arg), &experts, &bytes)) {
                fprintf(stderr,
                        "ds4-bench: --ssd-streaming-cache-experts must be a positive count or <number>GB\n");
                exit(2);
            }
            c.ssd_streaming_cache_experts = experts;
            c.ssd_streaming_cache_bytes = bytes;
        } else if (!strcmp(arg, "--ssd-streaming-full-layers")) {
            int v = parse_nonnegative_int(need_arg(&i, argc, argv, arg), arg);
            c.ssd_streaming_full_layers = (uint32_t)v;
            c.ssd_streaming_full_layers_set = true;
        } else if (!strcmp(arg, "--ssd-streaming-preload-experts")) {
            int v = parse_int(need_arg(&i, argc, argv, arg), arg);
            if (v <= 0) {
                fprintf(stderr, "ds4-bench: --ssd-streaming-preload-experts must be positive\n");
                exit(2);
            }
            c.ssd_streaming_preload_experts = (uint32_t)v;
        } else if (!strcmp(arg, "--simulate-used-memory")) {
            if (!ds4_parse_gib_arg(need_arg(&i, argc, argv, arg),
                                   &c.simulate_used_memory_bytes)) {
                fprintf(stderr,
                        "ds4-bench: --simulate-used-memory must be a positive GiB value, e.g. 64GB\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--prefill-chunk")) {
            c.prefill_chunk = (uint32_t)parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--power")) {
            c.power_percent = parse_int(need_arg(&i, argc, argv, arg), arg);
            if (c.power_percent < 1 || c.power_percent > 100) {
                fprintf(stderr, "ds4-bench: --power must be between 1 and 100\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--warm-weights")) {
            c.warm_weights = true;
        } else {
            fprintf(stderr, "ds4-bench: unknown option: %s\n", arg);
            usage(stderr, NULL);
            exit(2);
        }
    }

    if (!!c.prompt_path == !!c.chat_prompt_path) {
        fprintf(stderr, "ds4-bench: specify exactly one of --prompt-file or --chat-prompt-file\n");
        exit(2);
    }
    if (c.ctx_start > c.ctx_max) {
        fprintf(stderr, "ds4-bench: --ctx-start must be <= --ctx-max\n");
        exit(2);
    }
    if (c.step_mul < 1.0) {
        fprintf(stderr, "ds4-bench: --step-mul must be >= 1\n");
        exit(2);
    }
    if (c.step_mul == 1.0 && c.step_incr <= 0) {
        fprintf(stderr, "ds4-bench: --step-incr must be positive when --step-mul is 1\n");
        exit(2);
    }
    if (c.ctx_max > INT_MAX - c.gen_tokens - 1) {
        fprintf(stderr, "ds4-bench: requested context is too large\n");
        exit(2);
    }
    if (c.ctx_alloc == 0) c.ctx_alloc = c.ctx_max + c.gen_tokens + 1;
    if (c.ctx_alloc <= c.ctx_max + c.gen_tokens) {
        fprintf(stderr, "ds4-bench: --ctx-alloc must be greater than ctx-max + gen-tokens\n");
        exit(2);
    }
    char dist_err[256];
    if (ds4_dist_prepare_engine_options(&c.dist, NULL, dist_err, sizeof(dist_err)) != 0) {
        fprintf(stderr, "ds4-bench: %s\n", dist_err);
        exit(2);
    }
    if (c.dist.role == DS4_DISTRIBUTED_WORKER) {
        fprintf(stderr, "ds4-bench: --role worker is a serving mode; start workers with ./ds4\n");
        exit(2);
    }
    return c;
}

static void json_write_string(FILE *fp, const char *s) {
    fputc('"', fp);
    if (s) {
        for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
            switch (*p) {
            case '"':  fputs("\\\"", fp); break;
            case '\\': fputs("\\\\", fp); break;
            case '\b': fputs("\\b", fp); break;
            case '\f': fputs("\\f", fp); break;
            case '\n': fputs("\\n", fp); break;
            case '\r': fputs("\\r", fp); break;
            case '\t': fputs("\\t", fp); break;
            default:
                if (*p < 0x20) fprintf(fp, "\\u%04x", (unsigned)*p);
                else fputc((char)*p, fp);
                break;
            }
        }
    }
    fputc('"', fp);
}

/* ds4-bench is built with -ffast-math, under which libc-style isfinite checks
 * may be folded away.  Inspecting IEEE-754 exponent bits keeps JSON strict even
 * if a diagnostic logits vector contains NaN or infinity. */
static bool json_f32_is_finite(const float *value) {
    uint32_t bits = 0;
    unsigned char *dst = (unsigned char *)&bits;
    const volatile unsigned char *src =
        (const volatile unsigned char *)(const void *)value;
    for (size_t i = 0; i < sizeof(bits); i++) dst[i] = src[i];
    return (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}

static int write_frontier_logits_json(
        const bench_config *cfg,
        ds4_engine         *engine,
        ds4_session        *session,
        int                 frontier,
        int                 previous) {
    if (!cfg->dump_frontier_logits_dir) return 0;

    const int vocab = ds4_engine_vocab_size(engine);
    float *logits = malloc((size_t)vocab * sizeof(logits[0]));
    if (!logits) {
        fprintf(stderr, "ds4-bench: out of memory copying frontier logits\n");
        return 1;
    }
    if (ds4_session_copy_logits(session, logits, vocab) != vocab) {
        fprintf(stderr, "ds4-bench: failed to copy frontier logits at %d\n", frontier);
        free(logits);
        return 1;
    }

    char path[PATH_MAX];
    const int n = snprintf(path,
                           sizeof(path),
                           "%s/frontier_%06d.logits.json",
                           cfg->dump_frontier_logits_dir,
                           frontier);
    if (n <= 0 || (size_t)n >= sizeof(path)) {
        fprintf(stderr, "ds4-bench: frontier logits path is too long\n");
        free(logits);
        return 1;
    }

    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "ds4-bench: failed to open %s: %s\n", path, strerror(errno));
        free(logits);
        return 1;
    }

    const int argmax = ds4_session_argmax(session);
    fprintf(fp, "{\n  \"source\":\"ds4-bench\",\n  \"model\":");
    json_write_string(fp, cfg->model_path);
    fprintf(fp,
            ",\n  \"backend\":\"%s\",\n  \"quality\":%s,\n"
            "  \"quant_bits\":%d,\n  \"prompt_tokens\":%d,\n"
            "  \"frontier_tokens\":%d,\n  \"prefill_tokens\":%d,\n"
            "  \"ctx\":%d,\n  \"vocab\":%d,\n"
            "  \"argmax_id\":%d,\n  \"argmax_logit\":%.9g,\n  \"logits\":[",
            ds4_backend_name(cfg->backend),
            cfg->quality ? "true" : "false",
            ds4_engine_routed_quant_bits(engine),
            frontier,
            frontier,
            frontier - previous,
            cfg->ctx_alloc,
            vocab,
            argmax,
            logits[argmax]);
    for (int i = 0; i < vocab; i++) {
        if (i) fputc(',', fp);
        if ((i % 8) == 0) fputs("\n    ", fp);
        if (json_f32_is_finite(&logits[i])) fprintf(fp, "%.9g", logits[i]);
        else fputs("null", fp);
    }
    fputs("\n  ]\n}\n", fp);
    if (fclose(fp) != 0) {
        fprintf(stderr, "ds4-bench: failed to close %s\n", path);
        free(logits);
        return 1;
    }
    free(logits);
    return 0;
}

#define DS4_BENCH_DECODE_EVIDENCE_SCHEMA "ds4.qwen.decode-evidence/1"

/* The decode timer must measure inference, not evidence serialization.  The
 * caller therefore records token IDs into a preallocated RAM buffer and copies
 * the final logits only after gen_t1.  This helper receives only immutable RAM
 * snapshots and cannot accidentally touch the live session or GPU. */
static int write_decode_evidence_json(
        const bench_config *cfg,
        int                 frontier,
        const int          *token_ids,
        int                 token_count,
        int                 final_argmax,
        const float        *final_logits,
        int                 vocab) {
    if (!cfg->dump_decode_evidence_dir) return 0;
    if (token_count < 0 || (token_count > 0 && !token_ids) ||
        !final_logits || vocab <= 0 || final_argmax < 0 || final_argmax >= vocab)
    {
        fprintf(stderr,
                "ds4-bench: invalid in-memory decode evidence at frontier %d\n",
                frontier);
        return 1;
    }
    for (int i = 0; i < token_count; i++) {
        if (token_ids[i] < 0 || token_ids[i] >= vocab) {
            fprintf(stderr,
                    "ds4-bench: invalid decoded token ID %d at frontier %d\n",
                    token_ids[i],
                    frontier);
            return 1;
        }
    }

    char path[PATH_MAX];
    char tmp_path[PATH_MAX];
    const int path_n = snprintf(path,
                                sizeof(path),
                                "%s/frontier_%06d.decode.json",
                                cfg->dump_decode_evidence_dir,
                                frontier);
    if (path_n <= 0 || (size_t)path_n >= sizeof(path)) {
        fprintf(stderr, "ds4-bench: decode evidence path is too long\n");
        return 1;
    }
    const int tmp_n = snprintf(tmp_path, sizeof(tmp_path), "%s.tmp", path);
    if (tmp_n <= 0 || (size_t)tmp_n >= sizeof(tmp_path)) {
        fprintf(stderr, "ds4-bench: decode evidence temporary path is too long\n");
        return 1;
    }

    FILE *fp = fopen(tmp_path, "wb");
    if (!fp) {
        fprintf(stderr,
                "ds4-bench: failed to open %s: %s\n",
                tmp_path,
                strerror(errno));
        return 1;
    }

    fprintf(fp,
            "{\n  \"schema\":\"%s\",\n"
            "  \"frontier_tokens\":%d,\n"
            "  \"token_ids\":[",
            DS4_BENCH_DECODE_EVIDENCE_SCHEMA,
            frontier);
    for (int i = 0; i < token_count; i++) {
        if (i) fputc(',', fp);
        if ((i % 16) == 0) fputs("\n    ", fp);
        fprintf(fp, "%d", token_ids[i]);
    }
    fprintf(fp,
            "%s  ],\n  \"final_argmax_id\":%d,\n  \"final_logits\":[",
            token_count ? "\n" : "",
            final_argmax);
    for (int i = 0; i < vocab; i++) {
        if (i) fputc(',', fp);
        if ((i % 8) == 0) fputs("\n    ", fp);
        if (json_f32_is_finite(&final_logits[i])) fprintf(fp, "%.9g", final_logits[i]);
        else fputs("null", fp);
    }
    fputs("\n  ]\n}\n", fp);

    /* A partially written JSON file must never match the campaign glob.  Close
     * and validate the temporary file first, then publish it atomically. */
    int write_errno = 0;
    if (ferror(fp)) write_errno = errno ? errno : EIO;
    if (fflush(fp) != 0 && write_errno == 0) write_errno = errno ? errno : EIO;
    if (fclose(fp) != 0 && write_errno == 0) {
        write_errno = errno ? errno : EIO;
    }
    if (write_errno != 0) {
        fprintf(stderr,
                "ds4-bench: failed to write %s: %s\n",
                tmp_path,
                strerror(write_errno));
        remove(tmp_path);
        return 1;
    }
    if (rename(tmp_path, path) != 0) {
        const int rename_errno = errno;
        fprintf(stderr,
                "ds4-bench: failed to publish %s: %s\n",
                path,
                strerror(rename_errno));
        remove(tmp_path);
        return 1;
    }
    return 0;
}

static int next_frontier(const bench_config *c, int cur) {
    if (cur >= c->ctx_max) return c->ctx_max;
    int next;
    if (c->step_mul == 1.0) {
        if (cur > INT_MAX - c->step_incr) next = c->ctx_max;
        else next = cur + c->step_incr;
    } else {
        const double v = ceil((double)cur * c->step_mul);
        next = v > (double)INT_MAX ? c->ctx_max : (int)v;
        if (next <= cur) next = cur + 1;
    }
    if (next > c->ctx_max) next = c->ctx_max;
    return next;
}

static void log_context_memory(ds4_backend backend,
                               int         ctx_size,
                               uint32_t    prefill_chunk) {
    ds4_context_memory m =
        ds4_context_memory_estimate_with_prefill(backend,
                                                 ctx_size,
                                                 prefill_chunk);
    fprintf(stderr,
            "ds4-bench: context buffers %.2f MiB (ctx=%d, backend=%s, prefill_chunk=%u, raw_kv_rows=%u, compressed_kv_rows=%u)\n",
            (double)m.total_bytes / (1024.0 * 1024.0),
            ctx_size,
            ds4_backend_name(backend),
            m.prefill_cap,
            m.raw_cap,
            m.comp_cap);
}

static int wait_distributed_route(ds4_session *session) {
    char err[256] = {0};
    char last[256] = {0};
    unsigned ticks = 0;
    const struct timespec delay = {0, 250000000L};

    for (;;) {
        int ready = ds4_session_distributed_route_ready(session, err, sizeof(err));
        if (ready > 0) {
            if (ticks) fprintf(stderr, "ds4-bench: distributed route ready\n");
            return 0;
        }
        if (ready < 0) {
            fprintf(stderr,
                    "ds4-bench: distributed route readiness failed: %s\n",
                    err[0] ? err : "unknown error");
            return 1;
        }
        const char *why = err[0] ? err : "route incomplete";
        if (strcmp(last, why) != 0 || (ticks % 20u) == 0) {
            fprintf(stderr, "ds4-bench: waiting for distributed route: %s\n", why);
            snprintf(last, sizeof(last), "%s", why);
        }
        nanosleep(&delay, NULL);
        ticks++;
    }
}

static void maybe_warn_distributed_step_shape(const bench_config *cfg, ds4_session *session) {
    if (!cfg || !session || cfg->dist.role != DS4_DISTRIBUTED_COORDINATOR) return;
    uint32_t chunk = cfg->dist.prefill_chunk;
    if (chunk == 0) {
        const int cap = ds4_session_prefill_cap(session);
        if (cap > 0) chunk = (uint32_t)cap;
    }
    if (chunk == 0) return;
    if (cfg->step_mul == 1.0 &&
        cfg->step_incr > 0 &&
        (uint32_t)cfg->step_incr < chunk &&
        cfg->ctx_start < cfg->ctx_max)
    {
        fprintf(stderr,
                "ds4-bench: note: --step-incr=%d is smaller than distributed prefill chunk %u; "
                "suffix rows will not show multi-chunk pipeline overlap\n",
                cfg->step_incr,
                chunk);
    }
}

int main(int argc, char **argv) {
    if (ds4_build_info_requested(argc, argv)) {
        ds4_build_info_print(stdout);
        return 0;
    }
    bench_config cfg = parse_options(argc, argv);

    ds4_engine_options opt = {
        .model_path = cfg.model_path,
        .backend = cfg.backend,
        .n_threads = cfg.threads,
        .context_size = (uint32_t)cfg.ctx_alloc,
        .prefill_chunk = cfg.prefill_chunk,
        .ssd_streaming_cache_experts = cfg.ssd_streaming_cache_experts,
        .ssd_streaming_cache_bytes = cfg.ssd_streaming_cache_bytes,
        .ssd_streaming_full_layers = cfg.ssd_streaming_full_layers,
        .ssd_streaming_preload_experts = cfg.ssd_streaming_preload_experts,
        .simulate_used_memory_bytes = cfg.simulate_used_memory_bytes,
        .power_percent = cfg.power_percent,
        .warm_weights = cfg.warm_weights,
        .quality = cfg.quality,
        .residency = cfg.residency,
        .ssd_streaming_cold = cfg.ssd_streaming_cold,
        .ssd_streaming_full_layers_set = cfg.ssd_streaming_full_layers_set,
        .expert_profile_path = cfg.expert_profile_path,
        .distributed = cfg.dist,
    };
    char dist_err[256];
    if (ds4_dist_prepare_engine_options(&cfg.dist, &opt, dist_err, sizeof(dist_err)) != 0) {
        fprintf(stderr, "ds4-bench: %s\n", dist_err);
        return 2;
    }
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0) return 1;
    log_context_memory(cfg.backend, cfg.ctx_alloc, cfg.prefill_chunk);

    char *text = read_file(cfg.prompt_path ? cfg.prompt_path : cfg.chat_prompt_path);
    ds4_tokens prompt = {0};
    const bool tokenized = cfg.chat_prompt_path ?
        ds4_encode_chat_prompt_checked(engine, cfg.system, text, DS4_THINK_NONE, &prompt) :
        ds4_tokenize_text_checked(engine, text, &prompt);
    free(text);
    if (!tokenized) {
        fprintf(stderr, "ds4-bench: failed to tokenize prompt\n");
        ds4_tokens_free(&prompt);
        ds4_engine_close(engine);
        return 1;
    }

    if (prompt.len < cfg.ctx_max) {
        fprintf(stderr,
                "ds4-bench: prompt has %d tokens, need at least --ctx-max=%d\n",
                prompt.len,
                cfg.ctx_max);
        ds4_tokens_free(&prompt);
        ds4_engine_close(engine);
        return 1;
    }

    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, cfg.ctx_alloc) != 0) {
        fprintf(stderr, "ds4-bench: failed to create session\n");
        ds4_tokens_free(&prompt);
        ds4_engine_close(engine);
        return 1;
    }
    if (cfg.dist.role == DS4_DISTRIBUTED_COORDINATOR &&
        wait_distributed_route(session) != 0)
    {
        ds4_session_free(session);
        ds4_tokens_free(&prompt);
        ds4_engine_close(engine);
        return 1;
    }
    maybe_warn_distributed_step_shape(&cfg, session);

    FILE *out = stdout;
    if (cfg.csv_path) {
        out = fopen(cfg.csv_path, "wb");
        if (!out) {
            fprintf(stderr, "ds4-bench: failed to open %s: %s\n", cfg.csv_path, strerror(errno));
            ds4_session_free(session);
            ds4_tokens_free(&prompt);
            ds4_engine_close(engine);
            return 1;
        }
    }
    fprintf(out,
            "ctx_tokens,prefill_tokens,prefill_tps,prefill_wall_ms,ttft_ms,"
            "prefill_hit_rate,prefill_evictions,prefill_expert_loads,"
            "prefill_unique_experts,prefill_unique_expert_gib,"
            "prefill_read_amplification,prefill_pread_syscalls,"
            "prefill_pread_kib_per_syscall,prefill_pread_gib,"
            "prefill_pread_gib_per_tok,prefill_pread_ms,"
            "prefill_pread_ms_per_tok,prefill_split_resident_wait_ms,"
            "prefill_split_resident_wait_ms_per_tok,gen_tokens,gen_tps,"
            "gen_wall_ms,gen_tpot_p50_ms,gen_tpot_p95_ms,kvcache_bytes,"
            "gen_hit_rate,gen_evictions,gen_expert_loads,gen_unique_experts,"
            "gen_unique_expert_gib,gen_read_amplification,"
            "gen_pread_syscalls,gen_pread_kib_per_syscall,"
            "gen_pread_gib,gen_pread_gib_per_tok,gen_pread_ms,"
            "gen_pread_ms_per_tok,gen_split_resident_wait_ms,"
            "gen_split_resident_wait_ms_per_tok\n");
    fflush(out);

    int *decode_token_ids = NULL;
    float *decode_final_logits = NULL;
    double *decode_token_ms = NULL;
    int decode_vocab = 0;
    if (cfg.gen_tokens > 0) {
        if ((size_t)cfg.gen_tokens > SIZE_MAX / sizeof(decode_token_ms[0])) {
            fprintf(stderr, "ds4-bench: invalid decode latency buffer size\n");
            if (out != stdout) fclose(out);
            ds4_session_free(session);
            ds4_tokens_free(&prompt);
            ds4_engine_close(engine);
            return 1;
        }
        decode_token_ms = malloc(
            (size_t)cfg.gen_tokens * sizeof(decode_token_ms[0]));
        if (!decode_token_ms) {
            fprintf(stderr, "ds4-bench: out of memory allocating decode latency buffer\n");
            if (out != stdout) fclose(out);
            ds4_session_free(session);
            ds4_tokens_free(&prompt);
            ds4_engine_close(engine);
            return 1;
        }
    }
    if (cfg.dump_decode_evidence_dir) {
        decode_vocab = ds4_engine_vocab_size(engine);
        if (decode_vocab <= 0 ||
            (size_t)decode_vocab > SIZE_MAX / sizeof(decode_final_logits[0]) ||
            (cfg.gen_tokens > 0 &&
             (size_t)cfg.gen_tokens > SIZE_MAX / sizeof(decode_token_ids[0])))
        {
            fprintf(stderr, "ds4-bench: invalid decode evidence buffer size\n");
            free(decode_token_ms);
            if (out != stdout) fclose(out);
            ds4_session_free(session);
            ds4_tokens_free(&prompt);
            ds4_engine_close(engine);
            return 1;
        }
        if (cfg.gen_tokens > 0) {
            decode_token_ids = malloc((size_t)cfg.gen_tokens * sizeof(decode_token_ids[0]));
        }
        decode_final_logits = malloc((size_t)decode_vocab * sizeof(decode_final_logits[0]));
        if ((cfg.gen_tokens > 0 && !decode_token_ids) || !decode_final_logits) {
            fprintf(stderr, "ds4-bench: out of memory allocating decode evidence buffers\n");
            free(decode_token_ids);
            free(decode_final_logits);
            free(decode_token_ms);
            if (out != stdout) fclose(out);
            ds4_session_free(session);
            ds4_tokens_free(&prompt);
            ds4_engine_close(engine);
            return 1;
        }
    }

    const int eos = ds4_token_eos(engine);
    const bool distributed = cfg.dist.role == DS4_DISTRIBUTED_COORDINATOR;
    const bool replay_restore = ds4_engine_is_qwen35(engine);
    ds4_session_snapshot snap = {0};
    char err[256];
    int previous = 0;
    int rc = 0;

    for (int frontier = cfg.ctx_start; ; frontier = next_frontier(&cfg, frontier)) {
        ds4_tokens prefix = {
            .v = prompt.v,
            .len = frontier,
            .cap = frontier,
        };

        /* Attribute storage traffic before entering sync.  These counters are
         * cumulative by design: taking both snapshots outside the timed call
         * adds no instrumentation to the engine hot path and keeps prefill and
         * decode I/O disjoint in the report. */
        uint64_t prefill_cache_hits = 0, prefill_cache_misses = 0;
        uint64_t prefill_cache_evictions = 0, prefill_expert_loads = 0;
        uint64_t prefill_pread_syscalls = 0;
        uint64_t prefill_pread_bytes = 0;
        uint64_t prefill_unique_experts = 0;
        uint64_t prefill_unique_bytes = 0;
        double prefill_pread_ms = 0.0;
        double prefill_split_resident_wait_ms = 0.0;
#ifndef DS4_NO_GPU
        ds4_gpu_stream_expert_cache_stats_v1 prefill_stats0 = {0};
        if (!ds4_gpu_stream_expert_cache_snapshot_v1(&prefill_stats0)) {
            fprintf(stderr, "ds4-bench: failed to snapshot prefill cache counters\n");
            rc = 1;
            break;
        }
        ds4_gpu_stream_expert_io_measurement_begin();
#endif
        const double prefill_t0 = bench_now_sec();
        const int prefill_rc =
            ds4_session_sync(session, &prefix, err, sizeof(err));
        const double prefill_t1 = bench_now_sec();
#ifndef DS4_NO_GPU
        ds4_gpu_stream_expert_io_measurement_end(
            &prefill_unique_experts, &prefill_unique_bytes);
#endif
        if (prefill_rc != 0) {
            fprintf(stderr, "ds4-bench: prefill to %d failed: %s\n", frontier, err);
            rc = 1;
            break;
        }
#ifndef DS4_NO_GPU
        {
            ds4_gpu_stream_expert_cache_stats_v1 prefill_stats1 = {0};
            if (!ds4_gpu_stream_expert_cache_snapshot_v1(&prefill_stats1)) {
                fprintf(stderr, "ds4-bench: failed to snapshot prefill cache counters\n");
                rc = 1;
                break;
            }
            prefill_cache_hits = prefill_stats1.hits - prefill_stats0.hits;
            prefill_cache_misses = prefill_stats1.misses - prefill_stats0.misses;
            prefill_cache_evictions =
                prefill_stats1.evictions - prefill_stats0.evictions;
            prefill_expert_loads =
                prefill_stats1.expert_loads - prefill_stats0.expert_loads;
            prefill_pread_syscalls =
                prefill_stats1.pread_syscalls - prefill_stats0.pread_syscalls;
            prefill_pread_bytes =
                prefill_stats1.pread_bytes - prefill_stats0.pread_bytes;
            prefill_pread_ms =
                prefill_stats1.pread_wall_ms - prefill_stats0.pread_wall_ms;
            prefill_split_resident_wait_ms =
                prefill_stats1.split_resident_wait_ms -
                prefill_stats0.split_resident_wait_ms;
        }
#endif
        if (rc != 0) break;
        const double prefill_sec = prefill_t1 - prefill_t0;
        const int prefill_tokens = frontier - previous;

        if (write_frontier_logits_json(&cfg, engine, session, frontier, previous) != 0) {
            rc = 1;
            break;
        }

        if (cfg.gen_tokens > 0 && !distributed && !replay_restore) {
            if (ds4_session_save_snapshot(session, &snap, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-bench: snapshot at %d failed: %s\n", frontier, err);
                rc = 1;
                break;
            }
        }

        /* Decode-window streaming-expert-cache attribution: snapshot the cumulative
           counters across exactly the gen loop so hit-rate and exposed miss-wait are
           scoped to decode (Metal streaming only; zero elsewhere). */
        uint64_t gen_cache_hits = 0, gen_cache_misses = 0;
        uint64_t gen_cache_evictions = 0, gen_expert_loads = 0;
        uint64_t gen_pread_syscalls = 0;
        uint64_t gen_pread_bytes = 0;
        uint64_t gen_unique_experts = 0;
        uint64_t gen_unique_bytes = 0;
        double gen_pread_ms = 0.0;
        double gen_split_resident_wait_ms = 0.0;
        int decode_token_count = 0;
        double first_token_ready_sec = 0.0;
#ifndef DS4_NO_GPU
        ds4_gpu_stream_expert_cache_stats_v1 gen_stats0 = {0};
        if (!ds4_gpu_stream_expert_cache_snapshot_v1(&gen_stats0)) {
            fprintf(stderr, "ds4-bench: failed to snapshot decode cache counters\n");
            rc = 1;
            break;
        }
        ds4_gpu_stream_expert_io_measurement_begin();
#endif
        const double gen_t0 = bench_now_sec();
        for (int i = 0; i < cfg.gen_tokens; i++) {
            if (ds4_session_pos(session) + 1 >= ds4_session_ctx(session)) {
                fprintf(stderr, "ds4-bench: generation would exceed allocated context at frontier %d\n", frontier);
                rc = 1;
                break;
            }
            const double token_t0 = bench_now_sec();
            const int token = ds4_session_argmax_excluding(session, eos);
            if (token < 0) {
                fprintf(stderr, "ds4-bench: failed to choose non-EOS token at frontier %d\n", frontier);
                rc = 1;
                break;
            }
            if (i == 0) {
                /* TTFT stops when the first token is selectable, before its
                 * eval.  Add this small selection interval to the isolated
                 * prefill timing, excluding snapshot/evidence serialization. */
                first_token_ready_sec = bench_now_sec() - gen_t0;
            }
            if (decode_token_ids) decode_token_ids[decode_token_count] = token;
            if (ds4_session_eval(session, token, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-bench: decode at frontier %d failed: %s\n", frontier, err);
                rc = 1;
                break;
            }
            decode_token_ms[decode_token_count] =
                (bench_now_sec() - token_t0) * 1000.0;
            decode_token_count++;
        }
        const double gen_t1 = bench_now_sec();
#ifndef DS4_NO_GPU
        ds4_gpu_stream_expert_io_measurement_end(
            &gen_unique_experts, &gen_unique_bytes);
        {
            ds4_gpu_stream_expert_cache_stats_v1 gen_stats1 = {0};
            if (!ds4_gpu_stream_expert_cache_snapshot_v1(&gen_stats1)) {
                fprintf(stderr, "ds4-bench: failed to snapshot decode cache counters\n");
                rc = 1;
                break;
            }
            gen_cache_hits = gen_stats1.hits - gen_stats0.hits;
            gen_cache_misses = gen_stats1.misses - gen_stats0.misses;
            gen_cache_evictions =
                gen_stats1.evictions - gen_stats0.evictions;
            gen_expert_loads =
                gen_stats1.expert_loads - gen_stats0.expert_loads;
            gen_pread_syscalls =
                gen_stats1.pread_syscalls - gen_stats0.pread_syscalls;
            gen_pread_bytes =
                gen_stats1.pread_bytes - gen_stats0.pread_bytes;
            gen_pread_ms =
                gen_stats1.pread_wall_ms - gen_stats0.pread_wall_ms;
            gen_split_resident_wait_ms =
                gen_stats1.split_resident_wait_ms -
                gen_stats0.split_resident_wait_ms;
        }
#endif
        if (rc != 0) break;

        if (cfg.dump_decode_evidence_dir) {
            if (decode_token_count != cfg.gen_tokens) {
                fprintf(stderr,
                        "ds4-bench: incomplete decode evidence at frontier %d\n",
                        frontier);
                rc = 1;
                break;
            }
            /* Both operations deliberately occur after gen_t1.  In particular,
             * serializing a ~1 MiB vocabulary must never depress reported TPS. */
            if (ds4_session_copy_logits(session,
                                        decode_final_logits,
                                        decode_vocab) != decode_vocab)
            {
                fprintf(stderr,
                        "ds4-bench: failed to copy final decode logits at frontier %d\n",
                        frontier);
                rc = 1;
                break;
            }
            const int final_argmax = ds4_session_argmax(session);
            if (write_decode_evidence_json(&cfg,
                                           frontier,
                                           decode_token_ids,
                                           decode_token_count,
                                           final_argmax,
                                           decode_final_logits,
                                           decode_vocab) != 0)
            {
                rc = 1;
                break;
            }
        }

        const bool need_restore = cfg.gen_tokens > 0 && frontier < cfg.ctx_max;
        if (cfg.gen_tokens == 0 || !need_restore) {
            /* Pure prefill benchmark: leave the live session at the frontier. */
        } else if (distributed || replay_restore) {
            if (ds4_session_sync(session, &prefix, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-bench: replay restore at %d failed: %s\n", frontier, err);
                rc = 1;
                break;
            }
        } else {
            if (ds4_session_load_snapshot(session, &snap, err, sizeof(err)) != 0) {
                fprintf(stderr, "ds4-bench: restore at %d failed: %s\n", frontier, err);
                rc = 1;
                break;
            }
        }

        const double gen_sec = gen_t1 - gen_t0;
        if (decode_token_count > 1) {
            qsort(decode_token_ms,
                  (size_t)decode_token_count,
                  sizeof(decode_token_ms[0]),
                  compare_double_ascending);
        }
        const double gen_tpot_p50_ms = nearest_rank_percentile(
            decode_token_ms, (size_t)decode_token_count, 0.50);
        const double gen_tpot_p95_ms = nearest_rank_percentile(
            decode_token_ms, (size_t)decode_token_count, 0.95);
        const double ttft_ms =
            (prefill_sec + first_token_ready_sec) * 1000.0;
        const uint64_t prefill_lookups =
            prefill_cache_hits + prefill_cache_misses;
        const double prefill_hit_rate = prefill_lookups ?
            (double)prefill_cache_hits / (double)prefill_lookups : 0.0;
        const double prefill_pread_gib =
            (double)prefill_pread_bytes / (1024.0 * 1024.0 * 1024.0);
        const double prefill_unique_expert_gib =
            (double)prefill_unique_bytes / (1024.0 * 1024.0 * 1024.0);
        const double prefill_read_amplification = prefill_unique_bytes > 0 ?
            (double)prefill_pread_bytes / (double)prefill_unique_bytes : 0.0;
        const double prefill_pread_kib_per_syscall =
            prefill_pread_syscalls > 0 ?
                ((double)prefill_pread_bytes / 1024.0) /
                    (double)prefill_pread_syscalls : 0.0;
        const double prefill_pread_gib_per_tok = prefill_tokens > 0 ?
            prefill_pread_gib / (double)prefill_tokens : 0.0;
        const double prefill_pread_ms_per_tok = prefill_tokens > 0 ?
            prefill_pread_ms / (double)prefill_tokens : 0.0;
        const double prefill_split_resident_wait_ms_per_tok =
            prefill_tokens > 0 ?
                prefill_split_resident_wait_ms / (double)prefill_tokens : 0.0;
        const uint64_t gen_lookups = gen_cache_hits + gen_cache_misses;
        const double gen_hit_rate = gen_lookups ?
            (double)gen_cache_hits / (double)gen_lookups : 0.0;
        const double gen_pread_kib_per_syscall =
            gen_pread_syscalls > 0 ?
                ((double)gen_pread_bytes / 1024.0) /
                    (double)gen_pread_syscalls : 0.0;
        const double gen_pread_gib =
            (double)gen_pread_bytes / (1024.0 * 1024.0 * 1024.0);
        const double gen_unique_expert_gib =
            (double)gen_unique_bytes / (1024.0 * 1024.0 * 1024.0);
        const double gen_read_amplification = gen_unique_bytes > 0 ?
            (double)gen_pread_bytes / (double)gen_unique_bytes : 0.0;
        const double gen_pread_gib_per_tok = cfg.gen_tokens > 0 ?
            gen_pread_gib / (double)cfg.gen_tokens : 0.0;
        const double gen_pread_ms_per_tok = cfg.gen_tokens > 0 ?
            gen_pread_ms / (double)cfg.gen_tokens : 0.0;
        const double gen_split_resident_wait_ms_per_tok = cfg.gen_tokens > 0 ?
            gen_split_resident_wait_ms / (double)cfg.gen_tokens : 0.0;
        fprintf(out,
                "%d,%d,%.2f,%.4f,%.4f,%.4f,%llu,%llu,%llu,%.6f,"
                "%.6f,%llu,%.3f,%.6f,%.9f,%.4f,%.6f,%.4f,%.6f,"
                "%d,%.2f,%.4f,%.4f,%.4f,%llu,%.4f,%llu,%llu,%llu,"
                "%.6f,%.6f,%llu,%.3f,%.6f,%.9f,%.4f,%.4f,%.4f,%.4f\n",
                frontier,
                prefill_tokens,
                prefill_sec > 0.0 ? (double)prefill_tokens / prefill_sec : 0.0,
                prefill_sec * 1000.0,
                ttft_ms,
                prefill_hit_rate,
                (unsigned long long)prefill_cache_evictions,
                (unsigned long long)prefill_expert_loads,
                (unsigned long long)prefill_unique_experts,
                prefill_unique_expert_gib,
                prefill_read_amplification,
                (unsigned long long)prefill_pread_syscalls,
                prefill_pread_kib_per_syscall,
                prefill_pread_gib,
                prefill_pread_gib_per_tok,
                prefill_pread_ms,
                prefill_pread_ms_per_tok,
                prefill_split_resident_wait_ms,
                prefill_split_resident_wait_ms_per_tok,
                cfg.gen_tokens,
                gen_sec > 0.0 ? (double)cfg.gen_tokens / gen_sec : 0.0,
                gen_sec * 1000.0,
                gen_tpot_p50_ms,
                gen_tpot_p95_ms,
                (unsigned long long)(distributed ? 0 : snap.len),
                gen_hit_rate,
                (unsigned long long)gen_cache_evictions,
                (unsigned long long)gen_expert_loads,
                (unsigned long long)gen_unique_experts,
                gen_unique_expert_gib,
                gen_read_amplification,
                (unsigned long long)gen_pread_syscalls,
                gen_pread_kib_per_syscall,
                gen_pread_gib,
                gen_pread_gib_per_tok,
                gen_pread_ms,
                gen_pread_ms_per_tok,
                gen_split_resident_wait_ms,
                gen_split_resident_wait_ms_per_tok);
        fflush(out);

        previous = frontier;
        if (frontier >= cfg.ctx_max) break;
    }

    if (out != stdout) fclose(out);
    free(decode_token_ms);
    free(decode_token_ids);
    free(decode_final_logits);
    ds4_session_snapshot_free(&snap);
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
    ds4_engine_close(engine);
    return rc;
}
