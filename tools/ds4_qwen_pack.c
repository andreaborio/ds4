#include "ds4_qwen_expert_pack.h"
#include "ds4_qwen_native_gguf.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { DEFAULT_FILESYSTEM_RESERVE_GIB = 1 };

typedef struct {
    ds4_qwen_expert_pack_phase phase;
    unsigned last_percent;
    bool started;
} progress_state;

static const char *phase_name(ds4_qwen_expert_pack_phase phase) {
    switch (phase) {
    case DS4_QWEN_EXPERT_PACK_HASH_SOURCE: return "hash GGUF";
    case DS4_QWEN_EXPERT_PACK_WRITE_DATA: return "write experts";
    case DS4_QWEN_EXPERT_PACK_VERIFY_DATA: return "verify pack hash";
    case DS4_QWEN_EXPERT_PACK_VERIFY_SOURCE_SPANS: return "compare GGUF spans";
    case DS4_QWEN_EXPERT_PACK_WRITE_NATIVE_GGUF: return "write native GGUF";
    case DS4_QWEN_EXPERT_PACK_VERIFY_NATIVE_GGUF: return "verify native GGUF";
    }
    return "work";
}

static void progress_report(
        void *context,
        ds4_qwen_expert_pack_phase phase,
        uint64_t completed,
        uint64_t total) {
    progress_state *state = context;
    const unsigned percent = total == 0 ? 100 :
        (unsigned)((completed * UINT64_C(100)) / total);
    if (!state->started || state->phase != phase) {
        if (state->started) fputc('\n', stderr);
        state->started = true;
        state->phase = phase;
        state->last_percent = UINT32_MAX;
    }
    if (percent == state->last_percent ||
        (percent != 100 && percent % 5 != 0)) {
        return;
    }
    fprintf(stderr, "\r%-20s %3u%%", phase_name(phase), percent);
    fflush(stderr);
    state->last_percent = percent;
}

static void progress_finish(progress_state *state) {
    if (state->started) fputc('\n', stderr);
    state->started = false;
}

static bool parse_bytes(const char *text, uint64_t *out) {
    errno = 0;
    char *end = NULL;
    const unsigned long long value = strtoull(text, &end, 10);
    if (errno != 0 || end == text) return false;
    uint64_t multiplier = 1;
    if (*end != '\0') {
        if (strcmp(end, "KiB") == 0) multiplier = UINT64_C(1024);
        else if (strcmp(end, "MiB") == 0) multiplier = UINT64_C(1024) * 1024;
        else if (strcmp(end, "GiB") == 0) {
            multiplier = UINT64_C(1024) * 1024 * 1024;
        } else {
            return false;
        }
    }
    if ((uint64_t)value > UINT64_MAX / multiplier) return false;
    *out = (uint64_t)value * multiplier;
    return true;
}

static void print_hash(const uint8_t digest[32]) {
    for (size_t i = 0; i < 32; i++) printf("%02x", digest[i]);
}

static void print_manifest_identity(
        const ds4_qwen_expert_pack_manifest *manifest) {
    if (!manifest) return;
    printf("format_version: %u\nsource_sha256: ",
           DS4_QWEN_EXPERT_PACK_FORMAT_VERSION);
    print_hash(manifest->source_sha256);
    printf("\npayload_sha256: ");
    print_hash(manifest->data_sha256);
    printf("\nsource_bytes: %" PRIu64
           "\npacked_bytes: %" PRIu64
           "\nentries: %" PRIu64 "\n",
           manifest->source_size,
           manifest->data_size,
           manifest->entry_count);
}

static void usage(FILE *stream, const char *program) {
    fprintf(stream,
            "usage:\n"
            "  %s build [--reserve-bytes N[KiB|MiB|GiB]] GGUF PACK\n"
            "  %s verify GGUF PACK\n"
            "  %s native [--reserve-bytes N[KiB|MiB|GiB]] GGUF PACK OUTPUT.gguf\n"
            "  %s verify-native GGUF OUTPUT.gguf\n\n"
            "The build command accepts only ds4's fixed Qwen3.6-35B-A3B "
            "Q4_K geometry.\n"
            "It writes PACK.tmp.*, verifies it completely, then atomically "
            "renames it. The native command replaces the canonical routed "
            "tensors with the verified pack inside one DS4-native GGUF.\n",
            program, program, program, program);
}

static int build_pack(int argc, char **argv) {
    uint64_t reserve = (uint64_t)DEFAULT_FILESYSTEM_RESERVE_GIB *
                       1024 * 1024 * 1024;
    int arg = 2;
    if (arg < argc && strcmp(argv[arg], "--reserve-bytes") == 0) {
        if (arg + 1 >= argc || !parse_bytes(argv[arg + 1], &reserve)) {
            fprintf(stderr, "invalid --reserve-bytes value\n");
            return 2;
        }
        arg += 2;
    }
    if (argc - arg != 2) {
        usage(stderr, argv[0]);
        return 2;
    }

    progress_state progress = {0};
    const ds4_qwen_expert_pack_build_options options = {
        .geometry = ds4_qwen35_expert_pack_geometry(),
        .filesystem_reserve_bytes = reserve,
        .progress = progress_report,
        .progress_context = &progress,
    };
    char error[512] = {0};
    const bool ok = ds4_qwen_expert_pack_build(
        argv[arg], argv[arg + 1], &options, error, sizeof(error));
    progress_finish(&progress);
    if (!ok) {
        fprintf(stderr, "ds4-qwen-pack: %s\n",
                error[0] ? error : "build failed");
        return 1;
    }
    if (error[0]) {
        fprintf(stderr, "ds4-qwen-pack: warning: %s\n", error);
    }
    printf("expert pack installed atomically: %s\n", argv[arg + 1]);

    ds4_qwen_expert_pack *pack = NULL;
    const ds4_qwen_expert_pack_geometry geometry =
        ds4_qwen35_expert_pack_geometry();
    error[0] = '\0';
    if (ds4_qwen_expert_pack_open(
            &pack, argv[arg + 1], &geometry, error,
            sizeof(error)) != DS4_QWEN_EXPERT_PACK_OK) {
        fprintf(stderr,
                "ds4-qwen-pack: installed pack could not be reopened: %s\n",
                error[0] ? error : "unknown error");
        return 1;
    }
    print_manifest_identity(ds4_qwen_expert_pack_manifest_get(pack));
    ds4_qwen_expert_pack_close(pack);
    return 0;
}

static int verify_pack(int argc, char **argv) {
    if (argc != 4) {
        usage(stderr, argv[0]);
        return 2;
    }
    char error[512] = {0};
    ds4_qwen_expert_pack *pack = NULL;
    const ds4_qwen_expert_pack_geometry geometry =
        ds4_qwen35_expert_pack_geometry();
    if (ds4_qwen_expert_pack_open(
            &pack, argv[3], &geometry, error,
            sizeof(error)) != DS4_QWEN_EXPERT_PACK_OK) {
        fprintf(stderr, "ds4-qwen-pack: %s\n", error);
        return 1;
    }
    fprintf(stderr, "hashing source GGUF for an exact identity check...\n");
    if (ds4_qwen_expert_pack_validate_source_file(
            pack, argv[2], error,
            sizeof(error)) != DS4_QWEN_EXPERT_PACK_OK) {
        fprintf(stderr, "ds4-qwen-pack: %s\n", error);
        ds4_qwen_expert_pack_close(pack);
        return 1;
    }
    fprintf(stderr, "hashing packed payload...\n");
    if (ds4_qwen_expert_pack_verify_payload(
            pack, error, sizeof(error)) != DS4_QWEN_EXPERT_PACK_OK) {
        fprintf(stderr, "ds4-qwen-pack: %s\n", error);
        ds4_qwen_expert_pack_close(pack);
        return 1;
    }
    const ds4_qwen_expert_pack_manifest *manifest =
        ds4_qwen_expert_pack_manifest_get(pack);
    printf("expert pack valid\n");
    print_manifest_identity(manifest);
    ds4_qwen_expert_pack_close(pack);
    return 0;
}

static int build_native(int argc, char **argv) {
    uint64_t reserve = (uint64_t)DEFAULT_FILESYSTEM_RESERVE_GIB *
                       1024 * 1024 * 1024;
    int arg = 2;
    if (arg < argc && strcmp(argv[arg], "--reserve-bytes") == 0) {
        if (arg + 1 >= argc || !parse_bytes(argv[arg + 1], &reserve)) {
            fprintf(stderr, "invalid --reserve-bytes value\n");
            return 2;
        }
        arg += 2;
    }
    if (argc - arg != 3) {
        usage(stderr, argv[0]);
        return 2;
    }
    progress_state progress = {0};
    const ds4_qwen_native_gguf_options options = {
        .geometry = ds4_qwen35_expert_pack_geometry(),
        .filesystem_reserve_bytes = reserve,
        .progress = progress_report,
        .progress_context = &progress,
    };
    char error[512] = {0};
    const bool ok = ds4_qwen_native_gguf_build(
        argv[arg], argv[arg + 1], argv[arg + 2],
        &options, error, sizeof(error));
    progress_finish(&progress);
    if (!ok) {
        fprintf(stderr, "ds4-qwen-pack: %s\n",
                error[0] ? error : "native GGUF build failed");
        return 1;
    }
    if (error[0]) {
        fprintf(stderr, "ds4-qwen-pack: warning: %s\n", error);
    }
    printf("DS4-native expert-major GGUF installed atomically: %s\n",
           argv[arg + 2]);
    return 0;
}

static int verify_native(int argc, char **argv) {
    if (argc != 4) {
        usage(stderr, argv[0]);
        return 2;
    }
    progress_state progress = {0};
    const ds4_qwen_native_gguf_options options = {
        .geometry = ds4_qwen35_expert_pack_geometry(),
        .progress = progress_report,
        .progress_context = &progress,
    };
    char error[512] = {0};
    const bool ok = ds4_qwen_native_gguf_verify(
        argv[2], argv[3], &options, error, sizeof(error));
    progress_finish(&progress);
    if (!ok) {
        fprintf(stderr, "ds4-qwen-pack: %s\n",
                error[0] ? error : "native GGUF verification failed");
        return 1;
    }
    printf("DS4-native expert-major GGUF valid: %s\n", argv[3]);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        usage(stderr, argv[0]);
        return 2;
    }
    if (strcmp(argv[1], "build") == 0) return build_pack(argc, argv);
    if (strcmp(argv[1], "verify") == 0) return verify_pack(argc, argv);
    if (strcmp(argv[1], "native") == 0) return build_native(argc, argv);
    if (strcmp(argv[1], "verify-native") == 0) {
        return verify_native(argc, argv);
    }
    if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        usage(stdout, argv[0]);
        return 0;
    }
    usage(stderr, argv[0]);
    return 2;
}
