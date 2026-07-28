#include "ds4.h"
#include "ds4_expert_store.h"
#include "hebrus_identity.h"

#include <string.h>

#ifndef DS4_BUILD_GIT_SHA
#define DS4_BUILD_GIT_SHA "unknown"
#endif

#define DS4_BUILD_STATIC_ASSERT(name, condition) \
    typedef char ds4_build_static_assert_##name[(condition) ? 1 : -1]

DS4_BUILD_STATIC_ASSERT(expert_store_v2_version,
                        DS4_EXPERT_STORE_V2_VERSION == 2);
DS4_BUILD_STATIC_ASSERT(expert_store_ggml_wire_value,
                        DS4_EXPERT_STORE_STORAGE_GGML == 0);
DS4_BUILD_STATIC_ASSERT(expert_store_mlx_affine4_wire_value,
                        DS4_EXPERT_STORE_STORAGE_MLX_AFFINE4 == 1);
DS4_BUILD_STATIC_ASSERT(expert_store_qwen_affine2_wire_value,
                        DS4_EXPERT_STORE_STORAGE_QWEN_AFFINE2_G32_IQ_DOWN == 2);

const char *ds4_build_backend(void) {
#ifdef DS4_NO_GPU
    return "cpu";
#else
    return "metal";
#endif
}

const char *ds4_build_arch(void) {
#if defined(__aarch64__) || defined(__arm64__)
    return "arm64";
#elif defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#else
    return "unknown";
#endif
}

const char *ds4_build_git_sha(void) {
    return DS4_BUILD_GIT_SHA;
}

void ds4_build_info_print(FILE *fp, const char *argv0) {
    if (!fp) fp = stdout;
    fprintf(fp,
            "%s build\n"
            "git:     %s\n"
            "backend: %s\n"
            "arch:    %s\n",
            hebrus_is_canonical_invocation(argv0) ? "hebrus" : "ds4",
            ds4_build_git_sha(),
            ds4_build_backend(),
            ds4_build_arch());
}

bool ds4_build_info_requested(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        if (argv[i] && strcmp(argv[i], "--build-info") == 0) return true;
    }
    return false;
}

bool ds4_capabilities_requested(int argc, char **argv) {
    for (int i = 1; i < argc; i++) {
        if (argv[i] && strcmp(argv[i], "--capabilities=json") == 0) return true;
    }
    return false;
}

static const char *ds4_executable_role_name(ds4_executable_role role) {
    switch (role) {
    case DS4_EXECUTABLE_ROLE_CLI:
        return "cli";
    case DS4_EXECUTABLE_ROLE_SERVER:
        return "server";
    case DS4_EXECUTABLE_ROLE_AGENT:
        return "agent";
    case DS4_EXECUTABLE_ROLE_BENCH:
        return "bench";
    case DS4_EXECUTABLE_ROLE_EVAL:
        return "eval";
    }
    return "unknown";
}

static void ds4_json_string_print(FILE *fp, const char *value) {
    fputc('"', fp);
    for (const unsigned char *p = (const unsigned char *)value; *p; p++) {
        switch (*p) {
        case '"':
            fputs("\\\"", fp);
            break;
        case '\\':
            fputs("\\\\", fp);
            break;
        case '\b':
            fputs("\\b", fp);
            break;
        case '\f':
            fputs("\\f", fp);
            break;
        case '\n':
            fputs("\\n", fp);
            break;
        case '\r':
            fputs("\\r", fp);
            break;
        case '\t':
            fputs("\\t", fp);
            break;
        default:
            if (*p < 0x20) {
                fprintf(fp, "\\u%04x", (unsigned)*p);
            } else {
                fputc(*p, fp);
            }
            break;
        }
    }
    fputc('"', fp);
}

void ds4_capabilities_print(FILE *fp, ds4_executable_role role,
                            const char *argv0) {
    if (!fp) fp = stdout;
    fputs("{\n"
          "  \"schema_version\": 1,\n"
          "  \"engine_id\": ", fp);
    ds4_json_string_print(fp,
        hebrus_is_canonical_invocation(argv0) ? "hebrus" : "ds4");
    fputs(",\n  \"build_git_sha\": ", fp);
    ds4_json_string_print(fp, ds4_build_git_sha());
    fputs(",\n  \"backend\": ", fp);
    ds4_json_string_print(fp, ds4_build_backend());
    fputs(",\n  \"executable_role\": ", fp);
    ds4_json_string_print(fp, ds4_executable_role_name(role));
    fprintf(fp,
            ",\n"
            "  \"model_families\": [\"deepseek4\", \"glm-dsa\", \"qwen35moe\"],\n"
            "  \"expert_major\": {\n"
            "    \"version\": %u,\n"
            "    \"tensor\": \"%s\",\n"
            "    \"storage_formats\": [\n"
            "      {\"id\": \"ggml\", \"wire_value\": %u, \"group_sizes\": []},\n"
            "      {\"id\": \"mlx-affine4\", \"wire_value\": %u, \"group_sizes\": [64]},\n"
            "      {\"id\": \"qwen-affine2-iq-down\", \"wire_value\": %u, \"group_sizes\": [32]}\n"
            "    ]\n"
            "  }\n"
            "}\n",
            (unsigned)DS4_EXPERT_STORE_V2_VERSION,
            DS4_EXPERT_STORE_V2_TENSOR,
            (unsigned)DS4_EXPERT_STORE_STORAGE_GGML,
            (unsigned)DS4_EXPERT_STORE_STORAGE_MLX_AFFINE4,
            (unsigned)DS4_EXPERT_STORE_STORAGE_QWEN_AFFINE2_G32_IQ_DOWN);
}
