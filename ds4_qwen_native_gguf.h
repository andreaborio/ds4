#ifndef DS4_QWEN_NATIVE_GGUF_H
#define DS4_QWEN_NATIVE_GGUF_H

#include "ds4_qwen_expert_pack.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define DS4_QWEN_NATIVE_EXPERT_TENSOR "ds4.expert_major.v1"

typedef struct {
    ds4_qwen_expert_pack_geometry geometry;
    uint64_t filesystem_reserve_bytes;
    ds4_qwen_expert_pack_progress_fn progress;
    void *progress_context;
} ds4_qwen_native_gguf_options;

/* Replace the 120 canonical Qwen routed matrices with one opaque I8 tensor
 * containing the already verified expert-major store. Non-routed tensors and
 * metadata are copied byte-for-byte; routed weights are present exactly once.
 * The destination is written, fully verified, fsynced, and atomically renamed
 * in its own directory. */
bool ds4_qwen_native_gguf_build(
        const char                           *source_gguf_path,
        const char                           *expert_pack_path,
        const char                           *destination_gguf_path,
        const ds4_qwen_native_gguf_options   *options,
        char                                 *error,
        size_t                                error_size);

/* Offline verifier for publication gates. It proves metadata and every
 * non-routed tensor match the source, validates the embedded index, hashes the
 * embedded payload, and binds the store manifest to the source GGUF digest. */
bool ds4_qwen_native_gguf_verify(
        const char                           *source_gguf_path,
        const char                           *native_gguf_path,
        const ds4_qwen_native_gguf_options   *options,
        char                                 *error,
        size_t                                error_size);

#endif
