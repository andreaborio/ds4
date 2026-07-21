#ifndef DS4_H
#define DS4_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "ds4_ssd.h"

/* Public engine boundary.
 *
 * The CLI and server should treat ds4_engine as the loaded model and
 * ds4_session as one mutable inference timeline.  A session owns the live KV
 * cache and logits; callers provide full token prefixes and let
 * ds4_session_sync() reuse, extend, or rebuild the graph state.  Keep this
 * header narrow so HTTP/CLI code does not depend on tensor internals. */

typedef enum {
    DS4_BACKEND_METAL = 0,
    /* Value 1 belonged to the frozen CUDA backend. Keep the public numeric
     * contract stable even though that backend is absent from this tree. */
    DS4_BACKEND_CPU = 2,
} ds4_backend;

typedef enum {
    DS4_THINK_NONE,
    DS4_THINK_HIGH,
    DS4_THINK_MAX,
} ds4_think_mode;

typedef enum {
    DS4_CHAT_FORMAT_DEEPSEEK_V4,
    DS4_CHAT_FORMAT_QWEN36,
} ds4_chat_format;

typedef enum {
    DS4_CHAT_ROLE_SYSTEM,
    DS4_CHAT_ROLE_USER,
    DS4_CHAT_ROLE_ASSISTANT,
    DS4_CHAT_ROLE_TOOL,
} ds4_chat_role;

typedef struct {
    const char *name;
    /* Decoded text for strings; canonical JSON for every other type. */
    const char *value;
    bool is_string;
} ds4_chat_tool_arg;

typedef struct {
    const char *name;
    const ds4_chat_tool_arg *args;
    size_t n_args;
} ds4_chat_tool_call;

typedef struct {
    ds4_chat_role role;
    /* Borrowed, NUL-terminated text. Multimodal content is outside this
     * renderer's deliberately text-only Qwen3.6 subset. */
    const char *content;
    /* NULL means the field was absent and enables the template's fallback
     * extraction from an embedded <think>...</think> block. */
    const char *reasoning;
    const ds4_chat_tool_call *tool_calls;
    size_t n_tool_calls;
} ds4_chat_message;

typedef struct {
    const ds4_chat_message *messages;
    size_t n_messages;
    /* Borrowed canonical JSON C strings in API order, including the function
     * wrapper. Tool names and argument names/values are borrowed as well. */
    const char *const *tools_json;
    size_t n_tools;
    ds4_think_mode think_mode;
    bool add_generation_prompt;
    bool preserve_thinking;
} ds4_chat_request;

typedef enum {
    DS4_LOG_DEFAULT,
    DS4_LOG_PREFILL,
    DS4_LOG_GENERATION,
    DS4_LOG_KVCACHE,
    DS4_LOG_TOOL,
    DS4_LOG_WARNING,
    DS4_LOG_TIMING,
    DS4_LOG_OK,
    DS4_LOG_ERROR,
} ds4_log_type;

typedef struct {
    int *v;
    int len;
    int cap;
} ds4_tokens;

typedef struct {
    int id;
    float logit;
    float logprob;
} ds4_token_score;

#define DS4_DEFAULT_TEMPERATURE 1.0f
#define DS4_DEFAULT_TOP_P 1.0f
#define DS4_DEFAULT_MIN_P 0.05f

typedef struct ds4_engine ds4_engine;
typedef struct ds4_session ds4_session;

typedef void (*ds4_session_progress_fn)(void *ud, const char *event, int current, int total);
typedef bool (*ds4_session_cancel_fn)(void *ud);

#define DS4_SESSION_SYNC_INTERRUPTED 2

typedef struct {
    const char *model_path;
    const char *mtp_path;
    ds4_backend backend;
    int n_threads;
    /* Context hint used by AUTO residency planning.  Zero uses the normal
     * 32K startup context; front-ends should pass their configured context. */
    uint32_t context_size;
    uint32_t prefill_chunk;
    int mtp_draft_tokens;
    float mtp_margin;
    const char *directional_steering_file;
    const char *expert_profile_path;
    float directional_steering_attn;
    float directional_steering_ffn;
    int power_percent;
    uint32_t ssd_streaming_cache_experts;
    uint64_t ssd_streaming_cache_bytes;
    uint32_t ssd_streaming_preload_experts;
    uint64_t simulate_used_memory_bytes;
    bool warm_weights;
    bool quality;
    ds4_residency_mode residency;
    /* Legacy source-compatible opt-in. New callers should set residency to
     * DS4_RESIDENCY_SSD; true here is normalized to that mode by the core. */
    bool ssd_streaming;
    bool ssd_streaming_cold;
    bool inspect_only;
    bool first_token_test;
    bool metal_graph_test;
} ds4_engine_options;

typedef void (*ds4_token_emit_fn)(void *ud, int token);
typedef void (*ds4_generation_done_fn)(void *ud);

/* Optional exact evidence for Qwen's canonical greedy generation loop.  The
 * caller owns both buffers and must size them before generation: token_ids for
 * n_predict entries and final_logits for ds4_engine_vocab_size().  Output
 * counts and frontier fields are committed only after the live session is
 * proven to contain every visible token. */
typedef struct {
    int   *token_ids;
    int    token_capacity;
    float *final_logits;
    int    final_logits_capacity;

    int prompt_tokens;
    int token_count;
    int final_argmax_id;
    int final_logits_count;
    int session_position;
} ds4_qwen_generation_evidence;

typedef struct {
    uint64_t total_bytes;
    uint64_t raw_bytes;
    uint64_t compressed_bytes;
    uint64_t scratch_bytes;
    uint32_t prefill_cap;
    uint32_t raw_cap;
    uint32_t comp_cap;
} ds4_context_memory;

typedef struct {
    uint8_t *ptr;
    uint64_t len;
    uint64_t cap;
} ds4_session_snapshot;

typedef struct {
    char *path;
    uint64_t bytes;
} ds4_session_payload_file;

int ds4_engine_open(ds4_engine **out, const ds4_engine_options *opt);
void ds4_engine_close(ds4_engine *e);
void ds4_engine_summary(ds4_engine *e);
/* Physical output width, including any model-defined padding rows. */
int ds4_engine_vocab_size(ds4_engine *e);
/* Token ids in [0, effective_vocab_size) are valid inputs and sampling
 * candidates.  Raw-logits callers must use this bound with ds4_sample_logits. */
int ds4_engine_effective_vocab_size(ds4_engine *e);
int ds4_engine_set_power(ds4_engine *e, int power_percent);
const char *ds4_engine_model_name(ds4_engine *e);
ds4_chat_format ds4_engine_chat_format(const ds4_engine *e);
bool ds4_engine_prompt_is_rendered_chat(
        const ds4_engine *e, const char *prompt);
/* Render the text-only subset of the pinned Qwen3.6 template while keeping
 * client data separate from trusted control tokens. On failure, out and
 * *rendered_out are unchanged. On success, the old malloc-owned *rendered_out
 * is freed and replaced. */
bool ds4_render_qwen36_chat_checked(
        ds4_engine *e,
        const ds4_chat_request *request,
        ds4_tokens *out,
        char **rendered_out,
        char *err,
        size_t errlen);
/* Build the canonical visible prefix used to key a later turn against a live
 * Qwen checkpoint that may contain hidden reasoning. The text and tokens must
 * be the exact generation prompt returned by the renderer. Failure leaves both
 * outputs unchanged; success replaces tokens_out and frees/replaces the
 * previous malloc-owned *out value. */
bool ds4_qwen36_visible_checkpoint_checked(
        ds4_engine *e,
        const char *generation_prompt,
        const ds4_tokens *generation_tokens,
        ds4_think_mode think_mode,
        const char *assistant_content,
        ds4_tokens *tokens_out,
        char **out);
int ds4_engine_layer_count(ds4_engine *e);
uint32_t ds4_engine_layer_compress_ratio(ds4_engine *e, uint32_t layer);
uint64_t ds4_engine_hidden_f32_values(ds4_engine *e);
/* Stable id for cache compatibility.  0 is the original Flash shape, so old
 * KV files with the previously-zero reserved byte remain Flash-compatible;
 * Pro and later shapes must use nonzero ids. */
int ds4_engine_model_id(ds4_engine *e);
bool ds4_engine_is_glm_dsa(ds4_engine *e);
const char *ds4_backend_name(ds4_backend backend);
typedef enum {
    DS4_EXECUTABLE_ROLE_CLI,
    DS4_EXECUTABLE_ROLE_SERVER,
    DS4_EXECUTABLE_ROLE_AGENT,
    DS4_EXECUTABLE_ROLE_BENCH,
    DS4_EXECUTABLE_ROLE_EVAL,
} ds4_executable_role;
const char *ds4_build_backend(void);
const char *ds4_build_arch(void);
const char *ds4_build_git_sha(void);
void ds4_build_info_print(FILE *fp, const char *argv0);
bool ds4_build_info_requested(int argc, char **argv);
bool ds4_capabilities_requested(int argc, char **argv);
void ds4_capabilities_print(FILE *fp, ds4_executable_role role,
                            const char *argv0);
bool ds4_think_mode_enabled(ds4_think_mode mode);
const char *ds4_think_mode_name(ds4_think_mode mode);
const char *ds4_think_max_prefix(void);
const char *ds4_glm_reasoning_effort_text(ds4_think_mode mode);
uint32_t ds4_think_max_min_context(void);
ds4_think_mode ds4_think_mode_for_context(ds4_think_mode mode, int ctx_size);
/* Uses the active DeepSeek shape selected by ds4_engine_open(); call after
 * opening the GGUF so Flash/Pro dimensions are known.  Model-aware callers
 * should prefer ds4_engine_context_memory_estimate_with_prefill(). */
ds4_context_memory ds4_context_memory_estimate(ds4_backend backend, int ctx_size);
ds4_context_memory ds4_context_memory_estimate_with_prefill(
        ds4_backend backend,
        int ctx_size,
        uint32_t prefill_chunk);
/* Estimate the buffers owned by a session for the engine's actual model
 * family.  Returns false for an invalid context or output pointer. */
bool ds4_engine_context_memory_estimate_with_prefill(
        const ds4_engine *engine,
        int ctx_size,
        uint32_t prefill_chunk,
        ds4_context_memory *out);
bool ds4_log_is_tty(FILE *fp);
void ds4_log(FILE *fp, ds4_log_type type, const char *fmt, ...);
int ds4_engine_generate_argmax(ds4_engine *e, const ds4_tokens *prompt,
                               int n_predict, int ctx_size,
                               ds4_token_emit_fn emit,
                               ds4_generation_done_fn done,
                               void *emit_ud,
                               ds4_session_progress_fn progress,
                               void *progress_ud);
/* Evidence is supported only by the Qwen canonical argmax path.  Passing NULL
 * is behaviorally identical to ds4_engine_generate_argmax(). */
int ds4_engine_generate_argmax_with_evidence(
        ds4_engine *e, const ds4_tokens *prompt,
        int n_predict, int ctx_size,
        ds4_token_emit_fn emit,
        ds4_generation_done_fn done,
        void *emit_ud,
        ds4_session_progress_fn progress,
        void *progress_ud,
        ds4_qwen_generation_evidence *evidence);
/* Publish a complete in-memory snapshot as
 * ds4.qwen.generation-evidence/1 using a same-directory temporary file and an
 * atomic rename.  Returns zero on success. */
int ds4_qwen_generation_evidence_write_json_atomic(
        const char *path,
        const ds4_qwen_generation_evidence *evidence,
        char *err,
        size_t errlen);
int ds4_engine_collect_imatrix(ds4_engine *e,
                               const char *dataset_path,
                               const char *output_path,
                               int ctx_size,
                               int max_prompts,
                               int max_tokens);
void ds4_engine_dump_tokens(ds4_engine *e, const ds4_tokens *tokens);
int ds4_dump_text_tokenization(const char *model_path, const char *text, FILE *fp);
int ds4_engine_head_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_first_token_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_metal_graph_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_metal_graph_full_test(ds4_engine *e, const ds4_tokens *prompt);
int ds4_engine_metal_graph_prompt_test(ds4_engine *e, const ds4_tokens *prompt, int ctx_size);

void ds4_tokens_push(ds4_tokens *tv, int token);
void ds4_tokens_free(ds4_tokens *tv);
void ds4_tokens_copy(ds4_tokens *dst, const ds4_tokens *src);
bool ds4_tokens_starts_with(const ds4_tokens *tokens, const ds4_tokens *prefix);

/* Checked tokenization APIs are transactional: false leaves the destination
 * token vector exactly unchanged.  The void forms remain compatibility
 * wrappers and report failures to stderr. */
bool ds4_tokenize_text_checked(
        ds4_engine *e, const char *text, ds4_tokens *out);
void ds4_tokenize_text(ds4_engine *e, const char *text, ds4_tokens *out);
bool ds4_tokenize_rendered_chat_checked(
        ds4_engine *e, const char *text, ds4_tokens *out);
void ds4_tokenize_rendered_chat(ds4_engine *e, const char *text, ds4_tokens *out);
void ds4_chat_begin(ds4_engine *e, ds4_tokens *tokens);
bool ds4_encode_chat_prompt_checked(
        ds4_engine *e,
        const char *system,
        const char *prompt,
        ds4_think_mode think_mode,
        ds4_tokens *out);
void ds4_chat_append_max_effort_prefix(ds4_engine *e, ds4_tokens *tokens);
bool ds4_chat_append_message_checked(
        ds4_engine *e,
        ds4_tokens *tokens,
        const char *role,
        const char *content);
void ds4_chat_append_message(ds4_engine *e, ds4_tokens *tokens, const char *role, const char *content);
bool ds4_chat_append_assistant_prefix_checked(
        ds4_engine *e,
        ds4_tokens *tokens,
        ds4_think_mode think_mode);
void ds4_chat_append_assistant_prefix(ds4_engine *e, ds4_tokens *tokens, ds4_think_mode think_mode);

char *ds4_token_text(ds4_engine *e, int token, size_t *len);
int ds4_token_eos(ds4_engine *e);
bool ds4_token_is_stop(ds4_engine *e, int token);
bool ds4_token_is_thinking_control(ds4_engine *e, int token);
bool ds4_token_is_stop_for_think_mode(ds4_engine *e,
                                      int token,
                                      ds4_think_mode mode);
int ds4_token_user(ds4_engine *e);
int ds4_token_assistant(ds4_engine *e);

int ds4_session_create(ds4_session **out, ds4_engine *e, int ctx_size);
void ds4_session_free(ds4_session *s);
int ds4_session_power(ds4_session *s);
int ds4_session_set_power(ds4_session *s, int power_percent);
void ds4_session_set_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud);
/* UI-only progress. It may report fine-grained progress inside a prefill chunk;
 * callers must not treat it as a durable KV checkpoint boundary. */
void ds4_session_set_display_progress(ds4_session *s, ds4_session_progress_fn fn, void *ud);
/* Optional cooperative cancellation.  ds4_session_sync() checks it only at
 * safe boundaries where the live checkpoint is either unchanged or represents a
 * valid token prefix, and returns DS4_SESSION_SYNC_INTERRUPTED when it stops. */
void ds4_session_set_cancel(ds4_session *s, ds4_session_cancel_fn fn, void *ud);
void ds4_session_report_progress(ds4_session *s, const char *event, int current, int total);
/* On-edge adaptive imatrix: accumulate aggregate routed-MoE second-moment statistics
 * from live prefills, so a locally-run quantized model can self-calibrate to its real
 * workload. No prompt text is ever stored — only per-expert importance vectors. */
int  ds4_session_imatrix_enable(ds4_session *s);
int  ds4_session_imatrix_save(ds4_session *s, const char *path);
uint64_t ds4_session_imatrix_observed_tokens(const ds4_session *s);
void ds4_session_imatrix_disable(ds4_session *s);
typedef enum {
    DS4_SESSION_REWRITE_ERROR = -1,
    DS4_SESSION_REWRITE_OK = 0,
    /* The live backend state cannot be rewritten safely in place.  The caller should
     * restore an older checkpoint if it has one, then sync to the prompt. */
    DS4_SESSION_REWRITE_REBUILD_NEEDED = 1,
} ds4_session_rewrite_result;

/* Synchronize the live session to a full prompt token prefix.  If the current
 * checkpoint is a prefix, only the suffix is evaluated; otherwise the backend
 * state is refilled from scratch. */
int ds4_session_sync(ds4_session *s, const ds4_tokens *prompt, char *err, size_t errlen);
bool ds4_session_rewrite_requires_rebuild(int live_len, int canonical_len, int common);
ds4_session_rewrite_result ds4_session_rewrite_from_common(
        ds4_session *s, const ds4_tokens *prompt, int common,
        char *err, size_t errlen);
int ds4_session_common_prefix(ds4_session *s, const ds4_tokens *prompt);
int ds4_session_argmax(ds4_session *s);
int ds4_session_argmax_excluding(ds4_session *s, int excluded_id);
int ds4_sample_logits(const float *logits, int n_vocab, float temperature,
                      int top_k, float top_p, float min_p, uint64_t *rng);
int ds4_session_sample(ds4_session *s, float temperature, int top_k, float top_p, float min_p, uint64_t *rng);
int ds4_session_top_logprobs(ds4_session *s, ds4_token_score *out, int k);
int ds4_session_token_logprob(ds4_session *s, int token, ds4_token_score *out);
int ds4_session_copy_logits(ds4_session *s, float *out, int cap);
int ds4_session_set_logits(ds4_session *s, const float *logits, int n);
int ds4_session_eval(ds4_session *s, int token, char *err, size_t errlen);
int ds4_session_eval_speculative_argmax(ds4_session *s, int first_token,
                                        int max_tokens, int eos_token,
                                        int *accepted, int accepted_cap,
                                        char *err, size_t errlen);
void ds4_session_invalidate(ds4_session *s);
void ds4_session_rewind(ds4_session *s, int pos);
int ds4_session_pos(ds4_session *s);
int ds4_session_ctx(ds4_session *s);
int ds4_session_prefill_cap(ds4_session *s);
int ds4_engine_routed_quant_bits(ds4_engine *e);
bool ds4_engine_has_output_head(ds4_engine *e);
bool ds4_engine_has_mtp(ds4_engine *e);
int ds4_engine_mtp_draft_tokens(ds4_engine *e);
/* Maximum exact greedy speculative span exposed by the active model/runtime.
 * This intentionally differs from ds4_engine_mtp_draft_tokens(): Qwen can use
 * prompt lookup without having an MTP model, while callers need one common
 * capability gate for the generation loop. */
int ds4_engine_speculative_draft_tokens(ds4_engine *e);
bool ds4_engine_is_qwen35(ds4_engine *e);
const ds4_tokens *ds4_session_tokens(ds4_session *s);

/* Disk KV payload helpers.  HTTP/agent code owns the outer file header and
 * persistence policy; the engine owns the DS4-specific serialized graph state. */
#define DS4_SESSION_PAYLOAD_MAGIC UINT32_C(0x34565344) /* "DSV4" */
#define DS4_SESSION_PAYLOAD_VERSION UINT32_C(2)
#define DS4_SESSION_PAYLOAD_U32_FIELDS 13u
uint64_t ds4_session_payload_bytes(ds4_session *s);
int ds4_session_stage_payload(ds4_session *s, ds4_session_payload_file *out,
                              char *err, size_t errlen);
int ds4_session_write_staged_payload(const ds4_session_payload_file *payload,
                                     FILE *fp, char *err, size_t errlen);
void ds4_session_payload_file_free(ds4_session_payload_file *payload);
int ds4_session_save_payload(ds4_session *s, FILE *fp, char *err, size_t errlen);
int ds4_session_load_payload(ds4_session *s, FILE *fp, uint64_t payload_bytes, char *err, size_t errlen);
int ds4_session_save_snapshot(ds4_session *s, ds4_session_snapshot *snap, char *err, size_t errlen);
int ds4_session_load_snapshot(ds4_session *s, const ds4_session_snapshot *snap, char *err, size_t errlen);
void ds4_session_snapshot_free(ds4_session_snapshot *snap);

#endif
