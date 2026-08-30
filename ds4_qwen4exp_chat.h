#ifndef DS4_QWEN4EXP_CHAT_H
#define DS4_QWEN4EXP_CHAT_H

#include <stdbool.h>
#include <stddef.h>

/* Standalone C99 renderer for the pinned Qwen4Exp text template.  It does not
 * tokenize, load a model, or make the family runnable.  The segment stream is
 * the security boundary: caller-owned content remains DATA even when its bytes
 * spell a tokenizer control token, while only template-authored atoms are
 * TRUSTED_CONTROL. */

typedef enum {
    DS4_QWEN4EXP_CHAT_SEGMENT_DATA = 0,
    DS4_QWEN4EXP_CHAT_SEGMENT_TRUSTED_CONTROL = 1,
} ds4_qwen4exp_chat_segment_kind;

typedef struct {
    ds4_qwen4exp_chat_segment_kind kind;
    size_t offset;
    size_t length;
} ds4_qwen4exp_chat_segment;

typedef enum {
    DS4_QWEN4EXP_CHAT_ROLE_SYSTEM = 0,
    DS4_QWEN4EXP_CHAT_ROLE_USER = 1,
    DS4_QWEN4EXP_CHAT_ROLE_ASSISTANT = 2,
    DS4_QWEN4EXP_CHAT_ROLE_TOOL = 3,
} ds4_qwen4exp_chat_role;

typedef enum {
    DS4_QWEN4EXP_CHAT_PART_TEXT = 0,
    DS4_QWEN4EXP_CHAT_PART_IMAGE = 1,
    DS4_QWEN4EXP_CHAT_PART_IMAGE_URL = 2,
    DS4_QWEN4EXP_CHAT_PART_VIDEO = 3,
} ds4_qwen4exp_chat_part_kind;

typedef struct {
    ds4_qwen4exp_chat_part_kind kind;
    const char *text;
} ds4_qwen4exp_chat_part;

typedef enum {
    DS4_QWEN4EXP_CHAT_CONTENT_NONE = 0,
    DS4_QWEN4EXP_CHAT_CONTENT_TEXT = 1,
    DS4_QWEN4EXP_CHAT_CONTENT_PARTS = 2,
} ds4_qwen4exp_chat_content_kind;

typedef struct {
    ds4_qwen4exp_chat_content_kind kind;
    const char *text;
    const ds4_qwen4exp_chat_part *parts;
    size_t part_count;
} ds4_qwen4exp_chat_content;

/* Values are already serialized exactly as the template would print them:
 * string arguments are their raw string contents; all other JSON types use
 * the pinned Transformers insertion-ordered, UTF-8 JSON representation. */
typedef struct {
    const char *name;
    const char *value;
} ds4_qwen4exp_chat_tool_argument;

typedef struct {
    const char *name;
    const ds4_qwen4exp_chat_tool_argument *arguments;
    size_t argument_count;
} ds4_qwen4exp_chat_tool_call;

typedef struct {
    ds4_qwen4exp_chat_role role;
    ds4_qwen4exp_chat_content content;
    const char *reasoning_content;
    const ds4_qwen4exp_chat_tool_call *tool_calls;
    size_t tool_call_count;
} ds4_qwen4exp_chat_message;

typedef enum {
    DS4_QWEN4EXP_CHAT_EFFORT_DEFAULT = 0,
    DS4_QWEN4EXP_CHAT_EFFORT_XHIGH = 1,
    DS4_QWEN4EXP_CHAT_EFFORT_MEDIUM = 2,
    DS4_QWEN4EXP_CHAT_EFFORT_LOW = 3,
} ds4_qwen4exp_chat_effort;

typedef enum {
    DS4_QWEN4EXP_CHAT_TRISTATE_DEFAULT = 0,
    DS4_QWEN4EXP_CHAT_TRISTATE_FALSE = 1,
    DS4_QWEN4EXP_CHAT_TRISTATE_TRUE = 2,
} ds4_qwen4exp_chat_tristate;

typedef struct {
    ds4_qwen4exp_chat_effort reasoning_effort;
    bool enable_thinking;
    ds4_qwen4exp_chat_tristate preserve_thinking;
    bool add_generation_prompt;
    bool add_vision_id;
    /* Each entry is the exact JSON object text supplied to the template.
     * Mapping order and non-ASCII bytes are significant. */
    const char *const *tools_json;
    size_t tool_count;
} ds4_qwen4exp_chat_options;

typedef struct {
    const ds4_qwen4exp_chat_message *messages;
    size_t message_count;
    ds4_qwen4exp_chat_options options;
} ds4_qwen4exp_chat_request;

typedef struct {
    char *rendered;
    size_t rendered_length;
    ds4_qwen4exp_chat_segment *segments;
    size_t segment_count;
} ds4_qwen4exp_chat_output;

typedef enum {
    DS4_QWEN4EXP_CHAT_ERROR_NONE = 0,
    DS4_QWEN4EXP_CHAT_ERROR_INVALID_ARGUMENT,
    DS4_QWEN4EXP_CHAT_ERROR_ALLOCATION,
    DS4_QWEN4EXP_CHAT_ERROR_EMPTY_CONVERSATION,
    DS4_QWEN4EXP_CHAT_ERROR_NO_USER_QUERY,
    DS4_QWEN4EXP_CHAT_ERROR_SYSTEM_NOT_FIRST,
    DS4_QWEN4EXP_CHAT_ERROR_UNEXPECTED_ROLE,
    DS4_QWEN4EXP_CHAT_ERROR_INVALID_REASONING_EFFORT,
    DS4_QWEN4EXP_CHAT_ERROR_STRUCTURED_MEDIA,
} ds4_qwen4exp_chat_error_code;

typedef struct {
    ds4_qwen4exp_chat_error_code code;
    const char *message;
} ds4_qwen4exp_chat_error;

void ds4_qwen4exp_chat_options_init(ds4_qwen4exp_chat_options *options);
void ds4_qwen4exp_chat_output_init(ds4_qwen4exp_chat_output *output);
void ds4_qwen4exp_chat_output_reset(ds4_qwen4exp_chat_output *output);

/* Failure leaves ``output`` byte-for-byte unchanged.  Success releases its
 * previous initialized contents and atomically replaces them.  ``error`` is
 * optional and refers only to static strings. */
bool ds4_qwen4exp_chat_render(
        const ds4_qwen4exp_chat_request *request,
        ds4_qwen4exp_chat_output *output,
        ds4_qwen4exp_chat_error *error);

#endif
