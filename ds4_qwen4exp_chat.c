#include "ds4_qwen4exp_chat.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>


typedef struct {
    char *bytes;
    size_t length;
    size_t capacity;
    ds4_qwen4exp_chat_segment *segments;
    size_t segment_count;
    size_t segment_capacity;
} q4e_chat_sink;

static const char q4e_xhigh_instruction[] =
    "Reasoning effort is set to xhigh. Please think carefully through the task, "
    "validate key assumptions, consider plausible alternatives, and prioritize "
    "correctness, consistency, and clarity in the final answer.";

static const char q4e_low_instruction[] =
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving "
    "directly to the conclusion without unnecessary elaboration.";

static const char q4e_error_invalid_argument[] = "Invalid Qwen4Exp chat argument.";
static const char q4e_error_allocation[] = "Qwen4Exp chat allocation failed.";
static const char q4e_error_empty[] =
    "Cannot apply chat template to an empty conversation. Provide at least one message.";
static const char q4e_error_no_query[] = "No user query found in messages.";
static const char q4e_error_system[] = "System message must be at the beginning.";
static const char q4e_error_role[] = "Unexpected message role.";
static const char q4e_error_effort[] =
    "Unexpected reasoning effort invalid. Supported types are xhigh (default), medium, and low.";
static const char q4e_error_media[] =
    "structured image/video content is excluded by the qwen4exp-base-v1 text-only contract";


static void q4e_set_error(
        ds4_qwen4exp_chat_error *error,
        ds4_qwen4exp_chat_error_code code,
        const char *message) {
    if (!error) return;
    error->code = code;
    error->message = message;
}

static bool q4e_add_size(size_t left, size_t right, size_t *result) {
    if (!result || left > SIZE_MAX - right) return false;
    *result = left + right;
    return true;
}

static bool q4e_mul_size(size_t left, size_t right, size_t *result) {
    if (!result || (left != 0 && right > SIZE_MAX / left)) return false;
    *result = left * right;
    return true;
}

static bool q4e_grow_capacity(size_t current, size_t required, size_t *grown) {
    size_t capacity = current ? current : 64u;
    while (capacity < required) {
        if (capacity > SIZE_MAX / 2u) {
            capacity = required;
            break;
        }
        capacity *= 2u;
    }
    *grown = capacity;
    return true;
}

static bool q4e_sink_reserve_bytes(q4e_chat_sink *sink, size_t additional) {
    size_t required;
    if (!q4e_add_size(sink->length, additional, &required) ||
        !q4e_add_size(required, 1u, &required)) {
        return false;
    }
    if (required <= sink->capacity) return true;
    size_t capacity;
    q4e_grow_capacity(sink->capacity, required, &capacity);
    char *replacement = (char *)realloc(sink->bytes, capacity);
    if (!replacement) return false;
    sink->bytes = replacement;
    sink->capacity = capacity;
    return true;
}

static bool q4e_sink_reserve_segments(q4e_chat_sink *sink, size_t additional) {
    size_t required;
    if (!q4e_add_size(sink->segment_count, additional, &required)) return false;
    if (required <= sink->segment_capacity) return true;
    size_t capacity;
    size_t bytes;
    q4e_grow_capacity(sink->segment_capacity, required, &capacity);
    if (!q4e_mul_size(capacity, sizeof(*sink->segments), &bytes)) return false;
    ds4_qwen4exp_chat_segment *replacement =
        (ds4_qwen4exp_chat_segment *)realloc(sink->segments, bytes);
    if (!replacement) return false;
    sink->segments = replacement;
    sink->segment_capacity = capacity;
    return true;
}

static bool q4e_sink_append(
        q4e_chat_sink *sink,
        ds4_qwen4exp_chat_segment_kind kind,
        const char *text,
        size_t length) {
    if (length == 0) return true;
    if (!text || !q4e_sink_reserve_bytes(sink, length)) return false;
    const bool merge = sink->segment_count != 0 &&
        sink->segments[sink->segment_count - 1u].kind == kind;
    if (!merge && !q4e_sink_reserve_segments(sink, 1u)) return false;

    const size_t offset = sink->length;
    memcpy(sink->bytes + sink->length, text, length);
    sink->length += length;
    sink->bytes[sink->length] = '\0';
    if (merge) {
        sink->segments[sink->segment_count - 1u].length += length;
    } else {
        ds4_qwen4exp_chat_segment *segment = &sink->segments[sink->segment_count++];
        segment->kind = kind;
        segment->offset = offset;
        segment->length = length;
    }
    return true;
}

static bool q4e_sink_data_n(q4e_chat_sink *sink, const char *text, size_t length) {
    return q4e_sink_append(sink, DS4_QWEN4EXP_CHAT_SEGMENT_DATA, text, length);
}

static bool q4e_sink_data(q4e_chat_sink *sink, const char *text) {
    return q4e_sink_data_n(sink, text, text ? strlen(text) : 0u);
}

static bool q4e_sink_control(q4e_chat_sink *sink, const char *text) {
    return q4e_sink_append(
        sink, DS4_QWEN4EXP_CHAT_SEGMENT_TRUSTED_CONTROL,
        text, text ? strlen(text) : 0u);
}

static void q4e_sink_reset(q4e_chat_sink *sink) {
    if (!sink) return;
    free(sink->bytes);
    free(sink->segments);
    memset(sink, 0, sizeof(*sink));
}

static bool q4e_utf8_next(
        const char *bytes,
        size_t length,
        size_t *offset,
        uint32_t *codepoint) {
    if (!bytes || !offset || !codepoint || *offset >= length) return false;
    const unsigned char lead = (unsigned char)bytes[*offset];
    size_t width;
    uint32_t value;
    if (lead < 0x80u) {
        width = 1u;
        value = lead;
    } else if (lead >= 0xc2u && lead <= 0xdfu) {
        width = 2u;
        value = lead & 0x1fu;
    } else if (lead >= 0xe0u && lead <= 0xefu) {
        width = 3u;
        value = lead & 0x0fu;
    } else if (lead >= 0xf0u && lead <= 0xf4u) {
        width = 4u;
        value = lead & 0x07u;
    } else {
        return false;
    }
    if (width > length - *offset) return false;
    for (size_t index = 1u; index < width; index++) {
        const unsigned char next = (unsigned char)bytes[*offset + index];
        if ((next & 0xc0u) != 0x80u) return false;
        value = (value << 6u) | (uint32_t)(next & 0x3fu);
    }
    if ((width == 3u && value < 0x800u) ||
        (width == 4u && value < 0x10000u) ||
        value > 0x10ffffu || (value >= 0xd800u && value <= 0xdfffu)) {
        return false;
    }
    *offset += width;
    *codepoint = value;
    return true;
}

static bool q4e_python_strip_space(uint32_t codepoint) {
    return (codepoint >= 0x0009u && codepoint <= 0x000du) ||
           (codepoint >= 0x001cu && codepoint <= 0x0020u) ||
           codepoint == 0x0085u || codepoint == 0x00a0u ||
           codepoint == 0x1680u ||
           (codepoint >= 0x2000u && codepoint <= 0x200au) ||
           codepoint == 0x2028u || codepoint == 0x2029u ||
           codepoint == 0x202fu || codepoint == 0x205fu ||
           codepoint == 0x3000u;
}

static bool q4e_copy_trimmed_bytes(
        const char *bytes,
        size_t length,
        char **trimmed_out) {
    if ((!bytes && length != 0u) || !trimmed_out) return false;
    size_t offset = 0u;
    size_t begin = length;
    size_t end = 0u;
    while (offset < length) {
        const size_t start = offset;
        uint32_t codepoint;
        if (!q4e_utf8_next(bytes, length, &offset, &codepoint)) return false;
        if (!q4e_python_strip_space(codepoint)) {
            if (begin == length) begin = start;
            end = offset;
        }
    }
    if (begin == length) end = length;
    const size_t trimmed_length = end - begin;
    if (trimmed_length == SIZE_MAX) return false;
    char *trimmed = (char *)malloc(trimmed_length + 1u);
    if (!trimmed) return false;
    if (trimmed_length) memcpy(trimmed, bytes + begin, trimmed_length);
    trimmed[trimmed_length] = '\0';
    *trimmed_out = trimmed;
    return true;
}

static bool q4e_content_has_media(const ds4_qwen4exp_chat_content *content) {
    if (!content || content->kind != DS4_QWEN4EXP_CHAT_CONTENT_PARTS) return false;
    if (content->part_count && !content->parts) return false;
    for (size_t index = 0; index < content->part_count; index++) {
        const ds4_qwen4exp_chat_part_kind kind = content->parts[index].kind;
        if (kind == DS4_QWEN4EXP_CHAT_PART_IMAGE ||
            kind == DS4_QWEN4EXP_CHAT_PART_IMAGE_URL ||
            kind == DS4_QWEN4EXP_CHAT_PART_VIDEO) {
            return true;
        }
    }
    return false;
}

static bool q4e_content_valid(const ds4_qwen4exp_chat_content *content) {
    if (!content) return false;
    if (content->kind == DS4_QWEN4EXP_CHAT_CONTENT_NONE) return true;
    if (content->kind == DS4_QWEN4EXP_CHAT_CONTENT_TEXT) return true;
    if (content->kind != DS4_QWEN4EXP_CHAT_CONTENT_PARTS ||
        (content->part_count && !content->parts)) {
        return false;
    }
    for (size_t index = 0; index < content->part_count; index++) {
        const ds4_qwen4exp_chat_part_kind kind = content->parts[index].kind;
        if (kind < DS4_QWEN4EXP_CHAT_PART_TEXT ||
            kind > DS4_QWEN4EXP_CHAT_PART_VIDEO) {
            return false;
        }
    }
    return true;
}

static bool q4e_content_trimmed(
        const ds4_qwen4exp_chat_content *content,
        char **trimmed_out) {
    if (content->kind == DS4_QWEN4EXP_CHAT_CONTENT_NONE) {
        return q4e_copy_trimmed_bytes("", 0u, trimmed_out);
    }
    if (content->kind == DS4_QWEN4EXP_CHAT_CONTENT_TEXT) {
        const char *text = content->text ? content->text : "";
        return q4e_copy_trimmed_bytes(text, strlen(text), trimmed_out);
    }

    size_t length = 0;
    for (size_t index = 0; index < content->part_count; index++) {
        const ds4_qwen4exp_chat_part *part = &content->parts[index];
        if (part->kind != DS4_QWEN4EXP_CHAT_PART_TEXT) return false;
        const char *text = part->text ? part->text : "";
        if (!q4e_add_size(length, strlen(text), &length)) return false;
    }
    if (length == SIZE_MAX) return false;
    char *joined = (char *)malloc(length + 1u);
    if (!joined) return false;
    size_t at = 0;
    for (size_t index = 0; index < content->part_count; index++) {
        const char *text = content->parts[index].text ? content->parts[index].text : "";
        const size_t part_length = strlen(text);
        memcpy(joined + at, text, part_length);
        at += part_length;
    }
    joined[length] = '\0';
    const bool ok = q4e_copy_trimmed_bytes(joined, length, trimmed_out);
    free(joined);
    return ok;
}

static bool q4e_is_tool_response(const char *content) {
    static const char open[] = "<tool_response>";
    static const char close[] = "</tool_response>";
    const size_t length = strlen(content);
    const size_t open_length = sizeof(open) - 1u;
    const size_t close_length = sizeof(close) - 1u;
    return length >= open_length + close_length &&
           memcmp(content, open, open_length) == 0 &&
           memcmp(content + length - close_length, close, close_length) == 0;
}

static bool q4e_render_tools_system(
        q4e_chat_sink *sink,
        const ds4_qwen4exp_chat_request *request,
        const char *instruction,
        const char *system_content) {
    if (!q4e_sink_control(sink, "<|im_start|>") ||
        !q4e_sink_data(sink, "system\n")) {
        return false;
    }
    if (instruction[0] &&
        (!q4e_sink_data(sink, instruction) || !q4e_sink_data(sink, "\n\n"))) {
        return false;
    }
    if (!q4e_sink_data(sink,
            "# Tools\n\nYou have access to the following functions:\n\n<tools>")) {
        return false;
    }
    for (size_t index = 0; index < request->options.tool_count; index++) {
        if (!q4e_sink_data(sink, "\n") ||
            !q4e_sink_data(sink, request->options.tools_json[index])) {
            return false;
        }
    }
    if (!q4e_sink_data(sink,
            "\n</tools>"
            "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:"
            "\n\n") ||
        !q4e_sink_control(sink, "<tool_call>") ||
        !q4e_sink_data(sink,
            "\n<function=example_function_name>"
            "\n<parameter=example_parameter_1>"
            "\nvalue_1"
            "\n</parameter>"
            "\n<parameter=example_parameter_2>"
            "\nThis is the value for the second parameter"
            "\nthat can span"
            "\nmultiple lines"
            "\n</parameter>"
            "\n</function>"
            "\n") ||
        !q4e_sink_control(sink, "</tool_call>") ||
        !q4e_sink_data(sink,
            "\n\n<IMPORTANT>"
            "\nReminder:"
            "\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within ") ||
        !q4e_sink_control(sink, "<tool_call>") ||
        !q4e_sink_control(sink, "</tool_call>") ||
        !q4e_sink_data(sink,
            " XML tags"
            "\n- Required parameters MUST be specified"
            "\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after"
            "\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls"
            "\n</IMPORTANT>")) {
        return false;
    }
    if (system_content[0] &&
        (!q4e_sink_data(sink, "\n\n") || !q4e_sink_data(sink, system_content))) {
        return false;
    }
    return q4e_sink_control(sink, "<|im_end|>") && q4e_sink_data(sink, "\n");
}

static bool q4e_render_plain_system(
        q4e_chat_sink *sink,
        const char *instruction,
        const char *system_content,
        bool has_system) {
    if (!has_system && !instruction[0]) return true;
    if (has_system && !system_content[0] && !instruction[0]) return true;
    if (!q4e_sink_control(sink, "<|im_start|>") ||
        !q4e_sink_data(sink, "system\n")) {
        return false;
    }
    if (instruction[0] && !q4e_sink_data(sink, instruction)) return false;
    if (instruction[0] && system_content[0] && !q4e_sink_data(sink, "\n\n")) {
        return false;
    }
    if (system_content[0] && !q4e_sink_data(sink, system_content)) return false;
    return q4e_sink_control(sink, "<|im_end|>") && q4e_sink_data(sink, "\n");
}

static bool q4e_render_tool_call(
        q4e_chat_sink *sink,
        const ds4_qwen4exp_chat_tool_call *call,
        bool content_before,
        bool first) {
    if ((!first || content_before) &&
        !q4e_sink_data(sink, first ? "\n\n" : "\n")) {
        return false;
    }
    if (!q4e_sink_control(sink, "<tool_call>") ||
        !q4e_sink_data(sink, "\n<function=") ||
        !q4e_sink_data(sink, call->name) ||
        !q4e_sink_data(sink, ">\n")) {
        return false;
    }
    for (size_t index = 0; index < call->argument_count; index++) {
        const ds4_qwen4exp_chat_tool_argument *argument = &call->arguments[index];
        if (!q4e_sink_data(sink, "<parameter=") ||
            !q4e_sink_data(sink, argument->name) ||
            !q4e_sink_data(sink, ">\n") ||
            !q4e_sink_data(sink, argument->value) ||
            !q4e_sink_data(sink, "\n</parameter>\n")) {
            return false;
        }
    }
    return q4e_sink_data(sink, "</function>\n") &&
           q4e_sink_control(sink, "</tool_call>");
}

static bool q4e_validate_tool_calls(const ds4_qwen4exp_chat_message *message) {
    if (message->tool_call_count && !message->tool_calls) return false;
    for (size_t call_index = 0; call_index < message->tool_call_count; call_index++) {
        const ds4_qwen4exp_chat_tool_call *call = &message->tool_calls[call_index];
        if (!call->name || (call->argument_count && !call->arguments)) return false;
        for (size_t argument_index = 0;
             argument_index < call->argument_count; argument_index++) {
            const ds4_qwen4exp_chat_tool_argument *argument =
                &call->arguments[argument_index];
            if (!argument->name || !argument->value) return false;
        }
    }
    return true;
}

static bool q4e_preserve_thinking(
        ds4_qwen4exp_chat_tristate configured,
        size_t message_index,
        size_t last_query) {
    return configured != DS4_QWEN4EXP_CHAT_TRISTATE_FALSE ||
           message_index > last_query;
}

static bool q4e_render_messages(
        q4e_chat_sink *sink,
        const ds4_qwen4exp_chat_request *request,
        size_t last_query) {
    for (size_t index = 0; index < request->message_count; index++) {
        const ds4_qwen4exp_chat_message *message = &request->messages[index];
        if (message->role == DS4_QWEN4EXP_CHAT_ROLE_SYSTEM) continue;

        char *content = NULL;
        if (!q4e_content_trimmed(&message->content, &content)) return false;
        bool ok = true;
        if (message->role == DS4_QWEN4EXP_CHAT_ROLE_USER) {
            ok = q4e_sink_control(sink, "<|im_start|>") &&
                 q4e_sink_data(sink, "user\n") &&
                 q4e_sink_data(sink, content) &&
                 q4e_sink_control(sink, "<|im_end|>") &&
                 q4e_sink_data(sink, "\n");
        } else if (message->role == DS4_QWEN4EXP_CHAT_ROLE_ASSISTANT) {
            const char *raw_reasoning = message->reasoning_content ?
                message->reasoning_content : "";
            char *reasoning = NULL;
            ok = q4e_copy_trimmed_bytes(
                raw_reasoning, strlen(raw_reasoning), &reasoning);
            if (ok) {
                ok = q4e_sink_control(sink, "<|im_start|>") &&
                     q4e_sink_data(sink, "assistant\n");
            }
            if (ok && q4e_preserve_thinking(
                    request->options.preserve_thinking, index, last_query)) {
                ok = q4e_sink_control(sink, "<think>") &&
                     q4e_sink_data(sink, "\n") &&
                     q4e_sink_data(sink, reasoning) &&
                     q4e_sink_data(sink, "\n") &&
                     q4e_sink_control(sink, "</think>") &&
                     q4e_sink_data(sink, "\n\n") &&
                     q4e_sink_data(sink, content);
            } else if (ok) {
                ok = q4e_sink_data(sink, content);
            }
            for (size_t call_index = 0;
                 ok && call_index < message->tool_call_count; call_index++) {
                ok = q4e_render_tool_call(
                    sink, &message->tool_calls[call_index], content[0] != '\0',
                    call_index == 0);
            }
            if (ok) {
                ok = q4e_sink_control(sink, "<|im_end|>") &&
                     q4e_sink_data(sink, "\n");
            }
            free(reasoning);
        } else {
            const bool starts_group = index == 0 ||
                request->messages[index - 1u].role != DS4_QWEN4EXP_CHAT_ROLE_TOOL;
            const bool ends_group = index + 1u == request->message_count ||
                request->messages[index + 1u].role != DS4_QWEN4EXP_CHAT_ROLE_TOOL;
            if (starts_group) {
                ok = q4e_sink_control(sink, "<|im_start|>") &&
                     q4e_sink_data(sink, "user");
            }
            if (ok) {
                ok = q4e_sink_data(sink, "\n") &&
                     q4e_sink_control(sink, "<tool_response>") &&
                     q4e_sink_data(sink, "\n") &&
                     q4e_sink_data(sink, content) &&
                     q4e_sink_data(sink, "\n") &&
                     q4e_sink_control(sink, "</tool_response>");
            }
            if (ok && ends_group) {
                ok = q4e_sink_control(sink, "<|im_end|>") &&
                     q4e_sink_data(sink, "\n");
            }
        }
        free(content);
        if (!ok) return false;
    }
    return true;
}


void ds4_qwen4exp_chat_options_init(ds4_qwen4exp_chat_options *options) {
    if (!options) return;
    memset(options, 0, sizeof(*options));
    options->reasoning_effort = DS4_QWEN4EXP_CHAT_EFFORT_DEFAULT;
    options->enable_thinking = true;
    options->preserve_thinking = DS4_QWEN4EXP_CHAT_TRISTATE_DEFAULT;
}

void ds4_qwen4exp_chat_output_init(ds4_qwen4exp_chat_output *output) {
    if (output) memset(output, 0, sizeof(*output));
}

void ds4_qwen4exp_chat_output_reset(ds4_qwen4exp_chat_output *output) {
    if (!output) return;
    free(output->rendered);
    free(output->segments);
    memset(output, 0, sizeof(*output));
}

bool ds4_qwen4exp_chat_render(
        const ds4_qwen4exp_chat_request *request,
        ds4_qwen4exp_chat_output *output,
        ds4_qwen4exp_chat_error *error) {
    q4e_set_error(error, DS4_QWEN4EXP_CHAT_ERROR_NONE, "");
    if (!request || !output ||
        (request->message_count && !request->messages) ||
        (request->options.tool_count && !request->options.tools_json) ||
        request->options.preserve_thinking < DS4_QWEN4EXP_CHAT_TRISTATE_DEFAULT ||
        request->options.preserve_thinking > DS4_QWEN4EXP_CHAT_TRISTATE_TRUE) {
        q4e_set_error(error, DS4_QWEN4EXP_CHAT_ERROR_INVALID_ARGUMENT,
                      q4e_error_invalid_argument);
        return false;
    }
    if (request->message_count == 0) {
        q4e_set_error(error, DS4_QWEN4EXP_CHAT_ERROR_EMPTY_CONVERSATION,
                      q4e_error_empty);
        return false;
    }
    for (size_t index = 0; index < request->options.tool_count; index++) {
        if (!request->options.tools_json[index]) {
            q4e_set_error(error, DS4_QWEN4EXP_CHAT_ERROR_INVALID_ARGUMENT,
                          q4e_error_invalid_argument);
            return false;
        }
    }
    for (size_t index = 0; index < request->message_count; index++) {
        const ds4_qwen4exp_chat_message *message = &request->messages[index];
        if (!q4e_content_valid(&message->content) ||
            !q4e_validate_tool_calls(message)) {
            q4e_set_error(error, DS4_QWEN4EXP_CHAT_ERROR_INVALID_ARGUMENT,
                          q4e_error_invalid_argument);
            return false;
        }
        if (q4e_content_has_media(&message->content)) {
            q4e_set_error(error, DS4_QWEN4EXP_CHAT_ERROR_STRUCTURED_MEDIA,
                          q4e_error_media);
            return false;
        }
    }

    const char *instruction = "";
    if (request->options.enable_thinking) {
        if (request->options.reasoning_effort == DS4_QWEN4EXP_CHAT_EFFORT_DEFAULT ||
            request->options.reasoning_effort == DS4_QWEN4EXP_CHAT_EFFORT_XHIGH) {
            instruction = q4e_xhigh_instruction;
        } else if (request->options.reasoning_effort == DS4_QWEN4EXP_CHAT_EFFORT_LOW) {
            instruction = q4e_low_instruction;
        } else if (request->options.reasoning_effort !=
                   DS4_QWEN4EXP_CHAT_EFFORT_MEDIUM) {
            q4e_set_error(error, DS4_QWEN4EXP_CHAT_ERROR_INVALID_REASONING_EFFORT,
                          q4e_error_effort);
            return false;
        }
    }

    size_t last_query = SIZE_MAX;
    for (size_t reverse = request->message_count; reverse != 0; reverse--) {
        const size_t index = reverse - 1u;
        const ds4_qwen4exp_chat_message *message = &request->messages[index];
        if (message->role != DS4_QWEN4EXP_CHAT_ROLE_USER) continue;
        char *content = NULL;
        if (!q4e_content_trimmed(&message->content, &content)) {
            q4e_set_error(error, DS4_QWEN4EXP_CHAT_ERROR_ALLOCATION,
                          q4e_error_allocation);
            return false;
        }
        const bool is_response = q4e_is_tool_response(content);
        free(content);
        if (!is_response) {
            last_query = index;
            break;
        }
    }
    if (last_query == SIZE_MAX) {
        q4e_set_error(error, DS4_QWEN4EXP_CHAT_ERROR_NO_USER_QUERY,
                      q4e_error_no_query);
        return false;
    }

    for (size_t index = 0; index < request->message_count; index++) {
        const ds4_qwen4exp_chat_role role = request->messages[index].role;
        if (role < DS4_QWEN4EXP_CHAT_ROLE_SYSTEM ||
            role > DS4_QWEN4EXP_CHAT_ROLE_TOOL) {
            q4e_set_error(error, DS4_QWEN4EXP_CHAT_ERROR_UNEXPECTED_ROLE,
                          q4e_error_role);
            return false;
        }
        if (role == DS4_QWEN4EXP_CHAT_ROLE_SYSTEM && index != 0) {
            q4e_set_error(error, DS4_QWEN4EXP_CHAT_ERROR_SYSTEM_NOT_FIRST,
                          q4e_error_system);
            return false;
        }
    }

    char *system_content = NULL;
    const bool has_system =
        request->messages[0].role == DS4_QWEN4EXP_CHAT_ROLE_SYSTEM;
    if (has_system) {
        if (!q4e_content_trimmed(&request->messages[0].content, &system_content)) {
            q4e_set_error(error, DS4_QWEN4EXP_CHAT_ERROR_ALLOCATION,
                          q4e_error_allocation);
            return false;
        }
    } else {
        system_content = (char *)malloc(1u);
        if (system_content) system_content[0] = '\0';
        if (!system_content) {
            q4e_set_error(error, DS4_QWEN4EXP_CHAT_ERROR_ALLOCATION,
                          q4e_error_allocation);
            return false;
        }
    }

    q4e_chat_sink sink;
    memset(&sink, 0, sizeof(sink));
    bool ok;
    if (request->options.tool_count) {
        ok = q4e_render_tools_system(
            &sink, request, instruction, system_content);
    } else {
        ok = q4e_render_plain_system(
            &sink, instruction, system_content, has_system);
    }
    free(system_content);
    if (ok) ok = q4e_render_messages(&sink, request, last_query);
    if (ok && request->options.add_generation_prompt) {
        ok = q4e_sink_control(&sink, "<|im_start|>") &&
             q4e_sink_data(&sink, "assistant\n") &&
             q4e_sink_control(&sink, "<think>");
        if (ok && request->options.enable_thinking) {
            ok = q4e_sink_data(&sink, "\n");
        } else if (ok) {
            ok = q4e_sink_data(&sink, "\n\n") &&
                 q4e_sink_control(&sink, "</think>") &&
                 q4e_sink_data(&sink, "\n\n");
        }
    }
    if (!ok) {
        q4e_sink_reset(&sink);
        q4e_set_error(error, DS4_QWEN4EXP_CHAT_ERROR_ALLOCATION,
                      q4e_error_allocation);
        return false;
    }

    ds4_qwen4exp_chat_output replacement;
    replacement.rendered = sink.bytes;
    replacement.rendered_length = sink.length;
    replacement.segments = sink.segments;
    replacement.segment_count = sink.segment_count;
    ds4_qwen4exp_chat_output_reset(output);
    *output = replacement;
    q4e_set_error(error, DS4_QWEN4EXP_CHAT_ERROR_NONE, "");
    return true;
}
