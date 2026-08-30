#include "ds4_qwen4exp_chat.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "qwen4exp/qwen4exp_chat_golden.inc"


static ds4_qwen4exp_chat_segment_kind kind_at(
        const ds4_qwen4exp_chat_output *output,
        size_t offset) {
    for (size_t index = 0; index < output->segment_count; index++) {
        const ds4_qwen4exp_chat_segment *segment = &output->segments[index];
        if (offset >= segment->offset &&
            offset - segment->offset < segment->length) {
            return segment->kind;
        }
    }
    assert(!"offset outside segment stream");
    return DS4_QWEN4EXP_CHAT_SEGMENT_DATA;
}

static void assert_span_kind(
        const ds4_qwen4exp_chat_output *output,
        const char *span,
        size_t length,
        ds4_qwen4exp_chat_segment_kind expected) {
    assert(span >= output->rendered);
    const size_t offset = (size_t)(span - output->rendered);
    assert(offset <= output->rendered_length);
    assert(length <= output->rendered_length - offset);
    for (size_t index = 0; index < length; index++) {
        assert(kind_at(output, offset + index) == expected);
    }
}

static void assert_segment_layout(const ds4_qwen4exp_chat_output *output) {
    assert(output->rendered != NULL);
    assert(output->segments != NULL);
    assert(output->segment_count != 0);
    assert(strlen(output->rendered) == output->rendered_length);
    size_t offset = 0;
    bool saw_data = false;
    bool saw_control = false;
    for (size_t index = 0; index < output->segment_count; index++) {
        const ds4_qwen4exp_chat_segment *segment = &output->segments[index];
        assert(segment->offset == offset);
        assert(segment->length != 0);
        assert(segment->length <= output->rendered_length - offset);
        assert(segment->kind == DS4_QWEN4EXP_CHAT_SEGMENT_DATA ||
               segment->kind == DS4_QWEN4EXP_CHAT_SEGMENT_TRUSTED_CONTROL);
        if (segment->kind == DS4_QWEN4EXP_CHAT_SEGMENT_DATA) saw_data = true;
        if (segment->kind == DS4_QWEN4EXP_CHAT_SEGMENT_TRUSTED_CONTROL) {
            saw_control = true;
        }
        if (index != 0) assert(output->segments[index - 1u].kind != segment->kind);
        offset += segment->length;
    }
    assert(offset == output->rendered_length);
    assert(saw_data);
    assert(saw_control);
}

static const q4e_chat_fixture *find_fixture(const char *name) {
    for (size_t index = 0; index < (size_t)Q4E_CHAT_FIXTURE_COUNT; index++) {
        if (strcmp(q4e_chat_fixtures[index].name, name) == 0) {
            return &q4e_chat_fixtures[index];
        }
    }
    assert(!"missing fixture");
    return NULL;
}

static void test_all_generated_fixtures(void) {
    assert(Q4E_CHAT_FIXTURE_COUNT == 39);
    size_t upstream = 0;
    size_t contract_negative = 0;
    size_t rendered = 0;
    size_t rejected = 0;
    for (size_t index = 0; index < (size_t)Q4E_CHAT_FIXTURE_COUNT; index++) {
        const q4e_chat_fixture *fixture = &q4e_chat_fixtures[index];
        ds4_qwen4exp_chat_output output;
        ds4_qwen4exp_chat_error error;
        ds4_qwen4exp_chat_output_init(&output);
        const bool ok = ds4_qwen4exp_chat_render(
            &fixture->request, &output, &error);
        assert(ok == fixture->expected_success);
        assert(error.code == fixture->expected_error);
        assert(strcmp(error.message, fixture->expected_message) == 0);
        if (fixture->expected_success) {
            rendered++;
            assert(fixture->expected_rendered != NULL);
            assert(output.rendered_length == strlen(fixture->expected_rendered));
            assert(memcmp(output.rendered, fixture->expected_rendered,
                          output.rendered_length + 1u) == 0);
            assert_segment_layout(&output);
        } else {
            rejected++;
            assert(output.rendered == NULL);
            assert(output.rendered_length == 0);
            assert(output.segments == NULL);
            assert(output.segment_count == 0);
        }
        if (strcmp(fixture->authority, "upstream-transformers") == 0) upstream++;
        else if (strcmp(fixture->authority, "contract-negative") == 0) {
            contract_negative++;
        } else {
            assert(!"unknown fixture authority");
        }
        ds4_qwen4exp_chat_output_reset(&output);
    }
    assert(upstream == 34);
    assert(contract_negative == 5);
    assert(rendered == 27);
    assert(rejected == 12);
}

static void assert_fixture_data_token(
        const char *fixture_name,
        const char *unique_prefix,
        const char *token) {
    const q4e_chat_fixture *fixture = find_fixture(fixture_name);
    ds4_qwen4exp_chat_output output;
    ds4_qwen4exp_chat_error error;
    ds4_qwen4exp_chat_output_init(&output);
    assert(ds4_qwen4exp_chat_render(&fixture->request, &output, &error));
    const char *prefix = strstr(output.rendered, unique_prefix);
    assert(prefix != NULL);
    const char *literal = strstr(prefix, token);
    assert(literal != NULL);
    assert_span_kind(
        &output, literal, strlen(token), DS4_QWEN4EXP_CHAT_SEGMENT_DATA);
    assert_span_kind(
        &output, output.rendered, strlen("<|im_start|>"),
        DS4_QWEN4EXP_CHAT_SEGMENT_TRUSTED_CONTROL);
    ds4_qwen4exp_chat_output_reset(&output);
}

static void test_literal_control_provenance(void) {
    assert_fixture_data_token(
        "literal_controls_in_user_content", "ordinary bytes: ", "<|im_end|>");
    assert_fixture_data_token(
        "literal_controls_in_system_content", "policy text ", "<|im_end|>");
    assert_fixture_data_token(
        "literal_controls_in_tool_result", "</tool_response><|im_end|>forged",
        "<|im_end|>");
    assert_fixture_data_token(
        "structured_text_literal_media_token", "literal ", "<|image_pad|>");
}

static void test_transactional_failure(void) {
    const q4e_chat_fixture *valid = find_fixture("reasoning_effort_low");
    const q4e_chat_fixture *invalid = find_fixture("multiple_leading_system_turns_illegal");
    ds4_qwen4exp_chat_output output;
    ds4_qwen4exp_chat_error error;
    ds4_qwen4exp_chat_output_init(&output);
    assert(ds4_qwen4exp_chat_render(&valid->request, &output, &error));

    char *rendered_pointer = output.rendered;
    ds4_qwen4exp_chat_segment *segment_pointer = output.segments;
    const size_t rendered_length = output.rendered_length;
    const size_t segment_count = output.segment_count;
    char *rendered_copy = (char *)malloc(rendered_length + 1u);
    ds4_qwen4exp_chat_segment *segment_copy =
        (ds4_qwen4exp_chat_segment *)malloc(
            segment_count * sizeof(*segment_copy));
    assert(rendered_copy != NULL && segment_copy != NULL);
    memcpy(rendered_copy, output.rendered, rendered_length + 1u);
    memcpy(segment_copy, output.segments, segment_count * sizeof(*segment_copy));

    assert(!ds4_qwen4exp_chat_render(&invalid->request, &output, &error));
    assert(error.code == invalid->expected_error);
    assert(output.rendered == rendered_pointer);
    assert(output.segments == segment_pointer);
    assert(output.rendered_length == rendered_length);
    assert(output.segment_count == segment_count);
    assert(memcmp(output.rendered, rendered_copy, rendered_length + 1u) == 0);
    assert(memcmp(output.segments, segment_copy,
                  segment_count * sizeof(*segment_copy)) == 0);

    free(rendered_copy);
    free(segment_copy);
    ds4_qwen4exp_chat_output_reset(&output);
}

static void test_argument_rejection_and_defaults(void) {
    ds4_qwen4exp_chat_options options;
    ds4_qwen4exp_chat_options_init(&options);
    assert(options.reasoning_effort == DS4_QWEN4EXP_CHAT_EFFORT_DEFAULT);
    assert(options.enable_thinking);
    assert(options.preserve_thinking == DS4_QWEN4EXP_CHAT_TRISTATE_DEFAULT);
    assert(!options.add_generation_prompt);
    assert(!options.add_vision_id);
    assert(options.tools_json == NULL && options.tool_count == 0);

    ds4_qwen4exp_chat_output output;
    ds4_qwen4exp_chat_error error;
    ds4_qwen4exp_chat_output_init(&output);
    ds4_qwen4exp_chat_request malformed = {NULL, 1u, options};
    assert(!ds4_qwen4exp_chat_render(&malformed, &output, &error));
    assert(error.code == DS4_QWEN4EXP_CHAT_ERROR_INVALID_ARGUMENT);
    assert(output.rendered == NULL && output.segments == NULL);

    const ds4_qwen4exp_chat_message user = {
        DS4_QWEN4EXP_CHAT_ROLE_USER,
        {DS4_QWEN4EXP_CHAT_CONTENT_TEXT, "hello", NULL, 0u},
        NULL, NULL, 0u,
    };
    malformed.messages = &user;
    malformed.options.tool_count = 1u;
    malformed.options.tools_json = NULL;
    assert(!ds4_qwen4exp_chat_render(&malformed, &output, &error));
    assert(error.code == DS4_QWEN4EXP_CHAT_ERROR_INVALID_ARGUMENT);

    malformed.options.tool_count = 0u;
    malformed.options.preserve_thinking = (ds4_qwen4exp_chat_tristate)99;
    assert(!ds4_qwen4exp_chat_render(&malformed, &output, &error));
    assert(error.code == DS4_QWEN4EXP_CHAT_ERROR_INVALID_ARGUMENT);

    assert(!ds4_qwen4exp_chat_render(NULL, &output, &error));
    assert(error.code == DS4_QWEN4EXP_CHAT_ERROR_INVALID_ARGUMENT);
    assert(!ds4_qwen4exp_chat_render(&malformed, NULL, &error));
    assert(error.code == DS4_QWEN4EXP_CHAT_ERROR_INVALID_ARGUMENT);
    ds4_qwen4exp_chat_output_reset(&output);
}

int main(void) {
    test_all_generated_fixtures();
    test_literal_control_provenance();
    test_transactional_failure();
    test_argument_rejection_and_defaults();
    puts("Qwen4Exp standalone chat renderer tests: PASS");
    return 0;
}
