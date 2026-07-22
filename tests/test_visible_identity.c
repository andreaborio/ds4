#include "hebrus_identity.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

static void expect_string(const char *label,
                          const char *actual,
                          const char *expected) {
    if (strcmp(actual, expected) == 0) return;
    fprintf(stderr,
            "visible-identity: %s mismatch\n  expected: %s\n  actual:   %s\n",
            label,
            expected,
            actual);
    failures++;
}

static void expect_agent_prompt(const char *argv0, const char *expected) {
    char buf[64];
    hebrus_agent_format_prompt_for(argv0, buf, sizeof(buf));
    expect_string(argv0, buf, expected);
}

static void expect_agent_banner(const char *argv0,
                                bool tty,
                                const char *expected) {
    char buf[192];
    hebrus_agent_format_welcome_banner_for(argv0, tty, "100K", buf, sizeof(buf));
    expect_string(argv0, buf, expected);
}

int main(void) {
    const char *canonical_system =
        hebrus_agent_system_prompt_for("/tmp/hebrus-agent");
    const char *legacy_system =
        hebrus_agent_system_prompt_for("/tmp/ds4-agent");
    const char *expected_system =
        "You are a helpful coding assistant running inside ds4-agent.";

    expect_string("canonical CLI prompt",
                  hebrus_cli_prompt_for("/tmp/hebrus"),
                  "hebrus> ");
    expect_string("legacy CLI prompt",
                  hebrus_cli_prompt_for("/tmp/ds4"),
                  "ds4> ");

    expect_agent_prompt("/tmp/hebrus-agent", "hebrus-agent> ");
    expect_agent_prompt("/tmp/ds4-agent", "ds4-agent> ");
    expect_agent_banner(
        "/tmp/hebrus-agent",
        true,
        "\x1b[1;97mHebrus\x1b[0m 🐋 Agent, context 100K tokens\n\n");
    expect_agent_banner(
        "/tmp/ds4-agent",
        true,
        "\x1b[1;97mDwarf\x1b[1;94mStar\x1b[0m 🐋 Agent, context 100K tokens\n\n");
    expect_agent_banner("/tmp/hebrus-agent",
                        false,
                        "Hebrus Agent, context 100K tokens\n\n");
    expect_agent_banner("/tmp/ds4-agent",
                        false,
                        "DwarfStar Agent, context 100K tokens\n\n");

    expect_string("canonical eval title",
                  hebrus_eval_command_for("/tmp/hebrus-eval"),
                  "hebrus-eval");
    expect_string("legacy eval title",
                  hebrus_eval_command_for("/tmp/ds4-eval"),
                  "ds4-eval");
    expect_string("canonical benchmark source",
                  hebrus_bench_command_for("/tmp/hebrus-bench"),
                  "hebrus-bench");
    expect_string("legacy benchmark source",
                  hebrus_bench_command_for("/tmp/ds4-bench"),
                  "ds4-bench");

    expect_string("canonical agent system prompt", canonical_system, expected_system);
    expect_string("legacy agent system prompt", legacy_system, expected_system);
    if (strlen(canonical_system) != strlen(legacy_system) ||
        memcmp(canonical_system, legacy_system, strlen(canonical_system)) != 0) {
        fputs("visible-identity: canonical and legacy system prompts differ\n", stderr);
        failures++;
    }

    if (failures) return EXIT_FAILURE;
    puts("visible-identity: PASS");
    return EXIT_SUCCESS;
}
