#ifndef HEBRUS_IDENTITY_H
#define HEBRUS_IDENTITY_H

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

static inline const char *hebrus_invocation_name(const char *argv0) {
    const char *name = argv0 ? strrchr(argv0, '/') : NULL;
    return name ? name + 1 : argv0;
}

static inline bool hebrus_is_canonical_invocation(const char *argv0) {
    const char *name = hebrus_invocation_name(argv0);
    return name &&
        (strcmp(name, "hebrus") == 0 || strcmp(name, "hebrus-server") == 0 ||
         strcmp(name, "hebrus-agent") == 0 || strcmp(name, "hebrus-bench") == 0 ||
         strcmp(name, "hebrus-eval") == 0);
}

static inline const char *hebrus_cli_prompt_for(const char *argv0) {
    return hebrus_is_canonical_invocation(argv0) ? "hebrus> " : "ds4> ";
}

static inline const char *hebrus_agent_command_for(const char *argv0) {
    return hebrus_is_canonical_invocation(argv0) ? "hebrus-agent" : "ds4-agent";
}

static inline const char *hebrus_agent_brand_for(const char *argv0) {
    return hebrus_is_canonical_invocation(argv0) ? "Hebrus" : "DwarfStar";
}

static inline const char *hebrus_agent_system_prompt_for(const char *argv0) {
    (void)argv0;
    return "You are a helpful coding assistant running inside ds4-agent.";
}

static inline void hebrus_agent_format_prompt_for(const char *argv0,
                                                  char *buf,
                                                  size_t len) {
    if (len == 0) return;
    snprintf(buf, len, "%s> ", hebrus_agent_command_for(argv0));
}

static inline void hebrus_agent_format_welcome_banner_for(const char *argv0,
                                                          bool tty,
                                                          const char *ctx,
                                                          char *buf,
                                                          size_t len) {
    if (len == 0) return;
    if (tty && hebrus_is_canonical_invocation(argv0)) {
        snprintf(buf, len,
                 "\x1b[1;97mHebrus\x1b[0m 🐋 Agent, context %s tokens\n\n",
                 ctx);
    } else if (tty) {
        snprintf(buf, len,
                 "\x1b[1;97mDwarf\x1b[1;94mStar\x1b[0m 🐋 Agent, context %s tokens\n\n",
                 ctx);
    } else {
        snprintf(buf, len, "%s Agent, context %s tokens\n\n",
                 hebrus_agent_brand_for(argv0), ctx);
    }
}

static inline const char *hebrus_eval_command_for(const char *argv0) {
    return hebrus_is_canonical_invocation(argv0) ? "hebrus-eval" : "ds4-eval";
}

static inline const char *hebrus_bench_command_for(const char *argv0) {
    return hebrus_is_canonical_invocation(argv0) ? "hebrus-bench" : "ds4-bench";
}

#endif
