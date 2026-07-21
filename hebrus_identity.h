#ifndef HEBRUS_IDENTITY_H
#define HEBRUS_IDENTITY_H

#include <stdbool.h>
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

#endif
