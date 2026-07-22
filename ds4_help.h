#ifndef DS4_HELP_H
#define DS4_HELP_H

#include <stdbool.h>
#include <stdio.h>

typedef enum {
    DS4_HELP_DS4,
    DS4_HELP_SERVER,
    DS4_HELP_AGENT,
    DS4_HELP_BENCH,
    DS4_HELP_EVAL,
} ds4_help_tool;

/* Select the public command family from argv[0]. Canonical Hebrus binaries and
 * legacy command symlinks share one object graph, so help text must use the name
 * through which that graph was invoked. */
void hebrus_help_set_invocation(const char *argv0);

/* Return true after printing the canonical error for a retired distributed
 * option. Both the exact spelling and --option=value forms are recognized. */
bool ds4_help_reject_retired_distributed_option(
        FILE *fp, ds4_help_tool tool, const char *arg);
void ds4_help_print(FILE *fp, ds4_help_tool tool, const char *topic);

#endif
