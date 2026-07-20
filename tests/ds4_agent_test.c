#include <stdio.h>

#include "internal/ds4_agent_unit.h"

int main(void) {
    int failures = ds4_agent_unit_tests_run();
    if (failures) {
        fprintf(stderr, "ds4-agent tests: %d failure(s)\n",
                failures);
        return 1;
    }
    puts("ds4-agent tests: ok");
    return 0;
}
