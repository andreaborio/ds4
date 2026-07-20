static int agent_test_failures;

static void agent_test_assert(bool cond, const char *expr,
                              const char *file, int line) {
    if (cond) return;
    fprintf(stderr, "%s:%d: assertion failed: %s\n", file, line, expr);
    agent_test_failures++;
}

#define AGENT_TEST_ASSERT(expr) \
    agent_test_assert((expr), #expr, __FILE__, __LINE__)

static void test_agent_edit_upto_tail_newline_is_not_part_of_anchor(void) {
    const char *data =
        "CFLAGS = -Wall -Wextra -g\n"
        "LDFLAGS =\n"
        "\n"
        "all: bc\n"
        "\n"
        "bc: main.c\n"
        "\t$(CC) $(CFLAGS) -o bc main.c $(LDFLAGS)\n"
        "\n"
        "clean:\n"
        "\trm -f bc\n";
    const char *old =
        "CFLAGS = -Wall -Wextra -g\n"
        "LDFLAGS =\n"
        "\n"
        "all: bc\n"
        "\n"
        "bc: main.c\n"
        "\t$(CC) $(CFLAGS) -o bc main.c $(LDFLAGS)\n"
        "\n"
        "[upto]\n"
        "clean:\n";

    const char *match = NULL;
    size_t match_len = 0;
    bool anchored = false;
    char err[128] = {0};
    AGENT_TEST_ASSERT(agent_edit_find_old_span(data, strlen(data), old,
                                              &match, &match_len, &anchored,
                                              err, sizeof(err)));
    AGENT_TEST_ASSERT(anchored);
    AGENT_TEST_ASSERT(match == data);
    AGENT_TEST_ASSERT(match_len == strlen(data) - strlen("\trm -f bc\n"));
}

static void test_agent_edit_upto_requires_tail_after_newline_strip(void) {
    const char *data = "head\nbody\ntail\n";
    const char *old = "head\n[upto]\n";
    const char *match = NULL;
    size_t match_len = 0;
    bool anchored = false;
    char err[128] = {0};

    AGENT_TEST_ASSERT(!agent_edit_find_old_span(data, strlen(data), old,
                                               &match, &match_len, &anchored,
                                               err, sizeof(err)));
    AGENT_TEST_ASSERT(strstr(err, "must include a unique tail anchor") != NULL);
}

static void test_agent_returned_user_text_preserves_ownership_and_order(void) {
    agent_worker w = {0};
    w.wake_fd[0] = -1;
    w.wake_fd[1] = -1;
    pthread_mutex_init(&w.mu, NULL);

    worker_return_user_text(&w, xstrdup("first rejected message"));
    worker_return_user_text(&w, xstrdup("second rejected message"));
    char *returned = worker_take_returned_user_text(&w);
    AGENT_TEST_ASSERT(returned != NULL);
    AGENT_TEST_ASSERT(returned && !strcmp(
        returned,
        "first rejected message\n\nsecond rejected message"));
    AGENT_TEST_ASSERT(worker_take_returned_user_text(&w) == NULL);

    free(returned);
    pthread_mutex_destroy(&w.mu);
}

int ds4_agent_unit_tests_run(void) {
    agent_test_failures = 0;
    test_agent_edit_upto_tail_newline_is_not_part_of_anchor();
    test_agent_edit_upto_requires_tail_after_newline_strip();
    test_agent_returned_user_text_preserves_ownership_and_order();
    return agent_test_failures;
}
