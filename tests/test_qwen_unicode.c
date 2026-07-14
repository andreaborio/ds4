#include "../ds4_qwen_unicode.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(cond) do {                                                     \
    if (!(cond)) {                                                           \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n",                     \
                __FILE__, __LINE__, #cond);                                  \
        return false;                                                        \
    }                                                                        \
} while (0)

static bool expect_nfc(
        const char *input,
        size_t      input_len,
        const char *expected,
        size_t      expected_len) {
    char *normalized = (char *)(uintptr_t)1;
    size_t normalized_len = SIZE_MAX;
    CHECK(ds4_qwen_nfc_normalize(
        input, input_len, &normalized, &normalized_len));
    CHECK(normalized != NULL);
    CHECK(normalized_len == expected_len);
    CHECK(memcmp(normalized, expected, expected_len) == 0);
    CHECK(normalized[normalized_len] == '\0');

    char *again = NULL;
    size_t again_len = 0;
    CHECK(ds4_qwen_nfc_normalize(
        normalized, normalized_len, &again, &again_len));
    CHECK(again_len == normalized_len);
    CHECK(memcmp(again, normalized, again_len) == 0);
    free(again);
    free(normalized);
    return true;
}

static bool expect_invalid_nfc(const unsigned char *input, size_t input_len) {
    char *normalized = (char *)(uintptr_t)1;
    size_t normalized_len = SIZE_MAX;
    CHECK(!ds4_qwen_nfc_normalize(
        (const char *)input, input_len, &normalized, &normalized_len));
    CHECK(normalized == NULL);
    CHECK(normalized_len == 0);
    return true;
}

static bool test_utf8(void) {
    static const char input[] = "A\0\xC3\xA9\xF0\x90\x97\x92";
    static const uint32_t expected[] = {'A', 0, 0x00e9, 0x105d2};
    size_t offset = 0;
    for (size_t i = 0; i < sizeof(expected) / sizeof(expected[0]); i++) {
        uint32_t cp = UINT32_MAX;
        CHECK(ds4_qwen_utf8_next(
            input, sizeof(input) - 1, &offset, &cp) == 1);
        CHECK(cp == expected[i]);
    }
    uint32_t cp = UINT32_MAX;
    CHECK(ds4_qwen_utf8_next(input, sizeof(input) - 1, &offset, &cp) == 0);
    CHECK(offset == sizeof(input) - 1);

    static const unsigned char bad[][4] = {
        {0x80, 0, 0, 0},
        {0xc0, 0x80, 0, 0},
        {0xe0, 0x80, 0x80, 0},
        {0xed, 0xa0, 0x80, 0},
        {0xf0, 0x80, 0x80, 0x80},
        {0xf4, 0x90, 0x80, 0x80},
        {0xf5, 0x80, 0x80, 0x80},
    };
    static const size_t bad_len[] = {1, 2, 3, 3, 4, 4, 4};
    for (size_t i = 0; i < sizeof(bad_len) / sizeof(bad_len[0]); i++) {
        offset = 0;
        cp = UINT32_MAX;
        CHECK(ds4_qwen_utf8_next(
            (const char *)bad[i], bad_len[i], &offset, &cp) == -1);
        CHECK(offset == 0);
        CHECK(cp == UINT32_MAX);
    }
    offset = 0;
    CHECK(ds4_qwen_utf8_next("\xE2\x82", 2, &offset, &cp) == -1);
    CHECK(offset == 0);
    offset = 2;
    CHECK(ds4_qwen_utf8_next("A", 1, &offset, &cp) == -1);
    CHECK(ds4_qwen_utf8_next(NULL, 1, &offset, &cp) == -1);
    return true;
}

static bool test_categories(void) {
    CHECK(ds4_qwen_unicode_classify('A') == DS4_QWEN_UNICODE_LETTER);
    CHECK(ds4_qwen_unicode_classify(0x105d2) == DS4_QWEN_UNICODE_LETTER);
    CHECK(ds4_qwen_unicode_classify(0x0301) == DS4_QWEN_UNICODE_MARK);
    CHECK(ds4_qwen_unicode_classify('7') == DS4_QWEN_UNICODE_NUMBER);
    CHECK(ds4_qwen_unicode_classify(0x2167) == DS4_QWEN_UNICODE_NUMBER);
    CHECK(ds4_qwen_unicode_classify(0x00bd) == DS4_QWEN_UNICODE_NUMBER);
    CHECK(ds4_qwen_unicode_classify(0x200d) == DS4_QWEN_UNICODE_OTHER);
    CHECK(ds4_qwen_unicode_classify(0x10ffff) == DS4_QWEN_UNICODE_OTHER);
    CHECK(ds4_qwen_unicode_classify(0x110000) == DS4_QWEN_UNICODE_OTHER);

    static const uint32_t whitespace[][2] = {
        {0x0009, 0x000d}, {0x0020, 0x0020}, {0x0085, 0x0085},
        {0x00a0, 0x00a0}, {0x1680, 0x1680}, {0x2000, 0x200a},
        {0x2028, 0x2029}, {0x202f, 0x202f}, {0x205f, 0x205f},
        {0x3000, 0x3000},
    };
    for (size_t i = 0; i < sizeof(whitespace) / sizeof(whitespace[0]); i++) {
        CHECK(ds4_qwen_unicode_is_space(whitespace[i][0]));
        CHECK(ds4_qwen_unicode_is_space(whitespace[i][1]));
        if (i == 0 || whitespace[i - 1][1] + 1 < whitespace[i][0]) {
            CHECK(!ds4_qwen_unicode_is_space(whitespace[i][0] - 1));
        }
        if (i + 1 == sizeof(whitespace) / sizeof(whitespace[0]) ||
            whitespace[i][1] + 1 < whitespace[i + 1][0]) {
            CHECK(!ds4_qwen_unicode_is_space(whitespace[i][1] + 1));
        }
    }
    CHECK(!ds4_qwen_unicode_is_space(0x200b));
    CHECK(!ds4_qwen_unicode_is_space(0xfeff));
    return true;
}

static bool test_nfc(void) {
    CHECK(expect_nfc(NULL, 0, "", 0));
    static const char ascii_nul[] = {'A', '\0', 'B'};
    CHECK(expect_nfc(ascii_nul, sizeof(ascii_nul),
                     ascii_nul, sizeof(ascii_nul)));

    CHECK(expect_nfc("e\xCC\x81", 3, "\xC3\xA9", 2));
    CHECK(expect_nfc("A\xCC\x95\xCC\x80", 5,
                     "\xC3\x80\xCC\x95", 4));
    CHECK(expect_nfc("A\xCC\x81\xCC\x80", 5,
                     "\xC3\x81\xCC\x80", 4));
    CHECK(expect_nfc("\xCC\x81!I", 4, "\xCC\x81!I", 4));

    CHECK(expect_nfc("\xE2\x84\xAA", 3, "K", 1));
    CHECK(expect_nfc("\xE2\x84\xAB", 3, "\xC3\x85", 2));

    CHECK(expect_nfc("\xE1\x84\x80\xE1\x85\xA1\xE1\x86\xA8", 9,
                     "\xEA\xB0\x81", 3));
    CHECK(expect_nfc("\xE1\x84\x80\xE1\x85\xA1\xE1\x86\xA7", 9,
                     "\xEA\xB0\x80\xE1\x86\xA7", 6));
    CHECK(expect_nfc("\xEA\xB0\x81", 3, "\xEA\xB0\x81", 3));

    /* Astral composition proves that the packed 21+21+21-bit table is not
     * accidentally truncated to BMP code points. */
    CHECK(expect_nfc(
        "\xF0\x91\x82\x99\xF0\x91\x82\xBA", 8,
        "\xF0\x91\x82\x9A", 4));

    /* Hugging Face tokenizers 0.22.2 normalizes with Unicode 9.0: Todhri was
     * unassigned then, so this Unicode-16 composition must remain decomposed.
     * The regex classifier above must nevertheless recognize U+105D2 as L. */
    CHECK(expect_nfc(
        "\xF0\x90\x97\x92\xCC\x87", 6,
        "\xF0\x90\x97\x92\xCC\x87", 6));
    CHECK(expect_nfc("\xF0\x90\x97\x89", 4,
                     "\xF0\x90\x97\x89", 4));

    static const unsigned char invalid0[] = {0xc0, 0x80};
    static const unsigned char invalid1[] = {0xe2, 0x82};
    static const unsigned char invalid2[] = {0xed, 0xa0, 0x80};
    static const unsigned char invalid3[] = {0xf4, 0x90, 0x80, 0x80};
    CHECK(expect_invalid_nfc(invalid0, sizeof(invalid0)));
    CHECK(expect_invalid_nfc(invalid1, sizeof(invalid1)));
    CHECK(expect_invalid_nfc(invalid2, sizeof(invalid2)));
    CHECK(expect_invalid_nfc(invalid3, sizeof(invalid3)));

    char *out = (char *)(uintptr_t)1;
    CHECK(!ds4_qwen_nfc_normalize("x", 1, &out, NULL));
    CHECK(out == NULL);
    CHECK(!ds4_qwen_nfc_normalize("x", 1, NULL, &(size_t){0}));
    size_t out_len = SIZE_MAX;
    CHECK(!ds4_qwen_nfc_normalize(NULL, 1, &out, &out_len));
    CHECK(out == NULL);
    CHECK(out_len == 0);
    return true;
}

static bool test_large_adversarial_canonical_order(void) {
    /* Alternating CCC 232/230 makes insertion sorting quadratic: every new
     * low mark would cross the complete high-class suffix accumulated so far.
     * Alternate two distinct CCC-230 marks as well to verify stable order. */
    const size_t pairs = 32768;
    const size_t len = 1 + pairs * 4;
    char *input = malloc(len);
    char *expected = malloc(len);
    CHECK(input != NULL && expected != NULL);
    input[0] = '#';
    expected[0] = '#';
    for (size_t i = 0; i < pairs; i++) {
        input[1 + 4 * i] = (char)0xcc;       /* U+0315, CCC 232 */
        input[1 + 4 * i + 1] = (char)0x95;
        input[1 + 4 * i + 2] = (char)0xcc;  /* CCC 230 */
        input[1 + 4 * i + 3] = (char)(i % 2 == 0 ? 0x80 : 0x81);

        expected[1 + 2 * i] = (char)0xcc;
        expected[1 + 2 * i + 1] = (char)(i % 2 == 0 ? 0x80 : 0x81);
        expected[1 + 2 * pairs + 2 * i] = (char)0xcc;
        expected[1 + 2 * pairs + 2 * i + 1] = (char)0x95;
    }

    char *normalized = NULL;
    size_t normalized_len = 0;
    CHECK(ds4_qwen_nfc_normalize(
        input, len, &normalized, &normalized_len));
    CHECK(normalized_len == len);
    CHECK(memcmp(normalized, expected, len) == 0);
    free(normalized);
    free(expected);
    free(input);
    return true;
}

int main(void) {
    if (!test_utf8() || !test_categories() || !test_nfc() ||
        !test_large_adversarial_canonical_order()) return 1;
    puts("Qwen Unicode tests: OK");
    return 0;
}
