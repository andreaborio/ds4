#include "ds4_qwen_unicode.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_qwen_unicode_data.inc"

#define QWEN_ARRAY_LEN(a) (sizeof(a) / sizeof((a)[0]))
#define QWEN_UNICODE_MASK 0x1fffffu

int ds4_qwen_utf8_next(
        const char *input,
        size_t      input_len,
        size_t     *offset,
        uint32_t   *codepoint) {
    if (!offset || !codepoint || *offset > input_len ||
        (!input && input_len != 0)) {
        return -1;
    }
    if (*offset == input_len) return 0;

    const unsigned char *s = (const unsigned char *)input;
    const size_t pos = *offset;
    const unsigned char c0 = s[pos];
    uint32_t cp;
    size_t width;

    if (c0 <= 0x7f) {
        cp = c0;
        width = 1;
    } else if (c0 >= 0xc2 && c0 <= 0xdf) {
        if (input_len - pos < 2 || (s[pos + 1] & 0xc0) != 0x80) return -1;
        cp = ((uint32_t)(c0 & 0x1f) << 6) |
             (uint32_t)(s[pos + 1] & 0x3f);
        width = 2;
    } else if (c0 >= 0xe0 && c0 <= 0xef) {
        if (input_len - pos < 3 ||
            (s[pos + 1] & 0xc0) != 0x80 ||
            (s[pos + 2] & 0xc0) != 0x80 ||
            (c0 == 0xe0 && s[pos + 1] < 0xa0) ||
            (c0 == 0xed && s[pos + 1] >= 0xa0)) {
            return -1;
        }
        cp = ((uint32_t)(c0 & 0x0f) << 12) |
             ((uint32_t)(s[pos + 1] & 0x3f) << 6) |
             (uint32_t)(s[pos + 2] & 0x3f);
        width = 3;
    } else if (c0 >= 0xf0 && c0 <= 0xf4) {
        if (input_len - pos < 4 ||
            (s[pos + 1] & 0xc0) != 0x80 ||
            (s[pos + 2] & 0xc0) != 0x80 ||
            (s[pos + 3] & 0xc0) != 0x80 ||
            (c0 == 0xf0 && s[pos + 1] < 0x90) ||
            (c0 == 0xf4 && s[pos + 1] >= 0x90)) {
            return -1;
        }
        cp = ((uint32_t)(c0 & 0x07) << 18) |
             ((uint32_t)(s[pos + 1] & 0x3f) << 12) |
             ((uint32_t)(s[pos + 2] & 0x3f) << 6) |
             (uint32_t)(s[pos + 3] & 0x3f);
        width = 4;
    } else {
        return -1;
    }

    *offset = pos + width;
    *codepoint = cp;
    return 1;
}

enum ds4_qwen_unicode_class ds4_qwen_unicode_classify(uint32_t codepoint) {
    if (codepoint >= 0x110000u) return DS4_QWEN_UNICODE_OTHER;

    size_t lo = 0;
    size_t hi = QWEN_ARRAY_LEN(qwen_uc_category_boundaries);
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        if ((qwen_uc_category_boundaries[mid] >> 2) <= codepoint) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo == 0) return DS4_QWEN_UNICODE_OTHER;
    return (enum ds4_qwen_unicode_class)
        (qwen_uc_category_boundaries[lo - 1] & 0x3u);
}

bool ds4_qwen_unicode_is_space(uint32_t codepoint) {
    size_t lo = 0;
    size_t hi = QWEN_ARRAY_LEN(qwen_uc_space_ranges) / 2;
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        const uint32_t first = qwen_uc_space_ranges[2 * mid];
        const uint32_t last = qwen_uc_space_ranges[2 * mid + 1];
        if (codepoint < first) {
            hi = mid;
        } else if (codepoint > last) {
            lo = mid + 1;
        } else {
            return true;
        }
    }
    return false;
}

static uint8_t qwen_unicode_ccc(uint32_t codepoint) {
    size_t lo = 0;
    size_t hi = QWEN_ARRAY_LEN(qwen_uc_ccc_boundaries);
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        if ((qwen_uc_ccc_boundaries[mid] >> 8) <= codepoint) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo == 0 ? 0 : (uint8_t)(qwen_uc_ccc_boundaries[lo - 1] & 0xffu);
}

typedef struct {
    uint32_t *value;
    size_t len;
    size_t cap;
} qwen_codepoints;

static bool qwen_codepoints_reserve_one(qwen_codepoints *points) {
    if (points->len < points->cap) return true;
    size_t cap = points->cap ? points->cap + points->cap / 2 : 32;
    if (cap <= points->cap) return false;
    if (cap > SIZE_MAX / sizeof(points->value[0])) return false;
    uint32_t *value = realloc(points->value, cap * sizeof(points->value[0]));
    if (!value) return false;
    points->value = value;
    points->cap = cap;
    return true;
}

static bool qwen_codepoints_append(qwen_codepoints *points, uint32_t cp) {
    if (!qwen_codepoints_reserve_one(points)) return false;
    points->value[points->len++] = cp;
    return true;
}

/* Canonical ordering only permutes consecutive non-starters.  CCC is one
 * byte, so a stable counting sort makes each run linear without imposing a
 * maximum run length.  Already ordered runs avoid scratch allocation. */
static bool qwen_canonical_order(qwen_codepoints *points) {
    uint32_t *scratch = NULL;
    size_t scratch_cap = 0;
    size_t pos = 0;

    while (pos < points->len) {
        while (pos < points->len && qwen_unicode_ccc(points->value[pos]) == 0) {
            pos++;
        }
        const size_t start = pos;
        uint8_t previous = 0;
        bool ordered = true;
        while (pos < points->len) {
            const uint8_t combining = qwen_unicode_ccc(points->value[pos]);
            if (combining == 0) break;
            if (previous > combining) ordered = false;
            previous = combining;
            pos++;
        }

        const size_t run_len = pos - start;
        if (ordered || run_len < 2) continue;
        if (run_len > scratch_cap) {
            if (run_len > SIZE_MAX / sizeof(scratch[0])) {
                free(scratch);
                return false;
            }
            uint32_t *grown = realloc(scratch, run_len * sizeof(scratch[0]));
            if (!grown) {
                free(scratch);
                return false;
            }
            scratch = grown;
            scratch_cap = run_len;
        }

        size_t bucket[256] = {0};
        for (size_t i = start; i < pos; i++) {
            bucket[qwen_unicode_ccc(points->value[i])]++;
        }
        size_t next = 0;
        for (size_t combining = 0; combining < QWEN_ARRAY_LEN(bucket);
             combining++) {
            const size_t count = bucket[combining];
            bucket[combining] = next;
            next += count;
        }
        for (size_t i = start; i < pos; i++) {
            const uint8_t combining = qwen_unicode_ccc(points->value[i]);
            scratch[bucket[combining]++] = points->value[i];
        }
        memcpy(points->value + start, scratch,
               run_len * sizeof(points->value[0]));
    }

    free(scratch);
    return true;
}

static bool qwen_decomposition_lookup(
        uint32_t        cp,
        const uint32_t **mapping,
        size_t          *mapping_len) {
    size_t lo = 0;
    size_t hi = QWEN_ARRAY_LEN(qwen_uc_decomposition_cps);
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        if (qwen_uc_decomposition_cps[mid] < cp) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo == QWEN_ARRAY_LEN(qwen_uc_decomposition_cps) ||
        qwen_uc_decomposition_cps[lo] != cp) {
        return false;
    }
    const uint16_t meta = qwen_uc_decomposition_meta[lo];
    const size_t offset = meta & 0x0fffu;
    const size_t len = (meta >> 12) + 1u;
    *mapping = qwen_uc_decomposition_pool + offset;
    *mapping_len = len;
    return true;
}

static bool qwen_decompose_append(qwen_codepoints *points, uint32_t cp) {
    enum {
        S_BASE = 0xac00,
        L_BASE = 0x1100,
        V_BASE = 0x1161,
        T_BASE = 0x11a7,
        V_COUNT = 21,
        T_COUNT = 28,
        N_COUNT = V_COUNT * T_COUNT,
        S_COUNT = 19 * N_COUNT,
    };
    if (cp >= S_BASE && cp < S_BASE + S_COUNT) {
        const uint32_t index = cp - S_BASE;
        const uint32_t leading = L_BASE + index / N_COUNT;
        const uint32_t vowel = V_BASE + (index % N_COUNT) / T_COUNT;
        const uint32_t trailing = index % T_COUNT;
        return qwen_codepoints_append(points, leading) &&
               qwen_codepoints_append(points, vowel) &&
               (trailing == 0 ||
                qwen_codepoints_append(points, T_BASE + trailing));
    }

    const uint32_t *mapping = NULL;
    size_t mapping_len = 0;
    if (!qwen_decomposition_lookup(cp, &mapping, &mapping_len)) {
        return qwen_codepoints_append(points, cp);
    }
    for (size_t i = 0; i < mapping_len; i++) {
        if (!qwen_codepoints_append(points, mapping[i])) return false;
    }
    return true;
}

static bool qwen_compose_hangul(uint32_t first, uint32_t second, uint32_t *out) {
    enum {
        S_BASE = 0xac00,
        L_BASE = 0x1100,
        V_BASE = 0x1161,
        T_BASE = 0x11a7,
        L_COUNT = 19,
        V_COUNT = 21,
        T_COUNT = 28,
        N_COUNT = V_COUNT * T_COUNT,
        S_COUNT = L_COUNT * N_COUNT,
    };
    if (first >= L_BASE && first < L_BASE + L_COUNT &&
        second >= V_BASE && second < V_BASE + V_COUNT) {
        *out = S_BASE + (first - L_BASE) * N_COUNT +
               (second - V_BASE) * T_COUNT;
        return true;
    }
    if (first >= S_BASE && first < S_BASE + S_COUNT &&
        (first - S_BASE) % T_COUNT == 0 &&
        second > T_BASE && second < T_BASE + T_COUNT) {
        *out = first + second - T_BASE;
        return true;
    }
    return false;
}

static bool qwen_compose_table(uint32_t first, uint32_t second, uint32_t *out) {
    size_t lo = 0;
    size_t hi = QWEN_ARRAY_LEN(qwen_uc_composition);
    while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        const uint64_t packed = qwen_uc_composition[mid];
        const uint32_t candidate_first = (uint32_t)(packed & QWEN_UNICODE_MASK);
        const uint32_t candidate_second =
            (uint32_t)((packed >> 21) & QWEN_UNICODE_MASK);
        if (candidate_first < first ||
            (candidate_first == first && candidate_second < second)) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo == QWEN_ARRAY_LEN(qwen_uc_composition)) return false;
    const uint64_t packed = qwen_uc_composition[lo];
    if ((uint32_t)(packed & QWEN_UNICODE_MASK) != first ||
        (uint32_t)((packed >> 21) & QWEN_UNICODE_MASK) != second) {
        return false;
    }
    *out = (uint32_t)((packed >> 42) & QWEN_UNICODE_MASK);
    return true;
}

static bool qwen_compose_pair(uint32_t first, uint32_t second, uint32_t *out) {
    return qwen_compose_hangul(first, second, out) ||
           qwen_compose_table(first, second, out);
}

static size_t qwen_compose_in_place(uint32_t *points, size_t len) {
    size_t write = 0;
    size_t starter = 0;
    uint8_t last_ccc = 0;
    bool have_starter = false;

    for (size_t read = 0; read < len; read++) {
        const uint32_t cp = points[read];
        const uint8_t combining = qwen_unicode_ccc(cp);
        uint32_t composed = 0;
        if (have_starter && (last_ccc == 0 || last_ccc < combining) &&
            qwen_compose_pair(points[starter], cp, &composed)) {
            points[starter] = composed;
            continue;
        }

        if (combining == 0) {
            starter = write;
            last_ccc = 0;
            have_starter = true;
        } else if (have_starter) {
            last_ccc = combining;
        }
        points[write++] = cp;
    }
    return write;
}

static size_t qwen_utf8_width(uint32_t cp) {
    if (cp <= 0x7f) return 1;
    if (cp <= 0x7ff) return 2;
    if (cp <= 0xffff) return 3;
    return 4;
}

static char *qwen_utf8_put(char *out, uint32_t cp) {
    if (cp <= 0x7f) {
        *out++ = (char)cp;
    } else if (cp <= 0x7ff) {
        *out++ = (char)(0xc0u | (cp >> 6));
        *out++ = (char)(0x80u | (cp & 0x3fu));
    } else if (cp <= 0xffff) {
        *out++ = (char)(0xe0u | (cp >> 12));
        *out++ = (char)(0x80u | ((cp >> 6) & 0x3fu));
        *out++ = (char)(0x80u | (cp & 0x3fu));
    } else {
        *out++ = (char)(0xf0u | (cp >> 18));
        *out++ = (char)(0x80u | ((cp >> 12) & 0x3fu));
        *out++ = (char)(0x80u | ((cp >> 6) & 0x3fu));
        *out++ = (char)(0x80u | (cp & 0x3fu));
    }
    return out;
}

bool ds4_qwen_nfc_normalize(
        const char *input,
        size_t      input_len,
        char      **output,
        size_t     *output_len) {
    if (output) *output = NULL;
    if (output_len) *output_len = 0;
    if (!output || !output_len) return false;
    if (!input && input_len != 0) return false;

    size_t first_non_ascii = 0;
    while (first_non_ascii < input_len &&
           (unsigned char)input[first_non_ascii] < 0x80) {
        first_non_ascii++;
    }
    if (first_non_ascii == input_len) {
        if (input_len == SIZE_MAX) return false;
        char *copy = malloc(input_len + 1);
        if (!copy) return false;
        if (input_len != 0) memcpy(copy, input, input_len);
        copy[input_len] = '\0';
        *output = copy;
        *output_len = input_len;
        return true;
    }

    qwen_codepoints points = {0};
    size_t offset = 0;
    while (offset < input_len) {
        uint32_t cp = 0;
        const int status = ds4_qwen_utf8_next(
            input, input_len, &offset, &cp);
        if (status != 1 || !qwen_decompose_append(&points, cp)) {
            free(points.value);
            return false;
        }
    }

    if (!qwen_canonical_order(&points)) {
        free(points.value);
        return false;
    }
    points.len = qwen_compose_in_place(points.value, points.len);
    size_t bytes = 0;
    for (size_t i = 0; i < points.len; i++) {
        const size_t width = qwen_utf8_width(points.value[i]);
        if (bytes > SIZE_MAX - width) {
            free(points.value);
            return false;
        }
        bytes += width;
    }
    if (bytes == SIZE_MAX) {
        free(points.value);
        return false;
    }
    char *normalized = malloc(bytes + 1);
    if (!normalized) {
        free(points.value);
        return false;
    }
    char *cursor = normalized;
    for (size_t i = 0; i < points.len; i++) {
        cursor = qwen_utf8_put(cursor, points.value[i]);
    }
    normalized[bytes] = '\0';
    free(points.value);
    *output = normalized;
    *output_len = bytes;
    return true;
}
