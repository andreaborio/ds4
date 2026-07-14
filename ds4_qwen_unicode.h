#ifndef DS4_QWEN_UNICODE_H
#define DS4_QWEN_UNICODE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Qwen3.6's tokenizer combines two deliberately different Unicode data sets:
 * NFC from Hugging Face tokenizers uses Unicode 9.0, while its Oniguruma
 * regular expression uses Unicode 16.0 categories.  Keep these helpers
 * model-specific instead of inheriting the host C library's Unicode version. */
enum ds4_qwen_unicode_class {
    DS4_QWEN_UNICODE_OTHER  = 0,
    DS4_QWEN_UNICODE_LETTER = 1,
    DS4_QWEN_UNICODE_MARK   = 2,
    DS4_QWEN_UNICODE_NUMBER = 3,
};

/* Decode one strict UTF-8 scalar.  Returns 1 for a scalar, 0 at end, and -1
 * for malformed input.  On error, offset is not advanced. */
int ds4_qwen_utf8_next(
        const char *input,
        size_t      input_len,
        size_t     *offset,
        uint32_t   *codepoint);

/* Unicode 16.0 General_Category classes used by \p{L}, \p{M}, and \p{N}. */
enum ds4_qwen_unicode_class ds4_qwen_unicode_classify(uint32_t codepoint);

/* Unicode 16.0 White_Space, matching Oniguruma's UTF-8 \s class. */
bool ds4_qwen_unicode_is_space(uint32_t codepoint);

/* Normalize valid UTF-8 to Unicode 9.0 NFC.  The result is malloc-owned,
 * length-delimited, and also NUL-terminated for convenience.  Embedded NULs
 * are preserved.  Failure leaves *output NULL and *output_len zero. */
bool ds4_qwen_nfc_normalize(
        const char *input,
        size_t      input_len,
        char      **output,
        size_t     *output_len);

#endif
