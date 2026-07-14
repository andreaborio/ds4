# Qwen Unicode data provenance

The Qwen3.6 tokenizer pinned by this branch is
`Qwen/Qwen3.6-35B-A3B` revision
`995ad96eacd98c81ed38be0c5b274b04031597b0`. Its `tokenizer.json` declares
an NFC normalizer and this pre-tokenizer expression:

```text
(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
```

The official Python stack used for the committed tokenizer fixtures uses
`tokenizers` 0.22.2. That release resolves NFC through
`unicode-normalization-alignments` 0.1.12, whose normalization tables are
Unicode 9.0.0, and regular expressions through `onig_sys` 69.9.1, whose
Oniguruma property and case-fold tables are Unicode 16.0.0. The version split
is observable: NFC leaves `U+105D2 U+0307` decomposed because Todhri was not in
Unicode 9, while the regex recognizes U+105D2 as a letter under Unicode 16.

`qwen_unicode_ucd_cache.txt` is a lossless filter of only the source records
needed by ds4. It was generated from the official archives below:

- Unicode 9.0.0 `UCD.zip`:
  `df9e028425816fd5117eaea7173704056f88f7cd030681e457c6f3827f9390ec`
- Unicode 16.0.0 `UCD.zip`:
  `c86dd81f2b14a43b0cc064aa5f89aa7241386801e35c59c7984e579832634eb2`

The cache contains Unicode-16 L/M/N and White_Space ranges plus Unicode-9
canonical combining classes, canonical decompositions, and the relevant full
composition exclusions. It is covered by `UNICODE_DATA_LICENSE.txt`.

Refresh the cache and generated include only from archives matching those
digests:

```sh
python3 tests/gen_qwen_unicode.py --refresh-cache \
  --ucd9 /path/to/Unicode-9.0.0-UCD.zip \
  --ucd16 /path/to/Unicode-16.0.0-UCD.zip
```

Normal verification is fully offline and deterministically regenerates the C
include in memory from the committed semantic cache. The generator pins the
cache itself to SHA-256
`400d9a7d10217d81727248529d2297e47da60ef2451b654b68c0fead4528a88e`, so
changing both the cache and generated include cannot make `--check` pass:

```sh
python3 tests/gen_qwen_unicode.py --check
```

The generated immutable tables occupy 43,408 bytes before compiler/linker
metadata: 7,848 bytes of category boundaries, 80 bytes of whitespace ranges,
1,984 bytes of combining-class boundaries, 25,976 bytes of canonical
decomposition data, and 7,520 bytes of composition pairs.
