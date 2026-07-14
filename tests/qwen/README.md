# Qwen3.6 reference fixtures

`qwen36_tokenizer_chat_golden.json` is collected from the official
`Qwen/Qwen3.6-35B-A3B` `tokenizer.json` at the pinned revision recorded in the
file.  That JSON is authoritative for the GGUF `qwen35` pre-tokenizer; the
seven official audio/TTS controls from `tokenizer_config.json` are appended at
their assigned IDs.  Transformers' `Qwen2Tokenizer` is otherwise used only to
render the canonical Jinja chat template before the JSON tokenizer encodes it.
It covers byte-BPE splitting, Unicode, whitespace, code, special tokens,
thinking controls, the canonical chat template, and a tool-call round trip.
The chat vectors also freeze Qwen's reasoning-retention boundary, embedded
`<think>` fallback, every JSON argument type, multi-call and grouped-response
formatting, content before a tool call, `preserve_thinking`, and Unicode/tool
schema serialization order.

Refresh it intentionally with:

```sh
uv run \
  --with 'transformers==5.13.1' \
  --with 'tokenizers==0.22.2' \
  --with 'jinja2==3.1.6' \
  --with 'huggingface-hub==1.23.0' \
  python tests/qwen/collect_reference.py
```

Verify a checked-in fixture against the pinned source with the same command and
`--check`.  This networked collector is not part of `make model-free-test`;
the C tokenizer tests consume the frozen data without contacting the Hub.

`qwen36_tokenizer_fixture.inc` is the compact C closure of those golden cases:
the required final symbols, every ranked merge candidate encountered on the
official path, and all expected token IDs.  Keeping every candidate is
intentional; a fixture containing only winning merges would not catch a BPE
implementation that ignores rank.  Regenerate or verify it against the pinned
`tokenizer.json` with:

```sh
TOK=/path/to/Qwen3.6-35B-A3B/tokenizer.json
uv run --with 'tokenizers==0.22.2' \
  python tests/qwen/collect_tokenizer_fixture.py \
  --tokenizer-json "$TOK" --check
```

Official cases containing literal added tokens are marked as trusted text.
The additional `untrusted_literal_controls_and_pad_are_data` safety case is
derived from the pinned tokenizer with only its added-token matcher removed;
it proves that production user-text tokenization keeps the same bytes as BPE
data.  Only a trusted, already-rendered chat prompt may turn them into control
IDs.  Four supplemental TEXT vectors freeze the scanner's ordered-regex edge
cases: Unicode case-folded contractions, punctuation prefixes, whitespace
backtracking across CR/LF, Unicode whitespace, and non-ASCII number classes.

`literal_controls_in_user_content_reference` is intentionally different: the
JSON records the canonical Jinja rendered text and its official whole-string
token IDs, but the compact trusted-rendered C fixture skips it.  Those literal
spellings originate in user data, so a structured security renderer must encode
them as ordinary BPE data instead of granting them control-token meaning.  The
case is marked `reference_only_untrusted_content` to make that distinction
machine-checked by the fixture collector.

`literal_controls_as_data` freezes that ordinary-BPE counterpart for the exact
same user payload.  It is collected with the pinned tokenizer's
`Tokenizer.encode_special_tokens=True` mode, checked to contain no control-token
IDs, and included in the compact C closure as a normal `TEXT` case.  Some
canonical prompt atoms are declared as non-special added tokens in the source
JSON, so the collector first promotes all known controls to special in a private
clone while verifying that every official ID remains unchanged; the trusted
official tokenizer itself is not modified.

The tokenizer's Unicode behavior is frozen separately in
`qwen_unicode_ucd_cache.txt`: Unicode 9.0 NFC plus Unicode 16.0 `L/M/N` and
`White_Space` properties, matching the two libraries used by the pinned
`tokenizers` build.  See `UNICODE_DATA_PROVENANCE.md` for the source hashes,
license, refresh command, and the observable version split.  Its normal
offline gate is:

```sh
make qwen-unicode-test
```

`qwen36_chat_template.jinja` is the byte-exact canonical template from the
same pinned model revision.  Its SHA-256 is
`e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259`;
the metadata gate rejects template drift because the C renderer implements
these fixed semantics rather than executing arbitrary Jinja from a GGUF.

`qwen36_gdn_golden.inc` is a small scalar Gated DeltaNet oracle derived from
the official Transformers fallback equations after GGUF conversion.  It
covers causal convolution, the recurrent delta update, runtime V-head mapping,
gated RMSNorm, Qwen's F32
softmax/top-8 routing and shared-expert gate, plus identical state across
decode and different prefill chunk boundaries.  It has no Python package or
network dependency.  Verify it with:

```sh
python3 tests/qwen/collect_gdn_reference.py --check
```

`qwen36_attention_golden.inc` freezes the other token-mixer path: per-head
Q/K RMSNorm using the converter-shifted weights, the fused per-head `[Q, gate]`
projection layout, text-only split-half partial RoPE, contiguous GQA head
repetition, F32 causal softmax, and the elementwise sigmoid output gate.  It
also proves that prefix and one-shot causal results agree.  Refresh or verify
it with:

```sh
python3 tests/qwen/collect_attention_reference.py --check
```

These scalar fixtures are correctness oracles only.  Passing them does not
enable Qwen CPU, Metal, or SSD-streaming inference; the runtime remains
fail-closed until the complete model graph passes end-to-end logits gates.
`test_v_tiling_contract.py` separately freezes the pinned converter's V-side
permutation, including QKV/conv, Z, controls, and output-projection columns.
It does not execute the converter, bind a GGUF, test quantization, or validate
the complete Qwen graph.

## Model-backed DS4/llama.cpp logits gate

`compare_logits.py` compares one full DS4 `--dump-logits` JSON file with the
final-position logits saved by llama.cpp `llama-debug`.  The gate is
fail-closed before it calculates metrics: the DS4 model path must identify the
same local file passed as `--model`, the llama.cpp checkout must be at pinned
revision `bf2c86ddc0685f580595954056c2e77ebabfab4f`, the llama output basename
must match that model, the llama-debug load log must name the same absolute
GGUF file, and the exact DS4 and llama.cpp prompt token IDs must be equal.  It
reports finite coverage, top-1 agreement, top-5/20/64 overlap, cosine
similarity, RMSE, and maximum absolute error.

Qwen IDs `248077..248319` are padding beyond the effective vocabulary.  They
are excluded from numeric metrics and from every argmax/top-k selection.  A
DS4 dump that reports one of them as argmax is rejected.

### Required pinned llama-debug patch

At the pinned revision, `examples/debug/debug.cpp` ignores the common
`--parse-special` setting: both its tokenization sites call
`common_tokenize(ctx, params.prompt, add_bos)` and therefore tokenize the
canonical chat controls as ordinary text.  Such output is not a valid oracle
for DS4.  In the isolated reference checkout, change **both** calls to:

```cpp
common_tokenize(ctx, params.prompt, add_bos, params.parse_special)
```

Rebuild `llama-debug` and invoke it with `--parse-special`.  The comparator
checks the pinned Git HEAD and checks that both source calls contain the patch;
it refuses an unpatched checkout.  Keep this two-line diagnostic patch local
to the reference checkout and record its diff with the model-backed result.

### Reproducible run

Use one absolute model path and one byte-exact, already-rendered prompt for both
engines.  This example extracts the frozen `plain_thinking` canonical prompt
without adding a trailing byte:

```sh
ROOT=$PWD
MODEL=/absolute/path/to/Qwen3.6-35B-A3B-DS4-Q4_K.gguf
LLAMA_SOURCE=/absolute/path/to/llama.cpp-qwen36
LLAMA_DEBUG=/absolute/path/to/llama-debug
OUT=/tmp/qwen36-logits-oracle
mkdir -p "$OUT/llama"

python3 -c \
  'import json,sys; d=json.load(open(sys.argv[1])); sys.stdout.write(next(v["rendered"] for v in d["chat_vectors"] if v["name"] == "plain_thinking"))' \
  "$ROOT/tests/qwen/qwen36_tokenizer_chat_golden.json" > "$OUT/prompt.txt"

./ds4 -m "$MODEL" --dump-tokens \
  --prompt-file "$OUT/prompt.txt" > "$OUT/ds4-tokens.txt"
./ds4 -m "$MODEL" --cpu --ctx 256 \
  --dump-logits "$OUT/ds4-logits.json" \
  --prompt-file "$OUT/prompt.txt"

# llama.cpp -f removes one trailing newline.  Preserve the rendered prompt's
# exact final newline by carrying a non-newline sentinel through command
# substitution, then removing only that sentinel.
PROMPT_TEXT=$(cat "$OUT/prompt.txt"; printf x)
PROMPT_TEXT=${PROMPT_TEXT%x}
"$LLAMA_DEBUG" -m "$MODEL" -p "$PROMPT_TEXT" \
  --parse-special --save-logits --logits-output-dir "$OUT/llama" \
  -ngl 0 -c 256 > "$OUT/llama-debug.log" 2>&1

MODEL_STEM=$(basename "$MODEL")
MODEL_STEM=${MODEL_STEM%.*}
python3 tests/qwen/compare_logits.py \
  --model "$MODEL" \
  --llama-source "$LLAMA_SOURCE" \
  --ds4-logits "$OUT/ds4-logits.json" \
  --ds4-tokens "$OUT/ds4-tokens.txt" \
  --llama-logits "$OUT/llama/llamacpp-$MODEL_STEM.bin" \
  --llama-prompt "$OUT/llama/llamacpp-$MODEL_STEM-prompt.txt" \
  --llama-log "$OUT/llama-debug.log" \
  --output "$OUT/comparison.json"
```

The binary file is preferred because `llama-debug` writes all native F32
values; its documented `index: value` `.txt` output is also accepted.  The
load log proves which local path llama.cpp opened but does not embed a model
hash, so preserve the exact command above and the independently verified GGUF
SHA-256 with the result.

The comparator and its complete synthetic fixture generator use only the
Python standard library.  Their offline gate is:

```sh
python3 tests/qwen/test_compare_logits.py -v
```
