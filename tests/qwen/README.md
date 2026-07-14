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

These scalar fixtures are correctness oracles only.  Passing them alone does
not enable Qwen CPU, Metal, or SSD-streaming inference.  The model-backed
runtimes have separate fail-closed model, backend, layout, and opt-in gates.
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

## Experimental Qwen Metal + SSD runtime

The current experimental runtime supports exactly one text model and one
normalized layout: `Qwen3.6-35B-A3B-ds4-Q4_K_S.gguf`, GGUF architecture
`qwen35moe`, with 733 tensors and no vision or MTP payload.  It is not a
generic Qwen runner.  A community GGUF with the same architecture label is not
accepted unless its metadata, tensor inventory, types, and offsets match this
layout.  Qwen3.7 and separate Coder variants are not aliases for this model.

The Metal path is deliberately double opt-in.  It requires an Apple Metal build,
the exact environment guard, explicit Metal, and explicit SSD residency:

```sh
export DS4_QWEN_EXPERIMENTAL_METAL=1
./ds4 -m /absolute/path/to/Qwen3.6-35B-A3B-ds4-Q4_K_S.gguf \
  --metal --ssd-streaming --power 100 ...
```

Omitting the environment guard, `--metal`, or `--ssd-streaming` fails closed.
Resident Qwen Metal inference is not enabled by this experiment.  `--power`
must remain 100.

### Expert-cache tiers

Each routed Q4_K expert instance contains gate, up, and down matrices and is
1,769,472 bytes (1.6875 MiB).  One token selects 40 x 8 = 320 instances.
The safe floor is therefore 321 complete experts, or 568,000,512 bytes
(0.529 GiB): one full route plus one slot so the next load cannot evict an
in-flight expert.  Requests below 321 are rejected.

The recommended starting tier is 640 experts (1.055 GiB), covering two complete
routes.  A smaller accepted cache emits an anti-thrashing warning.  This is a
starting policy, not a guarantee that 640 experts fit every machine under
current memory pressure: the startup planner may reject an explicit request
that exceeds its safe host-memory budget.  If no count or byte budget is
supplied, AUTO selects a safe value at or above the 321-expert floor.

### Reproduce the one-token Metal/logits smoke

Use the same normalized GGUF and exact one-token rendered prompt as the pinned
CPU/llama.cpp oracle.  `printf` is intentional here: the prompt must not gain a
trailing newline.

```sh
MODEL=/absolute/path/to/Qwen3.6-35B-A3B-ds4-Q4_K_S.gguf
OUT=/tmp/qwen36-metal-one-token
mkdir -p "$OUT"
printf '%s' '<|im_start|>' > "$OUT/prompt.txt"

DS4_QWEN_EXPERIMENTAL_METAL=1 \
DS4_METAL_MEMORY_REPORT=1 \
DS4_MOE_RECORD_SELECTED_IDS="$OUT/selected-ids.bin" \
  /usr/bin/time -l ./ds4 -m "$MODEL" \
    --metal --ssd-streaming --ssd-streaming-cold \
    --ssd-streaming-cache-experts 321 \
    --power 100 -c 2 \
    --dump-logits "$OUT/metal-logits.json" \
    --prompt-file "$OUT/prompt.txt" \
    > "$OUT/stdout.log" 2> "$OUT/stderr.log"
```

At commit `70d515e51ebe7e168f36bf718c8ccff9d6393f3f` on an Apple M5
Pro with 64 GiB unified memory, this command completed a real Metal graph and
SSD-selected-expert forward.  The backend recorded 40 routed-MoE selections,
320 cold misses, 0.53 GiB of logical expert `pread` demand, no non-finite
logits, and argmax token 846 (`user`) at 25.9613953.  Wall time was 0.67 s
and `/usr/bin/time -l` reported 704,462,848 bytes maximum RSS.

The Metal vector was compared offline over all 248,077 non-padding IDs with the
CPU vector and the pinned llama.cpp oracle described above.  The Metal vector
is from `70d515e51ebe`; the unchanged CPU reference vector was captured at
`c2e34ccd579d`, and llama.cpp was pinned to
`bf2c86ddc0685f580595954056c2e77ebabfab4f`.

| Pair | Top-1 | Top-5 | Top-20 | Top-64 | Cosine | RMSE | Max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Metal / llama.cpp | match | 5/5 | 20/20 | 63/64 | 0.9999464514 | 0.02020756 | 0.11475825 |
| Metal / DS4 CPU | match | 5/5 | 20/20 | 63/64 | 0.9999474300 | 0.02007011 | 0.10157680 |
| DS4 CPU / llama.cpp | match | 5/5 | 20/20 | 63/64 | 0.9999761156 | 0.01349266 | 0.07191539 |

These are backend-tolerance results, not bit identity.  Metal evaluates
quantized weights against F32 activations while the scalar CPU reference also
quantizes some activations, so small deterministic drift is expected.

### Reproduce the bounded coding smoke

This deterministic run exercises multiple recurrent and full-attention
positions and uses the 640-expert tier:

```sh
MODEL=/absolute/path/to/Qwen3.6-35B-A3B-ds4-Q4_K_S.gguf
OUT=/tmp/qwen36-metal-coder-n96
mkdir -p "$OUT"

DS4_QWEN_EXPERIMENTAL_METAL=1 \
DS4_METAL_MEMORY_REPORT=1 \
DS4_MOE_RECORD_SELECTED_IDS="$OUT/selected-ids.bin" \
  /usr/bin/time -l ./ds4 -m "$MODEL" \
    --metal --ssd-streaming --ssd-streaming-cold \
    --ssd-streaming-cache-experts 640 \
    --power 100 -c 160 -n 96 -t 18 \
    --temp 0 --top-p 1 --min-p 0 --seed 1 --nothink \
    -p 'Scrivi solo codice Python: una funzione fibonacci(n) iterativa, con validazione per n negativo.' \
    > "$OUT/stdout.log" 2> "$OUT/stderr.log"
```

At commit `f0f62b2f0850` on the same M5 Pro 64 GiB machine, the prompt
contained 43 tokens and the 96-token cap produced a complete iterative
`fibonacci(n)` implementation through `return curr`.  The closing Markdown
fence was not emitted before the cap; this is recorded as truncation, not
presented as an EOS-complete answer.

The same prompt, sampling arguments, context, model, and commit were also run
through the separately built CPU reference.  This diagnostic mapped
13.59 GB maximum RSS and should not be repeated on a 16 GiB target; use the
CPU-only binary below only on a machine with sufficient headroom.

```sh
DS4_QWEN_EXPERIMENTAL_CPU=1 \
  /usr/bin/time -l build/cpu-$(uname -m)/bin/ds4 -m "$MODEL" \
    --cpu --power 100 -c 160 -n 96 -t 18 \
    --temp 0 --top-p 1 --min-p 0 --seed 1 --nothink \
    -p 'Scrivi solo codice Python: una funzione fibonacci(n) iterativa, con validazione per n negativo.' \
    > "$OUT/cpu-stdout.log" 2> "$OUT/cpu-stderr.log"
```

| Backend | Prefill | Generation | Wall | Maximum RSS |
| --- | ---: | ---: | ---: | ---: |
| Metal + SSD, cache 640 | 14.62 t/s | 15.61 t/s | 9.48 s | 1,276,805,120 B |
| DS4 CPU reference | 25.54 t/s | 25.47 t/s | 5.65 s | 13,585,776,640 B |

The two stdout files are byte-identical, both with SHA-256
`a650b56ceb47dc8715f87c125c7eeab506bc4a510512cedbd190e38c46df5f33`.
This deterministic 96-token equality is stronger continuation evidence than
plausible-looking text alone, although it remains one bounded prompt.

The Metal cache held 640 experts (1.05 GiB), locked the full buffer, and
reported 20,728 hits, 23,432 misses, a 0.469 hit rate, and 38.61 GiB logical
`pread` demand.  The 5,520 selected-ID records equal 138 forward positions x
40 layers: 43 prompt evaluations plus 95 decode evaluations, with the first of
96 output tokens sampled from the final prompt logits.

`miss_pread` is logical loader demand, not measured physical SSD traffic.
Likewise, the zero swap and block-I/O fields in the captured stderr are process
counters reported by `/usr/bin/time -l`, not device-level storage telemetry.

### Evidence identity and current limits

The normalized GGUF used by both model-backed runs has SHA-256
`c33efb67bde86c9ba1f9e79c2dc42627170963bef0e915ab9b91a55cfb6d0fcd`.
The local evidence bundle used to prepare this record was kept outside the
worktree at `../qwen36-ds4-artifact/logs/metal-ssd-smoke-20260714T213430/`.
It is intentionally not committed because it contains model-run artifacts.
Its primary files have these hashes:

- `metal-logits.json`:
  `decc87665fc08665946d296d0936a5c89913511cb88098f805fd6587b61d450c`;
- `metal-logits.stderr.log`:
  `06cf1f3705cd4df98f304b6d75fa7cfab625d96b859ee8b226f0e6af0c1927fa`;
- `coding-640-n96.stdout.log`:
  `a650b56ceb47dc8715f87c125c7eeab506bc4a510512cedbd190e38c46df5f33`;
- `coding-640-n96.stderr.log`:
  `df8609e3c1db2ff3313adb4c61c6375dc53ea41d306ddb9d7cdd8d1e3a155f91`;
- `coding-cpu-n96.stdout.log`:
  `a650b56ceb47dc8715f87c125c7eeab506bc4a510512cedbd190e38c46df5f33`;
- `coding-cpu-n96.stderr.log`:
  `d45694627341cf4da06e5d7ee5baecf742ebb65a2d9f3edb6210fe01135627f1`.

Both measurements are from an M5 Pro with 64 GiB unified memory.  They do not
prove operation on a physical 16 GiB Mac.  Prefill is scalar
(`prefill_cap=1`), prioritizing correctness over throughput.  The experimental
runtime currently rejects or leaves unsupported:

- `--quality` and MTP;
- session payloads/snapshots and layer slices/distributed execution;
- imatrix collection;
- expert profiles/hotlists, expert preloading, and directional steering;
- non-routed pin profiles and power settings below 100.

Longer continuations, resident/SSD equivalence, disk snapshot restore, server
tool-call sessions, 8K/32K context, and physical 16 GiB pressure/swap tests
remain release gates.  The one-token and bounded coding runs prove an alive,
numerically close Metal+SSD path; they do not by themselves promote it to the
general Qwen release path.
