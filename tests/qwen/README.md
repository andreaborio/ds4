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

## Model-backed DS4/llama.cpp logits comparator and smoke gate

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
MODEL=/absolute/path/to/Qwen3.6-35B-A3B-ds4-Q4_K_S.gguf
LLAMA_SOURCE=/absolute/path/to/llama.cpp-qwen36
LLAMA_DEBUG=/absolute/path/to/llama-debug
OUT=/tmp/qwen36-logits-oracle
mkdir -p "$OUT/llama"

python3 -c \
  'import json,sys; d=json.load(open(sys.argv[1])); sys.stdout.write(next(v["rendered"] for v in d["chat_vectors"] if v["name"] == "plain_thinking"))' \
  "$ROOT/tests/qwen/qwen36_tokenizer_chat_golden.json" > "$OUT/prompt.txt"

./ds4 -m "$MODEL" --dump-tokens \
  --prompt-file "$OUT/prompt.txt" > "$OUT/ds4-tokens.txt"
DS4_QWEN_EXPERIMENTAL_CPU=1 ./ds4 -m "$MODEL" --cpu --ctx 256 \
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

## Experimental Qwen Metal AUTO runtime

The current experimental runtime qualifies one text model and one normalized
layout: `Qwen3.6-35B-A3B-ds4-Q4_K_S.gguf`, GGUF architecture `qwen35moe`, with
733 tensors and no vision or MTP payload.  It is not a generic Qwen runner.
Runtime validation checks the required Qwen3.6 geometry and tokenizer controls,
the complete tensor name/shape/type inventory, the blessed static payload size,
and aligned, in-bounds, non-overlapping tensor ranges.  Absolute offsets need
not match the reference artifact, and runtime acceptance is not a file-SHA
attestation.  Qwen3.7 and separate Coder variants are not aliases for this
model.

### Qualified artifact provenance

The measured artifact starts from the prebuilt Unsloth
`unsloth/Qwen3.6-35B-A3B-GGUF` repository at revision
`a483e9e6cbd595906af30beda3187c2663a1118c`, file
`Qwen3.6-35B-A3B-UD-Q4_K_S.gguf`.  The 20,893,015,008-byte source has SHA-256
`a8138f183e3993f12cdc23afd2babb8cdb084e64088ce4a256d49101d47b949c`.
It was used as the base rather than downloading the full BF16 safetensors.

Normalization restores the official padding ID and chat template, converts
three routed `ffn_down_exps` banks from Q6_K to the uniform Q4_K cache layout,
and converts `output.weight` from Q6_K to Q8_0.  The other 729 tensor payloads
are byte-identical to the Unsloth source.  The resulting 20,808,563,424-byte
artifact has SHA-256
`c33efb67bde86c9ba1f9e79c2dc42627170963bef0e915ab9b91a55cfb6d0fcd`.
This exact hash is the artifact qualified by the evidence below; the raw
Unsloth GGUF is not a drop-in substitute for this experimental DS4 path.

The literal environment guard is the experimental opt-in.  The Metal backend,
AUTO residency, and power 100 already default on Apple builds; the backend and
power are shown for clarity and reproducibility:

```sh
export DS4_QWEN_EXPERIMENTAL_METAL=1
./ds4 -m /absolute/path/to/Qwen3.6-35B-A3B-ds4-Q4_K_S.gguf \
  --metal --power 100 ...
```

Omitting the environment guard fails closed, as does selecting a non-Metal
backend or an effective power setting below 100.  AUTO resolves to resident
only when both the fixed Metal working-set budget and a live host-pressure
preflight prove it safe; otherwise it selects bounded SSD streaming.
`--resident` is a strict request and fails unless both admission checks pass;
the point-in-time pressure snapshot cannot guarantee future memory availability.
`--ssd-streaming` remains the explicit forced-streaming mode.  Cache, cold, and
other SSD-only options continue to imply SSD and conflict with `--resident`.
Resident mode maps the complete tensor payload, disables DS4's explicit expert
cache/`pread` path, and uses full-tensor Metal kernels.  Metal residency requests
are budgeting hints rather than whole-file pre-faults, so this mode name alone
does not prove that every GGUF page remained physically resident throughout a
run.

The supported path does not implement neural CPU+GPU hybrid inference.  Metal
runs the dense, recurrent, attention, router, and routed-expert math.  Resident
mode consumes the router's top-8 IDs and weights on the active Metal command
timeline without a host readback.  The CPU still handles tokenization and
sampling; selected-ID readback, expert-cache policy, and GGUF `pread` I/O are
confined to SSD mode and explicit route trace/replay tooling.  A future CPU/GPU
expert split would be a separate performance experiment and must not be
inferred from the current orchestration.

### Expert-cache tiers

Each routed Q4_K expert instance contains gate, up, and down matrices and is
1,769,472 bytes (1.6875 MiB).  One token selects 40 x 8 = 320 instances.
The safe floor is therefore 321 complete experts, or 568,000,512 bytes
(0.529 GiB): one full route plus one slot so the next load cannot evict an
in-flight expert.  Requests below 321 are rejected.  The floor is checked
against the effective locked budget, not only the requested count: the Metal
kernel test forces the first lazy `mlock` to fail and verifies rejection before
readahead, `pread`, cache installation, miss accounting, or token accounting.

The controlled comparison tier is 640 experts (1.055 GiB), covering two
complete routes.  A smaller accepted cache emits an anti-thrashing warning.
This is a benchmark tier, not the automatic production choice or a guarantee
that 640 experts fit every machine under current memory pressure.  When AUTO
selects SSD and no count or byte budget is supplied, the Qwen-specific strict
planner charges the 2.50 GiB static page set, context/runtime,
current-pressure margin, and Metal headroom. Above 16 GiB those reserves remain
independent. On a 16 GiB Mac the unpinned static pages share ordinary headroom,
AUTO consumes the largest complete 320-expert cycle admitted by the live and
platform budgets, and bounded file-backed inactive pages receive full credit
only under normal macOS pressure. Unknown or elevated pressure retains
half-credit and fails closed near the boundary. Expert
slots are populated and locked lazily, but Metal cache storage is allocated in
321-expert slabs (about 0.529 GiB), so the first route allocates one complete
working set plus its safety slot and later routes grow storage incrementally.
The planner charges the complete cache budget rather than only the currently
populated slots. The generic DeepSeek 4 GiB slab default remains unchanged.

### AUTO and resident validation

The resident top-8 Metal path is covered model-free against the same Q4_K
fixture as SSD streaming.  It consumes the GPU-produced route in one expert
pass and reduces the eight weighted expert outputs in one dispatch.  Its output
is checked against the compatibility path using the exact same GPU-produced
IDs and weights.  The test also proves one GPU-only route and zero host
readbacks; the compatibility run proves the readback counter separately.  SSD
streaming retains its two top-4 selected-slot passes, and invalid resident
routes leave all outputs and cache counters unchanged.

The autoregressive Gated DeltaNet has a separate `key_dim=128` Metal kernel.
One 32x4 threadgroup advances four value rows, keeps four adjacent state cells
per SIMD lane in registers, and reads/writes each recurrent-state element once.
The generic kernel remains the fallback for other shapes.  The model-free test
forces the specialized path with an odd seven-row tail and compares output and
mutated state with the scalar oracle; maximum absolute errors were 1.86e-9 and
1.49e-8 respectively.

Decode cannot reuse prompt-style token batching: token `n+1` depends on the
sampled result and mutated recurrent/KV state from token `n`.  The structural
decode optimization is therefore parallelism *within* one token.  Qwen's
previous top-8 router computed exponentials in parallel but selected eight
experts with a serial `8 x 256` scan in one Metal thread.  The resident path now
uses two-level SIMD reductions for those eight selections, preserving the
reference normalization and the lower-expert-ID tie break.  The old serial
kernel remains available with `DS4_QWEN_DISABLE_PARALLEL_ROUTER=1` as a precise
diagnostic fallback.

The model-free gate compares the parallel and serial production batch ABIs on
13 adversarial rows, including all-equal logits, exact maximum ties, and finite
softmax-underflow extremes.  Selected IDs and weights are bit-identical between
the two Metal kernels; the maximum weight difference from the CPU oracle is
5.96e-8.  Undersized buffers, guard regions, and non-finite input behavior are
checked separately.

The opt-in stage profiler can isolate a layer and decode position:

```sh
DS4_QWEN_METAL_DECODE_STAGE_PROFILE=1 \
DS4_QWEN_METAL_DECODE_STAGE_PROFILE_LAYER=17 \
DS4_QWEN_METAL_DECODE_STAGE_PROFILE_POSITION=64 \
  ./ds4 -m "$MODEL" --metal --resident [normal generation arguments]
```

Every reported boundary closes and waits for the active Metal command buffer.
Its timings are diagnostic attribution, not normal pipeline throughput.  On an
Apple M5 Pro, it measured the serial top-8 selection at 0.451-0.455 ms per
profiled layer and the parallel selection at 0.159-0.174 ms.

With the resident model warm, the deterministic n96 command below was run in
the order serial/parallel/parallel/serial/serial/parallel.  Serial decode
measured 36.97, 37.00, and 37.19 t/s; parallel decode measured 62.72, 63.05,
and 62.96 t/s.  The medians are 37.00 and 62.96 t/s, a 70.2% improvement.  All
six outputs have SHA-256
`a650b56ceb47dc8715f87c125c7eeab506bc4a510512cedbd190e38c46df5f33`,
and `/usr/bin/time -l` recorded zero process swaps in every run.

After the rejected prototype below was removed and the tree rebuilt cleanly,
five quiet-desktop confirmations measured 232.37/66.02, 233.33/66.18,
220.06/64.04, 223.64/65.63, and 223.73/65.20 t/s for
prefill/generation.  The medians are 223.73 and 65.63 t/s; the best decode run
is 66.18 t/s.  The same binary measured a 185.35/50.20 t/s median while the
desktop compositor and Codex renderer were actively contending for the GPU.
The quiet run hid those application windows temporarily but did not terminate
their processes.  This gap is reported explicitly: 65.63 t/s is an achievable
machine-local resident result, not a promise under arbitrary interactive GPU
load.  All ten outputs retained the hash above and every process reported zero
swaps.

A Q8_0 paired-matvec prototype was also tested and deliberately not retained.
After clock and page-cache warmup, its balanced A/B median was 63.315 t/s
against 63.09 t/s for the two standard matvecs (+0.36%), with fully overlapping
ranges.  That result is below run-to-run noise and does not justify another
kernel.  Neither experiment changes or further quantizes any model weights.

Thinking mode exposed a separate server-side cost.  Its fixed sampling policy
is temperature 1, top-p 1, and min-p 0.05; the old full-vocabulary path called
`expf()` twice across all 248,077 selectable Qwen tokens for every generated
token.  The optimized sampler rejects logits conservatively below the relative
min-p boundary before exponentiation, retains the exact eligible probabilities
once, and samples them in the original vocabulary order.  A model-free
reference gate covers six cases, including three temperatures around the min-p
boundary, and 128 seeds each: all 768 selected tokens and post-sample RNG states
are identical to the prior algorithm.  The logarithmic pre-screen keeps only a
32-epsilon rounding guard; the former full-natural-log guard still evaluated
thousands of probabilities that the exact min-p check could never retain on
flat distributions.  On a
31-token server prompt, two pre-change Thinking runs measured 39.55 and 39.86
t/s; three post-change runs measured 59.01, 58.59, and 58.48 t/s.  The medians
are 39.71 and 58.59 t/s, a 47.6% improvement.  After narrowing the guard, an
active-desktop parity check measured 46.61 t/s without Thinking and 46.54 t/s
with Thinking at short context; at a 1,403-token prefix the same pair measured
33.03 and 33.19 t/s.  The lower absolute pair reflects shared compositor/Codex
GPU load; the near-identical pairs show that full-vocabulary sampling no longer
adds a material decode penalty.

For context positions at or above 256, full-attention decode now uses a second
structural optimization.  Eight SIMD groups scan independent cache slices,
keep their query and stable online-softmax accumulators in registers, then
merge the eight partial states once per query head.  This removes three
threadgroup barriers per cached token from the former serial-context kernel.
`DS4_QWEN_DISABLE_PARALLEL_GQA_DECODE=1` restores that kernel for controlled
diagnosis.  The standalone Metal oracle covers cache frontiers 1, 7, 257,
1,025, and 4,097; maximum absolute error is 8.39e-5 at the longest frontier,
and K/V/query guard regions remain unchanged.

A model-backed adjacent A/B used the same 1,428-token prompt, 200-token greedy
generation, and warm resident model.  Serial and parallel prefill measured
320.44 and 321.79 t/s respectively, making the host/GPU state comparable;
decode measured 36.18 and 44.06 t/s, a 21.8% improvement.  Both responses have
SHA-256
`7111fd2b619195bd56b85b2d1baf3bb2b6aea377dea5d6da394e43a6b2c9bbf5`.
A DSBox-like Thinking request spanning positions 1,426 through 1,930 measured
320.30 prefill and 41.04 generation t/s; its 50-token decode windows declined
gradually from 43.22 to 38.70 t/s as the attention prefix grew.  The complete
504-token response has SHA-256
`868c9de51f2154ccd092768aa1a112fd5660140fb8174a2b0eef2cba03fe94d8`.
These changes optimize sampling and attention scheduling only; they do not
change or further quantize dense model weights.

Model-backed AUTO is pressure-dependent by design.  A 64 GiB Mac is not an
unconditional resident tier: if other applications consume unified memory,
AUTO can correctly choose SSD.  A physical 16 GiB machine is expected to use
SSD for this 19.37 GiB tensor payload. AUTO caps that host tier at the safe
budget computed from a normal-pressure snapshot and rounds down to complete
320-expert cycles; the selected count is pressure-dependent. A bounded physical
M1 Pro 16 GiB server smoke is now recorded below; the full sustained cold/warm
gate remains open.

### Physical M1 Pro 16 GiB admission smoke

On 2026-07-15, the branch based on `1fdfe080ea63` was built natively on a
MacBook Pro (`MacBookPro18,3`), Apple M1 Pro 8-core, 16 GiB, macOS 26.5
(`25F71`). The machine was on battery at 38%. The normalized Q4_K_S artifact
had the SHA-256 recorded below. The server used Metal AUTO, context 8,192,
power 100, eight CPU threads, and the automatically capped 321-expert cache.

The preflight reported 5.69 GiB reclaimable, a 2.50 GiB shared
static/headroom reserve, 0.25 GiB pressure margin, 0.37 GiB runtime, and a
2.56 GiB safe expert budget. AUTO resolved to SSD and readiness succeeded.
Two identical non-thinking server requests contained 32 prompt tokens and
reached the 64-token generation cap:

| Run | Prefill | Generation | Total | Result |
| --- | ---: | ---: | ---: | --- |
| First | 4.17 t/s | 8.71 t/s | 15.025 s | 64 tokens |
| Immediate repeat | 5.45 t/s | 8.83 t/s | 13.120 s | byte-identical content |

macOS pressure remained normal; the lowest sampled availability was 50%. The
swapout counter remained exactly 2,010,446 and reported swap use remained
395.31 MiB before and after both requests. The first run followed model copy and
hash activity, so this is admission, generation, and no-new-swapout evidence,
not a controlled cold-device benchmark or the complete preregistered sustained
16 GiB gate.

The first admission build above still capped Qwen AUTO at the floor despite its
larger safe budget. A follow-up removed that model-specific cap while leaving
the pressure reserves and DeepSeek's separately measured low-RAM ceiling
unchanged. The same host then ran a warm-cache B/A/B, two identical 32+64-token
requests per arm:

| Arm | Cache | Generation runs | Mean | Process RSS |
| --- | ---: | ---: | ---: | ---: |
| B1 | AUTO 1,281 (2.11 GiB) | 9.85 / 9.71 t/s | 9.78 t/s | 2.48 GiB |
| A | Explicit 321 (0.53 GiB) | 8.45 / 8.87 t/s | 8.66 t/s | 0.86 GiB |
| B2 | AUTO 1,281 (2.11 GiB) | 9.38 / 9.59 t/s | 9.49 t/s | 2.51 GiB |

The combined AUTO mean was 9.63 t/s, 11.2% above the control. Pressure stayed
normal with at least 46% reported availability; swap use stayed at 387.62 MiB
and the swapout counter remained exactly 2,010,466. One output from each cache
size was byte-identical, SHA-256
`81a77f323f8fafb9d1e7d68038c198a54ed0948b5cc0ffdd2d66df7c78e0d3fd`.
This comparison was on battery with a warm file cache (61% and discharging at
the end) and does not replace the preregistered cold/warm sustained gate.

### Same-artifact llama.cpp comparison

The official llama.cpp b10016 macOS arm64 release was tested with the same
normalized GGUF. On the M1 Pro 16 GiB host, default Metal mmap/autofit did not
complete its first `pp32` test before macOS entered elevated pressure; swapouts
rose by 178,744 pages and swap use grew by about 2.61 GiB. Minimal scalar
`pp1`/`tg4` attempts with all MoE layers on CPU and with a 4 GiB Metal fit
margin also entered elevated pressure. They were stopped by the pressure
watchdog, so no unsafe partial t/s value is reported.

On the M5 Pro 64 GiB host, the full artifact fit and both runtimes stayed under
normal pressure with zero new swapouts. Three page-touched DS4 resident runs
measured a 218.30 t/s prefill median and 63.94 t/s generation median. Three
llama.cpp CLI runs on the same 43-token rendered prompt measured 252.1 and
60.3 t/s. DS4 was 13.4% slower in prefill, 6.0% faster in generation, and 4.0%
faster by the derived complete 43+96-token time. The canonical rendered prompt
produced the same exact 43 token IDs in both tokenizers, and both CLIs produced
the same visible continuation through `return curr`.

The official `llama-bench` reference measured median `pp43` 508.969 t/s,
`tg96` 57.596 t/s, and combined `pp43+tg96` 80.032 t/s. Its synthetic prompt
microbenchmark excludes real chat-template/final-logits/sampling work and is
recorded separately rather than compared directly with DS4's sampled CLI
prefill. Exact commands, all samples, memory evidence, and the 16 GiB failure
probes are in
[`../../docs/benchmarks/2026-07-15-qwen-ds4-vs-llamacpp.md`](../../docs/benchmarks/2026-07-15-qwen-ds4-vs-llamacpp.md).

### Reproduce the resident coding benchmark

This is the resident counterpart of the bounded coding smoke below.  It uses
the same 43-token prompt and 96-token generation cap, but admits the complete
19.37 GiB model mapping only while the live pressure preflight can preserve the
configured headroom:

```sh
MODEL=/absolute/path/to/Qwen3.6-35B-A3B-ds4-Q4_K_S.gguf

DS4_QWEN_EXPERIMENTAL_METAL=1 \
DS4_METAL_MEMORY_REPORT=1 \
  ./ds4 -m "$MODEL" --metal --resident \
    -c 160 -n 96 -t 18 \
    --temp 0 --top-p 1 --min-p 0 --seed 1 --nothink \
    -p 'Scrivi solo codice Python: una funzione fibonacci(n) iterativa, con validazione per n negativo.'
```

On an Apple M5 Pro with 64 GiB unified memory, the GPU-only top-8 route before
the Gated DeltaNet specialization measured 32.71/30.24, 32.55/30.21, and
33.02/30.01 t/s for prefill/generation: medians of 32.71 and 30.21 t/s.  With
the parallel GDN-128 kernel, three cool-state runs measured 40.48/37.08,
40.12/37.38, and 41.13/37.25 t/s: medians of 40.48 and 37.25 t/s.  This is a
23.3% generation improvement from the immediately preceding implementation
and a 39.9% improvement from the earlier 26.62 t/s resident confirmation.

Resident prefill now runs layer-major, in chunks of at most 64 tokens.  Dense
projections use the existing batched matrix kernels, while dedicated sequence
kernels keep causal-convolution and Gated DeltaNet state in registers across a
chunk.  Full-attention queries scan only their causal K/V prefix; batched
router and routed-MoE kernels keep all top-8 routes on Metal.  The fixed batch
scratch allocation is included in the resident admission budget and the graph
lifetime test.  `DS4_QWEN_DISABLE_RESIDENT_BATCH_PREFILL=1` selects the scalar
fallback for differential diagnosis.

With the model pages warm and memory pressure normal, three consecutive n96
runs of the command above measured 170.29/30.40, 179.63/30.47, and
175.77/30.32 t/s for prefill/generation: medians of 175.77 and 30.40 t/s.  An
immediately following scalar-fallback run measured 31.93/29.31 t/s, making the
controlled prompt-prefill improvement 5.50x without a decode regression in
that run order.  Throughput is prompt-length, temperature, and page-cache
dependent; this is a local implementation A/B, not a cross-runtime benchmark.

For the 43-token prompt, batch/scalar next-token logits had the same top-1,
20/20 top-20 overlap, 98/100 top-100 overlap, RMSE 0.07970, and maximum absolute
difference 0.40870.  A 122-token prompt forced two chunks and retained the same
top-1, 5/5 top-5, 19/20 top-20, and 97/100 top-100 overlap (RMSE 0.10999,
maximum absolute difference 0.64091).  The drift is expected because the
batched quantized matrix kernels use half-width activation tiles; it is not
presented as bit-identical logits.  Model-free sequence tests independently
compare convolution, DeltaNet state, and causal GQA with repeated scalar
oracles, with maximum absolute errors no larger than 2.98e-8.

Every complete n96 run recorded 5,520 GPU-only routed-MoE token/layer calls,
zero route readbacks, and 4,140 specialized GDN token/layer calls.  Generated
stdout remained byte-identical across all three runs and to the prior resident
and SSD output at SHA-256
`a650b56ceb47dc8715f87c125c7eeab506bc4a510512cedbd190e38c46df5f33`.

The earlier bounded scalar queue remains as a diagnostic fallback when batched
prefill is disabled.  It caps resident scalar prefill at eight command buffers
in flight; `DS4_QWEN_DISABLE_RESIDENT_PREFILL_QUEUE=1` disables that queue as
well.  SSD mode and route trace/replay remain synchronous because expert-cache
selection is still token-dependent.

A controlled implementation A/B on the same machine separated the two changes:
the compatibility readback/two-top-4 path generated at 22.90 t/s; keeping the
route on GPU while retaining two top-4 passes reached 27.00 t/s; the
single-top-8 path reached 28.17 t/s before the GDN work.  The temporary MoE A/B
switches were removed from the production source.  These are short,
deterministic local measurements rather than a cross-runtime `llama-bench`
comparison.

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

At runtime build `04fb4cc2a3e8` on an Apple M5 Pro with 64 GiB unified memory,
this command completed a real Metal graph and
SSD-selected-expert forward.  The backend recorded 40 routed-MoE selections,
320 cold misses, 0.53 GiB of logical expert `pread` demand, no non-finite
logits, and argmax token 846 (`user`) at 25.9613953.  The confirmatory rerun
completed in 0.29 s and `/usr/bin/time -l` reported 704,495,616 bytes maximum
RSS and zero swap.  The OS page cache was not flushed after preceding model
runs, so this timing is correctness/footprint evidence, not a cold-start
benchmark.

The Metal vector was compared offline over all 248,077 non-padding IDs with a
CPU vector captured from the same `04fb4cc2a3e8` runtime build and the pinned
llama.cpp oracle described above.  llama.cpp was pinned to
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

At runtime build `04fb4cc2a3e8` on the same M5 Pro 64 GiB machine, the prompt
contained 43 tokens and the 96-token cap produced a complete iterative
`fibonacci(n)` implementation through `return curr`.  The closing Markdown
fence was not emitted before the cap; this is recorded as truncation, not
presented as an EOS-complete answer.

The same prompt, sampling arguments, context, model, and runtime build were also
run through the separately built CPU reference.  This diagnostic reported
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
| Metal + SSD, cache 640 | 19.43 t/s | 19.94 t/s | 7.14 s | 1,276,952,576 B |
| DS4 CPU reference | 27.20 t/s | 27.09 t/s | 5.32 s | 13,585,743,872 B |

These post-rebase confirmations followed earlier model runs without flushing
the macOS page cache; `--ssd-streaming-cold` clears DS4's expert cache, not the
OS cache.  Treat the throughput as warm-page-cache confirmation, not a fresh
cold-disk benchmark.

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
The post-rebase runtime-build evidence bundle used to prepare this record was
kept outside the worktree at
`../qwen36-ds4-artifact/logs/metal-ssd-final-04fb4cc/`.
It is intentionally not committed because it contains model-run artifacts.
Its primary files have these hashes:

- `metal-logits.json`:
  `decc87665fc08665946d296d0936a5c89913511cb88098f805fd6587b61d450c`;
- `stderr.log`:
  `98267a0ec5e62c46b294fbb0961ebcaf425b64e7f0ac2ff4e371ca475773e204`;
- `coding-640-n96.stdout.log`:
  `a650b56ceb47dc8715f87c125c7eeab506bc4a510512cedbd190e38c46df5f33`;
- `coding-640-n96.stderr.log`:
  `420dcf20d2c0139417c443419e2f550dbe6b26edcbbf30e3ab3d7186972f70af`;
- `coding-cpu-n96.stdout.log`:
  `a650b56ceb47dc8715f87c125c7eeab506bc4a510512cedbd190e38c46df5f33`;
- `coding-cpu-n96.stderr.log`:
  `d873b429fc4aa373e7a3c78ca156f00244a968a71d1b166f97cb8369c9f9376b`.

Both measurements are from an M5 Pro with 64 GiB unified memory.  They do not
prove operation on a physical 16 GiB Mac.  The evidence bundle in this section
predates resident layer-major prefill; its SSD path remains scalar
(`prefill_cap=1`), prioritizing bounded expert-cache behavior over throughput.
The opt-in runtime currently rejects or leaves unsupported:

- `--quality` and MTP;
- session payloads/snapshots and layer slices/distributed execution;
- imatrix collection;
- expert profiles/hotlists, expert preloading, and directional steering;
- non-routed pin profiles and power settings below 100.

Normalized-vs-original-Unsloth scoring, a broader multi-position oracle, disk
snapshot restore, server tool-call sessions, longer contexts, cold device-I/O,
and physical 16 GiB pressure/swap measurements remain useful follow-up work;
they are not extra promotion gates beyond the standard model/backend checks.
The current evidence qualifies this exact text-only artifact on Metal, not
generic Qwen support or arbitrary community GGUFs.
