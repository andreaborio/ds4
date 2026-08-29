# Test oracles, fixtures and qualification

## 1. Oracle hierarchy

1. Pinned Transformers `42ca970...` with the exact HF revision is the model
   oracle.
2. Small explicit Python/numpy calculations validate fixture generation.
3. `ds4_qwen4exp_ref` is an independent scalar C oracle.
4. Simple Metal path is compared to C.
5. Fused/quantized/SSD paths are compared to simple Metal and end-to-end model
   fixtures.
6. llama.cpp and MLX-VLM are differential signals only; disagreement triggers
   investigation, not automatic conformance.

Fixtures include provenance JSON: source/Transformers commit, package versions,
config hash, seed, dtype, device, input IDs, positions/masks, tensor role, output
shape/dtype and SHA-256 of binary arrays. Store compact arrays in the established
test fixture mechanism, not base64 blobs in source.

## 2. Tolerance policy

Define tolerance per operation and precision before writing an optimized kernel.
Suggested starting gates, to be calibrated on pinned reference data:

| Path | Check |
|---|---|
| exact integer/hash/IDs/state metadata | bit-exact |
| F32 scalar versus F32 Metal | `atol<=2e-5`, `rtol<=2e-5` unless reduction-sized bound documented |
| BF16/F16 activations | role-specific max/mean error plus downstream logits |
| quantized dense/expert/PLE/KV | per-tensor error, routing/selection overlap, logits KL and greedy agreement |

Never hide a large local error behind final argmax agreement. Router must report
top-10 ID agreement and weight error. QSA reports top-block Jaccard and exact ID
agreement for correctness codecs. Promotion thresholds are frozen from baseline
data, not relaxed after seeing a regression.

## 3. Primitive matrix

### Norm and GR

- all-zero zero-centered stored weight;
- conventional GDN gated norm with weight one;
- zero, constant, alternating, extreme and subnormal inputs;
- four branches deliberately different, proving group width 2560;
- `/4` before SiLU and before injection sigmoid independently perturbed;
- injection derived from normalized wide input, never block output;
- final mixer has no injection and no extra output norm.

### GDN

- Q/K/V projection segment sentinels and exact repeat-interleave 16->48;
- convolution kernel/history boundaries 1,2,3,4,5 tokens;
- sigmoid output gate versus SiLU negative fixture;
- F32 recurrence after 1, 10, 1K and long repeated steps;
- decode one-by-one equals chunks 1/2/3/4/63/64/65/2048/2049;
- padding masks, unequal sequences, reset/fork/copy/rollback;
- state nonfinite input fails without committing.

### Router/MoE

- equal logits choose smaller IDs deterministically;
- full F32 softmax, top-10 and renormalization;
- huge positive/negative logits without overflow;
- exactly 512 expert IDs and unique selected IDs;
- selected `(ID,weight)` pair preservation through group/scatter;
- top-10 sum versus wrong top-8/8+2 order negative controls;
- expert quant tails at dimensions 640/2560;
- routed plus sigmoid-gated shared expert.

### QSA

Lengths: `0,1,2,3,4,5,2047,2048,2049,2051,2052`, page/chunk boundaries,
65,535/65,536/65,537, 100K and near 262,144.

- raw key cached before pooling/norm/RoPE;
- only complete causal groups; group position is first token;
- `sum(ReLU(dot_h))`, with negative fixture `ReLU(sum(dot_h))`;
- top-512 tie rule; expansion and per-query tail <=3;
- width equals all visible keys and dense result for `n_kv<=2051`;
- no future token and mutation of later token cannot change earlier output;
- one-shot/chunk/decode equality;
- left-padding begins grouping at first real token;
- multi-sequence slot reuse/holes/wrap/rewind;
- raw-cache and incremental-pooled-cache identical selected IDs;
- KV quantization cannot alter index position ownership;
- allocation instrumentation proves no dense QxK mask.

### PLE

- SplitMix64 constants/multipliers bit-exact, forced odd and never float;
- all 16 primes, offsets, bigram/trigram heads and shard/page boundaries;
- missing predecessors use EOS 248044;
- EOS at current, previous and previous-previous position;
- two interleaved sequences and chunk boundaries;
- 64-bit overflow/wrap vectors;
- 16 rows flatten in correct head order;
- signed square root near zero/positive/negative;
- dilation-3 kernel-4 history uses nine positions;
- resident row, cache hit and cold SSD miss decode identically;
- corrupt/truncated/wrong-digest page and cancelled I/O;
- prompt dedup/scatter preserves token/head order.

## 4. Tokenizer/template matrix

Freeze byte-level expected IDs and rendered text for: ASCII, accents/NFC/NFD,
CJK, Arabic, emoji/astral, combining marks, tabs/newlines/spaces, code, JSON/XML,
invalid UTF-8 handling, empty strings and every special-token literal. Cover all
roles, multiple system messages if legal/illegal, tool definitions/calls/results,
thinking efforts `xhigh/medium/low`, thinking disabled, preservation policy and
multiturn assistant reasoning.

Security/provenance tests ensure literal user text cannot synthesize an internal
control token. Text-only profile rejects image/video content and reserved media
IDs from structured input, while ordinary safe literal rendering remains stable.

## 5. Loader/converter negative matrix

Each fixture alters exactly one field from `03-artifact-converter-and-admission.md`.
Assert failure phase, diagnostic key and zero GPU/bulk allocation. Fuzz GGUF and
both manifests with bounded input. Test checked multiplication near `UINT64_MAX`,
unaligned/truncated/overlapping extents, duplicate identities and path rename/
fsync failure. Converter subprocess failure leaves the prior target untouched.

## 6. Transaction and failure matrix

Inject failure after: PLE hash plan, cache reservation, each pread worker, digest,
PLE install, expert route readback, expert reservation/read/install, every Metal
command encoding category, command commit/completion, KV append reservation,
index append, GDN update, PLE conv update, logits copy and checkpoint publish.

Postconditions:

```text
public frontier/checkpoint/logits unchanged;
GDN/conv, PLE history/conv and QSA KV/index semantically unchanged;
no leaked lease/ticket/staging/buffer/thread;
no stale worker can install after generation reuse;
next clean token produces the same logits as a fresh session.
```

Run cancellation races repeatedly under ThreadSanitizer-compatible host tests
where possible and Metal validation on physical hardware.

## 7. Full-model fixtures

Start with short text-only prompts whose layer intermediates can be captured from
Transformers: embeddings, PLE rows/output, GR mixed/injection values, GDN/QSA
outputs, selected experts/weights, selected QSA blocks, post-MoE wide residual,
final hidden and logits. Use at least multilingual prose, code, mathematics,
thinking and tool-call prompts.

Qualification then compares:

- resident reference artifact versus Transformers;
- production mixed quant versus reference quality/logits;
- streamed versus resident PLE;
- normal AUTO versus forced SSD control;
- single-shot prefill, varied chunks and token decode;
- cold process and warm process kept separate.

Store exact generated text, token IDs, per-step logit hashes/selected probes and
artifact manifest hash. A fluent sample alone is not evidence.

## 8. Long-context and concurrency

Required prompt domains: ordinary text and a routing/QSA-stressing second domain.
Run 32K, 65,536, 100,000 and near 262,144 with probes at group/page boundaries.
Use repeated sentinel facts to detect lost/aliased cache positions. At least
8-16 turn chat catches index/KV drift.

Concurrency tests run two requests with distinct secret tokens, interleaved
PLE hashes, cache slots and cancellation. No PLE/index/speculative state crosses
request/session identity. Repeated load/unload and a long decode must not grow
pipelines, cache tables or unreclaimed resources.

## 9. Quality gates for aggressive quantization

Because a 64 GB profile likely uses mixed 1/2/3-bit experts, evaluate more than
perplexity:

- WikiText-like perplexity against reference;
- held-out multilingual/code/math/task suites allowed by project policy;
- next-token top-1 and top-k agreement over multiple domains;
- logit KL/cosine/error distribution;
- router top-10 overlap and weight error per layer;
- QSA selected-block overlap per layer/context;
- PLE row decode error;
- long generation repetition, collapse and tool-format validity.

Freeze acceptance limits with maintainers before choosing the release codec.
Report failures, not only aggregate means. MTP and vision require separate gates.

## 10. Repository gates

At every phase run focused tests plus repository-required format/build checks.
Before support merge run all Qwen3.6 suites and affected DeepSeek/GLM/shared
runtime tests, Metal model-free tests, loader/converter/download/release tests,
sanitizers and physical-hardware gates. `git diff --check` and documentation
contract validators are mandatory but not substitutes for runtime evidence.

