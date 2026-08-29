# Risks, open decisions and prohibited shortcuts

## 1. Decisions already fixed by this plan

- Target name is Qwen3.8-Flash-Next, HF `qwen4_exp`, not the cloud-only name.
- First support is text-only, Metal production, base next-token model.
- Distinct Hebrus family/profile; no masquerading as Qwen3.6.
- ExpertMajor v2 embedded routed store remains the accepted expert container.
- PLE gets a separately owned fixed-page store/extent and explicit SSD cache.
- Transformers pin is the mathematical oracle; llama.cpp/MLX are comparisons.
- QSA production is physically sparse; no dense QxK mask.
- GDN recurrent state starts FP32.
- Routed experts should be resident for primary M5 64 GB; PLE is streamed.
- MTP and vision are later, optional artifacts/phases.

Changing any item requires an ADR/plan update, not an opportunistic code branch.

## 2. Decisions that require measurements/review

| Decision | Required evidence | Owner/deadline |
|---|---|---|
| exact GGUF architecture/profile naming | ADR plus converter/llama metadata mapping | Phase 0 |
| PLE v1 codec/page stride/rows per page | decode error, random-I/O amplification, cache data | Phase 2/7 |
| release routed codec/mix | quality suite + exact bytes + M5 speed | Phase 8/9 |
| dense/shared role codecs | per-role error and full logits/quality | Phase 8 |
| KV BF16/Q8/Q4 and pooled index | long-context parity/quality/memory/speed | Phase 6/9 |
| MTLIO versus pread selector | physical capability/cold-warm A/B/failure behavior | Phase 7 |
| first advertised maximum context | normal AUTO endpoint evidence | Phase 9 |
| other Apple tiers | physical device qualification | Phase 10 |
| MTP artifact/algorithm | acceptance/speed + complete rollback | Phase 11 |

## 3. Correctness risks

### PLE hash drift

Any float conversion, signed overflow, wrong prime/head order or EOS semantics
selects unrelated rows while still producing fluent-looking output. Store arrays
as uint64, generate golden IDs for all heads and test wraparound.

### Norm convention ambiguity

Most Qwen4Exp norms are zero-centered `(1+w)`, but GDN's gated RMSNorm is
conventional `w`. Some third-party conversions pre-add one. Manifest each role's
convention and reject ambiguous artifacts; never blanket-transform all norms.

### GR equation drift

Injection weights derive from normalized wide input before the sub-block and use
`/4`; they do not depend on the block output. Branch group norm width is 2560,
not 10240. Final mixer has no injection/extra output norm.

### QSA near-correctness

Common failures: cache processed keys instead of raw, pool physical rather than
logical groups, apply ReLU after summing heads, wrong group start RoPE, reuse one
tail for a prefill tile, or desynchronize index and main KV slots. These often
appear only above 2051 or after multi-turn wrap/rewind.

### GDN state/head mapping

Q/K repeat-interleave 16->48 and conventional gated norm differ from simple
reuse assumptions. FP16 state may drift slowly. Segment/chunk/padding bugs can
survive one-token tests.

### Quant selection discontinuities

Small router/indexer error changes expert/block IDs and cascades. Role-specific
selection metrics are required in addition to logit/perplexity averages.

## 4. Memory and performance risks

### Q4 does not fit

The backbone/experts dominate even after external PLE. Continuing to optimize a
Q4 resident assumption wastes time. Mixed low-bit expert quality is the central
feasibility question.

### “SSD experts are fast” fallacy

A cold top-10 record set over 48 layers can exceed a GiB/token at Q4. Expert SSD
is a lower-memory/control path, not the primary speed design unless hit-rate data
proves nearly resident behavior.

### Warm page-cache benchmark illusion

PLE warm results hide random I/O. Release evidence requires fresh cold process/
cache methodology plus natural warm runs, physical read bytes and layer-1 wait.

### `mmap`/Metal registration wires PLE

Accidental first-touch, preload, residency advice or Metal buffer registration
can consume tens of GB. The PLE extent is excluded from warmup/page spans and is
accessed through explicit bounded pages. Mapping metadata alone is not residency.

### Dense sparse-attention implementation

Both young upstream ports can be mathematically sparse yet backend-dense. Hebrus
must instrument gather/selected bytes and prohibit QxK masks.

### Dispatch explosion

Ten experts x48 layers plus GR/GDN/QSA can become command/dispatch-bound. Profile
before deep arithmetic tuning; fuse safely and group prompt tokens by expert.

### Long-context cache capacity

BF16 main+raw-index QSA reaches ~6.75 GiB at 262K. Pooled index and quant KV are
important, but must not precede the BF16 reference/state-correctness path.

## 5. Runtime/ownership risks

- late PLE/expert worker installs into reused generation;
- eviction of page/record referenced by uncompleted Metal command;
- partial publish of GDN, QSA, PLE history or public checkpoint;
- cancellation leaks a lease/staging buffer or leaves pending worker ownership;
- multi-request PLE/index state contamination;
- cache rewind across a four-token group or nine-position PLE conv boundary;
- pressure shrink below in-flight minimum;
- huge command buffer causes GPU timeout or holds resources too long.

The response is explicit tickets, generations, lease/inflight sequences,
transactional private state and forced-failure tests—not more cleanup comments.

## 6. Upstream maturity risk

llama.cpp support landed only days before this research. Relevant primary issue/
PR inputs include multi-segment/rotated cache behavior, concurrent crashes,
deep-context slowdown and speculative cache reuse. MLX-LM support was still in
review and MLX-VLM dense-masks QSA. Pin revisions and treat fixes as test ideas;
do not chase mutable `main` during implementation.

## 7. Legal/distribution risk

The checkpoint uses Qwen Community License 1.0. Distribution of original or
quantized weights, naming, attribution and usage terms require project/legal
review before downloader/release work. The implementation may be developed from
public interfaces, but artifact publication is a separate approval.

## 8. Prohibited shortcuts

- accepting a generic `qwen4_exp` based only on architecture string;
- using Qwen3.6 IDs/constants/store family/cache classification;
- allowing canonical routed tensors or PLE sidecar discovery as fallback;
- treating ignored vision/MTP tensors as harmless extras;
- converting all 180B parameters to BF16 in RAM;
- applying `+1` to every norm or none;
- hashing through float/Metal fast math;
- calling dense SDPA with a sparse mask in production;
- caching only QSA selected IDs without raw/pooled index state semantics;
- reading selected experts synchronously one by one;
- allowing OS page faults to be the PLE I/O scheduler;
- using forced SSD as the M5 product benchmark;
- claiming 262K because config declares it;
- enabling server/download support before physical qualification;
- mixing MTP/vision into the base correctness milestone.

## 9. Stop conditions

Pause and escalate rather than weakening the contract if:

- source/config/tensor inventory no longer matches the pinned revision;
- no artifact meets both quality and 64 GB AUTO budget;
- PLE cold I/O cannot be bounded/overlapped enough for useful decode;
- QSA implementation cannot prove sparse physical work;
- a required change breaks current Qwen3.6/DeepSeek/GLM semantics;
- license review blocks artifact distribution;
- requested lower-memory tier cannot hold minimum exact state/cache floors.

The fallback is to narrow the advertised context/tier or keep the work
experimental, not silently alter output semantics.

