# Upstream engine research: llama.cpp, MLX and Metal

Research snapshot: 2026-08-29. This document separates observed upstream
behavior from proposed Hebrus behavior. Pins and source URLs are registered in
`00-source-register.md`.

## 1. llama.cpp implementation

Qwen4Exp support landed in [PR #27742](https://github.com/ggml-org/llama.cpp/pull/27742)
on 2026-08-27. The plan inspected current commit
`cc83d7b4824f73cfdda4dfbb47ee39804f71b328`, with the original integration also
available at immutable merge commit
[`6c84c7d`](https://github.com/ggml-org/llama.cpp/commit/6c84c7d5d8833c6e0df69628f75a0f599797934e).
It was less than three days old during research; treat it as valuable but young.

### 1.1 Loader and graph

`src/models/qwen4exp.cpp` validates one PLE layer and the GR/GDN/QSA geometry,
loads the ordinary graph tensors and marks the giant per-layer token embedding
with `TENSOR_READ_LAZY`. It builds the four-stream residual, injects PLE before
layer index 1, performs GR read -> mixer -> GR injection twice per layer, and
uses a final no-injection mixer.

The converter performs nontrivial operations that must become Hebrus fixtures:

- split combined index Q/K projection;
- reorder GDN value-head layout;
- squeeze/transcode convolution layouts;
- preserve uint64 PLE constants instead of generic float metadata conversion;
- apply zero-centered norm conversion only to appropriate weights;
- concatenate 128 PLE shards lazily;
- exclude MTP from the base model.

llama.cpp's GGUF namespace `qwen4exp` includes metadata for hyper-connections,
indexer/compression, PLE layers/ngram/hash arrays and per-layer input embedding
width. Hebrus may reuse registered semantic names after ADR review, but physical
artifact/store fields remain Hebrus contracts.

### 1.2 Hybrid cache and QSA

`llama-memory-hybrid-idx.cpp` makes the indexer cache mirror the attention cache:
the same cell positions, sequence IDs, remove/copy operations and state restore.
This ownership design is directly transferable. State restore failure drops the
whole hybrid state rather than exposing partial divergence.

The QSA path stores raw index keys, forms logical complete groups, pools/norms/
rotates, scores with a ReLU per query-index head, selects groups and expands IDs.
It constructs a sparse mask around the generic attention graph. That proves
semantics, not necessarily sparse backend work: depending on backend, masked KV
may still be traversed. Hebrus therefore keeps compact IDs and gathers/visits
only selected KV.

llama.cpp's index-cache implementation is a conservative correctness reference.
Hebrus's proposed pooled-complete-group cache is an exact optimization: completed
group keys are immutable, so only a maximum-three raw tail is retained.

### 1.3 PLE lazy access

The model loader can mark tensors over 4 GiB lazy, avoid ordinary prefetch/
`WILLNEED`, and advise random access. GPU/row gather then relies on mapped file
pages and OS faults. This makes the model functional without eagerly loading a
95 GiB BF16 table, but does not give Hebrus deterministic page budget, explicit
overlap, physical-byte telemetry or priority. Hebrus uses a fixed-page extent,
bounded cache and asynchronous authoritative reads.

llama.cpp/Unsloth experiments around
[PR #27794](https://github.com/ggml-org/llama.cpp/pull/27794),
[PR #27837](https://github.com/ggml-org/llama.cpp/pull/27837) and
[discussion #27864](https://github.com/ggml-org/llama.cpp/discussions/27864)
are useful for page-cache/prefetch hypotheses. Their reported throughput depends
on machine, quant, context and OS cache; it is not a Hebrus target.

### 1.4 Upstream correctness signals

The integration PR reported reference-like perplexity and high short-run token/
selection agreement, but its support is evolving. Turn these issue classes into
Hebrus tests:

- [#27797](https://github.com/ggml-org/llama.cpp/issues/27797): multi-segment/
  attention-rotation behavior;
- [#27774](https://github.com/ggml-org/llama.cpp/pull/27774): rotated quantized KV;
- [#27835](https://github.com/ggml-org/llama.cpp/issues/27835): concurrent crash;
- [#27852](https://github.com/ggml-org/llama.cpp/issues/27852): speculative cache
  state reused across requests;
- [#27856](https://github.com/ggml-org/llama.cpp/issues/27856): deep-context QSA
  slowdown;
- [#27871](https://github.com/ggml-org/llama.cpp/issues/27871): 262K crash report.

Observed constraints such as one PLE layer, limited shared-token/multi-sequence
behavior and no merged MTP support are not Hebrus API constraints; they indicate
which cases need explicit design/tests.

## 2. MLX-VLM implementation

The most complete Apple-oriented port is
[MLX-VLM PR #2032](https://github.com/Blaizzy/mlx-vlm/pull/2032), beginning at
commit `505267caa84fb7ba89851719fbc2655a454ab2c8`. Its `qwen4_exp` package covers
language graph, checkpoint mapping, FP8 handling, external PLE and experimental
MTP. It is the best comparison for Python/MLX shape mapping on Apple unified
memory, not a low-level performance template.

### 2.1 Useful mapping behavior

- remove `model.language_model.` prefixes deliberately;
- separate base language, vision, MTP and PLE roles;
- split fused expert gate/up on the exact projection dimension;
- transpose depthwise convolutions to runtime layout;
- preserve conventional GDN norm versus zero-centered other norms;
- stream official FP8 tensors with `_scale_inv` rather than assuming ordinary
  FP8 payload;
- normalize PLE shard names and avoid eager concatenated copies.

Older community quantizations had already added one to zero-centered weights.
This demonstrates why the Hebrus manifest needs an explicit norm convention per
role and cannot infer it from a filename.

### 2.2 Dense-mask QSA limitation

MLX-VLM computes the QSA top blocks correctly, then constructs a dense boolean
`[batch,query,key]` mask and calls normal SDPA. The result is mathematically
sparse but can remain dense in memory/work. It also retains raw index keys and
repools context in composed MLX operations. Hebrus must instead:

- cache incremental pooled complete groups;
- keep compact selected block/token IDs;
- scan scores in bounded tiles;
- perform indexed KV gather/online softmax;
- avoid retracing/allocating a growing dense mask per decode token.

### 2.3 External PLE experiment

[MLX-VLM PR #2045](https://github.com/Blaizzy/mlx-vlm/pull/2045) uses a row-
interleaved external memory-mapped PLE, deduplicates indices, maintains a row
LRU, copies selected rows CPU->MLX and dequantizes on GPU. It provides strong
capacity evidence:

- roughly 30 GB for a Q4 PLE;
- even with PLE external, reported active Q4 model memory around 71.68 GB;
- a cold 65,545-token PLE lookup reportedly spent about 9.68 seconds in lookup;
- a reported M2 Ultra 128 GB decode was about 19 token/s in that experiment.

These measurements are upstream-specific, but establish two design facts: PLE
must be external to the active working set, and external PLE alone does not make
a Q4 backbone fit 64 GB. Hebrus changes the synchronous mmap/row-LRU hot path to
early hash planning, fixed-page cache, coalesced async I/O and layer-0 overlap.

MLX's approximate row payloads also inform candidates: affine Q4 group32 width
160 is about 100 bytes/row including scale/bias; an NVFP4-style row about 90
bytes. Sixteen useful rows are only ~1.4-1.6 KiB/token, but unrelated 4 KiB pages
can amplify physical reads toward 64 KiB/token.

### 2.4 MTP experiment

[MLX-VLM PR #2040](https://github.com/Blaizzy/mlx-vlm/pull/2040) reported a large
speedup and high acceptance on an M5 Max 128 GB 3-bit configuration. This is not
portable to M5 Pro 64 GB. Transferable lessons are: MTP consumes the complete
four-stream state, cache changes are transactional, and adaptive draft depth may
beat a fixed depth. Hebrus defers MTP until base capacity/correctness is solved.

## 3. MLX-LM status and review-derived tests

Text-only support was still open in
[MLX-LM PR #1788](https://github.com/ml-explore/mlx-lm/pull/1788). Review findings
provide a valuable negative-test list:

- exclude embedded MTP intentionally;
- split fused gate/up along the correct dimension;
- use zero-centered norm only where specified;
- preserve per-slot position offsets and QSA index cache during batch/cache merge;
- do not form an all-masked QSA row for a short history;
- begin group pooling at the first real token under left padding;
- ensure GDN/PLE convolution and n-gram history ignore padding;
- preserve index state when quantizing main KV.

MLX lazy evaluation/unified memory are useful scheduling concepts, but Hebrus
must avoid per-token host materialization (`item`/`tolist` equivalents), enormous
command buffers and length-dependent graph retracing.

## 4. MNN and lower-level Metal references

MNN did not provide a verified complete `qwen4_exp` graph during research, but
its primary releases are relevant to kernel design:

- [3.4.1](https://github.com/alibaba/MNN/releases/tag/3.4.1): Metal Gated
  DeltaNet building blocks;
- [3.6.0](https://github.com/alibaba/MNN/releases/tag/3.6.0): Metal 2/3-bit
  GEMV comparisons;
- [3.6.1](https://github.com/alibaba/MNN/releases/tag/3.6.1): fused Q4/Q8
  dequant+GEMM and an M5-oriented tile variant.

These are kernel references only. They do not supply QSA, PLE, four-stream GR,
checkpoint mapping or Hebrus ownership/admission semantics. Benchmark their
ideas against existing Hebrus `metal/moe.metal` rather than importing a second
runtime abstraction.

Apple's [Metal I/O command queue](https://developer.apple.com/documentation/metal/mtliocommandqueue)
is a Tier-A PLE candidate for direct file-to-resource scheduling. The portable
contract remains batched `pread` into bounded `MTLStorageModeShared` rings;
capability absence or an MTLIO error must fall back or fail transactionally, not
change model output.

## 5. What Hebrus should copy, adapt and reject

| Upstream idea | Decision |
|---|---|
| llama hybrid index/main-KV slot ownership | copy semantics and test more deeply |
| llama exact uint64 hash metadata/host planning | copy mathematical contract |
| llama/MLX converter split/reorder rules | convert into pinned fixtures |
| llama lazy PLE mapping | use as functional control, replace scheduler |
| MLX external PLE separation/dedup | adapt to embedded isolated extent + page cache |
| MLX dense QSA mask | reject for production |
| MLX raw repooling on every call | replace with exact incremental pooled cache |
| MLX synchronous CPU PLE lookup | reject for decode hot path |
| MNN low-bit/Tensor-oriented tiles | benchmark as specialized candidates |
| upstream throughput numbers | record as context only; never claim for Hebrus |
| upstream generic model acceptance | reject; Hebrus remains exact-profile fail-closed |

## 6. Research conclusion for M5 Pro 64 GB

The combined evidence narrows the viable design:

```text
mixed very-low-bit routed backbone resident
+ sensitive dense/router/indexer roles at higher precision
+ GDN state FP32
+ PLE around Q4 in an isolated SSD extent with bounded async page cache
+ true compact sparse QSA and eventually quantized KV
+ fused GR/GDN/expert Metal paths selected by capabilities
+ MTP and vision absent from the base working set
```

No upstream engine currently supplies this complete Hebrus design. The value of
the research is identifying exact semantics, proven failure modes and useful
kernel/storage experiments while retaining Hebrus's stricter artifact,
transaction and performance contracts.

