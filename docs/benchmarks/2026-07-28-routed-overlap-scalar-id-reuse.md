# Routed SSD-overlap scalar selected-ID reuse

Date: 2026-07-28

Status: directional fourth-tranche candidate. Model-free Metal, Qwen Q2_K_XL
scalar decode, and DeepSeek batch-prefill short exactness pass. The required
context/model qualification matrices remain pending.

Decision: retain this change after the selected-ID ownership and expert-cache
scan corrections on their branch. Do not promote the combined stack from this
record alone.

Related records:

- [initial selected-ID ownership reuse](2026-07-28-routed-overlap-selected-id-reuse.md);
- [streaming cache expert-width bound](2026-07-28-streaming-cache-expert-width-bound.md).

## Inefficiency and correction

Qwen SSD decode reads its eight selected expert IDs when the router becomes
ready so expert storage reads can overlap the shared-expert GPU work. The
scalar routed-MoE consumer then read the same immutable 32-byte selected tensor
again. The first ownership tranche removed its per-layer heap allocation but
left this duplicate tensor readback in place.

The pending overlap ticket now hands the synchronized IDs to both consumers:

- a batch/prefill route transfers its heap array exactly as before;
- a scalar top-8 route copies its fixed eight-element inline array into the
  consumer's stack before pending state is reset;
- a non-overlap fallback still performs its one required selected-tensor
  readback;
- count mismatch, undersized inline storage, or a missing output owner fails
  closed and reset retains responsibility for any pending heap allocation.

The bounded inline copy avoids exposing a pointer into reusable global pending
state. Replay still overwrites the consumer stack, so selected-ID trace
semantics are unchanged.

## Deterministic resource effect

The Qwen check generated 128 tokens through 40 routed layers, exercising 5,120
scalar route-ready calls.

| Resource | Previous tranche | Candidate | Delta |
| --- | ---: | ---: | ---: |
| Selected-tensor readbacks per scalar routed layer | 2 | 1 | -50.00% |
| Selected-tensor bytes read back per scalar routed layer | 64 | 32 | -50.00% |
| Duplicate selected-tensor readbacks over the run | 5,120 | 0 | -100.00% |
| Scalar selected-ID heap allocations | 0 | 0 | 0.00% |

The consumer still performs one bounded 32-byte host-to-host copy from pending
inline storage to its stack. The result removes repeated tensor access; it is
not a throughput claim.

## Runtime evidence

| Condition | Value |
| --- | --- |
| Host | Apple M5 Pro, 64 GiB unified memory, macOS 26.5.2 |
| Base revision | `hebrus/main` `572e6a6df07e` |
| Candidate | `codex/metal-ssd-attention-audit`, uncommitted combined four-tranche diff |
| Final production benchmark SHA-256 | `ff223b93854b04de0ec21b5c42a952dd37196c702de8b610a7702badaae3f3ba` |
| Qwen model | published Q2_K_XL ExpertMajor v2, 12,290,632,032 bytes, SHA-256 `30c22f70aff0f05986b517ee4ad8fef554a1b5aab6971c9ca09f999566d30143` |
| DeepSeek model | V4 Flash ExpertMajor v2, 86,720,114,272 bytes, SHA-256 `8378080263eb9224f7228d72e2afa4ac3cf74a116023fdec2c596ff228a33e3f` |
| Qwen runtime | forced SSD, cold preload, 1 GiB cache, strict full stack, 65,536-token allocation |
| DeepSeek runtime | forced SSD, cold preload, 2 GiB cache, required asynchronous prefill overlap, 129-token allocation |
| Raw evidence | local scratch CSV, frontier logits, decode evidence, and runtime telemetry; intentionally not committed |

The host was not isolated from desktop and system work, and no A/B/B/A cohort
was run for this refinement. Timing deltas are therefore `N/A`. The rows are
chronological and report throughput only to identify the exercised run.

| Started | Model / lane | Prefill t/s | Decode t/s | TPOT p50 / p95 | Delta vs tested baseline | Delta vs previous comparable | SSD reads | Result |
| --- | --- | ---: | ---: | --- | --- | --- | ---: | --- |
| 2026-07-28T22:08:31+02:00 | Qwen, candidate before final fail-closed audit | 105.66 | 35.25 | 28.007 / 32.478 ms | N/A; correctness-only arm | N/A; no comparable cohort | 18.329071 GiB | exact pass |
| 2026-07-28T22:10:14+02:00 | DeepSeek, candidate before final fail-closed audit | 26.53 | N/A | N/A | N/A; correctness-only arm | N/A; no comparable cohort | 37.738037 GiB | exact pass |
| 2026-07-28T22:11:43+02:00 | Qwen, final production binary | 107.51 | 34.98 | 28.225 / 32.471 ms | N/A; correctness-only arm | N/A; no comparable cohort | 18.329071 GiB | exact pass |
| 2026-07-28T22:12:10+02:00 | DeepSeek, final production binary | 26.47 | N/A | N/A | N/A; correctness-only arm | N/A; no comparable cohort | 37.738037 GiB | exact pass |

Both final Qwen documents are byte-identical to the previous combined-stack
reference: frontier logits SHA-256
`37fc78e2e0dc64dec160d4ea901348ffc6efb33cbdb53c805c24d11a42ffd8a7`
and decode evidence SHA-256
`16d915d7e6fc1ef4c5f550a416e7b0b18ae347da9205d1f29d3c92d31d6615f1`.
The final run retained 5,120 selected calls, 62,648 cache hits, 19,272 misses,
18,483 evictions, 57,816 successful storage reads, and 18.329071 GiB read.

DeepSeek final frontier logits are byte-identical to the original control and
all earlier tranches, SHA-256
`2a1b42ec08d1657a319c124f0d721f5a1c132d2efa4d911ac7ca8f9e4d483471`.
The run retained 27,299 hits, 5,725 misses, 5,422 evictions, 17,175 successful
storage calls, and 37.738037 GiB read.

## Validation

| Started | Revision / experiment | Test command or lane | Result |
| --- | --- | --- | --- |
| 2026-07-28T22:07:24+02:00 | scalar selected-ID reuse candidate | warning-clean `make ds4_test` | pass |
| 2026-07-28T22:07:33+02:00 | scalar selected-ID reuse candidate | model-free kernel suite on Apple Metal | pass |
| 2026-07-28T22:08:15+02:00 | scalar selected-ID reuse candidate | warning-clean production Metal build | pass |
| 2026-07-28T22:10:14+02:00 | scalar selected-ID reuse candidate | DeepSeek 128 pure-prefill, required SSD overlap | pass; exact logits |
| 2026-07-28T22:11:04+02:00 | final fail-closed ownership candidate | warning-clean test build plus model-free kernel suite on Apple Metal | pass |
| 2026-07-28T22:11:26+02:00 | final fail-closed ownership candidate | warning-clean production Metal build | pass |
| 2026-07-28T22:11:43+02:00 | final production binary | Qwen 128 prefill plus 128 decode, strict SSD stack | pass; exact frontier and decode evidence |
| 2026-07-28T22:12:10+02:00 | final production binary | DeepSeek 128 pure-prefill, required SSD overlap | pass; exact logits |
| 2026-07-28T22:13:41+02:00 | final combined candidate | `make premerge` | pass; repository, docs, build isolation, model-free Metal/CPU, install layout, and diff gates |

The ownership regression covers heap transfer, exact count validation, inline
copy and capacity validation, reset cleanup, and the missing-output-owner
failure. The exercised Qwen and DeepSeek outputs show that selected experts,
route order, expert weights, and model arithmetic did not change.

## Remaining promotion gates

- Stable-host Qwen and DeepSeek A/B/B/A at 128/2K/8K/32K, plus the second 32K
  routing/I/O prompt domain.
- DeepSeek 65,536/100K safety and Qwen near-262K endpoint coverage.
- Qualified GLM execution because the combined branch modifies common expert
  cache code. The published `ds4-v0.2.0` object was remotely verified with its
  expected 262,147,193,504-byte size and SHA-256
  `7f5017e3076e706c78f2a5322b035a9e2f6519c65ff5b6be8b2d91aeff61505d`,
  but it is not present on a mounted local filesystem and the internal volume
  has only about 51 GiB available.
- Final combined-stack comparison against the original `hebrus/main`, full
  merge-base diff review, residue audit, and clean `premerge`.
