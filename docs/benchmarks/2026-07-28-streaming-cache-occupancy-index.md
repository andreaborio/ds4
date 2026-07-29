# Streaming cache occupancy index

Date: 2026-07-28

Status: directional fifth-tranche candidate. Model-free Metal plus short Qwen
and DeepSeek exactness pass. The required context/model qualification matrices
remain pending.

Decision: retain this change after the four earlier routed-I/O and cache-scan
corrections on their branch. Do not promote the combined stack from this
record alone.

Related records:

- [streaming cache expert-width bound](2026-07-28-streaming-cache-expert-width-bound.md);
- [routed overlap scalar selected-ID reuse](2026-07-28-routed-overlap-scalar-id-reuse.md).

## Inefficiency and correction

The previous tranches skipped empty layers and bounded occupied rows to the
model's validated expert width. A Qwen decode eviction still inspected 9,026.4
array slots on average even though the cache held at most 789 valid entries.
Most remaining checks therefore read `valid == 0`.

Each layer now maintains six 64-bit occupancy words for the maximum 384-expert
geometry. Installation sets the matching bit and successful removal clears it.
Victim selection enumerates set bits in increasing expert order inside the
existing increasing layer order. Route hotness, recency comparisons, protected
entries, inflight handling, and tie behavior are unchanged.

Before using the sparse iterator, the runtime checks that:

- no bit lies outside the validated expert width;
- the bitmap population equals the existing layer entry count.

A mismatch takes the exact previous dense path. Full cache and layer cleanup
still traverse the canonical arrays and then clear the index, so teardown does
not rely on auxiliary state. The index is used by per-layer prune, reusable
buffer selection, batch reuse, memory-lock relief, and global budget prune.

## Qwen structural result

| Condition | Value |
| --- | --- |
| Host | Apple M5 Pro, 64 GiB unified memory, macOS 26.5.2 |
| Base revision | `hebrus/main` `572e6a6df07e` |
| Candidate | `codex/metal-ssd-attention-audit`, uncommitted combined five-tranche diff |
| Measured production binary SHA-256 | `b2abdda5b686f8322f39032d0046e86dfc6b1063f123796a083071f4cdfbacf9` |
| Model | published Qwen Q2_K_XL ExpertMajor v2, SHA-256 `30c22f70aff0f05986b517ee4ad8fef554a1b5aab6971c9ca09f999566d30143` |
| Runtime | forced SSD, cold preload, 1 GiB cache, strict full stack, 65,536-token allocation |
| Frontier | 128 prefill plus 128 greedy decode tokens |
| Host isolation | invalid for performance; animated wallpaper and desktop load remained active |

All comparable structural runs retained 18,483 reuse scans, 62,648 cache hits,
19,272 misses, 18,483 evictions, 57,816 successful storage reads, and
18.329071 GiB read from SSD.

| Resource | Original `hebrus/main` | Width-bound stack | Occupancy-index stack | Delta vs original | Delta vs previous |
| --- | ---: | ---: | ---: | ---: | ---: |
| Average entry checks per reuse scan | 30,720.0 | 9,026.4 | 671.2 | -97.82% | -92.56% |
| Profiled reuse-scan wall | 438.352 ms | 168.933 ms | 67.895 ms | -84.51% | -59.81% |

Entry checks are the deterministic resource result. The wall measurement is
diagnostic only because the host was not isolated.

Rows are chronological. There is no valid performance baseline or previous
comparable performance arm, so throughput deltas are `N/A`.

| Started | Model / lane | Prefill t/s | Decode t/s | TPOT p50 / p95 | Delta vs tested baseline | Delta vs previous comparable | SSD reads | Result |
| --- | --- | ---: | ---: | --- | --- | --- | ---: | --- |
| 2026-07-28T22:40:22+02:00 | Qwen, occupancy-index candidate | 107.38 | 34.48 | 28.527 / 33.050 ms | N/A; contaminated correctness lane | N/A; no comparable cohort | 18.329071 GiB | exact pass |
| 2026-07-28T22:40:53+02:00 | DeepSeek, occupancy-index candidate | 25.99 | N/A | N/A | N/A; contaminated correctness lane | N/A; no comparable cohort | 37.738037 GiB | exact pass |

Qwen frontier logits are byte-identical to the preceding combined stack,
SHA-256
`37fc78e2e0dc64dec160d4ea901348ffc6efb33cbdb53c805c24d11a42ffd8a7`.
Decode evidence is also byte-identical, SHA-256
`16d915d7e6fc1ef4c5f550a416e7b0b18ae347da9205d1f29d3c92d31d6615f1`.

DeepSeek frontier logits are byte-identical to the original control and all
preceding tranches, SHA-256
`2a1b42ec08d1657a319c124f0d721f5a1c132d2efa4d911ac7ca8f9e4d483471`.

## Validation

| Started | Revision / experiment | Test command or lane | Result |
| --- | --- | --- | --- |
| 2026-07-28T22:34:02+02:00 | occupancy-index candidate | warning-clean `make ds4_test` | pass |
| 2026-07-28T22:39:52+02:00 | occupancy-index candidate | model-free kernel suite on Apple Metal | pass |
| 2026-07-28T22:40:09+02:00 | occupancy-index candidate | warning-clean production Metal build | pass |
| 2026-07-28T22:40:22+02:00 | occupancy-index production binary | Qwen 128 prefill plus 128 decode, strict SSD stack | pass; exact outputs and -92.56% checks versus previous tranche |
| 2026-07-28T22:40:53+02:00 | occupancy-index production binary | DeepSeek 128 pure-prefill, required SSD overlap | pass; exact logits and identical cache/I/O counters |
| 2026-07-28T22:48:15+02:00 | final fifth-tranche source | all `premerge` gates except `context-audit`, plus `git diff --check` | pass; documentation, brand boundary, release contract, generated fixtures, build isolation, model-free Metal/CPU, install-layout, and diff gates |

The model-free regression covers 128/256/384 width bounds, population
mismatch, a partial final bitmap word, bits outside the model width, sparse
iteration across word boundaries, termination, and dense fallback.
`context-audit` is intentionally not reported as passing: the task is still
active and its required handoff remains under `docs/work/active/`. It will run
only after the model qualification work is complete and that handoff is
removed.

## Remaining promotion gates

- Stable-host Qwen and DeepSeek A/B/B/A at 128/2K/8K/32K, plus the second 32K
  routing/I/O prompt domain.
- DeepSeek 65,536/100K safety and Qwen near-262K endpoint coverage.
- Qualified GLM execution because the occupancy index is common cache code.
  The published `ds4-v0.2.0` object was remotely verified with its expected
  262,147,193,504-byte size and SHA-256
  `7f5017e3076e706c78f2a5322b035a9e2f6519c65ff5b6be8b2d91aeff61505d`,
  but it is not present on a mounted local filesystem.
- Final combined-stack comparison against the original `hebrus/main`, full
  merge-base diff review, active-handoff removal, and clean `premerge`.
