# Streaming cache empty-layer scan skip

Date: 2026-07-28

Status: directional combined-stack candidate. Qwen and DeepSeek short
correctness plus the model-free Metal gate pass; the required context/model
qualification matrices remain pending.

Decision: retain the empty-layer skip with the selected-ID ownership changes on
their branch. Do not promote either optimization from this record alone.

Related records:

- [routed SSD-overlap selected-ID reuse](2026-07-28-routed-overlap-selected-id-reuse.md);
- [occupied-layer expert-width bound](2026-07-28-streaming-cache-expert-width-bound.md).

## Inefficiency and correction

Every global reusable-buffer eviction scanned the cache's compile-time maximum
geometry: 80 layers by 384 experts, or 30,720 entry checks. Qwen3.6 has 40
routed layers; DeepSeek V4 Flash has 43. The scan therefore visited 37 to 40
empty layer rows on every cache miss after reaching the byte budget.

The cache already maintains `g_stream_expert_cache_layer_count` at the only
entry installation and clear points, under the single Metal encoder owner.
Global reusable-buffer, batch-reuse, memory-lock relief, and budget-prune scans
now skip a layer only when this count is zero. Candidate ordering within every
occupied layer and across every real LFU/LRU eviction candidate is unchanged.

This is deliberately narrower than adding a heap or mutable eviction index.
It removes known-empty rows without creating synchronization, invalidation, or
tie-order state. It does not remove the remaining linear search through the
occupied model layers.

## Qwen structural and diagnostic result

| Condition | Value |
| --- | --- |
| Host | Apple M5 Pro, 64 GiB unified memory, macOS 26.5.2 |
| Base revision | `hebrus/main` `572e6a6df07e` |
| Immediate comparison | selected-ID ownership/inline candidate before the empty-layer skip |
| Final combined candidate binary SHA-256 | `1d2bfe37dd7e4565f7e38b12fffbbd88b069716be0490d2f92faf90e036d4e6a` |
| Model | published Qwen Q2_K_XL ExpertMajor v2, SHA-256 `30c22f70aff0f05986b517ee4ad8fef554a1b5aab6971c9ca09f999566d30143` |
| Prompt | `speed-bench/promessi_sposi.txt`, SHA-256 `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f` |
| Runtime | forced SSD, cold preload, 1 GiB cache, strict full stack, 65,536-token allocation |
| Frontier | 128 prefill plus 128 greedy decode tokens |
| Host isolation | invalid for performance; Wallpaper Aerials about 55% CPU, WindowServer about 50%, Codex service about 19%, and a VideoToolbox decoder about 14% |

The compared runs had the same 18,483 reuse scans, 62,648 cache hits, 19,272
misses, 18,483 evictions, 57,816 successful `pread` calls, and 18.329071 GiB of
SSD reads.

| Resource | Immediate control | Candidate | Delta |
| --- | ---: | ---: | ---: |
| Average entry checks per reuse scan | 30,720.0 | 13,539.5 | -55.93% |
| Profiled reuse-scan wall | 438.352 ms | 218.398 ms | -50.18% |
| Reuse-clear wall | 3.214 ms | 3.333 ms | +3.70% |

The entry-check reduction is the bounded resource result. The timing is
diagnostic support: background load invalidates end-to-end performance claims.

Rows are chronological. Since the host was contaminated and the candidate has
only one final-stack arm, there is no tested performance baseline or previous
comparable arm; those deltas are `N/A`.

| Started | Arm | Prefill t/s | Decode t/s | TPOT p50 / p95 | Delta vs tested baseline | Delta vs previous comparable | Result |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| 2026-07-28T21:30:31+02:00 | immediate control, selected-ID stack | 248.26 | 34.85 | 28.260 / 32.238 ms | N/A; contaminated diagnostic | N/A; no comparable retained cohort | exact pass |
| 2026-07-28T21:37:58+02:00 | combined candidate | 266.97 | 36.87 | 26.751 / 29.762 ms | N/A; contaminated diagnostic | N/A; no comparable retained cohort | exact pass |

The apparent +5.80 percent decode throughput is not a speed claim. The complete
frontier-logit JSON is byte-identical to `hebrus/main`, SHA-256
`37fc78e2e0dc64dec160d4ea901348ffc6efb33cbdb53c805c24d11a42ffd8a7`.
The complete post-decode token/final-logit JSON is also byte-identical,
SHA-256
`16d915d7e6fc1ef4c5f550a416e7b0b18ae347da9205d1f29d3c92d31d6615f1`.

## DeepSeek short correctness

The final combined source was also exercised on the qualified 86,720,114,272
byte DeepSeek V4 Flash artifact, SHA-256
`8378080263eb9224f7228d72e2afa4ac3cf74a116023fdec2c596ff228a33e3f`.
The lane used forced SSD, cold preload, a 2 GiB cache, required asynchronous
prefill overlap, a 129-token allocation, and 128 pure-prefill tokens.

| Started | Arm | Prefill t/s | Decode t/s | TPOT p50 / p95 | Delta vs tested baseline | Delta vs previous comparable | SSD reads | Result |
| --- | --- | ---: | ---: | --- | --- | --- | ---: | --- |
| 2026-07-28T21:39:58+02:00 | combined candidate correctness arm | 26.17 | N/A | N/A | N/A; contaminated single arm | N/A; no comparable final-stack arm | 37.738037 GiB | exact pass |

The candidate retained the control's 27,299 hits, 5,725 misses, 5,422
evictions, 17,175 `pread` calls, and SSD bytes. Its frontier logits are
byte-identical to `hebrus/main`, SHA-256
`2a1b42ec08d1657a319c124f0d721f5a1c132d2efa4d911ac7ca8f9e4d483471`.

## Validation

| Started | Revision / experiment | Test command or lane | Result |
| --- | --- | --- | --- |
| 2026-07-28T21:37:33+02:00 | combined candidate | `make ds4_test`; `ds4_test --metal-kernels` on Apple Metal | pass |
| 2026-07-28T21:37:58+02:00 | combined candidate | Qwen 128 prefill + 128 decode, strict SSD stack | pass; exact outputs and -55.93% cache entry checks |
| 2026-07-28T21:39:58+02:00 | combined candidate | DeepSeek 128 pure-prefill, required SSD overlap | pass; exact logits and identical cache/I/O counters |
| 2026-07-28T21:42:36+02:00 | combined candidate | `make premerge` | pass; repository, documentation, build-isolation, model-free Metal/CPU, install-layout, and diff gates |

The model-free cache-pressure cases exercise occupied-layer reuse and preserve
their cold/warm/duplicate, growth-guard, memory-lock, floor, and resident replay
expectations.

## Remaining promotion gates

- Stable-host Qwen and DeepSeek A/B/B/A at 128/2K/8K/32K, plus the second 32K
  routing/I/O prompt domain.
- DeepSeek 65,536/100K safety and Qwen near-262K endpoint coverage.
- Qualified GLM model execution because the empty-layer skip is common cache
  code. No qualified GLM GGUF is present on the attached local filesystems.
- Final combined-stack comparison against the original `hebrus/main`, full
  merge-base diff review, and residue audit.
