# Streaming cache occupied-layer expert-width bound

Date: 2026-07-28

Status: directional third-tranche candidate. Model-free Metal plus short Qwen
and DeepSeek exactness pass. The required context/model qualification matrices
remain pending.

Decision: retain this change after the selected-ID ownership and empty-layer
skip changes on their branch. Do not promote the combined stack from this
record alone.

A later tranche replaces traversal of every slot inside this bound with a
validated occupancy index; see
[streaming cache occupancy index](2026-07-28-streaming-cache-occupancy-index.md).

Related records:

- [routed SSD-overlap selected-ID reuse](2026-07-28-routed-overlap-selected-id-reuse.md);
- [streaming cache empty-layer skip](2026-07-28-streaming-cache-empty-layer-skip.md).

## Inefficiency and correction

The cache reserves its maximum supported geometry, 80 layers by 384 experts.
After empty layers were skipped, all four global victim scans still traversed
384 slots in every occupied layer. The qualified Qwen3.6 and DeepSeek V4 Flash
artifacts each expose 256 experts, so one third of each occupied row was known
to be outside the model.

Each occupied layer now records the model's already validated
`n_total_expert`. The value is published only after the first successful cache
installation, every later installation in that layer must present the same
value, and clearing the last entry resets it. A missing, zero, oversized, or
otherwise out-of-range scan value falls back to all 384 slots. A later
installation with a different validated width fails closed.

Reusable-buffer, batch-reuse, memory-lock relief, and global budget-prune scans
use the recorded width. Expert iteration and LFU/LRU comparison order within
that width are unchanged. Qwen and DeepSeek therefore scan 256 slots per
occupied layer; a 384-expert GLM layer receives no width reduction.

## Qwen structural result

| Condition | Value |
| --- | --- |
| Host | Apple M5 Pro, 64 GiB unified memory, macOS 26.5.2 |
| Base revision | `hebrus/main` `572e6a6df07e` |
| Immediate comparison | selected-ID ownership plus empty-layer skip |
| Final candidate binary SHA-256 | `ee5900aa7b5c0b601b1eaaf0b302dda2d150cdd683fd462acf0a93d23f5f6ea3` |
| Model | published Qwen Q2_K_XL ExpertMajor v2, SHA-256 `30c22f70aff0f05986b517ee4ad8fef554a1b5aab6971c9ca09f999566d30143` |
| Model geometry | 40 routed layers by 256 experts |
| Runtime | forced SSD, cold preload, 1 GiB cache, strict full stack, 65,536-token allocation |
| Frontier | 128 prefill plus 128 greedy decode tokens |
| Host isolation | invalid for performance; material non-inference system and desktop load remained active |

Every complete arm retained 18,483 reuse scans, 62,648 cache hits, 19,272
misses, 18,483 evictions, 57,816 successful storage reads, and 18.329071 GiB
read from SSD.

| Resource | Original `hebrus/main` | Empty-layer stack | Width-bound stack | Delta vs original | Delta vs previous |
| --- | ---: | ---: | ---: | ---: | ---: |
| Average entry checks per reuse scan | 30,720.0 | 13,539.5 | 9,026.4 | -70.62% | -33.33% |
| Profiled reuse-scan wall | 438.352 ms | 218.398 ms | 172.343 ms | -60.68% | -21.09% |

The entry-check deltas are the bounded resource result. Scan wall is diagnostic
support only because the host was not isolated.

Rows are chronological. There is no valid performance baseline or previous
comparable performance arm, so throughput deltas are `N/A`.

| Started | Arm | Prefill t/s | Decode t/s | TPOT p50 / p95 | Delta vs tested baseline | Delta vs previous comparable | Result |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| 2026-07-28T21:51:26+02:00 | width-bound candidate, missing evidence directory | N/A | N/A | N/A | N/A; incomplete arm | N/A; incomplete arm | invalid; stopped after prefill before evidence publication |
| 2026-07-28T21:51:52+02:00 | width-bound candidate, canonical executable | 270.01 | 35.39 | 28.105 / 32.015 ms | N/A; contaminated diagnostic | N/A; no comparable cohort | exact numeric logits and byte-identical decode evidence |
| 2026-07-28T21:52:56+02:00 | width-bound candidate, compatibility alias matching control identity | 275.04 | 38.63 | 25.440 / 28.255 ms | N/A; contaminated diagnostic | N/A; no comparable cohort | complete evidence byte-identical to immediate control |
| 2026-07-28T21:57:50+02:00 | final binary, compatibility alias matching control identity | 104.04 | 34.63 | 28.680 / 32.270 ms | N/A; heavily contaminated diagnostic | N/A; no comparable cohort | complete evidence byte-identical to immediate control |

The first complete arm differed from the control JSON only in the descriptive
executable-source field; all 248,320 logits and every other field were exactly
equal. The repeated arm used the same compatibility alias as the control and
made the entire frontier document byte-identical, SHA-256
`37fc78e2e0dc64dec160d4ea901348ffc6efb33cbdb53c805c24d11a42ffd8a7`.
Its decode evidence is also byte-identical, SHA-256
`16d915d7e6fc1ef4c5f550a416e7b0b18ae347da9205d1f29d3c92d31d6615f1`.

## DeepSeek short correctness

The third-tranche source also ran on the qualified 86,720,114,272-byte
DeepSeek V4 Flash artifact, SHA-256
`8378080263eb9224f7228d72e2afa4ac3cf74a116023fdec2c596ff228a33e3f`.
The lane used forced SSD, cold preload, a 2 GiB cache, required asynchronous
prefill overlap, a 129-token allocation, and 128 pure-prefill tokens.

| Started | Arm | Prefill t/s | Decode t/s | TPOT p50 / p95 | Delta vs tested baseline | Delta vs previous comparable | SSD reads | Result |
| --- | --- | ---: | ---: | --- | --- | --- | ---: | --- |
| 2026-07-28T21:54:32+02:00 | width-bound candidate before private-symbol cleanup | 26.46 | N/A | N/A | N/A; contaminated single arm | N/A; no comparable final-stack arm | 37.738037 GiB | exact pass |
| 2026-07-28T21:58:11+02:00 | final width-bound combined candidate | 26.43 | N/A | N/A | N/A; contaminated single arm | N/A; no comparable final-stack arm | 37.738037 GiB | exact pass |

The candidate retained 27,299 hits, 5,725 misses, 5,422 evictions, 17,175
successful storage calls, and the same SSD byte count. Frontier logits are
byte-identical to the immediate candidate and original control, SHA-256
`2a1b42ec08d1657a319c124f0d721f5a1c132d2efa4d911ac7ca8f9e4d483471`.

## Validation

| Started | Revision / experiment | Test command or lane | Result |
| --- | --- | --- | --- |
| 2026-07-28T21:50:06+02:00 | width-bound candidate | warning-clean production and test compilation | pass |
| 2026-07-28T21:50:21+02:00 | width-bound candidate | model-free kernel suite on Apple Metal | pass |
| 2026-07-28T21:52:56+02:00 | width-bound candidate | Qwen 128 prefill + 128 decode, strict SSD stack | pass; exact outputs and -33.33% entry checks versus previous tranche |
| 2026-07-28T21:54:32+02:00 | width-bound candidate | DeepSeek 128 pure-prefill, required SSD overlap | pass; exact logits and identical cache/I/O counters |
| 2026-07-28T21:57:21+02:00 | final candidate after private-symbol cleanup | warning-clean test compilation | pass |
| 2026-07-28T21:57:30+02:00 | final candidate after private-symbol cleanup | model-free kernel suite on Apple Metal | pass |
| 2026-07-28T21:57:50+02:00 | final candidate binary | Qwen 128 prefill + 128 decode, strict SSD stack | pass; exact outputs and identical cache/I/O counters |
| 2026-07-28T21:58:11+02:00 | final candidate binary | DeepSeek 128 pure-prefill, required SSD overlap | pass; exact logits and identical cache/I/O counters |
| 2026-07-28T21:59:10+02:00 | final combined candidate | `make premerge` | pass; repository, documentation, build-isolation, model-free Metal/CPU, install-layout, and diff gates |

The model-free regression covers 128-, 256-, and 384-expert bounds, fail-open
handling for zero and oversized values, acceptance of an empty or matching
occupied layer, and rejection of an inconsistent occupied-layer width.

## Remaining promotion gates

- Stable-host Qwen and DeepSeek A/B/B/A at 128/2K/8K/32K, plus the second 32K
  routing/I/O prompt domain.
- DeepSeek 65,536/100K safety and Qwen near-262K endpoint coverage.
- Qualified GLM execution because the width bookkeeping and scans are common
  cache code. The published `ds4-v0.2.0` object was remotely verified on
  2026-07-28 with its expected 262,147,193,504-byte size and SHA-256
  `7f5017e3076e706c78f2a5322b035a9e2f6519c65ff5b6be8b2d91aeff61505d`,
  but it is not present on a mounted local filesystem and the internal volume
  has only about 51 GiB available.
- Final combined-stack comparison against the original `hebrus/main`, full
  merge-base diff review, residue audit, and clean `premerge`.
