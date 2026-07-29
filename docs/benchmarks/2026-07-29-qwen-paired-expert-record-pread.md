# Qwen Metal/SSD cumulative optimization stack

Date: 2026-07-29

Status: promotion candidate. The final flag-free source passes Qwen resident
and SSD qualification, including both 32K prompt domains, 65K/100K safety
lanes, and the 262,015-token model-window endpoint. DeepSeek and GLM preserve
their qualified behavior and exact output. Repository premerge and integration
review remain the only pending gates at the time this record was written.

Decision: promote the eight changes below as a Qwen3.6-only fast path. Keep
DeepSeek on three component reads and its original dense cache traversal.
Keep GLM on its qualified one-read full-record path and original cache
behavior. Remove all experiment switches: accepted behavior is the default,
and rejected behavior has no dormant release branch.

Supersedes: the temporary
`DS4_METAL_ENABLE_PAIRED_EXPERT_RECORD_PREAD` and
`DS4_METAL_STREAM_EVICT_LAYER_STALENESS` experiments. This record is the
final baseline-versus-combined-stack decision for the seven dated 2026-07-28
records indexed beside it.

## Intent, mechanism, effect, and risk

| Rank | Change | What the code does | Expected effect | Important risk and containment |
| ---: | --- | --- | --- | --- |
| 1 | Paired Qwen record reads | Reads contiguous gate+up data with one `pread` and down data with a second | One third fewer Qwen SSD syscalls | Larger DeepSeek records regress; the classifier requires validated Qwen 40-layer/256-expert ExpertMajor v2 geometry |
| 2 | Qwen forward-layer eviction | Among equally hot cross-layer victims, evicts the layer furthest from reuse in the forward decode sweep | 351 fewer decode loads/evictions, 1,053 fewer reads, and 0.337 GiB less SSD traffic in the isolated tranche | A wrong model transfer changes cache behavior; DeepSeek and GLM are explicitly excluded |
| 3 | Sparse occupancy traversal | Scans a validated per-layer bitmap in original expert order | Average victim checks fall from 30,720.0 to 671.2 | Bitmap drift could select a different victim; population validation fails open to the original dense traversal |
| 4 | Scalar selected-ID reuse | Preserves Qwen's first top-8 router readback through scalar overlap ownership | Duplicate tensor readbacks fall from 5,120 to zero | Lifetime errors could corrupt routing; inline/heap ownership and cleanup have a model-free regression |
| 5 | Batch selected-ID ownership reuse | Transfers the already synchronized selected-ID allocation into the pending overlap object | Removes duplicate allocation/readback | Ownership must survive asynchronous overlap; the regression exercises both transfer and cleanup |
| 6 | Dense synthetic-ID publication reuse | Initializes Qwen's constant dense ID buffer only on allocation or growth | Host zero/fill/`didModifyRange:` publications fall from 32,250 to 1 | Reallocation must republish; a 128-to-512 growth test covers it |
| 7 | Validated expert-width bound | Stops cache scans at the installed model width | Avoids scanning compile-time capacity beyond 256 Qwen experts | Missing/corrupt metadata fails open to the old maximum |
| 8 | Empty-layer skip | Avoids victim traversal for a layer whose validated count is zero | Removes provably empty work | Count inconsistency could hide entries; Qwen-only occupancy validation and dense fallback preserve correctness |

For example, a missing Qwen expert previously submitted gate, up, and down as
three storage operations. Gate and up are adjacent both in the ExpertMajor v2
file and in the destination slab, so the final plan fills them with one
contiguous operation and reads down separately. The comparison stops at
record geometry: DeepSeek's larger requests lose parallelism when joined, so
the same grouping is not transferable.

## Experiment identity

| Condition | Value |
| --- | --- |
| Host | Apple M5 Pro, 64 GiB unified memory, macOS build `25F84`, AC power |
| Tested `main` | clean detached `572e6a6df07ef09bff4bd9d4ef54ffbfbad43aad`; binary SHA-256 `87d19070b4a4c07561604f27fb63e5ab63ef051686e4f2fab6a20ca23dfecebd` |
| Final measured stack | same parent plus the eight changes; measured diff SHA-256 `2ab7b9d85eebecbe2d9f852de29a5ee11c1d85ea6148a2a8662c9792e8c73ffd`; source-state SHA-256 `99cb34eeed6c35628befb7e2107ce12662adf4c218e2e199d98d63001aa0dfda`; binary SHA-256 `63af27e4e8705d265c46845ed3579c042cfd0d7a6df8c768d72acf1aa6c5b056` |
| Qwen artifact | Qwen3.6 35B A3B ExpertMajor v2 Q2_K_XL; 12,290,632,032 bytes; SHA-256 `30c22f70aff0f05986b517ee4ad8fef554a1b5aab6971c9ca09f999566d30143` |
| DeepSeek artifact | DeepSeek V4 Flash ExpertMajor v2; 86,720,114,272 bytes; SHA-256 `8378080263eb9224f7228d72e2afa4ac3cf74a116023fdec2c596ff228a33e3f` |
| GLM artifact | GLM 5.2 ExpertMajor v2 Q2_K; 262,147,193,504 bytes; SHA-256 `7f5017e3076e706c78f2a5322b035a9e2f6519c65ff5b6be8b2d91aeff61505d` |
| Primary prompt | `promessi_sposi.txt`; SHA-256 `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f` |
| Second 32K prompt | `tests/long_context_security_prompt.txt`; SHA-256 `e7c1a2cadf781d274cc26bd251d532fe1b9e632080da97e3eb4684741e7cc308` |
| Qwen SSD plan | forced SSD; 789-record application cache; no preload; warm OS page cache; 128 greedy decode tokens |
| Qwen resident plan | forced resident; production 4,096 preload argument; warm OS page cache; 128 greedy decode tokens |
| Ordering | separate-process A/B/B/A; A is tested `main`, B is the final stack |
| Isolation | no competing inference process; wallpaper PID 562 and video decoder PID 650 suspended; every retained arm has zero new swapout |
| Raw evidence | `/private/tmp/hebrus-qwen-gated-final-*`, `/private/tmp/hebrus-deepseek-qwen-only-gated-128-check*`, and `/private/tmp/hebrus-glm-gold-*` |

The Qwen application cache starts empty in every SSD process. macOS does not
offer a safe page-cache flush, so discarded warm-ups establish cache state and
crossed controls reject drift. Sustained 32K and resident 8K arms use four
minutes of idle cooldown because back-to-back runs showed large cliffs without
a macOS thermal warning.

## Final Qwen SSD crossed cohorts

### 2K + 128

| Started (Europe/Rome) | Arm | Prefill t/s | Decode t/s | TPOT p50 / p95 | Delta vs tested `main` mean | Delta vs previous comparable | Result |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| 2026-07-29T02:46:25+02:00 | A1, `main` | 791.45 | 35.44 | 27.121 / 34.909 ms | -0.23% / -0.92% | N/A; first retained arm | exact pass |
| 2026-07-29T02:46:48+02:00 | B1, final stack | 850.59 | 39.93 | 24.192 / 29.093 ms | +7.23% / +11.63% | +7.47% / +12.67% | exact pass |
| 2026-07-29T02:47:10+02:00 | B2, final stack | 853.89 | 39.70 | 24.425 / 28.884 ms | +7.65% / +10.99% | +0.39% / -0.58% | exact pass |
| 2026-07-29T02:47:30+02:00 | A2, `main` | 795.02 | 36.10 | 26.700 / 33.801 ms | +0.23% / +0.92% | -6.89% / -9.07% | exact pass |

| 2K mean | Tested `main` | Final stack | Delta | Time saved |
| --- | ---: | ---: | ---: | ---: |
| Prefill | 793.24 t/s | 852.24 t/s | +7.44% | 178.77 ms |
| Decode | 35.77 t/s | 39.82 t/s | +11.31% | 364.17 ms |
| TPOT p50 | 26.911 ms | 24.309 ms | -9.67% | 2.602 ms/token |
| TPOT p95 | 34.355 ms | 28.989 ms | -15.62% | 5.367 ms/token |
| Prefill plus decode | 6,160.72 ms | 5,617.78 ms | -8.81% | 542.93 ms |
| Total `pread` calls | 42,822 | 28,690 | -33.00% | 14,132 calls |
| Cumulative `pread` time | 511.55 ms | 471.23 ms | -7.88% | 40.32 ms |

Control spread is 0.45% prefill and 1.86% decode; candidate spread is 0.39%
and 0.58%.

### 8K + 128

| Started (Europe/Rome) | Arm | Prefill t/s | Decode t/s | TPOT p50 / p95 | Delta vs tested `main` mean | Delta vs previous comparable | Result |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| 2026-07-29T02:53:52+02:00 | A1, `main` | 789.52 | 31.92 | 30.407 / 36.128 ms | +0.06% / -0.34% | N/A; first retained arm | exact pass |
| 2026-07-29T02:54:23+02:00 | B1, final stack | 803.91 | 35.33 | 27.576 / 31.275 ms | +1.88% / +10.30% | +1.82% / +10.68% | exact pass |
| 2026-07-29T02:54:55+02:00 | B2, final stack | 804.82 | 35.23 | 27.595 / 31.303 ms | +1.99% / +9.99% | +0.11% / -0.28% | exact pass |
| 2026-07-29T02:55:29+02:00 | A2, `main` | 788.65 | 32.14 | 30.204 / 36.381 ms | -0.06% / +0.34% | -2.01% / -8.77% | exact pass |

| 8K mean | Tested `main` | Final stack | Delta | Time saved |
| --- | ---: | ---: | ---: | ---: |
| Prefill | 789.09 t/s | 804.37 t/s | +1.94% | 197.29 ms |
| Decode | 32.03 t/s | 35.28 t/s | +10.15% | 367.86 ms |
| TPOT p50 | 30.306 ms | 27.586 ms | -8.97% | 2.720 ms/token |
| TPOT p95 | 36.255 ms | 31.289 ms | -13.70% | 4.966 ms/token |
| Prefill plus decode | 14,377.64 ms | 13,812.49 ms | -3.93% | 565.15 ms |
| Total `pread` calls | 79,428 | 53,016 | -33.25% | 26,412 calls |
| Cumulative `pread` time | 782.10 ms | 747.55 ms | -4.42% | 34.54 ms |

Control spread is 0.11% prefill and 0.69% decode; candidate spread is 0.11%
and 0.28%. An earlier closing control swapped eight pages; that entire cohort
is invalid and excluded.

### 32K + 128, primary prompt

| Started (Europe/Rome) | Arm | Prefill t/s | Decode t/s | TPOT p50 / p95 | Delta vs tested `main` mean | Delta vs previous comparable | Result |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| 2026-07-29T03:26:13+02:00 | A1, `main` | 540.96 | 25.93 | 37.556 / 44.723 ms | -0.13% / 0.00% | N/A; first retained arm | exact pass |
| 2026-07-29T03:31:30+02:00 | B1, final stack | 544.63 | 27.78 | 35.295 / 39.544 ms | +0.55% / +7.13% | +0.68% / +7.13% | exact pass |
| 2026-07-29T03:36:53+02:00 | B2, final stack | 542.67 | 27.82 | 35.098 / 39.657 ms | +0.19% / +7.29% | -0.36% / +0.14% | exact pass |
| 2026-07-29T03:42:15+02:00 | A2, `main` | 542.33 | 25.93 | 37.663 / 44.713 ms | +0.13% / 0.00% | -0.06% / -6.79% | exact pass |

| 32K mean | Tested `main` | Final stack | Delta | Time saved |
| --- | ---: | ---: | ---: | ---: |
| Prefill | 541.65 t/s | 543.65 t/s | +0.37% | 223.46 ms |
| Decode | 25.93 t/s | 27.80 t/s | +7.21% | 331.26 ms |
| TPOT p50 | 37.610 ms | 35.197 ms | -6.42% | 2.413 ms/token |
| TPOT p95 | 44.718 ms | 39.601 ms | -11.44% | 5.118 ms/token |
| Prefill plus decode | 65,433.08 ms | 64,878.35 ms | -0.85% | 554.72 ms |
| Total `pread` calls | 72,444 | 48,326 | -33.29% | 24,118 calls |
| Cumulative `pread` time | 751.51 ms | 724.90 ms | -3.54% | 26.61 ms |

Control spread is 0.25% prefill and 0.00% decode; candidate spread is 0.36%
and 0.14%. An earlier cohort is invalid because its closing main control
drifted 4.20% in decode; rendering benchmark output was removed from the
measurement path and the complete crossed cohort was restarted.

## Qwen resident crossed cohorts

| Mode/frontier | Tested `main` prefill / decode | Final stack prefill / decode | TPOT p50 / p95, main -> stack | End-to-end delta | Control / candidate spread | Resource decision |
| --- | --- | --- | --- | ---: | --- | --- |
| Resident 2K + 128 | 894.03 / 72.49 t/s | 891.44 / 72.44 t/s | 13.633 / 14.357 -> 13.644 / 14.338 ms | +7.92 ms / +0.20% | 0.09% / 0.08%; 0.23% / 0.47% | throughput neutral; keep 32,250 -> 1 publication reduction |
| Resident 8K + 128 | 865.06 / 66.05 t/s | 864.62 / 65.96 t/s | 15.095 / 15.420 -> 15.113 / 15.423 ms | +7.64 ms / +0.07% | 0.23% / 0.06%; 0.31% / 0.08% | throughput neutral; keep 32,250 -> 1 publication reduction |

| Started (Europe/Rome) | Resident arm | Delta vs tested `main` mean | Delta vs previous comparable | Result |
| --- | --- | --- | --- | --- |
| 2026-07-29T05:40:54+02:00 | 2K A1, `main` | -0.05% prefill / +0.04% decode | N/A; first retained arm | exact pass |
| 2026-07-29T05:44:13+02:00 | 2K B1, final stack | -0.41% / +0.17% | -0.36% / +0.12% | exact pass |
| 2026-07-29T05:44:37+02:00 | 2K B2, final stack | -0.17% / -0.30% | +0.23% / -0.47% | exact pass |
| 2026-07-29T05:44:59+02:00 | 2K A2, `main` | +0.05% / -0.04% | +0.22% / +0.26% | exact pass |
| 2026-07-29T05:53:20+02:00 | 8K A1, `main` | +0.12% / -0.03% | N/A; first retained 8K arm | exact pass |
| 2026-07-29T05:57:45+02:00 | 8K B1, final stack | +0.10% / -0.18% | -0.01% / -0.15% | exact pass |
| 2026-07-29T06:02:19+02:00 | 8K B2, final stack | -0.20% / -0.11% | -0.31% / +0.08% | exact pass |
| 2026-07-29T06:06:45+02:00 | 8K A2, `main` | -0.12% / +0.03% | +0.09% / +0.14% | exact pass |

Two uncooled resident 8K cohorts are discarded: the first had a 5.2% main
decode warm-up drift, and the second hit a 47% thermal prefill cliff on B2.
macOS emitted no thermal warning. Four-minute spacing made the retained cohort
stable.

## Correctness and long-context matrix

These final-stack-only safety rows have no matching final-source `main`
performance arm, so their speed delta is `N/A`; they prove output, memory, and
endpoint safety and are not used for a speed claim.

| Started (Europe/Rome) | Lane | Prefill / decode | TPOT p50 / p95 | Peak RSS | Pressure min | Delta vs tested `main` | Delta vs previous comparable | Result |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| 2026-07-29T02:40:30+02:00 | SSD 128 + 128 | 113.34 / 34.28 t/s | 28.563 / 33.688 ms | 3,757 MiB | 71% | N/A; final policy check, not a crossed speed arm | N/A; first final check | exact; two reads/load |
| 2026-07-29T03:48:28+02:00 | SSD 32K security/coding + 128 | 542.63 / 23.96 t/s | 41.183 / 48.986 ms | 3,757 MiB | 72% | N/A; safety/domain arm | N/A; different prompt | exact vs prior `main`; SHA-256 logits `a3b057...`, decode `2ea829...` |
| 2026-07-29T03:54:11+02:00 | SSD 65,536 + 128 | 219.66 / 19.25 t/s | 49.618 / 63.871 ms | 6,318 MiB | 69% | N/A; safety arm | N/A; different frontier | exact; zero swap |
| 2026-07-29T04:03:35+02:00 | SSD 100,000 + 128 | 155.66 / 15.34 t/s | 62.914 / 76.412 ms | 6,318 MiB | 69% | N/A; safety arm | N/A; different frontier | exact; zero swap |
| 2026-07-29T04:19:10+02:00 | SSD endpoint 262,015 + 128 | 57.00 / 8.19 t/s | 119.718 / 126.218 ms | 11,475 MiB | 60% | N/A; endpoint safety arm | N/A; different frontier | exact; 4,596.41 s prefill; zero swap |

The endpoint deterministically expanded the source prompt, recorded generated
prompt SHA-256 `1bdb4f5dd07621f411b53d62980e5570109c70858d0a73a3cabf274674cd1d7b`,
and produced logits SHA-256
`5d87850465d4ca90eb2b7d427b196fd3b9749babd68a428204411fd6a11ad556`
and decode-evidence SHA-256
`3a7e2d8f7394798631658faaa84ffacdfda3c616eda29defdf8276a64979861a`.

## DeepSeek rejection and final exclusion

Clean production A/B/B/A exploration showed that joining DeepSeek gate+up
reduced syscall count by one third but regressed end-to-end time. The policy
is rejected for DeepSeek:

| Frontier | Tested `main` prefill / decode | Experimental paired prefill / decode | End-to-end delta | `pread` delta | `pread` time delta | Decision |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 2K + 128 | 151.32 / 10.35 t/s | 150.13 / 10.34 t/s | +113.26 ms / +0.44% | -33.33% | +150.08 ms / +8.68% | reject |
| 8K + 128 | 184.03 / 9.56 t/s | 183.10 / 9.55 t/s | +229.46 ms / +0.40% | -33.33% | +191.01 ms / +10.37% | reject |

| Started (Europe/Rome) | Final DeepSeek lane | Prefill / decode | Delta vs tested `main` | Delta vs previous comparable | Result |
| --- | --- | --- | --- | --- | --- |
| 2026-07-29T02:41:21+02:00 | Qwen-only classifier exclusion, 128 + 128 | 19.79 / 11.66 t/s | N/A; policy/correctness check after unlike page-cache state | N/A; unlike model | exact; 2,037 decode loads, 6,111 `pread`, exactly three/load; zero swap |

The final source therefore leaves DeepSeek's qualified three-component I/O,
cache scans, eviction, and overlap behavior unchanged.

## GLM qualification

The qualified 244 GiB object was available on external `SSD1`. It could not be
copied to the internal disk, which had about 51 GiB free; all retained GLM
tests read it directly and completed before the device was released.

The generic `ds4-bench` GLM session produced different 128-token decode
documents on repeated `main` runs as well as repeated candidate runs. This is
a pre-existing nondeterminism in that generic session/bench path, not
candidate-specific drift, so those rows are invalid for promotion:

| Started (Europe/Rome) | Generic GLM arm | Decode SHA-256 prefix | Delta vs tested `main` | Delta vs previous comparable | Result |
| --- | --- | --- | --- | --- | --- |
| 2026-07-29T01:47:42+02:00 | A1, `main` | `1b84f775...` | N/A; no deterministic reference | N/A; first arm | invalid output repeatability |
| 2026-07-29T01:54:18+02:00 | B1, candidate | `9b767c2b...` | N/A; unlike output | N/A; unlike output | invalid output repeatability |
| 2026-07-29T02:01:28+02:00 | A2, `main` | `797f254e...` | N/A; unlike A1 output | N/A; unlike output | invalid output repeatability |
| 2026-07-29T02:11:22+02:00 | B2, candidate | `cd4486a...` | N/A; unlike output | N/A; unlike output | invalid output repeatability |

Qualification instead used the canonical exact 288+32 GLM Gold CLI prompt.
Every stdout document matches the historical SHA-256
`2803fda8b47acff3aedd24bd7609b0c649602ca1fa6d908368b57fe2a586a5c2`.

| Started (Europe/Rome) | GLM Gold arm | Prefill / decode | Wall | Delta vs tested `main` wall mean | Delta vs previous comparable | Result |
| --- | --- | --- | ---: | ---: | ---: | --- |
| 2026-07-29T02:19:39+02:00 | A1, `main` | 3.99 / 0.52 t/s | 136.08 s | +0.48% | N/A; first arm | exact; zero swap |
| 2026-07-29T02:24:11+02:00 | B1, pre-gate candidate | 3.97 / 0.51 t/s | 135.32 s | -0.08% | -0.56% | exact; zero swap |
| 2026-07-29T02:28:46+02:00 | B2, pre-gate candidate | 3.97 / 0.52 t/s | 133.80 s | -1.20% | -1.12% | exact; zero swap |
| 2026-07-29T02:33:16+02:00 | A2, `main` | 3.98 / 0.52 t/s | 134.78 s | -0.48% | +0.73% | exact; zero swap |
| 2026-07-29T02:42:21+02:00 | B3, final Qwen-only stack | 3.92 / 0.53 t/s | 134.78 s | -0.48% | 0.00% | exact; zero swap |

The crossed control mean is 135.43 s and the pre-gate candidate mean is
134.56 s (-0.64%). Control spread is 0.97% and candidate spread is 1.14%, so
the timing is neutral. Final B3 confirms the Qwen-only classifier preserves
the GLM full-record plan and exact output.

## Invalid cohorts and environmental controls

| Approximate order | Lane | Invalidating observation | Disposition |
| --- | --- | --- | --- |
| Before final 8K | Qwen SSD 8K closing `main` | 8 new swapout pages | discard complete cohort and repeat |
| Before retained 32K | Qwen SSD 32K | back-to-back candidate thermal cliff without macOS warning | discard complete cohort; four-minute spacing |
| Before retained 32K-r2 | Qwen SSD 32K closing `main` | 4.20% decode control drift and p95 outlier | discard complete cohort; remove log rendering from measured tail |
| Before retained resident 8K-r3 | Qwen resident 8K | warm-up drift, then a separate 47% B2 thermal cliff | discard both cohorts; four-minute spacing |

No retained arm swapped, resolved a different plan, or observed an inference
competitor. Wallpaper/video processes were suspended only for the benchmark
window and must be resumed after integration.

## Future exploration notes

These are commit-note candidates, not dormant release branches:

- Qwen's fixed maximum-size expert slots reserve about 249.64 MiB more than
  the live logical record bytes. Reclaiming that 24.4% internal fragmentation
  needs compactable/reclaimable slabs or a per-size-class capacity contract;
  merely shrinking slots can violate the one-layer/all-256-experts lease.
- The Qwen/DeepSeek read-grouping split shows that record size and device
  scheduling matter. Revisit only with model/device-qualified crossed cohorts,
  not user-visible autotuning.
- Profile Qwen split-K GQA workgroup geometry at 8K/32K/65K without profiler
  fences; prior fences added about 0.13-0.17 ms per boundary.
- The apparent double `n_kv_head` factor in the FlashAttention tail allocation
  is required by the sparse head-major shader ABI. Allocation, pad offsets,
  and consumer shader layout must change together.
- Diagnose generic GLM `ds4-bench` session nondeterminism separately. The
  canonical GLM Gold CLI path is exact and remains the release gate.

## Post-merge Affine4 SSD correction and overlap measurement

A post-merge audit found that commit `7b5a3eb6f315` accidentally narrowed two
Qwen SSD capability classifiers to Q2_K_XL. The published Stable
Affine4/group-64 artifact therefore lost its validated selected-address
overlap admission. Restoring that exact Affine4 profile exposed a second
wiring error in scalar decode: Qwen executes top-8 as two four-route halves,
but the Affine4 down projection called the IQ2 sum6 encoder. That encoder
correctly rejects `nei0 != 6`, so decode failed before the down kernel was
submitted.

The correction restores only the admitted Affine4/group-64 profile; it does
not restore the retired generic Q4 path. Affine4 down dispatch now uses the
existing 1..8-route Qwen sum8 encoder, while IQ2 remains on its six-route
encoder. The timing summary also separates route publication/copy/planning
from finish publication, exposed SSD wait, `didModifyRange:`, and cache
installation. This instrumentation is observational and remains useful for
the next expert-cache/overlap tranche.

| Started (Europe/Rome) | Revision / lane | Result |
| --- | --- | --- |
| N/A; the failure start was not captured | clean `main` `7b5a3eb6f315`; Affine4 SSD 128 + first decode token | FAIL reproduced independently: token 128 stopped at FFN |
| N/A; the diagnostic start was not captured | corrected classifier, wrong sum6 wrapper retained; layer-0 stage profile | FAIL after gate/up and activation; no down stage was submitted |
| N/A; the diagnostic start was not captured | sum8 wiring correction; layer-0 stage profile | PASS: both four-route halves completed gate/up, down, and sum |
| 2026-07-29T15:05:30+0200 | final source without temporary capability logging; Affine4 SSD 128 + 8 | PASS; full-stack overlap enabled; frontier logits and eight-token decode evidence byte-identical to the synchronous ablation |

The two short arms below are a correctness diagnostic, not a promotion-speed
cohort. Tested `main` has no comparable timing because it aborts at the first
decode token. The final arm followed the synchronous arm on the same host, so
its timing is retained only to expose the observed direction.

| Started (Europe/Rome) | Arm | Prefill t/s | Decode t/s | TPOT p50 / p95 | Delta vs tested `main` | Delta vs previous comparable | Correctness |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| N/A; start was not captured (evidence file created at `2026-07-29T15:04:11+0200`) | Affine4 SSD, `io_overlap` ablated, 128 + 8 | 206.94 | 19.73 | 51.535 / 64.553 ms | N/A; `main` aborts | N/A; first successful arm | exact evidence SHA-256 `56951276…` frontier / `a9b421eb…` decode |
| 2026-07-29T15:05:30+0200 | Affine4 SSD, final full stack, 128 + 8 | 229.86 | 20.80 | 49.448 / 60.922 ms | N/A; `main` aborts | +11.08% / +5.42%; TPOT -4.05% / -5.62% | byte-identical to synchronous arm |

The measurement-first DeepSeek check remained valid because the Affine4
classifier correction is Qwen-only:

| Started (Europe/Rome) | Revision / lane | Prefill t/s | Decode t/s | TPOT p50 / p95 | Delta vs tested `main` | Delta vs previous comparable | Result |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| 2026-07-29T14:51:31+0200 | `7b5a3eb6f315` plus timing-only instrumentation; DeepSeek cold SSD, 2 GiB cache, pure prefill 128 | 28.26 | N/A | N/A | N/A; diagnostic-only arm | N/A; first breakdown | PASS; 4,529.033 ms wall, 2,742.252 ms exposed finish wait, 396.016 ms route-ready wait, 178.760 ms planning |

On this deliberately cold, undersized-cache DeepSeek arm, exposed SSD wait was
about 60.6% of prefill wall time and the router-ready barrier another 8.7%.
That points the next optimization work toward whole-expert prefetch/cache
policy and SSD/GPU overlap, not partial-expert prediction.

## Final validation

| Started (Europe/Rome) | Revision / experiment | Command or lane | Result |
| --- | --- | --- | --- |
| 2026-07-29T00:53:20+02:00 | earlier final Qwen-only source | warning-clean `make ds4_test ds4-bench`; `git diff --check` | pass |
| N/A; start was not captured | earlier final Qwen-only source | complete unrestricted `make model-free-test` | pass |
| 2026-07-29T02:40:30+02:00 | final measured source | Qwen real-model 128 + 128 policy check | exact; two reads/load |
| 2026-07-29T02:41:21+02:00 | final measured source | DeepSeek real-model 128 + 128 exclusion check | exact; three reads/load |
| 2026-07-29T02:42:21+02:00 | final measured source | GLM canonical Gold post-classifier check | exact |
| 2026-07-29T03:48:28+02:00 | final measured source | second-domain Qwen 32K + 128 | exact |
| 2026-07-29T03:54:11+02:00 | final measured source | Qwen 65K safety | pass |
| 2026-07-29T04:03:35+02:00 | final measured source | Qwen 100K safety | pass |
| 2026-07-29T04:19:10+02:00 | final measured source | Qwen 262,015 endpoint | pass |
| 2026-07-29T06:12:20+02:00 | post-benchmark source | first `make premerge` attempt | rejected by the brand-boundary audit; three new private legacy-prefixed names were renamed without changing behavior |
| N/A; exact second-run start was not retained (final binary emitted 2026-07-29T06:16:02+02:00) | final source, binary `f16ede1...` | complete `make premerge` | pass |
| 2026-07-29T06:17:58+02:00 | final source, binary `f16ede1...` | Qwen real-model 128 + 128 post-rename identity check | exact; two reads/load; zero swap; no competitor |
| 2026-07-29T06:18:08+02:00 | final source, binary `f16ede1...` | DeepSeek deliberately undersized cache admission check | expected fail-closed before inference; zero swap; no competitor |
| 2026-07-29T06:18:28+02:00 | final source, binary `f16ede1...` | DeepSeek real-model AUTO 128 + 128 post-rename identity check | exact; three reads/load; zero swap; no competitor |
| 2026-07-29T15:07:23+0200 | Affine4 corrective source plus unrelated in-progress v3 prototype files | complete unrestricted `make model-free-test` | PASS; Metal kernels, ExpertMajor v2, Qwen Affine4/IQ fixtures, SSD residency, metadata, server, and repository model-free lanes |
| 2026-07-29T15:11:30+0200 | Affine4 corrective source plus unrelated in-progress v3 prototype files | complete unrestricted `make premerge` | PASS; repository, documentation, brand boundary, build isolation, model-free Metal, install layout, and final diff checks |

The full diff from merge base `572e6a6` was reviewed for correctness,
Qwen-only gating, experimental residue, generated artifacts, and unsupported
compatibility. The active handoff was removed, `git diff --check` is clean,
and the eight benchmark documents plus this cumulative decision record are
the only new untracked source-tree files intended for integration.
