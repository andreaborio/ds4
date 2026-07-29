# Qwen layer-staleness cache eviction

Date: 2026-07-28

Status: directional sixth-tranche candidate. Model-free Metal, Qwen
128-prefill/128-decode exactness, and DeepSeek 128-prefill non-regression pass.
The required context and GLM qualification matrices remain pending.

Decision: retain the Qwen-only policy after the five earlier routed-I/O and
cache-scan corrections on this branch. Remove the experimental environment
switch and make the successful policy the Qwen default. Do not promote or
merge the combined stack from this short, contaminated cohort alone.

Supersedes: the diagnostic
`DS4_METAL_STREAM_EVICT_LAYER_STALENESS` path. This record does not supersede
the canonical long-context acceptance matrix.

Affected path: Qwen3.6 SSD inference with its embedded ExpertMajor v2 store.
DeepSeek keeps its existing LFU/LRU tie-break. GLM is explicitly excluded and
also keeps LFU/LRU.

## Inefficiency and correction

The cache first compares route hotness and previously used LRU for equal
hotness. During scalar decode, however, model layers execute in increasing
order. Under cache pressure an expert from layer 21 while layer 20 is running
will be needed almost immediately, whereas an expert from layer 19 has just
missed its opportunity and will not be needed until the next model sweep.
Plain cross-layer LRU can prefer the upcoming expert as the victim.

For Qwen only, equal-hotness candidates from different layers now compare
their forward distance from the current layer. The farther candidate is the
better victim. Equal-hotness candidates in the same layer still use LRU, and
route hotness remains the primary key in all cases.

The runtime identifies Qwen from the active, validated embedded-store geometry:
40 layers by 256 experts, using the canonical Qwen shape constants. It does not
infer the model from the protected-ID count of one cache operation. That
distinction matters because a Qwen seed/prune operation may temporarily
protect fewer than its normal top eight routes. DeepSeek's 43-layer store and
GLM model mode cannot enter the new tie-break.

## Experiment identity

| Condition | Value |
| --- | --- |
| Host | Apple M5 Pro, 64 GiB unified memory, macOS 26.5.2 |
| Parent / remote-main identity | `572e6a6df07ef09bff4bd9d4ef54ffbfbad43aad` |
| Immediate control | uncommitted combined five-tranche stack, experimental switch absent |
| Candidate | uncommitted combined six-tranche stack; Qwen layer-staleness default |
| Qwen model | `Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-Q2_K_XL.gguf`, 12,290,632,032 bytes, SHA-256 `30c22f70aff0f05986b517ee4ad8fef554a1b5aab6971c9ca09f999566d30143` |
| Qwen prompt | `speed-bench/promessi_sposi.txt`, SHA-256 `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f` |
| Qwen runtime | forced SSD, cold preload, 1 GiB cache, strict full stack, 65,536-token allocation |
| Completed Qwen tier | 128 prefill plus 128 greedy decode tokens |
| DeepSeek runtime | forced SSD, cold preload, 2 GiB cache, required asynchronous prefill overlap, 129-token allocation |
| Completed DeepSeek tier | 128 pure-prefill tokens |
| Warm-up policy | no discarded warm-up; every arm started a fresh process with a cold expert cache |
| Resolved plan | forced SSD in every measured model arm |
| Raw evidence | `/private/tmp/hebrus-qwen-record-io-*.csv`, `/private/tmp/hebrus-qwen-layer-stale-*.csv`, `/private/tmp/hebrus-qwen-layer-stale-final*`, and `/private/tmp/hebrus-deepseek-layer-stale-final*` |
| Host isolation | invalid for performance: WindowServer about 51%, Wallpaper Aerials about 52%, and a VideoToolbox decoder about 17% CPU |

The exact parent-only `main` binary was not measured under the same final
conditions. Therefore every delta against tested `main` is `N/A`. The A arms
below are the immediately preceding five-tranche stack and isolate this policy,
but are not a substitute for the final original-main cohort.

## Qwen A/B/B/A diagnostic

Rows are chronological. Candidate and control output evidence is byte-identical
in every complete arm. Timing is retained only to reject a clear regression;
the contaminated host does not support a speed claim.

| Started | Arm | Prefill t/s | Decode t/s | Decode TPOT p50 / p95 | Delta vs tested `main` | Delta vs five-tranche A mean | Delta vs previous comparable | Result |
| --- | --- | ---: | ---: | --- | --- | --- | --- | --- |
| 2026-07-28T22:56:57+02:00 | A1, five-tranche LFU/LRU | 308.01 | 39.57 | 25.025 / 27.205 ms | N/A; no matching `main` arm | -0.15% decode | N/A; first arm | exact pass |
| 2026-07-28T22:58:40+02:00 | B1, layer staleness | 294.12 | 39.44 | 25.006 / 27.092 ms | N/A; no matching `main` arm | -0.48% decode | -0.33% decode vs A1 | exact pass |
| 2026-07-28T22:59:04+02:00 | B2, layer staleness | 297.65 | 39.76 | 24.891 / 26.968 ms | N/A; no matching `main` arm | +0.33% decode | +0.81% decode vs B1 | exact pass |
| 2026-07-28T22:59:18+02:00 | A2, five-tranche LFU/LRU | 301.52 | 39.69 | 24.897 / 27.199 ms | N/A; no matching `main` arm | +0.15% decode | -0.18% decode vs B2 | exact pass |

The control mean is 39.63 decode tokens/s and the candidate mean is 39.60
tokens/s, a diagnostic -0.08 percent. Mean TPOT p95 changes from 27.202 to
27.030 ms. These timing differences are smaller than the cohort noise.

The deterministic decode resources do change:

| Decode resource | Five-tranche LFU/LRU | Qwen layer staleness | Delta |
| --- | ---: | ---: | ---: |
| Cache hit rate | 0.6376 | 0.6461 | +0.0085 |
| Evictions | 14,377 | 14,026 | -351 (-2.44%) |
| Expert loads | 14,845 | 14,494 | -351 (-2.36%) |
| Unique experts read | 3,347 | 3,341 | -6 (-0.18%) |
| `pread` syscalls | 44,535 | 43,482 | -1,053 (-2.36%) |
| SSD read | 14.079872 GiB | 13.742409 GiB | -0.337463 GiB (-2.40%) |
| Read amplification | 4.440822 | 4.342348 | -2.22% |

Prefill remains identical: 4,106 evictions, 4,427 expert loads, 13,281
`pread` calls, and 4.249199 GiB read. Across the full run, misses and loads fall
from 19,272 to 18,921 and evictions from 18,483 to 18,132.

The final flag-free Qwen run at 2026-07-28T23:13:42+02:00 reproduced all
candidate structural counters. Its complete frontier logits are byte-identical
to the five-tranche reference, SHA-256
`37fc78e2e0dc64dec160d4ea901348ffc6efb33cbdb53c805c24d11a42ffd8a7`.
Its decode evidence is also byte-identical, SHA-256
`16d915d7e6fc1ef4c5f550a416e7b0b18ae347da9205d1f29d3c92d31d6615f1`.

## Rejected record-read schedules

The same profile showed expert-record `pread` time dominating the already
reduced victim-scan time, so two existing diagnostic schedules were checked
before changing the eviction policy.

| Started | Schedule | Prefill t/s | Decode t/s | TPOT p50 / p95 | Delta vs tested `main` | Delta vs previous comparable | Decode `pread` time | Result / decision |
| --- | --- | ---: | ---: | --- | --- | --- | ---: | --- |
| 2026-07-28T22:56:57+02:00 | A1, component reads | 308.01 | 39.57 | 25.025 / 27.205 ms | N/A; no matching `main` arm | N/A; first arm | 405.458 ms | exact control |
| 2026-07-28T22:57:11+02:00 | B1, one full-record `pread` | N/A | N/A | N/A | N/A; incomplete arm | N/A; failed arm | N/A | rejected: first decode token failed in Qwen FFN |
| 2026-07-28T22:57:34+02:00 | B1, balanced record reads | 296.58 | 39.46 | 25.024 / 27.361 ms | N/A; no matching `main` arm | -0.28% decode vs A1 | 413.209 ms | exact but no resource benefit; rejected |

The full-record path is not runtime-correct for this Qwen artifact. The
balanced schedule retains the same load, syscall, and byte counters and did
not reduce measured `pread` time. Neither schedule is part of this candidate.

## DeepSeek non-regression

| Started | Revision / experiment | Prefill t/s | Decode t/s | TPOT p50 / p95 | Delta vs tested `main` | Delta vs previous comparable | SSD read | Result |
| --- | --- | ---: | ---: | --- | --- | --- | ---: | --- |
| 2026-07-28T23:00:29+02:00 | diagnostic policy enabled globally | 26.77 | N/A | N/A | N/A; contaminated short arm | N/A; first arm | 37.738037 GiB | exact pass |
| 2026-07-28T23:14:15+02:00 | final Qwen-only default | 26.76 | N/A | N/A | N/A; contaminated short arm | -0.04% prefill | 37.738037 GiB | exact pass; unchanged counters |

The final DeepSeek run retains 27,299 hits, 5,725 misses, 5,422 evictions,
5,725 expert loads, and 17,175 `pread` calls. Frontier logits remain
byte-identical to the original control and all five preceding tranches,
SHA-256
`2a1b42ec08d1657a319c124f0d721f5a1c132d2efa4d911ac7ca8f9e4d483471`.

## Validation

| Started | Revision / experiment | Test command or lane | Result |
| --- | --- | --- | --- |
| 2026-07-28T23:03:58+02:00 | initial policy candidate | warning-clean `make ds4_test` | pass |
| 2026-07-28T23:04:14+02:00 | initial policy candidate, sandbox | `ds4_test --metal-kernels` | invalid; sandbox exposed no Metal device |
| 2026-07-28T23:04:21+02:00 | initial policy candidate, Apple Metal | `ds4_test --metal-kernels` | pass |
| 2026-07-28T23:13:17+02:00 | final model-geometry discriminator | warning-clean `make ds4_test` plus production `make metal` | pass |
| 2026-07-28T23:13:31+02:00 | final model-geometry discriminator | model-free kernel suite on Apple Metal | pass |
| 2026-07-28T23:13:42+02:00 | final production binary | Qwen 128 prefill plus 128 decode, strict SSD stack | pass; exact outputs and candidate structural counters |
| 2026-07-28T23:14:15+02:00 | final production binary | DeepSeek 128 pure-prefill, required SSD overlap | pass; exact logits and unchanged cache/I/O counters |
| 2026-07-28T23:17:34+02:00 | first final-source audit | documentation links | pass, 192 local links |
| 2026-07-28T23:17:34+02:00 | first final-source audit | brand-boundary check | fail; new helper and duplicate Makefile dependency increased the migration-pending prefix |
| 2026-07-28T23:18:06+02:00 | Hebrus-prefixed helper, compiler-generated header dependency | documentation, brand, release-contract, generator, and build-isolation gates | pass |
| 2026-07-28T23:18:42+02:00 | final sixth-tranche source | complete model-free suite, including Apple Metal | pass |
| 2026-07-28T23:19:38+02:00 | final sixth-tranche source | install-layout and `git diff --check` | pass |

The model-free comparator regression proves that route hotness stays primary,
the disabled policy stays LRU, cross-layer Qwen ties select the farther
just-passed layer, the reverse ordering does not, and same-layer ties stay LRU.
`context-audit` is intentionally not reported as passing: the task remains
active and its required handoff must stay under `docs/work/active/` until model
qualification is complete.

## Remaining promotion gates

- Stable-host Qwen and DeepSeek A/B/B/A at 2K/8K/32K, plus the second 32K
  routing/I/O prompt domain.
- DeepSeek 65,536/100K safety and Qwen near-262K endpoint coverage.
- Qualified GLM execution because the earlier five tranches modify common
  cache code. The policy in this record is excluded in GLM mode, but the
  combined stack still requires the complete structural qualification.
- Final combined-stack comparison against original `hebrus/main`, full
  merge-base diff review, active-handoff removal, and clean `make premerge`.
