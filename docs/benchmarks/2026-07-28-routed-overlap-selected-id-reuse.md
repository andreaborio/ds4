# Routed SSD-overlap selected-ID reuse

Date: 2026-07-28

Status: directional candidate; short DeepSeek and Qwen Q2_K_XL A/B/B/A
prefill cohorts, a Qwen decode exactness lane, and the model-free Metal gate
pass, but the required 2K/8K/32K qualification matrices remain pending.

Decision: retain the ownership correction on its branch. Do not promote it from
this record alone.

Supersedes: no accepted benchmark decision. The change removes duplicate host
work from an existing overlap schedule; it does not enable DeepSeek overlap,
change the Qwen feature policy, or alter routed arithmetic.

## Inefficiency and correction

The router-ready overlap path copied the selected-expert tensor to host memory
to validate IDs, group routes, and launch SSD reads. The later batch routed-MoE
preparation discarded that authenticated copy, allocated another array, and
copied the same immutable Metal buffer again.

The pending overlap ticket now owns the first host array until one of two exact
endpoints:

- normal batch consumption transfers the array to routed-MoE preparation;
- abort, mismatch cleanup, or engine cleanup frees it through the existing
  pending-ticket reset.

The consumer validates the expected `n_tokens * n_selected` count before taking
ownership. The synchronous fallback is unchanged and still performs its one
required readback.

Scalar Qwen decode selects exactly eight experts. Its overlap ticket now keeps
that first 32-byte read in an eight-element inline array instead of allocating
and freeing a heap array for every routed layer and token. The later scalar
consumer in this first tranche deliberately retained its stack-based second
read. A later ownership refinement safely removed that duplicate tensor
readback; see
[scalar selected-ID reuse](2026-07-28-routed-overlap-scalar-id-reuse.md).

Affected paths are Qwen Q2_K_XL SSD prefill with the accepted I/O-overlap
feature and the opt-in DeepSeek SSD prefill-overlap experiment. Published Qwen
Affine4 does not advertise this overlap capability and is unchanged.

## Deterministic resource effect

The measured DeepSeek frontier has 43 routed layers, 128 tokens, top-6 routing,
and 3,072 selected-ID bytes per layer.

| Resource | Control | Candidate | Delta |
| --- | ---: | ---: | ---: |
| Selected-ID host copies | 86 | 43 | -50.00% |
| Selected-ID host bytes copied | 264,192 | 132,096 | -50.00% |
| Selected-ID heap allocations | 86 | 43 | -50.00% |
| Profiled second-read section, A/B arm | 0.021 ms | 0.006 ms | -71.43% |
| Profiled second-read section, A2/B2 arm | 0.023 ms | 0.001 ms | -95.65% |

The timer still includes branch and clock overhead, so its sub-microsecond
values are not used as a speed claim. The copy/allocation counts follow
directly from the exercised call graph.

The final Qwen decode check generated 128 tokens through 40 routed layers, for
5,120 scalar route-ready calls. The inline array changes those calls as follows.

| Resource | Control | Candidate | Delta |
| --- | ---: | ---: | ---: |
| Scalar selected-ID heap allocations | 5,120 | 0 | -100.00% |
| Scalar selected-ID heap frees | 5,120 | 0 | -100.00% |
| Selected-ID bytes stored per call | 32 | 32 | 0.00% |
| Selected-ID readbacks per call | 2 | 2 | 0.00% |

The resource reduction is structural and bounded by the runtime-reported 5,120
selected calls. It is not a throughput claim.

## DeepSeek short-context A/B/B/A

| Condition | Value |
| --- | --- |
| Host | Apple M5 Pro, 64 GiB unified memory, macOS 26.5.2 |
| Metal runtime | Apple M5 Pro, Metal 4 tensor API enabled |
| Baseline revision | `hebrus/main` `572e6a6df07e` |
| Baseline binary SHA-256 | `ca0ebb53a2d0e28bbd110a010a77913638a7a5920d4571f7906d9cef5e464a6f` |
| Candidate | `codex/metal-ssd-attention-audit`, uncommitted selected-ID ownership diff on `572e6a6df07e` |
| Measured candidate binary SHA-256 | `e06e690596b263aa3756c0be6c88bcc93d391403976047aecbf4dd2fb5dea057` |
| Post-audit candidate binary SHA-256 | `81bed92c80f4187ee99eebb085e08188c5c747809a5a6213d9ca02565cb1add5`; differs by the brand-audit-driven private-symbol rename applied after measurement |
| Model | DeepSeek V4 Flash ExpertMajor v2, 86,720,114,272 bytes, SHA-256 `8378080263eb9224f7228d72e2afa4ac3cf74a116023fdec2c596ff228a33e3f` |
| Prompt SHA-256 | `29363eab21bbbccaeea8e13f669e7ce05e8eafc48e31fcf9b725edabb2058666` |
| Runtime | forced SSD, cold preload, 2 GiB expert cache, 129-token allocation |
| Frontier | 128 pure-prefill tokens; no decode |
| Overlap | required, capable, ready, and asynchronous on all 43 routed layers |
| Raw evidence | local scratch CSV, stderr profile, `/usr/bin/time -l`, and frontier-logit JSON; intentionally not committed |

Rows are chronological. The tested baseline mean is 26.140 prefill tokens/s.
“Previous comparable” compares candidate arms with the previous candidate and
control arms with the previous control. Decode and TPOT are unavailable because
this is a pure-prefill diagnostic.

| Started | Arm | Prefill t/s | Decode t/s | TPOT p50 / p95 | Delta vs tested baseline | Delta vs previous comparable | SSD reads | Selected-read section | Result |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 2026-07-28T20:54:48+02:00 | A1 control | 25.94 | N/A | N/A | -0.77% | N/A; first control | 37.738037 GiB | 0.021 ms | pass |
| 2026-07-28T20:55:13+02:00 | B1 candidate | 26.00 | N/A | N/A | -0.54% | +0.23% vs A1 | 37.738037 GiB | 0.006 ms | pass |
| 2026-07-28T20:55:51+02:00 | B2 candidate | 26.13 | N/A | N/A | -0.04% | +0.50% vs B1 | 37.738037 GiB | 0.001 ms | pass |
| 2026-07-28T20:56:06+02:00 | A2 control | 26.34 | N/A | N/A | +0.77% | +1.54% vs A1 | 37.738037 GiB | 0.023 ms | pass |

The candidate mean is 26.065 tokens/s, or -0.29 percent versus the control
mean. Control spread is 1.54 percent and candidate spread is 0.50 percent, so
the short run supports performance neutrality rather than a throughput gain.
Every arm reported zero swapout. Peak memory footprint ranged from
2,432,223,920 to 2,439,400,184 bytes with no candidate-only increase.

## Correctness and validation

| Started | Revision / experiment | Test command or lane | Result |
| --- | --- | --- | --- |
| 2026-07-28T20:50:36+02:00 | candidate | `make ds4_test` | pass; warning-clean Metal test build |
| 2026-07-28T20:50:54+02:00 | candidate, sandbox | `ds4_test --metal-kernels` | invalid; sandbox exposed no Metal device |
| 2026-07-28T20:51:07+02:00 | candidate, Apple Metal | `ds4_test --metal-kernels` | pass |
| 2026-07-28T20:55:51+02:00 | candidate B2 | DeepSeek 128-token frontier logits | byte-identical to A2 control |
| 2026-07-28T20:56:06+02:00 | control A2 | DeepSeek 128-token frontier logits | SHA-256 `2a1b42ec08d1657a319c124f0d721f5a1c132d2efa4d911ac7ca8f9e4d483471` |
| 2026-07-28T20:58:36+02:00 | candidate | `make premerge` | fail; new private test symbols exceeded the brand-token baseline |
| 2026-07-28T20:59:11+02:00 | candidate after test-symbol rename | `make premerge` | fail; the modified private reset still used the migration-pending prefix |
| 2026-07-28T21:00:29+02:00 | candidate after complete private-symbol rename | `make premerge` | pass; context/docs, brand, contracts, generators, build isolation, model-free Metal/CPU, install, and diff checks |
| 2026-07-28T21:28:25+02:00 | candidate with inline top-8 storage, sandbox | `ds4_test --metal-kernels` | invalid; sandbox exposed no Metal device |
| 2026-07-28T21:28:25+02:00 | candidate with inline top-8 storage, Apple Metal | `ds4_test --metal-kernels` | pass |
| 2026-07-28T21:30:31+02:00 | final candidate | Qwen 128 prefill + 128 greedy decode | pass; 5,120 scalar selected calls, final logits and decode evidence byte-identical to control |
| 2026-07-28T21:30:51+02:00 | `hebrus/main` control `572e6a6df07e` | Qwen 128 prefill + 128 greedy decode | pass; logits SHA-256 `37fc78e2e0dc64dec160d4ea901348ffc6efb33cbdb53c805c24d11a42ffd8a7`, decode evidence SHA-256 `16d915d7e6fc1ef4c5f550a416e7b0b18ae347da9205d1f29d3c92d31d6615f1` |
| 2026-07-28T21:32:13+02:00 | final candidate | `git diff --check`; `make doc-links context-audit brand-boundary-audit` | pass |
| 2026-07-28T21:33:30+02:00 | final candidate | `make premerge` | pass; complete repository, build-isolation, model-free Metal/CPU, install, and diff gate |

The new model-free regression covers successful pointer transfer, count
mismatch fail-closed behavior, heap reset ownership after transfer, and
non-transferable inline reset. The exact frontier-logit matches demonstrate
that route order, selected IDs, expert weights, generated tokens, and final
arithmetic did not change in the exercised DeepSeek and Qwen lanes.

## Qwen Q2_K_XL short-context A/B/B/A

The exact published Beta artifact was found in local scratch after the initial
audit. Its 12,290,632,032 bytes and SHA-256
`30c22f70aff0f05986b517ee4ad8fef554a1b5aab6971c9ca09f999566d30143`
match `docs/contracts/qwen-release.json`.

All arms used forced SSD, cold preload, a 1 GiB expert cache, a 129-token
allocation, strict full-stack admission, and the same prompt as the DeepSeek
cohort. The runtime reported `io_overlap` implemented, expected, and enabled.
All 40 routed layers exercised the selected-address batch path.

At 128 tokens with top-8 routing, each layer has 4,096 selected-ID bytes.
Control therefore performs 80 copies / 327,680 bytes and 80 allocations;
candidate performs 40 copies / 163,840 bytes and 40 allocations. This is the
same exact 50-percent reduction demonstrated on DeepSeek.

Rows are chronological. The tested baseline mean is 209.330 prefill tokens/s.
Decode and TPOT are unavailable because this is a pure-prefill diagnostic.

| Started | Arm | Prefill t/s | Decode t/s | TPOT p50 / p95 | Delta vs tested baseline | Delta vs previous comparable | SSD reads | Selected-read section | Result |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 2026-07-28T21:12:35+02:00 | A1 control | 210.25 | N/A | N/A | +0.44% | N/A; first control | 5.340103 GiB | 0.023 ms | pass |
| 2026-07-28T21:12:51+02:00 | B1 candidate | 207.00 | N/A | N/A | -1.11% | -1.55% vs A1 | 5.340103 GiB | 0.000 ms | pass |
| 2026-07-28T21:13:06+02:00 | B2 candidate | 209.11 | N/A | N/A | -0.11% | +1.02% vs B1 | 5.340103 GiB | 0.000 ms | pass |
| 2026-07-28T21:13:18+02:00 | A2 control | 208.41 | N/A | N/A | -0.44% | -0.88% vs A1 | 5.340103 GiB | 0.025 ms | pass |

The candidate mean is 208.055 tokens/s, or -0.61 percent versus control.
Control spread is 0.88 percent and candidate spread is 1.02 percent, so the
cohort supports performance neutrality rather than a throughput claim. Every
arm reported zero swapout and identical SSD bytes. Candidate B2 and control A2
frontier logits are byte-identical, SHA-256
`482004af3b5902efaae234b5e3bc544345785662d7c26a5d9cc3ad9aa46ce8b8`.

## Qwen final scalar-decode correctness lane

This B/A lane validates the final combined ownership and inline-storage source,
whose benchmark executable SHA-256 is
`dffeaf29ac4f6a449359f67b157aeb268e5406970c11393c5ccc3e6d231cfafb`.
Both arms used the published Q2_K_XL artifact, forced SSD, cold preload, a
1 GiB cache, strict full-stack admission, a 65,536-token allocation, 128-token
prefill, and 128 greedy decode tokens. Runtime route/cache counters, SSD bytes,
and exact-stack resolution were identical.

The host snapshot contained material non-inference load (`WindowServer` about
48 percent CPU and `syspolicyd` about 47 percent CPU). The two arms therefore
exist only as correctness and resource-path evidence. Their timing deltas are
`N/A`: a contaminated B/A pair is neither an acceptance cohort nor comparable
to the retained A/B/B/A baseline.

| Started | Arm | Prefill t/s | Decode t/s | TPOT p50 / p95 | Delta vs tested baseline | Delta vs previous comparable | SSD reads | Result |
| --- | --- | ---: | ---: | --- | --- | --- | ---: | --- |
| 2026-07-28T21:30:31+02:00 | B final candidate | 248.26 | 34.85 | 28.260 / 32.238 ms | N/A; contaminated correctness lane | N/A; first final-candidate arm | 18.329071 GiB | exact pass |
| 2026-07-28T21:30:51+02:00 | A `hebrus/main` control | 243.46 | 34.88 | 28.175 / 31.736 ms | N/A; contaminated correctness lane | N/A; no comparable prior control | 18.329071 GiB | exact pass |

Both full frontier-logit JSON documents are byte-identical, SHA-256
`37fc78e2e0dc64dec160d4ea901348ffc6efb33cbdb53c805c24d11a42ffd8a7`.
Both post-decode token/final-logit documents are byte-identical, SHA-256
`16d915d7e6fc1ef4c5f550a416e7b0b18ae347da9205d1f29d3c92d31d6615f1`.
Each run reported 5,120 selected calls, 62,648 cache hits, 19,272 misses,
57,816 `pread` calls, and no runtime fallback.

## Invalid Qwen 2K timing attempts

Two 2K A/B/B/A attempts completed before the scalar inline addition. They are
retained only to explain why no 2K speed result is claimed. The first cohort's
decode controls drifted by 5.94 percent. The second ran under the same material
background load and its candidate effect did not exceed the control drift.
Once a cohort is invalidated, no row has a tested baseline or previous
comparable arm; the required deltas are therefore `N/A`.

| Started | Cohort / arm | Prefill t/s | Decode t/s | TPOT p50 / p95 | Delta vs tested baseline | Delta vs previous comparable | Result |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| 2026-07-28T21:18:52+02:00 | first A1 | 789.27 | 35.19 | 27.420 / 35.317 ms | N/A; invalid cohort | N/A; invalid cohort | invalid |
| 2026-07-28T21:19:16+02:00 | first B1 | 792.39 | 35.97 | 26.993 / 33.853 ms | N/A; invalid cohort | N/A; invalid cohort | invalid |
| 2026-07-28T21:19:45+02:00 | first B2 | 783.62 | 34.99 | 27.458 / 35.256 ms | N/A; invalid cohort | N/A; invalid cohort | invalid |
| 2026-07-28T21:20:06+02:00 | first A2 | 816.26 | 37.28 | 25.870 / 33.153 ms | N/A; invalid cohort | N/A; invalid cohort | invalid |
| 2026-07-28T21:21:01+02:00 | second A1 | 802.10 | 35.67 | 27.093 / 34.989 ms | N/A; invalid cohort | N/A; invalid cohort | invalid |
| 2026-07-28T21:21:27+02:00 | second B1 | 807.53 | 35.80 | 27.063 / 34.083 ms | N/A; invalid cohort | N/A; invalid cohort | invalid |
| 2026-07-28T21:21:43+02:00 | second B2 | 811.90 | 35.57 | 27.331 / 34.699 ms | N/A; invalid cohort | N/A; invalid cohort | invalid |
| 2026-07-28T21:22:19+02:00 | second A2 | 782.87 | 34.85 | 27.962 / 34.843 ms | N/A; invalid cohort | N/A; invalid cohort | invalid |

All arms reported zero swap and the same 7.656487 GiB of SSD reads. Where
captured, candidate and control frontier logits and decode evidence were
byte-identical. Those correctness observations do not rehabilitate the timing
cohorts.

## Remaining promotion gates

- DeepSeek and Qwen 2K/8K/32K matrices, plus the second 32K prompt domain for
  the routing/I/O change.
- DeepSeek 65,536/100K safety and Qwen near-262K endpoint coverage required by
  their long-context contracts.
- Qualified GLM model run because this modifies common SSD ownership code,
  even though GLM does not call the changed route-ready path.
- Independent merge-base diff review and final residue audit before merge.
