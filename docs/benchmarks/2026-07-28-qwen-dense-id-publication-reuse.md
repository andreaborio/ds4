# Qwen dense synthetic-ID publication reuse

Date: 2026-07-28

Status: directional seventh-tranche candidate. The complete model-free suite,
Qwen 128-prefill/128-decode exactness, and a 128-to-512-token buffer-growth
comparison pass. The required context and GLM qualification matrices remain
pending.

Decision: retain the Qwen Q2_K_XL host-work reduction after the six earlier
routed-I/O and cache corrections on this branch. Do not claim a throughput
gain or merge the combined stack from this short, contaminated cohort.

Supersedes: repeated zeroing and publication of the dense routed-kernel
synthetic expert-ID buffer before every GGML-K projection. It does not
supersede the canonical long-context acceptance matrix.

Affected path: Qwen3.6 Q2_K_XL resident and SSD inference. The only caller of
`ds4_gpu_matmul_ggml_k_tensor()` is the Qwen dense projection dispatcher.
DeepSeek and GLM do not enter this function.

## Inefficiency and correction

Qwen's Q4_K/Q5_K/Q6_K dense projections reuse a routed Metal kernel with one
synthetic expert. Every token or batch row therefore has expert ID zero. The
old host path nevertheless cleared the entire ID buffer and called
`didModifyRange:` before every projection, even though the kernels only read
the buffer and no dispatch can change it.

The scratch allocator retains a buffer while its capacity is sufficient and
replaces it when a larger allocation is required. The corrected path records
whether allocation or growth is needed before calling the allocator, then
zeros and publishes the new buffer only in that case. A reused buffer requires
no host write or publication. No Metal kernel, dispatch geometry, graph order,
weight, activation, cache, or routing decision changes.

The production Q2_K_XL decode topology makes the redundant work explicit:

| Projection group | Layers / calls | GGML-K calls per token |
| --- | --- | ---: |
| Gated DeltaNet QKV, gate, and output | 30 layers × 3 | 90 |
| Full-attention Q, K, V, and output | 10 layers × 4 | 40 |
| Shared-FFN gate and up | 40 layers × 2 | 80 |
| Shared-FFN down | 39 Q6_K layers; layer 1 is Q8_0 | 39 |
| Vocabulary output | 1 Q4_K projection | 1 |
| Total | 40 transformer layers plus output | 250 |

For the qualified 128-prefill/128-decode run, prefill is one 128-row batch.
The old path therefore made 250 publications of 512 bytes during prefill and
32,000 publications of four bytes during decode. The new path initializes the
512-byte buffer once in prefill and performs no decode publication:

| Host operation for the complete run | Immediate control | Candidate | Reduction |
| --- | ---: | ---: | ---: |
| Zero-and-`didModifyRange:` publications | 32,250 | 1 | -32,249 (-99.997%) |
| Bytes explicitly rewritten as zero | 256,000 | 512 | -255,488 (-99.800%) |
| Steady-state publications per decode token | 250 | 0 | -250 (-100%) |
| Steady-state zero bytes per decode token | 1,000 | 0 | -1,000 (-100%) |

These are deterministic graph/work counts, not elapsed-time estimates.
`didModifyRange:` cost is driver and storage-mode dependent, and the
contaminated host did not resolve a trustworthy throughput effect.

## Experiment identity

| Condition | Value |
| --- | --- |
| Host | Apple M5 Pro, 64 GiB unified memory, macOS 26.5.2 |
| Parent / remote-main identity | `572e6a6df07ef09bff4bd9d4ef54ffbfbad43aad` |
| Immediate control | uncommitted combined six-tranche stack, binary SHA-256 `61e0f65a25474f4fefca02482d2c1d362a61751390b7b745681a78641ae3acce` |
| Timed candidate | same stack plus allocation-only ID publication before the final state simplification, binary SHA-256 `a2a80bfa8920d0793482fe81ae7e865597a308a0c60fc2bfe6700c5b8b7ea33e` |
| Final candidate | semantically equivalent allocation-only implementation without a redundant prefix-length global, binary SHA-256 `2e7e0476de05224d2419df4e6b665024c415f0b6c6c4f6025f98cea9814483fa` |
| Qwen model | `Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-Q2_K_XL.gguf`, 12,290,632,032 bytes, SHA-256 `30c22f70aff0f05986b517ee4ad8fef554a1b5aab6971c9ca09f999566d30143` |
| Prompt | `speed-bench/promessi_sposi.txt`, SHA-256 `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f` |
| Runtime | forced SSD, cold preload, 1 GiB cache, 65,536-token allocation |
| Timed frontier | 128 prefill plus 128 greedy decode tokens |
| Growth frontier | 128 then 512 tokens in one process, one greedy decode token at each frontier |
| Warm-up policy | no discarded warm-up; every arm started a fresh process with a cold expert cache |
| Resolved plan | forced SSD in every completed model arm |
| Raw evidence | `/private/tmp/hebrus-qwen-dense-id-*.csv`, `/private/tmp/hebrus-qwen-dense-id-*-frontier`, and `/private/tmp/hebrus-qwen-dense-id-*-decode` |
| Host isolation | invalid for performance: Wallpaper Aerials about 51%, WindowServer about 45%, VideoToolbox decoder about 18%, and Codex about 33% CPU before the cohort |

The exact parent-only `main` binary was not measured under the same final
conditions. Therefore every delta against tested `main` is `N/A`. The A arms
isolate this seventh change from the immediately preceding six-tranche stack
but do not replace the required final original-main comparison.

## Contaminated A/B/B/A diagnostic

Rows are chronological. Start timestamps were not captured for this
exploratory micro-cohort, so the ISO timestamps below are file-completion
times. Timing is retained only to show that the host cannot resolve the
effect; it is not promotion evidence.

| Completed | Arm | Prefill t/s | Decode t/s | Decode TPOT p50 / p95 | Delta vs tested `main` | Delta vs previous comparable | Result |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| 2026-07-28T23:38:15+02:00 | A1, repeated publication | 293.63 | 39.68 | 24.941 / 27.181 ms | N/A; no matching `main` arm | N/A; first arm | exact control |
| 2026-07-28T23:38:26+02:00 | B1, allocation-only publication | 300.74 | 38.52 | 25.241 / 31.085 ms | N/A; no matching `main` arm | +2.42% prefill / -2.92% decode vs A1 | exact candidate |
| 2026-07-28T23:38:37+02:00 | B2, allocation-only publication | 297.22 | 39.44 | 24.951 / 27.392 ms | N/A; no matching `main` arm | -1.17% prefill / +2.39% decode vs B1 | exact candidate |
| 2026-07-28T23:38:47+02:00 | A2, repeated publication | 291.43 | 39.64 | 24.960 / 27.230 ms | N/A; no matching `main` arm | -1.95% prefill / +0.51% decode vs B2 | exact control |

Control means are 292.53 prefill and 39.66 decode tokens/s. Candidate means
are 298.98 and 38.98 tokens/s: a diagnostic +2.20 percent prefill and -1.71
percent decode. Candidate TPOT p50 is only 0.58 percent slower, while its p95
is 7.47 percent slower because one arm contains a 31.085 ms outlier. A later
final-source exactness run completed at 296.22 prefill and 39.73 decode
tokens/s. This contradictory ordering is consistent with external host noise,
so no timing delta is accepted.

Cache hit rate, evictions, loads, unique experts, `pread` calls, and SSD bytes
are identical in all completed 128+128 arms. The final source retains the
sixth-tranche values: decode hit rate 0.6461, 14,026 evictions, 14,494 expert
loads, 43,482 `pread` calls, and 13.742409 GiB read.

## Exactness and growth evidence

The final allocation-only source reproduces the established combined-stack
128+128 evidence:

| Evidence | SHA-256 | Comparison |
| --- | --- | --- |
| 128-token frontier logits | `37fc78e2e0dc64dec160d4ea901348ffc6efb33cbdb53c805c24d11a42ffd8a7` | byte-identical to the immediate control |
| 128-token decode evidence | `16d915d7e6fc1ef4c5f550a416e7b0b18ae347da9205d1f29d3c92d31d6615f1` | byte-identical to the immediate control |
| 512-token frontier logits after in-process growth | `c27e7d8270207bdca5620a171dff7d08c6cc9929e3685f822a94996bba8480ac` | byte-identical to the immediate control |
| 512-token decode evidence after in-process growth | `75b0c84448617fc7539fe349682582433483f21e5f8a471c54fadb852819c521` | byte-identical to the immediate control |

The growth comparison is important: it proves that replacing a 128-row ID
buffer with a 384-row incremental-prefill buffer initializes the new storage
before the routed kernel reads it.

## Validation

| Started | Revision / experiment | Test command or lane | Result |
| --- | --- | --- | --- |
| 2026-07-28T23:40:25+02:00 | final state simplification | warning-clean `make ds4_test ds4-bench` | pass |
| 2026-07-28T23:40:42+02:00 | final candidate | Qwen 128 prefill plus 128 decode, forced cold SSD | pass; exact frontier/decode and unchanged SSD counters |
| 2026-07-28T23:41:59+02:00 | final candidate | bare `./ds4_test` | invalid invocation; attempted absent default model `ds4flash.gguf`, no code test ran |
| 2026-07-28T23:42:13+02:00 | final candidate | canonical `make model-free-test`, including Apple Metal kernels | pass |
| 2026-07-28T23:43:24+02:00 | final candidate | Qwen in-process 128-to-512 buffer growth | pass; all frontier/decode evidence byte-identical to control |

## Remaining promotion gates

- Stable-host Qwen and DeepSeek A/B/B/A at 2K/8K/32K, plus the second 32K
  routing/I/O prompt domain.
- DeepSeek 65,536/100K safety and Qwen near-262K endpoint coverage.
- Qualified GLM execution because the earlier common cache tranches still
  affect it; this Qwen-only change does not remove that combined-stack gate.
- Final combined-stack comparison against original `hebrus/main`, full
  merge-base diff review, active-handoff removal, and clean `make premerge`.
