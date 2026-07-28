# Qwen3.6 35B-A3B Q2_K_XL performance per weight

Date: 2026-07-28

Status: accepted dual-profile runtime and performance decision; Q2_K_XL is
published as an opt-in Beta at immutable revision
`bdb363efaeb227bfd702c9145cb224fffa456891`, with 64 GiB/32K scope and
near-262K Stable/full-window qualification pending.

Decision: select the exact 12,290,632,032-byte Hebrus-native Q2_K_XL
ExpertMajor v2 representation. It is 40.93 percent smaller than the previous
20,808,566,880-byte Qwen affine4 artifact, reaches condition-matched
short-context decode parity with pinned llama.cpp, and passes resident/SSD
128/2K/8K/32K correctness. Preserve the published Affine4 profile and use one
Qwen graph/session pipeline for both formats. Keep only their physical weight
decoders specialized; remove the unqualified Affine2 experiment instead of
creating a third codec.

Supersedes: the temporary experiment conclusion that Q2_K_XL should replace
Affine4. It does not supersede the published Affine4 artifact or its immutable
release identity.

## Scope and identities

Host: Apple M5 Pro, 64 GiB unified memory, Apple Metal.

| Item | Identity |
| --- | --- |
| Canonical Q2_K_XL source bytes | 12,290,628,576 |
| Canonical Q2_K_XL source SHA-256 | `96b9c0af5c77a4ecaabe3983175112b5ece763261c1ece12b2494b692a70dad7` |
| Hebrus-native bytes | 12,290,632,032 |
| Hebrus-native SHA-256 | `30c22f70aff0f05986b517ee4ad8fef554a1b5aab6971c9ca09f999566d30143` |
| Embedded ExpertMajor payload SHA-256 | `ccc3fbc2405d1dd73f8ac15741b0277514de4f46b80818531297ea9ffa0c6a3c` |
| Routed layer classes | 36 × IQ2_XS/IQ3_XXS; 3 × IQ2_XS/IQ4_XS; 1 × IQ3_XXS/IQ4_XS |
| Published Affine4 bytes | 20,808,566,880 |
| Published Affine4 SHA-256 | `dd17266185833a9f05531ce366fd7284ddca1ed64aa3dcf06e321e8c72c9ea3d` |
| Pinned llama.cpp revision | `bf2c86ddc0685f580595954056c2e77ebabfab4f` |
| Prose prompt SHA-256 | `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f` |
| Security/coding prompt SHA-256 | `e7c1a2cadf781d274cc26bd251d532fe1b9e632080da97e3eb4684741e7cc308` |

External model-card performance and quality claims were not used. Candidate
files were pinned, hashed, inspected with the local GGUF reader, and measured
on the target machine.

## Representation decision

The preregistered deeper quality gate scored eight identical 4,096-token chunks
from each of Italian literature, security/code, and narrative corpora with the
same pinned llama.cpp evaluator.

| Artifact | Geometric-mean PPL | Delta vs Q4 control | Delta vs previous candidate | Exact GiB |
| --- | ---: | ---: | ---: | ---: |
| Q4 control | 4.5175 | baseline | N/A | 19.458 |
| UD-IQ2_M | 5.0455 | +11.69% | +11.69% vs Q4 | 10.731 |
| UD-Q2_K_XL | 4.6317 | +2.53% | -8.20% vs IQ2_M | 11.447 |

Q2_K_XL beat IQ2_M in all three domains. Relative to Q4 it cuts exact file
weight by 41.17 percent while giving up 2.53 percent geometric-mean PPL. This
gate selected an implementation candidate; the runtime matrices below made the
product decision.

## Runtime rationalization

The result is one Qwen implementation with two exact weight profiles, not two
parallel models:

| Shared for Affine4 and Q2_K_XL | Specialized by codec |
| --- | --- |
| Session/model state and graph topology | Complete tensor/profile admission |
| Gated DeltaNet, full attention, RoPE, and KV | Affine4 scale/bias decoding |
| Router, expert selection, cache keys, and output ordering | IQ2/IQ3/IQ4 and Q4/Q5/Q6/Q8 decoding |
| Resident/SSD planning, expert I/O, slabs, and long-prefill scheduling | Codec-specific matvec/grouped-MM primitive |

The profile is bound once from the logical tensor inventory, tokenizer
metadata, and ExpertMajor storage marker. Hot kernels do not branch per block.
The implementation boundary and its publication consequences are recorded in
[`ADR 0006`](../adr/0006-qwen-dual-weight-codecs.md).

## Accepted exact performance stack

The surviving changes preserve the qualified arithmetic or introduce only a
separately measured legal reduction order:

- exact Q5_K embedding and Q5_K/Q6_K dense decode matvecs;
- compact per-expert routed-IQ prefill work lists from 128 tokens;
- fixed-geometry Qwen GDN recurrence;
- fused full-attention Q/gate split and Q RMSNorm from 2K, plus the equivalent
  single-token fusion;
- one resident decode command-buffer split after layer 4;
- strict legacy GQA decode below 2K and F32 split-K GQA from 2K;
- one shared resident/SSD GDN Q/K pre-normalization order;
- type-aware routed-IQ SSD route-ready and selected-address admission.

There are no runtime enable flags for the accepted paths.

The largest isolated decode gain was the frontier-aware GQA decision:

| Started | Frontier / path | Prefill t/s | Decode t/s | TPOT p50 / p95 | Delta vs path control | Correctness |
| --- | --- | ---: | ---: | --- | ---: | --- |
| 2026-07-28T02:27:34+02:00 | p128 strict legacy oracle | 401.57 | 72.58 | 13.791 / 14.307 ms | +1.30% decode vs old reuse | same 128 greedy IDs |
| 2026-07-28T02:28:19+02:00 | p2K F32 split-K | 933.25 | 76.58 | 12.967 / 13.340 ms | +73.49% decode vs legacy | same 128 IDs; final max abs `9.18e-6`, RMS `1.86e-6` |
| 2026-07-28T02:28:36+02:00 | p2K strict legacy control | 931.60 | 44.14 | 22.627 / 23.045 ms | baseline | strict control |

The model-free split-K fixture matched its CPU oracle with maximum absolute
error `3.2e-9`.

## Affine4 final-stack A/B/B/A

The common Qwen changes were tested against authoritative `hebrus/main`
`b8251e465d3db77e2db85c1b1d3aeaa1fddccf46` on the same published Affine4
artifact. Every arm was a new resident process on the Apple M5 Pro 64 GiB,
AC power, warm cache, 65,536-token allocation, and 128 greedy decode tokens.
The prose prompt SHA-256 was
`f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`.
The main binary SHA-256 was `371afa47…`; the candidate binary was
`fc79103c…`. Every retained arm reported zero swapout, no competing inference
process, and the same resolved plan within its binary.

| Frontier | `main` mean prefill / decode t/s | Candidate mean prefill / decode t/s | Candidate TPOT p50 / p95 | Delta vs tested `main` prefill / decode | TPOT delta p50 / p95 | Max within-arm spread |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 497.64 / 56.47 | 519.33 / 59.39 | 16.821 / 17.306 ms | +4.36% / +5.17% | -4.93% / -5.86% | 2.58% |
| 2K | 1,663.64 / 59.27 | 1,897.02 / 61.85 | 16.137 / 16.425 ms | +14.03% / +4.35% | -3.80% / -6.23% | 1.34% |
| 8K | 1,419.95 / 52.79 | 1,579.05 / 55.02 | 18.103 / 18.538 ms | +11.20% / +4.22% | -3.33% / -7.30% | 2.26% |
| 32K | 730.00 / 36.27 | 789.85 / 38.18 | 26.113 / 26.663 ms | +8.20% / +5.27% | -4.10% / -9.18% | 2.62% |

The complete chronological cohort is retained below. Deltas in the `main`
column use the two-arm `main` mean for that frontier. “Previous” means the
previous arm of the same implementation and frontier.

| Started | Frontier / arm | Prefill t/s | Decode t/s | TPOT p50 / p95 | Delta vs tested `main` mean, prefill / decode | Delta vs previous comparable | Correctness |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| 2026-07-28T16:02:23+02:00 | 8K A1 `main` | 1,417.13 | 52.75 | 18.712 / 20.086 ms | -0.20% / -0.08% | N/A; first 8K control | retained |
| 2026-07-28T16:02:48+02:00 | 8K B1 candidate | 1,561.17 | 54.96 | 18.113 / 18.546 ms | +9.95% / +4.11% | N/A; first 8K candidate | exact frontier and decode evidence |
| 2026-07-28T16:03:29+02:00 | 8K B2 candidate | 1,596.93 | 55.08 | 18.093 / 18.529 ms | +12.46% / +4.34% | +2.29% / +0.22% vs B1 | byte-identical to B1 and `main` |
| 2026-07-28T16:03:54+02:00 | 8K A2 `main` | 1,422.77 | 52.83 | 18.740 / 19.907 ms | +0.20% / +0.08% | +0.40% / +0.15% vs A1 | byte-identical control |
| 2026-07-28T16:04:30+02:00 | 32K A1 `main` | 728.19 | 36.72 | 26.951 / 28.113 ms | -0.25% / +1.25% | N/A; first 32K control | retained |
| 2026-07-28T16:06:52+02:00 | 32K B1 candidate | 800.20 | 38.27 | 26.073 / 26.513 ms | +9.62% / +5.53% | N/A; first 32K candidate | exact frontier and decode evidence |
| 2026-07-28T16:07:59+02:00 | 32K B2 candidate | 779.50 | 38.08 | 26.153 / 26.812 ms | +6.78% / +5.01% | -2.59% / -0.50% vs B1 | byte-identical to B1 and `main` |
| 2026-07-28T16:09:09+02:00 | 32K A2 `main` | 731.80 | 35.81 | 27.506 / 30.604 ms | +0.25% / -1.25% | +0.50% / -2.48% vs A1 | byte-identical control |
| 2026-07-28T16:10:50+02:00 | 128 A1 `main` | 503.23 | 56.43 | 17.694 / 18.442 ms | +1.12% / -0.06% | N/A; first 128 control | retained |
| 2026-07-28T16:11:11+02:00 | 128 B1 candidate | 526.03 | 59.40 | 16.795 / 17.288 ms | +5.71% / +5.20% | N/A; first 128 candidate | same 128 greedy IDs |
| 2026-07-28T16:15:14+02:00 | 128 B2 candidate | 512.62 | 59.37 | 16.847 / 17.324 ms | +3.01% / +5.15% | -2.55% / -0.05% vs B1 | byte-identical to B1 |
| 2026-07-28T16:15:32+02:00 | 128 A2 `main` | 492.05 | 56.50 | 17.693 / 18.323 ms | -1.12% / +0.06% | -2.22% / +0.12% vs A1 | byte-identical control |
| 2026-07-28T16:15:55+02:00 | 2K A1 `main` | 1,652.59 | 59.20 | 16.777 / 17.577 ms | -0.66% / -0.11% | N/A; first 2K control | retained |
| 2026-07-28T16:16:20+02:00 | 2K B1 candidate | 1,889.78 | 61.86 | 16.134 / 16.425 ms | +13.59% / +4.38% | N/A; first 2K candidate | exact frontier and decode evidence |
| 2026-07-28T16:16:39+02:00 | 2K B2 candidate | 1,904.25 | 61.83 | 16.139 / 16.425 ms | +14.46% / +4.33% | +0.77% / -0.05% vs B1 | byte-identical to B1 and `main` |
| 2026-07-28T16:17:07+02:00 | 2K A2 `main` | 1,674.68 | 59.33 | 16.771 / 17.456 ms | +0.66% / +0.11% | +1.34% / +0.22% vs A1 | byte-identical control |

At 128, the candidate deliberately uses strict GQA rather than `main`'s
approximate reuse path. All 128 greedy IDs and the final argmax match; final
logits differ by maximum absolute `6.68e-6`, RMS `1.357e-6`, cosine similarity
`0.9999999999999257`, and identical top-1/5/10/100 sets. The strict path is the
closer oracle, so this is explained numerical improvement rather than
unattributed drift. At 2K and above the retained evidence is byte-identical.

## Condition-matched llama.cpp decode

The comparison used the same canonical Q2_K_XL source, exact 276-token binary
prompt, F32 K/V, temperature zero, and generic quant arithmetic
(`GGML_METAL_TENSOR_DISABLE=1` for llama.cpp). llama.cpp reported 71.99 and
72.01 eval t/s, a 72.00 mean. The final Hebrus run reported 74.22 generated
t/s and retained evidence SHA-256
`317894d69fcdcb6af5005c7e42d223233fbe069372b8bb7713e845e68f51027f`.

The nominal Hebrus delta is +3.08 percent. The runtimes do not define timing
identically: llama.cpp reports 255 target evaluations excluding its first
sampled token, while Hebrus divides 256 visible tokens by the wall time for 255
target evaluations plus sampling and emission. This supports parity, not a
sub-percent cross-runtime speed claim.

The much higher default llama.cpp prefill numbers use Metal TensorOps and a
different precision/reduction contract. Every attempted Hebrus routed-IQ
TensorOps arm changed deterministic output and was removed. The exact generic
prefill discrepancy was separately closed under matched arithmetic.

## Final resident matrix

Every row is an isolated process with 128 greedy decode tokens. The p128/p2K/p8K
rows use the narrative prompt. That file is shorter than 32K Qwen tokens, so the
p32K row uses the prose prompt and is not compared with the shorter routing
distribution.

| Started | Frontier / prompt | Prefill t/s | Decode t/s | TPOT p50 / p95 | Delta vs tested `main` | Delta vs previous comparable | Evidence |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| 2026-07-28T02:30:32+02:00 | p128/n128, narrative | 407.62 | 72.72 | 13.737 / 14.293 ms | N/A; `main` rejects Q2_K_XL | N/A; first final p128 cohort | `af8b2ffa…` |
| 2026-07-28T02:30:48+02:00 | p2K/n128, narrative | 934.49 | 76.59 | 12.970 / 13.350 ms | N/A; `main` rejects Q2_K_XL | N/A; first final p2K cohort | `03a5887f…` |
| 2026-07-28T02:31:02+02:00 | p8K/n128, narrative | 872.92 | 66.43 | 14.955 / 15.275 ms | N/A; `main` rejects Q2_K_XL | N/A; first final p8K cohort | `01afff55…` |
| 2026-07-28T02:33:09+02:00 | p32K/n128, prose | 568.21 | 42.35 | 22.950 / 26.326 ms | N/A; `main` rejects Q2_K_XL | N/A; first final p32K cohort | `541a3b2c…` |
| 2026-07-28T04:11:52+02:00 | p128/n128 after affine-path removal, narrative | 403.49 | 72.95 | 13.680 / 14.383 ms | N/A; `main` rejects Q2_K_XL | -1.01% prefill / +0.32% decode vs 02:30 p128 | `af8b2ffa…`; byte-identical output |
| 2026-07-28T04:33:36+02:00 | p128/n128 after final exact-layout admission cleanup, first process after Metal rebuild | 357.53 | 72.92 | 13.630 / 14.055 ms | N/A; `main` rejects Q2_K_XL | -11.39% prefill / -0.04% decode vs 04:11 p128 | `af8b2ffa…`; byte-identical output; first run paid the rebuilt Metal-library cache cost |
| N/A; immediate isolated replica after the 04:33:36 arm | p128/n128 after final exact-layout admission cleanup, warm Metal cache | 405.44 | 73.26 | 13.645 / 14.047 ms | N/A; `main` rejects Q2_K_XL | +0.48% prefill / +0.42% decode vs 04:11 p128 | `af8b2ffa…`; byte-identical output; 0 swaps |
| 2026-07-28T16:34:39+02:00 | p8K/n128 after Affine2 removal, prose | 869.68 | 66.31 | 15.050 / 15.295 ms | N/A; `main` rejects Q2_K_XL | N/A; earlier p8K used the narrative prompt | exact artifact hash; 0 swaps; no contamination |
| 2026-07-28T17:16:54+02:00 | p8K/n128, final binary, prose | 874.68 | 66.30 | 14.972 / 15.270 ms | N/A; `main` rejects Q2_K_XL | +0.57% prefill / -0.02% decode vs 16:34 p8K | byte-identical evidence; 0 swaps; no contamination |
| 2026-07-28T17:17:29+02:00 | p32K/n128, final binary, cooled arm 1, prose | 581.50 | 43.42 | 22.952 / 23.409 ms | N/A; `main` rejects Q2_K_XL | +2.34% prefill / +2.53% decode vs 02:33 p32K | evidence `541a3b2c…`; 0 swaps; no contamination |
| 2026-07-28T17:29:10+02:00 | p32K/n128, final binary, cooled arm 2, prose | 588.13 | 43.63 | 22.814 / 23.330 ms | N/A; `main` rejects Q2_K_XL | +1.14% prefill / +0.48% decode vs cooled arm 1 | byte-identical to cooled arm 1; 0 swaps; no contamination |

The original `main` baseline is structurally unavailable because it cannot
admit this artifact. Incremental A/B evidence was retained for each accepted
optimization; unlike-condition affine4 results are not relabeled as a baseline.
The post-rebuild p128 pair is retained to expose the one-time Metal-library
cache effect rather than selecting the faster replica silently. The warm-cache
replica is comparable with the earlier final p128 rows; its prefill and decode
remain within 0.5 percent of the 04:11 cleanup run.

The two cooled final-binary p32K arms average 584.82 prefill t/s, 43.53 decode
t/s, and 22.883/23.370 ms TPOT p50/p95. Their prefill and decode spreads are
1.14 and 0.48 percent, respectively, and their logits and generated evidence
are byte-identical. This cooled mean is the qualified final-binary p32K result.

Three nearby p32K arms are retained only as invalidation evidence:

| Started | Prefill / decode t/s | Delta vs previous identical arm | Invalidation |
| --- | ---: | ---: | --- |
| 2026-07-28T16:35:05+02:00 | 550.18 / 42.92 | N/A; first post-100K arm | Host not isolated after sustained 65K/100K work |
| 2026-07-28T16:36:38+02:00 | 505.62 / 38.07 | -8.10% / -11.30% | Confirmed hidden application load: later inspection found a VS Code Node helper at 99% CPU and a Spark helper at 94% although both visible parents were suspended |
| 2026-07-28T17:18:49+02:00 | 372.75 / 30.13 | -35.90% / -30.61% vs cooled arm 1 | Helpers were suspended, but this immediate back-to-back arm followed a sustained p32K run; host power/thermal state was not comparable |

All three used the same exact artifact and plan, retained byte-identical output
evidence, reported zero swapout, and saw no competing inference process. The
first two expose why parent-only application suspension is insufficient. The
third isolates inference-induced power/thermal drift: after an idle cooldown,
the 17:29 arm returned to within 1.14/0.48 percent of the 17:17 arm. None of
these invalid rows contributes to the qualified mean.

## Final cold-SSD 1 GiB matrix

The initial routed-IQ p2K SSD failure had two causes:

1. route-ready probed hard-coded Q4 types, applied the Q4 760-row cap, and
   returned no I/O ticket for a 2,048-row IQ batch;
2. after that was fixed, SSD still used the legacy GDN Q/K normalization order,
   causing material final-output drift from resident.

Passing actual layer types and sharing the accepted pre-normalized GDN order
made p2K post-prefill logits bit-identical to resident. A controlled
pure-prefill pair improved from 759.09 to 830.96 t/s (+9.47 percent). The old
output is correctness-invalid and is not a normal performance baseline.

All final SSD token sequences exactly match their corresponding resident rows:

| Started | Frontier / prompt | Prefill t/s | Decode t/s | TPOT p50 / p95 | Delta vs matching resident prefill / decode | Correctness |
| --- | --- | ---: | ---: | --- | --- | --- |
| 2026-07-28T03:15:46+02:00 | p2K/n128, narrative | 829.89 | 34.49 | 28.737 / 31.766 ms | -11.20% / -54.97% | all 128 IDs; max abs `7.868e-6`, RMS `1.639e-6` |
| 2026-07-28T03:17:27+02:00 | p128/n128, narrative | 209.00 | 34.36 | 28.819 / 31.981 ms | -48.73% / -52.75% | all 128 IDs; max abs `1.049e-5`, RMS `1.849e-6` |
| 2026-07-28T03:17:57+02:00 | p8K/n128, narrative | 839.76 | 29.97 | 33.082 / 36.130 ms | -3.80% / -54.88% | all 128 IDs; max abs `7.870e-6`, RMS `1.364e-6` |
| 2026-07-28T03:18:33+02:00 | p32K/n128, prose | 634.80 | 28.00 | 35.025 / 41.537 ms | +11.72% / -33.88% | all 128 IDs; max abs `6.920e-6`, RMS `1.496e-6` |

The p32K prefill reversal is thermally sensitive and is not generalized as an
SSD speed claim. Decode is the remaining SSD performance gap.

A strict-stack p8K pure-prefill run started at
2026-07-28T03:28:40+02:00 with `macro_prefill`, `phase_budget`, `layer_pin`,
`io_overlap`, `expert_pack`, and `gqa_reuse` all expected and enabled. It
completed at 839.21 t/s, only -0.07 percent versus the normal p8K row.

## Second-domain 32K diagnostic

Routing, expert-cache, and expert-I/O changes require a different prompt
domain. Throughput is not compared or averaged across domains.

| Started | Profile / mode / security-coding prompt | Prefill t/s | Decode t/s | TPOT p50 / p95 | Correctness / pressure |
| --- | --- | ---: | ---: | --- | --- |
| 2026-07-28T03:32:57+02:00 | Q2_K_XL resident p32K/n128 | 599.80 | 43.95 | 22.697 / 23.064 ms | evidence `85066e6f…`; 0 swaps |
| 2026-07-28T03:34:14+02:00 | Q2_K_XL SSD cold 1 GiB p32K/n128 | 611.94 | 23.11 | 43.358 / 45.352 ms | all 128 IDs match resident; same argmax; max abs `1.907e-5`, RMS `2.598e-6`; 0 swaps |
| 2026-07-28T16:31:38+02:00 | Affine4 final shared runtime, resident p32K/n128 | 575.48 | 33.67 | 28.436 / 38.259 ms | N/A vs prose domain; exact artifact hash; 0 swaps; no contamination |

An earlier resident attempt was excluded because its evidence directory did not
exist. It is not used in either table or claim.

## Affine4 extended-context stability

The shared attention/KV changes require isolated 65,536- and 100,000-token
lanes. These are candidate-only safety/correctness arms, not speed comparisons
with `main`.

| Started | Frontier / allocation / decode | Prefill t/s | Decode t/s | TPOT p50 / p95 | Delta vs tested `main` | Delta vs previous comparable | Safety |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| 2026-07-28T16:17:41+02:00 | 65,536 / 131,072 / 128 | 277.09 | 23.14 | 41.586 / 56.578 ms | N/A; no matching `main` arm | N/A; first 65K final-stack arm | 0 swaps; pressure minimum 41%; no contamination |
| 2026-07-28T16:22:17+02:00 | 100,000 / 131,072 / 128 | 190.74 | 18.63 | 52.757 / 64.636 ms | N/A; no matching `main` arm | N/A; 65K is a different frontier | 0 swaps; pressure minimum 40%; no contamination |

The validated model metadata declares a 262,144-token window. An Affine4
endpoint arm started at 2026-07-28T17:36:13+02:00 with prompt 262,015,
allocation 262,144, and 128 requested decode tokens. It was owner-interrupted
after roughly one hour to finish the merge; the runner had not emitted a CSV
row, logits, or decode evidence, so the arm contributes no performance or
correctness result. The exact end timestamp was not captured by the interrupted
runner. The Q2_K_XL endpoint arm was not started.

| Started | Profile / frontier | Result | Qualification effect |
| --- | --- | --- | --- |
| 2026-07-28T17:36:13+02:00 | Affine4 resident, 262,015 / allocation 262,144 / requested n128 | ABORTED by owner before prefill completion; no CSV result, logits, or decode evidence | No endpoint evidence |
| N/A; not started | Q2_K_XL resident endpoint | NOT RUN | No endpoint evidence |

Under the endpoint-pending merge rule in `CONTRIBUTING.md`, this permits the
additive implementation and a separately labelled Beta artifact without
changing the Stable Affine4 download or making a full-window claim. Both
endpoint arms remain blocking before Q2_K_XL Stable/full-window promotion or
the next release qualification that advertises the model's full window.

## Speculative decoding decision

Zero-weight prompt lookup was rejected and completely removed.

| Started | Mode | Generated | Decode t/s | Delta vs ordinary decode | Correctness |
| --- | --- | ---: | ---: | ---: | --- |
| 2026-07-28T01:11:51+02:00 | resident ordinary decode | 256 | 62.98 | baseline | production evidence |
| 2026-07-28T00:59:52+02:00 | exact sequential verifier, draft ≤8 | 256 | 40.67 mean | -35.35% | exact output |
| 2026-07-28T01:03:55+02:00 | SSD ordinary decode | 128 | 32.075 mean | baseline | SSD control |
| 2026-07-28T01:03:55+02:00 | SSD exact sequential verifier | 128 | 25.18 mean | -21.50% | exact output |

A faster macro verifier reached 53.30 t/s but left final-logit maximum absolute
error 0.681588 and RMS 0.12853. Identical sampled IDs were insufficient. The
feature bit, n-gram index, verifier, telemetry, dispatch, and multi-row
persistent workspace were removed. Qwen's source MTP tail is 899,008,512 bytes
and is excluded from the selected 40-layer artifact; no separately qualified
draft model exists.

## Rejected optimization families

- Affine2/group-32 plus IQ down was rejected as a product profile. Its
  14,171,777,376-byte artifact is 15.31% larger than Q2_K_XL, had no matching
  three-domain quality qualification, and would require a third storage marker
  and permanent kernels. Its first 32K tile experiment was invalid: identical
  arms varied by 69–122%. A cooled A/B/B/A then showed that widening its
  paired route tile from N16 to N32 reduced 32K prefill from a 616.245 t/s mean
  to 582.425 t/s (-5.49%) with byte-identical logits. The format, kernels,
  admission, converter support, tests, and selectors were removed.
- Routed-IQ TensorOps: up to +45.81 percent p128 prefill, but deterministic
  decode diverged at the sixth generated token.
- Dense Q4_K/Q5_K/Q6_K TensorOps: warm throughput improved, but the p2K
  frontier diverged.
- Fused routed-IQ gate/up SwiGLU and sum variants: throughput regressions.
- GDN decay precompute: exact but -2.19 percent at p512 and -1.04 percent at
  p2K.
- Direct Q6_K dense dispatch: +0.02/-0.12 percent, within noise.
- Additional command-buffer splits and unretained buffers: neutral or slower.

All rejected runtime switches, kernels, selectors, and scaffolding were
removed. Permanent numeric fixtures remain only where they cover the selected
artifact.

## Validation

| Started | Revision / experiment | Test lane | Result |
| --- | --- | --- | --- |
| 2026-07-28T03:27:20+02:00 | final Q2_K_XL stack | `build/metal-arm64/bin/ds4_test --metal-kernels` | PASS; complete Metal lane, split-K GQA max error `3.2e-9` |
| 2026-07-28T03:27:31+02:00 | final Q2_K_XL stack | `make model-free-test` | PASS |
| 2026-07-28T03:40:10+02:00 | DeepSeek V4 Flash v2 smoke | SSD ctx128/n8 | PASS; coherent continuation, 0 swaps |
| 2026-07-28T03:40:28+02:00 | GLM 5.2 v2 smoke | strict full-graph SSD ctx128/n8 | PASS; 0 swaps |
| 2026-07-28T09:04:03+02:00 | exact-layout admission and IQ SSD fixture cleanup | `make model-free-test` | PASS; Qwen top-8 cache fixture uses IQ2_XS/IQ3_XXS unequal component geometry; 16/24/32/64 GiB residency matrix passes |
| 2026-07-28T09:05:39+02:00 | final converter/store implementation | `python3 gguf-tools/ds4-expert-major.py verify SOURCE NATIVE` | PASS; source SHA `96b9c0af…`, payload SHA `ccc3fbc2…`, all 40 layer records byte-exact |
| 2026-07-28T17:07:25+02:00 | reviewed final tree | DeepSeek V4 Flash v2 SSD smoke, ctx128/n8 | PASS; coherent continuation, 0 swaps |
| 2026-07-28T17:08:04+02:00 | reviewed final tree | GLM 5.2 v2 strict full-graph SSD smoke, ctx128/n8 | PASS; coherent continuation, 0 swaps |
| 2026-07-28T17:12:38+02:00 | reviewed final converter/store | `python3 gguf-tools/ds4-expert-major.py verify SOURCE NATIVE` | PASS; source SHA `96b9c0af…`, payload SHA `ccc3fbc2…`, all 40 layer records byte-exact |
| 2026-07-28T17:35:53+02:00 | reviewed dual-profile runtime | `test_qwen_session` and `test_ssd_residency` | PASS; Q2_K_XL and Affine4 profile/cache geometry accepted |
| 2026-07-28T18:44:15+02:00 | reviewed dual-profile fixtures | `make qwen-metadata-test` | PASS; both tokenizer/tensor/store profiles and crossed-profile rejection cases |
| 2026-07-28T18:45:19+02:00 | reviewed merge candidate | `make premerge` | PASS; repository, docs, generated tables, build isolation, model-free Metal, install, and diff checks |

The DeepSeek and GLM rows are model-backed structural smoke, not throughput
promotion claims. Physical 16/24/32 GiB Qwen qualification remains outside this
64 GiB host.

## Publication boundary

The two upstream artifacts preserve different tokenizer metadata. Affine4 uses
padding/BOS 248044 and its 7,764-byte chat template. Q2_K_XL uses BOS 248044,
EOS 248046, padding 248055, and its 8,057-byte template. Runtime admission binds
each tokenizer contract to its exact tensor and storage profile; neither is a
fallback for the other.

The Beta publication completed the non-endpoint identity gates:

1. the reviewed dual-profile runtime merged while preserving Affine4;
2. the exact Q2_K_XL native artifact was published and independently hashed;
3. immutable revision
   `bdb363efaeb227bfd702c9145cb224fffa456891`, bytes, hash, and runtime commit
   were pinned in the machine-readable Qwen release contract without deleting
   the Affine4 identity;
4. the separate `download_model.sh qwen-q2-beta` selector was added rather than
   changing what `qwen-v2` returns;
5. the public Beta boundary was restricted to nonrecommended, minimum 64 GiB,
   qualified through 32K, and explicitly not full-window qualified.

Before Stable/full-window promotion, complete and retain valid Affine4 and
Q2_K_XL near-262K endpoint arms, then update the release contract and public
claims from that evidence. Affine4 remains the Stable/recommended release.
