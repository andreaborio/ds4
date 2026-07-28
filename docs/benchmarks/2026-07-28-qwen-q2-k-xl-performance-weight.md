# Qwen3.6 35B-A3B Q2_K_XL performance per weight

Date: 2026-07-28

Status: accepted implementation and performance decision; publication identity
pending the reviewed runtime commit and immutable artifact revision.

Decision: select the exact 12,290,632,032-byte Hebrus-native Q2_K_XL
ExpertMajor v2 representation. It is 40.93 percent smaller than the previous
20,808,566,880-byte Qwen affine4 artifact, reaches condition-matched
short-context decode parity with pinned llama.cpp, and passes resident/SSD
128/2K/8K/32K correctness. The old affine4 runtime path must be removed rather
than retained as compatibility.

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

The original `main` baseline is structurally unavailable because it cannot
admit this artifact. Incremental A/B evidence was retained for each accepted
optimization; unlike-condition affine4 results are not relabeled as a baseline.
The post-rebuild p128 pair is retained to expose the one-time Metal-library
cache effect rather than selecting the faster replica silently. The warm-cache
replica is comparable with the earlier final p128 rows; its prefill and decode
remain within 0.5 percent of the 04:11 cleanup run.

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

| Started | Mode / security-coding prompt | Prefill t/s | Decode t/s | TPOT p50 / p95 | Correctness / pressure |
| --- | --- | ---: | ---: | --- | --- |
| 2026-07-28T03:32:57+02:00 | resident p32K/n128 | 599.80 | 43.95 | 22.697 / 23.064 ms | evidence `85066e6f…`; 0 swaps |
| 2026-07-28T03:34:14+02:00 | SSD cold 1 GiB p32K/n128 | 611.94 | 23.11 | 43.358 / 45.352 ms | all 128 IDs match resident; same argmax; max abs `1.907e-5`, RMS `2.598e-6`; 0 swaps |

An earlier resident attempt was excluded because its evidence directory did not
exist. It is not used in either table or claim.

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

The DeepSeek and GLM rows are model-backed structural smoke, not throughput
promotion claims. Physical 16/24/32 GiB Qwen qualification remains outside this
64 GiB host.

## Publication boundary

The selected tokenizer metadata uses BOS 248044, EOS 248046, and padding
248055. The old affine4 artifact records padding 248044 and is intentionally
incompatible with the selected singular contract. Publication must:

1. remove the affine4 Qwen runtime path and its obsolete release documentation;
2. commit the minimum compatible runtime;
3. publish and independently hash the exact Q2_K_XL native artifact;
4. pin its immutable repository revision, bytes, hash, and runtime commit in
   the machine-readable Qwen release contract.

Until those steps complete, this document records an accepted implementation
decision, not an already published artifact identity.
