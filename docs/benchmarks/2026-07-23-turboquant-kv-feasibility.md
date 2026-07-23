# TurboQuant KV cross-engine feasibility — 2026-07-23

Status: active Qwen research evidence; measured on the 32 GiB M1 Pro lane,
not promoted as a production speed or universal-RAM claim.

## Decision

Proceed with a native DS4 packed-KV investigation, beginning with Qwen full
attention and retaining explicit DeepSeek/GLM surface ownership from the first
change. Do not import `turbovec` as a runtime dependency and do not enable a
packed production cache on planning arithmetic alone.

The durable design and qualification contract are in
[`../architecture/KV_QUANTIZATION.md`](../architecture/KV_QUANTIZATION.md).

## Source Audit

The reviewed `turbovec` main identity was
`1e7200cfd8f26c92ce2855652db64bc7f85bc039`. Its public Rust/Python product is
an ANN/vector-search index, not an online KV-cache implementation. Search
initialization retains packed and blocked search layouts; serialized-size
compression is therefore not a runtime-memory result that transfers to DS4.

The algorithm audit also covered the TurboQuant paper, merged vLLM Qwen/hybrid
serving work, and upstream long-context/speculative bugs. The resulting DS4
candidate uses a deterministic WHT/Lloyd-Max key reference and uniform 4-bit
value reference, but remains unqualified for all three DS4 production models.

## Checked Allocation Baseline

At 32K Qwen's full-attention K/V buffers account for exactly 1.250 GiB:

| Surface | Vector stride | 32K total |
| --- | ---: | ---: |
| F32 key, 10 layers * 2 heads | 1,024 bytes | 640.000 MiB |
| F32 value, 10 layers * 2 heads | 1,024 bytes | 640.000 MiB |
| TQ4 v1 key payload + F16 norm | 130 bytes | 81.250 MiB |
| TQ4 v1 value payload + F16 scale/min | 132 bytes | 82.500 MiB |

The packed calculation totals 163.750 MiB (0.160 GiB), or 1.090 GiB less than
the F32 K/V allocation. It excludes allocator rounding, graph state outside
K/V, transient conversion, and attention scratch.

Cross-engine planner fixtures also lock:

- DeepSeek V4 Flash raw-ring example: 43 layers, 4,224 rows, width 512, F32 =
  371,982,336 bytes;
- GLM 5.2 32K `kv_lora` surface alone: 78 layers, width 512, F16 =
  2,617,245,696 bytes;
- complete GLM 32K compact-cache formula, including `k_rope` and the 21
  indexer layers = 3,120,562,176 bytes (2.906 GiB).

These numbers establish testable allocation geometry only. DeepSeek raw MLA
and GLM `kv_lora` do not inherit Qwen K/V semantics.

## Model-Free And Metal Evidence

The scalar reference passes a strict C99 `-Wall -Wextra -Werror` build and
verifies:

- checked invalid/overflow geometry;
- exact Qwen, DeepSeek, and GLM fixture totals;
- deterministic TQ4 key packing;
- zero-vector behavior;
- key reconstruction cosine greater than 0.985 on the fixed fixture;
- normalized value RMSE below 0.04;
- generated Lloyd-Max centroid provenance through a `--check` gate.

The Qwen Metal research tranche now also verifies:

- graph allocation at exactly 130 key bytes and 132 value bytes per
  256-dimensional KV-head vector;
- direct packed stores both at cache origin and at a non-zero continuation
  frontier;
- Metal key encoding against the scalar reconstruction and byte-identical
  uniform value encoding;
- causal packed-cache prefill and decode against attention computed from the
  scalar-decoded packed rows;
- no global F32 dequantized cache in the persistent graph.

`ds4_test --metal-kernels` passes on the 64 GiB M5 Pro model-free host. The
fixture crosses both the 64-key staging tile and the 2,048-key split frontier.
All retained decode strategies (`serial`, `parallel`, `split`, `reuse8`, and
F16-staged `flash`) pass against attention over the scalar-decoded cache.
Maximum absolute errors are `5.16e-06` for the F16-staged path and at most
`3.49e-09` for the direct packed paths.

Every model-free command except the Expert Store probe also passes. The full
`make model-free-test` target is currently blocked by a pre-existing starting
tree failure: `tests/test_expert_store.c` references the absent
`DS4_EXPERT_STORE_GLM_AFFINE2_GROUP_SIZE`. Neither that file nor the Expert
Store contract is changed by this branch. This is not waived; the complete
gate must be rerun after the owning change repairs or rebases that test.

## Hardware Lanes

Model-backed work must not run on the 64 GiB development host while another
thread owns its inference lane.

- 16 GiB: physical M1 Pro on the local LAN, SSD-streaming qualification.
- 32 GiB: physical Apple Silicon host through Tailscale, resident/AUTO
  qualification; exact SoC is recorded only after login.

Before every cohort, record host identity, power/thermal state, memory
pressure, swapout counter, artifact SHA-256, source SHA, resolved runtime plan,
and absence of another DS4 process. Each host runs one model process at a time.

The operational endpoint and access state remain in the active handoff rather
than this durable record.

## 32 GiB M1 Pro Exploratory Cohorts

The reachable Tailscale lane is an Apple M1 Pro with 32 GiB unified memory.
It ran on AC with zero swap and no competing inference process. The Qwen
artifact used for these cohorts is `20,808,566,880` bytes with SHA-256
`dd17266185833a9f05531ce366fd7284ddca1ed64aa3dcf06e321e8c72c9ea3d`.
Every A/B arm used a fresh process and the same binary within its cohort.

| Context / strategy | Prefill tok/s | Decode tok/s | Task footprint |
| --- | ---: | ---: | ---: |
| 2K F32 baseline | 387.09 | 27.49 | 2.28 GiB |
| 2K TQ4 serial | 166.82 | 11.78 | 2.13 GiB |
| 2K TQ4 parallel | 166.78 | 23.40 | 2.13 GiB |
| 2K TQ4 + Flash prefill + `reuse8` decode | 388.07 | 17.27 | not retained by that runner |
| 2K TQ4 + Flash prefill + direct split-K | 389.21 | 27.02 | 2.14 GiB |
| 8K F32 baseline | 335.88 | 25.15 | 2.88 GiB |
| 8K TQ4 direct split-K | 336.34 | 23.11 | 2.33 GiB |
| 8K TQ4 paired-nibble split-K | 336.45 | 23.85 | 2.33 GiB |
| 8K TQ4 pre-rotated-query split-K | not separately material | 23.89 | 2.33 GiB |
| 8K TQ4 F16-staged Flash decode | 336.03 | 23.80 | 2.34 GiB |

The clean 2K direct comparison measured about 140 MiB lower task footprint,
with prefill `+0.55%` and decode `-1.71%`. At 8K the persistent-cache saving
grew to about 550 MiB, prefill was effectively flat, and direct decode
remained about 5–8% slower depending on the retained implementation. The
`reuse8` result is specifically an M1 Pro result: its 512-thread groups and
barriers can lose there while still being a plausible M5 policy candidate.
It therefore remains selectable rather than being deleted mid-campaign.

The quantized 2K logits retained all top five entries, 19/20 of the top 20,
58/64 of the top 64, and 97/100 of the top 100. RMSE was `0.19499`, MAE
`0.15565`, maximum absolute difference `1.09785`, and greedy generation first
diverged at token 27 of 128. The 128-token exploratory run first diverged at
token 34 and had logit RMSE around `0.230`. This is explained lossy-format
movement, not permission to promote: corpus continuation-NLL evidence is still
required.

These cohorts establish a real memory reduction and expose an M1 decode-cost
tradeoff. They do not decide the final dispatch. In particular, no strategy
will be removed until the 16 GiB LAN lane and the M5 lane have run the same
selectable implementations.

### Single-binary strategy confirmation

After restoring every implementation behind `DS4_QWEN_TQ4_DECODE`, the 32 GiB
host repeated baseline plus all six choices from remote source
`1d3972eca07a76fdd9ffc3e3deeb62587c6e7581` and binary SHA-256
`5481339d8be43b587f5020a37b82ebcde1f892d98d6e65fd1d30a4c3015f1b5d`.
The process, model, allocation, and runner were fresh for every arm.

| Context | Strategy | Prefill tok/s | Decode tok/s | Decode vs F32 | Task footprint |
| --- | --- | ---: | ---: | ---: | ---: |
| 2K | F32 baseline | 388.53 | 27.45 | control | 2.28 GiB |
| 2K | TQ4 `serial` | 389.35 | 11.78 | -57.09% | 2.14 GiB |
| 2K | TQ4 `parallel` | 388.20 | 24.21 | -11.80% | 2.14 GiB |
| 2K | TQ4 `split` | 386.79 | 27.25 | -0.73% | 2.14 GiB |
| 2K | TQ4 `reuse8` | 386.98 | 17.24 | -37.19% | 2.14 GiB |
| 2K | TQ4 `flash` | 389.05 | 26.82 | -2.30% | 2.14 GiB |
| 2K | TQ4 `auto` | 388.04 | 27.50 | +0.18% | 2.14 GiB |
| 8K | F32 baseline | 335.89 | 25.13 | control | 2.88 GiB |
| 8K | TQ4 `serial` | 336.54 | 4.29 | -82.93% | 2.34 GiB |
| 8K | TQ4 `parallel` | 337.64 | 16.21 | -35.50% | 2.34 GiB |
| 8K | TQ4 `split` | 336.78 | 23.83 | -5.17% | 2.34 GiB |
| 8K | TQ4 `reuse8` | 336.49 | 7.88 | -68.64% | 2.34 GiB |
| 8K | TQ4 `flash` | 337.27 | 24.36 | -3.06% | 2.33 GiB |
| 8K | TQ4 `auto` | 337.23 | 24.41 | -2.87% | 2.34 GiB |

Every arm completed with zero swapout delta and no competing process.
System-wide free-memory pressure stayed at or above 24% for the 8K F32
control and 26% for the TQ4 arms. Within each context all TQ4 strategies
produced byte-identical logits artifacts: strategy selection changes execution
only, not the quantized-cache result.

The table also explains the M1 degradation. F32 attention reads a
compute-ready cache; TQ4 saves persistent bandwidth and allocation but must
unpack nibbles, load F16 metadata, reconstruct values, and perform the key
rotation contract during every attention use. `serial` and `reuse8` add
barrier/occupancy costs that dominate on M1. Split-K and Flash expose enough
parallelism to recover most, but not all, of that cost at 8K. M5 must be
measured rather than inferred because its tensor/Metal capabilities and
scheduling balance differ materially from this pre-M5 lane.

### 32K admission and retained-strategy frontier

The F32 resident control could not be run safely at 32K on this 32 GiB host.
With capacity 33,024, resident preflight calculated 25.23 GiB required against
the profile's fixed 24.96 GiB working-set budget and rejected the request
before model loading. A deliberately over-allocated 65,536-capacity attempt
was also rejected (26.47 GiB required). Both failures had normal initial
pressure, zero swap, and no inference allocation; they are admission evidence,
not failed inference runs.

TQ4 resident was admitted at capacity 33,024 and completed every retained
strategy:

| Strategy | Prefill tok/s | Decode tok/s | Task footprint | Pressure minimum |
| --- | ---: | ---: | ---: | ---: |
| TQ4 `auto` | 262.22 | 17.08 | 2.84 GiB | 23% |
| TQ4 `flash` | 262.45 | 17.10 | 2.84 GiB | 23% |
| TQ4 `split` | 262.55 | 15.80 | 2.84 GiB | 23% |
| TQ4 `parallel` | 262.63 | 6.90 | 2.84 GiB | 23% |
| TQ4 `reuse8` | 262.27 | 2.49 | 2.84 GiB | 23% |
| TQ4 `serial` | 262.50 | 1.21 | 2.84 GiB | 23% |

Every 32K arm completed with zero swapout delta, no process contamination,
and the same full-logit artifact hash. This is not a same-plan F32/TQ4 speed
comparison because no F32 resident arm was admitted. It does establish that
the packed cache changes a 32K resident request from rejected to runnable on
this host. It also preserves the M1 execution-policy evidence: Flash wins
among the retained strategies at the long frontier.

### 65K and 100K functional frontiers

The `auto`/Flash TQ4 resident path completed both required long-context
frontiers with 128 greedy decode tokens:

| Frontier | Capacity | Prefill tok/s | Decode tok/s | Task footprint | Pressure minimum | Wired peak | Swapout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 65,536 | 65,792 | 201.79 | 12.13 | 3.56 GiB | 19% | 24.996 GiB | 0 |
| 100,000 | 100,256 | 161.81 | 9.43 | 3.12 GiB | 16% | 25.970 GiB | 0 |

Both produced complete logits and decode-evidence artifacts without an
out-of-bounds failure, changed resolved plan, concurrent inference process, or
swap. They are functional exploratory evidence only: their pressure minima
fell below the 20% launch threshold, so neither can promote the candidate.
The result is nevertheless useful for the original feasibility question:
packed resident storage remains bounded enough to execute 100K on the 32 GiB
M1 Pro, whereas F32 resident is already rejected at 32K.

## 16 GiB LAN Status

The 16 GiB M1 Pro is identified as `macbookpro.lan` / `192.168.1.212`.
The host still answers ICMP and is present in ARP, but TCP/22 times out. A
magic packet did not restore Remote Login. The earlier direct-link endpoint
`169.254.83.36`, previously used with the same physical Mac, is also
unreachable. No source was copied and no benchmark was started there.

This is an external access blocker, not permission to substitute the
32 GiB/Tailscale machine or the local M5. The 16 GiB fresh-process SSD cohort
remains required as soon as either LAN endpoint returns.
