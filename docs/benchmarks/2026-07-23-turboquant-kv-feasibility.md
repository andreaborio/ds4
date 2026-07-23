# TurboQuant KV cross-engine feasibility — 2026-07-23

Status: active Qwen research evidence; measured on the 16 GiB and 32 GiB
M1 Pro lanes, not promoted as a production speed, quality, or universal-RAM
claim.

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

## 16 GiB M1 Pro SSD Cohorts

Remote Login returned on the LAN host at `192.168.1.212`. The physical host is
an Apple M1 Pro with 16 GiB unified memory, running macOS 26.5 build `25F71`
and connected to AC power. It had 93.75 MiB of pre-existing swap allocation,
so the runner used the cumulative `vm_stat` swapout counter as the invalidation
signal. Every retained arm completed with zero new swapouts.

The isolated remote source commit is
`264904f210b555f98f42080b7bb30a78b5f6e80e`, created from local research
commit `61b75cc`. The benchmark binary SHA-256 is
`8d0692012d4f0348ffa01b4cbddad2c36989bc63a3b6bc1760822220780ac504`.
The model is the same `20,808,566,880`-byte artifact used on the 32 GiB lane,
with SHA-256
`dd17266185833a9f05531ce366fd7284ddca1ed64aa3dcf06e321e8c72c9ea3d`.
Before the model-backed cohort, `make kv-quant-test`, `make ds4_test`,
`ds4_test --metal-kernels`, and the `ds4-bench` build passed on this host.
All retained Metal strategies passed the decoded-cache reference fixture:
direct paths were within `4.48e-09`, and F16-staged Flash was within
`5.16e-06`.

Each model-backed arm used a fresh process, explicit SSD streaming, a
one-second pressure/swap watchdog, and no preload. The guard terminated an arm
if free-memory pressure fell below 20%, a new swapout appeared, or the 32K
runtime exceeded 30 minutes. No valid arm triggered it. Two setup attempts are
not evidence: one ran from the wrong working directory and failed before Metal
or model allocation; another was immediately terminated after a monitor-shell
error, before inference and without a swapout.

### Short and adaptive-policy observations

The 128-token smoke completed baseline and TQ4 `auto` with zero swapouts.
Baseline/TQ4 prefill was 35.12/57.85 tok/s, decode was 12.06/12.07 tok/s, and
task footprint was 4.88/4.87 GiB. The baseline was the cold first process and
the adaptive expert-cache plans differed, so this is a compatibility smoke,
not a speed A/B.

At 2K, normal SSD policy reinvested the TQ4 memory saving in the expert cache:
the baseline settled at 2,881 cached experts while every TQ4 arm settled at
3,521. This is valid end-to-end AUTO behavior, but it is not a same-plan
comparison:

| 2K policy / strategy | Expert cache | Prefill tok/s | Decode tok/s | Task footprint |
| --- | ---: | ---: | ---: | ---: |
| F32 adaptive baseline | 2,881 | 157.61 | 14.15 | 5.47 GiB |
| TQ4 `auto` | 3,521 | 221.03 | 14.08 | 6.47 GiB |
| TQ4 `flash` | 3,521 | 221.29 | 14.49 | 6.46 GiB |
| TQ4 `split` | 3,521 | 221.62 | 13.90 | 6.47 GiB |
| TQ4 `parallel` | 3,521 | 219.96 | 13.47 | 6.46 GiB |
| TQ4 `reuse8` | 3,521 | 221.15 | 11.07 | 6.47 GiB |
| TQ4 `serial` | 3,521 | 221.22 | 8.44 | 6.46 GiB |

All TQ4 strategies produced the same full-logit artifact. Their relative
decode result is therefore execution-policy evidence, while the F32/TQ4 task
footprints above mainly show how AUTO spends available RAM.

### Exact-cache 2K and 8K comparisons

Fixing both paths to exactly 2,881 expert entries produces a clean 2K pair.
F32/TQ4 Flash measured 158.32/221.90 prefill tok/s and 14.33/14.19 decode
tok/s. Runtime tensors fell from 722.76 to 648.60 MiB and task footprint from
5.47 to 5.41 GiB. TQ4 therefore saved 74.16 MiB of live runtime tensors,
increased prefill by 40.16% through its Flash path, and reduced decode by
0.98%. Both arms had zero swapout delta.

The complete exact-cache 8K matrix was:

| 8K strategy | Prefill tok/s | Decode tok/s | Decode vs F32 | Runtime tensors | Task footprint | Pressure minimum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F32 baseline | 93.88 | 13.12 | control | 962.78 MiB | 5.74 GiB | 34% |
| TQ4 `auto` | 276.00 | 12.64 | -3.66% | 679.33 MiB | 5.50 GiB | 36% |
| TQ4 `flash` | 275.16 | 12.80 | -2.44% | 679.33 MiB | 5.50 GiB | 36% |
| TQ4 `split` | 277.05 | 12.22 | -6.86% | 679.33 MiB | 5.51 GiB | 36% |
| TQ4 `parallel` | 277.04 | 10.10 | -23.02% | 679.33 MiB | 5.50 GiB | 36% |
| TQ4 `reuse8` | 278.58 | 6.23 | -52.52% | 679.33 MiB | 5.51 GiB | 36% |
| TQ4 `serial` | 276.87 | 3.79 | -71.11% | 679.33 MiB | 5.50 GiB | 36% |

All arms completed with zero new swapouts and all TQ4 strategies again emitted
byte-identical full-logit artifacts. The packed path saved 283.45 MiB, or
29.44%, of live runtime tensors. Flash scratch grew from 2.75 to 50.93 MiB,
but task footprint still fell by about 0.24 GiB. The large prefill increase is
the combined packed-cache/Flash research path versus the pre-M5 F32 SSD path;
it must not be generalized to resident mode or another SoC.

### Exact-cache 32K comparison

The 32K pair used capacity 32,897, exactly 2,881 expert entries, and 128 greedy
decode tokens:

| 32K path | Prefill tok/s | Decode tok/s | Runtime tensors live / peak | Task footprint | Scratch | Pressure minimum | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| F32 baseline | 29.18 | 10.67 | 1,922.88 / 2,497.94 MiB | 6.68 GiB | 2.75 MiB | 25% | 1,136 s |
| TQ4 Flash | 231.77 | 10.16 | 802.23 / 1,377.30 MiB | 5.77 GiB | 195.02 MiB | 36% | 155 s |

Both completed without a guard, competing process, or new swapout. TQ4 saved
1,120.65 MiB of live and peak runtime tensors, reduced reported task footprint
by 0.91 GiB, and raised minimum free-memory pressure by 11 percentage points
despite 192.27 MiB of additional Flash staging scratch. Persistent runtime
preflight fell from 3.28 to 2.19 GiB. Prefill was 7.94 times baseline, while
decode was 4.78% slower.

The 32K frontier argmax remained identical. Top-set overlap was 5/5, 18/20,
60/64, and 97/100; full-vocabulary RMSE was `0.16947`, MAE `0.13516`, and
maximum absolute movement `0.87727`. Greedy generation nevertheless first
diverged at token 9 of 128. This is explicit lossy-quality evidence and blocks
promotion without the required continuation-NLL and cross-model qualification.

### 16 GiB interpretation

The 16 GiB lane confirms that packed KV creates material memory headroom and
that Flash is the only retained M1 policy close to the F32 decode rate. It also
explains why deleting other strategies before the M5 comparison would be
premature: the M1 ordering is consistent at 2K, 8K, and 32K, but it says
nothing definitive about M5 scheduling. No implementation is removed, and no
production default, admission rule, or snapshot format changes from these
cohorts.
