# DeepSeek MLX affine2 and SSD-streaming feasibility (2026-07-22)

Status: research / HOLD; not qualified for promotion.

Decision: keep `codex/mlx-deepseek-ssd-study` as an experimental branch. Do
not merge the affine2 runtime into a supported DeepSeek path until the artifact
writer, residency contract, SSD planner, end-to-end correctness, quality, and
required performance matrices are complete.

Supersedes: none.

## Question and result

This audit asks whether MLX implementation ideas can improve DeepSeek V4 Flash
in DS4, including the 64 GiB SSD-streaming path. The answer is split:

- MLX's affine quantization semantics, expert-indexed quantized matrix
  multiplication, and route sorting are credible implementation ideas for a
  compute-bound MoE path.
- The current DeepSeek affine2 candidate does not establish an end-to-end
  benefit. It has no writable model artifact, rejects resident execution,
  moves 18.52% more bytes per expert miss than the qualified IQ2/Q2 store, and
  has a planner/backend mismatch that can request full-store prefill reads in
  addition to selected-record loads.
- No DeepSeek affine2 timing or quality result is reported here. The only
  attempted model A/B was stopped before inference by the repository's AC
  power guard. Treating that stopped run as performance evidence would violate
  the benchmark contract.

This is therefore a feasibility and rejection record for the current slice,
not a claim that affine2 can never win. A warm-cache or future resident kernel
could still recover more compute time than the added byte cost, but that must
be demonstrated on the final artifact.

## Source and host identities

| Item | Identity |
| --- | --- |
| Control source | `ec6322ed022be13f7ff67915701ffc86ebfcda50` |
| Candidate source | `70d9164451da2fb3a8b2f352d0bbf5b7dbce17da` |
| Candidate branch | `codex/mlx-deepseek-ssd-study` |
| Local MLX source inspected | `57c66cac7cb3e5b1eb350488a61f1506b40d39f8` |
| Local MLX-LM source inspected | `a790972f0f844d81067ed45c28b524220a10c019` |
| Host | Apple M5 Pro, 64 GiB unified memory, 18 logical CPUs |
| OS | macOS 26.5.2, build `25F84` |
| Available internal storage | about 14 GiB at the time of the audit |
| Power during attempted model A/B | battery, 39% and discharging |

The clean comparison worktrees produced these diagnostic build identities:

| Arm | `ds4-bench` SHA-256 | Metal source-set SHA-256 |
| --- | --- | --- |
| Control | `e7ba823e89a79567539ddda21687f90f306a5a3404b52f04eddb47cb0cdb1104` | `25bf26675ca668248a2a45081856765d4a5befcabf822d2dc299695f3e3eeeea` |
| Candidate | `e5183834f30e3b82536e012a4097fe322fcf0d336529cc808eb34078ccfa7b8e` | `8328fbf1ef540518db3269f3e1ab9218946e6889380e54e1a87f45a7b5a47a44` |

The currently qualified DeepSeek v2 control artifact remains:

| Artifact property | Value |
| --- | --- |
| Filename | `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-DS4-ExpertMajor-v2.gguf` |
| Bytes | `86,720,114,272` |
| SHA-256 | `8378080263eb9224f7228d72e2afa4ac3cf74a116023fdec2c596ff228a33e3f` |

The hash above is the published project identity; this audit did not spend an
additional full-file pass rehashing the 86.7 GB local copy.

## What transfers from MLX

MLX exposes lazy evaluation and unified memory, while MLX-LM's current MoE
layer uses `gather_qmm` for expert-indexed quantized multiplication and sorts
expert indices once the route set is large enough. Those are useful kernel and
scheduling references. They are not, by themselves, an eviction-controlled
expert SSD cache: DS4 still needs explicit slot budgeting, `pread` accounting,
mapping policy, and cold/warm qualification.

Primary references inspected:

- [MLX unified memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html)
  and [lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html);
- [MLX-LM switch layers](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/switch_layers.py),
  including expert route sorting and `gather_qmm`;
- [MLX array loader implementation](https://github.com/ml-explore/mlx/blob/main/mlx/backend/common/load.cpp);
- the assessed [DeepSeek V4 Flash affine2 donor repository](https://huggingface.co/mlx-community/DeepSeek-V4-Flash-2bit-DQ/tree/main)
  and its [quantization configuration](https://huggingface.co/mlx-community/DeepSeek-V4-Flash-2bit-DQ/blob/main/config.json).

The transfer hypothesis is supported only indirectly by the qualified Qwen
results already in this repository. On the same M5 Pro, Qwen's resident
MLX-affine NAX path improved routed prefill by 29.68% at 2K and 20.70% at 8K
against its affine SIMD control. The final Qwen affine runtime improved 35.02%
at 2K and 39.55% at 8K over the earlier Q4 path. Its 32K SSD macro-prefill
reduced expert reads by 72.24% with a 0.63% throughput cost. Those results show
that MLX-derived storage and kernel ideas can transfer, but they do not predict
DeepSeek affine2: Qwen has an equal-size affine repack and a qualified resident
TensorOps path, while this DeepSeek candidate has neither property.

## Exact storage geometry

The candidate's canonical per-expert record is gate g32 plus up/down g64:

| Quantity | Qualified IQ2/Q2 v2 | Candidate MLX affine2 | Delta |
| --- | ---: | ---: | ---: |
| Gate bytes per expert | - | 3,145,728 | - |
| Up bytes per expert | - | 2,621,440 | - |
| Down bytes per expert | - | 2,621,440 | - |
| Complete expert record | 7,077,888 (6.75 MiB) | 8,388,608 (8 MiB) | +18.5185% |
| One 256-expert layer | 1,811,939,328 (1.6875 GiB) | 2,147,483,648 (2 GiB) | +18.5185% |
| Routed payload, 43 layers | 77,913,391,104 (72.5625 GiB) | 92,341,796,864 (86 GiB) | +13.4375 GiB |

Replacing the routed payload in the qualified v2 container would produce an
estimated `101,148,520,032`-byte (94.2019 GiB) GGUF before any additional
format overhead. The larger record also fits 15.625% fewer cache slots in a
fixed byte budget. Affine2 must therefore save enough compute to cover both
the extra bytes per miss and any increase in reloads.

## Implementation audit

### Artifact production is incomplete

`plan_deepseek_mlx_affine2()` explicitly stops at a physical layout plan. The
CLI requires `--dry-run` and says that the payload writer is not implemented.
There is no 86 GiB writer, complete GGUF/manifest construction, atomic publish,
end-to-end digest verification, or artifact to score. The current provenance
checks pin the donor Git revision and validate shard headers/shapes, but do not
hash every hydrated shard against its Git LFS OID or prove that routed donor
and non-routed GGUF bytes derive from the same checkpoint.

### Residency contract is incomplete

The runtime admits the affine2 store and exact g32/g64 geometry, then explicitly
rejects it when SSD streaming is not active. Thus this slice cannot deliver a
resident MLX-derived benefit, and AUTO can select resident before installation
later fails. That conflicts with the supported DeepSeek AUTO/resident/SSD
contract unless either resident execution is implemented and qualified or the
new format is made explicitly SSD-only in the residency planner and contract.

### SSD planner and Metal dispatch disagree

The C prefill predicate recognizes selected-address batching only for the
qualified IQ2/Q2 tensor types. Full-layer `pread` preparation is enabled by
default. When that predicate is false, `metal_graph_stream_layer_spans()` adds
the complete physical ExpertMajor layer. The Metal backend independently
forces every affine2 batch onto selected-address compute.

The resulting plan is full-layer preparation plus selected-record compute.
Under the default 4,096-token Flash chunk cap, the full-layer preparation alone
would request the following routed bytes:

| Prompt frontier | Prefill chunks | Full-layer prepare requests |
| ---: | ---: | ---: |
| 128 | 1 | 86 GiB |
| 2,048 | 1 | 86 GiB |
| 8,192 | 2 | 172 GiB |
| 32,768 | 8 | 688 GiB |

These are code-path byte requests, not measured physical NAND traffic. Page
cache hits can reduce device I/O. Conversely, an 86 GiB routed store exceeds
the host's 64 GiB unified memory, so repeated scans are likely to be
cache-expulsive. Selected-address cache loads are additional requests. Current
Metal `pread_bytes` telemetry counts the expert cache loader but not the layer
prepare worker, so it cannot by itself reveal this amplification.

A safe correction is not merely adding affine2 to the existing IQ2 predicate:
that predicate has a 760-token automatic ceiling and respects disable/quality
conditions that the forced Metal affine2 path does not. One canonical
"mandatory affine2 selected-address" decision must control both the C mapping
and Metal dispatch for every batch. I/O overlap and cache policy should remain
a separate change and a separate experiment.

Two secondary gaps also need explicit treatment:

- selected page-in/readahead diagnostics derive the up-component size from the
  gate size, which is invalid for affine2's 3 MiB gate and 2.5 MiB up records;
- the existing shared-expert/router I/O overlap and DeepSeek scheduling gates
  are IQ2/Q2-only, so affine2 currently uses the synchronous selected loader
  and excludes the resident NAX/TensorOps advantage behind the Qwen results.

## Experiments executed

### Clean build and model-free correctness

The candidate passed:

```sh
make clean
make -j8
make model-free-test
make premerge
```

This included the Metal kernel suite, `--metal-expert-pack`, the ExpertMajor
Python tests, SSD/cache policy tests, server/flag checks, and the other
model-free gates. The complete premerge gate also passed, including build
isolation, documentation links, deterministic dataset/prompt checks, and a
fresh model-free run. Its CPU-only isolation build emitted the repository's
existing unused-function warnings and completed successfully. The affine2
kernel coverage exercises exact selected-address
g32/g64 dequantization with F32/F16 right-hand sides and token counts
1/33/255/256. The sparse test maps the full 43 x 256, 86 GiB address geometry
without materializing that storage. It is numerical and structural coverage,
not a throughput measurement or store-to-output end-to-end test.

Runtime Metal source SHA-256 reported by the suite:
`96a7d8a4f37593fc1153d359de99ca40dfd9e0d8a146b289ff0600bd1b928c67`.

### Controlled model A/B attempt

Clean detached worktrees for the control and candidate were built separately.
The planned first regression lane used the unchanged qualified IQ2/Q2 artifact,
forced SSD, exact 3,000-record cache/preload, a 128-token prefill plus 128
greedy decode tokens, 32K context allocation, and A/B/B/A ordering. This lane
could only detect a regression introduced by the scaffolding; it could not
prove an affine2 benefit.

The first A arm terminated immediately with:

```text
M5 benchmark requires AC power
```

No model was loaded, no inference process ran, and no timing row or partial
cohort was retained. The guard was not bypassed. A second blocker is that no
affine2 GGUF exists; the roughly 14 GiB free on the internal disk is also far
below the estimated 94.2 GiB output plus safe conversion scratch.

## Required experiment before reconsideration

1. Implement a fail-closed writer/verifier with full donor and base provenance,
   generate the exact artifact on storage with at least 190-210 GiB safe
   scratch, and publish its byte count and SHA-256.
2. Resolve the resident-versus-SSD contract before model admission. Add C and
   Python fixtures plus store -> `pread` -> cache -> address table -> routed MoE
   end-to-end tests and AUTO/resident/SSD startup tests.
3. Align the C mapping plan and Metal dispatch. First compare current
   full-layer prepare against mandatory selected-only mapping without changing
   the affine kernels. Add separate counters for layer-prepare bytes, cache
   loader bytes, and physical disk I/O.
4. Run the official continuation scorer against the pinned donor, preserve
   greedy evidence, and qualify the quantization independently of timing.
5. On AC power with no competing inference process and zero swapout, run
   isolated cold and warm A/B/B/A cohorts at 128, 2,048, 8,192, and 32,768,
   each with at least 128 greedy decode tokens. Use both the prose and
   security/coding prompts at 32K. Record plan, control drift, within-arm
   spread, TPOT p50/p95, route uniqueness, cache hits/misses/evictions,
   `pread` calls/bytes/time, layer-prepare bytes, pressure, and swap.
6. If the correction touches shared SSD/Metal policy, repeat the required
   DeepSeek, GLM, and Qwen regression matrices. A synthetic kernel win may
   reject or justify further work, but may not promote the runtime.

Promotion requires correctness and quality gates, control drift below 3%, an
effect larger than noise and spread, zero unsafe memory pressure/swapout, every
frontier at least 95% of its qualified gold, and a geometric mean at least 98%.
For this candidate the causal break-even condition is:

```text
compute_ms_saved > affine2_extra_io_ms + reload_penalty_ms + noise_margin_ms
```

Until that condition is measured on the final artifact, the current affine2
SSD path remains HOLD.
