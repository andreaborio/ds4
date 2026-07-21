# Qwen unified MLX-affine AUTO and SSD promotion (2026-07-21)

Status: promoted to `main`.

## Runtime contract

Qwen3.6-35B-A3B now has one optimized Mac configuration and one model file for
both resident and SSD execution:

| Contract | Value |
|---|---|
| Container | GGUF with embedded `ds4.expert_major.v2` |
| Routed storage | MLX affine 4-bit, group 64 |
| Physical group | 32 packed bytes + BF16 scale + BF16 bias |
| Artifact | `Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-MLX-Affine4-G64.gguf` |
| Bytes | 20,808,566,880 |
| SHA-256 | `dd17266185833a9f05531ce366fd7284ddca1ed64aa3dcf06e321e8c72c9ea3d` |
| AUTO selection | Resident when admitted by Metal/live-memory gates; SSD otherwise |
| User tuning flags | None |

The affine repack replaces the equal-size routed payload inside v2. It does not
create ExpertMajor v3 or a non-GGUF sidecar. Qwen startup deliberately rejects
the old ExpertMajor v2 GGML/Q4 payload; backward compatibility is not part of
this engine contract.

Normal startup is therefore one command:

```sh
./ds4 -m /absolute/path/to/Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-MLX-Affine4-G64.gguf
```

`--resident` and `--ssd-streaming` remain qualification overrides only.

## Final throughput matrix

Host: Apple M5 Pro, 64 GiB unified memory. Backend: Metal. Prompt: fixed
64K-token prose corpus. Pure-prefill runs use F32 context state and the complete
strict Qwen feature stack.

| Frontier | Resident t/s | Resident wall | SSD t/s | SSD wall | SSD expert read | SSD amplification |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 523.72 | 244.404 ms | 210.70 | 607.503 ms | 8.898926 GiB | 1.000x |
| 2,048 | 1,661.18 | 1,232.862 ms | 551.57 | 3,713.042 ms | 15.596191 GiB | 1.000x |
| 8,192 | 1,421.90 | 5,761.314 ms | 269.10 | 30,442.711 ms | 16.703613 GiB | 1.000x |
| 32,768 | 877.34 | 37,349.238 ms | 83.69 | 391,552.292 ms | 16.875000 GiB | 1.000x |

AUTO resolves to resident on this 64 GiB host for these allocations. The SSD
column is forced only to qualify the same artifact's low-memory path.

## Remaining gap to local MLX

The local MLX comparison uses the same model family and the earlier fixed 2K/8K
benchmark lane. These are prefill comparisons, not decode claims.

| Frontier | Local MLX | Previous DS4 Q4 | Final DS4 affine resident | Gain vs Q4 | Remaining gap to MLX |
|---:|---:|---:|---:|---:|---:|
| 2K | 1,969.55 | 1,230.33 | 1,661.18 | +35.02% | 15.66% |
| 8K | 2,214.43 | 1,018.93 | 1,421.90 | +39.55% | 35.79% |

The large remaining 8K gap is no longer principally routed-weight format or
router time. Gated DeltaNet recurrence, full attention, and graph scheduling
remain the next resident optimization frontier.

## SSD macro-prefill promotion

The final planner may make one macro tile as large as the context admitted by
its RAM calculation. The Metal graph still processes bounded 2,048-token micro
batches; the larger macro changes layer/expert reuse order, not residency.

| 32K forced-SSD metric | 8K-capped multi-tile control | Adaptive single macro | Delta |
|---|---:|---:|---:|
| Throughput | 84.22 t/s | 83.69 t/s | -0.63% |
| Wall | 389,057.348 ms | 391,552.292 ms | +0.64% |
| Expert reads | 60.776367 GiB | 16.875000 GiB | -72.24% |
| `pread` calls | 110,640 | 30,720 | -72.23% |
| Expert loads | 36,880 | 10,240 | -72.23% |
| Read amplification | 3.7845x | 1.0000x | -73.58% |
| `pread` time | 1,380.387 ms | 376.437 ms | -72.73% |
| Macro workspace | about 262 MiB | 742.42 MiB | about +480 MiB |

This meets the promotion policy: a structural resource reduction of at least
40% is accepted when measured performance regresses by less than 2%. The RAM
planner reduces macro width in 2,048-token steps when the full workspace does
not fit, so this does not turn SSD streaming into full model residency.

## SSD micro-batch and cache selection

The selected-address affine grouped-MM path is automatic at 32+ tokens. The
final SSD micro width is 2,048; 4,096 was rejected because it gained only 1.13%
at 8K while doubling the bounded scratch frontier.

| 2K SSD micro width | Throughput |
|---:|---:|
| 64 | 278.01 t/s |
| 128 | 364.59 t/s |
| 256 | 448.39 t/s |
| 512 | 512.40 t/s |
| 1,024 | 553.95 t/s |
| 2,048 | 580.74 t/s |

| 8K SSD path | Throughput | Delta vs prior row |
|---|---:|---:|
| Affine selected-address matvec, micro 64 | 145.78 t/s | - |
| Grouped MM, micro 64 | 171.75 t/s | +17.81% |
| Grouped MM, micro 1,024 | 266.59 t/s | +55.22% |
| Grouped MM, micro 2,048 | 273.75 t/s | +2.69% |
| Grouped MM, micro 4,096 | 276.84 t/s | +1.13% |

Increasing the 2K prefill cache from the 321-expert complete-cycle floor to
640 or 1,280 did not change expert reads or loads and measured 281.73/281.36
t/s. The default therefore remains 321 for prefill; the decode target may grow
separately when the memory planner admits it.

## Decode and resident/SSD consistency

The final 128+16 lane uses the same GGUF in both modes.

| Mode | Prefill | Decode | p50 TPOT | p95 TPOT | Expert reads: prefill / decode |
|---|---:|---:|---:|---:|---:|
| Resident | 523.21 t/s | 57.43 t/s | 17.352 ms | 18.753 ms | 0 / 0 GiB |
| SSD, cold 321-expert cache | 212.60 t/s | 23.94 t/s | 41.881 ms | 43.850 ms | 8.898926 / 6.117188 GiB |

All 16 greedy token IDs and the final argmax were identical. Final-logit
comparison was top-5 4/5, top-20 16/20, top-64 58/64, cosine 0.995156569,
RMSE 0.373369, and maximum absolute difference 2.451809.

| Prefill frontier | Top-1 | Top-5 | Top-20 | Top-64 | Cosine | RMSE | Max abs |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | same | 2/5 | 17/20 | 49/64 | 0.989774760 | 0.529453 | 3.080565 |
| 2K | same | 3/5 | 18/20 | 56/64 | 0.990038559 | 0.428820 | 2.382765 |
| 8K | same | 4/5 | 15/20 | 36/64 | 0.917458747 | 1.002387 | 6.529112 |

Resident and SSD use different accumulation/scheduling kernels, so their full
logit vectors are not expected to be bit-identical. Greedy parity is retained
for every qualified frontier and for the complete 16-token decode lane.

## Context and regression gates

The Qwen metadata declares 262,144 tokens. Actual graph/allocation smokes passed
at 100,000 and 262,144; the latter estimated 12.03 GiB of context runtime on the
64 GiB host. Context support is therefore the model-declared maximum, not a
2,048-token or 100K engine wall.

| Context allocation | Runtime estimate | 128-token smoke | Strict stack |
|---:|---:|---:|---:|
| 100,000 | 5.84 GiB | 227.83 t/s | pass |
| 262,144 | 12.03 GiB | 207.28 t/s | pass |

`make model-free-test` passes with the final source, including Metal kernels,
ExpertMajor converter/reader, Qwen session maximum-context validation, metadata,
reference math, and SSD residency tests.
