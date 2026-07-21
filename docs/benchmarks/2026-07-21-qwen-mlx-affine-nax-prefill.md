# Qwen MLX-affine NAX routed-prefill canary (2026-07-21)

## Scope

- Host: Apple M5 Pro, 64 GiB unified memory.
- Model: Qwen3.6-35B-A3B, MLX affine 4-bit/group-64 weights repacked in
  ExpertMajor v2 without changing their scale/bias semantics.
- Runtime: Metal resident, F32 KV, pure prefill (`--gen-tokens 0`).
- Candidate: affine-only Metal 4 `matmul2d` routed kernel, enabled at 2048+
  tokens.
- Control: the existing affine SIMD-group routed kernel selected with
  `DS4_METAL_DISABLE_MLX_AFFINE_NAX=1`.
- Ordering: A/B/B/A in separate processes. Reported values are arithmetic
  means of the two observations for each arm.

The canary deliberately does not alter the GGML Q4_K path. The historical
TensorOps drift concern therefore remains isolated from the existing GGML
arithmetic baseline.

## Throughput

| Shape | Control B1 | Control B2 | Control mean | NAX A1 | NAX A2 | NAX mean | Delta |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2K full prefill | 1113.51 | 1113.33 | 1113.42 | 1458.33 | 1429.43 | 1443.88 | +29.68% |
| 8K full prefill | 1025.42 | 1030.01 | 1027.72 | 1241.68 | 1239.30 | 1240.49 | +20.70% |
| 2K to 8K segment (6144 tokens) | 1000.44 | 1005.45 | 1002.95 | 1223.34 | 1188.92 | 1206.13 | +20.26% |

All throughput values are tokens per second.

## Gap against the local MLX baseline

| Frontier | MLX | DS4 promoted Q4 | Q4 gap | Affine SIMD control | Control gap | Affine NAX | NAX gap | Original gap recovered |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2K full | 1969.55 | 1096.68 | 44.31% | 1113.42 | 43.47% | 1443.88 | 26.69% | 39.78% |
| 8K full | 2214.43 | 1018.93 | 53.99% | 1027.72 | 53.59% | 1240.49 | 43.98% | 18.53% |

The recovered-gap column uses the promoted Q4 DS4 result as the starting
point. It is not a claim of parity: attention, routing overhead, activation,
and remaining tile/scheduling differences still account for a substantial
8K gap.

## Layer-0 routed-MoE profile at 8K

| Stage | Affine SIMD ms | Affine NAX ms | Delta |
|---|---:|---:|---:|
| map | 1.028 | 0.996 | -3.11% |
| gate | 23.260 | 11.956 | -48.60% |
| up | 23.295 | 12.197 | -47.64% |
| activation/route weight | 3.916 | 3.784 | -3.37% |
| down | 30.179 | 20.601 | -31.74% |
| sum | 3.389 | 3.402 | +0.38% |

The speedup is therefore in the intended kernels rather than in model load,
attention, or benchmark bookkeeping.

## Logit drift

| Frontier | Top-1 | Top-5 overlap | Top-20 overlap | RMS | Max abs | Cosine | Repeatability |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2K | same | 5/5 | 19/20 | 0.046057 | 0.228110 | 0.999895878 | bit-identical per path |
| 8K | same | 4/5 | 20/20 | 0.035172 | 0.260485 | 0.999926012 | bit-identical per path |

The comparison is NAX versus the affine SIMD control. Both repeated NAX runs
and both repeated control runs produced exactly identical dumped logits.

## Promotion boundary

The affine NAX kernel is selected by default only for resident MLX-affine
ExpertMajor batches of at least 2048 tokens and remains disable-only bisectable
with `DS4_METAL_DISABLE_MLX_AFFINE_NAX=1`.

This is not yet a complete runtime promotion. MLX-affine decode and selected-
expert SSD address kernels are still required. The existing Q4_K SSD streaming
path is unchanged and remains covered by the model-free and SSD tests.
