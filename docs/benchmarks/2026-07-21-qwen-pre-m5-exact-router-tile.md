# Qwen pre-M5 exact F32 router token tile (2026-07-21)

## Status

- Status: directional evidence; the current timing cohort is contaminated.
- Decision: correctness-qualified through 8K. Do not use these timings as the
  clean promotion cohort.
- Candidate commit: `d612c0c`.
- Control: `DS4_METAL_DISABLE_F32_NAX_PREFILL=1`.
- Host: M1 Pro, 16 GiB, AC power, automatic SSD residency.
- Model: Qwen3.6-35B-A3B DS4 ExpertMajor v2 MLX affine4/group-64 GGUF,
  20,808,566,880 bytes,
  SHA-256 `dd17266185833a9f05531ce366fd7284ddca1ed64aa3dcf06e321e8c72c9ea3d`.

The candidate stages two `2048`-element F32 router rows once per threadgroup
and evaluates an eight-token tile. It preserves the scalar backend's float4
dot order, lane assignment, SIMD reduction, and cross-SIMD reduction. The
isolated scalar/candidate comparison is byte-identical on M5 and M1.

## Directional performance

These measurements are deliberately retained because the speedup is large,
but the cohort is not promotion-grade: `audiomxd` consumed 69.4--82.9% of one
CPU core and `configd` consumed 33.4--36.4% during the runs.

| Context | Arm | Prefill t/s | Delta | Prefill wall ms | Wall delta | TTFT ms | Decode t/s | Generation wall ms | TPOT p50 / p95 ms |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1K | scalar | 66.35 | - | 15,433.219 | - | 15,434.498 | 7.48 | 2,138.034 | 122.323 / 194.011 |
| 1K | exact token tile | 142.57 | +114.9% | 7,182.611 | -53.5% | 7,183.756 | 7.59 | 2,106.958 | 121.420 / 189.308 |
| 2K | scalar | 94.64 | - | 21,639.230 | - | 21,640.809 | 6.85 | 2,336.072 | 149.357 / 195.101 |
| 2K | exact token tile | 155.51 | +64.3% | 13,169.742 | -39.1% | 13,171.023 | 6.89 | 2,320.585 | 145.736 / 197.994 |
| 8K | scalar | 38.12 | - | 214,899.401 | - | 214,900.956 | 6.97 | not retained | not retained |
| 8K | exact token tile | 39.97 | +4.85% | 204,962.404 | -4.62% | 204,964.687 | 3.80 | not retained | p95 353.7 |

The 8K decode difference is not attributable to the candidate: the decode
path is unchanged, while the competing system load varied during generation.
Only the prefill result is directionally relevant.

## Correctness and I/O

| Context | Greedy tokens | Prefill logits | Post-decode logits | Max abs / RMSE | Expert pread per arm | Swapout delta |
|---:|---:|---|---|---:|---:|---:|
| 1K | 16/16 | byte-identical | byte-identical | 0 / 0 | 12.649658 GiB | 0 / 0 |
| 2K | 16/16 | byte-identical | byte-identical | 0 / 0 | 14.605774 GiB | 0 / 0 |
| 8K | 16/16 | byte-identical | byte-identical | 0 / 0 | 16.565186 GiB | 0 / 0 |

The equal pread totals show that the directional gain comes from router
execution rather than reduced SSD work. Decode is intentionally unchanged.

## Rejected approximate tile

Commit `26c02af` used half-operand SIMD-group MMA below a total-context wall.
It was rejected before performance qualification because correctness was not
monotonic with context.

| Context | Selector | Greedy tokens | Prefill max abs / RMSE | Post-decode max abs / RMSE | Decision |
|---:|---|---:|---:|---:|---|
| 128 | approximate tile | 16/16 | 0.200718 / 0.031331 | 0.228804 / 0.042801 | insufficient guarantee |
| 512 | approximate tile | 16/16 | 0.371407 / 0.062834 | 0.204328 / 0.036684 | insufficient guarantee |
| 1,024 | approximate tile | 10/16 | 0.641499 / 0.091578 | 11.583725 / 1.586271 | reject |
| 2,048 | approximate tile | 16/16 | 0.282220 / 0.052654 | 0.318121 / 0.036509 | insufficient guarantee |
| 8,192 | scalar fallback | 16/16 | 0 / 0 | 0 / 0 | fallback exact |

The first 1K mismatch occurred at generated token 11 (`2264` versus `32666`).
The approximate kernel and context-wall plumbing were removed rather than
retained as an opt-in path.

## Raw evidence

- Rejected candidate summary:
  `ds4-router-total-prefill-26c02af-20260721/qualification-summary.md`
- Rejected candidate CSV:
  `ds4-router-total-prefill-26c02af-20260721/correctness-table.csv`
- Exact-tile summary:
  `ds4-router-exact-tiled-d612c0c-20260721/qualification-summary.md`
- Exact-tile directional CSV:
  `ds4-router-exact-tiled-d612c0c-20260721/directional-table.csv`
- Exact-tile raw evidence:
  `ds4-router-exact-tiled-d612c0c-20260721/results/`

All evidence directories above are retained on the qualified M1 Pro host.
