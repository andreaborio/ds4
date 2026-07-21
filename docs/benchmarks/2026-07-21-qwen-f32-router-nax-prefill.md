# Qwen F32 router NAX prefill promotion (2026-07-21)

## Scope

- Host: Apple M5 Pro, 64 GiB unified memory.
- Runtime: Metal, F32 KV, pure prefill unless noted.
- Models:
  - Qwen3.6-35B-A3B ExpertMajor v2 with MLX affine 4-bit/group-64 experts.
  - Qwen3.6-35B-A3B ExpertMajor v2 with GGML Q4_K_S experts.
- Candidate: Metal 4 direct-RHS TensorOps for the F32 `2048x256`
  Qwen router, with an N128 token tile.
- Control: `DS4_METAL_DISABLE_F32_NAX_PREFILL=1`.

The existing F32 path evaluated every prompt row as an independent matvec and
reloaded the router matrix for every token. The promoted kernel stages each F32
weight tile as F16 and evaluates the whole prompt as a tiled matrix multiply.
Decode is unchanged because the promotion starts at 128 prompt rows.

## Resident throughput

All throughput values are tokens per second. The steady A/B qualification
excludes the first call that compiled a new Metal pipeline.

| Model / shape | Control observations | Control mean | Candidate observations | Candidate mean | Delta |
|---|---:|---:|---:|---:|---:|
| MLX-affine v2, 2K | 1445.49 / 1436.90 / 1443.36 | 1441.92 | 1641.22 / 1675.42 | 1658.32 | +15.01% |
| MLX-affine v2, 8K | 1238.82 / 1238.97 | 1238.90 | 1410.69 / 1415.98 | 1413.34 | +14.08% |
| Q4_K_S v2, 2K | 1106.50 / 1101.21 | 1103.86 | 1221.74 / 1238.91 | 1230.33 | +11.46% |

Final clean-tree observations after rejected-kernel removal were 1687.28 t/s
at 2K and 1426.64 t/s at 8K on the MLX-affine model.

## Gap against local MLX

| Frontier | Local MLX | Previous DS4 affine NAX | Previous gap | Final DS4 | Final gap | Original Q4-to-MLX gap recovered |
|---|---:|---:|---:|---:|---:|---:|
| 2K | 1969.55 | 1443.88 | 26.69% | 1687.28 | 14.33% | 67.66% |
| 8K | 2214.43 | 1240.49 | 43.98% | 1426.64 | 35.58% | 34.10% |

The recovered-gap denominator starts from the previously promoted Q4 results:
1096.68 t/s at 2K and 1018.93 t/s at 8K.

## Router attribution

| Layer-0 router at 8K | Time | Delta |
|---|---:|---:|
| F32 matvec control | 21.487 ms | - |
| F32 NAX N128 | 1.244 ms | -94.21% |

This removes about 20.24 ms per routed layer. The remaining large 8K hotspots
are GDN recurrent/QKV work and full attention, so further MoE-only changes
cannot close the remaining MLX gap.

## Tile selection

| Router tile at 2K | Throughput | Delta vs 1441.92 control |
|---|---:|---:|
| N32 | 1551.82 | +7.62% |
| N64 | 1585.27 | +9.94% |
| N128 | 1687.35 | +17.02% |

N128 is retained. `DS4_METAL_F32_NAX_TILE_N=32|64|128` remains available for
hardware qualification.

## Numerical qualification

| Model / frontier | Top-1 | Top-5 overlap | Top-20 overlap | Top-64 overlap | Cosine | RMSE | Max abs |
|---|---:|---:|---:|---:|---:|---:|---:|
| MLX-affine v2, 2K | same | 5/5 | 20/20 | 61/64 | 0.999816715 | 0.061025 | 0.301249 |
| MLX-affine v2, 8K | same | 5/5 | 19/20 | 61/64 | 0.999879142 | 0.045607 | 0.250775 |
| Q4_K_S v2, 2K | same | 4/5 | 20/20 | not recorded | 0.999726701 | 0.073425 | 0.394355 |
| Q4_K_S v2, SSD 128 | bit-identical | 5/5 | 20/20 | 64/64 | 1.000000000 | 0.000000 | 0.000000 |

At the first routed layer of the 2K affine run, 16,383 of 16,384 selected
expert slots were unchanged (99.99390%), and 99.95117% of tokens retained the
exact top-8 set. Across all 40 layers after error propagation, route-slot
retention was 98.93402% and exact-set retention was 91.77734%.

## SSD streaming / GGUF

The promoted router path reads a normal F32 tensor from the GGUF model mapping;
it does not alter expert addressing or the SSD cache. A forced Q4_K_S SSD run
with a 321-expert cache produced the same I/O behavior in both arms:

| SSD 128-token lane | Control | Candidate |
|---|---:|---:|
| Warm prefill throughput | 206.42 t/s | 206.93 t/s mean |
| Expert loads | 4239 | 4239 |
| Expert-cache hit rate | 89.65% | 89.65% |
| Expert data read | 6.985657 GiB | 6.985657 GiB |
| Logits | baseline | bit-identical |

The +0.25% SSD throughput difference is neutral because this small-cache lane
is I/O-bound. The important result is that GGUF Q4 SSD streaming remains fully
active and unchanged. MLX-affine ExpertMajor SSD addressing is still separate
unfinished work; the affine artifact remains resident-only.

## Rejected candidates

| Candidate | Observation | Result |
|---|---:|---|
| Routed affine N64/K64 | 894.31 t/s at 2K vs about 1444 N32 | reject |
| Routed affine NAX gate/up pair N32 | 1372.12 t/s at 2K | reject |
| Routed affine NAX gate/up pair N16 | 1322.68 t/s at 2K | reject |
| Compact route map N32 | 1318.54 t/s at 2K | reject |
| Compact route map N64 | 1216.63 t/s at 2K | reject |
| Full-F32 tiled router | 1534.91 t/s mean at 2K | reject: slower and 19/20 top-20 |

The rejected kernels and host branches were removed rather than left behind as
inactive experimental code.

## Promotion boundary

The NAX router is automatic only when all of these are true:

- Metal 4 TensorOps are available;
- the dense tensor shape is exactly `2048x256`;
- the batch contains at least 128 tokens and is divisible by 32.

All other F32 matmuls retain the existing path. The promotion can be disabled
with `DS4_METAL_DISABLE_F32_NAX_PREFILL=1` or enabled for a non-default canary
with `DS4_METAL_ENABLE_F32_NAX_PREFILL=1`.
