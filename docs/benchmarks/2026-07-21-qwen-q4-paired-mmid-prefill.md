# Qwen Q4 paired routed-MM prefill candidate

Date: 2026-07-21

Status: retained on `codex/qwen-mlx-batch-prefill` for further qualification.
Short, 2K, 8K, model-free, and explicit SSD-streaming gates pass. The mandatory
32K promotion gate is still pending, so this note does not claim a production
promotion.

## Change

Qwen3.6 resident prefill already used the grouped routed `mm_id` path. The
useful MLX-style transfer was therefore not another resident layout or a
matvec-to-matmul switch. It was to reduce intermediate traffic inside the
existing large-batch path:

- dequantize Q4_K gate and up projections in one Metal dispatch;
- reuse the staged F16 activation tile for both projections;
- apply SwiGLU and route weight before writing the intermediate;
- write only the F16 down-projection RHS instead of two large F32 gate/up
  temporaries;
- use a 16-route output tile on Apple GPUs to preserve occupancy with the
  doubled accumulator set.

The pre-existing IQ2/Q2_K specialization remains available and keeps its N32
default. Qwen Q4_K uses the new N16 specialization by default, with disable
switches retained for A/B qualification.

The first Q4 N32 implementation was rejected immediately: its 2K resident
prefill was about 1.4% slower than the split control. N16 changes that result
because it avoids the occupancy loss caused by the paired accumulator set.

## Resident results

Model: `Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-Q4_K_S.gguf`, complete artifact
SHA-256 `d7c43a6388ec20e6fe5530850350f96fdb0ac37c5ce36d3e5f92b172c447f56b`.
All runs were sequential on the same Apple M5 Pro with 64 GiB unified memory,
explicit resident mode, AC power, and no competing inference process.

| Lane | Split control | Q4 paired N16 | Prefill delta | Decode result |
| --- | ---: | ---: | ---: | --- |
| short, 43+96 A/B/B/A | 232.56 t/s mean | 276.32 t/s mean | +18.82% | neutral |
| 2K+16 directional pair | 318.96 t/s mean | 358.58 t/s mean | +12.42% | neutral |
| 8K+16 A/B/B/A | 180.10 t/s mean | 192.58 t/s mean | +6.93% | 55.03 vs 55.23 t/s, neutral |

The 8K arms were 180.10, 192.54, 192.61, and 180.10 t/s in A/B/B/A
order. Both control values were identical to the displayed precision. All four
frontier-logit files were byte-identical with SHA-256
`1fd9da6ef0dfe052c32e8b2e2737f1afc6989d37922d7c66b44cf723e47cf540`.
All four decode-evidence files were byte-identical with SHA-256
`a1777614ba73bc6022f0b5b2ac29ab69a1a1b2b40e44027ae3eef3156b172c6d`.
Swap usage stayed at 3,932.56 MiB before and after every retained arm; the
campaign delta was zero.

Raw 8K evidence is under
`/private/tmp/ds4-qwen-q4-pair-8k.vaqfbE` on the qualification host.

## SSD-streaming isolation gate

The paired kernel is intentionally ineligible whenever the selected-expert
address streaming path is active. An explicit 321-expert-cache canary confirmed
the runtime selected `qwen_q4_expert_group_stream_addr` with an F32
intermediate, not the new resident kernel.

The 128+16 SSD canary completed at 207.04/28.67 prefill/decode t/s with zero
swap delta. Repeating it with the paired-kernel disable switch produced
byte-identical frontier logits and decode evidence:

- logits SHA-256:
  `933534128069d6dc03c56a1e3432815e0d538bbbcccd1add8b9990015b5968e9`;
- decode-evidence SHA-256:
  `58f3038ea6de549fb0d37a09a3f128e8ce1d49601c24c4c3b79d48b7b44d5258`.

This proves the candidate does not intercept or numerically alter the current
SSD-streaming implementation. Resident and SSD modes retain their existing
F16-versus-F32 intermediate distinction; their generated token IDs and final
argmax matched in the cross-mode check.

## Tests and next gate

`make model-free-test`, the Qwen reference tests, Qwen metadata tests, Metal
kernel compilation, and `git diff --check` pass on the candidate.

Before promotion, run the exact 32K resident A/B/B/A lane with frontier logits,
decode evidence, memory-pressure capture, and swap abort rules. If that passes,
the next model-family experiment is DeepSeek IQ2: compare its existing paired
N32 specialization with N16 without changing the default. GLM needs a separate
geometry audit because its routed-expert count and prefill pipeline are not the
Qwen top-8 shape.
