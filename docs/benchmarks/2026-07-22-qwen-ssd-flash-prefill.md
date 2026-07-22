# Qwen SSD implicit-causal FlashAttention candidate (2026-07-22)

Status: directional candidate; not promotion evidence.

Decision: keep the implementation on `codex/qwen-ssd-flash-prefill` and finish
the clean 128/2K/8K/32K AC cohort plus pre-M5 transfer qualification before
promotion.  The first isolated canary is strongly positive, but the host was
on battery and the arms were not interleaved.

Supersedes: no current qualification record.  If promoted, this changes the
Qwen MLX-affine ExpertMajor v2 resident and SSD prefill executor only; the GGUF
container, routed-weight encoding, F32 persistent KV cache, decode path, and
SSD expert streaming contract remain unchanged.

## Implementation boundary

The previous selector rejected Qwen's existing F16 GQA FlashAttention whenever
SSD streaming was active.  At 8K the SSD executor therefore used the exact F32
fallback for all ten full-attention layers.  The candidate:

- permits the same FlashAttention pipeline in resident and SSD modes;
- computes the causal boundary inside the Metal kernel, including non-zero
  prefixes and partial 64-key tails;
- skips key blocks that are wholly in the future;
- removes the dense F16 `[query][key]` mask, its block classifier, and its tail
  mask copy from this Qwen path;
- preserves the persistent token-major F32 KV cache and stages only one shared
  live F16 K/V frontier;
- charges the linear F16 frontier and bounded tail pad to automatic runtime
  admission.

## Scratch frontier

| Admitted context | Staged F16 K/V | Tail pad upper bound | Dense causal mask | Total new accounted Flash scratch |
|---:|---:|---:|---:|---:|
| 2,048 | 4 MiB | 0.25 MiB | 0 | 4.25 MiB |
| 8,192 | 16 MiB | 0.25 MiB | 0 | 16.25 MiB |
| 32,768 | 64 MiB | 0.25 MiB | 0 | 64.25 MiB |
| 100,000 | 195.31 MiB | 0.25 MiB | 0 | 195.56 MiB |
| 262,144 model maximum | 512 MiB | 0.25 MiB | 0 | 512.25 MiB |

The tail pad is conservative: exact multiples of 64 allocate only a dummy
byte.  The 512 MiB maximum-context value is asserted by a model-free planner
test.

## Directional forced-SSD A/B

Host: Apple M5 Pro, 64 GiB.  Power state: battery, 55% at control start and 53%
after the candidate.  Model:
`Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-MLX-Affine4-G64.gguf`, the current
20,808,566,880-byte affine v2 artifact.  Input: `speed-bench/promessi_sposi.txt`.
Both runs used Metal, forced SSD, pure prefill, strict full stack, context
allocation 8,193, and the 2K/8K frontiers.  Control and candidate are the same
dirty build; the control alone sets
`DS4_METAL_DISABLE_QWEN_FLASH_PREFILL=1`.

| Frontier | Control t/s | Candidate t/s | Throughput delta | Control wall | Candidate wall | Wall delta |
|---:|---:|---:|---:|---:|---:|---:|
| 2K full | 399.56 | 879.97 | +120.23% | 5.126 s | 2.327 s | -54.59% |
| 2K→8K segment (6,144 tokens) | 221.37 | 930.51 | +320.34% | 27.755 s | 6.603 s | -76.21% |
| 8K aggregate full | 249.14 | 917.34 | +268.20% | 32.881 s | 8.930 s | -72.84% |

## SSD I/O invariance

| Frontier | Metric | Control | Candidate | Delta |
|---:|---|---:|---:|---:|
| 2K | Expert reads | 12.895203 GiB | 12.900146 GiB | +0.038% |
| 2K | Expert loads | 7,825 | 7,828 | +0.038% |
| 2K→8K | Expert reads | 15.129822 GiB | 15.124878 GiB | -0.033% |
| 2K→8K | Expert loads | 9,181 | 9,178 | -0.033% |

The route-dependent three-load difference is negligible.  The magnitude and
location of the speedup, while expert bytes stay flat, support the attention
fallback diagnosis.  `pread` time is not compared because the non-interleaved
ordering and OS page-cache state make that field invalid for attribution.

## Correctness and gates

| Gate | Result |
|---|---|
| Production geometry | 16 query heads, 2 KV heads, head dimension 256 |
| Causal coverage | Non-zero prefix, 65 query rows, 70-key partial tail |
| Resident versus SSD Flash output | Byte-identical |
| Flash versus legacy F32 max absolute difference | 0.00249 |
| Runtime memory at 8K canary | 16 MiB F16 KV, 0-byte Flash mask |
| `make premerge` | Pass |
| Model-declared maximum accounting | 262,144 tokens, pass |

The F16 comparison tolerance covers the existing resident Flash staging
semantics; this candidate does not introduce a new quantization or model
format.

## Pending promotion matrix

| Required lane | Current state |
|---|---|
| 128 smoke and greedy decode | Pending clean AC cohort |
| 2K A/B/B/A plus logits | Pending clean AC cohort |
| 8K A/B/B/A plus logits | Pending clean AC cohort |
| 32K A/B/B/A plus logits and memory telemetry | Pending clean AC cohort |
| M1 Pro 16/32 GiB SSD canary | Pending reachable host and artifact |
| M2/M3/M4 transfer coverage | Pending hardware availability; model-free Metal compile is portable |

Raw directional CSVs are retained in
[`raw/2026-07-22-qwen-ssd-flash-control-battery.csv`](raw/2026-07-22-qwen-ssd-flash-control-battery.csv)
and
[`raw/2026-07-22-qwen-ssd-flash-candidate-battery.csv`](raw/2026-07-22-qwen-ssd-flash-candidate-battery.csv).
