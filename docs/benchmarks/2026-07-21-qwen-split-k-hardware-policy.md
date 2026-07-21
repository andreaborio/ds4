# Qwen split-K and hardware-aware Metal policy — 2026-07-21

Status: current accepted Qwen qualification evidence.

Decision: retain the F32 split-K GQA decode path already committed as
`c7d00e6` and accept the hardware-aware Qwen AUTO policy. The split-K result is
a compute speedup. The memory-policy result is an admission and safety change;
it is not credited with kernel throughput that it does not implement.

Supersedes the Qwen performance row in
`2026-07-20-long-context-metal-stack.md`. That earlier record remains the
baseline identity and rejection evidence for the removed paired-Q8 prototype.

## Definitions and artifact

`Prefill` is prompt ingestion throughput. `Decode` or `generation` is
autoregressive output-token throughput after prefill; TPOT is its inverse
latency distribution. These columns are never compared without also naming the
prompt-token and generated-token counts.

All model-backed lanes used the 20,808,566,880-byte Qwen3.6-35B-A3B
ExpertMajor v2 Q4_K_S artifact with expected SHA-256
`d7c43a6388ec20e6fe5530850350f96fdb0ac37c5ce36d3e5f92b172c447f56b`.
Greedy evidence used 128 generated tokens unless a row explicitly says
otherwise.

## M5 Pro 64 GiB split-K A/B

The controlled lane used macOS build 25F84, AC power, AUTO resolving to
resident, a 65,536-token allocation, and the security prompt with SHA-256
`e7c1a2cadf781d274cc26bd251d532fe1b9e632080da97e3eb4684741e7cc308`.
The control was clean `e439753`; the candidate is the source committed as
`c7d00e6`. Both contexts completed with zero swapout and the same 128 greedy
token IDs in control and candidate.

| Context | Path | Prefill | Decode | TPOT p50 / p95 | Decode change |
| ---: | --- | ---: | ---: | ---: | ---: |
| 8,192 | serial GQA control | 161.60 t/s | 11.03 t/s | 90.650 / 91.699 ms | baseline |
| 8,192 | F32 split-K | 159.94 t/s | **52.62 t/s** | 18.857 / 20.027 ms | **4.77x** |
| 32,768 | serial GQA control | 64.06 t/s | 3.16 t/s | 316.037 / 318.419 ms | baseline |
| 32,768 | F32 split-K | 61.89 t/s | **37.42 t/s** | 26.562 / 27.834 ms | **11.84x** |

The candidate changes prefill by -1.03% at 8K and -3.39% at 32K while decode
improves by 377% and 1,084% respectively. Split-K is selected automatically at
2,048 or more cached tokens; the environment switch is retained only for
ablation.

Raw local evidence is under `/private/tmp/ds4-qwen-longctx-20260721-*` and
`/private/tmp/ds4-qwen-split-auto8k-20260721.*`. The candidate Metal source
identity is
`4a1df29617f83d6b6281b57ec27c990d3e7fdb5817a9061358a8a4f68591f2ea`.

## M1 Pro 16 GiB transfer and safety

The focused 2K A/B used the current ExpertMajor v2 candidate, AC power, AUTO
resolving to SSD, 2,048 prompt tokens, and 128 generated tokens. Both arms used
321 cached experts during prefill and a 3,521-expert decode target. Pressure
remained normal and swapout stayed unchanged.

| Path | Prefill | Decode | TPOT p50 / p95 | Decode change |
| --- | ---: | ---: | ---: | ---: |
| split-K disabled | 80.90 t/s | 9.70 t/s | 96.528 / 140.464 ms | baseline |
| split-K enabled | 80.77 t/s | **15.00 t/s** | 60.070 / 114.581 ms | **1.55x** |

The two arms produced identical greedy token IDs and the same final argmax.
Their final logits differed by at most `4.77e-06` with RMS `8.96e-07`.
Evidence is retained under `/private/tmp/ds4-hw-policy-final16.4am7Pe` and
`/private/tmp/ds4-splitk-off16.xCQjoz`.

A separate persistent-server check repeated one cold plus four distinct warm
short prompts at context allocation 8,192 and generation cap 64. AUTO selected
321 prefill / 2,241 decode experts. The four warm medians were **18.43 t/s
prefill** and **11.06 t/s generation**, with normal pressure and zero swapout.
The historical 2026-07-18 medians were 15.04 and 9.77 t/s, but the new run is
only directional: power, artifact generation, tokenized prompt lengths, and
EOS lengths were not identical. It must not be presented as a controlled
speedup. Raw evidence is under `/private/tmp/ds4-shortwarm16-current.3SNbw1`.

## M1 Pro 32 GiB policy isolation

At 2K, both clean `c7d00e6` and the memory-policy candidate resolved AUTO to
resident, used the same Metal source, issued no expert-cache `pread`, and
completed with zero swapout and byte-identical decode evidence.

| Arm | Resident requirement / Metal budget | Prefill | Decode | TPOT p50 / p95 |
| --- | ---: | ---: | ---: | ---: |
| `c7d00e6` control | 24.67 / 24.96 GiB | 147.54 t/s | 29.00 t/s | 34.716 / 34.989 ms |
| policy candidate | 22.17 / 24.96 GiB | 143.33 t/s | 28.67 t/s | 34.892 / 35.575 ms |

The single sequential sample is -2.85% prefill and -1.14% decode, with only
0.51% p50 and 1.68% p95 TPOT movement. Because the shader and hot path are
identical and no repeated cohort was run, this is treated as order/clock noise,
not as a performance change. The policy's measured effect is the larger fixed
admission margin: 2.79 GiB instead of 0.29 GiB.

A same-candidate 2K decode ablation measured 29.00 t/s with split-K and
14.19 t/s without it, a 2.04x gain. Prefill was unchanged at 147.54/147.60
t/s. Token IDs and final argmax matched; final-logit max absolute difference
was `6.2e-06`, with RMS `1.106e-06`.

The same candidate then ran a controlled 32K split-K ablation on the 32 GiB
host. Both arms used AUTO resident, AC power with low-power mode disabled,
32,768 prompt tokens, and 128 generated tokens. Both had zero expert `pread`,
zero resident split wait, zero swapout, and no thermal or performance warning.

| Path | Prefill | Decode | TPOT p50 / p95 | Decode change |
| --- | ---: | ---: | ---: | ---: |
| split-K disabled | 31.70 t/s | 1.63 t/s | 613.307 / 614.574 ms | baseline |
| split-K enabled | 30.54 t/s | **19.02 t/s** | 52.564 / 53.297 ms | **11.67x** |

Split-K reduces decode wall time and TPOT by about 91.4% while prefill differs
by -3.66%. The two arms generated the same 128 token IDs and final argmax;
final-logit max absolute difference was `1.598e-05`, with RMS `2.976e-06`.
The complete local evidence bundle is
`/tmp/ds4-32gb-policy-ab-evidence/evidence-bundle.tgz`, SHA-256
`f55ef1a299862efe5e1635f03159ddf66eedda672cf597144aa22a6b3ae0526e`.

## Hardware policy and gates

Qwen exposes 16/24/32/36/48/64/96/128 GiB labels while sizing from exact
physical RAM, Metal's reported working-set recommendation, context runtime, and
live pressure. AUTO resolved to SSD on the physical 16 GiB host and resident on
the idle, AC-powered 32 GiB host. A busy 64 GiB host correctly fell back to SSD
when its live-pressure gate did not admit the mapped payload.

The final source passes `make premerge`, including context and documentation
audits, build isolation, model-free CPU and Metal tests, Qwen reference tests,
ExpertMajor v2 validation, and the SSD residency-policy suite. The latter tests
every named memory cut, continuous reserves, cold/warm file-cache equivalence,
monotonic cache capacity, resident fixed/live gate agreement, and the measured
16 GiB cache ceiling.
