# Qwen Affine4 128K on a physical M1 Pro with 16 GiB

Date: 2026-07-30

Status: accepted physical capacity and safety qualification for the guarded
16 GiB SSD path at its 128K ceiling; not a performance promotion. The candidate
completed a 131,072-token prefill and 128 greedy decode tokens with zero swap
on the physical 16 GiB floor. This record does not promote a speed claim or
establish output parity with a matching `main` run. The artifact's larger
metadata window is not a gate for this RAM tier.

Decision: retain the Qwen-only path and cap the public 16 GiB contract at this
measured capacity frontier. The run qualifies bounded completion and admission
safety, not comparative throughput. Refreshed lower-frontier timing remains a
separate performance gate and cannot turn the numbers below into a speed claim.
On guarded 16/24 GiB tiers, cap one macro-prefill arena at 32K tokens, reserve
arithmetic room for the minimum cache plus one complete route cycle, and allow
decode entry to re-sample macOS pressure for at most 30 seconds after transient
prefill storage is released. The gate still fails closed if pressure does not
return to normal. The 16 GiB admission ceiling is 131,201 allocated tokens: a
131,072-token prompt, 128 decode tokens, and one bookkeeping slot. Refreshed
8K and 32K timing lanes remain pending before any performance promotion.

Supersedes: none. This extends the physical 16 GiB evidence in
[`2026-07-29-qwen-m1-pro-16g-main.md`](2026-07-29-qwen-m1-pro-16g-main.md)
without changing the support boundary.

## Intent, mechanism, expected effect, and risk

The earlier 128K candidate completed prefill but observed non-normal pressure
at decode entry and correctly stopped before allocating the larger decode
cache. A second exploratory process showed that a warmer GGUF page cache could
make the planner choose a still larger single macro arena. It was terminated
before producing a result because that plan did not test the intended bounded
memory profile.

The selected candidate makes four related Qwen-only changes:

1. Prefill-only tensors and rollback storage belong to a transient phase arena
   and are released before decode.
2. Guarded planning preserves room for 641 experts: the 321-slot minimum plus
   one 320-expert route cycle (40 routed layers times 8 selected experts).
3. One guarded macro-prefill arena is capped at 32,768 tokens. A hotter page
   cache therefore cannot turn extra reclaimable credit into a larger
   transient allocation.
4. Decode entry waits up to 30 seconds for a fresh normal-pressure signal. It
   never retries prefill and never proceeds on a stale or non-normal signal.

For example, the successful 128K run used four 32K macro arenas sequentially,
not one 128K allocation. The comparison stops at allocation lifetime: the
model still evaluates the complete 128K prompt and retains the context state
required for decode.

The main risk is trading throughput for a bounded footprint. More macro
boundaries repeat setup and make later chunks progressively slower as the
context grows. A short-context run could also regress even though 128K is
capacity-qualified, so this record is not sufficient to promote a performance
claim.

## Experiment identity

| Condition | Value |
| --- | --- |
| Host | `MacBookPro18,3`; Apple M1 Pro; 16 GiB unified memory |
| OS | macOS 26.5 build `25F71` |
| Mode | AUTO resolved to guarded SSD |
| Candidate source | `fe495acd2342b3b700f2ba850921f4b6446072c8`; tree `4df8646c6f97838e15be2863ac34f6455d0dda17` |
| Candidate binary | SHA-256 `e9e93da00848732c070d7b333bb3d6ce3bceae113cafad5b852777e6bacb0347` |
| Runtime Metal source | SHA-256 `06de75f42895665f97153105c5a1de931973a551af4b29769dbce4a783c75098`; no overrides |
| Artifact identity | Canonical path `Qwen3.6-35B-A3B-Hebrus-ExpertMajor-v2-MLX-Affine4-G64.gguf`; 20,808,566,880 bytes. The physical run used its pre-migration local basename; the canonical Hub copy is byte-identical. |
| Artifact SHA-256 | `dd17266185833a9f05531ce366fd7284ddca1ed64aa3dcf06e321e8c72c9ea3d` |
| Prompt | `long_context_security_prompt.repeat4.txt`; SHA-256 `1a98e040a0057e9ddd01daa4139a239d8930b8b7fb75fb08894991e4275a9a09` |
| Frontier | 131,072 prompt tokens; context allocation 131,201; 128 greedy decode tokens |
| Isolation | Headless SSH session; one inference process; GUI and indexing competitors disabled |
| Validity guards | Minimum 10% free-memory floor; zero new swapout required; swap disabled and zero before/after |
| Cache state | Warm macOS page cache after an identical discarded warmup; fresh application cache |

The benchmark harness records `main_revision` equal to the candidate because
the earlier clean `main` build was no longer present after reboot. This is a
candidate-only cohort. No baseline result is inferred from those identity
fields.

## Chronological run results

| Started (Europe/Rome) | Revision / plan | Prefill throughput | Decode throughput | Decode TPOT p50 / p95 | Delta vs tested `main` | Delta vs previous comparable | Result |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| 2026-07-30T10:06:34+02:00 | `7869b7c`; macro 81,920; decode target 1,281 | 84.534 t/s derived from phase telemetry | N/A | N/A | N/A; no completed matching `main` | N/A; first 128K attempt | rejected: prefill completed, pressure non-normal at decode entry, zero swap |
| 2026-07-30T10:36:49+02:00 | `7b57702`; warm-cache plan selected macro 131,072 | N/A | N/A | N/A | N/A; no completed matching `main` | N/A; intentionally stopped before a comparable result | aborted by operator after plan inspection; exit 143, zero swap |
| 2026-07-30T10:41:33+02:00 | `fe495ac`; four 32,768-token macros; decode target 1,921 | 70.06 t/s | 4.01 t/s | 233.897 / 278.205 ms | N/A; no completed matching `main` | prefill -17.122% vs `7869b7c`; decode N/A | pass: 128 decode tokens, watchdog clear, zero swap |

The comparable prefill wall time increased from 1,550.520 seconds in the first
attempt to 1,870.960 seconds in the passing run (+20.667%). This is the
measured stability cost, not a performance win. The first run has no result
CSV because it stopped before decode; its prefill throughput is derived from
the timestamped phase events.

The four passing-run macro completions occurred at cumulative frontiers 32,768,
65,536, 98,304, and 131,072. At decode entry, the first pressure snapshot was
not admissible. The bounded recheck resumed after 11 total samples and 1,000
ms:

```json
{"event":"memory_pressure","phase":"decode","normal":true,"cache_budget_changed":true,"attempts":11,"waited_ms":1000,"action":"resume_phase"}
```

During decode, live headroom denied another 541.69 MiB slab. The guarded cache
froze at 963 expert slots and inference continued. This is expected bounded
behavior, not an admission bypass.

## Refreshed 8K correctness and rejected timing cohort

The first refreshed lower-frontier sequence used the same physical host,
artifact, prompt bytes, 8,192-token frontier, 8,321-token allocation, 128
greedy decode tokens, warm-cache declaration, and zero-swap watchdog. It ran
candidate/main/candidate/main in chronological order.

| Started (Europe/Rome) | Revision / arm | Prefill throughput | Decode throughput | Decode TPOT p50 / p95 | Delta vs tested `main` | Delta vs previous comparable | Result |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| 2026-07-30T11:24:55+02:00 | `23256f9`, candidate B1 | 307.30 t/s | 9.98 t/s | 92.636 / 153.239 ms | -0.111% / +2.464% vs main A1 | N/A; first candidate | exact output vs main; zero swap |
| 2026-07-30T11:28:36+02:00 | `d824833`, main A1 | 307.64 t/s | 9.74 t/s | 96.427 / 158.795 ms | baseline | N/A; first main | exact output; zero swap |
| 2026-07-30T11:30:31+02:00 | `23256f9`, candidate B2 | 307.31 t/s | 6.34 t/s | 143.026 / 231.677 ms | -0.107% / -34.908% vs main A1 | +0.003% / -36.473% vs candidate B1 | exact output vs main; zero swap |
| 2026-07-30T11:33:17+02:00 | `d824833`, main A2 | 124.83 t/s | 5.78 t/s | 163.495 / 254.068 ms | -59.423% / -40.657% vs main A1 | N/A; closing control drifted | exact output; zero swap |

Every arm produced byte-identical frontier logits (SHA-256
`e45185067de6a9117941819bcd0c6f62c9f5a4cf21ef21bbf9077097babfd4d5`)
and decode evidence (SHA-256
`bb1d5089158c1da0f3636877cc7fc377a89155f08999254d7e4172d54f6cb72f`).
The closing main control lost 59.423% prefill throughput relative to the
opening main run, and the two candidate decode runs also diverged by 36.473%.
The complete timing cohort is therefore rejected for performance comparison.
It remains valid additive evidence for exact output, completion, pressure
safety, and zero swap.

The first 32K control attempt was interrupted after this drift was recognized
and is not a result. A fresh post-reboot 8K/32K timing cohort is still required
before promotion.

## Memory and storage evidence

| Started (Europe/Rome) | Revision / result | Minimum free | Watchdog max RSS | `/usr/bin/time` max RSS | Peak footprint | Swap delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2026-07-30T10:06:34+02:00 | `7869b7c`; stopped before decode | 29% | 3,121,184 KiB | 3,316,858,880 bytes | 8,050,725,832 bytes | 0 |
| 2026-07-30T10:36:49+02:00 | `7b57702`; operator-aborted plan | 50% | 3,536,176 KiB | N/A | N/A | 0 |
| 2026-07-30T10:41:33+02:00 | `fe495ac`; pass | 29% | 3,626,240 KiB | 3,960,766,464 bytes | 7,246,697,416 bytes | 0 |

Against the first attempt, the passing process's sampled maximum RSS rose
19.413%, while peak memory footprint fell 9.987%. These metrics describe
different accounting boundaries and are reported separately. The important
acceptance observations are that minimum free memory stayed at 29%, swap
remained exactly zero, pressure returned to normal before decode, and the
process released all memory after completion (system free memory returned
from 90% before the cohort to 92% after it).

## Evidence and remaining gates

The passing bundle is retained on the physical host under the private
benchmark root as
`runs/explore-128k-phase-floor32k-recheck-headless-no-swap-20260730-r3`.

Key evidence hashes:

| Evidence | SHA-256 |
| --- | --- |
| Result CSV | `a4fb5c3cae36d93733cc986cbc71bb4d9ea555dac7ce5cbb5b67e2ce45b4e26b` |
| Qwen telemetry JSONL | `3aaf8d38b2f0a124f9479cd6ad55cfda02d84ec605c0bea44e2475a1cb544b2c` |
| 128K frontier logits | `3297675dc6c146bb2f21bd7f4b1e1a151b87aa9dfffea0de5a553a2037ec897d` |
| Decode evidence | `0398d83411209621502f25dc28bb352152255e7a29958180a0b5fc9eee675247` |

The result proves that this artifact and candidate can complete 128K plus 128
decode tokens on this physical 16 GiB host without swap; that bounded capacity
and safety result is the basis for the public 16 GiB ceiling. It does not prove
byte-identical output against a matching `main` run because no such run
completed, and it supports no comparative speed claim. Before any performance
promotion, run the refreshed 8K and 32K lanes with exact output comparison.
Context allocations above 131,201 are deliberately outside the 16 GiB
contract; higher-memory profiles retain their own endpoint gates.
