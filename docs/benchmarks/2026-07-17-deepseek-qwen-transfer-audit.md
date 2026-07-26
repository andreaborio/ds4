# DeepSeek Flash: Qwen transfer audit

Date: 2026-07-17

## Verdict

The useful Qwen lesson is not the expert-major file layout by itself.  On both
models that layout is effectively throughput-neutral once the runtime already
issues selected-expert reads.  Qwen's large wins came from removing arithmetic
and scheduling bottlenecks: paired gate/up work, parallel GQA, and a faster
router.  DeepSeek Flash already has the analogous paired IQ2 gate/up path and a
parallel router, while its attention geometry and SSD working set are very
different.

The significant remaining DeepSeek SSD opportunity is **route locality**.  The
built-in Flash hotlist was seeded with rank-sized LFU priorities.  At a 3,097
record cache budget those priorities ranged from 3,097 down to 1, while an
expert selected during decode gained only one point.  A hotlist learned from a
different workload could therefore remain effectively unevictable for a whole
short generation.

Changing the built-in seed to priority 1 keeps its useful preload order but
allows real selections to rise above unused entries immediately.  On the 5K
workload this cut generation reads by 41% and improved the clean screening run
from the 6.74 t/s control geomean to 7.08 t/s, about 5%.  A hotlist trained on
the workload reached 7.90 t/s, about 17% above the nearby 6.75 t/s legacy
control.  That is an upper bound, not a general benchmark claim, but it proves
where a material SSD-streaming gain exists.

## Fixed test envelope

- Apple M5 Pro, 64 GiB, AC power, power mode 2;
- Metal SSD streaming, 32,768-token allocation;
- 86,720,114,272-byte native expert-major v2 Flash GGUF;
- exact 3,097-record expert cache unless stated otherwise;
- 5,120 prompt tokens and 64 generated tokens for the long-context decode
  campaign;
- frontier-logit SHA-256
  `a9843bd4d3750fde2411f6000ec1372df717e643c7536d5c2bd660a2877ce924`;
- zero new swapout in every retained run.

The machine developed substantial whole-system drift during the long A/B/B/A
sequence: prefill fell from about 160 to 103 t/s even though the hotlist change
does not affect prefill.  Those four rates are retained for transparency but
are not used as a clean promotion mean.  The I/O counters, exact outputs,
single-screen comparisons, and short-context control are the stronger evidence.

## What transferred from Qwen, and what did not

| Qwen mechanism | DeepSeek status | Audit result |
|---|---|---|
| Native expert-major artifact | Implemented in the v2 GGUF | No duplicate payload, but layout alone was parity; the best grouped 256-token result was only +0.49% |
| Paired gate/up compute | Already implemented for IQ2 gate/up with fused SwiGLU | Warm decode profile: about 0.31 ms gate/up and 0.23 ms Q2 down per layer |
| Parallel GQA | Not directly portable to Flash's 64-head, 512-dim compressed/indexed attention | An eight-part indexed-attention probe saved about 0.09 ms on one active ratio-4 layer and only +1.25% end to end |
| Router acceleration | DeepSeek router is already parallel | The remaining router SIMD ceiling is below 1%; unlike the old Qwen path, there is no serial top-8 scan to remove |
| Cache/victim knowledge | Shared LFU/LRU machinery existed | The built-in seed priority defeated adaptation; this is the useful transfer target |

## No-go experiments

### More cache is not safely available at 5K

Explicit 3,600, 3,900, and 4,387-record attempts all crossed the runner's 20%
free-memory guard during long prefill.  They produced no new swapout and the
host recovered immediately, but none is a valid generation result.  The
3,097-record tier is the robust operating point for this workload on the
current 64 GiB host.

### Static residency hints are not the old pin result

A Metal residency set covering 89 static allocations / 8.20 GiB produced a
noisy +5.4% A/B/B/A decode result.  Explicitly page-touching the same 8.20 GiB
took 501 ms, grew RSS as expected, and did not improve generation.  This does
not reproduce the older `mlock` result, and the unsafe static pin remains
default-off after the documented watchdog incident.

### Parallel indexed attention has a small ceiling

The first synthetic microbenchmark accidentally used a 2,048-row raw window.
Flash actually keeps only 128 raw rows; at 5K the ratio-4 layers attend to about
128 raw plus top-512 compressed rows.  In the real active-path A/B/B/A, serial
decode was 6.77 / 6.71 t/s and the parallel candidate was 6.91 / 6.74 t/s,
only +1.25% by geometric mean.  Layer 2 attention fell from 0.869 to 0.778 ms,
consistent with roughly two milliseconds per token across the eligible layers.
The probe is therefore removed rather than retained as another production flag.

## Route-locality result

| Policy | Decode t/s | Hit rate | Expert loads | Routed reads | p50 / p95 TPOT |
|---|---:|---:|---:|---:|---:|
| Legacy built-in rank priority, clean A1 | 6.75 | 50.15% | 8,231 | 54.26 GiB | 109.98 / 126.17 ms |
| No preload hotlist | 6.78 | 65.50% | 5,697 | 37.55 GiB | 108.62 / 267.48 ms |
| Built-in order, adaptive priority 1, clean screen | 7.08 | 70.78% | 4,825 | 31.81 GiB | 103.70 / 133.59 ms |
| Workload-trained hotlist | 7.90 | 81.69% | 3,024 | 19.93 GiB | 92.47 / 117.74 ms |

Disabling the hotlist proved the stale-priority diagnosis by improving hit rate
and reducing reads, but its cold first tokens damaged tail latency.  Adaptive
priority keeps the preload and removes the lock-in.  A workload-trained hotlist
shows the larger domain-specific ceiling without increasing the RAM budget.

The 64-token route profile contained 16,512 selections over 4,913 distinct
layer/expert pairs.  The current built-in top 3,097 entries covered only 6,925
selections, or 41.94%.  The top 3,097 entries from this workload covered 89.00%
offline; the live run reached 81.69% after cache dynamics and startup effects.

The short-context 128-prompt / 128-generation A/B/B/A was neutral, as expected:
the legacy arms were 13.06 / 13.01 t/s and adaptive arms 12.83 / 12.93 t/s,
with identical hit counts, loads, read volume, logits, and zero swapout.  The
small rate difference is ordinary run noise; the policy only matters when the
seed and the long-context route distribution diverge.

The final canary after removing all experimental attention/residency code
reproduced the adaptive counters exactly: 70.78% hit rate, 4,825 loads, and
31.81 GiB of routed reads, with the same logits and zero swapout.  Absolute
decode was only 6.10 t/s while prefill had also fallen to 139.61 t/s.  This is
additional evidence that the machine-wide rate drift is larger than the
general adaptive-policy gain in this session.  The deterministic I/O reduction
is promoted; a precise throughput percentage still requires a fresh mirrored
cold/warm campaign.

The historical flag integration at commit `ba2a729` received a separate
model-backed functional smoke after the parser, help text, and rollback path
were added. On binary SHA-256
`6301541dce570c80660e78c61f46a1e4dfbe68220dbfa7d04a8f0022e2f8f70b`, a
768-token native-AUTO request seeded 259 entries under both the omitted/default
`adaptive` policy and explicit `legacy`. The logs reported
`priority=adaptive:1` and `priority=legacy:0` respectively; both runs completed
with zero new swapout and produced byte-identical frontier files with SHA-256
`99ba046837ba10d1ff7a1ba555d8a48299d696d1d09c32b92bb959c86b5596ff`.
Live pressure selected different post-prefill cache budgets, so their timings
are deliberately not compared. In that build, invalid policy values failed
startup, and the model-free suite covered the default, legacy, fixed-positive,
overflow, zero, and malformed parser cases.

## Practical path to larger gains

The existing profiler can build a domain hotlist without changing model math:

```sh
DS4_EXPERT_PROFILE=/tmp/deepseek-profile.json \
DS4_EXPERT_HOTLIST=/tmp/deepseek-hotlist.txt \
./build/metal-arm64/bin/ds4-bench ...

DS4_METAL_STREAMING_EXPERT_HOTLIST=/tmp/deepseek-hotlist.txt \
./build/metal-arm64/bin/ds4-server ...
```

The production path treats that file as an ordered preload prior and seeds
every entry at priority one. This record's tested binary also exposed
`DS4_METAL_STREAMING_EXPERT_HOTLIST_PRIORITY=legacy|N` for its A/B arms. The
current runtime has elided that experiment switch; reproduce those historical
arms from commit `ba2a729` rather than carrying their policy into release
inference.

Do not train and report on the same prompt as a production claim.  The next
qualification campaign should aggregate representative DSBox conversations,
freeze the hotlist, and benchmark held-out prompts in mirrored order.  The
promotion gate should require exact outputs, zero new swapout, better p50 and
p95 TPOT, and a gain that survives both warm and cold file-cache states.

After route locality, the next structural target is routed-MoE compute.  Warm
profiling puts the paired IQ2 gate/up and Q2 down kernels at roughly 0.54 ms per
layer, or about 23 ms/token across 43 layers before smaller activation/reduction
costs.  A materially faster down path or a validated fusion can therefore have
a double-digit theoretical ceiling.  By contrast, further expert-major I/O
repacking, static page warming, and indexed-attention partitioning do not.
