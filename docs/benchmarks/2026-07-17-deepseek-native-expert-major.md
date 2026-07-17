# DeepSeek Flash DS4-native expert-major qualification tranche

Date: 2026-07-17
Host: Apple M5 Pro, 64 GiB unified memory, AC power
Branch: `codex/deepseek-expert-major-v2`

This is the first model-backed qualification tranche for
`ds4.expert_major.v2`. It validates format identity, the physical long-prefill
mapping, grouped Metal math, SSD decode translation, and the interaction
between record size and I/O queue depth. It is not yet the 2K/8K/16K
publication gate.

## Artifact identity

| Artifact | Bytes |
|---|---:|
| Canonical GGUF | 86,720,111,488 |
| Native expert-major v2 GGUF | 86,720,114,272 |

The native file is 2,784 bytes larger. Routed weights occupy 77,913,391,104
bytes in both files; v2 changes their order and adds only its aligned manifest
container. It does not carry a canonical duplicate.

Manifest identities:

- source SHA-256: `efc7ed607ff27076e3e501fc3fefefa33c0ed8cf1eff483a2b7fdc0c2e616668`
- routed payload SHA-256: `e5d34567e2d5397d37127ee1af5dfb8e70c2327a7eb4bd6a5a7162b6f53abf36`
- manifest SHA-256: `9ee2cc83c6d9c553ca0423074ab78d07156e512c40855c3fe96a9089facb0107`

The converter's full byte-level verifier passed before the native file was
installed.

## Correctness

The final fused IQ2 gate/up kernel, its split-kernel control, the canonical
GGUF, and the native GGUF produced the same complete frontier-logit hash:

| Context | Logits SHA-256 |
|---:|---|
| 128 | `84482c0ada999d2e7ff41d1a5f54dc5df4bab96be64b5eca8ecb0ef6b4976b23` |
| 768 | `84ddca7a8356d053627115f333ec26aeff04998cd998347fd282875c9385ce43` |

The first fused implementation exposed a cross-simdgroup race at the K-block
boundary. A threadgroup barrier now prevents the next RHS tile from replacing
shared memory while another simdgroup is still accumulating the current up
projection. Two consecutive native runs, the canonical run, and the split
control are exact after the fix.

## Long-prefill result

With 768 prompt tokens, one generated token, AUTO cache, and no new swapout:

| Layout | Prefill | Decode misses | `pread` calls | Wired peak |
|---|---:|---:|---:|---:|
| Native v2 | 95.82 t/s | 258 | 258 in the diagnostic full-record arm | 13,942 MiB |
| Canonical | 95.74 t/s | 258 | 774 | 10,902 MiB |

The native path maps one real 1.6875 GiB expert-major layer for grouped
prefill instead of addressing synthetic canonical tensor ranges. During long
prefill, native AUTO temporarily contracts the decode cache to the 259-record
correctness floor and grows it lazily afterwards. The 95.82/95.74 result shows
layout parity for the compute path; it is not presented as a material speedup.

After the AUTO policy was finalized, a production-schedule native run of the
same 768-token shape measured 103.69 t/s prefill. It selected 4,387 records,
contracted to 259 for grouped prefill, restored the larger decode budget
lazily, issued the normal 774 component reads for the 258 cold records, and
completed with zero new swapout. Because page-cache and thermal state differed
from the paired table above, that final run is a policy validation, not a new
native-versus-canonical speedup claim.

## Why one record-sized `pread` is slower

Each Flash expert record is 6.75 MiB: adjacent gate, up, and down components.
One synchronous `pread` reduces syscall count by 3x, but also turns three
independent I/O tasks into one outstanding request. The normal loader can run
up to nine tasks concurrently, which better feeds the Apple SSD queue.

At context 128, 256 generated tokens, 3,097 cached experts, and pinned static
weights, the warm comparison was:

| Native I/O schedule | Decode | Cumulative I/O wait | Syscalls |
|---|---:|---:|---:|
| Three component tasks | 14.09 t/s | 2,668 ms | 24,387 |
| One full-record task | 13.61 t/s | 3,033 ms | 8,129 |

The causal control forced the loader to one thread for 64 generated tokens:

| One-thread schedule | Decode | Cumulative I/O wait |
|---|---:|---:|
| Three sequential components | 12.70 t/s | 967 ms |
| One full record | 13.27 t/s | 786 ms |

With parallelism removed, coalescing wins; with the production pool, it loses.
This rules out bad record alignment or an extra host copy as the primary cause.
A gate+up/down two-task hybrid reduced calls by one third but remained inside
run noise (14.20/14.04 versus 14.09/14.01 t/s) and increased measured I/O wait,
so it was not retained.

Production therefore keeps the established three-task schedule. Full-record
I/O remains an explicit diagnostic for other storage:

```sh
DS4_METAL_ENABLE_COALESCED_EXPERT_RECORD_PREAD=1 ./ds4-bench ...
```

After stabilizing the 64 GiB normal-pressure AUTO reserve, a final unpinned
run selected the intended 4,387-record / 28.92 GiB cache tier despite the warm
GGUF page cache. It completed with zero new swapout, 41,735 MiB peak wired
memory, and normal memory pressure. The 9/16 Metal envelope, not the transient
free-page count, is the upper bound.

## Decode parity with the production schedule

The final native/canonical/native-style measurements at context 128 and 256
generated tokens used the same 3,097-record cache budget. Native production
runs measured 14.09 and 14.01 t/s; the intervening canonical run measured
13.99 t/s. All runs had zero new swapout and identical frontier logits. The
layout is therefore decode-neutral in this tranche; the native format's
measured gain is the long-prefill physical view, not fewer production reads.

## Selected-address scheduling campaign

The follow-up campaign tested the optimization stack as a whole first, then
removed one component at a time. This matters for these kernels: expert
grouping can change the value of I/O overlap or threadgroup tiling even when an
isolated run of either feature looks neutral. No experimental implementation
was removed merely because its first combined result lost; every non-promoted
path remains available behind an explicit switch for future hardware and
storage qualification.

All numbers below are from the native v2 artifact, Apple Metal SSD streaming,
the 3,097-record pinned cache, `DS4_M5_PRELOAD_EXPERTS=1`, a 32,768-token KV
allocation, and one generated token. Mirrored comparisons use A/B/B/A order.
Every cited run completed with zero new swapout. Results must be compared only
inside the same binary cohort:

| Cohort | Binary SHA-256 | Repository diff SHA-256 | Use |
|---|---|---|---|
| Frozen scheduling campaign | `47664dad045fa1123554fdf2984939af79ea0067198b4f784f3b5e45ed8f4aa5` | `99a1f162afcd84e98e417d797f0b8c7926aabfe2f22cccd7f816c76e450c4ad9` | Grouping, overlap, balanced reads, and the 760-token stage profile |
| Final auto-policy campaign | `b8780571b3601fdd7ae7da05a60963ff78a6d5bfe1fd975dba33633dd64b3df6` | `6b77956615823e1e32236bdc472981467a43c63d1cc4a9df6c7c9cc2f2bcb3c4` | Production auto/disable check, boundary checks, and the 2K NR16 control |

The earlier expert-tile diagnostic summaries predate binary/diff sealing in the
runner. They are reported separately below and are not mixed into either
promotion mean.

### Exactness ledger

Every arm within a context produced the same complete frontier-logit hash:

| Context | Frontier logits SHA-256 | Covered arms |
|---:|---|---|
| 128 | `71fd3be0732e0fe97b9f104112911dc937896257c604c64eae851e36fa142441` | Default, grouped, tiled, final lower-boundary smoke |
| 256 | `1c8499a11442c05e77d7ae37fb164265b41b63afd25538780f337f166f45b35a` | Group, overlap, balanced, all tile variants, auto and disabled |
| 512 | `a26ededaeaefe028328a9bc292a46b36bc8880f2420919a3bfa6167f801343d4` | Default, grouped, route-4 tile, shared-decode tile |
| 760 | `f5cf769b1869fea8df284678aa1e31aa960bda287922a74cfa54bdf04966cc99` | Default and grouped stage-profile arms |
| 768 | `468728cb84f4126c53c8b6012ccaf5b3fd7abedd200c8955fbf544bb60267684` | Final upper-boundary smoke |
| 2,048 | `bc6620c18c1087bd391cfc8c639c3406da7720f3e55861f48c9a4463cc85827d` | NR32 default and NR16 control |

These hashes qualify equality between arms in this campaign. They should not
be compared with the first converter-qualification hashes earlier in this
document: that tranche used a different benchmark input/configuration cohort.

### Expert-group schedule

The grouped selected-address kernel orders routes expert-major while retaining
the exact route weights and output positions. Its pipeline is compiled lazily,
so a model or shape that does not select this schedule pays no startup pipeline
cost.

| Context | Default | Grouped | Relative result | Decision |
|---:|---:|---:|---:|---|
| 128 | 23.325 t/s | 23.265 t/s | -0.26% | Do not select automatically |
| 256 | 37.015 t/s | 37.345 t/s | +0.89% | Candidate for auto |
| 512 | 45.115 t/s | 45.700 t/s | +1.30% | Candidate for auto |

Whole-run results at 760 tokens were too sensitive to thermal and page-cache
order to use as a promotion number. The isolated layer-0 gate/up stage was
stable enough to identify the direction: grouped reduced elapsed time from
194.299 ms to 191.437 ms, equivalent to roughly +1.5% stage throughput. At
768 tokens the runtime crosses to the existing `mm_id` path, so the selected-
address group schedule is intentionally no longer considered.

The final binary then repeated the actual production decision at 256 tokens:

| A/B/B/A arm | Runs | Mean prefill | Relative result |
|---|---|---:|---:|
| Auto disabled | 32.93, 33.88 t/s | 33.405 t/s | control |
| Auto enabled | 33.65, 33.49 t/s | 33.570 t/s | +0.49% |

The smaller final delta is still positive under the exact shipped selection
logic. Boundary smokes verified that 128 and 768 tokens remain on their old
paths; they produced 23.20 and 92.10 t/s respectively, zero new swapout, and
the exact hashes recorded above. Those two single runs verify routing, not a
throughput comparison.

Production auto-selection is therefore deliberately narrow. It requires all
of the following:

- a native `ds4.expert_major.v2` DeepSeek model;
- Apple Metal SSD streaming, not full-resident or quality mode;
- 256 routed experts, top-6 routing, `IQ2_XXS` gate/up, and `Q2_K` down;
- a prefill batch from 256 through 760 tokens, with graph dumping disabled.

Canonical DeepSeek, Qwen, GLM, other quantizations, decode (`n_tokens == 1`),
full-resident machines, and batches outside that window retain their previous
schedule. The immediate production rollback is:

```sh
DS4_METAL_DISABLE_DEEPSEEK_EXPERT_GROUP_PREFILL=1 ./ds4-bench ...
```

The post-campaign integration adds one safety guard without changing the
eligible grouped kernel measured above: automatic GROUP is requested only when
the backend reports that its ordinary selected-address path is enabled under
the current runtime knobs. Disabling selected-address or pair fusion therefore
keeps the historical fallback. Explicit GROUP/TILE requests remain fail-closed
through the exact `schedule_used` mask.

A functional 256-token routing smoke covered both sides of that contract on
binary `b8efcf6d20b4ed5a81d982acd2b46c5141523419dec9ccbfcc808fb09511d28a`
(pre-documentation diff `06e72f054753c9e69b43187d57391e8c188f2c4917ab79a58636075c8d044bda`).
With `DS4_METAL_DISABLE_STREAMING_PREFILL_BATCH_SELECTED_ADDR=1`, the native
auto-policy completed through the historical fallback (`rc=0`, 28.77 t/s,
zero new swapout, frontier hash
`457a81854338fed8b5128c8294ccaa7ee66c5704136f2f5d6b1c63e7c8c332e7`).
Adding explicit `DS4_METAL_ENABLE_DEEPSEEK_EXPERT_GROUP_PREFILL=1` rejected the
same unavailable schedule at layer 0 with `used=0x0` and `rc=1`, as intended.
These are routing/rollback smokes, not throughput comparisons with the paired
promotion cohort.

Explicit enable remains useful for short-context and canonical-DeepSeek SSD
experiments without widening the native-only auto-policy:

```sh
DS4_METAL_ENABLE_DEEPSEEK_EXPERT_GROUP_PREFILL=1 ./ds4-bench ...
```

### I/O overlap and balanced reads

Starting selected-expert reads immediately after router readback does overlap
them with the shared-expert encode, but on this unified-memory machine the
extra CPU/SSD activity also contends with Metal. Two mirrored 256-token
ablations kept their own controls inside each cohort:

| Comparison | Control mean | Combined mean | Relative result |
|---|---:|---:|---:|
| Overlap (O) vs group + overlap (GO) | O: 33.395 t/s | GO: 33.245 t/s | -0.45% |
| Group + overlap (GO) vs balanced reads added (GOB) | GO: 33.120 t/s | GOB: 33.300 t/s | +0.54% |

Balanced assignment can recover a small part of an overlap run, but it does
not make overlap beat the simpler grouped schedule. Neither feature is in the
auto-policy. Both remain opt-in so they can be retested on a different SSD,
memory tier, or CPU/GPU balance:

```sh
DS4_METAL_ENABLE_DEEPSEEK_PREFILL_IO_OVERLAP=1 ./ds4-bench ...
DS4_METAL_ENABLE_BALANCED_EXPERT_RECORD_PREAD=1 ./ds4-bench ...
```

### Expert-route tiles

The route-tile experiments shared compressed IQ2 weights or decoded values
inside each threadgroup. Reuse was real--3.400 routes/tile at context 256 and
3.632 at context 512--but the grouped control already benefits from GPU cache.
Threadgroup copies, barriers, register pressure, and reduced occupancy cost
more than the saved loads.

| Context | Grouped control | Best route-4 tile | Relative result |
|---:|---:|---:|---:|
| 128 | 24.81 t/s | 23.20 t/s | -6.5% |
| 256 | 36.68 t/s | 35.48 t/s | -3.3% |
| 512 | 51.52 t/s | 49.42 t/s | -4.1% |

At 256 tokens, the alternatives made the tradeoff explicit:

| Tile variant | Prefill | Versus grouped control | Main cost |
|---|---:|---:|---|
| Route 4, row 8 | 33.75 t/s | -8.0% | 19 KiB threadgroup allocation and lower occupancy |
| Route 4, decoded shared values | 34.31 t/s | -6.5% | Decode-loader work and eight barriers |
| Route 2, full shared values | 31.37 t/s | -14.5% | Too few active threads per resident group |
| Route 4, staged | 34.04 t/s | -7.2% | Smaller shared allocation, but eight barriers |
| Route 2, staged | 33.98 t/s | -7.4% | Barrier cost plus small groups |

All tile runs had zero new swapout and exact context-matched hashes. The code
remains lazy and opt-in for architectural follow-up; no production model pays
its buffer upload or pipeline compilation cost:

```sh
DS4_METAL_ENABLE_DEEPSEEK_EXPERT_TILE_PREFILL=1 ./ds4-bench ...
```

The row-8, decoded-shared, route-pair, and staged variants have independent
enable switches for controlled retesting, but none is selected automatically:

```sh
DS4_METAL_ENABLE_DEEPSEEK_EXPERT_TILE_ROW8=1
DS4_METAL_ENABLE_DEEPSEEK_EXPERT_TILE_DECODE_SHARED=1
DS4_METAL_ENABLE_DEEPSEEK_EXPERT_TILE_PAIR=1
DS4_METAL_ENABLE_DEEPSEEK_EXPERT_TILE_STAGED=1
```

### Long-prefill NR16 control

The paired `mm_id` kernel's narrower N tile had shown gains in earlier short
diagnostics, so it was retained through the combined campaign and tested again
at 2,048 tokens on the final binary. The mirrored result reversed direction:

| `mm_id` pair tile | Runs | Mean prefill | Relative result |
|---|---|---:|---:|
| NR32 default | 160.65, 152.52 t/s | 156.585 t/s | control |
| NR16 | 154.41, 155.29 t/s | 154.850 t/s | -1.11% |

Both arms produced the exact 2K hash and zero new swapout. NR16 therefore
remains explicit opt-in and default-off:

```sh
DS4_METAL_ENABLE_MOE_MM_ID_PAIR_NR16=1 ./ds4-bench ...
```

The resulting production change is intentionally one-dimensional: native
DeepSeek SSD prefill gets expert grouping only in the measured 256--760 window.
Overlap, balanced reads, route tiles, and NR16 remain available for combined
future experiments without changing decode or any other model/machine path.

## Remaining release gate

Before replacing the canonical publication, run alternating canonical/native
measurements at 2K, 8K, and at least 16K context, record decode evidence rather
than frontier logits alone, and repeat on every advertised RAM tier. Resident
qualification requires a host that can hold this 80.76 GiB file plus runtime
headroom; it must not be inferred from the 64 GiB SSD run.
