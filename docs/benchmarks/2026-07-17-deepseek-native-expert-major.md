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

## Remaining release gate

Before replacing the canonical publication, run alternating canonical/native
measurements at 2K, 8K, and at least 16K context, record decode evidence rather
than frontier logits alone, and repeat on every advertised RAM tier. Resident
qualification requires a host that can hold this 80.76 GiB file plus runtime
headroom; it must not be inferred from the 64 GiB SSD run.
