# Qwen exact K/V pair-blit promotion (2026-07-22)

Status: accepted for merge by explicit owner direction as an exact structural
optimization.  The full performance-promotion matrix is incomplete; the valid
cohorts establish directional non-regression, not a statistically significant
SSD speedup or general release-performance claim.

Decision: replace the two consecutive full-attention K/V cache blit encoders
with one encoder containing the same two ordered byte copies.  Enable the path
for both resident and SSD Qwen execution.  Keep the M1 Pro 16 GiB timing lane
marked invalid because unrelated host stalls moved between control and
candidate phases; its correctness evidence remains valid.  This owner-directed
code merge is recorded separately from completion of the repository's release
performance gate.

Supersedes: no model format or storage contract.  The accepted artifact remains
the 20,808,566,880-byte
`Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-MLX-Affine4-G64.gguf`, SHA-256
`dd17266185833a9f05531ce366fd7284ddca1ed64aa3dcf06e321e8c72c9ea3d`.
SSD expert streaming, persistent F32 K/V state, copied bytes, copy order, and
command-buffer commit count are unchanged.

## Structural change

`ds4_gpu_tensor_copy_pair()` validates both source/destination ranges before
recording either operation, closes the active compute encoder once, records K
and V in one `MTLBlitCommandEncoder`, and ends that encoder once.  Invalid
second-copy bounds reject the complete pair without partially recording the
first copy.

Qwen has ten full-attention layers.  The benchmarked 2K prefill uses one batch
per layer; the SSD macro 8K prefill uses four 2K chunks per layer.  Decode
performs one K/V pair per full-attention layer and generated token.

| Work unit | Baseline blit encoders | Pair-blit | Reduction |
| --- | ---: | ---: | ---: |
| One K/V cache update | 2 | 1 | 50% |
| 2K prefill | 20 | 10 | 50% |
| SSD macro 8K prefill | 80 | 40 | 50% |
| Decode, 128 tokens | 2,560 | 1,280 | 50% |
| Copy commands, copied bytes, command-buffer commits | unchanged | unchanged | 0% |

The 50% encoder-count reduction is exact from the call graph and candidate
disassembly.  It does **not** mean 50% fewer SSD reads or Metal command-buffer
commits.

## Artifact identity and protocol

All retained arms used Metal, the `--power 100` GPU duty target, eight CPU
helper threads, AC power, and the same story prompt (SHA-256
`29363eab21bbbccaeea8e13f669e7ce05e8eafc48e31fcf9b725edabb2058666`),
2,177 context allocation at the 2K frontier, 128 greedy decode tokens, strict
Qwen full-stack admission, and separate logits/decode-evidence directories.
The M5 cohorts used discarded baseline and candidate warm-ups and interleaved
orders; the M1 32 GiB retained warm 2K and interleaved 8K cohorts.

| Artifact | SHA-256 |
| --- | --- |
| Baseline M5 executable | `5b33b52dfa77feaa1dc99354126a0f6ec7723ccb0186732af0c6ae7aa93b28c6` |
| Candidate M5 executable | `86acf37c3caf889e7b8ec2cd18f1d2fd55ffaff333c0a014e793a8d448e4490b` |
| Candidate `ds4.c` | `462ebdc2b185c1a95051ba5ff5daf7fe66d504217e4d3a611d1f6c763b353d43` |
| Candidate `ds4_gpu.h` | `d889e99fe6a819f06ee7f66389617d25e0c228a3a2dcfd4ac58d4df99e075d4c` |
| Candidate `ds4_metal.m` | `32701593a371cb43299f01146830b89a61d632c9d1b956d7809dd67474fe6af9` |
| Runtime Metal library | `1e26e4135c620326fb1d8d6bc23aafd43183b78353734cb8c5d76ddfc7781719` |

The integrated source is byte-identical to the frozen candidate for all three
implementation files.  A rebuilt Mach-O has a different link UUID/ad-hoc
signature, but its `__text`, `__cstring`, symbols, and pair-blit callsites match
the frozen candidate.

## M5 Pro 64 GiB results

Host: Mac17,9, Apple M5 Pro 18-core, 64 GiB, macOS 26.5.2 build
25F84, AC at 100%.  Swap stayed at 156.25 MiB before and after every arm.

### Forced SSD, 2K+128, ABBA-BAAB (n=4 per side)

| Metric | Baseline | Pair-blit | Delta |
| --- | ---: | ---: | ---: |
| Prefill throughput | 896.665 t/s | 890.553 t/s | -0.682% |
| Prefill wall | 2,284.064 ms | 2,299.855 ms | +0.691% |
| TTFT | 2,284.812 ms | 2,300.611 ms | +0.691% |
| Decode throughput | 32.620 t/s | 32.865 t/s | +0.751% |
| Decode wall | 3,925.062 ms | 3,895.231 ms | -0.760% |
| TPOT p50 | 28.115 ms | 27.728 ms | -1.377% |
| TPOT p95 | 43.403 ms | 43.513 ms | +0.253% |
| Request wall | 6,209.126 ms | 6,195.086 ms | -0.226% |
| Prefill `pread` | 27,891 / 15.320984 GiB / 325.102 ms | 27,891 / 15.320984 GiB / 333.058 ms | bytes 0%; time +2.447% |
| Decode `pread` | 19,017 / 10.446350 GiB / 325.816 ms | 19,017 / 10.446350 GiB / 328.579 ms | bytes 0%; time +0.848% |

All eight arms produced logits SHA-256
`d0c0d7f23fa53717fc2bcb8c173f0f06f7b1d97489805ac2662339e93de49276`
and decode evidence SHA-256
`cf4e29dcc2e20bc4f75f38ce136d5d02e218f9aca06b53ea8bbd8c743b74ea81`.

### Resident, 2K+128, ABBA (n=2 per side)

| Metric | Baseline | Pair-blit | Delta |
| --- | ---: | ---: | ---: |
| Prefill throughput | 1,660.300 t/s | 1,651.965 t/s | -0.502% |
| Prefill wall | 1,233.596 ms | 1,240.181 ms | +0.534% |
| TTFT | 1,234.351 ms | 1,240.905 ms | +0.531% |
| Decode throughput | 56.520 t/s | 58.055 t/s | +2.716% |
| Decode wall | 2,264.858 ms | 2,204.782 ms | -2.653% |
| TPOT p50 | 17.670 ms | 17.039 ms | -3.568% |
| TPOT p95 | 18.639 ms | 18.330 ms | -1.658% |
| Request wall | 3,498.454 ms | 3,444.963 ms | -1.529% |
| Expert `pread` | 0 | 0 | unchanged |

All four retained arms produced logits SHA-256
`0c53d64c958b9830fa5743d90a3ddab600ccfcb696950077e8162e0e511a34e2`
and decode evidence SHA-256
`2e324d8b792b61e4090d57088e8cc4377f94c2b90aacc748c5a1345dd9f1e3c9`.

## M1 Pro 32 GiB forced-SSD results

Host: MacBookPro18,1, Apple M1 Pro 10-core, 32 GiB, macOS 26.5.2 build
25F84, AC.  Swap was zero throughout.  `audiomxd`, WindowServer, `configd`,
and SecurityAgent remained active at similar load in every retained arm, so
these cohorts support exactness and directional non-regression, not a precise
speedup claim.

### Warm 2K+128, ABBA (n=2 per side)

| Metric | Baseline | Pair-blit | Delta |
| --- | ---: | ---: | ---: |
| Prefill throughput | 285.835 t/s | 285.260 t/s | -0.201% |
| Prefill wall | 7,165.122 ms | 7,179.881 ms | +0.206% |
| TTFT | 7,166.385 ms | 7,181.103 ms | +0.205% |
| Decode throughput | 16.780 t/s | 17.010 t/s | +1.371% |
| Decode wall | 7,629.025 ms | 7,524.033 ms | -1.376% |
| TPOT p50 | 57.162 ms | 56.373 ms | -1.380% |
| TPOT p95 | 77.689 ms | 77.285 ms | -0.520% |
| Request wall | 14,794.146 ms | 14,703.914 ms | -0.610% |
| Prefill `pread` | 27,948 / 15.352295 GiB / 1,541.977 ms | 27,948 / 15.352295 GiB / 1,531.547 ms | bytes 0%; time -0.676% |
| Decode `pread` | 17,835 / 9.797058 GiB / 1,103.407 ms | 17,835 / 9.797058 GiB / 1,109.940 ms | bytes 0%; time +0.592% |

Logits SHA-256 is
`64a8d7a3551898e50cb99e874f4344bfa9b13975d98d3548ce1bd7b8242e47f7`;
decode evidence SHA-256 is
`4a092796f4a33eb154f99c0836302884838e7f41be60a144d5e013c234de41b5`.
A separate first cold baseline measured 208.17 prefill t/s, 9.839 s TTFT,
16.76 decode t/s, and 3,913.4 ms prefill `pread`; it is not included above.

### 8K+128, ABBA (n=2 per side)

| Metric | Baseline | Pair-blit | Delta |
| --- | ---: | ---: | ---: |
| Prefill throughput | 314.880 t/s | 314.370 t/s | -0.162% |
| Prefill wall | 26,016.767 ms | 26,059.003 ms | +0.162% |
| TTFT | 26,017.894 ms | 26,060.331 ms | +0.163% |
| Decode throughput | 15.360 t/s | 15.330 t/s | -0.195% |
| Decode wall | 8,332.882 ms | 8,351.328 ms | +0.221% |
| TPOT p50 | 61.731 ms | 62.062 ms | +0.535% |
| TPOT p95 | 88.829 ms | 88.636 ms | -0.217% |
| Request wall | 34,349.649 ms | 34,410.331 ms | +0.177% |
| Prefill `pread` | 29,466 / 16.186157 GiB / 1,919.100 ms | 29,466 / 16.186157 GiB / 1,940.294 ms | bytes 0%; time +1.104% |
| Decode `pread` | 17,856 / 9.808594 GiB / 1,339.251 ms | 17,856 / 9.808594 GiB / 1,339.947 ms | bytes 0%; time +0.052% |

Logits SHA-256 is
`ad070919ac16a9eb5c77c4fcd933a422721c20d58454668d990004f360e7a75f`;
decode evidence SHA-256 is
`57eb0d9c430194bc70d4215d94c8de6d009d0c5a165dfeb3de4acaacdb5164ff`.

## M1 Pro 16 GiB forced-SSD gate

Host: MacBookPro18,3, Apple M1 Pro 8-core, 16 GiB, macOS 26.5 build
25F71, AC at 100%.  Swap stayed at 93.75 MiB and `pmset` reported no thermal
or performance warning.  Correctness passed, but the timing cohort is invalid.

The first adjacent pair without a long stall is retained only as provisional
descriptive data:

| Metric | Baseline A1 | Pair-blit B1 | Raw delta |
| --- | ---: | ---: | ---: |
| Prefill throughput | 187.61 t/s | 203.71 t/s | +8.582% |
| TTFT | 10.918 s | 10.055 s | -7.906% |
| Decode throughput | 10.90 t/s | 10.93 t/s | +0.275% |
| TPOT p50 / p95 | 84.982 / 139.192 ms | 84.525 / 138.571 ms | -0.538% / -0.446% |
| Request wall | 22.660 s | 21.762 s | -3.962% |
| Prefill `pread` | 15.352295 GiB / 3,916.386 ms | 15.352295 GiB / 3,625.956 ms | bytes 0%; time -7.416% |
| Decode `pread` | 13.981201 GiB / 4,501.204 ms | 13.981201 GiB / 4,515.006 ms | bytes 0%; time +0.307% |

The apparent prefill win is not attributable to pair-blit: baseline A1 to A2
drifted by about +9.1%, while candidate B1 to the later non-stalled B3 drifted
by about -9.4%.  More importantly, unrelated pauses moved between variants:

| Invalid arm | Prefill | Decode | Non-I/O stall |
| --- | ---: | ---: | ---: |
| Candidate B2 | 20.82 t/s | 10.91 t/s | about 88 s during prefill |
| Baseline A2 | 204.64 t/s | 0.69 t/s | about 174 s in one decode interval |
| Baseline A3 | 10.61 t/s | 10.89 t/s | about 189 s during prefill |

`pread` time remained about 3.6--4.6 s while phase walls grew to 98--193 s.
The stalls coincided with sustained `audiomxd`/`configd` and intermittent
`airportd`/SecurityAgent activity and even delayed the watchdog sleep.  No 8K
timing was retained after the 2K gate became invalid.  All six completed
logits files nevertheless have SHA-256
`60ec3917227e536e588554bfa6842862e400e8b3c718c4288676e5bd7adb2d3f`;
all decode evidence has SHA-256
`4a092796f4a33eb154f99c0836302884838e7f41be60a144d5e013c234de41b5`.

## Correctness, safety, and merge rationale

| Gate | Result |
| --- | --- |
| Pair versus two independent exact copies | Byte-identical |
| Invalid second range | Complete pair rejected before first copy is recorded |
| Qwen logits/evidence on every completed arm | Identical within each host/context cohort |
| Metal kernel suite | Pass on M5 Pro and both M1 Pro hosts |
| Qwen session, attention-reference, and state suites | Pass |
| New swap | None |
| SSD expert reads, loads, hit plan | Unchanged within each A/B cohort |
| Valid mean end-to-end inference regressions | All below 2.0% |
| Bounded resource reduction | K/V blit-encoder creation reduced exactly 50% |

The speed effect on forced SSD is smaller than cohort noise.  The owner-directed
merge rests on exact output, the deterministic 50% reduction in the targeted
encoder resource, zero safety/I/O change, sub-2% retained end-to-end
regressions, and the measured resident decode improvement.  The +2.447% M5 SSD
prefill `pread` time is reported above as variable resource telemetry;
bytes/syscalls were identical and TTFT changed only +0.691%.  This is a
control-overhead optimization, not an SSD-byte optimization.

The 128, 32K, 65K, and 100K acceptance arms were not run for this isolated
candidate, the M1 16 GiB timing lane was invalid, and the M1 32 GiB lanes were
background-contaminated.  Those omissions prevent claiming completion of the
normal release performance matrix even though the code is merged by explicit
direction.

Raw result bundles at decision time:

- M5 SSD: `/private/tmp/ds4-pair-blit-results-20260722/m5-64-ssd2k-abba-baab`
- M5 resident: `/private/tmp/ds4-pair-blit-results-20260722/m5-64-resident2k-abba`
- M1 32 GiB local copy: `/private/tmp/ds4-pair-blit-m1-32-results-20260722`
- M1 16 GiB local copy: `/private/tmp/ds4-pairblit16-results-20260722`
