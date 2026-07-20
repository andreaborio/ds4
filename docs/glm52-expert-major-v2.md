# GLM 5.2 DS4 ExpertMajor v2

DS4 runs the 244.14 GiB GLM-5.2 Q2_K artifact on Apple Metal while streaming
the routed experts from local storage. The qualified machine is an M5 Pro with
64 GiB unified memory. No claim is made for Macs below 64 GiB.

The GGUF contains one `ds4.expert_major.v2` store. There is no sidecar, second
routed payload, startup repack, or per-machine conversion. For GLM this is the
only supported DS4 execution layout: canonical GLM inference, CPU, CUDA, ROCm,
distributed slices, and resident mode are rejected rather than falling back to
a slower or ambiguous path. `--inspect` remains available to identify files.
There is deliberately no backward-compatibility promise for older GLM files,
sidecars, ExpertMajor revisions, or retired tuning modes. A future layout
revision may require a newly published GGUF.

## Start it

```sh
make -j8
./ds4 \
  -m /absolute/path/to/GLM-5.2-DS4-ExpertMajor-v2-Q2_K.gguf \
  --ctx 8192
```

This starts the interactive CLI. A one-shot request only adds a prompt:

```sh
./ds4 \
  -m /absolute/path/to/GLM-5.2-DS4-ExpertMajor-v2-Q2_K.gguf \
  --ctx 8192 \
  -p "Explain why locality matters in mixture-of-experts inference."
```

The local API uses the same contract:

```sh
./ds4-server \
  -m /absolute/path/to/GLM-5.2-DS4-ExpertMajor-v2-Q2_K.gguf \
  --ctx 8192
```

Do not add `--metal`, `--ssd-streaming`, a cache size, a preload count, a
full-layer count, or ExpertMajor environment variables. On Apple Silicon DS4
selects Metal, detects that the file cannot be resident, activates the GLM Gold
profile, validates the embedded store, and chooses the cache policy. The
explicit 8K context is the predictable 64 GiB starting point; larger contexts
should be admitted and measured independently.

The artifact is hosted at
[`andreaborio/GLM-5.2-DS4-GGUF`](https://huggingface.co/andreaborio/GLM-5.2-DS4-GGUF).

| Property | Value |
| --- | --- |
| File | `GLM-5.2-DS4-ExpertMajor-v2-Q2_K.gguf` |
| Bytes | 262,147,193,504 |
| SHA-256 | `7f5017e3076e706c78f2a5322b035a9e2f6519c65ff5b6be8b2d91aeff61505d` |
| Routed store | 76 layers x 256 experts, 224.44 GiB |
| Expert record | gate + up + down, 12,386,304 contiguous bytes |
| Qualified backend | One local Apple Metal process, SSD residency |
| Minimum qualified memory | 64 GiB unified memory |

## Prefill

Prefill and decode use different schedules:

1. Startup validates every manifest range and reconstructs logical gate, up,
   and down descriptors. The canonical offsets are identities, not physical
   tensor spans.
2. Long prefill resolves one physical ExpertMajor layer and exposes that span
   to Metal. It never tries to map three absent canonical projections.
3. A complete GPU address table maps each expert ID to its gate, up, and down
   position inside the physical record. The grouped Q2_K kernel evaluates the
   prompt routes without a token-time repack.
4. Indexed preparation for layer `L+1` overlaps compute for layer `L`.
5. The multi-GiB layer wrapper is released after the referencing command
   buffer completes; wrappers do not accumulate across 76 layers.

At `--ctx 8192`, two final simple-command gates processed the 288-token
reference prompt at 10.63-10.91 t/s. The best retained run of the same optimized
stack measured 10.75 t/s prefill and 1.81 t/s decode with detailed profiling
enabled. Prefill is no longer the release bottleneck.

## Decode

For each generated token and routed layer:

1. Metal computes the router and selects eight experts.
2. Cache hits bind an existing slab. Each miss resolves the logical expert to
   its adjacent physical ExpertMajor record.
3. GLM reads the complete 11.81 MiB gate+up+down record with one `pread`, not
   three independent component reads. The measured run recorded one successful
   syscall per completed expert load.
4. Advisory router-ahead hints use the same logical-to-physical translation.
   Sending canonical offsets to `F_RDADVISE` was the main port regression: it
   prefetched unrelated file regions and competed with the authoritative read.
5. Cache slabs remain pageable for this GLM path. Avoiding per-buffer `mlock`
   removed roughly 200 ms of allocation preparation without adding swap.
6. The selected address table feeds paired gate/up and down kernels; compact
   DSA KV stays resident. There is no full-layer probe, repack, or whole-model
   remap in decode.

The combined repair moved the broken mainline lane from 1.27 to 1.77-1.81 t/s
decode. The final command after compatibility cleanup measured 1.79 t/s. A
prior rested-storage qualification of the same ExpertMajor runtime had a 1.90
t/s median. A same-condition comparison after repeated I/O measured 1.75 t/s
on the old qualified commit and 1.74 t/s on the new port, showing runtime
parity rather than a new decode regression.

The 64 GiB AUTO policy holds 601 experts: one complete 75 x 8 route plus an
in-flight slot. A larger 1,801-expert cache cut misses by about 22% but slowed
decode from 1.81 to 1.73 t/s. Fewer misses did not compensate for narrower
per-layer I/O concurrency and the larger cache-management footprint.

The predicted-expert install prototype was also removed. It reached only about
75% prediction accuracy, contended for the same pool, and measured 1.15 t/s.
Advisory hints remain because a wrong hint cannot mutate cache or compute state.

## Memory tiers

| Unified memory | Policy | Evidence |
| --- | --- | --- |
| 64 GiB | SSD streaming, 601-expert AUTO cache, 8K recommended starting context | Qualified on M5 Pro; no new swap activity in the release lane |
| 96-128 GiB | SSD streaming with the pressure-admitted adaptive candidate | Runtime path supported, throughput and best cache not yet published |
| 192-256 GiB | SSD streaming remains the safe default; the 244.14 GiB file still needs runtime and OS headroom | Not qualified |
| 384 GiB and above | Resident execution is a separate future qualification lane | Not qualified and currently rejected by the GLM-only SSD contract |

Do not extrapolate the 64 GiB cache result to larger hosts. Each tier needs its
own paired cache sweep and must preserve the same prompt, output, power state,
memory pressure, and storage condition.

## Gates

`make model-free-test` covers canonical/native numeric equivalence for direct,
scalar-batch, and grouped-batch Q2_K kernels, plus manifest corruption and
address translation. `make build-isolation-test` proves Metal and CPU build
products cannot mix. Model-backed commands, incremental timings, root-cause
evidence, and the cross-engine research plan are in
[`benchmarks/2026-07-20-glm52-expert-major-v2.md`](benchmarks/2026-07-20-glm52-expert-major-v2.md).
