# ExpertMajor v2 runtime and qualification roadmap

`ds4.expert_major.v2` is the single routed-expert storage contract for every
MoE family supported by this fork: DeepSeek V4, GLM 5.2, and
Qwen3.6-35B-A3B. The former Qwen fixed-geometry v1 layout, external sidecars,
and canonical inference paths are retired. There is deliberately no backwards
compatibility layer.

## Shared format contract

The offline converter replaces the physical canonical routed tensors with one
opaque GGUF tensor and a checksummed manifest. For every routed layer it
records:

- model-family identity and format version;
- layer and expert counts, including selected-expert count;
- gate, up, and down dimensions, GGML types, and quant block sizes;
- record, component, layer, and payload offsets with explicit alignment;
- source GGUF, payload, and manifest SHA-256 identities.

DS4 reconstructs canonical logical descriptors for graph binding and cache
identity only. Resident mapping, prefill, decode, and SSD cache fills resolve
physical bytes through the manifest. Unknown families or versions, canonical
routed tensors beside a v2 store, incomplete inventories, unsupported types,
overlapping ranges, and geometry mismatches fail before inference.

## Runtime boundary

The release runtime accepts ExpertMajor v2 artifacts only on local Apple Metal.
The normal CLI is the same for all three families:

```sh
./ds4 -m /absolute/path/to/MODEL-DS4-ExpertMajor-v2.gguf --ctx 8192
```

AUTO chooses the family-qualified resident or SSD consumer. No ExpertMajor,
sidecar, backend, cache, preload, or power flag is required. Canonical GGUFs
remain inputs to `inspect`, `build`, and `verify`, but are not executable by
this runtime. CPU, CUDA, ROCm, distributed, v1, and sidecar paths fail closed.

ExpertMajor applies only to routed experts. Embeddings, attention, routers,
shared experts, normalization, and output tensors keep their ordinary GGUF
layout.

## Family status

### Qwen3.6-35B-A3B

- **Implemented:** distinct `qwen35moe` family ID, 40-layer fail-closed
  geometry, generic v2 conversion, logical reconstruction, resident mapping,
  and SSD translation.
- **Qualified on M5 Pro 64 GiB:** resident output matches the retired v1
  control. The 2K v2/v1/v2 lane measured 318.96/29.54, 320.59/29.59, and
  318.83/29.54 prefill/decode t/s.
- **Correctness:** all three evidence files are byte-identical with SHA-256
  `399504c6ce3d4531ee0f2207702e96e2324c9b5c8dbf98adf47dfb9e64cae54d`;
  no new swapout was observed.
- **Release artifact:** 20,808,566,880 bytes, SHA-256
  `d7c43a6388ec20e6fe5530850350f96fdb0ac37c5ce36d3e5f92b172c447f56b`.

### DeepSeek V4

- **Implemented:** manifest-driven mixed-quant conversion, resident and SSD
  consumers, phase-aware prefill cache policy, paired IQ2 gate/up execution,
  and selected-address grouped prefill.
- **Validated:** C/Python corruption, bounds, payload, mixed-type, and
  cross-family tests plus the recorded M5 Pro SSD parity tranches.
- **Release rule:** each Flash or PRO artifact must pass byte verification,
  model-backed output equality, alternating throughput, and swap/memory gates.
  A canonical source is never a runtime fallback while an artifact is awaiting
  qualification.

### GLM 5.2

- **Qualified and published:** distinct family checks, non-zero routed-layer
  prefix, full address tables, grouped prefill, contiguous selected-expert
  decode reads, compact DSA KV, and the measured 601-expert 64 GiB policy.
- **Performance:** the corrected 288+32 lane moved decode from 1.27 to
  1.77-1.81 t/s with exact output and no new swap activity. The prior
  rested-storage median remains 11.08/1.90 t/s.
- **Release artifact:** 262,147,193,504 bytes, SHA-256
  `7f5017e3076e706c78f2a5322b035a9e2f6519c65ff5b6be8b2d91aeff61505d`.

## Next optimization tranches

The common format removes layout fallbacks; performance remains
family-specific. Every new optimization is measured alone and in combination
with the current stack.

1. Qwen: preserve paired gate/up and parallel decode while checking resident
   first-token page behavior at 2K, 8K, and 16K.
2. DeepSeek: qualify each published v2 artifact and retain the measured
   three-task SSD decode schedule unless a storage-specific A/B wins.
3. GLM: evaluate protected-hot/second-hit cache admission, then reduce host
   synchronization without changing the one-record read path.
4. Shared: keep prefill/decode schedules separate, overlap safe I/O with Metal
   work, and reject any token-time repack or accidental canonical range read.

The dated evidence belongs under `docs/benchmarks/`. A format conversion alone
is not a speed claim; promotion requires correctness, throughput, and memory
evidence on every advertised machine tier.
