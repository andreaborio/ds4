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

Hebrus reconstructs canonical logical descriptors for graph binding and cache
identity only. Resident mapping, prefill, decode, and SSD cache fills resolve
physical bytes through the manifest. Unknown families or versions, canonical
routed tensors beside a v2 store, incomplete inventories, unsupported types,
overlapping ranges, and geometry mismatches fail before inference.

## Runtime boundary

The release runtime accepts ExpertMajor v2 artifacts only on local Apple Metal.
The normal CLI is the same for all three families:

```sh
./hebrus -m /absolute/path/to/MODEL-DS4-ExpertMajor-v2.gguf --ctx 8192
```

`./ds4` remains a byte-identical compatibility alias; it is no longer the
primary command in new documentation.

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
- **One runtime, two codecs:** model/session state, Gated DeltaNet, attention,
  KV, routing, cache, resident/SSD policy, and scheduling are shared. Only the
  Affine4 and routed-IQ physical weight primitives differ, as required by
  [`ADR 0006`](adr/0006-qwen-dual-weight-codecs.md).
- **Qualified and published:** MLX affine4/group-64 resident and SSD execution
  on the hardware lanes recorded in the current benchmark index.
- **Qualified implementation, publication pending:** exact Q2_K_XL
  mixed-GGML inventory, 12,290,632,032 bytes, SHA-256
  `30c22f70aff0f05986b517ee4ad8fef554a1b5aab6971c9ca09f999566d30143`.
  The immutable distribution revision and release-contract entry remain
  pending; the Affine4 release is preserved.
- **Correctness:** the final resident and SSD lane retains deterministic token
  and logit comparisons with no new swapout in the recorded qualification.
- **Release artifact:**
  `Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-MLX-Affine4-G64.gguf`,
  20,808,566,880 bytes, SHA-256
  `dd17266185833a9f05531ce366fd7284ddca1ed64aa3dcf06e321e8c72c9ea3d`.
  The immutable repository revision is
  `7bf9c3f7f6136aeb2599d75ee61c0cc2f18e2b02`; its manifest requires runtime
  artifact-format floor `73a332fef82a0bcdd567d17e0de17aa004cad85d`;
  release safety still requires the current runtime-support policy.
  The former Q4_K_S artifact is retained only as a fail-closed negative case.
  Experimental Affine2 was rejected and removed rather than becoming a third
  storage codec.

### DeepSeek V4

- **Implemented:** manifest-driven mixed-quant conversion, resident and SSD
  consumers, phase-aware prefill cache policy, paired IQ2 gate/up execution,
  and selected-address grouped prefill.
- **Validated:** C/Python corruption, bounds, payload, mixed-type, and
  cross-family tests plus the recorded M5 Pro SSD parity tranches.
- **Adaptive 64 GiB lane:** after isolating GLM-only victim reuse, AUTO selects
  up to 4,387 records (17 complete route cycles) under the live pressure
  ceiling. The final 128+256 gate measured 25.21/14.19 t/s prefill/decode,
  exact frontier logits, zero swapout, and 42,018 MiB peak wired memory.
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
- **Adaptive memory contract:** the qualified 64 GiB DS4-managed pageable tier
  remains 601 records; unused process budget stays available to the macOS file
  cache. Larger explicit tiers reduced misses but regressed end-to-end decode. Different
  physical-memory tiers keep a pressure-derived ceiling and require their own
  qualification.
- **Release artifact:** 262,147,193,504 bytes, SHA-256
  `7f5017e3076e706c78f2a5322b035a9e2f6519c65ff5b6be8b2d91aeff61505d`.

## Next optimization tranches

The common format removes layout fallbacks; performance remains
family-specific. Every new optimization is measured alone and in combination
with the current stack.

1. Qwen: preserve the shared graph and profile-specific weight decoders; focus
   next on exact Affine4 SSD decode and long-prefill work reduction without
   introducing a third codec or permanent selector.
2. DeepSeek: qualify each published v2 artifact and retain the measured
   three-task SSD decode schedule unless a storage-specific A/B wins.
3. GLM: evaluate protected-hot/second-hit cache admission, then reduce host
   synchronization without changing the one-record read path.
4. Shared: keep prefill/decode schedules separate, overlap safe I/O with Metal
   work, and reject any token-time repack or accidental canonical range read.

The dated evidence belongs under `docs/benchmarks/`. A format conversion alone
is not a speed claim; promotion requires correctness, throughput, and memory
evidence on every advertised machine tier.
