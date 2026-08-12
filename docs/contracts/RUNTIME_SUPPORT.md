# Runtime Support Contract

This document defines the current qualified inference contract. It is an
admission and maintenance contract, not a list of code that happens to compile.
Testing and release evidence remain governed by
[`CONTRIBUTING.md`](../../CONTRIBUTING.md) and
[`QA_BEFORE_RELEASES.md`](../../QA_BEFORE_RELEASES.md).
The architectural decisions behind this boundary are
[`ADR 0002: Apple Metal Is The Production Runtime`](../adr/0002-apple-metal-production-runtime.md)
and Qwen's lower-memory extension in
[`ADR 0004: Qwen Metal Uses Hardware-Aware Memory Profiles`](../adr/0004-qwen-metal-hardware-memory-policy.md).

## Supported Matrix

| Model family | Minimum unified memory | Qualified Metal modes | Release startup |
| --- | ---: | --- | --- |
| DeepSeek V4 Flash | 64 GiB | AUTO resolving to resident or SSD; explicit resident and SSD according to the artifact gate | `./download_model.sh deepseek-v2`, then `./hebrus -m DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-DS4-ExpertMajor-v2.gguf` |
| Qwen3.6-35B-A3B | 16 GiB for Stable Affine4; 64 GiB for Q2_K_XL Beta | 16 GiB Affine4 is qualified with guarded SSD through a 128K prompt frontier plus 128 decode tokens. The allocation-time release candidate extends deterministic guarded SSD through 24 GiB; above 24 GiB AUTO may resolve to resident or SSD according to the Qwen admission gates. Q2_K_XL Beta has recorded 64 GiB resident/SSD evidence through 32768 tokens | Stable: `./hebrus -m Qwen3.6-35B-A3B-Hebrus-ExpertMajor-v2-MLX-Affine4-G64.gguf`; Beta: `./hebrus -m Qwen3.6-35B-A3B-Hebrus-ExpertMajor-v2-Q2_K_XL.gguf` |
| GLM 5.2 | 64 GiB | AUTO resolving to SSD streaming only; resident is rejected | `./download_model.sh glm-v2`, then `./hebrus -m GLM-5.2-DS4-ExpertMajor-v2-Q2_K.gguf --ctx 8192` |

All rows require Apple Metal and a validated embedded `ds4.expert_major.v2`
store. Explicit residency options are controlled qualification tools; normal
release startup uses AUTO. GLM's explicit SSD/cache controls are diagnostic
rather than an additional qualified startup contract. Detailed planners and gates are in
[`GOLD_METAL_SSD.md`](../../GOLD_METAL_SSD.md) and the family documents.

Qwen has one model/session/graph runtime and two exact weight profiles:
the published MLX affine4/group-64 inventory and the opt-in `published-beta`
Q2_K_XL mixed-GGML inventory. Attention, Gated DeltaNet, KV, routing,
resident/SSD scheduling, cache ownership, and output execution are shared.
Only physical weight decoding and the corresponding dense/routed primitives
differ. The binding is fail-closed and happens once from the complete tensor,
tokenizer, and ExpertMajor storage inventory; it is not a runtime flag. See
[`ADR 0006`](../adr/0006-qwen-dual-weight-codecs.md).

Qwen has named 16/24/32/36/48/64/96/128 GiB policy profiles, but selection uses
the active device's exact physical RAM, `recommendedMaxWorkingSetSize`, context
runtime, and live pressure. The current 20.81 GB MLX affine4/group-64 artifact
is qualified at the established 16 GiB floor only in guarded SSD mode. The
allocation-time hardening candidate applies the same deterministic
SSD-only rule through 24 GiB: AUTO resolves to SSD and an explicit resident
request is rejected on that tier. It must not be described as release-qualified
on 24 GiB until the physical sustained Studio gate passes and the managed
runtime pin advances. A 32 GiB host may use resident for shorter contexts when
both gates pass and falls back to SSD otherwise. The retired
Q4_K_S payload is rejected rather than decoded through a compatibility path.
Q2_K_XL uses the same policy arithmetic with its exact smaller non-routed,
per-component, and per-expert byte geometry; it does not inherit Affine4's
physical lower-memory qualification. The named policy is unit-tested at every
cut; performance claims remain limited to the physical hosts, artifact, and
exact workloads in the release evidence.

Qwen SSD plans on 16 and 24 GiB are guarded tiers. They require an affirmative
normal-pressure signal at admission and every prefill/decode phase entry,
including entries where the configured budget is unchanged. Immediately before
every proposed new Metal slab, normally up to 321 experts but possibly smaller
for the final target tail, the runtime also rechecks live reclaimable memory,
pressure, and the device working-set ceiling. A denied allocation
freezes the effective cache at its already allocated slab capacity and reuses
those slots; it cannot escape through a fresh combined or per-expert buffer.
The routed-expert target remains capped at 3,521 experts (about 5.80 GiB for
the published artifact). A denial before the minimum route floor exists fails
the request closed.

The 16 GiB Stable Affine4 contract stops at a 131,072-token prompt frontier
with 128 decode tokens and one bookkeeping slot (`--ctx 131201`). Larger
allocations fail admission on that tier even though the artifact metadata
declares 262,144 tokens. That larger metadata window remains available only on
higher-memory profiles that pass their own qualified endpoint gates; it is not
a release requirement for the 16 GiB profile.

The allocation-time amendment described above is a publication candidate, not
yet a new physical-hardware qualification claim. The 21 GiB planner
fixture and real-Metal fault injection prove policy mechanics. A physical
24 GiB Mac must still pass the versioned five-request gate before release.

The lower-memory extension is Qwen-specific. Hosts below 64 GiB remain outside
the DeepSeek and GLM qualified contract. Do not infer support for another
family from Qwen's successful admission.

Runtime support and public artifact distribution are separate gates. The
Stable/recommended Qwen store remains the `published`
`Qwen3.6-35B-A3B-Hebrus-ExpertMajor-v2-MLX-Affine4-G64.gguf`, a
20,808,566,880-byte MLX affine4/group-64 artifact with SHA-256
`dd17266185833a9f05531ce366fd7284ddca1ed64aa3dcf06e321e8c72c9ea3d`.
`download_model.sh qwen-v2` pins immutable repository revision
`e002665becd2db618897effb213030fdf92e7e98`; its manifest requires runtime
commit `73a332fef82a0bcdd567d17e0de17aa004cad85d` as its artifact-format
compatibility floor. That field does not qualify an older descendant for the
current hardware policy. A release runtime must also contain the current
accepted safety fixes; the Studio/runtime pin for the 24 GiB lane must advance
to the eventual published commit containing this allocation-time hardening. The
older `Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-Q4_K_S.gguf` store is
`negative-only`, not a downloadable runtime fallback. The machine-readable
[Qwen release contract](qwen-release.json) is the canonical identity record.

The runtime additionally admits the exact opt-in `published-beta` Q2_K_XL
artifact `Qwen3.6-35B-A3B-Hebrus-ExpertMajor-v2-Q2_K_XL.gguf`:
12,290,632,032 bytes, SHA-256
`30c22f70aff0f05986b517ee4ad8fef554a1b5aab6971c9ca09f999566d30143`,
with embedded payload SHA-256
`ccc3fbc2405d1dd73f8ac15741b0277514de4f46b80818531297ea9ffa0c6a3c`.
`download_model.sh qwen-q2-beta` pins immutable repository revision
`e002665becd2db618897effb213030fdf92e7e98` and requires runtime commit
`42e2fec2a7dbb14a42e7a5612dfec00e33d443ca`. It is nonrecommended, has a
64 GiB minimum, and is qualified only through exactly 32768 context tokens.
It is not a `qwen-v2` replacement and makes no full-window claim. The near-262K
endpoint lane remains mandatory before Stable/full-window promotion.

## Model Artifact Admission

Qualified inference accepts the embedded ExpertMajor v2 artifact for
each supported family. It must fail closed for:

- canonical routed-weight GGUFs used as converter inputs;
- ExpertMajor v1;
- external sidecars;
- missing, malformed, or mismatched ExpertMajor v2 metadata;
- Qwen ExpertMajor v2 stores whose logical tensor inventory, tokenizer
  metadata, and routed payload do not match either exact MLX
  affine4/group-64 or Q2_K_XL profile;
- artifact/model combinations that have not passed their family gates.

Normal startup has no ExpertMajor admission bypass and needs no sidecar,
backend, cache, preload, or power environment flag. See
[`0001-expert-major-v2-only.md`](../adr/0001-expert-major-v2-only.md) and
[`0003-no-model-backward-compatibility.md`](../adr/0003-no-model-backward-compatibility.md).

## Non-Production And Frozen Paths

| Path | Current policy |
| --- | --- |
| CPU | Reference/debug and compile-isolation path; never a production model fallback |
| CUDA | Frozen; backend source absent from the active tree; no inference claim |
| ROCm | Frozen; backend source absent from the active tree; no inference claim |
| Distributed | Retired; implementation source absent; former CLI flags fail closed before model loading |

The last pre-removal CUDA, ROCm, and distributed source is recoverable from Git
commit `d8d673858f90834522bbe878951a534d8c6508b4` if work resumes. Do not restore
it opportunistically and do not infer support from historical build or benchmark
results. The canonical QA document still governs release sign-off: record these
paths as source-absent, or run the complete applicable reactivation gate if any
source/build integration returns.

## Compatibility Boundary

The project intentionally provides no backward compatibility for old model
containers, old routed-expert layouts, sidecars, or removed model admission
flags. New format versions require explicit admission, conversion tooling,
provenance, validation, model-backed correctness evidence, and performance
evidence before they replace ExpertMajor v2.

This model-artifact policy does not silently waive API, disk-KV, tokenizer, or
snapshot correctness. Incompatible cached state must be rejected explicitly;
protocol changes follow their own tests and release gates.

## Changing This Contract

A support change requires all of the following in one reviewed tranche:

1. An accepted ADR with motivation and consequences.
2. Updated admission and fail-closed tests.
3. Updated converter/download/user documentation where applicable.
4. Correctness and performance evidence required by `CONTRIBUTING.md`.
5. The complete applicable release evidence from `QA_BEFORE_RELEASES.md`.
6. Updates to this document, `CODEMAP.md`, and `FORK_NOTES.md` when the
   fork/upstream boundary changes.

A code path, build target, published file, historical benchmark, or successful
ad-hoc run does not change this contract by itself.
