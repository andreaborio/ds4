# Runtime Support Contract

This document defines what the current fork supports in production. It is an
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
| DeepSeek V4 Flash | 64 GiB | AUTO resolving to resident or SSD; explicit resident and SSD according to the artifact gate | `./hebrus -m DEEPSEEK-DS4-ExpertMajor-v2.gguf` |
| Qwen3.6-35B-A3B | 16 GiB | AUTO resolving to resident or SSD; explicit resident and SSD according to the Qwen admission gates | `./hebrus -m Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-MLX-Affine4-G64.gguf` |
| GLM 5.2 | 64 GiB | AUTO resolving to SSD streaming only; resident is rejected | `./hebrus -m GLM-DS4-ExpertMajor-v2.gguf --ctx 8192` |

All rows require Apple Metal and a validated embedded `ds4.expert_major.v2`
store. Explicit residency options are controlled qualification tools; normal
release startup uses AUTO. GLM's explicit SSD/cache controls are diagnostic
rather than an additional qualified startup contract. Detailed planners and gates are in
[`GOLD_METAL_SSD.md`](../../GOLD_METAL_SSD.md) and the family documents.

Qwen has named 16/24/32/36/48/64/96/128 GiB policy profiles, but selection uses
the active device's exact physical RAM, `recommendedMaxWorkingSetSize`, context
runtime, and live pressure. The current 20.81 GB MLX affine4/group-64 artifact
necessarily uses SSD on 16 GiB. A 32 GiB host may use resident for shorter
contexts when both gates pass and falls back to SSD otherwise. The retired
Q4_K_S payload is rejected rather than decoded through a compatibility path.
The named policy is unit-tested at every cut; performance claims remain limited
to the physical hosts and exact workloads in the release evidence.

Qwen SSD plans on 16 and 24 GiB are guarded tiers. They require an affirmative
normal-pressure signal at admission and every prefill/decode phase entry,
including entries where the configured budget is unchanged but lazy slabs can
still populate, and cap the routed-expert cache at 3,521 experts (about
5.80 GiB for the published artifact). This prevents a sustained decode from
turning an optimistic warm file-cache snapshot into unsafe wired slab growth.
A higher 24 GiB ceiling may replace it only after the complete physical
context/cache safety matrix passes; the current support boundary is unchanged.

The lower-memory extension is Qwen-specific. Hosts below 64 GiB remain outside
the DeepSeek and GLM production contract. Do not infer support for another
family from Qwen's successful admission.

Runtime support and public artifact distribution are separate gates. The only
runnable Qwen store is the `published`
`Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-MLX-Affine4-G64.gguf`, a
20,808,566,880-byte MLX affine4/group-64 artifact with SHA-256
`dd17266185833a9f05531ce366fd7284ddca1ed64aa3dcf06e321e8c72c9ea3d`.
`download_model.sh qwen-v2` pins immutable repository revision
`7bf9c3f7f6136aeb2599d75ee61c0cc2f18e2b02`; its manifest requires runtime
commit `73a332fef82a0bcdd567d17e0de17aa004cad85d` or a compatible descendant. The
older `Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-Q4_K_S.gguf` store is
`negative-only`, not a downloadable runtime fallback. The machine-readable
[Qwen release contract](qwen-release.json) is the canonical identity record.

## Model Artifact Admission

Production inference accepts the qualified embedded ExpertMajor v2 artifact for
each supported family. It must fail closed for:

- canonical routed-weight GGUFs used as converter inputs;
- ExpertMajor v1;
- external sidecars;
- missing, malformed, or mismatched ExpertMajor v2 metadata;
- Qwen ExpertMajor v2 stores whose routed payload is not MLX affine4/group-64;
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
