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
| DeepSeek V4 Flash | 64 GiB | AUTO resolving to resident or SSD; explicit resident and SSD according to the artifact gate | `./ds4 -m DEEPSEEK-DS4-ExpertMajor-v2.gguf` |
| Qwen3.6-35B-A3B | 16 GiB | AUTO resolving to resident or SSD; explicit resident and SSD according to the Qwen admission gates | `./ds4 -m QWEN-DS4-ExpertMajor-v2.gguf` |
| GLM 5.2 | 64 GiB | AUTO resolving to SSD streaming only; resident is rejected | `./ds4 -m GLM-DS4-ExpertMajor-v2.gguf --ctx 8192` |

All rows require Apple Metal and a validated embedded `ds4.expert_major.v2`
store. Explicit residency options are controlled qualification tools; normal
release startup uses AUTO. GLM's explicit SSD/cache controls are diagnostic
rather than an additional qualified startup contract. Detailed planners and gates are in
[`GOLD_METAL_SSD.md`](../../GOLD_METAL_SSD.md) and the family documents.

Qwen has named 16/24/32/36/48/64/96/128 GiB policy profiles, but selection uses
the active device's exact physical RAM, `recommendedMaxWorkingSetSize`, context
runtime, and live pressure. The current 20.81 GB Q4_K_S artifact necessarily
uses SSD on 16 GiB. A 32 GiB host may use resident for shorter contexts when
both gates pass and falls back to SSD otherwise. The named policy is unit-tested
at every cut; performance claims remain limited to the physical hosts and exact
workloads in the release evidence.

The lower-memory extension is Qwen-specific. Hosts below 64 GiB remain outside
the DeepSeek and GLM production contract. Do not infer support for another
family from Qwen's successful admission.

## Model Artifact Admission

Production inference accepts the qualified embedded ExpertMajor v2 artifact for
each supported family. It must fail closed for:

- canonical routed-weight GGUFs used as converter inputs;
- ExpertMajor v1;
- external sidecars;
- missing, malformed, or mismatched ExpertMajor v2 metadata;
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
