# ADR 0002: Apple Metal Is The Production Runtime

- Status: Accepted
- Date: 2026-07-20
- Amendment: ADR 0004 changes only Qwen's minimum-memory policy.

## Context

The actively optimized and qualified runtime is Apple Metal. Carrying equal
maintenance obligations for CPU, CUDA, ROCm, and distributed inference consumes
review and agent context while those paths do not accept the production
ExpertMajor v2 artifacts. It also obscures which performance and correctness
claims are real.

## Decision

Apple Metal on Apple Silicon Macs with at least 64 GB unified memory is the
production runtime. DeepSeek V4 and Qwen3.6 ExpertMajor v2 artifacts have
qualified AUTO, resident, and SSD-streaming paths subject to their family
admission gates. GLM 5.2 is qualified only through normal AUTO resolving to SSD
streaming; an explicit resident request is rejected.

CPU remains reference/debug and build-isolation code, not an inference fallback.
CUDA and ROCm are frozen and their backend source is removed from the active
tree. Distributed inference is retired, its implementation source is removed,
and its former CLI flags remain only as centralized fail-closed tombstones. Git
commit `d8d673858f90834522bbe878951a534d8c6508b4` is the recovery point if active
backend development resumes.

## Consequences

- New runtime optimization work targets Metal unless a later ADR reopens a
  backend.
- Residency behavior remains model-specific; a mode qualified for DeepSeek or
  Qwen must not be inferred for GLM.
- Frozen code is not opportunistically refactored or used as design authority.
- A successful non-Metal build does not imply supported model inference.
- Canonical QA release-reporting requirements remain authoritative: normal
  releases confirm the frozen source stays absent, and any restoration triggers
  the complete former backend lane.
- The removal requires a reviewable commit boundary, documentation update,
  clean supported build/tests, and this recorded recovery commit.
- Reintroducing a backend requires an ADR, an owner, supported model artifacts,
  correctness tests, performance baselines, and release qualification.

This decision narrows product support; it does not permit weakening the
regression rules in [`CONTRIBUTING.md`](../../CONTRIBUTING.md) or
[`QA_BEFORE_RELEASES.md`](../../QA_BEFORE_RELEASES.md).
