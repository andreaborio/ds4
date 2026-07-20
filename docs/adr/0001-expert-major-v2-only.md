# ADR 0001: ExpertMajor v2 Is The Only Production Expert Store

- Status: Accepted
- Date: 2026-07-20

## Context

Maintaining canonical routed tensors, ExpertMajor v1, external sidecars, and
embedded ExpertMajor v2 in one runtime multiplies admission, tensor-addressing,
cache, and performance paths. That ambiguity makes optimizations hard to prove
and lets partially compatible artifacts reach inference.

ExpertMajor v2 embeds a self-describing expert store in the model artifact and
is the format qualified by the current DeepSeek, GLM, and Qwen Metal paths.

## Decision

Production inference accepts only a validated embedded
`ds4.expert_major.v2` store. Canonical model files may be inspected or converted
offline, but they are not inference fallbacks. ExpertMajor v1 and external
sidecars are rejected.

There is one release path per supported model family. A new store version must
replace v2 through a new ADR and full qualification; it must not become a
permanent parallel semantic path behind a flag.

## Consequences

- Loader and model admission fail before inference on missing or incompatible
  ExpertMajor v2 metadata.
- Startup commands need no admission or sidecar override.
- Tests assert correct admission and rejection as well as resident/SSD behavior.
- Conversion tools record input/output hashes and validate the completed store.
- Code dedicated only to v1, sidecars, or canonical routed-weight fallback may
  be removed rather than maintained.
- Published model artifacts need explicit format/version naming and provenance.

The executable performance and behavior details remain in
[`GOLD_METAL_SSD.md`](../../GOLD_METAL_SSD.md); the current support matrix is in
[`RUNTIME_SUPPORT.md`](../contracts/RUNTIME_SUPPORT.md).
