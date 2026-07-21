## Summary

Explain the user-visible outcome and why this change belongs in Hebrus.

## Scope

- [ ] Behavioral or runtime change
- [ ] Mechanical rename or compatibility change
- [ ] Documentation, tests, tooling, or build-only change
- [ ] Performance-sensitive change

Primary components changed:

## Upstreamability

Select one and explain the boundary:

- [ ] General, reproducible, and safe for the affected backends
- [ ] Potentially general, but not yet proven across the affected paths
- [ ] Model-, quant-, or hardware-specific research
- [ ] Equivalent change already exists upstream
- [ ] Measured regression without a necessary correctness fix

Rationale and any upstream issue/patch reference:

## Compatibility contract

- [ ] `hebrus*` remains the canonical public interface, or the intentional change is documented.
- [ ] Applicable `ds4*` aliases and `DS4_*` environment variables remain compatible, or migration coverage is included.
- [ ] ExpertMajor/GGUF serialized identifiers and metadata remain compatible, or the format change has an ADR, versioning plan, and migration tests.
- [ ] Applicable disk-KV data, server/API behavior, and exit codes remain compatible, or the break is explicitly documented and tested.
- [ ] Not applicable; this change cannot affect public or serialized compatibility.

Compatibility notes:

## Validation

List exact commands and results. Do not write only “tests pass.”

```text
# command
# result
```

Model-free coverage:

- [ ] Targeted unit/integration tests
- [ ] Compatibility or CLI checks where applicable
- [ ] Documentation links/context audit where applicable
- [ ] `git diff --check`

## Model-backed evidence

- [ ] Required and provided below
- [ ] Not applicable; explain why

When required, include:

- Exact model family, filename, byte size, SHA-256, provenance/conversion, quantization, and manifest identity.
- Exact hardware, OS build, power state, build identity, backend, and resolved resident/SSD plan.
- Successful decoded-token evidence plus parity/cosine/logit or other appropriate correctness checks.
- Short (128), medium (2048), large (8192), and long (32768) lanes with at least 128 decoded tokens at each frontier when making a general performance claim.
- 65K/100K evidence when changing long-context memory behavior or scheduling.
- A/B/B/A ordering, repeated samples, separate cold/warm results, and memory-pressure/swap observations for performance promotion.

Evidence or reason it is not applicable:

## Documentation and release impact

- [ ] User-facing behavior and current support status are documented.
- [ ] Release QA or support-matrix updates are included when this changes a release contract.
- [ ] No documentation or release change is needed; explain why.

## Repository hygiene and security

- [ ] No model weights/blobs, local logs, benchmark scratch, binaries, secrets, private prompts, or personal absolute paths are committed.
- [ ] Generated files have reproducible provenance and are intentionally tracked.
- [ ] This pull request does not publicly disclose an uncoordinated vulnerability; security reports follow [`SECURITY.md`](../SECURITY.md).
- [ ] I reviewed the complete diff, including generated and compatibility changes.

## Reviewer notes

Call out risky assumptions, deferred work, and the smallest useful review order.
