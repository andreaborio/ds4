# ADR 0003: No Model Artifact Backward Compatibility

- Status: Accepted
- Date: 2026-07-20

## Context

Compatibility branches for old containers, tensor layouts, sidecars, and
admission flags preserve code that is not part of the current release path.
They increase the number of states every loader, cache, graph, test, and coding
agent must understand, while receiving little or no current qualification.

## Decision

The runtime does not preserve backward compatibility with old model artifacts.
It accepts the current qualified model-family contract and fails closed for old,
ambiguous, or partially compatible artifacts. There is no model fallback or
environment-variable bypass.

Migration is performed offline by explicit conversion tooling into the current
qualified format. When a future format is accepted, the project may remove its
predecessor after publishing the converter and validation evidence; it need not
run both formats indefinitely.

## Consequences

- Compatibility-only loader, graph, cache, flag, and test code should be
  deleted once it has no current contract owner.
- Error messages identify the unsupported artifact and required conversion;
  they must not silently reinterpret tensors or metadata.
- Converters and published models carry provenance, version, and hashes.
- Old model containers remain recoverable through published artifacts and Git
  history, not through permanent production branches.
- Disk-KV/session state with incompatible model or layout identity is rejected,
  not adapted implicitly.

This ADR applies to model artifact compatibility. API protocol changes and
user-data compatibility remain governed by their own contracts and regression
tests. The active model boundary is defined in
[`RUNTIME_SUPPORT.md`](../contracts/RUNTIME_SUPPORT.md).
