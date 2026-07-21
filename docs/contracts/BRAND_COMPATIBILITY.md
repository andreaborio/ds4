# Hebrus Brand Compatibility Contract

Status: bridge contract for the unreleased Hebrus naming transition.

This document defines which public names are canonical, which historical names
remain accepted, and which identifiers must never be rewritten merely for
branding. Runtime and model support remain governed by
[`RUNTIME_SUPPORT.md`](RUNTIME_SUPPORT.md).

## Command names

Every command pair resolves to one real executable and one object graph. The
name used to invoke the executable controls its help/build identity, public
prompt identity, and `engine_id`; it does not select a different runtime.
Runtime diagnostic prefixes remain in the legacy namespace during this bridge
phase.

| Canonical command | Compatibility command | Capability `engine_id` |
| --- | --- | --- |
| `hebrus` | `ds4` | `hebrus` / `ds4` |
| `hebrus-server` | `ds4-server` | `hebrus` / `ds4` |
| `hebrus-agent` | `ds4-agent` | `hebrus` / `ds4` |
| `hebrus-bench` | `ds4-bench` | `hebrus` / `ds4` |
| `hebrus-eval` | `ds4-eval` | `hebrus` / `ds4` |

The compatibility commands remain supported through at least the complete 1.x
release line. Removing them would require usage evidence, a new accepted
decision record, release notes, and a separately tested migration.

Options, defaults, streams, exit codes, model admission, HTTP protocols,
generated tokens, and runtime plans must remain equivalent across each pair.
Only the invoked command identity may differ. `make command-alias-test` checks
the active build profile; macOS `make premerge` also exercises the CPU aliases
through the build-isolation gate.

## Structured runtime identity

All ten command names expose `--capabilities=json` without loading a model.
Consumers must accept `engine_id` values `hebrus` and `ds4` during the bridge,
reject unknown schema versions, and validate the remaining fields exactly.
They must not infer support by scanning source filenames, C symbols, diagnostic
sentences, or binary strings.

Changing `engine_id` does not change the schema-1 model-family list or the
ExpertMajor contract. In particular, both identities report:

- `expert_major.version: 2`;
- `expert_major.tensor: "ds4.expert_major.v2"`;
- GGML storage wire value `0`;
- MLX affine4 storage wire value `1`, with Qwen group size `64`.

## Permanently preserved identifiers

The following are data or provenance, not product copy. They remain unchanged:

- `ds4.expert_major.v2` and the retired `ds4.expert_major.v1` rejection marker;
- ExpertMajor magic, version, offsets, digest semantics, and numeric wire
  values;
- disk-KV magic, version, and payload ABI;
- existing model filenames, immutable revisions, byte counts, and SHA-256
  values containing `DS4`;
- historical benchmark, release, issue, and pull-request links;
- Git history, authorship, copyright, and the fork relationship to
  [`antirez/ds4`](https://github.com/antirez/ds4).

A branded display label may say “Hebrus ExpertMajor v2”; serialized bytes and
identifiers do not change.

## Environment variables and paths

`DS4_*` environment variables remain the accepted runtime namespace in the
bridge release. `HEBRUS_*` aliases are deferred until a central resolver can
enforce one policy consistently. When aliases are introduced, conflicting old
and new values must fail closed; a textual mass replacement is not permitted.

Existing checkouts, model paths, caches, benchmark records, and application
data are not copied or renamed. A repository redirect is transport
compatibility, not permission to reset an absolute local path.

## Repository and application bridge

Until the administrative rename occurs, the engine repository remains
`andreaborio/ds4`. Bridge consumers accept both `andreaborio/ds4` and the
planned `andreaborio/hebrus` identity over HTTPS or SSH. The existing fork is
renamed in place; no history-less replacement is created and the old repository
name must not be reused.

DSBox remains the companion application's visible and persisted identity until
its separate name is selected. The bridge application resolves
`hebrus-server` first, falls back to `ds4-server`, accepts both structured
engine identities, and permits source/string fallback only for a legacy
capability-less `ds4-server`. A capability-less `hebrus-server`, malformed JSON,
unknown schema, wrong backend/role, dirty or wrong revision, and contradictory
ExpertMajor fields fail closed.

## Migration rules for maintainers

1. Add compatibility before changing a public name.
2. Keep each rename tranche separate from inference, storage, kernel, and
   performance changes.
3. Run `make brand-boundary-audit`; reductions pass, while every new or
   increased legacy token group requires an exact reviewed classification.
4. Run the complete model-free gates for identity-only work and the canonical
   model-backed release gates for any change that can affect code generation or
   inference behavior.
5. Do not begin internal file/symbol migration until the DSBox bridge is
   published and the Qwen MLX-affine baseline is qualified.

The architectural rationale and the acceptance conditions for the rename are
recorded in
[`ADR 0005`](../adr/0005-hebrus-naming-and-compatibility-boundary.md).
