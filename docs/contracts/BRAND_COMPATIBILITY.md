# Hebrus Brand Compatibility Contract

Status: bridge contract for the unreleased Hebrus naming transition.

This document defines which public names are canonical, which historical names
remain accepted, and which identifiers must never be rewritten merely for
branding. Runtime and model support remain governed by
[`RUNTIME_SUPPORT.md`](RUNTIME_SUPPORT.md).

## Canonical logo asset

The public Hebrus mark is the maintainer-supplied
[`docs/media/hebrus-logo.png`](../media/hebrus-logo.png). Its source pixels stay
unchanged: the repository freezes the 1254 x 1254 RGBA PNG at SHA-256
`4be8949c73bd52e7abef58396dcd57f636165a8bb6cd6d536a600bcbf880594c`.
Presentation surfaces may add a CSS `drop-shadow()` at render time; they must
not bake shadows, text, color changes, crops, or other edits into this master.

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

The compatibility commands are implemented and tested in the current bridge.
Their proposed complete-1.x support horizon remains unaccepted in
[`ADR 0005`](../adr/0005-hebrus-naming-and-compatibility-boundary.md); it is not
yet a release-line promise. Removing the current aliases would require usage
evidence, a new accepted decision record, release notes, and a separately
tested migration.

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
- published model bytes, byte counts, SHA-256 values, and embedded identifiers;
- historical benchmark, release, issue, and pull-request links;
- Git history, authorship, copyright, and the fork relationship to
  [`antirez/ds4`](https://github.com/antirez/ds4).

A branded display label may say “Hebrus ExpertMajor v2”; serialized bytes and
identifiers do not change.

## Published artifact filenames

New canonical public artifact filenames use the Hebrus product name. A filename
migration is permitted only when the Hub publishes a server-side, byte-identical
copy and the same change:

1. proves the original byte count and SHA-256 remain unchanged;
2. pins the new path at an immutable repository revision;
3. updates the machine-readable release contract, downloader, tests, and
   current documentation together; and
4. retains the pre-migration path only as a historical object, never as a
   canonical download target or a second runtime format.

The Qwen Stable and Beta artifacts completed this migration under
[`ADR 0007`](../adr/0007-qwen-hebrus-artifact-filenames.md). Runtime admission
does not depend on either basename: it validates the GGUF contents and embedded
ExpertMajor contract. Historical benchmarks may therefore name the path that
was actually used, but current commands and release surfaces must use the
canonical Hebrus path.

## Environment variables and paths

`DS4_*` environment variables remain the accepted runtime namespace in the
bridge release. `HEBRUS_*` aliases are deferred until a central resolver can
enforce one policy consistently. When aliases are introduced, conflicting old
and new values must fail closed; a textual mass replacement is not permitted.

Existing checkouts, model paths, caches, benchmark records, and application
data are not copied or renamed. A repository redirect is transport
compatibility, not permission to reset an absolute local path.

## Repository and application bridge

The engine repository was renamed in place to `andreaborio/hebrus`. Bridge
consumers accept both the previous `andreaborio/ds4` identity and the canonical
identity over HTTPS or SSH. No history-less replacement was created, and the
old repository name must not be reused.

The companion application's public name is **Hebrus Studio**. Its persisted
DSBox identity remains a compatibility boundary for the bridge release:
`com.dsbox.desktop`, `$HOME/.dsbox`, `DSBOX_*`, browser storage keys, and the
legacy Electron user-data directory do not move. The bridge application resolves
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
5. Do not begin internal file/symbol migration until the Hebrus Studio bridge is
   published and the Qwen MLX-affine baseline is qualified.

The architectural rationale and the acceptance conditions for the rename are
recorded in
[`ADR 0005`](../adr/0005-hebrus-naming-and-compatibility-boundary.md); the
Qwen artifact-filename decision is
[`ADR 0007`](../adr/0007-qwen-hebrus-artifact-filenames.md).
