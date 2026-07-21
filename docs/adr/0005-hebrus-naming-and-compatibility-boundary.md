# ADR 0005: Hebrus Naming And Compatibility Boundary

- Status: Proposed
- Date: 2026-07-21

## Context

This repository began as a fork of [`antirez/ds4`](https://github.com/antirez/ds4)
and still preserves that ancestry, attribution, and a path for generally useful
changes to return upstream. The fork now has a materially different product
focus: Apple Metal inference, mmap-backed execution, embedded ExpertMajor
stores, and SSD streaming across several model families. Continuing to present
the fork itself as DS4 obscures that boundary and makes application/runtime
ownership harder to explain.

A textual rename is unsafe. Existing installations and automation depend on
`ds4` commands, `DS4_*` environment variables, repository URLs, C names, model
filenames, and application paths. Published GGUFs and disk-KV files also contain
identifiers whose bytes are part of a durable format. The current DSBox release
additionally probes exact source paths, C symbols, and diagnostic strings when
admitting an ExpertMajor runtime.

## Decision

After namespace and legal screening is complete, the public engine name will be
**Hebrus**. The existing GitHub fork will be renamed rather than copied, and the
README will continue to identify `antirez/ds4` prominently as the origin of the
fork and of substantial implementation work. Acknowledgments will also name
ggml, llama.cpp, MLX, and other material sources with precise links and scope.

The migration separates public brand from compatibility identity:

- canonical executable names become `hebrus`, `hebrus-server`,
  `hebrus-agent`, `hebrus-bench`, and `hebrus-eval`;
- all current `ds4` executable names remain aliases to the same object graph
  through at least the complete 1.x release line;
- new `HEBRUS_*` environment aliases may be introduced only through one
  resolver; existing `DS4_*` variables remain accepted, and conflicting values
  fail closed;
- the companion application must consume a versioned, model-free capability
  document before internal source names move. Legacy source/binary probe markers
  remain a temporary compatibility shim, not the new admission API;
- source, header, symbol, and compile-time macro renames occur later in
  mechanical subsystem-sized changes with no inference or performance change.

The following identifiers do not change as part of the brand migration:

- the `ds4.expert_major.v2` tensor name and the retired
  `ds4.expert_major.v1` rejection marker;
- ExpertMajor v2 magic, version, manifest layout, offsets, digest semantics,
  storage wire values (`GGML = 0`, `MLX_AFFINE4 = 1`), and qualified Qwen group
  size (`64`);
- published model filenames, byte counts, revisions, and SHA-256 values;
- disk-KV magic, version, and payload ABI;
- Git history, tags, authorship, copyright, fork ancestry, and historical issue,
  PR, benchmark, and release links.

Old model artifacts are not revived by this compatibility policy. In
particular, the retired Qwen GGML/Q4 ExpertMajor store remains rejected under
both command brands. Model-format lifecycle remains governed by ADR 0003.

## Consequences

- A bridge release must precede internal source renaming and the companion
  application's visible rebrand.
- Every remaining DS4 identifier must be classified as serialized, historical,
  compatibility-owned, or erroneous; unclassified additions fail a repository
  check.
- Canonical and legacy commands must have parity for options, exit status,
  protocols, deterministic output, resolved runtime plan, and performance.
- Existing installations, checkouts, models, caches, and application data do
  not need to be copied or rewritten for branding.
- The GitHub repository rename happens only after the compatibility application
  recognizes both old and new remote identities.
- This ADR becomes Accepted only after the Hebrus identity is reserved, the
  machine-readable capability contract is merged, the DSBox bridge is proven,
  and the pre-rebrand release baseline is green.
