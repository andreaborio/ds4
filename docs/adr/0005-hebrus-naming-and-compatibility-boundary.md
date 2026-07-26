# ADR 0005: Hebrus Naming And Compatibility Boundary

- Status: Accepted
- Date: 2026-07-21
- Accepted: 2026-07-24

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
identifiers whose bytes are part of a durable format. The pre-rename DSBox release
additionally probes exact source paths, C symbols, and diagnostic strings when
admitting an ExpertMajor runtime.

## Decision

The public engine name is **Hebrus**. The existing GitHub fork is renamed in
place rather than copied, and the README continues to identify `antirez/ds4`
prominently as the origin of the fork and of substantial implementation work.
Acknowledgments also name ggml, llama.cpp, MLX, and other material sources with
precise links and scope.

This record decides the software naming and compatibility boundary. It is not a
trademark opinion or a record of legal clearance; the maintainer's use of the
name is an external project decision.

The migration separates public brand from compatibility identity:

- canonical executable names are `hebrus`, `hebrus-server`,
  `hebrus-agent`, `hebrus-bench`, and `hebrus-eval`;
- all current `ds4` executable names are aliases to the same object graph
  through at least the complete 1.x release line;
- new `HEBRUS_*` environment aliases may be introduced only through one
  resolver; existing `DS4_*` variables remain accepted, and conflicting values
  fail closed;
- the companion application consumes the versioned, model-free capability
  document before internal source names move. Legacy source/binary probe markers
  are a temporary compatibility shim, not the admission API;
- source, header, symbol, and compile-time macro renames may occur later in
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

- A published Hebrus Studio bridge release with green persisted-identity and
  rollback tests must precede internal source renaming.
- Every remaining DS4 identifier must be classified as serialized, historical,
  compatibility-owned, or erroneous; unclassified additions fail a repository
  check.
- Canonical and legacy commands must have parity for options, exit status,
  protocols, deterministic output, resolved runtime plan, and performance.
- Existing installations, checkouts, models, caches, and application data do
  not need to be copied or rewritten for branding.
- The GitHub repository was renamed in place only after the compatibility
  application recognized both old and new remote identities.

## Acceptance evidence

Technical acceptance is anchored by the machine-readable capability contract
(`f0c8502648adb63e2d18062360cfabba393cd508`), the engine bridge merge
(`7686f70ac0d4e0de0319a9fe30555924d6ac2b82`), and the verified in-place
repository rename record
(`8796cff5fec8ac8ad63c36999bd1a640e4c8b793`). The companion bridge was merged
in [`andreaborio/hebrus-studio#11`](https://github.com/andreaborio/hebrus-studio/pull/11)
at `dd13cb5`; its dual engine identity, persisted DSBox state, and rollback
boundary were verified together.

The pre-rebrand v0.2.0 evidence commit
`57acfd408a3154851a0c59be432904300abb3b6c` is the historical runtime baseline
for this decision, not the current support contract.

Acceptance of this architecture does not publish an engine or application
release. Engine release qualification, artifact checksums, and model-backed
evidence remain governed by
[`QA_BEFORE_RELEASES.md`](../../QA_BEFORE_RELEASES.md). Companion signing,
notarization, stapling, and clean-machine verification belong to the Hebrus
Studio release process and are outside this repository's authority.
