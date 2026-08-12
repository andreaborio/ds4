# Changelog

This file records user-visible changes. Historical benchmark documents and
release notes remain evidence for their exact commits; they are not rewritten
into current support claims here.

## Unreleased

The entries below describe the local Hebrus compatibility bridge after the
current published history. They have no release version or release date and do
not represent a published release.

### Added

- Added deterministic, model-free `--capabilities=json` output to the CLI,
  server, agent, benchmark, and evaluation executables. The versioned document
  reports build identity, backend, executable role, supported model-family IDs,
  and the immutable ExpertMajor v2 storage contract.
- Added canonical `hebrus`, `hebrus-server`, `hebrus-agent`, `hebrus-bench`, and
  `hebrus-eval` commands. Existing `ds4*` command names remain symlink aliases
  to the same binaries for command-surface parity.
- Added model-free tests for capability-schema determinism, cross-executable
  consistency, command aliases, build-profile isolation, and canonical/legacy
  output parity.
- Added Proposed ADR 0005 to document the Hebrus naming layer and the proposed
  long-term compatibility boundary without changing durable model, cache, or
  historical identifiers.
- Added the maintainer-supplied Hebrus logo as a hash-frozen, unchanged RGBA
  master shared with Hebrus Studio. Repository tests reject pixel or encoding
  drift; web presentation effects remain CSS-only.
- Added fail-closed source-release tooling that binds a deterministic archive,
  JSON provenance manifest, and SHA-256 set to one immutable clean commit, then
  rebuilds and smoke-installs the archive outside Git before publication.

### Changed

- Help headings, usage lines, examples, and retired-option diagnostics now use
  the executable name that was invoked. Canonical `hebrus*` commands present
  the Hebrus name, while the `ds4*` compatibility aliases preserve their
  legacy command identity and the same options, defaults, streams, and exit
  codes.
- Canonical commands report `engine_id: "hebrus"` and `hebrus build`; legacy
  aliases retain `engine_id: "ds4"` and `ds4 build`. Both identities use the
  same schema-1 capability fields and immutable ExpertMajor wire contract.
- Current contributor, release, and engine-reference documentation now presents
  Hebrus commands first and links an explicit brand compatibility contract.
  Canonical CLI/agent/evaluation prompts and new benchmark evidence use the
  invoked Hebrus identity; legacy aliases preserve their existing labels.

### Compatibility

- The engine repository was renamed in place and is now canonical at
  <https://github.com/andreaborio/hebrus>. Historical documents may retain the
  repository identity that was current when their evidence was recorded.
- Existing `DS4_*` environment variables and `ds4`-owned serialized identifiers
  remain unchanged. This bridge does not introduce `HEBRUS_*` environment
  aliases or rename source-level C symbols.
- ExpertMajor tensor names, wire values, model artifact identities, disk-KV
  formats, Git history, and historical links are unchanged.
