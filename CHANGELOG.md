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
- Added Proposed ADR 0005 to document the intended Hebrus naming and
  compatibility boundary without changing durable model, cache, or historical
  identifiers.

### Changed

- Help headings, usage lines, examples, and retired-option diagnostics now use
  the executable name that was invoked. Canonical `hebrus*` commands present
  the Hebrus name, while the `ds4*` compatibility aliases preserve their
  legacy command identity and the same options, defaults, streams, and exit
  codes.

### Compatibility

- The repository remains at <https://github.com/andreaborio/ds4> until an
  administrative rename is actually performed.
- Existing `DS4_*` environment variables and `ds4`-owned serialized identifiers
  remain unchanged. This bridge does not introduce `HEBRUS_*` environment
  aliases or rename source-level C symbols.
- ExpertMajor tensor names, wire values, model artifact identities, disk-KV
  formats, Git history, and historical links are unchanged.
