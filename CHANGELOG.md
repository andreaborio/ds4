# Changelog

This file records user-visible changes. Historical benchmark documents and
release notes remain evidence for their exact commits; they are not rewritten
into current support claims here.

## Unreleased

No user-visible changes recorded yet.

## 0.3.0 - 2026-08-13

Hebrus 0.3.0 is the first Hebrus-named source release. It ships source and
provenance files only; it does not include prebuilt engine binaries or a
Hebrus Studio application bundle.

### Added

- Added canonical `hebrus`, `hebrus-server`, `hebrus-agent`, `hebrus-bench`,
  and `hebrus-eval` commands. The corresponding `ds4*` names remain
  compatibility symlinks to the same binaries in this release.
- Added deterministic, model-free schema-1 `--capabilities=json` output to all
  five executable roles and both command identities. The document reports
  build identity, backend, executable role, supported model-family IDs, and
  the immutable ExpertMajor v2 storage contract.
- Added staged `make install` and `make uninstall` support with `PREFIX`,
  `BINDIR`, and `DESTDIR`, including versioned Metal resources and an
  install-layout test.
- Added the exact Qwen3.6 Q2_K_XL profile as an opt-in `published-beta`
  artifact. It has a 64 GiB floor and a 32,768-token qualified boundary; it is
  neither recommended nor full-window qualified.
- Added model-free tests for capability-schema determinism, cross-executable
  consistency, command aliases, build-profile isolation, installed resources,
  and canonical/compatibility output parity.
- Added fail-closed source-release tooling that binds a deterministic archive,
  JSON provenance manifest, and SHA-256 set to one immutable clean commit, then
  rebuilds and smoke-installs the archive outside Git before publication.

### Changed

- Renamed the existing GitHub fork in place to `andreaborio/hebrus` while
  preserving its history and upstream attribution.
- Replaced the retired Qwen Q4_K_S release path with the exact Stable and
  recommended MLX Affine4/group-64 artifact. The retired Q4_K_S artifact is
  rejection-only, not a fallback.
- Qualified the Stable Qwen Affine4 guarded-SSD 16 GiB capacity and safety lane
  through a 131,072-token prompt plus 128 greedy decode tokens, without making
  a speed claim. The separate 24 GiB policy remains a publication candidate,
  not a qualified hardware claim.
- Changed short DeepSeek conversational prefills to retain the warm expert
  cache by default through 4,096 tokens.
- Updated current commands, documentation, Qwen artifact names, Qwen download
  metadata and immutable repository paths, and release-contract tests to use
  the Hebrus identity while retaining compatibility-owned serialized
  identifiers.
- Added Proposed ADR 0005 to document the naming layer and a possible future
  compatibility horizon. This release does not accept or promise that horizon.

### Fixed

- Corrected the host-side hyper-connection combination strides and added a
  three-prompt DeepSeek generated-text regression check.
- Made installed Metal commands discover their complete shader source set
  relative to the executable instead of depending on the current working
  directory.
- Hardened Qwen guarded-SSD allocation so live memory and working-set limits
  are checked before cache growth; denied growth reuses the bounded cache or
  fails closed.
- Restored the qualified Qwen Affine4 SSD execution path after the dual-profile
  integration while keeping Qwen-specific cache behavior isolated from
  DeepSeek and GLM.
- Made an unsafe explicit DeepSeek expert-cache budget fail closed instead of
  allowing unconfigured routed experts to read as zero.

### Compatibility

- This release preserves the five `ds4*` command aliases, the `DS4_*`
  namespace for documented runtime settings, `ds4.expert_major.v2`, disk-KV
  magic, version and payload ABI, model bytes, and historical identifiers.
- It does not promise an alias horizon for later release lines: ADR 0005
  remains Proposed.
- Old model containers and the retired Qwen Q4_K_S payload remain unsupported;
  see the runtime support and brand compatibility contracts.
