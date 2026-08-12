# Hebrus Bridge Release: Launch Candidate Notes

- Status: local launch candidate; not published
- Version: to be assigned at release cut
- Release date: to be assigned at release cut
- Candidate commit: to be recorded after all release gates pass

These notes separate behavior already implemented in source from external
release operations that still require an explicit decision, credentials, or a
published artifact. They are not a release announcement and do not promise a
repository URL, signed package, or availability date.

> [!IMPORTANT]
> Hebrus began as a fork of
> [`antirez/ds4`](https://github.com/antirez/ds4) and retains substantial core
> implementation, architecture, utilities, and Git history from that project.
> [llama.cpp](https://github.com/ggml-org/llama.cpp),
> [GGML](https://github.com/ggml-org/ggml), and
> [MLX](https://github.com/ml-explore/mlx) remain material implementation and
> validation references. This lineage does not imply endorsement. See
> [`ACKNOWLEDGMENTS.md`](../../ACKNOWLEDGMENTS.md) and
> [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md).

## Candidate scope implemented in source

### Canonical Hebrus command surface

The bridge adds `hebrus`, `hebrus-server`, `hebrus-agent`, `hebrus-bench`, and
`hebrus-eval` as canonical commands. The corresponding `ds4*` names remain
relative symlinks to the same five binaries and preserve their invocation
identity. Options, protocols, model admission, generated tokens, and runtime
planning are shared; help text, prompts, build labels, and structured
`engine_id` reflect the command actually invoked.

`make install` supports `PREFIX`, `BINDIR`, and `DESTDIR`. A Metal install also
places its named shader sources in a versioned executable-relative resource
directory. Uninstall removes the ten public command paths and only those named
resources, leaving unrelated files intact.

### Versioned integration contract

All ten commands expose deterministic `--capabilities=json` output without
loading a model. Schema 1 reports:

- the invocation-aware `engine_id`;
- exact build Git SHA, backend, and executable role;
- the supported model-family identifiers;
- ExpertMajor version 2, the stable `ds4.expert_major.v2` tensor, and exact
  GGML/MLX-affine storage wire values.

Consumers can therefore validate a server without scanning source filenames,
C symbols, diagnostics, or binary strings. The bridge contract accepts
`engine_id` values `hebrus` and `ds4`; malformed documents, unknown schemas,
wrong roles or backends, and contradictory ExpertMajor fields fail closed.

### Metal and SSD runtime focus

The runtime remains deliberately narrow: qualified Apple Metal execution,
mmap-backed GGUF containers, checksummed embedded ExpertMajor v2 routed
weights, hardware-aware AUTO admission, and either resident Metal execution or
bounded SSD expert streaming. It is not a general GGUF runner.

The current support authority is
[`RUNTIME_SUPPORT.md`](../contracts/RUNTIME_SUPPORT.md). Exact artifact names,
revisions, byte counts, hashes, and measured lanes remain in the model and
benchmark records; these launch-candidate notes do not replace them.

### Companion bridge identity

**Hebrus Studio** is the selected public name for the companion application.
The bridge resolves `hebrus-server` before `ds4-server` and validates the
structured capability contract while retaining a narrow legacy fallback for a
capability-less `ds4-server`.

The visible name is separate from persisted compatibility identity. The bridge
release keeps the existing `com.dsbox.desktop` bundle identifier, `$HOME/.dsbox`
data root, legacy `$HOME/Library/Application Support/DSBox` Electron `userData`
directory, and `DSBOX_*` configuration namespace so installed models,
configuration, downloads, and local conversations are not moved. The desktop
application is delivered separately from the engine and is not embedded in
this repository's command package.

`DSBox.app` and `Hebrus Studio.app` have different Finder names, so installing
the latter does not replace the former. They still share the app
identifier, state, and control port and must never run simultaneously. The
verified upgrade quits DSBox first, installs and validates Hebrus Studio, and
then removes the old bundle or retains it only in an offline rollback archive.

### Canonical visual identity

Both repositories use the same maintainer-supplied 1254 x 1254 RGBA Hebrus
master without pixel, crop, color, or encoding edits. Its SHA-256 is
`4be8949c73bd52e7abef58396dcd57f636165a8bb6cd6d536a600bcbf880594c`.
Hebrus Studio derives its macOS icon from that exact file during packaging and
adds a drop shadow only through CSS on web surfaces. The engine and application
release gates reject a modified master or a mismatched packaged copy.

### Open-source project surface

The candidate includes contribution, security, governance, conduct,
acknowledgment, third-party notice, changelog, citation, issue-template, and
release-checklist material. Linux and macOS hosted workflows cover the
model-free and compile boundaries appropriate to those runners. Hosted macOS
compilation is not evidence for qualified Metal kernels, model artifacts,
throughput, memory pressure, or SSD behavior.

## Preserved compatibility

This bridge intentionally does not rename:

- `ds4.expert_major.v2`, the retired v1 rejection marker, or any ExpertMajor
  on-disk bytes;
- published GGUF filenames, immutable revisions, sizes, or checksums;
- disk-KV magic, version, or payload ABI;
- existing `DS4_*` engine variables;
- Hebrus Studio's `DSBOX_*` variables, data root, or bundle identifier;
- source-level C identifiers, Git history, authorship, tags, or historical
  benchmark and release links.

Compatibility aliases remain supported through at least the complete 1.x
release line. Any future removal requires usage evidence, a new accepted
decision record, release notes, and its own migration tests.

## Evidence required on the final candidate commit

Before replacing the placeholders at the top of this document, the release
owner must attach evidence for the exact commit being published:

- clean context, documentation-link, brand-boundary, model-free, build
  isolation, command-alias, staged-install, and download-manifest gates;
- green SHA-pinned Linux and macOS hosted jobs for that commit;
- qualified Apple Silicon model-backed correctness and performance lanes for
  every supported artifact, including exact output parity, resolved plan,
  memory pressure, swap deltas, and SSD telemetry;
- exact source archive, engine package, model manifest, and companion-package
  checksums;
- a clean-install and no-data-loss upgrade/rollback exercise for Hebrus Studio
  that proves the legacy Electron `userData` and `$HOME/.dsbox` state are
  reused, only one app owns the control port, and Finder's two bundle names do
  not permit concurrent launch;
- an updated support matrix and benchmark evidence index that cite the same
  immutable commit and artifacts.

Follow [`QA_BEFORE_RELEASES.md`](../../QA_BEFORE_RELEASES.md); a passing narrow
smoke test cannot stand in for this matrix.

## External and administrative gates still pending

The following are not implied by source readiness and must remain described as
pending until independently verified:

- namespace and legal screening, reservation, and acceptance of the naming ADR;
- re-verification of the completed repository rename redirects and publication
  of the compatibility app;
- final version/tag selection and remote push;
- GitHub private vulnerability reporting or another verified private intake;
- Apple Developer ID signing, notarization, stapling, and clean-machine
  Gatekeeper verification for Hebrus Studio;
- public download URLs, release artifacts, checksums, screenshots, and launch
  announcement;
- model-backed release evidence on the exact final commit and qualified
  hardware.

An ad-hoc-signed local application bundle proves package structure and seal
integrity only. It must not be described as Developer ID signed or notarized.

## Upgrade and rollback

Users and package maintainers should follow
[`Migrating From DS4 To Hebrus`](../guides/MIGRATING_TO_HEBRUS.md). The guide
keeps existing commands and persisted identities operational, validates both
capability identities before consumers switch, and provides a rollback that
does not rewrite models or user data.

## Release-owner cut procedure

1. Choose and record the immutable candidate commit.
2. Run the complete checklist and archive exact raw evidence.
3. Verify the release version appears consistently in packages and notes.
4. Replace only the version, date, and commit placeholders above.
5. Confirm every pending item is either completed with evidence or explicitly
   disclosed as unavailable; do not silently omit a gate.
6. Publish only after the engine, Hebrus Studio bridge, migration guide,
   checksums, provenance, and rollback instructions agree.
