# Hebrus <version>

<!--
Copy this file to docs/releases/v<version>.md. Replace every angle-bracket
placeholder, delete these comments, and keep claims limited to evidence from
the exact release commit. The canonical release gate is QA_BEFORE_RELEASES.md;
do not copy that checklist into these notes.
-->

| Release field | Value |
| --- | --- |
| Release date | <YYYY-MM-DD> |
| Source commit | `<full-40-character-commit>` |
| Source archive | `hebrus-<version>.tar.gz` |
| Source SHA-256 | `<64-lowercase-hex-digest>` |

## Summary

<Describe the user-visible outcome in two or three factual sentences. State
the supported production boundary instead of implying general platform or model
support.>

## Changes

<!-- Derive this section from the matching entries in CHANGELOG.md. -->

### Added

- <User-visible addition, or remove this subsection.>

### Changed

- <User-visible change, or remove this subsection.>

### Fixed

- <User-visible fix, or remove this subsection.>

## Supported configurations

The exact model artifacts, hardware floors, qualified modes, and context
frontiers for this release are recorded in the
[runtime support contract](../contracts/RUNTIME_SUPPORT.md) at the source
commit above.

<Call out only support changes introduced by this release. Do not reproduce the
whole matrix or infer support from compilation.>

## Source integrity and build provenance

Download the archive together with `hebrus-<version>-source.json` and
`SHA256SUMS`, then verify the two checksummed files:

```sh
# Linux
sha256sum -c SHA256SUMS

# macOS
shasum -a 256 -c SHA256SUMS
```

The JSON source manifest binds the version, full Git commit, commit timestamp,
archive size, member count, and SHA-256. Builds made from the archive have no
`.git` directory, so package recipes must pass the manifest's 12-character
commit explicitly:

```sh
make NATIVE_CPU_FLAG= BUILD_GIT_SHA=<12-character-commit>
```

`NATIVE_CPU_FLAG=` is appropriate for redistributable packages. A local build
may retain the Makefile's native-machine default.

## Compatibility and migration

<Describe command, configuration, API, model-format, or persisted-data impact.
If there is none, state that directly and link the applicable contract rather
than promising unspecified compatibility.>

See the [migration guide](../guides/MIGRATING_TO_HEBRUS.md) for the established
command-name transition and rollback boundary.

## Validation

The release was evaluated with the canonical
[release checklist](../../QA_BEFORE_RELEASES.md) on the exact source commit
above.

| Evidence | Result or link |
| --- | --- |
| Hosted Linux and macOS jobs | <permanent run links> |
| Source archive reproducibility and install smoke | <result or evidence link> |
| Qualified model-backed lanes | <dated evidence links> |
| Manual server, cache, and agent lanes | <dated evidence links> |

<Record every skipped or non-applicable lane with its reason. Do not replace a
missing physical-hardware result with hosted compilation or a simulated lane.>

## Known limitations

- <Current limitation supported by the release commit, or “No additional
  limitations beyond the runtime support contract.”>

## Security and provenance

Security fixes should be described only after coordinated disclosure. Report
new vulnerabilities through the [security policy](../../SECURITY.md).

Hebrus retains substantial implementation and history from its upstream fork.
See [Acknowledgments](../../ACKNOWLEDGMENTS.md) and
[Third-party notices](../../THIRD_PARTY_NOTICES.md); attribution does not imply
endorsement.

## Contributors

<List contributors supported by the release commit or link the exact compare
view. Do not infer authorship from generated release text.>
