# Hebrus rebrand record

- Snapshot: 2026-07-22
- Status: historical record; not a live launch plan
- Scope: Hebrus engine, Hebrus Studio, the project website, and the qualified
  Hugging Face model repositories

This record preserves the decisions and outcomes of the DS4-to-Hebrus rename.
It must not be used to infer current repository settings, CI state, counters,
downloads, redirects, release readiness, or application availability. Recheck
those external facts when performing a release or administrative operation.

Current authority is split deliberately:

- [`BRAND_COMPATIBILITY.md`](contracts/BRAND_COMPATIBILITY.md) defines canonical
  names, compatibility aliases, and identifiers that branding must not change;
- [`RUNTIME_SUPPORT.md`](contracts/RUNTIME_SUPPORT.md) defines qualified runtime
  and model support;
- [`qwen-release.json`](contracts/qwen-release.json) records exact current Qwen
  artifact identities;
- [`QA_BEFORE_RELEASES.md`](../QA_BEFORE_RELEASES.md) defines release evidence.

## Recorded outcome

| Surface | Recorded identity | Outcome at the snapshot date |
| --- | --- | --- |
| Engine | `andreaborio/hebrus`, merge `7686f70` | Repository renamed in place; canonical `hebrus*` commands and `ds4*` compatibility aliases implemented |
| Studio | `andreaborio/hebrus-studio`, merge `dd13cb5` | Repository renamed in place; visible Hebrus Studio identity and persisted DSBox compatibility bridge implemented |
| Website | `andreaborio/hebrus-site`, revision `086ade3` | Canonical source and release links recorded in the site source |
| Model Hub | Three `andreaborio/*-Hebrus-GGUF` repositories | In-place moves recorded and checked against the pre-move inventory; see the [historical Hub inventory](HF_MODEL_RENAME_INVENTORY.md) |

These revisions identify the rename work only. They are not current release
pins and do not replace the exact runtime and artifact identities in the live
contracts above.

## Durable compatibility boundary

| Canonical public identity | Compatibility implemented in this snapshot | Preserved data or history |
| --- | --- | --- |
| `Hebrus`, `hebrus`, and five `hebrus*` commands | Five `ds4*` command aliases | `ds4.expert_major.v2`, wire values, GGUF metadata, disk-KV ABI, published bytes and hashes |
| `Hebrus Studio` visible name | Persisted DSBox application identity, data paths, browser keys, and `DSBOX_*` configuration | Existing user models, conversations, settings, downloads, release artifacts, and tags |
| `andreaborio/hebrus` and `andreaborio/hebrus-studio` | Old repository identities accepted by the recorded bridge | Git history, authorship, upstream fork ancestry, licenses, acknowledgments, and historical links |
| `*-Hebrus-GGUF` repository display identities | Historical repository IDs and already-installed repository directory names | Model bytes, immutable revisions, LFS/Xet objects, checksums, and embedded identifiers |
| A possible future `HEBRUS_*` namespace | Existing `DS4_*` variables | No textual rewrite of environment variables, C symbols, paths, or serialized identifiers |

The aliases present in this snapshot are tested compatibility behavior. The
proposal to retain them through the complete 1.x line remains unaccepted in
[`ADR 0005`](adr/0005-hebrus-naming-and-compatibility-boundary.md); this record
does not turn that proposed horizon into a release promise.

## Completed operations recorded on 2026-07-22

- The engine and Studio repositories were renamed in place rather than copied
  into history-less replacements.
- Three Hugging Face repositories were moved in place. The recorded comparison
  covered revisions, sibling inventories, object identities, ranged reads, and
  redirects; it did not change model bytes or embedded ExpertMajor identifiers.
- Current product copy moved to Hebrus while upstream DS4 origin, authorship,
  licenses, and substantial implementation credit remained explicit.
- The compatibility bridge separated public display names from command,
  application-state, environment, and serialized-data identities.

Redirect and download observations above are dated evidence, not guarantees
about present external service behavior.

## Remaining publication gates

Before describing a new build as a public release, the maintainer must still:

- satisfy the acceptance conditions in ADR 0005 before promoting its proposed
  release-line compatibility horizon;
- obtain green CI and every model-backed lane required for the exact candidate;
- provide a verified private security-reporting route;
- verify current repository links, redirects, downloads, artifact hashes, and
  release assets rather than relying on this snapshot; and
- claim Developer ID signing, notarization, stapling, or Gatekeeper acceptance
  only when each property has been checked on the published artifact.

Do not remove aliases, move application data, rename serialized identifiers, or
rewrite historical attribution as part of routine brand work. Hebrus remains a
fork of [`antirez/ds4`](https://github.com/antirez/ds4); this lineage does not
imply upstream endorsement or partnership.
