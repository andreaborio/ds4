# Hebrus Rebrand Rollout Plan

- Status: active integration plan; repository renames not yet authorized
- Last updated: 2026-07-22
- Scope: Hebrus engine, Hebrus Studio, the Hebrus Studio website, and the
  qualified Hugging Face model repositories
- Release posture: public development preview; final notarized macOS release pending

This is the cross-repository source of truth for completing the Hebrus rebrand
without breaking runtime compatibility, installed application state, published
model artifacts, Git history, or upstream attribution. Detailed engine release
evidence remains in [`QA_BEFORE_RELEASES.md`](../QA_BEFORE_RELEASES.md); the
stable naming boundary remains in
[`BRAND_COMPATIBILITY.md`](contracts/BRAND_COMPATIBILITY.md).

> [!IMPORTANT]
> Hebrus began as a fork of
> [`antirez/ds4`](https://github.com/antirez/ds4) and retains substantial code,
> architecture, utilities, and history from that project. Salvatore
> Sanfilippo/antirez must remain prominently credited in the engine, Studio,
> website, release notes, and future repository descriptions. `llama.cpp`,
> GGML, MLX, and other material references remain credited with links and
> scope. A repository rename must preserve this lineage rather than presenting
> Hebrus as a clean-room implementation.

## Current snapshot

| Surface | Canonical work | State | Immediate constraint |
| --- | --- | --- | --- |
| Engine | `andreaborio/ds4`, branch `codex/hebrus-phase-0` | Rebased on current `origin/main`; 39 commits ahead and 0 behind; complete model-free premerge gate green | Publish the integration branch for review; model-backed release lanes remain required before release |
| Studio | `andreaborio/dsbox`, branch `codex/hebrus-engine-bridge` plus two newer `main` brand commits | Bridge is pushed and the `0.4.0-dev.1` prerelease exists; branch is 13 commits ahead and 2 behind `main` | Rebuild the bridge on current `main` and clear all CI failures |
| Website | `andreaborio/hebrus-site`, `main` | Published on GitHub Pages; Pages workflow green | Final logo, canonical post-rename links, and final signed download remain pending |
| Model Hub | Three consolidated `andreaborio/*-DS4-GGUF` repositories | Immutable qualified revisions are consumed by Studio and engine release contracts | Add dual repository-ID compatibility before moving any Hugging Face repository |

Release work must use the active engine rebrand branch, the active Studio bridge
branch, and the GitHub Pages site checkout identified in the table above. A
separate empty workspace checkout named `ds4` is unrelated and must not be used
for release work. The uncommitted modal-portal change in an older Studio
checkout must be reviewed and either ported deliberately to current Studio
`main` or discarded by its owner; it must not be accidentally included in the
rebrand integration.

## Non-breaking identity matrix

| May become canonical | Must remain compatible | Must not be renamed for branding |
| --- | --- | --- |
| `Hebrus`, `hebrus`, and the five `hebrus*` commands | Five `ds4*` command aliases through the complete 1.x line | `ds4.expert_major.v2`, wire values, GGUF metadata, disk-KV ABI, published filenames and hashes |
| `Hebrus Studio` visible product and Finder name | `com.dsbox.desktop`, `$HOME/.dsbox`, legacy Electron `userData`, browser keys, `DSBOX_*` configuration | Existing user models, conversations, settings, downloads, release artifacts, tags, and historical links |
| Future `andreaborio/hebrus` and `andreaborio/hebrus-studio` repository names | Old GitHub URLs through in-place rename redirects; consumers accept old and new engine identities | Git history, authorship, upstream remote, fork ancestry, licenses, acknowledgments, citations |
| `*-Hebrus-GGUF` as the canonical Hugging Face repository display identity | Historical `*-DS4-GGUF` repository IDs and already-installed repository directory names | GGUF filenames, bytes, immutable revisions, LFS/Xet objects, SHA-256 values, `ds4.expert_major.v2` metadata |
| Future `HEBRUS_*` aliases through one resolver | Existing `DS4_*` variables; conflicting old/new values fail closed | A textual replacement of environment variables, C symbols, paths, or serialized identifiers |

## Phase 0 — Reconcile the three active repositories

Goal: establish one reproducible head per product before changing another
public name.

1. Fetch all remotes and record immutable starting SHAs for engine, Studio, and
   website.
2. Rebase or reconstruct `codex/hebrus-phase-0` on current engine `main`. Keep
   the three incoming Qwen Flash/SSD commits intact and resolve conflicts as
   integration work, not as opportunities for extra kernel changes.
3. Reconstruct the Studio bridge on current Studio `main`, preserving the two
   new visual-brand commits and the thirteen bridge/release-hardening commits.
   Avoid a blind merge if it duplicates the brand-boundary inventory.
4. Review the older checkout's `createPortal(..., document.body)` modal change
   against current Studio. Port it in a separate bug-fix commit only if the
   current modal still has the stacking-context defect.
5. Mark the empty `Documents/ds4` checkout as non-canonical locally so builds,
   scripts, and release notes cannot accidentally reference it.

Exit gate: both integration branches are clean, based on current `main`, have
an explicit upstream tracking branch, and contain no unexplained duplicate or
dropped commits.

## Phase 1 — Restore green CI before further renaming

### Studio blockers already observed

1. Update the vulnerable `fast-uri` dependency resolution and rerun
   `npm audit --audit-level=high` from a clean lockfile install.
2. Fix the macOS package verifier false positive that classifies the font path
   `/dist/fonts/hebrus` as an embedded engine executable. The verifier must
   distinguish executable payloads from font/assets by type and location, not
   by basename alone.
3. Make the persisted-artifact resume test deterministic: the current failure
   compares `[null, null]` with `[null, undefined]`. Decide and document the
   boundary representation, then assert that contract consistently rather than
   weakening the compatibility check.
4. Rerun typecheck, brand audit, the complete test suite, production build,
   ad-hoc macOS packaging contract, and legacy upgrade/rollback exercise on the
   same candidate SHA.

### Engine gates

The current brand audit is green with 29,877 legacy occurrences classified in
370 groups and no unclassified increase. After rebasing, rerun at minimum:

- context and documentation-link audits;
- brand-boundary and Qwen release-contract checks;
- CPU/Linux and macOS arm64 builds;
- visible identity, capabilities, command-alias, install/uninstall, and server
  alias parity tests;
- qualified Apple Silicon model-backed correctness and performance lanes for
  every supported artifact affected by the incoming Metal changes.

### Website gates

Keep lint, static export tests, dependency audit, GitHub Pages deployment, HTTP
200 verification, canonical/OG metadata, and the exact GitHub Release download
URL green. The website must not host the DMG inside the Pages repository.

Exit gate: required checks are green on the exact engine and Studio candidate
SHAs; the site deploy for its exact content SHA is green.

## Phase 2 — Merge the bridge without renaming repositories

1. Open focused integration PRs for engine and Studio. Keep inference changes,
   visual polish, dependency fixes, and naming mechanics in reviewable commits.
2. Merge the engine capability contract and canonical `hebrus*` commands while
   retaining `ds4*` aliases and invocation-aware identity.
3. Merge Hebrus Studio's `hebrus-server`-first discovery, `ds4-server`
   fallback, dual `engine_id` acceptance, visible product identity, and
   persisted DSBox compatibility identity.
4. Publish a fresh development candidate from the merged Studio `main`; verify
   no-data-loss upgrade from an installed DSBox build and rollback to it.
5. Change ADR 0005 from Proposed to Accepted only after the merged bridge,
   compatibility evidence, and namespace decision are all recorded.

Exit gate: users can install Hebrus Studio and invoke Hebrus while old DS4
commands, models, state, paths, and integrations continue to work.

## Phase 3 — Freeze the public launch surface

1. Select the final logo master. The website currently uses a temporary
   text-based `H`; regenerate the favicon, application icons, and Open Graph
   card together once the master is accepted.
2. Freeze the typography, colors, spacing, screenshot count, English copy, and
   accessibility behavior across Studio documentation and the website.
3. Capture fresh screenshots from the exact release candidate, with no stale
   DSBox visible copy except where a migration explanation requires it.
4. Synchronize README hero copy, repository descriptions, topics, social card,
   release notes, install guide, support matrix, and changelog.
5. Verify every public surface carries the upstream DS4 origin and material
   technical references without implying endorsement.

Exit gate: visual assets and copy are versioned, consistent, accessible, and
traceable to the same candidate release.

## Phase 4 — Administrative repository renames

Perform in-place GitHub renames only; do not create history-less replacement
repositories and do not reuse the old names.

1. Rename `andreaborio/ds4` to `andreaborio/hebrus`.
2. Verify clone, fetch, submodule/reference, issue, PR, release, and Actions
   behavior through both the new canonical URL and GitHub's old-URL redirect.
3. Update active remotes, badges, repository topics/descriptions, security and
   issue-template links, package metadata, CI references, Studio engine-source
   admission, and website source links. Do not rewrite historical documents.
4. Rename `andreaborio/dsbox` to `andreaborio/hebrus-studio` only after Studio
   consumes both old and new engine remote identities.
5. Repeat redirect and automation verification for Studio, then update the
   website and release download URLs. Keep `hebrus-site` unchanged unless a
   separate brand decision justifies moving the Pages base path.
6. Run repository-wide broken-link scans and fresh clone/build/install tests
   from the new canonical URLs.

Exit gate: new URLs are canonical, old URLs redirect, Actions and Pages are
green, releases and stars remain attached to the renamed repositories, and no
published artifact or historical attribution link was rewritten destructively.

## Phase 5 — Rename the qualified Hugging Face repositories

Hugging Face documents an in-place move operation that redirects the old URL
and preserves download counts and likes. Use that operation rather than copying
or re-uploading weights. The initial public move set is:

| Current repository | Future canonical repository |
| --- | --- |
| `andreaborio/DeepSeek-V4-Flash-DS4-GGUF` | `andreaborio/DeepSeek-V4-Flash-Hebrus-GGUF` |
| `andreaborio/GLM-5.2-DS4-GGUF` | `andreaborio/GLM-5.2-Hebrus-GGUF` |
| `andreaborio/Qwen3.6-35B-A3B-DS4-GGUF` | `andreaborio/Qwen3.6-35B-A3B-Hebrus-GGUF` |

Retired, experimental, lab, v1, and superseded v2 repositories are historical
records. Do not bulk-rename or revive them; classify each separately and keep
them hidden from the Studio catalog.

1. Before any Hub move, update Studio catalog contracts so the new Hebrus ID is
   canonical and the current DS4 ID is included in `previousRepositories`.
   Installed paths using either repository name must continue to resolve to the
   same model.
2. Update engine download manifests and tooling to accept both repository IDs
   while pinning the same immutable revision, GGUF filename, byte size, and
   SHA-256. Repository identity is transport metadata, not a model-format
   change.
3. Add tests for API discovery, revision resolution, ranged download/resume,
   local installed-path recognition, hidden retired repositories, and fallback
   through the old redirected URL.
4. Record each source repository, destination repository, visibility, gated
   status, main revision, refs, file inventory, file sizes, LFS/Xet object
   identifiers, model-card metadata, downloads, and likes immediately before
   the move.
5. Move one repository at a time using the authenticated Hub move operation.
   Start with the least operationally critical model, validate it completely,
   and stop the sequence if any invariant changes.
6. After each move, verify the old web, API, `resolve`, and Git URLs redirect;
   the new URL resolves the same commit; range requests and Studio downloads
   work; the exact GGUF hash is unchanged; the model card and attribution are
   intact; and download counts/likes remain attached.
7. Update live model cards, Hebrus/Studio READMEs, catalog source labels,
   website links, release manifests, and support tables to the new canonical
   IDs. Preserve historical release notes and benchmark links unless they are
   intended as live download instructions.
8. Do not rename files inside the repositories. A friendly Hebrus display name
   may be added to the model card and manifest, but existing published GGUF
   filenames and their checksums remain immutable.

Exit gate: all three new Hub IDs are canonical, every old URL redirects, Studio
recognizes both old and new installed paths, exact revisions/files/hashes are
unchanged, and no model is duplicated or re-uploaded.

## Phase 6 — Release and launch

Two release lanes remain deliberately separate:

- **Community/development preview:** ad-hoc signed, clearly labelled, includes
  the Control-click installation path, checksums, provenance, SBOM, known
  limitations, and no notarization claim.
- **Public macOS release:** Developer ID signed, notarized, stapled, verified on
  a clean Mac under Gatekeeper, with the final DMG checksum and attestation.

For either lane, cut tags only from the green immutable candidate SHAs, publish
engine and Studio notes together, update the website download atomically, and
run a post-publish smoke test covering download, install, first launch, runtime
discovery, one resident model, one SSD-streamed model, upgrade, and rollback.

The launch announcement should lead with the differentiated product direction:
an open-source, Apple Metal-first inference engine with adaptive SSD streaming.
It must also state the DS4 fork origin prominently and link the acknowledgments,
compatibility guide, source, supported-artifact matrix, and reproducible
benchmark evidence.

## Stop conditions

Do not rename repositories, remove aliases, move application data, or describe
a build as a final release when any of the following is true:

- either engine or Studio CI is red;
- a branch is not reconciled with current `main`;
- a model-backed Metal/SSD gate required by the diff is missing;
- upstream attribution, license inventory, checksums, provenance, or rollback
  evidence is incomplete;
- the website points to an unavailable or differently qualified artifact;
- the macOS build is described as notarized without Developer ID, notarization,
  stapling, and clean-machine Gatekeeper evidence.

## Completion definition

The rebrand is complete when the canonical repositories, commands, application,
website, release assets, documentation, and community surfaces say Hebrus or
Hebrus Studio; every intentionally retained DS4/DSBox identifier is classified
and tested as compatibility, serialized data, or history; old URLs and commands
still work within their promised window; CI and model-backed evidence are green;
and a new user can understand the product, its support boundary, its provenance,
and its installation path from the public landing page alone.
