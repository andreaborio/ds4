# Governance

## Project Model

This repository uses a lightweight maintainer model. There is no foundation,
steering committee, technical committee, or voting body represented by this
document. Maintainer status is determined by the repository's actual write and
release permissions; this document does not create honorary roles.

Maintainers have final responsibility for:

- reviewing, merging, and rejecting changes;
- protecting correctness, compatibility, security, and release quality;
- accepting or rejecting architectural decisions;
- changing the runtime support contract;
- coordinating private security reports; and
- creating tags and publishing releases or artifacts.

That authority does not override project licenses, preserve a right to remove
authorship or attribution, or weaken the repository's documented validation
gates.

## Contributions And Review

Anyone may propose a change through a pull request. The contribution and test
requirements in [`CONTRIBUTING.md`](CONTRIBUTING.md) apply to every change.
Maintainers may request narrower scope, more evidence, changes to ownership
boundaries, or an independent review before merging.

Reviewers and contributors influence decisions through technical evidence and
discussion, but review activity alone does not grant merge or release
authority. Repository permissions remain the factual source of maintainer
authority.

## Decision Records And Runtime Authority

Routine, reversible implementation choices may be decided in pull-request
review. Changes to architecture, compatibility, supported models or backends,
serialized formats, or release policy require a written decision record and
the validation required by the repository guides.

Accepted records under [`docs/adr/`](docs/adr/) define architectural decisions.
[`docs/contracts/RUNTIME_SUPPORT.md`](docs/contracts/RUNTIME_SUPPORT.md) defines
the current production support boundary. Proposed ADRs describe proposals, not
active policy. In particular,
[`ADR 0005`](docs/adr/0005-hebrus-naming-and-compatibility-boundary.md) remains
Proposed until its acceptance conditions are met.

When normative documents disagree, the contradiction must be resolved in the
same change. A PR description, benchmark note, or temporary handoff cannot
silently override an accepted ADR, the runtime contract, `CONTRIBUTING.md`, or
`QA_BEFORE_RELEASES.md`.

## Upstream Relationship

The repository remains a fork of
[`antirez/ds4`](https://github.com/antirez/ds4). Changes that are general,
reproducible, and applicable to an upstream-supported path follow the
upstream-first classification and submission policy in
[`CONTRIBUTING.md`](CONTRIBUTING.md). Upstream review does not transfer
maintainer authority over this repository, and this governance document does
not imply endorsement by or formal affiliation with upstream.

## Releases And Security

A maintainer may publish a release only after the applicable gates in
[`QA_BEFORE_RELEASES.md`](QA_BEFORE_RELEASES.md) are satisfied and the release
scope is documented. Local qualification, an unmerged branch, or an entry under
`Unreleased` is not a published release.

Undisclosed vulnerabilities follow [`SECURITY.md`](SECURITY.md) and must not be
debated in public issues. GitHub private vulnerability reporting must be
enabled before launch; once it is available, maintainers coordinate
investigation and disclosure through that advisory flow. No response-time or
release-frequency guarantee is created by this document.

## Changing Governance

Governance changes use the same public pull-request and review process as other
repository changes. Only a maintainer with the corresponding repository
permissions can merge a governance change or alter access and release rights.
