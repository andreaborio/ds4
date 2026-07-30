# ADR 0007: Qwen Canonical Artifacts Use Hebrus Filenames

- Status: Accepted
- Date: 2026-07-30

## Context

The Qwen repository and engine are publicly named Hebrus, but the Stable and
Beta GGUF basenames still carried the pre-rebrand product token. That mismatch
leaked into current download commands and benchmark metadata, where the brand
boundary audit correctly rejected a new legacy token group.

The basename is not part of Qwen runtime admission. Hebrus validates the GGUF
tensor inventory, ExpertMajor marker, storage profile, geometry, byte count,
and digest. Renaming or rewriting the GGUF itself would therefore add risk
without adding compatibility value.

## Decision

The canonical Qwen Stable and Beta artifact basenames use `Hebrus`:

- `Qwen3.6-35B-A3B-Hebrus-ExpertMajor-v2-MLX-Affine4-G64.gguf`;
- `Qwen3.6-35B-A3B-Hebrus-ExpertMajor-v2-Q2_K_XL.gguf`.

Both were created in the existing public Hub repository as server-side copies
of their accepted source objects. The copies preserve exact bytes, byte counts,
SHA-256 values, embedded tensor identifiers, and runtime behavior. Both paths
are pinned at immutable repository revision
`e002665becd2db618897effb213030fdf92e7e98`.

The machine-readable Qwen release contract is the sole source for these
basenames and revisions. The downloader, current support/QA documentation, and
contract tests derive or verify the same identities. The pre-migration object
paths remain available only for reproducibility of historical records; they are
not canonical targets and do not create a second runtime format or fallback.

The former Q4_K_S artifact remains a historical, negative-only identity. This
decision does not publish it, admit it, or alter the fail-closed rejection
required by ADR 0003 and the Qwen release contract.

## Consequences

- New Qwen downloads and current documentation present Hebrus consistently.
- Existing local files continue to load because admission is content-based,
  not basename-based.
- No model bytes, serialized identifiers, kernels, scheduling, or inference
  output change.
- Future artifact filename migrations require a byte-identical server-side
  copy, immutable revision pin, digest/size proof, and an atomic update of the
  release contract, downloader, tests, and current documentation.
- Historical evidence may retain the name used by the original run, but it
  cannot silently become a current download recommendation.
