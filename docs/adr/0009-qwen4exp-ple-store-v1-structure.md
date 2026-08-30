# ADR 0009: Qwen4Exp PLE v1 Freezes Structure Before Codec Promotion

- Status: Accepted
- Date: 2026-08-30
- Support state: Structural converter contract only (`pinned-not-supported`).
  No PLE codec, artifact, runtime path, downloader or release is admitted.

## Context

The pinned Qwen4Exp text profile owns 320,001,446 logical n-gram rows split
across sixteen cumulative prime-sized head segments. The checkpoint pads that
extent to 320,001,536 rows at an alignment of 128. A per-row offset table would
contain more than 320 million entries and is not an acceptable runtime index.
The PLE extent must also remain independently verifiable and physically
isolated from dense tensors and the ExpertMajor store.

The production row codec is intentionally unresolved. Quality and M5 evidence
must choose it later, so the structural format cannot imply that a synthetic
test codec or the current ExpertMajor affine candidate is release-qualified.

## Decision

1. The embedded extent is `ds4.ple_rows.v1`, wire version 1, family 4,
   profile `qwen4exp-base-v1`, and hash identity
   `SplitMix64-Qwen4Exp-v1`. All integers are little-endian.
2. The manifest header is 512 bytes. It closes the sixteen primes and offsets,
   logical/padded row geometry, row alignment, exact caller-supplied codec
   descriptor, fixed-page geometry, physical extents, and whole-payload and
   manifest SHA-256 digests. Reserved bytes are zero.
3. Physical pages have a 64-byte header and fixed stride. Each header repeats
   page index, first row, valid row count, width, encoded row bytes, rows per
   page, codec version/group size, and family. A compact table stores one
   SHA-256 per page; there is no per-row index.
4. Row location is checked affine arithmetic:

   ```text
   page = row / rows_per_page
   slot = row % rows_per_page
   offset = payload_offset + page * page_stride
   ```

   The complete page must lie inside the caller-owned embedded extent and the
   underlying regular file. All add, multiply and alignment operations reject
   overflow before I/O.
5. The page digest covers the complete physical stride. The payload digest
   covers all complete pages in order. The manifest digest covers the header
   with its own digest field zeroed, the page-digest table, and alignment
   padding before the payload.
6. Opening validates structure and the manifest without reading the bulk
   payload. Offline publication verifies every page and the whole payload.
   Reading a row takes one page snapshot, verifies its duplicated header and
   digest, then publishes the requested encoded row; failure leaves the caller
   output unchanged.
7. The writer uses a sibling temporary file, bounded one-page memory, fsync,
   closes and reopens through the same parser, verifies the complete payload
   and boundary pages, then renames atomically and fsyncs the parent directory.
   Rename is the commit point; a post-rename directory-fsync error is reported
   without attempting an unsafe rollback.
8. `row_alignment` and physical `page_alignment` are distinct. Admission
   requires `padded_rows == align_up(last_offset + last_prime, row_alignment)`.
   For the pinned profile this is
   `align_up(320001446, 128) == 320001536`, exactly 90 padding rows.
9. Codec ID, codec version, group size, encoded row bytes, rows per page and
   page alignment are exact fields of a future codec-specific artifact
   profile. Phase 2 tests use an explicitly non-production byte fixture only.

## Consequences

- `ds4_ple_store.[ch]` can validate miniature and embedded extents without
  allocating a giant table or touching all PLE pages at startup.
- The structural version is stable while codec promotion remains fail-closed.
  Changing wire offsets, digest coverage or page-header meaning requires a new
  format version; selecting a production codec requires a reviewed artifact
  profile and quality/performance evidence.
- No production loader registers, warms or presents this extent to Metal in
  Phase 2. Runtime ownership and admission remain Phase 3 work.
