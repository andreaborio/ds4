# ADR 0008: DeepSeek DSpark Uses A Separate Embedded Support Store

- Status: Proposed
- Date: 2026-08-02

## Context

The final DeepSeek V4 Flash checkpoint adds a three-stage DSpark drafter. A
preview-derived reference quantization measures about 5.58 GiB: about 5.06 GiB
of routed experts and 0.52 GiB of static tensors. This establishes the expected
storage geometry, not provenance for a final artifact; publication requires a
support file regenerated from the pinned final 0731 shards. The normal product
target is an Apple Silicon Mac with 64 GiB unified memory where AUTO resolves
the main model to SSD streaming. On that target, support-weight residency
competes directly with the qualified target expert cache.

The available upstream implementations do not settle this product case. The
current antirez README describes DSpark alongside SSD streaming, but its source
still rejects an external support model when target SSD streaming is active.
llama.cpp, vLLM, and the published DeepSpec measurements exercise resident
accelerator memory rather than Hebrus' mmap/pread ExpertMajor path. Copying one
of those schedulers would therefore import assumptions that are false on the
normal Hebrus target.

Appending the DSpark stages to the target `ds4.expert_major.v2` payload would
also change its existing layer and source-inventory contract. That would turn a
DeepSeek-only acceleration into a new target-store ABI and force unrelated GLM
and Qwen requalification.

## Proposed Decision

When DSpark is admitted, publish one self-contained GGUF containing two opaque
ExpertMajor v2 stores:

- `ds4.expert_major.v2` remains the byte-compatible target store;
- `ds4.dspark.expert_major.v2` contains only the three DSpark routed-MoE
  stages;
- the DSpark non-routed tensors remain ordinary tensors in the same GGUF.

The second tensor reuses the validated ExpertMajor v2 codec and manifest. It is
not an external sidecar, target-store extension, or new generic store version.
The loader gives the two stores distinct identities, cache quotas, eviction
accounting, read-byte counters, and wait-time counters while charging both to
one startup memory envelope. Neither cache may consume the other's allocation
implicitly.

Production execution remains the native Apple Metal graph. MLX may generate
development fixtures for the projection, HC, Markov, confidence, routing, and
sampling equations, but it is not linked into production and its timings do
not select a DSpark depth.

The normal runtime chooses a draft depth from zero through five. The model's
per-position confidence logits are passed through sigmoid and the prefix ends
before the first value below threshold, matching official DeepSpec semantics.
The threshold is then selected using native target-cache and I/O measurements;
a copied fixed upstream value is not a product policy. Depth zero is ordinary
target decode and is required whenever memory pressure, swap, tail length,
measured cost, or unsupported sampling semantics make speculation unsafe or
unprofitable.

There is no permanent user-facing experiment flag or separate release path.
Before DSpark is qualified, development builds may inspect and test the
embedded support store, but public artifacts and downloader defaults remain on
the already qualified target-only path.

## Admission And Promotion Gates

The combined artifact must fail before graph allocation when any DSpark
metadata, stage, tensor, source identity, store manifest, record geometry, or
memory charge is missing, duplicated, or inconsistent. The final checkpoint
contract is three stages, block size five, target layers 40/41/42, Markov rank
256, noise token 128799, 256 experts, and six selected experts per stage.
Shape compatibility with a preview-derived support file is insufficient: its
source revision and digest must trace reproducibly to the pinned final 0731
shards before packaging can be enabled.

The offline support generator therefore pins and authenticates the official
revision, config, index, and shards 46--48 both before and after conversion.
It embeds all six identities as GGUF metadata. The composer compares that
metadata against its own hardcoded production constants and separately
requires a manually reviewed digest of the complete generated support GGUF.
The generator never writes or derives that production pin, so a self-declared
manifest cannot enable packaging. Temporary output is exclusive and
no-follow, size/hash checked through its open descriptor, and installed from
that descriptor with no-clobber semantics before the directory is synced. An
output that appears during conversion is preserved. The tools provide no
destructive replacement mode; regeneration requires an explicit external remove
or a new artifact name.

Successful conversion establishes reproducibility, not model quality. Before
the complete support digest can be admitted, an independent final-0731
converter must confirm the tensor inventory and compare either exact bytes or
documented dequantization error per tensor. DSpark logits and draft decisions
must then match the official implementation and the MLX oracle before the
normal Apple AUTO-to-SSD performance matrix is allowed to make a promotion
decision.

The first performance gate is normal M5 Pro 64 GiB AUTO resolving to SSD at an
8K prompt frontier. Target-only and DSpark runs use equal total memory budgets
and comparable cache/page state. Promotion requires all of the following:

- exact greedy output and verifier/rollback parity;
- mathematically exact sampled acceptance and residual sampling;
- no swapout and no admission outside the shared envelope;
- separate target/support cache and SSD telemetry;
- lower full-request time and at least a material representative decode win;
- no material TPOT p95 regression;
- no increase in total target-plus-support SSD bytes per emitted token;
- the canonical 32K lane and the largest admitted endpoint lane;
- qualified DeepSeek, GLM, and Qwen regression runs for shared runtime changes.

A resident-only win does not qualify normal AUTO. If a bounded support cache is
slower, fully resident support may be tested as the one contingency, with its
complete footprint subtracted from the target cache. If neither configuration
passes the 8K gate, AUTO keeps DSpark parked and the acceleration code is not
promoted.

## Consequences

- The target ExpertMajor v2 format and its three-family runtime remain
  unchanged.
- Packaging needs a deterministic two-source converter that writes one GGUF
  atomically and verifies both payloads byte for byte without a combined
  canonical intermediate.
- The Metal cache layer needs explicit namespaces or equivalent ownership for
  target and support records; relying on the macOS page cache is insufficient.
- Server, CLI, and agent emission must not advance session state beyond a
  stopped or partially consumed speculative block.
- The final DeepSeek checkpoint can be published target-only before DSpark is
  qualified. DSpark is an acceleration gate, not a prerequisite for correct
  model support.
- Accepting this proposal after implementation requires corresponding updates
  to ADR 0001, the runtime support contract, CODEMAP, user documentation, QA,
  converter provenance, and the durable benchmark decision.
