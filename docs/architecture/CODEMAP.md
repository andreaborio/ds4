# Code Map

This map describes the repository as it exists today. It is a navigation and
ownership index, not the target refactor layout. Update it in the same change
when files gain, lose, or transfer a responsibility.

Start with [`AGENTS.md`](../../AGENTS.md), the
[`runtime contract`](../contracts/RUNTIME_SUPPORT.md), and only the subsystem
below that is relevant to the task. Historical benchmark notes are evidence,
not architecture specifications. The production-backend boundary is recorded
in [`ADR 0002`](../adr/0002-apple-metal-production-runtime.md). Qwen's
RAM/working-set adaptation is isolated in
[`ADR 0004`](../adr/0004-qwen-metal-hardware-memory-policy.md). The proposed
Hebrus public-name migration and its immutable compatibility boundary are
recorded in [`ADR 0005`](../adr/0005-hebrus-naming-and-compatibility-boundary.md).
Qwen's shared runtime and dual weight-codec boundary are recorded in
[`ADR 0006`](../adr/0006-qwen-dual-weight-codecs.md).
Qwen's canonical, byte-identical Hebrus artifact basenames are recorded in
[`ADR 0007`](../adr/0007-qwen-hebrus-artifact-filenames.md).
For the runtime data and admission path at a glance, see the accessible
[`mmap → ExpertMajor → AUTO → Metal/SSD flow`](hebrus-runtime-flow.svg).

## Runtime Entry Points

The Makefile links one canonical `hebrus*` executable per role from these
entrypoints and publishes the corresponding `ds4*` name as a symlink to the
same file. It also owns the `DESTDIR`/`PREFIX`/`BINDIR` install boundary:
canonical executables are copied, compatibility aliases stay relative, and
the Metal build's exact runtime source set is installed under the versioned
resource root derived from `BINDIR`. Uninstall names every removable command
and resource path explicitly. There are no alias-specific wrappers or object
graphs.

| Path | Primary responsibility |
| --- | --- |
| `ds4_cli.c` | One-shot CLI, interactive transcript, and command-line orchestration |
| `ds4_server.c` | HTTP server, OpenAI/Responses/Anthropic protocols, request parsing, generation jobs, streaming, and disk-KV policy |
| `ds4_agent.c` | Stateful coding-agent TUI, tools, session commands, and interruption state |
| `ds4_bench.c` | Context-frontier prefill/decode benchmark driver |
| `ds4_eval.c` | Evaluation driver and extractor checks |
| `ds4_build.c` | Build identity and the versioned machine-readable capability contract reported by executables |

These programs should use the public engine/session API in `ds4.h`. They should
not acquire tensor, ExpertMajor, or Metal-kernel ownership.

## Core Runtime

| Path | Current responsibility |
| --- | --- |
| `ds4.h` | Public engine, token, generation, evidence, and session API |
| `ds4.c` | Model profiles; GGUF parsing/mapping; quant blocks and CPU reference kernels; tensor binding; tokenizer/chat rendering; model-family dispatch; common Metal graph scheduling; ExpertMajor/SSD orchestration; engine/session implementation; diagnostic paths |
| `ds4_expert_store.[ch]` | Embedded ExpertMajor store metadata, validation, and indexed range access |
| `ds4_ssd.[ch]` | SSD residency and read-policy support shared with the runtime |
| `ds4_profile.[ch]` | Metal/SSD profiling records and summaries |
| `ds4_kvstore.[ch]` | Disk-backed server/agent KV checkpoint store |
| `ds4_help.[ch]` | Shared structured command help and centralized rejection of retired CLI flags |

`ds4.c` remains a refactor hotspot. Use its section headers and search by API
or model-family name; do not read the entire file by default. Model-specific
graph implementations may be textual partitions under `runtime/` so hot static
helpers remain in the same translation unit while agents can load one family.

## Model-Specific Support

| Path | Primary responsibility |
| --- | --- |
| `ds4_qwen.[ch]` | Qwen state, metadata, shapes, and model-specific helpers; the runtime profile binding remains beside the complete tensor inventory in `ds4.c` |
| `ds4_qwen_expert_group.[ch]` | Qwen expert grouping and slab planning |
| `ds4_qwen_ref.[ch]` | Qwen numeric reference implementations used by tests |
| `ds4_qwen4exp.[ch]` | Immutable, model-free descriptor and checked logical tensor-payload formulas for the pinned text-only Qwen3.8-Flash-Next/Qwen4Exp profile; this is not runtime support admission |
| `ds4_qwen4exp_ref.[ch]` | Allocation-free scalar C oracle for Qwen4Exp norms, gated residuals, GDN, routing, sparse-index selection, PLE hashing/gating/convolution, and tiny state checkpoint semantics |
| `ds4_qwen4exp_chat.[ch]` | Standalone exact Qwen4Exp chat-template renderer with transactional output and contiguous client-data versus trusted-control provenance segments; normal runtime/server support remains disabled |
| `ds4_ple_store.[ch]` | Structural parser, verifier and atomic writer for the embedded fixed-page `ds4.ple_rows.v1` extent; codec admission remains caller/profile-explicit |
| `ds4_qwen_unicode.[ch]` | Qwen Unicode/tokenizer data access |
| `ds4_qwen_unicode_data.inc` | Generated Unicode data; provenance lives under `tests/qwen/` |
| `ds4_streaming_hotlist.inc` | DeepSeek streaming hotlist data included by `ds4.c` |
| `ds4_streaming_hotlist_glm52.inc` | GLM 5.2 streaming hotlist data included by `ds4.c` |
| `runtime/ds4_deepseek_cache_phase.inc` | DeepSeek adaptive ExpertMajor cache transitions around batched prefill; textually included by `ds4.c` |
| `runtime/ds4_glm_graph.inc` | GLM Metal graph state, allocation, prefill/decode scheduling, routed MoE/SSD orchestration, and GLM generation; textually included by `ds4.c` |
| `runtime/ds4_qwen4exp_loader.inc` | Closed Qwen4Exp metadata, tokenizer-provenance, physical inventory, ExpertMajor/PLE manifest and whole-file ownership admission; production has no registered physical profile and the only positive is a CPU-only structural test hook |
| `runtime/ds4_metal_glm.inc` | GLM-specific Metal encoders and tensor wrappers; textually included by `ds4_metal.m` |

Qwen keeps one Metal graph and scheduler across its two accepted weight codecs.
The graph separates session-lifetime core/KV/logits from the SSD prefill arena
and macro rollback workspace, which are owned only by an active prefill phase;
resident mode retains its wide arena and rollback workspace for the session.
SSD admission remains conservative at the logical prefill limit, while live
tensor accounting follows the arena's current physical capacity. On guarded
16/24 GiB tiers, macro-prefill planning also preserves arithmetic room for one
complete additional route cycle at the next phase entry; the runtime still
requires a fresh normal-pressure signal before decode and before every slab.
The guarded macro frontier is capped at 32K so a hotter GGUF page cache cannot
turn additional reclaimable credit into a larger transient allocation than the
small-memory profile is intended to carry.
After releasing transient prefill storage, guarded decode entry may re-sample
pressure for at most 30 seconds; it proceeds only after macOS reports a fresh
normal signal, otherwise it fails closed and rolls the transaction back.
The physical 16 GiB policy admits at most a 128K prompt frontier plus 128
decode tokens and one bookkeeping slot; higher-memory profiles retain their
separate context contracts.

DeepSeek and non-graph GLM binding/reference/session support still live directly
in `ds4.c`; this is current-state debt, not permission to add another family
there without an explicit decision.

The `runtime/*.inc` files are implementation partitions, not headers or new
translation units. They intentionally preserve static linkage, lexical order,
inlining, and generated code. Keep their Makefile dependencies explicit and
prove codegen identity for move-only edits before behavior work continues.

## Metal Production Backend

| Path | Primary responsibility |
| --- | --- |
| `ds4_gpu.h` | Shared GPU-facing interface used by core graph scheduling |
| `ds4_metal.m` | Metal device/runtime state, versioned installed/build-tree source discovery, runtime library compilation, buffers, generic command encoding, tensor transfers, ExpertMajor resident/SSD paths, non-partitioned model-family wrappers, and one-time Qwen codec dispatch |
| `metal/*.metal` | Metal compute kernels grouped by operation or model family, including separate Affine4 and routed-IQ weight decoders under the shared Qwen graph; the required set plus the generated IQ table include is installed as runtime data for Metal builds |

`ds4_metal.m` is the second refactor hotspot. Keep Objective-C runtime calls
there. Before moving hot functions across translation units, compare generated
code and run the applicable performance gates; the current build does not rely
on link-time optimization to restore lost inlining.

## Retired And Non-Production Backends

| Path | Current status |
| --- | --- |
| CUDA | Frozen backend source removed from the active tree; recover from Git commit `d8d673858f90834522bbe878951a534d8c6508b4` if a new decision reopens it |
| ROCm | Frozen backend source removed from the active tree; recover from Git commit `d8d673858f90834522bbe878951a534d8c6508b4` if a new decision reopens it |
| Distributed | Retired source absent from the active tree; former CLI flags are centralized fail-closed tombstones; recover implementation from Git commit `d8d673858f90834522bbe878951a534d8c6508b4` only after a new accepted decision |

Do not use historical backend code as design authority for new production work.
The runtime contract defines what may execute; the QA document remains
authoritative for release evidence and for reporting lanes that were skipped or
not validated.

## Tools, Tests, And Evidence

| Path | Primary responsibility |
| --- | --- |
| `tests/` | Model-free, model-backed, kernel, tokenizer, server, and build-isolation regressions |
| `tests/test_capabilities.py` | Exact schema and cross-executable checks for the model-free build/capability contract |
| `tests/test_command_aliases.py` | Canonical/legacy symlink layout, binary identity, and CLI-output parity checks |
| `tests/test_install.sh` | Temporary-root install/uninstall layout, path portability, capability, versioned Metal-resource discovery from a clean working directory, model-free library initialization when a device is available, and explicit-removal checks |
| `tools/release_source.py` + `tests/test_release_source.py` | Fail-closed deterministic source-archive generation and verification: full-commit/clean-tree provenance, isolated Git-object archiving, canonical and traversal-safe tar/gzip metadata, strict JSON manifest and SHA-256 set, race-safe overwrite refusal, and tamper fixtures |
| `tests/test_release_source_smoke.sh` + `.github/workflows/release-source.yml` | Double-build reproducibility, extraction outside Git, explicit build provenance, staged install smoke, and read-only retention of the three source-release files; the workflow does not publish a release |
| `tools/brand_boundary.json` + `tools/brand_boundary_audit.py` | Exact canonical, bridged, and permanently preserved identity contract plus explicit per-file legacy `ds4`/`DS4`/`DwarfStar` classification and monotonic count ceilings; `--check` rejects contract drift, new groups, and increases, while `--refresh` requires exact authorizations before widening a ceiling |
| `tests/test_brand_boundary_audit.py` | Fail-closed fixtures for new files and tokens, increases, reductions, deterministic refresh, and invalid manifests |
| `docs/contracts/qwen-release.json` + `tools/qwen_release_contract.py` | Canonical Hebrus-named Stable, opt-in Beta, and historical negative-only Qwen artifact identities plus the model-free gate that parses their documentation, downloader, and test surfaces for drift |
| `tests/test_qwen_release_contract.py` | Fail-closed fixtures for prose, table, downloader, schema, status, and negative-only Qwen release-contract drift |
| `tests/qwen/` | Qwen fixtures, provenance, reference collectors, and model-specific gates |
| `tests/qwen4exp/` | Pinned Qwen4Exp source inventory, upstream-backed scalar/tokenizer/chat captures, independent controls, per-case provenance, sparse Phase 3 GGUF admission builder, black-box negative battery, and fail-closed regeneration checks |
| `gguf-tools/qwen4exp-profile.py` | Header/index-only Qwen4Exp conversion dry run that maps and byte-accounts every pinned source identity without loading checkpoint tensors |
| `tests/test-vectors/` | Official and local continuation vectors plus provenance |
| `gguf-tools/` | Quantization, ExpertMajor conversion, imatrix, and quality-scoring tools |
| `speed-bench/` | Benchmark prompt, driver helpers, plots, and historical results |
| [`docs/benchmarks/`](../benchmarks/README.md) | Indexed dated benchmark decisions and measurements |

Tests must link production objects through public or narrow internal APIs. Tests
that include a complete production `.c` file are known coupling debt and must
not be copied into new tests.

## Documentation Authority

| Path | Authority |
| --- | --- |
| `CONTRIBUTING.md` | Contribution, regression, and upstream-coordination gates |
| `QA_BEFORE_RELEASES.md` | Complete release checklist |
| `docs/contracts/RUNTIME_SUPPORT.md` | Current supported runtime/model matrix |
| `docs/contracts/BRAND_COMPATIBILITY.md` | Canonical Hebrus names, compatibility aliases, and permanently stable identifiers |
| `docs/adr/` | Accepted architectural decisions and their consequences |
| `GOLD_METAL_SSD.md` | Metal/SSD planner details and performance gates; support authority remains `RUNTIME_SUPPORT.md` |
| `FORK_NOTES.md` | Time-stamped fork/upstream boundary ledger |
| `docs/work/` | Temporary cross-session operational state; never product truth |

When two normative documents overlap or disagree, fix the conflict rather than
adding another note. Superseded research belongs in history; it must not remain
worded as a current instruction.

## Change Impact Routes

- GGUF metadata, tensors, or admission: inspect `ds4.c`, `ds4_expert_store.*`,
  the model-family helper, converter tests, and runtime support contract.
- Prefill/decode or routing: inspect the family graph code in `ds4.c` or its
  `runtime/` partition, `ds4_gpu.h`, `ds4_metal.m` or its family partition,
  relevant kernels, gold gates, and family tests.
- SSD/cache behavior: inspect `ds4_ssd.*`, ExpertMajor access in `ds4.c`, Metal
  ExpertMajor wrappers, profiling, and SSD/model tests.
- Tokenizer/template: inspect tokenizer sections in `ds4.c`, Qwen Unicode code,
  golden vectors, server rendering, and API tests.
- Server/session behavior: inspect `ds4_server.c`, `ds4_kvstore.*`, public
  session APIs, protocol tests, and agent session behavior when shared.
- CLI/configuration: inspect all executable parsers, `ds4_help.*`, environment
  reads in runtime/backend code, README startup commands, and release help checks.
- Source packaging: inspect `tools/release_source.py`, its unit and extracted
  install tests, the read-only workflow, the README package contract, and the
  canonical source-bundle section in `QA_BEFORE_RELEASES.md`.
