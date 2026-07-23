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
[`ADR 0004`](../adr/0004-qwen-metal-hardware-memory-policy.md).

## Runtime Entry Points

| Path | Primary responsibility |
| --- | --- |
| `ds4_cli.c` | One-shot CLI, interactive transcript, and command-line orchestration |
| `ds4_server.c` | HTTP server, OpenAI/Responses/Anthropic protocols, request parsing, generation jobs, streaming, and disk-KV policy |
| `ds4_agent.c` | Stateful coding-agent TUI, tools, session commands, and interruption state |
| `ds4_bench.c` | Context-frontier prefill/decode benchmark driver |
| `ds4_eval.c` | Evaluation driver and extractor checks |
| `ds4_build.c` | Build identity reported by executables |

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
| `ds4_kv_quant.[ch]` | Internal cross-engine KV surface geometry and deterministic TQ4 conformance reference; production cache ownership remains model-specific |
| `ds4_kvstore.[ch]` | Disk-backed server/agent KV checkpoint store |
| `ds4_help.[ch]` | Shared structured command help and centralized rejection of retired CLI flags |

`ds4.c` remains a refactor hotspot. Use its section headers and search by API
or model-family name; do not read the entire file by default. Model-specific
graph implementations may be textual partitions under `runtime/` so hot static
helpers remain in the same translation unit while agents can load one family.

## Model-Specific Support

| Path | Primary responsibility |
| --- | --- |
| `ds4_qwen.[ch]` | Qwen state, metadata, shapes, and model-specific helpers |
| `ds4_qwen_expert_group.[ch]` | Qwen expert grouping and slab planning |
| `ds4_qwen_ref.[ch]` | Qwen numeric reference implementations used by tests |
| `ds4_qwen_unicode.[ch]` | Qwen Unicode/tokenizer data access |
| `ds4_qwen_unicode_data.inc` | Generated Unicode data; provenance lives under `tests/qwen/` |
| `ds4_streaming_hotlist.inc` | DeepSeek streaming hotlist data included by `ds4.c` |
| `ds4_streaming_hotlist_glm52.inc` | GLM 5.2 streaming hotlist data included by `ds4.c` |
| `runtime/ds4_deepseek_cache_phase.inc` | DeepSeek adaptive ExpertMajor cache transitions around batched prefill; textually included by `ds4.c` |
| `runtime/ds4_glm_graph.inc` | GLM Metal graph state, allocation, prefill/decode scheduling, routed MoE/SSD orchestration, and GLM generation; textually included by `ds4.c` |
| `runtime/ds4_metal_glm.inc` | GLM-specific Metal encoders and tensor wrappers; textually included by `ds4_metal.m` |

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
| `ds4_metal.m` | Metal device/runtime state, buffers, generic command encoding, tensor transfers, ExpertMajor resident/SSD paths, and non-partitioned model-family wrappers |
| `metal/*.metal` | Metal compute kernels grouped by operation or model family |

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
| `tests/qwen/` | Qwen fixtures, provenance, reference collectors, and model-specific gates |
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
- Packed KV work: read
  [`KV_QUANTIZATION.md`](KV_QUANTIZATION.md), then inspect
  `ds4_kv_quant.*` and the owning Qwen, DeepSeek, or GLM graph. The shared
  module does not authorize cross-family tensor reinterpretation.
- Tokenizer/template: inspect tokenizer sections in `ds4.c`, Qwen Unicode code,
  golden vectors, server rendering, and API tests.
- Server/session behavior: inspect `ds4_server.c`, `ds4_kvstore.*`, public
  session APIs, protocol tests, and agent session behavior when shared.
- CLI/configuration: inspect all executable parsers, `ds4_help.*`, environment
  reads in runtime/backend code, README startup commands, and release help checks.
