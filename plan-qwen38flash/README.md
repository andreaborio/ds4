# Qwen3.8-Flash-Next in Hebrus: implementation dossier

Status: implementation specification, not a support claim. Baseline: `hebrus/main`
at `7600fe183325b0c56fbdd8cc31120789724293ba` (2026-08-29).

To start a lead implementation agent, use
[START_IMPLEMENTATION_PROMPT.md](START_IMPLEMENTATION_PROMPT.md) verbatim or as
the authoritative base prompt.

## 1. Exact target

The target is the open-weight Hugging Face checkpoint
`Qwen/Qwen3.8-Flash-Next`, architecture `qwen4_exp`, revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540`. It is not the separately hosted
Alibaba service named `Qwen3.8-Flash`. The first Hebrus support increment is:

- text input and text output on Apple Metal;
- the checkpoint's base next-token head, without MTP acceleration initially;
- the complete 262,144-token model context contract, admitted in phases;
- a native `qwen4_exp` graph rather than pretending the model is `qwen35moe`;
- an embedded ExpertMajor v2 routed-expert store and a checksummed PLE row store;
- M5 Pro 64 GB as the primary performance target, with fail-closed portable
  Metal paths for other Apple GPU families;
- CPU routines only as small tensor/reference oracles, never as a production
  fallback.

The initial artifact is deliberately text-only. The upstream checkpoint also
contains a vision tower. Converter and loader must either build/admit the exact
documented text artifact or reject it; they must not silently accept images,
video tokens, missing vision weights, an unknown `qwen4_exp` variant, or a
generic Qwen checkpoint.

## 2. Definition of done

No code is “supported” until all of these are true:

1. The converter pins the source revision and emits a deterministic manifest,
   checksums, the dense tensors, one embedded ExpertMajor v2 store, and one PLE
   store with validated row geometry.
2. Admission checks every closed constant, tensor name, shape, type, byte
   extent, tokenizer control token, layer pattern, PLE hash parameter and
   checksum before allocating the execution graph.
3. Tokenizer and template fixtures match the pinned upstream tokenizer.
4. Each new primitive has an independent float32 oracle and adversarial tests.
5. Prefill, decode, reset, fork/copy if exposed, cancellation and error unwind
   preserve all GDN, QSA index, QSA KV, PLE-history and convolution state.
6. Full-model logits agree with the pinned Transformers reference at declared
   tolerances on resident and SSD paths.
7. The normal `AUTO` run on an M5 Pro 64 GB has measured memory ownership,
   produces the correct evidence text, does not swap, and beats the declared
   baseline under the repository performance protocol.
8. Lower-memory/general-Metal claims are enabled only after their own physical
   qualification. A generic kernel existing is not a support claim.

## 3. Required reading order

1. [00-source-register.md](00-source-register.md): pinned sources and evidence
   discipline.
2. [01-model-contract-and-math.md](01-model-contract-and-math.md): exact model,
   equations and state semantics.
3. [02-hebrus-integration-map.md](02-hebrus-integration-map.md): current Hebrus
   components, reuse boundaries and proposed files.
4. [03-artifact-converter-and-admission.md](03-artifact-converter-and-admission.md):
   deterministic artifacts, tensor mapping and loader rejection rules.
5. [04-metal-kernels-and-graph.md](04-metal-kernels-and-graph.md): kernel and
   graph implementation contract.
6. [05-ssd-streaming-and-memory-policy.md](05-ssd-streaming-and-memory-policy.md):
   ExpertMajor and PLE I/O, budgets, concurrency and pressure behavior.
7. [06-implementation-phases.md](06-implementation-phases.md): ordered changes
   sized for implementation agents.
8. [07-test-oracles-and-qualification.md](07-test-oracles-and-qualification.md):
   fixtures, tolerances, negative tests and release gates.
9. [08-performance-plan-m5-pro.md](08-performance-plan-m5-pro.md): measurement
   and optimization plan.
10. [09-risks-and-decisions.md](09-risks-and-decisions.md): open decisions,
    risks and explicit non-decisions.
11. [10-agent-playbook.md](10-agent-playbook.md): snippets and review checklists
    for agents implementing the plan.
12. [11-upstream-engine-research.md](11-upstream-engine-research.md): detailed
    llama.cpp, MLX-VLM/MLX-LM and Metal comparison, including what not to copy.

## 4. Non-negotiable design rules

- Do not weaken existing Qwen3.6, DeepSeek or GLM contracts to fit the new
  model. Shared changes rerun all affected suites.
- Do not add a canonical GGUF routed-expert fallback, v1 store compatibility,
  sidecar discovery, bypass environment flag, CPU production path, or model
  family guessing.
- Keep arithmetic that defines routing, decay, normalization, PLE hashes and
  sparse selection in the precision and order specified here. Optimize only
  behind an oracle and an A/B proof.
- Every state buffer has one named owner, a lifetime, a byte formula and reset/
  copy semantics. Allocation success is transactional.
- Treat SSD as asynchronous storage, not slow RAM. Decode must not require a
  cold multi-megabyte expert read for every selected expert. PLE reads are
  deduplicated, page-aligned, overlapped and measurable.
- Never cite an upstream throughput number as Hebrus performance. Upstream
  results are hypotheses and comparison points only.

## 5. Proposed delivery shape

The work is intentionally split so that early pull requests establish contracts
and oracles without touching the giant runtime hot spots. The new graph should
live in textual include partitions such as `runtime/ds4_qwen4exp_graph.inc` and
`runtime/ds4_metal_qwen4exp.inc`, included from the existing translation units;
they are not separate objects. The exact phase sequence and allowed parallelism
are in `06-implementation-phases.md`.
