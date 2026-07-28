# Coding Agent Guide

This file is the entry point for coding agents working in this repository. It
is deliberately short. Follow its links instead of loading every design note
or historical benchmark into context.

## Read First

For every task, read these documents in order:

1. [`docs/contracts/RUNTIME_SUPPORT.md`](docs/contracts/RUNTIME_SUPPORT.md) --
   current product and model support boundary.
2. [`docs/architecture/CODEMAP.md`](docs/architecture/CODEMAP.md) -- ownership and
   navigation map; then read only the documents for the subsystem being changed.
3. The sections of [`CONTRIBUTING.md`](CONTRIBUTING.md) and
   [`QA_BEFORE_RELEASES.md`](QA_BEFORE_RELEASES.md) applicable to the realistic
   failure modes of the task.
4. An active task handoff under `docs/work/active/`, if one exists.

Read the entire `QA_BEFORE_RELEASES.md` only when preparing or signing off a
release. Non-release tasks still apply every relevant QA section; they do not
load unrelated release lanes into context.

`CONTRIBUTING.md` and `QA_BEFORE_RELEASES.md` are canonical. An agent note,
handoff, plan, PR description, or convenience target may add evidence but may
not weaken, replace, or silently skip their gates. If documents disagree, stop
and resolve the contradiction in the same change.

## Runtime Contract

The production path is Apple Metal inference with an embedded ExpertMajor v2
store. DeepSeek V4 and Qwen3.6 have qualified AUTO/resident/SSD modes. GLM 5.2
is SSD-only: normal AUTO resolves to SSD streaming and a resident request is
rejected. DeepSeek and GLM require at least 64 GiB of unified memory; Qwen has a
separate hardware-aware 16 GiB minimum and necessarily uses SSD at that tier.
See the runtime support contract and accepted ADRs for exact boundaries.

CUDA and ROCm are frozen and their backend source is intentionally absent from
the active tree. Do not restore or extend it without a new accepted decision;
recover it from Git if active work resumes. The canonical QA document remains
authoritative: normal releases confirm the frozen sources remain absent, while
any restoration triggers the complete reactivation gate. Distributed inference
is likewise retired and its source is absent; the shared CLI policy keeps its
former flags as explicit fail-closed tombstones. Recover retired backend code
from Git only after a new accepted decision. CPU code is reference/debug code,
not a production inference fallback.

## Working Rules

- Preserve correctness before speed. Do not retain unexplained logits,
  attention, routing, KV, or generated-output drift.
- Keep the release path singular. A successful experiment becomes the default
  and loses its flag; a rejected experiment and its scaffolding are removed.
- Do not add model backward compatibility, sidecars, canonical routed-weight
  fallback, ExpertMajor v1 inference, or admission bypasses.
- Do not add C++. Objective-C belongs only where Metal requires it.
- Keep public APIs narrow. CLI and server layers must not depend on tensor or
  kernel internals.
- Keep model loading mmap-backed. SSD expert loading, cache ownership, and
  overlap boundaries must remain explicit and measurable.
- Do not run multiple huge model processes concurrently. Use an isolated
  worktree/process lane when another agent or inference run may interfere.
- Do not make a move-only refactor and a behavior change in the same commit.
- Files under `runtime/*.inc` are textual implementation partitions owned by
  their including translation unit. Do not compile them separately or turn
  them into public headers; preserve lexical order and explicit build deps.
- Never hide a warning with `MAYBE_UNUSED` or a test-only compiler suppression
  without a documented, narrowly scoped reason.
- Generated files need a reproducible generator, input provenance, and a check
  mode. Do not commit local logs, binaries, model files, or benchmark scratch.

### Implementation Explanations

When proposing, attempting, or reporting an implementation, explain the intent,
mechanism, expected effect, and important risks in clear, plain language. Use a
small concrete example whenever it makes the change easier to understand, and
keep the example focused on one idea at a time.

Analogies and metaphors must preserve the relevant technical relationships.
Explicitly map their parts to the real code, runtime state, or data flow, and
state where the comparison stops being accurate. Do not use a memorable
metaphor if it would hide an important constraint or imply behavior the system
does not have.

## Knowledge Across Sessions

Use the lifecycle in [`docs/work/README.md`](docs/work/README.md). An active
handoff records operational state, not permanent truth. Promote durable facts
before merge:

- architecture or ownership changes -> `CODEMAP.md`;
- support changes -> `RUNTIME_SUPPORT.md` and, when needed, an ADR;
- user-visible behavior -> README/help/model documentation;
- accepted or rejected optimizations -> a concise benchmark decision record;
- fork/upstream boundary changes -> `FORK_NOTES.md`.

Update documents in the same change as the code. Remove superseded text instead
of adding a second explanation. Active handoffs, stale test hooks, debug flags,
temporary logs, commented code, and obsolete fixtures must not survive merge.

## Validation And Merge

Choose tests by realistic impact using `CONTRIBUTING.md`; use the complete
`QA_BEFORE_RELEASES.md` gate for releases. A structural change to common model,
session, ExpertMajor, SSD, tokenizer, Metal graph, prefill, or decode code must
rerun the qualified DeepSeek, GLM, and Qwen models before merge.

### Test Result Reporting

Whenever an agent reports test or benchmark results in chat, a handoff, commit
or PR notes, or a benchmark record, present the results in Markdown tables
rather than prose alone. Keep rows in chronological order and identify each new
run with an ISO 8601 timestamp including the timezone, captured at the start of
the run whenever practical, so the experiment sequence is unambiguous.

For inference performance results, use separate columns for prefill throughput
and decode throughput, and include decode TPOT p50/p95 when collected. Every
comparable row must show the delta against the tested `main` baseline and
against the most recent comparable experiment, with the commit or experiment
identity for both references. Keep model, artifact, mode, prompt/frontier,
hardware, and other acceptance conditions visible in the table or an adjacent
metadata table.

Never manufacture a comparison between unlike conditions. If a matching
`main` baseline, previous experiment, timestamp, or phase-specific metric is
unavailable, write `N/A` and explain why. For non-performance tests, use a
table containing at least the timestamp, revision or experiment, test
command/lane, and result.

A short-context run may reject an inference optimization for correctness,
safety, or a clear regression, but it may never promote the change or
generalize a speed claim. Apply the canonical short/medium/large/long matrix in
`CONTRIBUTING.md`, keep its frontiers separate, and require the 32K long-context
lane for every surviving promotion candidate.

Long context is a primary product workload, not an edge case. The 32K lane is
only the minimum long-context screen. Every surviving model-backed inference
performance candidate must also complete an isolated endpoint lane at the
largest admitted prompt frontier that leaves room for at least 128 greedy
decode tokens and runtime bookkeeping. Derive the limit from the locally
validated artifact metadata and runtime admission, not a model-card claim. For
the current Qwen3.6 profile, whose validated metadata declares a 262,144-token
context, qualification therefore includes a near-262K endpoint lane. If a
qualified hardware/mode profile cannot complete its advertised endpoint
safely, either fix the runtime or narrow the public context contract; do not
waive the lane. A win at a smaller frontier may never hide a regression at a
larger or endpoint frontier.

For context-sensitive inference work, start performance exploration at 8K;
use shorter frontiers as secondary correctness, safety, and low-context-cost
checks. At exact-output parity and with no measured regression in any qualified
lane, prefer an implementation with demonstrably lower asymptotic work, memory
traffic, dispatch count, synchronization, or bounded resource use even when
the current host's wall-clock difference is inside measurement noise. This is
a tie-breaker for a singular production path, not permission to retain a
parallel kernel, flag, or merely theoretical optimization whose cost reduction
has not been demonstrated.

The durable benchmark record must include the final
baseline-versus-combined-stack comparison. An abort, swapout, changed resolved
plan, or competing inference process invalidates the complete comparison
cohort.

Every merge candidate needs a review of the full diff from its merge base. The
reviewer should be independent of the implementer when practical and must look
for correctness/performance regressions, unsupported compatibility, unused
code, experimental residue, test-only code in release builds, unregistered
flags, generated artifacts, stale documentation, and personal absolute paths.

A task is complete only when required tests and model runs are recorded, the
documentation describes the resulting code, temporary handoffs are removed,
the residue review passes, and `git diff --check` is clean.

Run `make premerge` for the repository, documentation, build-isolation, and
model-free gates. It does not replace the qualified DeepSeek, GLM, and Qwen
model runs required above.
