# Cross-Session Work Protocol

Files in this directory transfer operational state between coding-agent
sessions. They are not architecture, product, or release authorities. Use
[`HANDOFF_TEMPLATE.md`](HANDOFF_TEMPLATE.md) for a task-specific file under
`docs/work/active/`.

No file under `docs/work/active/` may be merged into `main`. Before merge,
promote durable facts to their canonical destination and delete the handoff.

## Session Start

1. Read `AGENTS.md`, the runtime support contract, the relevant CODEMAP route,
   and the applicable `CONTRIBUTING.md`/`QA_BEFORE_RELEASES.md` sections. Read
   the complete QA document when the task prepares or signs off a release.
2. Verify the actual repository path, branch, HEAD, remotes, worktree status,
   and merge base. Do not trust a stale handoff over Git.
3. Check for other agents, worktrees, builds, or huge inference processes that
   could modify shared files or contaminate measurements.
4. Read the active handoff and verify every claimed artifact, test, benchmark,
   and blocker that is cheap and safe to check.
5. State the task boundary, owned files, baseline, required gates, and stop
   condition before changing code.

## During Work

- Keep the handoff concise and current; replace disproven assumptions instead
  of appending a diary.
- Label statements as verified fact, inference, failed attempt, or pending work.
- Record exact commits, model hashes, commands, host/backend, cache/page state,
  thermals when relevant, output identity, and result paths.
- Record failed experiments only when the conclusion prevents useful repetition:
  hypothesis, controlled comparison, result, and rejection reason.
- Update canonical documentation in the same tranche as the code it describes.
- Remove superseded documentation, flags, debug code, test scaffolding, and
  generated scratch as soon as they lose their purpose.

## Durable Knowledge Destinations

| Knowledge | Canonical destination |
| --- | --- |
| File ownership or execution flow | `docs/architecture/CODEMAP.md` or a focused architecture document |
| Product/backend/model support | `docs/contracts/RUNTIME_SUPPORT.md` plus an ADR when policy changes |
| Long-lived architectural decision | `docs/adr/` |
| Contribution or regression requirement | `CONTRIBUTING.md` |
| Release procedure | `QA_BEFORE_RELEASES.md` |
| User-visible startup/behavior | README, help, or focused model documentation |
| Accepted/rejected performance claim | Dated record under `docs/benchmarks/` with reproducible evidence |
| Fork/upstream boundary | `FORK_NOTES.md` |

Do not copy a canonical checklist into another document. Link to it and record
only task-specific evidence.

## Session End

1. Recheck status and record the exact HEAD, dirty files, and uncommitted work.
2. Record what changed, what was verified, exact commands/results, unresolved
   risks, and the next safe command.
3. Update all canonical documents affected by the resulting code.
4. Remove false leads and stale instructions from the handoff.
5. Stop background model/build processes started by the task and record any
   intentionally retained process.
6. If the task is merge-ready, perform the pre-merge review below and delete
   the active handoff after knowledge promotion.

## Pre-Merge Review

Review the complete diff from its merge base, preferably with an agent that did
not implement it. The review must apply, not duplicate, the full requirements
in `CONTRIBUTING.md` and the applicable `QA_BEFORE_RELEASES.md` sections.

The reviewer must also inspect specifically for merge residue:

- active handoffs, scratch plans, debug output, commented code, and stale TODOs;
- test-only branches, hooks, fixtures, compiler suppressions, or default flags;
- rejected experimental paths, unused compatibility code, and orphan options;
- unregistered environment variables and contradictory enable/disable switches;
- generated files without generator/provenance/check mode;
- local paths, model files, logs, benchmark scratch, and accidental binaries;
- dead declarations, includes, fields, helpers, documents, and duplicated rules;
- documentation that describes the plan instead of the resulting code.

The review result is `PASS` or `BLOCK`. A follow-up is acceptable only for work
independent of the current change; cleanup required by the diff is blocking.

Before merge, record required tests and model-backed runs, run
`git diff --check`, verify documentation links, and prove that no active handoff
is included. Structural changes to shared model, session, ExpertMajor, SSD,
tokenizer, Metal graph, prefill, or decode code require qualified DeepSeek, GLM,
and Qwen reruns.
