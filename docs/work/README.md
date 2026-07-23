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
  thermals when relevant, output identity, and result paths. Performance work
  also records which canonical short/medium/large/long tiers are complete,
  whether a cohort is cold or warm, and every invalidation or restart. It may
  reject a candidate early for correctness, safety, or a clear regression, but
  may not mark a promotion decision complete while the 32K long-context tier is
  pending. Record the additional 64K/100K tiers when the changed path touches
  attention, KV, cache, RoPE, allocation, or context scaling.
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
| Identified commit lineages of a closed experiment | Pushed annotated `experiments/...` Git tags; never a substitute for the canonical result document |
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

## Closing And Archiving Experiment Branches

Working branches and worktrees exist only while their task is active. Preserve
every identified lineage of a closed experiment with immutable annotated tags,
then remove its working refs so inactive experiments do not accumulate as
branches. Use exactly one of these namespaces:

```text
experiments/accepted/YYYY-MM-DD-<slug>
experiments/rejected/YYYY-MM-DD-<slug>
experiments/abandoned/YYYY-MM-DD-<slug>
```

Use the plain name for the cleaned terminal lineage. If the experiment has a
side tip that is not its ancestor, preserve it with a descriptive suffix such
as `-side-<name>` or `-part-<number>` in the same outcome namespace.

The outcome means:

- `accepted`: the qualified result has been promoted to the production path
  and merged or is otherwise durably reachable from its integration branch;
- `rejected`: evidence supports a negative decision and all rejected code,
  flags, tests, and scaffolding have been removed from the terminal commit;
- `abandoned`: work stopped without an acceptance or rejection claim. The tag
  preserves provenance, but its contents are not evidence for a product or
  performance conclusion.

Close an experiment in this order:

1. Resolve the exact branches, worktrees, terminal commit, merge base, owner,
   and dirty state. Inspect the experiment's branch and worktree refs, relevant
   branch reflogs, and any known rewritten, reset, detached, or side tips. Stop
   if another task still owns any part or the intended archive scope is
   ambiguous. A tag preserves only the ancestors of its target.
2. Promote durable facts to the canonical documents above, delete the active
   handoff and experimental residue, complete the applicable review and checks,
   and commit that cleaned terminal state. An accepted experiment must first
   reach its intended integration branch; a useful negative performance result
   must retain its dated `docs/benchmarks/` record.
3. Confirm the intended tag names are unused locally and remotely, and inspect
   the selected remote's push URL. Never overwrite or force-update an existing
   archive tag. Create an annotated primary tag at the terminal commit. Its
   message records the outcome, decision-record path or reason none exists, and
   whether runtime code was promoted:

   ```sh
   git remote get-url --push <remote>
   git show-ref --verify refs/tags/experiments/<outcome>/YYYY-MM-DD-<slug>
   git ls-remote --exit-code <remote> \
     refs/tags/experiments/<outcome>/YYYY-MM-DD-<slug>
   git tag -a experiments/<outcome>/YYYY-MM-DD-<slug> <commit> \
     -m "<outcome>, evidence location, and promotion status"
   ```

   A missing-name check is expected to report no matching ref; any existing
   local or remote ref must be inspected instead of replaced. For every
   identified experiment tip, prove it is an ancestor of the primary tag's
   peeled commit with `git merge-base --is-ancestor`. If it is not, create and
   document a companion annotated tag for that tip. Do not delete any source
   ref until every intended tip is reachable from at least one archive tag.

4. Record the local annotated-tag object OID and peeled commit OID. Push only
   each exact tag to the canonical writable remote, then query the same exact
   refs with `--exit-code`:

   ```sh
   git rev-parse refs/tags/experiments/<outcome>/YYYY-MM-DD-<slug>
   git rev-parse 'refs/tags/experiments/<outcome>/YYYY-MM-DD-<slug>^{}'
   git push <remote> refs/tags/experiments/<outcome>/YYYY-MM-DD-<slug>
   git ls-remote --exit-code <remote> \
     refs/tags/experiments/<outcome>/YYYY-MM-DD-<slug> \
     'refs/tags/experiments/<outcome>/YYYY-MM-DD-<slug>^{}'
   ```

   Compare both remote OIDs exactly with the two local OIDs: the first is the
   annotated tag object and the second is its peeled commit. A missing line,
   mismatch, unexpected push URL, or pre-existing tag with different annotation
   blocks deletion. Repeat the reachability and exact-OID checks for every
   companion tag.

5. Resolve the exact worktree path from `git worktree list --porcelain`. Audit
   tracked and untracked state with
   `git status --porcelain=v1 --untracked-files=all`, then separately inspect
   ignored files with `git status --short --ignored`. Preserve every needed
   model, log, benchmark artifact, or generated file at its declared retention
   location outside the worktree. Stop if any file's ownership or retention is
   uncertain.
6. Only after all archive refs and files pass those checks, remove the exact
   worktree without `--force` and delete the local working branch. Delete a
   remote working branch only when it exists, is conclusively closed, and no
   collaborator still uses it. Never use a bulk or pattern-based deletion for
   experiment branches.

Tags in this namespace are immutable archive identities: do not retarget,
overwrite, or reuse them. Reopen an archived experiment under a new working
branch:

```sh
git switch -c codex/<new-task> \
  experiments/<outcome>/YYYY-MM-DD-<slug>
```

If a new clone does not fetch an archive tag automatically, fetch that exact
tag before reopening it:

```sh
git fetch <remote> \
  refs/tags/experiments/<outcome>/YYYY-MM-DD-<slug>:refs/tags/experiments/<outcome>/YYYY-MM-DD-<slug>
```

The pushed tags preserve every explicitly audited implementation lineage
off-machine and across garbage collection. Canonical Markdown records preserve
the conclusion; neither one replaces the other.

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
