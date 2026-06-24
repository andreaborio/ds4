# Upstream merge log

Record of syncs of this fork (`andreaborio/ds4`) with upstream `antirez/ds4`, and how the
fork's engine additions were preserved across upstream refactors. See
[`FORK_NOTES.md`](FORK_NOTES.md) for per-change upstreamability and [`README.md`](README.md)
("This fork") for what each addition does.

---

## 2026-06-24 — publish sync with upstream `main` @ `80ebbc3`

**Result: ✅ landed on `origin/main`.** `sync-fix` still contains upstream
`main` plus the fork's five mainline additions. The README now also points to the
separate experimental GLM-5.2 branch
`wip/glm52-metal64-strict-probe`, clearly scoped as Apple Silicon / Metal
short-context bring-up rather than a mainline feature.

No new upstream commits were present after `80ebbc3` during the 2026-06-24
refresh; the work here is publishing the already-validated sync and bringing the
fork README up to date before pushing `andreaborio/ds4`.

### 2026-06-24 verification on M5 Pro / 64 GB

The available full-size DeepSeek V4 Flash GGUF was larger than resident memory,
so model tests were run through Metal SSD streaming instead of default full
residency. A global `DS4_TEST_SSD_STREAMING=1 make test` is not a valid local
substitute for upstream's normal full-residency suite: it over-applies streaming
to tests that open multiple engines and, with a 16 GiB cache under macOS lock
pressure, reduced the runtime cache to pathological sizes. The accepted local
matrix was therefore split by test surface:

| check | local command/config | result |
|---|---|---|
| build | `make clean && make && make ds4_test` | ✅ OK |
| CPU portability | `make cpu` | ✅ OK (CPU-only unused warnings) |
| eval extractor self-test | `./ds4-eval --self-test-extractors` | ✅ OK |
| agent unit tests | `./ds4_agent_test` | ✅ OK |
| Q4_K dot unit tests | `make q4k-dot-test` | ✅ OK |
| server unit tests | `./ds4_test --server` | ✅ OK |
| Metal kernel unit tests | `./ds4_test --metal-kernels` | ✅ OK |
| official logprob vectors | SSD streaming, requested 40 GiB cache, capped to 36 GiB | ✅ OK |
| streaming decode/prefill correctness | SSD streaming, 11 GiB cache | ✅ OK |
| Metal tensor equivalence | SSD streaming, 11 GiB cache | ✅ OK |
| local golden vectors | SSD streaming, 11 GiB cache | ✅ OK |
| Metal short prefill | SSD streaming, 11 GiB cache | ✅ OK |
| long-context recall | SSD streaming, 11 GiB cache | ✅ OK |
| Metal SSD cache-pressure repro | test's built-in 16 GiB cache | ✅ OK |
| MTP verify depth | no local `DS4_TEST_MTP` GGUF | ✅ OK / self-skipped |

Two model-behavior tests were **not** green in SSD streaming on this 64 GB Mac:

- `./ds4_test --tool-call-quality`: the exact path tried to access dense Metal
  model ranges that are not mapped by this SSD-streaming configuration.
- `./ds4_test --think-tool-recovery`: recovery triggered and generated the
  expected-looking DSML stanza, but the parser saw `calls=0`.

Those two should be re-run on a machine/configuration that can execute the
normal full-residency upstream suite. They are documented here so the fork sync
does not claim a stronger test result than the local hardware can support.

---

## 2026-06-18 — sync with upstream `main` @ `80ebbc3`

**Result: ✅ synced and re-validated.** Branch `sync-fix` = upstream `main` + the fork's
non-streaming features (imatrix-collection API, expert prune mask, PROFILE_FULL). The fork's
streaming feature (#4) was **dropped** because it converged upstream. Passes the same
`ds4_test` correctness suite as pre-merge `main`.

### Divergence

- Fork `main` (`d2101a5`) was **11 commits ahead, 18 behind** upstream `main` (merge-base `8384adf`).
- The fork's `ds4.c` carried **+203 / −19 lines across 17 hunks** — four interwoven features in
  exactly the areas upstream refactored (streaming, session, prefill, imatrix):
  1. `DS4_EXPERT_PROFILE_FULL` toggle
  2. expert **prune mask** (`DS4_EXPERT_PRUNE_MASK`)
  3. **streaming "experts-uniform"** detection + span-decode helpers
  4. **imatrix-collection API** (`ds4_session_imatrix_*`, collector, prefill hooks)

### What went wrong with the naive merge (and how it was caught)

A plain `git merge upstream/main` produced **one** textual conflict
(`weights_streaming_layer_experts_uniform`); the other 16 hunks auto-merged and it **built**.
But the CONTRIBUTING acceptance suite caught a regression:

| test | naive merge (`720aad1`) | pre-merge `main` (`d2101a5`) |
|---|---|---|
| `ds4_test --streaming-decode-prefill-correctness` | ❌ **65 failures**, garbage decode logits (`cand≈1.9e37`) | ✅ OK (`cand==ref`, rms=0) |

→ a **merge-introduced regression** in streaming decode. Root cause: the streaming feature (#4)
had **fully converged upstream** — antirez independently added `model_map_span_vec_include_layer_decode`,
`weights_streaming_layer_experts_uniform`, `streaming_layer_routed_expert_bytes` and refactored
~10 streaming functions. The auto-merge **textually combined our streaming edits with his** in the
same functions (non-overlapping lines, so no conflict flagged) → incoherent streaming → corrupt
expert reads. (A first try keeping the whole fork `ds4.c` instead **failed to build**: upstream
changed function signatures, `error: too many arguments … expected 3, have 11`.)

### Fix

Take upstream's `ds4.c` (its streaming) and re-apply **only the 13 non-streaming fork hunks**
(profile + prune + imatrix API) — they land in functions upstream didn't touch, so they graft
cleanly (`git apply --3way`, 0 conflicts) and link. The 5 streaming hunks are **dropped** (now
upstream). Commits: merge `5800f15` (take-upstream-`ds4.c`) → `8236528` (re-graft non-streaming).

### Verification (CONTRIBUTING acceptance) — on the fix

| check | result |
|---|---|
| `make` (Metal, M5 Pro) | ✅ `ds4` + `ds4-server` built |
| `make cpu` (CPU portability) | ✅ rc=0 |
| `ds4_test --server` | ✅ OK |
| `ds4_test --streaming-decode-prefill-correctness` | ✅ **OK** — `cand==ref`, rms=0, `cold_warm_neq=0` (**matches `main`**) |
| `ds4_test --metal-ssd-streaming-cache-pressure` | ✅ OK (was in the 65 failures; now passes) |
| imatrix collection (`ds4 --imatrix-dataset … --imatrix-out`) | ✅ valid 430 MB `.dat`, 387k routed observations |
| expert prune mask | ✅ `expert prune mask ACTIVE (430 experts pruned)`, generation coherent |
| `ds4-bench` (speed, default path) | ✅ runs, sane t/s (ctx2048: prefill 153.7, gen 6.57). No regression expected: the decode path is now upstream's (the streaming test gives **bit-identical logits to `main`**), and the fork features are off by default. |

**imatrix speed:** exempt from the no-speed-regression rule by project decision (on-edge imatrix
is inherently slower when active; default path unaffected).

> Still recommended before landing on `main`: `make test` (full upstream suite) and, if you want a
> formal speed number, a `ds4-bench` before/after sweep vs `main`. The smoke + streaming-equivalence
> tests above already establish the default path is unchanged.

### To reproduce

```sh
git checkout -b sync-fix main
git merge upstream/main
git checkout --theirs ds4.c && git add ds4.c && git commit      # take upstream streaming
# re-apply only the non-streaming fork hunks (profile + prune + imatrix), NOT streaming:
git diff 8384adf main -- ds4.c | <keep non-stream hunks> | git apply --3way
make && DS4_TEST_SSD_STREAMING=1 ./ds4_test --streaming-decode-prefill-correctness   # must == main
```
