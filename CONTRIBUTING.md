# Contributing

Hebrus changes should be tested against the failure mode they can realistically
affect. The project has two regression tracks: correctness and speed. Please
include the commands you ran, the machine/backend, the model quant, and any
notable failures in the PR or commit notes.

## Co-development with `antirez/ds4`

Hebrus is an increasingly independent inference engine that began as a
transparent research fork of
[`antirez/ds4`](https://github.com/antirez/ds4). It retains substantial upstream
implementation and history while focusing on Apple Metal, embedded ExpertMajor
storage, and SSD streaming across several model families. It is not a
replacement for the upstream project. General improvements that remain
applicable to upstream-supported paths continue to be contributed back after
validation.

Every change applicable to an upstream-supported path **must be opened as an
upstream PR** once it is scoped and validated. This includes model- or
hardware-specific work when the same path exists upstream. Continuing research
in the fork while that PR is reviewed is expected.

Every contribution must be classified before it is promoted:

| Classification | Required action |
| --- | --- |
| General, reproducible, and safe for the affected backends | Open an upstream PR once correctness and performance evidence is complete |
| Potentially general, but not yet proven across the affected paths | Keep it isolated and clearly experimental while collecting the missing evidence |
| Model-, quant-, or hardware-specific research | Keep it labelled while incomplete; if it applies to an upstream-supported path, open an upstream PR after validation |
| Equivalent change already exists upstream | Adopt the upstream implementation and remove the redundant fork delta |
| Measured regression without a necessary correctness fix | Do not promote it |

Before opening a fork PR:

- check current upstream commits, PRs, and issues so the work is not duplicated;
- state the upstreamability classification in the PR description;
- include the exact test commands, hardware/backend, model and quantization,
  before/after numbers, and correctness evidence;
- link the upstream PR or issue when one exists, or explain concretely why the
  change remains fork-only;
- update [`FORK_NOTES.md`](FORK_NOTES.md) if the fork/upstream boundary changes.

An upstream review may continue while follow-up research proceeds here. If
upstream later lands an equivalent or better implementation, this fork should
converge on it rather than preserve a competing version.

Do not send PRs affecting one or more inference backends without checking if the
resulting code is still correct and fast. The only acceptable speed regression
is when an important correctness bug is fixed and it requires some speed penalty.

## Correctness Regression Tests

Build the default backend first:

```sh
make clean
make
```

The C test runner is `ds4_test`. Running it without arguments is equivalent to
`--all`:

```sh
make test
```

Useful narrower checks:

```sh
./ds4_test --server
./ds4_test --logprob-vectors
./ds4_test --long-context
./ds4_test --tool-call-quality
./ds4_test --metal-kernels
```

What they cover:

- `--server`: request parsing, chat rendering, streaming, tool-call parsing,
  thinking controls, KV disk-cache bookkeeping, and other server-side logic.
  This is the best quick check for API and prompt-rendering changes.
- `--logprob-vectors`: compares local token bytes and top-logprob slices against
  official DeepSeek V4 Flash continuation vectors. This catches tokenizer,
  template, attention, and logits regressions.
- `--long-context`: runs a long-context story fact-recall regression from
  `tests/long_context_story_prompt.txt`. The model must retrieve spelled-out
  person-number assignments from a long prose prompt and return `Name=number`
  lines that the test parses.
- `--tool-call-quality`: exercises actual model behavior for DSML tool-call
  emission in both fast and exact paths.
- `--metal-kernels`: isolated Metal kernel numeric checks.

The runner retains `ds4flash.gguf` only as a historical local convenience.
Every model-backed regression or release gate must override it with the exact
qualified ExpertMajor v2 artifact:

```sh
DS4_TEST_MODEL=/path/to/model.gguf ./ds4_test --logprob-vectors
DS4_TEST_VECTOR_FILE=/path/to/official.vec ./ds4_test --logprob-vectors
DS4_TEST_LONG_PROMPT=/path/to/prompt.txt ./ds4_test --long-context
```

### Artifact publication boundary

Runtime qualification does not make a local artifact publicly downloadable.
A runtime download target may be enabled only after an immutable repository
revision records the exact filename, byte count, complete output SHA-256,
manifest contract, and compatible runtime commit. Until then, contributor QA
may use an explicitly labelled local candidate, but release and download checks
must report public distribution as unavailable.

The machine-readable
[Qwen release contract](docs/contracts/qwen-release.json) is the canonical
source for every repeated identity below. The current Qwen release is
`published` as
`Qwen3.6-35B-A3B-Hebrus-ExpertMajor-v2-MLX-Affine4-G64.gguf`, 20,808,566,880
bytes, SHA-256
`dd17266185833a9f05531ce366fd7284ddca1ed64aa3dcf06e321e8c72c9ea3d`.
It is published at immutable repository revision
`e002665becd2db618897effb213030fdf92e7e98`, and
its manifest records artifact-format compatibility floor
`73a332fef82a0bcdd567d17e0de17aa004cad85d`. That field proves the runtime can
read the store; it does not supersede later hardware-safety policy. The release
runtime must also contain every fix required by `RUNTIME_SUPPORT.md`.
`download_model.sh qwen-v2` must pin and validate that exact identity. The older
`Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-Q4_K_S.gguf` object is `negative-only`, not
a runnable fallback. See
[`docs/qwen-expert-major-store.md`](docs/qwen-expert-major-store.md).

The additive Q2_K_XL artifact is `published-beta` and must remain opt-in through
`download_model.sh qwen-q2-beta`; it never replaces `qwen-v2`. Its exact
identity is
`Qwen3.6-35B-A3B-Hebrus-ExpertMajor-v2-Q2_K_XL.gguf`, 12,290,632,032 bytes,
SHA-256
`30c22f70aff0f05986b517ee4ad8fef554a1b5aab6971c9ca09f999566d30143`,
embedded payload SHA-256
`ccc3fbc2405d1dd73f8ac15741b0277514de4f46b80818531297ea9ffa0c6a3c`.
It is pinned at immutable revision
`e002665becd2db618897effb213030fdf92e7e98` with minimum compatible runtime
commit `42e2fec2a7dbb14a42e7a5612dfec00e33d443ca`. A Beta publication may expose
only its measured boundary: minimum 64 GiB, 32768 qualified context tokens,
nonrecommended, and explicitly not full-window qualified. The near-262K lane
remains blocking before Stable/full-window promotion.

Change the JSON contract first, update every intentional human-readable mirror,
then run `make release-contract release-contract-test`; do not update a mirror
as an independent release authority.

CUDA and ROCm are frozen and their backend source and build targets are absent
from the active tree. Ordinary changes must not restore them accidentally. A
change that intentionally reactivates either backend must restore and pass the
complete former backend lane before promotion: clean build on its designated
hardware, model-free/backend regression suite, long-context and synthetic
kernel coverage, ExpertMajor v2 admission or fail-closed checks, and a Metal
cross-check proving that shared warning or runtime changes did not regress the
production path. It also requires an accepted ADR, an updated runtime support
contract, an owner, and current performance evidence. Do not infer backend
support from compilation alone.

For CPU portability, at least verify that the CPU target still builds:

```sh
make cpu
```

On macOS, also run `make build-isolation-test`. The CPU-only binaries live in
`build/cpu-$(uname -m)/bin`; `make cpu` intentionally leaves the root Metal
commands unchanged.

The GitHub Actions workflow runs the full Linux CPU/model-free premerge gate
and a macOS arm64 lane that repeats the CPU suite, builds the Metal command
surface, and checks Metal/CPU artifact isolation. Hosted compilation is not
Metal kernel or qualified-model evidence; the hardware release gates below
remain mandatory.

The CPU backend is a reference/debug path, not the production performance
target. Remember that executing the CPU path on Metal can crash the system
because of a kernel bug in macOS.

## Quality Checks For Quantization Changes

For GGUF or quantization work, use the official-continuation scorer in
`gguf-tools/quality-testing`. The test compares how much probability a local
GGUF assigns to official DeepSeek V4 Flash continuations, token by token.

Build the scorer:

```sh
make -C gguf-tools quality-score
```

Then score old and new GGUFs against the same manifest and compare:

```sh
gguf-tools/quality-testing/score_official OLD.gguf \
  gguf-tools/quality-testing/data/manifest.tsv /tmp/old.tsv 4096

gguf-tools/quality-testing/score_official NEW.gguf \
  gguf-tools/quality-testing/data/manifest.tsv /tmp/new.tsv 4096

python3 gguf-tools/quality-testing/compare_scores.py /tmp/old.tsv /tmp/new.tsv
```

Lower `avg_nll` is better. See
`gguf-tools/quality-testing/README.md` for collecting or refreshing official
continuations.

## Speed Regression Tests

Use `ds4-bench` for throughput regressions. It reports instantaneous prefill and
generation speed at context frontiers, not one whole-run average. Prefill is
incremental: each row measures only the newly processed suffix since the
previous frontier.

### Product relevance gate

Pass this gate before implementing, live-tuning, or retaining an inference
performance candidate. First record a product-target declaration in the active
task plan or handoff. Promote the final declaration and decision to the dated
benchmark record with the completed or rejected experiment. The declaration
must identify:

- the exact qualified model, artifact, and quantization profile; a new support
  candidate additionally requires explicit authorization, its accepted ADR,
  and the proposed runtime-contract change;
- the physical Mac/chip and unified-memory tier, OS/power state, prompt and
  context frontier, representative decode horizon, and the documented or
  minimum admitted context allocation needed for that workload;
- the normal `AUTO` startup command and the mode/cache plan that clean `main`
  resolves under normal pressure on that target;
- a measured end-to-end bottleneck or bounded capacity/safety constraint in
  that normal path, not only a hit rate, byte counter, kernel time, or
  forced-mode observation;
- the candidate's conservative worst-case incremental resource bound and the
  immediate rejection conditions.

Here, normal `AUTO` means the documented product command with the required
model, prompt/request, and context arguments, but no residency, cache, reserve,
pressure, or tuning override. The context allocation must be the documented
default or the minimum admitted allocation that fits the recorded prompt,
decode horizon, and bookkeeping. Oversizing it, inflating an external reserve,
adding background load, or inducing pressure to make `AUTO` select another
mode invalidates the target.

A mode is an optimization target only on a qualified physical lane, or on an
explicitly authorized support-candidate physical lane following the ADR,
runtime-contract, and physical-qualification process above. Clean `main` normal
`AUTO` must select it for the representative workload. If a newly authorized
artifact cannot be admitted by `main`, record the closest accepted runnable
product baseline without manufacturing a cross-artifact delta, and use the
support lane's approved qualification plan to establish a condition-matched
comparator. Withhold positive support and performance claims until that
physical qualification completes. If `AUTO` admits resident execution with
every required reserve intact, forced SSD on that lane is correctness and
non-regression evidence only: do not implement or promote an SSD optimization
for it. SSD performance work requires a physical hardware/context lane where
normal `AUTO` naturally selects SSD, including a family whose production
contract is SSD-only.

The narrow exception is a candidate whose declared purpose is to change the
normal `AUTO` resolution itself, such as removing enough resident memory to
make resident execution safely admissible. Treat that as a residency/admission
policy change: compare clean-main and candidate normal `AUTO` on the same
physical target, account for both plans, and complete every applicable
admission, hardware-policy, support-contract, and release gate. A forced mode
cannot establish the transition. A candidate may never create its own
justification by adding enough footprint or pressure to make `AUTO` fall from
resident to SSD.

Explicit `--resident` or `--ssd-streaming`, cache-size overrides, tuning
environment variables, simulated RAM or external reserves, fault injection,
induced pressure, and manually constrained larger hosts create diagnostic
control lanes. They may validate arithmetic, policy mechanics, correctness,
fail-closed behavior, or path isolation after a real target is named. They may
not define the target, supply its positive result, reproduce its unified-memory
pressure/swap behavior, or be generalized to another physical hardware or
mode. If the target hardware is unavailable, restrict work to offline or
model-free hypothesis testing; do not produce a live promotion candidate or
hardware performance claim.

Before behavior code, account for baseline and candidate worst-case bytes by
owner and lifetime: model/context state, Metal private and shared allocations,
authoritative cache, staging or speculative storage, locked/wired memory, and
pageable mappings. Use exact artifact geometry where known and otherwise a
conservative proved upper bound; capture exact executed telemetry before
promotion. Slot counts or logical bytes alone are insufficient. The candidate
must fit the target `AUTO` plan's production admission and cache caps while
preserving every context, runtime, pressure, and safety reserve. Exceeding
remaining admitted headroom, exceeding the current guarded cache cap, assuming
reclaim beyond the production planner's credited budget, relying on swap, or
fitting only on a larger host is an immediate veto for an ordinary
optimization. A proposed cap change is a separate hardware-policy/support
change and must pass its applicable ADR, admission, physical-safety, and
release gates before it can support a performance claim.

For a speed claim, the positive result must be exact end-to-end evidence from
the target's normal `AUTO` path. Report full-request wall time, prefill/TTFT,
decode throughput and TPOT, memory pressure, swap, and the representative
break-even horizon. A decode-only gain that loses the representative request,
or lower SSD bytes or higher cache hit rate alone, is not a speed win. A
resource/capacity candidate with no resolved wall-time improvement must instead
demonstrate a user-level admission, pressure, or bounded-capacity improvement
and be reported as such, not as a speedup. Required resident, SSD, and
cross-model control lanes still protect shared paths, but a control cannot
supply the win for a different target. If no product-relevant lane passes this
gate, stop and reject the idea before implementation; retain only a concise
decision record when it prevents repetition.

### Performance acceptance matrix

Every inference-performance change must be evaluated at short, medium, large,
and long prompt context before promotion. This includes kernels, graph fusion,
cache or residency policy, SSD loading, adaptivity, and changes advertised as
prefill-only or decode-only. A microbenchmark or a short-context run may guide
an experiment and may reject it immediately for a correctness failure, unsafe
memory behavior, or an unambiguous regression. It cannot promote a change or
generalize a speed claim on its own.

The table defines workload frontiers, not target hardware. Run these minimum
frontiers on the physical target selected by the Product relevance gate unless
the model's record defines a larger model-specific lane. A qualified 64 GiB
host supplies positive promotion evidence only for a normal `AUTO` lane it
naturally represents; otherwise it is an additional correctness or
non-regression control, not a substitute for the physical target:

| Tier | Prompt tokens | Greedy decode tokens | Role |
| --- | ---: | ---: | --- |
| Short | 128 | 128 or more | Smoke, correctness, safety, and low-context cost; may reject but never promote or generalize |
| Medium | 2,048 | 128 or more | Normal interactive workload |
| Large | 8,192 | 128 or more | Larger working-set and routing behavior; still not the long-context gate |
| Long | 32,768 | 128 or more | Mandatory promotion lane for every surviving performance candidate |

Changes to attention, KV layout or compression, cache policy, RoPE, context
allocation, or context scaling must additionally run isolated 65,536- and
100,000-token frontiers with at least 128 greedy decode tokens. Use a context
allocation strictly larger than the frontier plus generated tokens; 131,072 is
the standard allocation for the 65,536- and 100,000-token lanes.

Long context is a primary product workload, not an edge case. The 32,768-token
row is the minimum merge screen, not the model-window qualification endpoint.
Full-window publication and release qualification additionally require an
isolated endpoint arm at the largest admitted prompt frontier that leaves room
for at least 128 greedy decode tokens and runtime bookkeeping. Derive that
limit from the locally validated artifact metadata and the runtime's admission
result, not from a model card. Record the declared context length, the exact
prompt frontier, context allocation, generated-token count, resolved
hardware/mode plan, peak memory pressure, and swapout delta.

The currently validated Qwen3.6 metadata declares a 262,144-token context, so
full-window qualification requires a near-262K endpoint arm in addition to the
standard matrix and the 65,536/100,000-token diagnostic arms where applicable.
An additive profile may merge or be distributed as an explicitly labelled,
nonrecommended Beta while that arm is pending only if it does not replace the
Stable artifact/downloader, pins an immutable exact identity, advertises no
context beyond its completed long-context lane, and records the endpoint as a
blocking Stable/full-window gate. If a qualified hardware or residency profile
cannot safely complete the context endpoint it advertises, either fix the
runtime or narrow that profile's public context contract and fail larger
requests closed. Never silently substitute 32K/100K evidence for the release
endpoint.

Routing, expert-cache, residency, and expert-I/O changes must also complete one
second 32,768-token diagnostic lane from a different prompt domain. Use
`speed-bench/promessi_sposi.txt` for the prose/locality lane and
`tests/long_context_security_prompt.txt` for the security/coding lane. The
primary promotion prompt still uses A/B/B/A; one final-stack arm is sufficient
for the second prompt unless a prompt-specific speed claim is made. Never
compare or average throughput across the two prompts: record both complete
prompt hashes and interpret the different routing working sets separately.

For context-sensitive inference work, begin throughput exploration at the
8,192-token Large tier so tuning is driven by a credible working set. Use the
Short tier as a secondary correctness, safety, and low-context-cost check.
Stop early and reject the candidate if any measured tier finds wrong output,
unsafe pressure or swap, a crash, or a clear regression. Every candidate that
survives and is proposed for promotion must still complete all four mandatory
tiers; an early rejection does not turn an incomplete matrix into positive or
general performance evidence.

Do not average the tiers into one score. Report prefill throughput, decode
throughput, decode TPOT p50 and p95, and correctness evidence separately at each
frontier. A long-context win must remain visible even if a short smoke is
neutral; a short win must never hide a medium-, long-, or endpoint-context
regression.

Performance comparisons must also follow these rules:

- Use an interleaved A/B/B/A order for the baseline and candidate. The two
  control arms must remain within 3% in the target metric, but that ceiling is
  not an acceptance threshold: the candidate effect must also exceed the
  absolute control drift, the within-arm spread, and known measurement
  resolution. Otherwise the result is inconclusive. Cool or stabilize the
  machine and rerun the complete cohort.
- Keep the hardware, OS, power state, model bytes, prompt prefix, sampling,
  context allocation, generated-token count, and every non-target setting
  identical. Normally the backend mode, cache budget, and resolved plan must
  also match. For a predeclared normal-`AUTO` plan-transition candidate, keep
  the product command identical, require one stable plan across both A arms and
  one stable intended plan across both B arms, and record the A-to-B transition
  as the measured target rather than contamination. Any undeclared plan change,
  abort, new swapout, or competing inference process invalidates the entire
  cohort, including already completed arms. Recover or stabilize the host,
  correct the unsafe policy when applicable, then restart from the first A arm;
  do not retain a convenient retry.
- Inspect application descendants, not only their visible parent processes,
  when isolating a benchmark host. Suspending an Electron application parent
  does not necessarily suspend its renderer or Node helpers. Record a
  process/load snapshot before the cohort, and treat a helper consuming
  material CPU/GPU or two byte-identical arms drifting beyond the control
  ceiling as host contamination. After a sustained long-context arm, require
  an idle cooldown before the next comparable arm unless active thermal
  control and its restoration are both recorded.
- Make acceptance environments hermetic. The bounded M5 runner rejects
  inherited `DS4_*` runtime flags outside its own controls and records every
  admitted `DS4_*` variable. The exact `DS4_QWEN_TELEMETRY_JSONL` sink is the
  only optional runtime-output exception: it must be the fresh dedicated
  `$DS4_M5_PREFIX.qwen-telemetry.jsonl` path, is retained and hashed with the
  arm, and must pass JSONL, terminal `runtime_close`, and runtime-failure-marker
  validation. Use `DS4_M5_EXPLORATORY=1` for an intentional flag experiment;
  exploratory evidence cannot promote a default.
- Record cold and warm/page-cache cohorts separately and never average them
  together. A retained warm cohort uses the same discarded warm-up for each arm
  before A/B/B/A. A cold or first-run observation uses fresh processes and is
  labelled separately. `--ssd-streaming-cold` describes expert-preload policy,
  not a guaranteed cold macOS file cache. Record memory pressure and swap
  before, during, and after every retained arm.
- Record the repository HEAD, dirty-diff SHA-256 when applicable, `--build-info`,
  executable SHA-256, qualified GGUF identity, and the SHA-256 of the complete
  runtime Metal source set. Before cache-state preparation, the bounded M5
  runner must compute the complete GGUF SHA-256 once and bind its evidence to
  the resolved path, device, inode, byte count, and modification time. Every
  arm verifies, copies, and hashes that evidence without rereading the GGUF;
  merely recording an expected digest is invalid. Retain the runtime
  `ds4: metal_library` identity
  line, which fingerprints the assembled source including diagnostic source
  overrides and records compile-mode macros. Metal sources are compiled at
  runtime, so a frozen executable run from a different source checkout is not a
  valid frozen arm.
- Compare `--dump-decode-evidence-dir` outputs at every frontier. Unexplained
  token or final-logit drift is a correctness failure, regardless of speed.
- Evaluate adaptive choices per hardware/context lane. Only a product-relevant
  normal-`AUTO` target lane may supply positive selection evidence. Forced-mode
  and simulated-hardware controls cannot. Preserve an exact fallback for the
  other lanes until the default policy is proven there.
- After the Product relevance gate passes, a candidate that provably removes
  at least 40% of a bounded runtime resource that is a real capacity or safety
  cost in the target `AUTO` lane
  (for example transferred bytes, storage reads, allocations, syscalls, or
  encoder creation) may be promoted even when its throughput effect is within
  measurement noise.  This resource-efficiency rule applies only when output
  remains exact, no safety or swap regression appears, and every retained
  end-to-end inference throughput and latency metric (prefill/decode
  throughput, phase wall, TTFT, and TPOT) regresses by less than 2.0% in each
  qualified lane.  Resource-specific timing remains mandatory telemetry but is
  not itself this end-to-end threshold unless it is the optimization target.
  When wall time does not improve, the reduction must also demonstrate a
  user-level admission, pressure, or bounded-capacity benefit in the target
  workload and must be reported as a resource/capacity result, not a speedup.
  The rule does not waive the context matrix, make an `AUTO`-unselected forced
  path product-relevant, or permit averaging tiers to hide a larger regression;
  record both the structural/resource reduction and the complete per-tier
  performance table.
- When exact outputs and every qualified end-to-end metric are at parity or
  better, use demonstrated long-term scalability as the tie-breaker. Prefer the
  implementation with lower asymptotic work, memory traffic, dispatch count,
  synchronization, or bounded resource use even if the present host measures
  the wall-clock effect inside noise. The reduction must be measured or derived
  from the executed geometry, and the result must replace the old production
  path rather than add a permanent alternate kernel or flag. Extra complexity
  justified only by a hoped-for future gain is not an optimization.
- Measure optimizations incrementally and then measure the final combined
  stack against the original baseline. Interactions can create or erase a win;
  do not add percentages from isolated experiments to predict the stack.
- Keep only the winning production path. Revert rejected code, flags, and test
  scaffolding, while retaining a concise dated decision record under
  `docs/benchmarks/` when it prevents the same failed experiment being repeated.

For shared Metal graph, prefill, decode, cache, or tensor changes, run this
matrix on qualified DeepSeek, GLM, and Qwen artifacts before merge. A
model-specific change runs the matrix on every affected qualified normal-`AUTO`
hardware/context lane, plus every explicit mode required as a correctness or
non-regression control. An authorized new support lane completes its separate
qualification gate. Control lanes cannot compensate for a loss or supply the
positive win in the target lane.

Run acceptance frontiers in separate processes so every row measures the full
prompt from an equivalent initial state. For example:

```sh
for frontier in 128 2048 8192 32768; do
  ./ds4-bench \
    -m /absolute/path/to/QUALIFIED-DEEPSEEK-FLASH-DS4-ExpertMajor-v2.gguf \
    --prompt-file speed-bench/promessi_sposi.txt \
    --ctx-start "$frontier" \
    --ctx-max "$frontier" \
    --ctx-alloc 65536 \
    --gen-tokens 128 \
    --csv "/tmp/ds4-speed-$frontier.csv" \
    --dump-decode-evidence-dir "/tmp/ds4-speed-$frontier-evidence"
done
```

A multi-frontier process remains useful for exploratory curves, but later rows
process only the suffix since the previous frontier and inherit the same
session/cache history. Do not compare those rows with isolated full-prefill
arms or use them as the final acceptance matrix.

Capture the artifact identities beside each retained arm. On macOS, one
reproducible command set is:

```sh
git rev-parse HEAD
git diff --binary --no-ext-diff | shasum -a 256
shasum -a 256 ./ds4-bench /absolute/path/to/QUALIFIED-MODEL.gguf
find metal -type f -name '*.metal' -print0 | \
  xargs -0 shasum -a 256 | LC_ALL=C sort | shasum -a 256
```

Use the same machine, backend, model file, context sweep, power/thermal state,
and background load when comparing two commits. For backend work, run at least
one interleaved baseline/candidate sweep and compare `prefill_tps`, `gen_tps`,
`gen_tpot_p50_ms`, and `gen_tpot_p95_ms`. Generation is greedy and skips EOS so
each frontier gets the same number of generated tokens. The acceptance matrix
above, including the long-context frontier, is mandatory even when the expected
effect is confined to one phase.

To generate a graph for a CSV:

```sh
python3 speed-bench/plot_speed.py /tmp/ds4-speed.csv --title "Machine t/s"
```

## Reporting sessions bugs

For debugging a failing generation, keep the trace:

```sh
./ds4-server --trace /tmp/ds4-trace.txt ...
```
