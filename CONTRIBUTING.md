# Contributing

DwarfStar4 changes should be tested against the failure mode they can realistically
affect. The project has two regression tracks: correctness and speed. Please
include the commands you ran, the machine/backend, the model quant, and any
notable failures in the PR or commit notes.

## Co-development with `antirez/ds4`

This repository is a transparent research fork of
[`antirez/ds4`](https://github.com/antirez/ds4), not a replacement for it. The
goal is to co-develop DwarfStar: use the fork to investigate complementary
hardware and model paths without blocking on review latency, then contribute
general improvements back upstream.

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

The current Qwen release is
`Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-MLX-Affine4-G64.gguf`, 20,808,566,880
bytes, SHA-256
`dd17266185833a9f05531ce366fd7284ddca1ed64aa3dcf06e321e8c72c9ea3d`.
It is published at immutable repository revision
`7bf9c3f7f6136aeb2599d75ee61c0cc2f18e2b02`, and
`download_model.sh qwen-v2` must pin that exact identity. The older Q4_K_S
object is an incompatible negative fixture, not a runnable fallback. See
[`docs/qwen-expert-major-store.md`](docs/qwen-expert-major-store.md).

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

### Performance acceptance matrix

Every inference-performance change must be evaluated at short, medium, large,
and long prompt context before promotion. This includes kernels, graph fusion,
cache or residency policy, SSD loading, adaptivity, and changes advertised as
prefill-only or decode-only. A microbenchmark or a short-context run may guide
an experiment and may reject it immediately for a correctness failure, unsafe
memory behavior, or an unambiguous regression. It cannot promote a change or
generalize a speed claim on its own.

Use these minimum frontiers on the qualified 64 GiB Metal host unless a model's
record defines a larger model-specific lane:

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

Routing, expert-cache, residency, and expert-I/O changes must also complete one
second 32,768-token diagnostic lane from a different prompt domain. Use
`speed-bench/promessi_sposi.txt` for the prose/locality lane and
`tests/long_context_security_prompt.txt` for the security/coding lane. The
primary promotion prompt still uses A/B/B/A; one final-stack arm is sufficient
for the second prompt unless a prompt-specific speed claim is made. Never
compare or average throughput across the two prompts: record both complete
prompt hashes and interpret the different routing working sets separately.

Run the tiers incrementally. Stop early and reject the candidate if a tier finds
wrong output, unsafe pressure or swap, a crash, or a clear regression. Every
candidate that survives and is proposed for promotion must complete all four
mandatory tiers; an early rejection does not turn an incomplete matrix into
positive or general performance evidence.

Do not average the tiers into one score. Report prefill throughput, decode
throughput, decode TPOT p50 and p95, and correctness evidence separately at each
frontier. A long-context win must remain visible even if a short smoke is
neutral; a short win must never hide a medium- or long-context regression.

Performance comparisons must also follow these rules:

- Use an interleaved A/B/B/A order for the baseline and candidate. The two
  control arms must remain within 3% in the target metric, but that ceiling is
  not an acceptance threshold: the candidate effect must also exceed the
  absolute control drift, the within-arm spread, and known measurement
  resolution. Otherwise the result is inconclusive. Cool or stabilize the
  machine and rerun the complete cohort.
- Keep the hardware, OS, power state, model bytes, prompt prefix, sampling,
  context allocation, generated-token count, backend mode, cache budget, and
  other non-target settings identical. Record the resolved adaptive plan. An
  abort, any new swapout, a different resolved plan, or a competing inference
  process invalidates the entire cohort, including already completed arms.
  Recover or stabilize the host, correct the unsafe policy when applicable,
  then restart from the first A arm; do not retain a convenient retry.
- Make acceptance environments hermetic. The bounded M5 runner rejects
  inherited `DS4_*` runtime flags outside its own controls and records every
  admitted `DS4_*` variable. Use `DS4_M5_EXPLORATORY=1` for an intentional
  flag experiment; exploratory evidence cannot promote a default.
- Record cold and warm/page-cache cohorts separately and never average them
  together. A retained warm cohort uses the same discarded warm-up for each arm
  before A/B/B/A. A cold or first-run observation uses fresh processes and is
  labelled separately. `--ssd-streaming-cold` describes expert-preload policy,
  not a guaranteed cold macOS file cache. Record memory pressure and swap
  before, during, and after every retained arm.
- Record the repository HEAD, dirty-diff SHA-256 when applicable, `--build-info`,
  executable SHA-256, qualified GGUF identity, and the SHA-256 of the complete
  runtime Metal source set. Retain the runtime `ds4: metal_library` identity
  line, which fingerprints the assembled source including diagnostic source
  overrides and records compile-mode macros. Metal sources are compiled at
  runtime, so a frozen executable run from a different source checkout is not a
  valid frozen arm.
- Compare `--dump-decode-evidence-dir` outputs at every frontier. Unexplained
  token or final-logit drift is a correctness failure, regardless of speed.
- Evaluate adaptive choices per hardware/context lane. A candidate may be
  selected only in lanes where it wins; preserve an exact fallback for the
  other lanes until the default policy is proven there.
- A candidate that provably removes at least 40% of a bounded runtime resource
  (for example transferred bytes, storage reads, allocations, syscalls, or
  encoder creation) may be promoted even when its throughput effect is within
  measurement noise.  This resource-efficiency rule applies only when output
  remains exact, no safety or swap regression appears, and every retained
  end-to-end inference throughput and latency metric (prefill/decode
  throughput, phase wall, TTFT, and TPOT) regresses by less than 2.0% in each
  qualified lane.  Resource-specific timing remains mandatory telemetry but is
  not itself this end-to-end threshold unless it is the optimization target.
  The rule does not waive the context matrix or permit averaging tiers to hide
  a larger regression; record both the structural/resource reduction and the
  complete per-tier performance table.
- Measure optimizations incrementally and then measure the final combined
  stack against the original baseline. Interactions can create or erase a win;
  do not add percentages from isolated experiments to predict the stack.
- Keep only the winning production path. Revert rejected code, flags, and test
  scaffolding, while retaining a concise dated decision record under
  `docs/benchmarks/` when it prevents the same failed experiment being repeated.

For shared Metal graph, prefill, decode, cache, or tensor changes, run this
matrix on qualified DeepSeek, GLM, and Qwen artifacts before merge. A
model-specific change runs the matrix on every mode and machine class that can
select the changed path.

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
