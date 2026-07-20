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

Default linear sweep:

```sh
./ds4-bench \
  -m /absolute/path/to/QUALIFIED-DEEPSEEK-FLASH-DS4-ExpertMajor-v2.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128 \
  --csv /tmp/ds4-speed.csv
```

Use the same machine, backend, model file, context sweep, power/thermal state,
and background load when comparing two commits. For backend work, run at least
one before/after CSV and compare both `prefill_tps` and `gen_tps`. Generation is
greedy and skips EOS so each frontier gets the same number of generated tokens.

To generate a graph for a CSV:

```sh
python3 speed-bench/plot_speed.py /tmp/ds4-speed.csv --title "Machine t/s"
```

## Reporting sessions bugs

For debugging a failing generation, keep the trace:

```sh
./ds4-server --trace /tmp/ds4-trace.txt ...
```
