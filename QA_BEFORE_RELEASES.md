# QA Before Releases

This is the release gate for DwarfStar.  Run it before tagging or pushing a
release build.  The goal is not to prove every code path exhaustively; it is to
exercise the paths that have historically regressed: Metal graph inference,
model-family residency, SSD streaming, distributed rejection, disk KV cache,
server APIs, and the agent TUI/tool state machine. CUDA and ROCm are frozen and
absent; sections 6 and 7 prevent accidental restoration and preserve their full
reactivation gates.

Do not run multiple huge model processes at the same time.  Record the commit,
hardware, GGUF file, context size, and any non-default flags for every manual
run.

DeepSeek V4, GLM 5.2, and Qwen3.6 release inference is ExpertMajor v2-only on
local Apple Metal. Canonical files may be inspected or converted offline, but
ExpertMajor v1, sidecars, CPU, and distributed inference must fail closed. CUDA
and ROCm source/build targets must remain absent unless a reactivation release
passes the conditional gates below. No ExpertMajor or Qwen admission environment
flag belongs in a release startup command.

## Release Artifact Identity

Resolve these variables to absolute paths before any model-backed command. Never
point them at a canonical converter input, sidecar, v1 file, symlink with unknown
target, or an artifact whose complete output hash is missing.

| Variable | Required identity |
| --- | --- |
| `DEEPSEEK_V2` | `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-DS4-ExpertMajor-v2.gguf`; 86,720,114,272 bytes; SHA-256 `8378080263eb9224f7228d72e2afa4ac3cf74a116023fdec2c596ff228a33e3f` |
| `DEEPSEEK_MIXED_V2` | Non-applicable until a mixed-quant DeepSeek ExpertMajor v2 artifact has a publication record with exact filename, bytes, and complete output SHA-256; do not resolve or use this variable before qualification |
| `GLM_V2` | `GLM-5.2-DS4-ExpertMajor-v2-Q2_K.gguf`; 262,147,193,504 bytes; SHA-256 `7f5017e3076e706c78f2a5322b035a9e2f6519c65ff5b6be8b2d91aeff61505d` |
| `QWEN_V2` | `Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-MLX-Affine4-G64.gguf`; immutable repository revision `7bf9c3f7f6136aeb2599d75ee61c0cc2f18e2b02`; 20,808,566,880 bytes; SHA-256 `dd17266185833a9f05531ce366fd7284ddca1ed64aa3dcf06e321e8c72c9ea3d`; MLX affine4/group-64 routed storage |
| `QWEN_RETIRED_Q4_NEGATIVE` | Rejection-only input: `Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-Q4_K_S.gguf`; 20,808,566,880 bytes; SHA-256 `d7c43a6388ec20e6fe5530850350f96fdb0ac37c5ce36d3e5f92b172c447f56b`; it must fail before inference |

Record the test machine by hardware model, unified memory, OS build, and power
state in the release evidence. Do not encode local hostnames, addresses, or
network routes in this checklist.

The Qwen runtime and downloader paths must both resolve the exact `QWEN_V2`
identity above. The immutable revision, filename, byte count, complete output
SHA-256, manifest contract, and compatible runtime commit are one release gate.

## 1. Repository And Build Sanity

- Start from a clean tree except intentional release notes:
  `git status --short`.
- Build the normal local target:
  `make clean && make`.
- Prove macOS Metal/CPU artifact isolation:
  `make build-isolation-test`.
- Build CPU-only binaries as a compile check. On macOS they remain under
  `build/cpu-$(uname -m)/bin` and must not replace the root Metal commands:
  `make cpu`.
- Record `./ds4 --build-info` and the CPU binary's `--build-info` output.
- Run `--capabilities=json` on all five Metal executables and the CPU profile;
  validate schema version 1, executable roles, backend identity, model-family
  claims, and ExpertMajor wire values with `make capabilities-test` and
  `python3 tests/test_capabilities.py --bin-dir build/cpu-$(uname -m)/bin --backend cpu`.
- Run whitespace checks before committing:
  `git diff --check`.
- Confirm `./ds4 --help`, `./ds4-server --help`, and `./ds4-agent --help` render
  cleanly, with readable section colors and no broken wrapping.

## 2. Core Regression Tests

- Run the default suite:
  `DS4_TEST_MODEL="$DEEPSEEK_V2" make test`.
- Without a local GGUF, run the complete model-free gate:
  `make model-free-test`.
- Run the vector checks explicitly after any tokenizer, template, KV, kernel,
  quantization, or prompt-rendering change:
  `DS4_TEST_MODEL="$DEEPSEEK_V2" ./ds4_test --logprob-vectors`
  and `DS4_TEST_MODEL="$DEEPSEEK_V2" ./ds4_test --local-golden-vectors`.
- Run server tests when HTTP, SSE, prompt rendering, cache policy, or tool-call
  replay changed:
  `./ds4_test --server`.
- Run `./ds4-eval --self-test-extractors`.

## Performance Promotion Gate

Apply the complete performance acceptance matrix in
[`CONTRIBUTING.md`](CONTRIBUTING.md#performance-acceptance-matrix) to every
release change intended to alter inference speed. Short-context results are
smoke evidence only: they may reject a candidate for correctness, safety, or a
clear regression, but they cannot promote it or generalize a speed claim. A
release performance claim is blocked until the affected qualified models and
modes have retained isolated 128-token short, 2,048-token medium, 8,192-token
large, and mandatory 32,768-token long-context arms, including exact decode
evidence, TPOT p50/p95, memory pressure, zero new swapout, and at least 128
greedy decode tokens per frontier. Attention, KV, cache, RoPE, allocation, and
context-scaling changes also require isolated 65,536- and 100,000-token arms.

- Use interleaved A/B/B/A comparisons and keep cold and warm cohorts separate.
  Use identical discarded warm-ups for retained warm cohorts. An abort, new
  swapout, changed resolved plan, or competing inference process invalidates the
  entire cohort; restart from its first arm after recovery or correction.
- For routing, expert-cache, residency, or expert-I/O changes, keep prose and
  security/coding 32K prompt lanes separate. Record both prompt hashes, never
  compare or average their throughput, and require a dedicated A/B/B/A before
  making a speed claim for either prompt class.
- Bind every arm to its executable, repository/diff, Metal runtime source, and
  GGUF hashes. A copied binary beside another checkout's `metal/*.metal` files
  is not the same benchmark artifact.
- Reject inherited `DS4_*` runtime flags in acceptance arms. Intentional flag
  experiments must be labelled exploratory and preserve their complete
  `DS4_*` environment artifact.
- Report tiers independently; do not use a cross-context average to hide a
  regression. The measured effect must exceed control drift, within-arm spread,
  and known measurement noise. If adaptivity selects different paths, invalidate
  the cohort and qualify each resolved hardware/context lane separately.
- For a stack of optimizations, retain both the incremental evidence and a final
  original-baseline versus complete-stack comparison. Only the measured stack,
  not the sum of isolated percentage gains, is a release result.
- A shared Metal graph, prefill, decode, cache, or tensor change requires the
  qualified DeepSeek, GLM, and Qwen short/medium/large/long matrices before
  merge.
- Record accepted and informative rejected results in a dated file under
  `docs/benchmarks/`; remove rejected implementation and experimental residue.

## 3. Metal Flash Path

Use the qualified DeepSeek Flash ExpertMajor v2 GGUF that users run. Set
`DEEPSEEK_V2` to its absolute path; a canonical download is not equivalent.

- One-shot CLI:
  `./ds4 -m "$DEEPSEEK_V2" --ctx 32768 --nothink -p "Explain C pointers in one paragraph."`
- Thinking and max-thinking prompts:
  run one short coding prompt with default thinking and one with max thinking.
- Long-context recall:
  run the long name/number or archive recall test used for catching attention
  and MoE routing drift.
- Logprob sanity:
  `./ds4 -m "$DEEPSEEK_V2" --nothink --temp 0 --dump-logprobs /tmp/ds4-logprobs.json --logprobs-top-k 20 -p "..."`
  and inspect that the continuation is sane.
- Speed sanity:
  run `ds4-bench -m "$DEEPSEEK_V2"` with
  `speed-bench/promessi_sposi.txt` and compare prefill, generation speed, and KV
  bytes with the last known good numbers for the same machine. This run must
  include the mandatory long-context frontier from the performance promotion
  gate; a short prompt does not sign off the lane.

## 4. Metal PRO Path

This checklist does not currently record a qualified PRO ExpertMajor v2
filename, complete output SHA-256, and model-backed baseline. Therefore PRO is
non-applicable for release inference: do not substitute a canonical or split
PRO file and do not execute an old PRO command.

- Verify canonical/split PRO and distributed requests fail closed.
- Build all current binaries and review changes touching PRO model shape,
  tensor lookup, routed expert mapping, template logic, and KV compatibility.
- Re-enable model-backed PRO QA only after its release record supplies an exact
  v2 artifact identity/hash and the runtime support contract admits it.

## 5. SSD Streaming

SSD streaming is a capacity path, so test both correctness and user experience.

- Flash q2/q2-q4 streaming:
  `./ds4 -m "$DEEPSEEK_V2" --ssd-streaming --ssd-streaming-cache-experts 32GB -p "..."`
- Mixed-quant Flash SSD streaming is currently non-applicable because no
  qualified mixed-quant ExpertMajor v2 artifact identity or model-backed
  baseline is recorded. Do not substitute a canonical, v1, sidecar, or
  unpublished file. Restore this regression lane only after its publication
  record supplies the exact filename, bytes, complete output SHA-256, prompt,
  and expected selected-address result.
- Cold streaming measurement:
  run once with `--ssd-streaming-cold` and verify no deadlock, missing expert,
  or impossible slowdown.
- Confirm startup reports cache budget and that generation does not stall on
  repeated expert misses for a small interactive prompt.
- If streaming cache internals changed, test the same prompt twice and compare
  first-token/logprob sanity between runs.

### GLM 5.2 Metal SSD lane

Use the verified `GLM_V2` identity above and start it with AUTO and the qualified
context. Do not add an explicit residency, cache, preload, or ExpertMajor flag.

- Verify AUTO resolves to local Metal SSD streaming and selects the qualified
  GLM Gold cache/prefill/decode policy.
- Run `./ds4 -m "$GLM_V2" --ctx 8192` for the normal flag-free AUTO smoke.
- Verify an explicit resident request fails closed; more host memory does not
  turn resident GLM into a qualified path.
- Run the deterministic GLM prompt and greedy continuation recorded in
  `docs/benchmarks/2026-07-20-glm52-expert-major-v2.md` and compare prefill,
  decode, output bytes, expert-read behavior, memory pressure, and swap with its
  same-condition gold evidence.
- Exercise both the indexed long-prefill path and multi-token decode. Confirm
  the runtime does not probe canonical component views or the retired full-layer
  decode resolver.
- Reject canonical GLM, sidecars, old ExpertMajor revisions, CPU, and
  distributed execution before inference.

### Qwen3.6 Metal lane

The Qwen path on `main` follows the same repository, build, core-test, and
regression rules as the other model paths. Use the verified normalized
`QWEN_V2` above and run the relevant model-backed smoke;
canonical, v1, sidecar, and community GGUFs are not equivalent inputs.

- Run `make model-free-test` and `./ds4_test --metal-kernels`. The latter must
  retain resident/SSD top-8 output equivalence, zero resident cache/`pread`
  accounting, malformed-route fail-closed behavior, and slab-growth checks.
- Run `./ds4 -m "$QWEN_V2" --ctx 8192` for the normal flag-free
  AUTO smoke.
- Run AUTO with the normal flag-free startup command; record both admission plans,
  their point-in-time inputs, resolved mode, cache tier, configured 321-expert
  slab target, cache `buffer_allocs`, task physical footprint, and system swap
  before/during/after. Exact slab count/capacity is asserted by the Metal
  kernel test; it is not a public runtime counter.
- Never bypass a failed resident admission to obtain a benchmark. On a host
  where both checks pass, compare the same deterministic prompt and logits in
  model-backed resident and forced-SSD modes.
- In SSD mode, verify the first route allocates one 321-expert slab (about
  0.529 GiB), later growth remains within the admitted cache budget, and no new
  swap appears. Separate warm page-cache evidence from cold device-I/O evidence.
- Resident mode proves complete model mapping and full-tensor Metal execution,
  not that every mapped GGUF page remained physically resident. Measure the
  stronger claim separately if it is used in release language.
- Run the model-free ExpertMajor admission fixture for a Qwen storage value of
  GGML/Q4, and, when the exact retired file is available, run
  `./ds4 -m "$QWEN_RETIRED_Q4_NEGATIVE" --ctx 8192`. Both must reject before
  inference; a command that produces tokens is a release blocker.
- Physical 16 GiB measurements and normalized-vs-source research comparisons
  improve hardware and artifact characterization, but are not additional
  release gates beyond the standard model/backend checks above. Do not claim
  measurements for hardware or artifacts that were not actually tested.

## 6. Frozen CUDA Reactivation Gate

For a normal release, confirm CUDA source, tests, and build targets remain
absent from the release tree and record the lane as `frozen - source absent`.
This is stronger and more precise than calling CUDA merely unvalidated.
The recovery commit recorded in `docs/contracts/RUNTIME_SUPPORT.md` contains
the former source and build/test recipes; restoring only part of that lane is
not reactivation.

If any change restores CUDA source or build integration, the release is blocked
until the complete reactivation lane passes:

- accept an ADR, update the runtime support contract, and identify an owner;
- put the exact release commit on the current designated CUDA validation host;
- complete a clean backend build and the restored model-free, backend,
  long-context, and synthetic-kernel regression suites;
- verify DeepSeek, GLM, and Qwen ExpertMajor v2 fail closed before inference
  unless the new ADR separately qualifies those exact CUDA paths;
- record build identity, hardware, compiler/architecture, commands, outputs,
  failures, and before/after performance evidence;
- verify shared and warning-cleanup changes are also clean on macOS and do not
  alter Metal correctness or speed.

A successful compile alone is not reactivation and must not be published as a
supported CUDA inference claim.

## 7. Frozen ROCm Reactivation Gate

For a normal release, confirm ROCm source, tests, and build targets remain
absent from the release tree and record the lane as `frozen - source absent`.
Use the recovery commit in `docs/contracts/RUNTIME_SUPPORT.md` as the source and
test-history boundary; a partial restore is not reactivation.

If any change restores ROCm source or build integration, the release is blocked
until the complete reactivation lane passes:

- accept an ADR, update the runtime support contract, and identify an owner;
- put the exact release commit on the current designated ROCm validation host;
- complete a clean backend build and the restored model-free, backend,
  long-context, and synthetic-kernel regression suites;
- verify DeepSeek, GLM, and Qwen ExpertMajor v2 fail closed before inference
  unless the new ADR separately qualifies those exact ROCm paths;
- record build identity, backend initialization, commands, outputs, failures,
  and before/after performance evidence;
- verify shared changes remain correct and fast on Metal.

A successful compile alone is not reactivation and must not be published as a
supported ROCm inference claim.

## 8. Retired Distributed Inference

Distributed implementation source is absent from the release tree. Its former
command-line surface remains only as a centralized fail-closed policy. Run the
model-free retired-option gate and confirm every executable rejects the options
before model loading:

`sh tests/test_retired_distributed_flags.sh`

The gate covers `ds4`, `ds4-server`, `ds4-agent`, `ds4-bench`, and `ds4-eval`,
and all nine retired flags: `--role`, `--layers`, `--listen`, `--coordinator`,
`--dist-prefill-chunk`, `--dist-prefill-window`,
`--dist-activation-bits`, `--dist-replay-check`, and `--debug`. Also confirm:

- normal and topic help do not advertise an active distributed setup;
- `ds4_distributed.c`, `ds4_distributed.h`, `ds4_distributed.o`, `ds4_dist_*`,
  `DS4_DISTRIBUTED_*`, layer-slice execution, and distributed layer-payload
  APIs are absent from production source and build dependencies;
- no canonical or split model is used as a compatibility fallback.

Restoring any distributed implementation or build integration requires a new
accepted ADR, an owner, updated support and security contracts, model-free
protocol tests, full affected-model correctness/performance evidence, and the
applicable release lanes. The recovery boundary is Git commit
`d8d673858f90834522bbe878951a534d8c6508b4`; a partial restore is not support.

## 9. Disk KV Cache

Disk KV cache bugs are high impact for server users.

- Start the server with:
  `./ds4-server -m "$DEEPSEEK_V2" --ctx 100000 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192`.
- Run the same request twice and verify the second request hits cache.
- Fill the cache enough to trigger eviction; verify the newly-written entry is
  not evicted and useful anchors are retained.
- Test rejection of incompatible checkpoints when model, quantization, context,
  or raw/compressed KV layout changes.
- Test stripped agent sessions: `/strip <id>` then `/switch <id>` should rebuild
  by prefill and render sane history.

## 10. Server APIs

The server must keep compatibility across OpenAI, Responses, and Anthropic
clients.

- `GET /v1/models/deepseek-v4-flash` and `GET /v1/models/deepseek-v4-pro`
  should both serve whichever GGUF is loaded.
- Test OpenAI chat completion, OpenAI Responses, and Anthropic messages.
- Test SSE streaming with thinking enabled and disabled.
- Test keepalive during long prefill and confirm clients do not time out.
- Test `--trace` and confirm rendered prompts, cache decisions, generated text,
  and tool-parser events are useful without leaking unrelated state.

## 11. ds4-agent

The agent is the most stateful component.  Test it manually, not only by build.

- Startup banner, status bar, help, `/power`, `/save`, `/list`, `/switch`,
  `/history`, `/compact`, `/new`, `/del`, and `/strip`.
- Ctrl+C during generation, during prefill, during a web fetch, and during a
  long tool call.  After `Stopped by user`, typing a new prompt must work.
- Queue messages while the model is busy.  Queued messages must not skip tool
  execution; after tool results, the queued user text must be provided.
- Read/search/edit/write tools:
  create a temp project, ask for edits, verify old/new and `[upto]` anchored
  edits fail safely on ambiguous matches and do not require retyping whole files.
- Real coding edit loop:
  delete `/tmp/mymandel`, ask ds4-agent to create a small C ASCII Mandelbrot
  program there, build and run it, then in a second user turn ask for a small
  modification that should naturally use the edit tool, such as changing the
  ASCII character ramp or output dimensions.  Verify the agent edits the
  existing file instead of rewriting the whole project, and that the final
  program still builds and runs.
- Bash tools:
  test short output, large output truncation, non-zero exit output, long-running
  jobs, `bash_status`, and `bash_stop`.
- Web tools:
  `google_search` and `visit_page` should ask for visible Chrome approval with
  timeout, open pages without stealing focus when possible, extract Markdown,
  close tabs, and handle consent/privacy walls as tool errors the model can see.
- TUI:
  test multiline prompt editing, history navigation, queued prompt display,
  status bar fill to terminal width, syntax highlighting in Markdown/code blocks,
  and SSH/remote terminal flicker.

## 12. Download Script And Model Files

- Test `download_model.sh` in a temporary directory so local weights are not
  overwritten.
- Verify `deepseek-v2`, `glm-v2`, and `qwen-v2` resolve to their exact qualified
  repository, immutable revision, and ExpertMajor v2 filename. Qwen must pin
  revision `7bf9c3f7f6136aeb2599d75ee61c0cc2f18e2b02` and must not resolve to the
  retired Q4_K_S object.
- Treat every `offline-*` target as a converter input, not a runnable artifact.
  Verify resume and file naming without launching inference from the source.
- Verify the script never creates or changes `./ds4flash.gguf`, exposes no
  distributed slice target, and contains none of the retired distributed flags.
- Verify each published runtime model has a distinct ExpertMajor v2 filename,
  complete converter verification, recorded source/output hashes, and no
  canonical routed-weight fallback.
- Verify legacy removed targets fail clearly.
- Verify README model names match the script and Hugging Face repository.

## 13. Performance And Power

- Run `ds4-bench` separately with `-m "$DEEPSEEK_V2"`, `-m "$GLM_V2"`, and
  `-m "$QWEN_V2"`, using each family's tracked workload and CSV baseline.
- Test `--power 100` is not throttled.
- Test `--power 50` visibly reduces duty cycle in CLI, server, agent, eval, and
  bench where practical.
- Confirm context buffer size, raw KV rows, compressed KV rows, and mmap behavior
  match expectations for 32k, 100k, and any release-advertised context size.

## 14. Release Sign-off

Do not sign off until:

- macOS Metal Flash passed.
- The qualified DeepSeek, GLM, and Qwen release artifacts passed their
  model-backed lanes with the residency modes defined by
  `docs/contracts/RUNTIME_SUPPORT.md`.
- The Qwen immutable publication record and downloader gate matched the same
  exact affine artifact used by the runtime lane.
- The retired Qwen Q4_K_S store remained fail-closed and was not presented as a
  runnable or downloadable fallback.
- CUDA source/tests/build targets were confirmed absent and recorded as frozen;
  if any were restored, the complete section 6 reactivation gate passed.
- ROCm source/tests/build targets were confirmed absent and recorded as frozen;
  if any were restored, the complete section 7 reactivation gate passed.
- Disk KV cache was exercised.
- Server API streaming was exercised.
- Agent interruption and tool loops were exercised manually.
- Speed is within expected variance for the same hardware and model.
- Any skipped item is written down with the reason.
