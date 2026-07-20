# QA Before Releases

This is the release gate for DwarfStar.  Run it before tagging or pushing a
release build.  The goal is not to prove every code path exhaustively; it is to
exercise the paths that have historically regressed: Metal graph inference,
CUDA, ROCm, SSD streaming, distributed execution, disk KV cache, server APIs, and the
agent TUI/tool state machine.

Do not run multiple huge model processes at the same time.  Record the commit,
hardware, GGUF file, context size, and any non-default flags for every manual
run.

DeepSeek V4, GLM 5.2, and Qwen3.6 release inference is ExpertMajor v2-only on
local Apple Metal. Canonical files may be inspected or converted offline, but
ExpertMajor v1, sidecars, CPU, CUDA, ROCm, and distributed inference must fail
closed. No ExpertMajor or Qwen admission environment flag belongs in a release
startup command.

Preferred release test hosts:

- CUDA / DGX Spark: `toor@192.168.0.180`.
- Metal / distributed Mac testing: `mac-m5max-it` and `mac-m5max-us`.
- ROCm: The Strix Halo system at antirez@strixhalo (Framework Desktop).

The Mac hosts have DNS entries and are reached through an internet VPN.  They
are connected to each other over WiFi and also through a Thunderbolt 5
point-to-point link.  The TB5 route is the preferred distributed-inference
network when it is available, but it can be fragile and sometimes only works
when `ds4` is executed in the foreground.  Prefer these machines for release
testing, especially distributed inference.  Local fallback testing on this
machine is acceptable when needed; it is an M3 Max with 128 GB RAM.
The Strix Halo system is reachable via the VPN as well and has a local WiFi
address in the same lan of the M5 Max systems. The CUDA hosts are in a
different remote lan and are accessible via a different VPN active
in this system.

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
- Run whitespace checks before committing:
  `git diff --check`.
- Confirm `./ds4 --help`, `./ds4-server --help`, and `./ds4-agent --help` render
  cleanly, with readable section colors and no broken wrapping.

## 2. Core Regression Tests

- Run the default suite:
  `make test`.
- Without a local GGUF, run the complete model-free gate:
  `make model-free-test`.
- Run the vector checks explicitly after any tokenizer, template, KV, kernel,
  quantization, or prompt-rendering change:
  `./ds4_test --logprob-vectors`
  and `./ds4_test --local-golden-vectors`.
- Run server tests when HTTP, SSE, prompt rendering, cache policy, or tool-call
  replay changed:
  `./ds4_test --server`.
- Run `./ds4-eval --self-test-extractors`.

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
  `./ds4 --nothink --temp 0 --dump-logprobs /tmp/ds4-logprobs.json --logprobs-top-k 20 -p "..."`
  and inspect that the continuation is sane.
- Speed sanity:
  run `ds4-bench` with `speed-bench/promessi_sposi.txt` and compare prefill,
  generation speed, and KV bytes with the last known good numbers for the same
  machine.

## 4. Metal PRO Path

PRO support is experimental, but release builds must not break it silently.

- If a PRO-capable machine is available, run a short PRO q2 prompt and verify
  the correct template, thinking behavior, and endpoint aliases.
- For PRO Q4 distributed builds, test only on the intended high-memory machines.
- If PRO cannot be run locally, at least build all binaries and review changes
  touching model shape, tensor lookup, routed expert mapping, template logic,
  and KV payload compatibility.

## 5. SSD Streaming

SSD streaming is a capacity path, so test both correctness and user experience.

- Flash q2/q2-q4 streaming:
  `./ds4 -m "$DEEPSEEK_V2" --ssd-streaming --ssd-streaming-cache-experts 32GB -p "..."`
- Regression test mixed-quant Flash SSD streaming. Use the mixed q2/q4 GGUF
  with boosted Q4 routed-expert layers and a prompt long enough to exercise the
  selected-address prefill path; it must not fail with "model range is not
  covered by mapped model views":
  `./ds4 -m "$DEEPSEEK_MIXED_V2" --ssd-streaming --ssd-streaming-cache-experts 16GB --ctx 4096 --tokens 1 --nothink --prompt-file /tmp/ds4_600tok_prompt.txt`.
- Cold streaming measurement:
  run once with `--ssd-streaming-cold` and verify no deadlock, missing expert,
  or impossible slowdown.
- Confirm startup reports cache budget and that generation does not stall on
  repeated expert misses for a small interactive prompt.
- If streaming cache internals changed, test the same prompt twice and compare
  first-token/logprob sanity between runs.

### Qwen3.6 Metal lane

The Qwen path on `main` follows the same repository, build, core-test, and
regression rules as the other model paths. Use the normalized
Qwen3.6-35B-A3B ExpertMajor v2 Q4_K_S artifact, record its SHA-256, and run the
relevant model-backed smoke; canonical, v1, sidecar, and community GGUFs are
not equivalent inputs.

- Run `make model-free-test` and `./ds4_test --metal-kernels`. The latter must
  retain resident/SSD top-8 output equivalence, zero resident cache/`pread`
  accounting, malformed-route fail-closed behavior, and slab-growth checks.
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
- Physical 16 GiB measurements and normalized-vs-source research comparisons
  improve hardware and artifact characterization, but are not additional
  release gates beyond the standard model/backend checks above. Do not claim
  measurements for hardware or artifacts that were not actually tested.

## 6. CUDA / DGX Spark

Before a release, ask the user for CUDA access if it is not already configured.
Use the DGX Spark / GB10 host `toor@192.168.0.180`.  Do not claim CUDA is
release-ready without this pass.

- Fetch or push the exact release commit to the CUDA machine.
- Build:
  `make clean && make cuda-spark`.
- Run `make cuda-regression` for model-free/backend coverage.
- Verify a Qwen, DeepSeek, or GLM ExpertMajor v2 artifact is rejected before
  inference. Do not publish generation throughput or present CUDA as a current
  MoE runtime lane.
- If CUDA kernels or build hooks changed, run their backend-specific synthetic
  tests without weakening the ExpertMajor v2 admission boundary.
- Verify that any CUDA-only warning fixes are also clean on macOS and do not
  change Metal behavior.

## 7. ROCm / Strix Halo

Use the Strix Halo Framework Desktop via the VPN hostname `strixhalo`
(`antirez@strixhalo`).  This host validates the ROCm backend; do not use it as
a substitute for CUDA or Metal release testing.

- Fetch or push the exact release commit to the Strix Halo machine.
- Build:
  `make clean && make strix-halo`.
- Run model-free/backend tests only. Verify a Qwen, DeepSeek, or GLM
  ExpertMajor v2 artifact is rejected before inference; do not attempt a
  canonical fallback or publish ROCm model throughput.
- Record build identity and backend initialization for synthetic tests. A
  successful ROCm compile is not a supported-model inference claim.

## 8. Distributed Inference

Distributed inference is outside the current ExpertMajor v2 runtime contract.
For Qwen, DeepSeek, and GLM, verify coordinator/worker options reject the model
before inference; do not use canonical or split files as a fallback. If shared
protocol code changes, run model-free protocol tests and compile checks only.

## 9. Disk KV Cache

Disk KV cache bugs are high impact for server users.

- Start the server with:
  `./ds4-server --ctx 100000 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192`.
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
- Treat its canonical Flash/PRO downloads as offline converter inputs, not
  runnable artifacts. Verify URL, resume, file naming, and symlink policy
  without launching inference from the downloaded source.
- Verify each published runtime model has a distinct ExpertMajor v2 filename,
  complete converter verification, recorded source/output hashes, and no
  canonical routed-weight fallback.
- Verify legacy removed targets fail clearly.
- Verify README model names match the script and Hugging Face repository.

## 13. Performance And Power

- Run `ds4-bench` on the release machine and compare with tracked CSV baselines.
- Test `--power 100` is not throttled.
- Test `--power 50` visibly reduces duty cycle in CLI, server, agent, eval, and
  bench where practical.
- Confirm context buffer size, raw KV rows, compressed KV rows, and mmap behavior
  match expectations for 32k, 100k, and any release-advertised context size.

## 14. Release Sign-off

Do not sign off until:

- macOS Metal Flash passed.
- CUDA was tested on the CUDA machine or the release notes explicitly say CUDA
  was not validated.
- ROCm was tested on Strix Halo or the release notes explicitly say ROCm was
  not validated.
- Disk KV cache was exercised.
- Server API streaming was exercised.
- Agent interruption and tool loops were exercised manually.
- Speed is within expected variance for the same hardware and model.
- Any skipped item is written down with the reason.
