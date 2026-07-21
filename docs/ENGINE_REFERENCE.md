# DwarfStar engine reference

> [!NOTE]
> This is the long-form engine guide inherited and evolved from
> [`antirez/ds4`](https://github.com/antirez/ds4). Start with the
> [fork landing page](../README.md) for current fork scope, measured deltas,
> model maturity, safety caveats, and the upstream co-development policy.

> [!IMPORTANT]
> The current inference contract for DeepSeek V4, GLM 5.2, and
> Qwen3.6-35B-A3B is a validated embedded `ds4.expert_major.v2` GGUF on Apple
> Metal. Canonical GGUFs are offline converter inputs only. ExpertMajor v1,
> external sidecars, CPU, CUDA, ROCm, and distributed inference are not runtime
> fallbacks. Historical benchmark and architecture sections below may describe
> those retired paths, but they are not current startup instructions.

## Motivations

* Very capable open weight models finally exist. DeepSeek v4 Flash feels quasi-frontier. The PRO is even better. Both resist 2 bit quantization very well.
* Very capable high-memory Apple Silicon computers now exist.
* DeepSeek v4 KV cache design makes it practical to run very large contexts. Other vendors are using this approach.
* These few-hundred-billion-parameter models are strictly better than smaller (even if dense) models, regardless of what benchmarks say.

That said, a few important things about this project:

* The local inference landscape contains many excellent projects, but new models are released continuously, and the attention immediately gets captured by the next model to implement. This project takes a deliberately narrow bet: one model at a time, official-vector validation (logits obtained with the official implementation), long-context tests, and enough agent integration to know if it really works. The exact model may change as the landscape evolves, but the constraint remains: credible local Apple Metal inference, with family-specific memory limits. DeepSeek and GLM currently start at 64 GiB; Qwen's SSD-backed policy starts at 16 GiB.
* This software is developed with **strong assistance from GPT 5.5** and with humans leading the ideas, testing, and debugging. We say this openly because it shaped how the project was built. If you are not happy with AI-developed code, this software is not for you. The acknowledgement below is equally important: this would not exist without `llama.cpp` and GGML, largely written by hand.
* This implementation is based on the idea that compressed KV caches like the one of DeepSeek v4 and the fast SSD disks of modern MacBooks should change our idea that KV cache belongs to RAM. **The KV cache is actually a first-class disk citizen**. Fast SSD disks also changed the inference game from the point of view of "model needs to fit RAM": while having more RAM than the model size is still preferred, SSD streaming turns the available amount of RAM from a hard cutoff (can I run this model or not?) into a continuous spectrum of speed levels.
* Our vision is that local inference should be a set of three things working well together, out of the box: A) inference engine with HTTP API + B) GGUF specially crafted to run well under a given engine and given assumptions + C) testing and validation with coding agents implementations. D) Purpose built agents for specific models and execution environments. DwarfStar only runs with the GGUF files provided. It gets tested against officially obtained logits at different context sizes. This project exists because we wanted to make one local model feel finished end to end, not just runnable. However this is beta quality code, so probably we are not still there, especially since SSD streaming and the additional model families are recent additions.
* The production graph path targets **Metal on macOS**. The CPU path is only for correctness checks and model/tokenizer diagnostics. For CPU-only Linux builds, use `make cpu`; it builds the normal `./ds4` and `./ds4-server` binaries without a GPU backend. On macOS, **warning: current macOS versions have a bug in the virtual memory implementation that can crash the kernel** if you try to run very large CPU model inference. Do not use CPU as a production fallback.

## Acknowledgements to llama.cpp and GGML

`ds4.c` does not link against GGML, but it **exists thanks to the path opened by the
llama.cpp project and the kernels, quantization formats, GGUF ecosystem, and hard-won
engineering knowledge developed there**.
We are thankful and indebted to [`llama.cpp`](https://github.com/ggml-org/llama.cpp)
and its contributors. Their implementation, kernels, tests, and design choices were
an essential reference while building this DeepSeek V4 specific inference path.
Some source-level pieces are retained or adapted here under the MIT license: GGUF
quant layouts and tables, CPU quant/dot logic, and certain kernels. For this
reason, and because we are genuinely grateful, we keep the GGML authors copyright
notice in our `LICENSE` file.

## Status

The code and GGUF files are **beta quality**. Inference and model serving are
complicated, the supported model paths are evolving quickly, and not every
backend receives the same validation at the same time. We try to keep tested
paths usable. If you hit an issue, use `--trace` to log the session and include
the full trace in the report.

The `ds4-agent` was added later and remains alpha quality.

## More Documentation

If you are looking for very specific things, we have other
sub-README files. Otherwise for normal usage keep reading the
next sections.

- [CONTRIBUTING.md](../CONTRIBUTING.md): correctness and speed regression testing
  guide for contributors. **Read this before sending a pull request**.
- [docs/contracts/RUNTIME_SUPPORT.md](contracts/RUNTIME_SUPPORT.md): authoritative
  current backend, model, artifact, and residency support matrix.
- [GOLD_METAL_SSD.md](../GOLD_METAL_SSD.md): Metal/SSD planner details and
  benchmark promotion gates.
- [gguf-tools/README.md](../gguf-tools/README.md): offline GGUF generation,
  imatrix collection, quantization tooling, and quality checks.
- [gguf-tools/imatrix/README.md](../gguf-tools/imatrix/README.md): how the
  routed-MoE imatrix is collected and used.
- [gguf-tools/imatrix/dataset/README.md](../gguf-tools/imatrix/dataset/README.md):
  how the calibration prompt corpus is generated.
- [gguf-tools/quality-testing/README.md](../gguf-tools/quality-testing/README.md):
  how local GGUFs are scored against official DeepSeek V4 Flash/PRO continuations.
- [dir-steering/README.md](../dir-steering/README.md): directional steering data,
  vector generation, and usage.
- [speed-bench/README.md](../speed-bench/README.md): benchmark commands, charts,
  and CSV generation.
- [tests/test-vectors/README.md](../tests/test-vectors/README.md): official
  continuation vectors used for regression checks.

## Model Weights

This implementation only works with the ExpertMajor v2 GGUFs published for
this project. It is not a general GGUF loader, and arbitrary Qwen, DeepSeek,
GLM, or community GGUF files will not have the validated embedded store,
tensor layout, quantization mix, or metadata expected by the engine. The 2 bit
DeepSeek quantizations provided here are not
a joke: they behave well, work under coding agents, call tools in a reliable way.
The 2 bit quants use a very asymmetrical quantization: only the routed MoE
experts are quantized, up/gate at `IQ2_XXS`, down at `Q2_K`. They are the
majority of all the model space: the other components (shared experts,
projections, routing) are left untouched to guarantee quality.

Obtain a qualified ExpertMajor v2 release artifact and verify the exact size and
complete output SHA-256 in its publication record:

- DeepSeek V4 Flash: use only a filename containing `DS4-ExpertMajor-v2` and the
  output identity required by
  [`deepseek-expert-major-v2.md`](deepseek-expert-major-v2.md). The current
  canonical mirror is an offline converter source, not the runtime artifact.
- [GLM 5.2 DS4 GGUF](https://huggingface.co/andreaborio/GLM-5.2-DS4-GGUF)
- [Qwen3.6-35B-A3B DS4 GGUF](https://huggingface.co/andreaborio/Qwen3.6-35B-A3B-DS4-GGUF)

Use the exact artifact, size, and SHA-256 recorded in the family documentation.
`download_model.sh` exposes the three qualified runtime artifacts as explicit
`*-v2` targets. Its clearly named `offline-*` targets download only complete
canonical converter inputs; distributed slices and legacy runtime targets are
not exposed. The script never creates `./ds4flash.gguf`. Convert and verify an
offline source with `gguf-tools/ds4-expert-major.py` before publication or
inference.

If you want to regenerate GGUF files or collect a new imatrix, see
[gguf-tools/README.md](../gguf-tools/README.md). Those tools are meant for offline
model-building work and can take a long time on the full DeepSeek V4 Flash
weights. Flash GGUF generation is supported by the local tools. PRO GGUF
production currently still depends on the external `llama.cpp`-based workflow;
native tooling can be added later.

The optional MTP/speculative decoding path remains experimental and
correctness-gated. It is not part of normal startup, is not qualified for GLM
or SSD streaming, and currently has no meaningful generation-speed win.

Then build:

```sh
make                  # macOS Metal
make cpu              # CPU-only diagnostics in build/cpu-<arch>/bin on macOS
```

On macOS, Metal and CPU objects/binaries live in separate build profiles.
`make cpu` never replaces the root Metal commands.  Use
`build/cpu-$(uname -m)/bin/ds4` for the CPU-only binary and `./ds4 --build-info`
to verify build provenance.

Do not rely on the historical `./ds4flash.gguf` symlink for runtime identity.
Pass `-m` with an absolute qualified ExpertMajor v2 path and verify its complete
published output SHA-256 before inference. Run `./ds4 --help` and
`./ds4-server --help` for the full flag list.

## Speed

Except for the explicitly historical DGX row, these are single-run Metal CLI
numbers with `--ctx 32768`, `--nothink`, greedy decoding, and `-n 256`. The
short prompt is a normal small Italian story prompt. The long prompts exercise
chunked prefill plus long-context decode. Q4 requires the larger-memory machine
class, so M3 Max Q4 numbers are `N/A`. The DGX result is retained as historical
benchmark context and is not a current CUDA support claim.

| Machine | Quant | Prompt | Prefill | Generation |
| --- | ---: | ---: | ---: | ---: |
| MacBook Pro M3 Max, 128 GB | q2 | short | 58.52 t/s | 26.68 t/s |
| MacBook Pro M3 Max, 128 GB | q2 | 11709 tokens | 250.11 t/s | 21.47 t/s |
| MacBook Pro M3 Max, 128 GB | q4 | short | N/A | N/A |
| MacBook Pro M3 Max, 128 GB | q4 | long | N/A | N/A |
| MacBook Pro M5 Max, 128 GB | q2 | short | 87.25 t/s | 34.27 t/s |
| MacBook Pro M5 Max, 128 GB | q2 | 11707 tokens | 463.44 t/s | 25.90 t/s |
| Mac Studio M3 Ultra, 512 GB | q2 | short | 84.43 t/s | 36.86 t/s |
| Mac Studio M3 Ultra, 512 GB | q2 | 11709 tokens | 468.03 t/s | 27.39 t/s |
| Mac Studio M3 Ultra, 512 GB | q4 | short | 78.95 t/s | 35.50 t/s |
| Mac Studio M3 Ultra, 512 GB | q4 | 12018 tokens | 448.82 t/s | 26.62 t/s |
| Mac Studio M3 Ultra, 512 GB | PRO q2 | 32768 tokens | 138.82 t/s | 9.56 t/s |
| DGX Spark GB10, 128 GB | q2 | 7047 tokens | 343.81 t/s | 13.75 t/s |

![M3 Max t/s](../speed-bench/m3_max_ts.svg)
![PRO model M3 Ultra t/s](../speed-bench/pro_model_m3_ultra_ts.svg)

## Running models larger than RAM

The normal macOS invocation uses Metal with **AUTO residency**. For DeepSeek and
Qwen, it estimates the model plus context/KV/scratch requirement and may keep
the model resident when the family admission gates pass; otherwise it selects
**SSD streaming**. GLM AUTO always selects its qualified SSD-streaming path and
rejects resident mode. In streaming mode the non-routed model weights stay
resident, while routed MoE experts are kept in an in-memory cache and loaded
from the GGUF file on cache misses.

Streaming is not as fast as fitting the full model in RAM. It still needs memory
for non-routed weights, KV cache, graph scratch, activations, and the routed
expert cache. It is useful because routed experts dominate model size and modern
Mac SSDs are fast enough to make cache misses tolerable. Long prefills can still
be fast; generation is more sensitive to cache misses because every new token
routes through experts again.

Start with AUTO residency and the automatic cache budget:

```sh
./ds4 -m /absolute/path/to/QUALIFIED-DEEPSEEK-OR-QWEN-DS4-ExpertMajor-v2.gguf
```

For DeepSeek and Qwen qualification, use `--ssd-streaming` to force streaming
or `--resident` to request the full-model mapped mode. Startup logs report the
resolved mode and memory-plan reason. GLM release startup remains flag-free
AUTO; `--resident` is rejected and explicit SSD/cache controls are diagnostics,
not alternate startup instructions.

The supported `qwen35moe` candidate contract is the normalized
Qwen3.6-35B-A3B ExpertMajor v2 MLX affine4/group-64 artifact. It remains a
single GGUF with the standard v2 container and activates automatically in both
resident and SSD modes. The former v2 GGML/Q4 payload is rejected. AUTO requires both the
normal working-set calculation and a live unified-memory pressure snapshot; if
either cannot admit resident mode, it uses bounded SSD streaming. Qwen's cache
planner charges its complete non-routed page set separately and grows cache
storage in 321-expert (about 0.529 GiB) slabs. The DeepSeek resident/SSD planner
and GLM SSD-only planner remain independent. Exact artifact and validation
details live in
[`qwen-expert-major-store.md`](qwen-expert-major-store.md) and the consolidated
[`affine AUTO/SSD gate`](benchmarks/2026-07-21-qwen-unified-affine-auto-ssd.md).

Qwen numerical inference is currently Metal-only. AUTO exposes named
16/24/32/36/48/64/96/128 GiB profiles but sizes resident headroom and SSD cache
from exact physical RAM, Metal's reported working-set limit, context runtime,
and current pressure. The 16 GiB profile is necessarily SSD for the current
19.37 GiB tensor payload; 32 GiB can be resident for shorter contexts when both
admission gates pass. The CPU performs tokenizer,
sampling, selected-route readback, cache bookkeeping, and GGUF I/O in streamed
mode; there is no CPU/GPU split of neural layers or routed experts.  Resident
mode disables DS4's expert-cache `pread`, but Metal's residency request remains
a budget hint rather than proof that every mapped page stayed physically in RAM.

If startup reports that the expert cache is too large, or if you want to reserve
more memory for context, set the routed expert cache explicitly:

```sh
./ds4 \
  -m /absolute/path/to/QUALIFIED-DEEPSEEK-OR-QWEN-DS4-ExpertMajor-v2.gguf \
  --ssd-streaming --ssd-streaming-cache-experts 32GB
```

The `32GB` value is a memory budget for complete routed experts, not a generic
byte cache. DwarfStar converts it to the number of full experts that fit for the
current GGUF. Non-routed weights, KV cache, graph scratch, and activations need
additional memory. Only the automatic cache budget does the full subtraction
for you: it reserves context/KV/scratch, external pressure, and 20% backend
headroom (at least 2 GiB), then subtracts non-routed weights and uses the safe
remainder for routed experts. Leave the hot expert preload enabled for
normal use; use `--ssd-streaming-cold` and `--ssd-streaming-preload-experts N`
only for measurements.

### Practical SSD streaming examples

On a qualified 64 GB host, run the published DeepSeek Flash ExpertMajor v2
artifact with its recorded complete output SHA-256 and a moderate expert cache:

```sh
./ds4 \
  -m /absolute/path/to/QUALIFIED-DEEPSEEK-FLASH-DS4-ExpertMajor-v2.gguf \
  --ssd-streaming \
  --ssd-streaming-cache-experts 32GB \
  --ctx 32768 \
  --nothink
```

Historical PRO q2 streaming experiments used canonical artifacts and measured
automatic and manual cache budgets on 128 GB hosts. They are retained in dated
benchmark history only. No complete PRO ExpertMajor v2 output identity and
release gate is specified here, so there is no executable PRO command in the
current guide; canonical and split PRO files must fail closed.

### Historical canonical Flash SSD-streaming tiers (July 2026)

> [!NOTE]
> These rows predate the ExpertMajor v2-only runtime and use a canonical Flash
> GGUF. They are historical performance context, not current hardware support,
> cache policy, artifact identity, or executable startup guidance.

These measurements use the same 86.72 GB (80.76 GiB) DeepSeek V4 Flash
IQ2XXS/SExpQ8 GGUF on AC power, without static-weight pinning. The workload is
shown because short decode rates depend heavily on prompt length and the macOS
file-backed page cache; rows with different workloads are not directly
comparable.

| Mac / tested ds4 build | Cache and context | Bounded workload | Prefill | Generation |
| --- | --- | --- | ---: | ---: |
| M1 Pro, 16 GB / [`2f95e67`](https://github.com/andreaborio/ds4/commit/2f95e67fdec1db988fe8b1a699330f387de66004) | exact 259, 8,192 | DSBox API, 9 prompt + 2 output tokens | — | 0.30 t/s cold; 0.53 / 0.51 / 0.51 t/s warm (~0.52 t/s) |
| M1 Pro, 16 GB / [`bf4201c`](https://github.com/andreaborio/ds4/commit/bf4201c47b901f0f479dc4af3f3df77330fabacf) | exact 259, 8,192 | extremely hot CLI, 14 + 2 tokens | 1.02–1.64 t/s | 2.13–2.46 t/s |
| M5 Pro, 64 GB / `6aa496d` | AUTO 3,613, 32,768 | DSBox API, two sequential 22–23 prompt + 64 output-token requests | — | 9.88 / 12.86 t/s |
| M5 Pro, 64 GB / [`f4e0e64`](https://github.com/andreaborio/ds4/commit/f4e0e64e76ab62151700f9ea404297ea1769c550) | AUTO 4,387, 32,768 | `ds4-bench`, 128 + 64 tokens, ABBA legs A1/A2 | 21.63 / 22.21 t/s | 13.05 / 13.59 t/s (13.3173 geomean) |
| M5 Pro, 64 GB / [`f4e0e64`](https://github.com/andreaborio/ds4/commit/f4e0e64e76ab62151700f9ea404297ea1769c550) | exact 4,342, 32,768 | same bounded ABBA, legs B1/B2 | 22.22 / 22.13 t/s | 13.74 / 13.78 t/s (13.7600 geomean) |

The M5 exact 4,342 arm was 3.32% faster in decode than AUTO 4,387, with
identical frontier logits and zero new swapout. The generic default remains
AUTO: the small gap does not justify applying one GGUF's exact expert count to
other quantizations. On 16 GB, AUTO uses the 259-expert floor only when the
live memory budget can safely admit it; otherwise startup fails closed.

The current DSBox server canary was run on the same M5 Pro under a lower live
memory budget, so AUTO selected 3,613 experts / 23.82 GiB rather than 4,387.
Both 64-token requests completed at normal macOS pressure with zero new
swapout; the second, warmer but different prompt reached 12.86 t/s. This row is
a service-path observation, while the 4,387/4,342 rows are controlled
`ds4-bench` comparisons.

The M1 `2f95e67` server build was later reverted by
[`8a2a53f`](https://github.com/andreaborio/ds4/commit/8a2a53f323d29e5afd99010852f99019ef0cc8f4)
because its startup bridge could admit the cache under insufficient sustained
headroom. The token loop and effective 259-entry cache in that bounded trace
were unchanged, but the row is historical rather than a current-release
guarantee. The 2.13–2.46 t/s row is a two-token, extremely hot micro-canary,
not sustained DSBox throughput; the repeatable short-server observation before
pressure was about 0.5 t/s.

## Retired Distributed Inference

> [!CAUTION]
> Distributed inference is retired. Its implementation source, layer-slice
> execution, transport protocol, and distributed session payloads are absent
> from the active tree. Former distributed CLI flags fail closed before model
> loading and are not tuning controls.

The historical implementation split canonical DeepSeek layers across hosts and
pipelined long prefill chunks. Two directly connected M5 Max systems measured
higher long-prefill throughput, while decode was slower because every
autoregressive token crossed the route. Those measurements describe removed
research code, not a supported topology or an ExpertMajor v2 execution path.

There is no current coordinator, worker, split-model, or network setup. Recover
the last pre-removal implementation from Git commit
`d8d673858f90834522bbe878951a534d8c6508b4` only after a new accepted ADR,
ownership, security review, protocol tests, and complete model correctness and
performance qualification.

## Reducing heat, power usage and fan noise

Long local inference runs can keep the GPU busy for extended periods. If you
care more about heat, fan noise, battery life on MacBooks, or reducing thermal
stress on the hardware than about maximum throughput, use `--power N`.

`--power 100` is the default and means full speed. Lower values ask DwarfStar to target
that percentage of GPU usage: `--power 70` targets about 70%, `--power 50`
targets about half usage, and so forth. DwarfStar does this by measuring GPU work time
and inserting small sleeps between work units: during prefill it sleeps between
layers, and during generation it sleeps between decoded tokens. This reduces
sustained load without changing model output.

The option is available on the CLI, server, agent, eval, and benchmark tools,
for example:

```sh
./ds4 -m /absolute/path/to/QUALIFIED-DS4-ExpertMajor-v2.gguf --power 50
./ds4-agent -m /absolute/path/to/QUALIFIED-DS4-ExpertMajor-v2.gguf --power 70
./ds4-server -m /absolute/path/to/QUALIFIED-DS4-ExpertMajor-v2.gguf --power 40 --ctx 100000
```

## Native agent

DwarfStar features a native coding agent that works in a different way
than most other systems: the inference is controlled from within the agent
itself, without socket/API boundaries, so the session is represented
by the on-disk KV cache itself. Moreover the tools and the system prompt
are all designed vertically for DeepSeek v4 Flash and PRO. This provides a
few advantages:

* Low latency experience, bounded mainly by the prefill speed limits. Displaying of generated text, tool calling, start of a new session are always instantaneous.
* Live progress bar during prefill time.
* No DSML tool calling conversion, the tools are handled natively in the LLM format.
* KV cache mismatch are impossible by construction, the current state is always the truth.
* Everything is tuned for this model.
* Ability to switch saved sessions with `/list` and `/switch`; full KV sessions resume without a prefill stage.

Agent sessions are stored in `~/.ds4/kvcache`. Use `/save` to persist the
current session, `/list` to show saved sessions sorted by recent update time,
and `/switch <sha>` to resume one of them. The session ID is stable across
future saves and is derived from the first user prompt and creation time.
`/del <sha>` removes a saved session. `/strip <sha>` keeps the rendered
conversation text and title but removes the heavy KV payload; switching to a
stripped session rebuilds the KV cache by prefilling the saved text.

The agent compacts long conversations before they become brittle. At roughly
85% context use, or when no more than 8192 tokens remain, it asks the live model
for a durable task-state summary, keeps a recent verbatim tail (10% of context,
capped at 50000 tokens and aligned to a user turn when possible), and rebuilds
the session from system contract, summary, and tail. If a tool result does not
fit, the same compaction runs immediately and the append is retried once.
Private compaction instructions never enter the visible transcript or execute
tools. `/compact` requests the same operation manually, and the terminal shows
`COMPACTING` while the summary and rebuilt prefill are in progress.

Use `--chdir /path/to/ds4` when launching `ds4-agent` from another directory,
so relative runtime files such as `metal/*.metal` resolve from the project tree.

However while the system already works, there is a lot of work to do
in order to make it ready for prime time. When finally the agent will reach
the wanted shape, we will *likely* split the server and the client creating a stateful
session-based protocol that can recreate all that in a client-server way.

## Benchmarking

`ds4-bench` measures instantaneous prefill and generation throughput at context
frontiers instead of reporting one whole-run average. It loads the model once,
walks a fixed token sequence to frontiers such as 2048, 4096, 6144, and uses
incremental prefill so each row measures only the newly-added token interval.
After each frontier it saves the live KV state to memory, generates a fixed
greedy non-EOS probe, restores the memory snapshot, and continues prefill.

```sh
./ds4-bench \
  -m /absolute/path/to/QUALIFIED-DEEPSEEK-FLASH-DS4-ExpertMajor-v2.gguf \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 \
  --ctx-max 65536 \
  --step-incr 2048 \
  --gen-tokens 128
```

The example file is a cleaned public-domain Project Gutenberg text of
Alessandro Manzoni's *I Promessi Sposi* (ebook #45334), with the Gutenberg
header and footer removed: <https://www.gutenberg.org/ebooks/45334>.

Use `--step-incr N` for different linear spacing, or `--step-mul F` for
exponential sweeps. Output is CSV with one row per frontier: latest prefill
interval tokens/sec, generation tokens/sec at that frontier, and
`kvcache_bytes`.

Sessions prefill long prompts in 4096-token chunks by default. Set
`DS4_METAL_PREFILL_CHUNK=N` to compare another chunk size, for example `2048`
to match the strict official-vector checkpoint path, or
`DS4_METAL_PREFILL_CHUNK=0` to prefill a prompt as one whole batch when memory
allows. Changing the chunk changes the KV checkpoint/logit path, so compare it
as an explicit run configuration.
Chunked Metal prefill reuses the same range-capable layer-major graph for each
chunk, preserving absolute compressor/indexer boundaries while avoiding the old
per-layer chunk dispatch path.

## Capability Evaluation

`ds4-eval` is a small real-model integration benchmark. It is not a leaderboard
runner and should not be reported as an official GPQA, SuperGPQA, AIME, or
security benchmark score: the questions are an embedded 92-item subset chosen
to make local regression testing useful and visually inspectable. The program
loads the real GGUF, renders DeepSeek chat prompts, streams sampled tokens in a split-screen TUI, grades
the final answer, and prints a per-question report with prompt tokens,
generated tokens, pass/fail state, the model answer, and the correct answer.

```sh
./ds4-eval \
  -m /absolute/path/to/QUALIFIED-DEEPSEEK-FLASH-DS4-ExpertMajor-v2.gguf \
  --trace /tmp/ds4-eval.txt
```

The default run uses `--tokens 16000`, thinking mode enabled, and a soft/hard
`</think>` budget cutoff so the model has room to produce a visible answer.
`ds4-eval` sizes the context internally from the largest selected prompt plus
the generation budget, and refuses runs that would need more than 1M context
tokens. Press `p` to pause, `q` to exit and print the report, Up/Down to
inspect or select another question, and Enter to run the selected question next.
`--plain` disables the TUI.

Use `--regrade-trace /path/to/trace.txt` to replay the current answer
extractor and scorer against a prior `--trace` file without loading the model
or regenerating tokens. This is useful when auditing evaluator changes: it
shows which cases changed, the old picked answer, the new picked answer, and a
pass/fail summary.

For inference changes that can affect generation drift, keep this deterministic
q1..q4 token-count gate in the test plan:

```sh
./ds4-eval \
  -m /absolute/path/to/QUALIFIED-DEEPSEEK-FLASH-DS4-ExpertMajor-v2.gguf \
  --plain \
  --questions 4 \
  --tokens 2048 \
  --temp 0 \
  --seed 1
```

The generated-token counts must stay aligned with the baseline:

| Question | Expected state | Expected generated tokens | Expected given/correct |
|---:|---|---:|---|
| 1 | `PASSED` | 2048 | `B` / `B` |
| 2 | `PASSED` | 438 | `C` / `C` |
| 3 | `PASSED` | 666 | `70` / `70` |
| 4 | `FAILED` | 2048 | `A` / `C` |

The first 75 embedded questions are interleaved as 25 GPQA Diamond, 25 audited
SuperGPQA, and 25 AIME 2025 problems. The final 17 are an audited COMPSEC
subset of reduced single-function C/C++ vulnerability-localization questions.
The model is asked for the single best source line, or the smallest exact line
set only when the bug cannot be localized to one line; the scorer accepts small
audited ranges only when adjacent lines are equivalent locations for the same
bug. The order is
intentionally progressive: early questions are useful smoke tests, while later
questions are hard enough that a strong reasoning model should still miss some
of them. The SuperGPQA slice is curated rather than blind: upstream rows with
wrong keys, missing figures, or underspecified prompts are replaced with cleaner
rows.

The set should be treated as a hard capability regression suite rather than
a pass/fail unit test.

- **GPQA Diamond** contributes graduate-level science questions with
  multiple-choice answers. DeepSeek's model card reports strong results
  on full GPQA Diamond in thinking mode, but individual items still require
  careful physics, chemistry, or biology reasoning and are easy to lose with a
  small prompt/rendering or sampling regression.
- **SuperGPQA** contributes broad specialist knowledge and domain-transfer
  questions. The model-card SuperGPQA number is much lower than GPQA Diamond,
  so these items are expected to be uneven: some look mundane, others require
  niche professional knowledge or exact interpretation of a translated-style
  exam question.
- **AIME 2025** contributes exact-answer contest math. These are often the most
  unforgiving items in the set: no multiple-choice prior, no partial credit, and
  a single arithmetic or algebraic slip changes the grade.
- **COMPSEC** contributes single-function C/C++ security reasoning items
  reduced from public CVE writeups. These are not exploit prompts: the task is
  to identify the best source line where the defensive code flaw is introduced,
  or return `0` for a safe function.

In practice this means `ds4-eval` should not be expected to produce a perfect
92/92 run. It is meant to answer a more useful engineering question: after a
kernel, quantization, prompt-rendering, KV-cache, or tool-streaming change, does
DeepSeek V4 Flash still solve a representative mix of hard science, broad
knowledge, exact math, and security-code problems while using the same inference
path users run?

## CLI

One-shot prompt:

```sh
./ds4 \
  -m /absolute/path/to/QUALIFIED-DS4-ExpertMajor-v2.gguf \
  -p "Explain Redis streams in one paragraph."
```

No `-p` starts the interactive prompt:

```sh
./ds4 -m /absolute/path/to/QUALIFIED-DS4-ExpertMajor-v2.gguf
ds4>
```

The interactive CLI is a real multi-turn chat. It keeps the rendered chat
transcript and the live graph KV checkpoint, so each turn extends the previous
conversation. Useful commands are `/help`, `/think`, `/think-max`, `/nothink`,
`/ctx N`, `/read FILE`, and `/quit`. Ctrl+C interrupts the current generation
and returns to `ds4>`.

The CLI defaults to thinking mode. Use `/nothink` or `--nothink` for direct
answers. `--mtp MTP.gguf --mtp-draft 2` enables the optional MTP speculative
path; it is useful only for greedy decoding, currently uses a confidence gate
(`--mtp-margin`) to avoid slow partial accepts, and should be treated as an
experimental slight-speedup path.

## Server

Start a local OpenAI/Anthropic-compatible server:

```sh
./ds4-server \
  -m /absolute/path/to/QUALIFIED-DS4-ExpertMajor-v2.gguf \
  --ctx 100000 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
```

Use `--chdir /path/to/ds4` when launching `ds4-server` from another directory,
so relative runtime files such as `metal/*.metal` resolve from the project tree.

The server keeps one mutable backend/KV checkpoint in memory,
so stateless clients that resend a longer version of the same prompt can reuse
the shared prefix instead of pre-filling from token zero.

Request parsing and sockets run in client threads, but inference itself is
serialized through one graph worker. The current server does not batch multiple
independent requests together; concurrent requests wait their turn on the single
live graph/session.

Supported endpoints:

- `GET /v1/models`
- `GET /v1/models/deepseek-v4-flash`
- `GET /v1/models/deepseek-v4-pro`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `POST /v1/completions`
- `POST /v1/messages`

The Flash and PRO model endpoints are compatibility aliases. They both report
the model currently loaded from the GGUF passed with `-m`; the endpoint name does
not select a different model.

`/v1/chat/completions` accepts the usual OpenAI-style `messages`,
`max_tokens`/`max_completion_tokens`, `temperature`, `top_p`, `top_k`, `min_p`,
`seed`, `stream`, `stream_options.include_usage`, `tools`, and `tool_choice`.
Tool schemas are rendered into DeepSeek's DSML tool format, and generated DSML
tool calls are mapped back to OpenAI tool calls.

`/v1/responses` accepts OpenAI Responses-style `input`, `instructions`,
`tools`, `tool_choice`, `max_output_tokens`, `temperature`, `top_p`, `stream`,
and `reasoning`. It is the preferred endpoint for Codex CLI. The server keeps
tool outputs bound to the exact sampled KV frontier by `call_id`. If that live
binding is gone, a request that replays the prior function call can use normal
DSML/KV prefix recovery; an orphan tool output is rejected. DS4 does not persist
OpenAI response objects, so non-null `previous_response_id` and `conversation`
are rejected instead of pretending the state exists.

`/v1/messages` is the Anthropic-compatible endpoint used by Claude Code style
clients. It accepts `system`, `messages`, `tools`, `tool_choice`, `max_tokens`,
`temperature`, `top_p`, `top_k`, `stream`, `stop_sequences`, and thinking
controls. Tool uses are returned as Anthropic `tool_use` blocks. Matching
`tool_result.tool_use_id` values continue from the live sampled frontier,
including multi-tool turns; otherwise the request must replay the corresponding
assistant `tool_use` blocks or it is rejected.

Default sampled API generation uses `temperature=1`, `top_p=1`, and
`min_p=0.05`, so the default filter is relative probability rather than
nucleus mass. In thinking mode DwarfStar uses those fixed sampling defaults and
ignores client sampling knobs, matching DeepSeek's fixed-thinking API behavior.

The chat, Responses, and Anthropic endpoints support SSE streaming. In thinking
mode, reasoning is streamed in the native API shape instead of being mixed into
final text. OpenAI chat streaming
also streams tool calls as soon as the DSML invocation is recognized: the tool
header is sent first, then parameter bytes are forwarded as
`tool_calls[].function.arguments` deltas while generation continues. The
Anthropic endpoint streams thinking and text live, then emits structured
`tool_use` blocks when the generated tool block is complete.
The Responses endpoint streams the Responses event lifecycle expected by Codex,
including `response.output_text.delta`, function-call argument events, and
terminal `response.completed` / `response.incomplete` / `response.failed`
events.

For a streaming request, the server sends the SSE response headers when
prefill progress begins and writes an SSE comment keepalive every five seconds
until the first generated event. This keeps slow long-context prefill from
tripping ordinary client or proxy idle timers without changing inference or
socket-stall limits. Non-streaming requests cannot use this mechanism; clients
running long local agent prompts should stream and set an appropriately long
stream-idle timeout.

For browser JavaScript clients served from another origin, start the server with
`--cors` to emit `Access-Control-Allow-*` headers. This only changes HTTP
headers; it does not expose the server on the LAN. Use `--host 0.0.0.0`
explicitly when remote machines should be able to connect.

### Tool call handling and canonicalization

DeepSeek V4 emits tool calls as [DSML text](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/encoding/README.md). Agent clients do not send that
same text back on the next request: they send normalized OpenAI/Anthropic JSON
tool-call objects. **If the server re-rendered those objects slightly
differently, the rendered byte prefix would no longer match the live KV
checkpoint** and the next turn would have to be rebuilt.

The first line of defense is exact replay. Every tool call gets an unguessable
API tool ID, and the server remembers `tool id -> exact sampled DSML block` in
a bounded in-memory map backed by radix trees. When the client later sends that
tool ID back, the prompt renderer uses the exact DSML bytes the model sampled,
not a freshly formatted approximation. This map can also be saved inside KV
cache files, so exact replay survives server restarts for cached histories.

**Canonicalization is only the backup path for stateless chat-style replay**.
If exact DSML is unavailable, the server renders a deterministic form, compares
it with cached state, and may restore an older disk checkpoint before replaying
the suffix. Live Responses and Anthropic tool-result turns instead trust their
matching protocol IDs and append only the new suffix to the sampled KV frontier;
they do not rebuild that frontier to resemble client-visible JSON.

During generation, the server also treats DSML syntax differently from payload.
When the model is emitting stable protocol structure such as DSML tags,
parameter headers, JSON punctuation, or closing markers, sampling is forced to
`temperature=0` so the tool call stays parseable. This greedy mode does **not**
apply to argument payloads: `string=true` parameter bodies and JSON string
values, including file contents and edit text, use the request's normal sampling
settings. That separation is important: deterministic decoding is helpful for
syntax, but can create repeated text when applied to long code or file bodies.

Minimal OpenAI example:

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"deepseek-v4-flash",
    "messages":[{"role":"user","content":"List three Redis design principles."}],
    "stream":true
  }'
```

### Agent Client Usage

`ds4-server` can be used by local coding agents that speak OpenAI-compatible
chat completions. Start the server first, and set the client context limit no
higher than the `--ctx` value you started the server with:

```sh
./ds4-server \
  -m /absolute/path/to/QUALIFIED-DS4-ExpertMajor-v2.gguf \
  --ctx 100000 --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
```

You can use larger context and larger cache if you wish. Full context of
1M tokens is going to use more or less 26GB of memory (compressed indexer
alone will be like 22GB), so configure a context which makes sense in
your system. With 128GB of RAM you would run the 2-bit quants, which are
already 81GB, 26GB are going to be likely too much, so a context window
of 100~300k tokens is wiser. However users reported being able to run 2bit
quants with 250k ctx window in a Macs with just 96GB of system memory: make sure
to kill processes that use too much memory, if you plan doing so ;)

The `384000` output limit below avoids token caps since the model is able
to generate very long replies otherwise (up to 384k tokens). The server
still stops when the configured context window is full.

For **opencode**, add a provider and agent entry to
`~/.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ds4": {
      "name": "ds4.c (local)",
      "npm": "@ai-sdk/openai-compatible",
      "options": {
        "baseURL": "http://127.0.0.1:8000/v1",
        "apiKey": "dsv4-local"
      },
      "models": {
        "deepseek-v4-flash": {
          "name": "DeepSeek V4 Flash (ds4.c local)",
          "limit": {
            "context": 100000,
            "output": 384000
          }
        }
      }
    }
  },
  "agent": {
    "ds4": {
      "description": "DeepSeek V4 Flash served by local ds4-server",
      "model": "ds4/deepseek-v4-flash",
      "temperature": 0
    }
  }
}
```

For **Pi**, add a provider to `~/.pi/agent/models.json`:

```json
{
  "providers": {
    "ds4": {
      "name": "ds4.c local",
      "baseUrl": "http://127.0.0.1:8000/v1",
      "api": "openai-completions",
      "apiKey": "dsv4-local",
      "compat": {
        "supportsStore": false,
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": true,
        "supportsUsageInStreaming": true,
        "maxTokensField": "max_tokens",
        "supportsStrictMode": false,
        "thinkingFormat": "deepseek",
        "requiresReasoningContentOnAssistantMessages": true
      },
      "models": [
        {
          "id": "deepseek-v4-flash",
          "name": "DeepSeek V4 Flash (ds4.c local)",
          "reasoning": true,
          "thinkingLevelMap": {
            "off": null,
            "minimal": "low",
            "low": "low",
            "medium": "medium",
            "high": "high",
            "xhigh": "xhigh"
          },
          "input": ["text"],
          "contextWindow": 100000,
          "maxTokens": 384000,
          "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0
          }
        }
      ]
    }
  }
}
```

Optionally make it the default Pi model in `~/.pi/agent/settings.json`:

```json
{
  "defaultProvider": "ds4",
  "defaultModel": "deepseek-v4-flash"
}
```

For **Codex CLI**, use the Responses wire API:

```toml
[model_providers.ds4]
name = "DS4"
base_url = "http://127.0.0.1:8000/v1"
wire_api = "responses"
stream_idle_timeout_ms = 1000000
```

Then run:

```sh
codex --model deepseek-v4-flash -c model_provider=ds4
```

For **Claude Code**, use the Anthropic-compatible endpoint. A wrapper like this
matches the local `~/bin/claude-ds4` setup:

```sh
#!/bin/sh
unset ANTHROPIC_API_KEY

export ANTHROPIC_BASE_URL="${DS4_ANTHROPIC_BASE_URL:-http://127.0.0.1:8000}"
export ANTHROPIC_AUTH_TOKEN="${DS4_API_KEY:-dsv4-local}"
export ANTHROPIC_MODEL="deepseek-v4-flash"

export ANTHROPIC_CUSTOM_MODEL_OPTION="deepseek-v4-flash"
export ANTHROPIC_CUSTOM_MODEL_OPTION_NAME="DeepSeek V4 Flash local ds4"
export ANTHROPIC_CUSTOM_MODEL_OPTION_DESCRIPTION="ds4.c local GGUF"

export ANTHROPIC_DEFAULT_SONNET_MODEL="deepseek-v4-flash"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="deepseek-v4-flash"
export ANTHROPIC_DEFAULT_OPUS_MODEL="deepseek-v4-flash"
export CLAUDE_CODE_SUBAGENT_MODEL="deepseek-v4-flash"

export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK=1
export CLAUDE_STREAM_IDLE_TIMEOUT_MS=600000

exec "$HOME/.local/bin/claude" "$@"
```

Claude Code may send a large initial prompt, often around 25k tokens, before it
starts doing useful work. Keep `--kv-disk-dir` enabled: after the first expensive
prefill, the disk KV cache lets later continuations or restarted sessions reuse
the saved prefix instead of processing the whole prompt again.

## Thinking Modes

DeepSeek V4 Flash has distinct non-thinking, thinking, and Think Max modes.
The server defaults to thinking mode. `reasoning_effort=max` requests Think
Max, but it is only applied when the context size is large enough for the model
card recommendation; smaller contexts fall back to normal thinking. OpenAI
`reasoning_effort=xhigh` still maps to normal thinking, not Think Max.

For direct replies, use `thinking: {"type":"disabled"}`, `think:false`, or a
non-thinking model alias such as `deepseek-chat`.

## Disk KV Cache

Chat/completion APIs are stateless: agent clients usually resend the whole
conversation every request. `ds4-server` first tries the cheap exact token-prefix
check, then falls back to comparing rendered prompt bytes with decoded
checkpoint bytes. The live in-memory checkpoint covers the current session; the
disk KV cache makes useful prefixes survive session switches and server
restarts.

For RAM reasons there is currently only one live KV cache in memory. When a new
unrelated session replaces it, the old checkpoint can only be resumed without
re-processing if it was written to the disk KV cache. In other words, memory
cache handles the active session; disk cache is the resume mechanism for
different sessions.

Enable it with:

```sh
./ds4-server \
  -m /absolute/path/to/QUALIFIED-DS4-ExpertMajor-v2.gguf \
  --kv-disk-dir /tmp/ds4-kv --kv-disk-space-mb 8192
```

The cache key is the SHA1 of the rendered byte prefix, and files are named
`<sha1>.kv`. The DS4 payload still stores the exact token IDs and graph state
for that prefix. This matters for continued chats: the model may have generated
one token whose decoded text is later sent back by a client as two canonical
prompt tokens. A rendered byte-prefix hit can still reuse the checkpoint and
tokenize only the new suffix.
The file is intentionally written with ordinary `read`/`write` I/O, not
`mmap`, so restoring cache entries does not add more VM mappings to a process
that already maps the model.

Tool calls also keep a bounded exact-DSML replay map keyed by unguessable tool
IDs, so client JSON history can be rendered back to the exact sampled text. The
RAM map keeps up to 100000 IDs by default; tune it with `--tool-memory-max-ids`.
Use `--disable-exact-dsml-tool-replay` to disable this and fall back to
canonical JSON-to-DSML rendering.

On disk, a cache file is:

```text
KVC fixed header, 48 bytes
u32 rendered_text_bytes
rendered_text_bytes of UTF-8-ish token text
DS4 session payload, payload_bytes from the KVC header
optional tool-id map section
```

The fixed header is little-endian:

```text
0   u8[3]  magic = "KVC"
3   u8     version = 1
4   u8     routed expert quant bits, currently 2 or 4
5   u8     save reason: 0 unknown, 1 cold, 2 continued, 3 evict, 4 shutdown
6   u8     extension flags, bit 0 = appended tool-id map
7   u8     reserved
8   u32    cached token count
12  u32    hit count
16  u32    context size the snapshot was written for
20  u8[4]  reserved
24  u64    creation Unix time
32  u64    last-used Unix time
40  u64    DS4 session payload byte count
```

The rendered text is the tokenizer-decoded text for the cached token prefix.
It is both the human-inspectable prefix and the lookup identity: its SHA1 is
the filename, and a file is reusable only when those bytes are a prefix of the
incoming rendered prompt. After load, the exact checkpoint tokens from the DS4
payload remain authoritative, and only the incoming text suffix after the cached
bytes is tokenized.

The optional tool-id map is present only when header extension bit 0 is set.
Appended sections use fixed bit order, so future extension bits can add fields
without ambiguity. The map stores unguessable API tool call IDs back to the
exact DSML block the model sampled. Only mappings whose DSML block is present
in the rendered cached text are stored. This lets restarted servers render
later client history byte-for-byte like the original model output, even if the
client reorders JSON arguments.

The current tool-id map section is:

```text
0   u8[3]  magic = "KTM"
3   u8     version = 1
4   u32    entry count

For each entry:
0   u32    tool id byte length
4   u32    sampled DSML byte length
8   bytes  tool id
... bytes  exact sampled DSML block
```

The section is auxiliary replay memory, not model state. A cache hit restores
the session payload first, then loads the map if present. Before rendering a
request, the server can also scan cache files for the tool IDs present in the
client history and load just those mappings, so an exact DSML replay can survive
server restarts even when the matching KV snapshot is not the one ultimately
used for the rendered-prefix hit.

The DS4 session payload starts with thirteen little-endian `u32` fields:

```text
0   magic = "DSV4"
1   payload version = 2
2   saved context size
3   prefill chunk size
4   raw KV ring capacity
5   raw sliding-window length
6   compressed KV capacity
7   checkpoint token count
8   layer count
9   raw/head KV dimension
10  indexer head dimension
11  vocabulary size
12  live raw rows serialized below
```

Then it stores:

- `u32[token_count]` checkpoint token IDs.
- `float32[vocab_size]` logits for the next token after that checkpoint.
- `u32[layer_count]` compressed attention row counts.
- `u32[layer_count]` ratio-4 indexer row counts.
- For every layer: the live raw sliding-window KV rows, written in logical
  position order rather than physical ring order.
- For compressed layers: live compressed KV rows and compressor frontier
  tensors.
- For ratio-4 compressed layers: live indexer compressed rows and indexer
  frontier tensors.

The logits are raw IEEE-754 `float32` values from the host `ds4_session`
buffer. They are saved immediately after the checkpoint tokens so a loaded
snapshot can sample or continue from the exact next-token distribution without
running one extra decode step. MTP draft logits/state are not persisted; after
loading a disk checkpoint the draft state is invalidated and rebuilt by normal
generation.

The tensor payload is DS4-specific KV/session state, not a generic inference
graph dump. It is expected to be portable only across compatible `ds4.c`
builds for this model layout.

The cache stores checkpoints at four moments:

- `cold`: after a long first prompt reaches a stable prefix, before generation.
- `continued`: when prefill or generation reaches the next absolute aligned frontier.
- `evict`: before an unrelated request replaces the live in-memory session.
- `shutdown`: when the server exits cleanly.

Cold saves intentionally trim a small token suffix and align down to a prefill
chunk boundary. This avoids common BPE boundary retokenization misses when a
future request appends text to the same prompt. The defaults are conservative:
store prefixes of at least 512 tokens, cold-save prompts up to 30000 tokens,
trim 32 tail tokens, and align to 2048-token chunks. The important knobs are:

Continued saves use the same alignment and are written only when the live graph
naturally reaches an absolute frontier. With the defaults this means roughly
every 10k tokens, independent of where the first cold checkpoint landed, so long
generations leave restart points behind without persisting the fragile final few
tokens.

- `--kv-cache-min-tokens`
- `--kv-cache-cold-max-tokens`
- `--kv-cache-continued-interval-tokens`
- `--kv-cache-boundary-trim-tokens`
- `--kv-cache-boundary-align-tokens`
- `--tool-memory-max-ids`
- `--disable-exact-dsml-tool-replay`

By default, checkpoints may be reused across the 2-bit and 4-bit routed-expert
variants if the rendered prefix matches. Use `--kv-cache-reject-different-quant`
when you want strict same-quant reuse only.

The cache directory is disposable. If behavior looks suspicious, stop the
server and remove it. You can investigate what is cached with hexdump as
the kv cache files include the verbatim prompt cached.

## Backends

Production inference for Qwen, DeepSeek, and GLM ExpertMajor v2 artifacts is
local Apple Metal:

```sh
./ds4 -m /absolute/path/to/MODEL-DS4-ExpertMajor-v2.gguf -p "Hello"
```

CPU remains useful for build and model-free diagnostics:

```sh
make cpu
build/cpu-$(uname -m)/bin/ds4 --build-info
```

Do not treat a successful CPU build as model admission. ExpertMajor v2 inference
fails closed on CPU; there is no canonical, v1, or sidecar fallback.

CUDA and ROCm source, tests, and build targets are frozen outside the active
tree. Their last pre-removal implementation is recoverable from Git commit
`d8d673858f90834522bbe878951a534d8c6508b4`. Restoring either backend requires
the ADR, ownership, correctness, performance, and complete reactivation gates
in `QA_BEFORE_RELEASES.md`; a historical build or benchmark is not support.

## Steering

This project supports steering with single-vector activation directions; see the
`dir-steering` directory for more information. This follows the core idea of the
[Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)
paper. You can use it to make the model more or less verbose, less likely to
answer programming questions if it is a chatbot for your car rental web site,
and so forth, much faster than fine-tuning.
This is also useful for cybersecurity researchers who want to reduce a model's
willingness to provide dual-use or offensive security guidance.

## Test Vectors

`tests/test-vectors` contains short and long-context continuation vectors
captured from the official DeepSeek V4 Flash API. The requests use
`deepseek-v4-flash`, greedy decoding, thinking disabled, and the maximum
`top_logprobs` slice exposed by the API. Local vectors are generated with
`./ds4 --dump-logprobs` and compared by token bytes, so tokenizer/template or
attention regressions show up before they become long generation failures. The
C runner pins `DS4_METAL_PREFILL_CHUNK=2048` for this strict API-vector
comparison.

All project tests are driven by the C runner, with a small `ds4-eval`
extractor self-test run first:

```sh
DS4_TEST_MODEL=/absolute/path/to/QUALIFIED-DEEPSEEK-FLASH-DS4-ExpertMajor-v2.gguf make test
DS4_TEST_MODEL=/absolute/path/to/QUALIFIED-DEEPSEEK-FLASH-DS4-ExpertMajor-v2.gguf ./ds4_test --logprob-vectors
./ds4_test --server
```

## Debugging Notes

When a generation looks wrong, three small tools are usually enough to get a
first answer:

```sh
./ds4 -m /absolute/path/to/QUALIFIED-DS4-ExpertMajor-v2.gguf --dump-tokens -p "..."
./ds4 -m /absolute/path/to/QUALIFIED-DS4-ExpertMajor-v2.gguf --dump-logprobs /tmp/out.json --logprobs-top-k 20 --temp 0 -p "..."
./ds4 -m /absolute/path/to/QUALIFIED-DS4-ExpertMajor-v2.gguf --dump-logits /tmp/logits.json --nothink --prompt-file prompt.txt
./ds4-server -m /absolute/path/to/QUALIFIED-DS4-ExpertMajor-v2.gguf --trace /tmp/ds4-trace.txt ...
```

- `--dump-tokens` tokenizes the `-p` or `--prompt-file` string exactly as
  written, recognizes DS4 protocol specials, and then exits before inference
  starts. For example, the DSML tool close marker starts as two tokens: `</`
  and `｜DSML｜`.
- `--dump-logprobs` stores a greedy continuation with the top local
  alternatives at each step, which helps separate sampling choices from
  logit/model issues.
- `ds4-server --trace` writes the rendered prompts, cache decisions, generated
  text, and tool-parser events for a whole agent session.
