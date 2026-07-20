# GLM 5.2 ExpertMajor v2 release qualification

Date: 2026-07-20

Host: Apple M5 Pro, 64 GiB unified memory

Backend: Apple Metal, AUTO resolved to SSD

Measured hot-path commit: `23a8446`

Release cleanup: `55d2bab` (deleted the unreachable install prototype)
Artifact: `GLM-5.2-DS4-ExpertMajor-v2-Q2_K.gguf`

## Scope

This gate answers three questions:

1. Does main consume the embedded interleaved ExpertMajor payload correctly?
2. Which prefill/decode loads are avoidable?
3. Which combination of proven changes gives the best 64 GiB policy without
   exposing a forest of startup flags?

This is not a cold-cache SSD benchmark. The model already existed on the Mac's
internal APFS volume. The campaign did not copy or rehash the complete file,
drop the system page cache, or run a synthetic storage stress loop.

```text
size    262147193504
sha256  7f5017e3076e706c78f2a5322b035a9e2f6519c65ff5b6be8b2d91aeff61505d
```

The exact remote Hugging Face object has the same size and content identifier;
the 244 GiB payload does not need another upload.

## Simple release command

Normal startup has no GLM tuning flags:

```sh
./ds4 \
  -m /absolute/path/to/GLM-5.2-DS4-ExpertMajor-v2-Q2_K.gguf \
  --ctx 8192
```

The deterministic benchmark adds only its prompt, sampling, and token cap:

```sh
./ds4 \
  -m /absolute/path/to/GLM-5.2-DS4-ExpertMajor-v2-Q2_K.gguf \
  --ctx 8192 \
  --prompt-file /tmp/glm-prefill-288-exact.txt \
  --temp 0 -n 32
```

Prompt SHA-256:
`8370b23334333501f547cc398225149c246e4523be24b53bf84a1baa8a1fa970`.
DS4 reported exactly 288 input tokens. Every accepted arm began with this exact
continuation:

```text
The user wants me to complete the function. The function computes count, mean,
and variance per numeric column using Welford's online algorithm. The return should
```

## Incremental result

| Arm | Cache | Prefill | Decode | Expert `pread` time | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Broken mainline port | 601 | 10.74 t/s | 1.27 t/s | 16.248 s | Correct text, physical advisory offsets wrong |
| Physical advisory translation + full ExpertMajor records | 601 | 10.60 t/s | 1.77 t/s | 9.178 s | Exact output |
| Above + no GLM slab `mlock` | 601 | 10.75 t/s | **1.81 t/s** | 9.276 s | Exact output, no new swap activity |
| AUTO adaptive candidate | 1,801 | 10.53 t/s | 1.73 t/s | 9.424 s | 22% fewer misses, slower end to end |
| Final simple command | 601 | **10.91 t/s** | 1.77 t/s | Profiling off | Exact output, AUTO selected every policy |
| Final source gate after compatibility cleanup | 601 | 10.63 t/s | **1.79 t/s** | Profiling off | Exact output; removed code was unreachable |

The first four rows were diagnostic runs and include detailed timing
instrumentation. The last two rows use the public command with no cache,
residency, profile, or ExpertMajor flags. Values are single controlled
observations, not medians; storage/cache state was not reset between them.

A prior qualification on rested internal storage recorded a median of 11.075
t/s prefill and 1.900 t/s decode at 528 experts. To check whether main had lost
that runtime performance, the old commit `08f3ebed` and the new port were built
cleanly and run back-to-back under the current post-campaign conditions:

| Runtime | Cache | Prefill | Decode |
| --- | ---: | ---: | ---: |
| Qualified `08f3ebed` | 528 | 10.65 t/s | 1.75 t/s |
| Mainline port | 528 | 10.63 t/s | 1.74 t/s |

This is 0.6% decode separation and demonstrates parity. It also shows why the
historical 1.90 median must not be relabeled as a current one-shot result. The
601-expert default is retained because it is the safe full-route floor and was
faster in the current lane.

## Root cause: an avoidable second I/O stream

ExpertMajor exposes canonical gate/up/down offsets to the graph but stores each
expert physically as one adjacent record. The authoritative cache loader
already translated those logical offsets. The ported router-ahead helper did
not: it issued `F_RDADVISE` on the model descriptor at the canonical offset.

That hint therefore warmed unrelated ranges while eight real expert reads were
competing for storage. Translating the hint through the embedded manifest cut
recorded expert `pread` time from 16.248 to 9.178 seconds and moved decode from
1.27 to 1.77 t/s. An unmappable hint is now skipped; the real read still fails
closed instead of falling back to canonical bytes.

## Prefill data flow

The long-prompt path performs these loads:

1. map and retain the fixed non-routed tensors;
2. validate the embedded manifest once;
3. for each routed layer, prepare one physical ExpertMajor layer span;
4. overlap preparation of layer `L+1` with grouped compute for layer `L`;
5. bind a complete physical address table to the grouped Q2_K kernel;
6. release the wrapper after its command buffer completes.

The important removal is three attempted canonical projection views per layer.
They do not exist in an ExpertMajor-only GLM. Layer-0 embedding is already
mapped, so the indexed prepare path also skips rereading it. Full-layer wrapper
lifetime is command-buffer bounded rather than model-lifetime bounded.

## Decode data flow

Each generated token routes 75 layers x 8 experts. At every routed layer:

1. Metal computes top-8 IDs;
2. cached records are rebound immediately;
3. missing records allocate/reuse contiguous slabs;
4. one worker reads one complete 12,386,304-byte gate+up+down record;
5. the address table is installed and paired gate/up plus down kernels run;
6. the slot becomes evictable only after the referencing GPU work is safe.

The new counters distinguish logical expert loads from OS calls. The full-record
run recorded exactly one successful `pread` syscall per completed load, proving
there was no hidden component fragmentation. GLM ExpertMajor slabs are not
`mlock`ed: on this path the lock setup cost was measurable while the process
still completed without new swap activity.

The old full-layer resolver also ran during selected-cache decode and generated
thousands of guaranteed misses. It is now bypassed entirely for layers that do
not have a live full-layer mapping.

## Cache policy on 64 GiB

AUTO first proves that current memory pressure can fit the generic 601-expert
floor. It then applies the measured GLM ExpertMajor 64 GiB policy:

- prefill cache: 601 experts;
- decode cache: 601 experts;
- routed cache target: 6.93 GiB;
- 8K startup plan: 17.28 GiB including KV, buffers, token embedding, routed
  cache, and overlapped-prefill reserve.

The 1,801-expert experiment is the useful counterexample. Its hit rate improved
and total misses fell, but decode regressed from 1.81 to 1.73 t/s. At 601, a
layer more often presents enough adjacent misses to keep the read workers busy;
at 1,801, fewer small miss sets serialize the same per-layer barrier. Larger
cache metadata and memory pressure add cost as well. Hit rate is therefore not
the optimization objective; end-to-end token latency is.

No 96 or 128 GiB number is inferred from the 64 GiB host. Those tiers retain
the ordinary pressure-admitted adaptive candidate until measured physically.

## Rejected decode arms

The campaign reused earlier GLM evidence rather than rerunning known losers:

| Arm | Evidence | Decision |
| --- | --- | --- |
| Predicted-expert install | About 75% accuracy, pool contention, 1.15 t/s | Removed from parser, backend API, and Metal implementation |
| Advisory lookahead > 1 | Prediction accuracy falls as layer distance grows | Keep lookahead 1 |
| More cache at 64 GiB | 1,801 experts: 1.73 t/s vs 1.81 at 601 | AUTO uses 601 |
| `F_NOCACHE` / aggressive `DONTNEED` | Prior campaigns regressed throughput materially | Removed from release path |
| MTLIO, QoS, sub-chunking, alternate LRU families | No repeatable end-to-end win in recorded campaigns | Do not expose as startup choices |
| MTP at about 55% acceptance | Extra expert I/O cost exceeds accepted-token gain | Off |

Wrong advisory predictions remain semantically safe because they never install
cache state. The install variant mutated cache state and therefore paid for
prediction misses twice; it is no longer selectable.

The release intentionally provides no compatibility route for that prototype,
older GLM sidecars, canonical GLM files, or prior ExpertMajor revisions.

## Qwen transfer analysis

The Qwen optimizations transfer as principles, not as identical constants:

- the embedded expert-major store removes sidecars and startup repacks;
- prefill and decode require separate schedules;
- physical addressing must be resolved once and reused by kernels and I/O;
- paired gate/up execution and grouped prefill reduce dispatch and traffic;
- phase-specific memory planning matters more than maximizing cache size.

One Qwen decision does not transfer: Qwen's smaller records benchmark best as
parallel component reads, while GLM's 11.81 MiB ExpertMajor record is fastest
as one contiguous read. The runtime now chooses by model/layout rather than a
global “best” I/O mode.

## Cross-engine audit

Open engines reinforce the phase split, but their multi-GPU CUDA/ROCm numbers
are not comparable to one 64 GiB Mac.

| Open-engine evidence | DS4 conclusion |
| --- | --- |
| [vLLM's GLM-5.2 recipe](https://recipes.vllm.ai/zai-org/GLM-5.2) combines compressed KV, fused expert work, and gated MTP | Compact DSA KV is already shipped; speculation stays off until acceptance repays expert I/O |
| [SGLang supports separate prefill/decode attention backends](https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/attention_backend.md) | Keep distinct Metal schedules and benchmark them independently |
| [SGLang releases](https://github.com/sgl-project/sglang/releases) and [vLLM releases](https://github.com/vllm-project/vllm/releases) repeatedly remove MoE copies and host synchronization | Profile selected-ID readback, address installation, and queue fences before adding another cache algorithm |
| [llama.cpp's two-tier expert-cache RFC](https://github.com/ggml-org/llama.cpp/issues/20757) explores protected hot slots, second-hit admission, and double buffering | Replay recorded GLM routes offline first; do not enable an unmeasured live policy |
| [MLX-LM large-model guidance](https://github.com/ml-explore/mlx-lm#large-models) recommends wiring weights/cache only when the model fits | A 244.14 GiB model cannot be wired into 64 GiB; selective pageable expert streaming remains necessary |

## Next incremental campaign

Future work should preserve the 601-expert baseline and combine only independent
winners:

1. profile decode stage walls without changing policy;
2. replay recorded routes offline for protected-hot/second-hit admission;
3. test a separate advisory lane or a gate/up-then-down bifasic load only if
   it does not contend with authoritative reads;
4. remove a host synchronization only after a model-free ordering proof;
5. test route/top-k, DSA indexer Q/K, and down-projection fusion separately;
6. combine accepted changes and measure each component's marginal gain;
7. repeat the cache sweep on real 96 and 128 GiB hosts rather than simulating
   their performance on this machine.

Promotion requires exact greedy text, numeric kernel parity, normal memory
pressure, zero new swap, and a repeatable end-to-end gain. A better isolated
hit rate or microkernel is not sufficient.
