# Qwen3.6 DS4 versus llama.cpp on 16 and 64 GiB Macs — 2026-07-15

> [!IMPORTANT]
> This is historical benchmark evidence. ExpertMajor v1 and lower-memory
> runtime claims in this document are superseded by the ExpertMajor v2-only
> contract; they do not describe current support. Environment names and
> commands retained below describe the retired test lane, not options accepted
> or recommended by current `main`.

This note records a same-artifact comparison between DS4 and the official
llama.cpp macOS arm64 release. It separates a real, identical-prompt CLI run
from `llama-bench`: the latter is useful as a llama.cpp microbenchmark, but its
synthetic prompt path is not directly comparable with DS4's sampled CLI path.

## Common artifact and runtime identity

- Model: `Qwen3.6-35B-A3B-ds4-Q4_K_S.gguf`.
- GGUF size: 20,808,563,424 bytes; mapped tensor payload: 19.37 GiB.
- GGUF SHA-256:
  `c33efb67bde86c9ba1f9e79c2dc42627170963bef0e915ab9b91a55cfb6d0fcd`.
- DS4: `fc4badded643`, Metal arm64.
- llama.cpp: [official release `b10016`](https://github.com/ggml-org/llama.cpp/releases/tag/b10016),
  commit `32b741c33`; release archive SHA-256:
  `845211ba3fd3fe5cf365de8ceaa0d73ef85d89830cf5dbddd5d645a2cdb8e09c`.
- No weights were changed or further quantized for this comparison.

## M5 Pro, 64 GiB: identical rendered prompt

The host was a MacBook Pro with an Apple M5 Pro (18 CPU cores, 20 GPU cores),
64 GiB unified memory, macOS 26.5.1 (`25F80`), internal SSD, AC power, and High
Power mode. No other model runtime was active.

Both runtimes used Metal, all model layers on the GPU, a 160-token context,
18 CPU helper threads, greedy sampling, and the same 43-token rendered prompt.
The canonical prompt tokenized to the same exact 43 token IDs in DS4 and
llama.cpp. Both reached the 96-token cap and produced the same visible Python
continuation through `return curr`; llama.cpp's CLI added one harness newline.

DS4 was explicitly page-touched before every retained run:

```sh
MODEL=/absolute/path/to/Qwen3.6-35B-A3B-ds4-Q4_K_S.gguf

DS4_QWEN_EXPERIMENTAL_METAL=1 \
DS4_METAL_MEMORY_REPORT=1 \
  /usr/bin/time -l ./ds4 \
    -m "$MODEL" --metal --resident --warm-weights --power 100 \
    -c 160 -n 96 -t 18 \
    --temp 0 --top-p 1 --min-p 0 --seed 1 --nothink \
    -p 'Scrivi solo codice Python: una funzione fibonacci(n) iterativa, con validazione per n negativo.'
```

llama.cpp used the model's canonical chat template with thinking disabled:

```sh
MODEL=/absolute/path/to/Qwen3.6-35B-A3B-ds4-Q4_K_S.gguf

/usr/bin/time -l ./llama-cli \
  -m "$MODEL" -c 160 -n 96 -t 18 -tb 18 \
  -ngl all -fa on -b 64 -ub 64 \
  --conversation --single-turn --reasoning off \
  -sys 'You are a helpful assistant' \
  -p 'Scrivi solo codice Python: una funzione fibonacci(n) iterativa, con validazione per n negativo.' \
  --temp 0 --top-k 0 --top-p 1 --min-p 0 -s 1 \
  --no-display-prompt -co off --show-timings
```

Three complete runs per runtime were retained. Aggregate throughput is
`139 / (43 / prefill + 96 / generation)`; it is not a rate emitted directly by
either CLI.

| Runtime | Prefill samples | Generation samples | Median prefill | Median generation | Median aggregate |
| --- | --- | --- | ---: | ---: | ---: |
| llama.cpp b10016 | 206.2 / 252.1 / 254.4 | 60.3 / 60.1 / 60.5 | **252.1 t/s** | 60.3 t/s | 78.62 t/s |
| DS4 fc4badd, page-touched resident | 216.40 / 218.77 / 218.30 | 63.94 / 63.46 / 64.19 | 218.30 t/s | **63.94 t/s** | **81.76 t/s** |

Relative to llama.cpp, DS4 was 13.4% slower in prompt prefill, 6.0% faster in
generation, and 4.0% faster on the complete 43+96-token workload. The first
llama.cpp prompt run was visibly colder, but it is retained and the median is
reported rather than discarded.

Both runtimes remained under normal macOS pressure. The global swapout counter
stayed exactly 196,863 and reported swap use stayed at 114.94 MiB. Maximum RSS
was 20,942,258,176 bytes for DS4 and 21,172,076,544 bytes for llama.cpp; both
reported zero process swaps. DS4's page-touch pass covered 19.37 GiB before
inference and is excluded from its prefill/generation rates.

### llama-bench reference

For reproducibility against other llama.cpp machines, the official benchmark
was also run with three repetitions:

```sh
./llama-bench \
  -m "$MODEL" -p 43 -n 96 -pg 43,96 -r 3 \
  -t 18 -ngl 99 -fa on -b 64 -ub 64 -mmp 1 -o json --progress
```

| llama-bench test | Samples | Mean | Median |
| --- | --- | ---: | ---: |
| `pp43` | 508.969 / 520.699 / 500.893 | 510.187 t/s | 508.969 t/s |
| `tg96` | 57.467 / 58.408 / 57.596 | 57.824 t/s | 57.596 t/s |
| `pp43+tg96` | 78.763 / 80.032 / 80.783 | 79.859 t/s | 80.032 t/s |

The `pp43` row is a synthetic batched prompt microbenchmark. It excludes the
real chat-template, final-logits, and sampling work represented by the CLI
table, so it must not be presented as a direct 508.969-versus-218.30 DS4
comparison. The combined and actual CLI rows are the more useful local-use
numbers.

## M5 Pro resident-prefill follow-up: paired Q4 gate/up

The production resident path was profiled on the same 43-token request before
changing its dispatch policy. The prompt is submitted as one Metal command
buffer and waits only after the complete graph. Its GPU duration was
209.238 ms versus about 212.65 ms of measured prefill wall time, so another
CPU queue or synchronization change was not the useful target. One chunk
contains 1,143 compute dispatches and 20 blits. Representative routed-MoE
stages took 2.649 ms in a Gated DeltaNet layer and 3.619 ms in a full-attention
layer, while causal convolution, the recurrent DeltaNet stage, and GQA took
0.136, 0.405, and 0.284 ms respectively in the same trace.

Inside the routed MoE, the Q4 gate and up projections were separate selected-
expert passes of about 0.9 ms each. The resident Qwen geometry can instead
reuse the existing paired Q4 selected-expert matvec: it loads each activation
row and selected expert ID once for both projections. The dispatch policy now
selects that kernel only when all of the following hold:

- quality mode and SSD streaming are off;
- gate and down tensors are Q4_K;
- the graph has 256 routed experts, top-8 selection, and more than four tokens;
- the already existing paired-Q4 pipeline is available.

No shader or model format changed. Decode remains on the one-token path, SSD
streaming cannot select the new branch, and
`DS4_QWEN_DISABLE_RESIDENT_PREFILL_PAIR_MV=1` restores the previous resident
dispatch for differential tests.

The final clean binary was tested in `A1/B1/B2/A2/A3/B3` order, where A set the
fallback variable and B used the new default. The machine was active during
this follow-up, so the controlled same-session delta is more useful than the
absolute rate or a comparison with the cleaner baseline above.

| Path | Prefill samples | Generation samples | Median prefill | Median generation |
| --- | --- | --- | ---: | ---: |
| Previous resident dispatch (A) | 212.93 / 208.46 / 209.34 | 58.51 / 58.85 / 58.71 | 209.34 t/s | 58.71 t/s |
| Paired Q4 gate/up (B) | 253.81 / 263.21 / 258.08 | 57.73 / 59.00 / 57.81 | **258.08 t/s** | 57.81 t/s |

The resident-prefill median improved by **23.3%**. The 0.90 t/s generation
difference is within observed run noise, and the changed condition is false
for decode. All six visible continuations were byte-identical with SHA-256
`a650b56ceb47dc8715f87c125c7eeab506bc4a510512cedbd190e38c46df5f33`.
Global swap use remained exactly 114.94 MiB, every process reported zero
swaps, and no benchmark process remained afterward.

The final next-token comparison against the fallback retained the same
argmax token, 20/20 top-20 overlap, 64/64 top-64 overlap, 98/100 top-100
overlap, cosine similarity 0.999253315635, RMSE 0.080182463, and maximum
absolute difference 0.410542610 over the 248,077 valid vocabulary entries.
Those values match the established resident batch-versus-scalar tolerance
(RMSE 0.07970 and maximum 0.40870 for this prompt).

Two scope checks were retained. A 28-token prompt improved by 5.0%, and a
136-token prompt spanning three chunks improved by 5.1%; the canonical
43-token request improved by 24.7% in that sweep. A second prototype fused
SwiGLU into the paired projection, but its complete-run median was 254.11 t/s
versus 255.51 t/s for pair-only. That roughly 1 t/s difference is noise and
did not justify the extra kernel wiring, so the fused prototype was removed.

### Physical 16 GiB SSD regression smoke

The final source was also copied to an isolated directory and built natively
on the connected M1 Pro 16 GiB host. DSBox reported `runtime: idle` before and
after the run, and no existing `ds4` process was touched. The canonical prompt
was run with forced SSD streaming, a conservative 321-expert cache, context
160, and a 32-token generation cap. It completed at 4.06 t/s prefill and
7.03 t/s generation. System-wide free memory started at 80%, reached a minimum
of 55%, and recovered to 76%; the global swapout counter did not move and swap
use decreased from 1,740.25 to 1,732.19 MiB. The sampled process RSS peaked at
about 678 MiB, DSBox remained idle, and no `ds4` process remained.

This is a no-regression smoke, not an SSD speed claim: the paired resident
branch is explicitly false in SSD mode, whose prefill remains scalar.

### 2026-07-18 production-main recheck

The same physical M1 Pro 16 GiB host was rechecked over the home LAN with the
unified production runtime at `bd62a0b`, macOS 26.5, and the machine on battery
at 56%. The test used the canonical migration artifact
`Qwen3.6-35B-A3B-ds4-Q4_K_S.gguf` (20,808,563,424 bytes, SHA-256
`c33efb67bde86c9ba1f9e79c2dc42627170963bef0e915ab9b91a55cfb6d0fcd`),
context 8,192, AUTO SSD streaming, and macro prefill 64. AUTO began with 321
cached experts for prefill and selected a 2,241-expert decode target. Different
short non-thinking prompts were used for every sample so that live KV reuse
could not inflate later measurements.

| Run | State | Prompt tokens | Prefill | Generated tokens | Generation |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | cold | 43 | 10.56 t/s | 64 | 8.24 t/s |
| 2 | warm | 45 | 13.85 t/s | 64 | 9.71 t/s |
| 3 | warm | 44 | 16.24 t/s | 64 | 9.83 t/s |
| 4 | warm | 46 | 12.61 t/s | 58 | 9.25 t/s |
| 5 | warm | 46 | 18.41 t/s | 53 | 10.29 t/s |

The four warm runs had medians of 15.04 t/s prefill and 9.77 t/s generation;
the five-run medians including the cold request were 13.85 and 9.71 t/s. Memory
pressure stayed normal, free memory moved from 76% to 49%, process RSS was about
3.70 GiB, swap use stayed at 830.06 MiB, and the global swapout counter remained
exactly 2,209,613 before and after the run. The first two repeated reference
prompts also preserved the output hashes produced by the earlier runtime.

Only the canonical migration GGUF was tested on this host. The native
ExpertMajor v1 artifact was published and runtime-supported in that historical
cohort, but was not copied
to this Mac because only 3.6 GB of disk space was free; these numbers therefore
must not be represented as a direct native-v1 speed measurement. The historical
4.06/7.03 smoke above remains useful as a deliberately constrained lower bound,
not as production-main throughput.

## M1 Pro, 16 GiB: bounded streaming versus mmap pressure

The second host was a physical MacBook Pro with an Apple M1 Pro 8-core,
16 GiB unified memory, and macOS 26.5. It used the same 20.81 GB GGUF.

DS4's pressure-aware AUTO mode selected SSD streaming with a bounded
1,281-expert cache (2.11 GiB). In the interleaved B/A/B cache experiment, four
AUTO generations averaged 9.63 t/s; pressure remained normal, availability
never fell below 46%, and neither swap use nor the global swapout counter
increased.

llama.cpp has [mmap and CPU-MoE placement](https://github.com/ggml-org/llama.cpp/blob/master/tools/cli/README.md),
but no bounded, persistent expert cache equivalent to DS4's SSD streaming
path; that remains an [open feature request](https://github.com/ggml-org/llama.cpp/issues/20757).
Three guarded attempts were stopped as soon as macOS entered elevated
pressure:

| llama.cpp b10016 setting | Smallest attempted workload | Outcome |
| --- | --- | --- |
| Default Metal autofit, mmap | `pp32`, then `tg8` | Did not complete benchmark 1/2; pressure elevated; swapouts rose by 178,744 pages and swap use grew by about 2.61 GiB |
| All 40 MoE layers on CPU, Metal for the rest | `pp1`, then `tg4`, scalar batch | Did not complete; pressure elevated; 376 additional swapout pages |
| Metal autofit with 4 GiB target margin | `pp1`, then `tg4`, scalar batch | Did not complete; pressure elevated; 1,824 additional swapout pages |

The primary llama.cpp command was:

```sh
./llama-bench \
  -m "$MODEL" -p 32 -n 8 -r 1 -t 8 \
  -ngl -1 -fa auto -mmp 1 --progress
```

The CPU-MoE and larger-fit-margin probes were:

```sh
./llama-bench \
  -m "$MODEL" -p 1 -n 4 -r 1 -t 8 \
  -ngl 99 -ncmoe 40 -b 1 -ub 1 -fa on -mmp 1 \
  --no-warmup --progress

./llama-bench \
  -m "$MODEL" -p 1 -n 4 -r 1 -t 8 \
  -ngl -1 -fitt 4096 -fitc 8192 \
  -b 1 -ub 1 -fa on -mmp 1 \
  --no-warmup --progress
```

There is therefore no safe llama.cpp throughput number for this artifact on
that 16 GiB host. “No result” is the measured result: mmap makes the model
file-backed, but it does not provide DS4's admission-controlled expert working
set, cache eviction, and expert-specific reads.

## Interpretation and limits

- On the 64 GiB host, resident decode is not presently a llama.cpp catch-up
  problem: DS4 is already 6.0% faster on the real deterministic request.
- Resident prefill still has measurable headroom. llama.cpp is 15.5% faster on
  the identical CLI prompt, while its much larger synthetic `pp43` number shows
  that harness choice can exaggerate the apparent gap.
- On the 16 GiB host, DS4's explicit expert cache is the enabling feature.
  llama.cpp's current mmap and CPU-MoE controls did not establish a safe
  runnable configuration for this GGUF.
- These results cover one Qwen3.6 Q4_K_S artifact, short context, Metal, and two
  Apple hosts. They are not general claims about every model, quant, context,
  backend, or llama.cpp release.
- No llama.cpp run was allowed to continue under elevated pressure merely to
  obtain a throughput number.
