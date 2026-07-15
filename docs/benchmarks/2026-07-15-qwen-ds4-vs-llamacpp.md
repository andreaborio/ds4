# Qwen3.6 DS4 versus llama.cpp on 16 and 64 GiB Macs — 2026-07-15

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
