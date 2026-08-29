# Source register and evidence policy

## 1. Evidence labels

Every implementation comment or plan claim should be mentally tagged as one of:

- **UPSTREAM CONTRACT**: directly observed in a pinned official config, weight
  index, tokenizer, model implementation or technical report.
- **UPSTREAM IMPLEMENTATION**: behavior of llama.cpp, MLX or another engine;
  useful evidence, but not automatically the Hebrus contract.
- **HEBRUS CONTRACT**: required by repository ADRs, runtime support contract or
  this accepted implementation plan.
- **DERIVED**: arithmetic derived from closed dimensions; retain the formula.
- **HYPOTHESIS**: an optimization that must survive correctness and benchmark
  gates.

Do not turn a hypothesis into a constant merely because another engine uses it.

## 2. Pinned authoritative model sources

| Source | Pin | Use |
|---|---|---|
| [Qwen model repository](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) | revision `de4b8e4d43b917e7706784d8bb445c9af86a3540` | artifact identity, files, weights |
| [model config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json) | same revision | closed dimensions and layer types |
| [tokenizer config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/tokenizer_config.json) | same revision | regex, control IDs, template |
| [safetensors index](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/model.safetensors.index.json) | same revision | exact tensor inventory and source shards |
| [official code/report repository](https://github.com/QwenLM/Qwen3.8-Flash-Next) | retrieved 2026-08-29; implementation should record an exact commit when vendoring fixtures | architecture narrative and report |
| [official launch article](https://qwen.ai/blog?id=qwen3.8-flash-next) | 2026-08-27 | intended architecture and deployment positioning |

The source index declares `359,999,963,128` bytes in BF16 and 1,658 tensor
entries across 131 shards. The Hugging Face metadata reports roughly 180B
parameters: 125B main model plus 51B PLE table and auxiliary parameters. These
numbers are identity/sanity checks, not allocation requests.

## 3. Executable reference

The mathematical oracle is Hugging Face Transformers `qwen4_exp` at commit
`42ca97014c85d71a88ad60d55f08cb9fb4d26e2c`:

- [configuration_qwen4_exp.py](https://github.com/huggingface/transformers/blob/42ca97014c85d71a88ad60d55f08cb9fb4d26e2c/src/transformers/models/qwen4_exp/configuration_qwen4_exp.py)
- [modeling_qwen4_exp.py](https://github.com/huggingface/transformers/blob/42ca97014c85d71a88ad60d55f08cb9fb4d26e2c/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py)

Fixture generation records this commit, Python/package versions, device, dtype,
input IDs and configuration. A future Transformers update is not silently a new
oracle; change the pin and regenerate fixtures in a reviewed commit.

## 4. llama.cpp comparison source

Research used llama.cpp commit
`cc83d7b4824f73cfdda4dfbb47ee39804f71b328`:

- [`qwen4exp.cpp`](https://github.com/ggml-org/llama.cpp/blob/cc83d7b4824f73cfdda4dfbb47ee39804f71b328/src/models/qwen4exp.cpp)
- [`llama-memory-hybrid-idx.cpp`](https://github.com/ggml-org/llama.cpp/blob/cc83d7b4824f73cfdda4dfbb47ee39804f71b328/src/llama-memory-hybrid-idx.cpp)
- [`conversion/qwen4exp.py`](https://github.com/ggml-org/llama.cpp/blob/cc83d7b4824f73cfdda4dfbb47ee39804f71b328/conversion/qwen4exp.py)
- [Qwen3.8 Flash Next performance discussion](https://github.com/ggml-org/llama.cpp/discussions/27864)

Transferable ideas: cell-for-cell indexer cache ownership, raw-key caching,
lazy treatment of the huge PLE table, exact host-side hash arithmetic and strict
shape validation. Non-transferable without proof: generic ggml graph choices,
`mmap`-fault-driven random I/O, backend-specific sparse-mask behavior, and any
reported speed. Upstream issues around multi-segment state and long-context
backend behavior are explicit adversarial-test inputs.

## 5. MLX and Metal sources

Use official Apple/MLX sources for API and hardware behavior, and pin third-party
model ports separately as comparison implementations:

- [MLX repository](https://github.com/ml-explore/mlx)
- [MLX LM repository](https://github.com/ml-explore/mlx-lm)
- [MLX fast Metal kernels](https://github.com/ml-explore/mlx/tree/main/mlx/backend/metal/kernels)
- [Apple Metal resource-storage modes](https://developer.apple.com/documentation/metal/mtlresourceoptions)
- [Apple sparse heaps](https://developer.apple.com/documentation/metal/mtlsparsepagetables)
- [Metal I/O command queue](https://developer.apple.com/documentation/metal/mtliocommandqueue)
- [Apple M5 Pro announcement](https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/)

The most complete Apple-oriented port found is the merged
[MLX-VLM qwen4_exp implementation](https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/qwen4_exp),
introduced by commit
[`505267c`](https://github.com/Blaizzy/mlx-vlm/commit/505267caa84fb7ba89851719fbc2655a454ab2c8).
MLX-LM text-only support was still an open
[PR #1788](https://github.com/ml-explore/mlx-lm/pull/1788) during research.
MLX-VLM is useful for Apple mapping, batching and external PLE experiments, but
Transformers remains the mathematical oracle. Its QSA constructs a dense mask
before ordinary SDPA and its PLE path is synchronous/CPU-driven; neither is the
Hebrus performance design.

Additional primary comparison records:

- [MLX-VLM external PLE PR #2045](https://github.com/Blaizzy/mlx-vlm/pull/2045),
  including memory and cold-lookup measurements;
- [MLX-VLM MTP PR #2040](https://github.com/Blaizzy/mlx-vlm/pull/2040), useful
  only for later speculative hypotheses;
- [MNN 3.6.1 release](https://github.com/alibaba/MNN/releases/tag/3.6.1), an
  implementation source for M5-oriented low-bit Metal kernels, not a Qwen4Exp
  model oracle.

## 6. Hebrus authority

Repository-local authority, in precedence order:

1. `docs/contracts/RUNTIME_SUPPORT.md`.
2. Accepted ADRs, especially 0001, 0002, 0003, 0004 and 0006.
3. `CONTRIBUTING.md`, `QA.md`, `docs/qwen-expert-major-store.md`,
   `GOLD_METAL_SSD.md` and `docs/architecture/CODEMAP.md`.
4. This dossier once reviewed and accepted.
5. Current code behavior where it does not conflict with the above.

## 7. Reproducibility record required in every benchmark

Record: Hebrus commit and dirty state; artifact manifest hash; upstream model
revision; machine identifier; chip/GPU/core count; RAM; macOS and Xcode/Metal
versions; filesystem and free SSD space; power/thermal state; command; prompt
hash; context/decode counts; AUTO decision and byte budget; resident/SSD mode;
quant codecs; cold/warm status; wall times; token/s; peak memory; swap; PLE page
cache hit rate; expert cache hit rate; physical `pread` bytes/syscalls/time; and
the exact generated text/evidence record.
