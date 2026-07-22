# DS4 GGUF Tools

This directory contains the offline tools used to build and evaluate the
DeepSeek V4, GLM 5.2, and Qwen3.6 GGUF files supported by `ds4`.

The important pieces are:

- `deepseek4-quantize.c`: C HF-safetensors to GGUF quantizer.
- `quants.[ch]`: the deliberately small local quantization implementation used
  by the quantizer.  It implements the DS4 output formats we actually ship:
  `q8_0`, `q4_K`, `q2_K`, and `iq2_xxs`.
- `ds4-expert-major.py`: deterministic canonical-to-native layout converter
  and byte-level verifier for DeepSeek, GLM, and Qwen
  `ds4.expert_major.v2` GGUFs.
- `imatrix/`: dataset and instructions for collecting routed-MoE activation
  importance with `ds4`.
- `quality-testing/`: prompts and scripts used to compare local GGUF variants
  against official DeepSeek V4 Flash continuations.

## Build

```sh
make -C gguf-tools
```

The quantizer is plain C and does not link GGML.  GGUF metadata handling,
safetensors loading, FP4/FP8 dequantization, and the quantizers used by our Q2
and Q4 recipes live in this directory.

## Build a DS4-native ExpertMajor v2 GGUF

Reorder an already qualified DeepSeek, GLM, or Qwen GGUF without changing
quantization:

```sh
python3 gguf-tools/ds4-expert-major.py inspect MODEL.gguf
python3 gguf-tools/ds4-expert-major.py build \
  MODEL.gguf MODEL-DS4-ExpertMajor-v2.gguf
python3 gguf-tools/ds4-expert-major.py verify \
  MODEL.gguf MODEL-DS4-ExpertMajor-v2.gguf
```

The build includes a full verification unless `--skip-verify` is explicitly
used for disposable development output. Publication must never use that flag.
The output stores routed weights once and is executable by this fork only on
Apple Metal. Current inference deliberately rejects canonical routed layouts,
ExpertMajor v1, external sidecars, CPU, CUDA, ROCm, and distributed execution.
Canonical files remain offline converter inputs. Format invariants, runtime
limits, and family qualification are documented in
[`../docs/expert-major-v2-roadmap.md`](../docs/expert-major-v2-roadmap.md),
[`../docs/deepseek-expert-major-v2.md`](../docs/deepseek-expert-major-v2.md),
and [`../docs/qwen-expert-major-store.md`](../docs/qwen-expert-major-store.md).

The supported routed quant types are Q2_K, Q4_K, Q5_K, Q6_K, and IQ2_XXS.
Family identity, routed layer inventory, expert counts, component geometry,
manifest digest, payload digest, source identity, and every copied tensor are
verified fail-closed. Build outputs are installed atomically on the destination
filesystem.

Run a completed artifact without conversion flags:

```sh
./ds4 -m /absolute/path/to/MODEL-DS4-ExpertMajor-v2.gguf --ctx 8192
```

## Build the pinned DeepSeek MLX affine2 routed store

The affine2 writer consumes the exact pinned
`mlx-community/DeepSeek-V4-Flash-2bit-DQ` checkout. It first validates the Git
origin and revision, config and index hashes, all committed Git LFS object
identities, every routed safetensor dtype/shape/extent, and destination free
space. Hydrate all 19 shards before writing; the command never downloads them.

```sh
python3 gguf-tools/ds4-expert-major.py \
  repack-deepseek-mlx-affine2 \
  --dry-run \
  --expected-revision 722bf559b7de93575b2320973cf2002e05bfe6c9 \
  /path/to/DeepSeek-V4-Flash-2bit-DQ

python3 gguf-tools/ds4-expert-major.py \
  repack-deepseek-mlx-affine2 \
  --resume \
  --expected-revision 722bf559b7de93575b2320973cf2002e05bfe6c9 \
  /path/to/DeepSeek-V4-Flash-2bit-DQ \
  /Volumes/SSD/DeepSeek-V4-Flash-MLX-Affine2-ExpertMajor-v2.store
```

The writer keeps only one expert component in memory, checkpoints completed
layers when `--resume` is enabled, verifies the final payload byte-for-byte by
default, and installs it with a same-filesystem rename. Ordinary failures
without `--resume` remove the partial output. `--skip-verify` is diagnostic
only and must not be used for publication.

The result is the standalone opaque `DS4EXPV2` store, not an executable GGUF.
Verify it against the same hydrated donor with:

```sh
python3 gguf-tools/ds4-expert-major.py \
  verify-deepseek-mlx-affine2 \
  --expected-revision 722bf559b7de93575b2320973cf2002e05bfe6c9 \
  /path/to/DeepSeek-V4-Flash-2bit-DQ \
  /Volumes/SSD/DeepSeek-V4-Flash-MLX-Affine2-ExpertMajor-v2.store
```

Embed that verified store as the sole `ds4.expert_major.v2` I8 tensor in an
existing qualified DeepSeek ExpertMajor v2 GGUF. This reconstructs the GGUF
directory and aligned tensor offsets into a new file; it never edits the base
model or standalone store in place:

```sh
python3 gguf-tools/ds4-expert-major.py \
  embed-deepseek-mlx-affine2 --dry-run \
  /path/to/DeepSeek-base-ExpertMajor-v2.gguf \
  /Volumes/SSD/DeepSeek-V4-Flash-MLX-Affine2-ExpertMajor-v2.store

python3 gguf-tools/ds4-expert-major.py \
  embed-deepseek-mlx-affine2 \
  /path/to/DeepSeek-base-ExpertMajor-v2.gguf \
  /Volumes/SSD/DeepSeek-V4-Flash-MLX-Affine2-ExpertMajor-v2.store \
  /Volumes/SSD/DeepSeek-V4-Flash-MLX-Affine2-DS4.gguf
```

The default publication path hashes the base GGUF, validates the standalone
payload digest while copying, then byte-compares every non-routed tensor and
the complete embedded store before hashing and atomically installing the final
GGUF. The compatibility gate requires identical DeepSeek layer/expert metadata
and component dimensions, one input store, one output store, and no duplicate
canonical routed tensors. The standalone `.store` can be removed after the
final GGUF and its recorded hashes have been independently retained.

```sh
python3 gguf-tools/ds4-expert-major.py \
  verify-deepseek-mlx-affine2-gguf \
  /path/to/DeepSeek-base-ExpertMajor-v2.gguf \
  /Volumes/SSD/DeepSeek-V4-Flash-MLX-Affine2-ExpertMajor-v2.store \
  /Volumes/SSD/DeepSeek-V4-Flash-MLX-Affine2-DS4.gguf
```

## Generate An Imatrix

First regenerate or inspect the calibration dataset:

```sh
python3 gguf-tools/imatrix/dataset/build_ds4_imatrix_dataset.py
```

Then collect activation statistics with the DS4 runtime:

```sh
./ds4 \
  -m gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf \
  --imatrix-dataset gguf-tools/imatrix/dataset/rendered_prompts.txt \
  --imatrix-out gguf/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4.dat \
  --ctx 32768
```

The imatrix file is useful immediately with this DS4 quantizer.  Generic GGUF
tools need DS4-specific tensor-name mapping and per-expert slicing before they
can use it correctly.  The accepted imatrix format is the legacy llama.cpp
binary `.dat` file emitted by `ds4 --imatrix-out`.

Generating this `.dat` file locally is possible, but slow: it runs the DS4
prefill graph over the full calibration corpus and reads routed-MoE activation
statistics back from the GPU.  The latest published imatrix-generated GGUF files
are available in the antirez Hugging Face repository:

```text
https://huggingface.co/antirez/deepseek-v4-gguf/tree/main
```

## Generate Q2 And Q4 GGUFs

The template GGUF supplies metadata, tokenizer, tensor order, and logical
shapes.  Tensor bytes are regenerated from the Hugging Face safetensors.  Full
generation is intentionally offline and heavy: expect roughly 80-90 GB outputs
for the 2-bit template family and roughly 150-170 GB for the 4-bit routed-expert
family, plus enough free disk for the temporary output.  Use `--dry-run` and
`--compare-tensor` before starting a full write, and use `--overwrite` only when
you really mean to replace an existing GGUF.

Q2 routed experts with imatrix:

```sh
gguf-tools/deepseek4-quantize \
  --hf ../deepseek-v4-quants/hf/DeepSeek-V4-Flash \
  --template gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf \
  --out gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --imatrix gguf/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4.dat
```

Q4 routed experts with imatrix:

```sh
gguf-tools/deepseek4-quantize \
  --hf ../deepseek-v4-quants/hf/DeepSeek-V4-Flash \
  --template gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf \
  --out gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2-imatrix.gguf \
  --imatrix gguf/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4.dat
```

You can override tensor families:

```sh
--experts iq2_xxs
--routed-w2 q2_k
--attention-proj q8_0
--shared q8_0
--output q8_0
```

Useful checks before writing a full model:

```sh
gguf-tools/deepseek4-quantize \
  --hf ../deepseek-v4-quants/hf/DeepSeek-V4-Flash \
  --template MODEL.gguf \
  --compare-tensor blk.0.attn_q_a.weight
```

`--compare-tensor` regenerates a single tensor and byte-compares it against the
template or `--compare-gguf`.  `--threads N` controls routed-expert workers.

## When No Imatrix Is Given

`iq2_xxs` requires an importance vector.  If `--imatrix` is not provided and
the target type requires one, `deepseek4-quantize` computes a synthetic fallback
from the dequantized weight itself:

```text
importance[column] = sum(row[column]^2) over all rows
```

This is a weight-energy heuristic.  It is not as good as measuring real DS4
activations, but it gives the quantizer a stable column weighting and was good
enough for the first working 2-bit GGUFs.

## Quality Testing

See `quality-testing/README.md`.  The short version is:

```sh
python3 gguf-tools/quality-testing/collect_official.py
make -C gguf-tools quality-score
gguf-tools/quality-testing/score_official MODEL.gguf gguf-tools/quality-testing/data/manifest.tsv /tmp/model.tsv 4096
python3 gguf-tools/quality-testing/compare_scores.py /tmp/old.tsv /tmp/new.tsv
```
