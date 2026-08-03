# Hebrus GGUF Tools

This directory contains the offline tools used to build and evaluate the
DeepSeek V4, GLM 5.2, and Qwen3.6 GGUF files supported by Hebrus.

The important pieces are:

- `deepseek4-quantize.c`: C HF-safetensors to GGUF quantizer.
- `quants.[ch]`: the deliberately small local quantization implementation used
  by the quantizer. It implements the Hebrus output formats we actually ship:
  `q8_0`, `q4_K`, `q2_K`, and `iq2_xxs`.
- `ds4-expert-major.py`: deterministic canonical-to-native layout converter
  and byte-level verifier for DeepSeek, GLM, and Qwen
  `ds4.expert_major.v2` GGUFs.
- `imatrix/`: dataset and instructions for collecting routed-MoE activation
  importance with Hebrus.
- `quality-testing/`: prompts and scripts used to compare local GGUF variants
  against official DeepSeek V4 Flash continuations.

## Build

```sh
make -C gguf-tools
```

The quantizer is plain C and does not link GGML.  GGUF metadata handling,
safetensors loading, FP4/FP8 dequantization, and the quantizers used by our Q2
and Q4 recipes live in this directory.

## Build a Hebrus ExpertMajor v2 GGUF

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
./hebrus -m /absolute/path/to/MODEL-DS4-ExpertMajor-v2.gguf --ctx 8192
```

### DSpark provenance gate

`deepseek4-quantize --dspark-support-only` is the sole generator for the
standalone composer input. It has a fixed 81-tensor inventory and quantization
recipe, and authenticates the final checkpoint revision, config, safetensors
index, and shards 46-48 before writing. It does not accept a template,
quantization override, imatrix, or provenance override.

The generated GGUF carries the independently pinned source revision plus the
complete SHA-256 values for config, index, and each source shard. The composer
checks those values against its own constants; metadata merely copied from a
support file is not accepted as proof of provenance.

Inspect the exact 5,989,114,912-byte output plan without reading the shards:

```sh
gguf-tools/deepseek4-quantize \
  --dspark-support-only \
  --hf /path/to/DeepSeek-V4-Flash-0731 \
  --dry-run
```

Authenticate all three local shards without generating output, then build the
support input atomically:

```sh
gguf-tools/deepseek4-quantize \
  --dspark-support-only \
  --hf /path/to/DeepSeek-V4-Flash-0731 \
  --check

gguf-tools/deepseek4-quantize \
  --dspark-support-only \
  --hf /path/to/DeepSeek-V4-Flash-0731 \
  --out /path/to/DeepSeek-V4-Flash-0731-DSpark-support.gguf
```

The source contract pins revision
`7872f01b1d1fe23eabc4c98b48bffcef5a386062`, config SHA-256
`6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023`,
index SHA-256
`98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`,
and shard SHA-256 values beginning `5db924ca`, `62816173`, and `cc43742b`.
The resulting GGUF is an authenticated offline input to
`ds4-expert-major.py`; it is not a production sidecar or runnable artifact.
Generation reauthenticates every input after conversion and before install.
The temporary output is exclusively created without following symlinks,
size/hash checked through its still-open descriptor, and installed directly
from that descriptor with true no-clobber semantics. Apple builds use
`fclonefileat`, falling back to an exclusive descriptor-to-descriptor copy
when the filesystem does not support cloning; non-Apple builds use the same
copy path. Cleanup removes the temporary pathname only while it
still identifies the owned inode, and directory installation is followed by
`fsync`. There is no destructive replacement mode: use a new output name or
remove an obsolete artifact explicitly before invoking the tool.

`build --dspark-support` and `verify --dspark-support` accept only the reviewed
final-0731 support artifact whose complete SHA-256 is
`aa2bd4b5b916e1aa0a01392d69cbdd9798a3f3050c29c22973c8ee4233af0413`.
It was generated from the pinned official inputs above. An independent
conversion with `antirez/deepseek-v4-gguf` commit
`54b36ed9ba42da31b24f2d1a5feb075c2475dbb1` reproduced the descriptors and
payload bytes of all 81 tensors exactly. The 640-byte whole-file difference is
authenticated Hebrus provenance metadata, not a weight difference.

The 5,989,114,272-byte support GGUF with SHA-256
`8b3adf5942bec22ae2ea867cd7079cf13530ba83ffcffaf00f5de48664a1a34e`
was published before the final 0731 checkpoint and identifies the preview
`DeepSeek-V4-Flash-DSpark` source. It is admitted only as a structural test
reference and is rejected for combined-artifact publication. Matching its
name, metadata, tensor inventory, and geometry does not prove that its weights
come from the final checkpoint.

No CLI hash override is provided. Runtime support remains gated by the
architecture in
[`../docs/adr/0008-deepseek-dspark-embedded-support-store.md`](../docs/adr/0008-deepseek-dspark-embedded-support-store.md).

Successful conversion and exact tensor reproduction do not qualify the
runtime. DSpark logits and draft decisions must still match the official
implementation and the MLX oracle. Then run target-only versus combined
quality and exact-output lanes, followed by the Apple 64 GiB AUTO-to-SSD 8K,
32K, and admitted-endpoint matrix. The reviewed digest is coded manually;
generation must never edit or derive the composer production pin.

## Generate An Imatrix

First regenerate or inspect the calibration dataset:

```sh
python3 gguf-tools/imatrix/dataset/build_ds4_imatrix_dataset.py
```

Then collect activation statistics with the Hebrus runtime:

```sh
./hebrus \
  -m gguf/DeepSeek-V4-Flash-Q4KExperts-F16HC-F16Compressor-F16Indexer-Q8Attn-Q8Shared-Q8Out-chat-v2.gguf \
  --imatrix-dataset gguf-tools/imatrix/dataset/rendered_prompts.txt \
  --imatrix-out gguf/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4.dat \
  --ctx 32768
```

The imatrix file is useful immediately with the Hebrus quantizer. Generic GGUF
tools need Hebrus-specific tensor-name mapping and per-expert slicing before they
can use it correctly.  The accepted imatrix format is the legacy llama.cpp
binary `.dat` file emitted by `hebrus --imatrix-out`.

Generating this `.dat` file locally is possible, but slow: it runs the Hebrus
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
`--compare-tensor` before starting a full write. Output creation is exclusive;
use a new name or explicitly remove an obsolete artifact before generation.

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

This is a weight-energy heuristic. It is not as good as measuring real Hebrus
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
