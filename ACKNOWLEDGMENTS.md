# Acknowledgments

Hebrus is the result of an open-source lineage, not a clean-room rewrite. This
document records the principal projects and implementations that materially
shaped the current repository. It describes technical provenance; it does not
claim endorsement by any upstream author, project, company, or model provider.

## Project origin: `antirez/ds4`

Hebrus began as a fork of
[`antirez/ds4`](https://github.com/antirez/ds4) and retains substantial
core-engine implementation, architecture, utilities, and Git history from that
project. Salvatore Sanfilippo (antirez) and the upstream contributors created
the foundation on which this fork continues to build.

The fork has since concentrated on Apple Metal execution, embedded ExpertMajor
stores, adaptive residency, SSD streaming, additional qualified model families,
and the evidence and tooling required to maintain those paths. The repositories
have diverged, but the contribution policy still asks whether general,
reproducible improvements should also be proposed upstream. See
[`FORK_NOTES.md`](FORK_NOTES.md) for the current boundary.

## GGML and llama.cpp

[`llama.cpp`](https://github.com/ggml-org/llama.cpp) and
[`GGML`](https://github.com/ggml-org/ggml) are essential engineering references
for this repository. Their work informs GGUF handling, quantization formats and
tables, CPU quantized dot products, Metal kernels, converters, test methods, and
many implementation decisions.

Hebrus does not link against GGML or llama.cpp as a runtime dependency. Some
source-level pieces are retained or adapted under the MIT license, including
quant layouts and tables, CPU quant/dot logic, and selected kernel techniques.
The GGML authors' copyright notice remains in [`LICENSE`](LICENSE), and
source-specific attribution remains in the relevant files.

Pinned llama.cpp converter implementations are also used as independent Qwen
validation references. A reference implementation or fixture oracle is not the
same thing as a linked dependency.

## MLX and MLX-LM

[`MLX`](https://github.com/ml-explore/mlx) and
[`MLX-LM`](https://github.com/ml-explore/mlx-lm) informed the current Qwen MLX
affine4/group-64 storage layout, scheduling comparisons, and validation work.
The local Qwen candidate repacks routed weights from an MLX-format source into
the existing checksummed ExpertMajor v2 GGUF container.

Hebrus does not link the MLX libraries at runtime. This credit does not imply
Apple authorship, sponsorship, or endorsement of Hebrus.

## YaRN

The Metal RoPE implementation in `metal/dsv4_rope.metal` contains an algorithm
based on [`YaRN`](https://github.com/jquesnelle/yarn), with the source header's
MIT attribution to Jeffrey Quesnelle and Bowen Peng preserved.

## Bundled utilities

The repository includes:

- [`linenoise`](https://github.com/antirez/linenoise), by Salvatore Sanfilippo
  and Pieter Noordhuis, for terminal line editing;
- Rax, by Salvatore Sanfilippo, for radix-tree storage used by the server path.

These files carry BSD-style terms, including the applicable no-endorsement
language. Their exact bundled notices are collected in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and remain in the source
headers.

## Validation references and model ecosystems

Qwen numerical fixtures use a pinned
[Hugging Face Transformers implementation](https://github.com/huggingface/transformers/blob/4626421dc6b741a329300682a6408246ee465490/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py)
and a pinned
[llama.cpp Qwen converter](https://github.com/ggml-org/llama.cpp/blob/657e01125aa49577a62a5531fde24cbcc007006d/conversion/qwen.py)
as independent validation references. The collectors use local deterministic
fixtures and do not make those projects runtime dependencies.

The project also depends intellectually on the researchers and open-weight
model teams behind the DeepSeek, Qwen, and GLM families, and on the wider GGUF,
Metal, and local-inference communities. Exact model, artifact, revision, and
benchmark provenance belongs in the corresponding model documentation and
dated evidence records rather than in a general credit list.

## Development process

The long-form engine reference discloses substantial AI assistance during
development, with humans directing design, testing, benchmarking, and release
decisions. That assistance does not replace the authorship, copyright, or
license notices attached to upstream and third-party work.

## Licenses and model terms

The project-level software license is [`LICENSE`](LICENSE). Bundled third-party
files retain their own notices and conditions. Model weights, datasets, model
cards, and external repositories have their own terms; this repository's
software license does not grant rights to those assets.
