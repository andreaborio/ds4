# Qwen ExpertMajor v2 weight profiles on Hebrus

Qwen3.6-35B-A3B uses the same self-describing
`ds4.expert_major.v2` container as DeepSeek and GLM. Hebrus admits two exact
Qwen weight profiles inside that container:

| Profile | Routed storage | Product role |
| --- | --- | --- |
| `MLX_AFFINE4_G64` | MLX affine4, group 64 | Stable/recommended higher-quality profile |
| `Q2_K_XL` | Exact GGML IQ2_XS/IQ3_XXS/IQ4_XS layer inventory | Opt-in Beta performance-per-weight profile; 64 GiB and 32K qualified boundary |

Both artifacts store every routed weight exactly once, in physical
expert-major order:

```text
record(layer, expert) = gate bytes | up bytes | down bytes
```

For Affine4, every 64-value group is:

```text
32 packed bytes | BF16 scale | BF16 bias
```

For Q2_K_XL, each component keeps its admitted GGML block encoding. Thirty-six
layers use IQ2_XS gate/up and IQ3_XXS down; layer 1 uses IQ3_XXS gate/up and
IQ4_XS down; layers 34, 38, and 39 use IQ2_XS gate/up and IQ4_XS down. Dense
weights retain their exact Q4_K/Q5_K/Q6_K/Q8_0 inventory.

The 120 physical canonical gate/up/down tensors are replaced by one opaque I8
tensor plus a checksummed manifest. At startup Hebrus validates the Qwen family
ID, 40-layer inventory, complete tensor and tokenizer profile, component types,
offsets, alignment, and manifest digest before initializing Metal. Logical
tensor names are reconstructed only as graph and cache identities; every
physical map or read is translated through the validated v2 descriptor.

> [!IMPORTANT]
> This is a closed two-profile contract, not generic quantization support.
> Canonical/community GGUFs, Affine2, the former v2 Q4_K_S payload,
> `ds4.expert_major.v1`, sidecars, and non-Metal inference are rejected. There
> is no slower compatibility fallback.

## One runtime, two physical codecs

The quantization profiles are not parallel model implementations. Hebrus binds
the profile once from the complete tensor inventory, tokenizer metadata, and
ExpertMajor storage marker. The rest of the runtime is shared:

| Shared Qwen path | Codec-specific boundary |
| --- | --- |
| Model/session state and chat/tokenizer orchestration | Exact tensor-inventory validation |
| Gated DeltaNet and full-attention graph | Affine4 scale/bias weight decoding |
| KV state, RoPE, router, sampling, and output sequencing | IQ2/IQ3/IQ4 and Q4/Q5/Q6/Q8 weight decoding |
| Resident/SSD planning, cache ownership, I/O, and scheduling | Codec-appropriate Metal matvec/grouped-MM primitive |

Dispatch selects the physical primitive outside its inner loop. There is no
per-block codec branch, user flag, or duplicated Qwen graph. This boundary is
normative in
[`ADR 0006`](adr/0006-qwen-dual-weight-codecs.md).

## Runtime behavior

A valid profile activates automatically:

```sh
make -j8

./hebrus \
  -m /absolute/path/to/Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-MLX-Affine4-G64.gguf \
  --ctx 8192

./hebrus \
  -m /absolute/path/to/Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-Q2_K_XL.gguf \
  --ctx 8192
```

The first command uses the Stable/recommended artifact. The second is valid only
for the exact published Beta Q2_K_XL bytes below; its 262K endpoint remains
pending. `download_model.sh qwen-v2` intentionally continues to download
Affine4, while `download_model.sh qwen-q2-beta` opts into Q2_K_XL.

Normal startup needs no experimental guard, explicit Metal selection, sidecar,
payload hash flag, cache geometry, `--resident`, or `--ssd-streaming`. Through
24 GiB, the published Affine4 hardware policy selects guarded SSD and rejects
explicit resident mode. On larger qualified hosts AUTO chooses resident when
the complete artifact-specific working set passes both Metal and live-memory
gates; otherwise it uses the bounded SSD expert cache.

Cache byte accounting is derived from the selected profile's exact static,
component, record, and slab geometry. The 3,521-expert guarded target is about
5.80 GiB for Affine4 and is not reused as a hard-coded Q2 byte estimate. Before
each proposed new slab, live host pressure and Metal working-set limits admit
the exact allocation. Denial freezes the cache at its allocated capacity and
uses eviction/reuse; a new combined, per-component, or mmap fallback cannot
bypass it.

In resident mode every expert-major layer is exposed as one read-only Metal
buffer with the manifest record size as expert stride. In SSD mode logical
tensor identities remain cache keys, while selected experts translate to their
adjacent gate/up/down records. The common layer-major scheduler processes
bounded micro-batches so long prefills can reuse selected records without
token-time repacking.

## Build and verify

The generic converter recognizes `general.architecture=qwen35moe`, requires
complete routed layers `0..39`, validates the supported profile inventory, and
writes the embedded store atomically. Build the selected GGML Q2_K_XL source
directly:

```sh
python3 gguf-tools/ds4-expert-major.py inspect CANONICAL-Q2-K-XL.gguf
python3 gguf-tools/ds4-expert-major.py build \
  CANONICAL-Q2-K-XL.gguf \
  Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-Q2_K_XL.gguf
python3 gguf-tools/ds4-expert-major.py verify \
  CANONICAL-Q2-K-XL.gguf \
  Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-Q2_K_XL.gguf
```

The published Affine4 artifact additionally repacks the routed payload from the
pinned MLX source:

```sh
python3 gguf-tools/ds4-expert-major.py build \
  CANONICAL-QWEN.gguf QWEN-DS4-EXPERT-MAJOR-V2.gguf
python3 gguf-tools/ds4-expert-major.py repack-mlx-affine \
  QWEN-DS4-EXPERT-MAJOR-V2.gguf \
  /path/to/mlx-community-Qwen3.6-35B-A3B-4bit \
  Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-MLX-Affine4-G64.gguf
```

`verify` checks source provenance, byte-identical metadata and non-routed
tensors, every routed component, all 40 descriptors, and the complete payload
digest including alignment padding. `--skip-verify` is for disposable
development output and must never be used for publication.

## Release identity

The published Stable, Beta, and rejection-only identities are:

| Item | Value |
| --- | --- |
| Publication state | `published` |
| Repository | `andreaborio/Qwen3.6-35B-A3B-Hebrus-GGUF` |
| Artifact | `Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-MLX-Affine4-G64.gguf` |
| Artifact bytes | 20,808,566,880 |
| Artifact SHA-256 | `dd17266185833a9f05531ce366fd7284ddca1ed64aa3dcf06e321e8c72c9ea3d` |
| Immutable revision | `7bf9c3f7f6136aeb2599d75ee61c0cc2f18e2b02` |
| Minimum compatible runtime commit | `73a332fef82a0bcdd567d17e0de17aa004cad85d` |
| Storage | `mlx-affine4/group-64` |
| Beta publication state | `published-beta` |
| Beta download target | `qwen-q2-beta` |
| Beta artifact | `Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-Q2_K_XL.gguf` |
| Beta artifact bytes | 12,290,632,032 |
| Beta artifact SHA-256 | `30c22f70aff0f05986b517ee4ad8fef554a1b5aab6971c9ca09f999566d30143` |
| Beta embedded payload SHA-256 | `ccc3fbc2405d1dd73f8ac15741b0277514de4f46b80818531297ea9ffa0c6a3c` |
| Beta immutable revision | `bdb363efaeb227bfd702c9145cb224fffa456891` |
| Beta minimum compatible runtime commit | `42e2fec2a7dbb14a42e7a5612dfec00e33d443ca` |
| Beta storage | `ggml/group-0` |
| Beta profile | `q2-k-xl` |
| Beta qualified context tokens | 32768 |
| Beta minimum unified memory GiB | 64 |
| Beta full-window qualified | `false` |
| Beta recommended | `false` |
| Beta canonical source bytes | 12,290,628,576 |
| Beta canonical source SHA-256 | `96b9c0af5c77a4ecaabe3983175112b5ece763261c1ece12b2494b692a70dad7` |
| Negative fixture state | `negative-only` |
| Negative fixture | `Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-Q4_K_S.gguf` |
| Negative fixture bytes | 20,808,566,880 |
| Negative fixture SHA-256 | `d7c43a6388ec20e6fe5530850350f96fdb0ac37c5ce36d3e5f92b172c447f56b` |

The machine-readable
[Qwen release contract](contracts/qwen-release.json) remains canonical for
downloadable artifacts. Q2_K_XL is intentionally marked Beta, nonrecommended,
64 GiB minimum, and qualified through 32K only. Its smaller bytes do not inherit
Affine4's lower-memory qualification, and the near-262K endpoint remains
mandatory before Stable/full-window promotion.

## Qualification

On the same M5 Pro 64 GiB host and exact Affine4 artifact, the final shared
runtime improved the tested `main` mean by 11.20% prefill and 4.22% decode at
8K, and by 8.20% prefill and 5.27% decode at 32K. Control drift stayed below 3%,
2K/8K/32K output evidence matched, and the 65,536- and 100,000-token stability
lanes completed with zero swapout.

Q2_K_XL is 40.93% smaller than Affine4 by exact native file bytes. Its
three-domain pinned llama.cpp quality gate measured 2.53% geometric-mean PPL
above the Q4 control and 8.20% below IQ2_M. The final resident and cold-SSD
128/2K/8K/32K matrices retained matching greedy IDs; the exact generic
short-context comparison reached decode parity with pinned llama.cpp.

Complete commands, invalidations, TPOT, hashes, and rejected Affine2/TensorOps
experiments are recorded in
[`benchmarks/2026-07-28-qwen-q2-k-xl-performance-weight.md`](benchmarks/2026-07-28-qwen-q2-k-xl-performance-weight.md).

## Lifecycle and failure policy

The engine owns a duplicated descriptor for the embedded extent. Metal owns
read-only wrappers over private mappings of the same file. Shutdown fences GPU
work, releases wrappers and residency sets, unmaps their bytes, drains page-in
workers, and then closes the descriptor.

Unknown versions, wrong family IDs, canonical routed tensors beside the opaque
store, missing layers, unsupported profile/type combinations, inconsistent
dimensions, tokenizer/profile mismatches, overlapping extents, reserved bits,
and manifest corruption fail before inference. Because a v2 artifact contains
no canonical routed copy, any store-install failure is fatal.
