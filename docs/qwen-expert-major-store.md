# Qwen ExpertMajor v2 GGUF on Hebrus

Qwen3.6-35B-A3B uses the same self-describing `ds4.expert_major.v2` container
as DeepSeek and GLM. The optimized Mac artifact stores every routed weight
exactly once, inside the GGUF, in the physical expert-major order consumed by
Metal resident and SSD execution. Its payload is sourced from the local MLX
4-bit/group-64 model rather than the retired GGML Q4 blocks.

```text
record(layer, expert) = gate affine4-g64 | up affine4-g64 | down affine4-g64
affine4-g64 group = 32 packed bytes | BF16 scale | BF16 bias
```

The 120 physical canonical gate/up/down tensors are replaced by one opaque I8
tensor plus a checksummed manifest. At startup DS4 validates the Qwen family ID,
40-layer inventory, expert geometry, component types, offsets, alignment, and
manifest digest before initializing Metal. It then reconstructs canonical
logical tensor names only as graph and cache identities. Every physical mapping
or read is translated through the validated v2 descriptor.

> [!IMPORTANT]
> The current runtime contract is intentionally v2 plus MLX affine4/group-64
> only. Qwen canonical GGUFs, v2 GGML/Q4 payloads, `ds4.expert_major.v1`,
> `.experts.pack` sidecars, and CPU, CUDA, ROCm, or distributed inference are
> rejected. There is no slower inference fallback or migration window.

## Runtime behavior

A valid Qwen v2 file activates automatically:

```sh
make -j8
./hebrus \
  -m /absolute/path/to/Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-MLX-Affine4-G64.gguf \
  --ctx 8192
```

Use the same model with `./hebrus-server` for the local API. The legacy
`./ds4` and `./ds4-server` names remain aliases to the same build. Normal startup does
not need an experimental guard, explicit Metal selection, a sidecar path,
payload hashes, cache geometry, `--resident`, or `--ssd-streaming`. AUTO chooses
resident mapping when the full working set fits its Metal and host-memory
budgets; otherwise it selects the SSD expert cache. Explicit residency flags
remain diagnostics, not release instructions.

In resident mode each complete expert-major layer is exposed as a read-only
Metal buffer with the manifest record size as its expert stride. Only
non-routed GGUF spans and those 40 layer buffers are registered for residency;
there is no token-time host lookup, repack, or duplicate routed mapping.

In SSD mode the runtime keeps logical tensor identities as cache keys but
translates a selected expert to its adjacent gate/up/down record. Selected-
address affine grouped-MM kernels consume those records directly. A RAM-planned
macro scheduler processes bounded 2,048-token micro batches layer-major so long
prefills can read each routed expert once instead of once per macro tile.

## Build and verify

The generic converter recognizes `general.architecture=qwen35moe` and requires
complete routed layers `0..39`, complete Qwen geometry metadata, and a supported
quant layout. It writes a same-filesystem temporary output, performs full
verification by default, fsyncs it, and installs it atomically.

```sh
python3 gguf-tools/ds4-expert-major.py inspect CANONICAL.gguf
python3 gguf-tools/ds4-expert-major.py build \
  CANONICAL.gguf QWEN-DS4-EXPERT-MAJOR-V2.gguf
python3 gguf-tools/ds4-expert-major.py verify \
  CANONICAL.gguf QWEN-DS4-EXPERT-MAJOR-V2.gguf
python3 gguf-tools/ds4-expert-major.py repack-mlx-affine \
  QWEN-DS4-EXPERT-MAJOR-V2.gguf /path/to/mlx-community-Qwen3.6-35B-A3B-4bit \
  Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-MLX-Affine4-G64.gguf
```

`verify` checks:

- the canonical source identity stored in the manifest;
- byte-identical metadata and non-routed tensors;
- every routed component against the canonical expert bytes;
- Qwen family identity, all 40 layer descriptors, and quant geometry;
- the complete ExpertMajor payload digest, including alignment padding.

`--skip-verify` exists only for disposable development output and must never be
used for publication.

## Release identity

| Item | Value |
|---|---|
| Publication state | `published` |
| Repository | `andreaborio/Qwen3.6-35B-A3B-DS4-GGUF` |
| Artifact | `Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-MLX-Affine4-G64.gguf` |
| Artifact bytes | 20,808,566,880 |
| Artifact SHA-256 | `dd17266185833a9f05531ce366fd7284ddca1ed64aa3dcf06e321e8c72c9ea3d` |
| Immutable revision | `7bf9c3f7f6136aeb2599d75ee61c0cc2f18e2b02` |
| Minimum compatible runtime commit | `73a332fef82a0bcdd567d17e0de17aa004cad85d` |
| Storage | `mlx-affine4/group-64` |
| Negative fixture state | `negative-only` |
| Negative fixture | `Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-Q4_K_S.gguf` |
| Negative fixture bytes | 20,808,566,880 |
| Negative fixture SHA-256 | `d7c43a6388ec20e6fe5530850350f96fdb0ac37c5ce36d3e5f92b172c447f56b` |
| Canonical source bytes | 20,808,563,424 |
| Canonical source SHA-256 | `c33efb67bde86c9ba1f9e79c2dc42627170963bef0e915ab9b91a55cfb6d0fcd` |
| Storage marker | `MLX_AFFINE4`, group size 64 |

The machine-readable
[Qwen release contract](contracts/qwen-release.json) is canonical for the
repository, publication states, artifact identities, revision, and runtime
compatibility floor in this table.

The v2 output is 3,456 bytes larger than its canonical converter input. It does
not contain a second copy of the routed weights.

These exact bytes and their matching manifest are published at immutable
repository revision `7bf9c3f7f6136aeb2599d75ee61c0cc2f18e2b02`.
The manifest requires runtime commit
`73a332fef82a0bcdd567d17e0de17aa004cad85d` or a compatible descendant;
`download_model.sh qwen-v2` pins that revision and validates this relationship.
The older Q4_K_S object is incompatible with the current runtime contract and
is retained only for fail-closed testing.

## Qualification

On an M5 Pro with 64 GiB, final resident prefill measured 1,661.18 t/s at 2K,
1,421.90 t/s at 8K, and 877.34 t/s at 32K. Forced SSD measured 551.57, 269.10,
and 83.69 t/s. At 32K the adaptive macro read 16.875 GiB of expert data once,
72.2% less than the multi-tile control, for a 0.63% throughput reduction. The
same artifact produced identical resident/SSD greedy IDs in the final 128+16
decode lane. Full tables, logit comparisons, context-max allocation, and test
scope are recorded in
[`benchmarks/2026-07-21-qwen-unified-affine-auto-ssd.md`](benchmarks/2026-07-21-qwen-unified-affine-auto-ssd.md).

## Lifecycle and failure policy

The engine owns a duplicated descriptor for the embedded extent. Metal owns
read-only wrappers over private mappings of the same file. Shutdown fences GPU
work, releases wrappers and residency sets, unmaps their bytes, drains page-in
workers, and then closes the descriptor.

Unknown format versions, wrong family IDs, canonical routed tensors beside the
opaque store, missing layers, unsupported types, inconsistent dimensions,
overlapping extents, reserved bits, and manifest corruption all fail before
inference. Because a v2 artifact deliberately contains no canonical routed
copy, any store-install failure is fatal.
