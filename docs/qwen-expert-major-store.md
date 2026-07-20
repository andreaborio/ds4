# Qwen DS4-native ExpertMajor v2 GGUF

Qwen3.6-35B-A3B now uses the same self-describing
`ds4.expert_major.v2` container as DeepSeek and GLM. The release artifact
stores every routed weight exactly once, inside the GGUF, in the physical
expert-major order consumed by Metal resident and SSD execution. Conversion
does not requantize or otherwise change a weight byte.

```text
record(layer, expert) = gate Q4_K | up Q4_K | down Q4_K
```

The 120 physical canonical gate/up/down tensors are replaced by one opaque I8
tensor plus a checksummed manifest. At startup DS4 validates the Qwen family ID,
40-layer inventory, expert geometry, component types, offsets, alignment, and
manifest digest before initializing Metal. It then reconstructs canonical
logical tensor names only as graph and cache identities. Every physical mapping
or read is translated through the validated v2 descriptor.

> [!IMPORTANT]
> The current runtime contract is intentionally v2-only. Qwen canonical GGUFs,
> `ds4.expert_major.v1`, `.experts.pack` sidecars, and CPU, CUDA, ROCm, or
> distributed inference are rejected. Canonical files remain valid offline
> converter inputs, but there is no inference fallback or migration window.

## Runtime behavior

A valid Qwen v2 file activates automatically:

```sh
make -j8
./ds4 \
  -m /absolute/path/to/Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-Q4_K_S.gguf \
  --ctx 8192
```

Use the same model with `./ds4-server` for the local API. Normal startup does
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
translates a selected expert to its adjacent gate/up/down record. The existing
Qwen prefill and parallel decode schedules consume the translated addresses;
the format adds no canonical reread or fallback path.

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
| Hugging Face repository | [`andreaborio/Qwen3.6-35B-A3B-DS4-GGUF`](https://huggingface.co/andreaborio/Qwen3.6-35B-A3B-DS4-GGUF) |
| Artifact | `Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-Q4_K_S.gguf` |
| Artifact bytes | 20,808,566,880 |
| Artifact SHA-256 | `d7c43a6388ec20e6fe5530850350f96fdb0ac37c5ce36d3e5f92b172c447f56b` |
| Canonical source bytes | 20,808,563,424 |
| Canonical source SHA-256 | `c33efb67bde86c9ba1f9e79c2dc42627170963bef0e915ab9b91a55cfb6d0fcd` |
| ExpertMajor payload SHA-256 | `d8bbe3731e4ac4f0117b24f8e8cb0ebaaf1a84cbfa7f264e4b297290946ee49f` |

The v2 output is 3,456 bytes larger than its canonical converter input. It does
not contain a second copy of the routed weights.

## Qualification

The M5 Pro 64 GiB resident smoke completed with output identical to the v1
control. The controlled 2K publication lane alternated v2/v1/v2 with 2,048
prefill tokens and 16 generated tokens:

| Arm | Format | Prefill | Decode |
|---|---|---:|---:|
| A1 | ExpertMajor v2 | 318.96 t/s | 29.54 t/s |
| B | retired ExpertMajor v1 control | 320.59 t/s | 29.59 t/s |
| A2 | ExpertMajor v2 | 318.83 t/s | 29.54 t/s |

All three decode-evidence files were byte-identical with SHA-256
`399504c6ce3d4531ee0f2207702e96e2324c9b5c8dbf98adf47dfb9e64cae54d`.
The run produced no new swapout. The v1 arm is retained only as benchmark
evidence; it is not an accepted runtime artifact after the v2-only cleanup.

Full gate scope and interpretation are recorded in
[`benchmarks/2026-07-20-qwen-expert-major-v2.md`](benchmarks/2026-07-20-qwen-expert-major-v2.md).

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
