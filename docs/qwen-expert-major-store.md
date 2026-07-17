# Qwen DS4-native expert-major GGUF

The release format stores the Qwen routed weights exactly once, inside the
GGUF, in the expert-major order used by both resident and SSD execution. It
does not requantize or otherwise change any weight byte.

```text
record(layer, expert) = gate Q4_K | up Q4_K | down Q4_K
```

The physical GGUF tensor is named `ds4.expert_major.v1` and has GGUF type I8.
Its bytes contain the same versioned header, checksummed index, and payload
previously used by the `.experts.pack` sidecar. The 120 canonical routed tensor
descriptors are absent; all metadata and all 613 non-routed tensors are copied
byte-for-byte from the source. This keeps the resulting file approximately the
same size as the canonical GGUF instead of shipping both layouts.

At load time DS4 validates the embedded index and reconstructs the canonical
733-tensor logical inventory in memory. Those routed descriptors are identity
keys only. SSD mode translates every selected expert span to the embedded
record. Resident mode maps one complete expert-major layer per Metal buffer
and passes the record size as the expert stride to the existing ID kernels.
There is no repack or host lookup in the token path.

> [!IMPORTANT]
> The file is a valid GGUF container but its routed-weight layout is a DS4
> extension. A loader that does not implement `ds4.expert_major.v1` cannot run
> it. Removing the canonical tensor names makes incompatible loaders fail
> visibly instead of interpreting interleaved bytes as ordinary matrices.

## Resident invariants

- A native GGUF activates its embedded store automatically. No sidecar path or
  hash environment variable is required.
- Legacy canonical GGUFs and the explicit `DS4_QWEN_EXPERT_PACK_*` sidecar
  remain supported during the migration window.
- All 40 canonical layer bindings must validate before resident mapping.
- Each layer is mapped from the preceding host-page boundary and keeps its
  exact inner offset. This supports the format's 4 KiB data alignment on Macs
  with 16 KiB host pages without changing the on-disk format.
- Gate, up, and down byte geometry must exactly match the canonical tensors.
- Resident kernels keep selected IDs on GPU. There is no host lookup, cache
  lookup, copy, or repack in the per-token path.
- Only non-routed GGUF spans and the embedded expert-major layer buffers are
  exposed to Metal.
- A sidecar failure falls back to the canonical tensors. An embedded-store
  failure is fatal because a native GGUF deliberately contains no canonical
  routed copy.

## Lifecycle

The engine owns a duplicated descriptor for the embedded extent. Metal owns
read-only, no-copy layer wrappers over private mappings of the same GGUF.
Shutdown fences GPU work, releases the wrappers, unmaps their bytes, drains SSD
page-in workers, and only then closes the descriptor.

Canonical model views and the 40 expert-major layer buffers are registered as
one Metal residency set before inference. The ordinary coarse Metal warmup
samples both stores, so page-table validation and residency accounting remain
in model load rather than inflating first-token latency. `--warm-weights`
still skips the canonical routed tensors when the pack is active; touching
those pages would make both physical layouts resident at once.

Clearing a resident pack first ends the residency set, then releases every
Metal wrapper, then unmaps the file. This order also makes fallback safe after
a partially completed resident setup.

## Building and verifying the native GGUF

Build the sidecar once as a deterministic staging artifact, then embed it. Both
steps write a same-directory temporary file, verify it completely, fsync it,
and atomically rename it:

```sh
make ds4-qwen-pack
./ds4-qwen-pack build --reserve-bytes 2GiB model.gguf model.experts.pack
./ds4-qwen-pack verify model.gguf model.experts.pack
./ds4-qwen-pack native --reserve-bytes 2GiB \
  model.gguf model.experts.pack model.ds4-expert-major.gguf
./ds4-qwen-pack verify-native model.gguf model.ds4-expert-major.gguf
```

`native` performs four publication gates before rename:

1. the sidecar source SHA-256 must match the exact canonical GGUF;
2. the sidecar payload SHA-256 and every index entry must validate;
3. metadata and every non-routed output tensor must match the source bytes;
4. the embedded payload is reopened at its non-zero GGUF offset and rehashed.

The temporary output is allocated on the destination filesystem. Free-space
preflight includes the complete output plus the requested reserve and does not
count an old destination as reclaimable, because the old file remains alive
until the atomic rename.

The release artifact produced from the canonical Q4_K_S model has this exact
identity:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `Qwen3.6-35B-A3B-DS4-ExpertMajor-v1-Q4_K_S.gguf` | 20,808,970,240 | `fb2b344d49f0c3dfd854cfc11d92ffc873cc93a1d30bf4664e5aea6f1bfef839` |

The canonical input is 20,808,563,424 bytes, so the single-layout native file
adds 406,816 bytes (397.3 KiB, about 0.002%). It does not include a second copy
of the routed weights.

Run the native file directly:

```sh
DS4_QWEN_EXPERIMENTAL_METAL=1 ./ds4 --metal --resident \
  -m model.ds4-expert-major.gguf -p 'Hello'
```

The old sidecar activation remains available for A/B comparison. `SHA256` is
the payload digest stored in the manifest; `GGUF_SHA256` identifies the exact
canonical source:

```sh
DS4_QWEN_EXPERT_PACK_PATH=model.experts.pack \
DS4_QWEN_EXPERT_PACK_SHA256=<payload-sha256> \
DS4_QWEN_EXPERT_PACK_VERSION=1 \
DS4_QWEN_GGUF_SHA256=<gguf-sha256> \
./ds4 --metal --resident -m model.gguf -p 'Hello'
```

Missing, malformed, stale, or unsupported packs are optimization misses. The
runtime logs the reason, clears alternate state, and uses the canonical GGUF.
This fallback applies only to canonical GGUF plus sidecar, never to the native
single-layout artifact.

## Compatibility and rollout

- `v1` is fixed to the measured Qwen3.6-35B-A3B Q4_K_S geometry: 40 layers,
  256 routed experts, Q4_K gate/up/down records.
- Canonical Qwen GGUFs continue to run unchanged. The native artifact is the
  recommended DS4 download after the release that introduces this loader.
- CUDA, ROCm, llama.cpp, MLX, and generic GGUF tools are not claimed to execute
  this extension. Tools may inspect the container, but only a loader that
  understands the opaque tensor can perform inference.
- DeepSeek and GLM will use a later generic store version because their
  per-layer expert quant types and geometries can vary. They must not be
  mislabeled as v1-compatible or published before model-backed correctness and
  performance gates pass.

## Validation

The expert-major payload was validated first as a sidecar on an M5 Pro with
64 GiB. The native GGUF uses the identical payload and execution path. Both
measured arms below used the same integrated optimization stack, resident
mode, greedy decode, and full post-decode evidence. Rates are tokens per
second:

| Context | Canonical prefill | Expert-major prefill | Canonical decode | Expert-major decode |
|---:|---:|---:|---:|---:|
| 2,048 | 246.97 | 247.70 | 38.58 | 38.32 |
| 8,192 | 132.14 | 132.23 | 17.42 | 17.51 |
| 16,384 | 100.76 | 100.72 | 10.23 | 10.26 |

Evidence files were byte-identical at all three frontiers. The 16K runs
reported zero swap events, and system swap usage was unchanged. The
expert-major process peak footprint was 876,741,832 bytes versus 876,823,776
for canonical.

The final single-file publication gate then ran native/sidecar/native at 2K,
4K, 8K, and 16K. All 12 decode-evidence files were byte-identical, including
sampled token IDs and complete final logits. Commands, evidence hashes, timing
spread, and memory observations are recorded in
[`benchmarks/2026-07-17-qwen-native-expert-major.md`](benchmarks/2026-07-17-qwen-native-expert-major.md).

The first implementation registered only the non-routed GGUF views. That made
the first 2K prefill pay deferred pack VM validation and measured 125.99 to
148.83 t/s. Including the 40 pack buffers in the residency set moved that
one-time work back to load and restored 247.70 t/s. This regression is covered
by the model-free `--metal-expert-pack` lifecycle test.

On the M1 Pro with 16 GiB, the same branch builds cleanly and passes the pack,
expert-group, and Metal pack-lifecycle tests. An SSD-streaming cold smoke kept
the complete feature stack enabled, selected a 321-expert prefill cache with a
1,921-expert decode target, and preserved the low-RAM policy's single 2 GiB
request reserve.
