# Qwen expert-major store

The Qwen expert pack is one versioned weight store with two access modes. It
does not change or requantize any weight bytes.

```text
record(layer, expert) = gate Q4_K | up Q4_K | down Q4_K
```

SSD mode translates canonical GGUF expert spans to records in the pack and
loads selected records into the bounded expert cache. Resident mode maps one
complete layer per Metal buffer and passes the record size as the expert
stride to the existing ID kernels. Both modes therefore share the same file,
geometry validation, hashes, and `(layer, expert)` ordering.

## Resident invariants

- The pack is opt-in through the existing `DS4_QWEN_EXPERT_PACK_*` identity
  variables. Runs without a pack keep the canonical resident path unchanged.
- All 40 canonical layer bindings must validate before resident mapping.
- Each layer is mapped from the preceding host-page boundary and keeps its
  exact inner offset. This supports the format's 4 KiB data alignment on Macs
  with 16 KiB host pages without changing the on-disk format.
- Gate, up, and down byte geometry must exactly match the canonical tensors.
- Resident kernels keep selected IDs on GPU. There is no host lookup, cache
  lookup, copy, or repack in the per-token path.
- Only non-routed GGUF spans are exposed as Metal model views while the pack is
  resident. Canonical routed pages are not touched, so the two physical copies
  are not charged to unified memory simultaneously.
- Any pack validation or mapping failure clears the alternate store and falls
  back to the canonical GGUF path.

## Lifecycle

The engine owns the verified pack descriptor. Metal owns read-only, no-copy
layer wrappers over private file mappings. Shutdown fences GPU work, releases
the wrappers, unmaps their bytes, drains SSD page-in workers, and only then
closes the descriptor.

Canonical model views and the 40 expert-major layer buffers are registered as
one Metal residency set before inference. The ordinary coarse Metal warmup
samples both stores, so page-table validation and residency accounting remain
in model load rather than inflating first-token latency. `--warm-weights`
still skips the canonical routed tensors when the pack is active; touching
those pages would make both physical layouts resident at once.

Clearing a resident pack first ends the residency set, then releases every
Metal wrapper, then unmaps the file. This order also makes fallback safe after
a partially completed resident setup.

## Building and enabling the store

The builder writes a same-directory temporary file, verifies the payload and
every source span, fsyncs it, and atomically renames it:

```sh
make ds4-qwen-pack
./ds4-qwen-pack build --reserve-bytes 2GiB model.gguf model.experts.pack
./ds4-qwen-pack verify model.gguf model.experts.pack
```

Runtime activation is explicit and identity-bound. `SHA256` is the payload
digest stored in the manifest; `GGUF_SHA256` identifies the exact source:

```sh
DS4_QWEN_EXPERT_PACK_PATH=model.experts.pack \
DS4_QWEN_EXPERT_PACK_SHA256=<payload-sha256> \
DS4_QWEN_EXPERT_PACK_VERSION=1 \
DS4_QWEN_GGUF_SHA256=<gguf-sha256> \
./ds4 --metal --resident -m model.gguf -p 'Hello'
```

Missing, malformed, stale, or unsupported packs are optimization misses. The
runtime logs the reason, clears alternate state, and uses the canonical GGUF.

## Validation

The production Qwen3.6-35B-A3B Q4_K_S artifact was validated on an M5 Pro with
64 GiB. Both arms used the same integrated optimization stack, resident mode,
greedy decode, and full post-decode evidence. Rates are tokens per second:

| Context | Canonical prefill | Expert-major prefill | Canonical decode | Expert-major decode |
|---:|---:|---:|---:|---:|
| 2,048 | 246.97 | 247.70 | 38.58 | 38.32 |
| 8,192 | 132.14 | 132.23 | 17.42 | 17.51 |
| 16,384 | 100.76 | 100.72 | 10.23 | 10.26 |

Evidence files were byte-identical at all three frontiers. The 16K runs
reported zero swap events, and system swap usage was unchanged. The
expert-major process peak footprint was 876,741,832 bytes versus 876,823,776
for canonical.

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
