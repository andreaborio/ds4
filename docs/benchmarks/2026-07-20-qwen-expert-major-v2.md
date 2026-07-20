# Qwen3.6 ExpertMajor v2 publication gate — 2026-07-20

This report records the model-backed gate that replaces Qwen's retired
fixed-geometry ExpertMajor v1 artifact with the shared
`ds4.expert_major.v2` format. It is a storage/runtime parity test, not a claim
that changing the container alone accelerates Qwen compute.

## Artifact identity

| Item | Bytes | SHA-256 |
|---|---:|---|
| Canonical converter input | 20,808,563,424 | `c33efb67bde86c9ba1f9e79c2dc42627170963bef0e915ab9b91a55cfb6d0fcd` |
| ExpertMajor payload | 18,119,393,280 | `d8bbe3731e4ac4f0117b24f8e8cb0ebaaf1a84cbfa7f264e4b297290946ee49f` |
| `Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-Q4_K_S.gguf` | 20,808,566,880 | `d7c43a6388ec20e6fe5530850350f96fdb0ac37c5ce36d3e5f92b172c447f56b` |

The v2 GGUF is 3,456 bytes larger than its canonical input. Routed weights
exist once; the physical canonical gate/up/down tensors are replaced by one
opaque store and its manifest.

## Test lane

- Host: Apple M5 Pro, 64 GiB unified memory.
- Backend: Metal resident.
- Shape: 2,048 prefill tokens and 16 generated tokens.
- Order: ExpertMajor v2, retired ExpertMajor v1 control, ExpertMajor v2.
- Sampling/evidence: the same deterministic configuration in all three arms.
- Safety observation: no new swapout during the lane.

The v1 binary and artifact were retained only to provide a same-machine control
for this gate. They are not part of the v2-only release runtime.

## Results

| Arm | Format | Prefill | Decode | Prefill wall | Decode wall |
|---|---|---:|---:|---:|---:|
| A1 | ExpertMajor v2 | 318.96 t/s | 29.54 t/s | 6,420.853 ms | 541.626 ms |
| B | ExpertMajor v1 control | 320.59 t/s | 29.59 t/s | 6,388.206 ms | 540.784 ms |
| A2 | ExpertMajor v2 | 318.83 t/s | 29.54 t/s | 6,423.421 ms | 541.630 ms |

The two v2 arms differ by 0.13 prefill t/s and agree at the reported decode
precision. Their mean is 318.895 prefill t/s and 29.54 decode t/s. Relative to
the interleaved v1 control, that is -0.53% prefill and -0.17% decode. The result
qualifies v2 as performance-neutral for this lane; it does not identify a
format-only speedup.

## Correctness

The resident v2 smoke produced output identical to the control. All three
complete decode-evidence files were byte-identical:

```text
399504c6ce3d4531ee0f2207702e96e2324c9b5c8dbf98adf47dfb9e64cae54d
```

This digest covers each arm's recorded decode evidence. Matching rendered text
alone was not used as the publication gate.

## Runtime decision

The release accepts the Qwen family only from the embedded v2 store on Apple
Metal. Normal startup is:

```sh
./ds4 \
  -m /absolute/path/to/Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-Q4_K_S.gguf \
  --ctx 8192
```

`DS4_QWEN_EXPERIMENTAL_METAL`, ExpertMajor v1, `DS4_QWEN_EXPERT_PACK_*`,
canonical Qwen inference, and non-Metal backends are outside the release
contract. The canonical GGUF remains an offline byte source for `inspect`,
`build`, and `verify` only.
