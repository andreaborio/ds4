# Generic expert-major GGUF roadmap: DeepSeek and GLM

`ds4.expert_major.v1` is deliberately fixed to the measured Qwen3.6-35B-A3B
Q4_K_S geometry. DeepSeek and GLM must not reuse that identifier: their routed
tensor types, shapes, layer participation, and quantization can vary.

The next format is a generic, versioned store discovered from the canonical
GGUF rather than a second set of family-specific hard-coded offsets.

## Format contract

The converter will emit one opaque GGUF tensor plus a checksummed manifest.
For every routed layer the manifest records:

- the canonical logical tensor identity and component role;
- expert count, selected-expert count, dimensions, GGML type, and block size;
- record, component, layer, and payload offsets with explicit alignment;
- source GGUF identity, payload digest, manifest digest, and format version.

DS4 reconstructs logical canonical descriptors for binding and diagnostics,
but every physical read is translated through the manifest. Unknown versions,
overlapping ranges, unsupported quant types, incomplete layers, and geometry
mismatches fail before backend initialization. Native files never fall back to
missing canonical routed weights.

## Runtime boundary

The store API owns validation and logical-to-physical translation. Model-family
code supplies only the expected tensor roles and supported geometry. Resident
and SSD paths consume the same validated records, so there is no per-token
repack and no separate format implementation in Metal, CUDA, or ROCm.

Canonical GGUF loading remains unchanged. The native extension becomes the
recommended artifact for a family only after its own correctness and
performance gates pass. Existing loaders that do not understand the extension
must use the canonical GGUF.

## DeepSeek tranche

1. Inventory Flash and PRO routed layers and every quant-type combination from
   the actual release artifacts.
2. Generalize the converter around manifest descriptors rather than Qwen's
   fixed gate/up/down sizes.
3. Add CPU/model-free fixtures for mixed quant types, skipped layers, corrupt
   offsets, truncated extents, and cross-family rejection.
4. Run canonical-versus-native SSD and full-resident parity where the model
   fits, including long-context prefill and decode.
5. Promote only if throughput is neutral within the predeclared variance band
   and memory/swap behavior does not regress on both low- and high-RAM Macs.

DeepSeek stays on its current mainline kernels and cache policy; adopting the
container must not silently inherit Qwen's 16 GiB planner or feature flags.

## GLM tranche

1. Rebase the generic store onto the qualified GLM branch rather than pulling
   the experimental GLM runtime wholesale into `main`.
2. Keep the existing GLM normalized/native model contract distinct from this
   expert-major container version; similar names do not imply compatible
   layouts.
3. Validate all routed layers, indexed-prefill preparation, SSD eviction, and
   long-context decode against the canonical artifact.
4. Repeat the parity and performance matrix on the target 64 GiB Mac before
   publishing a native model.

GLM support is promoted independently because its current runtime line is
experimental and has separate DeepSeek regression constraints.

## Release sequence

1. Ship Qwen `v1` and its model as the narrow proven implementation.
2. Land the generic manifest and converter behind a new identifier.
3. Qualify and publish DeepSeek artifacts.
4. Qualify and publish GLM artifacts.
5. Deprecate sidecars only after every supported family has a native artifact
   and the canonical migration path has been documented for other runtimes.
