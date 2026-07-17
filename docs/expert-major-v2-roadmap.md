# Generic expert-major GGUF roadmap: DeepSeek and GLM

`ds4.expert_major.v1` is deliberately fixed to the measured Qwen3.6-35B-A3B
Q4_K_S geometry. DeepSeek and GLM must not reuse that identifier: their routed
tensor types, shapes, layer participation, and quantization can vary.

The generic, versioned store is now implemented for DeepSeek behind
`ds4.expert_major.v2`. It is discovered from the canonical GGUF rather than a
second set of family-specific hard-coded offsets. Model-backed publication
qualification remains open; GLM adoption remains a separate tranche.

## Format contract

The converter emits one opaque GGUF tensor plus a checksummed manifest.
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
code supplies only the expected tensor roles and supported geometry. The first
consumer is local Apple Metal: resident and SSD paths consume the same
validated records without per-token repack. Native CPU, CUDA, ROCm, and
distributed execution fail early until their own translators are implemented;
their canonical GGUF paths are unchanged.

Canonical GGUF loading remains unchanged. The native extension becomes the
recommended artifact for a family only after its own correctness and
performance gates pass. Existing loaders that do not understand the extension
must use the canonical GGUF.

## DeepSeek tranche

1. **Implemented:** discover the complete routed inventory and quant geometry
   from the source GGUF.
2. **Implemented:** manifest-driven converter instead of Qwen's fixed
   gate/up/down sizes.
3. **Implemented:** C/Python fixtures for mixed quant types, corrupt manifests,
   corrupt payloads, structural bounds, and cross-family rejection.
4. **In progress:** the first M5 Pro SSD parity tranche passes at 128 and 768
   tokens; the 2K/8K/16K alternating gate and full-resident qualification on a
   host where the model fits remain open.
5. **Open:** promote only if throughput is neutral within the predeclared variance band
   and memory/swap behavior does not regress on both low- and high-RAM Macs.

DeepSeek keeps an independently qualified runtime policy: the native store
enables its paired IQ2 grouped-prefill kernel, a phase-aware long-prefill cache
floor, and a normal-pressure AUTO tier only on 64--96 GiB Macs. Canonical
DeepSeek, Qwen's 16 GiB planner, other model families, explicit cache budgets,
and larger-memory hosts retain their existing policy boundaries.

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
