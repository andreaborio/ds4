# Generic expert-major GGUF roadmap: DeepSeek and GLM

`ds4.expert_major.v1` is deliberately fixed to the measured Qwen3.6-35B-A3B
Q4_K_S geometry. DeepSeek and GLM must not reuse that identifier: their routed
tensor types, shapes, layer participation, and quantization can vary.

The generic, versioned store is implemented for DeepSeek and the independently
qualified GLM release behind `ds4.expert_major.v2`. It is discovered from the
canonical GGUF rather than a second set of family-specific hard-coded offsets.
Each family keeps its own runtime admission and performance gate.

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
code supplies only the expected tensor roles and supported geometry. Local
Apple Metal is the first consumer. Native CPU, CUDA, ROCm, and distributed
execution fail early until their own translators are implemented.

DeepSeek retains its independent canonical migration path. GLM does not: after
qualification, DS4 deliberately admits GLM only from ExpertMajor v2 on local
Metal SSD streaming. Existing external loaders that do not understand the
extension must use a different canonical artifact and runtime.

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

1. **Implemented:** port the qualified GLM runtime onto fork `main` with strict
   family checks instead of changing the DeepSeek/Qwen schedules.
2. **Implemented:** keep canonical logical tensor identity distinct from the
   physical expert-major container and fail closed on incomplete geometry.
3. **Implemented:** direct record strides, full expert address tables for
   grouped prefill, selected-expert SSD translation, in-flight-safe wrapper
   lifetime, and a model-free canonical/native numeric regression.
4. **Qualified:** the corrected 288+32 lane moved decode from 1.27 to
   1.77-1.81 t/s with exact output and no new swap activity. The prior
   rested-storage qualification remains 11.08/1.90 t/s median; an old/new
   same-condition A/B measured 1.75/1.74 and proves mainline runtime parity.
5. **Published:** the single-payload artifact and its exact SHA-256 live at
   `andreaborio/GLM-5.2-DS4-ExpertMajor-v2-GGUF`.

GLM is an independently admitted, ExpertMajor-only Apple Metal SSD path. CPU,
CUDA, ROCm, resident/distributed execution, canonical GLM artifacts, other
model families, and Macs below 64 GiB do not inherit its policy or artifact
compatibility.

## Release sequence

1. Ship Qwen `v1` and its model as the narrow proven implementation.
2. Land the generic manifest and converter behind a new identifier.
3. Qualify and publish DeepSeek artifacts.
4. **Complete:** qualify and publish the GLM artifact.
5. Keep family-specific admission explicit; GLM has no sidecar or canonical
   fallback in DS4 main.
