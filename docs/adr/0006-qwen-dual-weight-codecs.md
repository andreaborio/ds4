# ADR 0006: Qwen Shares One Runtime Across Two Weight Codecs

- Status: Accepted
- Date: 2026-07-28

## Context

The published Qwen3.6-35B-A3B artifact stores routed experts as MLX
affine4/group-64. A performance-per-weight investigation selected a second,
smaller artifact whose exact tensor inventory mixes GGML IQ2_XS, IQ3_XXS, and
IQ4_XS routed weights with Q4_K, Q5_K, Q6_K, and Q8_0 dense weights.

Treating these artifacts as two model implementations would duplicate Qwen
attention, Gated DeltaNet, KV, routing, cache, residency, SSD, session, and
tokenizer orchestration. Treating their weight bytes as one universal codec
would instead put storage-format branches inside hot matrix kernels and obscure
the different numerical contracts. Replacing the published affine4 artifact
would also remove a qualified higher-quality option without technical need.

An experimental affine2/group-32 representation was also evaluated. It needed
another physical format and dedicated kernels, produced a 14,171,777,376-byte
artifact, and did not establish a quality or end-to-end performance advantage
over the selected Q2_K_XL profile. Keeping it would create a third permanent
codec without a product benefit.

## Decision

Qwen3.6-35B-A3B has one runtime graph and two admitted weight profiles:

1. `MLX_AFFINE4_G64`, the published affine4/group-64 artifact;
2. `Q2_K_XL`, the exact mixed GGML inventory selected by the dated benchmark
   decision.

The runtime binds the profile once from the complete logical tensor inventory,
tokenizer metadata, and ExpertMajor v2 storage marker. Both profiles then use
the same model/session implementation, graph topology, Gated DeltaNet and
full-attention scheduling, KV state, router, resident/SSD policy, cache
ownership, and output path.

Only operations that interpret physical weight bytes remain codec-specific.
Affine4 uses its scale/bias grouped kernels and Q8/F16 dense path. Q2_K_XL uses
the admitted IQ2/IQ3/IQ4 routed kernels and Q4/Q5/Q6/Q8 dense kernels. Dispatch
selects these primitives outside their inner loops; there is no permanent
runtime flag and no universal kernel with a per-block codec branch.

ExpertMajor v2 remains the single container and mapping contract. Affine2,
generic community GGUF inventories, the former Q4_K_S store, canonical routed
weights, sidecars, and mismatched tokenizer/profile combinations fail closed.
The affine2 implementation and its experiment controls are removed.

Artifact publication is independent from runtime implementation. The existing
affine4 immutable release and `download_model.sh qwen-v2` remain unchanged.
Q2_K_XL is published separately as an opt-in, nonrecommended Beta through
`download_model.sh qwen-q2-beta` after its exact native bytes, immutable
repository revision, and compatible runtime commit were added to the
machine-readable release contract. Its public boundary is limited to the
completed 64 GiB/32K evidence; the near-262K endpoint remains mandatory before
Stable/full-window promotion.

## Consequences

- Users retain the published four-bit quality profile and gain a smaller
  two-bit profile without two parallel Qwen implementations.
- Shared Qwen optimizations apply to both codecs and must be validated on both.
  The final affine4 A/B/B/A cohorts therefore remain part of the Q2_K_XL
  promotion evidence.
- A new weight codec is not admitted merely because the converter can encode
  it. It needs a distinct product benefit, exact layout admission, focused
  numeric kernels, model-backed quality/correctness, the complete context
  matrix, and an amendment to this decision.
- Storage-specific kernels may evolve independently, but accepted scheduling
  and graph behavior stay singular. A successful experiment replaces its
  predecessor rather than adding a permanent selector.
- Physical lower-memory qualification remains artifact-specific. Affine4 keeps
  its published 16 GiB guarded-SSD contract; Q2_K_XL initially carries only the
  hardware/modes recorded in its benchmark decision until additional physical
  tiers pass.
