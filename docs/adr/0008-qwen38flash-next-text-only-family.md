# ADR 0008: Qwen3.8-Flash-Next Is a New Text-Only Qwen4Exp Family

- Status: Accepted
- Date: 2026-08-29
- Support state: Pinned contract only (`pinned-not-supported`); no runtime,
  artifact-distribution, downloader, server, or release support is qualified.

## Context

`Qwen/Qwen3.8-Flash-Next` at revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540` is a native multimodal
`Qwen4ExpForConditionalGeneration` (`model_type: qwen4_exp`, text config type
`qwen4_exp_text`, GGUF architecture spelling `qwen4exp`). It is not Qwen3-8B,
not Qwen3.5-Flash, and not the separately hosted proprietary `Qwen3.8-Flash`
service. Its pinned checkpoint contains 1,658 tensors in 131 shards totalling
`359,999,963,128` bytes, of which 333 tensors belong to the vision tower and
31 to a one-layer MTP head. The 1,294 tensors retained by the base profile are
`lm_head.weight` plus the `model.language_model.*` text backbone, including a
51,200,245,760-parameter PLE n-gram embedding table whose rows are hashed with
exact 64-bit integer multipliers.

Architecturally the model cannot reuse Qwen3.6's family/profile identity or
graph as-is. It uses a four-stream Gated Residual hyper-connection, a Gated
DeltaNet variant with 16 key heads expanded to 48 value heads and a sigmoid
output gate, Qwen Sparse Attention with a pooled index key block and a
2,048-token selected budget, a 512-expert top-10 MoE with F32 softmax and
renormalised weights, and one paged/embedded n-gram lookup that injects into
every residual stream before runtime layer 1. Generic matrix, allocation and
transactional-I/O primitives may be reused only after their dimensions and
semantics are made profile-explicit. Qwen3.6's admitted identifiers, caches,
and ExpertMajor family embed 40-layer, 256-expert, top-8 assumptions that are
false here.

Reusing those identifiers would either reject the correct artifact or admit a
near match. Reusing the hosted service's advertised one-million-token window
would advertise capacity this repository has not measured. Keeping the vision
and MTP weights in the base artifact would retain several gigabytes of source
tensor data that the first supported scope does not use.

## Decision

Add a distinct, fail-closed family for this model. No execution or support
claim follows from this decision; support requires the gates below.

1. **Identity.** New model family `DS4_MODEL_FAMILY_QWEN4EXP`, GGUF
   architecture value `qwen4exp` with metadata namespace `qwen4exp.`, and
   artifact profile ID `qwen4exp-base-v1`. Admission compares every field for
   exact equality. Fuzzy aliases and the forbidden identities `qwen3`,
   `qwen3-8b`, `qwen3.5-flash`, `qwen3.8-flash`, `qwen35moe` and `qwen4` are
   rejected. A near match is a failure, not a fallback.
2. **Text-only scope.** The defined base artifact profile carries only text weights,
   tokenizer, chat template and closed metadata. It explicitly excludes
   `model.visual.*` and `mtp.*` while the converter accounts for every
   excluded source tensor. Runtime admission rejects unexpected tensors, and
   structured image/video input fails before inference. Vision and MTP are
   separate future ADRs and artifacts. MTP is not executed and no speculative
   path is admitted.
3. **Closed contract.** `docs/contracts/qwen4exp-profile.json` is canonical
   for the 48-entry layer-type array, every closed dimension, tokenizer and
   template digests, the PLE hash multipliers, head primes, head offsets and
   the padded row count, and the source pins. Model constants, the layer array
   and the logical tensor-role inventory are admission-time equalities checked
   before any GPU allocation. Converter preflight separately checks the pinned
   source names, dtypes, shapes and byte extents. A later codec-specific
   artifact profile must close its output physical types and extents before it
   can be admitted; this ADR does not freeze them. `config.json` values are not
   re-derived at runtime from a Python package.
4. **Source evidence.** Phase 0 evidence is the weight-free
   inventory fixture `tests/qwen4exp/fixtures/qwen38flash-next-inventory-v1.json`
   with its collector. It pins 131 shard digests, all 1,658 tensor names,
   dtypes, shapes, owning shards and byte extents, the inventory digest
   `a639efc7a5147b04200e870d7e320335527f4361a8327b137feca2683b1dc434`, the
   config/tokenizer/template/index digests, and the three PLE hash buffers
   observed in the checkpoint. The mathematical reference is Transformers
   commit `42ca97014c85d71a88ad60d55f08cb9fb4d26e2c`.
5. **ExpertMajor v2 retained.** Routed experts remain one embedded opaque
   `ds4.expert_major.v2` store in one GGUF I8 tensor, with the same header,
   manifest, component and record layout, and the same v2 reader semantics.
   The Qwen4Exp profile reserves family
   `DS4_EXPERT_STORE_FAMILY_QWEN4EXP = 4` with 48 layers, 512 experts per layer
   and gate|up|down record order. One structural maximum rises with it:
   `DS4_EXPERT_STORE_V2_MAX_EXPERTS` from 384 to 512. That Phase 2 admission
   change does not alter the v2 container layout, must rerun the Qwen3.6 store
   and converter suites, and does not itself admit a Qwen4Exp artifact. The
   archived ExpertMajor v3 prototype stays archived.
6. **PLE v1 embedded extent.** PLE rows are one embedded opaque
   `ds4.ple_rows.v1` fixed-page extent, isolated from dense spans and from
   the expert store, with its own manifest, digests, fixed page geometry and
   checksums. Row location uses `page = row / rows_per_page`,
   `slot = row % rows_per_page`,
   `offset = payload_offset + page * page_stride`; no 320-million-entry
   offset table exists. The extent is excluded from warmup, `WILLNEED`, Metal
   registration and whole-tensor first touch, and is read through its file
   descriptor plus validated extent. A sidecar PLE store requires a new ADR
   and version. The codec, rows per page, page stride and checksum-generation
   details remain pending converter and PLE-I/O qualification; no PLE artifact
   is admitted by this decision alone.
7. **One graph, role-specific codecs.** One runtime graph will serve any
   eventually admitted codec-specific artifact profile for these base-model
   semantics. Physical weight interpretation stays codec-specific.
   Correctness begins at BF16/F32 reference parity. Release candidates may use
   a mixed, role-specific profile, including routed-expert candidates in the
   1.5–3-bit range, higher precision for sensitive smaller roles, and an
   independently quantised PLE profile. Ideal routed payload is 56.25 GiB at
   4 bits and 28.125 GiB at 2 bits for `120,795,955,200` parameters, so no
   global-Q4 assumption is permitted. No codec is currently qualified or
   frozen; selection requires quality bakeoff, exact artifact bytes and actual
   M5 Pro measurements.
8. **Precision.** F32 for norms, reductions, GDN controls and state, router
   logits/softmax/top-10, QSA score accumulation and GR gates and injection;
   exact unsigned 64-bit arithmetic for PLE hashes. Hash constants never pass
   through floating point or signed overflow. The maximum product
   `(248320 - 1) × 23703573157769 = 5,886,047,582,964,040,311` is below
   `2^63 - 1`, so the pinned checkpoint's signed `remainder` coincides with
   the unsigned remainder; the runtime uses unsigned arithmetic and asserts
   bit 63 is never set.
9. **Norm conventions.** Most norms are zero-centred:
   `y = (x * rsqrt(mean(x^2) + 1e-6)) * (1 + w)`. The GDN gated output norm is
   conventional: normalise, multiply by the stored weight, then apply the
   sigmoid gate; no `1 +` is added there. The convention is recorded per role
   in the manifest, and an ambiguous third-party quantization is rejected
   rather than guessed.
10. **Memory policy and context rollout.** Apple Metal remains the only
    production backend and `AUTO` remains the product path, with the M5 Pro
    64 GB tier as the primary target; other Apple Metal tiers may be admitted
    only through their own capability-gated profiles. Context is advertised by
    measured qualification, not by the checkpoint: qualify 8,192, 32,768,
    65,536 and 100,000 in order before the near-262,144 endpoint lane. Only the
    largest passing and explicitly bounded frontier is published. A full
    262,144-token claim requires the near-endpoint lane; a narrower public
    contract must fail larger requests closed and pass its own endpoint gate.
    The hosted service's one-million-token claim is not a Hebrus claim.
11. **Support gates.** User-facing support documentation, the runtime support
    matrix, downloader/server aliases and release contracts must not advertise
    Qwen3.8-Flash-Next as supported until the applicable gates pass: scalar
    oracle parity against the pinned Transformers reference; artifact/admission
    gates including negative fixtures that fail before GPU work; transactional
    state and failure-injection gates; a real M5 Pro `AUTO` run with
    owner-by-owner byte accounting, zero swap delta and exact output evidence;
    the 32K merge screen plus the endpoint lane for the public context contract;
    and the performance record for the target tier. An endpoint-pending
    additive implementation may merge only without a downloader/default or
    full-window claim. Narrowing an advertised context is allowed; weakening a
    correctness gate is not.
12. **License.** Distribution of converted or quantized weights requires a
    recorded review of the Qwen Community License 1.0 covering the
    redistribution terms for derived weights. Apache-2.0 is not assumed.
    The Phase 0 review status is `pending`; until it is resolved, converted or
    quantized weights may be used only as non-distributed private research
    evidence, with no public artifact or downloader entry.

## Consequences

- `qwen4exp` is a permanent second Qwen family. Qwen3.6 keeps its existing
  family, profile, cache semantics and published artifacts unchanged, and any
  shared change to common model, session, ExpertMajor, SSD, tokenizer or
  Metal graph code reruns the qualified DeepSeek, GLM and Qwen3.6 suites.
- An admitted base artifact must not contain vision or MTP weights, canonical
  routed tensors, or a canonical giant PLE tensor, and runtime must not discover
  a sidecar or fallback layout.
- In an admitted artifact, the 320,001,536 padded PLE rows exist only as fixed
  pages inside the validated extent, so no runtime materialises a 102.4 GB BF16
  table.
- Raising `MAX_EXPERTS` to 512 is a shared admission change, so Phase 2 owns
  it together with the Qwen3.6 regressions, and Phase 0/1 do not touch the
  store.
- User-facing documentation and CLI stay silent about support until the gates
  in item 11 pass. The first milestone is the frozen, independently tested
  semantic and artifact foundation: the closed contract, the pinned inventory
  and the scalar oracles built on them.
- Later vision or MTP support requires its own accepted ADR, its own artifact
  profile and its own qualification; neither may silently widen this scope.
