# Fork notes — andreaborio/hebrus

This is a transparent co-development fork of **antirez/ds4** (DwarfStar). This
file records grouped fork deltas, their evidence, and their upstream status so
contributors do not confuse fork experiments with upstream behavior.

**Policy:** every change that can be cleanly applied to an upstream-supported
path will be opened as an upstream PR once it is scoped and validated. Model- or
hardware-specific work is not automatically fork-only: it stays isolated only
while its upstream fit or evidence is incomplete. If upstream lands an
equivalent implementation, this fork converges on it and removes the duplicate.

> antirez's bar (see upstream `CONTRIBUTING.md`): *"Do not send PRs affecting one or more
> inference backends without checking if the resulting code is still correct and fast. The
> only acceptable speed regression is when an important correctness bug is fixed and it
> requires some speed penalty."* Read every entry below against that rule.

Ledger snapshot: fork-wide ExpertMajor v2-only migration for DeepSeek, GLM, and
Qwen, including the earlier GLM promotion and Metal lifecycle milestones;
upstream `main` `f9602e5`, audited 2026-07-21.
Commit links and live branch differences remain authoritative if this dated
snapshot drifts.

## Upstream status of fork changes

| change | what | upstream status |
|---|---|---|
| `gguf-tools/Makefile` quality-score link (`d2101a5`) | historical linkage fix for `ds4_distributed.o ds4_ssd.o` after the streaming refactor | **OBSOLETE IN THIS FORK** — distributed source and its object were retired; `ds4_ssd.o` remains linked. Upstream history: [antirez/ds4#434](https://github.com/antirez/ds4/pull/434). |
| Live server imatrix (`ee7181f`) | allow `ds4-server --imatrix-out` to aggregate routed-MoE statistics from live traffic without storing prompts | **UPSTREAM PR REQUIRED after final privacy/API review** — default-off and broadly applicable; design and verification in `ONEDGE_IMATRIX.md`. |
| `deepseek4-quantize --reuse` (`ef80754`, `f330c3b`) | incremental re-quantize: copy byte-identical tensors from a prior build and regenerate only changed tensors | **UPSTREAM PR REQUIRED after reuse-key hardening** — quantizer/tooling only. Blind the key with the quantizer implementation/version and add an exhaustive, reproducible, fail-closed byte verifier before upstreaming; the retired exercise-only sampler was not an acceptance gate. |
| Re-calibration reuse (`db96c2b`, `69787f1`, `324cc5a`) | reuse imatrix-independent tensors when only the calibration matrix changes | **UPSTREAM PR REQUIRED after the same key hardening** — keep stacked with, or follow, the base `--reuse` proposal. The 2026-06-12 exercise copied 1,199/1,328 tensors, regenerated 129 routed-expert tensors in about 45 versus 80 minutes, and sampled 40/40 unchanged regular plus 16/16 changed expert tensors with identical tensor tables. This is historical tooling evidence, not a current runtime or performance claim. |
| `score_official.c` SSD streaming | enable `ssd_streaming` + a cache budget so the scorer runs when the model > RAM | **UPSTREAM PR REQUIRED after rework; not mergeable as-is** — the 40 GiB cap is hardcoded for a 64 GB box. Rework it as a default-off CLI flag using `ds4_parse_streaming_cache_experts_arg`. |
| Metal AUTO gold path (`c58e91f`) | backend-isolated builds, build provenance, AUTO residency, SSD planning tests, and a Metal-first default policy | **MIXED; UPSTREAM PRS REQUIRED for reusable pieces** — split build isolation, residency safety, and tests from fork policy/defaults; validate every affected backend before proposing. |
| Adaptive expert-cache planner (`c8ea867`, `8a21edc`, `5816022`, `fc28b9c`, `b4b2036`) | size/admit the routed-expert cache from model geometry, context needs, live host memory, and backend headroom; fail closed at unsafe low-RAM budgets | **UPSTREAM PR REQUIRED after cross-backend validation** — direct DeepSeek A/B is performance-neutral on Metal; CUDA/ROCm behavior must be measured before a backend-affecting PR. The reverted startup bridge (`2f95e67`, reverted by `8a2a53f`) is not part of the promoted behavior. |
| Opt-in non-routed pinning (`517a11f`) | research whether static non-routed model state benefits from explicit pinning | **FORK EXPERIMENT, default-off** — one bounded microbenchmark was positive, but host-wide eviction risk and sustained behavior are unresolved. Do not propose or enable by default without a safe admission policy and a broader A/B. |
| Bounded cache benchmark and telemetry (`f4e0e64`, `2ffda62`, `09e29f5`) | abort on unsafe pressure/swap/wired-memory thresholds and report decode-scoped expert I/O/hit metrics | **UPSTREAM PR REQUIRED after interface cleanup** — tooling/observability is broadly reusable; preserve the no-hot-path-overhead default. |
| Serialize async expert-cache mutation (`bf4201c`) | prevent submitted Metal work from racing cache ownership changes | **UPSTREAM PR REQUIRED after a focused race regression test** — Metal correctness fix; no measured DeepSeek throughput regression in the current whole-fork A/B. |
| Safe Metal teardown (`1523b26`) | release GPU graph/session state before unmapping model-backed memory | **UPSTREAM PR REQUIRED after lifecycle regression coverage** — small Metal correctness/lifecycle fix. |
| RAM guard: refuse resident model maps larger than physical RAM | fail the non-streaming load when a map (or span-set total) exceeds 90% of `hw.memsize`, suggesting `--ssd-streaming`; `DS4_ALLOW_MODEL_OVERCOMMIT=1` opts out | **UPSTREAM PR REQUIRED; PR-ready** — branch `marmyx77/ds4:fix/refuse-oversized-resident-maps` (`06fd005`) from upstream main, tested (guard fires in ~3 s on an 80.8 GiB GGUF / 64 GB box, streaming + Metal kernels + streamed logprob vectors + cache-pressure suites pass). No hot-path code. |
| Expert prune/profile hooks (`aef72ee`) | default-off full expert rankings and CPU-router prune masks for domain analysis | **RETIRED AND REMOVED** — a historical coding evaluation kept 154/256 experts (about 40% removed) and measured pass@1 74% → 72%, within noise; the 2026-06-18 sync at `5800f15` and `8236528` separately verified a 430-expert mask and coherent generation. These are research observations, not current support or performance claims. Recover the hooks from Git only for a new isolated research campaign. |
| Mixed-precision routed-expert streaming (`fefa426`, later sync) | serve per-layer boosted expert quants under SSD streaming | **CONVERGED UPSTREAM** — the fork takes upstream's implementation after `5800f15`; no competing delta should be preserved. |
| Fork-wide ExpertMajor v2-only admission (2026-07-20 tranche) | use the family-tagged, checksummed `ds4.expert_major.v2` store as the only physical routed-expert layout for DeepSeek, GLM, and Qwen; reconstruct logical tensor identity while translating all Metal mappings and reads through the manifest | **FORK MAIN CONTRACT** — local Apple Metal only. Canonical GGUFs are offline converter inputs, not inference artifacts. ExpertMajor v1, sidecars, CPU, CUDA, ROCm, and distributed execution have no compatibility path. Normal startup is `./ds4 -m MODEL.gguf --ctx N`; invalid family, inventory, geometry, or manifest state fails before inference. |
| GLM 5.2 ExpertMajor v2 (`55d2bab` release gate) | #520 correctness + #528 indexed prepare, compact DSA KV, physical grouped prefill, one-record selected decode reads, translated advisory I/O, tokenizer/prompt routing, measured 601-expert 64 GiB policy, embedded single-payload GGUF, and removal of the retired predicted-install path | **FORK MAIN, EXPERTMAJOR-ONLY METAL CONTRACT** — the broken port moved from 1.27 to 1.77-1.81 decode t/s; the final simple gate was 1.79 t/s and the old/new same-condition A/B was 1.75/1.74, restoring runtime parity with the qualified path. The earlier rested-storage median remains 11.08/1.90 t/s. Model-backed output is coherent and deterministic; synthetic canonical/native direct, scalar-batch, and grouped-batch math passes. #520/#528 remain open upstream. GLM canonical/non-Metal/distributed execution and old GLM layouts have no compatibility path. |
| Qwen3.6-35B-A3B ExpertMajor v2 (`qwen35moe`, current affine contract) | distinct v2 family ID, generic converter, one embedded MLX affine4/group-64 routed store, hardware-aware AUTO resident/SSD admission, affine Metal kernels, and model-backed resident/SSD qualification | **FORK MAIN RUNTIME; PUBLISHED ARTIFACT; UPSTREAM ISSUE [#462](https://github.com/antirez/ds4/issues/462)** — the only runnable release is `Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-MLX-Affine4-G64.gguf`, 20,808,566,880 bytes, SHA-256 `dd17266185833a9f05531ce366fd7284ddca1ed64aa3dcf06e321e8c72c9ea3d`, pinned at repository revision `7bf9c3f7f6136aeb2599d75ee61c0cc2f18e2b02`. The previously published Q4_K_S v2 artifact and its SHA-256 `d7c43a6388ec20e6fe5530850350f96fdb0ac37c5ce36d3e5f92b172c447f56b` remain dated 2026-07-20 measurement/control evidence only; the current runtime rejects that store before inference. The experimental guard, v1 loader, sidecar, and canonical inference paths remain retired. |
| Qwen hardware-aware Metal policy (2026-07-21; 24 GiB guard amended 2026-07-24) | named 16/24/32/36/48/64/96/128 GiB profiles, continuous resident reserve, normal-pressure cold/warm file-cache equivalence, guarded 16/24 GiB SSD cache, and live resident/SSD fallback | **FORK MAIN CONTRACT; 24 GiB PHYSICAL RECHECK REQUIRED** — AUTO resolves to SSD on M1 Pro 16 GiB and resident on AC-powered M1 Pro 32 GiB with zero swap. Split-K transfer is independently confirmed at 2K on 16 GiB (15.00 versus 9.70 decode t/s) and 32K on 32 GiB (19.02 versus 1.63 decode t/s), with identical greedy token IDs in each A/B. A reported M5 24 GiB sustained decode reached macOS `WARNING` under the formerly uncapped cache; the corrected policy reuses the 3,521-expert 16 GiB ceiling and pressure-gates phase growth. Model-free regression coverage passes, but the exact Hebrus Studio physical reproducer must pass before that fix is signed off. |
| Qwen exact K/V pair-blit (2026-07-22) | validate both full-attention K/V cache copies transactionally, then record them in one ordered Metal blit encoder in resident and SSD modes | **FORK MAIN, OWNER-AUTHORIZED STRUCTURAL MERGE; RELEASE MATRIX INCOMPLETE** — removes exactly 50% of K/V blit-encoder creation without changing copy commands, bytes, commits, SSD I/O, logits, or decode evidence. M5 resident 2K decode improved 56.52 to 58.06 t/s (+2.72%); M5 SSD 2K decode improved +0.75%; M1 Pro 32 GiB measured +1.37% at 2K and -0.20% at contaminated 8K. Every valid mean end-to-end inference regression stayed below 2%; the M1 Pro 16 GiB timing gate is explicitly invalid due alternating 88--189 s external host stalls, while its correctness and zero-swap gates passed. The 128/32K/65K/100K release arms remain unqualified, so this merge is not recorded as completion of the normal performance-promotion gate. |
| Exact paired Q8 Metal decode (2026-07-20 long-context experiment) | compute two independent Q8 matvec outputs in one exact Metal dispatch while preserving bit-identical accumulation order | **REJECTED AND REMOVED** — the final zero-swap Qwen 32K A/B/B/A preserved exact evidence but did not show a speed win. Scalar means were 67.03 prefill t/s, 40,217.705 ms decode wall, 314.818 ms p50, and 317.563 ms p95; paired means were 40,273.322 ms, 315.395 ms, and 317.665 ms. The differences were neutral-to-worse and smaller than cohort spread, so the M5 kernel, dispatch, and dedicated test were removed. The clean post-removal source then measured 65.76 prefill t/s, about 3.18 decode t/s, and 315.193/317.518 ms p50/p95 with exact evidence and zero swap. |
| DeepSeek phase-adaptive long-context cache (2026-07-20 candidate) | shrink the ExpertMajor cache only during large prefill, restore 4,129 records through 32K or 2,065 at 65K+, pressure-gate growth, seed once after success, and enter lower-memory tiers 128 tokens before each hard frontier | **FORK RELEASE CANDIDATE; M5 64 GB SAFETY/CORRECTNESS** — valid 2K/8K cohorts cut p95 by 53.3%/42.2% with exact output; 32K candidate arms completed with zero swap while both original-baseline cohorts were invalidated by A2 swap, so no 32K speedup percentage is promoted. Final 65K/100K AUTO-to-SSD measured 137.29/6.59 and 145.11/6.44 prefill/decode t/s with zero swap. Non-M5 Apple systems retain the generic cache path. |
| DeepSeek SIMD top-6 finalize/weights fusion (2026-07-20) | replace the existing exact two-dispatch one-token router fast path with one SIMD-heavy dispatch | **REJECTED AND REMOVED** — warm 128-token A/B/B/A decode was 12.95/13.12/13.01/13.49 t/s. Control drift exceeded 3%, and the candidate mean did not beat the control mean, so no long-context promotion run was warranted. |
| GLM compact-indexer SSD transition (2026-07-20 candidate) | map the 21 exact norm/bias view pairs required after final prefill output mapping and invalidate static-map state before remap | **FORK RELEASE CORRECTNESS CANDIDATE** — fixes the 32K failure at the full-attention/indexed frontier without changing KV or attention geometry. Final same-prompt 32K+128 measured 44.73 prefill t/s, 1.87 overall decode t/s and about 2.12 t/s p50 steady, with zero swap and content hashes identical to the earlier corrected arm. A harder security/coding prompt completes at 45.53/1.33 because cache hit rate falls from 36.90% to 13.15%. The original baseline emits no logits, so this is not a speedup claim. |

## Work not on `main`

| work | current status | promotion/upstream rule |
|---|---|---|
| Resident-map overcommit guard | Published at `marmyx77/ds4:fix/refuse-oversized-resident-maps` (`06fd005`); tested, not yet on this fork's `main` | Open upstream as a standalone PR; do not claim mainline protection until it lands here or upstream. |

## DO NOT UPSTREAM (without clearing the bar below): per-expert / mixed-precision expert quantization

**Status: fork-only research. NOT mergeable into antirez/ds4 under his current requirements.**

The natural next quantization lever (MoPEQ-style, arXiv:2509.02512) is **per-expert** bit
allocation: within a routed-expert layer, give the few sensitive experts more bits and the
rest fewer. Our own A/B (see `forgequant`, the layer-selection harness) shows the
**per-LAYER** boost saturates — boosting ~6 hot layers captures essentially all the
recoverable fidelity; which-6 and how-many barely move it. The remaining headroom is
**per-expert**, which the per-layer `--tensor-type` boost cannot express (GGUF stores one
quant type per fused expert tensor).

**Why it cannot be merged upstream (the speed wall):**

1. ds4's Metal expert kernel dispatches on the **(gate_type, down_type)** pair and only
   supports specific combinations (`ds4_metal.m`, the `"unsupported Metal routed MoE quant
   types gate=%u down=%u"` guard). Uniform per-layer types work (base = iq2/q2_k; boost =
   q4/q4); **mixed combos like (gate=iq2, down=q4) are rejected and prefill fails.**
   (Empirically confirmed: a `coder-w2q4` build — down-proj at Q4, gate/up at iq2 — builds
   fine but dies at runtime with "metal prefill failed".)
2. Per-expert (or mixed-type) serving therefore needs **new Metal + CUDA + ROCm kernels**.
3. **antirez already tried grouped-Q4 experts and measured it SLOWER** (`ds4_metal.m`, the
   grouped-Q4 path: *"walks every expert window in the layer. On PRO Q4 this measured far
   slower than the active selected-slot path, so keep it opt-in for profiling"*).
4. This fork's target is a **64 GB Apple Silicon box that is decode compute-bound (~15
   tok/s ceiling)**. A slower expert kernel is exactly the regression we cannot absorb, and
   it violates antirez's "no speed regression" rule.

So: per-expert quantization is **research that lives in this fork only**. Do not open a PR
to antirez/ds4 for it, and do not assume the quantizer producing a combo means the runtime
can serve it — **the kernel's supported-combo table is the binding constraint, not the
quantizer.**

### The one exception (when slowness becomes acceptable)

Per antirez's own rule, a speed penalty is acceptable **if it buys an important correctness
gain**. So per-expert quantization could become upstream-worthy **only if** it delivers a
**clearly significant quality/correctness improvement** — not a marginal one. Concretely,
before reconsidering:

- prove the quality gain first with a **quality-only prototype** (correctness over speed,
  e.g. a CPU or non-optimized Metal path), measured with the official-continuation scorer
  and the `forgequant` paired A/B;
- the improvement must be **large and unambiguous** (well beyond the ~1 % / non-significant
  deltas we saw between per-layer selection strategies), enough that trading decode speed
  for it is obviously worth it;
- only then design the production kernels and quantify the exact speed cost with
  `ds4-bench` (before/after CSV, same machine) for the PR.

Marginal quality + slower kernel = **do not merge, do not propose.** Significant quality
that justifies the slowdown = revisit, with numbers, against antirez's correctness-vs-speed
rule.
