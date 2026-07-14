# Fork notes — andreaborio/ds4

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

Ledger snapshot: fork `main` `1523b26`, upstream `main` `80ebbc3`, audited
2026-07-14. Commit links and live branch differences remain authoritative if
this dated snapshot drifts.

## Upstream status of fork changes

| change | what | upstream status |
|---|---|---|
| `gguf-tools/Makefile` quality-score link (`d2101a5`) | add `ds4_distributed.o ds4_ssd.o` to `QUALITY_OBJS`; the scorer stopped linking after the streaming refactor | **PR OPEN** — [antirez/ds4#434](https://github.com/antirez/ds4/pull/434). Pure build fix, no backend hot path. |
| Live server imatrix (`ee7181f`) | allow `ds4-server --imatrix-out` to aggregate routed-MoE statistics from live traffic without storing prompts | **UPSTREAM PR REQUIRED after final privacy/API review** — default-off and broadly applicable; design and verification in `ONEDGE_IMATRIX.md`. |
| `deepseek4-quantize --reuse` (`ef80754`, `f330c3b`) | incremental re-quantize: copy byte-identical tensors from a prior build and regenerate only changed tensors | **UPSTREAM PR REQUIRED after reuse-key hardening** — quantizer/tooling only. Blind the key with the quantizer implementation/version and lead with the byte-verifier. |
| Re-calibration reuse (`db96c2b`, `69787f1`, `324cc5a`) | reuse imatrix-independent tensors when only the calibration matrix changes | **UPSTREAM PR REQUIRED after the same key hardening** — keep stacked with, or follow, the base `--reuse` proposal; measured byte checks are already documented. |
| `score_official.c` SSD streaming | enable `ssd_streaming` + a cache budget so the scorer runs when the model > RAM | **UPSTREAM PR REQUIRED after rework; not mergeable as-is** — the 40 GiB cap is hardcoded for a 64 GB box. Rework it as a default-off CLI flag using `ds4_parse_streaming_cache_experts_arg`. |
| Metal AUTO gold path (`c58e91f`) | backend-isolated builds, build provenance, AUTO residency, SSD planning tests, and a Metal-first default policy | **MIXED; UPSTREAM PRS REQUIRED for reusable pieces** — split build isolation, residency safety, and tests from fork policy/defaults; validate every affected backend before proposing. |
| Adaptive expert-cache planner (`c8ea867`, `8a21edc`, `5816022`, `fc28b9c`, `b4b2036`) | size/admit the routed-expert cache from model geometry, context needs, live host memory, and backend headroom; fail closed at unsafe low-RAM budgets | **UPSTREAM PR REQUIRED after cross-backend validation** — direct DeepSeek A/B is performance-neutral on Metal; CUDA/ROCm behavior must be measured before a backend-affecting PR. The reverted startup bridge (`2f95e67`, reverted by `8a2a53f`) is not part of the promoted behavior. |
| Opt-in non-routed pinning (`517a11f`) | research whether static non-routed model state benefits from explicit pinning | **FORK EXPERIMENT, default-off** — one bounded microbenchmark was positive, but host-wide eviction risk and sustained behavior are unresolved. Do not propose or enable by default without a safe admission policy and a broader A/B. |
| Bounded cache benchmark and telemetry (`f4e0e64`, `2ffda62`, `09e29f5`) | abort on unsafe pressure/swap/wired-memory thresholds and report decode-scoped expert I/O/hit metrics | **UPSTREAM PR REQUIRED after interface cleanup** — tooling/observability is broadly reusable; preserve the no-hot-path-overhead default. |
| Serialize async expert-cache mutation (`bf4201c`) | prevent submitted Metal work from racing cache ownership changes | **UPSTREAM PR REQUIRED after a focused race regression test** — Metal correctness fix; no measured DeepSeek throughput regression in the current whole-fork A/B. |
| Safe Metal teardown (`1523b26`) | release GPU graph/session state before unmapping model-backed memory | **UPSTREAM PR REQUIRED after lifecycle regression coverage** — small Metal correctness/lifecycle fix. |
| RAM guard: refuse resident model maps larger than physical RAM | fail the non-streaming load when a map (or span-set total) exceeds 90% of `hw.memsize`, suggesting `--ssd-streaming`; `DS4_ALLOW_MODEL_OVERCOMMIT=1` opts out | **UPSTREAM PR REQUIRED; PR-ready** — branch `fix/refuse-oversized-resident-maps` (`06fd005`) on upstream main, tested (guard fires in ~3 s on an 80.8 GiB GGUF / 64 GB box, streaming + Metal kernels + streamed logprob vectors + cache-pressure suites pass). No hot-path code. |
| Expert prune/profile hooks (`aef72ee`) | default-off full expert rankings and CPU-router prune masks for domain analysis | **FORK EXPERIMENT** — research instrumentation, not a default inference feature. Reassess upstream fit only with a general diagnostic use case and neutral overhead. |
| Mixed-precision routed-expert streaming (`fefa426`, later sync) | serve per-layer boosted expert quants under SSD streaming | **CONVERGED UPSTREAM** — the fork takes upstream's implementation after `5800f15`; no competing delta should be preserved. |
| GLM 5.2 streaming line (branch `codex/glm52-upstream-clean-bench` = `bd89932` + 11 commits) | #520 fixes + #528 prepare + always-active ds4-native GLM GGUF layout support (`a0e234a`, the substrate every GLM number runs on) + a copy of the RAM guard + default-off experiments (router-ahead prefetch, prune/profile hooks, virtual resident decode layers, eviction tie-break) | **MIXED** — #520/#528 are open upstream; the RAM guard is tracked separately above. Keep the remaining experiments isolated while incomplete, but open upstream PRs for each piece that applies to the upstream GLM path after validation. Measured verification is in `SSD_STREAMING_VERIFICATION.md`. |

## Work not on `main`

| work | current status | promotion/upstream rule |
|---|---|---|
| Resident-map overcommit guard | Published branch `fix/refuse-oversized-resident-maps` at `06fd005`; tested, not yet on fork `main` | Open upstream as a standalone PR; do not claim mainline protection until it lands here or upstream. |
| GLM 5.2 | Published experimental branch; streamed prefill fixes and optimization measured on M5 Pro 64 GB | Keep whole-line claims separate from DeepSeek `main`; #520 and #528 are already open upstream. |
| Qwen3.6-35B-A3B (`qwen35moe`) | Experimental branch `feat/qwen-support`; the normalized Q4_K_S artifact passes a one-token CPU/llama.cpp logits smoke gate and real Metal + SSD logits/generation on M5 Pro 64 GB | Keep isolated and opt-in. No 16 GB support claim until a physical 8K cold/warm zero-swap run passes; the normalized-vs-Unsloth multi-position quality gate, server tools, exact rewind/snapshots, long context, and cross-backend production parity remain release gates. Review reusable primitives for focused upstream PRs only after those measurements. |

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
