# Fork notes — andreaborio/ds4

This is a fork of **antirez/ds4** (DwarfStar4). This file records, per fork-specific
change, whether it is **upstreamable to antirez/ds4** or **fork-only** — so future
contributors (and AI agents) don't waste effort proposing changes that cannot be merged
under antirez's requirements, and don't get confused about what lives here vs upstream.

> antirez's bar (see upstream `CONTRIBUTING.md`): *"Do not send PRs affecting one or more
> inference backends without checking if the resulting code is still correct and fast. The
> only acceptable speed regression is when an important correctness bug is fixed and it
> requires some speed penalty."* Read every entry below against that rule.

## Upstream status of fork changes

| change | what | upstream status |
|---|---|---|
| `gguf-tools/Makefile` quality-score link | add `ds4_distributed.o ds4_ssd.o` to QUALITY_OBJS (build was broken after the streaming refactor) | **MERGED-CANDIDATE** — opened as antirez/ds4 #434. Pure build fix, no backend code. |
| `deepseek4-quantize --reuse` | incremental re-quantize: copy byte-identical tensors from a prior build, regenerate only changed ones | **UPSTREAMABLE, separate PR** — quantizer/tooling only, no inference-backend impact. Must blind the reuse-key (hash the quantizer/quants.c version too) and lead with the byte-verifier before proposing. |
| `score_official.c` SSD streaming | enable `ssd_streaming` + a cache budget so the scorer runs when the model > RAM | **NOT mergeable as-is** — the 40 GiB cap is hardcoded for a 64 GB box. Rework as a CLI flag (reuse `ds4_parse_streaming_cache_experts_arg`) defaulting to off before any PR. |

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
