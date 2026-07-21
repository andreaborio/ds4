# Upstream Metal transfer audit

Date: 2026-07-21

Status: research complete; isolated GLM candidate rejected and removed.

Decision: do not merge the upstream M5 dense NAX port because the exploratory
campaign established no qualified benefit. The 4K cohort was inconclusive, the
8K observation was directional only, and the 32K canary did not establish a
new win against the non-contemporaneous retained gold. Do not cherry-pick the
mixed upstream optimization commit: most of it is specific to DeepSeek-V4 DSA
and does not enter the GLM graph.

Supersedes: no retained runtime path. This record extends
`2026-07-20-long-context-metal-stack.md` with the post-release upstream audit.

Affected paths and modes: the proposed change was shared Metal 4 Q8/F16 dense
batched prefill code, potentially reachable by GLM, Qwen, and DeepSeek. Only
GLM 5.2 ExpertMajor v2 AUTO-to-SSD was exercised with a real model. The Q8
single-token decode export and all model-specific MoE paths were held constant.

## Upstream sources and scope

The audit compared fork commit `57acfd408a3154851a0c59be432904300abb3b6c`
with:

- [`antirez/ds4` commit `427e281`](https://github.com/antirez/ds4/commit/427e281d24be2d9dfb6031189d1d171c18934cc5),
  a mixed 17-file integration containing Metal, CUDA, ROCm, and tests;
- [PR #555](https://github.com/antirez/ds4/pull/555), the primary source of
  the Metal work and its measured evidence;
- [M5 dense NAX commit `97efe609`](https://github.com/antirez/ds4/commit/97efe609fac1711ad6055767039dcbad0cfefc6c),
  the narrowest transferable subset;
- [PR #578](https://github.com/antirez/ds4/pull/578), which separates a later
  streaming slowdown caused by lost `mlock` coverage from the Metal kernels.

The final upstream commit is not a clean merge of PR #555 and has no isolated
parent-versus-child benchmark. Its published performance numbers therefore
describe PR revisions and hardware cohorts, not an attributable speedup for
`427e281` as a whole.

## Applicability to this fork

| Upstream family | GLM 5.2 ExpertMajor v2 on M5 | Decision |
| --- | --- | --- |
| Dense direct-RHS MPP double buffering and `dequantize_q8_0_pairs` | Directly reachable through GLM Q8/F16 batched matmuls | Port and test in isolation |
| Paired Q8 matvec | Not a GLM path; the shared dispatch would also re-enable the Qwen path already measured neutral-to-worse | Reject for GLM; keep Qwen scalar |
| DSV4 compressor, HC, gathered KV, indexer NAX, router, and top-k work | DeepSeek-V4/DSA-specific call sites and shapes | Do not port to GLM |
| DSV4 RoPE in-place/pair kernels | GLM decode uses its dedicated fused RoPE path; the generic GLM prefill calls do not meet the M5 `n_tok == 1` gate | Defer; requires a separately profiled GLM-kernel change |
| Generic MoE batch pairs | GLM already uses ExpertMajor-specialized pair kernels with different routing/layout contracts | Research concept only; not a direct port |

The upstream `mlock` recovery is also not transferred blindly. The retained
GLM v2 policy intentionally keeps its 601-slot expert cache pageable because
the local 64 GiB campaign measured that policy faster with zero new swapout.
DeepSeek retains its own pinning and slab-relief behavior.

## Isolated dense NAX experiment

The uncommitted candidate changed only `metal/dense.metal` and `ds4_metal.m`:

- double-buffered the prefill weight tile, increasing threadgroup scratch from
  4 KiB to 8 KiB;
- added paired aligned Q8 dequant loads;
- kept `kernel_mul_mm_q8_0_f32_nax_direct_rhs` as the exact legacy
  single-buffered N=1 decode export;
- added a separate N=32 prefill export and changed only the N=32/64/128 prefill
  dispatches;
- did not port DSV4, attention-output, indexer, router, or MoE changes.

Candidate dirty-diff SHA-256 was
`fb111a012fa7de42d387a9ce3e17dc429571eefaddd0b07dc10b0c72c48ab53b`.
The contemporaneous release and candidate `ds4-bench` SHA-256 values were
respectively
`116bd9e5a30c48825b0bd1025e1876f20073c10bbdeb0fcacfcafd597f1bdb79`
and
`35a1cbbd29e5bed3abbda26a5ac2ef58af187e5037c89a0050966390087bc6e6`.
Their runtime Metal source SHA-256 values were respectively
`e57502f1ebab8dbb1b797a76dc50e7947537f4ea00910704c50bdefeb29b4c08`
and
`ee6e5008b494d1fb0950ea939cb1b703ca55f2f6d2d6bf12c9f979a69ed18691`.

Before model-backed measurements the candidate passed:

```text
make clean
make -j10
make -j10 ds4_test
./ds4_test --metal-kernels
git diff --check
```

The machine was the qualified Apple M5 Pro with 64 GiB unified memory,
macOS build 25F84, and AC power. Every arm used the same qualified 262 GB GLM
ExpertMajor v2 GGUF, prose prompt SHA-256
`f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`,
AUTO residency, a 65,536-token allocation, eight greedy decode tokens, and no
competing inference process. These were exploratory rejection arms, not release
qualification cohorts.

The runner recorded the GGUF identity as `expected-only`: the complete artifact
had already been qualified as
`7f5017e3076e706c78f2a5322b035a9e2f6519c65ff5b6be8b2d91aeff61505d`,
but it was not rehashed inside every arm. Every new arm used
`cache_state=exploratory`; no cold or warm claim was made and no discarded
warm-up preceded the measurements. The sequential order therefore allowed
page-cache and thermal state to drift and is interpreted conservatively.

Both source variants resolved to the same runtime parameters: SSD because GLM
is qualified only for that mode, profile `glm52-metal-ssd-gold-v1`, 601 cached
experts / 6.93 GiB, 5.81 GiB compressed KV, 1.47 GiB buffers, 1.77 GiB resident
model data, and a 5.91 GiB prefill reserve. The recorded plan SHA-256 differs
because its first line includes the clean/dirty build identity: release
`cda1cc144195da5623de63c87108b514d7d357deeb8a4e6f27316f228d35dac3`,
candidate
`416518df8c92b4732ccc4acdda4c698d69d23f3a2a0f712943978477dc024dd4`.

| Context and order | Release prefill | Candidate prefill | Interpretation |
| --- | ---: | ---: | --- |
| 4K A/B/B/A | 48.36, 46.54 t/s | 47.39, 46.67 t/s | Means 47.45 vs 47.03 t/s; candidate about 0.9% slower. Control spread was 1.82 t/s / 3.84% and candidate spread 0.72 t/s / 1.53%; the control exceeded the 3% gate, so the whole cohort is inconclusive |
| 8K B/A | 46.35 t/s | 47.46 t/s | Candidate about 2.4% faster, but only a directional pair and smaller than observed host drift |
| 32K canary vs historical retained gold | 44.73 t/s historical gold | 44.50 t/s | No positive signal, but not a performance comparison: the gold was warm, used 128 decode tokens, commit `5479c42`, and different Metal sources; the candidate was exploratory with 8 decode tokens on dirty `57acfd4` |

All new arms had zero swapout. The candidate 32K run reached a 27% minimum
memory-pressure reading and a 37,314 MiB wired peak. Its frontier-logits content
SHA-256 was
`fc048bce4865b2c6a0df8951fef892d64e2058fa1980876895a46f300c91cc0e`,
exactly matching the retained 32K release result. Decode throughput is not
interpreted because the candidate deliberately preserved the release decode
kernel and used only eight exploratory decode tokens. The decode-evidence files
from that historical comparison are not comparable because the generated-token
counts differ.

Within the contemporaneous arms, release and candidate files matched exactly
at every tested frontier:

| Frontier | Logits file SHA-256 | Decode-evidence file SHA-256 |
| --- | --- | --- |
| 4K | `65673446afd745e2e0e383646965e71fe81e4e527ce5639c22086ea7d33bf7f9` | `0c6763db868c3213e98302b79f222885ecdcaf29087a7031a7e13f9b4cb4d74c` |
| 8K | `458b3837f1447c43f646f79d024dc8f8bd6a86fefe1fd8afb3b13162adbfff56` | `119a4c8c13081547cb9921a15526f69e0dc65ef0857e2950564920ff17a6f6c9` |

Raw summaries, CSV, environment, Metal identity, resolved plan, logits, decode
evidence, and memory telemetry remain under
`/tmp/ds4-upstream-nax-{a1,b1,b2,a2,b8k,a8k,b32k}.*`. The historical retained
gold is under
`/tmp/ds4-final-source.KJ9YHm/release-glm-32k-promessi.*`.

The campaign supplied no qualified positive result: the only complete cohort
was inconclusive because control drift exceeded the 3% gate, the 8K pair was
directional, and the 32K arm was comparable to historical output but not to
historical speed. Starting a
roughly one-hour 32K A/B/B/A cohort was therefore not justified by the evidence.
The implementation was removed instead of leaving dormant kernel duplication
in the runtime. This is a conservative no-benefit decision, not a claim that
the candidate caused a measured long-context regression.

## Follow-up boundary

Future work may test the DeepSeek-only pieces on their actual call sites. A
shared optimization must still pass the complete per-model context matrix in
`CONTRIBUTING.md`; isolated percentage gains are never added arithmetically.
When several independently correct candidates survive, test their complete
stack against the original baseline because interactions may amplify or cancel
the individual effects.
