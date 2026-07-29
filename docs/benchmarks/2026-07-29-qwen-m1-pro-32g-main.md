# Qwen `main` on a physical M1 Pro with 32 GiB

Date: 2026-07-29

Status: additive physical-hardware evidence. Stable Affine4 and Beta Q2_K_XL
completed fresh-process 128, 2K, 8K, and 32K AUTO lanes with exact evidence,
zero new swapout, and no watchdog abort. A condition-matched rerun of pre-fix
`main` `7b5a3eb6f315` shows performance movement within 0.41% in every valid
pair and reproduces the old Affine4 SSD 32K decode failure. This is
hardware/fix characterization, not a performance-change promotion or a
lower-memory qualification for Q2_K_XL.

Decision: keep normal AUTO and tight context allocation. On this 32 GiB host,
Affine4 resolves to resident through 8K and guarded SSD at 32K; Q2_K_XL remains
resident through 32K. Affine4 remains Stable and recommended. Q2_K_XL remains
an opt-in Beta with its published 64 GiB minimum and pending near-262K
full-window gate.

Supersedes: none. This record adds physical M1 Pro 32 GiB evidence for tested
`main` `dc987860e9e93d3ad54513decc1485ffafd65706`.

## Intent, mechanism, expected effect, and risk

The run tests the exact immutable Affine4 and Q2_K_XL artifacts with the same
production Qwen graph and a clean binary built from current `main`. AUTO makes
the memory decision before allocation. For example, Q2_K_XL at 32K estimates
17.28 GiB including model, context/KV/scratch, and resident reserve, so it stays
resident inside the 24.96 GiB recommended working-set budget. Affine4 estimates
25.22 GiB at the same frontier and selects guarded SSD instead.

The important risks are hidden swap traffic and background CPU load. Every arm
records the macOS cumulative swapout counter and aborts after any increase or
below 20% free memory. The owner processes `node` and `mdbulkimport` were
suspended and no competing inference ran. Root-owned `audiomxd` and `configd`
remained active and intermittently consumed CPU; they could not be isolated
without administrator access. The timing values are therefore descriptive of
this physical machine, not clean-room precision measurements.

## Experiment identity

| Condition | Value |
| --- | --- |
| Host | `MacBookPro18,1`; Apple M1 Pro, 10 cores (8 performance, 2 efficiency); 32 GiB unified memory |
| OS and power | macOS 26.5.2 build `25F84`; AC power, battery 95% at initial capture |
| Tested runtime | clean isolated worktree at `dc987860e9e93d3ad54513decc1485ffafd65706` |
| Binary | `hebrus-bench`; SHA-256 `363382d7c2f6a3c57a83c18c3ae138ec5316bd33058bed4e770b29afcc110250` |
| Runtime Metal source | SHA-256 `06de75f42895665f97153105c5a1de931973a551af4b29769dbce4a783c75098`; no overrides; tensor mode off on M1 |
| Previous-main control | clean isolated worktree at `7b5a3eb6f315`; binary SHA-256 `918dd3bcfceee3963cb131bb76c104750a5e2dbf17b16b07323d4a426cde00db`; Metal source-set SHA-256 `908102a6a3ea0d46f4a392f25d15a53fd67e7c65c00a45ca82e2080d4f177b3f` |
| Stable artifact | `Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-MLX-Affine4-G64.gguf`; 20,808,566,880 bytes; SHA-256 `dd17266185833a9f05531ce366fd7284ddca1ed64aa3dcf06e321e8c72c9ea3d` |
| Beta artifact | `Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-Q2_K_XL.gguf`; 12,290,632,032 bytes; SHA-256 `30c22f70aff0f05986b517ee4ad8fef554a1b5aab6971c9ca09f999566d30143` |
| Artifact verification | Independent full OpenSSL SHA-256 on the 32 GiB host; Q2 transfer additionally verified the local source, each complete reconstruction, and final promoted filename |
| Prompt domains | `speed-bench/promessi_sposi.txt` at 2K; `tests/long_context_security_prompt.txt` otherwise |
| Decode | 128 greedy tokens except the 128+8 short safety arms |
| Context allocation | 8192 for short safety; otherwise frontier plus 129 tokens |
| Cache state | Fresh process and application cache per arm; warm macOS page cache; no safe system page-cache flush |
| Environment | Inherited `DS4_QWEN_EXPERIMENTAL_METAL=1` explicitly unset before each arm; production defaults plus diagnostic memory/timing variables only |
| Isolation caveat | One inference process and no model transfer during an arm; owner processes suspended; root `audiomxd`/`configd` CPU load remained |

The Q2 file could not be reconstructed from Affine4 because the routed weights
use different physical codecs. The exact Q2 bytes were copied from the
independently verified 16 GiB host copy, reconstructed from four transport
parts, verified again, and only then promoted to the final filename.

## Stable Affine4 physical 32 GiB lanes

Rows are chronological. The later rerun of pre-fix `main` supplies matching
2K, 8K, and attempted 32K controls on this exact host. The short safety row has
no matching control and remains `N/A`.

| Started (Europe/Rome) | Frontier / resolved plan | Prefill t/s | Decode t/s | TPOT p50 / p95 (ms) | Free min / peak footprint | Delta vs tested `main` | Delta vs previous comparable | Result |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| 2026-07-29T17:02:09+02:00 | 128+8, AUTO→resident | 203.07 | 25.80 | 34.085 / 71.456 | 26% / 2.15 GiB | N/A; physical characterization on tested `main` | N/A; first matching-host arm | exact; zero swap |
| 2026-07-29T17:02:54+02:00 | 8K+128, AUTO→resident | 416.09 | 26.95 | 37.033 / 37.135 | 25% / 2.41 GiB | +0.309% / +0.037% vs `7b5a3eb` | same matching control | exact and byte-identical; zero swap |
| 2026-07-29T17:04:04+02:00 | 2K+128, AUTO→resident | 448.45 | 29.72 | 33.613 / 33.691 | 26% / 2.20 GiB | -0.156% / +0.067% vs `7b5a3eb` | same matching control | exact and byte-identical; zero swap |
| 2026-07-29T17:06:16+02:00 | 32K+128, AUTO→SSD | 266.02 | 12.95 | 73.297 / 101.900 | 50% / 12.48 GiB | N/A; `7b5a3eb` failed first decode FFN | functional fix; no valid old timing row | exact; zero swap |

AUTO resident preflight required 24.27 GiB at the short allocation, 24.28 GiB
at 8K, and 24.04 GiB at 2K. At 32K the conservative admission requirement rose
to 25.22 GiB, above the 24.96 GiB recommended working-set budget, so AUTO
selected guarded SSD. The 32K SSD plan mapped the 19.37 GiB model, started with
321 prefill experts, and raised the decode target to all 10,240 routed experts
without swap.

The retained logits / decode-evidence SHA-256 pairs are:

| Frontier | Logits SHA-256 | Decode evidence SHA-256 |
| --- | --- | --- |
| 128 | `1e78d50f7cddf69a3083543ab0c7451415d8101399b6f7ebe1acef8cd478faea` | `a03dcf1ab041be000b6b182bf28248418642324f155e97b8ce6c481990785e5b` |
| 2K | `8c17d3035c63f8bd30200d8aab852f425a236d18362bac2692f5bbacacab4e6a` | `6b4e0a4f646db6e7228062ad229bd236d05c94ee34be1706d5ba5c1e78a76fce` |
| 8K | `c0e704737120df6ae7f20badeb07000a63996d3768d0a0e86cdc2271feac3ac1` | `f25dff2c21dda8559575a565393ac1ed3ee2304248997734c4671588a71e3a5b` |
| 32K | `17747cba1bf38fabdcadd7dbc34c64c0145084cf82ccb13f4603ceb59dfa18ca` | `1edd7e4c99528707e828413783aa0b6603485c9a363352d4b4d6e2ddc0404d52` |

## Q2_K_XL physical 32 GiB technical lanes

These are technical lower-memory observations only. Q2_K_XL remains published
with a 64 GiB Beta minimum because this single 32 GiB machine does not supply
the required qualification cohort or near-262K endpoint evidence.

| Started (Europe/Rome) | Frontier / resolved plan | Prefill t/s | Decode t/s | TPOT p50 / p95 (ms) | Free min / peak footprint | Delta vs tested `main` | Delta vs previous comparable | Result |
| --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |
| 2026-07-29T18:08:07+02:00 | 128+8, AUTO→resident | 106.68 | 28.77 | 26.917 / 89.537 | 50% / 2.40 GiB | N/A; physical characterization on tested `main` | N/A; first matching-host arm | exact; zero swap |
| 2026-07-29T18:08:25+02:00 | 8K+128, AUTO→resident | 247.02 | 33.44 | 29.891 / 30.023 | 50% / 2.41 GiB | -0.403% / +0.210% vs `7b5a3eb` | same matching control | exact and byte-identical; zero swap |
| 2026-07-29T18:09:18+02:00 | 2K+128, AUTO→resident | 266.15 | 37.61 | 26.523 / 26.663 | 51% / 2.20 GiB | -0.060% / +0.027% vs `7b5a3eb` | same matching control | exact and byte-identical; zero swap |
| 2026-07-29T18:09:49+02:00 | 32K+128, AUTO→resident | 192.59 | 22.94 | 43.542 / 43.650 | 47% / 3.35 GiB | +0.068% / +0.087% vs `7b5a3eb` | same matching control | exact and byte-identical; zero swap |

Q2_K_XL required 16.34 GiB at the short allocation, 16.35 GiB at 8K, 16.11
GiB at 2K, and 17.28 GiB at 32K. Every arm remained resident with the 9.80 GiB
embedded routed-expert payload and performed no expert `pread`.

The retained logits / decode-evidence SHA-256 pairs are:

| Frontier | Logits SHA-256 | Decode evidence SHA-256 |
| --- | --- | --- |
| 128 | `6d00ef456f0301e276b6acd6f30542c3f3999fadd6c08dd6710aef7f78aa2479` | `d58c0f39f8e8cc2a526c0a747fdd06c5779449716d58d73dcaf3e4b3fd073677` |
| 2K | `710f92b4d7dc77c6012540afd9790cbd5159fd3f46d8befff8870d7b25e09c20` | `04e22efdf5b1a9f3841a24d898a2f998f9d8a101eff970773942edacea36cad1` |
| 8K | `854281f9600c0249c751a47e3ab94efbb50055404d8df6cd0097e1515d20023d` | `82b5a305df597793c45360d8e624cc8b067123f742ccfeff3718c19d302f138f` |
| 32K | `17d7c204629ef3ecd52735b9ed22e91e7cb032fddbd46e30f32b13bddffe398a` | `e52ad087698894b3624b5d57202f62c675c57107cc2f35f8a2cd6a9fa7c3ec2e` |

## Condition-matched previous-main control

After the new-main matrix, pre-fix `main` `7b5a3eb6f315` was rebuilt in a
second isolated worktree and run with the same host, artifacts, prompts, AUTO
mode, allocations, generation count, environment guard, and watchdog. Rows are
ordered by the new-main run timestamp. All five completed pairs have
byte-identical logits and decode-evidence hashes across revisions.

| New run / frontier | Old control started | Prefill old → new (delta) | Decode old → new (delta) | TPOT p50/p95 old → new (delta) | TTFT delta | Total wall old → new (delta) |
| --- | --- | --- | --- | --- | ---: | --- |
| 2026-07-29T17:02:54+02:00 · Affine4 8K | 2026-07-29T18:20:40+02:00 | 414.81 → 416.09 t/s (+0.309%) | 26.94 → 26.95 t/s (+0.037%) | 37.026/37.129 → 37.033/37.135 ms (+0.019%/+0.016%) | -0.309% | 24,499.963 → 24,438.234 ms (-0.252%) |
| 2026-07-29T17:04:04+02:00 · Affine4 2K | 2026-07-29T18:20:05+02:00 | 449.15 → 448.45 t/s (-0.156%) | 29.70 → 29.72 t/s (+0.067%) | 33.623/33.750 → 33.613/33.691 ms (-0.030%/-0.175%) | +0.155% | 8,869.984 → 8,874.014 ms (+0.045%) |
| 2026-07-29T17:06:16+02:00 · Affine4 32K | 2026-07-29T18:21:30+02:00 | old prefill completed; no valid CSV | old decode failed at token 32768 | N/A | N/A | N/A; functional fix |
| 2026-07-29T18:08:25+02:00 · Q2 8K | 2026-07-29T18:24:27+02:00 | 248.02 → 247.02 t/s (-0.403%) | 33.37 → 33.44 t/s (+0.210%) | 29.908/30.009 → 29.891/30.023 ms (-0.057%/+0.047%) | +0.408% | 36,864.851 → 36,992.128 ms (+0.345%) |
| 2026-07-29T18:09:18+02:00 · Q2 2K | 2026-07-29T18:23:53+02:00 | 266.31 → 266.15 t/s (-0.060%) | 37.60 → 37.61 t/s (+0.027%) | 26.543/26.645 → 26.523/26.663 ms (-0.075%/+0.068%) | +0.060% | 11,094.469 → 11,097.931 ms (+0.031%) |
| 2026-07-29T18:09:49+02:00 · Q2 32K | 2026-07-29T18:25:17+02:00 | 192.46 → 192.59 t/s (+0.068%) | 22.92 → 22.94 t/s (+0.087%) | 43.568/43.676 → 43.542/43.650 ms (-0.060%/-0.060%) | -0.066% | 175,845.760 → 175,728.173 ms (-0.067%) |

The valid performance movements are all inside ±0.41%. They are descriptive
single pairs, not a precision speed-promotion cohort. The material result is
the Affine4 32K correction: old `main` exits `1` with
`Metal Qwen token 32768 failed at FFN`, while new `main` completes prefill and
128-token decode with exact evidence and zero swapout.

## Condition-matched profile tradeoff

At 2K and 8K, the revision, hardware, prompt, resident plan, context
allocation, and generated-token count match; only the artifact and its physical
weight decoder change. This is a profile tradeoff, not a `main` speed delta and
not a model-quality comparison.

| Frontier | Metric | Q2_K_XL | Affine4 | Q2 observed delta |
| --- | --- | ---: | ---: | ---: |
| 2K | Prefill throughput | 266.15 t/s | 448.45 t/s | -40.651% |
| 2K | Decode throughput | 37.61 t/s | 29.72 t/s | +26.548% |
| 2K | TPOT p50 / p95 | 26.523 / 26.663 ms | 33.613 / 33.691 ms | -21.093% / -20.860% |
| 2K | Prefill plus decode wall | 11,097.931 ms | 8,874.014 ms | +25.061% |
| 8K | Prefill throughput | 247.02 t/s | 416.09 t/s | -40.633% |
| 8K | Decode throughput | 33.44 t/s | 26.95 t/s | +24.082% |
| 8K | TPOT p50 / p95 | 29.891 / 30.023 ms | 37.033 / 37.135 ms | -19.286% / -19.152% |
| 8K | Prefill plus decode wall | 36,992.128 ms | 24,438.234 ms | +51.370% |

At 32K the resolved plans differ: Q2 remains resident while Affine4 uses SSD.
The observed values (192.59/22.94 t/s for Q2 and 266.02/12.95 t/s for
Affine4) are useful hardware behavior, but no mode-matched delta is claimed.
Q2's smaller weights favor decode residency; Affine4's kernels still produce
substantially faster prefill on this M1 Pro.

## Stability conclusion and settings

- All eight retained arms began and ended at swapout counter 2408 pages; delta
  was zero for every arm.
- Minimum reported free memory was 25% for resident Affine4, 50% for Affine4
  SSD at 32K, and 47% for Q2 resident at 32K.
- Use normal AUTO. Do not force Affine4 resident at 32K and do not add an
  admission bypass.
- Allocate only the context the request needs. Tight allocation makes the
  context-dependent AUTO decision explicit and avoids reserving the full model
  window unnecessarily.
- Affine4 remains the Stable/recommended artifact. Q2_K_XL remains a technical
  Beta choice for faster resident decode per stored byte; its slower prefill
  and lower-memory publication gates remain visible.
- No near-262K endpoint was attempted. This record cannot support a
  full-window claim.

## Validation and raw evidence

| Started (Europe/Rome) | Revision / command lane | Result |
| --- | --- | --- |
| N/A; build start was not captured | `dc98786`; clean isolated `make hebrus-bench` | PASS; no build warnings; binary SHA-256 `363382d7…` |
| 2026-07-29T17:02:09+02:00 | Affine4 AUTO 128/8, 8K/128, 2K/128, 32K/128 | PASS; exact evidence, zero new swapout in all four arms |
| 2026-07-29T18:08:07+02:00 | Q2_K_XL AUTO 128/8, 8K/128, 2K/128, 32K/128 | PASS; exact evidence, zero new swapout in all four arms |
| 2026-07-29T18:20:05+02:00 | `7b5a3eb` matching Affine4 2K/8K/32K controls | 2K/8K PASS and byte-identical; 32K FAIL at first decode FFN; zero new swapout |
| 2026-07-29T18:23:53+02:00 | `7b5a3eb` matching Q2_K_XL 2K/8K/32K controls | PASS and byte-identical; zero new swapout in all three arms |

The retained remote result directories are:

- `/private/tmp/hebrus-qwen32-dc98786/affine4-auto-p128-n8-a1`
- `/private/tmp/hebrus-qwen32-dc98786/affine4-auto-p8192-n128-a1`
- `/private/tmp/hebrus-qwen32-dc98786/affine4-auto-p2048-n128-a1`
- `/private/tmp/hebrus-qwen32-dc98786/affine4-auto-p32768-n128-a1`
- `/private/tmp/hebrus-qwen32-dc98786/q2-auto-p128-n8-a1`
- `/private/tmp/hebrus-qwen32-dc98786/q2-auto-p8192-n128-a1`
- `/private/tmp/hebrus-qwen32-dc98786/q2-auto-p2048-n128-a1`
- `/private/tmp/hebrus-qwen32-dc98786/q2-auto-p32768-n128-a1`
- `/private/tmp/hebrus-qwen32-7b5a3eb/affine4-auto-p2048-n128-a1`
- `/private/tmp/hebrus-qwen32-7b5a3eb/affine4-auto-p8192-n128-a1`
- `/private/tmp/hebrus-qwen32-7b5a3eb/affine4-auto-p32768-n128-a1`
- `/private/tmp/hebrus-qwen32-7b5a3eb/q2-auto-p2048-n128-a1`
- `/private/tmp/hebrus-qwen32-7b5a3eb/q2-auto-p8192-n128-a1`
- `/private/tmp/hebrus-qwen32-7b5a3eb/q2-auto-p32768-n128-a1`
