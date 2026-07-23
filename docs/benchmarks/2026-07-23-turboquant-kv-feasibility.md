# TurboQuant KV feasibility — rejected archive — 2026-07-23

Status: rejected and archived. No implementation, selector, storage format,
planner, generator, or active handoff from this experiment remains in the
branch tip.

Supersedes: the experimental commits `132fefb`, `36634b4`, `61b75cc`, and
`8159d09`, the temporary `KV_QUANTIZATION.md` architecture note, and the
`turboquant-kv-cross-engine.md` active handoff. It does not supersede any
production baseline or current model qualification record.

Affected path: experimental Qwen Metal KV storage and consumers in explicit SSD
streaming and resident research lanes. DeepSeek and GLM were analysis-only;
their runtime paths were not changed.

## Decision

Do not promote the TurboQuant-derived Qwen KV-cache experiment.

The 4-bit candidate created substantial long-context memory headroom and a fast
Flash-prefill path, but changed greedy output. Raising key precision to F16 and
keeping values in F32 recovered the 128-token control output at 2K and 8K when
only one full-attention layer was changed, but saved too little memory and
regressed 8K prefill/decode by about 10–12%. Applying that conservative format
to all ten full-attention layers produced a useful projected 32K saving but
regressed 8K prefill by 54% and decode by 71%.

The performance matrix permits an early rejection for correctness or a clear
regression. The K16/V32 candidate therefore did not proceed to 32K, M5,
DeepSeek, or GLM qualification. Production allocation, snapshots, admission,
and dispatch are unchanged.

## Scope And Provenance

The source audit used `RyanCodrai/turbovec` commit
`1e7200cfd8f26c92ce2855652db64bc7f85bc039`. TurboVec is an ANN/vector-search
index rather than an online attention-cache runtime, so DS4 did not import its
Rust or Python packages.

The discarded native DS4 tranche was recorded by commits:

- `132fefb` — initial checked planner, scalar reference, Qwen packed store and
  Metal consumers;
- `36634b4`, `61b75cc`, and `8159d09` — cross-hardware evidence and remaining
  gates;
- final uncommitted research snapshot:
  `fc131ed2283b01ec7fa8202d3191a466303d8c90171071113c9aa5fe668a2b3f`;
- final 16 GiB remote source diff:
  `452ab094cdd84151de03fbfff1682d1b3636572dd40c24f3fc3f679184533651`.

The retained measurements came from distinct rejection-only builds:

- the 16 GiB packed-TQ4 table used source
  `264904f210b555f98f42080b7bb30a78b5f6e80e` and executable SHA-256
  `8d0692012d4f0348ffa01b4cbddad2c36989bc63a3b6bc1760822220780ac504`;
- the 32 GiB strategy/capacity cohort used source
  `1d3972eca07a76fdd9ffc3e3deeb62587c6e7581` and executable SHA-256
  `5481339d8be43b587f5020a37b82ebcde1f892d98d6e65fd1d30a4c3015f1b5d`;
- the final 16 GiB K16/V32 isolation used `ds4-bench` SHA-256
  `e9dd38dc318f19e5be30afdab1b2db691ed0c9c3f86ca0928c3c8e44f62b5dc9`
  (`git=264904f210b5-dirty`, Metal/arm64).

The final quality-isolation runs used:

- Apple M1 Pro with 16 GiB unified memory, macOS 26.5 build `25F71`, AC power;
- explicit Qwen SSD streaming with exactly 2,881 cached experts;
- the 20,808,566,880-byte Qwen ExpertMajor v2 MLX affine4/group-64 artifact,
  SHA-256
  `dd17266185833a9f05531ce366fd7284ddca1ed64aa3dcf06e321e8c72c9ea3d`;
- `speed-bench/promessi_sposi.txt`, fresh process per arm, 128 greedy decode
  tokens, capacity 2,177 at 2K and 8,321 at 8K;
- a one-second pressure/swap watchdog and no competing DS4 process.

Every retained 16 GiB arm below completed with zero new swapout. The raw CSV,
logits, decode evidence, stderr, and watchdog summaries remain outside the
repository at the remote-home-relative path
`BEEP/ds4-tq4-kv-16g-20260723/results`, under the
`16g-*-exact2881-*` run prefixes. The final K16/V32 Metal-library SHA-256 was
`126ce81f2e9048e90ef1e6c86cccb4057564c641f74c556a8c873afea2e50e92`.
The earlier 16 GiB TQ4 and 32 GiB Metal identities were not transcribed into
this durable record and are therefore unavailable for promotion-grade reuse.
The exact 32 GiB raw-artifact path was likewise not promoted into durable
provenance.

## Evidence Classification

This is a negative feasibility record, not a qualification or promotion
cohort. The retained lanes were:

| Candidate and host | Resolved plan | Completed tiers | Not run | Cohort state |
| --- | --- | --- | --- | --- |
| TQ4, 16 GiB M1 Pro | AUTO/SSD at 128 with different adaptive expert plans; explicit Metal SSD streaming with 2,881 cached experts for the retained comparison tables | 128-context compatibility smoke; 2K, 8K, 32K | M5; DeepSeek; GLM | Fresh process per arm; the 128 F32 arm was the cold first process and is not a speed A/B; no controlled cold/warm classification, discarded warm-up, A/B/B/A timing sequence, or within-arm spread |
| TQ4, 32 GiB M1 Pro | Metal AUTO resolved to resident for the retained capacity frontier | 2K, 8K, 32K; functional TQ4-only 65K and 100K | M5; DeepSeek; GLM | Fresh process per arm; F32 32K failed admission, so there is no same-plan comparison; no controlled cold/warm classification, discarded warm-up, A/B/B/A sequence, or within-arm spread |
| K16/V32, 16 GiB M1 Pro | Explicit Metal SSD streaming, 2,881 cached experts | 2K and 8K | 128-context short tier; 32K; 65K; 100K; 32 GiB; M5; DeepSeek; GLM | Fresh process per arm; adjacent-build directional timing only; no controlled cold/warm classification, discarded warm-up, A/B/B/A sequence, or within-arm spread |

Filesystem/page-cache temperature was not controlled, and no timing warm-up
was designated and discarded. The repeated F32 2K quality control was
byte-identical, but that establishes output determinism only; timing drift was
not measured. These omissions prohibit speed promotion but do not weaken the
correctness and clear-regression grounds for rejection.

## Packed TQ4 Result

The broad candidate stored each 256-dimensional key as signed
Hadamard/Lloyd-Max 4-bit codes plus F16 metadata and each value as per-vector
affine 4-bit codes plus F16 metadata. At a logical 32K frontier this reduced
Qwen's ten-layer full-attention KV payload from 1.250 GiB to about 0.160 GiB.

On the controlled 16 GiB SSD lane:

| Context | Path | Prefill tok/s | Decode tok/s | Live runtime tensors | Task footprint |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2K | F32 control | 158.32 | 14.33 | 722.76 MiB | 5.47 GiB |
| 2K | TQ4 Flash | 221.90 | 14.19 | 648.60 MiB | 5.41 GiB |
| 8K | F32 control | 93.88 | 13.12 | 962.78 MiB | 5.74 GiB |
| 8K | TQ4 Flash | 275.16 | 12.80 | 679.33 MiB | 5.50 GiB |
| 32K | F32 control | 29.18 | 10.67 | 1,922.88 MiB | 6.68 GiB |
| 32K | TQ4 Flash | 231.77 | 10.16 | 802.23 MiB | 5.77 GiB |

The 32K TQ4 arm saved 1,120.65 MiB of live runtime tensors and completed with
36% minimum free pressure versus 25% for F32. Its frontier argmax stayed the
same, with top-set overlap 5/5, 18/20, 60/64, and 97/100, but full-vocabulary
RMSE was `0.16947` and greedy generation first diverged at generated token 9.
That is a correctness rejection, not a quality-neutral speed result.

The large prefill difference above compares the experimental Flash-staged path
with the pre-M5 F32 SSD path. It must not be generalized to another SoC,
resident mode, or a production default.

## Quality Isolation

A repeated F32 2K control was byte-identical, ruling out baseline
nondeterminism. The investigation then separated key/value precision,
correction metadata, layer position, and consumer precision.

The initial norm correction preserved the reconstructed centroid norm. The
TurboVec main algorithm instead stores:

```text
scale = ||k|| / <R(k / ||k||), reconstructed_centroids>
```

so that `<k, reconstructed_k>` is approximately `||k||²`. Implementing this
zero-byte correction and adding an independent self-inner-product test removed
the estimator bias but did not remove greedy divergence.

| Context | Candidate | Frontier RMSE vs F32 | Top-100 overlap | Common greedy prefix |
| ---: | --- | ---: | ---: | ---: |
| 2K | original K4/V4 | 0.228615 | not retained | 20/128 |
| 2K | corrected K4/V4 | 0.215993 | not retained | 26/128 |
| 2K | K4/V32, only layer 39 | 0.009635 | 99/100 | 128/128 |
| 8K | K4/V32, only layer 39 | 0.006718 | 100/100 | 79/128 |
| 2K | K16/V32, only layer 39 | 0.000307 | 100/100 | 128/128 |
| 8K | K16/V32, only layer 39 | 0.000264 | 100/100 | 128/128 |

The K4/V32 result proves that 4-bit key error alone can change the
autoregressive path even with exact F32 values and only one affected layer.
The final 8K K16/V32 logit RMSE after 128 matching generated tokens was
`0.0000352`, with the same final argmax.

The investigation also found that the experimental hybrid wrapper silently
rewrote explicit `serial` decode to F16-staged Flash at 2,048 or more cached
tokens. The final `serialfix` cohort preserved the direct consumer and added a
model-free K16/V32 regression beyond that boundary. Earlier nominally
“direct” hybrid decode labels are not used in the final decision.

## Conservative K16/V32 Performance

K16/V32 stores keys as F16 and values as F32. It is not a 4-bit TurboQuant
format; it was the conservative precision frontier used to determine whether
any useful exact-output compromise remained.

| Context | Scope | Prefill tok/s | Delta vs F32 | Decode tok/s | Delta vs F32 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 2K | F32 control | 154.05 | — | 14.18 | — |
| 2K | only layer 39 | 153.55 | -0.3% | 13.44 | -5.2% |
| 2K | all 10 layers | 105.79 | -31.3% | 8.11 | -42.8% |
| 8K | F32 control | 90.35 | — | 11.47 | — |
| 8K | only layer 39 | 81.09 | -10.2% | 10.07 | -12.2% |
| 8K | all 10 layers | 41.60 | -54.0% | 3.35 | -70.8% |

The F32 rows came from the immediately preceding build. The candidate build
changed only rejected research-format parsing/dispatch and its model-free
tests, not the F32 execution path. This is sufficient directional evidence for
rejection—especially for the all-layer loss—but it is not a same-binary
A/B/B/A promotion cohort and must not be cited as one.

At exactly 32,768 rows, converting one layer's keys from F32 to F16 saves
32 MiB; all ten layers save 320 MiB. At the benchmark allocation of 32,897
rows those figures are 32.126 MiB and 321.260 MiB. The one-layer arm's measured
8K peak RSS fell by about 7.4 MiB, consistent with its approximately 8.1 MiB
persistent allocation reduction.

The one-layer saving is too small for its 8K regression. The all-layer saving
is material, but its direct/serial consumer is far outside an acceptable
performance envelope. The all-layer run is therefore rejection evidence only;
its decode output was not used to make a positive quality claim.

## 32 GiB Capacity Evidence

On the exploratory 32 GiB M1 Pro resident lane, the original F32 request at
32K was rejected by admission: 25.23 GiB required versus the profile's
24.96 GiB budget. Full TQ4 was admitted and completed all retained strategies
with about 2.84 GiB task footprint, 23% minimum pressure, and zero swapout.

TQ4 also completed functional 65,536- and 100,000-token frontiers, but minimum
free pressure fell to 19% and 16%. Those runs proved bounded access rather than
safe promotion. They do not override the greedy-output failure or establish a
same-plan F32/TQ4 speed comparison.

## DeepSeek And GLM Boundary

No DeepSeek or GLM packed-cache implementation was qualified or retained.
Qwen's ordinary full K/V layout does not transfer directly:

- DeepSeek has raw MLA, model-native compressed attention, and indexer
  surfaces with different ownership and selection semantics.
- GLM has compact `kv_lora`, `k_rope`, and indexer-key surfaces; `kv_lora`
  reconstructs both attention operands and is not a Qwen-style value vector.
- GLM remains SSD-only and expanded full K/V must remain disabled.

This archive prevents repeating the Qwen K4/V4 or direct/serial K16/V32
experiment as a presumed cross-engine optimization. Any future cache work
requires a new decision, family-specific formats, independent quality gates,
and the complete short/medium/large/long matrix.

## Cleanup

The experimental commits remain recoverable from Git history, but their
implementation was reverted from the branch tip. The following were removed:

- shared planner/reference files and centroid generator;
- Qwen packed cache, Metal kernels, staging buffers, and storage flags;
- `DS4_QWEN_KV_TQ4` and `DS4_QWEN_TQ4_DECODE`;
- model-free research fixtures and Makefile targets;
- the temporary architecture plan and active task handoff.

The only retained repository artifact is this dated negative decision record
and its benchmark-index entry.
