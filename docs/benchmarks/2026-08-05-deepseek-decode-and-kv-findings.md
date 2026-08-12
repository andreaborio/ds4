# DeepSeek decode and disk-KV findings

Date: 2026-08-05

Status: retained operational evidence and rejected performance directions on
one Apple M5 Pro with 64 GiB unified memory. Results are specific to the
DeepSeek V4 Flash 0731 artifact and workloads described below. They do not
change the runtime support contract or establish a cross-engine comparison.

Decision: keep the conversational cache retention default; retain GPU-wait
decode and n-gram verification as opt-in research paths; keep the measured
dense/routed bandwidth probes as diagnostic evidence. Do not publish the small
GPU-wait speed delta: it is below observed run-to-run spread.

## Conversational warm-cache retention

`DS4_DEEPSEEK_PREFILL_CACHE_KEEP_MAX_TOKENS` now defaults to 4096. Setting it
to `0` restores the earlier behavior. On the qualified host and artifact, a
second conversational turn of roughly 2.4K tokens otherwise tore down the warm
expert cache and reread about 27 GiB from SSD.

| Path | Second-turn TTFT | Steady decode |
| --- | ---: | ---: |
| Retain warm cache (current default) | 17.8 s | 20.9 char/s |
| Always tear down (value `0`) | 21.5 s | 10.9 char/s |

This is a single-host, character-rate observation, not a general throughput
claim. Prompts above the 4096-token boundary retain the previous teardown path
so large prefill keeps its memory headroom.

## Disk-KV restart reuse

The local server was run in AUTO resolving to SSD streaming with disk KV
enabled. A repeated prompt after server restart restored the saved prefix
automatically.

| Prompt lane | Cold | After restart | Restore operation |
| --- | ---: | ---: | ---: |
| About 1.7K tokens (`--ctx 4096`) | 17.8 s | 2.1 s | 7.7 ms |
| About 26.6K saved tokens (`--ctx 32768`) | 185.4 s | 23.0 s | 74.1 ms |

The 32K warm request still processed roughly 1.6K suffix tokens and rebuilt a
cold process-local expert cache; the 74.1 ms restore was not the complete 23.0
second request. Two cold state payloads differed by three header bytes while
the 47.8 MB KV state region was byte-identical. Sampled disk use was about 14
KB per saved token (372 MiB for 26.6K tokens). These values describe this
artifact and configuration only.

## GPU-wait decode: correct, not promoted for speed

The opt-in `DS4_METAL_ENABLE_DEEPSEEK_DECODE_GPU_WAIT` path moves the expert
load wait into the Metal command sequence. Output identity, repeatability, slab
residency, and cancellation gates passed at the tested frontiers. Its measured
speed effect did not survive repetition:

| Cohort | Opt-in versus legacy |
| --- | ---: |
| Two paired morning runs at 32K | +3.9% |
| Four-run A/B/B/A evening cohort at 32K | +1.3% |

The latter cohort had 10.77–12.03 t/s spread inside one arm (11.7%), larger
than the observed effect. The mode therefore remains opt-in and carries no
promoted performance claim.

## Tiny-batch and bandwidth diagnostics

A cache-resistant probe on the same host measured a real-shape Q8_0 dense
matvec at 247.4 GiB/s and Q4_K at 248.5 GiB/s. The 1.89x kernel-time difference
matched the byte ratio, but no production artifact uses the diagnostic Q4_K
dense path and no model-quality result supports changing those weights.

A separate routed-MoE probe measured about 186 GiB/s for the six selected
experts per layer, versus 244–250 GiB/s for dense Q8_0. Reconstructing the
tested token budget gave roughly 29.4 ms of dense weight traffic and 9.1 ms of
routed weight traffic. This rejected routed-MoE bandwidth as the primary
bottleneck for the tiny-batch verifier; it did not establish an end-to-end
speedup.

The opt-in n-gram verifier became deterministic on its layer-major two-row
path and accepted 71 of 85 drafted rounds in the sampled run. End to end it
did not beat sequential decode: the two-row verification cost about 141.1 ms
against 77.3 ms for a sequential pass, with nearly all verifier time in the 43
layer loop. The path remains research-only (`--ngram-spec`) and a verifier
failure invalidates the session rather than continuing from uncertain state.

## Claim boundary

- The timings above are retained to prevent repeating rejected directions and
  to document current defaults; they are not release-wide performance claims.
- Raw prompts and private host paths are intentionally omitted. Every value is
  scoped to the recorded M5 Pro 64 GiB lane and DeepSeek 0731 artifact.
- Q4_K dense execution is a diagnostic entry point, not a supported artifact
  profile.
- GPU-wait and n-gram verification remain opt-in. Neither changes AUTO startup
  or the support matrix.
