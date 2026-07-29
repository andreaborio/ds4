# Qwen `main` on a physical M1 Pro with 16 GiB

Date: 2026-07-29

Status: additive physical-hardware evidence. Stable Affine4 completed the
short, 2K, 8K, and 16K guarded-SSD lanes with exact evidence and zero new
swapout. The separate 32K arm completed inference but is rejected because 32
pages were swapped during finalization. Q2_K_XL completed the documented short
and 8K technical lanes, but this does not lower its published 64 GiB Beta
minimum.

Decision: keep normal AUTO startup, the default 4096-token prefill chunk, tight
context allocation, and the guarded adaptive expert cache on this machine. Do
not force resident mode or a manual cache size. The Stable Affine4 artifact
remains the recommended 16 GiB profile. No support boundary changes in this
record.

Supersedes: none. This record adds a physical post-fix run for commit
`a1d7398799e91681aca6cb54b56aaf9b42dcedac`.

## Intent, mechanism, expected effect, and risk

The run checks that the Affine4 SSD-overlap fix reaches decode on the physical
16 GiB floor, compares the two published Qwen weight profiles under one shared
runtime, and observes how guarded cache capacity changes as context grows.
AUTO must resolve to SSD on this tier. Before every larger expert slab, the
runtime rechecks host and Metal headroom; a denial freezes the existing cache
and reuses its slots.

The important risk is hidden virtual-memory traffic. Expert streaming already
uses storage, so a concurrent macOS swapout makes storage latency a second
uncontrolled variable and signals that the requested memory envelope touched
system pressure. The watchdog therefore rejects an arm after any new swapout
or below 20% free memory. This is a benchmark-validity rule, not a claim that a
few swapped pages necessarily corrupt output or crash normal inference.

## Experiment identity

| Condition | Value |
| --- | --- |
| Host | `MacBookPro18,3`; Apple M1 Pro, 8 cores (6 performance, 2 efficiency); 16 GiB unified memory |
| OS and power | macOS 26.5 build `25F71`; AC power, battery 96% and charging at final capture |
| Tested runtime | clean remote `main` at `a1d7398799e91681aca6cb54b56aaf9b42dcedac` |
| Binary | `hebrus-bench`; SHA-256 `cfab1d4a68293e58337e159f2a559f6e7d9339098df09a7b10a2fc47830e221a` |
| Runtime Metal source | SHA-256 `06de75f42895665f97153105c5a1de931973a551af4b29769dbce4a783c75098`; no overrides |
| Stable artifact | `Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-MLX-Affine4-G64.gguf`; 20,808,566,880 bytes; SHA-256 `dd17266185833a9f05531ce366fd7284ddca1ed64aa3dcf06e321e8c72c9ea3d`; HF revision `7bf9c3f7f6136aeb2599d75ee61c0cc2f18e2b02` |
| Beta artifact | `Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-Q2_K_XL.gguf`; 12,290,632,032 bytes; SHA-256 `30c22f70aff0f05986b517ee4ad8fef554a1b5aab6971c9ca09f999566d30143`; HF revision `bdb363efaeb227bfd702c9145cb224fffa456891` |
| Artifact verification | Downloader byte/SHA gate plus independent OpenSSL SHA-256 for both files |
| Resolved plan | AUTO or explicit technical SSD request resolving to guarded SSD; resident Q2 request rejected before model allocation |
| Prompt domains | `speed-bench/promessi_sposi.txt` at 2K; `tests/long_context_security_prompt.txt` otherwise |
| Decode | 128 greedy tokens except the 128+8 short safety arms |
| Cache state | Fresh application cache per process; warm macOS page cache; no safe system page-cache flush |
| Isolation | One inference process; GUI applications terminated; user Spotlight workers suspended; models moved under `HebrusModels.noindex`; per-arm swap and memory watchdog |

The legacy Q4_K_S placeholder was not used. The Stable Affine4 file was
downloaded from the immutable HF revision above and independently verified.

## Q2_K_XL physical 16 GiB technical lanes

The first row is the matching pre-fix `main` control retained earlier on the
same host. It is the only condition-matched tested-main baseline. Rows at other
frontiers therefore report `N/A` rather than manufacturing a comparison.

| Started (Europe/Rome) | Revision / mode / frontier | Prefill t/s | Decode t/s | TPOT p50 / p95 | Free min / peak footprint | Delta vs tested `main` | Delta vs previous comparable | Result |
| --- | --- | ---: | ---: | --- | --- | --- | --- | --- |
| 2026-07-29T15:13:45+02:00 | `7b5a3eb`, AUTO→SSD, 2K+128 | 217.76 | 17.54 | 49.768 / 111.151 ms | 30% / N/A | baseline | N/A; first retained arm | exact; zero swap |
| 2026-07-29T15:27:21+02:00 | `a1d7398`, explicit SSD cold, 128+8 | 70.98 | 13.14 | 74.531 / 90.540 ms | N/A / 3.36 GiB | N/A; different frontier | N/A; different frontier | exact; zero swap |
| 2026-07-29T15:28:05+02:00 | `a1d7398`, explicit SSD cold, 8K+128 | 219.15 | 13.77 | 68.163 / 105.907 ms | observed 37% / 5.42 GiB | N/A; no matching 8K control | N/A; first 8K arm | exact; zero swap |
| 2026-07-29T15:29:57+02:00 | `a1d7398`, explicit SSD default preload, 8K+128 | 220.11 | 13.96 | 67.959 / 106.186 ms | 35% / 5.42 GiB | N/A; no matching 8K control | +0.438% / +1.380% vs preceding cold arm | exact; zero swap; single descriptive pair only |
| 2026-07-29T15:31:56+02:00 | `a1d7398`, AUTO→SSD, 2K+128 | 221.38 | 18.84 | 48.650 / 81.462 ms | 38% / 5.19 GiB | +1.662% / +7.412% | +1.662% / +7.412% vs `7b5a3eb` | exact; zero swap |

For the matching 2K row, TTFT changed from 9,406.013 to 9,252.104 ms
(-1.636%), TPOT p50 improved 2.246%, and TPOT p95 improved 26.711%. The logits
and decode-evidence SHA-256 values match the pre-fix control:
`7a55bf859cee17ab139fffd41335b66258d2cc0b0b222a8a85dcf0aa5c9d84f1`
and
`1454b16ce05a299b68adc6798063f07315a31d2cd766cc66edb6ad412af3197f`.
This is a single same-host before/after observation, not a crossed A/B/B/A
cohort or a runtime speed-promotion claim.

An explicit resident request at
`2026-07-29T15:31:06+02:00` failed closed before allocation:

```text
refusing explicit Qwen resident mode: hosts at or below 24 GiB are qualified
only for guarded SSD streaming
```

This is the intended production policy. The file's 12.29 GB size alone does
not include dense/static tensors, Metal working-set limits, context state,
workspace, and required safety headroom.

## Stable Affine4 physical 16 GiB lanes

The old runtime failed at the first FFN after prefill, so it has no valid
condition-matched performance baseline. Deltas against tested `main` and the
previous experiment are `N/A` at these distinct frontiers. Each row below is a
fresh process using AUTO and the default 4096-token prefill chunk.

| Started (Europe/Rome) | Frontier | Prefill t/s | Decode t/s | TPOT p50 / p95 | Free min / peak footprint | Effective cache | Delta vs tested `main` | Delta vs previous comparable | Result |
| --- | ---: | ---: | ---: | --- | --- | ---: | --- | --- | --- |
| 2026-07-29T15:57:15+02:00 | 128+8, ctx alloc 8192 | 32.37 | 7.17 | 125.728 / 172.101 ms | 48% / 4.10 GiB | initial 321 | N/A; old control failed decode | N/A; first valid post-fix arm | exact; zero swap |
| 2026-07-29T15:57:57+02:00 | 2K+128, ctx alloc 2177 | 261.68 | 14.43 | 60.105 / 119.231 ms | 35% / 5.49 GiB | froze at 2889 | N/A; old control failed decode | N/A; different frontier | exact; zero swap |
| 2026-07-29T15:58:35+02:00 | 8K+128, ctx alloc 8321 | 311.66 | 9.58 | 98.265 / 166.703 ms | 34% / 5.72 GiB | froze at 2889 | N/A; old control failed decode | N/A; different frontier | exact; zero swap |
| 2026-07-29T16:07:17+02:00 | 16K+128, ctx alloc 16513 | 293.22 | 8.64 | 110.276 / 169.778 ms | 37% / 4.98 GiB | froze at 2247 | N/A; old control failed decode | N/A; different frontier | exact; zero swap |

The 16K logits and decode-evidence SHA-256 values are
`3f2519365295c52d5ca9d42d77ab88b1ce8fbfd0175efd818744a801ae492e2e`
and
`a26bb9251e624c75f2200b992ab463feb2c12a60d8bd8406fe871adfeea872bb`.

## Condition-matched 2K profile comparison

This comparison changes the artifact and physical decoder, not the runtime
revision, hardware, prompt, AUTO→SSD plan, context, or generated-token count.
It is a profile tradeoff, not a `main` speed delta.

| Metric | Q2_K_XL Beta | Stable Affine4 | Observed tradeoff |
| --- | ---: | ---: | --- |
| Prefill throughput | 221.38 t/s | 261.68 t/s | Affine4 +18.204% |
| Decode throughput | 18.84 t/s | 14.43 t/s | Q2 +30.561% |
| TPOT p50 | 48.650 ms | 60.105 ms | Q2 -19.058% |
| TPOT p95 | 81.462 ms | 119.231 ms | Q2 -31.677% |
| Prefill plus decode wall | 16,043.501 ms | 16,696.174 ms | Q2 -3.909% |

These numbers do not compare model quality. Affine4 remains Stable and
recommended; Q2_K_XL remains an opt-in Beta with a 64 GiB published minimum.

## Rejected and invalid arms

The manual prefill-chunk sequence used 4096, 8192, 2048, then 4096. The two
candidate arms had exact output and no new swap, but the closing 4096 control
swapped 80 pages (1.25 MiB). Repository policy therefore invalidates the whole
timing cohort, including already completed arms. It cannot support a tuning
claim.

| Started (Europe/Rome) | Invalid lane | Raw prefill / decode | Raw TPOT p50 / p95 | Failure and decision |
| --- | --- | --- | --- | --- |
| 2026-07-29T16:04:40+02:00 | closing 8K default control | 311.45 / 9.37 t/s | 101.084 / 165.790 ms | 80 new swapout pages; watchdog exit 143; entire chunk cohort invalid |
| 2026-07-29T16:09:49+02:00 | 32K+128 AUTO | 239.76 / 7.70 t/s | 126.535 / 182.025 ms | inference/evidence completed, then 32 new swapout pages (512 KiB); watchdog exit 143; no 32K performance claim |

The 32K raw logits and decode evidence are retained, but the arm is not a
qualification result. This session's highest clean new frontier is 16K. That
observation does not narrow the existing product contract or replace its prior
qualification evidence.

## Recommended 16 GiB settings

- Use normal AUTO startup. It resolves to guarded SSD on this hardware.
- Keep the default prefill chunk. The invalid tuning cohort supplies no valid
  evidence to replace 4096.
- Allocate only the context needed by the workload. For benchmark frontiers,
  use `ctx + generated tokens + 1`; for normal CLI use, avoid reserving the full
  model window when the workload needs much less.
- Leave expert-cache sizing to the guarded planner. It safely froze at 2889
  slots at 2K/8K and 2247 at 16K instead of forcing another 541.69 MiB slab.
- Keep large GGUFs in a non-indexed local directory so Spotlight does not
  compete for CPU and storage.
- Do not force Q2 resident mode or use admission bypasses.

## Raw evidence

The retained bundles on the physical host are:

- `/private/tmp/hebrus-q2-16g-a1d7398-safety-20260729T152721+0200`
- `/private/tmp/hebrus-q2-16g-a1d7398-p8k-cold-20260729T152805+0200`
- `/private/tmp/hebrus-q2-16g-a1d7398-p8k-preload-20260729T152957+0200`
- `/private/tmp/hebrus-q2-16g-a1d7398-resident-admission-20260729T153106+0200`
- `/private/tmp/hebrus-q2-16g-a1d7398-p2k-auto-20260729T153156+0200`
- `/private/tmp/hebrus-affine4-16g-a1d7398-safety-20260729T155715+0200`
- `/private/tmp/hebrus-affine4-16g-a1d7398-p2k-auto-20260729T155757+0200`
- `/private/tmp/hebrus-affine4-16g-a1d7398-p8k-auto-20260729T155835+0200`
- `/private/tmp/hebrus-affine4-16g-a1d7398-p16k-auto-20260729T160717+0200`
- `/private/tmp/hebrus-affine4-16g-a1d7398-p32k-auto-20260729T160949+0200`
