# Long-context Metal optimization stack

Date: 2026-07-20

Status: release-candidate evidence from the preceding candidate binary; the
final-source reruns and repository/DSBox release gates remain open.

Decision: retain the exact paired-Q8 decode kernel, the DeepSeek long-context
cache-phase policy, and the GLM compact-indexer mapping correction. The
DeepSeek SIMD router remains provisional until the final combined-stack cohort;
this record does not assign it an isolated percentage gain.

Supersedes: current performance interpretation in
`2026-07-20-agent-friendly-refactor-validation.md` and the short-only release
interpretation in `2026-07-20-glm52-expert-major-v2.md`. Those records remain
authoritative for artifact provenance and earlier rejected experiments.

## Scope and rules

This campaign introduced the short/medium/large/long acceptance matrix now in
`CONTRIBUTING.md`. Short runs may reject a candidate, but only the complete
context evidence can promote it. Arms ran sequentially on the same Apple M5 Pro
with 64 GiB unified memory, macOS build 25F84, AC power, and no synthetic SSD
stress. Cold and warm observations were not averaged together.

The qualified ExpertMajor v2 artifacts were:

| Family | Bytes | Complete artifact SHA-256 from the campaign manifest |
| --- | ---: | --- |
| Qwen3.6-35B-A3B | 20,808,566,880 | `d7c43a6388ec20e6fe5530850350f96fdb0ac37c5ce36d3e5f92b172c447f56b` |
| DeepSeek V4 Flash | 86,720,114,272 | `8378080263eb9224f7228d72e2afa4ac3cf74a116023fdec2c596ff228a33e3f` |
| GLM 5.2 | 262,147,193,504 | `7f5017e3076e706c78f2a5322b035a9e2f6519c65ff5b6be8b2d91aeff61505d` |

The three complete GGUF files were rehashed once after the campaign and matched
the identities above. They were not rehashed inside every arm. The runner
records the expected identity, model byte count and mtime; it does not claim
that `model_sha256_expected` is a fresh per-arm full-file hash.

The retained candidate Metal source identity was
`1c2413b76fa34300c5f473b87288540c6d71e2d975b53d48d86ecf87df60e02d`.
Qwen's clean baseline used
`e57502f1ebab8dbb1b797a76dc50e7947537f4ea00910704c50bdefeb29b4c08`.

## Qwen paired-Q8 result

The Qwen arm compares the original scalar Q8 decode calls with one exact
paired call. Separate discarded warm-ups preceded a warm A/B/B/A cohort.
All four retained 32K arms had zero swapout, identical resolved resident plans,
no competing inference process, and identical logits/evidence content hashes:

- logits: `2d383f416ae85249b6ff91e04e92bf05d61e1b35c6b864b6a1bcf78a80f0d4c8`;
- decode evidence: `d73b694d169cf2a5ce1e7563bb844e6b2714cede5d9ba69213cf4aebb4fd3c9e`.

| 32K arm | Binary | Prefill | Decode wall | TPOT p50 | TPOT p95 |
| --- | --- | ---: | ---: | ---: | ---: |
| A1 baseline | `d963d673…` | 60.44 t/s | 42,579.471 ms | 333.207 ms | 337.661 ms |
| B1 paired | `7ef6e565…` | 60.88 t/s | 42,480.777 ms | 332.998 ms | 334.233 ms |
| B2 paired | `7ef6e565…` | 60.21 t/s | 42,583.592 ms | 333.251 ms | 334.227 ms |
| A2 baseline | `d963d673…` | 59.92 t/s | 42,533.631 ms | 333.060 ms | 336.997 ms |

Mean decode throughput changed by only +0.057%, less than within-arm spread,
so throughput is neutral. Mean p95 fell from 337.329 to 334.230 ms (-0.92%).
That 3.099 ms effect exceeds both the 0.664 ms baseline p95 spread and the
0.006 ms candidate spread. The production decision is therefore a measured
long-tail reduction, not a general tokens-per-second claim.

The earlier isolated 128, 2K and 8K A/B/B/A tiers also preserved exact output.
Their decode means were 43.97/44.35, 21.90/21.93 and 10.12/10.18 t/s for
baseline/candidate. These are neutral-to-small observations; the 32K p95 result
is the acceptance reason.

## DeepSeek phase-adaptive cache result

The unsafe long-prompt AUTO path shrank the 4,387-record cache to the 259-record
prefill floor and restored an empty 4,387-record decode cache. It produced
about 3.6 t/s at 8K and could drive swap. The corrected policy separates total
context from suffix work:

1. start at the normal short-context 4,387-record target;
2. shrink to 259 records only while a large batched prefill needs the memory;
3. grow to 4,129 records through 32K, or 2,065 records at 65K and above,
   after successful prefill;
4. seed the hotlist once; cancellation/error restores capacity without seeding.

The lower-memory tier is admitted 128 tokens before each hard frontier. This
bounded guard avoids a near-8K or near-65K session growing a larger cache only
to drain and shrink it again during the same decode window.

Both targets are bounded by the live plan and complete per-token working-set
cycles. A live macOS pressure check also blocks post-prefill growth whenever
pressure is unavailable or non-normal. Tiny resume suffixes use total context
for memory policy and suffix length only for the work schedule.

| Warm candidate lane | Prefill | Decode | TPOT p50 | TPOT p95 | Pressure min | Swapout |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 8K+128 | 161.68 t/s | 8.46 t/s | 91.425 ms | 125.570 ms | 18% | 0 |
| 32K+128 | 137.04 t/s | 7.46 t/s | 102.211 ms | 140.538 ms | 19% | 0 |
| 65K+128 | 141.61 t/s | 5.72 t/s | 157.035 ms | 183.723 ms | 39% | 0 |
| 100K+128 | 126.72 t/s | 5.58 t/s | 158.477 ms | 192.283 ms | 35% | 0 |

The 8K content hashes were
`0af1ae94765dc97e722cb426b58749ae2d2f14e47a50dd9ec682d22903e1c9ec`
for logits and
`82ee4a713c50589a81ddd69cb7184281aade7f283570e084a4db2b05648ef24a`
for decode evidence. The 32K hashes were
`c95c0928f92fd6538894ad81ea218a03c03fdb9b7dc71c3cb38bd4fc917e83b5`
and
`5693ea1becef5be76f19370125b5e70b0dd54a146b743a3b4b7fdd5c69af42ba`.

An explicit safe 4,129-record 8K reference measured 161.36/8.24 t/s with
93.364/128.168 ms p50/p95. The combined candidate is better in that single
matched observation, but it is not a retained A/B/B/A cohort. This record
therefore treats the change primarily as a memory-safety and phase-correctness
fix and does not publish a standalone speedup percentage.

The isolated 65K and 100K arms used runtime AUTO, which resolved to SSD, a
131,072-token allocation, and deterministic extensions of the checked-in
security prompt. Their prompt SHA-256 values were respectively
`b03c9d3458bf5fe4d5943d7a66a442df50adff7ef7403e44809e44f5a015193e`
and
`a433c538bedf7df589e3fbeeefee9f8867bd0639ddf58d2af30e4ef463034cb6`.
Both used binary
`5ea1e65fb42adbe81661740ab056841df3ea82c996fd108ecee241dd76f8bdec`,
produced complete logits/evidence, and saw no competing inference process.
At 65K the logits/evidence content hashes were
`2406b1dfd16989e652bacb3b0f2995e763824605551683c21ac79d99396846e3`
and
`edc2adbce5bddde908e9ba269aeb452721596e07044fe9dcf3bb71ee2181528b`;
at 100K they were
`cf43fb3cf5ff7c1499a5ce634e0fa294c920c7ff268ad56ac8445e69403740a5`
and
`2d6179263ac0791311c45de838a7bba257cb2cf2e3a4fd0bea99ec425298a610`.

## GLM compact-indexer transition

The first GLM 32K attempt completed all prefill work, then failed before logits:

```text
ds4: Metal model range 0.41..0.41 GiB is not covered by mapped model views
ds4-bench: prefill to 32768 failed: metal GLM compact indexer warmup failed
```

The final indexed-prefill chunk had replaced the current per-layer view with
the output-head view. The frontier warmup then read `indexer_k_norm` and bias
for every full-indexer layer. Those 42 F32 tensors were valid but no longer
covered by a Metal model view.

The correction constructs the exact cross-layer span set, validates every
required norm/bias pair, and installs it before the batched warmup. Page
alignment collapses the 42 adjacent tensors into 21 disjoint views totaling
about 0.33 MiB. It does not map the multi-GiB decode-static set or add a model
copy. The static-decode map state is invalidated before the remap so a failed
view rebuild cannot leave a stale cache bit.

Model-free coverage now includes an actual Metal transition from an output-only
view to two disjoint norm views followed by an indexer store at position 4096.

Incremental real-model gates used `ctx_alloc=65536`; a smaller allocation would
raise the internal full-attention cap and mask this exact boundary.

| Candidate lane | Prefill | Decode | TPOT p50 | TPOT p95 | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| 4K+1 boundary | 39.61 t/s | cold first token 0.23 t/s | 4,351.509 ms | 4,351.509 ms | exact transition passes; zero swap delta |
| 8K+8 canary | 40.29 t/s | 0.95 t/s overall | 551.587 ms | 4,691.184 ms | indexed prefill passes; p50 is about 1.81 t/s steady decode |
| 32K+128 acceptance | 36.19 t/s | 1.62 t/s overall | 558.710 ms | 731.082 ms | `rc=0`; p50 is about 1.79 t/s steady decode |

The final 32K binary SHA-256 was
`f3e68d9e8c69b6a33d67e03ad976941b6149205557e8dab14be51a7c94d79c81`.
The arm recorded pressure 83% before, 23% minimum and 84% after; wired memory
peaked at 37,871 MiB; swapout delta was zero; no competing inference process
appeared. Logits/evidence content hashes were
`fc048bce4865b2c6a0df8951fef892d64e2058fa1980876895a46f300c91cc0e`
and
`fd41083cb825ef9e754052398f79c82d2141dfd12d388e04e03cc8ad50a99320`.

There is no meaningful 32K speedup percentage: the baseline fails the
correctness transition and produces no logits. Performance acceptance is that
the fixed path preserves the established roughly 1.8 t/s steady GLM decode
while making the long-context lane complete successfully.

## Invalid and discarded runs

- Qwen candidate and baseline warm-ups were discarded symmetrically before the
  retained A/B/B/A cohort.
- A DeepSeek 8K arm was aborted by the earlier runner when transient free
  percentage crossed 20%. It had no swapout but the complete cohort is invalid.
  The runner now records transient pressure and aborts on swapout, wired-memory,
  timeout or process contamination.
- The original GLM 32K arm is invalid because the compact-indexer warmup failed
  and produced no evidence.
- A fixed GLM boundary invocation passed the warmup but then failed because its
  dump directory did not exist. It is invocation evidence only.
- The first corrected GLM 32K run was stopped deliberately after independent
  review found the stale-state failure-path ordering. No result from that arm
  is retained.
- The first DeepSeek 65K fixture contained only 36,831 model tokens and failed
  before inference. The runner now extends undersized long prompts
  deterministically and records source/effective hashes.
- The first complete DeepSeek 65K prefill restored 4,129 experts, drove free
  pressure to 13%, and added 35,772 swapout pages before decode. The runner
  aborted it. Reducing the 65K+ tier to eight complete routing cycles (2,065
  experts) cut peak wired memory from 42,016 to 28,194 MiB and completed with
  zero swapout; the failed arm contributes no performance result.
- A final-source 8,190-token decode canary exposed a 259-to-4,387-to-4,129
  grow/shrink cycle and 36 page-outs. The runner invalidated that arm. The
  128-token transition guard above removes that churn; its replacement canary
  remains part of the final-source gate.

## Remaining release gates

The measured code paths are retained, but this record alone does not authorize
release. Before merge:

- run `make premerge` on the frozen source tree and complete a final full-diff
  and residue review;
- run fresh final-binary short smokes for Qwen, DeepSeek and GLM;
- rerun DeepSeek 65K/100K and a decode canary that crosses 8K on the final
  source, then complete the required combined-stack comparison;
- keep manual API, agent, disk-KV, packaging, DSBox and Hugging Face publication
  checks separate from performance acceptance.

The GLM mapping correction does not alter attention, KV geometry, RoPE,
context allocation or context scaling, so it does not create a new 65K/100K
performance claim. The 4K boundary and 8K/32K indexed lanes exercise the
changed transition directly.

Raw local campaign artifacts are under
`/tmp/ds4-q8-final-matrix.MquCV1` and
`/tmp/ds4-long-adaptive.VnKgWS`, with final extended-context evidence under
`/tmp/ds4-release-validation.zZvXI7`, for the release session. This record
carries the durable metrics and content hashes; temporary paths are not
artifact identity.
