# Long-context Metal optimization stack

Date: 2026-07-20

Status: final-source performance and safety evidence is complete for the
retained Metal paths. DeepSeek carries no promoted 32K speedup percentage
because the original AUTO controls can swap. Manual API, agent, and persistent
disk-KV release gates passed on the frozen build. Revision-pinned packaging and
normal startup also passed. Model-artifact tagging on Hugging Face is complete;
DSBox, main, and the final Hugging Face cards/runtime revisions remain open.

Decision: retain the DeepSeek long-context cache-phase policy and the GLM
compact-indexer mapping correction. Reject and remove both the exact paired-Q8
decode experiment and the experimental DeepSeek SIMD router because their
A/B/B/A cohorts did not show a win.

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

The retained runtime code is commit
`5479c42e52116ee15766b6e0e00391f940d77215`. Its clean `ds4-bench` SHA-256 is
`1decc95495f91b7d478d514cf508bc48f2db5cc50525f1c3557115dce260c69e`;
the assembled Metal source SHA-256 is
`f143a7e86f3b5e4b7a4b8c92ec3e7cee0a8e1164d5f0fa969dcb056eb1ae2e67`.
The later hermetic-runner commit `25466e5f4a9c` changes build provenance and
benchmark tooling only, not C/Metal runtime objects. Its clean diagnostic
`ds4-bench` binary is
`144f6be8550a7b754290d97b250dd964da1db126922d86f8d72213d36281a9c4`.
The instrumented original baseline binary is
`d963d673f642d467206a5aff697322131babf9365f284f1013cf15a9ac3427d4`.

## Qwen paired-Q8 rejection

The final Qwen arm compared the original scalar Q8 decode calls with one exact
paired call. Separate discarded warm-ups preceded a warm A/B/B/A cohort. All
four retained 32K arms had zero swapout, identical resolved resident plans, no
competing inference process, and identical logits/evidence content hashes:

- logits: `2d383f416ae85249b6ff91e04e92bf05d61e1b35c6b864b6a1bcf78a80f0d4c8`;
- decode evidence: `d73b694d169cf2a5ce1e7563bb844e6b2714cede5d9ba69213cf4aebb4fd3c9e`.

| 32K arm | Path | Prefill | Decode wall | TPOT p50 | TPOT p95 |
| --- | --- | ---: | ---: | ---: | ---: |
| A1 | scalar | 67.03 t/s | 40,179.787 ms | 314.468 ms | 317.460 ms |
| B1 | paired | 67.06 t/s | 40,291.136 ms | 315.576 ms | 317.506 ms |
| B2 | paired | 67.04 t/s | 40,255.507 ms | 315.213 ms | 317.824 ms |
| A2 | scalar | 67.03 t/s | 40,255.622 ms | 315.168 ms | 317.666 ms |

The scalar means were 67.03 prefill t/s, 40,217.705 ms decode wall (about
3.18 t/s), 314.818 ms p50, and 317.563 ms p95. The paired means were 67.05
prefill t/s, 40,273.322 ms decode wall, 315.395 ms p50, and 317.665 ms p95.
The paired path was neutral-to-worse in every decode measure, and each mean
difference was smaller than the relevant cohort spread. It therefore has no
measured production benefit. The paired kernel, dispatch implementation, and
dedicated test were rejected and removed; Qwen retains the exact scalar path.

Earlier isolated 128, 2K, and 8K cohorts also preserved exact output but did
not establish a durable win. They remain exploratory history rather than an
acceptance basis.

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

The primary security/coding prompt used SHA-256
`e7c1a2cadf781d274cc26bd251d532fe1b9e632080da97e3eb4684741e7cc308`.
Each comparison used the same prompt, model, 128-token decode, allocation,
warm-state label and sequential host lane. The 128/2K/8K A/B/B/A cohorts had
zero swapout, no process contamination and exact output parity between the
original baseline and complete stack. The 32K candidate arms also preserved
exact output, but both attempted original-baseline cohorts were invalidated by
an A2 swap abort; no 32K speedup percentage is promoted.

| Security/coding lane | Baseline prefill/decode | Stack prefill/decode | Baseline/stack TPOT p95 | Interpretation |
| --- | ---: | ---: | ---: | --- |
| 128+128 | 24.64 / 13.06 t/s | 24.72 / 12.97 t/s | 83.95 / 84.84 ms | Neutral; candidate spread dominates |
| 2K+128 | 196.85 / 10.02 t/s | 156.22 / 10.99 t/s | 188.53 / 88.05 ms | p95 -53.3%; throughput controls drifted, so no t/s percentage |
| 8K+128 | 211.80 / 8.57 t/s | 195.07 / 8.68 t/s | 239.01 / 138.26 ms | p95 -42.2%; throughput controls drifted, so no t/s percentage |
| 32K+128 | 187.34 / 7.58 t/s | 177.85 / 7.78 t/s | 288.25 / 145.47 ms | Directional only; A2 swap invalidated the cohort |

The valid 2K/8K p95 control drifts were 2.03% and 1.23%; the 53.3% and 42.2%
tail reductions exceed control drift and candidate spread. Decode throughput
is neutral or inconclusive under the stricter rule. At 32K two separate
attempts again showed roughly half the candidate p95, but both ended with an
unsafe original A2 and are retained only as directional/safety evidence. The
stack also pays an honest prefill cost to seed 4,096 hotlist records instead of
the baseline's 259-record floor: about 20.6%, 7.9%, and 5.1% at 2K, 8K, and
32K in the first attempt. The production decision retains that cost because the
old AUTO policy can swap at 32K, while the corrected candidate completes 32K,
65K, and 100K with exact output and zero swap.

The logits/decode-evidence content hashes for 128, 2K, 8K, and 32K were:

| Frontier | Logits content SHA-256 | Decode evidence content SHA-256 |
| --- | --- | --- |
| 128 | `20753418e04ef834524319d2d5fc48fdc5d6886a2ed905cc21b91b325d4e587f` | `a6e232d06f43e280b8dba70b1053e74136839cd14ee0c254be1e8024c2969c2b` |
| 2K | `b428054a77166c234f69cfa5a5828ef194198561483761ba7d67aaf859d2c8c3` | `ebd2378a1f4eb41de71b5b1b2be6cc29e6d7fb3c241efb3b5d57d765a315ae3b` |
| 8K | `d05af5406613ff2a93223720ae5ad96f64d2365a0c10ce36b24a41db2e84e6b2` | `705014d39271c41722e3b301d7bc9c081671700d9d8b3230cb2cbd05c90e68ea` |
| 32K | `56a6592cdad1ab94c072d1406a1d4872922802b1254ae59c929fa9aaee14923c` | `a4d3324620bba4ba3f9e47a2d1248322dc1460ef14476710b097dca7c4aa0f3d` |

The final 8,190+128 canary crossed the 8K boundary directly from the 259-record
prefill floor to 4,129 records. It measured 162.49 t/s prefill, 8.36 t/s
decode, 93.559/145.948 ms p50/p95, 16% minimum pressure, and zero swapout.

A later hermetic 32K A/B/B/A retry gave every arm an identical 128+128
discarded warm-up. A1 completed at 151.42/6.48 t/s with 341.226 ms p95 and zero
swap. B1/B2 completed at 147.79/6.75 and 148.46/6.57 t/s with 166.359 and
173.808 ms p95, exact content and zero swap. A2 completed prefill but the guard
stopped it before decode after 6,376 swapout pages and 13% minimum pressure.
The complete cohort is invalid for speed claims; it independently confirms that
the original AUTO plan is unsafe in a sequential 32K lane while the combined
stack remains bounded.

A second 32K diagnostic used the prose/locality prompt
`f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`
on clean runner commit `25466e5`. It measured 164.43/7.27 t/s,
105.410/149.943 ms p50/p95, an 86.73% decode hit rate, 15% minimum pressure,
and zero swapout. This is a separate final-stack workload check, not a speed
comparison with the security cohort or an earlier differently warmed arm.

The experimental SIMD top-6 finalize/weights fusion was rejected before the
long matrix. Its warm 128-token A/B/B/A decode results were 12.95, 13.12,
13.01, and 13.49 t/s, where A is the existing two-dispatch fast path and B the
SIMD candidate. The A controls differed by more than 3%, and the B mean was not
faster than the A mean. A short lane may reject a candidate under the campaign
rules, so the kernel, dispatch and dedicated test scaffolding were removed.

The final isolated 65K and 100K arms used runtime AUTO, which resolved to SSD,
a 131,072-token allocation, and deterministic extensions of the checked-in
security prompt. They measured 137.29/6.59 and 145.11/6.44 prefill/decode t/s,
with 133.411/169.059 and 133.580/174.592 ms p50/p95. Pressure minima were 35%
and 34%, wired peaks 27,727 and 27,860 MiB, and both had zero swapout. Their
prompt SHA-256 values were respectively
`b03c9d3458bf5fe4d5943d7a66a442df50adff7ef7403e44809e44f5a015193e`
and
`a433c538bedf7df589e3fbeeefee9f8867bd0639ddf58d2af30e4ef463034cb6`.
Both used the final runtime binary
`1decc95495f91b7d478d514cf508bc48f2db5cc50525f1c3557115dce260c69e`,
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

| GLM lane | Prefill | Decode | TPOT p50 | TPOT p95 | Decode cache hit | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 4K+1 boundary | 39.61 t/s | cold first token 0.23 t/s | 4,351.509 ms | 4,351.509 ms | n/a | exact transition passes; zero swap delta |
| 8K+8 canary | 40.29 t/s | 0.95 t/s overall | 551.587 ms | 4,691.184 ms | n/a | indexed prefill passes; p50 is about 1.81 t/s steady decode |
| Earlier 32K prose | 36.19 t/s | 1.62 t/s overall | 558.710 ms | 731.082 ms | 36.90% | corrected candidate; zero swap |
| Final-source 32K prose | 44.73 t/s | 1.87 t/s overall | 472.624 ms | 621.302 ms | 36.90% | about 2.12 t/s p50 steady; zero swap |
| Final-source 32K security/coding | 45.53 t/s | 1.33 t/s overall | 734.406 ms | 819.176 ms | 13.15% | adversarial routing lane; zero swap |

The same-prompt final-source prose observation uses prompt SHA-256
`f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`
and final runtime binary
`1decc95495f91b7d478d514cf508bc48f2db5cc50525f1c3557115dce260c69e`.
Its logits and decode-evidence
content hashes exactly match the earlier corrected arm:
`fc048bce4865b2c6a0df8951fef892d64e2058fa1980876895a46f300c91cc0e`
and
`fd41083cb825ef9e754052398f79c82d2141dfd12d388e04e03cc8ad50a99320`.
Pressure fell from 84% to a 22% minimum, wired memory peaked at 37,389 MiB,
swapout stayed zero, and no competing inference process appeared.

The security/coding prompt has SHA-256
`e7c1a2cadf781d274cc26bd251d532fe1b9e632080da97e3eb4684741e7cc308`.
It routed through 14,309 unique records instead of 11,698, reduced the 601-slot
cache hit rate from 36.90% to 13.15%, and increased decode expert reads from
559.92 to 769.44 GiB. The additional I/O explains the lower throughput; compute
time outside `pread` did not regress. Its logits/evidence content hashes were
`c9fcbda2fe8c6353a0eaf32d803b74112ad8d5fa0bf63825a1a2b0ebb55ccc26`
and
`a11298a3bd91c2dd04d8d5dd90266acf596ea447046a2d4311940707136df7dc`.

The same-prompt observation did not show a material final-source GLM decode
regression and exceeds the earlier top. It did not use a matched warm-up or an
A/B/B/A speed cohort, so this record does not publish a percentage gain. There
is also no
correctness-baseline speedup percentage: the original baseline fails the 32K
transition and produces no logits. GLM 32K is feasible on the 64 GiB host; the
cost is test duration, not a RAM-capacity failure.

## Invalid and discarded runs

- Qwen paired and scalar warm-ups were discarded symmetrically before the
  retained A/B/B/A cohort.
- A DeepSeek 8K arm was aborted by the earlier runner when transient free
  percentage crossed 20%. It had no swapout but the complete cohort is invalid.
  The runner now records transient pressure and aborts on swapout, wired-memory,
  timeout or process contamination.
- The first final-source DeepSeek 128 cohort began with a cold 11.77 t/s A1 and
  ended with a 13.45 t/s A2, so it was discarded before the retained warm
  cohort.
- In the first DeepSeek 32K comparison, A2 was stopped before inference after
  20 system swapout pages. A later zero-swap A2 retry is useful diagnostic data
  but cannot repair an interrupted cohort. The fully restarted hermetic cohort
  above was also invalidated when the original A2 added 6,376 swapout pages.
  Neither cohort supports a 32K speedup percentage.
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

## Functional release validation

The functional gate used a clean detached worktree at runner commit
`25466e5f4a9c`, whose runtime code is commit `5479c42e5211`. Binary SHA-256
values were `a24a0d96f8451be6ae74c6dc4b6a4e79d003a165fa2c2d14ce9a8127300eea5b`
for `ds4`, `bd061c0dd8aa6368f9c0e1183350feb80247e3fa00cbebb4a26c8809a50ec1f1`
for `ds4-server`, and
`967427b29cf98b14914074645a05241a419173832a969ef9340f1693deb8638e`
for `ds4-agent`. The assembled Metal source identity remained
`f143a7e86f3b5e4b7a4b8c92ec3e7cee0a8e1164d5f0fa969dcb056eb1ae2e67`.

With the qualified DeepSeek v2 artifact and normal AUTO startup,
`ds4-server` resolved to SSD streaming and passed:

- both `deepseek-v4-flash` and `deepseek-v4-pro` model aliases;
- OpenAI Chat, OpenAI Responses, Anthropic Messages, and SSE streaming;
- trace creation and a clean, draining shutdown;
- disk-KV writes followed by a fresh-process restart hit from the same cache
  directory (`tokens=1`, `load=1.2 ms`), then a new shutdown checkpoint.

The model downloader pins `ds4-v0.2.0` and verifies the qualified byte count
and complete SHA-256. The annotated tag was created and peeled back to the
qualified model-repository commit for DeepSeek, GLM, and Qwen. Public
`hf download --dry-run` checks resolved exactly one 86.7, 262.1, and 20.8 GB
v2 file respectively, without downloading another model copy.

The native DeepSeek agent passed a non-interactive one-turn smoke and a real
two-turn tool loop in a temporary project: `write` plus `bash` created, built,
and ran a C program; the next turn used the anchored `edit` tool rather than a
full rewrite, rebuilt it, and produced the changed output. A separate live TUI
gate exercised `/help`, `/power 50`, `/power 100`, `/save`, `/list`,
`/history`, `/strip`, `/new`, `/switch`, and `/del`. Stripped-session switch
rebuilt 1,987 tokens from rendered text. Ctrl+C interrupted a live prefill and
the same process accepted and completed the next prompt. The temporary session
was deleted after the gate.

Normal flag-free AUTO CLI startup also passed on all three qualified artifacts
at an 8,192-token allocation. Qwen resolved resident and emitted `QWEN_OK`;
DeepSeek resolved SSD and emitted `DEEPSEEK_OK`; GLM resolved to its required
SSD-only Gold profile and emitted `GLM_OK`. These tiny generations are startup
and prompt-rendering checks, not performance evidence.

The exhaustive interactive web-approval, queued-message, long-running bash,
and remote-terminal rendering matrix was not repeated because this release
delta does not change agent tools, TUI, server protocols, or disk-KV code. The
manual gates above validate normal startup and the historically critical
interruption/tool/session paths without presenting the untouched surfaces as
newly qualified evidence.

## Remaining release gates

The retained runtime paths have passed `make premerge`, independent full-diff
review, final-binary Qwen/DeepSeek/GLM short smokes, DeepSeek 8K-crossing and
65K/100K gates, both 32K prompt domains, and final-source GLM 32K. DeepSeek is
promoted as a safety/correctness and long-tail fix, not with a 32K throughput
percentage, because the original AUTO controls can swap.

This record alone still does not authorize publication. Before merge/release:

- rerun `make premerge` after the final documentation-only commit and complete
  one last residue review;
- validate the DSBox catalog/startup surface against the release commit;
- publish/verify main and Hugging Face only after those functional gates pass.

The GLM mapping correction does not alter attention, KV geometry, RoPE,
context allocation or context scaling, so it does not create a new 65K/100K
performance claim. The 4K boundary and 8K/32K indexed lanes exercise the
changed transition directly.

This record carries the durable metrics, prompt identities, binary identities,
and content hashes. Session-local scratch paths are intentionally omitted
because they are not durable artifact identity.
