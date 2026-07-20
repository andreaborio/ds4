# SSD-streaming campaign — independent verification (2026-07-10)

> [!IMPORTANT]
> This is superseded historical campaign evidence, not a current runtime or
> tuning guide. Its branches, environment switches, artifacts, and relative
> `main` references describe the 2026-07-10 code only. Current support and
> release commands are defined by `docs/contracts/RUNTIME_SUPPORT.md`,
> `GOLD_METAL_SSD.md`, and `QA_BEFORE_RELEASES.md`.

Independent re-verification of the claims in an external 2026-07-10 campaign
handoff that was not retained in this repository. Every claim was re-run from
the documented reproduction on the same box, paired A/B back-to-back, per the
speed-regression methodology in `CONTRIBUTING.md`.

- Machine: MacBook M5 Pro 64 GB, macOS 26 (Darwin 25.5.0), Metal, SSD AP1024Z 1TB.
- GLM model: GLM 5.2 `glm52-ds4-native-64g.gguf` (ds4-native layout), `--ssd-streaming`.
- DeepSeek model: DeepSeek-V4-Flash IQ2XXS (w2Q2K, AProj/SExp/Out Q8, chat-v2 imatrix), `--ssd-streaming`.
- Fork binaries: worktree at `feat/streaming-staging-opt` e02cf53, `make ds4 ds4-bench` up to date.
- All runners executed as `.sh` files via `sh` (zsh word-splitting trap).
- Runs strictly sequential, one model process at a time, no builds during
  benches. End state restored: upstream clone back on `main`, fork worktree
  back on `feat/streaming-staging-opt` e02cf53 and rebuilt.
- Related PRs: antirez/ds4#520, antirez/ds4#528, antirez/ds4#434.

## Summary

| claim | expected | measured | verdict |
|---|---|---|---|
| C1 regression: glm5.2 line slows DS decode ~2.8x | main 7.3-7.8 vs glm5.2 2.2-2.9 t/s | main 6.8-8.3 vs glm5.2 line 1.9-2.8 t/s | CONFIRMED |
| C2 GLM indexed prefill prepare (PR #528) | 3.5-3.9 -> 8.2-8.9 t/s (x2.4) | 3.88/4.10 -> 9.60/10.28 t/s (x2.5) | CONFIRMED |
| C3 DS chat: advisory per-miss OFF +7-13% | 4.9-6.0 -> 5.6-6.5 t/s | warm steady state: 6.88 vs 6.78 (no gain) | NOT REPRODUCED at steady state |
| C4 GLM decode bottleneck = router readback | sync_avg ~3.3 ms x 7125 calls (~37%) | sync_avg 4.42 ms x 7125 calls (~40%) | CONFIRMED (stronger) |
| C5 the 4 opt-in patches are default-neutral | identical behavior with envs unset | code-audit: all gated; one always-on extra fd | CONFIRMED (one caveat) |
| C6 GLM MTP gate: acceptance 56-58% -> NO-GO | acc 0.55-0.58, ppl ~2.7 | acc 0.5516 final (0.53-0.58 running), ppl 2.689 | CONFIRMED |

## C2 — GLM indexed prefill prepare (PR #528): CONFIRMED

288-token prompt (`bench-logs/symptom-prompt.txt`; the handoff says 271 —
the CLI reports 288 input tokens after templating), `-n 16 --temp 0`, base env
`DS4_GLM_ROUTER_AHEAD_PREFETCH=1 DS4_METAL_DISABLE_STREAMING_EXPERT_READAHEAD=1`,
OFF/ON interleaved x2:

| arm | prefill t/s | gen t/s |
|---|---|---|
| OFF rep1 | 4.10 | 1.13 |
| ON  rep1 | 10.28 | 1.04 |
| OFF rep2 | 3.88 | 1.01 |
| ON  rep2 | 9.60 | 0.99 |

Prefill x2.5 (claim x2.4), decode unchanged within variance, greedy output
byte-identical across all four arms. Absolute numbers slightly above the
handoff bands (warm page cache).

## C3 — DS chat advisory per-miss OFF: NOT REPRODUCED at steady state

`-p "Spiega in due frasi cos'e' un mutex." -n 48 --temp 0`, default vs
`DS4_METAL_DISABLE_STREAMING_EXPERT_READAHEAD=1`, 5 interleaved pairs:

| pair | base gen | advisory-OFF gen |
|---|---|---|
| 1 (cache warming) | 6.04 | 6.64 |
| 2 | 6.91 | 6.88 |
| 3 | 6.83 | 6.76 |
| 4 | 6.83 | 6.66 |
| 5 | 6.94 | 6.83 |

The +10% shows only on the first (cache-warming) pair. At steady state the
two arms are equal (means 6.88 vs 6.78, advisory-OFF marginally worse).
Mechanism is consistent: with a warm page cache there are few expert misses,
so a per-miss advisory has nothing to cost. Downgrade: not a standing config
recommendation; at most a cold/mixed-regime effect. Note both steady-state
arms sit above the handoff's expected bands, so page-cache state dominates
the effect being claimed.

## C4 — GLM decode bottleneck is the router readback: CONFIRMED

Same command as C2 with `-n 96` + `DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1`:

- `selected_calls=7125` — exactly 75 MoE layers x 95 decode tokens.
- `sync_avg=4.419 ms` (read_avg 4.420: the read IS the sync wait) -> ~331 ms/token.
- gen 1.20 t/s -> ~833 ms/token -> sync is ~40% of the token budget.
- Cross-check: `miss_pread=534.91 GiB` matches the "~535 GiB per-run miss
  traffic" figure in the e02cf53 commit message to the decimal.

Handoff said 3.3 ms / ~244 ms / 37%; this run measured it higher (4.4 ms /
331 ms / 40%). Same conclusion, stronger: the per-layer selected-readback
sync is the single dominant serial cost of streamed GLM decode. The
sync-elimination proposal draft (2026-07-09) still cites 150 ms / 33% —
update it to the measured range before publishing.

## C5 — the four opt-in patches are default-neutral: CONFIRMED (one caveat)

Code audit of 3bcf775..e02cf53 (93264df QoS+chunk, 33e97cf single-cb split,
e02cf53 F_NOCACHE fd; 355 insertions):

- Every behavioral change sits behind its env gate, all default-off:
  `DS4_METAL_ENABLE_STREAMING_IO_QOS`, `DS4_METAL_STREAMING_EXPERT_PREAD_CHUNK_KB`
  (0/unset = whole-task pulls), `DS4_METAL_ENABLE_STREAMING_SPLIT_SINGLE_CB`,
  `DS4_METAL_STREAMING_SPLIT_MIN_MISSING` (default 3 = the old hardcoded
  threshold), `DS4_METAL_STREAMING_EXPERT_NOCACHE`.
- The inflight check now also reads an atomic async-done sequence; it stays 0
  unless single-cb mode publishes it — same result with envs unset.
- Caveat (resource, not behavior): `model_open` now always opens a second
  F_NOCACHE file description, even with the env unset. The fd is only used
  when `DS4_METAL_STREAMING_EXPERT_NOCACHE=1` (F_NOCACHE is per-description,
  so the default fd's caching is unaffected), but the open is unconditional.
- Runtime consistency: the C2 OFF arms and C3/C4 default arms on e02cf53
  reproduce the historical 3bcf775-line baselines.

All four measured neutral-or-negative by the campaign; do not enable them
expecting gains (F_NOCACHE arms lose 23-45%).

## C1 — glm5.2 line regresses DS decode ~2.8x: CONFIRMED

`ds4-bench`, promessi_sposi.txt, ctx 2048(-4096), gen 32, DeepSeek Flash
IQ2XXS `--ssd-streaming`. Two probes rebuilt from scratch in the upstream
clone (`bisect2.sh`), then three arms interleaved x2
(`final-vs-upstream2.sh`), then a probe of current upstream main.

Bisect points (upstream glm5.2 code):

| commit | gen t/s | gen_first_ms |
|---|---|---|
| 34b3736 (pre pro-correctness merge) | 2.76 | 5342 |
| 173a9a3 "Add GLM 5.2 runtime support" | 1.94 | 5346 |

Interleaved arms, gen t/s at ctx 2048 / 4096:

| arm | rep1 | rep2 |
|---|---|---|
| U  = upstream main (5b95fa1) | 7.27 / 7.15 | 6.83 / 8.30 |
| F0 = fork GLM line (e02cf53), default env | 2.42 / 2.62 | 1.88 / 2.67 |
| F1 = F0 + DISABLE_STREAMING_EXPERT_READAHEAD | 1.91 / 2.73 | 2.35 / 2.78 |

Extra probe not in the handoff: current upstream main 80ebbc3 does 6.49 gen
t/s at ctx 2048 — the regression has NOT reached main, and the issue draft
is still current. The regression lands with 173a9a3 itself, persists through
the whole line including the fork's GLM branch, and no env configuration
rescues it (F1 = F0). First token ~5.2-7.2 s vs sub-second on main. Prefill:
no clear regression once warm (rep2: F0 107/120 vs U 118/116 t/s); rep-1
prefill deltas are cache-warming noise.

## C6 — GLM MTP gate (nextn acceptance): CONFIRMED, NO-GO stands

Branch `feat/glm-mtp-probe` (fbde3a0) rebuilt in place, then
`DS4_GLM_NEXTN_PROBE=1 ./ds4 -m GLM --ssd-streaming
--perplexity-file bench-logs/code-probe.txt`:

- `acc_vs_main`: 0.5516 final over 640 tokens (running values 0.53-0.58);
  handoff band 0.55-0.58, endpoint marginally below but equivalent.
- `ppl=2.689` on the code probe (expected ~2.7): the probe does not perturb
  the main path.
- acc+1=0.51, acc+2/acc+3 ~0.02: no depth to recover either.

Against the 75% acceptance threshold for I/O-bound MTP to pay off, 55% keeps
the K=2 speculative decode NO-GO for GLM. Worktree restored to
`feat/streaming-staging-opt` e02cf53 and rebuilt afterwards.

## Handoff corrections found during verification

1. Handoff says fork main = upstream main "(5b95fa1)" + 19 commits. The base
   is actually current upstream main 80ebbc3 (5b95fa1 is 54 commits older);
   the "+19 fork commits, clean" part is correct.
2. The PR #520 worktree carries an UNPUSHED extra commit 7c9f033 ("metal:
   refuse resident model maps larger than physical RAM") on top of the PR
   head b85f6f8. Not mentioned in the handoff. Good standalone upstream
   candidate (no overlapping upstream PR/issue found); needs its own branch,
   rebase and test pass before proposing.
3. C3's recommended config does not reproduce at steady state (see above).
4. C4's cost figure is machine-state dependent: 3.3-4.4 ms/call across the
   two measurement days; quote the range, not a point.

## Upstream-relevant items (verified, no PR overlap)

- PR #520 (fix, pushed b85f6f8 = local): independently validated by a third
  party on an M4 Max 128 GB (full `make test` pass + 17,623-token and
  86,361-token prefill repros). Mergeable.
- PR #528 (pushed 77a1732 = local; patch byte-identical to the fork-line
  commit 3bcf775 modulo context offsets): body carries machine, backend,
  quant, correctness and speed evidence per CONTRIBUTING. Stacked on #520;
  rebase to a single commit if #520 lands first.
- External DS regression issue draft from the campaign:
  no overlapping upstream issue found, and the fresh 80ebbc3 probe (6.49 gen
  t/s) confirms it is still current. Two edits before publishing: the draft
  cites main as 5b95fa1 (add the 80ebbc3 datapoint) and its per-commit
  numbers moved a little on re-run (173a9a3: 2.21 -> 1.94; 34b3736:
  2.88 -> 2.76) — quote ranges.
- Sync-elimination proposal draft: update the readback cost to the measured
  range (3.3-4.4 ms/call, 33-40% of token) before publishing.
- 173a9a3 (the regressing commit) is NOT in upstream main; upstream/glm5.2
  is still at bd89932. The regression is confined to the glm5.2 line.
