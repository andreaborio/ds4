# Qwen 24 GiB sustained-decode cache guard — 2026-07-24

Status: draft-publication candidate implemented; exact-commit Qwen and DeepSeek
smokes plus model-free gates pass. Physical M5 24 GiB confirmation and the
managed-runtime pin advance remain pending.

Decision candidate: Qwen through 24 GiB must use guarded SSD mode. AUTO resolves
to SSD and an explicit resident request is rejected. The established 16 GiB
floor remains qualified; this 24 GiB allocation-time amendment remains
unreleased and requires the physical gate and Studio runtime-pin advance
before release.
Require
affirmative normal memory pressure at admission and every prefill/decode phase
entry, including an unchanged configured budget whose lazy slabs can still
populate. Before every proposed new slab (up to 321 experts; the final target
tail may be smaller), admit its exact bytes against a fresh host-pressure
snapshot and the Metal working-set ceiling. Denial freezes
the cache at already allocated slab capacity and forces eviction/reuse. The
planned routed-expert ceiling remains 3,521 experts. This is a correctness and
safety change, not a throughput claim.

Supersedes the uncapped 24 GiB SSD-cache portion and the 24 GiB resident-mode
eligibility in `2026-07-21-qwen-split-k-hardware-policy.md`. The resident
reserve formula itself, named hardware profiles, 16/32/64 GiB evidence, and
split-K results remain unchanged.

## Reported incident

The supplied Hebrus Studio event log shows Qwen producing tokens normally until
macOS memory pressure reached `WARNING`. Studio then deliberately sent
`SIGTERM`; Hebrus Server drained requests, and `client stream write failed` was
a secondary consequence of that shutdown. The exact source revision, artifact
hash, context allocation, host pressure history, and timezone were not captured,
so this report is reproduction evidence rather than a comparable benchmark.

| Timestamp | Revision / experiment | Host / model / mode | Prompt frontier | Prefill | Decode | TPOT p50 / p95 | Delta vs `main` | Delta vs previous | Result |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| N/A (event display: 07/24 10:46:36; timezone unavailable) | N/A, reported pre-fix build | Apple M5, 24 GiB / reported Qwen / AUTO | observed context 2,146; 1,719 generated tokens | N/A | 14.27 t/s rolling average | N/A | N/A | N/A | **ABORT** — macOS pressure `WARNING`; watchdog `SIGTERM`; stream failed |

Thinking `medium` reached the partial answer before the abort; thinking `high`
aborted earlier. That difference is consistent with cache slabs being populated
according to routed-expert diversity and decode duration. It is not evidence
that thinking mode itself failed.

## Planner defect and correction

Before this change, only Qwen hosts at or below 16 GiB received the measured
3,521-expert ceiling and the mandatory normal-pressure phase gate. A 24 GiB
host received full warm file-cache credit, then lazily allocated 321-expert
Metal slabs toward a much larger one-shot target.

The deterministic 24 GiB fixture uses an 18 GiB
`recommendedMaxWorkingSetSize`, 20 GiB reclaimable memory, 512 MiB persistent
runtime, and 2.5 GiB pageable static coverage. It isolates the policy
arithmetic; it is not hardware throughput evidence.

| Policy identity | Cache experts | Cache bytes | Delta vs pre-fix `main` | Delta vs previous |
| --- | ---: | ---: | ---: | ---: |
| `main` `d61a6d73f5c38e92e433beb9e404d06d79b153b1`, uncapped 24 GiB calculation | 8,641 | 14.24 GiB | baseline | N/A |
| working-tree guard, 16/24 GiB target ceiling | 3,521 | 5.80 GiB | -59.3% experts; -8.44 GiB | -59.3% experts; -8.44 GiB |

The cap is expressed as eleven complete 320-expert routing cycles plus the
in-flight slot. Byte admission still uses the artifact's exact per-expert size.
Hosts above 24 GiB retain the continuous byte planner.

## 2026-07-27 allocation-time hardening

The phase-entry gate alone could not prove that memory was still safe at the
later instant when a lazy slab became real Metal storage. The hardened path
therefore separates the requested cache target from a monotonic effective
growth cap:

1. AUTO cannot enter resident mode through 24 GiB, even if a transient
   point-in-time snapshot appears optimistic.
2. Every new slab is charged as exact bytes against live reclaimable memory,
   affirmative normal pressure, the complete runtime/static/cache envelope,
   and `recommendedMaxWorkingSetSize`.
3. On denial, already allocated slabs remain valid, the effective cache cap is
   frozen at their slot capacity, and misses evict/reuse slots. Combined,
   per-component, and mmap-backed fresh-buffer fallbacks are not available.
4. A denial before the minimum 321-expert route slab exists fails before expert
   I/O.

Hebrus Studio now reapplies its hardware token profile at every managed Qwen
launch and command preview, so an older version-2 configuration requesting 32K
on a 21/24 GiB Mac is launched at 16K context and 8K maximum output without
silently rewriting the stored values. Managed ExpertMajor launches also remove
all inherited or configured `DS4_*` tuning variables except the documented
memory-report and Qwen-telemetry diagnostics.

The synthetic 21 GiB fixture is deliberately not a process memory limit. It
feeds the planner an explicit physical-memory, Metal-budget, reclaimable-page,
and pressure snapshot. A separate real-Metal fault injection denies slab growth
at the allocation boundary. Together they make the safety mechanics
deterministic in CI; neither can reproduce macOS unified-memory contention or
qualify physical 24 GiB throughput.

The artifact manifest's `runtimeCommit=73a332f...` remains the physical-format
compatibility floor and predates this safety policy. The hardened source is a
publication candidate, not a released policy. Before release, land it, advance
Hebrus Studio's minimum managed runtime pin to that commit (or a reviewed
descendant), and rerun the gates below. Until then the Studio release runtime
is not qualified for this 24 GiB policy even though it can decode the artifact
format.

## Model-free evidence

Rows are chronological. These tests do not replace the physical model-backed
gate.

| Timestamp | Revision / experiment | Command / lane | Result |
| --- | --- | --- | --- |
| 2026-07-24T11:13:39+02:00 | `main` `d61a6d73f5c38e92e433beb9e404d06d79b153b1` | `make build/metal-arm64/bin/test_ssd_residency && build/metal-arm64/bin/test_ssd_residency` | PASS — pre-change baseline |
| 2026-07-24T11:15:27+02:00 | working-tree 24 GiB guard | same SSD residency-policy lane | PASS — 24 GiB ceiling, normal/elevated/unavailable pressure, and 16/24 boundary assertions |
| 2026-07-24T11:26:10+02:00 | working-tree 24 GiB guard | `make build/metal-arm64/bin/test_ssd_residency build/metal-arm64/bin/test_qwen_session && build/metal-arm64/bin/test_ssd_residency && build/metal-arm64/bin/test_qwen_session` | PASS — SSD policy and Qwen session regression |
| 2026-07-24T11:26:58+02:00 | working-tree 24 GiB guard | `build/metal-arm64/bin/ds4_test --server` | PASS — server contract lane |
| 2026-07-24T12:37:48+02:00 | working-tree guard plus bounded 24 GiB runner | shell syntax, runner fixture, generated brand-boundary check and regression | PASS — one-shot artifact binding, 24/64 GiB profiles, preload omission, watchdog aborts, telemetry and competitor guards |
| 2026-07-24T12:39:08+02:00 | pre-review working-tree guard | `make premerge` with Apple M5 Pro Metal access | PASS — repository, documentation, brand, build-isolation, model-free Metal and install gates before the final review corrections |
| 2026-07-24T12:56:23+02:00 | runner corrections, before the persistent-phase pressure fix | `make premerge` with Apple M5 Pro Metal access | FAIL — all core and Metal lanes passed; install alias parity straddled a wall-clock second and compared unequal timestamp prefixes |
| 2026-07-24T13:02:05+02:00 | guarded unchanged-budget phase check | server compile; `test_ssd_residency`; `test_qwen_session`; runner fixture; `git diff --check` | PASS — 16 GiB guarded changed/unchanged pressure decisions, 32 GiB unguarded isolation, Qwen session contract, and runner fail-closed cases |
| 2026-07-24T13:03:46+02:00 | same phase check plus alias timestamp normalization | command-alias profile lane and complete install test | PASS — canonical and compatibility aliases compare semantic diagnostics without a wall-clock race |
| 2026-07-24T13:09:29+02:00 | final reviewed working-tree candidate | `make premerge` with Apple M5 Pro Metal access | PASS — complete repository, documentation, brand, build-isolation, model-free Metal, runner, alias, and install gates |
| 2026-07-27T09:43:29+02:00 | `main` `595301760cda5ebc368636876a992ff68fc6d95e` plus dirty allocation-time candidate | build `test_ssd_residency` and `ds4_test`; `make qwen-24g-fixture-test`; run SSD resolver; `git diff --check` | PASS — synthetic 21 GiB equality, pressure, host, Metal and overflow cases; fixture hashes/order/runner syntax; clean diff |
| 2026-07-27T09:43:44+02:00 | same local candidate | `build/metal-arm64/bin/ds4_test --metal-kernels` on Apple M5 Pro | PASS — real Metal admitted one test slab, denied the next and reused slots without a new buffer; denial before the first slab remained at cap zero across a second attempt with no SSD I/O or fallback |
| 2026-07-27T09:50:34+02:00 | same local candidate | `make premerge` with Apple M5 Pro Metal access | PASS — complete repository, documentation, contracts, brand, fixture, build-isolation, model-free Metal, benchmark guard, alias and install gates |
| 2026-07-27T11:06:47+02:00 | immutable candidate `51a0e0e3df18c1b6fd9640f81546137b91c75dae`; clean detached worktree | `make premerge` with Apple M5 Pro Metal access | PASS — exact-commit repository, documentation, contracts, brand, fixture, build-isolation, model-free Metal, benchmark guard, alias and install gates |

## Immutable draft-publication smokes

These short arms verify Qwen and the shared DeepSeek SSD path on the exact
allocation-time hardening commit intended for a draft pull request. They are
correctness and isolation smokes, not a promotion cohort: there is no matching
clean-`main` baseline, no medium/large/long context matrix, and no physical
24 GiB host in this cohort.

Shared identity:

| Field | Value |
| --- | --- |
| Host | Apple M5 Pro, 64 GiB unified memory, macOS build `25F84`, AC power |
| Source | clean commit `51a0e0e3df18c1b6fd9640f81546137b91c75dae`; empty tracked diff and zero untracked files |
| Source-state SHA-256 | `b6085995b88d05c1d0605c876e621ab1061476ba7de1e4cea0816f1e1adb62e5` |
| Executable SHA-256 | `79dc56edd0ba2c541e7cda816b55776993f3d6a33c4e341d2066e38842668afb` |
| Metal file-set SHA-256 | `5274486a015b61eb0e385b5ef793565267dc3113338e1bac7330e7b27e490b60` |
| Prompt | `tests/long_context_security_prompt.txt`, SHA-256 `e7c1a2cadf781d274cc26bd251d532fe1b9e632080da97e3eb4684741e7cc308` |
| Runner policy | warm cache state; no new swapout allowed; no competing inference process allowed |

Artifact and launch identity:

| Model | Artifact | Residency / allocation |
| --- | --- | --- |
| Qwen3.6-35B-A3B | 20,808,566,880 bytes; SHA-256 `dd17266185833a9f05531ce366fd7284ddca1ed64aa3dcf06e321e8c72c9ea3d` | explicit SSD, preload omitted, 8,192-token context allocation |
| DeepSeek V4 Flash | 86,720,114,272 bytes; SHA-256 `8378080263eb9224f7228d72e2afa4ac3cf74a116023fdec2c596ff228a33e3f` | explicit SSD, 4,096-expert preload, adaptive cache 4,387 experts / 28.92 GiB, 32,768-token context allocation |

The most recent comparable references used for the previous-experiment deltas
were the 2026-07-27T10:44:36+02:00 Qwen arm and the
2026-07-27T10:45:09+02:00 DeepSeek arm. Both used `main`
`595301760cda5ebc368636876a992ff68fc6d95e` plus tracked-diff SHA-256
`7c34359fee3ba83cd2bb406416bbf41506dac77b9023ef813dd354f1d077115d`,
source-state SHA-256
`4063adfc59e6fff91929df084738d3d85d1ef5b724701035e1b57dbcb9a0ee54`,
and executable SHA-256
`84c561d5415b33f0a75b05702878dc08f87a041bf65b0e42d82134e1fdd5369f`
under otherwise matching model, prompt, residency, cache-state and frontier
conditions.

| Timestamp | Revision / model | Prompt frontier | Prefill | Decode | TPOT p50 / p95 | Delta vs `main` | Delta vs previous experiment | Result |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 2026-07-27T11:10:14+02:00 | `51a0e0e3df18`; Qwen explicit SSD | 128 + 128 generated | 66.72 t/s | 28.21 t/s | 32.326 / 54.812 ms | N/A — no matching clean-`main` arm | prefill -1.62%; decode +1.84%; p50 -0.73%; p95 +0.75% vs 2026-07-27T10:44:36+02:00 | PASS — `rc=0`, zero swapout, pressure 90%→minimum 70%→90%, peak RSS 9,850 MiB, peak wired 16,095 MiB, no competitor, 32,426 valid telemetry records |
| 2026-07-27T11:11:20+02:00 | `51a0e0e3df18`; DeepSeek explicit SSD | 128 + 128 generated | 19.30 t/s | 11.51 t/s | 65.923 / 89.296 ms | N/A — no matching clean-`main` arm | prefill +4.66%; decode -0.60%; p50 -1.25%; p95 +0.53% vs 2026-07-27T10:45:09+02:00 | PASS — `rc=0`, zero swapout, pressure 90%→minimum 24%→84%, peak RSS 29,689 MiB, peak wired 42,561 MiB, no competitor |

| Model | Resolved-plan SHA-256 | Telemetry SHA-256 | Frontier-logits content SHA-256 | Decode-evidence content SHA-256 |
| --- | --- | --- | --- | --- |
| Qwen | `c97262c14544ccff3ea48989fb47fedb5e8cf7cf9d2a068b2dcacf0e7237093a` | `eda66c5cdfbcccede330a8599c6ab4a0c2c46ebc7e2cebda8764125d6f9bc4f9` | `b6821f4cd32c1d6c2a00ce5048683ec81d2bd86f5ba8856e5a84610e95ab520d` | `c26eec8b5a52283383e509dadc282f5f891fb8f05bd1af1008d0b3db91e69061` |
| DeepSeek | `af3c10b93269b2fe34ec80147c43106917f65df7cdc3bb5a5ae9cf1f5b115caf` | N/A — DeepSeek does not emit Qwen telemetry | `e5528272f265a8d8e5b7462ff16e8500f9ee48dc56a7ca039296a4c251b8747f` | `a6e232d06f43e280b8dba70b1053e74136839cd14ee0c254be1e8024c2969c2b` |

Both models reproduced the same frontier-logits and decode-evidence content
hashes as their immediately preceding comparable arms. The short Qwen arm ran
on a 64 GiB host and therefore does not exercise or qualify the guarded
24 GiB allocation policy; that remains covered mechanically by the real-Metal
fault test and pending on the required physical sustained gate.

## Local M5 Pro model-backed context checks

These are real 16K- and 32K-context checks on this 64 GiB Mac. They are not
physical 16/32 GiB memory-tier evidence. The deterministic planner test above
covers those RAM profiles; only matching hardware can qualify their pressure
and throughput behavior.

Shared identity:

| Field | Value |
| --- | --- |
| Host | Apple M5 Pro, 64 GiB unified memory, macOS build `25F84`, AC power |
| Source | `d61a6d73f5c38e92e433beb9e404d06d79b153b1` plus dirty source-state SHA-256 `f160ff7f55319135d70821ade979e60aa9863a52745ab0054508a57fc559aba7` |
| Executable SHA-256 | `9b4c05b6938ac16933599803548640782daef9d4fa423be70c3b9af9d120a4a2` |
| Metal source identity | `1e26e4135c620326fb1d8d6bc23aafd43183b78353734cb8c5d76ddfc7781719` |
| Model | published affine4/group-64 artifact, 20,808,566,880 bytes, SHA-256 `dd17266185833a9f05531ce366fd7284ddca1ed64aa3dcf06e321e8c72c9ea3d` |
| Model-hash evidence SHA-256 | `05ec92f429c1bfaac1c9b476b0729016363e54b3666defad7fb1dd0c3d73abc0` |
| Prompt SHA-256 | `e7c1a2cadf781d274cc26bd251d532fe1b9e632080da97e3eb4684741e7cc308` |
| Residency | AUTO resolved to resident in both arms |

The rows are separate safety checks, not a comparable performance cohort.
There is no matching clean-`main` or previous-context baseline, so both deltas
are `N/A`.

| Timestamp | Revision / experiment | Prompt frontier | Prefill | Decode | TPOT p50 / p95 | Delta vs `main` | Delta vs previous | Result |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 2026-07-24T13:05:05+02:00 | local candidate, 16K context / 32K allocation | 16,384 + 128 generated | 1,153.76 t/s | 45.39 t/s | 21.798 / 23.012 ms | N/A | N/A | PASS — `rc=0`, zero swapout, pressure 91%→minimum 51%→86%, no competitor, 1,515 valid telemetry records |
| 2026-07-24T13:06:00+02:00 | local candidate, 32K context / 65K allocation | 32,768 + 128 generated | 732.46 t/s | 36.41 t/s | 27.253 / 28.293 ms | N/A | N/A — different frontier | PASS — `rc=0`, zero swapout, pressure 87%→minimum 48%→86%, no competitor, 1,617 valid telemetry records |

| Frontier | Telemetry SHA-256 | Frontier-logits content SHA-256 | Decode-evidence content SHA-256 |
| ---: | --- | --- | --- |
| 16,384 | `c45cd5820ccebec346d90c1621667bad87a7a44a66a791d06465c67d843615c8` | `6a9c5fd6a1bebd13433419521ea123f16675281a0de72b4498d274f4f02b5b4b` | `31f5e4684bfd291a387e58a0c0b1905feba19cee35341101761e1f5c4349104b` |
| 32,768 | `cae25437403861904ef4f51cb689aec8bb0fcc57fc0451fbb1a68bae6ddcd05e` | `b18e77059f603acfd11f7cad5c8bcd5a4c0da60257850caa7604b631d365256b` | `5712bc829d550b9010cc5301e2eb6e89adb7039c5fe14cf1ae7668f9b18e71a9` |

## DeepSeek isolation check

The SSD implementation is shared, so the candidate also received a real
DeepSeek regression arm even though the planner and pressure-policy change is
Qwen-only. This is a short correctness and isolation check, not a promotion
performance cohort. There is no matching clean-`main` or previous-experiment
baseline, so both deltas are `N/A`.

| Field | Value |
| --- | --- |
| Host | Apple M5 Pro, 64 GiB unified memory, macOS build `25F84`, AC power |
| Source | `d61a6d73f5c38e92e433beb9e404d06d79b153b1` plus dirty source-state SHA-256 `b994df096d827231856d8903477a7ca783077246db83c7e83d0a254bb4ec0dd7` |
| Executable SHA-256 | `c69fd88c38a6d139c00dda02479025ab97ade395166d278c99cdb541ccc84151` |
| Metal source identity | `1e26e4135c620326fb1d8d6bc23aafd43183b78353734cb8c5d76ddfc7781719` |
| Model | qualified DeepSeek V4 Flash ExpertMajor v2 artifact, 86,720,114,272 bytes, SHA-256 `8378080263eb9224f7228d72e2afa4ac3cf74a116023fdec2c596ff228a33e3f` |
| Model-hash evidence SHA-256 | `23fefba71783e281d5166d2f0507dd10d27523c9de7406cc3a98857b43d8faaa` |
| Prompt SHA-256 | `e7c1a2cadf781d274cc26bd251d532fe1b9e632080da97e3eb4684741e7cc308` |
| Residency | explicit SSD; adaptive cache resolved to 4,387 experts (28.92 GiB) |

| Timestamp | Revision / experiment | Prompt frontier | Prefill | Decode | TPOT p50 / p95 | Delta vs `main` | Delta vs previous | Result |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 2026-07-24T13:16:27+02:00 | local candidate, DeepSeek SSD isolation | 128 + 128 generated | 18.69 t/s | 11.43 t/s | 66.057 / 85.771 ms | N/A | N/A | PASS — `rc=0`, zero swapout, pressure 87%→minimum 19%→79%, no competitor |

The 64 GiB runner treats 20% free memory as an admission threshold and records,
but does not abort on, a transient lower live sample; the arm produced no
swapout and recovered to 79%. Frontier-logits content SHA-256 is
`e5528272f265a8d8e5b7462ff16e8500f9ee48dc56a7ca039296a4c251b8747f`;
decode-evidence content SHA-256 is
`a6e232d06f43e280b8dba70b1053e74136839cd14ee0c254be1e8024c2969c2b`.

## Hebrus Studio integration evidence

Hebrus Studio remains a separate worktree. Its regression uses the 24 GiB
defaults, selects the canonical `hebrus-server` binary even when the persistent
compatibility alias is present, and verifies that resident/SSD/cache overrides
cannot escape canonical Qwen AUTO. The 2026-07-27 extension also launches an
oversized persisted Qwen configuration through a synthetic 21 GiB profile,
asserts effective 16K/8K limits without rewriting it, and strips active DS4
tuning variables while preserving documented diagnostics. It does not simulate
macOS memory pressure or replace the physical gate.

| Timestamp | Revision / experiment | Command / lane | Result |
| --- | --- | --- | --- |
| 2026-07-24T11:33:52+02:00 | Studio `5720685d415c` plus pre-existing user worktree and Qwen regression | `npm test` | PARTIAL — 500/502 tests passed; two unrelated web-search middleware tests in the pre-existing dirty worktree expected a different control-header spelling |
| 2026-07-24T11:38:25+02:00 | same Studio state | `npm test -- tests/runtime.test.ts --configLoader runner --no-cache`; `npm run typecheck`; `npm run check:brand` | PASS — 70/70 targeted runtime tests, both TypeScript projects, and brand boundary |
| 2026-07-27T09:44:22+02:00 | Studio `main` `c6c48825de219ee0f980adc4789cec349a6fa4c0` plus dirty Qwen integration candidate and unrelated user worktree | `npm test -- tests/runtime.test.ts tests/config.test.ts` | PASS — 80/80; synthetic 21 GiB launch clamps persisted 32K/32K to 16K/8K without rewriting storage and strips non-allowlisted tuning environment |
| 2026-07-27T09:45:33+02:00 | same Studio state | `npm run check:brand`; `git diff --check` | PASS — exact compatibility groups classified; clean diff |
| 2026-07-27T09:45:39+02:00 | same Studio state | `npm run typecheck` | FAIL outside this candidate — unrelated untracked artifact/research qualification files require type exports and runtime-build helpers absent from the shared worktree |
| 2026-07-27T09:45:47+02:00 | same Studio state | strict targeted `tsc --noEmit` over `hardware-profile`, engine arguments, config, and runtime | PASS — no TypeScript error in the modified launch-policy modules |

## Required physical confirmation

On the reported M5 24 GiB host, verify the published affine artifact by byte
size and SHA-256, then use the same Hebrus Studio persistent streaming endpoint
and the five requests, in order, from
`tests/qwen/fixtures/qwen-24g-release-v1.json`. The original incident prompt,
context allocation, and seed were not captured. The checked-in Sarajevo prompt
is therefore a qualitative reconstruction, not a byte-identical replay; the
separate deterministic companions provide the quantitative sustained-decode
gate.

| Prompt | SHA-256 | Purpose |
| --- | --- | --- |
| `qwen-24g-sarajevo-v1.txt` | `39f8f9bcfcc5f99b5fcc6a6d7ca303322db2d7a02ddb702e6485e570f6b63ba6` | Natural medium/high completion, ordinary EOS required |
| `qwen-24g-sustained-v1.txt` | `6942d59a685679dc5e404bdc4b51cc2ae97f3b3a79b5aabf0effe44a358d30b2` | Medium/high length stop after at least 1,720 generated tokens |
| `qwen-24g-followup-v1.txt` | `833c43d43d72d6e66ec778a66ccb22dfcfe8464be2501fbd8dbd0059ba71024e` | Same-process post-guard liveness |

With Hebrus Studio already running on the physical host, execute the sequence
with the checked-in loopback-only runner:

```sh
HEBRUS_API_KEY=... python3 tests/qwen/run_24g_release_gate.py \
  --output-dir /absolute/private/path/qwen-24g-release-evidence
```

The output directory must not already exist. It contains raw model responses;
treat it as potentially sensitive evidence and do not commit it.

Require AUTO to resolve to SSD and a separate explicit-resident launch to fail.
Record the resolved AUTO plan, requested decode target (never above 3,521),
every proposed/admitted or denied slab, the monotonic effective cap, allocated
slab capacity, pressure at admission, every phase entry and every slab decision,
`buffer_allocs`, task physical footprint, system swap, prompt and generation
counts, prefill/decode throughput, and TPOT p50/p95. Run with
`DS4_METAL_MEMORY_REPORT=1` so successful allocation-time decisions are
observable. Confirm Studio launches an oversized persisted configuration at
16,384 context / 8,192 output and removes every non-allowlisted `DS4_*`
variable. Any pressure `WARNING`, new swapout, watchdog `SIGTERM`, stream error,
changed resolved plan, fresh-buffer cap bypass, or competing inference process
invalidates the cohort.

Until those rows exist, this record does not claim that the original physical
failure is closed or that performance on 24 GiB has been measured.
