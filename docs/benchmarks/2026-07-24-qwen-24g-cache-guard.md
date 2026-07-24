# Qwen 24 GiB sustained-decode cache guard — 2026-07-24

Status: safety fix implemented; physical M5 24 GiB confirmation pending.

Decision: treat Qwen SSD plans through 24 GiB as guarded. Require affirmative
normal memory pressure at admission and every prefill/decode phase entry,
including an unchanged configured budget whose lazy slabs can still populate,
and cap the routed-expert cache at 3,521 experts. This is a correctness and
safety change, not a throughput claim.

Supersedes only the uncapped 24 GiB SSD-cache portion of
`2026-07-21-qwen-split-k-hardware-policy.md`. The named hardware profiles,
resident admission formula, 16/32/64 GiB evidence, and split-K results remain
unchanged.

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
| working-tree guard, 16/24 GiB ceiling | 3,521 | 5.80 GiB | -59.3% experts; -8.44 GiB | -59.3% experts; -8.44 GiB |

The cap is expressed as eleven complete 320-expert routing cycles plus the
in-flight slot. Byte admission still uses the artifact's exact per-expert size.
Hosts above 24 GiB retain the continuous byte planner.

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
cannot escape canonical Qwen AUTO. It does not simulate macOS memory pressure
or replace the physical gate.

| Timestamp | Revision / experiment | Command / lane | Result |
| --- | --- | --- | --- |
| 2026-07-24T11:33:52+02:00 | Studio `5720685d415c` plus pre-existing user worktree and Qwen regression | `npm test` | PARTIAL — 500/502 tests passed; two unrelated web-search middleware tests in the pre-existing dirty worktree expected a different control-header spelling |
| 2026-07-24T11:38:25+02:00 | same Studio state | `npm test -- tests/runtime.test.ts --configLoader runner --no-cache`; `npm run typecheck`; `npm run check:brand` | PASS — 70/70 targeted runtime tests, both TypeScript projects, and brand boundary |

## Required physical confirmation

On the reported M5 24 GiB host, verify the published affine artifact by byte
size and SHA-256, then use the same Hebrus Studio persistent streaming endpoint.
The original context allocation and seed were not captured; use Studio's
configured 16,384-token candidate context and record the seed:

1. run the Sarajevo travel prompt with thinking `medium` through a normal final
   stream, treating an ordinary EOS as successful completion;
2. repeat with thinking `high`;
3. for each thinking setting, run a deterministic sustained-decode companion
   past 1,719 generated tokens;
4. complete another request in the same Hebrus Server process.

Record the resolved AUTO plan, 3,521-expert ceiling, pressure at admission and
every phase entry, `buffer_allocs`, task physical footprint, system swap,
prompt and generation counts, prefill/decode throughput, and TPOT p50/p95. Any
pressure `WARNING`, new swapout, watchdog `SIGTERM`, stream error, changed
resolved plan, or competing inference process invalidates the cohort.

Until those rows exist, this record does not claim that the original physical
failure is closed or that performance on 24 GiB has been measured.
