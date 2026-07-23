# TurboQuant KV Cross-Engine Handoff

## Identity

- **Objective:** evaluate and implement a native online quantized-KV path without
  importing the `turbovec` Rust/Python ANN implementation. The storage and
  admission contract must cover Qwen3.6, DeepSeek V4, and GLM 5.2 even though
  Qwen is the first implementation and qualification lane.
- **Repository:** `/private/tmp/ds4-turboquant-kv-cross-engine`
- **Branch:** `codex/turboquant-kv-cross-engine`
- **Starting HEAD:** `72d13941d0d4f2c021c0fa9c14183a7f3a487aa4`
- **Merge base:** `70d9164451da2fb3a8b2f352d0bbf5b7dbce17da`
- **Upstream boundary:** the branch starts eleven commits ahead of the merge
  base on the current DeepSeek SSD work line.

## Scope And Ownership

- Add one narrow, model-independent KV quantization contract and reference
  implementation. Keep tensor shapes, cache ownership, Metal dispatch, and
  snapshot compatibility in their existing model/runtime owners.
- Qwen: quantize only the ten full-attention-layer K/V caches. Recurrent state
  is a separate surface and is not covered by a K/V compression claim.
- DeepSeek: distinguish the raw sliding-window K cache from the existing
  model-native compressed attention cache. Never reinterpret compressed rows
  as ordinary K/V heads.
- GLM: distinguish compact `kv_lora`, `k_rope`, and indexer-key surfaces. GLM
  has no enabled expanded full-K/V cache and remains SSD-only.
- Do not add a sidecar, a canonical-weight fallback, an admission bypass, a
  public tensor/kernel API, or a Rust/Python runtime dependency.

## Compatibility Contract

- The production default and snapshot format remain unchanged until all
  affected model lanes qualify.
- A quantized cache must carry an explicit format/version identity. Restore
  must fail closed on a missing, unknown, or geometry-incompatible identity.
- Cache-byte planning must use checked arithmetic and must include packed
  payload, per-group metadata, alignment, staging, and scratch peaks. A smaller
  steady-state payload is not sufficient if conversion or long-prefill peaks
  increase admission pressure.
- Any surviving implementation becomes the singular default and loses its
  experiment switch. During the active hardware campaign every decode
  strategy remains selectable; losing code and scaffolding are removed only
  after the combined 16 GiB, 32 GiB, and M5 decision.

## Baselines And Evidence

- Qwen3.6 production artifact: `20,808,566,880` bytes,
  SHA-256 `d7c43f4e89da129306d15c06d38b6ee248ae303ee22849e76a9ef5b033b8d53b`.
- DeepSeek V4 production artifact: `86,720,114,272` bytes,
  SHA-256 `837808422914912915013e1d78132c23381d543a9142dd7f30fd66dfa450894e`.
- GLM 5.2 production artifact: `262,147,193,504` bytes,
  SHA-256 `7f5017d5a30b8d1d762f9fe1da53538f7f784389a85a2ed8e0b46d52489aebd3`.
- Qwen's current cache geometry is 40,964 bytes per token: ten
  full-attention layers, K and V, two KV heads, head dimension 256, F32,
  plus four bytes of sequence metadata. This is 1.250 GiB at 32K and
  approximately 10.001 GiB at the 262,144-token model limit.
- A 3.5-bit payload-only estimate is 4,484 bytes per token, or 0.137 GiB at
  32K. This is only a lower-bound planning estimate; metadata and peak scratch
  must be measured before claiming the resulting 1.113 GiB saving.

## Required Gates

- Model-free: deterministic reference vectors; packing boundaries; overflow and
  invalid-geometry failures; planner agreement for all three model families;
  `make premerge`; `git diff --check`.
- Qwen first lane: baseline versus F16, Q8, and the selected packed format on
  the canonical short/medium/large/32K matrix plus 65K and 100K KV frontiers.
- Because the contract is shared, promotion also requires qualified DeepSeek,
  GLM, and Qwen runs under their supported AUTO/resident/SSD modes. GLM
  resident requests must continue to fail closed.
- Record resolved plan, first-token latency, decode throughput, RSS, Metal
  allocation, swap delta, cache bytes, transient scratch peak, token/logit
  agreement, and invalidating conditions. Abort, swapout, a changed plan, or a
  competing inference process invalidates a comparison cohort.

## Stop Conditions

- Do not promote a production path on short-context evidence.
- Stop on unexplained token/logit/attention/routing drift, increased memory
  admission risk, snapshot ambiguity, or a regression at any long-context
  frontier.
- If a native Metal implementation is not fully qualified, leave production
  unchanged and keep only the documented/reference tranche on this branch.

## Current State

- The original checkout contains unrelated tracked and untracked user work. It
  has not been modified by this task.
- This isolated worktree was clean at creation.
- No DS4 inference process was running when the lane was opened.
- Other detached benchmark worktrees and their processes are out of scope.
- `ds4_kv_quant.[ch]` now owns the checked surface planner and scalar TQ4
  conformance reference. `tests/test_kv_quant.c` covers Qwen, DeepSeek, and GLM
  geometry plus deterministic key/value reference behavior.
- `tests/gen_kv_quant_centroids.py --check` owns the generated Lloyd-Max table.
- The standalone strict C99 reference build and `make kv-quant-test` pass.
- The Qwen research path now allocates packed graph buffers and uses native
  Metal store plus causal prefill/decode kernels without a global dequantized
  cache. `DS4_QWEN_TQ4_DECODE` preserves `auto`, `serial`, `parallel`,
  `split`, `reuse8`, and F16-staged `flash` paths for cross-SoC testing.
  Model-free graph, incremental-store, scalar/Metal packing, prefill, and all
  strategy tests pass; direct packed decode is within `3.49e-09` and F16
  staging within `5.16e-06` of the decoded-cache reference fixture.
- All repository model-free commands except the Expert Store probe pass. The
  starting tree's unchanged `tests/test_expert_store.c` does not compile
  because `DS4_EXPERT_STORE_GLM_AFFINE2_GROUP_SIZE` is absent. This remains an
  external gate blocker, not a waiver.
- On the current research SHA, `make premerge` stops earlier in
  `tools/context_audit.py`: the active handoff is intentionally present and
  the two research environment names raise `direct_ds4_env_names` from its
  budget of 334 to 336. This is not waived. After the 16 GiB/M5 policy
  decision, remove the strategy selectors and active handoff, restore the
  audit budget, and rerun the gate; the unchanged Expert Store failure remains
  the next known external blocker.
- Durable design is in `docs/architecture/KV_QUANTIZATION.md`; initial evidence
  is in `docs/benchmarks/2026-07-23-turboquant-kv-feasibility.md`.

## Remote Hardware State

- The 64 GiB M5 Pro development host is reserved for another thread's model
  lane. This task runs only model-free commands there.
- The 16 GiB M1 Pro is the LAN lane at `192.168.1.212`
  (`macbookpro.lan`). Remote Login is restored. The isolated source and result
  roots, relative to the remote home directory, are
  `BEEP/ds4-tq4-kv-16g-20260723/src` and
  `BEEP/ds4-tq4-kv-16g-20260723/results`.
- The 32 GiB lane is the Tailscale benchmark Mac
  `cescos-macbook-pro-1` (`100.99.235.116`). It is authenticated through the
  active SSH control connection and is an M1 Pro with 32 GiB, on AC.
- The 32 GiB 2K and 8K fresh-process cohorts are recorded in
  `docs/benchmarks/2026-07-23-turboquant-kv-feasibility.md`. TQ4 reduced task
  footprint by about 140 MiB at 2K and 550 MiB at 8K. Prefill is flat after
  Flash staging; on the single-binary selector confirmation, M1 Pro decode is
  within noise at 2K under `auto` and 2.87% below F32 at 8K. `split` is 5.17%
  below at 8K; `reuse8` loses 68.64% there and remains present for the M5
  comparison. All arms have zero swapout delta and byte-identical TQ4 logits
  within each context.
- At 32K the F32 resident control fails closed at admission: 25.23 GiB is
  required against the 24.96 GiB fixed working-set budget at capacity 33,024.
  TQ4 resident is admitted and all retained strategies complete at about
  2.84 GiB task footprint, 23% minimum pressure, and zero swap. Flash/auto is
  the best M1 strategy at about 17.1 decode tok/s. This proves a resident
  capacity gain, not an F32/TQ4 same-plan speed comparison.
- TQ4 resident `auto` also completed 65,536 and 100,000 real-token frontiers
  with full logits/decode artifacts and zero swap. Prefill/decode were
  201.79/12.13 and 161.81/9.43 tok/s respectively. Minimum pressure fell to
  19% and 16%, so these are functional exploratory/OOB gates only and cannot
  promote the candidate.
- The 16 GiB source commit is
  `264904f210b555f98f42080b7bb30a78b5f6e80e`, synchronized from local
  `61b75cc`; the binary SHA-256 is
  `8d0692012d4f0348ffa01b4cbddad2c36989bc63a3b6bc1760822220780ac504`.
  The host is an M1 Pro on AC running macOS 26.5 build `25F71`. Its Qwen
  artifact matches the 32 GiB cohort SHA-256 `dd172661...c9ea3d`.
- All retained Metal strategies and model-free gates available before model
  loading pass on 16 GiB. Fresh-process 2K and exact-cache 8K matrices
  completed with zero new swapouts. At exact-cache 8K, TQ4 saved 283.45 MiB
  of live runtime tensors; Flash decode was 2.44% below F32 while `split`,
  `parallel`, `reuse8`, and `serial` lost 6.86%, 23.02%, 52.52%, and 71.11%.
- The exact-cache 32K pair also completed with zero new swapouts. F32/TQ4
  Flash prefill was 29.18/231.77 tok/s and decode 10.67/10.16 tok/s. TQ4
  reduced live runtime tensors from 1,922.88 to 802.23 MiB, task footprint
  from 6.68 to 5.77 GiB, and raised minimum free pressure from 25% to 36%.
  The 32K argmax was stable, but greedy output diverged at token 9; this
  remains unqualified lossy research.
- Run only one model process per host. Qwen uses AUTO/SSD on 16 GiB and
  AUTO/resident on 32 GiB; DeepSeek/GLM follow the support contract, and GLM
  resident must continue to fail closed.

## Next Safe Action

Preserve the completed 16 GiB SSD and 32 GiB resident artifacts. Compare every
retained strategy on M5 when its isolated inference lane becomes available,
then make the execution-policy decision from the combined hardware evidence.
The shared cache contract still requires real DeepSeek and GLM implementations
and qualified model runs before any promotion. Production allocation,
snapshots, admission, and default dispatch remain unchanged until that matrix
succeeds; strategy removal happens only after the hardware decision.
