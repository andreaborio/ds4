# Benchmark Evidence Index

[`CONTRIBUTING.md`](../../CONTRIBUTING.md#performance-acceptance-matrix) is the
canonical performance-change gate. This directory stores dated evidence and
decisions; historical records do not become current qualification merely
because their numbers remain useful.

| Record | Status | Scope |
| --- | --- | --- |
| [`2026-07-28-qwen-q2-k-xl-performance-weight.md`](2026-07-28-qwen-q2-k-xl-performance-weight.md) | Accepted dual-profile implementation; 262K endpoint and Q2_K_XL publication pending | Q2_K_XL quality/performance-per-weight decision, shared Qwen runtime, Affine4 A/B/B/A, evidence through 100K/32K, rejected Affine2 and speculative paths |
| [`2026-07-24-qwen-24g-cache-guard.md`](2026-07-24-qwen-24g-cache-guard.md) | Local safety candidate implemented; physical M5 24 GiB confirmation pending | Qwen 24 GiB sustained-decode watchdog incident, guarded cache policy, model-free regression, and versioned five-request Hebrus Studio physical gate |
| [`2026-07-22-qwen-kv-pair-blit.md`](2026-07-22-qwen-kv-pair-blit.md) | Owner-authorized merge; full release matrix incomplete, SSD speed effect within noise | Qwen K/V cache pair-blit on M5 resident/SSD and M1 Pro 16/32 GiB, including invalid 16 GiB timing evidence |
| [`2026-07-22-qwen-ssd-flash-prefill.md`](2026-07-22-qwen-ssd-flash-prefill.md) | Directional candidate; battery cohort, promotion pending | Qwen implicit-causal FlashAttention transfer to SSD, memory accounting, and first 2K/8K A/B |
| [`2026-07-21-qwen-pre-m5-exact-router-tile.md`](2026-07-21-qwen-pre-m5-exact-router-tile.md) | Directional; timing cohort contaminated | Exact F32 router weight reuse on M1 Pro through 8K, rejected approximate predecessor, and pending clean promotion lane |
| [`2026-07-21-qwen-split-k-hardware-policy.md`](2026-07-21-qwen-split-k-hardware-policy.md) | Current Qwen qualification evidence | F32 split-K A/B at 8K/32K on 64 GiB, transfer validation on 16 GiB, and hardware-aware AUTO validation on 16/32/64 GiB |
| [`2026-07-21-upstream-metal-transfer-audit.md`](2026-07-21-upstream-metal-transfer-audit.md) | Research/rejected | Post-release audit of upstream PR #555/`427e281`; isolated M5 dense NAX GLM tests through 32K and removal decision |
| [`2026-07-20-long-context-metal-stack.md`](2026-07-20-long-context-metal-stack.md) | Current release qualification evidence | Qwen 32K A/B/B/A, DeepSeek adaptive dual-prompt 32K plus 65K/100K safety gates, GLM compact-indexer fix through dual-prompt 32K, invalidations and remaining publication gates |
| [`2026-07-20-agent-friendly-refactor-validation.md`](2026-07-20-agent-friendly-refactor-validation.md) | Pre-policy correctness/artifact baseline | Qualified v2 artifacts and post-refactor Qwen, DeepSeek, and GLM smoke/gold lanes; not a current performance-acceptance matrix |
| [`2026-07-20-glm52-expert-major-v2.md`](2026-07-20-glm52-expert-major-v2.md) | Pre-policy GLM research baseline | GLM v2 SSD prefill/decode decisions and rejected arms; superseded for current long-context acceptance |
| [`2026-07-20-qwen-expert-major-v2.md`](2026-07-20-qwen-expert-major-v2.md) | Current Qwen artifact baseline | Qwen v2 identity and 2K resident comparison |
| [`2026-07-17-deepseek-native-expert-major.md`](2026-07-17-deepseek-native-expert-major.md) | Historical and research | DeepSeek native-layout qualification and SSD experiments |
| [`2026-07-17-deepseek-qwen-transfer-audit.md`](2026-07-17-deepseek-qwen-transfer-audit.md) | Research/rejected | Cross-model optimization experiments and rejection evidence |
| [`2026-07-17-qwen-native-expert-major.md`](2026-07-17-qwen-native-expert-major.md) | Historical | Retired Qwen ExpertMajor v1 comparison |
| [`2026-07-15-qwen-ds4-vs-llamacpp.md`](2026-07-15-qwen-ds4-vs-llamacpp.md) | Historical | Short-context runtime comparison and follow-up experiments |
| [`2026-07-14-m5-pro.md`](2026-07-14-m5-pro.md) | Historical and research | Early fork/upstream and GLM feature comparisons |

The short 16/32-token and single-context lanes above remain valid for the
specific correctness or artifact decisions they recorded. They do not satisfy
the performance acceptance matrix introduced afterward.

Every new performance record must state `Status`, `Decision`, `Supersedes`,
affected model paths/modes, baseline and candidate identities, the completed
context tiers, cold or warm cohort state, discarded warm-ups, resolved plan,
control drift and within-arm spread, and where the raw CSV, decode evidence,
runtime Metal identity, and memory telemetry are retained. Mark the entire
cohort invalid when any arm aborts, swaps, resolves a different plan, or sees a
competing inference process; a later retry does not repair the partial cohort.
Record measured facts and the resulting code, not an unfinished plan.
