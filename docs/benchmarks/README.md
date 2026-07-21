# Benchmark Evidence Index

[`CONTRIBUTING.md`](../../CONTRIBUTING.md#performance-acceptance-matrix) is the
canonical performance-change gate. This directory stores dated evidence and
decisions; historical records do not become current qualification merely
because their numbers remain useful.

| Record | Status | Scope |
| --- | --- | --- |
| [`2026-07-20-long-context-metal-stack.md`](2026-07-20-long-context-metal-stack.md) | Current release-candidate evidence | Qwen 32K A/B/B/A, DeepSeek adaptive 8K/32K/65K/100K, GLM compact-indexer fix through 32K, invalidations and remaining gates |
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
