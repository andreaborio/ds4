# ADR 0004: Qwen Metal Uses Hardware-Aware Memory Profiles

- Status: Accepted
- Date: 2026-07-21

## Context

Qwen3.6-35B-A3B ExpertMajor v2 can execute either with the complete tensor
payload mapped for Metal or with routed experts streamed from SSD. The previous
AUTO policy combined a generic 20% Metal reserve with a Qwen SSD exception that
applied only at 16 GiB. After a GGUF conversion warmed the macOS file cache,
that discontinuity could select fewer cached experts on a 32 GiB Mac than on a
16 GiB Mac. It also rejected safe short-context resident plans on 32 GiB even
when the device working-set limit and live pressure both had room.

Apple ships a finite set of common unified-memory cuts, but
`recommendedMaxWorkingSetSize` also varies by chip and GPU configuration. A
safe policy therefore cannot infer capacity from RAM labels alone or hardcode
one cache count for every Metal device.

## Decision

Qwen Metal exposes named 16, 24, 32, 36, 48, 64, 96, and 128 GiB memory
profiles for observability and tests. Safety arithmetic remains continuous in
the physical-memory and Metal working-set values reported by the active host.
Values between named cuts use the next containing profile without changing the
byte calculation.

Qwen's fixed resident reserve is:

```text
resident reserve = max(2 GiB, physical RAM / 16)
                 + max(0.25 GiB, physical RAM / 64)

resident required = model tensors + context/KV/scratch + resident reserve
```

AUTO selects resident only when both independent gates pass:

1. `resident required` fits `recommendedMaxWorkingSetSize` after any explicit
   simulated external reserve;
2. the same model, runtime, and reserve fit the point-in-time reclaimable-memory
   plan.

An explicit resident request also requires both gates. Failure selects bounded
SSD streaming for AUTO and rejects an explicit resident request. The current
19.37 GiB routed tensor payload alone exceeds 16 GiB, so that profile
necessarily uses SSD. A 24 GiB device is decided from its actual Metal
working-set report rather than an assumed ratio. The 32 GiB profile may use
resident for shorter contexts when current pressure allows it and falls back
to SSD otherwise.

For Qwen SSD streaming, pageable static weights share the larger ordinary
headroom envelope but remain fully charged. While macOS reports normal pressure,
the bounded file-backed page pool receives full reclaimable credit on every
Qwen profile. This makes equivalent cold and warm GGUF page states produce the
same cache plan. Elevated or unavailable pressure retains half credit; the
16 GiB profile additionally requires an affirmative normal-pressure signal.
The result is capped independently by live reclaimable memory and Metal's
recommended working set, then rounded down to complete `1 + 320*k` routing
cycles. The 16 GiB profile has an additional measured ceiling of eleven cycles
plus the in-flight slot: 3,521 experts for this 40-layer, top-8 model. This keeps
the previously retained zero-swap tier and rejects the next warm-cache tier,
which produced swap in validation. Because the cap is expressed in routing
cycles while byte admission uses the artifact's exact per-expert size, it does
not treat compact MLX affine4/group-64 storage as F32.

## Consequences

- AUTO adapts to RAM, context length, Metal's device-specific working-set
  recommendation, and current memory pressure instead of selecting by RAM
  label alone.
- A larger nominal memory profile cannot receive a smaller cache solely because
  its GGUF pages are warm rather than cold.
- Resident is a complete mapped-tensor execution mode, not proof that every
  mmap page remains physically resident after later system pressure changes.
- The named profile matrix is policy-tested on every cut. Performance claims
  remain tied to the exact physical hosts recorded in benchmark evidence.
- Physical 16/32/64 GiB validation and the retained split-K controls are
  recorded in
  [`2026-07-21-qwen-split-k-hardware-policy.md`](../benchmarks/2026-07-21-qwen-split-k-hardware-policy.md).
- DeepSeek and GLM retain their independent memory policies.
- Changing the reserve formula, normal-pressure credit, or routing-cycle
  rounding requires the context/cache acceptance matrix in
  `QA_BEFORE_RELEASES.md`.
