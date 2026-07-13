# DeepSeek Flash non-routed host pin screening (2026-07-13)

## Question

Does wiring the file-backed pages for token/output and every non-routed layer
tensor improve Metal SSD-streaming performance on the 64 GiB M5 Pro?

The residency monitor is not run during any benchmark leg.  Both arms use the
same binary; the only treatment is the startup-only environment flag
`DS4_METAL_STREAMING_PIN_NON_ROUTED=1`.
Unset the variable or set it to `0` for the control/default path.

## Screening configuration

- model: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`
- backend: Metal SSD streaming
- routed cache: exactly 4,342 expert entries, 4,096 preloaded
- allocated context: 32,768
- measured frontier: 128 prompt tokens plus 64 greedy decode tokens
- prompt: `tests/long_context_security_prompt.txt`
- order: `A1 B1 B2 A2`
- A: normal lazy file-backed non-routed pages
- B: the same pages faulted and wired once with `mlock()` before Metal/cache setup

This first matrix is explicitly a battery-power screening run.  It can reject
the idea, but cannot promote it without an AC-power confirmation.

## Metrics and stop rules

Primary metric: geometric mean decode ratio
`sqrt((B1_tps * B2_tps) / (A1_tps * A2_tps))`.

Secondary metrics: prefill t/s, startup wall time, maximum RSS, peak physical
footprint, page faults, page reclaims, and system swapout delta.

- stop immediately on mlock/allocation failure, process crash, new swapouts, or
  peak footprint above 48 GiB;
- reject if the balanced decode ratio is below 1.01;
- treat 1.01--1.03 or discordant pairs as inconclusive;
- candidate only if ratio is at least 1.03, both pairwise comparisons are
  positive, and prefill is at least 0.95x;
- regardless of screening result, require a fresh AC-power ABBA before adoption.
