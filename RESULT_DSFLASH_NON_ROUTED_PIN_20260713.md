# DeepSeek Flash non-routed host pin: AC-confirmed opt-in result

Date: 2026-07-13

## Verdict

The startup-only `mlock()` treatment produced a strong, internally consistent
battery screening signal and then passed a fresh AC-powered ABBA with a longer
256-token decode. It is suitable only as an opt-in fork feature; the default
remains unchanged.

Important follow-up: after the subsequent validation attempt, the Mac rebooted
with a kernel panic at `2026-07-13 19:22:43 +0200`. The panic report
`/Library/Logs/DiagnosticReports/Retired/panic-full-2026-07-13-192243.0002.panic`
records `bug_type=210`, macOS `26.5.1 (25F80)`, and
`watchdog timeout: no checkins from watchdogd in 91 seconds`; the stackshot
included a `ds4_test` process. Compressor state was reported OK, so this does
not look like a simple swap/OOM failure from the visible panic header. Treat
this as a serious safety blocker for any default-on policy and investigate
before running larger stress/soak tests.

Validation note: `make model-free-test` does not exercise the large GGUF or the
Metal SSD-streaming model path. The SSD-streaming evidence in this report comes
from the explicit `ds4-bench` AC ABBA run. The full `ds4_test` path only uses
SSD streaming when `DS4_TEST_SSD_STREAMING=1` is set, and should not be repeated
with the large model until the watchdog panic is understood.

AC confirmation, order A1/B1/B2/A2:

| Leg | Pin | Prefill t/s | Decode t/s | Wall s | Page faults | Swapout delta |
|---|---:|---:|---:|---:|---:|---:|
| A1 | no | 18.39 | 11.86 | 29.34 | 7,999 | 0 |
| B1 | yes | 19.01 | 12.47 | 29.21 | 979 | 0 |
| B2 | yes | 19.84 | 12.39 | 29.17 | 458 | 0 |
| A2 | no | 19.23 | 8.21 | 38.58 | 5,606 | 0 |

- decode geometric mean: `9.8677 -> 12.4299 tok/s` (`+25.97%`);
- both paired comparisons are positive: `+5.14%` and `+50.91%`;
- prefill geometric mean: `18.8053 -> 19.4206 tok/s` (`+3.27%`);
- wall-time geometric mean: `33.64 -> 29.19 s` (`-13.24%`);
- the two treatment runs are stable (`12.47/12.39 tok/s`) while the pageable
  control degrades on A2;
- all four full prefill-logit JSON files are byte-identical, SHA-256
  `476a31661beb5beecd4d1b5e4bd48611bf59464836843a15f43a7c6aaa4eb0ae`;
- no new swapout occurred.

Earlier battery screening:

- decode geometric mean: `11.8697 -> 14.5224 tok/s` (`1.22349x`, `+22.35%`);
- pair A1/B1: `11.96 -> 14.80 tok/s` (`+23.75%`);
- pair A2/B2: `11.78 -> 14.25 tok/s` (`+20.97%`);
- prefill geometric ratio: `1.09326x` (`+9.33%`);
- system swapouts: no increase across the matrix;
- peak physical footprint reported by `/usr/bin/time -l`: effectively unchanged
  at about `32.85 GiB`;
- maximum RSS: about `28.72 GiB` in A and `36.94 GiB` in B, the expected
  `+8.22 GiB` file-backed page coverage;
- page faults: A=`5906/6514`, B=`2167/320`;
- the full prefill-logit JSON was byte-identical between A and B, SHA-256
  `476a31661beb5beecd4d1b5e4bd48611bf59464836843a15f43a7c6aaa4eb0ae`.

A separate 256-token sustained-decode pair, ordered B then A, also preserved
the signal: `14.57 vs 11.29 tok/s` (`1.2905x`, `+29.05%`). B recorded 32 page
faults versus 6,571 for A; wall time was `26.25 vs 29.75 s`. This pair is not
balanced by itself, but shows that the gain did not disappear after the first
64 tokens.

The monitor was not running during any benchmark leg.  The treatment executes
only during startup and adds no token-loop work.

## Configuration

- source base: `c58e91f48910f16be92ac1338bb39de36317722d`;
- screening binary SHA-256:
  `5ce3ba0a5096184c10005721ff301b4c4317905441e82cfd3b7e92f591299169`;
- model: `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`;
- cache: 4,342 expert entries, 4,096 preloaded;
- allocated context: 32,768;
- measured prompt/decode: 128/64 tokens;
- order: A1, B1, B2, A2;
- power: battery, 72% at the preregistration check;
- B pin startup after priming: `0.217/0.247 s`; isolated cold inspect:
  `1.323 s`.

Raw compact rows are in `bench-logs/static-pin-screen-20260713.tsv`.
The sustained pair is in `bench-logs/static-pin-long-20260713.tsv`.

## Interpretation and scope

The repeatable drop in page faults, sustained 256-token result, and return to
the slower A result after two B legs argue against simple run-order warming as
the whole explanation.
The likely mechanism is that wiring the always-used non-routed pages prevents
VM churn while the routed cache occupies roughly 28.6 GiB.

The AC ABBA satisfies the preregistered adoption gate for an opt-in feature:
both pairs are positive, the balanced decode gain is far above 3%, prefill does
not regress, logits are identical, and swapouts remain unchanged. The feature
does not alter weights or arithmetic; it changes only Darwin residency policy
at startup and rolls back completely if `mlock()` cannot cover the requested
pages.

This is not a reason to enable the treatment automatically for every model or
memory size. The measured model has an 8.20 GiB non-routed island on a 64 GiB
M5 Pro. Larger static sets, contexts, or expert caches need their own memory
gate. GLM 5.2 was tested separately and did not benefit from a full static pin.

Still useful after publication: root-cause the watchdog panic, add a bounded
stress harness that records wired pages and watchdog-adjacent system symptoms,
then run multi-prompt/server soak, resume-after-idle, and a CLI-facing memory
guard before considering any default-on policy.
