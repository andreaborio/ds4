# DeepSeek Flash RAM-tiered SSD-streaming cache

Date: 2026-07-13

## Verdict

The expert cache must not use one policy on every Mac.  The adopted policy is
conservative on low-RAM hosts and grows only from the memory that is actually
safe at startup:

| Physical RAM | AUTO expert cache | Static non-routed pin |
|---|---|---|
| `<= 16 GiB` | fixed at the measured safe floor, 259 entries for Flash, if it fits; otherwise startup fails | rejected |
| `17-63 GiB` | derived from live reclaimable memory, Metal's recommended working set, runtime reserve, and the pageable static set | rejected |
| `>= 64 GiB` | same adaptive calculation, with a stable hardware-scaled ceiling | explicit opt-in only, after a preflight budget check |

AUTO never enables the static pin.  A manual cache override remains available
for controlled benchmarking, but the correctness floor is always enforced.

## Why the 64 GiB planner needed another constraint

On the M5 Pro, the old calculation treated file-backed pages as reclaimable
without reserving the approximately `8.197 GiB` strict non-routed set.  It
selected 4,903 entries in one state and 5,161 under warmer page-cache state.
The 5,161-entry run did not complete: the bounded runner stopped it at the
memory-pressure guard, so it has no valid throughput result.

The corrected planner:

1. protects the strict page-aligned non-routed working set above the low-RAM
   tier;
2. subtracts modeled runtime and pressure headroom;
3. respects `recommendedMaxWorkingSetSize`;
4. caps AUTO at `9/16` of that recommended set so a warm mapping cannot make
   a later launch grow beyond the stable envelope;
5. still shrinks below that ceiling whenever current pressure requires it.

The `9/16` ceiling is just above the measured explicit reference: 4,342
entries use about `55.21%` of the M5 Metal recommended set.  It maps AUTO to
the nearest complete Flash working-set tier, 4,387 entries / `28.918 GiB`.

## M5 Pro 64 GiB result

All legs used Metal SSD streaming, the same 128-token prompt, 64 decode tokens,
4,096 preloaded experts, context allocation 32,768, AC power, and zero new
swapout.  Order was A1/B1/B2/A2.

| Leg | Cache | Prefill t/s | Decode t/s | Peak wired GiB |
|---|---:|---:|---:|---:|
| A1 | AUTO 4,387 | 21.63 | 13.05 | 42.718 |
| B1 | exact 4,342 | 22.22 | 13.74 | 41.068 |
| B2 | exact 4,342 | 22.13 | 13.78 | 41.062 |
| A2 | AUTO 4,387 | 22.21 | 13.59 | 41.358 |

- AUTO decode geometric mean: `13.3173 t/s`;
- exact-4,342 decode geometric mean: `13.7600 t/s` (`+3.32%`);
- exact-4,342 prefill advantage: `+1.17%`;
- all four frontier-logit outputs have SHA-256
  `476a31661beb5beecd4d1b5e4bd48611bf59464836843a15f43a7c6aaa4eb0ae`;
- no new swapout occurred.

The AUTO selection at 4,387 is therefore stable across these launch orders and
close to the better of the two tested arms.  The small remaining gap does not
justify hard-coding a model-specific count into the generic default; exact
4,342 remains the explicit performance reference for this GGUF.

The synthetic 24/32 GiB planner fixtures validate arithmetic and boundaries,
not throughput on those machines.  With an 18 GiB Metal recommendation, the
24 GiB fixture selects 1,291 entries.  A 32 GiB fixture selects 1,807 with a
24 GiB recommendation and 2,065 with a 32 GiB recommendation.  These tiers
must be measured on real hardware before being described as performance
optima.

## Pin and safety status

The pin parser now accepts only `0` or `1`, rejects inspect mode and hosts below
64 GiB, measures the exact static page coverage before touching it, and checks
both live-pressure and fixed-platform budgets before `mlock()`.

A bounded AUTO+pin canary selected 4,387 entries, covered `8.197 GiB` of static
pages, produced `11.64 t/s`, peaked at `45.351 GiB` wired, kept system-wide
memory free at `88%` before and after, produced no new swapout, and preserved
the reference logits hash.  This shows in one bounded canary that the preflight
accounting does not collapse AUTO after pinning; it is not a reason to enable
pin automatically.  Pin performance remains model-specific: it met the
preregistered performance gate on this Flash GGUF but did not help GLM 5.2.

A later accidental full-resident `make test` with this 80.76 GiB model was
associated with the documented watchdog kernel panic.  The command was not
using SSD streaming and is not evidence that the pin arm caused the panic, but
it makes unbounded validation unacceptable.  Pin remains default-off pending
bounded server soak and concurrent-pressure validation.

The bounded runner `speed-bench/run_m5_dsflash_arm.sh` refuses to start beside
another inference process, requires AC power, checks initial pressure and wired
memory before launch, kills the process on pressure/wired/swap/timeout limits,
rechecks final swapout counters, and rejects runs without frontier-logit JSON.
It must be used instead of a full-resident `make test` with this 80.76 GiB
model.

## Validation

- `make -j8 model-free-test`: pass;
- `make -j8 cpu`: pass, with the seven existing unused warnings;
- Metal build and targeted SSD-residency tests: pass;
- runner syntax and `git diff --check`: pass;
- bounded real M5 runs: zero new swapout and identical logits.

Do not port the `9/16` envelope or Flash tier geometry blindly to the GLM
branch.  GLM needs its own measured cache curve; only the general principles
(RAM tiers, live pressure, static-set accounting, and default-off pin) transfer.
