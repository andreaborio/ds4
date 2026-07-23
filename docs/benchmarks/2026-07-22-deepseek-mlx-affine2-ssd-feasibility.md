# DeepSeek MLX affine2 and SSD-streaming feasibility (2026-07-22)

Status: research / HOLD for affine2; planner safety correction implemented.
The 2026-07-23 AC follow-up below separately validates the explicit-cache
correctness fix and records rejected IQ2/Q2 performance candidates. The safe
AUTO12 default remains unchanged.

Decision: keep `codex/mlx-deepseek-ssd-study` as an experimental branch. Do
not merge the affine2 runtime into a supported DeepSeek path until the artifact
writer, residency contract, SSD planner, end-to-end correctness, quality, and
required performance matrices are complete.

The follow-up planner correction in this record is a fail-closed safety fix. It
does not change the support contract or constitute evidence that affine2 is
faster than the qualified IQ2/Q2 artifact.

Supersedes: none.

## Question and result

This audit asks whether MLX implementation ideas can improve DeepSeek V4 Flash
in DS4, including the 64 GiB SSD-streaming path. The answer is split:

- MLX's affine quantization semantics, expert-indexed quantized matrix
  multiplication, and route sorting are credible implementation ideas for a
  compute-bound MoE path.
- The current DeepSeek affine2 candidate does not establish an end-to-end
  benefit. It has no writable model artifact, rejects resident execution, and
  moves 18.52% more bytes per expert miss than the qualified IQ2/Q2 store. The
  audit also found a planner/backend mismatch that could request full-store
  prefill reads in addition to selected-record loads; the follow-up change on
  this branch now removes that mismatch with one mandatory selected-address
  decision shared by C mapping and Metal dispatch.
- No DeepSeek affine2 timing or quality result is reported here. The only
  attempted model A/B was stopped before inference by the repository's AC
  power guard. Treating that stopped run as performance evidence would violate
  the benchmark contract.

This is therefore a feasibility and rejection record for the current slice,
not a claim that affine2 can never win. A warm-cache or future resident kernel
could still recover more compute time than the added byte cost, but that must
be demonstrated on the final artifact.

## Source and host identities

| Item | Identity |
| --- | --- |
| Control source | `ec6322ed022be13f7ff67915701ffc86ebfcda50` |
| Candidate source | `70d9164451da2fb3a8b2f352d0bbf5b7dbce17da` |
| Candidate branch | `codex/mlx-deepseek-ssd-study` |
| Local MLX source inspected | `57c66cac7cb3e5b1eb350488a61f1506b40d39f8` |
| Local MLX-LM source inspected | `a790972f0f844d81067ed45c28b524220a10c019` |
| Host | Apple M5 Pro, 64 GiB unified memory, 18 logical CPUs |
| OS | macOS 26.5.2, build `25F84` |
| Available internal storage | about 14 GiB at the time of the audit |
| Power during attempted model A/B | battery, 39% and discharging |

The clean comparison worktrees produced these diagnostic build identities:

| Arm | `ds4-bench` SHA-256 | Metal source-set SHA-256 |
| --- | --- | --- |
| Control | `e7ba823e89a79567539ddda21687f90f306a5a3404b52f04eddb47cb0cdb1104` | `25bf26675ca668248a2a45081856765d4a5befcabf822d2dc299695f3e3eeeea` |
| Candidate | `e5183834f30e3b82536e012a4097fe322fcf0d336529cc808eb34078ccfa7b8e` | `8328fbf1ef540518db3269f3e1ab9218946e6889380e54e1a87f45a7b5a47a44` |

The currently qualified DeepSeek v2 control artifact remains:

| Artifact property | Value |
| --- | --- |
| Filename | `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-DS4-ExpertMajor-v2.gguf` |
| Bytes | `86,720,114,272` |
| SHA-256 | `8378080263eb9224f7228d72e2afa4ac3cf74a116023fdec2c596ff228a33e3f` |

The hash above is the published project identity; this audit did not spend an
additional full-file pass rehashing the 86.7 GB local copy.

## What transfers from MLX

MLX exposes lazy evaluation and unified memory, while MLX-LM's current MoE
layer uses `gather_qmm` for expert-indexed quantized multiplication and sorts
expert indices once the route set is large enough. Those are useful kernel and
scheduling references. They are not, by themselves, an eviction-controlled
expert SSD cache: DS4 still needs explicit slot budgeting, `pread` accounting,
mapping policy, and cold/warm qualification.

Primary references inspected:

- [MLX unified memory](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html)
  and [lazy evaluation](https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html);
- [MLX-LM switch layers](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/switch_layers.py),
  including expert route sorting and `gather_qmm`;
- [MLX array loader implementation](https://github.com/ml-explore/mlx/blob/main/mlx/backend/common/load.cpp);
- the assessed [DeepSeek V4 Flash affine2 donor repository](https://huggingface.co/mlx-community/DeepSeek-V4-Flash-2bit-DQ/tree/main)
  and its [quantization configuration](https://huggingface.co/mlx-community/DeepSeek-V4-Flash-2bit-DQ/blob/main/config.json).

The transfer hypothesis is supported only indirectly by the qualified Qwen
results already in this repository. On the same M5 Pro, Qwen's resident
MLX-affine NAX path improved routed prefill by 29.68% at 2K and 20.70% at 8K
against its affine SIMD control. The final Qwen affine runtime improved 35.02%
at 2K and 39.55% at 8K over the earlier Q4 path. Its 32K SSD macro-prefill
reduced expert reads by 72.24% with a 0.63% throughput cost. Those results show
that MLX-derived storage and kernel ideas can transfer, but they do not predict
DeepSeek affine2: Qwen has an equal-size affine repack and a qualified resident
TensorOps path, while this DeepSeek candidate has neither property.

## Exact storage geometry

The candidate's canonical per-expert record is gate g32 plus up/down g64:

| Quantity | Qualified IQ2/Q2 v2 | Candidate MLX affine2 | Delta |
| --- | ---: | ---: | ---: |
| Gate bytes per expert | - | 3,145,728 | - |
| Up bytes per expert | - | 2,621,440 | - |
| Down bytes per expert | - | 2,621,440 | - |
| Complete expert record | 7,077,888 (6.75 MiB) | 8,388,608 (8 MiB) | +18.5185% |
| One 256-expert layer | 1,811,939,328 (1.6875 GiB) | 2,147,483,648 (2 GiB) | +18.5185% |
| Routed payload, 43 layers | 77,913,391,104 (72.5625 GiB) | 92,341,796,864 (86 GiB) | +13.4375 GiB |

Replacing the routed payload in the qualified v2 container would produce an
estimated `101,148,520,032`-byte (94.2019 GiB) GGUF before any additional
format overhead. The larger record also fits 15.625% fewer cache slots in a
fixed byte budget. Affine2 must therefore save enough compute to cover both
the extra bytes per miss and any increase in reloads.

## Implementation audit

### Artifact production is incomplete

`plan_deepseek_mlx_affine2()` explicitly stops at a physical layout plan. The
CLI requires `--dry-run` and says that the payload writer is not implemented.
There is no 86 GiB writer, complete GGUF/manifest construction, atomic publish,
end-to-end digest verification, or artifact to score. The current provenance
checks pin the donor Git revision and validate shard headers/shapes, but do not
hash every hydrated shard against its Git LFS OID or prove that routed donor
and non-routed GGUF bytes derive from the same checkpoint.

### Residency contract is incomplete

The runtime admits the affine2 store and exact g32/g64 geometry, then explicitly
rejects it when SSD streaming is not active. Thus this slice cannot deliver a
resident MLX-derived benefit, and AUTO can select resident before installation
later fails. That conflicts with the supported DeepSeek AUTO/resident/SSD
contract unless either resident execution is implemented and qualified or the
new format is made explicitly SSD-only in the residency planner and contract.

### SSD planner and Metal dispatch divergence: corrected, not qualified

Before the follow-up correction, the C prefill predicate recognized
selected-address batching only for the
qualified IQ2/Q2 tensor types. Full-layer `pread` preparation is enabled by
default. When that predicate is false, `metal_graph_stream_layer_spans()` adds
the complete physical ExpertMajor layer. The Metal backend independently
forced every affine2 batch onto selected-address compute.

The resulting plan was full-layer preparation plus selected-record compute.
Under the default 4,096-token Flash chunk cap, the full-layer preparation alone
would request the following routed bytes:

| Prompt frontier | Prefill chunks | Full-layer prepare requests |
| ---: | ---: | ---: |
| 128 | 1 | 86 GiB |
| 2,048 | 1 | 86 GiB |
| 8,192 | 2 | 172 GiB |
| 32,768 | 8 | 688 GiB |

These were code-path byte requests, not measured physical NAND traffic. Page
cache hits can reduce device I/O. Conversely, an 86 GiB routed store exceeds
the host's 64 GiB unified memory, so repeated scans are likely to be
cache-expulsive. Selected-address cache loads are additional requests. Current
Metal `pread_bytes` telemetry counts the expert cache loader but not the layer
prepare worker, so it cannot by itself reveal this amplification.

The implemented correction does not merely add affine2 to the existing IQ2
predicate: that predicate has a 760-token automatic ceiling and respects
disable/quality conditions that the affine2 path cannot safely use. Instead,
validated store metadata is now the authority for one mandatory affine2
selected-address decision. The C planner and Metal dispatch both use it for
every non-empty SSD batch. The C layer-span builder also rejects an affine2
full-layer request, and the address-table builder ignores its optional disable
bisect while constructing a mandatory affine2 selected batch. IQ2/Q2 thresholds
and diagnostic gates remain unchanged. I/O overlap and cache policy remain
separate, unmodified experiments.

Two secondary gaps also need explicit treatment:

- selected page-in/readahead diagnostics derive the up-component size from the
  gate size, which is invalid for affine2's 3 MiB gate and 2.5 MiB up records;
- the existing shared-expert/router I/O overlap and DeepSeek scheduling gates
  are IQ2/Q2-only, so affine2 currently uses the synchronous selected loader
  and excludes the resident NAX/TensorOps advantage behind the Qwen results.

## Experiments executed

### Clean build and model-free correctness

The candidate passed:

```sh
make clean
make -j8
make model-free-test
make premerge
```

This included the Metal kernel suite, `--metal-expert-pack`, the ExpertMajor
Python tests, SSD/cache policy tests, server/flag checks, and the other
model-free gates. The complete premerge gate also passed, including build
isolation, documentation links, deterministic dataset/prompt checks, and a
fresh model-free run. Its CPU-only isolation build emitted the repository's
existing unused-function warnings and completed successfully. The affine2
kernel coverage exercises exact selected-address
g32/g64 dequantization with F32/F16 right-hand sides and token counts
1/33/255/256. The sparse test maps the full 43 x 256, 86 GiB address geometry
without materializing that storage. It is numerical and structural coverage,
not a throughput measurement or store-to-output end-to-end test.

Runtime Metal source SHA-256 reported by the suite:
`96a7d8a4f37593fc1153d359de99ca40dfd9e0d8a146b289ff0600bd1b928c67`.

### Follow-up planner regression coverage

The correction adds model-free coverage at both sides of the C/Metal boundary:

- affine2 selects the mandatory address path at 1, 2, 760, 761, 2,048, and
  4,096 tokens, while an empty batch and non-SSD execution do not;
- the result remains mandatory with the ordinary selected-batch and address
  table disable flags set, and with the quality gate that disables optional
  IQ2 batching;
- an affine2 full-layer span request fails before any page-in or `pread` can be
  scheduled;
- the unchanged IQ2/Q2 policy remains off at 1 token, on at 2 and 760, and off
  at 761, 2,048, and 4,096 tokens;
- the Metal affine2 admission helper covers zero, boundary, and long batches,
  plus SSD, address-table, and geometry negatives.

The follow-up passed:

```sh
make -j8
./build/metal-arm64/bin/ds4_test --metal-expert-pack
./build/metal-arm64/bin/ds4_test --metal-kernels
make premerge
```

`make premerge` repeated build isolation and the complete model-free gate. The
CPU isolation build emitted only the repository's existing unused-function
warnings. The independent sparse-store tests prove planner/dispatch agreement
over the virtual 86 GiB geometry; they still do not execute a physical affine2
artifact.

### Controlled model A/B attempt

Clean detached worktrees for the control and candidate were built separately.
The planned first regression lane used the unchanged qualified IQ2/Q2 artifact,
forced SSD, exact 3,000-record cache/preload, a 128-token prefill plus 128
greedy decode tokens, 32K context allocation, and A/B/B/A ordering. This lane
could only detect a regression introduced by the scaffolding; it could not
prove an affine2 benefit.

The first A arm terminated immediately with:

```text
M5 benchmark requires AC power
```

No model was loaded, no inference process ran, and no timing row or partial
cohort was retained. The guard was not bypassed. A second blocker is that no
affine2 GGUF exists; the roughly 14 GiB free on the internal disk is also far
below the estimated 94.2 GiB output plus safe conversion scratch.

### Exploratory 64 GiB battery smoke after the planner correction

At the user's request, three direct `ds4-bench` smokes were run on the qualified
IQ2/Q2 artifact without AC power. They were deliberately outside the M5
acceptance harness and used `--power 50`, which adds idle time to cap duty
cycle. They are not an A/B cohort, do not exercise affine2, and their throughput
cannot qualify or promote this change. The first two runs are correctness
failures; a third, minimal AUTO-cache discriminant is finite.

The two explicit-cache failures used this code and runtime identity:

| Item | Identity |
| --- | --- |
| Source parent | `97f7a19e5361f0c74454d39fe3db4f2e56e246c4` |
| Dirty binary diff SHA-256 | `34da5e2512339797a7c1e39d45b13b3941bd0f9b1ff9a456d5376645d3698082` |
| `ds4-bench` SHA-256 | `603d3dd6e12e74eee895e6d9a53ab39552982ad5f1357a9d2b7053bc470ab5ac` |
| Runtime Metal source SHA-256 | `96a7d8a4f37593fc1153d359de99ca40dfd9e0d8a146b289ff0600bd1b928c67` |
| Prompt SHA-256 | `f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f` |
| Model | qualified DeepSeek v2 identity recorded above; published hash was not recomputed |

Both processes used SSD streaming, a 517-record cache (3.408 GiB), one
preloaded record, a 512-token allocation, greedy decode, and the prose prompt.

The lane labels are
`battery-power_exploratory-unqualified_ctx33_g8_p50_ssd-cache517_preload1`
and
`battery-power_exploratory-unqualified_ctx128_g16_p50_ssd-cache517_preload1_warm-biased-retry`.

```sh
DS4_METAL_MEMORY_REPORT=1 \
DS4_METAL_STREAMING_EXPERT_TIMING_SUMMARY=1 \
./build/metal-arm64/bin/ds4-bench --metal -m "$DEEPSEEK_V2" \
  --ssd-streaming --ssd-streaming-cache-experts 517 \
  --ssd-streaming-preload-experts 1 --power 50 \
  --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start "$N" --ctx-max "$N" --ctx-alloc 512 \
  --gen-tokens "$G" --csv "$OUT.csv" \
  --dump-frontier-logits-dir "$OUT.logits" \
  --dump-decode-evidence-dir "$OUT.evidence"
```

| Lane | Result | Prefill | Decode | Prefill cache hit | Prefill expert loads / evictions | Prefill `pread` |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 33 + 8, cache state unclassified | **FAIL: non-finite decode** | 5.01 tok/s | invalid 1.16 tok/s | 68.33% | 2,696 / 2,179 | 17.771 GiB, 8,088 calls, 593.340 ms |
| 128 + 16, uncontrolled warm-biased retry | **FAIL: non-finite decode** | 12.95 tok/s | invalid 1.45 tok/s | 85.24% | 4,875 / 4,358 | 32.135 GiB, 14,625 calls, 1,030.861 ms |

The planned footprint was 4.44 GiB. The host was at 26-25% battery and
discharging. System free-memory readings stayed between 78% and 93%; the peak
observed wired count was 697,091 16 KiB pages (10.637 GiB). The pre-existing
system `Swapouts` counter stayed exactly `660581` before, during, and after the
runs, and no competing inference process was present.

The first 128-token attempt completed its prefill and then stopped because the
requested logits directory had not been created. It produced no retained CSV
or evidence and did not change the swapout counter. The directory was created
and the same lane was rerun, which makes its prefill warm-biased in addition to
the decode correctness failure.

Retained artifact identities (scratch files were not committed) are:

| Lane | CSV SHA-256 | Frontier logits SHA-256 | Decode evidence SHA-256 |
| --- | --- | --- | --- |
| 33 + 8 | `7e545bb9eb10e5e64143793416a7ea45ff89a0b374d29e1316ca1d5f7e5e1431` | `2b7a9559a7afb48addd345f20a2495efa60c74b0bac7d787b7ff6446b5e7fc8d` | `daadf654c78a22200d3e74bbdc68d08e2b4fa0d825242c3e2336f7fdc2cbaf8c` |
| 128 + 16 | `1ce1873f1362f129b2adb63f44a4dad5d6fc5eea5b4ed23839324e4797bdbe20` | `c66812e6aa99ca6b450a8af7092a9dfd9d44ac957f4ee8396b94da626beb5c4c` | `d1392d510b53e8ff60b95491c4b062e459a49ce31bd9637d8dda34e2ec7df3d7` |

The frontier logits were finite, selecting token 54 at 33 tokens and token 14 at
128 tokens. The decode evidence then contained `[54, 0, 0, 0, 0, 0, 0, 0]` and
`[14, 0, ...]`, respectively, and all 129,280 final logits in each file were
non-finite (`null` in JSON). Thus every decode timing above is invalid.

This is a pre-existing explicit-cache phase bug, not an affine2 effect. An
explicit `--ssd-streaming-cache-experts 517` bypasses the AUTO cache planner,
leaving the four DeepSeek phase targets at zero. The post-prefill transition is
still called, reduces the configured cache from 517 to zero, and the first
decode evaluation performs no expert loads before producing non-finite logits.
Git history attributes the unconditional transition to `0432c121`. Fixing that
cache policy is intentionally kept out of this planner/dispatch correction.

One minimal 33 + 2 discriminant, labelled
`battery-power_exploratory-unqualified_ctx33_g2_p50_ssd-auto`, then omitted the
explicit cache and preload, letting the qualified AUTO policy select 4,387
records (28.92 GiB; 29.95 GiB total planned). It used this updated
implementation identity:

| Item | AUTO discriminant identity |
| --- | --- |
| Source parent | `97f7a19e5361f0c74454d39fe3db4f2e56e246c4` |
| Code-only dirty diff SHA-256 | `6bdd4a539c8444a9e454da4b9ccebaf98a529826ffe55b2d4fb1f1c3578f7fe5` |
| `ds4-bench` SHA-256 | `ad7c4dddfbee051fc3262b6cf03898d7f26bbd34949630241ce25412165fe0df` |
| Runtime Metal source SHA-256 | `96a7d8a4f37593fc1153d359de99ca40dfd9e0d8a146b289ff0600bd1b928c67` |

The AUTO lane produced tokens `[54, 93729]`, final argmax 14, and 129,280
finite final logits. Its prefill frontier file is byte-identical to the failed
33-token lane, isolating the break after prefill. Decode recorded an 83.91% hit
rate, 83 loads/evictions, and 0.547119 GiB over 249 `pread` calls. The measured
2.81 prefill and 0.32 decode tok/s remain non-promotable because this was one
two-token, duty-capped, battery-powered process with no control arm.

AUTO scratch identities were CSV
`7abf54beecd83334e7df328d3d954ebdd86dbf2c507afb36360c6f9dcce7c2bc`,
frontier logits
`2b7a9559a7afb48addd345f20a2495efa60c74b0bac7d787b7ff6446b5e7fc8d`,
and decode evidence
`bfbbf26af9bf7730023b8e1dd8382047400bcbd7376fb187bf3ed7649cbe3311`.
Battery fell from 21% to 20%; free-memory bottomed at 34%, wired memory peaked
at 2,114,499 16 KiB pages (32.265 GiB), and `Swapouts` remained `660581`.

## 2026-07-23 AC follow-up on the qualified IQ2/Q2 artifact

This follow-up does not time affine2: no physical affine2 artifact exists. It
uses the qualified IQ2/Q2 ExpertMajor v2 artifact to fix the cache-transition
bug found above, establish context measurements, and screen the simplest SSD
prefill/decode changes while the host is on AC power.

All retained successful measurement and cohort arms used separate processes,
128 greedy decode tokens, the published model identity above,
process-contamination checks, finite frontier and decode evidence, and a
zero-new-swap gate. Explicitly rejected/aborted screens are identified below
and excluded from performance aggregates. Scratch logs remain outside the
repository. The prose prompt SHA-256 was
`f53e0d80cb2d4492d24ebd63c7000c397b16ae70f9bf09b3763e5d8323ec209f`.
The 32K security/coding prompt is the checked-in fixture.

### Explicit cache correctness fix

The explicit-cache failure was corrected by making a DeepSeek phase transition
a no-op unless AUTO initialized a non-zero prefill target and a larger decode
target. Thus `--ssd-streaming-cache-experts 517` remains 517 after prefill
instead of becoming zero. The pure transition helper covers zero/equal/valid
targets, and the fixed 517-record canary produced 128 decode tokens and all
129,280 finite final logits. Its evidence matched the retained fixed-cache and
pre-MLX-control runs at the same prompt and context.

### Fixed-cache context measurements

These are exploratory context measurements, not an A/B performance promotion.
They use exact cache 517 and therefore emphasize SSD miss cost.

| Prompt | Context | Prefill | Decode | Decode p95 | Wired peak | New swap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prose | 128 | 25.81 tok/s | 9.20 tok/s | 126.453 ms | 20,250 MiB | 0 |
| Prose | 2,048 | 156.42 tok/s | 7.72 tok/s | 140.060 ms | 17,304 MiB | 0 |
| Prose | 8,192 | 172.83 tok/s | 8.89 tok/s | 124.739 ms | 16,775 MiB | 0 |
| Prose | 32,768 | 165.73 tok/s | 8.04 tok/s | 136.300 ms | 16,622 MiB | 0 |
| Security/coding | 32,768 | 173.52 tok/s | 6.55 tok/s | 165.101 ms | 16,745 MiB | 0 |

Prompt identity matters. The prose and security/coding 32K rows must not be
combined as repeat measurements. The security/coding row's raw frontier
evidence SHA-256 was
`5547c4434479e3fe61672df5523e994ee751986362a2465a32d3b2bc340de2d0`,
matching its historical oracle.

### Cache-size and eviction sweep

An exact 128-token screen shows why decode cache capacity dominates SSD decode:

| Cache | Prefill | Decode | Hit rate | Decode `pread` | Read amplification | Wired peak | New swap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 517 fixed | 25.81 tok/s | 9.20 tok/s | 0.604 | 86.208 GiB | 3.109 | 20,250 MiB | 0 |
| 3,097 fixed | 21.64 tok/s | 12.71 tok/s | 0.896 | 22.649 GiB | 1.137 | 33,584 MiB | 0 |

The larger fixed cache improves decode 38.2% and reduces measured decode reads
73.7%, but it slows this prefill screen because a fixed budget cannot contract
during prefill. That result motivated phase-specific AUTO targets rather than
one large fixed cache.

At 2K, AUTO14 and AUTO17 were then compared in a valid A/B/B/A cohort. Control
drift was 0.032% for prefill, 1.30% for decode throughput, and 2.41% for p95.
Every run added zero swap and produced identical evidence.

| 2K policy | Prefill mean | Decode mean | Decode p95 mean | Wired peak mean | New swap |
| --- | ---: | ---: | ---: | ---: | ---: |
| AUTO17 / 4,387 | 155.945 tok/s | 10.820 tok/s | **94.699 ms** | 42,314 MiB | 0 |
| AUTO14 / 3,613 | **161.000 tok/s** | 10.775 tok/s | 101.183 ms | 37,090 MiB | 0 |

AUTO14 improves the reported prefill interval 3.24%, but decode throughput is
effectively neutral and p95 is 6.85% worse. `ds4-bench` times phase restore and
hotlist seed inside its prefill interval; both policies use 259 records during
the actual batched prefill, while 3,613 versus 4,387 is the post-prefill target.
The table must therefore not be interpreted as a math-kernel prefill result.

The existing layer-staleness eviction tie-break was screened once at AUTO14:
160.11 prefill tok/s, 10.87 decode tok/s, and 99.980 ms p95. It did not recover
AUTO17's 94.699 ms cohort mean and was rejected without a full cohort.

### Context-tiered candidate: not promoted

One candidate kept 17 complete route cycles for short-context decode,
contracted to 12 cycles before decode at the 8K guard, and to eight cycles at
the 65K guard. Batched prefill stayed at the 259-record correctness floor.

| Resulting context | Post-prefill/decode target | Records |
| --- | ---: | ---: |
| 0 through 8,063 | 17 route cycles | 4,387 |
| 8,064 through 65,407 | 12 route cycles | 3,097 |
| 65,408 and above | 8 route cycles | 2,065 |

The 128-token guard made the candidate runtime boundaries 8,064 and 65,408;
the pure long-target helper was checked at 8,192 and 65,536. An explicit cache
still bypassed all phase targets.

Two final dirty-build canaries based on `6315de2` used binary SHA-256
`a6aabd7135be15ccc81d37f57fc434a4a68658c784ede410b732dc68138191c1`
and code diff SHA-256
`194c8d5e6798d29b36cf3e3784583ed7a6faeabe9f5822aad03e84282f4d3b61`.

| Canary | Initial ceiling | Prefill target | Decode target | Prefill | Decode | p95 | Wired peak | New swap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2K | 4,387 | 259 | 4,387 | 150.78 tok/s | 10.88 tok/s | 96.526 ms | 42,556 MiB | 0 |
| 8K | 4,387 | 259 | 3,097 | 195.96 tok/s | 9.76 tok/s | 113.087 ms | 33,952 MiB | 0 |

The 8K log explicitly records `4387 -> 259 -> 3097`. Against the earlier
fixed-517 8K canary, the tiered canary is 13.4% faster in prefill, 9.8% faster
in decode, and 9.3% lower in p95. This is encouraging but remains an
unpaired canary comparison; the percentages are not a promotion-grade A/B.

The required direct comparisons against the actual AUTO12 baseline did not
meet the stability gate. The AUTO12/AUTO17 2K A/B/B/A sequence was:

| Arm | Policy | Prefill | Decode | Decode p95 | New swap |
| --- | --- | ---: | ---: | ---: | ---: |
| A1 | AUTO12 | 144.98 tok/s | 10.00 tok/s | 112.292 ms | 0 |
| B1 | AUTO17 | 143.80 tok/s | 10.29 tok/s | 101.771 ms | 0 |
| B2 | AUTO17 | 147.00 tok/s | 10.66 tok/s | 95.575 ms | 0 |
| A2 | AUTO12 | 166.47 tok/s | 10.42 tok/s | 107.292 ms | 0 |

The controls drifted 14.8% in prefill. AUTO17's mean decode signal was 2.6%
higher and p95 10.1% lower, but its prefill mean was lower and the cohort is
invalid. It cannot establish an overall speedup or restore a tier previously
reduced for delayed swap.

AUTO14 was also compared directly against AUTO12 after the host reported 83%
free memory. Its A/B/B/A sequence was:

| Arm | Policy | Prefill | Decode | Decode p95 | New swap |
| --- | --- | ---: | ---: | ---: | ---: |
| A1 | AUTO12 | 164.96 tok/s | 10.43 tok/s | 105.261 ms | 0 |
| B1 | AUTO14 | 160.19 tok/s | 10.61 tok/s | 102.691 ms | 0 |
| B2 | AUTO14 | 156.06 tok/s | 10.55 tok/s | 104.694 ms | 0 |
| A2 | AUTO12 | 92.25 tok/s | 6.92 tok/s | 226.321 ms | 0 |

A2 slowed despite zero new swap and no competing inference process, making the
cohort unusable. Since neither candidate proves a stable prefill and decode
win over AUTO12, the context-tiered code/test diff was withdrawn. AUTO12 / 12
cycles (3,097 records) remains the runtime default; 65K+ continues to use the
existing eight-cycle long-context cap.

### Simple prefill fast-path screens

The three lowest-effort existing controls were screened after the cache sweep.
None is promoted.

| Candidate | Design | Result | Correctness / safety | Decision |
| --- | --- | --- | --- | --- |
| Batch HC+RMSNorm fusion | Exact 517, 2K A/B/B/A | Prefill means 189.88 A vs 190.78 B (+0.48%); p95 114.38 vs 120.89 ms | Evidence identical, swap zero, but mean-normalized control prefill drift 3.65% | Reject as below noise |
| Prefill chunk 8,192 | Exact 517, 8K screen | Candidate aborted before decode | **372 new swapout pages** | Hard reject |
| Layer prepare-ahead 2 | Exact 517, 2K A/B/A screen | 127.08 A1, 148.22 B, 154.45 A2 prefill tok/s | Evidence identical and zero added swap; A drift 21.5% | Reject as page-cache state, not causal |

The unsafe 8,192-chunk arm changed the system swapout counter from 897,897 to
898,269. The runner stopped it at the first sample over the zero-page limit;
no decode evidence was produced. Subsequent retained screens used 898,269 as
their immutable baseline and added zero pages. This rejected arm is never
mixed into safe performance aggregates.

### GLM and Qwen parallel audit

Two independent read-only agents checked for planner/dispatch and cache-policy
problems analogous to DeepSeek. They did not modify code or run a model.

| Family | Priority | Finding | Consequence |
| --- | --- | --- | --- |
| Qwen | P1 | SSD frontend requires grouped selected-address from batch 8, while Metal affine4 uses it from 32 | Batches 8 through 31 can return `expert_group_used=0` and abort |
| Qwen | P1/P2 | Cache growth rechecks pressure only on the low-RAM tier | 32 GiB AUTO and 64 GiB forced SSD can allocate slabs after a stale admission snapshot |
| Qwen | P2 | Planner charges an 8,192-token arena while SSD runtime uses 2,048 | About 1.47 GiB / 894 experts of conservative over-accounting |
| Qwen | P2 | Ordinary phase-finish failure lacks macro-prefill rollback semantics | Public prompt position and abort semantics can diverge |
| GLM | P1 | AUTO lacks a measured host gate below 64 GiB | An unqualified low-memory host may receive the generic aggressive plan |
| GLM | P1 | Admission does not require the qualified Q2_K routed format | Unsupported routed quantizations can reach incomplete dispatch coverage |
| GLM | P2 | `DS4_METAL_GLM_DISABLE_STREAMING_EXPERT_CACHE` can make mapping and dispatch disagree | Full-layer mapping plus cache loads can duplicate SSD work |
| GLM | P2 | Explicit numeric cache bypasses snapshot/planner admission | Lazy slab allocation can create delayed pressure or swap |

Neither family has the exact DeepSeek zero-target transition bug. Qwen's first
finding is the closest planner/dispatch mismatch and should be fixed with one
shared threshold, plus boundary tests at 7/8/31/32. GLM/Qwen runtime fixes and
their required model matrices are deliberately separate from this DeepSeek
branch result.

## Required experiment before reconsideration

1. Implement a fail-closed writer/verifier with full donor and base provenance,
   generate the exact artifact on storage with at least 190-210 GiB safe
   scratch, and publish its byte count and SHA-256.
2. Resolve the resident-versus-SSD contract before model admission. Add C and
   Python fixtures plus store -> `pread` -> cache -> address table -> routed MoE
   end-to-end tests and AUTO/resident/SSD startup tests.
3. Validate the now-shared C/Metal selected-address decision on the physical
   affine2 artifact at every required frontier. Prove that no full-layer
   prepare occurs, and add separate counters for layer-prepare bytes, cache
   loader bytes, and physical disk I/O before interpreting performance.
4. Preserve the now-fixed explicit-cache transition coverage and make
   non-finite logits fail closed before using any new cache control in a
   benchmark cohort.
5. Run the official continuation scorer against the pinned donor, preserve
   greedy evidence, and qualify the quantization independently of timing.
6. On AC power with no competing inference process and zero swapout, run
   isolated cold and warm A/B/B/A cohorts at 128, 2,048, 8,192, and 32,768,
   each with at least 128 greedy decode tokens. Use both the prose and
   security/coding prompts at 32K. Record plan, control drift, within-arm
   spread, TPOT p50/p95, route uniqueness, cache hits/misses/evictions,
   `pread` calls/bytes/time, layer-prepare bytes, pressure, and swap.
7. If the correction touches shared SSD/Metal policy, repeat the required
   DeepSeek, GLM, and Qwen regression matrices. A synthetic kernel win may
   reject or justify further work, but may not promote the runtime.

Promotion requires correctness and quality gates, control drift below 3%, an
effect larger than noise and spread, zero unsafe memory pressure/swapout, every
frontier at least 95% of its qualified gold, and a geometric mean at least 98%.
For this candidate the causal break-even condition is:

```text
compute_ms_saved > affine2_extra_io_ms + reload_penalty_ms + noise_margin_ms
```

Until that condition is measured on the final artifact, the current affine2
SSD path remains HOLD.
