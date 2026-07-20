# Agent-friendly refactor validation

Date: 2026-07-20

Branch: `codex/agent-friendly-refactor`

Base: `d8d673858f90834522bbe878951a534d8c6508b4`

Host: Apple M5 Pro, 64 GiB unified memory, macOS 26.5.2 build 25F84, AC power.
Model processes ran sequentially. The system page cache was not dropped and no
storage stress test was performed.

## Scope

This gate covers the repository simplification, ExpertMajor v2-only runtime,
Metal family isolation, and the final adaptive-cache correction. CUDA and ROCm
are intentionally absent from the active tree and remain recoverable from Git.
The anti-regression rules in `CONTRIBUTING.md` and `QA_BEFORE_RELEASES.md` stay
authoritative.

The release artifact identities were checked earlier in the publication lane;
the large files were not rehashed during every speed run:

| Family | Bytes | Complete output SHA-256 |
| --- | ---: | --- |
| Qwen3.6-35B-A3B | 20,808,566,880 | `d7c43a6388ec20e6fe5530850350f96fdb0ac37c5ce36d3e5f92b172c447f56b` |
| DeepSeek V4 Flash | 86,720,114,272 | `8378080263eb9224f7228d72e2afa4ac3cf74a116023fdec2c596ff228a33e3f` |
| GLM 5.2 | 262,147,193,504 | `7f5017e3076e706c78f2a5322b035a9e2f6519c65ff5b6be8b2d91aeff61505d` |

## Tested tree and commands

The working tree is intentionally uncommitted during this gate. The final
candidate was built by the last successful `make premerge`; no C, header,
Metal, implementation-include, or Makefile input is newer than the binaries.
The pre-report tracked diff SHA-256 recorded by the DeepSeek runner was
`929ebbbe147fea084f51b64c2561c3878053224857a141b61c8f82f0c1763367`.
The model-tested Metal binaries are:

| Binary | SHA-256 |
| --- | --- |
| `ds4`, GLM run | `d2eb963f3fc117b5da723a483fdd12ec6106c3f1dd2bd767bb463e4b336f092e` |
| `ds4-bench`, Qwen and DeepSeek runs | `3bd4fb85117b676340d6130d89261c47c78cf242dd6390fc306cda90703e983f` |

Documentation-only corrections after the runs change the repository diff
digest but not either tested binary. With `QWEN_V2`, `DEEPSEEK_V2`, and
`GLM_V2` resolved to the artifact identities above, the model commands were:

```sh
./build/metal-arm64/bin/ds4-bench -m "$QWEN_V2" --metal --resident \
  --power 100 -t 8 --prompt-file speed-bench/promessi_sposi.txt \
  --ctx-start 2048 --ctx-max 2048 --step-mul 2 --gen-tokens 16 \
  --csv "$QWEN_RESULT" --dump-decode-evidence-dir "$QWEN_EVIDENCE"

DS4_M5_MODEL="$DEEPSEEK_V2" DS4_M5_PREFIX="$DEEPSEEK_PREFIX" \
  DS4_M5_CTX_START=128 DS4_M5_CTX_MAX=128 DS4_M5_CTX_ALLOC=32768 \
  DS4_M5_PRELOAD_EXPERTS=4096 \
  speed-bench/run_m5_dsflash_arm.sh final-adaptive-4387 auto 256

./build/metal-arm64/bin/ds4 -m "$GLM_V2" --ctx 8192 \
  --prompt-file /tmp/glm-prefill-288-exact.txt --temp 0 -n 32
```

## Model-backed result

| Family and lane | Prefill | Decode | Correctness and memory | Gate |
| --- | ---: | ---: | --- | --- |
| Qwen resident, exact 2,048+16 lane | **315.93 t/s** | **29.48 t/s** | Decode-evidence SHA-256 `399504c6ce3d4531ee0f2207702e96e2324c9b5c8dbf98adf47dfb9e64cae54d`; swap unchanged | Pass: decode is -0.20% and prefill -0.95% against the first 318.96/29.54 v2 reference arm |
| DeepSeek AUTO, 128+256 cold | 20.13 t/s | 13.03 t/s | 4,387 records / 28.92 GiB; exact frontier-logit SHA-256 `71fd3be0732e0fe97b9f104112911dc937896257c604c64eae851e36fa142441`; zero swapout | Cold file-cache observation |
| DeepSeek AUTO, immediate warm repeat | **23.22 t/s** | **13.91 t/s** | Same artifact, policy, evidence and binary; page-ins fell from 5,168,431 to 4,399,996 pages; zero swapout | Pass: above the earlier current/base/current cohort and -1.97% from the isolated 14.19 single-run peak |
| GLM AUTO, exact 288+32 lane | **11.82 t/s** | **1.83 t/s** | Exact output SHA-256 `2803fda8b47acff3aedd24bd7609b0c649602ca1fa6d908368b57fe2a586a5c2`; 601-record / 6.93 GiB cache; 50.50 GiB reclaimable; swap unchanged | Pass: decode is above the 1.82 reference |

The final candidate therefore preserves byte-identical evidence for all three
families and requalifies every tracked performance lane. DeepSeek's 14.19 t/s
value remains a single best observation rather than a same-condition median;
the final 13.91 t/s result is compared primarily with the interleaved cohort
below.

### Same-condition base comparison

The base commit was built in an isolated clean worktree. Its Metal binary
SHA-256 values were `d28a2237345d8a52c5325b5e70d9c114ca22fdffab4b31d99422229a900ea14e`
for `ds4-bench` and
`1ec89fd44d1b2e8689f2430a87ad8bc07c06b3aa7e9ebea008fc77f1e2aec1db`
for `ds4`.

DeepSeek was interleaved current/base/current without changing the artifact,
prompt, 4,387-record AUTO target, or evidence. Decode measured 13.41, 13.64,
and 13.22 t/s; all three emitted the same frontier-logit SHA-256 and zero
swapout. The final 13.91 warm result is above all three interleaved observations
and passes the bounded regression gate while keeping exact evidence and zero
swapout.

The GLM base/current/base sequence measured 1.82, 1.28, and 1.06 t/s. All three
produced byte-identical output and unchanged swap. The severe monotonic collapse
across both binaries proves that this sequence became host-state contaminated;
it cannot attribute the slower second arm to the refactor. The final 1.83 t/s
candidate result above supplies the required rested current-binary gate.

## Structural QA

`make premerge` passed on the recorded diff. It includes the context budget,
97 local documentation links, the deterministic 7,124-prompt imatrix rebuild
check, Metal -> CPU -> Metal build isolation, the complete model-free suite,
Metal kernels, ExpertMajor v2 reader/converter verification, SSD residency
policy tests, server parser/state tests, agent unit tests, tokenizer/reference
fixtures, and `git diff --check`. The context audit reported 43,748 lines in the
largest source and zero personal absolute paths.

CUDA and ROCm source/build targets are absent and are recorded as frozen, with
their reactivation requirements retained in `QA_BEFORE_RELEASES.md`. The
complete manual release sign-off was not attempted: live server streaming,
model-backed disk-KV eviction, and manual agent interruption/tool-loop gates
remain outstanding. This report therefore cannot authorize a release by
itself.

## Adaptive-cache decisions

DeepSeek and GLM share the ExpertMajor container but do not share one cache
constant or one scheduling policy.

- DeepSeek's GLM-contaminated 3,097-record lane recovered from 9.x to 13.06
  t/s after victim reuse was family-gated. A clean sweep then measured 13.87 at
  4,387 and 14.22 at 4,645 records. The stable 9/16 envelope admits 4,387, so
  release AUTO selects at most 17 complete route cycles on the 64 GiB tier. At
  engine open, the point-in-time pressure plan can select fewer than 4,387;
  later phase switching only restores that startup-admitted target.
- The 4,903- and 5,500-record DeepSeek probes were terminated by the memory
  guard when transient free pressure crossed 20%. Neither swapped; neither is
  a performance result.
- GLM's larger 1,801-record cache reduced misses but previously regressed
  1.81 to 1.73 t/s. A combined 1,801-record/split-threshold probe reached only
  1.52 t/s. GLM therefore keeps the 601-record DS4-managed pageable tier on
  64 GiB.
- GLM still uses the rest of memory adaptively. A warm startup reported about
  43 GiB file-backed/inactive memory managed by macOS outside process RSS.
  Forcing that memory into a larger explicit expert cache would displace the OS tier
  and narrow useful per-layer I/O concurrency.

No user-facing cache flag was added. Explicit cache overrides remain diagnostic
and authoritative; the normal startup command remains only model plus context.

Here, `adaptive` means point-in-time admission at engine startup plus, for Qwen
SSD and DeepSeek Flash SSD, a phase-local prefill/decode transition within the
already admitted budget. It is not a continuous memory-pressure feedback
controller. On the qualified 64 GiB GLM lane, DS4 keeps a 601-record pageable
expert cache because larger explicit caches reduced end-to-end decode speed;
macOS independently grows and reclaims the remaining file-backed GGUF cache. At
96 and 128 GiB, GLM remains SSD-only and uses the generic startup pressure
candidate, but those cache optima are not yet physically qualified.

## Release decision

The structural, model-admission, correctness, memory-safety, and tracked
performance gates pass on the final candidate binaries. Qwen and GLM meet their
published lanes; DeepSeek's warm result is within 2% of its isolated peak and
above the prior same-condition interleaved cohort, with exact logits and zero
swapout.

This report removes the previous performance blocker to merge. It does not by
itself authorize publication: the remaining manual release checks named above
must still pass, and the committed tree must be rebuilt once to record clean
non-`dirty` artifact hashes before binaries or model cards are published.

Related evidence:

- [DeepSeek native ExpertMajor campaign](2026-07-17-deepseek-native-expert-major.md)
- [GLM ExpertMajor v2 qualification](2026-07-20-glm52-expert-major-v2.md)
- [ExpertMajor v2 runtime roadmap](../expert-major-v2-roadmap.md)
- [Runtime support contract](../contracts/RUNTIME_SUPPORT.md)
