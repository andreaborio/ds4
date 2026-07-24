# Gold Metal + SSD contract

This fork treats Apple Metal as the production runtime. The normal policy is
`AUTO`, interpreted by model family: DeepSeek and Qwen may resolve to the
qualified full-model mapped path or SSD streaming, while GLM always resolves to
its qualified SSD-streaming path.

## User-facing contract

```sh
make                         # publishes ./hebrus* plus legacy ./ds4* aliases
./hebrus -m DEEPSEEK-OR-QWEN-DS4-ExpertMajor-v2.gguf  # AUTO
./hebrus -m DEEPSEEK-OR-QWEN-DS4-ExpertMajor-v2.gguf --resident
./hebrus -m DEEPSEEK-OR-QWEN-DS4-ExpertMajor-v2.gguf --ssd-streaming
./hebrus -m GLM-5.2-DS4-ExpertMajor-v2-Q2_K.gguf --ctx 8192  # AUTO -> SSD
./hebrus --build-info
```

The `./ds4` and `./ds4-server` names remain byte-identical compatibility
aliases throughout the bridge and 1.x compatibility window.

DeepSeek V4, GLM 5.2, and Qwen3.6 inference accepts only a validated embedded
`ds4.expert_major.v2` store on Apple Metal. Canonical GGUFs remain offline
converter inputs; ExpertMajor v1, external sidecars, CPU, CUDA, ROCm, and
distributed inference fail closed. Normal startup requires no ExpertMajor,
sidecar, backend, cache, preload, or power flag.

For DeepSeek and Qwen, `--resident` (also `--no-ssd-streaming`) and
`--ssd-streaming` are explicit qualification overrides. Cache, cold-streaming,
or preload options imply SSD mode and are rejected when combined with
`--resident`. GLM release startup is AUTO-only: AUTO selects SSD streaming and
an explicit resident request is rejected. GLM SSD/cache controls remain
diagnostic rather than alternate release commands.

On macOS, `make cpu` writes reference/debug binaries only to
`build/cpu-<arch>/bin/` and never replaces the Metal root binaries. CUDA and
ROCm source and build targets are frozen outside the active tree; their
conditional reactivation gates are in `QA_BEFORE_RELEASES.md`.

## Build identity and isolation

macOS artifacts are separated completely:

```text
build/metal-<arch>/{obj,bin}
build/cpu-<arch>/{obj,bin}
```

Objects are not shared between profiles.  `make` and `make metal` publish the
five root commands as symlinks to the Metal profile; `make cpu` never changes
those symlinks.  Every executable supports `--build-info`, which reports git
revision (including `-dirty`), compiled backend, and architecture.

The invariant is executable:

```sh
make build-isolation-test
```

It performs Metal -> CPU -> Metal builds, checks linkage with `otool`, checks
all five root links, and verifies each binary's provenance.

## AUTO residency planner

For Metal, AUTO compares a conservative required-memory plan with
`recommendedMaxWorkingSetSize`. The generic DeepSeek calculation is:

```text
required = model tensors
         + optional MTP file
         + context / KV / scratch estimate
         + max(20% of recommended working set, 2 GiB)

budget   = recommended working set - explicitly simulated used memory
```

Qwen uses the device's physical RAM and recommended Metal working set through
the hardware policy in
[`ADR 0004`](docs/adr/0004-qwen-metal-hardware-memory-policy.md):

```text
qwen reserve  = max(2 GiB, physical RAM / 16)
              + max(0.25 GiB, physical RAM / 64)
qwen required = model tensors + context / KV / scratch + qwen reserve
```

The named 16/24/32/36/48/64/96/128 GiB profiles make the selected hardware
class visible, but the byte calculation is continuous and always uses the live
Metal device values. The Qwen live-pressure gate uses the same reserve as the
fixed gate.

For DeepSeek and Qwen, `required <= budget` may resolve to resident; otherwise
AUTO resolves to SSD. Model-specific pressure and artifact gates may still
reject resident mode. GLM applies its family override and resolves AUTO to SSD
regardless of the generic resident calculation. The resolved mode, reason,
components, and cache plan are printed at startup. There is no non-Metal AUTO
inference fallback.

If Metal cannot report a working-set recommendation, AUTO fails safely unless
the user supplies an explicit SSD cache budget.  SSD + MTP remains unsupported.

### ExpertMajor v2 family policy

Every supported MoE family uses the same self-describing ExpertMajor v2 storage
contract and its own qualified Metal consumer:

- DeepSeek AUTO may select its artifact-qualified resident or SSD path.
- Qwen3.6-35B-A3B AUTO admits the complete mapped-tensor path only when both the
  fixed working-set plan and a point-in-time live-memory pressure check pass;
  otherwise it selects SSD streaming. Explicit `--resident` fails unless both
  checks pass.
- GLM 5.2 AUTO always selects the qualified SSD path. Resident mode is not a
  capacity-dependent fallback and is rejected even on larger hosts.

Admission snapshots are conservative checks, not guarantees against later
memory-pressure changes.

Qwen's SSD planner charges static page coverage, session/runtime memory,
ordinary host headroom, pressure margin, and Metal headroom. Pageable static
pages may share the larger ordinary reserve but are never omitted. Under normal
macOS pressure, equivalent free and bounded file-backed pages receive equal
credit on every Qwen profile, so warming the GGUF cannot by itself shrink AUTO.
The planner then chooses a complete `1 + 320*k` expert tier and grows Metal
storage in 321-expert slabs (about 0.529 GiB) instead of the generic 4 GiB slab.
On 16 and 24 GiB hosts, Qwen's SSD plan is additionally guarded: it requires
an affirmative normal-pressure signal at admission and before each prefill or
decode cache phase, including when the configured budget is unchanged, and
caps the cache at 3,521 experts (about 5.80 GiB for the published artifact).
The unchanged-budget check matters because later requests can still populate
lazy expert slabs. Hebrus Studio's watchdog remains a last-resort safety
boundary, not the cache-sizing mechanism.
DeepSeek retains its independently qualified resident/SSD planners. GLM retains
its independent SSD-only planner and schedule.

For this path, `resident` means complete tensor mapping, full-tensor Metal
kernels, and no DS4 expert-cache `pread`.  Metal residency requests are budget
hints; they do not prove that all file pages remained physically resident.  See
[`tests/qwen/README.md`](tests/qwen/README.md) for current fixture checks, the
exact v2 release identity, and model-backed resident/SSD qualification commands.
The complete release requirements remain in `QA_BEFORE_RELEASES.md`.

## Reference lanes

Do not use the GLM runtime line as the DeepSeek reference until the known
DeepSeek decode regression is fixed.

- DeepSeek gold: `andreaborio/main` at the recorded benchmark commit, no
  experimental environment overrides.
- GLM 5.2 gold: the dedicated GLM line, Metal + SSD, with only independently
  verified family-specific winners enabled.

The GLM gold profile is allowed to enable:

- indexed-prefill next-layer prepare;
- router-ahead advisory mode level 1;
- disabling redundant expert-miss readahead only for GLM.

It must not change the DeepSeek defaults.  Router install mode, QoS/subchunk,
single-command-buffer splitting, `F_NOCACHE`, virtual full layers, MTP, and
oversized GLM caches remain experimental or rejected because measurements were
neutral or negative.

## Gold gates

Tests that do not require a model download:

```sh
make build-isolation-test
make model-free-test
```

`make test` additionally runs model-backed tests and therefore requires a
qualified DeepSeek ExpertMajor v2 GGUF at `ds4flash.gguf` or `DS4_TEST_MODEL`.

For performance A/Bs:

1. Record build revision, machine, macOS, model path/hash, quant, requested and
   resolved residency, context, cache slots, and page-cache regime.
2. Run arms sequentially and interleaved; never run two huge model processes.
3. Resolve the automatic cache once, then pin the same expert count in both
   arms.  Invalidate the pair after `mlock` relief, slot-count drift, OOM,
   thermal mismatch, or different output.
4. DeepSeek must retain at least 95% at every frontier and at least 98%
   geometric-mean throughput against its gold commit.
5. GLM indexed prepare must be at least 2.0x its true OFF arm, with decode no
   worse than 5% and greedy output byte-identical.

Cold and warm results are separate measurements; neither may be presented as
the other.
