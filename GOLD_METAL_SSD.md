# Gold Metal + SSD contract

This fork treats Apple Metal as the production default on macOS.  The default
runtime policy is `AUTO` residency: use the faster fully-resident path when the
model and requested context fit a conservative Metal budget, otherwise select
SSD streaming automatically.

## User-facing contract

```sh
make                         # namespaced Metal build; publishes ./ds4*
./ds4 -m MODEL.gguf          # Metal + AUTO residency
./ds4 -m MODEL.gguf --resident
./ds4 -m MODEL.gguf --ssd-streaming
./ds4 --build-info
```

`--resident` (also `--no-ssd-streaming`) and `--ssd-streaming` are explicit
overrides.  Cache, cold-streaming, or preload options imply SSD mode and are
rejected when combined with `--resident`.

Alternative backends remain explicit.  On macOS, `make cpu` writes only to
`build/cpu-<arch>/bin/` and never replaces the Metal root binaries.  Linux
keeps the explicit `cuda-spark`, `cuda-generic`, `cuda CUDA_ARCH=...`, `rocm`,
and `cpu` targets.

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
`recommendedMaxWorkingSetSize`:

```text
required = model tensors
         + optional MTP file
         + context / KV / scratch estimate
         + max(20% of recommended working set, 2 GiB)

budget   = recommended working set - explicitly simulated used memory
```

`required <= budget` resolves to resident; otherwise it resolves to SSD.  The
resolved mode, reason, components, and cache plan are printed at startup.
Outside Metal, AUTO preserves the existing resident behavior until an
equivalent backend-specific capacity planner is validated.

If Metal cannot report a working-set recommendation, AUTO fails safely unless
the user supplies an explicit SSD cache budget.  SSD + MTP remains unsupported.

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

`make test` additionally runs model-backed tests and therefore requires the
supported DeepSeek GGUF at `ds4flash.gguf` or `DS4_TEST_MODEL`.

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
