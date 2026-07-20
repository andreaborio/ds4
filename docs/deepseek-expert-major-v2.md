# DeepSeek DS4-native ExpertMajor v2 GGUF

`ds4.expert_major.v2` is the self-describing routed-expert container used by
DeepSeek V4, GLM 5.2, and Qwen3.6 in this fork. For DeepSeek it changes storage
order only: no tensor is requantized, and every non-routed tensor plus GGUF
metadata is copied byte-for-byte from the canonical converter input.

Each routed layer is stored as adjacent expert records:

```text
record(layer, expert) = gate | up | down
```

The manifest records family identity, GGML type, dimensions, quant block
geometry, component offsets, record size, and physical extent independently for
every layer. This supports Flash and PRO artifacts with different routed quant
types or record sizes without changing the format.

The physical canonical routed tensors are absent from the output. At load time
DS4 reconstructs their names as logical graph and cache identities, then
translates every physical map and selected-expert read through the validated
manifest. The file therefore stores the routed weights once.

> [!IMPORTANT]
> Current inference is ExpertMajor v2-only on Apple Metal. Canonical DeepSeek
> GGUFs are accepted by the offline converter, not by the inference runtime.
> There is no v1, sidecar, canonical, CPU, CUDA, ROCm, or distributed fallback.

## Build, inspect, and verify

The converter writes a same-filesystem temporary output, preflights the output
plus reserve, hashes the source, performs full verification by default, fsyncs
the result, and then atomically renames it.

```sh
python3 gguf-tools/ds4-expert-major.py inspect CANONICAL.gguf
python3 gguf-tools/ds4-expert-major.py build \
  CANONICAL.gguf DEEPSEEK-DS4-EXPERT-MAJOR-V2.gguf
python3 gguf-tools/ds4-expert-major.py verify \
  CANONICAL.gguf DEEPSEEK-DS4-EXPERT-MAJOR-V2.gguf
```

`inspect` is read-only and reports the routed inventory, record classes, and
predicted native size. `build` accepts complete `deepseek4`, `glm-dsa`, and
`qwen35moe` GGUF v3 sources and supports routed Q2_K, Q4_K, Q5_K, Q6_K, and
IQ2_XXS records. `verify` checks:

- source identity against the SHA-256 in the manifest;
- byte-identical metadata and non-routed tensors;
- every routed component against its canonical source bytes;
- every family, layer, dimension, type, block, record, alignment, and extent;
- the complete ExpertMajor payload SHA-256, including padding.

Publication must never use `--skip-verify`. The exact size and hashes printed by
the completed build are authoritative; do not publish estimates.

## Start the model

Native v2 activates automatically:

```sh
make -j8
./ds4 \
  -m /absolute/path/to/DEEPSEEK-DS4-EXPERT-MAJOR-V2.gguf \
  --ctx 32768
```

For the API, replace `./ds4` with `./ds4-server`. AUTO chooses resident mapping
when the complete model safely fits or SSD streaming otherwise. The release
command does not need an ExpertMajor variable, sidecar path, explicit backend,
cache budget, preload policy, or residency flag.

The runtime validates the embedded header and manifest before backend setup and
checks that family ID, layer count, expert count, selected-expert count, logical
tensor inventory, and active DeepSeek shape profile agree. A failure is fatal;
the artifact contains no canonical routed copy to fall back to.

## Prefill and decode memory flow

Resident mode maps non-routed GGUF spans and one complete physical
ExpertMajor layer per read-only Metal buffer. Kernels receive the manifest
record size as the expert stride. There is no token-time repack, CPU expert
lookup, or duplicate routed mapping.

SSD prefill and decode deliberately use different consumers:

- Long prefill maps a complete physical layer and feeds grouped selected-ID
  kernels with the record stride. It never treats virtual canonical offsets as
  physical tensor ranges.
- Decode keeps stable logical cache keys, then translates each miss to the
  adjacent expert-major components. DeepSeek retains three parallel component
  reads because a single synchronous 6.75 MiB record read reduced queue depth
  and was slower on the measured Apple storage.
- The phase-aware AUTO policy may contract an otherwise unused decode cache to
  the per-token correctness floor during long prefill, then restore the planned
  decode budget lazily.

Only the small manifest is hashed at startup. The multi-gigabyte payload digest
is an offline publication gate, avoiding a mandatory full-model startup read.

## DeepSeek-specific Metal schedules

For Flash `IQ2_XXS` gate/up plus `Q2_K` down geometry, grouped prefill uses the
paired gate/up Metal kernel and writes weighted SwiGLU rows directly in F16.
The paired and split paths are byte-identical at the recorded 128- and
768-token frontiers.

Native Flash SSD prefill also selects the expert-major selected-address
schedule for eligible 256-760 token batches. At 768 tokens the existing
`mm_id` path wins and remains the automatic choice. These are compute and
scheduling optimizations above v2; the file layout by itself is not advertised
as their speedup.

The measured production decode schedule keeps gate, up, and down page-ins as
parallel tasks. Full-record coalescing, I/O overlap, balanced record tasks,
route-tile kernels, and narrower NR16 prefill are retained only in historical
benchmark evidence because they did not beat the selected defaults on the M5
Pro.

## Qualification and publication

Every DeepSeek Flash or PRO v2 artifact must independently pass:

1. complete converter verification and exact source/output SHA-256 recording;
2. byte-identical frontier or decode evidence against its qualified source;
3. the official-continuation quality scorer where applicable;
4. alternating throughput runs at the contexts claimed by the release;
5. zero new swapout plus memory-pressure and page-fault accounting;
6. cold/warm repetition on every advertised memory tier.

A failed or incomplete artifact is rejected; DS4 does not broaden runtime
compatibility by falling back to a canonical file. Results belong in a dated
document under `docs/benchmarks/`.

The first M5 Pro native-v2 SSD tranche, including the full-record I/O ablation,
is in
[`benchmarks/2026-07-17-deepseek-native-expert-major.md`](benchmarks/2026-07-17-deepseek-native-expert-major.md).
The route-locality and adaptive-priority study is in
[`benchmarks/2026-07-17-deepseek-qwen-transfer-audit.md`](benchmarks/2026-07-17-deepseek-qwen-transfer-audit.md).
