# DeepSeek DS4-native expert-major v2 GGUF

`ds4.expert_major.v2` is the generic, self-describing expert-major container
used for DeepSeek V4. It changes storage order only: no tensor is requantized,
and every non-routed tensor plus the complete GGUF metadata block is copied
byte-for-byte from the canonical source.

For each routed layer the payload is laid out as complete expert records:

```text
record(layer, expert) = gate | up | down
```

The manifest records the GGML type, dimensions, block geometry, component
offset, record size, and physical extent independently for every layer. This is
the reason DeepSeek does not reuse Qwen's fixed-geometry
`ds4.expert_major.v1`: Flash and PRO artifacts can use different routed quant
types or size classes without changing the format.

The canonical gate/up/down tensors are removed from the physical GGUF and
reconstructed as logical descriptors at load time. Their virtual offsets remain
stable cache and binding identities; all selected-expert I/O is translated to
the validated expert-major record. The native file therefore stores routed
weights once, not once in canonical order plus a second optimized copy.

> [!IMPORTANT]
> This is a DS4 GGUF extension. A loader that does not implement
> `ds4.expert_major.v2` cannot execute the file. Incompatible runtimes fail on
> the missing canonical routed tensors instead of silently reading interleaved
> bytes with the wrong geometry. Keep the canonical GGUF for llama.cpp, MLX,
> and other runtimes.

## Build, inspect, and verify

The converter is deterministic and writes a same-filesystem temporary output.
It preflights the complete output plus the requested reserve, hashes the source,
fsyncs the file, performs a full byte-level verification by default, and only
then atomically renames the result.

```sh
python3 gguf-tools/ds4-expert-major.py inspect CANONICAL.gguf

python3 gguf-tools/ds4-expert-major.py build \
  --reserve-bytes 2GiB \
  CANONICAL.gguf DEEPSEEK-DS4-EXPERT-MAJOR-V2.gguf

python3 gguf-tools/ds4-expert-major.py verify \
  CANONICAL.gguf DEEPSEEK-DS4-EXPERT-MAJOR-V2.gguf
```

`inspect` is read-only and reports the routed inventory and predicted native
size. `build` accepts only DeepSeek4 GGUF v3 inputs with complete gate/up/down
inventories and currently supported routed types (`IQ2_XXS`, `Q2_K`, and
`Q4_K`). `verify` checks all of the following:

- source identity against the SHA-256 stored in the manifest;
- byte-identical GGUF metadata and non-routed tensors;
- every manifest dimension, quant block, record, alignment, and extent;
- every routed expert component against its canonical source bytes;
- the complete expert-major payload SHA-256, including alignment padding.

The output is normally only alignment and directory overhead larger than the
source. The exact byte delta is printed before conversion by `inspect` and
after installation by `build`; do not publish an estimated size.

## Runtime contract

Native v2 files activate automatically and are mandatory single-layout
artifacts: there is no sidecar variable and no fallback to absent canonical
routed weights.

```sh
./ds4 --metal --ssd-streaming \
  -m DEEPSEEK-DS4-EXPERT-MAJOR-V2.gguf \
  --ctx 32768 --nothink -p 'Hello'
```

The first implementation deliberately keeps its backend boundary narrow:

- complete local Apple Metal models are supported in resident and SSD modes;
- canonical DeepSeek GGUFs retain the existing CPU, CUDA, ROCm, and distributed
  behavior unchanged;
- native v2 is rejected before inference on those other paths until each has a
  model-backed translator and performance qualification;
- SSD mode currently requires one routed record size class across layers,
  because its cache owns one slab geometry; resident mode accepts per-layer
  geometry.

This separation prevents an optimization for the measured M5 path from
degrading higher-memory machines or alternate accelerators. Canonical files do
not enter the v2 address translator or its native cache phases. The grouped
IQ2 kernel shared by canonical and native DeepSeek has its own exactness gate;
CPU, CUDA, ROCm, distributed, Qwen, and GLM execution are unchanged.

## Metal behavior

At startup DS4 validates the embedded header and checksummed manifest, binds all
logical layer identities, and then selects one of two consumers:

- Long SSD prefill maps one complete physical expert-major layer and feeds the
  existing grouped-ID kernels with the record stride. It never treats the
  virtual canonical descriptors as physical tensor ranges.
- SSD decode retains canonical cache keys but translates every exact component
  miss to the adjacent expert-major record. The production loader deliberately
  keeps gate, up, and down as three parallel tasks: on the measured Apple SSD,
  one 6.75 MiB synchronous `pread` reduced queue depth and lost throughput.
  `DS4_METAL_ENABLE_COALESCED_EXPERT_RECORD_PREAD=1` is a diagnostic for other
  storage, not the default.
- Resident mode maps one complete expert-major layer per read-only Metal buffer
  and passes its record size as the expert stride to the existing ID kernels.
  There is no token-time repack or host expert lookup.

Native AUTO is phase-aware for long prefill. It contracts only the otherwise
unused decode cache to the complete per-token correctness floor, maps the
physical layer, then restores the previously planned decode budget lazily.
Explicit cache budgets, canonical GGUFs, short prompts, other model families,
and other backends do not enter this policy. Set
`DS4_METAL_DISABLE_DEEPSEEK_PHASE_CACHE=1` only for an A/B diagnostic.

### Hotlist admission priority

The preload list is an ordering prior, not a permanent LFU score. By default,
every built-in or file-backed hotlist entry starts at priority one, so a route
selected by the live request immediately outranks an unused seed. The same top
entries are still selected for preload, and their relative order is preserved
within each layer; after that, live route frequency and the LRU tie-breaker own
eviction.

The policy is explicit and reversible:

```sh
# Production default. Omitting the variable is equivalent.
DS4_METAL_STREAMING_EXPERT_HOTLIST_PRIORITY=adaptive ./ds4 ...

# Historical behavior: built-in rank or the file's raw hit count.
DS4_METAL_STREAMING_EXPERT_HOTLIST_PRIORITY=legacy ./ds4 ...

# Give every preload entry a fixed positive initial priority.
DS4_METAL_STREAMING_EXPERT_HOTLIST_PRIORITY=8 ./ds4 ...
```

With the adaptive default, a file hotlist's row order determines preload rank;
its hit-count column is retained for provenance but does not become a long-lived
cache score. Use `legacy` only to reproduce an older benchmark or a deliberately
sticky domain profile. Invalid values fail engine startup. This changes cache
admission and eviction only; routing, weights, and logits are unchanged.

On the measured 64 GiB tier, normal memory pressure also makes AUTO insensitive
to warm GGUF page-cache order: file-backed inactive pages receive full credit,
and the current-pressure reserve is fixed at 2 GiB including its pressure
margin. The independent Metal envelope remains capped at 9/16 of the device's
recommended working set, leaving roughly 14 GiB below that limit after the
Flash static and runtime working sets. Elevated pressure, hosts below 64 GiB or
at least 96 GiB, and every non-DeepSeek policy retain their prior accounting.

For Flash's `IQ2_XXS` gate/up plus `Q2_K` down geometry, grouped prefill uses a
paired gate/up kernel that writes weighted SwiGLU rows directly in F16. The
kernel has a threadgroup barrier at every K-block boundary: without it, a fast
simdgroup could replace the shared RHS tile while another was still consuming
it. The paired and split paths are byte-identical at the model frontier in the
recorded 128- and 768-token controls.

### Selected-address expert schedule

Native Flash SSD prefill has one additional, deliberately narrow auto-policy.
For batches from 256 through 760 tokens, the selected-address path orders
routes expert-major before dispatching the paired IQ2 gate/up kernel. Mirrored
M5 Pro measurements were neutral at 128 tokens, positive at 256 and 512, and
positive in the isolated 760-token gate/up stage. At 768 tokens the existing
`mm_id` path takes over, so the schedule stops before that crossover.

Auto-selection requires every condition below:

- Apple Metal SSD streaming with a native `ds4.expert_major.v2` store;
- normal inference rather than quality mode or graph dumping;
- exactly 256 routed experts with top-6 selection;
- `IQ2_XXS` gate/up and `Q2_K` down tensors;
- `n_tokens >= 256 && n_tokens <= 760`.

The grouped pipeline is resolved lazily on its first eligible dispatch.
Canonical DeepSeek files, resident execution on higher-memory Macs, decode,
other context shapes, other quantizations, Qwen, GLM, CPU, CUDA, ROCm, and
distributed execution therefore keep both their existing schedule and startup
cost. The disable switch is the complete production rollback:

```sh
DS4_METAL_DISABLE_DEEPSEEK_EXPERT_GROUP_PREFILL=1 ./ds4 ...
```

Automatic selection also probes the backend's current selected-address
eligibility. Existing fallback/debug controls such as disabling streaming
selected-address, the expert address table, or routed pair fusion therefore
continue to select their previous path instead of turning an optional schedule
into an inference error. Explicit GROUP or TILE enable flags remain strict: an
A/B fails if the requested schedule was not actually encoded.

`DS4_METAL_ENABLE_DEEPSEEK_EXPERT_GROUP_PREFILL=1` can request the grouped
schedule for shorter eligible SSD batches, or for a canonical DeepSeek SSD
artifact, during an A/B. This explicit diagnostic does not broaden the
native-only automatic policy.

The campaign also keeps four independent research paths without enabling them
by default:

- `DS4_METAL_ENABLE_DEEPSEEK_PREFILL_IO_OVERLAP=1` starts selected-expert I/O
  after router readback so it can overlap the shared-expert encode. It lost to
  the simple grouped schedule on the measured unified-memory system.
- `DS4_METAL_ENABLE_BALANCED_EXPERT_RECORD_PREAD=1` balances record tasks. It
  recovered only a small fraction of the overlap loss and is not useful alone
  in the measured cohort.
- `DS4_METAL_ENABLE_DEEPSEEK_EXPERT_TILE_PREFILL=1` enables the lazy route-tile
  kernels. All route-2/route-4, row-4/row-8, staged, and decoded-shared variants
  were exact but slower because barriers and occupancy outweighed reuse.
- `DS4_METAL_ENABLE_MOE_MM_ID_PAIR_NR16=1` selects the narrower long-prefill
  pair tile. Its final 2K mirrored mean was 1.11% below NR32, so NR32 remains
  the default.

These switches are intentionally composable. They preserve the ability to
repeat the full stack on a different memory tier or SSD and then ablate one
piece at a time, without adding work to production execution that does not
request them. Exact binary identities, paired results, zero-swap evidence, and
frontier-logit hashes are recorded in the dated benchmark document.

Only the small manifest is hashed at startup. The multi-gigabyte payload digest
is an offline publication gate so model startup does not acquire a mandatory
full-file read.

## Long requests and client timeouts

The server does not impose an inference deadline. For streaming chat,
Responses, and Anthropic requests it sends SSE headers as soon as prefill
progress begins and emits an SSE comment keepalive every five seconds while
prefill is running. This preserves existing model and machine behavior while
preventing ordinary HTTP idle timeouts from mistaking slow local prefill for a
dead request.

The client must still use streaming and configure a stream-idle timeout longer
than its network path permits. A non-streaming proxy cannot receive SSE
keepalives, so its own request timeout remains external to DS4. Do not raise
the server's socket-stall limits to compensate: those limits detect a client
that is no longer reading, not a slow model.

## Promotion gate

Support in source does not make an artifact release-qualified. Before a native
DeepSeek file replaces the canonical recommendation, the same canonical/native
pair must pass:

1. full converter verification and source/output SHA-256 recording;
2. byte-identical frontier logits and greedy decode evidence;
3. the official-continuation scorer against the same manifest;
4. alternating throughput measurements at 2K, 8K, and 16K or longer context;
5. zero new swapout plus memory-pressure and page-fault accounting;
6. a cold/warm repetition on every advertised RAM tier.

Results belong in a dated file under `docs/benchmarks/`. Until those gates are
recorded, v2 remains an experimental artifact format and the canonical GGUF is
the compatibility reference.

The first M5 Pro tranche, including the queue-depth ablation for full-record
I/O, is in
[`benchmarks/2026-07-17-deepseek-native-expert-major.md`](benchmarks/2026-07-17-deepseek-native-expert-major.md).
The route-locality and adaptive-priority evidence is in
[`benchmarks/2026-07-17-deepseek-qwen-transfer-audit.md`](benchmarks/2026-07-17-deepseek-qwen-transfer-audit.md).
