# SSD streaming, caches and AUTO memory policy

## 1. Two independent storage workloads

Qwen4Exp introduces two very different SSD access patterns:

| Store | Access unit | Requests/token | Reuse shape | Required strategy |
|---|---:|---:|---|---|
| routed experts | complete gate/up/down record, MiB-scale | up to 480 layer-expert selections | route locality across tokens/layers | resident low-bit payload or very large record cache; explicit pread for misses |
| PLE | fixed page containing width-160 rows | 16 logical rows once/token | hash/page collisions, prompt repetition | page cache, dedup/coalesce, async prefetch before layer 1 |

They have separate budgets, tables, leases, metrics and workers/priority. Sharing
only a safe low-level `pread` pool is allowed after priority and deadlock tests.
Do not let a large expert miss starve the latency-critical PLE wave needed at
layer 1.

“SSD streaming” does not mean KV-cache streaming. QSA KV, raw index keys and GDN
state remain live session memory in this plan.

## 2. Why expert SSD misses cannot be the primary 64 GB decode path

One expert has `4,915,200` parameters. At ideal 4 bits it is about 2.34 MiB.
There are 480 layer-expert selections/token. If all are cold, payload exceeds
1.09 GiB/token before metadata/copy. No consumer SSD can make that a fast decode
path. Therefore the M5 Pro 64 GB primary artifact must aim to keep routed experts
resident or attain an extremely high cache hit rate, likely through an
approximately two-bit routed codec. Forced-expert-SSD remains a correctness and
lower-memory control until measurements justify a claim.

The existing ExpertMajor v2 record layout remains useful: route-selected
gate/up/down components are contiguous, aligned and checksummed. Reuse its
generation ticket, lease, staged install, abort and telemetry concepts after
generalizing exact geometry to 48x512/top-10.

## 3. PLE random-I/O reality

At ideal four-bit payload, one width-160 row is about 80 bytes plus quant
metadata, but a filesystem read generally costs at least a page/cache operation.
Sixteen unrelated 4 KiB pages would be 64 KiB physical bytes/token. Decode is
then IOPS/latency-bound rather than bandwidth-bound. The design therefore:

1. uses fixed independently decodable pages containing multiple rows;
2. hashes the entire prompt batch early;
3. deduplicates identical pages and rows;
4. sorts/coalesces adjacent pages into bounded `pread` tasks;
5. begins reads before layer 0 and overlaps them with GPU work;
6. caches decoded or compressed pages under a strict budget;
7. separately records requested logical bytes and physical read bytes.

Do not rely on `mmap` page faults as the production scheduler. llama.cpp's lazy
tensor mode proves functional on-demand access, but OS fault timing cannot
provide Hebrus's ownership, budget, overlap or evidence contract.

## 4. PLE cache objects

```c
typedef enum { FREE, READING, READY, INFLIGHT_GPU } ds4_ple_page_state;

typedef struct {
    uint64_t page_id;
    uint64_t generation;
    ds4_ple_page_state state;
    uint32_t lease_count;
    uint64_t last_use_epoch;
    void *compressed_bytes;       /* fixed page stride */
    id<MTLBuffer> gpu_buffer;     /* optional decoded/cache tier */
    uint64_t inflight_seq;
    uint8_t digest_verified;
} ds4_ple_page_entry;

typedef struct {
    uint64_t generation;
    uint64_t owner_session;
    uint32_t request_count, unique_page_count, task_count;
    uint32_t *row_ids;
    ds4_ple_scatter *scatter;
    ds4_ple_page_entry **leases;
    ds4_pread_task *tasks;
    /* state owns every allocation until finish or abort */
} ds4_ple_io_ticket;
```

One cache entry is published `READY` only after a complete page read, digest
verification and buffer visibility. It cannot be evicted while reading, leased
or referenced by an uncompleted GPU sequence. Use a generation counter so an
old worker can never install into a recycled slot.

## 5. PLE planning algorithm

```text
input: token IDs for a tile, per-sequence two-token histories
for each token in logical order:
  compute exact 16 row IDs; update a private copy of history
  locate fixed page and row slot; append scatter(token, head, page, slot)
sort requests by page ID, preserving scatter associations
deduplicate page IDs
for each unique page:
  READY -> lease/hit
  READING same generation/store -> join/lease pending entry
  missing -> reserve cache entry without evicting leased/inflight entries
coalesce physically adjacent missing pages up to MAX_READ_BYTES/MAX_IOV
submit asynchronous pread wave with immutable task array
return ticket; do not publish private token history yet
```

Finish waits only when layer 1 needs the rows, verifies all entries, builds the
compact Metal scatter/descriptor buffer, and transfers leases to the command.
Commit publishes token history and releases leases after GPU completion. Abort
restores history, drains/cancels safe workers, invalidates failed entries and
releases every owned resource exactly once.

## 6. Cache replacement and priority

Use a bounded set-associative or clock/LRU approximation, not an unbounded hash
table. The key includes store/profile/digest and physical page ID. Candidate
victims must be READY, unleased and GPU-complete. Prefer retaining pages with
recent hits; sequential prompt prefetch must not evict the entire hot decode
set in one wave.

Two-tier candidate:

- compressed page cache in shared host/Metal-visible memory;
- optional decoded BF16/F16 row cache only if decode cost is material and budget
  permits.

Measure before enabling decoded caching; a four-bit row decoder may be cheaper
than doubling/quadrupling cache bytes. A page-cache budget change is
transactional and cannot shrink below live leases.

## 7. I/O scheduling

Bound worker count, bytes per wave, and in-flight staging. Priorities:

1. PLE pages on the critical path for the current layer-1 boundary;
2. missing experts needed by the current layer;
3. near-future PLE prompt tiles;
4. speculative/expert readahead.

Avoid spawning a thread per page. Use the existing bounded pread pool only after
making tickets store-agnostic and adding priority/fairness. Otherwise create a
small PLE pool. `pread` handles EINTR, short read and EOF explicitly. Task result
records syscall count, returned bytes and wall time; a digest failure is an I/O
failure, not a cache miss.

Coalescing must not read across a manifest extent or silently include a different
owner. Cap a coalesced task (candidate 1-4 MiB) and tune from cold-cache M5 data.
`F_RDADVISE` may supplement planned reads, never replace authoritative `pread`.

## 8. AUTO plan for M5 Pro 64 GB

AUTO is the release path. The provisional primary plan is:

- low-bit routed ExpertMajor payload resident if exact live admission passes;
- dense/shared/GR/GDN/QSA weights at a higher qualified codec;
- PLE quantized store file-backed and read only via bounded PLE cache;
- per-session GDN state and QSA/index caches sized to requested context;
- sufficient reserve for macOS, command/staging buffers and pressure changes;
- no swap before or during the measured run.

Do not assume “64 GB” is allocatable. Use `recommendedMaxWorkingSetSize`, host
free/inactive/purgeable accounting and repository reserve rules. Resolve once at
engine open and recheck guarded growth at phase changes. AUTO can select:

```text
Q4EXP_RESIDENT_ROUTED_PLE_STREAM
Q4EXP_HYBRID_EXPERT_CACHE_PLE_STREAM
UNSUPPORTED_FOR_REQUESTED_CONTEXT
```

Names are illustrative. Evidence prints the chosen plan and every byte owner.
The second plan is not promoted for M5 if decode is dominated by expert misses.

## 9. Context-dependent formulas

At BF16, QSA main KV is 6 GiB and raw index keys 0.75 GiB at 262,144 tokens.
An exact incremental pooled-key cache is a later production candidate and lowers
the index component to roughly 0.19 GiB while retaining at most three raw tail
keys per QSA layer/sequence.
Use exact checked formulas parameterized by requested context `C`:

```text
qsa_kv(C)    = 12 * C * 2 heads * 256 * 2(K,V) * cache_elem_bytes
qsa_index(C) = 12 * C * 128 * index_elem_bytes
gdn_state    = 36 * 48 * 128 * 128 * 4 + exact conv bytes
wide_tile(P) = P * 4 * 2560 * activation_bytes * live_buffer_count
routes(P)    = P * 10 * (sizeof(id)+sizeof(float)) + grouping metadata
qsa_work(P)  = bounded block scores/top IDs/gather tile, never P*C dense mask
ple_work(P)  = P*16 row descriptors + unique pages + bounded staging
```

Alignment, double buffers, snapshots and Metal allocation granularity are added
under their named owner. Print logical and allocated values. Context admission
must consider runtime graph plus static model/store caches together.

## 10. Phase budgets

Plan separately for:

- **decode:** tiny activation arena, full context state, hot routed/PLE caches;
- **micro prefill:** conservative physical tile for SSD/hybrid modes;
- **macro prefill:** larger tile, QSA and route grouping workspace, extra
  transactional snapshot/journal;
- **endpoint:** reduced tile if necessary, never reduced semantic context.

Cache growth is monotonic during a phase and rounded to a complete working set.
For experts the complete-route floor is 481 records. For PLE define a minimum
that covers the current tile's unique pages plus a hot decode reserve; if that
cannot fit, reduce physical prefill tile before failing. Never evict pages leased
by the current tile to admit the same tile.

## 11. Pressure and failure behavior

Before every material growth, snapshot host/device memory and require normal
pressure. Under pressure:

- stop readahead and decoded-page promotion;
- reduce future prefill tile/caches only at a safe phase boundary;
- retain current transaction resources until completion/abort;
- never silently change codec, context or output semantics;
- if the admitted minimum no longer fits, fail at a documented safe boundary
  with owner-by-owner byte diagnostics.

`ENOMEM`, short read, digest error, Metal allocation failure and cancellation
all execute the same ownership unwind discipline. Inject each failure in tests.

## 12. Required telemetry

Expert and PLE telemetry is separate and includes:

```text
budget/allocated/live/peak bytes; entries; hits/misses/hit rate;
requested logical rows/records; unique pages/experts;
pread syscalls/bytes/wall ms; coalesced tasks; short reads/EINTR/errors;
cache evictions and blocked victims; leases/inflight high-water;
prefetch submitted/completed/useful/late; exposed layer wait p50/p95;
decode versus prefill and cold versus warm counters;
per-layer expert misses; PLE physical amplification = pread/logical bytes;
AUTO plan, growth decisions, pressure snapshots and swap evidence.
```

Telemetry must not introduce a synchronization/readback in the normal hot path
when disabled. Counters used for release evidence are reset immediately before
the measured region.

## 13. Lower-memory/general-Metal policy

The fallback can use the generalized expert cache plus PLE page cache, but it is
not automatically a support tier. For each RAM/device tier, qualify an explicit
maximum context and artifact profile. If even the 481-record expert floor,
minimum PLE tile/cache, runtime state and reserve do not fit, fail at engine open.
Do not advertise “runs on Metal” based on a one-token forced-SSD smoke test.
