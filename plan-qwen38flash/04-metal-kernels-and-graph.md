# Metal kernels and execution graph

## 1. Implementation order: correct, bounded, fused

For every primitive, deliver in this order:

1. independent float32 CPU/reference routine and golden fixture;
2. simple Metal kernel with parameterized shapes and explicit bounds;
3. full-graph resident correctness at tiny context;
4. batch/prefill implementation with state equivalence tests;
5. M5-specialized or fused path selected only by exact profile/capability;
6. general-Metal fallback retained as a correctness and portability control.

A fast kernel without an independent oracle must not enter the model graph.
Compile-time constants are welcome in specialized functions; hidden assumptions
in supposedly generic functions are not.

## 2. Precision contract

| Operation | Minimum semantic precision |
|---|---|
| zero-centered RMS reductions | float32 |
| GDN `A_log`, softplus, sigmoid, decay and state update | float32 controls/state |
| router logits, full softmax, top-10 weights/renorm | float32 |
| QSA index dot/ReLU/sum and top-k scores | float32 |
| PLE 64-bit hashes | exact unsigned integer |
| Q/K normalization and RoPE | float32 arithmetic; cache type separately qualified |
| accumulators for quantized matmul/MoE | float32 unless oracle promotes a tested mixed path |
| GR gates and injection scalars | float32 |

BF16/F16 activations may reduce traffic after measured qualification. Never
silently narrow persistent GDN state or routing scores.

## 3. Buffer layout

All multidimensional layouts are named in header comments and statically checked
against Metal argument structs. Recommended logical layouts:

```text
wide residual      [token][branch=4][hidden=2560]
GR normalized      same (may alias bounded scratch, never persistent input)
GDN recurrent      [layer_gdn][value_head=48][key_dim=128][value_dim=128]
GDN conv history   [layer][qkv_channel][history_slot]
QSA KV             [layer_qsa][logical_slot][kv_head=2][head_dim=256]
QSA raw index K    [layer_qsa][logical_slot][128]
router logits      [token][512]
selected route     [token][10] IDs plus [token][10] F32 weights
PLE row IDs        [token][16]
PLE gathered       [token][16][160] or fused page/row descriptors
```

Prefer token-major activation layout for layer-major prefill so dense operations
are contiguous. Persistent QSA caches need slot maps and absolute positions;
their physical layout may be tiled for Metal but may not define semantics.
The correctness path stores every raw index key. A later qualified path stores
one pooled+normalized+rotated key per completed four-token group and only a raw
tail of at most three keys, reducing index-cache storage/traffic exactly.

## 4. GR kernels

### 4.1 Correctness path

Start with separate kernels for four ZCRMSNorms, low-rank down projection,
`/4+SiLU`, up projection+sigmoid, gated branch mean, the wide-to-four injection
projection with `/4` and sigmoid, and broadcast writes. This exposes each fixture.

### 4.2 M5 fused path

GR is bandwidth-bound: a BF16 wide residual is 20 KiB/token and is read/written
twice per layer. Fuse normalization + gate application + branch reduction when
the low-rank projection permits a tiled reduction. Calculate four injection
scalars from that normalized wide input before the sub-block, retain only those
scalars, and fuse the later branch updates so block output is not reread.

```metal
// Sketch only. injection_weight was computed as
// 2*sigmoid(W_inject(ZCRMSNorm(wide))/4) before the sub-block.
kernel void q4exp_gr_apply_injection_f32(
    device float *wide,
    device const float *block,
    device const float *injection_weight, // [token][4]
    constant gr_args &a,
    uint branch [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
  float scale = injection_weight[a.token*4 + branch];
  for (uint i = tid; i < 2560; i += a.tg_size)
    wide[(a.token*4+branch)*2560+i] += scale*block[a.token*2560+i];
}
```

The sketch intentionally uses float storage. Introduce BF16 load/store only in
a separately named variant. Check threadgroup memory against device limits and
avoid a global barrier between branches.

## 5. GDN kernels

### 5.1 Decode

Reuse parameterized Qwen3.6 causal-conv and gated-delta concepts, but add a
profile-qualified 16-key/48-value mapping. The reference mapping must decide
whether repeated heads use `v_head % 16` or a grouped repeat; encode it once in
a pure helper and test all 48 heads. Do not infer it from Qwen3.6's 32 heads.

Decode pipeline:

1. quant/dense projection QKV and controls;
2. fused depthwise conv step + SiLU, updating private conv history;
3. float32 controls (`decay`, `beta`);
4. one threadgroup/SIMDgroup per value head updates a 128x128 matrix tile;
5. normalize Q/K and compute recurrent read;
6. conventional weighted RMSNorm + sigmoid output gate;
7. output projection and GR injection.

Avoid allocating Q/K replicated to 48 heads. Map the 16-head source during the
state kernel. Persistent state writes belong to the private transaction until
command completion.

### 5.2 Prefill

First implement an exact sequential scan within bounded token tiles. Then add a
parallel associative/chunk scan only if it reproduces end state and every
requested output across tile sizes. Test token counts 1,2,3,4,63,64,65,2047,
2048,2049 and irregular final tiles. State equivalence at the last token is
necessary but insufficient: compare all layer outputs on small fixtures.

## 6. Exact 512/top-10 routing and MoE

### 6.1 Router

One token's 512 F32 logits fit a bounded Metal reduction. Use max subtraction,
float32 exponent/sum, then deterministic top-10. A simple reference-friendly
selection repeatedly finds `(prob, -expert_id)` maxima; an optimized path may
maintain per-thread sorted ten-element lists and merge them.

```text
tie rule: greater probability wins; equal bitwise probability -> smaller ID
postcondition: IDs unique, ordered by rule, finite nonnegative weights,
               abs(sum(weights)-1) <= tolerance
```

Tests force equal logits, NaN/Inf rejection behavior, extremely separated
logits and IDs across thread partitions. The production artifact should never
produce NaN; fail the token rather than selecting undefined experts.

### 6.2 Route grouping

For prefill, build `(expert_id, token_id, rank, weight)` records, stable-sort or
histogram by 512 experts, and dispatch contiguous expert runs. Retain a reverse
mapping or write weighted outputs directly to token accumulators with a race-
free plan. Do not allocate `tokens*512` activations.

Decode top-10 should issue at most ten expert operations but deduplicate
physical records across a multi-token batch. Route ID/weight buffers are owned
until all expert commands finish; CPU readback used for SSD planning must be one
bounded synchronized copy, then immutable.

### 6.3 Expert kernels

Keep gate+up paired/fused so input and quant metadata are reused, apply SiLU and
elementwise product, then down projection. Sum ten weighted results in a new
kernel; never call the existing `sum8` with overrun or split 8+2 in an order
that changes tolerance without tests. Shared expert can overlap routed SSD I/O.

M5 candidate: group tokens by expert, use matrix-matrix/TensorOps for prompt
tiles with enough rows, and quantized matrix-vector for decode. Selector inputs
are exact codec, dimensions, token count, alignment and device capability.

## 7. QSA indexer

### 7.1 State and logical grouping

Append raw width-128 keys using exactly the main KV cache's logical slot. A
kernel builds complete group descriptors from absolute positions rather than
physical adjacency. Reject/mask groups crossing a sequence boundary, hole or
nonconsecutive position.

After raw-cache parity, complete groups incrementally: on the fourth committed
consecutive key, mean in float32, zero-centered-normalize, apply RoPE at the
group's first position, append one pooled key and clear the private raw tail.
Rewind/fork/copy must update pooled groups and tail in the same transaction as
main KV. Compare selected IDs to the full raw-cache path at all boundaries.

### 7.2 Score/top-block kernel

For decode, calculate up to `ceil(context/4)` scores, then select 512. Avoid a
full sort: per-thread local top lists plus hierarchical merge, or radix/select,
but start with a deterministic bounded heap/control path. Workspace grows with
block count, not `context^2`.

For prefill, queries at different positions have different causal candidate
sets. Tile queries and blocks; compute scores, update each query's top-512, then
discard score tiles. Incomplete tail tokens are not candidates as a pooled
block and are appended directly.

Postconditions per query:

- no future group/token;
- no duplicate complete group;
- selected count is `min(512, complete_visible_groups)`;
- expanded IDs are logical causal positions and group-contiguous;
- tail contains exactly the visible incomplete suffix;
- total width `<=2051`.

### 7.3 Sparse attention kernel

Decode consumes selected logical token IDs and resolves them through cache slot
metadata. Fetch only their two K/V heads, compute online softmax over at most
2051 keys for 24 queries, then gate/output. For prefill, batch adjacent selected
blocks into indirect four-row reads. A gather-to-contiguous scratch path is a
good first Metal implementation because it bounds attention compute and makes
I/O measurable; later fuse gather+attention if profiling shows scratch traffic.

Do not copy llama.cpp's sparse mask mechanically: at the researched revision it
uses selected IDs to form a mask around a generic attention builder, which is
not proof that every backend avoids dense work.

## 8. PLE kernels and graph overlap

### 8.1 CPU hash planner

Hashing 16 addresses/token is cheap and exact on CPU. Plan input batches before
GPU layer 0: compute row IDs, map to pages, deduplicate pages, retain token/head
scatter destinations, and submit async reads. A Metal hash kernel is optional
only after it proves identical unsigned wraparound and reduces wall time.

### 8.2 Page decode/gather

The I/O subsystem returns leased page buffers and a compact descriptor per
requested row: `(page_address, slot, token, head)`. Metal decodes width-160 rows
directly into `[token][16][160]`, or fuses decode into the PLE key/value
projection if codec and alignment allow. Never expand the entire 25+ GB table.

### 8.3 PLE compute

Implement projections using existing dense primitives, then model-specific
group norm, signed-square-root gate, four-branch broadcast and depthwise
dilated-conv. Conv history semantics get a standalone test. PLE addition happens
before layer index 1 GR read and exactly once per token.

### 8.4 Timeline

```text
CPU: tokenize -> hash rows -> dedup/page plan -> async pread ........ wait
GPU: embedding + layer 0 GDN/GR ................................. | PLE compute
CPU/GPU: install leased pages ------------------------------------^
GPU: inject PLE -> layer 1 -> ...
```

If PLE I/O is not ready after layer 0, record the exposed wait separately. An
“overlap enabled” evidence line requires a real pending generation and nonzero
overlapped interval, not only an enabled flag.

## 9. Command lifecycle and failure semantics

One token/tile transaction owns:

- private GDN/conv and PLE-history updates;
- QSA append reservations for KV and raw index key;
- expert and PLE I/O tickets/leases;
- Metal command buffers/events and private logits.

Success waits for required I/O and GPU completion, checks status, advances the
single frontier and publishes logits/checkpoint once. Failure aborts tickets,
waits or cancels workers safely, releases leases, invalidates reserved cache
slots, restores/journals persistent states, discards logits, and preserves the
old public frontier. Test a forced failure after every numbered stage.

## 10. Device selectors

Primary specialized selector: M5 Pro, 64 GB system, supported Metal feature set,
normal pressure and validated alignments. Do not select solely on a marketing
string; query relevant capabilities and `recommendedMaxWorkingSetSize`.

General Metal fallback uses conservative threadgroup sizes and ordinary buffer
operations. It may be slower but must be correct. Older devices/lower memory
fall into guarded SSD only if the memory plan is admissible; otherwise engine
open reports required/available bytes and fails before model execution.

## 11. Kernel evidence counters

Expose at minimum: dispatch counts/time per GR/GDN/QSA/router/MoE/PLE category;
QSA candidate/selected/tail/gather bytes; GDN decode versus scan path; route
unique experts and group sizes; PLE requested/unique/hit/miss pages and exposed
wait; quant codec path; fallback/specialized selector reasons; command failures
and rollback count. Counters are diagnostic and resettable, with overflow-safe
accumulation.
