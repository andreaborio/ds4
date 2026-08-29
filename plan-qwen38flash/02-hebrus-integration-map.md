# Hebrus integration map and reuse boundaries

## 1. Baseline architecture

The implementation baseline is clean `main` at `7600fe1`. The archived
`feat/kv-cache-encryption` work and uncommitted ExpertMajor v3 experiments are
not inputs. Production remains Apple Metal plus an embedded ExpertMajor v2
store. Read `docs/contracts/RUNTIME_SUPPORT.md`, ADRs 0001/0002/0003/0004/0006,
`docs/qwen-expert-major-store.md` and `GOLD_METAL_SSD.md` before editing.

Qwen3.6 is implemented principally in:

| Concern | Current location | Qwen4Exp rule |
|---|---|---|
| immutable dimensions/reference | `ds4_qwen.[ch]`, `ds4_qwen_ref.[ch]` | add new descriptor/reference; do not replace constants |
| family, loader, graph/session | `ds4.c` | add exact family and partition new graph in `.inc` |
| Metal host/backend | `ds4_metal.m`, `ds4_gpu.h` | add profile-qualified APIs and selectors |
| kernels | `metal/qwen35.metal`, `metal/moe.metal` | generic correctness first; new Qwen4Exp source for new math |
| expert store | `ds4_expert_store.[ch]` | new family/profile, retain v2 container |
| converter | `gguf-tools/ds4-expert-major.py` | make shapes descriptor-driven |
| SSD policy | `ds4_ssd.[ch]`, Qwen portions of `ds4.c`/`ds4_metal.m` | generalize geometry; add separate PLE owner/cache |
| tokenizer/chat | Qwen functions in `ds4.c`, public enum in `ds4.h` | add pinned tokenizer/template without changing Qwen3.6 |
| serving | `ds4_server.c` | new alias and capability gates after core qualification |

The giant translation units are a merge and review risk. New graph code should
be textually included from `runtime/ds4_qwen4exp_graph.inc`; new Objective-C
host plumbing from `runtime/ds4_metal_qwen4exp.inc`. They inherit the including
file's static scope and are not compiled independently. Add them to code maps and
line-count budgets.

## 2. Family and immutable profile

Add a distinct family such as `DS4_MODEL_FAMILY_QWEN4_EXP_FLASH_NEXT`; do not
call it `QWEN35_MOE`. Add a model/profile ID that is included in cache/session/
artifact identity. Separate common operations from exact profiles:

```c
typedef struct {
    ds4_model_family family;
    uint32_t profile_id;
    const char *gguf_arch;
    uint32_t n_layer, n_embd, n_vocab, max_context;
    uint32_t n_q_head, n_kv_head, head_dim, rope_dim;
    uint32_t n_gdn_layer, n_qsa_layer;
    uint32_t n_expert, top_k, routed_ff, shared_ff;
    uint32_t residual_streams, gr_rank;
    uint32_t qsa_budget, qsa_group;
    uint32_t ple_rows, ple_row_width, ple_head_count;
    uint64_t tokenizer_hash, template_hash, tensor_inventory_hash;
} ds4_qwen_profile;
```

The actual descriptor also needs all GDN/indexer/PLE fields and exact layer-type
array. Store no pointer to unvalidated GGUF memory. A function returning the
profile first matches the exact architecture string, then validates every
closed field. There is no “closest” profile.

Required family touchpoints in `ds4.c` include the enum near the top,
`model_family_from_metadata`, metadata summary, validator dispatch, engine-open
dispatch, option validation, residency resolution, session creation, sync/
rollback and public capability predicates. Search every `qwen35` occurrence;
classify it as exact-profile behavior or family-common behavior before changing
it.

## 3. Weights and tensor binding

Create separate structs; do not extend `ds4_qwen35_layer_weights` until it is a
union of unrelated optional fields:

```c
typedef enum { DS4_Q4EXP_GDN, DS4_Q4EXP_QSA } ds4_q4exp_mixer;

typedef struct {
    ds4_q4exp_mixer mixer;
    ds4_tensor *gr_attn_norm, *gr_attn_down, *gr_attn_up;
    ds4_tensor *gr_attn_inject;
    ds4_tensor *gr_moe_norm, *gr_moe_down, *gr_moe_up;
    ds4_tensor *gr_moe_inject;
    /* union-like GDN or QSA tensors, all exact */
    ds4_tensor *router, *expert_gate, *expert_up, *expert_down;
    ds4_tensor *shared_gate, *shared_up, *shared_down, *shared_out;
} ds4_q4exp_layer_weights;

typedef struct {
    ds4_tensor *token_embd, *output;
    ds4_q4exp_layer_weights layer[48];
    /* PLE projections/norm/conv; PLE payload is a store handle, not tensor */
    ds4_tensor *final_gr_norm, *final_gr_down, *final_gr_up;
} ds4_q4exp_weights;
```

Use a generated or checked tensor-inventory table containing exact name, rank,
dimensions, accepted physical codecs, role and ownership. Binding does one
lookup per expected identity, marks it consumed, and finally rejects missing or
unconsumed model tensors. Optional MTP/vision tensors are controlled by artifact
profile, never ignored opportunistically.

## 4. Qwen3.6 primitives: what can and cannot be reused

Safe candidates after shape-parameter validation:

- quantized dense matrix-vector/matrix-matrix primitives;
- affine4 G64 decode and fused gate/up SwiGLU in `metal/moe.metal`;
- generic allocation, command-buffer and event ownership machinery;
- exact selected-ID ownership, generation tickets, leases and transactional
  staging concepts in SSD expert streaming;
- embedding lookup once physical type and vocabulary extents match;
- partial RoPE arithmetic once head count, dimensions and interleaving are
  arguments rather than hidden constants.

Must be new or explicitly generalized:

- Qwen3.6 top-8 router (hardcoded 256 experts/top-8) to exact 512/top-10;
- routed MoE buffers represented as two groups of four;
- GDN head mapping: Qwen3.6 has 16 key/32 value heads; Qwen4Exp has 16/48;
- GQA selectors hardcoded for 16Q/2KV versus 24Q/2KV;
- four-stream GR read/write and final mixer;
- QSA indexer, its mirrored raw-key state, top-block selection and indirect KV;
- PLE hashing, storage, gather, gating and dilated convolution;
- static memory constants and cache floor based on 40 layers/top-8;
- paired-read classifier restricted to family Qwen3.6 and 40x256;
- tokenizer/template/model alias and valid-vocabulary gates.

If a generalized function requires more than two exact-profile branches, use a
profile descriptor plus a selector. Retain specialized Qwen3.6 kernels and call
them only for its exact profile.

## 5. Graph state ownership

Define one `ds4_q4exp_graph` with explicit owners:

| State | Owner | Lifetime | Reset/copy/rollback |
|---|---|---|---|
| four residual streams | current command/arena | token or prefill tile | private until commit |
| GDN recurrence + conv | session graph | sequence | snapshot or transactional replay |
| QSA main K/V | session graph | sequence/context | frontier controls visibility |
| QSA raw index keys | session graph | same slots as K/V | identical cell operations |
| PLE token history | session graph | sequence | two IDs + valid count |
| PLE dilated-conv history | session graph | sequence | nine prior positions |
| PLE page cache | engine/store | model lifetime | leases protect in-flight rows |
| expert cache | Metal engine/store | model lifetime | existing lease/ticket rules |
| public logits/checkpoint | session | committed frontier | publish atomically |

The invariant remains `checkpoint.len == committed graph frontier`. A command
failure, I/O failure or cancellation cannot publish partially advanced state.
QSA KV and index cache always share logical sequence, absolute position and slot
mapping; do not allow one to compact independently.

## 6. Runtime byte accounting

Replace copied constants with checked formulae, then freeze measured/aligned
totals in tests. Owners include:

```text
static model bytes:
  non-routed dense mapped/resident spans
  embedded ExpertMajor store mapped/resident/cache
  PLE store file mapping metadata + page cache

per engine:
  Metal pipelines, quant tables, expert/PLE address tables
  async worker queues and staging slabs

per session/context:
  36 GDN recurrence matrices + conv state
  12 QSA K/V caches + 12 raw index-key caches
  PLE predecessor and dilated-conv state
  graph fixed buffers + logits

per prefill capacity/token:
  wide residuals and GR scratch
  GDN/QSA temporary tensors
  route IDs/weights and grouped routes
  QSA block scores/top IDs/gather tiles
  PLE row IDs/dedup map/gathered embeddings

transactional/transient:
  recurrent snapshot or journal
  expert and PLE read staging
  private command buffers/events
```

All multiplication/addition uses checked `uint64_t`. Admission calculates the
normal AUTO plan before allocating bulk buffers. Allocation failure unwinds in
reverse ownership order without altering the current engine/session.

## 7. SSD expert cache generalization

The existing cache floor is `routed_layers * top_k + 1`; Qwen4Exp therefore
needs at least `48*10+1 = 481` complete expert records, not 11 or 321. Phase
rounding should remain `1 + k*(routed_layers*top_k)` unless measurements and an
ADR change the invariant. Maximum selected unique experts in a batch is bounded
by `min(512, tokens*10)`, not 256.

Generalize pending structures from inline top-8 assumptions to bounded dynamic
storage whose capacity is calculated before publishing a read wave. Preserve:

- one encoder owner and monotonically nonzero generation ticket;
- no eviction of leased/in-flight records;
- publish only after every component read and validation succeeds;
- one-time transfer of selected-ID storage;
- abort/release on every early return;
- `async capable` means actual asynchronous I/O.

Paired gate+up reads are likely appropriate because ExpertMajor records are
`gate|up|down`, but select them only from validated component offsets/adjacency,
not from the model family name.

## 8. Tokenizer, chat and server boundary

Add a new chat format only if the pinned template semantics differ. Do not alter
`DS4_CHAT_FORMAT_QWEN36`. The safe renderer must preserve control-token
provenance and reject malformed structured messages. Sampling iterates only over
the valid vocabulary even if a physical output matrix is padded.

Core implementation and CLI qualification precede server exposure. Then add an
exact model alias, endpoint capability and evidence tests. Disk-KV remains out
of scope: current Qwen disk-KV is intentionally disabled, and “SSD streaming”
here means weights/PLE, not KV state.

## 9. New file proposal

```text
ds4_qwen4exp.h                 immutable descriptor and pure helpers
ds4_qwen4exp_ref.c/.h          scalar float32/hash/state oracles
ds4_ple_store.c/.h             PLE manifest, verification, row/page API
runtime/ds4_qwen4exp_graph.inc loader/graph/session textual partition
runtime/ds4_metal_qwen4exp.inc Metal host selectors/dispatch partition
metal/qwen4exp.metal           GR, GDN/QSA/index/PLE model-specific kernels
gguf-tools/qwen4exp-profile.py source inventory and artifact descriptor
tests/qwen4exp/...             model-free and model-backed fixtures/tests
docs/adr/NNNN-...md            support/artifact decision before merge
```

Do not create all files in one commit. The phase plan establishes the contracts
and tests first so later agents cannot quietly invent semantics in hot code.
