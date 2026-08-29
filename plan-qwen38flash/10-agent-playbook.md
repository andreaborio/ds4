# Implementation agent playbook and critical snippets

## 1. Before editing

Every agent must read repository `AGENTS.md`, runtime contract, relevant ADRs,
this README and the phase-specific documents. Then report:

```text
exact phase/prerequisite commit;
files to edit;
contract fields/equations implemented;
memory owners/lifetimes introduced;
tests and negative cases planned;
shared runtimes potentially affected;
explicit non-goals.
```

If a tensor name, dimension, hash rule or state transition is missing, stop and
consult the pinned sources. Never fill gaps from a different Qwen version.

## 2. Pure checked arithmetic pattern

All artifact/memory geometry uses checked unsigned arithmetic:

```c
static bool q4exp_u64_mul(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out || (a && b > UINT64_MAX / a)) return false;
    *out = a * b;
    return true;
}

static bool q4exp_u64_add(uint64_t a, uint64_t b, uint64_t *out) {
    if (!out || b > UINT64_MAX - a) return false;
    *out = a + b;
    return true;
}
```

No unchecked `count*stride+offset`, including tests. Validate range endpoint
against its owner extent after calculating it.

## 3. Exact uint64 PLE hash skeleton

Use pinned generated multipliers/primes/offsets, not placeholders below:

```c
static uint64_t q4exp_u64_mul_wrap(uint64_t a, uint64_t b) {
    return a * b; /* unsigned C arithmetic wraps modulo 2^64 */
}

static uint32_t q4exp_ple_row(uint32_t head, uint32_t current,
                              uint32_t prev1, uint32_t prev2,
                              const ds4_ple_hash_profile *p) {
    uint64_t h = q4exp_u64_mul_wrap(current, p->multiplier[0]) ^
                 q4exp_u64_mul_wrap(prev1,   p->multiplier[1]);
    if (head >= p->heads_per_ngram)
        h ^= q4exp_u64_mul_wrap(prev2, p->multiplier[2]);
    uint64_t local = h % p->prime[head];
    uint64_t row = p->offset[head] + local;
    DS4_ASSERT(row < p->row_count);
    return (uint32_t)row;
}
```

Predecessor normalization is separate and uses EOS 248044 for missing/truncated
history. Golden tests must force wraparound. Do not cast constants/tokens to
float, signed multiply or platform `long`.

## 4. Deterministic top-k comparator

```c
static bool q4exp_route_better(float pa, uint32_t ia,
                               float pb, uint32_t ib) {
    if (pa > pb) return true;
    if (pa < pb) return false;
    return ia < ib;
}
```

Reject nonfinite router values before selection. Softmax all 512 logits in F32,
select ten unique IDs under this comparator, then divide selected weights by
their F32 sum. Metal and C fixtures include all-equal logits.

## 5. QSA complete-group planner

Semantics are logical positions, never physical cache adjacency:

```c
for (uint64_t start = sequence_start; start + 3 <= query_position; start += 4) {
    ds4_slot s[4];
    if (!lookup_consecutive_same_sequence(start, s)) continue; /* or fail invariant */
    float pooled[128] = {0};
    for (unsigned j = 0; j < 4; ++j)
        for (unsigned d = 0; d < 128; ++d)
            pooled[d] += raw_index_key(s[j], d) * 0.25f;
    zc_rmsnorm_and_rope(pooled, start);
    score[start/4] = 0.0f;
    for (unsigned h = 0; h < 4; ++h)
        score[start/4] += fmaxf(dot(query[h], pooled), 0.0f);
    score[start/4] *= 1.0f / sqrtf(128.0f);
}
```

Production tiles this work and uses cached pooled keys, but fixture semantics do
not change. Select 512 groups, expand each to four logical IDs and append only
the incomplete causal suffix. Test `sum(relu)` against the wrong `relu(sum)`.

## 6. Transaction/ticket cleanup pattern

Use one cleanup path with explicit ownership flags. Publishing is last:

```c
bool q4exp_step(ds4_session *s, uint32_t token) {
    q4exp_tx tx = { .old_frontier = s->frontier };
    if (!snapshot_or_begin_journal(s, &tx)) goto fail;
    if (!ple_plan_begin(s, token, &tx.ple)) goto fail;
    if (!encode_embedding_layer0(s, token, &tx)) goto fail;
    if (!ple_finish_and_encode(s, &tx.ple, &tx)) goto fail;
    if (!encode_remaining_layers(s, &tx)) goto fail;
    if (!wait_and_validate_commands(&tx)) goto fail;
    commit_private_state(s, &tx);
    publish_logits_checkpoint_frontier(s, &tx); /* single final publication */
    tx.published = true;
fail:
    if (!tx.published) rollback_private_state(s, &tx);
    ple_ticket_abort_or_release(&tx.ple);
    expert_ticket_abort_or_release(&tx.expert);
    release_tx_resources_after_gpu_safe(&tx);
    return tx.published;
}
```

Actual asynchronous resource release may run in a completion handler; never free
a backing allocation while an unretained Metal reference is live. Force each
`goto fail` in a model-free test.

## 7. Fixed-page PLE locate/read pattern

```c
bool ds4_ple_locate(const ds4_ple_manifest *m, uint32_t row,
                    uint64_t *file_offset, uint32_t *slot) {
    if (!m || !file_offset || !slot || row >= m->row_count) return false;
    uint64_t page = row / m->rows_per_page;
    uint64_t rel, off, end;
    *slot = row % m->rows_per_page;
    return q4exp_u64_mul(page, m->page_stride, &rel) &&
           q4exp_u64_add(m->payload_offset, rel, &off) &&
           q4exp_u64_add(off, m->page_stride, &end) &&
           end <= m->payload_end && ((*file_offset = off), true);
}
```

Reads group missing adjacent pages only within `[payload_offset,payload_end)`.
Short reads loop or fail by documented policy; EOF/digest error cannot publish a
READY entry. Page cache key includes store digest/profile.

## 8. Converter streaming pattern

```python
with AtomicArtifactWriter(target) as out:
    for layer in range(profile.layers):
        for source_tile in source.iter_expert_tiles(layer, max_bytes=tile_cap):
            profile.validate_source_tile(source_tile)
            for expert, role, array in source_tile.iter_role_arrays():
                encoded = codec.encode(array, logical_shape=profile.shape(role))
                out.write_expert_component(layer, expert, role, encoded)
            source_tile.release()
    out.finish_manifests_and_hashes()
    out.reopen_and_verify_with_runtime_parser()
    out.fsync_and_install()
```

Do not use `.astype(bfloat16)` on an entire shard/model. For official FP8, read a
bounded tile, apply its `_scale_inv`, quantize immediately to final codec, write
and release. Assert zero unconsumed/missing source tensors at end.

## 9. Metal review checklist

- Host and Metal argument structs have byte-size static asserts.
- Every buffer offset/length is validated before encoding.
- Specialized selector names exact profile, codec, alignment and capability.
- General fallback remains reachable and tested.
- No CPU `.item`/readback/global `waitUntilCompleted` in token hot path except
  the one bounded route handoff required for authoritative expert SSD planning.
- Threadgroup memory/register assumptions checked on fallback device tier.
- Accumulators/fast-math comply with precision table.
- Command buffer bounded; pipeline states cached outside token loop.
- Resource lifetime extends past GPU completion; errors query command status.
- Counters prove sparse selected bytes and real I/O overlap.

## 10. SSD/concurrency review checklist

- Ticket has nonzero generation and exact owner session/store.
- Immutable task arrays after worker submission.
- Cache entry states have legal transitions and one publisher.
- Leased/reading/GPU-inflight entries cannot be victims.
- Old worker cannot install into recycled entry.
- Abort handles partially completed task wave and releases once.
- Worker count/bytes/queue depth are bounded.
- PLE current critical reads cannot starve behind expert readahead.
- Pressure change occurs only at safe phase boundary.
- Telemetry distinguishes logical requested, physical read and exposed wait.

## 11. Handoff evidence template

```text
Commit/dirty state:
Phase and accepted prerequisites:
Files changed and why:
Source/profile pins:
Semantic invariants implemented:
Memory owner/lifetime/byte changes:
Failure paths exercised:
Commands/tests with pass counts:
Golden fixture hashes/tolerances:
Physical hardware/capabilities if used:
Performance samples (if phase permits):
Known limitations/open questions:
Shared regressions run:
Next phase interface:
```

Do not hand off “works on my prompt”. Provide exact tests/evidence and point to
the next agent's stable interface. If a shortcut was necessary for a correctness
scaffold (dense QSA, resident tiny PLE), label it in code, test and handoff with
the phase that must remove it before support.

