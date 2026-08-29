# Artifact, converter and admission specification

## 1. Artifact set

The proposed text artifact is one GGUF v3 file containing:

1. exact Qwen4Exp metadata and tokenizer/template metadata;
2. accepted dense/non-routed tensors in a closed quant profile;
3. one opaque `ds4.expert_major.v2` tensor containing all routed gate/up/down
   matrices and its embedded v2 manifest;
4. one opaque `ds4.ple_rows.v1` tensor containing the quantized logical PLE row
   store and an embedded checksummed manifest;
5. no canonical routed-expert tensors, no PLE canonical tensor, no sidecars;
6. no vision tower for the text-only profile; the manifest explicitly records
   this exclusion and admission rejects image/video use;
7. no `mtp.*` tensors in the base artifact; they are accounted as deliberately
   excluded during conversion and may later form a separately identified artifact.

Adding `ds4.ple_rows.v1` is a new artifact format and needs an ADR/contract
update. Retaining ExpertMajor v2 avoids reviving the unaccepted v3 prototype.
The PLE store is a different access geometry and should not be disguised as
512 experts.

Although the PLE extent is embedded to preserve one fail-closed artifact, it is
physically page-aligned and isolated from dense spans. Hebrus reads it by file
descriptor/extent; it is never registered as one giant Metal buffer, warmed or
included in non-routed residency advice. If future evidence requires a sidecar,
that is a new ADR/artifact version rather than automatic discovery.

## 2. Source lock and preflight

Converter invocation requires an immutable source revision, not a mutable model
name. Before allocating an output file it verifies:

- repository/revision and source-file hashes;
- config/template/tokenizer hashes;
- exact 1,658-entry source weight map (or a newly reviewed pinned inventory);
- every tensor rank, shape and source dtype;
- all 131 declared shards present and matching index ownership;
- no duplicate tensor, missing source extent, overlapping shard extent or
  ignored unexpected tensor;
- enough free output space for temporary file plus fsync/rename;
- text-profile policy for vision and MTP tensors.

Produce a machine-readable conversion report listing every source tensor and
destination role. “Skipped” is legal only for an explicitly named profile rule
such as excluded vision or non-executed MTP, and its source bytes must still be
accounted.

## 3. ExpertMajor v2 profile

Add a new store family/profile ID; Qwen3.6's family encodes assumptions beyond
the container. Required geometry:

```text
layers             = 48
experts/layer      = 512
components         = gate, up, down
gate shape/expert  = [2560,640] logical orientation per Hebrus decoder contract
up shape/expert    = [2560,640]
down shape/expert  = [640,2560]
record order       = gate | up | down
record alignment   = manifest-declared and verified
layer alignment    = 4096 minimum
```

The converter must be descriptor-driven. Remove assumptions that MLX repacking
starts from a same-sized Q4_K file; direct build from pinned safetensors is the
canonical route. A minimal structure:

```python
@dataclass(frozen=True)
class ComponentSpec:
    role: str
    source_key: str
    source_shape: tuple[int, ...]
    logical_rows: int
    logical_cols: int

@dataclass(frozen=True)
class ExpertProfile:
    family_id: int
    layers: int
    experts: int
    components: tuple[ComponentSpec, ...]
    codec: str
    group_size: int

def convert_layer(profile, layer, writer):
    arrays = [load_exact(c.source_key.format(layer=layer),
                         c.source_shape) for c in profile.components]
    for expert in range(profile.experts):
        record_start = writer.tell()
        for component, array in zip(profile.components, arrays):
            payload = encode_exact(profile.codec, array[expert])
            writer.write_component(component.role, payload)
        writer.finish_record(record_start)
```

Do not hold all 48x512 source experts in RAM. Process one component/layer or a
bounded expert tile, issue sequential destination writes, hash as written, and
release source mappings promptly.

## 4. Quantization candidates, not assumptions

The 64 GB target makes a four-bit routed payload implausible as a fully resident
default: ideal routed payload alone is 56.25 GiB. Qualify at least:

- a high-quality reference artifact for logits/quality (may require SSD);
- a mixed approximately two-bit routed codec plus higher-bit dense tensors;
- an MLX affine candidate where a supported group size and measured Metal kernel
  make sense;
- a four-bit PLE candidate because PLE is table lookup rather than matmul.

Each codec is a closed profile with its own manifest ID, exact byte formula,
decoder, padding rules and tolerance/quality evidence. Do not declare one
canonical winner before perplexity/task and M5 measurements. The likely primary
shape is low-bit resident routed experts plus SSD-streamed PLE, but it remains a
promotion hypothesis.

## 5. PLE row store v1

### 5.1 Logical geometry

The manifest fixes:

```text
magic/version             = DS4 PLE v1
model family/profile       = exact Qwen3.8 Flash Next text profile
logical row count          = 320,001,536
logical width              = 160
logical head count         = 16
head moduli[16]            = exact primes from pinned reference
head offsets[16]           = cumulative exact offsets
hash algorithm/version     = SplitMix64-Qwen4Exp-v1
codec/group size           = closed profile
rows per physical page     = profile constant
page alignment             = >= 4096
row/page index extent      = explicit
payload extent             = explicit
per-page and whole digest  = explicit
```

Physical pages, rather than individual variable-sized rows, are checksum and
cache units. Choose rows/page so a requested row decodes without reading an
unbounded extent and adjacent rows amortize storage. For a 4-bit width-160 row,
raw payload is 80 bytes plus scales; 32 or 64 rows per 4-8 KiB page is a useful
candidate, not a frozen answer. Page header and quant tables must keep every
page independently decodable.

### 5.2 Indexing

Prefer an affine mapping `page = row / rows_per_page`, `slot = row % ...` with
fixed page stride. An explicit 320M-entry index is unacceptable. If final codec
uses variable sizes, redesign it to fixed pages or prove the smaller two-level
index budget and cache behavior. Checked arithmetic validates:

```c
bool ds4_ple_locate(const manifest *m, uint32_t row,
                    uint64_t *offset, uint32_t *slot) {
    if (row >= m->row_count) return false;
    uint64_t page = row / m->rows_per_page;
    *slot = row % m->rows_per_page;
    return checked_mul_add(page, m->page_stride, m->payload_offset, offset) &&
           *offset + m->page_stride <= m->payload_end;
}
```

### 5.3 Write protocol

Write to a sibling temporary file/extent, stream hashes, close/fsync, reopen and
run the same parser used by Hebrus, verify random boundary pages and the complete
manifest digest, then install via atomic rename. Embedding into GGUF repeats
extent validation and final fsync/rename. On failure retain neither a seemingly
valid target nor an unbounded temporary file.

## 6. GGUF metadata namespace

Use a new exact architecture namespace such as the upstream/llama.cpp registered
`qwen4exp`; the accepted spelling is an ADR decision. Metadata must carry every
closed model field in `01-model-contract-and-math.md`, including arrays for all
48 layer types, PLE primes/offsets/hash seed/version, GR and indexer fields.

Hebrus-specific metadata includes:

```text
ds4.model.profile_id
ds4.model.source_revision
ds4.model.tensor_inventory_digest
ds4.tokenizer.digest
ds4.chat_template.digest
ds4.text_only = true
ds4.expert_store.family/profile/codec digest references
ds4.ple_store.family/profile/codec digest references
ds4.mtp.present = false / ds4.mtp.executed = false
```

The manifest is authoritative for physical extents; GGUF keys duplicate only
the fields needed for early detection and cross-checking. Any disagreement is a
hard error.

## 7. Admission sequence

Admission is ordered to fail cheaply and before bulk allocation:

1. parse GGUF bounds with overflow-safe arithmetic;
2. exact architecture and profile ID;
3. source revision/inventory/tokenizer/template digests;
4. every scalar/array model constant and 48-layer pattern;
5. tokenizer vocabulary, merges, regex, special IDs and template;
6. exact tensor identity set, shape/type/extent and no overlaps;
7. one ExpertMajor v2 store, family/codec/manifest and 48x512 records;
8. one PLE v1 store, geometry/hash/codec/extents/checksums;
9. text-only/MTP policy;
10. exact non-routed page-span ownership and disjointness;
11. platform/Metal capabilities and AUTO memory plan;
12. only then allocate/load pipelines, graph and caches.

Every error names the first violated field, expected value, observed value and
artifact path, without exposing unchecked strings as format strings.

## 8. Tensor inventory and coverage proof

Construct exact non-routed spans from bound tensor identities, just as Qwen3.6
does, but driven by the Qwen4Exp inventory. Sort ranges and prove:

- each `[offset,end)` lies in the mapped GGUF data extent;
- no arithmetic overflow and required alignment holds;
- no overlap among dense tensors, ExpertMajor extent and PLE extent;
- every required physical byte is assigned to one owner or named padding;
- routed/PLE canonical weights are absent;
- page-rounded residency spans cannot accidentally include the huge PLE payload.

This last condition is critical: mapping or `mlock`-style warmup of a neighboring
dense span must not drag PLE pages into the primary working set.

## 9. Negative fixture matrix

Generate compact corrupt artifacts that independently alter: family, profile,
revision, context, layer pattern, each head dimension, top-k, expert count, GR
rank, PLE insertion layer, row count/width, one prime, one offset, hash version,
tokenizer/template hash, special ID, tensor rank/dimension/type, component
offset/length, overlapping extent, page stride, store checksum, missing/extra
tensor, second store, canonical routed tensor, vision tensor in text profile,
and MTP policy. Each must fail before GPU work and report the expected field.

## 10. Downloader/release boundary

Do not update production downloader aliases or `RUNTIME_SUPPORT.md` during
bring-up. First publish a reproducible artifact recipe and checksums in the work
area. After all gates pass, create a separate release contract rather than
rewriting the historical Qwen3.6 contract. Downloader verifies size plus digest,
supports resumable download safely, and runs offline admission verification
before presenting the artifact as installable.
