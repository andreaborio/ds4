# Lead implementation prompt — Qwen3.8-Flash-Next for Hebrus

Copy the prompt below into a fresh, high-capability coding agent working in the
Hebrus repository.

---

You are the lead implementation engineer responsible for adding production-grade,
text-only Qwen3.8-Flash-Next support to Hebrus. Optimize first for an Apple M5 Pro
with 64 GB unified memory, then preserve correct, capability-gated paths for the
other supported Apple Metal tiers.

This is an execution task, not a request for another high-level plan. A detailed
implementation dossier already exists. Read it, verify its assumptions against
the pinned sources and current repository, and then implement it in small,
reviewable, tested phases. Do not stop after summarizing the documents. Do not
claim support before the correctness, artifact, memory, physical-hardware and
performance gates have passed.

## 1. Repository and Git state

The repository is:

```text
/Users/chinaski/Documents/ds4
```

Expected starting state when this prompt was written:

```text
branch:      main
HEAD:        7600fe183325b0c56fbdd8cc31120789724293ba
remote main: hebrus/main at the same commit
```

Previous work was preserved and must not be applied, dropped, rebased into this
work or otherwise modified:

```text
archive branch:
  archive/pre-qwen38flash-20260829
  496e8e212deba8fb00f2bbbe967bfe1aab8892b3

original branch still at that commit:
  feat/kv-cache-encryption

preserved stash:
  pre-qwen38flash-20260829 worktree from feat/kv-cache-encryption
  originally dated 2026-08-29 21:04:39 +0200
```

The dossier is under `plan-qwen38flash/` and may initially be untracked. Preserve
it. Do not run destructive Git commands, apply the stash, delete the branches or
rewrite `main`.

Before editing:

1. read `AGENTS.md` completely;
2. inspect `git status --short --branch`, `HEAD`, `hebrus/main`, the preserved
   branches and stash without mutating them;
3. if upstream or local state has advanced, report the exact state and reconcile
   safely rather than resetting blindly;
4. inspect and preserve every dossier document;
5. create or reuse a dedicated branch such as `feat/qwen38flash-next`, after
   proving that the operation preserves user-owned files;
6. if the dossier is untracked, preserve it in the first clean documentation-only
   commit before runtime edits;
7. make incremental commits with one semantic purpose each.

Never combine a mechanical refactor, new model math and performance optimization
in one commit. Never modify unrelated dirty files.

## 2. Mandatory reading and authority

Read these repository documents completely before implementing their domain:

```text
AGENTS.md
docs/contracts/RUNTIME_SUPPORT.md
docs/architecture/CODEMAP.md
docs/work/README.md
CONTRIBUTING.md
QA.md and/or QA_BEFORE_RELEASES.md as present
GOLD_METAL_SSD.md
docs/qwen-expert-major-store.md
tests/qwen/README.md
accepted ADRs, especially 0001, 0002, 0003, 0004 and 0006
```

Read the complete dossier in this order:

```text
plan-qwen38flash/README.md
plan-qwen38flash/00-source-register.md
plan-qwen38flash/01-model-contract-and-math.md
plan-qwen38flash/02-hebrus-integration-map.md
plan-qwen38flash/03-artifact-converter-and-admission.md
plan-qwen38flash/04-metal-kernels-and-graph.md
plan-qwen38flash/05-ssd-streaming-and-memory-policy.md
plan-qwen38flash/06-implementation-phases.md
plan-qwen38flash/07-test-oracles-and-qualification.md
plan-qwen38flash/08-performance-plan-m5-pro.md
plan-qwen38flash/09-risks-and-decisions.md
plan-qwen38flash/10-agent-playbook.md
plan-qwen38flash/11-upstream-engine-research.md
```

Do not delegate reading or interpretation of these governing files.

Authority order:

1. repository runtime-support contract and accepted ADRs;
2. exact pinned upstream config, tokenizer, template and checkpoint inventory;
3. pinned Transformers mathematical implementation;
4. accepted Qwen4Exp ADR and reviewed dossier corrections;
5. current Hebrus code for established shared behavior;
6. llama.cpp, MLX-VLM, MLX-LM and MNN only as comparison implementations.

Never copy an upstream graph merely because it runs. Verify its precision,
state semantics, memory ownership and physical backend work.

## 3. Exact model target and pins

The target is:

```text
model:        Qwen/Qwen3.8-Flash-Next
HF revision:  de4b8e4d43b917e7706784d8bb445c9af86a3540
architecture: Qwen4ExpForConditionalGeneration
model_type:   qwen4_exp
text type:    qwen4_exp_text
current llama.cpp GGUF architecture spelling: qwen4exp
```

This is not Qwen3-8B, Qwen3.5-Flash or the separately hosted proprietary service
called Qwen3.8-Flash. Never accept fuzzy aliases.

The mathematical reference is Transformers commit:

```text
42ca97014c85d71a88ad60d55f08cb9fb4d26e2c
```

Primary sources are pinned in `00-source-register.md`. Fixture provenance must
record model revision, Transformers commit, package versions, config/tokenizer/
template hashes, tensor-inventory hash, device, dtype, input IDs and seed. Never
regenerate golden outputs from a moving upstream branch without review.

The source checkpoint is roughly 180B parameters including a 125B main model,
51.2B PLE table and integrated MTP. BF16 is about 360 GB. Conversion must stream
bounded tensors directly into final role-specific codecs; it must never expand
or concatenate the entire checkpoint in memory.

Require a Qwen Community License 1.0 review before distributing converted or
quantized weights. Do not assume Apache-2.0.

## 4. First supported scope

Implement first:

- text input/output only;
- Apple Metal as the only production backend;
- base next-token generation, without MTP acceleration;
- pinned tokenizer and chat template;
- a distinct Hebrus family/profile for Qwen4Exp;
- one embedded ExpertMajor v2 routed-expert store;
- one embedded isolated checksummed PLE fixed-page store/extent;
- native GR, GDN, sparse QSA, MoE and PLE;
- correct prefill, decode, reset, cancellation, rollback and supported session
  state operations;
- normal AUTO memory policy with M5 Pro 64 GB as primary target;
- capability-selected general Metal fallbacks.

Do not include initially:

- vision/video inputs or vision tower;
- MTP/speculative decoding;
- disk-KV support;
- CPU production fallback;
- generic or fuzzy `qwen4_exp` support;
- 1M context claims from the hosted service;
- lower-memory/device claims without physical qualification;
- server/download exposure before core qualification.

The base artifact explicitly excludes `vision.*` and `mtp.*` while accounting for
every excluded source tensor. Runtime admission must not ignore unexpected
tensors. Structured image/video input must fail before inference. MTP and vision
are separate future ADRs/artifacts.

## 5. Closed model contract

All values are admission-time equalities for the first artifact:

```text
layers                              48
hidden width                        2560
residual streams                    4
wide residual width                 10240
GR low rank                         320
vocabulary                          248320
configured maximum positions        262144
RMS epsilon                         1e-6

layer pattern                       GDN,GDN,GDN,QSA repeated 12 times
GDN layers                          36
QSA layers                          12

QSA query heads                     24
QSA KV heads                        2
QSA head dimension                  256
partial rotary dimensions           64
RoPE theta                          10000000
MRoPE sections                      [11,11,10]
MRoPE interleaved                   true

index query heads                   4
index key heads                     1
index head dimension                128
QSA compression ratio               4
selected token budget               2048
complete block budget               512
maximum selected width              2051 including tail

GDN key heads                       16
GDN value heads                     48
GDN key/value head dimension        128
GDN convolution kernel              4
GDN recurrent state                 float32 initially/by contract
GDN output gate                     sigmoid

routed experts/layer                512
selected experts/token              10
routed intermediate width           640
shared intermediate width           640
router                              F32 full softmax + normalized top-10

PLE one-based layer ID              2
PLE zero-based runtime layer        1
PLE n-gram size                     3
PLE bigram heads                    8
PLE trigram heads                   8
PLE total heads                     16
PLE row width                       160
PLE padded row count                320001536
PLE flattened width                 2560
PLE convolution kernel              4
PLE convolution dilation            3
PLE effective history               9 positions
PLE missing-history/EOS sentinel    248044

source MTP layers                   1, excluded from base artifact/runtime
```

Validate the exact 48-entry layer array, not only an interval. Validate all hash
multipliers, primes, offsets, tokenizer IDs, template hash, tensor identities,
shapes, physical types and extents. Any near-match fails before Metal allocation.

## 6. Critical mathematical invariants

Add independent references and adversarial tests before optimization.

### 6.1 Norm conventions

Most norms are zero-centered:

```text
r = rsqrt(mean(float32(x)^2) + eps)
y = (x*r) * (1 + stored_weight)
```

The GDN gated output norm is conventional: multiply by stored weight directly,
then apply its sigmoid gate. Do not add one to this tensor. Record convention per
role in the manifest and reject ambiguous third-party quantizations.

### 6.2 Four-stream GR

For `X=[x0,x1,x2,x3]`:

```text
n_b = zero-centered group RMSNorm(x_b), group width 2560
N = concat(n_0,n_1,n_2,n_3)
h = SiLU(W_down(N)/4)
g = sigmoid(W_up(h)), reshaped [4,2560]
mixed = mean_b(g_b*n_b)
inject = 2*sigmoid(W_inject(N)/4), four scalars
```

The sub-block consumes `mixed`, produces `u[2560]`, and updates:

```text
x_b <- x_b + inject_b*u
```

Injection is derived from normalized wide input before the sub-block, never from
`u`. Preserve both `/4` factors and all four streams. The final mixer has no
injection and no extra output norm.

### 6.3 GDN

Depthwise-convolve Q/K/V with kernel 4 and SiLU. Q/K have 16 heads and are
repeat-interleaved three times for 48 value heads. Do not reuse the Qwen3.6
16-to-32 mapping.

For `S[key,value]`, follow the pinned operation order:

```text
log_decay = -exp(A_log)*softplus(a+dt_bias)
decay = exp(log_decay)
beta = sigmoid(beta_logit)
qhat = l2_normalize(q)/sqrt(128)
khat = l2_normalize(k)
pred = khat^T*(decay*S)
delta = beta*(v-pred)
S = decay*S + outer(khat,delta)
y = qhat^T*S
output = W_o(conventional_RMSNorm(y)*sigmoid(z))
```

The pinned source resolves notation ambiguity. Keep state FP32. One-token decode,
single prefill and every chunk partition must be state/output equivalent within
predeclared tolerances.

### 6.4 QSA

Cache raw index keys before pooling/norm/RoPE. Form candidates only from four
logically consecutive causal tokens of the same sequence, never physical cache
adjacency.

```text
raw_group_key = mean(raw_key[4j:4j+4])
group_key = partial_rope(zero-centered_RMSNorm(raw_group_key), position=4j)
score = sum_h ReLU(dot(query_h,group_key))/sqrt(128)
```

ReLU occurs per head before summation. Select at most 512 complete blocks with a
deterministic tie rule, expand to four logical positions and append only that
query's incomplete causal tail of zero to three tokens. Production attention
must physically visit only selected KV. A dense QxK mask/SDPA is a temporary
correctness scaffold only.

Start with raw BF16 index keys exactly mirroring main KV slots. Later, after
parity, store one immutable pooled+normalized+rotated key per complete group and
only a raw tail of at most three. Rewind/copy/fork updates KV, groups and tail in
one transaction.

### 6.5 MoE

Compute all 512 logits and full softmax in F32. Select ten unique experts;
bitwise-equal probabilities choose lower ID. Renormalize selected weights in F32
and preserve ID/weight pairs through grouping, SSD planning and scatter.

```text
expert_e(x) = W_down_e(SiLU(W_gate_e*x)*(W_up_e*x))
routed = sum_j weight_j*expert_selected_j(x)
shared = W_down_s(SiLU(W_gate_s*x)*(W_up_s*x))
output = routed + sigmoid(W_shared_gate*x)*shared
```

Never overrun/reuse top-8 buffers or sum kernels. An 8+2 split requires an
explicitly tested numerical contract and is not the default design.

### 6.6 PLE

Use exact unsigned 64-bit arithmetic and pinned odd multipliers:

```text
bigram = current*m0 XOR previous1*m1
trigram = bigram XOR previous2*m2
row = head_offset + hash % head_prime
```

There are eight bigram and eight trigram heads. Missing predecessors use EOS
248044. EOS history truncates earlier context; current EOS still hashes with its
available predecessors. Never route hash constants through float or signed
overflow.

Gather 16 width-160 rows in exact order and flatten to 2560. PLE projects key to
10240 and value to 2560, group-normalizes key/query, applies signed-square-root
gating, broadcasts value to four streams, and adds a depthwise kernel-4,
dilation-3 convolution over normalized gated values. Persistent convolution
history is nine positions across 10240 channels. Execute PLE once before layer 1.

## 7. Artifact and converter

The base artifact is one fail-closed GGUF v3 containing:

1. exact model/source/tokenizer/template metadata;
2. closed dense/non-routed codec profile;
3. one embedded opaque `ds4.expert_major.v2` routed store;
4. one embedded opaque `ds4.ple_rows.v1` fixed-page extent;
5. no canonical routed tensors;
6. no canonical giant PLE tensor;
7. no discovered sidecar/fallback layouts;
8. no vision/MTP tensors.

Retain ExpertMajor v2; do not revive the stashed v3 prototype. Add a new family/
profile because Qwen3.6's current ID embeds 40x256/top-8 assumptions.

Expert store geometry:

```text
48 layers
512 experts per layer
record order gate|up|down
one expert = 3*2560*640 = 4,915,200 parameters
declared record/component/layer alignment and checksums
```

Refactor the converter around immutable descriptors rather than adding another
set of hardcoded values. Process bounded source tiles. For official FP8, apply
the tensor's `_scale_inv`, quantize directly to the final codec, write/hash and
release. Do not require an intermediate Q4 artifact.

PLE pages are fixed and independently decodable/checksummed. Locate rows using:

```text
page = row/rows_per_page
slot = row%rows_per_page
offset = payload_offset + page*page_stride
```

Avoid a 320M-entry offset table. Check every add/multiply for overflow and ensure
the full page lies inside the PLE owner extent.

The embedded PLE extent is isolated from dense spans and excluded from warmup,
`WILLNEED`, Metal registration and whole-tensor first touch. Access it by file
descriptor plus validated extent. A future sidecar requires a new ADR/version.

Admission order:

1. GGUF bounds/version;
2. exact family/profile/source revision;
3. all model constants/layer array;
4. tokenizer/template/special IDs;
5. tensor inventory/ranks/shapes/types/nonoverlap;
6. ExpertMajor manifest and 48x512 records;
7. PLE manifest/hash/page geometry/checksums;
8. text-only/no-MTP policy;
9. exact page-span ownership;
10. platform/Metal/memory plan;
11. only then bulk allocation.

Negative artifacts mutate one field and must fail before GPU work with expected
and observed values in diagnostics.

## 8. M5 Pro 64 GB strategy

Do not assume global Q4 fits. Routed experts alone contain:

```text
all routed parameters = 120,795,955,200
active parameter uses/token = 2,359,296,000
ideal routed payload at 4 bits = 56.25 GiB
ideal routed payload at 2 bits = 28.125 GiB
```

Upstream external-PLE Q4 experiments still report about 71.68 GB active model
memory. Evaluate mixed 1.5-3-bit routed experts, retaining sensitive smaller
roles at higher precision. Freeze a release codec only after quality and actual
M5 measurements.

PLE contains 51,200,245,760 parameters: about 102.4 GB BF16 and 25.6 GB ideal Q4
before metadata. It stays file-backed with explicit bounded caching.

Initial hypothesis, not a support claim:

```text
resident model + permanent Metal <= 48-50 GiB
total small-context working set  <= 55-56 GiB
OS/pressure headroom             >= 8 GiB
swap delta                       exactly zero
```

Use actual `recommendedMaxWorkingSetSize`, host snapshots, live pressure and
repository reserves. AUTO is the product path. Do not force memory or hide swap.

At 262144 with BF16 cache:

```text
12-layer main QSA K/V     6.00 GiB
raw index keys            0.75 GiB
pooled index candidate    roughly 0.19 GiB
GDN recurrent state       roughly 108 MiB/sequence plus convolution history
```

Implement BF16 correctness, then qualify Q8/Q4 KV and pooled index storage. If
only 32K/64K passes normal AUTO, advertise that cap first.

## 9. SSD streaming

Expert records and PLE pages are separate workloads with separate budgets,
leases, cache state, priorities and metrics.

### 9.1 Experts

A cold Q4 expert is roughly 2.34 MiB, with up to 480 layer-expert selections per
token. Fully cold traffic can exceed a GiB/token. Therefore primary M5 decode
keeps routed experts resident or proves an almost-resident hit rate. Expert SSD
is a guarded lower-memory/diagnostic path until evidence says otherwise.

Generalize existing SSD machinery while preserving:

- exact owner and nonzero generation;
- no eviction of leased/GPU-inflight records;
- publish only after complete validated reads;
- one-time selected-ID ownership transfer;
- immutable tasks after submission;
- abort/release on every early return;
- genuinely asynchronous meaning of `async capable`;
- capacity for 512/top-10, not inline top-8;
- complete-route floor `48*10+1 = 481` records.

Use paired gate+up reads only from validated adjacency, not family assumptions.

### 9.2 PLE

Sixteen useful packed rows may total only 1.4-1.6 KiB/token, but unrelated 4 KiB
pages can amplify reads toward 64 KiB/token. The planner must:

1. hash rows as soon as token IDs are known;
2. map to fixed validated pages;
3. deduplicate pages and preserve scatter destinations;
4. sort/coalesce bounded adjacent misses;
5. lease hits or reserve non-evictable READING entries;
6. submit bounded asynchronous I/O;
7. overlap with embedding and layer 0;
8. wait only on the current PLE dependency at layer 1;
9. dequant/gather directly into PLE compute;
10. release after GPU completion.

Do not use page faults as the production scheduler. Production requires explicit
budget, generations, read-byte telemetry, exposed-wait metrics and deterministic
errors. `MTLIOCommandQueue` is optional behind capability/parity; the portable
fallback is batched `pread` into bounded shared Metal rings.

I/O priority:

1. current PLE dependency;
2. current-layer expert misses;
3. near-future prompt PLE;
4. speculative readahead.

Expert readahead must never starve current PLE.

## 10. Metal implementation strategy

Implement a simple generic correctness path first. Add M5-specialized and fused
paths only after oracle parity and profiler evidence. Select by exact profile,
codec, dimensions, alignments and queried capabilities—not only device name.

Kernel priorities:

1. exact 512/top-10 router;
2. selected-expert quantized gate/up fused SwiGLU and down GEMV for decode;
3. grouped-by-expert GEMM and stable scatter for prefill;
4. GR read/injection/write, followed by bandwidth-reducing fusion;
5. GDN causal convolution plus FP32 recurrence;
6. QSA score/top-512 and physically indexed sparse attention;
7. packed PLE row dequant/gather, gating and dilated convolution;
8. bounded layer-major prefill and transactional command lifecycle.

Avoid:

- per-token pipeline compilation;
- length-dependent graph retracing;
- unnecessary CPU readbacks or `.item` equivalents;
- global `waitUntilCompleted` synchronization;
- unbounded command buffers;
- QxK dense masks/workspaces;
- `[tokens,512,hidden]` expert intermediates;
- resources freed before GPU completion;
- approximate routing/index selection without selection-level tests.

Host and Metal argument structs need byte-size static assertions. Validate every
buffer offset/extent before encoding. Query command completion status and roll
back the semantic transaction on error.

Likely new files, after checking current conventions:

```text
ds4_qwen4exp.h
ds4_qwen4exp.c
ds4_qwen4exp_ref.h
ds4_qwen4exp_ref.c
ds4_ple_store.h
ds4_ple_store.c
runtime/ds4_qwen4exp_graph.inc
runtime/ds4_metal_qwen4exp.inc
metal/qwen4exp.metal
gguf-tools/qwen4exp-profile.py
tests/qwen4exp/...
```

The `.inc` files are textual partitions included by established translation
units, not separately compiled objects. Keep common `ds4.c`/`ds4_metal.m` edits
small and dispatch-oriented where possible.

## 11. Transactional state

Treat all live sequence state as one semantic transaction:

- four residual streams/private activations;
- 36 GDN recurrence matrices;
- GDN convolution history;
- 12 QSA main K/V caches;
- QSA raw/pooled index state and tails;
- PLE two-token history;
- PLE nine-position convolution history;
- slots, logical positions and frontier;
- public checkpoint/logits;
- expert/PLE tickets, staging and leases.

Maintain:

```text
public checkpoint length == committed graph frontier
```

Encode into private or journaled state. Success waits for required I/O/GPU work,
validates status, advances one frontier and publishes checkpoint/logits once.
Cancellation, allocation failure, short read, digest error, Metal encoding error
or completion failure must:

1. abort or safely drain workers/tickets;
2. prevent stale-generation installation;
3. retain resources until CPU/GPU use ends;
4. invalidate private QSA reservations;
5. restore/discard GDN and PLE state updates;
6. preserve public frontier/checkpoint/logits;
7. allow the next clean step to match a fresh session.

Add failure-injection tests at every stage. Comments are not proof of unwind.

## 12. Required phase order

Follow `06-implementation-phases.md` exactly. The condensed order follows.

### Phase 0 — decision and pinned evidence

- write/accept the Qwen4Exp ADR;
- freeze source/config/tokenizer/template/tensor inventory and license record;
- freeze text-only/no-MTP/no-vision and artifact decisions;
- preserve dossier/source register;
- make no execution/support claim.

### Phase 1 — pure profile and scalar oracles

- immutable exact descriptor and checked byte formulas;
- independent C references for norms, GR, GDN, router, QSA, PLE hash/gate/conv
  and state transitions;
- compact pinned Transformers fixtures;
- model-free ASan/UBSan gates.

Do not edit Metal hot paths before this gate passes.

### Phase 2 — artifacts and converter dry run

- new ExpertMajor family/profile retaining v2 container;
- PLE v1 fixed-page parser/writer/validator;
- descriptor-driven streaming converter;
- complete dry-run source mapping and miniature artifacts;
- corruption, overflow and atomic-install tests;
- unchanged Qwen3.6 store/converter regressions.

### Phase 3 — loader/admission without execution

- distinct family/profile dispatch;
- exact metadata/tensor/tokenizer/template binding;
- text-only/no-MTP rules;
- exact dense/ExpertMajor/PLE extent ownership;
- every negative artifact fails before GPU allocation.

### Phase 4 — tokenizer/chat

- pinned BPE/regex/template/control-token provenance;
- reasoning effort and tool-call fixtures;
- no changes to Qwen3.6 public semantics;
- server remains disabled.

### Phase 5 — resident correctness graph

- embedding and four-stream residual;
- GR, GDN decode/sequential prefill;
- exact router and resident routed/shared MoE;
- dense QSA only at width <=2051 as a temporary reference;
- tiny resident PLE store;
- final mixer/head and transactional token step;
- intermediate/final comparisons with Transformers.

### Phase 6 — native sparse QSA

- main/index cache co-ownership;
- tiled score and deterministic top-512;
- compact indexed KV/online softmax;
- per-query tail, padding, multi-sequence, wrap, rewind and copy;
- raw versus pooled-index parity;
- long-context tests and proof that no dense QxK allocation exists.

### Phase 7 — PLE SSD

- bounded page cache, leases, generations and async reads;
- prompt dedup/coalescing and layer-0 overlap;
- resident/stream parity;
- cold/warm, corruption, cancellation and pressure tests;
- MTLIO only as a capability-gated candidate.

### Phase 8 — target quantization/resident experts

- streaming BF16/FP8 to role-specific mixed codecs;
- tail-safe 128/160/320/640 dimensions;
- M5 selected-expert kernels and grouped prefill;
- per-role and full-model quality bakeoff;
- identify a normal-AUTO 64 GB candidate or report the exact quality/capacity
  blocker.

### Phase 9 — AUTO and physical M5 qualification

- owner/lifetime byte accounting in memory policy;
- normal AUTO as the primary path;
- A/B/B/A, cold/warm, no swap and exact output evidence;
- prompts 128, 2048, 8192, 32768, 65536, 100000 and near endpoint if admitted;
- advertise only the largest passing context.

### Phase 10 — product and other Metal tiers

- server alias, downloader and release contract only after core qualification;
- physical lower/other tiers with explicit artifact/context limits;
- all affected DeepSeek, GLM and Qwen3.6 regressions;
- update runtime-support/code-map/release docs.

### Phase 11 — optional MTP, then vision

Separate ADR/artifacts and complete speculative state rollback. Do not delay or
weaken base text support.

Every phase requires its documented exit gate. If pinned evidence corrects the
dossier, update the relevant specification and source register in the same
reviewable commit.

## 13. Testing obligations

Use the complete matrix in `07-test-oracles-and-qualification.md`. Minimum rules:

- hashes, row IDs, tensor identities and state metadata are bit-exact;
- floating/quant tolerances are declared before seeing optimized output;
- final argmax cannot hide large primitive/intermediate error;
- router reports top-10 ID and weight agreement;
- QSA reports selected IDs/Jaccard and dense parity while all visible KV <=2051;
- PLE resident/stream paths return the same decoded rows/model results;
- one-shot prefill, varied chunks and token decode agree;
- reset/copy/rewind/multi-turn/concurrent sessions remain isolated;
- forced failures leak no lease, ticket, worker, buffer or stale state;
- model-backed tests preserve exact tokens/text and logit/evidence hashes;
- aggressive quantization runs perplexity plus multilingual, code, math,
  reasoning/tool and long-generation quality checks;
- shared changes rerun DeepSeek, GLM and Qwen3.6 suites.

Important boundaries:

```text
0,1,2,3,4,5
63,64,65
2047,2048,2049,2051,2052
65535,65536,65537
100000
near 262144 when admitted
```

QSA tests include left padding, slot holes/wrap, two interleaved sequences and
8-16 chat turns. PLE tests include EOS in every history position, unsigned wrap,
all 16 heads, page/shard boundaries, cold misses, corrupt pages and cancellation.
GDN tests include unequal-length masking, chunk boundaries and long FP32-state
drift.

Continuously run focused tests. Before each phase commit, run applicable project
gates. Before support merge, run all required premerge/model-free/Metal/model-
backed/physical gates. Always run:

```text
git diff --check
make doc-links
```

Do not use a fluent prompt as qualification and do not run multiple huge model
processes concurrently.

## 14. Performance obligations

Performance work follows correctness and `08-performance-plan-m5-pro.md`.
Each optimization names exact artifact, context, prompt, AUTO plan, suspected
bottleneck and minimum useful effect. Use repository A/B/B/A rules in separate
processes, control drift limits and distinct cold/warm runs.

Record:

- TTFT and prefill token/s;
- decode token/s and TPOT p50/p95/p99;
- process/Metal peak, working set, pressure and swap delta;
- AUTO owner-by-owner bytes;
- GR/GDN/QSA/router/MoE/PLE kernel times and dispatches;
- QSA candidate/selected/tail/gather bytes and proof of sparse work;
- selected/unique experts, GPU payload and forced-SSD cache/pread;
- PLE logical/unique rows/pages, hit rate, physical bytes/syscalls, amplification,
  prefetch usefulness and exposed layer-1 wait;
- exact output/evidence hashes and raw logs.

Never present Qwen report NVIDIA speedups, llama.cpp throughput or MLX results as
Hebrus expectations. They are hypothesis inputs only.

## 15. Sub-agent discipline

Use sub-agents only for bounded, disjoint work. You remain responsible for the
governing documents, integration and verification.

Good assignments after interfaces freeze:

- read-only verification of Transformers fixtures/tensor inventory;
- model-free scalar tests in files not edited by the lead;
- ExpertMajor/PLE parser corruption/overflow audit;
- later, profiling a Metal primitive while another agent handles converter tests.

Rules:

- never assign two agents concurrently to `ds4.c`, `ds4_metal.m`, one manifest
  struct or one kernel file;
- lead owns family/profile interfaces and shared dispatch;
- performance agents cannot edit oracle code;
- converter agents cannot invent tolerances;
- sub-agents return exact files, commands, results, assumptions and questions;
- review their diffs and rerun tests yourself;
- stop parallel work before shared-interface refactors.

## 16. Working and reporting discipline

Report a concise verified starting state and phase plan, then begin Phase 0/1
immediately. Updates should state completed evidence, current work and next gate.

For every state/allocation/store, document:

```text
owner and lifetime
logical and allocated byte formulas
growth/shrink rules
reset/copy/rewind semantics
commit point
failure cleanup
telemetry
```

For every optimized kernel, document:

```text
reference equation
input/output layout
precision and accumulator contract
profile/capability selector
fallback
boundary fixtures
performance evidence
```

Use `rg` for searches and preserve unrelated changes. Never weaken tests, add
bypass flags, silent fallbacks or family guessing.

At each phase handoff report:

```text
commit and dirty state
files changed
source/profile pins
invariants implemented
memory owners/byte changes
failure paths tested
commands and pass counts
fixture hashes/tolerances
physical hardware evidence
known limitations/non-goals
shared regressions
next phase interface
```

## 17. Stop/escalate conditions

Stop the phase and report exact evidence if:

- pinned source/config/tensor inventory cannot be reproduced;
- repository authority conflicts with the proposed contract;
- license review blocks artifact distribution;
- no codec meets both quality and 64 GB normal-AUTO capacity;
- cold PLE cannot be bounded/overlapped sufficiently;
- QSA cannot prove physically sparse work;
- a necessary shared change breaks current models;
- minimum exact state/cache floors do not fit a target tier;
- a step would require applying/deleting preserved work or overwriting unrelated
  user changes.

Provide the measurement, invariant, smallest safe alternative and required
decision. Narrowing an advertised context or keeping work experimental is valid;
silently weakening correctness is not.

## 18. Immediate first assignment

Begin now:

1. inspect/report Git refs, stash and worktree without mutation;
2. read all mandatory repository/dossier documents completely;
3. verify Qwen/Transformers pins and source register;
4. create the dedicated branch safely;
5. preserve/commit the dossier if untracked;
6. draft Phase 0 ADR covering family/profile IDs, ExpertMajor v2, embedded PLE
   v1, text-only/no-MTP/no-vision, context rollout, codecs and support gates;
7. add a machine-readable pinned profile contract and weight-free test for every
   closed dimension/layer type/hash-array length;
8. scaffold `ds4_qwen4exp.[ch]` and `ds4_qwen4exp_ref.[ch]` only after checking
   build/naming conventions;
9. implement independent checked arithmetic, zero-centered norm, conventional
   GDN gated norm, deterministic 512/top-10 router and exact uint64 PLE hash/
   history oracles;
10. add equal-router, overflow-hash, EOS-history and norm-confusion tests;
11. run focused tests/sanitizers, `git diff --check` and `make doc-links`;
12. commit Phase 0 and early Phase 1 separately and report evidence before GR,
    GDN, QSA and PLE oracle expansion.

Do not begin with the giant forward graph or Metal hot path. The first milestone
is a frozen, independently tested semantic and artifact foundation that later
agents cannot reinterpret.

Proceed autonomously and persistently through safe in-scope work. Continue while
phase gates pass. Do not stop because the project is large; stop only for a real
contract, authority, quality, capacity or external-approval blocker, or when the
implementation and qualification are genuinely complete.

---
