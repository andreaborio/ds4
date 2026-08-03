# DSpark development oracle

This directory is a model-free numerical oracle for the DeepSeek V4 Flash 0731
DSpark work. It does not participate in model loading or inference. Production
remains the native Hebrus Metal runtime.

The NumPy reference covers the complete model-free DSpark seam: ordered capture
after target layers 40/41/42, stage-zero `main_proj` plus `main_norm`, pending/noise
embedding layout, four-lane Hyper-Connection split/Sinkhorn/post/final-head
equations, three independent logical raw-cache rings, the sequential Markov
correction, the per-position confidence cutoff, and speculative-sampling
accept/reject and residual rules. Random uniforms are fixture inputs, so a
failure is reproducible without controlling a framework RNG.

`support_schema.py` separately defines the reviewed final standalone support
GGUF v3 header: 81 tensors split 26/24/31 across stages 0/1/2. `main_proj` and
`main_norm` exist only in stage zero; final norm, HC head, Markov, and confidence
weights exist only in stage two. Its parser stops after metadata and tensor
descriptors. For a real file it also checks aligned, non-overlapping,
in-bounds offsets and the structural file size. It never maps or reads weight
payloads. Complete SHA-256 and payload identity remain the responsibility of
`gguf-tools/ds4-expert-major.py verify --dspark-support`.

The confidence cutoff follows the pinned official DeepSpec evaluator: apply
`sigmoid` to each confidence logit and stop before the first position whose
probability is strictly below the threshold. Confidence probabilities are not
multiplied. DeepSpec does use `cumprod` elsewhere, on the binary sampling
acceptance mask so that tokens after the first rejection cannot be committed;
that separate operation must not be imported into confidence scheduling.

The official token shift is explicit. After target position `p` is evaluated,
target logits have already sampled pending token `y[p+1]`. DSpark embeds
`[y[p+1], noise, noise, noise, noise]` at absolute input positions
`p+1..p+5`; its five head rows propose outputs `p+2..p+6`. It never
re-predicts `y[p+1]`. The vanilla Markov head is sequential: position zero
uses that pending token, and every later position uses the token sampled at the
previous draft position. Sampled verification uses
`min(1, p(x) / max(q(x), 1e-8))`; on rejection it samples normalized
`max(p-q, 0)`, falling back to the target row when residual mass is at most
`1e-8`, exactly as in the pinned evaluator. Full acceptance samples the bonus
target row.

The raw-cache oracle pins the final 0731 semantic geometry to three independent
`[128, 512]` rings, not a backend allocation. Prompt prefill retains and
projects every one of the last 128 target capture rows, not just the frontier
tap. Every stage derives its own context rows directly as
`kv_norm(Wkv(main_x(t)))`, applies RoPE to the final 64 dimensions at absolute
position `t`, and FP8-simulates the other 448 dimensions. Target history does
not pass through the candidate HC-pre or attention-norm path.

When a draft begins, the ring is already committed through position `p`; its
target-derived `h[p]` row must not be appended a second time. A proposal view
is therefore committed-through-`p` history plus five transient candidate rows
for `p+1..p+5`. Once full, the pinned top-k indices enumerate physical ring
slots `0..127`; they do not rotate the view back into chronological order.
That distinction is observable because attention advances in 64-row blocks
and rounds each block's exponential numerator weights to BF16. Building the
view is pure. A verifier transaction validates
all captured target rows before appending every row in its accepted prefix;
rejected suffixes and zero-row rollback leave the prior ring byte-identical.
Closed samples cover prompt truncation to 128, wrap, multi-row acceptance, and
rollback across absolute positions 0, 127, 128, and 129.

The NumPy storage simulation follows the pinned official kernel boundaries:
BF16 storage around RMSNorm/RoPE, seven 64-wide non-RoPE groups, `amax` floored
at `1e-4`, UE8M0 power-of-two scale `2^ceil(log2(amax/448))`, E4M3FN clamp and
nearest-even rounding, dequantization, then BF16 storage. It does not claim to
reproduce the quantized-weight `Wkv` GEMM or Metal command ordering; those
still require real runtime captures.

The sparse-attention oracle additionally pins the final geometry and precision
schedule: five noncausal query rows, 64 heads, 512-wide shared K/V, online
softmax in physical 64-row blocks, FP32 running max/denominator/accumulator,
BF16 weights before every value product, a denominator-only FP32 sink, and a
final BF16 output. The deterministic full-ring fixture changes 10,073 output
lanes (maximum `0.00048828125`) if a plausible chronological staging shortcut
is substituted, so the rejected order cannot pass under a loose float bound.

The stage-zero attention-half fixture freezes the official ordered seam as
`HC-pre -> attention norm -> Q-A -> Q-A norm -> Q-B -> per-head norm -> RoPE`
in parallel with `KV -> KV norm -> RoPE -> non-RoPE FP8 Q/DQ`, followed by the
physical two-source attention, inverse RoPE, grouped output-A, output-B, and
HC-post. The stage input is first published to BF16; that reopened value is
used by both HC-pre and the HC-post residual. Every other named return to model
storage is also explicit BF16. Per-head Q normalization separately publishes
BF16 after square, mean, epsilon addition, reciprocal square root, and final
multiply. Absolute proposal positions must begin at the committed raw-cache
frontier, so RoPE cannot silently detach from cache history. The fixture covers
both a partial `C=2` ring and a physical wrapped `C=128` ring.

The compact Q-B projection retains all eight reduced Q ranks and gives every
one of the 64 heads a distinct exact signature. Rotating its head blocks
changes all 64 heads through Q, RoPE, and attention, including 1,429 wrapped
attention lanes. Output-A uses rank two, makes both ranks live in every group,
and exposes the required group-major/rank-minor flattening: a rank-major
mutation changes all 40 output-B lanes. Other negative controls are likewise
observable. Rebuilding the full ring chronologically changes 83 attention
lanes (maximum `0.000244140625`); skipping the input publication changes 14
HC-pre lanes, 640 attention lanes, and 92 HC-post lanes; substituting the raw
input only as the residual changes 88 HC-post lanes. Omitting BF16 before
attention norm changes 15 lanes, using a float32-until-final Q norm changes
52,096 lanes, and omitting the output-A publication changes 19 lanes.

That fixture does not claim synthetic reduced weights reproduce checkpoint
weights. Hidden width and Q-LoRA/output-LoRA ranks are reduced to keep the
fixture small; the semantic attention seam remains the final geometry
`[5, 64, 512]`, with eight output groups and the physical `[128, 512]` ring.
Its dense synthetic GEMMs do not simulate the activation or weight
quantization of Q-A, Q-B, KV, output-A, or output-B, and they do not exercise
the production Q8 Metal decoder. Those require real-weight runtime captures.
Deterministic compact projections only keep this boundary oracle numerically
well-conditioned. In particular, the fixture gives the synthetic V path a
small positive floor to avoid cancellation around zero. That conditioning is
not evidence about checkpoint value distributions; the separate raw-context
oracle covers general signed storage inputs and varying FP8 group scales. On
MLX 0.32 Metal every named boundary before attention, and every boundary after
inverse RoPE, is exact against NumPy. `C=2` attention and inverse RoPE are also
exact; at wrapped `C=128`, attention differs in two lanes and inverse RoPE in
ten. Every difference is one adjacent BF16 code with maximum absolute drift
`0.000244140625`.

`physical_fixture.py` adds a separate payload-first handoff fixture for the
native Metal white box. It uses Q8-compatible reduced geometry: hidden width
32, Q-LoRA rank 32, output rank four (so the flattened low dimension is 32),
while retaining five rows, 64 heads, width 512, eight output groups, and both
`C=2` and wrapped `C=128` cache states. The five Q8_0 projections and the F16
HC function are serialized first and reopened from their bytes; the oracle
never receives their ideal float matrices. Q8_0 follows the repository C
quantizer exactly: codes use float32 `d=amax/127` and `roundf` ties away from
zero, while dequantization uses the serialized little-endian F16 `d`. The
cross-language test compiles the current `gguf-tools/quants.c` into a temporary
dynamic library and compares its bytes with the Python packer for the scale
discriminator, 32-wide fixture rows, sparse 4096-wide rows with zero blocks, a
seeded dense matrix, all five fixture Q8 payloads, and the HC F16 payload. No
compiled test asset is retained.

The model-map handoff is 64-byte aligned, 1,281,856 bytes, and has SHA-256
`1388a4a205ae61c59a25df4a03af312e2dea1fb13d35f6503362f06dd0ee1492`;
the tests freeze every tensor offset, size, and payload digest. Small dyadic
values that quantize to zero distinguish every ideal Q8 matrix from its
reopened payload, and substituting the ideal matrix is observable separately
for Q-A, Q-B, KV, output-A, and output-B. The live payload uses one nonzero
code per row with `d=2^-8`. Every activation entering a Q8 projection is also
F16-exact, so the fixture is conditioned for the required generic half-staged
Q-A/Q-B/KV/output-B paths with one exact product and no reduction-order
ambiguity. The native hook is required to use direct grouped Q8 matvec for
output-A and to capture actual dispatch selection; the manifest records only
those required paths, not proof that the runtime selected them. Output-A
crosses an explicit BF16 boundary before output-B. The 256-pair Q-head
reduction and NumPy mean publish identical BF16 means on this fixture, but the
end-to-end construction does not make every internal square/mean/add
publication independently observable. The dedicated native `0x384A` helper
test covers those internal boundaries. The one-live-code construction
validates payload layout and named publications; it does not qualify the dense
Q8 accumulation exercised by checkpoint weights. That remains a separate
real-weight capture gate.

The raw `[5, 4, 32]` hidden source has deterministic `+/-2^-12` F32 dust in
all 640 lanes. Every lane rounds back to the existing exact BF16 boundary, so
all downstream boundary digests remain stable while a native hook that skips
the ingress publication is observable. Its separate 2,560-byte source digest
is `e15d38302793fb96779672dcc38a99ef59b0de2f07ea25c17c348d37335dad57`;
it is not part of the aligned weight blob.

The C handoff also freezes the complete physical stage-zero F32 ring buffer,
not a chronological reconstruction: `C=2` is capacity 128, start 0, length 2,
SHA-256 `e27748d96d6d36cd5b12f42a710eb76d0e29b27c774fc11d153e9536d4526c9d`;
wrapped `C=128` is capacity 128, start 2, length 128, SHA-256
`d085299feb54f6010b64b7a7550dbb3b90d4c03c800de9384db0aa2fa36ea338`.
Stage zero overwrites draft template row zero with its computed KV. Template
rows for stages one and two are validated but numerically irrelevant to this
stage-zero result, so they are not a required C handoff payload.

The physical fixture retains 29 distinct Q-A values, 52 Q-B values, 64
distinct head signatures, and all 160 output-A and output-B lanes. Rank and
head permutations, rank-major output flattening, payload-code mutation, and
physical-versus-chronological cache order are all observable. MLX 0.32 Metal
is exact for `C=2`; wrapped `C=128` differs in six attention and six
inverse-RoPE lanes, each one adjacent BF16 code and at most `0.0001220703125`.
This is still synthetic evidence. It does not claim checkpoint-weight parity,
real-model graph parity, or product readiness; those require the native hook
to consume the frozen payload blob and later real-weight captures.

The payload-first FFN fixture continues from that attention handoff without
reducing the product-facing axes: five candidate rows, four HC lanes, hidden
width 4096, 256 routed experts, and top six routing. Only the expert
intermediate width is reduced from 2048 to 256, which is one complete block for
both IQ2_XXS gate/up and Q2_K down storage. The Q8_0 router and shared
gate/up/down matrices, F16 HC function, and F32 norm/HC/router-bias tensors are
serialized and reopened before use. Routed records are constructed directly
as valid IQ2_XXS/Q2_K payloads and decoded lazily; there is no ideal routed
float matrix. Thirty distinct selected records occupy 26,542,080 bytes and
include expert ids 0, 29, 30, and 255. A separate duplicate-row mutation
reduces the five-row route union to 24 experts, so both the 30-record SUPPORT
floor and duplicate reuse are observable.

The FFN dtype schedule is copied as semantics, not implementation, from the
pinned official `inference/model.py` whose SHA-256 is recorded in
`provenance.json`. Stage input, HC-pre, and FFN RMSNorm publish BF16. Routing
promotes activation and logical weight to F32, applies
`sqrt(softplus(logit))`, adds the F32 bias only for selection, and normalizes
the six unbiased probabilities before multiplying by 1.5. The official
`torch.topk` does not define tie order; score-descending/expert-id-ascending is
the deterministic Hebrus/Metal policy. A non-finite row, F32 matmul overflow,
or exact six-way denominator of zero publishes no route (`-1` ids and zero
probabilities/weights), with no artificial denominator floor. Shared and routed
gate/up linear results return from their BF16 weights before F32 clamp/SwiGLU.
Gate has only the upper clamp 10;
up is clamped to `[-10, 10]`. The weighted routed mid and unweighted shared mid
publish BF16 immediately before down; each down result is BF16. Routed expert
results are accumulated in ascending expert-id order in F32, independent of
top-k slot order, shared down is added in F32, and only then does MoE publish
BF16. HC-post publishes the final BF16 four-lane state. Bias removal, a
seven-way exact tie, clamp removal, slot-order accumulation, payload-code
mutation, 30-distinct routing, and duplicate routing are separate negative
controls.

FFN fixture version two makes the asymmetric gate clamp a primary-path
invariant instead of a local-only control. Shared gate `[0,0]` reopens from Q8
as `-127/256`, and shared-down row 67 is one exact Q8 `d=1`, code-127 amplifier
of mid lane zero. Replacing the required upper-only gate clamp with a symmetric
clamp changes one shared-mid lane, 16 shared-down lanes, one final MoE lane and
four HC-post lanes. The shared payload digests and every affected publication
hash are frozen independently; this remains synthetic boundary evidence, not
real-checkpoint parity.

The fixture-v2 system oracle and its MLX 0.32.0 / mlx-metal 0.32.0 cross-check
both pass on the M5 Pro. The shared path, including the changed asymmetric
clamp discriminator, remains exact. The operation-specific ceilings below were
first frozen with fixture v1 and the complete v2 rerun passes those same
ceilings; they are limits, not a claim that every v1 maximum was re-observed.
Hidden/HC/norm and the complete shared path are exact. Eight router
logits differ by at most `0.015625`, their probabilities by
`0.0013456344604492188`, and the 30 normalized route weights by
`0.00004869699478149414`. The explicit pre-down BF16 boundary limits the
propagation to `0.00048828125` in routed mid and `0.001953125` in routed down
and the F32 routed sum; final MoE BF16 and HC-post BF16 are exact. These are
operation-specific synthetic-fixture limits, not checkpoint accuracy claims or
a tolerance for the future native graph.

`schema.json` maps the pinned official Hugging Face config to the canonical
combined-GGUF `dspark.*` records and to the standalone support header, including
its six authenticated source-provenance fields. The target keeps
`general.architecture=deepseek4`; the standalone support uses
`general.architecture=deepseek4-dspark`. It intentionally defines no aliases.
`provenance.json` pins the source revisions and hashes from which the 0731
constants and three-stage inventory were checked. It also pins the three
official source shards containing `mtp.*`, the reference quantizer revision,
and the size/SHA-256 of the inspected third-party support GGUF. That GGUF
predates the final 0731 release and lacks a reproducible source-shard manifest,
so it is explicitly reference-only and rejected for publication; a matching
model name or tensor geometry is not treated as weight provenance.

The generated fixture exercises non-degenerate HC and final-head paths, an
ordered synthetic stage-0-to-1-to-2 chain that shares one immutable `main_x`,
the production raw-cache boundaries, greedy and caller-uniform sampled Markov
feedback, confidence, and speculative sampling. Its expected HC, stage-chain,
final-head, Markov, and cache-boundary values are frozen declarations: the
generator does not evaluate `reference.py`, and a changed lane order,
flattening, stage order, normalization, output projection, or confidence input
does not regenerate a matching answer. The unit suite additionally contains
small hand-derived vectors for sequential feedback, per-position confidence
(including a vector that a cumulative product gets wrong), first/intermediate
rejection, the official numerical floors, residual sampling, and the target
bonus row. The generator imports neither `reference.py`, `metadata.py`, nor
`schema.json`.

The 81-name GGUF inventory is currently a reviewed declaration cross-checked
against the authenticated final file and converter manifest. A genuinely
independent generator from the three pinned source-safetensors headers has not
yet been implemented; that derivation remains pending rather than being
simulated from another copy of the same schema.

Regenerate or check the small synthetic fixture with:

```sh
python3 tools/dspark_oracle/generate_fixtures.py
python3 tools/dspark_oracle/generate_fixtures.py --check
python3 tests/dspark/test_oracle.py
```

To validate the authenticated final support file without reading payloads:

```sh
python3 -m tools.dspark_oracle.support_schema \
  /path/to/DeepSeek-V4-Flash-0731-DSpark-support.gguf
DS4_DSPARK_SUPPORT_GGUF=/path/to/DeepSeek-V4-Flash-0731-DSpark-support.gguf \
  python3 tests/dspark/test_oracle.py
```

The capture fixture keeps layer, token, HC, and hidden axes separate. Each of
the three ordered taps is `[token, 4, 4096]`; decode requires exactly one token
row. A 130-row prompt fixture separately exposes its last frontier tap and its
complete token-major last-128 history. Only after the four-lane mean does the
oracle stack layers 40, 41, and 42 into `[3, 4096]`; `main_proj` applies to
every retained position in that order. Compact generators and closed samples
avoid serializing those full matrices. The tests reject reordered or
duplicated layer identities, mismatched token counts, decode batches, wrong
phases, rank/HC/width aliases, frontier-only prompt seeding, and every pairwise
stage-zero capture swap.

`mlx_optional.py` can cross-check primitive equations on Apple Silicon when
`mlx` is installed. Importing the main oracle never imports MLX, and neither
`mlx` nor `mlx-lm` is a runtime dependency.

The hardware parity lane fails closed unless both `mlx` and `mlx-metal` are
exactly 0.32.0 and the default device is the Metal GPU. The system-Python lane
without MLX remains one explicit optional skip. Device parity uses separate
absolute drift ceilings for each float32 operation. The Markov matmul ceiling
is `1e-4` (measured maximum `9.765625e-05`); confidence projection plus sigmoid
uses `5e-8` (measured maximum `4.42772e-08`); main projection/norm uses `5e-8`
(measured `4.70172e-08`); HC split uses `1e-7` (measured maximum
`6.64673e-08`); HC mean uses `1e-7` (measured zero); and HC
reductions/expansion/final-head use `5e-7` (measured maximum `2.54492e-07`).
Full physical-ring attention uses a separate BF16 ceiling of
`0.000244140625`, measured in six output lanes after the final BF16 boundary.
HC post uses an explicit four-lane reduction:
MLX's generic tiny matmul selected reduced-precision accumulation and produced
`0.0024364` drift, which is not the elementwise production HC equation. The
test reports measured drift on failure. These are fixture-specific Metal
bounds, not permission to accept drift in production logits or sampled output.

## Evidence boundary

The official 0731 inference sources pin the equations and three-stage topology,
but their example `generate.py` invokes target generation only; it does not
exercise `forward_spec`. They are therefore architecture evidence, not a
qualified scheduler or output-parity run. Likewise, the final local GGUF proves
metadata and descriptor identity, not decoded weight numerics.

What is validated now:

- GGUF v3 metadata, all 81 tensor names/types/shapes, real-file offset layout,
  and structural size (but not payload identity in this oracle);
- stage-zero capture concatenation, projection, RMSNorm, and pending/noise layout;
- distinct decode/frontier taps plus full last-128 prompt history after the
  post-HC mean at ordered target layers 40/41/42;
- direct per-stage context `kv_norm(Wkv(main_x))`, absolute-position RoPE tail,
  and model-free official non-RoPE FP8 storage simulation;
- non-degenerate HC split, 20-step Sinkhorn, pre/post, and final-head equations;
- ordered three-stage topology with one shared immutable `main_x`;
- transactional three-ring `[128, 512]` ownership, accepted multi-row commit,
  rollback, wraparound, physical attention order, and no duplicate
  current-frontier row;
- five-row/64-head two-source attention with the pinned 64-row online-softmax,
  per-block BF16 numerator weights, denominator-only sinks, and BF16 output;
- ordered stage-zero attention-half boundaries from HC-pre through HC-post,
  including C=2 and C=128 wrapped physical-ring cases and boundary mutations;
- pending `y[p+1]`, candidate inputs `p+1..p+5`, and proposed outputs
  `p+2..p+6`;
- final-head wiring into greedy/sampled sequential Markov and per-position
  confidence;
- NumPy expected values and MLX 0.32.0 Metal parity on synthetic inputs.

Still requiring real runtime captures:

- target-layer 40/41/42 decode and prompt-frontier prefill values, plus the
  resulting stage-zero `main_x`, from the combined model with real weights;
- every stage's quantized KV/attention, HC, router, selected experts, and output;
- physical raw-cache slots and FP8 bytes across prefill, wrap, skip,
  acceptance, and rollback;
- final base/corrected logits and confidence against the real quantized weights;
- exact greedy/sampled target output, SSD bytes, TPOT, and scheduler economics;
- independent derivation of the 81-name support inventory directly from the
  pinned source-safetensors headers.

For a reproducible isolated device check:

```sh
uv venv /tmp/dspark-mlx-venv --python 3.13
uv pip install --python /tmp/dspark-mlx-venv/bin/python \
  -r tools/dspark_oracle/requirements-mlx.txt
PYTHONDONTWRITEBYTECODE=1 \
  /tmp/dspark-mlx-venv/bin/python tests/dspark/test_oracle.py
```
