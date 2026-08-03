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
