# DSpark development oracle

This directory is a model-free numerical oracle for the DeepSeek V4 Flash 0731
DSpark work. It does not participate in model loading or inference. Production
remains the native Hebrus Metal runtime.

The NumPy reference covers the complete model-free DSpark seam: capture of the
three target hidden rows, stage-zero `main_proj` plus `main_norm`, accepted/noise
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

The vanilla Markov head is sequential: position zero uses the accepted token
preceding the block, and every later position uses the token sampled at the
previous draft position. Sampled verification uses
`min(1, p(x) / max(q(x), 1e-8))`; on rejection it samples normalized
`max(p-q, 0)`, falling back to the target row when residual mass is at most
`1e-8`, exactly as in the pinned evaluator. Full acceptance samples the bonus
target row.

The raw-cache oracle pins the final 0731 semantic geometry to three independent
`[128, 512]` rings, not a backend allocation. Every stage owns a different ring
because every stage applies a different `attn_kv` projection to the same
`main_x`. A proposal sees the committed target window, the current
target-derived row, and all five draft-derived rows noncausally. Building that
view is pure: only committing the current target position advances the rings.
Closed boundary samples cover absolute positions 0, 127, 128, and 129 without
serializing complete matrices into the fixture. Physical FP8 packing, RoPE
bytes, and command-buffer ordering remain runtime responsibilities and need
separate captures.

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

The capture fixture exercises the production geometry `[3, 4, 4096]`, checks
that NumPy returns independent `[3, 4096]` storage, and rejects rank, HC-lane,
and hidden-width aliases instead of silently reshaping an undersized tensor.

`mlx_optional.py` can cross-check primitive equations on Apple Silicon when
`mlx` is installed. Importing the main oracle never imports MLX, and neither
`mlx` nor `mlx-lm` is a runtime dependency.

The hardware parity lane fails closed unless both `mlx` and `mlx-metal` are
exactly 0.32.0 and the default device is the Metal GPU. The system-Python lane
without MLX remains one explicit optional skip. Device parity uses separate
absolute drift ceilings for each float32 operation. The Markov matmul ceiling
is `1e-4` (measured maximum `9.765625e-05`); confidence projection plus sigmoid
uses `5e-8` (measured maximum `4.42772e-08`); main projection/norm uses `5e-8`
(measured `1.59703e-08`); HC split uses `1e-7` (measured maximum
`6.64673e-08`); HC mean uses `1e-7` (measured zero); and HC
reductions/expansion/final-head use `5e-7` (measured maximum `2.54492e-07`).
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
- stage-zero capture concatenation, projection, RMSNorm, and noise layout;
- non-degenerate HC split, 20-step Sinkhorn, pre/post, and final-head equations;
- ordered three-stage topology with one shared immutable `main_x`;
- transactional three-ring `[128, 512]` raw-cache ownership and wraparound;
- final-head wiring into greedy/sampled sequential Markov and per-position
  confidence;
- NumPy expected values and MLX 0.32.0 Metal parity on synthetic inputs.

Still requiring real runtime captures:

- target-layer 40/41/42 values and stage-zero `main_x` from the combined model;
- every stage's quantized KV/attention, HC, router, selected experts, and output;
- physical raw-cache slots across prefill, wrap, skip, acceptance, and rollback;
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
