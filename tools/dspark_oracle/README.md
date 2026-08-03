# DSpark development oracle

This directory is a model-free numerical oracle for the DeepSeek V4 Flash 0731
DSpark work. It does not participate in model loading or inference. Production
remains the native Hebrus Metal runtime.

The NumPy reference covers the sequential Markov correction, the DSpark
per-position confidence cutoff, and speculative-sampling accept/reject and
residual rules. Random uniforms are fixture inputs, so a failure is
reproducible without controlling a framework RNG.

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

`schema.json` maps the pinned official Hugging Face config to the six canonical
`dspark.*` records preserved by the self-contained GGUF converter; the target
keeps `general.architecture=deepseek4`. It intentionally defines no aliases.
`provenance.json` pins the source revisions and hashes from which the 0731
constants and three-stage inventory were checked. It also pins the three
official source shards containing `mtp.*`, the reference quantizer revision,
and the size/SHA-256 of the inspected third-party support GGUF. That GGUF
predates the final 0731 release and lacks a reproducible source-shard manifest,
so it is explicitly reference-only and rejected for publication; a matching
model name or tensor geometry is not treated as weight provenance.

The generated fixture exercises larger mixed values, while the unit suite also
contains small closed-form vectors whose expected values are written out by
hand: sequential Markov feedback, per-position confidence (including a vector
that a cumulative product gets wrong), first/intermediate rejection, the
official numerical floors, residual sampling, and the target bonus row. Those
vectors do not come from `reference.py` or the fixture generator. The generator
likewise serializes independently declared inputs and expected values: it
imports neither `reference.py`, `metadata.py`, nor `schema.json`.

Regenerate or check the small synthetic fixture with:

```sh
python3 tools/dspark_oracle/generate_fixtures.py
python3 tools/dspark_oracle/generate_fixtures.py --check
python3 tests/dspark/test_oracle.py
```

The capture fixture exercises the production geometry `[3, 4, 4096]`, checks
that NumPy returns independent `[3, 4096]` storage, and rejects rank, HC-lane,
and hidden-width aliases instead of silently reshaping an undersized tensor.

`mlx_optional.py` can cross-check primitive equations on Apple Silicon when
`mlx` is installed. Importing the main oracle never imports MLX, and neither
`mlx` nor `mlx-lm` is a runtime dependency.

The device parity gate uses separate absolute drift ceilings for each float32
operation. The Markov matmul ceiling is `1e-4` (measured maximum
`9.765625e-05`); confidence projection plus sigmoid uses `5e-8` (measured
maximum `3.43827e-08`). The test reports the measured drift on failure. These
are fixture-specific Metal bounds, not permission to accept that drift in
production logits or sampled output.

For a reproducible isolated device check:

```sh
uv venv /tmp/dspark-mlx-venv --python 3.13
uv pip install --python /tmp/dspark-mlx-venv/bin/python \
  -r tools/dspark_oracle/requirements-mlx.txt
PYTHONDONTWRITEBYTECODE=1 \
  /tmp/dspark-mlx-venv/bin/python tests/dspark/test_oracle.py
```
