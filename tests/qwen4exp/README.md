# Qwen4Exp pinned evidence and scalar fixtures

This directory contains model-free evidence for the text-only
`Qwen/Qwen3.8-Flash-Next` profile. It does not make the model runnable or
supported. The support boundary remains
[`RUNTIME_SUPPORT.md`](../../docs/contracts/RUNTIME_SUPPORT.md), and the closed
profile is [`qwen4exp-profile.json`](../../docs/contracts/qwen4exp-profile.json).

## Pinned sources

| Source | Immutable identity |
| --- | --- |
| Hugging Face checkpoint | `Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540` |
| Transformers equations | `huggingface/transformers@42ca97014c85d71a88ad60d55f08cb9fb4d26e2c` |
| Transformers `modeling_qwen4_exp.py` SHA-256 | `91e9b1e9c74efe373cd989fe1974a8fa305f4aad43628dbcbd03dac20437814f` |
| Checkpoint `config.json` SHA-256 | `889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b` |
| Checkpoint `tokenizer.json` SHA-256 | `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` |
| Checkpoint `tokenizer_config.json` SHA-256 | `b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27` |
| Checkpoint `chat_template.jinja` SHA-256 | `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041` |
| Source tensor inventory SHA-256 | `a639efc7a5147b04200e870d7e320335527f4361a8327b137feca2683b1dc434` |

`collect_source_inventory.py` and its fixture close the checkpoint inventory.
`collect_scalar_reference.py --write` imports the pinned Transformers source,
verifies its file digest, and runs its small primitives under Python `3.13.13`,
NumPy `2.4.6`, Torch `2.9.1`, CPU float32, one Torch thread and deterministic
algorithms. Inputs use PCG64 seed `0x4e455854` or explicit edge-case values.
The collector also evaluates an independent NumPy transcription and rejects
upstream captures outside the declared float tolerance. It never imports
Hebrus implementation code, downloads weights, or allocates a full model.

`qwen4exp_scalar_provenance.json` records every array's authority, exact source
operation, shape, dtype, element count, SHA-256 and NumPy comparison policy.
Arrays whose behavior is intentionally stricter than unspecified upstream
behavior are labelled `contract-control`, not misrepresented as Transformers
outputs. These include deterministic equal-score ties, invalid-token uint64
wrap controls, and reset/copy/rewind state transitions.

## Scalar fixture coverage

The generated `qwen4exp_scalar_golden.inc` is a compact C99-friendly fixture.
Its canonical name/type/shape/little-endian-byte digest is:

```text
9564a15b4fff26cc1db7c2e9872c0b033bbd4e8a9d1e644c50e649bd00122406
```

It covers:

- zero-centered RMSNorm with zero and nonzero stored weights, plus the
  conventional sigmoid-gated GDN norm;
- four-stream GR prepare, injection, application, and final mixing;
- upstream GDN causal convolution, controls, recurrent update, and the exact
  tiny analogue of the model's 16-to-48 repeat-interleave (`2` key heads to
  `6` value heads);
- the exact 512-expert router width with ordinary and extreme upstream top-10,
  plus a separate all-equal deterministic contract control;
- QSA complete-group pooling at head width `8`, partial RoPE width `4`,
  per-head ReLU scoring, selection/expansion, tie order, and proof that its
  two-token incomplete tail is appended unchanged;
- PLE SplitMix64 constants, primes/offsets, modular multiplication wraparound,
  bigram/trigram rows, missing history, current-EOS hashing and successor reset;
- PLE signed-square-root gates including exact-zero and near-zero positive and
  negative scores, plus dilation-3/kernel-4 convolution with the full
  nine-position history;
- explicit tiny reset, copy, advance and rewind state-control vectors.

For `ds4_qwen4exp_ple_history`, `token[0]` is the most recent predecessor and
`token[1]` is the second most recent. A current EOS (`248044`) is hashed with
the history that precedes it; committing that EOS resets the successor history
to count zero.

## Refresh and check

Regeneration is an intentional, networked upstream-source operation. Use the
complete pinned environment:

```sh
PYTHONDONTWRITEBYTECODE=1 uv run --python 3.13.13 \
  --with 'numpy==2.4.6' \
  --with 'torch==2.9.1' \
  --with 'transformers @ git+https://github.com/huggingface/transformers.git@42ca97014c85d71a88ad60d55f08cb9fb4d26e2c' \
  python tests/qwen4exp/collect_scalar_reference.py --write
```

Verify the checked-in fixture and provenance offline, without Transformers or
Torch and without changing either file:

```sh
PYTHONDONTWRITEBYTECODE=1 uv run --with 'numpy==2.4.6' \
  python tests/qwen4exp/collect_scalar_reference.py --check
```

The capture refuses any mismatched Python, NumPy, Torch, Transformers version
or modeling-source digest. Offline `--check` parses every C array, verifies its
per-array provenance hash, repeats the independent NumPy checks, and reproduces
the complete fixture and provenance text. Generated files must not be edited
by hand.

## Phase 2 artifact dry run

`gguf-tools/qwen4exp-profile.py --dry-run` reads only this directory's pinned
inventory and the closed profile contract. It maps and byte-accounts all 1,658
source identities exactly once: 96 routed tensors become 144 ExpertMajor
gate/up/down destinations; 1,061 remain dense; 137 belong to PLE; 333 vision
and 31 MTP tensors are explicitly excluded by the base-text policy. It does
not open weight shards, select a release codec, or emit weight payloads.

`ds4.ple_rows.v1` is exercised separately with a 100-row non-production byte
fixture. The test freezes the 512-byte manifest, 64-byte fixed page header,
per-page and whole SHA-256 rules, checked affine lookup, transactional row
read, and atomic writer protocol. Production codec, rows per page and page
stride remain fail-closed profile decisions; see
[`ADR 0009`](../../docs/adr/0009-qwen4exp-ple-store-v1-structure.md).

Run the Phase 2 model-free gates with:

```sh
make qwen4exp-converter-test qwen4exp-ple-store-test \
  expert-store-test qwen4exp-sanitizer-test
```

## Limitations

These vectors exercise semantic primitives at tiny dimensions. They do not
contain checkpoint weights, full-layer intermediates, tokenizer/template
vectors, BF16 rounding, quantization, Metal, SSD I/O, or long-running state.
Their outputs come from the pinned upstream primitives and are independently
checked with NumPy, but they are not a substitute for later checkpoint-backed
layer/logit captures, full-model parity, sanitizer, physical-Metal, memory, and
performance gates.
