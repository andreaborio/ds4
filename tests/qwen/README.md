# Qwen3.6 reference fixtures

`qwen36_tokenizer_chat_golden.json` is collected from the official
`Qwen/Qwen3.6-35B-A3B` `tokenizer.json` at the pinned revision recorded in the
file.  That JSON is authoritative for the GGUF `qwen35` pre-tokenizer; the
seven official audio/TTS controls from `tokenizer_config.json` are appended at
their assigned IDs.  Transformers' `Qwen2Tokenizer` is otherwise used only to
render the canonical Jinja chat template before the JSON tokenizer encodes it.
It covers byte-BPE splitting, Unicode, whitespace, code, special tokens,
thinking controls, the canonical chat template, and a tool-call round trip.

Refresh it intentionally with:

```sh
uv run \
  --with 'transformers==5.13.1' \
  --with 'jinja2>=3.1' \
  python tests/qwen/collect_reference.py
```

Verify a checked-in fixture against the pinned source with the same command and
`--check`.  This networked collector is not part of `make model-free-test`;
the eventual C tokenizer tests consume the frozen JSON without contacting the
Hub.

`qwen36_gdn_golden.inc` is a small scalar Gated DeltaNet oracle derived from
the official Transformers fallback equations after GGUF conversion.  It
covers causal convolution, the recurrent delta update, runtime V-head mapping,
gated RMSNorm, Qwen's F32
softmax/top-8 routing and shared-expert gate, plus identical state across
decode and different prefill chunk boundaries.  It has no Python package or
network dependency.  Verify it with:

```sh
python3 tests/qwen/collect_gdn_reference.py --check
```

`qwen36_attention_golden.inc` freezes the other token-mixer path: per-head
Q/K RMSNorm using the converter-shifted weights, the fused per-head `[Q, gate]`
projection layout, text-only split-half partial RoPE, contiguous GQA head
repetition, F32 causal softmax, and the elementwise sigmoid output gate.  It
also proves that prefix and one-shot causal results agree.  Refresh or verify
it with:

```sh
python3 tests/qwen/collect_attention_reference.py --check
```

These scalar fixtures are correctness oracles only.  Passing them does not
enable Qwen CPU, Metal, or SSD-streaming inference; the runtime remains
fail-closed until the complete model graph passes end-to-end logits gates.
`test_v_tiling_contract.py` separately freezes the pinned converter's V-side
permutation, including QKV/conv, Z, controls, and output-projection columns.
It does not execute the converter, bind a GGUF, test quantization, or validate
the complete Qwen graph.
