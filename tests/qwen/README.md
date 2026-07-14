# Qwen3.6 reference fixtures

`qwen36_tokenizer_chat_golden.json` is collected from the official
`Qwen/Qwen3.6-35B-A3B` tokenizer at the pinned revision recorded in the file.
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
