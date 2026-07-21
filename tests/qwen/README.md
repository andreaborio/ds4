# Qwen3.6 fixtures and ExpertMajor v2 gates

This directory contains frozen Qwen3.6 reference data, reproducible fixture
collectors, and narrow offline checks. Current production inference accepts
only the qualified embedded ExpertMajor v2 artifact on Apple Metal. The support
boundary is defined by
[`RUNTIME_SUPPORT.md`](../../docs/contracts/RUNTIME_SUPPORT.md); the complete
release lane remains
[`QA_BEFORE_RELEASES.md`](../../QA_BEFORE_RELEASES.md).

## Fixture inventory

| File | Purpose | Normal check |
| --- | --- | --- |
| `qwen36_tokenizer_chat_golden.json` | Pinned official tokenizer and rendered-chat vectors | consumed by the collectors and tokenizer tests |
| `qwen36_tokenizer_fixture.inc` | Compact C closure of required symbols, ranked merge candidates, and expected IDs | `make qwen-tokenizer-test` |
| `qwen36_chat_template.jinja` | Byte-exact canonical chat template | `make qwen-metadata-test` |
| `qwen_unicode_ucd_cache.txt` | Frozen Unicode normalization and property data | `make qwen-unicode-test` |
| `qwen36_gdn_golden.inc` | Scalar Gated DeltaNet state/output oracle | `make qwen-reference-test` |
| `qwen36_attention_golden.inc` | Scalar full-attention and causal-prefix oracle | `make qwen-reference-test` |
| `test_v_tiling_contract.py` | Converter V-side permutation contract | `make qwen-reference-test` |
| `test_compare_logits.py` | Offline logits-comparator validation | `python3 tests/qwen/test_compare_logits.py -v` |

The JSON fixture records its model revision and source identity. It covers
byte-BPE splitting, Unicode, whitespace, code, trusted controls, thinking
semantics, canonical chat rendering, tool calls, JSON argument types, grouped
responses, and serialization order. The compact C fixture keeps every ranked
merge candidate encountered by the official path; retaining only winning
merges would not detect an implementation that ignores BPE rank.

User-provided spellings of control tokens are data. The
`untrusted_literal_controls_and_pad_are_data` and `literal_controls_as_data`
cases prove they stay ordinary BPE bytes. Only an already-rendered trusted chat
prompt may produce control IDs. The reference-only rendered vector remains
separate so a fixture cannot accidentally turn untrusted content into syntax.

Unicode provenance, source hashes, licenses, and the pinned version split are
recorded in [`UNICODE_DATA_PROVENANCE.md`](UNICODE_DATA_PROVENANCE.md). The chat
template SHA-256 is
`e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259`.

## Refresh and verification

Refresh the network-derived JSON fixture only as an intentional source update:

```sh
uv run \
  --with 'transformers==5.13.1' \
  --with 'tokenizers==0.22.2' \
  --with 'jinja2==3.1.6' \
  --with 'huggingface-hub==1.23.0' \
  python tests/qwen/collect_reference.py
```

Add `--check` to verify the checked-in JSON against the pinned Hub revision
without rewriting it. This networked collector is intentionally outside
`make model-free-test`.

Verify or regenerate the compact tokenizer closure from the pinned
`tokenizer.json`:

```sh
TOKENIZER_JSON=/absolute/path/to/tokenizer.json
uv run --with 'tokenizers==0.22.2' \
  python tests/qwen/collect_tokenizer_fixture.py \
  --tokenizer-json "$TOKENIZER_JSON" --check
```

Remove `--check` only when the pinned source and the reviewed fixture are being
updated together. Scalar GDN and attention fixtures are deterministic and have
no network dependency:

```sh
python3 tests/qwen/collect_gdn_reference.py --check
python3 tests/qwen/collect_attention_reference.py --check
python3 tests/qwen/test_v_tiling_contract.py
```

Generated fixture changes require source provenance, a reviewed diff, and the
matching C/Python tests. Do not hand-edit generated token IDs or Unicode data.

## Model-free gate

The focused Qwen gate is:

```sh
make qwen-metadata-test \
     qwen-reference-test \
     qwen-unicode-test \
     qwen-tokenizer-test \
     qwen-expert-group-test
python3 tests/qwen/test_compare_logits.py -v
./ds4_test --metal-kernels
```

`make model-free-test` runs these checks as part of the wider repository gate.
The scalar fixtures prove numeric and tokenizer invariants; they do not qualify
a model artifact or a production inference mode. Metal Qwen cases in
`test_qwen35_metal.m` additionally cover resident/SSD top-8 equivalence,
malformed-route fail-closed behavior, cache accounting, and slab growth.

`compare_logits.py` remains an offline evidence checker. Its unit test validates
identity checks, padding exclusion, top-k metrics, and malformed-input failure.
It is not a substitute for the current ExpertMajor v2 model-backed lane.

## Qualified ExpertMajor v2 artifact

Use this exact release identity for every Qwen model-backed gate:

| Field | Value |
| --- | --- |
| Publication state | Published at immutable revision `7bf9c3f7f6136aeb2599d75ee61c0cc2f18e2b02` |
| File | `Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-MLX-Affine4-G64.gguf` |
| Bytes | `20,808,566,880` |
| SHA-256 | `dd17266185833a9f05531ce366fd7284ddca1ed64aa3dcf06e321e8c72c9ea3d` |
| Routed storage | MLX affine4, group size 64 |
| Hugging Face | [`andreaborio/Qwen3.6-35B-A3B-DS4-GGUF`](https://huggingface.co/andreaborio/Qwen3.6-35B-A3B-DS4-GGUF) |

The v2 file contains one routed-weight copy inside its checksummed embedded
store. Canonical Qwen GGUFs are offline converter inputs, not inference
substitutes. Admission validates model family, geometry, layer inventory,
component types, ranges, alignment, and manifest digest before Metal starts.
Format and conversion details are in
[`qwen-expert-major-store.md`](../../docs/qwen-expert-major-store.md).

The converter's full byte-identity verifier is the artifact gate:

```sh
python3 gguf-tools/ds4-expert-major.py inspect CANONICAL-QWEN.gguf
python3 gguf-tools/ds4-expert-major.py verify \
  CANONICAL-QWEN.gguf \
  Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-MLX-Affine4-G64.gguf
```

Never use `--skip-verify` for a release artifact.

## Model-backed Metal gate

Resolve `QWEN_V2` to the absolute path whose size and complete SHA-256 match the
table above. Normal startup is flag-free AUTO:

```sh
QWEN_V2=/absolute/path/to/Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-MLX-Affine4-G64.gguf
./ds4 -m "$QWEN_V2" --ctx 8192 \
  -n 32 --temp 0 \
  -p 'Scrivi solo una breve funzione Python che somma due interi.'
```

Record the admission inputs, resolved residency, cache tier, physical footprint,
and system swap before, during, and after the run. AUTO may choose resident or
SSD according to the current memory and pressure gates.

The policy suite covers named 16/24/32/36/48/64/96/128 GiB profiles. Model-backed
release evidence must additionally identify the real Metal device and its
`recommendedMaxWorkingSetSize`; a simulated profile is not hardware throughput
evidence. At minimum, validate the available lower-memory lanes as follows:

- 16 GiB: AUTO resolves to SSD, pressure is explicitly normal, and swap does
  not increase;
- 32 GiB: AUTO resolves to resident when both logged budgets pass, and falls
  back to SSD rather than forcing residency when either gate fails;
- 64 GiB reference: preserve the existing resident/SSD correctness lanes.

Explicit modes are qualification controls, not the release startup command. On
a host where resident admission succeeds, run the same deterministic prompt in
both modes:

```sh
./ds4 -m "$QWEN_V2" --ctx 8192 --resident \
  -n 32 --temp 0 \
  -p 'Scrivi solo una breve funzione Python che somma due interi.'

./ds4 -m "$QWEN_V2" --ctx 8192 --ssd-streaming \
  -n 32 --temp 0 \
  -p 'Scrivi solo una breve funzione Python che somma due interi.'
```

Compare deterministic output and logits, not plausibility alone. Resident mode
must use complete mapped tensors with zero DS4 expert-cache `pread` accounting.
SSD mode must allocate the first 321-expert slab within its admitted budget and
must not introduce swap. Keep warm page-cache evidence separate from cold
device-I/O evidence, and never bypass a failed admission to obtain a number.

The current affine qualification and interpretation are recorded in
[`2026-07-21-qwen-unified-affine-auto-ssd.md`](../../docs/benchmarks/2026-07-21-qwen-unified-affine-auto-ssd.md).
That dated record is qualification evidence; the immutable repository revision
is the publication manifest. `QA_BEFORE_RELEASES.md` remains authoritative for
the complete release gate.
