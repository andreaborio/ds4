# Qwen3.6 fixtures and ExpertMajor v2 gates

The internal `ds4_test` target retains its historical compatibility name;
runtime examples use the public `hebrus` executable.

This directory contains frozen Qwen3.6 reference data, reproducible fixture
collectors, and narrow offline checks. Current production inference accepts
only the two exact qualified embedded ExpertMajor v2 weight profiles on Apple
Metal: published MLX Affine4 G64 and Q2_K_XL. They share the Qwen graph and
session runtime while retaining codec-specific numeric fixtures. The support
boundary is defined by
[`RUNTIME_SUPPORT.md`](../../docs/contracts/RUNTIME_SUPPORT.md); the complete
release lane remains
[`QA_BEFORE_RELEASES.md`](../../QA_BEFORE_RELEASES.md).

## Fixture inventory

| File | Purpose | Normal check |
| --- | --- | --- |
| `qwen36_tokenizer_chat_golden.json` | Pinned official tokenizer and rendered-chat vectors | consumed by the collectors and tokenizer tests |
| `qwen36_tokenizer_fixture.inc` | Compact C closure of required symbols, ranked merge candidates, and expected IDs | `make qwen-tokenizer-test` |
| `qwen36_chat_template.jinja` | Byte-exact Q2_K_XL chat template | `make qwen-metadata-test` |
| `qwen36_chat_template_affine4.jinja` | Byte-exact published Affine4 chat template | `make qwen-metadata-test` |
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
`55d4931433fe502b794226ee7f4d206a6bdd436ac9f80eb7d8ebb4c639f9ea0c`.

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
     qwen-expert-group-test \
     qwen-iq-metal-kernel-test \
     qwen-q5-metal-kernel-test \
     qwen-gdn-controls-metal-kernel-test \
     qwen-fused-split-q-norm-metal-kernel-test
python3 tests/qwen/test_compare_logits.py -v
./ds4_test --metal-kernels
```

`make model-free-test` runs these checks as part of the wider repository gate.
The scalar fixtures prove numeric and tokenizer invariants; they do not qualify
a model artifact or a production inference mode. Metal Qwen cases in
`test_qwen35_metal.m` additionally cover resident/SSD top-8 equivalence,
malformed-route fail-closed behavior, cache accounting, and slab growth. The
focused Q5_K lane compares both one-row and batched embedding gathers against a
CPU oracle. `test_qwen35_iq_metal.m` independently compares resident and
six-slot SSD matvecs for IQ2_XS, IQ3_XXS, and IQ4_XS against decoded CPU values.
The incremental-growth case admits one slab, denies the next, and proves reuse
without a new buffer; a second case denies the first slab and proves no expert
I/O or allocation fallback. `test_ssd_residency` includes the synthetic 21 GiB
host snapshot. `make qwen-24g-fixture-test` validates the versioned physical
request manifest and prompt hashes.

`compare_logits.py` remains an offline evidence checker. Its unit test validates
identity checks, padding exclusion, top-k metrics, and malformed-input failure.
It is not a substitute for the current ExpertMajor v2 model-backed lane.

## Qualified ExpertMajor v2 artifacts

Use the published Affine4 identity for release and lower-memory gates:

| Field | Value |
| --- | --- |
| Publication state | Published at immutable revision `7bf9c3f7f6136aeb2599d75ee61c0cc2f18e2b02` |
| File | `Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-MLX-Affine4-G64.gguf` |
| Bytes | `20,808,566,880` |
| SHA-256 | `dd17266185833a9f05531ce366fd7284ddca1ed64aa3dcf06e321e8c72c9ea3d` |
| Routed storage | MLX affine4, group size 64 |
| Hugging Face | [`andreaborio/Qwen3.6-35B-A3B-Hebrus-GGUF`](https://huggingface.co/andreaborio/Qwen3.6-35B-A3B-Hebrus-GGUF) |

Use this exact Q2_K_XL identity for the second profile's 64 GiB gates:

| Field | Value |
| --- | --- |
| Publication state | Qualified implementation; immutable publication revision pending |
| File | `Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-Q2_K_XL.gguf` |
| Bytes | `12,290,632,032` |
| SHA-256 | `30c22f70aff0f05986b517ee4ad8fef554a1b5aab6971c9ca09f999566d30143` |
| Routed storage | Exact IQ2_XS/IQ3_XXS/IQ4_XS Q2_K_XL inventory |
| Canonical source SHA-256 | `96b9c0af5c77a4ecaabe3983175112b5ece763261c1ece12b2494b692a70dad7` |
| Embedded payload SHA-256 | `ccc3fbc2405d1dd73f8ac15741b0277514de4f46b80818531297ea9ffa0c6a3c` |

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

python3 gguf-tools/ds4-expert-major.py verify \
  CANONICAL-Q2-K-XL.gguf \
  Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-Q2_K_XL.gguf
```

Never use `--skip-verify` for a release artifact.

## Model-backed Metal gate

Resolve `QWEN_V2` to the published Affine4 path and `QWEN_Q2_V2` to the exact
Q2_K_XL path; verify each size and complete SHA-256 against the tables above.
Normal startup is flag-free AUTO:

```sh
QWEN_V2=/absolute/path/to/Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-MLX-Affine4-G64.gguf
QWEN_Q2_V2=/absolute/path/to/Qwen3.6-35B-A3B-DS4-ExpertMajor-v2-Q2_K_XL.gguf
./hebrus -m "$QWEN_V2" --ctx 8192 \
  -n 32 --temp 0 \
  -p 'Scrivi solo una breve funzione Python che somma due interi.'

./hebrus -m "$QWEN_Q2_V2" --ctx 8192 \
  -n 32 --temp 0 \
  -p 'Scrivi solo una breve funzione Python che somma due interi.'
```

Record the admission inputs, resolved residency, cache tier, physical footprint,
and system swap before, during, and after the run. Through 24 GiB, AUTO must
choose guarded SSD and an explicit resident request must fail. On larger hosts,
AUTO may choose resident or SSD according to the current memory and pressure
gates.

The policy suite covers named 16/24/32/36/48/64/96/128 GiB profiles. Model-backed
release evidence must additionally identify the real Metal device and its
`recommendedMaxWorkingSetSize`; a simulated profile is not hardware throughput
evidence. The lower-memory requirements below currently qualify Affine4 only;
Q2_K_XL must not inherit them from its smaller byte count. At minimum, validate
the available Affine4 lower-memory lanes as follows:

- 16 GiB: AUTO resolves to SSD, pressure is explicitly normal, and swap does
  not increase;
- 24 GiB: AUTO resolves to SSD, an explicit resident request fails, and the
  logged prefill/decode target never exceeds 3,521 experts (about 5.80 GiB for
  this artifact). Each phase entry, including one whose configured budget is
  unchanged, must carry a fresh normal-pressure signal. Every proposed new
  slab (up to 321 experts; the final target tail may be smaller) must receive a
  separate live host/pressure/Metal admission; denied
  growth must freeze at current slab capacity and use eviction/reuse without a
  fresh-buffer fallback. Run the five ordered, hash-verified requests in
  `fixtures/qwen-24g-release-v1.json`. The Sarajevo prompt is a versioned
  qualitative reconstruction, not the missing original incident text. Each
  medium/high natural response must finish normally; each sustained companion
  must generate at least 1,720 tokens, and the final follow-up must complete in
  the same Hebrus Server process. Studio must clamp an oversized persisted
  profile to 16K/8K and remove every non-allowlisted `DS4_*` variable. Require
  no pressure `WARNING`, new swapout, or watchdog `SIGTERM`;
- 32 GiB: AUTO resolves to resident when both logged budgets pass, and falls
  back to SSD rather than forcing residency when either gate fails;
- 64 GiB reference: preserve the existing resident/SSD correctness lanes.

With Hebrus Studio already running on the physical 24 GiB host, execute the
checked-in sequence without putting credentials on the command line:

```sh
HEBRUS_API_KEY=... python3 tests/qwen/run_24g_release_gate.py \
  --output-dir /absolute/private/path/qwen-24g-release-evidence
```

The output directory must not already exist. It contains raw model responses,
so treat it as potentially sensitive evidence and do not commit it.

Explicit modes are qualification controls, not the release startup command. On
a host above 24 GiB where resident admission succeeds, run the same
deterministic prompt in both modes:

```sh
./hebrus -m "$QWEN_V2" --ctx 8192 --resident \
  -n 32 --temp 0 \
  -p 'Scrivi solo una breve funzione Python che somma due interi.'

./hebrus -m "$QWEN_V2" --ctx 8192 --ssd-streaming \
  -n 32 --temp 0 \
  -p 'Scrivi solo una breve funzione Python che somma due interi.'
```

Compare deterministic output and logits, not plausibility alone. Resident mode
must use complete mapped tensors with zero Hebrus expert-cache `pread` accounting.
SSD mode must allocate the first 321-expert slab within its admitted budget and
must not introduce swap. Keep warm page-cache evidence separate from cold
device-I/O evidence, and never bypass a failed admission to obtain a number.

The current affine qualification and interpretation are recorded in
[`2026-07-21-qwen-unified-affine-auto-ssd.md`](../../docs/benchmarks/2026-07-21-qwen-unified-affine-auto-ssd.md).
That dated record is qualification evidence; the immutable repository revision
is the publication manifest. `QA_BEFORE_RELEASES.md` remains authoritative for
the complete release gate.
