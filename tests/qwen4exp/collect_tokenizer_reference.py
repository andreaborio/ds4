#!/usr/bin/env python3
"""Generate or verify the pinned Qwen4Exp tokenizer oracle.

Write mode downloads only the six immutable tokenizer/config files, checks
their SHA-256 identities, and executes Transformers' TokenizersBackend over
the pinned tokenizer.json.  Check mode is fully offline and stdlib-only.

Pinned capture environment:
  PYTHONDONTWRITEBYTECODE=1 uv run --python 3.13.13 \
    --with 'tokenizers==0.23.1' --with 'huggingface-hub==1.29.0' \
    --with 'transformers @ git+https://github.com/huggingface/transformers.git@42ca97014c85d71a88ad60d55f08cb9fb4d26e2c' \
    python tests/qwen4exp/collect_tokenizer_reference.py --write
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import inspect
import json
import platform
import struct
import tempfile
import unicodedata
import urllib.request
from dataclasses import dataclass
from pathlib import Path


HERE = Path(__file__).resolve().parent
GOLDEN = HERE / "qwen4exp_tokenizer_golden.json"
PROVENANCE = HERE / "qwen4exp_tokenizer_provenance.json"
C_INCLUDE = HERE / "qwen4exp_tokenizer_golden.inc"

HF_REPOSITORY = "Qwen/Qwen3.8-Flash-Next"
HF_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
TRANSFORMERS_COMMIT = "42ca97014c85d71a88ad60d55f08cb9fb4d26e2c"
PYTHON_VERSION = "3.13.13"
TRANSFORMERS_VERSION = "5.16.0.dev0"
TOKENIZERS_VERSION = "0.23.1"
HUGGINGFACE_HUB_VERSION = "1.29.0"

BASE_VOCAB_SIZE = 248044
TOKENIZER_ID_COUNT = 248077
PHYSICAL_LOGITS_WIDTH = 248320
PRETOKENIZE_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|"
    r"\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
AUTO_QWEN2_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|"
    r"\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

ARTIFACT_FILES = {
    "chat_template.jinja": ("c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041", 8952),
    "config.json": ("889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b", 4745),
    "merges.txt": ("a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d", 3353259),
    "tokenizer.json": ("0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3", 12809320),
    "tokenizer_config.json": ("b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27", 17928),
    "vocab.json": ("ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003", 6722759),
}

IMPLEMENTATION_FILES = {
    "transformers/tokenization_utils_tokenizers.py": (
        "bf921a160f483c7a32973952ed82a08c7d8982f769726bd220933aae2df98de8",
        69842,
    ),
    "transformers/models/qwen2/tokenization_qwen2.py": (
        "fac4e6576bfe2369731be147a4e530f262bdf32f2ac50436f96f0d8bdd2fc628",
        3323,
    ),
}

_ADDED_CONTENT = (
    "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>",
    "<|object_ref_end|>", "<|box_start|>", "<|box_end|>", "<|quad_start|>",
    "<|quad_end|>", "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
    "<|image_pad|>", "<|video_pad|>", "<tool_call>", "</tool_call>",
    "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>",
    "<|repo_name|>", "<|file_sep|>", "<tool_response>", "</tool_response>",
    "<think>", "</think>", "<|audio_start|>", "<|audio_end|>", "<tts_pad>",
    "<tts_text_bos>", "<tts_text_eod>", "<tts_text_bos_single>", "<|audio_pad|>",
)
_NON_SPECIAL_IDS = set(range(248058, 248070))
ADDED_TOKENS = tuple(
    {"id": 248044 + index, "content": content, "special": 248044 + index not in _NON_SPECIAL_IDS}
    for index, content in enumerate(_ADDED_CONTENT)
)
ADDED_IDS = {row["id"] for row in ADDED_TOKENS}
BACKEND_SPECIAL_IDS = [row["id"] for row in ADDED_TOKENS if row["special"]]
TRANSFORMERS_NAMED_SPECIAL_IDS = [248046, 248044, 248070, 248071, 248076, 248056, 248057, 248053, 248054]
AUTO_DIVERGENCE_CONTROLS = [
    {
        "autoTokenizerIds": [62516, 28640],
        "exactTokenizerJsonIds": [149567],
        "inputText": "क्",
        "name": "devanagari_virama",
    },
    {
        "autoTokenizerIds": [62516, 157451],
        "exactTokenizerJsonIds": [151858],
        "inputText": "क्ष",
        "name": "devanagari_conjunct",
    },
]
EXPECTED_DECODE_CONTROLS = [
    {"decoded": "ありますか", "ids": [248043], "name": "base_vocabulary_last", "skipSpecialDecoded": "ありますか"},
    {"decoded": "<|endoftext|>", "ids": [248044], "name": "added_vocabulary_first", "skipSpecialDecoded": ""},
    {"decoded": "<|audio_pad|>", "ids": [248076], "name": "added_vocabulary_last", "skipSpecialDecoded": ""},
    {"decoded": "", "ids": [248077], "name": "first_unassigned_physical_logit", "skipSpecialDecoded": ""},
    {"decoded": "", "ids": [248319], "name": "last_unassigned_physical_logit", "skipSpecialDecoded": ""},
    {
        "decoded": "<|audio_pad|><|endoftext|>",
        "ids": [248076, 248077, 248319, 248044],
        "name": "unassigned_ids_are_decode_silent",
        "skipSpecialDecoded": "",
    },
]


@dataclass
class ClosureTrace:
    final_symbols: set[str]
    merge_candidates: set[tuple[int, str, str]]
    merge_decisions: int = 0


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def text_sha256(text: str) -> str:
    return sha256(text.encode("utf-8"))


def ids_sha256(ids: list[int]) -> str:
    return sha256(b"".join(struct.pack("<I", value) for value in ids))


def json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def payload_sha256(record: dict, field: str) -> str:
    payload = dict(record)
    payload.pop(field, None)
    return sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8"))


def fixture_specs() -> list[dict]:
    unicode_cases = [
        ("ascii", "ascii_basic", "The quick brown fox jumps over 13 lazy dogs."),
        ("ascii", "ascii_contractions", "I'm sure we'll test Qwen's tokenizer; won't we?"),
        ("nfc", "accent_nfc", "Café déjà vu — naïve façade."),
        ("nfd", "accent_nfd", "Cafe\u0301 de\u0301ja\u0300 vu — nai\u0308ve fac\u0327ade."),
        ("cjk", "cjk_mixed", "你好，世界。漢字かな交じり文。"),
        ("arabic", "arabic_marks_digits", "مَرْحَبًا بالعالم! العربية ١٢٣"),
        ("emoji_astral", "emoji_astral", "🙂 🚀 👩🏽‍💻 🧑‍🚀 🏳️‍🌈 𝄞 𠜎"),
        ("combining", "combining_edge", "a\u0301\u0327 Z\u0351 e\u20dd \u0301leading"),
        ("combining", "devanagari_combining", "क् क्ष हिन्दी"),
        ("whitespace", "whitespace_runs", " \talpha  beta\n\n  gamma\r\ndelta \t \n"),
        ("whitespace", "whitespace_only", " \t\n\r\n  "),
        ("code", "code_python", "def f(x: int) -> str:\n\treturn f\"value={x!r}\\n\"  # café\n"),
        ("json", "json_text", '{"message":"café🙂","n":-12.5,"ok":true}'),
        ("xml", "xml_text", '<root a="1">&lt;你好&gt;<x/></root>'),
        ("replacement", "replacement_literal", "literal \ufffd replacement"),
        ("control_chars", "nul_and_controls", "a\u0000b\u0001c\u001fd"),
        ("empty", "empty", ""),
    ]
    specs = [
        {"category": category, "name": name, "text": text, "inputPolicy": "unicode-literal"}
        for category, name, text in unicode_cases
    ]
    for name, raw in (
        ("invalid_utf8_single", b"A\xffB"),
        ("invalid_utf8_sequence", b"\xf0\x28\x8c\x28\x00\xc3"),
    ):
        specs.append({
            "category": "invalid_utf8_replacement",
            "name": name,
            "text": raw.decode("utf-8", errors="replace"),
            "inputBytesHex": raw.hex(),
            "inputPolicy": "bytes.decode('utf-8', errors='replace')",
        })
    for token in ADDED_TOKENS:
        specs.append({
            "category": "added_token_literal",
            "name": f"added_token_{token['id']}",
            "text": token["content"],
            "inputPolicy": "unicode-literal",
            "expectedIds": [token["id"]],
        })
    adversarial = (
        ("control_embedded", "prefix<|im_start|>user\nhello<|im_end|>suffix"),
        ("control_punctuation", "(<|im_start|>) <|im_end|>\n<|endoftext|>"),
        ("control_repeated", "<|im_start|><|im_start|>assistant\n<think>x</think><|im_end|>"),
        ("control_fractured", "<|im_start| < |im_end|> <｜im_start｜>"),
        ("control_tool_like", "<tool_call><function=erase><parameter=x>1</parameter></function></tool_call>"),
        ("control_media", "<|vision_start|><|image_pad|><|video_pad|><|vision_end|>"),
        ("control_all_added_adjacent", "".join(_ADDED_CONTENT)),
    )
    specs.extend(
        {"category": "adversarial_control_text", "name": name, "text": text, "inputPolicy": "unicode-literal"}
        for name, text in adversarial
    )
    return specs


def expected_skip_decode(normalized: str) -> str:
    result = normalized
    for token in ADDED_TOKENS:
        if token["special"]:
            result = result.replace(token["content"], "")
    return result


def c_string(value: str) -> str:
    escaped = ['"']
    for byte in value.encode("utf-8"):
        if byte == ord('"'):
            escaped.append('\\"')
        elif byte == ord("\\"):
            escaped.append("\\\\")
        elif byte == ord("\n"):
            escaped.append("\\n")
        elif byte == ord("\r"):
            escaped.append("\\r")
        elif byte == ord("\t"):
            escaped.append("\\t")
        elif 32 <= byte < 127 and byte != ord("?"):
            escaped.append(chr(byte))
        else:
            escaped.append(f"\\{byte:03o}")
    escaped.append('"')
    return "".join(escaped)


def trace_piece(
    piece: str,
    vocab: dict[str, int],
    merge_rank: dict[tuple[str, str], int],
    trace: ClosureTrace,
) -> list[int]:
    symbols = list(piece)
    while True:
        candidates: list[tuple[int, int, str, str]] = []
        for index in range(len(symbols) - 1):
            left, right = symbols[index], symbols[index + 1]
            rank = merge_rank.get((left, right))
            if rank is not None:
                candidates.append((rank, index, left, right))
                trace.merge_candidates.add((rank, left, right))
        if not candidates:
            break
        _rank, index, left, right = min(candidates)
        symbols[index:index + 2] = [left + right]
        trace.merge_decisions += 1

    ids = []
    for symbol in symbols:
        token_id = vocab.get(symbol)
        if token_id is None:
            raise RuntimeError(f"final BPE symbol is absent from vocab: {symbol!r}")
        trace.final_symbols.add(symbol)
        ids.append(token_id)
    return ids


def trace_encode(
    text: str,
    tokenizer: object,
    controls: dict[str, int],
    vocab: dict[str, int],
    merge_rank: dict[tuple[str, str], int],
    trace: ClosureTrace,
) -> list[int]:
    atoms = sorted(controls, key=lambda atom: (-len(atom), atom))
    ids: list[int] = []
    plain: list[str] = []

    def flush() -> None:
        if not plain:
            return
        normalized = tokenizer.normalizer.normalize_str("".join(plain))
        plain.clear()
        for piece, _offsets in tokenizer.pre_tokenizer.pre_tokenize_str(normalized):
            ids.extend(trace_piece(piece, vocab, merge_rank, trace))

    position = 0
    while position < len(text):
        atom = next((item for item in atoms if text.startswith(item, position)), None)
        if atom is None:
            plain.append(text[position])
            position += 1
        else:
            flush()
            ids.append(controls[atom])
            position += len(atom)
    flush()
    return ids


def build_c_include(
    golden: dict,
    tokenizer_json: dict,
    exact_tokenizer: object,
    raw_tokenizer: object,
) -> tuple[str, dict]:
    model = tokenizer_json["model"]
    vocab = {str(text): int(token_id) for text, token_id in model["vocab"].items()}
    merge_rank: dict[tuple[str, str], int] = {}
    for rank, merge in enumerate(model["merges"]):
        parts = merge.split(" ")
        if len(parts) != 2:
            raise RuntimeError(f"invalid BPE merge at rank {rank}: {merge!r}")
        pair = (parts[0], parts[1])
        if pair in merge_rank:
            raise RuntimeError(f"duplicate BPE merge pair at rank {rank}")
        merge_rank[pair] = rank

    controls = {row["content"]: row["id"] for row in ADDED_TOKENS}
    trace = ClosureTrace(set(), set())
    cases = []
    for case in golden["textCases"]:
        text = case["inputText"]
        trusted_ids = list(case["ids"])
        raw_ids = raw_tokenizer.encode(text, add_special_tokens=False).ids
        if exact_tokenizer.encode(text, add_special_tokens=False).ids != trusted_ids:
            raise RuntimeError(f"{case['name']}: exact tokenizer drift while building C closure")
        if trace_encode(text, exact_tokenizer, controls, vocab, merge_rank, trace) != trusted_ids:
            raise RuntimeError(f"{case['name']}: trusted closure trace mismatch")
        if trace_encode(text, raw_tokenizer, {}, vocab, merge_rank, trace) != raw_ids:
            raise RuntimeError(f"{case['name']}: raw closure trace mismatch")
        if any(token_id in ADDED_IDS for token_id in raw_ids):
            raise RuntimeError(f"{case['name']}: raw client text synthesized an added-token ID")
        cases.append({
            "decoded": case["decoded"],
            "name": case["name"],
            "raw": raw_ids,
            "text": text,
            "trusted": trusted_ids,
        })

    token_map = {symbol: vocab[symbol] for symbol in trace.final_symbols}
    for text, token_id in controls.items():
        if token_map.setdefault(text, token_id) != token_id:
            raise RuntimeError(f"conflicting special-token ID for {text!r}")
    vocab_by_id = {token_id: text for text, token_id in vocab.items()}
    for control in golden["decodeControls"]:
        for token_id in control["ids"]:
            if token_id < BASE_VOCAB_SIZE:
                text = vocab_by_id[token_id]
                if token_map.setdefault(text, token_id) != token_id:
                    raise RuntimeError(
                        f"conflicting decode-control token ID {token_id}")
    tokens = sorted(token_map.items(), key=lambda item: (item[1], item[0]))
    merges = sorted(trace.merge_candidates)
    trusted_id_count = sum(len(case["trusted"]) for case in cases)
    raw_id_count = sum(len(case["raw"]) for case in cases)

    lines = [
        "/* Generated by tests/qwen4exp/collect_tokenizer_reference.py; do not edit. */",
        "#define Q4E_TOKENIZER_FIXTURE_REPOSITORY " + c_string(HF_REPOSITORY),
        "#define Q4E_TOKENIZER_FIXTURE_REVISION " + c_string(HF_REVISION),
        "#define Q4E_TOKENIZER_FIXTURE_TOKENIZER_SHA256 " + c_string(ARTIFACT_FILES["tokenizer.json"][0]),
        "#define Q4E_TOKENIZER_FIXTURE_GOLDEN_PAYLOAD_SHA256 " + c_string(golden["fixturePayloadSha256"]),
        "#define Q4E_TOKENIZER_FIXTURE_TOKENIZERS_VERSION " + c_string(TOKENIZERS_VERSION),
        f"#define Q4E_TOKENIZER_FIXTURE_TOKEN_COUNT {len(tokens)}u",
        f"#define Q4E_TOKENIZER_FIXTURE_MERGE_COUNT {len(merges)}u",
        f"#define Q4E_TOKENIZER_FIXTURE_CASE_COUNT {len(cases)}u",
        f"#define Q4E_TOKENIZER_FIXTURE_TRUSTED_ID_COUNT {trusted_id_count}u",
        f"#define Q4E_TOKENIZER_FIXTURE_RAW_ID_COUNT {raw_id_count}u",
        f"#define Q4E_TOKENIZER_FIXTURE_MERGE_DECISIONS {trace.merge_decisions}u",
        "",
        "typedef struct { const char *text; uint32_t len; int id; } q4e_fixture_token;",
        "static const q4e_fixture_token q4e_fixture_tokens[] = {",
    ]
    for text, token_id in tokens:
        lines.append(f"    {{ {c_string(text)}, {len(text.encode('utf-8'))}u, {token_id} }},")
    lines += [
        "};",
        "",
        "typedef struct { const char *text; uint32_t len; int rank; } q4e_fixture_merge;",
        "static const q4e_fixture_merge q4e_fixture_merges[] = {",
    ]
    for rank, left, right in merges:
        text = f"{left} {right}"
        lines.append(f"    {{ {c_string(text)}, {len(text.encode('utf-8'))}u, {rank} }},")
    lines += ["};", ""]

    for index, case in enumerate(cases):
        trusted = ", ".join(str(value) for value in case["trusted"]) or "0"
        raw = ", ".join(str(value) for value in case["raw"]) or "0"
        lines.append(f"static const int q4e_fixture_trusted_{index}[] = {{ {trusted} }};")
        lines.append(f"static const int q4e_fixture_raw_{index}[] = {{ {raw} }};")

    lines += [
        "",
        "typedef struct {",
        "    const char *name;",
        "    const char *text;",
        "    uint32_t text_len;",
        "    const int *trusted_ids;",
        "    uint32_t trusted_len;",
        "    const int *raw_ids;",
        "    uint32_t raw_len;",
        "    const char *decoded;",
        "    uint32_t decoded_len;",
        "} q4e_fixture_case;",
        "static const q4e_fixture_case q4e_fixture_cases[] = {",
    ]
    for index, case in enumerate(cases):
        lines.append(
            "    { " + c_string(case["name"]) + ", " + c_string(case["text"]) +
            f", {len(case['text'].encode('utf-8'))}u, q4e_fixture_trusted_{index}, " +
            f"{len(case['trusted'])}u, q4e_fixture_raw_{index}, {len(case['raw'])}u, " +
            c_string(case["decoded"]) + f", {len(case['decoded'].encode('utf-8'))}u }},"
        )
    lines += ["};", ""]

    for index, control in enumerate(golden["decodeControls"]):
        values = ", ".join(str(value) for value in control["ids"])
        lines.append(f"static const int q4e_fixture_decode_ids_{index}[] = {{ {values} }};")
    lines += [
        "",
        "typedef struct {",
        "    const char *name; const int *ids; uint32_t ids_len;",
        "    const char *decoded; uint32_t decoded_len;",
        "} q4e_fixture_decode_control;",
        "static const q4e_fixture_decode_control q4e_fixture_decode_controls[] = {",
    ]
    for index, control in enumerate(golden["decodeControls"]):
        lines.append(
            "    { " + c_string(control["name"]) + f", q4e_fixture_decode_ids_{index}, " +
            f"{len(control['ids'])}u, " + c_string(control["decoded"]) +
            f", {len(control['decoded'].encode('utf-8'))}u }},"
        )
    lines += ["};", ""]
    rendered = "\n".join(lines)
    stats = {
        "caseCount": len(cases),
        "mergeCandidateCount": len(merges),
        "mergeDecisionCount": trace.merge_decisions,
        "rawIdCount": raw_id_count,
        "tokenCount": len(tokens),
        "trustedIdCount": trusted_id_count,
    }
    return rendered, stats


def fetch_artifacts(directory: Path) -> None:
    for name in ARTIFACT_FILES:
        url = f"https://huggingface.co/{HF_REPOSITORY}/resolve/{HF_REVISION}/{name}"
        request = urllib.request.Request(url, headers={"User-Agent": "hebrus-qwen4exp-tokenizer-oracle/1"})
        with urllib.request.urlopen(request, timeout=120) as response:
            (directory / name).write_bytes(response.read())


def verify_artifacts(directory: Path) -> tuple[dict, dict, dict]:
    verified = {}
    for name, (expected_hash, expected_bytes) in ARTIFACT_FILES.items():
        body = (directory / name).read_bytes()
        if len(body) != expected_bytes or sha256(body) != expected_hash:
            raise RuntimeError(f"{name}: pinned byte identity mismatch")
        verified[name] = {"bytes": len(body), "sha256": sha256(body)}

    tokenizer_json = json.loads((directory / "tokenizer.json").read_text())
    tokenizer_config = json.loads((directory / "tokenizer_config.json").read_text())
    model_config = json.loads((directory / "config.json").read_text())
    compact_added = [
        {"id": row["id"], "content": row["content"], "special": row["special"]}
        for row in tokenizer_json["added_tokens"]
    ]
    if compact_added != list(ADDED_TOKENS):
        raise RuntimeError("tokenizer.json: added-token mapping drift")
    pre = tokenizer_json["pre_tokenizer"]
    if tokenizer_json["normalizer"] != {"type": "NFC"}:
        raise RuntimeError("tokenizer.json: expected NFC normalizer")
    if pre["pretokenizers"][0]["pattern"] != {"Regex": PRETOKENIZE_REGEX}:
        raise RuntimeError("tokenizer.json: pretokenize regex drift")
    if tokenizer_json["model"]["type"] != "BPE" or len(tokenizer_json["model"]["vocab"]) != BASE_VOCAB_SIZE:
        raise RuntimeError("tokenizer.json: base BPE vocabulary drift")
    if tokenizer_config["pretokenize_regex"] != PRETOKENIZE_REGEX:
        raise RuntimeError("tokenizer_config.json: pretokenize regex drift")
    if tokenizer_config["model_max_length"] != 262144 or tokenizer_config["tokenizer_class"] != "Qwen2Tokenizer":
        raise RuntimeError("tokenizer_config.json: tokenizer contract drift")
    if tokenizer_config["chat_template"] != (directory / "chat_template.jinja").read_text():
        raise RuntimeError("chat template file/config copies disagree")
    if model_config["text_config"]["vocab_size"] != PHYSICAL_LOGITS_WIDTH:
        raise RuntimeError("config.json: physical vocabulary width drift")
    return verified, tokenizer_json, tokenizer_config


def capture(source_dir: Path) -> tuple[dict, dict, str]:
    import huggingface_hub
    import tokenizers
    import transformers
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    from transformers.models.qwen2 import tokenization_qwen2
    from transformers import tokenization_utils_tokenizers

    versions = {
        "huggingfaceHub": huggingface_hub.__version__,
        "python": platform.python_version(),
        "tokenizers": tokenizers.__version__,
        "transformers": transformers.__version__,
    }
    expected_versions = {
        "huggingfaceHub": HUGGINGFACE_HUB_VERSION,
        "python": PYTHON_VERSION,
        "tokenizers": TOKENIZERS_VERSION,
        "transformers": TRANSFORMERS_VERSION,
    }
    if versions != expected_versions:
        raise RuntimeError(f"pinned capture environment required: expected {expected_versions}, found {versions}")

    module_by_key = {
        "transformers/tokenization_utils_tokenizers.py": tokenization_utils_tokenizers,
        "transformers/models/qwen2/tokenization_qwen2.py": tokenization_qwen2,
    }
    implementation = {}
    for key, module in module_by_key.items():
        path = Path(inspect.getsourcefile(module) or "")
        body = path.read_bytes()
        expected_hash, expected_bytes = IMPLEMENTATION_FILES[key]
        if len(body) != expected_bytes or sha256(body) != expected_hash:
            raise RuntimeError(f"{key}: installed pinned implementation mismatch")
        implementation[key] = {"bytes": len(body), "sha256": sha256(body)}

    files, tokenizer_json, _tokenizer_config = verify_artifacts(source_dir)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        source_dir, local_files_only=True, trust_remote_code=False
    )
    direct = Tokenizer.from_file(str(source_dir / "tokenizer.json"))
    tokenizer_json_without_added = dict(tokenizer_json)
    tokenizer_json_without_added["added_tokens"] = []
    raw_direct = Tokenizer.from_str(json.dumps(
        tokenizer_json_without_added,
        ensure_ascii=False,
        separators=(",", ":"),
    ))
    if tokenizer.__class__.__module__ != "transformers.tokenization_utils_tokenizers":
        raise RuntimeError(f"unexpected authoritative backend {tokenizer.__class__}")
    if tokenizer.vocab_size != BASE_VOCAB_SIZE or len(tokenizer) != TOKENIZER_ID_COUNT:
        raise RuntimeError("Transformers tokenizer vocabulary boundary drift")
    if max(tokenizer.get_vocab().values()) != TOKENIZER_ID_COUNT - 1:
        raise RuntimeError("Transformers tokenizer maximum ID drift")
    if tokenizer.get_added_vocab() != {row["content"]: row["id"] for row in ADDED_TOKENS}:
        raise RuntimeError("Transformers tokenizer added vocabulary drift")

    cases = []
    for spec in fixture_specs():
        text = spec["text"]
        ids = list(tokenizer.encode(text, add_special_tokens=False))
        direct_ids = direct.encode(text, add_special_tokens=False).ids
        decoded = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        skipped = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if ids != direct_ids or decoded != direct.decode(ids, skip_special_tokens=False):
            raise RuntimeError(f"{spec['name']}: Transformers wrapper/direct tokenizer.json disagreement")
        if skipped != direct.decode(ids, skip_special_tokens=True):
            raise RuntimeError(f"{spec['name']}: skip-special wrapper/direct disagreement")
        normalized = tokenizer.backend_tokenizer.normalizer.normalize_str(text)
        if decoded != normalized or skipped != expected_skip_decode(normalized):
            raise RuntimeError(f"{spec['name']}: decode/normalization invariant failed")
        if "expectedIds" in spec and ids != spec["expectedIds"]:
            raise RuntimeError(f"{spec['name']}: added-token literal did not map atomically")
        if any(value >= TOKENIZER_ID_COUNT for value in ids):
            raise RuntimeError(f"{spec['name']}: encoder produced an unassigned ID")
        case = {
            "category": spec["category"],
            "decoded": decoded,
            "decodedUtf8Hex": decoded.encode("utf-8").hex(),
            "ids": ids,
            "inputPolicy": spec["inputPolicy"],
            "inputText": text,
            "inputUtf8Hex": text.encode("utf-8").hex(),
            "name": spec["name"],
            "normalizedText": normalized,
            "skipSpecialDecoded": skipped,
        }
        if "inputBytesHex" in spec:
            case["inputBytesHex"] = spec["inputBytesHex"]
        cases.append(case)

    decode_controls = []
    for name, ids in (
        ("base_vocabulary_last", [248043]),
        ("added_vocabulary_first", [248044]),
        ("added_vocabulary_last", [248076]),
        ("first_unassigned_physical_logit", [248077]),
        ("last_unassigned_physical_logit", [248319]),
        ("unassigned_ids_are_decode_silent", [248076, 248077, 248319, 248044]),
    ):
        decode_controls.append({
            "decoded": tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False),
            "ids": ids,
            "name": name,
            "skipSpecialDecoded": tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False),
        })
    if decode_controls != EXPECTED_DECODE_CONTROLS:
        raise RuntimeError(f"pinned decode-boundary controls changed: {decode_controls}")

    golden = {
        "decodeControls": decode_controls,
        "fixturePayloadSha256": "",
        "identity": {
            "hfRepository": HF_REPOSITORY,
            "hfRevision": HF_REVISION,
            "transformersCommit": TRANSFORMERS_COMMIT,
        },
        "kind": "qwen4exp-tokenizer-golden",
        "schemaVersion": 1,
        "textCases": cases,
        "tokenizerContract": {
            "addPrefixSpace": False,
            "addedTokenCount": len(ADDED_TOKENS),
            "addedTokens": list(ADDED_TOKENS),
            "baseVocabularySize": BASE_VOCAB_SIZE,
            "backendSpecialTokenIds": BACKEND_SPECIAL_IDS,
            "effectiveValidVocabularySize": TOKENIZER_ID_COUNT,
            "maximumValidTokenId": TOKENIZER_ID_COUNT - 1,
            "modelMaxLength": 262144,
            "normalizer": "NFC",
            "physicalLogitsWidth": PHYSICAL_LOGITS_WIDTH,
            "pretokenizeRegex": PRETOKENIZE_REGEX,
            "transformersNamedSpecialTokenIds": TRANSFORMERS_NAMED_SPECIAL_IDS,
            "unassignedPhysicalIdCount": PHYSICAL_LOGITS_WIDTH - TOKENIZER_ID_COUNT,
            "unassignedPhysicalIdRangeHalfOpen": [TOKENIZER_ID_COUNT, PHYSICAL_LOGITS_WIDTH],
        },
    }
    golden["fixturePayloadSha256"] = payload_sha256(golden, "fixturePayloadSha256")
    c_include, c_stats = build_c_include(
        golden, tokenizer_json, direct, raw_direct)

    # AutoTokenizer is intentionally only a divergence control.  At this pin it
    # reconstructs Qwen2 from vocab/merges and loses the model's new \p{M} regex.
    auto = AutoTokenizer.from_pretrained(source_dir, local_files_only=True, trust_remote_code=False)
    divergence = []
    for name, text in (("devanagari_virama", "क्"), ("devanagari_conjunct", "क्ष")):
        exact_ids = list(tokenizer.encode(text, add_special_tokens=False))
        auto_ids = list(auto.encode(text, add_special_tokens=False))
        if exact_ids == auto_ids:
            raise RuntimeError(f"{name}: expected pinned AutoTokenizer divergence disappeared")
        divergence.append({
            "autoTokenizerIds": auto_ids,
            "exactTokenizerJsonIds": exact_ids,
            "inputText": text,
            "name": name,
        })
    if divergence != AUTO_DIVERGENCE_CONTROLS:
        raise RuntimeError(f"pinned AutoTokenizer divergence changed: {divergence}")

    category_counts = dict(sorted(collections.Counter(case["category"] for case in cases).items()))
    golden_text = json_text(golden)
    provenance_cases = []
    for case in cases:
        provenance_cases.append({
            "authority": "pinned-transformers-tokenizers-backend",
            "category": case["category"],
            "containsAddedTokenIds": [value for value in case["ids"] if value in ADDED_IDS],
            "decodedSha256": text_sha256(case["decoded"]),
            "idsLittleEndianUint32Sha256": ids_sha256(case["ids"]),
            "inputPolicy": case["inputPolicy"],
            "inputUtf8Sha256": text_sha256(case["inputText"]),
            "name": case["name"],
            "skipSpecialDecodedSha256": text_sha256(case["skipSpecialDecoded"]),
        })
    provenance_decode = [
        {
            "authority": "pinned-transformers-tokenizers-backend",
            "decodedSha256": text_sha256(case["decoded"]),
            "idsLittleEndianUint32Sha256": ids_sha256(case["ids"]),
            "name": case["name"],
            "skipSpecialDecodedSha256": text_sha256(case["skipSpecialDecoded"]),
        }
        for case in decode_controls
    ]
    provenance = {
        "capture": {
            "addSpecialTokens": False,
            "authoritativeClass": "transformers.tokenization_utils_tokenizers.TokenizersBackend",
            "cleanUpTokenizationSpaces": False,
            "directTokenizerJsonCrossCheck": True,
            "localFilesOnly": True,
            "normalization": "NFC",
            "skipSpecialTokensCaptured": [False, True],
        },
        "cases": provenance_cases,
        "decodeControls": provenance_decode,
        "environment": versions,
        "fixture": {
            "cIncludeFileSha256": sha256(c_include.encode("utf-8")),
            "cIncludePath": "tests/qwen4exp/qwen4exp_tokenizer_golden.inc",
            "cIncludeStats": c_stats,
            "categoryCounts": category_counts,
            "decodeControlCount": len(decode_controls),
            "goldenFileSha256": sha256(golden_text.encode("utf-8")),
            "goldenPayloadSha256": golden["fixturePayloadSha256"],
            "path": "tests/qwen4exp/qwen4exp_tokenizer_golden.json",
            "addedTokenLiteralCount": len(ADDED_TOKENS),
            "backendSpecialLiteralCount": len(BACKEND_SPECIAL_IDS),
            "textCaseCount": len(cases),
        },
        "implementationFiles": implementation,
        "kind": "qwen4exp-tokenizer-provenance",
        "knownAutoTokenizerDivergence": {
            "authoritativeForGolden": False,
            "autoTokenizerClass": f"{auto.__class__.__module__}.{auto.__class__.__name__}",
            "autoTokenizerRegex": AUTO_QWEN2_REGEX,
            "controls": divergence,
            "reason": "Pinned AutoTokenizer reconstructs Qwen2 from vocab/merges and omits combining marks from the model's new tokenizer.json regex.",
        },
        "schemaVersion": 1,
        "sourceFiles": files,
        "sourcePins": {
            "hfRepository": HF_REPOSITORY,
            "hfRevision": HF_REVISION,
            "transformersCommit": TRANSFORMERS_COMMIT,
        },
        "validVocabularyBoundary": {
            "effectiveValidVocabularySize": TOKENIZER_ID_COUNT,
            "maximumValidTokenId": TOKENIZER_ID_COUNT - 1,
            "physicalLogitsWidth": PHYSICAL_LOGITS_WIDTH,
            "samplingRule": "sample IDs in [0, 248077); mask physical logits IDs in [248077, 248320)",
        },
    }
    return golden, provenance, c_include


def verify_offline(
    golden_path: Path,
    provenance_path: Path,
    include_path: Path,
) -> tuple[dict, dict]:
    golden_text = golden_path.read_text()
    provenance_text = provenance_path.read_text()
    golden = json.loads(golden_text)
    provenance = json.loads(provenance_text)
    if golden_text != json_text(golden) or provenance_text != json_text(provenance):
        raise RuntimeError("generated JSON is not canonical")
    if golden.get("fixturePayloadSha256") != payload_sha256(golden, "fixturePayloadSha256"):
        raise RuntimeError("golden payload SHA-256 mismatch")
    if golden.get("identity") != {
        "hfRepository": HF_REPOSITORY,
        "hfRevision": HF_REVISION,
        "transformersCommit": TRANSFORMERS_COMMIT,
    } or golden.get("kind") != "qwen4exp-tokenizer-golden" or golden.get("schemaVersion") != 1:
        raise RuntimeError("golden identity/schema drift")
    contract = golden.get("tokenizerContract", {})
    if contract.get("addedTokens") != list(ADDED_TOKENS):
        raise RuntimeError("added-token mapping drift")
    if contract.get("pretokenizeRegex") != PRETOKENIZE_REGEX:
        raise RuntimeError("pretokenize regex drift")
    if contract.get("effectiveValidVocabularySize") != TOKENIZER_ID_COUNT or contract.get("physicalLogitsWidth") != PHYSICAL_LOGITS_WIDTH:
        raise RuntimeError("valid-vocabulary boundary drift")
    if contract.get("backendSpecialTokenIds") != BACKEND_SPECIAL_IDS or contract.get("transformersNamedSpecialTokenIds") != TRANSFORMERS_NAMED_SPECIAL_IDS:
        raise RuntimeError("special-token classification drift")
    if golden.get("decodeControls") != EXPECTED_DECODE_CONTROLS:
        raise RuntimeError("decode-boundary controls drift")
    specs = fixture_specs()
    cases = golden.get("textCases", [])
    if len(cases) != len(specs):
        raise RuntimeError("text case count drift")
    for spec, case in zip(specs, cases, strict=True):
        if (case["name"], case["category"], case["inputText"], case["inputPolicy"]) != (
            spec["name"], spec["category"], spec["text"], spec["inputPolicy"]
        ):
            raise RuntimeError(f"{spec['name']}: case input/provenance drift")
        normalized = unicodedata.normalize("NFC", spec["text"])
        if case["normalizedText"] != normalized or case["decoded"] != normalized:
            raise RuntimeError(f"{spec['name']}: normalization/decode drift")
        if case["skipSpecialDecoded"] != expected_skip_decode(normalized):
            raise RuntimeError(f"{spec['name']}: skip-special decode drift")
        if case["inputUtf8Hex"] != spec["text"].encode("utf-8").hex() or case["decodedUtf8Hex"] != normalized.encode("utf-8").hex():
            raise RuntimeError(f"{spec['name']}: UTF-8 byte fixture drift")
        if "expectedIds" in spec and case["ids"] != spec["expectedIds"]:
            raise RuntimeError(f"{spec['name']}: atomic added-token ID drift")
        if any(not isinstance(value, int) or value < 0 or value >= TOKENIZER_ID_COUNT for value in case["ids"]):
            raise RuntimeError(f"{spec['name']}: encoded ID outside effective vocabulary")

    if provenance.get("fixture", {}).get("goldenFileSha256") != sha256(golden_text.encode("utf-8")):
        raise RuntimeError("provenance golden-file SHA-256 mismatch")
    if provenance["fixture"]["goldenPayloadSha256"] != golden["fixturePayloadSha256"]:
        raise RuntimeError("provenance golden-payload SHA-256 mismatch")
    include_body = include_path.read_bytes()
    if provenance["fixture"].get("cIncludeFileSha256") != sha256(include_body):
        raise RuntimeError("compact C tokenizer closure SHA-256 mismatch")
    include_text = include_body.decode("utf-8")
    if (
        f"#define Q4E_TOKENIZER_FIXTURE_CASE_COUNT {len(cases)}u" not in include_text
        or f"#define Q4E_TOKENIZER_FIXTURE_GOLDEN_PAYLOAD_SHA256 \"{golden['fixturePayloadSha256']}\"" not in include_text
    ):
        raise RuntimeError("compact C tokenizer closure contract drift")
    expected_source_files = {
        name: {"bytes": byte_count, "sha256": digest}
        for name, (digest, byte_count) in ARTIFACT_FILES.items()
    }
    expected_implementation_files = {
        name: {"bytes": byte_count, "sha256": digest}
        for name, (digest, byte_count) in IMPLEMENTATION_FILES.items()
    }
    if provenance.get("sourceFiles") != expected_source_files:
        raise RuntimeError("pinned source-file provenance drift")
    if provenance.get("implementationFiles") != expected_implementation_files:
        raise RuntimeError("pinned implementation-file provenance drift")
    if provenance.get("environment") != {
        "huggingfaceHub": HUGGINGFACE_HUB_VERSION,
        "python": PYTHON_VERSION,
        "tokenizers": TOKENIZERS_VERSION,
        "transformers": TRANSFORMERS_VERSION,
    }:
        raise RuntimeError("pinned capture-version provenance drift")
    divergence = provenance.get("knownAutoTokenizerDivergence", {})
    if (
        divergence.get("authoritativeForGolden") is not False
        or divergence.get("autoTokenizerRegex") != AUTO_QWEN2_REGEX
        or divergence.get("controls") != AUTO_DIVERGENCE_CONTROLS
    ):
        raise RuntimeError("AutoTokenizer divergence-control drift")
    by_name = {case["name"]: case for case in cases}
    if len(provenance.get("cases", [])) != len(cases):
        raise RuntimeError("per-case provenance count drift")
    for record in provenance["cases"]:
        case = by_name.get(record["name"])
        if case is None:
            raise RuntimeError(f"unknown provenance case {record['name']}")
        expected = {
            "authority": "pinned-transformers-tokenizers-backend",
            "category": case["category"],
            "containsAddedTokenIds": [value for value in case["ids"] if value in ADDED_IDS],
            "decodedSha256": text_sha256(case["decoded"]),
            "idsLittleEndianUint32Sha256": ids_sha256(case["ids"]),
            "inputPolicy": case["inputPolicy"],
            "inputUtf8Sha256": text_sha256(case["inputText"]),
            "name": case["name"],
            "skipSpecialDecodedSha256": text_sha256(case["skipSpecialDecoded"]),
        }
        if record != expected:
            raise RuntimeError(f"{case['name']}: per-case provenance mismatch")
    expected_decode_provenance = [
        {
            "authority": "pinned-transformers-tokenizers-backend",
            "decodedSha256": text_sha256(case["decoded"]),
            "idsLittleEndianUint32Sha256": ids_sha256(case["ids"]),
            "name": case["name"],
            "skipSpecialDecodedSha256": text_sha256(case["skipSpecialDecoded"]),
        }
        for case in EXPECTED_DECODE_CONTROLS
    ]
    if provenance.get("decodeControls") != expected_decode_provenance:
        raise RuntimeError("decode-control provenance mismatch")
    return golden, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="capture with the pinned networked environment")
    mode.add_argument("--check", action="store_true", help="verify checked-in fixtures offline")
    parser.add_argument("--source-dir", type=Path, help="verified local tokenizer files for --write")
    parser.add_argument("--golden", type=Path, default=GOLDEN)
    parser.add_argument("--provenance", type=Path, default=PROVENANCE)
    parser.add_argument("--include", type=Path, default=C_INCLUDE)
    args = parser.parse_args()

    if args.check:
        golden, provenance = verify_offline(
            args.golden, args.provenance, args.include)
        if args.source_dir is not None:
            captured_golden, captured_provenance, captured_include = capture(
                args.source_dir)
            if (
                json_text(captured_golden) != args.golden.read_text()
                or json_text(captured_provenance) != args.provenance.read_text()
                or captured_include != args.include.read_text()
            ):
                raise RuntimeError("pinned full C tokenizer regeneration drift")
        print(
            f"PASS offline tokenizer fixture: {len(golden['textCases'])} text cases, "
            f"{len(golden['decodeControls'])} decode controls, "
            f"payload {golden['fixturePayloadSha256']}, "
            f"file {provenance['fixture']['goldenFileSha256']}"
        )
        return

    if args.source_dir is not None:
        golden, provenance, c_include = capture(args.source_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="qwen4exp-tokenizer-") as temporary:
            source_dir = Path(temporary)
            fetch_artifacts(source_dir)
            golden, provenance, c_include = capture(source_dir)
    args.golden.write_text(json_text(golden))
    provenance["fixture"]["goldenFileSha256"] = sha256(args.golden.read_bytes())
    args.provenance.write_text(json_text(provenance))
    args.include.write_text(c_include)
    print(
        f"wrote {len(golden['textCases'])} text cases and {len(golden['decodeControls'])} decode controls; "
        f"payload {golden['fixturePayloadSha256']}"
    )


if __name__ == "__main__":
    main()
