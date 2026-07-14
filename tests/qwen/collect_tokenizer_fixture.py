#!/usr/bin/env python3
"""Build the compact, model-free Qwen3.6 C tokenizer fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tokenizers as tokenizers_package
from tokenizers import AddedToken, Tokenizer


MODEL = "Qwen/Qwen3.6-35B-A3B"
REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
TOKENIZER_SHA256 = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
GOLDEN_SHA256 = "87606fc0f98911e4ccaba9f7179ed11dffda79d11a9b12795f6f9bb961218ec2"
TOKENIZERS_VERSION = "0.22.2"


@dataclass(frozen=True)
class Case:
    kind: str
    name: str
    text: str
    expected: tuple[int, ...]


@dataclass
class Trace:
    final_symbols: set[str]
    merge_candidates: set[tuple[int, str, str]]
    merge_decisions: int = 0


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_hash(path: Path, expected: str, label: str) -> None:
    got = sha256(path)
    if got != expected:
        raise RuntimeError(
            f"unexpected {label} SHA256 for {path}: got {got}, expected {expected}"
        )


def resolve_tokenizer_json(path: Path | None) -> Path:
    if path is not None:
        return path
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "pass --tokenizer-json or install huggingface_hub"
        ) from exc
    return Path(
        hf_hub_download(MODEL, "tokenizer.json", revision=REVISION)
    )


def load_golden(path: Path) -> tuple[dict[str, Any], list[Case]]:
    require_hash(path, GOLDEN_SHA256, "golden fixture")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("source") != {"model": MODEL, "revision": REVISION}:
        raise RuntimeError(f"unexpected source metadata in {path}")

    cases: list[Case] = []
    controls = set(data["tokenizer"]["special_token_ids"])
    for item in data["text_vectors"]:
        # The public text API treats user content as data.  Golden vectors that
        # intentionally contain added tokens exercise only the trusted rendered
        # prompt path, where those atoms must remain special and unnormalized.
        kind = (
            "TRUSTED_TEXT"
            if any(token in item["text"] for token in controls)
            else "TEXT"
        )
        cases.append(
            Case(
                kind=kind,
                name=item["name"],
                text=item["text"],
                expected=tuple(item["token_ids"]),
            )
        )
    for item in data["chat_vectors"]:
        cases.append(
            Case(
                kind="RENDERED_CHAT",
                name=item["name"],
                text=item["rendered"],
                expected=tuple(item["token_ids"]),
            )
        )
    return data, cases


def load_tokenizer(
    path: Path, controls: dict[str, int]
) -> tuple[Tokenizer, dict[str, int], dict[tuple[str, str], int]]:
    require_hash(path, TOKENIZER_SHA256, "official tokenizer")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("normalizer") != {"type": "NFC"}:
        raise RuntimeError("the pinned tokenizer no longer declares NFC normalization")

    model = raw.get("model", {})
    if model.get("type") != "BPE":
        raise RuntimeError("the pinned tokenizer is no longer a BPE tokenizer")
    vocab = model.get("vocab")
    merges = model.get("merges")
    if not isinstance(vocab, dict) or not isinstance(merges, list):
        raise RuntimeError("the pinned tokenizer has invalid vocab or merge data")

    merge_rank: dict[tuple[str, str], int] = {}
    for rank, merge in enumerate(merges):
        parts = merge.split(" ")
        if len(parts) != 2:
            raise RuntimeError(f"invalid BPE merge at rank {rank}: {merge!r}")
        pair = (parts[0], parts[1])
        if pair in merge_rank:
            raise RuntimeError(f"duplicate BPE merge pair at rank {rank}: {merge!r}")
        merge_rank[pair] = rank

    tokenizer = Tokenizer.from_file(str(path))
    for text, token_id in sorted(controls.items(), key=lambda item: item[1]):
        got = tokenizer.token_to_id(text)
        if got is None:
            tokenizer.add_special_tokens(
                [AddedToken(text, special=True, normalized=False)]
            )
            got = tokenizer.token_to_id(text)
        if got != token_id:
            raise RuntimeError(
                f"control token {text!r} has id {got}, expected {token_id}"
            )

    return tokenizer, {str(key): int(value) for key, value in vocab.items()}, merge_rank


def trace_piece(
    piece: str,
    vocab: dict[str, int],
    merge_rank: dict[tuple[str, str], int],
    trace: Trace,
) -> list[int]:
    symbols = list(piece)
    while True:
        candidates: list[tuple[int, int, str, str]] = []
        for index in range(len(symbols) - 1):
            left = symbols[index]
            right = symbols[index + 1]
            rank = merge_rank.get((left, right))
            if rank is None:
                continue
            candidates.append((rank, index, left, right))
            # Keep every candidate seen on the official path, not only the
            # winner.  Otherwise a first-match implementation could pass this
            # fixture without honoring the BPE ranks.
            trace.merge_candidates.add((rank, left, right))
        if not candidates:
            break

        _rank, index, left, right = min(candidates)
        symbols[index : index + 2] = [left + right]
        trace.merge_decisions += 1

    token_ids: list[int] = []
    for symbol in symbols:
        token_id = vocab.get(symbol)
        if token_id is None:
            raise RuntimeError(f"final BPE symbol is absent from vocab: {symbol!r}")
        trace.final_symbols.add(symbol)
        token_ids.append(token_id)
    return token_ids


def trace_encode(
    text: str,
    tokenizer: Tokenizer,
    controls: dict[str, int],
    vocab: dict[str, int],
    merge_rank: dict[tuple[str, str], int],
    trace: Trace,
) -> list[int]:
    # Every pinned control has single_word/lstrip/rstrip=false and
    # normalized=false.  Match it before NFC, as tokenizers does.
    atoms = sorted(controls, key=lambda item: (-len(item), item))
    token_ids: list[int] = []
    plain: list[str] = []

    def flush_plain() -> None:
        if not plain:
            return
        normalizer = tokenizer.normalizer
        if normalizer is None:
            raise RuntimeError("the pinned tokenizer unexpectedly has no normalizer")
        normalized = normalizer.normalize_str("".join(plain))
        plain.clear()
        for piece, _offsets in tokenizer.pre_tokenizer.pre_tokenize_str(normalized):
            token_ids.extend(trace_piece(piece, vocab, merge_rank, trace))

    position = 0
    while position < len(text):
        atom = next(
            (candidate for candidate in atoms if text.startswith(candidate, position)),
            None,
        )
        if atom is None:
            plain.append(text[position])
            position += 1
            continue
        flush_plain()
        token_ids.append(controls[atom])
        position += len(atom)
    flush_plain()
    return token_ids


def c_string(value: str) -> str:
    # Three-digit octal escapes are byte-exact and cannot absorb a following
    # hexadecimal source character the way C's variable-length \\x escape can.
    escaped: list[str] = ['"']
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


def render_fixture(
    cases: list[Case],
    controls: dict[str, int],
    vocab: dict[str, int],
    trace: Trace,
) -> str:
    token_ids = {symbol: vocab[symbol] for symbol in trace.final_symbols}
    for text, token_id in controls.items():
        previous = token_ids.setdefault(text, token_id)
        if previous != token_id:
            raise RuntimeError(f"conflicting fixture token id for {text!r}")

    tokens = sorted(token_ids.items(), key=lambda item: (item[1], item[0]))
    merges = sorted(trace.merge_candidates)
    expected_count = sum(len(case.expected) for case in cases)

    lines = [
        "/* Generated by tests/qwen/collect_tokenizer_fixture.py; do not edit. */",
        "#define QWEN36_TOKENIZER_FIXTURE_MODEL " + c_string(MODEL),
        "#define QWEN36_TOKENIZER_FIXTURE_REVISION " + c_string(REVISION),
        "#define QWEN36_TOKENIZER_FIXTURE_TOKENIZER_SHA256 "
        + c_string(TOKENIZER_SHA256),
        "#define QWEN36_TOKENIZER_FIXTURE_GOLDEN_SHA256 " + c_string(GOLDEN_SHA256),
        "#define QWEN36_TOKENIZER_FIXTURE_TOKENIZERS_VERSION "
        + c_string(TOKENIZERS_VERSION),
        f"#define QWEN36_TOKENIZER_FIXTURE_TOKEN_COUNT {len(tokens)}u",
        f"#define QWEN36_TOKENIZER_FIXTURE_MERGE_COUNT {len(merges)}u",
        f"#define QWEN36_TOKENIZER_FIXTURE_CASE_COUNT {len(cases)}u",
        f"#define QWEN36_TOKENIZER_FIXTURE_EXPECTED_ID_COUNT {expected_count}u",
        f"#define QWEN36_TOKENIZER_FIXTURE_MERGE_DECISIONS {trace.merge_decisions}u",
        "",
        "typedef struct {",
        "    const char *text;",
        "    uint32_t len;",
        "    int id;",
        "} qwen36_fixture_token;",
        "",
        "static const qwen36_fixture_token qwen36_fixture_tokens[] = {",
    ]
    for text, token_id in tokens:
        lines.append(
            f"    {{ {c_string(text)}, {len(text.encode('utf-8'))}u, {token_id} }},"
        )
    lines += [
        "};",
        "",
        "typedef struct {",
        "    const char *text;",
        "    uint32_t len;",
        "    int rank;",
        "} qwen36_fixture_merge;",
        "",
        "static const qwen36_fixture_merge qwen36_fixture_merges[] = {",
    ]
    for rank, left, right in merges:
        text = f"{left} {right}"
        lines.append(
            f"    {{ {c_string(text)}, {len(text.encode('utf-8'))}u, {rank} }},"
        )
    lines += ["};", ""]

    for case in cases:
        values = ", ".join(str(token_id) for token_id in case.expected)
        lines.append(
            f"static const int qwen36_fixture_ids_{case.name}[] = {{ {values} }};"
        )

    lines += [
        "",
        "typedef enum {",
        "    QWEN36_FIXTURE_TEXT = 0,",
        "    QWEN36_FIXTURE_TRUSTED_TEXT = 1,",
        "    QWEN36_FIXTURE_RENDERED_CHAT = 2,",
        "} qwen36_fixture_kind;",
        "",
        "typedef struct {",
        "    qwen36_fixture_kind kind;",
        "    const char *name;",
        "    const char *text;",
        "    uint32_t text_len;",
        "    const int *expected;",
        "    uint32_t expected_len;",
        "} qwen36_fixture_case;",
        "",
        "static const qwen36_fixture_case qwen36_fixture_cases[] = {",
    ]
    for case in cases:
        lines.append(
            "    { "
            f"QWEN36_FIXTURE_{case.kind}, {c_string(case.name)}, "
            f"{c_string(case.text)}, {len(case.text.encode('utf-8'))}u, "
            f"qwen36_fixture_ids_{case.name}, {len(case.expected)}u"
            " },"
        )
    lines += ["};", ""]
    return "\n".join(lines)


def collect(tokenizer_json: Path, golden_path: Path) -> tuple[str, Trace, int]:
    golden, cases = load_golden(golden_path)
    controls = {
        str(text): int(token_id)
        for text, token_id in golden["tokenizer"]["special_token_ids"].items()
    }
    tokenizer, vocab, merge_rank = load_tokenizer(tokenizer_json, controls)
    trace = Trace(final_symbols=set(), merge_candidates=set())

    for case in cases:
        official = tokenizer.encode(case.text, add_special_tokens=False).ids
        if tuple(official) != case.expected:
            raise RuntimeError(
                f"official tokenizer no longer matches golden case {case.name!r}"
            )
        traced = trace_encode(
            case.text, tokenizer, controls, vocab, merge_rank, trace
        )
        if tuple(traced) != case.expected:
            raise RuntimeError(
                f"merge trace does not match golden case {case.name!r}"
            )

    return render_fixture(cases, controls, vocab, trace), trace, len(cases)


def main() -> int:
    if tokenizers_package.__version__ != TOKENIZERS_VERSION:
        raise RuntimeError(
            "fixture generation requires tokenizers=="
            f"{TOKENIZERS_VERSION}, got {tokenizers_package.__version__}"
        )

    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Build the compact Qwen3.6 tokenizer fixture for C tests."
    )
    parser.add_argument(
        "--tokenizer-json",
        type=Path,
        help="pinned official tokenizer.json (downloaded when omitted)",
    )
    parser.add_argument(
        "--golden",
        type=Path,
        default=here / "qwen36_tokenizer_chat_golden.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "qwen36_tokenizer_fixture.inc",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail unless the checked-in C fixture is byte-for-byte current",
    )
    args = parser.parse_args()

    tokenizer_json = resolve_tokenizer_json(args.tokenizer_json)
    rendered, trace, case_count = collect(tokenizer_json, args.golden)
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"Qwen C tokenizer fixture is stale: {args.output}")
        print(
            f"Qwen C tokenizer fixture matches {MODEL}@{REVISION}: "
            f"{case_count} cases, {len(trace.final_symbols)} BPE outputs, "
            f"{len(trace.merge_candidates)} merge candidates"
        )
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(
        f"wrote {args.output}: {case_count} cases, "
        f"{len(trace.final_symbols)} BPE outputs, "
        f"{len(trace.merge_candidates)} merge candidates"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
