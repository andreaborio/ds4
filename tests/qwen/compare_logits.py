#!/usr/bin/env python3
"""Compare one DS4 Qwen logits dump with llama.cpp ``llama-debug``.

The comparison is deliberately fail-closed around provenance.  It checks that
both programs used the same local GGUF, consumed exactly the same token IDs,
and produced logits for the same final prompt position before reporting any
numeric result.  Only Python's standard library is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


QWEN_VOCAB = 248320
QWEN_EFFECTIVE_VOCAB = 248077
QWEN_PAD_START = QWEN_EFFECTIVE_VOCAB
QWEN_PAD_END = QWEN_VOCAB
TOP_K = (5, 20, 64)
LLAMA_CPP_REVISION = "bf2c86ddc0685f580595954056c2e77ebabfab4f"
LLAMA_UNPATCHED_CALL = "common_tokenize(ctx, params.prompt, add_bos)"
LLAMA_PARSE_SPECIAL_CALL = (
    "common_tokenize(ctx, params.prompt, add_bos, params.parse_special)"
)


class ComparisonError(ValueError):
    """The inputs cannot support a trustworthy logits comparison."""


@dataclass(frozen=True)
class Ds4Dump:
    model: Path
    prompt_tokens: int
    vocab: int
    reported_argmax: int
    logits: tuple[float, ...]


@dataclass(frozen=True)
class LlamaPrompt:
    n_tokens: int
    token_ids: tuple[int, ...]


def _reject_json_constant(value: str) -> None:
    raise ComparisonError(f"non-standard JSON constant in DS4 dump: {value}")


def _is_plain_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _required_int(data: dict[str, Any], key: str, *, minimum: int) -> int:
    value = data.get(key)
    if not _is_plain_int(value) or value < minimum:
        raise ComparisonError(f"DS4 field {key!r} must be an integer >= {minimum}")
    return value


def load_ds4_dump(path: Path) -> Ds4Dump:
    try:
        with path.open("r", encoding="utf-8") as source:
            data = json.load(source, parse_constant=_reject_json_constant)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"cannot read DS4 JSON {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ComparisonError("DS4 logits dump must be a JSON object")
    if data.get("source") != "ds4":
        raise ComparisonError("DS4 logits dump must contain source='ds4'")

    model_text = data.get("model")
    if not isinstance(model_text, str) or not model_text:
        raise ComparisonError("DS4 field 'model' must be a non-empty path")
    prompt_tokens = _required_int(data, "prompt_tokens", minimum=1)
    vocab = _required_int(data, "vocab", minimum=1)

    argmax_token = data.get("argmax_token")
    if not isinstance(argmax_token, dict):
        raise ComparisonError("DS4 field 'argmax_token' must be an object")
    reported_argmax = _required_int(argmax_token, "id", minimum=0)

    raw_logits = data.get("logits")
    if not isinstance(raw_logits, list):
        raise ComparisonError("DS4 field 'logits' must be an array")
    if len(raw_logits) != vocab:
        raise ComparisonError(
            f"DS4 logits length {len(raw_logits)} does not match vocab {vocab}"
        )

    logits: list[float] = []
    for index, value in enumerate(raw_logits):
        if value is None:
            logits.append(math.nan)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            logits.append(float(value))
        else:
            raise ComparisonError(
                f"DS4 logit {index} must be a number or null, got {type(value).__name__}"
            )

    return Ds4Dump(
        model=Path(model_text),
        prompt_tokens=prompt_tokens,
        vocab=vocab,
        reported_argmax=reported_argmax,
        logits=tuple(logits),
    )


def load_llama_binary(path: Path) -> tuple[float, ...]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ComparisonError(f"cannot read llama.cpp binary logits {path}: {exc}") from exc
    if not payload:
        raise ComparisonError("llama.cpp binary logits file is empty")
    if len(payload) % 4:
        raise ComparisonError(
            f"llama.cpp binary logits size {len(payload)} is not a multiple of float32"
        )
    # llama-debug writes the host float bytes directly.  The pinned Apple and
    # llama.cpp reference environment is little-endian IEEE-754 float32.
    return tuple(item[0] for item in struct.iter_unpack("<f", payload))


_LLAMA_TEXT_LINE = re.compile(
    r"^\s*(?P<index>[0-9]+)\s*:\s*(?P<value>\S+)\s*$"
)


def load_llama_text(path: Path) -> tuple[float, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ComparisonError(f"cannot read llama.cpp text logits {path}: {exc}") from exc

    logits: list[float] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        match = _LLAMA_TEXT_LINE.fullmatch(line)
        if match is None:
            raise ComparisonError(
                f"invalid llama.cpp text logits line {line_number}: {line!r}"
            )
        index = int(match.group("index"))
        if index != len(logits):
            raise ComparisonError(
                "llama.cpp text logits indices must be contiguous from zero: "
                f"expected {len(logits)}, got {index} on line {line_number}"
            )
        try:
            value = float(match.group("value"))
        except ValueError as exc:
            raise ComparisonError(
                f"invalid float on llama.cpp text logits line {line_number}"
            ) from exc
        logits.append(value)
    if not logits:
        raise ComparisonError("llama.cpp text logits file is empty")
    return tuple(logits)


def load_llama_logits(path: Path) -> tuple[float, ...]:
    suffix = path.suffix.lower()
    if suffix == ".bin":
        return load_llama_binary(path)
    if suffix == ".txt":
        return load_llama_text(path)
    raise ComparisonError(
        f"unsupported llama.cpp logits suffix {path.suffix!r}; expected .bin or .txt"
    )


_LLAMA_TOKEN_COUNT = re.compile(r"^n_tokens:\s*([0-9]+)\s*$", re.MULTILINE)
_LLAMA_TOKEN_IDS = re.compile(r"^token ids:\s*([^\r\n]*)$", re.MULTILINE)
_LLAMA_MODEL_LOG = re.compile(
    r"loaded meta data with [0-9]+ key-value pairs and [0-9]+ tensors "
    r"from (?P<path>.+) \(version .+\)$",
    re.MULTILINE,
)
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def load_llama_prompt(path: Path) -> LlamaPrompt:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ComparisonError(f"cannot read llama.cpp prompt metadata {path}: {exc}") from exc

    count_matches = _LLAMA_TOKEN_COUNT.findall(text)
    id_matches = _LLAMA_TOKEN_IDS.findall(text)
    if len(count_matches) != 1 or len(id_matches) != 1:
        raise ComparisonError(
            "llama.cpp prompt metadata must contain exactly one n_tokens and token ids line"
        )
    n_tokens = int(count_matches[0])
    raw_ids = id_matches[0].strip()
    try:
        token_ids = tuple(
            int(piece.strip()) for piece in raw_ids.split(",") if piece.strip()
        )
    except ValueError as exc:
        raise ComparisonError("invalid token ID in llama.cpp prompt metadata") from exc
    if len(token_ids) != n_tokens:
        raise ComparisonError(
            f"llama.cpp prompt says n_tokens={n_tokens}, but lists {len(token_ids)} IDs"
        )
    if any(token < 0 for token in token_ids):
        raise ComparisonError("llama.cpp prompt contains a negative token ID")
    return LlamaPrompt(n_tokens=n_tokens, token_ids=token_ids)


def load_llama_model_path(path: Path) -> Path:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ComparisonError(f"cannot read llama-debug log {path}: {exc}") from exc
    text = _ANSI_ESCAPE.sub("", text)
    model_paths = {match.group("path") for match in _LLAMA_MODEL_LOG.finditer(text)}
    if len(model_paths) != 1:
        raise ComparisonError(
            "llama-debug log must identify exactly one loaded GGUF model path"
        )
    model_path = Path(model_paths.pop())
    if not model_path.is_absolute():
        raise ComparisonError(
            "llama-debug must be invoked with an absolute GGUF path for provenance"
        )
    return model_path


def load_ds4_token_ids(path: Path) -> tuple[int, ...]:
    try:
        with path.open("r", encoding="utf-8") as source:
            first_line = source.readline()
    except (OSError, UnicodeError) as exc:
        raise ComparisonError(f"cannot read DS4 token dump {path}: {exc}") from exc
    try:
        raw_ids = json.loads(first_line)
    except json.JSONDecodeError as exc:
        raise ComparisonError(
            "the first line of the DS4 --dump-tokens output must be a JSON integer array"
        ) from exc
    if not isinstance(raw_ids, list) or not raw_ids:
        raise ComparisonError("DS4 token dump must contain a non-empty integer array")
    if not all(_is_plain_int(token) and token >= 0 for token in raw_ids):
        raise ComparisonError("DS4 token dump contains an invalid token ID")
    return tuple(raw_ids)


def _resolve_existing(path: Path, *, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise ComparisonError(f"{label} does not exist: {path}") from exc
    if not resolved.is_file():
        raise ComparisonError(f"{label} is not a regular file: {resolved}")
    return resolved


def validate_llama_source(path: Path) -> dict[str, object]:
    source = path.expanduser().resolve()
    debug_cpp = source / "examples" / "debug" / "debug.cpp"
    if not debug_cpp.is_file():
        raise ComparisonError(
            f"llama.cpp source does not contain examples/debug/debug.cpp: {source}"
        )
    try:
        completed = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ComparisonError(f"cannot read llama.cpp source revision: {exc}") from exc
    revision = completed.stdout.strip()
    if revision != LLAMA_CPP_REVISION:
        raise ComparisonError(
            f"llama.cpp revision mismatch: expected {LLAMA_CPP_REVISION}, got {revision}"
        )
    try:
        debug_source = debug_cpp.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ComparisonError(f"cannot inspect patched llama-debug source: {exc}") from exc
    try:
        baseline = subprocess.run(
            [
                "git",
                "-C",
                str(source),
                "show",
                f"{LLAMA_CPP_REVISION}:examples/debug/debug.cpp",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ComparisonError(f"cannot read pinned llama-debug source: {exc}") from exc
    if baseline.count(LLAMA_UNPATCHED_CALL) != 2:
        raise ComparisonError(
            "pinned llama-debug baseline no longer has the expected two tokenization calls"
        )
    expected_debug_source = baseline.replace(
        LLAMA_UNPATCHED_CALL, LLAMA_PARSE_SPECIAL_CALL
    )
    if debug_source != expected_debug_source:
        raise ComparisonError(
            "pinned llama-debug must contain exactly the two documented "
            "params.parse_special changes and no other debug.cpp drift"
        )
    return {
        "llama_cpp_source": str(source),
        "llama_cpp_revision": revision,
        "llama_debug_parse_special_patch": True,
        "llama_debug_source_sha256": hashlib.sha256(
            debug_source.encode("utf-8")
        ).hexdigest(),
    }


def validate_provenance(
    ds4: Ds4Dump,
    model: Path,
    llama_logits_path: Path,
    llama_model: Path,
    ds4_token_ids: Sequence[int],
    llama_prompt: LlamaPrompt,
) -> dict[str, object]:
    expected_model = _resolve_existing(model, label="comparison model")
    ds4_model = _resolve_existing(ds4.model, label="model recorded by DS4")
    try:
        same_model = os.path.samefile(expected_model, ds4_model)
    except OSError as exc:
        raise ComparisonError(f"cannot compare model file identity: {exc}") from exc
    if not same_model:
        raise ComparisonError(
            f"model mismatch: DS4 used {ds4_model}, comparator was given {expected_model}"
        )
    loaded_llama_model = _resolve_existing(
        llama_model, label="model recorded by llama-debug"
    )
    try:
        same_llama_model = os.path.samefile(expected_model, loaded_llama_model)
    except OSError as exc:
        raise ComparisonError(f"cannot compare llama-debug model identity: {exc}") from exc
    if not same_llama_model:
        raise ComparisonError(
            f"model mismatch: llama-debug used {loaded_llama_model}, comparator "
            f"was given {expected_model}"
        )

    expected_llama_name = (
        f"llamacpp-{model.expanduser().stem}{llama_logits_path.suffix.lower()}"
    )
    if llama_logits_path.name != expected_llama_name:
        raise ComparisonError(
            "llama.cpp logits filename does not match the model: expected "
            f"{expected_llama_name!r}, got {llama_logits_path.name!r}"
        )

    if tuple(ds4_token_ids) != llama_prompt.token_ids:
        mismatch = next(
            (
                index
                for index, pair in enumerate(zip(ds4_token_ids, llama_prompt.token_ids))
                if pair[0] != pair[1]
            ),
            min(len(ds4_token_ids), len(llama_prompt.token_ids)),
        )
        raise ComparisonError(
            "prompt token mismatch at index "
            f"{mismatch}: DS4 count={len(ds4_token_ids)}, "
            f"llama.cpp count={len(llama_prompt.token_ids)}"
        )
    if ds4.prompt_tokens != len(ds4_token_ids):
        raise ComparisonError(
            f"DS4 JSON says prompt_tokens={ds4.prompt_tokens}, but --dump-tokens "
            f"contains {len(ds4_token_ids)} IDs"
        )
    if ds4.prompt_tokens != llama_prompt.n_tokens:
        raise ComparisonError(
            f"token-position mismatch: DS4={ds4.prompt_tokens}, "
            f"llama.cpp={llama_prompt.n_tokens}"
        )

    return {
        "model": str(expected_model),
        "llama_debug_model": str(loaded_llama_model),
        "prompt_tokens": ds4.prompt_tokens,
        "logits_position": ds4.prompt_tokens - 1,
        "token_ids_match": True,
    }


def is_qwen_padding(token_id: int) -> bool:
    return QWEN_PAD_START <= token_id < QWEN_PAD_END


def selectable_ids(logits: Sequence[float]) -> list[int]:
    return [
        token_id
        for token_id, value in enumerate(logits)
        if not is_qwen_padding(token_id) and math.isfinite(value)
    ]


def top_token_ids(logits: Sequence[float], count: int) -> list[int]:
    candidates = selectable_ids(logits)
    if len(candidates) < count:
        raise ComparisonError(
            f"need {count} finite non-padding logits, found {len(candidates)}"
        )
    candidates.sort(key=lambda token_id: (-logits[token_id], token_id))
    return candidates[:count]


def compare_vectors(
    ds4_logits: Sequence[float],
    llama_logits: Sequence[float],
    *,
    ds4_reported_argmax: int | None = None,
) -> dict[str, object]:
    if len(ds4_logits) != len(llama_logits):
        raise ComparisonError(
            f"vocab mismatch: DS4 has {len(ds4_logits)} logits, "
            f"llama.cpp has {len(llama_logits)}"
        )

    eligible = [
        token_id
        for token_id in range(len(ds4_logits))
        if not is_qwen_padding(token_id)
    ]
    paired = [
        token_id
        for token_id in eligible
        if math.isfinite(ds4_logits[token_id])
        and math.isfinite(llama_logits[token_id])
    ]
    if not paired:
        raise ComparisonError("no paired finite non-padding logits")

    ds4_top = top_token_ids(ds4_logits, max(TOP_K))
    llama_top = top_token_ids(llama_logits, max(TOP_K))
    ds4_argmax = ds4_top[0]
    llama_argmax = llama_top[0]
    if ds4_reported_argmax is not None:
        if is_qwen_padding(ds4_reported_argmax):
            raise ComparisonError(
                f"DS4 reported padded Qwen token {ds4_reported_argmax} as argmax"
            )
        if ds4_reported_argmax != ds4_argmax:
            raise ComparisonError(
                f"DS4 reported argmax {ds4_reported_argmax}, but its eligible "
                f"logits select {ds4_argmax}"
            )

    products = [ds4_logits[index] * llama_logits[index] for index in paired]
    ds4_squares = [ds4_logits[index] ** 2 for index in paired]
    llama_squares = [llama_logits[index] ** 2 for index in paired]
    differences = [ds4_logits[index] - llama_logits[index] for index in paired]
    dot = math.fsum(products)
    norm_product = math.sqrt(math.fsum(ds4_squares) * math.fsum(llama_squares))
    cosine = dot / norm_product if norm_product else None
    squared_error = math.fsum(value * value for value in differences)
    max_abs_token = max(paired, key=lambda index: abs(ds4_logits[index] - llama_logits[index]))

    top_k: dict[str, object] = {}
    for count in TOP_K:
        ds4_ids = ds4_top[:count]
        llama_ids = llama_top[:count]
        overlap = len(set(ds4_ids).intersection(llama_ids))
        top_k[str(count)] = {
            "overlap": overlap,
            "overlap_fraction": overlap / count,
            "ds4_ids": ds4_ids,
            "llama_ids": llama_ids,
        }

    return {
        "finite_coverage": {
            "eligible": len(eligible),
            "ds4_finite": sum(math.isfinite(ds4_logits[index]) for index in eligible),
            "llama_finite": sum(math.isfinite(llama_logits[index]) for index in eligible),
            "paired_finite": len(paired),
            "paired_fraction": len(paired) / len(eligible),
            "excluded_qwen_padding_ids": max(
                0, min(len(ds4_logits), QWEN_PAD_END) - QWEN_PAD_START
            ),
        },
        "top_1": {
            "ds4_id": ds4_argmax,
            "llama_id": llama_argmax,
            "match": ds4_argmax == llama_argmax,
        },
        "top_k": top_k,
        "cosine": cosine,
        "rmse": math.sqrt(squared_error / len(paired)),
        "max_abs": abs(ds4_logits[max_abs_token] - llama_logits[max_abs_token]),
        "max_abs_token_id": max_abs_token,
    }


def compare_files(
    *,
    ds4_logits_path: Path,
    llama_logits_path: Path,
    llama_prompt_path: Path,
    llama_log_path: Path,
    ds4_tokens_path: Path,
    model_path: Path,
    llama_source_path: Path,
) -> dict[str, object]:
    llama_source = validate_llama_source(llama_source_path)
    ds4 = load_ds4_dump(ds4_logits_path)
    if ds4.vocab != QWEN_VOCAB:
        raise ComparisonError(
            f"expected Qwen3.6 vocab {QWEN_VOCAB}, got {ds4.vocab}"
        )
    llama_logits = load_llama_logits(llama_logits_path)
    llama_model = load_llama_model_path(llama_log_path)
    expected_prompt_name = f"llamacpp-{model_path.stem}-prompt.txt"
    if llama_prompt_path.name != expected_prompt_name:
        raise ComparisonError(
            "llama.cpp prompt filename does not match the model: expected "
            f"{expected_prompt_name!r}, got {llama_prompt_path.name!r}"
        )
    llama_prompt = load_llama_prompt(llama_prompt_path)
    ds4_token_ids = load_ds4_token_ids(ds4_tokens_path)
    provenance = validate_provenance(
        ds4,
        model_path,
        llama_logits_path,
        llama_model,
        ds4_token_ids,
        llama_prompt,
    )
    if ds4.vocab != len(llama_logits):
        raise ComparisonError(
            f"vocab mismatch: DS4 metadata={ds4.vocab}, llama.cpp={len(llama_logits)}"
        )
    comparison = compare_vectors(
        ds4.logits,
        llama_logits,
        ds4_reported_argmax=ds4.reported_argmax,
    )
    return {
        "schema": "ds4-qwen-logits-comparison-v1",
        "provenance": {**provenance, **llama_source},
        "comparison": comparison,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ds4-logits", type=Path, required=True)
    parser.add_argument("--llama-logits", type=Path, required=True)
    parser.add_argument("--llama-prompt", type=Path, required=True)
    parser.add_argument("--llama-log", type=Path, required=True)
    parser.add_argument("--ds4-tokens", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="the exact local GGUF passed to both DS4 and llama-debug",
    )
    parser.add_argument(
        "--llama-source",
        type=Path,
        required=True,
        help=(
            "pinned llama.cpp checkout with the documented llama-debug "
            "parse-special patch"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="also write the JSON report atomically to this path",
    )
    return parser


def write_report(path: Path, payload: str) -> None:
    destination = path.expanduser()
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, destination)
    except OSError as exc:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise ComparisonError(f"cannot write report {destination}: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = compare_files(
            ds4_logits_path=args.ds4_logits,
            llama_logits_path=args.llama_logits,
            llama_prompt_path=args.llama_prompt,
            llama_log_path=args.llama_log,
            ds4_tokens_path=args.ds4_tokens,
            model_path=args.model,
            llama_source_path=args.llama_source,
        )
        payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.output is not None:
            write_report(args.output, payload)
        sys.stdout.write(payload)
    except ComparisonError as exc:
        print(f"compare_logits: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
