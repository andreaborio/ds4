#!/usr/bin/env python3
"""Validate the Qwen release identity across its authoritative local surfaces."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = "docs/contracts/qwen-release.json"
SCHEMA_VERSION = 1
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)
LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")
QWEN_FILENAME_RE = re.compile(r"Qwen3\.6-35B-A3B-[A-Za-z0-9_.-]+\.gguf")


class ContractError(Exception):
    """A malformed contract or a release-surface mismatch."""


@dataclass(frozen=True)
class Artifact:
    status: str
    filename: str
    bytes: int
    sha256: str
    revision: str | None = None
    runtime_commit: str | None = None
    storage: str | None = None
    group_size: int | None = None


@dataclass(frozen=True)
class Contract:
    model_family: str
    download_target: str
    repository: str
    published: Artifact
    negative: Artifact


def require_exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual == expected:
        return
    details: list[str] = []
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing:
        details.append("missing " + ", ".join(missing))
    if extra:
        details.append("unknown " + ", ".join(extra))
    raise ContractError(f"{label} has invalid keys ({'; '.join(details)})")


def require_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{label} must be a non-empty string")
    return value


def require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{label} must be a positive integer")
    return value


def require_filename(value: Any, label: str) -> str:
    filename = require_string(value, label)
    pure = PurePosixPath(filename)
    if pure.name != filename or filename in {".", ".."}:
        raise ContractError(f"{label} must be a bare filename")
    return filename


def require_hex(value: Any, pattern: re.Pattern[str], label: str) -> str:
    digest = require_string(value, label)
    if pattern.fullmatch(digest) is None:
        width = 40 if pattern is HEX40_RE else 64
        raise ContractError(f"{label} must be {width} lowercase hexadecimal characters")
    return digest


def load_contract(path: Path) -> Contract:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractError(f"contract does not exist: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read contract {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ContractError("contract root must be a JSON object")
    require_exact_keys(
        document,
        {
            "schemaVersion",
            "modelFamily",
            "downloadTarget",
            "repository",
            "publishedArtifact",
            "negativeArtifact",
        },
        "contract",
    )
    if document["schemaVersion"] != SCHEMA_VERSION or isinstance(
        document["schemaVersion"], bool
    ):
        raise ContractError(
            f"contract schemaVersion must be {SCHEMA_VERSION}, "
            f"got {document['schemaVersion']!r}"
        )

    model_family = require_string(document["modelFamily"], "modelFamily")
    if model_family != "Qwen3.6-35B-A3B":
        raise ContractError("modelFamily must be Qwen3.6-35B-A3B")
    download_target = require_string(document["downloadTarget"], "downloadTarget")
    if download_target != "qwen-v2":
        raise ContractError("downloadTarget must be qwen-v2")
    repository = require_string(document["repository"], "repository")
    if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository) is None:
        raise ContractError("repository must be an owner/name identifier")

    published_raw = document["publishedArtifact"]
    negative_raw = document["negativeArtifact"]
    if not isinstance(published_raw, dict) or not isinstance(negative_raw, dict):
        raise ContractError("publishedArtifact and negativeArtifact must be objects")
    require_exact_keys(
        published_raw,
        {
            "status",
            "filename",
            "revision",
            "bytes",
            "sha256",
            "runtimeCommit",
            "storage",
            "groupSize",
        },
        "publishedArtifact",
    )
    require_exact_keys(
        negative_raw,
        {"status", "filename", "bytes", "sha256"},
        "negativeArtifact",
    )
    if published_raw["status"] != "published":
        raise ContractError("publishedArtifact.status must be published")
    if negative_raw["status"] != "negative-only":
        raise ContractError("negativeArtifact.status must be negative-only")
    if published_raw["storage"] != "mlx-affine4":
        raise ContractError("publishedArtifact.storage must be mlx-affine4")
    if published_raw["groupSize"] != 64 or isinstance(
        published_raw["groupSize"], bool
    ):
        raise ContractError("publishedArtifact.groupSize must be 64")

    published = Artifact(
        status="published",
        filename=require_filename(published_raw["filename"], "publishedArtifact.filename"),
        revision=require_hex(
            published_raw["revision"], HEX40_RE, "publishedArtifact.revision"
        ),
        bytes=require_positive_int(published_raw["bytes"], "publishedArtifact.bytes"),
        sha256=require_hex(
            published_raw["sha256"], HEX64_RE, "publishedArtifact.sha256"
        ),
        runtime_commit=require_hex(
            published_raw["runtimeCommit"],
            HEX40_RE,
            "publishedArtifact.runtimeCommit",
        ),
        storage="mlx-affine4",
        group_size=64,
    )
    negative = Artifact(
        status="negative-only",
        filename=require_filename(negative_raw["filename"], "negativeArtifact.filename"),
        bytes=require_positive_int(negative_raw["bytes"], "negativeArtifact.bytes"),
        sha256=require_hex(
            negative_raw["sha256"], HEX64_RE, "negativeArtifact.sha256"
        ),
    )
    if published.filename == negative.filename:
        raise ContractError("published and negative-only filenames must differ")
    if published.sha256 == negative.sha256:
        raise ContractError("published and negative-only SHA-256 values must differ")
    return Contract(model_family, download_target, repository, published, negative)


def read_text(root: Path, relative: str) -> str:
    path = root / relative
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ContractError(f"required release surface does not exist: {relative}") from exc
    except OSError as exc:
        raise ContractError(f"cannot read release surface {relative}: {exc}") from exc


def markdown_section(root: Path, relative: str, heading: str) -> str:
    text = read_text(root, relative)
    matches = list(HEADING_RE.finditer(text))
    selected = [match for match in matches if match.group(2).strip() == heading]
    if len(selected) != 1:
        raise ContractError(
            f"{relative}: expected one Markdown section {heading!r}, found {len(selected)}"
        )
    match = selected[0]
    level = len(match.group(1))
    end = len(text)
    for candidate in matches:
        if candidate.start() > match.start() and len(candidate.group(1)) <= level:
            end = candidate.start()
            break
    return text[match.end() : end]


def strip_code(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value.startswith("`") and value.endswith("`"):
        return value[1:-1]
    return value


def parse_table(section: str, relative: str) -> list[dict[str, str]]:
    lines = [line.strip() for line in section.splitlines()]
    for index in range(len(lines) - 1):
        header_line = lines[index]
        separator_line = lines[index + 1]
        if not header_line.startswith("|") or not separator_line.startswith("|"):
            continue
        separator = [cell.strip() for cell in separator_line.strip("|").split("|")]
        if not separator or not all(re.fullmatch(r":?-{3,}:?", cell) for cell in separator):
            continue
        headers = [cell.strip() for cell in header_line.strip("|").split("|")]
        if len(headers) != len(separator):
            raise ContractError(f"{relative}: malformed Markdown table header")
        rows: list[dict[str, str]] = []
        for line in lines[index + 2 :]:
            if not line.startswith("|"):
                break
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) != len(headers):
                raise ContractError(f"{relative}: malformed Markdown table row: {line}")
            rows.append(dict(zip(headers, cells, strict=True)))
        return rows
    raise ContractError(f"{relative}: expected a Markdown table in the release section")


def require_tokens(label: str, text: str, tokens: list[str]) -> None:
    missing = [token for token in tokens if token not in text]
    if missing:
        raise ContractError(f"{label}: missing contract value(s): {', '.join(missing)}")


def require_contract_link(
    root: Path, relative: str, section: str, manifest_path: Path
) -> None:
    source_parent = (root / relative).parent
    destinations: list[Path] = []
    for match in LINK_RE.finditer(section):
        raw = match.group(1).strip().split(maxsplit=1)[0]
        if raw.startswith(("#", "http://", "https://", "mailto:")):
            continue
        destinations.append((source_parent / raw.split("#", 1)[0]).resolve())
    if manifest_path.resolve() not in destinations:
        raise ContractError(
            f"{relative}: release section must link to the canonical contract"
        )


def require_qwen_filenames(
    label: str, text: str, expected: set[str]
) -> None:
    actual = set(QWEN_FILENAME_RE.findall(text))
    if actual != expected:
        raise ContractError(
            f"{label}: Qwen artifact identities differ: "
            f"expected {sorted(expected)!r}, got {sorted(actual)!r}"
        )


def check_prose_surface(
    root: Path,
    manifest_path: Path,
    contract: Contract,
    relative: str,
    heading: str,
) -> None:
    section = markdown_section(root, relative, heading)
    published = contract.published
    negative = contract.negative
    require_contract_link(root, relative, section, manifest_path)
    require_tokens(
        f"{relative}#{heading}",
        section,
        [
            contract.model_family,
            contract.download_target,
            published.status,
            published.filename,
            f"{published.bytes:,}",
            published.sha256,
            published.revision or "",
            published.runtime_commit or "",
            negative.status,
            negative.filename,
        ],
    )
    require_qwen_filenames(
        f"{relative}#{heading}", section, {published.filename, negative.filename}
    )


def require_table_row(
    rows: list[dict[str, str]], key_header: str, key: str, relative: str
) -> dict[str, str]:
    matches = [row for row in rows if strip_code(row.get(key_header, "")) == key]
    if len(matches) != 1:
        raise ContractError(f"{relative}: expected one table row for {key!r}")
    return matches[0]


def check_qa(root: Path, manifest_path: Path, contract: Contract) -> None:
    relative = "QA_BEFORE_RELEASES.md"
    section = markdown_section(root, relative, "Release Artifact Identity")
    require_contract_link(root, relative, section, manifest_path)
    rows = parse_table(section, relative)
    published = require_table_row(rows, "Variable", "QWEN_V2", relative)
    negative = require_table_row(rows, "Variable", "QWEN_RETIRED_Q4_NEGATIVE", relative)
    published_text = published.get("Required identity", "")
    negative_text = negative.get("Required identity", "")
    p = contract.published
    n = contract.negative
    require_tokens(
        f"{relative}:QWEN_V2",
        published_text,
        [
            p.status,
            p.filename,
            p.revision or "",
            f"{p.bytes:,}",
            p.sha256,
            p.runtime_commit or "",
            "MLX affine4/group-64",
        ],
    )
    require_tokens(
        f"{relative}:QWEN_RETIRED_Q4_NEGATIVE",
        negative_text,
        [n.status, n.filename, f"{n.bytes:,}", n.sha256],
    )


def rows_by_item(rows: list[dict[str, str]], relative: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in rows:
        item = strip_code(row.get("Item", ""))
        if not item or item in result:
            raise ContractError(f"{relative}: duplicate or empty release identity item")
        result[item] = strip_code(row.get("Value", ""))
    return result


def require_exact_item(items: dict[str, str], key: str, value: str, relative: str) -> None:
    actual = items.get(key)
    if actual != value:
        raise ContractError(
            f"{relative}: {key!r} must be {value!r}, got {actual!r}"
        )


def check_qwen_store(root: Path, manifest_path: Path, contract: Contract) -> None:
    relative = "docs/qwen-expert-major-store.md"
    section = markdown_section(root, relative, "Release identity")
    require_contract_link(root, relative, section, manifest_path)
    items = rows_by_item(parse_table(section, relative), relative)
    p = contract.published
    n = contract.negative
    require_exact_item(items, "Publication state", p.status, relative)
    require_exact_item(items, "Repository", contract.repository, relative)
    require_exact_item(items, "Artifact", p.filename, relative)
    require_exact_item(items, "Artifact bytes", f"{p.bytes:,}", relative)
    require_exact_item(items, "Artifact SHA-256", p.sha256, relative)
    require_exact_item(items, "Immutable revision", p.revision or "", relative)
    require_exact_item(
        items, "Minimum compatible runtime commit", p.runtime_commit or "", relative
    )
    require_exact_item(items, "Storage", f"{p.storage}/group-{p.group_size}", relative)
    require_exact_item(items, "Negative fixture state", n.status, relative)
    require_exact_item(items, "Negative fixture", n.filename, relative)
    require_exact_item(items, "Negative fixture bytes", f"{n.bytes:,}", relative)
    require_exact_item(items, "Negative fixture SHA-256", n.sha256, relative)


def parse_shell_assignments(text: str, relative: str) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        match = re.fullmatch(r"([A-Z][A-Z0-9_]*)=(.*)", line)
        if match is None:
            continue
        name, raw_value = match.groups()
        try:
            values = shlex.split(raw_value, posix=True)
        except ValueError as exc:
            raise ContractError(
                f"{relative}:{line_number}: invalid shell assignment: {exc}"
            ) from exc
        if len(values) != 1:
            continue
        if name in assignments:
            raise ContractError(f"{relative}: duplicate top-level assignment {name}")
        assignments[name] = values[0]
    return assignments


def case_block(text: str, target: str, relative: str) -> str:
    lines = text.splitlines()
    starts = [index for index, line in enumerate(lines) if line == f"    {target})"]
    if len(starts) != 1:
        raise ContractError(f"{relative}: expected one case arm for {target}")
    start = starts[0]
    for index in range(start + 1, len(lines)):
        if lines[index] == "        ;;":
            return "\n".join(lines[start + 1 : index])
    raise ContractError(f"{relative}: unterminated case arm for {target}")


def check_downloader(root: Path, contract: Contract) -> None:
    relative = "download_model.sh"
    text = read_text(root, relative)
    assignments = parse_shell_assignments(text, relative)
    p = contract.published
    expected = {
        "RUNTIME_QWEN_STATUS": p.status,
        "RUNTIME_QWEN_REPO": contract.repository,
        "RUNTIME_QWEN_FILE": p.filename,
        "RUNTIME_QWEN_BYTES": str(p.bytes),
        "RUNTIME_QWEN_SHA256": p.sha256,
        "RUNTIME_QWEN_REVISION": p.revision or "",
        "RUNTIME_QWEN_MIN_RUNTIME_COMMIT": p.runtime_commit or "",
    }
    for name, value in expected.items():
        if assignments.get(name) != value:
            raise ContractError(
                f"{relative}: {name} must be {value!r}, got {assignments.get(name)!r}"
            )
    block = case_block(text, contract.download_target, relative)
    required_wiring = {
        "MODEL_REPO=$RUNTIME_QWEN_REPO",
        "MODEL_FILE=$RUNTIME_QWEN_FILE",
        "MODEL_REVISION=$RUNTIME_QWEN_REVISION",
        "MODEL_BYTES=$RUNTIME_QWEN_BYTES",
        "MODEL_SHA256=$RUNTIME_QWEN_SHA256",
    }
    wiring = {line.strip() for line in block.splitlines() if line.strip()}
    if not required_wiring.issubset(wiring):
        missing = sorted(required_wiring - wiring)
        raise ContractError(
            f"{relative}: {contract.download_target} is missing wiring: {missing!r}"
        )
    if contract.negative.filename in text or contract.negative.sha256 in text:
        raise ContractError(
            f"{relative}: negative-only Qwen artifact must not be downloadable"
        )


def check_download_test(root: Path, manifest_path: Path, contract: Contract) -> None:
    relative = "tests/test_download_model.sh"
    text = read_text(root, relative)
    expected_manifest = manifest_path.resolve().relative_to(root.resolve()).as_posix()
    require_tokens(
        relative,
        text,
        [
            expected_manifest,
            "RUNTIME_QWEN_STATUS",
            "RUNTIME_QWEN_REPO",
            "RUNTIME_QWEN_FILE",
            "RUNTIME_QWEN_BYTES",
            "RUNTIME_QWEN_SHA256",
            "RUNTIME_QWEN_REVISION",
            "RUNTIME_QWEN_MIN_RUNTIME_COMMIT",
            "QWEN_NEGATIVE_FILE",
            "QWEN_NEGATIVE_SHA256",
        ],
    )
    duplicated = [
        value
        for value in (
            contract.repository,
            contract.published.filename,
            contract.published.sha256,
            contract.published.revision or "",
            contract.published.runtime_commit or "",
            contract.negative.filename,
            contract.negative.sha256,
        )
        if value and value in text
    ]
    if duplicated:
        raise ContractError(
            f"{relative}: must consume the canonical contract, not duplicate values: "
            + ", ".join(duplicated)
        )


def validate(root: Path, manifest_path: Path) -> Contract:
    try:
        manifest_path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ContractError("manifest must be inside the repository root") from exc
    contract = load_contract(manifest_path)
    for relative, heading in (
        ("README.md", "Supported models"),
        ("CONTRIBUTING.md", "Artifact publication boundary"),
        ("docs/contracts/RUNTIME_SUPPORT.md", "Supported Matrix"),
    ):
        check_prose_surface(root, manifest_path, contract, relative, heading)
    check_qa(root, manifest_path, contract)
    check_qwen_store(root, manifest_path, contract)
    check_downloader(root, contract)
    check_download_test(root, manifest_path, contract)
    return contract


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    try:
        contract = validate(root, manifest_path)
    except ContractError as exc:
        print(f"Qwen release contract: FAIL: {exc}", file=sys.stderr)
        return 1
    print(
        "Qwen release contract: PASS "
        f"({contract.published.status} {contract.published.filename}; "
        f"{contract.negative.status} {contract.negative.filename})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
