#!/usr/bin/env python3
"""Validate the numbered release metadata that is knowable before tagging."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEMVER_RE = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?")
DATE_RE = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


def fail(message: str) -> None:
    raise SystemExit(f"release-metadata: FAIL: {message}")


def cff_scalar(text: str, name: str) -> str:
    match = re.search(rf"(?m)^{re.escape(name)}:\s*['\"]?([^'\"\s]+)['\"]?\s*$", text)
    if match is None:
        fail(f"CITATION.cff lacks a single-line {name}")
    return match.group(1)


def main() -> int:
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    version = cff_scalar(citation, "version")
    release_date = cff_scalar(citation, "date-released")
    if SEMVER_RE.fullmatch(version) is None:
        fail(f"invalid citation version: {version}")
    if DATE_RE.fullmatch(release_date) is None:
        fail(f"invalid citation date-released: {release_date}")

    note_path = ROOT / "docs" / "releases" / f"v{version}.md"
    if not note_path.is_file():
        fail(f"missing numbered release note: {note_path.relative_to(ROOT)}")
    note = note_path.read_text(encoding="utf-8")
    required_note_tokens = (
        f"# Hebrus {version}",
        f"| Release date | {release_date} |",
        f"| Release tag | `v{version}` |",
        f"`hebrus-{version}.tar.gz`",
        f"`hebrus-{version}-source.json`",
        "`SHA256SUMS`",
    )
    for token in required_note_tokens:
        if token not in note:
            fail(f"{note_path.name} lacks {token!r}")
    placeholder_patterns = (
        r"<YYYY-MM-DD>",
        r"<version>",
        r"<full-40-character-commit>",
        r"<64-lowercase-hex-digest>",
        r"<permanent run links>",
    )
    for pattern in placeholder_patterns:
        if re.search(pattern, note, flags=re.IGNORECASE):
            fail(f"{note_path.name} retains placeholder {pattern!r}")

    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    heading = f"## {version} - {release_date}"
    if heading not in changelog:
        fail(f"CHANGELOG.md lacks {heading!r}")

    print(f"release-metadata: PASS (v{version}, {release_date}, three source assets)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
