#!/usr/bin/env python3
"""Check that local Markdown links resolve inside the repository."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")


def markdown_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.md"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted(ROOT / name for name in result.stdout.splitlines() if (ROOT / name).exists())


def local_target(raw: str) -> str | None:
    target = raw.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    target = target.split(maxsplit=1)[0]
    if not target or target.startswith(("#", "/", "http://", "https://", "mailto:")):
        return None
    return unquote(target.split("#", 1)[0])


def main() -> int:
    broken: list[str] = []
    checked = 0
    for source in markdown_files():
        text = source.read_text(encoding="utf-8")
        for match in LINK_RE.finditer(text):
            target = local_target(match.group(1))
            if target is None:
                continue
            checked += 1
            destination = (source.parent / target).resolve()
            try:
                destination.relative_to(ROOT)
            except ValueError:
                broken.append(f"{source.relative_to(ROOT)}: link escapes repository: {target}")
                continue
            if not destination.exists():
                line = text.count("\n", 0, match.start()) + 1
                broken.append(f"{source.relative_to(ROOT)}:{line}: missing {target}")

    if broken:
        print("documentation link check failed:", file=sys.stderr)
        for issue in broken:
            print(f"- {issue}", file=sys.stderr)
        return 1
    print(f"documentation link check passed ({checked} local links)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
