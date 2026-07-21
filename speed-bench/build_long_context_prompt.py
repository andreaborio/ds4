#!/usr/bin/env python3
"""Deterministically extend a benchmark prompt without storing a giant fixture."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def extend_prompt(source: bytes, minimum_bytes: int) -> bytes:
    if not source:
        raise ValueError("source prompt is empty")
    if minimum_bytes <= 0:
        raise ValueError("minimum byte count must be positive")

    chunks = [source]
    repeat = 1
    total = len(source)
    while total < minimum_bytes:
        separator = (
            f"\n\n[ds4 deterministic long-context repeat {repeat:06d}]\n\n"
        ).encode("ascii")
        chunks.extend((separator, source))
        total += len(separator) + len(source)
        repeat += 1
    return b"".join(chunks)


def self_check() -> None:
    source = b"alpha beta gamma\n"
    first = extend_prompt(source, 80)
    second = extend_prompt(source, 80)
    assert first == second
    assert first.startswith(source)
    assert len(first) >= 80
    assert first.count(source) > 1
    print("long-context prompt builder: OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--minimum-bytes", type=int)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.check:
        self_check()
        return
    if args.source is None or args.output is None or args.minimum_bytes is None:
        parser.error("--source, --output, and --minimum-bytes are required")
    if args.source.resolve() == args.output.resolve():
        parser.error("source and output must be different files")

    data = extend_prompt(args.source.read_bytes(), args.minimum_bytes)
    args.output.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    print(f"bytes={len(data)} sha256={digest} output={args.output}")


if __name__ == "__main__":
    main()
