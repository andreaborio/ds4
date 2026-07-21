#!/usr/bin/env python3
"""Prove canonical Hebrus commands and DS4 aliases are one executable surface."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys


PAIRS = (
    ("hebrus", "ds4", "cli"),
    ("hebrus-server", "ds4-server", "server"),
    ("hebrus-agent", "ds4-agent", "agent"),
    ("hebrus-bench", "ds4-bench", "bench"),
    ("hebrus-eval", "ds4-eval", "eval"),
)

PARITY_CASES = (
    ("--capabilities=json",),
    ("--build-info",),
    ("--help",),
    ("--capabilities",),
    ("--role",),
    ("--command-alias-invalid-test",),
)


def fail(message: str) -> None:
    raise AssertionError(message)


def run(binary: pathlib.Path, args: tuple[str, ...]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [os.fsencode(binary), *(os.fsencode(arg) for arg in args)],
        check=False,
        capture_output=True,
    )


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_layout(canonical: pathlib.Path, legacy: pathlib.Path, layout: str) -> None:
    if not canonical.is_file() or not legacy.is_file():
        fail(f"missing command pair: {canonical}, {legacy}")
    if not legacy.is_symlink():
        fail(f"legacy command is not a symlink: {legacy}")
    if layout == "profile":
        if canonical.is_symlink():
            fail(f"canonical profile command is unexpectedly a symlink: {canonical}")
        if os.readlink(legacy) != canonical.name:
            fail(f"{legacy} does not point directly to {canonical.name}")
    elif not canonical.is_symlink():
        fail(f"published canonical command is not a symlink: {canonical}")
    if not os.path.samefile(canonical, legacy):
        fail(f"command pair does not resolve to one file: {canonical}, {legacy}")
    if sha256(canonical) != sha256(legacy):
        fail(f"command pair has different binary content: {canonical}, {legacy}")


def validate_pair(
    canonical: pathlib.Path,
    legacy: pathlib.Path,
    role: str,
    backend: str,
) -> None:
    capabilities = run(canonical, ("--capabilities=json",))
    if capabilities.returncode != 0 or capabilities.stderr:
        fail(f"{canonical}: capability invocation failed")
    document = json.loads(capabilities.stdout)
    if document.get("engine_id") != "hebrus":
        fail(f"{canonical}: canonical engine_id is not hebrus")
    legacy_document = json.loads(run(legacy, ("--capabilities=json",)).stdout)
    if legacy_document.get("engine_id") != "ds4":
        fail(f"{legacy}: compatibility engine_id is not ds4")
    if document.get("backend") != backend:
        fail(f"{canonical}: expected backend {backend}")
    if document.get("executable_role") != role:
        fail(f"{canonical}: expected executable role {role}")

    for args in PARITY_CASES:
        canonical_result = run(canonical, args)
        legacy_result = run(legacy, args)
        canonical_name = os.fsencode(canonical.name)
        legacy_name = os.fsencode(legacy.name)

        def normalize(data: bytes) -> bytes:
            return (
                data.replace(canonical_name, legacy_name)
                .replace(b'"engine_id": "hebrus"', b'"engine_id": "ds4"')
                .replace(b"hebrus build", b"ds4 build")
            )

        canonical_record = (
            canonical_result.returncode,
            normalize(canonical_result.stdout),
            normalize(canonical_result.stderr),
        )
        legacy_record = (
            legacy_result.returncode,
            legacy_result.stdout,
            legacy_result.stderr,
        )
        if canonical_record != legacy_record:
            fail(f"{canonical.name}/{legacy.name} differ for {' '.join(args)}")

        if args == ("--help",):
            if not canonical_result.stdout.startswith(canonical_name + b"\n"):
                fail(f"{canonical.name}: help does not use canonical invocation name")
            if not legacy_result.stdout.startswith(legacy_name + b"\n"):
                fail(f"{legacy.name}: help does not preserve legacy invocation name")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-dir", required=True, type=pathlib.Path)
    parser.add_argument("--backend", required=True, choices=("metal", "cpu"))
    parser.add_argument("--layout", required=True, choices=("profile", "published"))
    args = parser.parse_args()

    bin_dir = args.bin_dir.absolute()
    resolved_targets: set[pathlib.Path] = set()
    for canonical_name, legacy_name, role in PAIRS:
        canonical = bin_dir / canonical_name
        legacy = bin_dir / legacy_name
        validate_layout(canonical, legacy, args.layout)
        validate_pair(canonical, legacy, role, args.backend)
        target = canonical.resolve()
        if target in resolved_targets:
            fail(f"roles unexpectedly share one executable: {target}")
        resolved_targets.add(target)

    print(f"command-aliases: PASS ({args.backend}, {len(PAIRS)} pairs, {args.layout})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, json.JSONDecodeError) as exc:
        print(f"command-aliases: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
