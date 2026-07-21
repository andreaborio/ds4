#!/usr/bin/env python3
"""Validate the public, model-free engine capability contract."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys


ROLES = {
    "ds4": "cli",
    "ds4-server": "server",
    "ds4-agent": "agent",
    "ds4-bench": "bench",
    "ds4-eval": "eval",
}

ROOT_KEYS = {
    "schema_version",
    "engine_id",
    "build_git_sha",
    "backend",
    "executable_role",
    "model_families",
    "expert_major",
}


def fail(message: str) -> None:
    raise AssertionError(message)


def run(binary: pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(binary), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def require_exact_int(value: object, expected: int, label: str) -> None:
    if type(value) is not int or value != expected:
        fail(f"{label}: expected integer {expected}, got {value!r}")


def validate(binary: pathlib.Path, backend: str, role: str) -> dict[str, object]:
    result = run(binary, "--capabilities=json")
    if result.returncode != 0:
        fail(f"{binary}: --capabilities=json exited {result.returncode}: {result.stderr}")
    if result.stderr:
        fail(f"{binary}: --capabilities=json wrote to stderr: {result.stderr!r}")
    try:
        document = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        fail(f"{binary}: invalid capability JSON: {exc}")

    if type(document) is not dict:
        fail(f"{binary}: capability document is not an object")
    if set(document) != ROOT_KEYS:
        fail(f"{binary}: unexpected root keys: {set(document) ^ ROOT_KEYS}")
    require_exact_int(document["schema_version"], 1, f"{binary}: schema_version")
    if document["engine_id"] != "ds4" or type(document["engine_id"]) is not str:
        fail(f"{binary}: invalid engine_id")
    if type(document["build_git_sha"]) is not str or not document["build_git_sha"]:
        fail(f"{binary}: invalid build_git_sha")
    if document["backend"] != backend or type(document["backend"]) is not str:
        fail(f"{binary}: expected backend {backend!r}, got {document['backend']!r}")
    if document["executable_role"] != role or type(document["executable_role"]) is not str:
        fail(f"{binary}: expected role {role!r}, got {document['executable_role']!r}")
    expected_families = ["deepseek4", "glm-dsa", "qwen35moe"]
    if document["model_families"] != expected_families:
        fail(f"{binary}: invalid model_families")
    if any(type(item) is not str for item in document["model_families"]):
        fail(f"{binary}: model_families entries must be strings")

    expert = document["expert_major"]
    if type(expert) is not dict or set(expert) != {"version", "tensor", "storage_formats"}:
        fail(f"{binary}: invalid expert_major object")
    require_exact_int(expert["version"], 2, f"{binary}: expert_major.version")
    if expert["tensor"] != "ds4.expert_major.v2" or type(expert["tensor"]) is not str:
        fail(f"{binary}: invalid ExpertMajor tensor")
    expected_storage = [
        {"id": "ggml", "wire_value": 0, "group_sizes": []},
        {"id": "mlx-affine4", "wire_value": 1, "group_sizes": [64]},
    ]
    if expert["storage_formats"] != expected_storage:
        fail(f"{binary}: invalid ExpertMajor storage formats")
    for index, storage in enumerate(expert["storage_formats"]):
        if type(storage) is not dict:
            fail(f"{binary}: storage format {index} is not an object")
        if type(storage["id"]) is not str:
            fail(f"{binary}: storage format {index} id is not a string")
        require_exact_int(storage["wire_value"], index, f"{binary}: storage wire_value")
        if type(storage["group_sizes"]) is not list:
            fail(f"{binary}: storage group_sizes is not an array")
        if any(type(size) is not int for size in storage["group_sizes"]):
            fail(f"{binary}: storage group_sizes must contain integers")

    build_info = run(binary, "--build-info")
    if build_info.returncode != 0:
        fail(f"{binary}: --build-info exited {build_info.returncode}")
    git_lines = [line[9:] for line in build_info.stdout.splitlines() if line.startswith("git:     ")]
    if git_lines != [document["build_git_sha"]]:
        fail(f"{binary}: capability SHA differs from --build-info")

    help_result = run(binary, "--help")
    if help_result.returncode != 0 or "--capabilities=json" not in help_result.stdout:
        fail(f"{binary}: help does not advertise the capability contract")

    invalid = run(binary, "--capabilities")
    if invalid.returncode != 2:
        fail(f"{binary}: bare --capabilities exited {invalid.returncode} instead of 2")
    if invalid.stdout:
        fail(f"{binary}: bare --capabilities unexpectedly wrote to stdout")
    if "unknown option: --capabilities" not in invalid.stderr:
        fail(f"{binary}: bare --capabilities did not take the invalid-option path")

    repeated = run(binary, "--capabilities=json")
    if repeated.stdout != result.stdout or repeated.stderr != result.stderr:
        fail(f"{binary}: capability output is not deterministic")
    return document


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin-dir", required=True, type=pathlib.Path)
    parser.add_argument("--backend", required=True, choices=("metal", "cpu"))
    args = parser.parse_args()

    shared: dict[str, object] | None = None
    for program, role in ROLES.items():
        binary = (args.bin_dir / program).resolve()
        if not binary.is_file():
            fail(f"missing executable: {binary}")
        document = validate(binary, args.backend, role)
        common = {key: value for key, value in document.items() if key != "executable_role"}
        if shared is None:
            shared = common
        elif common != shared:
            fail(f"{binary}: shared capability fields differ across executables")

    print(f"capabilities: PASS ({args.backend}, {len(ROLES)} executables)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"capabilities: FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
