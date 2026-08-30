#!/usr/bin/env python3
"""Offline tests for the Qwen4Exp conversion inventory dry-run."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "gguf-tools" / "qwen4exp-profile.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("qwen4exp_profile", TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def expect_profile_error(tool, inventory: dict, contract: dict | None,
                         tmp_path: Path, label: str,
                         expected_fragment: str) -> None:
    inventory_path = tmp_path / f"{label}-inventory.json"
    write_json(inventory_path, inventory)
    if contract is None:
        contract_path = tool.DEFAULT_CONTRACT
    else:
        contract_path = tmp_path / f"{label}-contract.json"
        write_json(contract_path, contract)
    try:
        tool.make_report(inventory_path, contract_path)
    except tool.ProfileError as exc:
        assert expected_fragment in str(exc), (label, str(exc))
        return
    raise AssertionError(f"mutation was admitted: {label}")


def main() -> int:
    tool = load_tool()
    report_a = tool.make_report()
    report_b = tool.make_report()
    assert report_a == report_b
    assert report_a["summary"] == {
        "classifications": {
            "PLE": 137,
            "dense": 1061,
            "excluded-MTP": 31,
            "excluded-vision": 333,
            "routed": 96,
        },
        "owningShards": 131,
        "routedDestinationComponentTensors": 144,
        "sourceBytes": 359999963128,
        "sourceBytesByClassification": {
            "PLE": 102466171160,
            "dense": 9829717760,
            "excluded-MTP": 5214301696,
            "excluded-vision": 897862112,
            "routed": 241591910400,
        },
        "sourceIdentities": 1658,
    }
    assert sum(report_a["summary"]["sourceBytesByClassification"].values()) \
        == report_a["summary"]["sourceBytes"]
    unsigned = dict(report_a)
    recorded_digest = unsigned.pop("reportSha256")
    assert hashlib.sha256(tool.canonical_json(unsigned)).hexdigest() == \
        recorded_digest

    entries = report_a["entries"]
    identities = [entry["source"]["identity"] for entry in entries]
    assert identities == sorted(identities)
    assert len(identities) == len(set(identities)) == 1658
    assert sum(entry["source"]["byteSpan"] for entry in entries) == \
        359999963128

    routed = [entry for entry in entries
              if entry["classification"] == "routed"]
    assert len(routed) == 96
    by_layer: dict[int, set[str]] = {}
    for entry in routed:
        records = entry["destination"]["records"]
        for record in records:
            identity = record["identity"]
            layer = int(identity.split(".")[1])
            by_layer.setdefault(layer, set()).add(record["component"])
            if record["component"] in ("gate", "up"):
                assert record["logicalShape"] == [2560, 640, 512]
            else:
                assert record["logicalShape"] == [640, 2560, 512]
    assert sorted(by_layer) == list(range(48))
    assert all(components == {"gate", "up", "down"}
               for components in by_layer.values())

    ple = [entry for entry in entries if entry["classification"] == "PLE"]
    ple_roles: dict[str, int] = {}
    for entry in ple:
        destination = entry["destination"]
        ple_roles[destination["role"]] = \
            ple_roles.get(destination["role"], 0) + 1
        assert destination["adapter"]["binaryEmission"] is False
        assert destination["adapter"]["codecStatus"] == \
            "descriptor-ready-page-codec-pending"
        assert (
            destination["adapter"]["manifestHeaderBytes"],
            destination["adapter"]["pageHeaderBytes"],
            destination["adapter"]["pageDigestBytes"],
            destination["adapter"]["minimumPageAlignment"],
            destination["adapter"]["rowAlignment"],
            destination["adapter"]["headRows"],
            destination["adapter"]["paddingRows"],
        ) == (512, 64, 32, 4096, 128, 320001446, 90)
    assert ple_roles == {
        "hash-control": 3,
        "row-shard": 128,
        "runtime-control": 6,
    }

    try:
        tool.QWEN4EXP_EXPERT_ADAPTER.experts = 384
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("ExpertMajor adapter descriptor is mutable")
    try:
        tool.QWEN4EXP_PLE_ADAPTER.binary_emission = True
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("PLE adapter descriptor is mutable")

    inventory = tool.read_json(tool.DEFAULT_INVENTORY)
    contract = tool.read_json(tool.DEFAULT_CONTRACT)
    with tempfile.TemporaryDirectory(prefix="qwen4exp-profile-test-") as tmp:
        tmp_path = Path(tmp)

        # Each failure mutates one ownership/admission axis. Exact fixture-file
        # pinning catches any remaining top-level drift after semantic checks.
        mutated = copy.deepcopy(inventory)
        mutated["tensors"][0]["shape"][1] -= 1
        expect_profile_error(
            tool, mutated, None, tmp_path, "shape", "byte span",
        )

        mutated = copy.deepcopy(inventory)
        mutated["tensors"][0]["dtype"] = "F32"
        expect_profile_error(
            tool, mutated, None, tmp_path, "dtype", "unsupported source dtype",
        )

        mutated = copy.deepcopy(inventory)
        mutated["tensors"][0]["shard"] = "model-00001-of-00131.safetensors"
        expect_profile_error(
            tool, mutated, None, tmp_path, "ownership", "owning shard set",
        )

        mutated = copy.deepcopy(inventory)
        mutated["tensors"][-1]["name"] = mutated["tensors"][0]["name"]
        expect_profile_error(
            tool, mutated, None, tmp_path, "duplicate", "duplicate/invalid",
        )

        mutated = copy.deepcopy(inventory)
        mutated["tensors"].pop()
        expect_profile_error(
            tool, mutated, None, tmp_path, "missing", "array length",
        )

        mutated = copy.deepcopy(inventory)
        mutated["paddingRowFacts"]["embedZero"] = \
            not mutated["paddingRowFacts"]["embedZero"]
        expect_profile_error(
            tool, mutated, None, tmp_path, "top-level", "fixture file SHA-256",
        )

        mutated_contract = copy.deepcopy(contract)
        mutated_contract["sourcePins"]["tensorCount"] -= 1
        expect_profile_error(
            tool, inventory, mutated_contract, tmp_path, "contract",
            "contract file SHA-256",
        )

        encoded = tool.canonical_json(report_a, pretty=True)
        report_path = tmp_path / "report.json"
        report_path.write_bytes(b"old report\n")
        tool.atomic_write(report_path, encoded)
        assert report_path.read_bytes() == encoded
        assert not list(tmp_path.glob(".report.json.tmp.*"))

        original_replace = tool.os.replace
        try:
            def fail_replace(_source, _destination):
                raise OSError("injected rename failure")

            tool.os.replace = fail_replace
            try:
                tool.atomic_write(report_path, b"replacement\n")
            except OSError as exc:
                assert "injected rename failure" in str(exc)
            else:
                raise AssertionError("injected atomic rename failure vanished")
        finally:
            tool.os.replace = original_replace
        assert report_path.read_bytes() == encoded
        assert not list(tmp_path.glob(".report.json.tmp.*"))

        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        cli_report = tmp_path / "cli-report.json"
        written = subprocess.run(
            [sys.executable, str(TOOL), "--dry-run",
             "--write", str(cli_report)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
            check=True,
        )
        assert written.stdout == cli_report.read_bytes() == encoded
        checked = subprocess.run(
            [sys.executable, str(TOOL), "--dry-run",
             "--check", str(cli_report)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
            check=True,
        )
        assert checked.stdout == encoded
        cli_report.write_bytes(encoded + b" ")
        drifted = subprocess.run(
            [sys.executable, str(TOOL), "--dry-run",
             "--check", str(cli_report)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
        )
        assert drifted.returncode != 0
        assert b"report drifted" in drifted.stderr

    print("qwen4exp header/index-only dry-run: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
