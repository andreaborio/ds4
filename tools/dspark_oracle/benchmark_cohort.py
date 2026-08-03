#!/usr/bin/env python3
"""Validate and render the model-backed DSpark 8K A/B/B/A plan.

This tool never opens a GGUF payload and never launches inference.  Its dry-run
output delegates each future arm to the repository's bounded M5 runner.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = Path(__file__).with_name("benchmark_8k_abba.json")
EXPECTED_SCHEMA = "hebrus.dspark.deepseek-8k-abba/1"
EXPECTED_ORDER = [
    ("A1", "baseline"),
    ("B1", "candidate"),
    ("B2", "candidate"),
    ("A2", "baseline"),
]
EXPECTED_ARTIFACTS = {
    "baseline": (
        86_720_114_240,
        "d89dd628ed786ecf14285cb886459eec01df89fd7e7bf3cbff1416a551bcd966",
        {"parent": 4129, "target": 4129, "support": 0},
    ),
    "candidate": (
        92_709_232_448,
        "c63860c8a1e49ff7a29765352cff7ef6ba8938bdf8d8d7e2be1647860d6154c8",
        {"parent": 4160, "target": 4129, "support": 31},
    ),
}
EMPTY_GIT_DIFF_SHA256 = hashlib.sha256(b"").hexdigest()


class ManifestError(ValueError):
    """The cohort declaration is incomplete or internally inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ManifestError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read manifest {path}: {exc}") from exc
    _require(isinstance(value, dict), "manifest root must be an object")
    return value


def validate_manifest(manifest: dict[str, Any], *, local: bool) -> None:
    _require(manifest.get("schema") == EXPECTED_SCHEMA, "unexpected schema")
    _require(
        manifest.get("status") == "prepared-not-executed",
        "manifest must not imply that a model cohort has run",
    )

    target = manifest.get("product_target", {})
    _require(target.get("hardware") == "Apple M5 Pro", "wrong physical target")
    _require(
        target.get("unified_memory_bytes") == 64 * 1024**3,
        "the first cohort is the exact 64 GiB lane",
    )
    _require(target.get("startup_residency") == "AUTO", "startup must be AUTO")
    _require(
        target.get("required_resolved_residency") == "SSD",
        "normal AUTO must resolve to SSD",
    )

    workload = manifest.get("workload", {})
    _require(workload.get("frontier_tokens") == 8192, "frontier must be 8192")
    _require(workload.get("decode_tokens") == 128, "decode horizon must be 128")
    _require(
        workload.get("context_allocation_tokens") == 8321,
        "context allocation must be the minimum 8192+128+1 allocation",
    )
    sampling = workload.get("sampling", {})
    _require(sampling.get("temperature") == 0.0, "benchmark must be greedy")
    _require(sampling.get("seed") == 0, "greedy seed declaration changed")
    _require(
        sampling.get("seed_consumed") is False,
        "ds4-bench greedy generation must not claim RNG consumption",
    )
    _require(
        sampling.get("implementation")
        == "ds4-bench transactional greedy argmax; EOS is an ordinary fixed-horizon token",
        "benchmark decode must use the transactional greedy path",
    )
    _require(
        sampling.get("eos_policy") == "ordinary_token_fixed_horizon",
        "benchmark EOS must remain an ordinary token for the fixed horizon",
    )
    _require(
        workload.get("discarded_identical_warmup_before_each_retained_arm")
        is True,
        "warm cohort requires one identical discarded warm-up per arm",
    )

    artifacts = manifest.get("artifacts", {})
    for variant, (size, digest, records) in EXPECTED_ARTIFACTS.items():
        artifact = artifacts.get(variant, {})
        _require(artifact.get("bytes") == size, f"{variant} byte count changed")
        _require(artifact.get("sha256") == digest, f"{variant} SHA-256 changed")
        _require(
            artifact.get("cache_records") == records,
            f"{variant} cache budget changed",
        )
        if local:
            environment_name = artifact.get("path_environment", "")
            _require(
                environment_name in os.environ,
                f"set {environment_name} for local artifact checks",
            )
            artifact_path = Path(os.environ[environment_name])
            _require(artifact_path.is_file(), f"missing local {variant} artifact")
            _require(
                artifact_path.stat().st_size == size,
                f"local {variant} artifact size mismatch",
            )

    candidate = artifacts["candidate"]
    _require(candidate.get("support_bytes") == 5_989_114_912, "support size changed")
    _require(
        candidate.get("support_sha256")
        == "aa2bd4b5b916e1aa0a01392d69cbdd9798a3f3050c29c22973c8ee4233af0413",
        "support SHA-256 changed",
    )
    if local:
        support_environment = candidate.get("support_path_environment", "")
        _require(
            support_environment in os.environ,
            f"set {support_environment} for local artifact checks",
        )
        support = Path(os.environ[support_environment])
        _require(support.is_file(), "missing local support artifact")
        _require(
            support.stat().st_size == candidate["support_bytes"],
            "local support artifact size mismatch",
        )

    memory = manifest.get("memory_comparison", {})
    _require(
        memory.get("comparison_kind")
        == "normal-AUTO product comparison; not equal instantaneous memory",
        "cohort must not be described as equal-memory",
    )
    _require(memory.get("expert_record_bytes") == 7_077_888, "record size changed")
    _require(
        memory.get("policy_reference_before_8k_tier", {}).get(
            "target_only_raw_records"
        ) == 4387,
        "pre-tier target-only policy reference changed",
    )
    _require(
        memory.get("candidate_selected_address_page_bytes") == 16_384
        and memory.get("candidate_incremental_runtime_bytes") == 29_671_424,
        "candidate production selected-address/runtime accounting changed",
    )
    _require(
        memory.get("candidate_incremental_static_plus_runtime_bytes")
        == 582_959_104,
        "candidate static/runtime increment changed",
    )
    for variant, records in EXPECTED_ARTIFACTS.items():
        expected_records = records[2]
        declared = memory.get("post_prefill_cache", {}).get(variant, {})
        for key in ("parent", "target", "support"):
            _require(
                declared.get(f"{key}_records") == expected_records[key],
                f"{variant} post-prefill {key} cache declaration changed",
            )

    runner = manifest.get("runner", {})
    _require(runner.get("mode_argument") == "auto", "runner mode must be auto")
    environment = runner.get("environment", {})
    expected_environment = {
        "DS4_M5_RESIDENCY": "auto",
        "DS4_M5_CACHE_STATE": "warm",
        "DS4_M5_CTX_START": "8192",
        "DS4_M5_CTX_MAX": "8192",
        "DS4_M5_CTX_ALLOC": "8321",
        "DS4_M5_MAX_SWAPOUT_PAGES": "0",
    }
    for name, expected in expected_environment.items():
        _require(environment.get(name) == expected, f"{name} changed")
    _require(
        [(arm.get("id"), arm.get("variant")) for arm in runner.get("order", [])]
        == EXPECTED_ORDER,
        "retained order must be A1,B1,B2,A2",
    )
    binary_identity = runner.get("binary_identity", {})
    _require(
        binary_identity.get("path_within_each_clean_worktree")
        == "build/metal-arm64/bin/ds4-bench",
        "benchmark binary must be the canonical build in each clean worktree",
    )
    _require(
        binary_identity.get("build_git")
        == "exact first 12 hexadecimal characters of that worktree HEAD, with no dirty suffix",
        "benchmark build-Git identity policy changed",
    )
    _require(
        binary_identity.get("backend") == "metal"
        and binary_identity.get("arch") == "arm64"
        and binary_identity.get("runtime_build_line_required") is True,
        "benchmark binary must prove a Metal arm64 runtime build identity",
    )
    _require(
        binary_identity.get("comparison_revision")
        == "A and B use byte-identical clean builds from the same exact main commit; only the authenticated model artifact changes",
        "A/B must use one identical clean product revision",
    )
    readiness = runner.get("execution_readiness", {})
    _require(
        readiness.get("status") == "blocked"
        and "production DSpark N>1 proposal and exact verifier"
        in readiness.get("blocker", ""),
        "the prepared cohort must expose the current production DSpark blocker",
    )
    _require(
        "one identical clean main commit" in readiness.get("required_preflight", "")
        and "transactional fixed-horizon decode"
        in readiness.get("required_preflight", ""),
        "identical-revision transactional preflight declaration is missing",
    )
    forbidden = set(runner.get("forbidden_acceptance_controls", []))
    _require(
        {"--resident", "--ssd-streaming", "--ssd-streaming-cache-experts"}
        <= forbidden,
        "acceptance controls do not fail closed",
    )

    prompt = ROOT / workload.get("prompt_path", "")
    _require(prompt.is_file(), "canonical prompt is missing")
    _require(_sha256(prompt) == workload.get("prompt_sha256"), "prompt SHA-256 drift")
    runner_path = ROOT / runner.get("path", "")
    _require(runner_path.is_file(), "bounded M5 runner is missing")

    oracle = manifest.get("oracle", {})
    _require(
        oracle.get("support_authentication")
        == "hash the external SUPPORT GGUF exactly once before either oracle; require device, inode, byte size and mtime to remain unchanged through both oracle runs",
        "SUPPORT oracle authentication policy changed",
    )
    _require(
        oracle.get("required_after_support_hash_and_before_benchmark_model_hash_or_warmup")
        is True,
        "oracle ordering policy changed",
    )
    mlx = oracle.get("mlx_parity", {})
    system_python = oracle.get("system_python", {})
    _require(
        system_python.get("expected_test_count") == 73
        and system_python.get("expected_skip_count") == 1
        and system_python.get("argv") == ["tests/dspark/test_oracle.py", "-v"],
        "system-Python oracle count/skip evidence contract changed",
    )
    _require(
        mlx.get("expected_test_count") == 73
        and mlx.get("expected_skip_count") == 0
        and mlx.get("argv") == ["tests/dspark/test_oracle.py", "-v"],
        "MLX oracle must execute all declared tests without skips",
    )
    _require(mlx.get("mlx_version") == "0.32.0", "MLX version must be 0.32.0")
    _require(
        mlx.get("mlx_metal_version") == "0.32.0",
        "mlx-metal version must be 0.32.0",
    )
    _require(mlx.get("numpy_version") == "2.4.6", "NumPy version must be 2.4.6")
    requirements = ROOT / mlx.get("requirements_path", "")
    _require(requirements.is_file(), "MLX requirements file is missing")
    pins = {
        line.strip()
        for line in requirements.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    _require(
        pins == {"mlx==0.32.0", "mlx-metal==0.32.0", "numpy==2.4.6"},
        "MLX requirements no longer match the manifest",
    )

    metrics = manifest.get("metrics", {})
    required_metrics = {
        "prefill_tps",
        "gen_tps",
        "gen_tpot_p50_ms",
        "gen_tpot_p95_ms",
        "prefill_pread_gib_per_tok",
        "prefill_pread_gib",
        "gen_pread_gib",
        "gen_pread_gib_per_tok",
    }
    _require(
        required_metrics <= set(metrics.get("csv_columns", [])),
        "required phase/TPOT/SSD metrics are missing",
    )
    _require(
        metrics.get("decode_ssd_bytes_per_token_must_not_increase") is True,
        "candidate SSD bytes/token veto is missing",
    )
    invalidation = " ".join(manifest.get("cohort_invalidation", [])).lower()
    for word in ("swapout", "abort_reason", "resolved plan", "parity", "order"):
        _require(word in invalidation, f"cohort invalidation omits {word}")


def _q(value: str | Path) -> str:
    return shlex.quote(str(value))


def _read_summary(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ManifestError(f"cannot read {path}: {exc}") from exc
    values: dict[str, str] = {}
    for line_number, line in enumerate(lines, 1):
        if "=" not in line:
            raise ManifestError(f"{path}:{line_number}: malformed summary line")
        key, value = line.split("=", 1)
        if not key or key in values:
            raise ManifestError(f"{path}:{line_number}: duplicate/empty key")
        values[key] = value
    return values


def _require_int(values: dict[str, str], key: str, expected: int | None = None) -> int:
    raw = values.get(key, "")
    if not raw.isdigit():
        raise ManifestError(f"summary field {key} is not an unsigned integer")
    value = int(raw)
    if expected is not None and value != expected:
        raise ManifestError(f"summary field {key}={value}, expected {expected}")
    return value


def _oracle_contract(manifest: dict[str, Any], lane: str) -> dict[str, Any]:
    oracle = manifest["oracle"]
    if lane == "system":
        return oracle["system_python"]
    if lane == "mlx":
        return oracle["mlx_parity"]
    raise ManifestError(f"unknown oracle lane {lane!r}")


def _parse_oracle_log(path: Path) -> tuple[int, int]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ManifestError(f"cannot read oracle log {path}: {exc}") from exc
    ran = re.findall(r"^Ran ([0-9]+) tests? in [^\n]+$", text, re.MULTILINE)
    success = re.findall(r"^OK(?: \(skipped=([0-9]+)\))?$", text, re.MULTILINE)
    if len(ran) != 1 or len(success) != 1 or "FAILED" in text:
        raise ManifestError(f"{path}: incomplete or unsuccessful unittest result")
    return int(ran[0]), int(success[0] or "0")


def _validated_oracle_row(
        manifest: dict[str, Any], lane: str, path: Path) -> dict[str, str]:
    tests, skipped = _parse_oracle_log(path)
    contract = _oracle_contract(manifest, lane)
    expected_tests = contract["expected_test_count"]
    expected_skips = contract["expected_skip_count"]
    if tests != expected_tests:
        raise ManifestError(
            f"{lane} oracle ran {tests} tests, expected {expected_tests}"
        )
    if skipped != expected_skips:
        raise ManifestError(
            f"{lane} oracle skipped {skipped} tests, expected {expected_skips}"
        )
    return {
        "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "lane": lane,
        "tests": str(tests),
        "skipped": str(skipped),
        "log_sha256": _sha256(path),
        "result": "pass",
    }


def _validate_oracle_results(root: Path, manifest: dict[str, Any]) -> None:
    path = root / "oracle-results.tsv"
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source, delimiter="\t"))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ManifestError(f"cannot read oracle result ledger: {exc}") from exc
    if [row.get("lane") for row in rows] != ["system", "mlx"]:
        raise ManifestError("oracle result ledger must contain system then MLX")
    previous: dt.datetime | None = None
    for row in rows:
        lane = row["lane"]
        log = root / f"oracle-{lane}.log"
        actual = _validated_oracle_row(manifest, lane, log)
        for key in ("tests", "skipped", "log_sha256", "result"):
            if row.get(key) != actual[key]:
                raise ManifestError(f"{lane} oracle ledger {key} mismatch")
        try:
            timestamp = dt.datetime.fromisoformat(row.get("timestamp", ""))
        except ValueError as exc:
            raise ManifestError(f"{lane} oracle ledger timestamp is invalid") from exc
        if timestamp.tzinfo is None:
            raise ManifestError(f"{lane} oracle ledger timestamp has no timezone")
        if previous is not None and timestamp < previous:
            raise ManifestError("oracle result timestamps are not chronological")
        previous = timestamp


def _read_csv_row(path: Path, required: set[str]) -> dict[str, float]:
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ManifestError(f"cannot read {path}: {exc}") from exc
    if len(rows) != 1 or reader.fieldnames is None:
        raise ManifestError(f"{path} must contain exactly one isolated frontier row")
    if not required <= set(reader.fieldnames):
        missing = sorted(required - set(reader.fieldnames))
        raise ManifestError(f"{path} is missing CSV fields {missing}")
    parsed: dict[str, float] = {}
    for name, raw in rows[0].items():
        if raw is None:
            raise ManifestError(f"{path}: missing value for {name}")
        try:
            value = float(raw)
        except ValueError as exc:
            raise ManifestError(f"{path}: invalid numeric {name}={raw}") from exc
        if not math.isfinite(value):
            raise ManifestError(f"{path}: non-finite numeric {name}")
        parsed[name] = value
    return parsed


def _canonical_json_hash(path: Path, *, exclude: set[str]) -> tuple[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot parse evidence {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"evidence root must be an object: {path}")
    value = {key: item for key, item in value.items() if key not in exclude}
    try:
        canonical = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"cannot canonicalize evidence {path}: {exc}") from exc
    return hashlib.sha256(canonical).hexdigest(), value


def _percent_delta(value: float, reference: float) -> float:
    if reference == 0.0:
        raise ManifestError("cannot calculate a percent delta from zero")
    return 100.0 * (value / reference - 1.0)


def _spread_percent(first: float, second: float) -> float:
    mean = (first + second) / 2.0
    return abs(_percent_delta(first, mean) - _percent_delta(second, mean))


def _validate_order_ledger(root: Path) -> dict[str, str]:
    path = root / "order.tsv"
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source, delimiter="\t"))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ManifestError(f"cannot read cohort order ledger: {exc}") from exc
    expected: list[tuple[str, str, str, str]] = []
    for arm, variant in EXPECTED_ORDER:
        expected.extend([
            (arm, variant, "warmup", "start"),
            (arm, variant, "warmup", "complete"),
            (arm, variant, "retained", "start"),
            (arm, variant, "retained", "complete"),
        ])
    actual = [
        (row.get("arm"), row.get("variant"), row.get("phase"), row.get("event"))
        for row in rows
    ]
    if actual != expected:
        raise ManifestError("cohort ledger is incomplete or not A1,B1,B2,A2")
    timestamps: dict[str, str] = {}
    previous: dt.datetime | None = None
    for row in rows:
        raw = row.get("timestamp", "")
        try:
            timestamp = dt.datetime.fromisoformat(raw)
        except ValueError as exc:
            raise ManifestError(f"invalid ledger timestamp {raw!r}") from exc
        if timestamp.tzinfo is None:
            raise ManifestError("ledger timestamp has no timezone")
        if previous is not None and timestamp < previous:
            raise ManifestError("cohort timestamps are not chronological")
        previous = timestamp
        if row["phase"] == "retained" and row["event"] == "start":
            timestamps[row["arm"]] = raw
    return timestamps


def _read_cohort_identity(root: Path) -> dict[str, str]:
    path = root / "identity.tsv"
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source, delimiter="\t"))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ManifestError(f"cannot read cohort identity: {exc}") from exc
    if len(rows) != 1:
        raise ManifestError("cohort identity must contain exactly one record")
    identity = rows[0]
    for key in ("baseline_head", "baseline_main_head", "candidate_head"):
        if re.fullmatch(r"[0-9a-f]{40}", identity.get(key, "")) is None:
            raise ManifestError(f"cohort identity has invalid {key}")
    if identity["baseline_head"] != identity["baseline_main_head"]:
        raise ManifestError("baseline HEAD is not the baseline worktree main ref")
    if identity["candidate_head"] != identity["baseline_head"]:
        raise ManifestError("A/B cohort does not use one identical product revision")
    return identity


def _validate_support_oracle_identity(
        root: Path, manifest: dict[str, Any]) -> None:
    candidate = manifest["artifacts"]["candidate"]
    evidence = _read_summary(root / "model-hash/support.txt")
    for key, expected in (
        ("schema", "1"),
        ("model_bytes", str(candidate["support_bytes"])),
        ("model_sha256_expected", candidate["support_sha256"]),
        ("model_sha256_actual", candidate["support_sha256"]),
    ):
        if evidence.get(key) != expected:
            raise ManifestError(f"SUPPORT oracle hash evidence {key} mismatch")
    for key in ("model_path_sha256",):
        if re.fullmatch(r"[0-9a-f]{64}", evidence.get(key, "")) is None:
            raise ManifestError(f"SUPPORT oracle hash evidence has invalid {key}")
    try:
        verified_at = dt.datetime.fromisoformat(evidence.get("verified_at", ""))
    except ValueError as exc:
        raise ManifestError("SUPPORT oracle hash timestamp is invalid") from exc
    if verified_at.tzinfo is None:
        raise ManifestError("SUPPORT oracle hash timestamp has no timezone")

    identity_path = root / "oracle-support-identity.tsv"
    try:
        with identity_path.open("r", encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source, delimiter="\t"))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ManifestError(f"cannot read SUPPORT oracle identity: {exc}") from exc
    if [row.get("phase") for row in rows] != ["before", "after"]:
        raise ManifestError("SUPPORT oracle identity must bracket both oracle runs")
    identity_fields = ("device", "inode", "bytes", "mtime_epoch")
    before = tuple(rows[0].get(key, "") for key in identity_fields)
    after = tuple(rows[1].get(key, "") for key in identity_fields)
    if before != after:
        raise ManifestError("SUPPORT oracle artifact identity changed during oracle runs")
    expected_identity = tuple(
        evidence.get(key, "")
        for key in ("model_device", "model_inode", "model_bytes", "model_mtime_epoch")
    )
    if before != expected_identity:
        raise ManifestError("SUPPORT oracle identity does not match its SHA evidence")
    if not all(value.isdigit() for value in before):
        raise ManifestError("SUPPORT oracle identity contains a non-integer field")
    for row in rows:
        try:
            timestamp = dt.datetime.fromisoformat(row.get("timestamp", ""))
        except ValueError as exc:
            raise ManifestError("SUPPORT oracle identity timestamp is invalid") from exc
        if timestamp.tzinfo is None:
            raise ManifestError("SUPPORT oracle identity timestamp has no timezone")


def _read_build_info(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ManifestError(f"cannot read build identity {path}: {exc}") from exc
    if len(lines) != 4 or lines[0] not in {"ds4 build", "hebrus build"}:
        raise ManifestError(f"{path}: malformed --build-info output")
    result: dict[str, str] = {}
    for line, key in zip(lines[1:], ("git", "backend", "arch"), strict=True):
        match = re.fullmatch(rf"{key}:\s+(.+)", line)
        if match is None:
            raise ManifestError(f"{path}: malformed {key} build identity")
        result[key] = match.group(1)
    return result


def _read_metal_library_identity(path: Path) -> dict[str, str]:
    try:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
    except (OSError, UnicodeError) as exc:
        raise ManifestError(f"cannot read runtime Metal identity {path}: {exc}") from exc
    lines = text.splitlines()
    if not text.endswith("\n") or len(lines) != 1:
        raise ManifestError(f"{path}: runtime Metal identity must be one complete line")
    match = re.fullmatch(
        r"ds4: metal_library source_sha256=([0-9a-f]{64}) "
        r"overrides=([0-9]+) tensor=(on|off) norm_unify=(on|off) "
        r"kv_raw_f32=(on|off) math=(safe|fast)",
        lines[0],
    )
    if match is None:
        raise ManifestError(f"{path}: malformed runtime Metal identity")
    return {
        "runtime_metal_library": lines[0],
        "runtime_metal_library_sha256": hashlib.sha256(raw).hexdigest(),
    }


def validate_results(root: Path, manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    if not root.is_absolute() or not root.is_dir():
        raise ManifestError("result root must be an existing absolute directory")
    timestamps = _validate_order_ledger(root)
    cohort_identity = _read_cohort_identity(root)
    _validate_support_oracle_identity(root, manifest)
    _validate_oracle_results(root, manifest)
    copied_manifest_path = root / "manifest.json"
    copied_manifest = load_manifest(copied_manifest_path)
    if copied_manifest != manifest:
        raise ManifestError("cohort manifest copy differs from the validated manifest")
    plan_path = root / "plan.sha256"
    try:
        plan_lines = plan_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ManifestError(f"cannot read cohort plan identity: {exc}") from exc
    if len(plan_lines) != 2 or any(
        re.fullmatch(r"[0-9a-f]{64}\s+.+", line) is None for line in plan_lines
    ):
        raise ManifestError("cohort plan identity must hash manifest and runner")
    if plan_lines[0].split()[0] != _sha256(copied_manifest_path):
        raise ManifestError("cohort manifest hash does not match plan identity")
    canonical_runner = ROOT / manifest["runner"]["path"]
    if plan_lines[1].split()[0] != _sha256(canonical_runner):
        raise ManifestError("cohort runner hash does not match the canonical runner")
    required_csv = set(manifest["metrics"]["csv_columns"]) | {
        "ctx_tokens", "prefill_tokens", "gen_tokens"
    }
    parity = manifest["parity"]
    artifacts = manifest["artifacts"]
    rows: list[dict[str, Any]] = []
    identities: dict[str, dict[str, str]] = {}
    errors: list[str] = []

    for arm, variant in EXPECTED_ORDER:
        prefix = root / arm
        summary = _read_summary(Path(f"{prefix}.summary"))
        for key, expected in (
            ("mode", "auto"),
            ("residency", "auto"),
            ("cache", "auto"),
            ("cache_state", "warm"),
            ("abort_reason", "none"),
            ("result_error", "none"),
            ("process_contamination", "none"),
        ):
            if summary.get(key) != expected:
                errors.append(f"{arm}: {key}={summary.get(key)!r}, expected {expected!r}")
        for key in (
            "process_rc", "rc", "swapout_pages_delta",
            "repo_untracked_count", "post_repo_untracked_count",
        ):
            try:
                _require_int(summary, key, 0)
            except ManifestError as exc:
                errors.append(f"{arm}: {exc}")
        if summary.get("repo_diff_sha256") != EMPTY_GIT_DIFF_SHA256:
            errors.append(
                f"{arm}: tracked worktree diff is not the canonical empty diff"
            )
        for before_key, after_key in (
            ("bin_sha256", "post_bin_sha256"),
            ("repo_head", "post_repo_head"),
            ("repo_diff_sha256", "post_repo_diff_sha256"),
            (
                "repo_untracked_manifest_sha256",
                "post_repo_untracked_manifest_sha256",
            ),
            ("repo_source_state_sha256", "post_repo_source_state_sha256"),
            (
                "metal_file_set_manifest_sha256",
                "post_metal_file_set_manifest_sha256",
            ),
            ("prompt_source_sha256", "post_prompt_source_sha256"),
            ("prompt_sha256", "post_prompt_sha256"),
            ("prompt_bytes", "post_prompt_bytes"),
        ):
            if (
                not summary.get(before_key)
                or summary.get(after_key) != summary.get(before_key)
            ):
                errors.append(
                    f"{arm}: post-arm {before_key} identity does not match pre-run"
                )
        try:
            _require_int(summary, "host_memory_bytes", 64 * 1024**3)
            pressure_min = _require_int(summary, "pressure_min")
            if pressure_min < 20:
                errors.append(f"{arm}: pressure_min={pressure_min}% is below 20%")
        except ManifestError as exc:
            errors.append(f"{arm}: {exc}")
        if "AC Power" not in summary.get("power_source", ""):
            errors.append(f"{arm}: arm was not recorded on AC power")
        expected_artifact = artifacts[variant]
        for key, expected in (
            ("model_sha256_expected", expected_artifact["sha256"]),
            ("model_sha256_actual", expected_artifact["sha256"]),
            ("model_bytes", str(expected_artifact["bytes"])),
            ("prompt_sha256", manifest["workload"]["prompt_sha256"]),
            ("ctx_start", "8192"),
            ("ctx_max", "8192"),
            ("ctx_alloc", "8321"),
            ("gen_tokens", "128"),
        ):
            if summary.get(key) != expected:
                errors.append(f"{arm}: {key} identity mismatch")

        stderr_path = Path(f"{prefix}.stderr")
        resolved_path = Path(f"{prefix}.resolved-plan")
        try:
            stderr = stderr_path.read_text(encoding="utf-8")
            resolved = resolved_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ManifestError(f"cannot read {arm} runtime plan artifacts: {exc}") from exc
        combined_plan = stderr + "\n" + resolved
        expected_repo_head = (
            cohort_identity["baseline_head"]
            if variant == "baseline"
            else cohort_identity["candidate_head"]
        )
        expected_build_git = expected_repo_head[:12]
        build_info = _read_build_info(Path(f"{prefix}.build-info"))
        if build_info != {
            "git": expected_build_git,
            "backend": "metal",
            "arch": "arm64",
        }:
            errors.append(
                f"{arm}: compiled build identity does not match clean repo_head "
                f"{expected_build_git} metal-arm64"
            )
        if not summary.get("bin", "").endswith(
            "/build/metal-arm64/bin/ds4-bench"
        ):
            errors.append(f"{arm}: benchmark binary is not the canonical worktree build")
        runtime_build = re.escape(expected_build_git)
        if re.search(
            rf"^ds4: build git={runtime_build} "
            rf"compiled=metal-arm64 runtime=metal$",
            combined_plan,
            re.MULTILINE,
        ) is None:
            errors.append(f"{arm}: runtime build line does not match repo_head")
        if not re.search(r"residency requested=auto resolved=ssd\b", combined_plan):
            errors.append(f"{arm}: AUTO did not record resolved SSD residency")
        records = expected_artifact["cache_records"]
        if variant == "baseline":
            pattern = rf"cached expert count:\s*{records['target']}\b"
        else:
            pattern = (
                rf"cycle-aligned parent\s*{records['parent']}\s*=\s*TARGET\s*"
                rf"{records['target']}\s*\+\s*SUPPORT\s*{records['support']}\b"
            )
        if not re.search(pattern, combined_plan):
            errors.append(f"{arm}: expected {variant} cache-record split is missing")
        if variant == "candidate" and not re.search(
            r"DSpark static payload\s+553290668 bytes;.*"
            r"speculative runtime\s+28\.30 MiB",
            combined_plan,
        ):
            errors.append(f"{arm}: candidate static/runtime accounting line is missing")

        try:
            csv_row = _read_csv_row(Path(f"{prefix}.csv"), required_csv)
        except ManifestError as exc:
            raise ManifestError(f"{arm}: {exc}") from exc
        for key, expected in (
            ("ctx_tokens", 8192.0),
            ("prefill_tokens", 8192.0),
            ("gen_tokens", 128.0),
        ):
            if csv_row[key] != expected:
                errors.append(f"{arm}: CSV {key}={csv_row[key]}, expected {expected}")
        for key in required_csv - {"ctx_tokens", "prefill_tokens", "gen_tokens"}:
            if csv_row[key] < 0.0:
                errors.append(f"{arm}: CSV {key} is negative")
        for key in ("prefill_tps", "gen_tps", "gen_tpot_p50_ms", "gen_tpot_p95_ms"):
            if csv_row[key] <= 0.0:
                errors.append(f"{arm}: CSV {key} must be positive")

        logit_files = sorted(Path(f"{prefix}.logits").glob(parity["frontier_logits"]["file_glob"]))
        evidence_files = sorted(Path(f"{prefix}.evidence").glob(parity["decode_output"]["file_glob"]))
        if len(logit_files) != 1 or len(evidence_files) != 1:
            raise ManifestError(f"{arm}: expected one logits and one decode JSON")
        logit_hash, logit_value = _canonical_json_hash(
            logit_files[0], exclude=set(parity["frontier_logits"]["exclude_fields"])
        )
        decode_hash, decode_value = _canonical_json_hash(evidence_files[0], exclude=set())
        if logit_value.get("frontier_tokens") != 8192:
            errors.append(f"{arm}: frontier-logit evidence is not the 8K row")
        if decode_value.get("schema") != parity["decode_output"]["runner_schema"]:
            errors.append(f"{arm}: unexpected decode-evidence schema")
        token_ids = decode_value.get("token_ids")
        if not isinstance(token_ids, list) or len(token_ids) != 128:
            errors.append(f"{arm}: decode evidence does not contain 128 token IDs")
        final_logits = decode_value.get("final_logits")
        if not isinstance(final_logits, list) or not final_logits:
            errors.append(f"{arm}: decode evidence has no final logits")

        identity_keys = (
            "repo_head", "repo_diff_sha256", "repo_source_state_sha256",
            "bin_sha256", "metal_file_set_manifest_sha256",
        )
        identities[arm] = {key: summary.get(key, "") for key in identity_keys}
        identities[arm]["build_git"] = build_info.get("git", "")
        identities[arm].update(
            _read_metal_library_identity(Path(f"{prefix}.metal-library"))
        )
        row = {
            "timestamp": timestamps[arm],
            "arm": arm,
            "variant": variant,
            **csv_row,
            "full_request_wall_ms": csv_row["prefill_wall_ms"] + csv_row["gen_wall_ms"],
            "prefill_ssd_bytes_per_token": csv_row["prefill_pread_gib_per_tok"] * 1024**3,
            "decode_ssd_bytes_per_token": csv_row["gen_pread_gib_per_tok"] * 1024**3,
            "total_ssd_bytes_per_emitted_token": (
                (csv_row["prefill_pread_gib"] + csv_row["gen_pread_gib"])
                * 1024**3 / csv_row["gen_tokens"]
            ),
            "frontier_logits_sha256": logit_hash,
            "decode_output_sha256": decode_hash,
            "repo_head": summary.get("repo_head", ""),
            "bin_sha256": summary.get("bin_sha256", ""),
            "model_sha256": summary.get("model_sha256_actual", ""),
        }
        rows.append(row)

    for first, second in (("A1", "A2"), ("B1", "B2")):
        if identities[first] != identities[second]:
            errors.append(f"{first}/{second}: executable/repository/Metal identity drift")
    if any(identities[arm] != identities["A1"] for arm in ("B1", "B2", "A2")):
        errors.append("cohort: A/B executable/repository/Metal identity differs")
    for arm in ("A1", "A2"):
        if identities[arm]["repo_head"] != cohort_identity["baseline_head"]:
            errors.append(f"{arm}: summary repo_head is not tested main")
    for arm in ("B1", "B2"):
        if identities[arm]["repo_head"] != cohort_identity["candidate_head"]:
            errors.append(f"{arm}: summary repo_head is not the declared candidate")
    common_summary_fields = ("os_build", "power_source", "host_memory_bytes", "prompt_sha256")
    summaries = [_read_summary(Path(f"{root / arm}.summary")) for arm, _ in EXPECTED_ORDER]
    for field in common_summary_fields:
        if len({summary.get(field) for summary in summaries}) != 1:
            errors.append(f"cohort: {field} differs across arms")

    for parity_key in ("frontier_logits_sha256", "decode_output_sha256"):
        if len({row[parity_key] for row in rows}) != 1:
            errors.append(f"cohort: exact {parity_key} parity failed")

    indexed = {row["arm"]: row for row in rows}
    baseline_wall = (indexed["A1"]["full_request_wall_ms"] + indexed["A2"]["full_request_wall_ms"]) / 2.0
    candidate_wall = (indexed["B1"]["full_request_wall_ms"] + indexed["B2"]["full_request_wall_ms"]) / 2.0
    control_spread = _spread_percent(
        indexed["A1"]["full_request_wall_ms"], indexed["A2"]["full_request_wall_ms"]
    )
    candidate_spread = _spread_percent(
        indexed["B1"]["full_request_wall_ms"], indexed["B2"]["full_request_wall_ms"]
    )
    if control_spread > manifest["metrics"]["control_drift_limit_percent"]:
        errors.append(f"cohort: A control drift {control_spread:.3f}% exceeds 3%")
    improvement = -_percent_delta(candidate_wall, baseline_wall)
    if improvement <= max(control_spread, candidate_spread):
        errors.append(
            "cohort: full-request improvement does not exceed control/candidate spread"
        )

    regression_limit = manifest["metrics"]["per_metric_regression_limit_percent"]
    higher_is_better = ("prefill_tps", "gen_tps")
    lower_is_better = (
        "prefill_wall_ms", "ttft_ms", "gen_wall_ms",
        "gen_tpot_p50_ms", "gen_tpot_p95_ms"
    )
    for arm in ("B1", "B2"):
        for key in higher_is_better:
            baseline = (indexed["A1"][key] + indexed["A2"][key]) / 2.0
            if _percent_delta(indexed[arm][key], baseline) < -regression_limit:
                errors.append(f"{arm}: {key} regresses by more than {regression_limit}%")
        for key in lower_is_better:
            baseline = (indexed["A1"][key] + indexed["A2"][key]) / 2.0
            if _percent_delta(indexed[arm][key], baseline) > regression_limit:
                errors.append(f"{arm}: {key} regresses by more than {regression_limit}%")
    baseline_ssd = (
        indexed["A1"]["decode_ssd_bytes_per_token"]
        + indexed["A2"]["decode_ssd_bytes_per_token"]
    ) / 2.0
    for arm in ("B1", "B2"):
        if indexed[arm]["decode_ssd_bytes_per_token"] > baseline_ssd:
            errors.append(f"{arm}: decode SSD bytes/token increased")
    baseline_total_ssd = (
        indexed["A1"]["total_ssd_bytes_per_emitted_token"]
        + indexed["A2"]["total_ssd_bytes_per_emitted_token"]
    ) / 2.0
    for arm in ("B1", "B2"):
        if indexed[arm]["total_ssd_bytes_per_emitted_token"] > baseline_total_ssd:
            errors.append(f"{arm}: total SSD bytes/emitted-token increased")
    return rows, errors


def print_results(
        rows: list[dict[str, Any]], errors: list[str], manifest: dict[str, Any]) -> None:
    indexed = {row["arm"]: row for row in rows}
    artifacts = manifest["artifacts"]
    memory = manifest["memory_comparison"]
    static_pages = memory["static_pages"]
    baseline_cache = memory["post_prefill_cache"]["baseline"]
    candidate_cache = memory["post_prefill_cache"]["candidate"]
    print("| Variant | GGUF mapped bytes | Static page bytes | Incremental DSpark runtime bytes | Static + runtime delta vs A | Post-prefill cache split | Cache bytes |")
    print("| --- | ---: | ---: | ---: | ---: | --- | ---: |")
    print(
        f"| A tested-main target-only | {artifacts['baseline']['bytes']} | "
        f"{static_pages['baseline_target_bytes']} | "
        "0 incremental (common runtime not normalized) | 0 | "
        f"parent {baseline_cache['parent_records']} = TARGET "
        f"{baseline_cache['target_records']} + SUPPORT "
        f"{baseline_cache['support_records']} | {baseline_cache['parent_bytes']} |"
    )
    print(
        f"| B complete DSpark stack | {artifacts['candidate']['bytes']} | "
        f"{static_pages['candidate_combined_bytes']} | "
        f"{memory['candidate_incremental_runtime_bytes']} | "
        f"{memory['candidate_incremental_static_plus_runtime_bytes']} | "
        f"parent {candidate_cache['parent_records']} = TARGET "
        f"{candidate_cache['target_records']} + SUPPORT "
        f"{candidate_cache['support_records']} | {candidate_cache['parent_bytes']} |"
    )
    print("\nNormal-AUTO product comparison; these rows are not instantaneous equal-memory arms and no win may be normalized by footprint.\n")
    print("| Variant | Retained arms | repo_head | Binary SHA-256 | Model SHA-256 | Cache plan |")
    print("| --- | --- | --- | --- | --- | --- |")
    print(
        f"| A tested main | A1,A2 | {indexed['A1']['repo_head']} | "
        f"{indexed['A1']['bin_sha256']} | {indexed['A1']['model_sha256']} | "
        "AUTO->SSD; parent 4129 = TARGET 4129 + SUPPORT 0 |"
    )
    print(
        f"| B candidate | B1,B2 | {indexed['B1']['repo_head']} | "
        f"{indexed['B1']['bin_sha256']} | {indexed['B1']['model_sha256']} | "
        "AUTO->SSD; parent 4160 = TARGET 4129 + SUPPORT 31 |"
    )
    print()
    baseline_prefill = (indexed["A1"]["prefill_tps"] + indexed["A2"]["prefill_tps"]) / 2.0
    baseline_decode = (indexed["A1"]["gen_tps"] + indexed["A2"]["gen_tps"]) / 2.0
    baseline_wall = (indexed["A1"]["full_request_wall_ms"] + indexed["A2"]["full_request_wall_ms"]) / 2.0
    print("| Timestamp | Arm | Variant | Prefill t/s | Δ vs A mean | Decode t/s | Δ vs A mean | TPOT p50 ms | TPOT p95 ms | SSD decode B/tok | Full wall ms | Δ vs A mean | Δ vs previous experiment | Parity |")
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    parity_ok = len({row["frontier_logits_sha256"] for row in rows}) == 1 and len(
        {row["decode_output_sha256"] for row in rows}
    ) == 1
    for index, row in enumerate(rows):
        if index == 0:
            previous_delta = "N/A - first arm"
        else:
            previous = rows[index - 1]
            previous_delta = (
                f"{_percent_delta(row['full_request_wall_ms'], previous['full_request_wall_ms']):+.3f}% "
                f"vs {previous['arm']}"
            )
        print(
            f"| {row['timestamp']} | {row['arm']} | {row['variant']} | "
            f"{row['prefill_tps']:.2f} | {_percent_delta(row['prefill_tps'], baseline_prefill):+.3f}% | "
            f"{row['gen_tps']:.2f} | {_percent_delta(row['gen_tps'], baseline_decode):+.3f}% | "
            f"{row['gen_tpot_p50_ms']:.3f} | {row['gen_tpot_p95_ms']:.3f} | "
            f"{row['decode_ssd_bytes_per_token']:.0f} | "
            f"{row['full_request_wall_ms']:.3f} | "
            f"{_percent_delta(row['full_request_wall_ms'], baseline_wall):+.3f}% | "
            f"{previous_delta} | {'exact' if parity_ok else 'FAIL'} |"
        )
    if errors:
        print("\nFail-closed findings:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
    else:
        print(
            "\n8K cohort is an exact, safe exploration survivor. This is not "
            "a promotion claim: measured resolution and remaining context/model "
            "lanes are still blocking."
        )


def render_dry_run(manifest_path: Path, manifest: dict[str, Any]) -> str:
    workload = manifest["workload"]
    artifacts = manifest["artifacts"]
    runner = manifest["runner"]
    environment = runner["environment"]
    cooldown = workload["cooldown_seconds_between_retained_arms"]
    prompt = ROOT / workload["prompt_path"]
    support_environment = artifacts["candidate"]["support_path_environment"]

    lines = [
        "#!/bin/zsh",
        "set -euo pipefail",
        "",
        ': "${HEBRUS_DSPARK_BASELINE_ROOT:?set clean tested-main worktree}"',
        ': "${HEBRUS_DSPARK_CANDIDATE_ROOT:?set clean identical-revision candidate worktree}"',
        ': "${HEBRUS_DSPARK_BASELINE_HEAD:?set exact tested-main commit}"',
        ': "${HEBRUS_DSPARK_CANDIDATE_HEAD:?set exact candidate commit}"',
        ': "${HEBRUS_DSPARK_MLX_PYTHON:?set pinned MLX Python path}"',
        ': "${HEBRUS_DSPARK_COHORT_ROOT:?set fresh absolute output directory}"',
        ': "${HEBRUS_DSPARK_BASELINE_MODEL:?set authenticated target-only GGUF path}"',
        ': "${HEBRUS_DSPARK_CANDIDATE_MODEL:?set authenticated combined GGUF path}"',
        ': "${HEBRUS_DSPARK_SUPPORT_MODEL:?set authenticated support GGUF path}"',
        "",
        f"MANIFEST={_q(manifest_path.resolve())}",
        f"RUNNER={_q((ROOT / runner['path']).resolve())}",
        f"PROMPT={_q(prompt.resolve())}",
        f"SUPPORT=${{{support_environment}}}",
        "BASELINE_MODEL=$HEBRUS_DSPARK_BASELINE_MODEL",
        "CANDIDATE_MODEL=$HEBRUS_DSPARK_CANDIDATE_MODEL",
        "BASELINE_BIN=$HEBRUS_DSPARK_BASELINE_ROOT/build/metal-arm64/bin/ds4-bench",
        "CANDIDATE_BIN=$HEBRUS_DSPARK_CANDIDATE_ROOT/build/metal-arm64/bin/ds4-bench",
        f"BASELINE_SHA256={_q(artifacts['baseline']['sha256'])}",
        f"CANDIDATE_SHA256={_q(artifacts['candidate']['sha256'])}",
        f"SUPPORT_SHA256={_q(artifacts['candidate']['support_sha256'])}",
        "OUT=$HEBRUS_DSPARK_COHORT_ROOT",
        "[[ $OUT == /* && ! -e $OUT ]] || {",
        "  print -u2 -- 'cohort output must be a fresh absolute path'",
        "  exit 2",
        "}",
        "mkdir -p -- $OUT/model-hash",
        "cp -- $MANIFEST $OUT/manifest.json",
        "shasum -a 256 $MANIFEST $RUNNER >$OUT/plan.sha256",
        "print -r -- $'timestamp\\tarm\\tvariant\\tphase\\tevent' >$OUT/order.tsv",
        "[[ $(sysctl -n hw.memsize) == 68719476736 ]]",
        "[[ $(sysctl -n machdep.cpu.brand_string) == *'M5 Pro'* ]]",
        "",
        "[[ $(git -C $HEBRUS_DSPARK_BASELINE_ROOT rev-parse HEAD) == $HEBRUS_DSPARK_BASELINE_HEAD ]]",
        "[[ $(git -C $HEBRUS_DSPARK_BASELINE_ROOT rev-parse main) == $HEBRUS_DSPARK_BASELINE_HEAD ]]",
        "[[ $(git -C $HEBRUS_DSPARK_CANDIDATE_ROOT rev-parse HEAD) == $HEBRUS_DSPARK_CANDIDATE_HEAD ]]",
        "[[ $(git -C $HEBRUS_DSPARK_CANDIDATE_ROOT rev-parse main) == $HEBRUS_DSPARK_CANDIDATE_HEAD ]]",
        "[[ $HEBRUS_DSPARK_CANDIDATE_HEAD == $HEBRUS_DSPARK_BASELINE_HEAD ]]",
        "[[ -z $(git -C $HEBRUS_DSPARK_BASELINE_ROOT status --porcelain) ]]",
        "[[ -z $(git -C $HEBRUS_DSPARK_CANDIDATE_ROOT status --porcelain) ]]",
        "check_transactional_bench() {",
        "  local source=$1/ds4_bench.c",
        "  [[ -f $source ]]",
        "  grep -q 'ds4_session_generation_block_begin' $source",
        "  grep -q 'ds4_session_generation_block_commit' $source",
        "}",
        "# The benchmark consumer is transactional, but product DSpark N>1 is",
        "# still admission-closed. Keep artifact hashing and inference unreachable.",
        "check_transactional_bench $HEBRUS_DSPARK_BASELINE_ROOT",
        "check_transactional_bench $HEBRUS_DSPARK_CANDIDATE_ROOT",
        "print -u2 -- 'cohort blocked: production DSpark N>1 is not connected'",
        "exit 2",
        "check_clean_build() {",
        "  local root=$1 bin=$2 expected info",
        "  [[ -x $bin && $bin == $root/build/metal-arm64/bin/ds4-bench ]]",
        "  expected=$(git -C $root rev-parse --short=12 HEAD)",
        "  info=$($bin --build-info)",
        "  [[ $(print -r -- $info | sed -n 's/^git:     //p') == $expected ]]",
        "  [[ $(print -r -- $info | sed -n 's/^backend: //p') == metal ]]",
        "  [[ $(print -r -- $info | sed -n 's/^arch:    //p') == arm64 ]]",
        "}",
        "check_clean_build $HEBRUS_DSPARK_BASELINE_ROOT $BASELINE_BIN",
        "check_clean_build $HEBRUS_DSPARK_CANDIDATE_ROOT $CANDIDATE_BIN",
        "print -r -- $'baseline_head\\tbaseline_main_head\\tcandidate_head' >$OUT/identity.tsv",
        "print -r -- \"$HEBRUS_DSPARK_BASELINE_HEAD\\t$(git -C $HEBRUS_DSPARK_BASELINE_ROOT rev-parse main)\\t$HEBRUS_DSPARK_CANDIDATE_HEAD\" >>$OUT/identity.tsv",
        "",
        "# Authenticate external SUPPORT exactly once before either oracle.",
        "DS4_M5_MODEL=$SUPPORT DS4_M5_MODEL_SHA256=$SUPPORT_SHA256 \\",
        "DS4_M5_MODEL_HASH_EVIDENCE=$OUT/model-hash/support.txt \\",
        "$RUNNER --prepare-model-hash-evidence",
        "support_before=$(stat -f '%d:%i:%z:%m' $SUPPORT)",
        "IFS=: read -r support_device support_inode support_bytes support_mtime <<<$support_before",
        "print -r -- $'timestamp\tphase\tdevice\tinode\tbytes\tmtime_epoch' >$OUT/oracle-support-identity.tsv",
        "print -r -- \"$(date -Iseconds)\tbefore\t$support_device\t$support_inode\t$support_bytes\t$support_mtime\" >>$OUT/oracle-support-identity.tsv",
        "print -r -- $'timestamp\tlane\ttests\tskipped\tlog_sha256\tresult' >$OUT/oracle-results.tsv",
        "",
        "# Model-free NumPy contract and pinned Apple-Metal MLX parity run next.",
        "if ! DS4_DSPARK_SUPPORT_GGUF=$SUPPORT python3 $HEBRUS_DSPARK_CANDIDATE_ROOT/tests/dspark/test_oracle.py -v >$OUT/oracle-system.log 2>&1; then",
        "  cat $OUT/oracle-system.log >&2; exit 2",
        "fi",
        f"python3 {_q(Path(__file__).resolve())} --manifest $MANIFEST --validate-oracle-log system $OUT/oracle-system.log >>$OUT/oracle-results.tsv",
        "if ! DS4_DSPARK_SUPPORT_GGUF=$SUPPORT $HEBRUS_DSPARK_MLX_PYTHON $HEBRUS_DSPARK_CANDIDATE_ROOT/tests/dspark/test_oracle.py -v >$OUT/oracle-mlx.log 2>&1; then",
        "  cat $OUT/oracle-mlx.log >&2; exit 2",
        "fi",
        f"python3 {_q(Path(__file__).resolve())} --manifest $MANIFEST --validate-oracle-log mlx $OUT/oracle-mlx.log >>$OUT/oracle-results.tsv",
        "support_after=$(stat -f '%d:%i:%z:%m' $SUPPORT)",
        "[[ $support_after == $support_before ]]",
        "IFS=: read -r support_device support_inode support_bytes support_mtime <<<$support_after",
        "print -r -- \"$(date -Iseconds)\tafter\t$support_device\t$support_inode\t$support_bytes\t$support_mtime\" >>$OUT/oracle-support-identity.tsv",
        "",
        "# Hash each benchmark artifact exactly once before page-cache preparation.",
        "DS4_M5_MODEL=$BASELINE_MODEL DS4_M5_MODEL_SHA256=$BASELINE_SHA256 \\",
        "DS4_M5_MODEL_HASH_EVIDENCE=$OUT/model-hash/baseline.txt \\",
        "$RUNNER --prepare-model-hash-evidence",
        "DS4_M5_MODEL=$CANDIDATE_MODEL DS4_M5_MODEL_SHA256=$CANDIDATE_SHA256 \\",
        "DS4_M5_MODEL_HASH_EVIDENCE=$OUT/model-hash/candidate.txt \\",
        "$RUNNER --prepare-model-hash-evidence",
        "",
        "run_arm() {",
        "  local arm=$1 variant=$2 phase=$3 root bin model digest evidence",
        "  if [[ $variant == baseline ]]; then",
        "    root=$HEBRUS_DSPARK_BASELINE_ROOT; bin=$BASELINE_BIN",
        "    model=$BASELINE_MODEL; digest=$BASELINE_SHA256",
        "    evidence=$OUT/model-hash/baseline.txt",
        "  else",
        "    root=$HEBRUS_DSPARK_CANDIDATE_ROOT; bin=$CANDIDATE_BIN",
        "    model=$CANDIDATE_MODEL; digest=$CANDIDATE_SHA256",
        "    evidence=$OUT/model-hash/candidate.txt",
        "  fi",
        "  local cache_state=warm exploratory=0 prefix=$OUT/$arm",
        "  if [[ $phase == warmup ]]; then",
        "    cache_state=exploratory; exploratory=1; prefix=$OUT/warmup-$arm",
        "  fi",
        "  print -r -- \"$(date -Iseconds)\\t$arm\\t$variant\\t$phase\\tstart\" >>$OUT/order.tsv",
        "  env \\",
        "    DS4_M5_ROOT=$root DS4_M5_BIN=$bin DS4_M5_MODEL=$model \\",
        "    DS4_M5_MODEL_SHA256=$digest DS4_M5_MODEL_HASH_EVIDENCE=$evidence \\",
        "    DS4_M5_PROMPT=$PROMPT DS4_M5_PREFIX=$prefix \\",
        f"    DS4_M5_RESIDENCY={_q(environment['DS4_M5_RESIDENCY'])} \\",
        "    DS4_M5_PRELOAD_POLICY=explicit \\",
        f"    DS4_M5_CACHE_STATE=$cache_state DS4_M5_EXPLORATORY=$exploratory \\",
        f"    DS4_M5_CTX_START={_q(environment['DS4_M5_CTX_START'])} \\",
        f"    DS4_M5_CTX_MAX={_q(environment['DS4_M5_CTX_MAX'])} \\",
        f"    DS4_M5_CTX_ALLOC={_q(environment['DS4_M5_CTX_ALLOC'])} \\",
        f"    DS4_M5_MAX_SECONDS={_q(environment['DS4_M5_MAX_SECONDS'])} \\",
        f"    DS4_M5_MIN_FREE_PERCENT={_q(environment['DS4_M5_MIN_FREE_PERCENT'])} \\",
        f"    DS4_M5_MAX_SWAPOUT_PAGES={_q(environment['DS4_M5_MAX_SWAPOUT_PAGES'])} \\",
        f"    DS4_M5_MAX_WIRED_GIB={_q(environment['DS4_M5_MAX_WIRED_GIB'])} \\",
        f"    $RUNNER dspark-8k-$arm-$phase auto {workload['decode_tokens']}",
        "  print -r -- \"$(date -Iseconds)\\t$arm\\t$variant\\t$phase\\tcomplete\" >>$OUT/order.tsv",
        "}",
        "",
    ]
    for index, arm in enumerate(runner["order"]):
        arm_id = arm["id"]
        variant = arm["variant"]
        lines.extend(
            [
                f"run_arm {_q(arm_id)} {_q(variant)} warmup",
                f"run_arm {_q(arm_id)} {_q(variant)} retained",
            ]
        )
        if index + 1 < len(runner["order"]):
            lines.append(f"sleep {cooldown}")
    lines.extend(
        [
            "",
            "# Required post-run review: compare semantic frontier-logit and",
            "# decode-evidence hashes exactly; then fail closed on every manifest gate.",
            f"python3 {_q(Path(__file__).resolve())} --validate-results $OUT",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--validate-results", type=Path, metavar="ABSOLUTE_ROOT")
    mode.add_argument(
        "--validate-oracle-log",
        nargs=2,
        metavar=("LANE", "LOG"),
        help="validate one retained unittest log and print its TSV evidence row",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="also stat the three retained local artifacts; never hash payloads",
    )
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        validate_manifest(manifest, local=args.local)
    except ManifestError as exc:
        print(f"dspark benchmark manifest: FAIL: {exc}", file=sys.stderr)
        return 1
    if args.check:
        scope = "manifest+local-artifact-size" if args.local else "manifest"
        print(
            f"dspark benchmark manifest: OK ({scope}; no model run; "
            "execution blocked until production DSpark N>1 is connected)"
        )
    elif args.dry_run:
        print(render_dry_run(args.manifest, manifest), end="")
    elif args.validate_oracle_log:
        lane, raw_path = args.validate_oracle_log
        try:
            row = _validated_oracle_row(manifest, lane, Path(raw_path))
        except ManifestError as exc:
            print(f"dspark benchmark oracle: FAIL: {exc}", file=sys.stderr)
            return 1
        print("\t".join(
            row[key]
            for key in (
                "timestamp", "lane", "tests", "skipped", "log_sha256", "result"
            )
        ))
    else:
        try:
            rows, errors = validate_results(args.validate_results, manifest)
        except ManifestError as exc:
            print(f"dspark benchmark cohort: FAIL: {exc}", file=sys.stderr)
            return 1
        print_results(rows, errors, manifest)
        if errors:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
