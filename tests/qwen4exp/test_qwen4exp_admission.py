#!/usr/bin/env python3
"""Black-box admission tests for the sparse Qwen4Exp GGUF fixture.

The input executable must be the dedicated ``DS4_NO_GPU`` plus
``DS4_TEST_HOOKS`` binary.  Every case rewrites the same sparse path in place,
then invokes the public ``--inspect -m`` boundary.  No checkpoint, model
payload, GPU object, or network access is used.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from qwen4exp_gguf_fixture import (
    BOUNDED_FUZZ_CASES_PER_REGION,
    BOUNDED_FUZZ_REGIONS,
    BOUNDED_FUZZ_SEED,
    PHYSICAL_PROFILE,
    PLE_PROFILE,
    SOURCE_BYTES,
    SOURCE_INVENTORY_SHA256,
    SOURCE_TENSORS,
    build_bounded_fuzz_fixture,
    build_fixture,
    self_check,
)


REPORT_KEYS = {
    "schemaVersion", "family", "profileId", "physicalProfileId", "stage",
    "admitted", "runtimeSupported", "payloadVerified", "textOnly",
    "mtpPresent", "visionPresent", "physicalTensors", "denseTensors",
    "expertStores", "pleStores", "fileBytes", "headerBytes", "denseBytes",
    "expertBytes", "pleBytes", "paddingBytes", "ownedBytes",
    "densePageBytes", "expertManifestVerified", "pleManifestVerified",
    "sourceTensorCount", "sourceBytes", "sourceInventorySha256",
    "tokenizerContentVerified", "rejection",
}


@dataclass(frozen=True)
class RejectCase:
    mutation: str
    stage: str
    field_fragment: str | None = None


REJECT_CASES = (
    # Identity and closed metadata typing/value admission.
    RejectCase("architecture", "identity", "general.architecture"),
    RejectCase("source_architecture", "identity", "source_architecture"),
    RejectCase("profile", "identity", "profile_id"),
    RejectCase("revision", "identity", "source_revision"),
    RejectCase("duplicate_kv", "identity", "num_hidden_layers"),
    RejectCase("physical_profile", "physical_profile", "physical_profile_id"),
    RejectCase("missing_metadata", "metadata", "metadata entry count"),
    RejectCase("extra_metadata", "metadata", "metadata entry count"),
    RejectCase("forbidden_metadata", "metadata", "bos_token_id"),
    RejectCase("context", "metadata", "max_position_embeddings"),
    RejectCase("layer_pattern", "metadata", "layer_pattern"),
    RejectCase("top_k", "metadata", "num_experts_per_tok"),
    RejectCase("expert_count", "metadata", "num_experts"),
    RejectCase("gr_rank", "metadata", "hc_lowrank"),
    RejectCase("qsa_query_heads", "metadata", "num_attention_heads"),
    RejectCase("qsa_kv_heads", "metadata", "num_key_value_heads"),
    RejectCase("qsa_head_dim", "metadata", "head_dim"),
    RejectCase("qsa_rotary_dim", "metadata", "partial_rotary_factor"),
    RejectCase("qsa_index_query_heads", "metadata", "indexer_n_heads"),
    RejectCase("qsa_index_kv_heads", "metadata", "indexer_kv_heads"),
    RejectCase("qsa_index_head_dim", "metadata", "indexer_head_dim"),
    RejectCase("gdn_key_heads", "metadata", "linear_num_key_heads"),
    RejectCase("gdn_value_heads", "metadata", "linear_num_value_heads"),
    RejectCase("gdn_key_dim", "metadata", "linear_key_head_dim"),
    RejectCase("gdn_value_dim", "metadata", "linear_value_head_dim"),
    RejectCase("ple_insertion_layer", "metadata", "ple_layer_ids"),
    RejectCase("ple_source_layer", "metadata", "ple_layer_ids"),
    RejectCase("ple_rows", "metadata", "ple.rows"),
    RejectCase("ple_multiplier_high32", "metadata", "layer_multipliers"),
    RejectCase("u64_wrong_type", "metadata", "layer_multipliers"),
    # Tokenizer/template identity is staged separately from model semantics.
    RejectCase("tokenizer_digest", "tokenizer", "tokenizer.digest"),
    RejectCase("template_digest", "tokenizer", "chat_template.digest"),
    RejectCase("special_id", "tokenizer", "image_pad_token_id"),
    # Exact physical inventory, including forbidden canonical/vision/MTP names.
    RejectCase("missing_tensor", "inventory", "physical tensor count"),
    RejectCase("extra_tensor", "inventory", "physical tensor count"),
    RejectCase("canonical_routed", "inventory", "tensor policy"),
    RejectCase("vision_tensor", "inventory", "tensor policy"),
    RejectCase("mtp_tensor", "inventory", "tensor policy"),
    RejectCase("second_expert_store", "inventory", "ds4.expert_major.v2"),
    RejectCase("second_ple_store", "inventory", "ds4.ple_rows.v1"),
    RejectCase("duplicate_tensor", "inventory", "lm_head.weight"),
    RejectCase("tensor_rank", "inventory", "lm_head.weight"),
    RejectCase("tensor_dimension", "inventory", "lm_head.weight.dim[0]"),
    RejectCase("tensor_type", "inventory", "lm_head.weight"),
    RejectCase("ple_runtime_layer", "inventory", "layers.1.ple.conv1d.weight"),
    RejectCase("unaligned_tensor", "inventory", "lm_head.weight"),
    # Embedded manifest and page-alignment admission.
    RejectCase("expert_family", "expert_store", "ds4.expert_major.v2"),
    RejectCase("expert_header_version", "expert_store", "ds4.expert_major.v2"),
    RejectCase("expert_component_offset", "expert_store", "ds4.expert_major.v2"),
    RejectCase("expert_component_length", "expert_store", "ds4.expert_major.v2"),
    RejectCase("expert_manifest_digest", "expert_store", "ds4.expert_major.v2"),
    RejectCase("truncated_expert_store", "expert_store", "ds4.expert_major.v2"),
    RejectCase("page_isolation", "ple_store", "ds4.ple_rows.v1"),
    RejectCase("unaligned_store", "ple_store", "ds4.ple_rows.v1"),
    RejectCase("ple_header_version", "ple_store", "ds4.ple_rows.v1"),
    RejectCase("ple_row_width", "ple_store", "ds4.ple_rows.v1"),
    RejectCase("ple_head_prime", "ple_store", "ds4.ple_rows.v1"),
    RejectCase("ple_head_offset", "ple_store", "ds4.ple_rows.v1"),
    RejectCase("ple_hash_version", "ple_store", "ds4.ple_rows.v1"),
    RejectCase("ple_manifest_rows", "ple_store", "ds4.ple_rows.v1"),
    RejectCase("ple_page_stride", "ple_store", "ds4.ple_rows.v1"),
    RejectCase("ple_manifest_digest", "ple_store", "ds4.ple_rows.v1"),
    RejectCase("truncated_ple_store", "ple_store", "ds4.ple_rows.v1"),
    # Policy and whole-file/page-rounded ownership are last.
    RejectCase("mtp_policy", "policy", "mtp.present"),
    RejectCase("exact_overlap", "ownership", "raw tensor spans"),
    RejectCase("store_overlap", "ownership", "raw tensor spans"),
)


FATAL_CASES = {
    "dimension_overflow": "tensor element count overflow",
    "offset_overflow": "tensor offset overflow",
    "gguf_bad_magic": "model is not a GGUF file",
    "gguf_bad_version": "only GGUF v3 is supported",
    "truncated_file": "tensor points outside GGUF file",
}


FORBIDDEN_RUNTIME_MARKERS = (
    "backend initialized", "metal graph", "gpu model", "gpu support",
)
SANITIZER_MARKERS = (
    "AddressSanitizer", "UndefinedBehaviorSanitizer", "runtime error:",
    "SUMMARY:",
)
_SUITE_LOCK_FILE: Path | None = None


def run(binary: Path, *arguments: str,
        timeout: float = 30) -> subprocess.CompletedProcess[str]:
    assert _SUITE_LOCK_FILE is not None, "admission suite lock is not configured"
    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    environment["DS4_LOCK_FILE"] = str(_SUITE_LOCK_FILE)
    result = subprocess.run(
        [str(binary), *arguments],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=environment,
    )
    combined = result.stdout + "\n" + result.stderr
    found = [marker for marker in SANITIZER_MARKERS if marker in combined]
    assert not found, (binary, arguments, found, combined)
    return result


def json_reports(result: subprocess.CompletedProcess[str]) -> list[dict]:
    reports: list[dict] = []
    for stream in (result.stdout, result.stderr):
        for line in stream.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and value.get("schemaVersion") == 1:
                reports.append(value)
    return reports


def assert_cpu_only_binary(binary: Path) -> None:
    capabilities = run(binary, "--capabilities=json")
    assert capabilities.returncode == 0, capabilities.stderr
    document = json.loads(capabilities.stdout)
    assert document["backend"] == "cpu", document

    build_info = run(binary, "--build-info")
    assert build_info.returncode == 0, build_info.stderr
    assert "backend: cpu" in build_info.stdout.lower(), build_info.stdout

    undefined = subprocess.run(
        ["nm", "-u", str(binary)], check=False, capture_output=True,
        text=True, timeout=30,
    )
    assert undefined.returncode == 0, undefined.stderr
    gpu_symbols = [
        line for line in undefined.stdout.splitlines()
        if re.search(r"(?:^|\s)_?ds4_gpu_", line)
    ]
    assert not gpu_symbols, (
        "the admission binary has unresolved GPU entry points", gpu_symbols,
    )


def assert_positive(binary: Path, fixture: Path) -> None:
    summary = self_check(fixture)
    result = run(binary, "--inspect", "-m", str(fixture))
    assert result.returncode == 0, (result.stdout, result.stderr)
    reports = json_reports(result)
    assert len(reports) == 1, (result.stdout, result.stderr)
    report = reports[0]
    assert set(report) == REPORT_KEYS, set(report) ^ REPORT_KEYS
    assert report == {
        "schemaVersion": 1,
        "family": "qwen4exp",
        "profileId": PLE_PROFILE,
        "physicalProfileId": PHYSICAL_PROFILE,
        "stage": "accepted",
        "admitted": True,
        "runtimeSupported": False,
        "payloadVerified": False,
        "textOnly": True,
        "mtpPresent": False,
        "visionPresent": False,
        "physicalTensors": 1069,
        "denseTensors": 1067,
        "expertStores": 1,
        "pleStores": 1,
        "fileBytes": summary.file_bytes,
        "headerBytes": summary.header_bytes,
        "denseBytes": summary.dense_bytes,
        "expertBytes": summary.expert_bytes,
        "pleBytes": summary.ple_bytes,
        "paddingBytes": summary.padding_bytes,
        "ownedBytes": (
            summary.dense_bytes + summary.expert_bytes + summary.ple_bytes
        ),
        "densePageBytes": summary.dense_page_bytes,
        "expertManifestVerified": True,
        "pleManifestVerified": True,
        "sourceTensorCount": SOURCE_TENSORS,
        "sourceBytes": SOURCE_BYTES,
        "sourceInventorySha256": SOURCE_INVENTORY_SHA256,
        "tokenizerContentVerified": False,
        "rejection": None,
    }


def assert_rejection(binary: Path, fixture: Path, case: RejectCase) -> None:
    build_fixture(fixture, case.mutation)
    result = run(binary, "--inspect", "-m", str(fixture))
    assert result.returncode != 0, f"mutation admitted: {case.mutation}"
    combined = result.stdout + "\n" + result.stderr
    assert str(fixture) in combined, (case.mutation, combined)
    reports = json_reports(result)
    assert len(reports) == 1, (
        case.mutation, result.returncode, result.stdout, result.stderr,
    )
    report = reports[0]
    assert set(report) == REPORT_KEYS, (case.mutation, set(report) ^ REPORT_KEYS)
    assert report["admitted"] is False, (case.mutation, report)
    assert report["stage"] == case.stage, (case.mutation, report)
    assert report["runtimeSupported"] is False, (case.mutation, report)
    assert report["payloadVerified"] is False, (case.mutation, report)
    rejection = report["rejection"]
    assert isinstance(rejection, dict), (case.mutation, report)
    assert set(rejection) == {"field", "expected", "observed"}, rejection
    assert all(isinstance(rejection[key], str) and rejection[key]
               for key in rejection), (case.mutation, rejection)
    if case.field_fragment is not None:
        assert case.field_fragment in rejection["field"], (
            case.mutation, rejection,
        )


def assert_fatal_directory_rejections(binary: Path, fixture: Path) -> None:
    for mutation, diagnostic in FATAL_CASES.items():
        build_fixture(fixture, mutation)
        result = run(binary, "--inspect", "-m", str(fixture))
        assert result.returncode != 0, f"mutation admitted: {mutation}"
        assert not json_reports(result), (mutation, result.stdout, result.stderr)
        assert diagnostic in result.stderr, (mutation, result.stderr)
        assert str(fixture) in result.stderr, (mutation, result.stderr)


def assert_execution_rejected_before_gpu(binary: Path, fixture: Path) -> None:
    build_fixture(fixture)
    result = run(binary, "-m", str(fixture), "-p", "structural fixture")
    assert result.returncode != 0, (result.stdout, result.stderr)
    assert "structural admission-only" in result.stderr, result.stderr
    assert str(fixture) in result.stderr, result.stderr
    reports = json_reports(result)
    assert len(reports) == 1 and reports[0]["admitted"] is True, (
        result.stdout, result.stderr,
    )
    combined = (result.stdout + "\n" + result.stderr).lower()
    assert not any(marker in combined for marker in FORBIDDEN_RUNTIME_MARKERS), \
        combined


def assert_bounded_fuzz_rejections(binary: Path, fixture: Path) -> int:
    count = 0
    for region in BOUNDED_FUZZ_REGIONS:
        for case_index in range(BOUNDED_FUZZ_CASES_PER_REGION):
            label = f"fuzz:{region}:{case_index}"
            build_bounded_fuzz_fixture(
                fixture, region, case_index, BOUNDED_FUZZ_SEED,
            )
            result = run(
                binary, "--inspect", "-m", str(fixture), timeout=5,
            )
            assert result.returncode > 0, (
                label, result.returncode, result.stdout, result.stderr,
            )
            combined = result.stdout + "\n" + result.stderr
            assert str(fixture) in combined, (label, combined)
            lowered = combined.lower()
            assert not any(
                marker in lowered for marker in FORBIDDEN_RUNTIME_MARKERS
            ), (label, combined)
            reports = json_reports(result)
            assert len(reports) <= 1, (label, result.stdout, result.stderr)
            if region in ("expert", "ple"):
                assert len(reports) == 1, (
                    label, result.stdout, result.stderr,
                )
                expected_stage = (
                    "expert_store" if region == "expert" else "ple_store"
                )
                assert reports[0]["stage"] == expected_stage, (
                    label, reports[0],
                )
            if reports:
                assert reports[0]["admitted"] is False, (label, reports[0])
                assert reports[0]["runtimeSupported"] is False, (
                    label, reports[0],
                )
                assert reports[0]["payloadVerified"] is False, (
                    label, reports[0],
                )
            count += 1
    return count


def assert_production_rejected(binary: Path, fixture: Path) -> None:
    """A normal binary must neither contain nor admit the test-only profile."""

    assert binary.is_file(), binary
    assert PHYSICAL_PROFILE.encode("ascii") not in binary.read_bytes(), (
        binary, "test-only physical profile leaked into production binary",
    )
    capabilities = run(binary, "--capabilities=json")
    assert capabilities.returncode == 0, (binary, capabilities.stderr)
    backend = json.loads(capabilities.stdout)["backend"]
    assert backend in ("cpu", "metal"), (binary, backend)
    build_fixture(fixture)
    backend_arguments = () if backend == "metal" else ("--cpu",)
    result = run(
        binary, *backend_arguments, "--inspect", "-m", str(fixture),
    )
    assert result.returncode != 0, (binary, result.stdout, result.stderr)
    combined = result.stdout + "\n" + result.stderr
    assert str(fixture) in combined, (binary, combined)
    reports = json_reports(result)
    assert len(reports) == 1, (binary, result.stdout, result.stderr)
    report = reports[0]
    assert set(report) == REPORT_KEYS, (binary, set(report) ^ REPORT_KEYS)
    assert report["admitted"] is False, (binary, report)
    assert report["stage"] == "physical_profile", (binary, report)
    assert report["runtimeSupported"] is False, (binary, report)
    assert report["payloadVerified"] is False, (binary, report)
    rejection = report["rejection"]
    assert isinstance(rejection, dict), (binary, report)
    assert "physical_profile_id" in rejection["field"], (binary, rejection)
    lowered = combined.lower()
    assert not any(marker in lowered for marker in FORBIDDEN_RUNTIME_MARKERS), (
        binary, combined,
    )


def main(argv: list[str] | None = None) -> int:
    global _SUITE_LOCK_FILE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", nargs="?", type=Path)
    parser.add_argument("--binary", dest="binary_option", type=Path)
    parser.add_argument(
        "--production-binary", action="append", default=[], type=Path,
        help="normal binary to prove rejects and does not embed the test profile",
    )
    args = parser.parse_args(argv)
    binary = args.binary_option or args.binary
    if binary is None:
        parser.error("the dedicated admission-test binary is required")
    binary = binary.resolve()
    assert binary.is_file(), binary
    production_binaries = [path.resolve() for path in args.production_binary]

    with tempfile.TemporaryDirectory(prefix="qwen4exp-admission-") as tmp:
        _SUITE_LOCK_FILE = Path(tmp) / "hebrus-admission.lock"
        try:
            fixture = Path(tmp) / "qwen4exp-structural.gguf"
            assert_cpu_only_binary(binary)
            assert_positive(binary, fixture)
            for case in REJECT_CASES:
                assert_rejection(binary, fixture, case)
            assert_fatal_directory_rejections(binary, fixture)
            assert_execution_rejected_before_gpu(binary, fixture)
            fuzz_count = assert_bounded_fuzz_rejections(binary, fixture)
            for production_binary in production_binaries:
                assert_production_rejected(production_binary, fixture)
        finally:
            _SUITE_LOCK_FILE = None

    total = (1 + len(REJECT_CASES) + len(FATAL_CASES) + 1 +
             fuzz_count + len(production_binaries))
    print(
        "qwen4exp admission: PASS "
        f"({total} cases, CPU-only test binary, "
        f"{len(production_binaries)} production binaries, "
        f"fuzz seed 0x{BOUNDED_FUZZ_SEED:016x})"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError, subprocess.SubprocessError,
            json.JSONDecodeError) as exc:
        print(f"qwen4exp admission: FAIL: {exc!r}", file=sys.stderr)
        raise SystemExit(1)
