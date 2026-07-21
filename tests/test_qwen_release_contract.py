#!/usr/bin/env python3
"""Fail-closed fixtures for the machine-readable Qwen release contract."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CHECKER = REPOSITORY_ROOT / "tools" / "qwen_release_contract.py"
MANIFEST = "docs/contracts/qwen-release.json"
SURFACES = (
    MANIFEST,
    "README.md",
    "CONTRIBUTING.md",
    "QA_BEFORE_RELEASES.md",
    "docs/contracts/RUNTIME_SUPPORT.md",
    "docs/qwen-expert-major-store.md",
    "download_model.sh",
    "tests/test_download_model.sh",
)


class Fixture:
    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="qwen-release-contract-")
        self.root = Path(self.temporary.name)
        for relative in SURFACES:
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPOSITORY_ROOT / relative, destination)

    def close(self) -> None:
        self.temporary.cleanup()

    def path(self, relative: str) -> Path:
        return self.root / relative

    def contract(self) -> dict[str, object]:
        return json.loads(self.path(MANIFEST).read_text(encoding="utf-8"))

    def replace_once(self, relative: str, old: str, new: str) -> None:
        path = self.path(relative)
        text = path.read_text(encoding="utf-8")
        if text.count(old) != 1:
            raise AssertionError(
                f"fixture expected exactly one {old!r} in {relative}, "
                f"found {text.count(old)}"
            )
        path.write_text(text.replace(old, new, 1), encoding="utf-8")

    def run(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(CHECKER),
                "--root",
                str(self.root),
                "--manifest",
                MANIFEST,
            ],
            text=True,
            capture_output=True,
            check=False,
        )


class QwenReleaseContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = Fixture()

    def tearDown(self) -> None:
        self.fixture.close()

    def assert_fails_with(self, expected: str) -> None:
        result = self.fixture.run()
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn(expected, result.stderr)

    def test_current_contract_and_surfaces_pass(self) -> None:
        result = self.fixture.run()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Qwen release contract: PASS", result.stdout)

    def test_every_human_surface_is_bound_to_the_manifest(self) -> None:
        contract = self.fixture.contract()
        published = contract["publishedArtifact"]
        negative = contract["negativeArtifact"]
        assert isinstance(published, dict)
        assert isinstance(negative, dict)
        cases = (
            ("README.md", published["filename"], "Qwen3.6-DRIFT.gguf"),
            ("CONTRIBUTING.md", published["revision"], "a" * 40),
            (
                "docs/contracts/RUNTIME_SUPPORT.md",
                published["runtimeCommit"],
                "b" * 40,
            ),
            ("QA_BEFORE_RELEASES.md", negative["sha256"], "c" * 64),
            (
                "docs/qwen-expert-major-store.md",
                contract["repository"],
                "example.invalid/Qwen-release",
            ),
        )
        for relative, old, new in cases:
            with self.subTest(relative=relative):
                fixture = Fixture()
                try:
                    fixture.replace_once(relative, str(old), new)
                    result = fixture.run()
                    self.assertEqual(result.returncode, 1, result.stdout)
                    self.assertIn(relative, result.stderr)
                finally:
                    fixture.close()

    def test_downloader_assignment_drift_fails(self) -> None:
        contract = self.fixture.contract()
        published = contract["publishedArtifact"]
        assert isinstance(published, dict)
        old = f"RUNTIME_QWEN_BYTES={published['bytes']}"
        self.fixture.replace_once("download_model.sh", old, "RUNTIME_QWEN_BYTES=1")
        self.assert_fails_with("download_model.sh: RUNTIME_QWEN_BYTES")

    def test_negative_artifact_cannot_become_downloadable(self) -> None:
        contract = self.fixture.contract()
        negative = contract["negativeArtifact"]
        assert isinstance(negative, dict)
        path = self.fixture.path("download_model.sh")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n# accidental target: {negative['filename']}\n")
        self.assert_fails_with("negative-only Qwen artifact must not be downloadable")

    def test_download_test_must_consume_instead_of_duplicate_contract(self) -> None:
        contract = self.fixture.contract()
        published = contract["publishedArtifact"]
        assert isinstance(published, dict)
        path = self.fixture.path("tests/test_download_model.sh")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n# stale test fixture {published['sha256']}\n")
        self.assert_fails_with("must consume the canonical contract")

    def test_manifest_statuses_fail_closed(self) -> None:
        for mutation, expected in (
            (("publishedArtifact", "status", "candidate"), "must be published"),
            (("negativeArtifact", "status", "retired"), "must be negative-only"),
        ):
            with self.subTest(mutation=mutation):
                fixture = Fixture()
                try:
                    document = fixture.contract()
                    group, field, value = mutation
                    artifact = document[group]
                    assert isinstance(artifact, dict)
                    artifact[field] = value
                    fixture.path(MANIFEST).write_text(
                        json.dumps(document, indent=2) + "\n", encoding="utf-8"
                    )
                    result = fixture.run()
                    self.assertEqual(result.returncode, 1, result.stdout)
                    self.assertIn(expected, result.stderr)
                finally:
                    fixture.close()

    def test_manifest_schema_and_digest_fail_closed(self) -> None:
        cases = (
            ("unknown-key", "contract has invalid keys"),
            ("invalid-digest", "must be 64 lowercase hexadecimal characters"),
        )
        for mutation, expected in cases:
            with self.subTest(mutation=mutation):
                fixture = Fixture()
                try:
                    document = fixture.contract()
                    if mutation == "unknown-key":
                        document["notes"] = "not part of schema 1"
                    else:
                        published = document["publishedArtifact"]
                        assert isinstance(published, dict)
                        published["sha256"] = "A" * 64
                    fixture.path(MANIFEST).write_text(
                        json.dumps(document, indent=2) + "\n", encoding="utf-8"
                    )
                    result = fixture.run()
                    self.assertEqual(result.returncode, 1, result.stdout)
                    self.assertIn(expected, result.stderr)
                finally:
                    fixture.close()

    def test_release_table_is_structural_not_an_unscoped_token_search(self) -> None:
        self.fixture.replace_once(
            "docs/qwen-expert-major-store.md",
            "| Immutable revision |",
            "| Unrelated note |",
        )
        self.assert_fails_with("'Immutable revision' must be")


if __name__ == "__main__":
    unittest.main()
