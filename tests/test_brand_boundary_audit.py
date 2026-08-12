#!/usr/bin/env python3
"""Regression tests for the monotonic legacy-brand inventory."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
AUDIT = REPOSITORY_ROOT / "tools" / "brand_boundary_audit.py"
CATEGORIES = (
    "serialized/permanent",
    "compatibility",
    "historical-attribution",
    "migration-pending",
)
LOCATION_ORDER = {"path": 0, "content": 1}
TOKEN_ORDER = {"ds4": 0, "DS4": 1, "DwarfStar": 2}
IDENTITY_CONTRACT = {
    "brand": "Hebrus",
    "canonical_commands": [
        "hebrus",
        "hebrus-server",
        "hebrus-agent",
        "hebrus-bench",
        "hebrus-eval",
    ],
    "command_aliases": {
        "ds4": "hebrus",
        "ds4-server": "hebrus-server",
        "ds4-agent": "hebrus-agent",
        "ds4-bench": "hebrus-bench",
        "ds4-eval": "hebrus-eval",
    },
    "accepted_engine_ids": ["hebrus", "ds4"],
    "environment_prefixes": {
        "preserved": ["DS4_"],
        "deferred": ["HEBRUS_"],
    },
    "permanent_literals": ["ds4.expert_major.v1", "ds4.expert_major.v2"],
    "repositories": {
        "historical_origin": "antirez/ds4",
        "pre_rename": "andreaborio/ds4",
    },
}


def manifest_document(entries: list[dict[str, object]]) -> dict[str, object]:
    ordered = sorted(
        entries,
        key=lambda entry: (
            entry["path"],
            LOCATION_ORDER[entry["location"]],
            TOKEN_ORDER[entry["token"]],
        ),
    )
    return {
        "schema_version": 1,
        "identity_contract": IDENTITY_CONTRACT,
        "scope": {
            "tokens": ["ds4", "DS4", "DwarfStar"],
            "locations": ["path", "content"],
            "excluded_paths": ["tools/brand_boundary.json"],
        },
        "category_definitions": {
            category: f"Test definition for {category}." for category in CATEGORIES
        },
        "refresh_policy": {
            "reductions": "python3 tools/brand_boundary_audit.py --refresh",
            "existing_increase": (
                "python3 tools/brand_boundary_audit.py --refresh "
                "--accept-increase PATH:LOCATION:TOKEN"
            ),
            "new_group": (
                "python3 tools/brand_boundary_audit.py --refresh "
                "--classify PATH:LOCATION:TOKEN=CLASSIFICATION"
            ),
        },
        "entries": ordered,
    }


def entry(
    path: str,
    location: str,
    token: str,
    maximum: int,
    classification: str = "migration-pending",
) -> dict[str, object]:
    return {
        "path": path,
        "location": location,
        "token": token,
        "classification": classification,
        "maximum": maximum,
    }


class Fixture:
    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="brand-boundary-")
        self.root = Path(self.temporary.name)
        self.manifest = self.root / "tools" / "brand_boundary.json"
        self.manifest.parent.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=self.root, check=True)

    def close(self) -> None:
        self.temporary.cleanup()

    def write(self, relative: str, content: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def track(self, *relative: str) -> None:
        subprocess.run(["git", "add", "--", *relative], cwd=self.root, check=True)

    def write_manifest(self, entries: list[dict[str, object]]) -> None:
        self.manifest.write_text(
            json.dumps(manifest_document(entries), indent=2) + "\n",
            encoding="utf-8",
        )

    def run(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(AUDIT),
                "--root",
                str(self.root),
                "--manifest",
                "tools/brand_boundary.json",
                *arguments,
            ],
            text=True,
            capture_output=True,
            check=False,
        )


class BrandBoundaryAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = Fixture()

    def tearDown(self) -> None:
        self.fixture.close()

    def seed_one_token(self) -> None:
        self.fixture.write("engine.c", "const char *engine = \"ds4\";\n")
        self.fixture.track("engine.c")
        self.fixture.write_manifest([entry("engine.c", "content", "ds4", 1)])

    def test_new_tracked_file_requires_exact_classification(self) -> None:
        self.seed_one_token()
        self.fixture.write("ds4_notes.txt", "brand bridge\n")
        self.fixture.track("ds4_notes.txt")

        checked = self.fixture.run("--check")
        self.assertEqual(checked.returncode, 1, checked.stderr)
        self.assertIn(
            "unclassified brand token group: ds4_notes.txt:path:ds4",
            checked.stderr,
        )

        refused = self.fixture.run("--refresh")
        self.assertEqual(refused.returncode, 1, refused.stderr)

        refreshed = self.fixture.run(
            "--refresh",
            "--classify",
            "ds4_notes.txt:path:ds4=migration-pending",
        )
        self.assertEqual(refreshed.returncode, 0, refreshed.stderr)
        self.assertEqual(self.fixture.run("--check").returncode, 0)

    def test_new_untracked_file_requires_exact_classification(self) -> None:
        self.seed_one_token()
        self.fixture.write("draft.c", "const char *legacy = \"DS4\";\n")

        checked = self.fixture.run("--check")
        self.assertEqual(checked.returncode, 1, checked.stderr)
        self.assertIn(
            "unclassified brand token group: draft.c:content:DS4",
            checked.stderr,
        )

    def test_worktree_deletion_removes_path_and_content_groups(self) -> None:
        self.seed_one_token()
        self.fixture.write("ds4_notes.txt", "DS4 bridge\n")
        self.fixture.track("ds4_notes.txt")
        self.fixture.write_manifest([
            entry("engine.c", "content", "ds4", 1),
            entry("ds4_notes.txt", "path", "ds4", 1),
            entry("ds4_notes.txt", "content", "DS4", 1),
        ])
        (self.fixture.root / "ds4_notes.txt").unlink()

        checked = self.fixture.run("--check")
        self.assertEqual(checked.returncode, 0, checked.stderr)
        self.assertIn("2 reductions", checked.stdout)

    def test_new_token_in_existing_file_is_unclassified(self) -> None:
        self.seed_one_token()
        self.fixture.write("engine.c", "const char *engine = \"ds4 DS4\";\n")

        checked = self.fixture.run("--check")
        self.assertEqual(checked.returncode, 1, checked.stderr)
        self.assertIn(
            "unclassified brand token group: engine.c:content:DS4",
            checked.stderr,
        )

    def test_increase_fails_until_exactly_authorized(self) -> None:
        self.seed_one_token()
        self.fixture.write("engine.c", "const char *engine = \"ds4 ds4\";\n")

        checked = self.fixture.run("--check")
        self.assertEqual(checked.returncode, 1, checked.stderr)
        self.assertIn("engine.c:content:ds4 2 > 1", checked.stderr)

        original = self.fixture.manifest.read_bytes()
        refused = self.fixture.run("--refresh")
        self.assertEqual(refused.returncode, 1, refused.stderr)
        self.assertEqual(self.fixture.manifest.read_bytes(), original)

        refreshed = self.fixture.run(
            "--refresh", "--accept-increase", "engine.c:content:ds4"
        )
        self.assertEqual(refreshed.returncode, 0, refreshed.stderr)
        self.assertEqual(self.fixture.run("--check").returncode, 0)
        document = json.loads(self.fixture.manifest.read_text(encoding="utf-8"))
        self.assertEqual(document["entries"][0]["maximum"], 2)

    def test_reduction_passes_and_refresh_tightens_deterministically(self) -> None:
        self.fixture.write("engine.c", "const char *engine = \"ds4\";\n")
        self.fixture.track("engine.c")
        self.fixture.write_manifest([entry("engine.c", "content", "ds4", 2)])

        checked = self.fixture.run("--check")
        self.assertEqual(checked.returncode, 0, checked.stderr)
        self.assertIn("1 reductions", checked.stdout)

        refreshed = self.fixture.run("--refresh")
        self.assertEqual(refreshed.returncode, 0, refreshed.stderr)
        first = self.fixture.manifest.read_bytes()
        self.assertEqual(self.fixture.run("--refresh").returncode, 0)
        self.assertEqual(self.fixture.manifest.read_bytes(), first)

        self.fixture.write("engine.c", "const char *engine = \"ds4 ds4\";\n")
        increased = self.fixture.run("--check")
        self.assertEqual(increased.returncode, 1, increased.stderr)
        self.assertIn("2 > 1", increased.stderr)

    def test_invalid_manifest_fails_closed(self) -> None:
        self.seed_one_token()
        document = json.loads(self.fixture.manifest.read_text(encoding="utf-8"))
        document["entries"][0]["path"] = "**/*.c"
        self.fixture.manifest.write_text(
            json.dumps(document, indent=2) + "\n", encoding="utf-8"
        )

        checked = self.fixture.run("--check")
        self.assertEqual(checked.returncode, 2, checked.stderr)
        self.assertIn("must be an explicit normalized path", checked.stderr)

    def test_identity_contract_is_exact(self) -> None:
        self.seed_one_token()
        document = json.loads(self.fixture.manifest.read_text(encoding="utf-8"))
        document["identity_contract"]["command_aliases"]["ds4-server"] = "other"
        self.fixture.manifest.write_text(
            json.dumps(document, indent=2) + "\n", encoding="utf-8"
        )

        checked = self.fixture.run("--check")
        self.assertEqual(checked.returncode, 2, checked.stderr)
        self.assertIn("identity_contract must exactly match", checked.stderr)

    def test_unused_refresh_authorization_fails_closed(self) -> None:
        self.seed_one_token()
        refreshed = self.fixture.run(
            "--refresh", "--accept-increase", "engine.c:content:ds4"
        )
        self.assertEqual(refreshed.returncode, 2, refreshed.stderr)
        self.assertIn("does not match a current increase", refreshed.stderr)


if __name__ == "__main__":
    unittest.main()
