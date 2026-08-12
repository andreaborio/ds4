#!/usr/bin/env python3
"""Regression tests for deterministic source-release bundles."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "release_source.py"
VERSION = "1.2.3-rc.1"
FIXED_DATE = "2024-01-02T03:04:05+00:00"


class ReleaseSourceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="hebrus-release-source-test-")
        self.base = Path(self.temporary.name)
        self.repository = self.base / "repository"
        self.repository.mkdir()
        self.git("init", "--quiet")
        self.git("config", "user.name", "Release Test")
        self.git("config", "user.email", "release-test@example.invalid")

        (self.repository / "Makefile").write_text(
            "all:\n\t@printf 'fixture build\\n'\n",
            encoding="utf-8",
            newline="\n",
        )
        executable = self.repository / "verify.sh"
        executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8", newline="\n")
        executable.chmod(0o755)
        (self.repository / "makefile-link").symlink_to("Makefile")
        (self.repository / "ignored.tmp").write_text("not archived\n", encoding="utf-8")
        (self.repository / ".gitignore").write_text("*.tmp\n", encoding="utf-8")
        self.git("add", "Makefile", "verify.sh", "makefile-link", ".gitignore")
        env = os.environ.copy()
        env.update({"GIT_AUTHOR_DATE": FIXED_DATE, "GIT_COMMITTER_DATE": FIXED_DATE})
        self.git("commit", "--quiet", "-m", "fixture", env=env)
        self.commit = self.git("rev-parse", "HEAD").stdout.strip()
        # The generator must not inherit archive modes from repository or user
        # configuration; choose a hostile value to exercise its fixed umask.
        self.git("config", "tar.umask", "0077")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def git(
        self,
        *args: str,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args],
            cwd=self.repository,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )

    def tool(
        self,
        *args: str,
        expect_success: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        completed = subprocess.run(
            [sys.executable, str(TOOL), *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if expect_success and completed.returncode != 0:
            self.fail(f"tool failed:\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
        if not expect_success and completed.returncode == 0:
            self.fail(f"tool unexpectedly passed:\n{completed.stdout}")
        return completed

    def build(self, output: Path) -> Path:
        self.tool(
            "build",
            "--repository",
            str(self.repository),
            "--version",
            VERSION,
            "--ref",
            self.commit,
            "--output-dir",
            str(output),
        )
        return output / f"hebrus-{VERSION}-source.json"

    def test_bundle_is_reproducible_and_self_verifying(self) -> None:
        first = self.base / "first"
        second = self.base / "second"
        first_manifest = self.build(first)
        second_manifest = self.build(second)

        names = [
            f"hebrus-{VERSION}.tar.gz",
            f"hebrus-{VERSION}-source.json",
            "SHA256SUMS",
        ]
        for name in names:
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes(), name)

        self.tool("verify", "--manifest", str(first_manifest))
        manifest = json.loads(first_manifest.read_text(encoding="utf-8"))
        self.assertEqual(manifest["git_commit"], self.commit)
        self.assertEqual(manifest["git_commit_short"], self.commit[:12])
        self.assertEqual(manifest["version"], VERSION)
        self.assertEqual(manifest["archive_format"], "git-archive-tar-gzip-v1")

        archive_path = first / f"hebrus-{VERSION}.tar.gz"
        with tarfile.open(archive_path, "r:gz") as archive:
            members = {member.name: member for member in archive}
        root = f"hebrus-{VERSION}"
        self.assertIn(root, members)
        self.assertIn(f"{root}/Makefile", members)
        self.assertIn(f"{root}/verify.sh", members)
        self.assertIn(f"{root}/makefile-link", members)
        self.assertNotIn(f"{root}/ignored.tmp", members)
        self.assertEqual(members[f"{root}/Makefile"].mode & 0o777, 0o644)
        self.assertEqual(members[f"{root}/verify.sh"].mode & 0o777, 0o755)
        self.assertTrue(members[f"{root}/makefile-link"].issym())
        self.assertEqual(members[f"{root}/makefile-link"].linkname, "Makefile")

    def test_dirty_tree_mutable_ref_and_invalid_version_fail_closed(self) -> None:
        invalid = self.tool(
            "build",
            "--repository",
            str(self.repository),
            "--version",
            "v1.2.3",
            "--ref",
            self.commit,
            "--output-dir",
            str(self.base / "invalid"),
            expect_success=False,
        )
        self.assertIn("SemVer", invalid.stderr)

        invalid_prerelease = self.tool(
            "build",
            "--repository",
            str(self.repository),
            "--version",
            "1.2.3-01",
            "--ref",
            self.commit,
            "--output-dir",
            str(self.base / "invalid-prerelease"),
            expect_success=False,
        )
        self.assertIn("SemVer", invalid_prerelease.stderr)

        mutable = self.tool(
            "build",
            "--repository",
            str(self.repository),
            "--version",
            "1.2.3",
            "--ref",
            "HEAD",
            "--output-dir",
            str(self.base / "mutable"),
            expect_success=False,
        )
        self.assertIn("full 40-character commit or an exact local tag", mutable.stderr)

        (self.repository / "Makefile").write_text("dirty\n", encoding="utf-8")
        dirty = self.tool(
            "build",
            "--repository",
            str(self.repository),
            "--version",
            "1.2.3",
            "--ref",
            self.commit,
            "--output-dir",
            str(self.base / "dirty"),
            expect_success=False,
        )
        self.assertIn("clean working tree", dirty.stderr)

    def test_exact_tag_is_accepted_and_non_head_commit_is_rejected(self) -> None:
        self.git("tag", "v1.2.3-test", self.commit)
        tagged = self.base / "tagged"
        self.tool(
            "build",
            "--repository",
            str(self.repository),
            "--version",
            "1.2.3",
            "--ref",
            "v1.2.3-test",
            "--output-dir",
            str(tagged),
        )
        manifest = json.loads(
            (tagged / "hebrus-1.2.3-source.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["git_commit"], self.commit)

        (self.repository / "second.txt").write_text("second\n", encoding="utf-8")
        self.git("add", "second.txt")
        env = os.environ.copy()
        env.update({"GIT_AUTHOR_DATE": FIXED_DATE, "GIT_COMMITTER_DATE": FIXED_DATE})
        self.git("commit", "--quiet", "-m", "second", env=env)
        non_head = self.tool(
            "build",
            "--repository",
            str(self.repository),
            "--version",
            "1.2.3",
            "--ref",
            self.commit,
            "--output-dir",
            str(self.base / "non-head"),
            expect_success=False,
        )
        self.assertIn("is not checked out at HEAD", non_head.stderr)

    def test_existing_output_and_tampering_are_rejected(self) -> None:
        output = self.base / "output"
        manifest = self.build(output)

        overwrite = self.tool(
            "build",
            "--repository",
            str(self.repository),
            "--version",
            VERSION,
            "--ref",
            self.commit,
            "--output-dir",
            str(output),
            expect_success=False,
        )
        self.assertIn("refusing to overwrite", overwrite.stderr)

        archive = output / f"hebrus-{VERSION}.tar.gz"
        data = bytearray(archive.read_bytes())
        data[-1] ^= 0x01
        archive.write_bytes(data)
        tampered = self.tool(
            "verify",
            "--manifest",
            str(manifest),
            expect_success=False,
        )
        self.assertIn("SHA-256 does not match", tampered.stderr)

    def test_noncanonical_gzip_wrapper_is_rejected(self) -> None:
        output = self.base / "wrapper"
        manifest_path = self.build(output)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        archive = output / f"hebrus-{VERSION}.tar.gz"

        with gzip.open(archive, "rb") as stream:
            tar_data = stream.read()
        with archive.open("wb") as destination:
            with gzip.GzipFile(
                filename="hebrus-noncanonical.tar",
                mode="wb",
                fileobj=destination,
                mtime=123,
            ) as compressed:
                compressed.write(tar_data)

        archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
        manifest["archive"]["bytes"] = archive.stat().st_size
        manifest["archive"]["sha256"] = archive_hash
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        checksum = output / "SHA256SUMS"
        checksum.write_text(
            f"{archive_hash}  {archive.name}\n"
            f"{hashlib.sha256(manifest_path.read_bytes()).hexdigest()}  {manifest_path.name}\n",
            encoding="utf-8",
        )

        result = self.tool(
            "verify",
            "--manifest",
            str(manifest_path),
            expect_success=False,
        )
        self.assertIn("gzip wrapper is not canonical", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
