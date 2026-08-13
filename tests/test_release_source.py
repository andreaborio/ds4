#!/usr/bin/env python3
"""Regression tests for deterministic source-release bundles."""

from __future__ import annotations

import gzip
import hashlib
import io
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

    def replace_archive(self, manifest_path: Path, archive_bytes: bytes) -> None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        archive = manifest_path.parent / manifest["archive"]["filename"]
        archive.write_bytes(archive_bytes)
        manifest["archive"]["bytes"] = archive.stat().st_size
        manifest["archive"]["sha256"] = hashlib.sha256(archive_bytes).hexdigest()
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (manifest_path.parent / "SHA256SUMS").write_text(
            f"{manifest['archive']['sha256']}  {archive.name}\n"
            f"{hashlib.sha256(manifest_path.read_bytes()).hexdigest()}  "
            f"{manifest_path.name}\n",
            encoding="utf-8",
        )

    def replace_with_tar_members(
        self,
        manifest_path: Path,
        members: list[tuple[str, str, bytes | str, dict[str, str] | None]],
    ) -> None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        tar_stream = io.BytesIO()
        with tarfile.open(
            fileobj=tar_stream,
            mode="w",
            format=tarfile.PAX_FORMAT,
            pax_headers={"comment": manifest["git_commit"]},
        ) as archive:
            for name, kind, value, pax_headers in members:
                member = tarfile.TarInfo(name)
                member.uid = 0
                member.gid = 0
                member.uname = "root"
                member.gname = "root"
                member.mtime = manifest["source_date_epoch"]
                member.pax_headers = pax_headers or {}
                if kind == "dir":
                    member.type = tarfile.DIRTYPE
                    member.mode = 0o755
                    archive.addfile(member)
                elif kind == "link":
                    member.type = tarfile.SYMTYPE
                    member.mode = 0o777
                    member.linkname = str(value)
                    archive.addfile(member)
                else:
                    data = bytes(value)
                    member.type = tarfile.REGTYPE
                    member.mode = 0o644
                    member.size = len(data)
                    archive.addfile(member, io.BytesIO(data))

        compressed = io.BytesIO()
        with gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=9,
            fileobj=compressed,
            mtime=0,
        ) as stream:
            stream.write(tar_stream.getvalue())
        manifest["archive"]["members"] = len(members)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.replace_archive(manifest_path, compressed.getvalue())

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

    def test_local_archive_attributes_cannot_change_commit_output(self) -> None:
        first = self.base / "attributes-first"
        second = self.base / "attributes-second"
        self.build(first)

        info_attributes = self.repository / ".git" / "info" / "attributes"
        info_attributes.write_text(
            "verify.sh export-ignore\nMakefile export-subst\n",
            encoding="utf-8",
        )
        self.assertEqual(self.git("status", "--porcelain").stdout, "")
        self.build(second)

        for name in (
            f"hebrus-{VERSION}.tar.gz",
            f"hebrus-{VERSION}-source.json",
            "SHA256SUMS",
        ):
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

        with tarfile.open(second / f"hebrus-{VERSION}.tar.gz", "r:gz") as archive:
            self.assertIn(f"hebrus-{VERSION}/verify.sh", archive.getnames())

    def test_replace_objects_cannot_substitute_the_release_commit(self) -> None:
        (self.repository / "replacement.txt").write_text("replacement\n", encoding="utf-8")
        self.git("add", "replacement.txt")
        env = os.environ.copy()
        env.update({"GIT_AUTHOR_DATE": FIXED_DATE, "GIT_COMMITTER_DATE": FIXED_DATE})
        self.git("commit", "--quiet", "-m", "replacement", env=env)
        replacement = self.git("rev-parse", "HEAD").stdout.strip()
        self.git("reset", "--hard", "--quiet", self.commit)
        self.git("replace", self.commit, replacement)

        output = self.base / "replaced"
        manifest = self.build(output)
        value = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(value["git_commit"], self.commit)
        with tarfile.open(output / f"hebrus-{VERSION}.tar.gz", "r:gz") as archive:
            self.assertNotIn(f"hebrus-{VERSION}/replacement.txt", archive.getnames())

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

        nonempty = self.base / "nonempty"
        nonempty.mkdir()
        (nonempty / "unrelated.txt").write_text("keep\n", encoding="utf-8")
        rejected = self.tool(
            "build",
            "--repository",
            str(self.repository),
            "--version",
            VERSION,
            "--ref",
            self.commit,
            "--output-dir",
            str(nonempty),
            expect_success=False,
        )
        self.assertIn("must be empty", rejected.stderr)
        self.assertEqual((nonempty / "unrelated.txt").read_text(encoding="utf-8"), "keep\n")

        real_output = self.base / "real-output"
        real_output.mkdir()
        linked_output = self.base / "linked-output"
        linked_output.symlink_to(real_output, target_is_directory=True)
        rejected = self.tool(
            "build",
            "--repository",
            str(self.repository),
            "--version",
            VERSION,
            "--ref",
            self.commit,
            "--output-dir",
            str(linked_output),
            expect_success=False,
        )
        self.assertIn("must not be a symlink", rejected.stderr)

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

    def test_gzip_crc_trailing_data_and_second_member_are_rejected(self) -> None:
        cases = ("crc", "trailing", "concatenated")
        for case in cases:
            with self.subTest(case=case):
                output = self.base / f"gzip-{case}"
                manifest = self.build(output)
                archive = output / f"hebrus-{VERSION}.tar.gz"
                data = bytearray(archive.read_bytes())
                if case == "crc":
                    data[-8] ^= 0x01
                    expected = "checksum or size is invalid"
                elif case == "trailing":
                    data.extend(b"unexpected")
                    expected = "trailing or concatenated"
                else:
                    data.extend(gzip.compress(b"second stream", mtime=0))
                    expected = "trailing or concatenated"
                self.replace_archive(manifest, bytes(data))
                result = self.tool(
                    "verify", "--manifest", str(manifest), expect_success=False
                )
                self.assertIn(expected, result.stderr)

    def test_unsafe_tar_structure_links_and_metadata_are_rejected(self) -> None:
        root = f"hebrus-{VERSION}"
        cases = {
            "path-alias": (
                [(root, "dir", b"", None), (f"{root}/./file", "file", b"x", None)],
                "escapes its top-level directory",
            ),
            "link-chain": (
                [
                    (root, "dir", b"", None),
                    (f"{root}/a", "link", "b", None),
                    (f"{root}/b", "link", "../..", None),
                ],
                "escapes its top-level directory",
            ),
            "missing-parent": (
                [
                    (root, "dir", b"", None),
                    (f"{root}/missing/child", "file", b"x", None),
                ],
                "preceding directory parent",
            ),
            "file-parent": (
                [
                    (root, "dir", b"", None),
                    (f"{root}/file", "file", b"x", None),
                    (f"{root}/file/child", "file", b"x", None),
                ],
                "preceding directory parent",
            ),
            "extended-metadata": (
                [
                    (root, "dir", b"", None),
                    (
                        f"{root}/file",
                        "file",
                        b"x",
                        {"SCHILY.xattr.user.release-test": "present"},
                    ),
                ],
                "unsupported extended metadata",
            ),
        }
        for name, (members, expected) in cases.items():
            with self.subTest(case=name):
                output = self.base / f"tar-{name}"
                manifest = self.build(output)
                self.replace_with_tar_members(manifest, members)
                result = self.tool(
                    "verify", "--manifest", str(manifest), expect_success=False
                )
                self.assertIn(expected, result.stderr)

    def test_duplicate_manifest_fields_are_rejected(self) -> None:
        output = self.base / "duplicate-json"
        manifest = self.build(output)
        text = manifest.read_text(encoding="utf-8")
        text = text.replace(
            '  "project": "hebrus",\n',
            '  "project": "hebrus",\n  "project": "hebrus",\n',
            1,
        )
        manifest.write_text(text, encoding="utf-8")
        archive = output / f"hebrus-{VERSION}.tar.gz"
        (output / "SHA256SUMS").write_text(
            f"{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n"
            f"{hashlib.sha256(manifest.read_bytes()).hexdigest()}  {manifest.name}\n",
            encoding="utf-8",
        )
        result = self.tool(
            "verify", "--manifest", str(manifest), expect_success=False
        )
        self.assertIn("duplicate field", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
