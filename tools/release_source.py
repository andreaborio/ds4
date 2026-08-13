#!/usr/bin/env python3
"""Build and verify deterministic Hebrus source-release bundles."""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import json
import os
import posixpath
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import zlib
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
PROJECT = "hebrus"
ARCHIVE_FORMAT = "git-archive-tar-gzip-v1"
PRERELEASE_IDENTIFIER = (
    r"(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
)
VERSION_RE = re.compile(
    r"(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)"
    rf"(?:-{PRERELEASE_IDENTIFIER}(?:\.{PRERELEASE_IDENTIFIER})*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
)
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
CANONICAL_MODES = {
    "file": frozenset({0o644, 0o755}),
    "dir": frozenset({0o755}),
    "link": frozenset({0o777}),
}
CANONICAL_GZIP_HEADER = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x02\xff"


class ReleaseSourceError(RuntimeError):
    """A release bundle failed a provenance or integrity check."""


def fail(message: str) -> None:
    raise ReleaseSourceError(message)


def clean_git_environment() -> dict[str, str]:
    """Return an environment without caller-selected Git repositories/config."""
    environment = os.environ.copy()
    for key in tuple(environment):
        if key.startswith("GIT_"):
            environment.pop(key)
    environment.update(
        {
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
        }
    )
    return environment


def run_git(repository: Path, *args: str, text: bool = True) -> str | bytes:
    try:
        completed = subprocess.run(
            ["git", "--no-replace-objects", *args],
            cwd=repository,
            env=clean_git_environment(),
            check=True,
            capture_output=True,
            text=text,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() if isinstance(exc.stderr, str) else ""
        fail(f"git {' '.join(args)} failed{': ' + detail if detail else ''}")
    return completed.stdout


def validate_version(version: str) -> None:
    if VERSION_RE.fullmatch(version) is None:
        fail(
            "version must be a SemVer value without a leading 'v' "
            "(for example 1.2.3 or 1.2.3-rc.1)"
        )


def resolve_immutable_ref(repository: Path, ref: str) -> str:
    normalized = ref.lower()
    if COMMIT_RE.fullmatch(normalized):
        commit = str(run_git(repository, "rev-parse", "--verify", f"{ref}^{{commit}}"))
        commit = commit.strip().lower()
        if commit != normalized:
            fail(f"full commit ref resolved unexpectedly: {ref} -> {commit}")
        return commit

    tag_ref = ref if ref.startswith("refs/tags/") else f"refs/tags/{ref}"
    try:
        run_git(repository, "show-ref", "--verify", "--quiet", tag_ref)
    except ReleaseSourceError:
        fail("ref must be a full 40-character commit or an exact local tag")
    return str(run_git(repository, "rev-parse", "--verify", f"{tag_ref}^{{commit}}")) \
        .strip().lower()


def require_tag_matches_version(ref: str, version: str) -> None:
    """Bind a tag-based final bundle to the same SemVer as its asset names."""
    if COMMIT_RE.fullmatch(ref.lower()):
        return
    expected = f"v{version}"
    normalized = ref.removeprefix("refs/tags/")
    if normalized != expected:
        fail(f"release tag must be {expected} for version {version}")


def require_release_checkout(repository: Path, commit: str) -> None:
    head = str(run_git(repository, "rev-parse", "--verify", "HEAD^{commit}")) \
        .strip().lower()
    if head != commit:
        fail(f"release ref {commit} is not checked out at HEAD {head}")

    status = str(
        run_git(repository, "status", "--porcelain", "--untracked-files=normal")
    )
    if status:
        fail("release source bundles require a clean working tree")

    tree = str(run_git(repository, "ls-tree", "-r", commit))
    submodules = [line for line in tree.splitlines() if line.startswith("160000 ")]
    if submodules:
        fail("release tree contains submodules that git archive would not expand")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def parse_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            fail(f"source manifest contains a duplicate field: {key}")
        value[key] = item
    return value


def require_plain_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        fail(f"{label} must be a regular file: {path}")


def safe_member_path(name: str, root: str) -> bool:
    if "\\" in name or "\x00" in name:
        return False
    parts = name.split("/")
    return (
        parts[0] == root
        and all(part not in ("", ".", "..") for part in parts)
        and (len(parts) > 1 or name == root)
    )


def validate_link_target(member: tarfile.TarInfo) -> None:
    target = member.linkname
    if (
        not target
        or target.startswith("/")
        or "\\" in target
        or "\x00" in target
        or posixpath.normpath(target) != target
    ):
        fail(f"archive has unsafe link target: {member.name} -> {target}")


def validate_link_graph(symlinks: dict[str, str], root: str) -> None:
    """Resolve archive symlinks component-wise so chained links cannot escape."""
    for link_name, initial_target in symlinks.items():
        pending = link_name.split("/")[:-1] + initial_target.split("/")
        resolved: list[str] = []
        followed: set[str] = set()
        while pending:
            part = pending.pop(0)
            if part == ".":
                continue
            if part == "..":
                if len(resolved) <= 1:
                    fail(
                        "archive link escapes its top-level directory: "
                        f"{link_name} -> {initial_target}"
                    )
                resolved.pop()
                continue

            resolved.append(part)
            candidate = "/".join(resolved)
            if candidate not in symlinks:
                continue
            if candidate in followed:
                fail(f"archive contains a symlink cycle involving {candidate}")
            followed.add(candidate)
            resolved.pop()
            pending = symlinks[candidate].split("/") + pending

        if not resolved or resolved[0] != root:
            fail(
                "archive link escapes its top-level directory: "
                f"{link_name} -> {initial_target}"
            )


@contextlib.contextmanager
def open_canonical_gzip_tar(path: Path):
    try:
        raw = path.open("rb")
    except OSError as exc:
        fail(f"cannot open source archive {path}: {exc}")
    try:
        header = raw.read(len(CANONICAL_GZIP_HEADER))
        if header != CANONICAL_GZIP_HEADER:
            fail(f"source archive gzip wrapper is not canonical: {path}")

        decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
        crc = 0
        expanded_size = 0
        trailer = b""
        with tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024) as expanded:
            while True:
                chunk = raw.read(1024 * 1024)
                if not chunk:
                    break
                data = decompressor.decompress(chunk)
                if data:
                    expanded.write(data)
                    crc = zlib.crc32(data, crc)
                    expanded_size += len(data)
                if decompressor.eof:
                    trailer = decompressor.unused_data + raw.read()
                    break

            if not decompressor.eof:
                fail(f"source archive has a truncated gzip stream: {path}")
            flushed = decompressor.flush()
            if flushed:
                expanded.write(flushed)
                crc = zlib.crc32(flushed, crc)
                expanded_size += len(flushed)
            if len(trailer) != 8:
                fail(f"source archive has trailing or concatenated gzip data: {path}")
            expected_crc, expected_size = struct.unpack("<II", trailer)
            if expected_crc != crc or expected_size != expanded_size & 0xFFFFFFFF:
                fail(f"source archive gzip checksum or size is invalid: {path}")

            expanded.seek(0)
            with tarfile.open(fileobj=expanded, mode="r:") as archive:
                yield archive, expanded, expanded_size
    except (gzip.BadGzipFile, tarfile.TarError, EOFError, OSError, zlib.error) as exc:
        fail(f"cannot read source archive {path}: {exc}")
    finally:
        raw.close()


def validate_archive(
    path: Path,
    *,
    root: str,
    source_date_epoch: int,
    git_commit: str,
) -> int:
    require_plain_regular_file(path, "source archive")

    count = 0
    member_kinds: dict[str, str] = {}
    symlinks: dict[str, str] = {}
    last_data_end = 0
    with open_canonical_gzip_tar(path) as (archive, expanded, expanded_size):
        if archive.pax_headers != {"comment": git_commit}:
            fail("archive global metadata does not match its Git commit")
        for member in archive:
            count += 1
            if not safe_member_path(member.name, root):
                fail(f"archive member escapes its top-level directory: {member.name}")
            if member.name in member_kinds:
                fail(f"archive contains a duplicate member: {member.name}")
            if count == 1:
                if member.name != root or not member.isdir():
                    fail("archive does not begin with its top-level directory")
            else:
                parent = member.name.rsplit("/", 1)[0]
                if member_kinds.get(parent) != "dir":
                    fail(f"archive member lacks a preceding directory parent: {member.name}")
            if ".git" in PurePosixPath(member.name).parts:
                fail(f"archive unexpectedly contains Git metadata: {member.name}")
            if member.mtime != source_date_epoch:
                fail(
                    f"archive member has non-canonical mtime: {member.name} "
                    f"({member.mtime} != {source_date_epoch})"
                )
            if (
                member.uid != 0
                or member.gid != 0
                or member.uname != "root"
                or member.gname != "root"
            ):
                fail(f"archive member has non-canonical ownership: {member.name}")
            if member.type not in {tarfile.REGTYPE, tarfile.DIRTYPE, tarfile.SYMTYPE}:
                fail(f"archive contains unsupported entry type: {member.name}")
            kind = (
                "file" if member.isfile() else "dir" if member.isdir() else "link"
            )
            mode = member.mode
            if mode not in CANONICAL_MODES[kind]:
                fail(f"archive member has non-canonical mode: {member.name} ({mode:o})")
            if member.devmajor != 0 or member.devminor != 0:
                fail(f"archive member has non-zero device numbers: {member.name}")
            if kind != "file" and member.size != 0:
                fail(f"archive non-file member has data: {member.name}")
            if not member.issym() and member.linkname:
                fail(f"archive non-link member has a link target: {member.name}")

            allowed_pax = {"comment", "path", "linkpath"}
            if set(member.pax_headers) - allowed_pax:
                fail(f"archive member has unsupported extended metadata: {member.name}")
            if member.pax_headers.get("comment") != git_commit:
                fail(f"archive member metadata does not match its Git commit: {member.name}")
            if "path" in member.pax_headers and member.pax_headers["path"] != member.name:
                fail(f"archive member has inconsistent extended path: {member.name}")
            if "linkpath" in member.pax_headers and (
                not member.issym() or member.pax_headers["linkpath"] != member.linkname
            ):
                fail(f"archive member has inconsistent extended link path: {member.name}")

            member_kinds[member.name] = kind
            if member.issym():
                validate_link_target(member)
                symlinks[member.name] = member.linkname

            last_data_end = member.offset_data + (
                (member.size + tarfile.BLOCKSIZE - 1) // tarfile.BLOCKSIZE
            ) * tarfile.BLOCKSIZE

        if count == 0:
            fail("archive is empty or lacks its top-level directory entry")
        validate_link_graph(symlinks, root)

        # git archive writes complete 10 KiB records. At least the two tar EOF
        # blocks must remain after the final member, extending to that boundary.
        expected_padding = (-last_data_end) % tarfile.RECORDSIZE
        if expected_padding < 2 * tarfile.BLOCKSIZE:
            expected_padding += tarfile.RECORDSIZE
        if expanded_size - last_data_end != expected_padding:
            fail("archive tar record padding is not canonical")
        expanded.seek(last_data_end)
        if any(expanded.read()):
            fail("archive has non-zero data after its final member")

    if count == 0:
        fail("archive is empty or lacks its top-level directory entry")
    return count


def artifact_names(version: str) -> tuple[str, str, str, str]:
    stem = f"{PROJECT}-{version}"
    return stem, f"{stem}.tar.gz", f"{stem}-source.json", "SHA256SUMS"


def expected_checksums(archive_path: Path, manifest_path: Path) -> str:
    return (
        f"{sha256_file(archive_path)}  {archive_path.name}\n"
        f"{sha256_file(manifest_path)}  {manifest_path.name}\n"
    )


def create_isolated_git_archive(
    repository: Path,
    commit: str,
    prefix: str,
    destination: Path,
    work: Path,
) -> None:
    """Archive a commit through a bare object view with no local attributes."""
    object_text = str(
        run_git(repository, "rev-parse", "--path-format=absolute", "--git-path", "objects")
    ).strip()
    if not object_text or "\n" in object_text or "\r" in object_text:
        fail("repository has an invalid Git object directory")
    object_directory = Path(object_text)
    if not object_directory.is_dir():
        fail(f"repository Git object directory is unavailable: {object_directory}")

    git_directory = work / "repository.git"
    (git_directory / "objects" / "info").mkdir(parents=True)
    (git_directory / "refs").mkdir()
    (git_directory / "config").write_text(
        "[core]\n\trepositoryformatversion = 0\n\tbare = true\n",
        encoding="utf-8",
        newline="\n",
    )
    (git_directory / "HEAD").write_text(
        "ref: refs/heads/unborn\n", encoding="utf-8", newline="\n"
    )
    (git_directory / "objects" / "info" / "alternates").write_text(
        str(object_directory.resolve()) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    try:
        subprocess.run(
            [
                "git",
                "--no-replace-objects",
                f"--git-dir={git_directory}",
                "-c",
                "tar.umask=0022",
                "archive",
                "--format=tar",
                f"--prefix={prefix}/",
                f"--output={destination}",
                commit,
            ],
            cwd=repository,
            env=clean_git_environment(),
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip()
        fail(f"git archive failed{': ' + detail if detail else ''}")


def publish_artifacts(sources: tuple[Path, ...], destinations: list[Path]) -> None:
    """Publish without ever replacing a destination created by another writer."""
    published: list[tuple[Path, Path]] = []
    try:
        for source, destination in zip(sources, destinations, strict=True):
            try:
                os.link(source, destination, follow_symlinks=False)
            except FileExistsError:
                fail(f"refusing to overwrite release artifact: {destination}")
            published.append((source, destination))
    except (OSError, ReleaseSourceError):
        for source, destination in reversed(published):
            try:
                source_stat = source.stat(follow_symlinks=False)
                destination_stat = destination.stat(follow_symlinks=False)
                if (
                    source_stat.st_dev == destination_stat.st_dev
                    and source_stat.st_ino == destination_stat.st_ino
                ):
                    destination.unlink()
            except (FileNotFoundError, OSError):
                pass
        raise


def verify_bundle(manifest_path: Path) -> dict[str, object]:
    require_plain_regular_file(manifest_path, "source manifest")
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=parse_json_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        fail(f"cannot read source manifest {manifest_path}: {exc}")
    if not isinstance(manifest, dict):
        fail("source manifest root must be an object")

    required = {
        "schema_version",
        "project",
        "version",
        "git_commit",
        "git_commit_short",
        "source_date_epoch",
        "archive_format",
        "archive",
    }
    if set(manifest) != required:
        fail(f"source manifest fields differ from schema {SCHEMA_VERSION}")
    if (
        type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != SCHEMA_VERSION
        or manifest["project"] != PROJECT
    ):
        fail("source manifest has an unsupported schema or project")

    version = manifest["version"]
    if not isinstance(version, str):
        fail("source manifest version must be a string")
    validate_version(version)
    stem, archive_name, expected_manifest_name, checksum_name = artifact_names(version)
    if manifest_path.name != expected_manifest_name:
        fail(f"source manifest filename must be {expected_manifest_name}")

    commit = manifest["git_commit"]
    short_commit = manifest["git_commit_short"]
    epoch = manifest["source_date_epoch"]
    if not isinstance(commit, str) or COMMIT_RE.fullmatch(commit) is None:
        fail("source manifest git_commit must be a lowercase full commit")
    if not isinstance(short_commit, str) or short_commit != commit[:12]:
        fail("source manifest short commit does not match git_commit")
    if type(epoch) is not int or epoch < 0:
        fail("source manifest source_date_epoch must be a non-negative integer")
    if manifest["archive_format"] != ARCHIVE_FORMAT:
        fail("source manifest archive format is unsupported")

    archive = manifest["archive"]
    if not isinstance(archive, dict) or set(archive) != {
        "filename", "root", "bytes", "sha256", "members"
    }:
        fail("source manifest archive fields differ from schema")
    if archive["filename"] != archive_name or archive["root"] != stem:
        fail("source manifest archive name or root does not match its version")
    if type(archive["bytes"]) is not int or archive["bytes"] <= 0:
        fail("source manifest archive byte count must be positive")
    if not isinstance(archive["sha256"], str) or SHA256_RE.fullmatch(archive["sha256"]) is None:
        fail("source manifest archive SHA-256 is invalid")
    if type(archive["members"]) is not int or archive["members"] <= 0:
        fail("source manifest archive member count must be positive")

    archive_path = manifest_path.parent / archive_name
    require_plain_regular_file(archive_path, "source archive")
    if archive_path.stat().st_size != archive["bytes"]:
        fail("source archive byte count does not match its manifest")
    if sha256_file(archive_path) != archive["sha256"]:
        fail("source archive SHA-256 does not match its manifest")
    members = validate_archive(
        archive_path,
        root=stem,
        source_date_epoch=epoch,
        git_commit=commit,
    )
    if members != archive["members"]:
        fail("source archive member count does not match its manifest")

    checksum_path = manifest_path.parent / checksum_name
    require_plain_regular_file(checksum_path, "checksum file")
    try:
        actual_checksums = checksum_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        fail(f"cannot read checksum file {checksum_path}: {exc}")
    expected = expected_checksums(archive_path, manifest_path)
    if actual_checksums != expected:
        fail("SHA256SUMS is missing, reordered, malformed, or stale")
    return manifest


def build_bundle(repository: Path, version: str, ref: str, output_dir: Path) -> Path:
    repository = repository.resolve()
    if output_dir.is_symlink():
        fail(f"output directory must not be a symlink: {output_dir}")
    output_dir = output_dir.resolve()
    validate_version(version)
    commit = resolve_immutable_ref(repository, ref)
    require_tag_matches_version(ref, version)
    require_release_checkout(repository, commit)

    stem, archive_name, manifest_name, checksum_name = artifact_names(version)
    destinations = [output_dir / name for name in (archive_name, manifest_name, checksum_name)]
    existing = [path for path in destinations if path.exists() or path.is_symlink()]
    if existing:
        fail("refusing to overwrite release artifact(s): " + ", ".join(map(str, existing)))

    epoch_text = str(run_git(repository, "show", "-s", "--format=%ct", commit)).strip()
    try:
        source_date_epoch = int(epoch_text)
    except ValueError:
        fail(f"commit has invalid timestamp: {epoch_text}")

    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        fail(f"release output directory must be empty: {output_dir}")
    with tempfile.TemporaryDirectory(prefix=".hebrus-release-", dir=output_dir) as temporary:
        work = Path(temporary)
        tar_path = work / f"{stem}.tar"
        archive_path = work / archive_name
        manifest_path = work / manifest_name
        checksum_path = work / checksum_name

        create_isolated_git_archive(repository, commit, stem, tar_path, work)

        with tar_path.open("rb") as source, archive_path.open("wb") as destination:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=9,
                fileobj=destination,
                mtime=0,
            ) as compressed:
                shutil.copyfileobj(source, compressed, length=1024 * 1024)

        members = validate_archive(
            archive_path,
            root=stem,
            source_date_epoch=source_date_epoch,
            git_commit=commit,
        )
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "project": PROJECT,
            "version": version,
            "git_commit": commit,
            "git_commit_short": commit[:12],
            "source_date_epoch": source_date_epoch,
            "archive_format": ARCHIVE_FORMAT,
            "archive": {
                "filename": archive_name,
                "root": stem,
                "bytes": archive_path.stat().st_size,
                "sha256": sha256_file(archive_path),
                "members": members,
            },
        }
        write_json(manifest_path, manifest)
        checksum_path.write_text(
            expected_checksums(archive_path, manifest_path),
            encoding="utf-8",
            newline="\n",
        )

        verify_bundle(manifest_path)
        publish_artifacts(
            (archive_path, manifest_path, checksum_path),
            destinations,
        )

    final_manifest = output_dir / manifest_name
    verify_bundle(final_manifest)
    return final_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="build a deterministic source bundle")
    build.add_argument("--version", required=True, help="SemVer without a leading v")
    build.add_argument(
        "--ref",
        required=True,
        help="full 40-character commit or exact local tag checked out at HEAD",
    )
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--repository", type=Path, default=ROOT)

    verify = subparsers.add_parser("verify", help="verify a source bundle and checksum set")
    verify.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "build":
            manifest = build_bundle(
                args.repository,
                args.version,
                args.ref,
                args.output_dir,
            )
            value = verify_bundle(manifest)
            print(
                f"release-source: built {value['archive']['filename']} "
                f"for {value['git_commit']}"
            )
            print(f"release-source: manifest {manifest}")
        else:
            value = verify_bundle(args.manifest.absolute())
            print(
                f"release-source: verified {value['archive']['filename']} "
                f"({value['archive']['sha256']})"
            )
    except (ReleaseSourceError, OSError) as exc:
        print(f"release-source: FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
