#!/usr/bin/env python3
"""Enforce the explicit legacy-brand inventory during the Hebrus migration."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = "tools/brand_boundary.json"
SCHEMA_VERSION = 1
TOKENS = ("ds4", "DS4", "DwarfStar")
LOCATIONS = ("path", "content")
CATEGORIES = (
    "serialized/permanent",
    "compatibility",
    "historical-attribution",
    "migration-pending",
)
TOP_LEVEL_KEYS = {
    "schema_version",
    "scope",
    "category_definitions",
    "refresh_policy",
    "entries",
}
ENTRY_KEYS = {"path", "location", "token", "classification", "maximum"}
GLOB_CHARS = frozenset("*?[]{}")


class AuditError(Exception):
    """A malformed manifest or unusable checkout."""


@dataclass(frozen=True)
class Identity:
    path: str
    location: str
    token: str


def identity_key(identity: Identity) -> tuple[str, int, int]:
    return (
        identity.path,
        LOCATIONS.index(identity.location),
        TOKENS.index(identity.token),
    )


def identity_text(identity: Identity) -> str:
    return f"{identity.path}:{identity.location}:{identity.token}"


def path_is_explicit(path: str) -> bool:
    if not path or any(char in path for char in GLOB_CHARS):
        return False
    pure = PurePosixPath(path)
    return not pure.is_absolute() and str(pure) == path and ".." not in pure.parts


def expected_exclusions(root: Path, manifest_path: Path) -> list[str]:
    try:
        relative = manifest_path.resolve().relative_to(root.resolve())
    except ValueError:
        return []
    return [relative.as_posix()]


def require_exact_keys(value: dict[str, Any], expected: set[str], context: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unknown " + ", ".join(extra))
        raise AuditError(f"{context} has invalid keys ({'; '.join(details)})")


def load_manifest(
    root: Path, manifest_path: Path
) -> tuple[dict[str, Any], dict[Identity, dict[str, Any]]]:
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AuditError(f"manifest does not exist: {manifest_path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditError(f"cannot read manifest {manifest_path}: {exc}") from exc

    if not isinstance(document, dict):
        raise AuditError("manifest root must be a JSON object")
    require_exact_keys(document, TOP_LEVEL_KEYS, "manifest")
    if (
        isinstance(document["schema_version"], bool)
        or not isinstance(document["schema_version"], int)
        or document["schema_version"] != SCHEMA_VERSION
    ):
        raise AuditError(
            f"manifest schema_version must be {SCHEMA_VERSION}, "
            f"got {document['schema_version']!r}"
        )

    scope = document["scope"]
    if not isinstance(scope, dict):
        raise AuditError("manifest scope must be an object")
    require_exact_keys(scope, {"tokens", "locations", "excluded_paths"}, "scope")
    if scope["tokens"] != list(TOKENS):
        raise AuditError(f"scope.tokens must be exactly {list(TOKENS)!r}")
    if scope["locations"] != list(LOCATIONS):
        raise AuditError(f"scope.locations must be exactly {list(LOCATIONS)!r}")
    exclusions = expected_exclusions(root, manifest_path)
    if scope["excluded_paths"] != exclusions:
        raise AuditError(
            "scope.excluded_paths may contain only the manifest itself: "
            f"expected {exclusions!r}"
        )

    definitions = document["category_definitions"]
    if not isinstance(definitions, dict) or set(definitions) != set(CATEGORIES):
        raise AuditError(
            "category_definitions must define exactly: " + ", ".join(CATEGORIES)
        )
    if any(not isinstance(definitions[name], str) or not definitions[name].strip()
           for name in CATEGORIES):
        raise AuditError("every category definition must be a non-empty string")

    policy = document["refresh_policy"]
    if not isinstance(policy, dict):
        raise AuditError("refresh_policy must be an object")
    require_exact_keys(
        policy,
        {"reductions", "existing_increase", "new_group"},
        "refresh_policy",
    )
    if any(not isinstance(value, str) or not value.strip() for value in policy.values()):
        raise AuditError("every refresh_policy command must be a non-empty string")

    raw_entries = document["entries"]
    if not isinstance(raw_entries, list):
        raise AuditError("manifest entries must be an array")

    entries: dict[Identity, dict[str, Any]] = {}
    order: list[Identity] = []
    for index, entry in enumerate(raw_entries):
        context = f"entries[{index}]"
        if not isinstance(entry, dict):
            raise AuditError(f"{context} must be an object")
        require_exact_keys(entry, ENTRY_KEYS, context)
        path = entry["path"]
        location = entry["location"]
        token = entry["token"]
        classification = entry["classification"]
        maximum = entry["maximum"]
        if not isinstance(path, str) or not path_is_explicit(path):
            raise AuditError(f"{context}.path must be an explicit normalized path")
        if path in exclusions:
            raise AuditError(f"{context}.path cannot be the excluded manifest")
        if location not in LOCATIONS:
            raise AuditError(f"{context}.location must be one of {LOCATIONS!r}")
        if token not in TOKENS:
            raise AuditError(f"{context}.token must be one of {TOKENS!r}")
        if classification not in CATEGORIES:
            raise AuditError(
                f"{context}.classification must be one of {CATEGORIES!r}"
            )
        if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
            raise AuditError(f"{context}.maximum must be a positive integer")
        identity = Identity(path, location, token)
        if identity in entries:
            raise AuditError(f"duplicate entry: {identity_text(identity)}")
        entries[identity] = entry
        order.append(identity)

    if order != sorted(order, key=identity_key):
        raise AuditError("manifest entries must be in deterministic path/location/token order")
    return document, entries


def tracked_paths(root: Path) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise AuditError(f"cannot enumerate tracked files under {root}: {exc}") from exc
    return sorted(
        item.decode("utf-8", "surrogateescape")
        for item in result.stdout.split(b"\0")
        if item
    )


def worktree_bytes(path: Path) -> bytes | None:
    try:
        if path.is_symlink():
            return os.readlink(path).encode("utf-8", "surrogateescape")
        if not path.is_file():
            return None
        return path.read_bytes()
    except OSError as exc:
        raise AuditError(f"cannot read tracked path {path}: {exc}") from exc


def count_tokens(blob: bytes) -> dict[str, int]:
    counts = {}
    for token in TOKENS:
        count = blob.count(token.encode("ascii"))
        if count:
            counts[token] = count
    return counts


def scan_tree(root: Path, excluded_paths: set[str]) -> dict[Identity, int]:
    observations: dict[Identity, int] = {}
    for relative in tracked_paths(root):
        if relative in excluded_paths:
            continue
        path_counts = count_tokens(relative.encode("utf-8", "surrogateescape"))
        for token, count in path_counts.items():
            observations[Identity(relative, "path", token)] = count

        content = worktree_bytes(root / relative)
        if content is None:
            continue
        for token, count in count_tokens(content).items():
            observations[Identity(relative, "content", token)] = count
    return observations


def parse_identity(value: str) -> Identity:
    try:
        path, location, token = value.rsplit(":", 2)
    except ValueError as exc:
        raise AuditError(
            f"invalid identity {value!r}; expected PATH:LOCATION:TOKEN"
        ) from exc
    identity = Identity(path, location, token)
    if not path_is_explicit(path):
        raise AuditError(f"identity path must be explicit, not a glob: {path!r}")
    if location not in LOCATIONS:
        raise AuditError(f"identity location must be one of {LOCATIONS!r}")
    if token not in TOKENS:
        raise AuditError(f"identity token must be one of {TOKENS!r}")
    return identity


def parse_classification(value: str) -> tuple[Identity, str]:
    try:
        identity_value, category = value.rsplit("=", 1)
    except ValueError as exc:
        raise AuditError(
            f"invalid classification {value!r}; expected "
            "PATH:LOCATION:TOKEN=CATEGORY"
        ) from exc
    if category not in CATEGORIES:
        raise AuditError(f"classification must be one of {CATEGORIES!r}")
    return parse_identity(identity_value), category


def render_violations(
    unknown: list[Identity],
    increases: list[tuple[Identity, int, int]],
) -> None:
    print("brand boundary audit failed:", file=sys.stderr)
    for identity in unknown:
        print(
            f"- unclassified brand token group: {identity_text(identity)}",
            file=sys.stderr,
        )
    for identity, actual, maximum in increases:
        print(
            f"- brand token count increased: {identity_text(identity)} "
            f"{actual} > {maximum}",
            file=sys.stderr,
        )


def inventory_delta(
    observations: dict[Identity, int],
    entries: dict[Identity, dict[str, Any]],
) -> tuple[list[Identity], list[tuple[Identity, int, int]], list[Identity]]:
    unknown = sorted(set(observations) - set(entries), key=identity_key)
    increases = sorted(
        (
            (identity, actual, entries[identity]["maximum"])
            for identity, actual in observations.items()
            if identity in entries and actual > entries[identity]["maximum"]
        ),
        key=lambda item: identity_key(item[0]),
    )
    reductions = sorted(
        (
            identity
            for identity, entry in entries.items()
            if observations.get(identity, 0) < entry["maximum"]
        ),
        key=identity_key,
    )
    return unknown, increases, reductions


def summary(
    observations: dict[Identity, int],
    entries: dict[Identity, dict[str, Any]],
    reductions: list[Identity],
) -> str:
    totals = {category: 0 for category in CATEGORIES}
    for identity, actual in observations.items():
        entry = entries.get(identity)
        if entry:
            totals[entry["classification"]] += actual
    categories = ", ".join(f"{name}={totals[name]}" for name in CATEGORIES)
    return (
        f"{sum(observations.values())} occurrences in {len(observations)} groups; "
        f"{len(reductions)} reductions; {categories}"
    )


def check_inventory(
    observations: dict[Identity, int],
    entries: dict[Identity, dict[str, Any]],
) -> int:
    unknown, increases, reductions = inventory_delta(observations, entries)
    if unknown or increases:
        render_violations(unknown, increases)
        return 1
    print("brand boundary audit passed: " + summary(observations, entries, reductions))
    return 0


def parse_refresh_authorizations(
    raw_increases: list[str], raw_classifications: list[str]
) -> tuple[set[Identity], dict[Identity, str]]:
    accepted: set[Identity] = set()
    for value in raw_increases:
        identity = parse_identity(value)
        if identity in accepted:
            raise AuditError(f"duplicate --accept-increase: {identity_text(identity)}")
        accepted.add(identity)

    classified: dict[Identity, str] = {}
    for value in raw_classifications:
        identity, category = parse_classification(value)
        if identity in classified:
            raise AuditError(f"duplicate --classify: {identity_text(identity)}")
        classified[identity] = category
    return accepted, classified


def refresh_inventory(
    manifest_path: Path,
    document: dict[str, Any],
    observations: dict[Identity, int],
    entries: dict[Identity, dict[str, Any]],
    raw_increases: list[str],
    raw_classifications: list[str],
) -> int:
    unknown, increases, reductions = inventory_delta(observations, entries)
    accepted, classified = parse_refresh_authorizations(
        raw_increases, raw_classifications
    )
    increase_ids = {identity for identity, _, _ in increases}
    unknown_ids = set(unknown)

    unused_increases = accepted - increase_ids
    unused_classifications = set(classified) - unknown_ids
    if unused_increases:
        raise AuditError(
            "--accept-increase does not match a current increase: "
            + ", ".join(identity_text(item) for item in sorted(unused_increases, key=identity_key))
        )
    if unused_classifications:
        raise AuditError(
            "--classify may name only a current unclassified group: "
            + ", ".join(
                identity_text(item)
                for item in sorted(unused_classifications, key=identity_key)
            )
        )

    missing_increases = increase_ids - accepted
    missing_classifications = unknown_ids - set(classified)
    if missing_increases or missing_classifications:
        render_violations(
            sorted(missing_classifications, key=identity_key),
            [item for item in increases if item[0] in missing_increases],
        )
        print(
            "refresh refused: authorize every increase with --accept-increase "
            "and every new group with --classify",
            file=sys.stderr,
        )
        return 1

    refreshed_entries = []
    for identity in sorted(observations, key=identity_key):
        category = (
            entries[identity]["classification"]
            if identity in entries
            else classified[identity]
        )
        refreshed_entries.append(
            {
                "path": identity.path,
                "location": identity.location,
                "token": identity.token,
                "classification": category,
                "maximum": observations[identity],
            }
        )

    refreshed = dict(document)
    refreshed["entries"] = refreshed_entries
    payload = json.dumps(refreshed, indent=2, ensure_ascii=True) + "\n"
    temporary = manifest_path.with_name(manifest_path.name + ".tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, manifest_path)
    except OSError as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise AuditError(f"cannot update manifest {manifest_path}: {exc}") from exc

    refreshed_map = {
        Identity(entry["path"], entry["location"], entry["token"]): entry
        for entry in refreshed_entries
    }
    print(
        "brand boundary manifest refreshed: "
        + summary(observations, refreshed_map, reductions)
    )
    return 0


def parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(
        description=(
            "Check the exact tracked ds4/DS4/DwarfStar inventory. Plain refresh "
            "only tightens the baseline; widening requires exact authorizations."
        )
    )
    action = argument_parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true", help="verify the inventory")
    action.add_argument(
        "--refresh",
        action="store_true",
        help="rewrite the manifest deterministically after explicit review",
    )
    argument_parser.add_argument(
        "--root",
        help="repository root, defaulting to the checkout containing this tool",
    )
    argument_parser.add_argument(
        "--manifest",
        help=f"manifest path relative to --root (default: {DEFAULT_MANIFEST})",
    )
    argument_parser.add_argument(
        "--accept-increase",
        action="append",
        default=[],
        metavar="PATH:LOCATION:TOKEN",
        help="authorize one existing group whose count increased; repeat as needed",
    )
    argument_parser.add_argument(
        "--classify",
        action="append",
        default=[],
        metavar="PATH:LOCATION:TOKEN=CATEGORY",
        help="classify one new exact group; repeat as needed",
    )
    return argument_parser


def main(argv: list[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    if arguments.check and (arguments.accept_increase or arguments.classify):
        print(
            "brand boundary audit error: refresh authorizations require --refresh",
            file=sys.stderr,
        )
        return 2

    root = Path(arguments.root).resolve() if arguments.root else DEFAULT_ROOT
    manifest_path = Path(arguments.manifest) if arguments.manifest else Path(DEFAULT_MANIFEST)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    manifest_path = manifest_path.resolve()

    try:
        document, entries = load_manifest(root, manifest_path)
        exclusions = set(document["scope"]["excluded_paths"])
        observations = scan_tree(root, exclusions)
        if arguments.check:
            return check_inventory(observations, entries)
        return refresh_inventory(
            manifest_path,
            document,
            observations,
            entries,
            arguments.accept_increase,
            arguments.classify,
        )
    except AuditError as exc:
        print(f"brand boundary audit error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
