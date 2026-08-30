#!/usr/bin/env python3
"""Offline Qwen4Exp source-inventory admission and conversion dry-run.

This tool reads only the committed safetensor header/index inventory.  It does
not open checkpoint shards, allocate model tensors, choose a routed/PLE codec,
or emit an artifact.  Every pinned source identity is validated and assigned
to exactly one destination owner before a future converter can write bytes.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "docs" / "contracts" / "qwen4exp-profile.json"
DEFAULT_INVENTORY = (
    ROOT / "tests" / "qwen4exp" / "fixtures" /
    "qwen38flash-next-inventory-v1.json"
)

HF_REPOSITORY = "Qwen/Qwen3.8-Flash-Next"
HF_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
TRANSFORMERS_COMMIT = "42ca97014c85d71a88ad60d55f08cb9fb4d26e2c"
INVENTORY_SHA256 = "a639efc7a5147b04200e870d7e320335527f4361a8327b137feca2683b1dc434"
INVENTORY_FILE_SHA256 = "56f08c30aeae33a46aa571b25fa7104837dc2d9fdd6462d62090dba34383af6e"
CONTRACT_FILE_SHA256 = "5d1e1a13b78bd2cba5e8aa8494c18efd825e21d85aad42d3d32a1e0728c8f75c"
EXPECTED_TENSORS = 1658
EXPECTED_SHARDS = 131
EXPECTED_SOURCE_BYTES = 359999963128
EXPECTED_CLASSIFICATIONS = {
    "PLE": 137,
    "dense": 1061,
    "excluded-MTP": 31,
    "excluded-vision": 333,
    "routed": 96,
}
DTYPE_BYTES = {"BF16": 2, "I64": 8}
ROUTED_SOURCE_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\."
    r"(gate_up_proj|down_proj)$"
)
PLE_SHARD_RE = re.compile(
    r"^model\.language_model\.layers\.1\.ple\.ple_embedding\."
    r"ngram_embedding\.shard_(\d+)\.weight$"
)
SHARD_RE = re.compile(r"^model-(\d{5})-of-00131\.safetensors$")


class ProfileError(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class ExpertMajorAdapter:
    family: int
    profile_id: str
    layers: int
    experts: int
    experts_used: int
    components: tuple[str, ...]
    source_gate_up_shape: tuple[int, ...]
    source_down_shape: tuple[int, ...]
    destination_gate_shape: tuple[int, ...]
    destination_up_shape: tuple[int, ...]
    destination_down_shape: tuple[int, ...]
    codec_status: str


@dataclasses.dataclass(frozen=True)
class PleAdapter:
    family: int
    profile_id: str
    magic: str
    hash_id: str
    rows: int
    head_rows: int
    width: int
    row_alignment: int
    source_shards: int
    manifest_header_bytes: int
    page_header_bytes: int
    page_digest_bytes: int
    minimum_page_alignment: int
    binary_emission: bool
    codec_status: str
    geometry_status: str


QWEN4EXP_EXPERT_ADAPTER = ExpertMajorAdapter(
    family=4,
    profile_id="qwen4exp-base-v1",
    layers=48,
    experts=512,
    experts_used=10,
    components=("gate", "up", "down"),
    source_gate_up_shape=(512, 1280, 2560),
    source_down_shape=(512, 2560, 640),
    destination_gate_shape=(2560, 640, 512),
    destination_up_shape=(2560, 640, 512),
    destination_down_shape=(640, 2560, 512),
    codec_status="unselected-fail-closed",
)

QWEN4EXP_PLE_ADAPTER = PleAdapter(
    family=4,
    profile_id="qwen4exp-base-v1",
    magic="ds4.ple_rows.v1",
    hash_id="SplitMix64-Qwen4Exp-v1",
    rows=320001536,
    head_rows=320001446,
    width=160,
    row_alignment=128,
    source_shards=128,
    manifest_header_bytes=512,
    page_header_bytes=64,
    page_digest_bytes=32,
    minimum_page_alignment=4096,
    binary_emission=False,
    codec_status="descriptor-ready-page-codec-pending",
    geometry_status="caller-supplied-fail-closed",
)


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProfileError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProfileError(f"{path}: top-level JSON value must be an object")
    return value


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory_digest(tensors: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in tensors:
        digest.update("\0".join((
            row["name"], row["dtype"],
            ",".join(str(value) for value in row["shape"]),
            row["shard"], str(row["begin"]), str(row["end"]),
        )).encode("utf-8") + b"\n")
    return digest.hexdigest()


def canonical_json(document: object, *, pretty: bool = False) -> bytes:
    if pretty:
        text = json.dumps(document, indent=2, sort_keys=True)
    else:
        text = json.dumps(document, separators=(",", ":"), sort_keys=True)
    return (text + "\n").encode("utf-8")


def require_equal(field: str, observed: object, expected: object) -> None:
    if observed != expected:
        raise ProfileError(
            f"{field}: expected {expected!r}, observed {observed!r}"
        )


def role_name(entry: dict) -> str:
    path = entry.get("path")
    leaf = entry.get("leaf")
    if not isinstance(path, str) or not path:
        raise ProfileError(f"invalid tensor-role path: {entry!r}")
    if leaf is not None and not isinstance(leaf, str):
        raise ProfileError(f"invalid tensor-role leaf for {path}")
    return path + (f".{leaf}" if leaf else "")


def expected_base_inventory(contract: dict) -> dict[str, tuple[str, tuple[int, ...]]]:
    try:
        roles = contract["tensorRoles"]
        text = contract["pinnedConfig"]["text"]
        layer_types = text["layer_types"]
        layer_count = text["num_hidden_layers"]
        ple_layer = roles["pleRuntimeLayer"]
    except (KeyError, TypeError) as exc:
        raise ProfileError(f"contract tensor roles are incomplete: {exc}") from exc
    require_equal("pinnedConfig.text.num_hidden_layers", layer_count, 48)
    if not isinstance(layer_types, list) or len(layer_types) != layer_count:
        raise ProfileError("pinnedConfig.text.layer_types must contain 48 entries")

    scopes: tuple[tuple[str, tuple[int | None, ...]], ...] = (
        ("global", (None,)),
        ("all", tuple(range(layer_count))),
        ("linear_attention", tuple(
            index for index, kind in enumerate(layer_types)
            if kind == "linear_attention"
        )),
        ("full_attention", tuple(
            index for index, kind in enumerate(layer_types)
            if kind == "full_attention"
        )),
        ("ple", (ple_layer,)),
    )
    expected: dict[str, tuple[str, tuple[int, ...]]] = {}
    for scope, layers in scopes:
        entries = roles.get(scope)
        if not isinstance(entries, list):
            raise ProfileError(f"tensorRoles.{scope} must be an array")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ProfileError(f"tensorRoles.{scope} entry is not an object")
            dtype = entry.get("dtype")
            shape = entry.get("shape")
            if dtype not in DTYPE_BYTES or not isinstance(shape, list) or \
                    any(not isinstance(dim, int) or isinstance(dim, bool) or
                        dim <= 0 for dim in shape):
                raise ProfileError(f"invalid role dtype/shape: {entry!r}")
            template = role_name(entry)
            count = entry.get("count")
            if count is not None and (
                    not isinstance(count, int) or isinstance(count, bool) or
                    count <= 0):
                raise ProfileError(f"invalid tensor-role count: {entry!r}")
            for layer in layers:
                name = template
                if layer is not None:
                    name = name.replace("layers.N", f"layers.{layer}")
                names = (name,) if count is None else tuple(
                    name.replace("shard_N", f"shard_{index}")
                    for index in range(count)
                )
                for identity in names:
                    if identity in expected:
                        raise ProfileError(
                            f"duplicate contract tensor identity: {identity}"
                        )
                    expected[identity] = (dtype, tuple(shape))
    require_equal(
        "tensorRoles.expectedBaseTensors", len(expected),
        roles.get("expectedBaseTensors"),
    )
    require_equal("closed base tensor count", len(expected), 1294)
    return expected


def validate_contract(contract: dict, contract_path: Path) -> None:
    require_equal("contract file SHA-256", file_sha256(contract_path),
                  CONTRACT_FILE_SHA256)
    require_equal("contract.schemaVersion", contract.get("schemaVersion"), 1)
    require_equal("contract.kind", contract.get("kind"),
                  "qwen4exp-profile-contract")
    require_equal("contract.status", contract.get("status"),
                  "pinned-not-supported")
    identity = contract.get("identity", {})
    pins = contract.get("sourcePins", {})
    require_equal("identity.hebrusFamily", identity.get("hebrusFamily"),
                  "qwen4exp")
    require_equal("identity.expertStoreFamily",
                  identity.get("expertStoreFamily"), 4)
    require_equal("identity.artifactProfileId",
                  identity.get("artifactProfileId"), "qwen4exp-base-v1")
    require_equal("identity.hfRepository", identity.get("hfRepository"),
                  HF_REPOSITORY)
    require_equal("identity.hfRevision", identity.get("hfRevision"),
                  HF_REVISION)
    require_equal("sourcePins.transformersCommit",
                  pins.get("transformersCommit"), TRANSFORMERS_COMMIT)
    require_equal("sourcePins.inventorySha256",
                  pins.get("inventorySha256"), INVENTORY_SHA256)
    require_equal("sourcePins.tensorCount", pins.get("tensorCount"),
                  EXPECTED_TENSORS)
    require_equal("sourcePins.shardCount", pins.get("shardCount"),
                  EXPECTED_SHARDS)
    require_equal("sourcePins.sourceBytes", pins.get("sourceBytes"),
                  EXPECTED_SOURCE_BYTES)
    require_equal("sourcePins.classification", pins.get("classification"), {
        "base": 1294, "mtp": 31, "ple": 137, "vision": 333,
    })
    require_equal("expertStore.family",
                  contract.get("expertStore", {}).get("family"), 4)
    require_equal("expertStore.layers",
                  contract.get("expertStore", {}).get("layers"), 48)
    require_equal("expertStore.expertsPerLayer",
                  contract.get("expertStore", {}).get("expertsPerLayer"), 512)
    require_equal("expertStore.components",
                  contract.get("expertStore", {}).get("components"),
                  ["gate", "up", "down"])
    require_equal("pinnedConfig.text.num_hidden_layers",
                  contract.get("pinnedConfig", {}).get("text", {}).get(
                      "num_hidden_layers"), 48)
    require_equal("pinnedConfig.text.num_experts",
                  contract.get("pinnedConfig", {}).get("text", {}).get(
                      "num_experts"), 512)
    require_equal("pinnedConfig.text.num_experts_per_tok",
                  contract.get("pinnedConfig", {}).get("text", {}).get(
                      "num_experts_per_tok"), 10)
    ple = contract.get("pleExtent", {})
    require_equal("pleExtent.magic", ple.get("magic"),
                  QWEN4EXP_PLE_ADAPTER.magic)
    require_equal("pleExtent.rows", ple.get("rows"),
                  QWEN4EXP_PLE_ADAPTER.rows)
    require_equal("pleExtent.rowWidth", ple.get("rowWidth"),
                  QWEN4EXP_PLE_ADAPTER.width)
    require_equal("pleExtent.embeddingShardCount",
                  ple.get("embeddingShardCount"),
                  QWEN4EXP_PLE_ADAPTER.source_shards)


def validate_inventory(inventory: dict, inventory_path: Path,
                       contract: dict) -> list[dict]:
    require_equal("inventory.schemaVersion", inventory.get("schemaVersion"), 1)
    require_equal("inventory.kind", inventory.get("kind"),
                  "qwen38flash-next-source-inventory")
    require_equal("inventory.repository", inventory.get("repository"),
                  HF_REPOSITORY)
    require_equal("inventory.revision", inventory.get("revision"), HF_REVISION)
    require_equal("inventory.transformersCommit",
                  inventory.get("transformersCommit"), TRANSFORMERS_COMMIT)
    require_equal("inventory.tensorCount", inventory.get("tensorCount"),
                  EXPECTED_TENSORS)
    require_equal("inventory.shardCount", inventory.get("shardCount"),
                  EXPECTED_SHARDS)
    tensors = inventory.get("tensors")
    if not isinstance(tensors, list):
        raise ProfileError("inventory.tensors must be an array")
    require_equal("inventory tensor array length", len(tensors),
                  EXPECTED_TENSORS)

    expected_keys = {"begin", "dtype", "end", "name", "shape", "shard"}
    names: set[str] = set()
    spans: dict[str, list[tuple[int, int, str]]] = {}
    bytes_by_dtype: dict[str, int] = {}
    previous_name = ""
    for index, row in enumerate(tensors):
        if not isinstance(row, dict) or set(row) != expected_keys:
            raise ProfileError(
                f"inventory.tensors[{index}] must have exactly "
                f"{sorted(expected_keys)}"
            )
        name, dtype, shape, shard = (
            row["name"], row["dtype"], row["shape"], row["shard"]
        )
        begin, end = row["begin"], row["end"]
        if not isinstance(name, str) or not name or name in names:
            raise ProfileError(f"duplicate/invalid tensor identity: {name!r}")
        if previous_name and name <= previous_name:
            raise ProfileError("inventory tensor identities are not sorted")
        previous_name = name
        names.add(name)
        if dtype not in DTYPE_BYTES:
            raise ProfileError(f"{name}: unsupported source dtype {dtype!r}")
        if not isinstance(shape, list) or not shape or any(
                not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0
                for dim in shape):
            raise ProfileError(f"{name}: invalid source shape {shape!r}")
        if not isinstance(shard, str) or not SHARD_RE.fullmatch(shard):
            raise ProfileError(f"{name}: invalid shard ownership {shard!r}")
        if not all(isinstance(value, int) and not isinstance(value, bool)
                   for value in (begin, end)) or begin < 0 or end <= begin:
            raise ProfileError(f"{name}: invalid header byte extent")
        byte_span = math.prod(shape) * DTYPE_BYTES[dtype]
        require_equal(f"{name} byte span", end - begin, byte_span)
        bytes_by_dtype[dtype] = bytes_by_dtype.get(dtype, 0) + byte_span
        spans.setdefault(shard, []).append((begin, end, name))

    shard_digests = inventory.get("shardSha256")
    if not isinstance(shard_digests, dict):
        raise ProfileError("inventory.shardSha256 must be an object")
    owning_shards = set(spans)
    pinned_shards = set(shard_digests)
    expected_shards = {
        f"model-{index:05d}-of-00131.safetensors"
        for index in range(1, EXPECTED_SHARDS + 1)
    }
    if pinned_shards != expected_shards:
        raise ProfileError(
            "pinned shard identity set mismatch; "
            f"missing={sorted(expected_shards - pinned_shards)} "
            f"unexpected={sorted(pinned_shards - expected_shards)}"
        )
    if owning_shards != pinned_shards:
        raise ProfileError(
            "inventory owning shard set mismatch; "
            f"missing={sorted(pinned_shards - owning_shards)} "
            f"unexpected={sorted(owning_shards - pinned_shards)}"
        )
    require_equal("inventory distinct owning shards", len(spans),
                  EXPECTED_SHARDS)
    files = inventory.get("files")
    if not isinstance(files, list):
        raise ProfileError("inventory.files must be an array")
    file_rows: dict[str, dict] = {}
    for row in files:
        if not isinstance(row, dict) or not isinstance(row.get("path"), str):
            raise ProfileError("inventory.files contains an invalid entry")
        if row["path"] in file_rows:
            raise ProfileError(f"duplicate source file identity: {row['path']}")
        file_rows[row["path"]] = row
    for shard, digest in sorted(shard_digests.items()):
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ProfileError(f"{shard}: invalid pinned shard SHA-256")
        file_row = file_rows.get(shard)
        if file_row is None:
            raise ProfileError(f"{shard}: owning shard is absent from files")
        require_equal(f"{shard} file SHA-256", file_row.get("sha256"), digest)
        if file_row.get("lfs") is not True or \
                not isinstance(file_row.get("size"), int) or \
                isinstance(file_row.get("size"), bool) or \
                file_row["size"] <= 0:
            raise ProfileError(f"{shard}: invalid pinned file ownership record")
        prior_end = -1
        for begin, end, name in sorted(spans[shard]):
            if begin < prior_end:
                raise ProfileError(f"{name}: overlapping tensor extent in {shard}")
            if end > file_row["size"]:
                raise ProfileError(f"{name}: tensor extent exceeds owning shard")
            prior_end = end

    require_equal("inventory bytesByDtype", bytes_by_dtype,
                  inventory.get("bytesByDtype"))
    require_equal("inventory total bytes", sum(bytes_by_dtype.values()),
                  int(inventory.get("totalBytes", -1)))
    require_equal("inventory source bytes", sum(bytes_by_dtype.values()),
                  EXPECTED_SOURCE_BYTES)
    digest = inventory_digest(tensors)
    require_equal("inventory tensor digest", digest, INVENTORY_SHA256)
    require_equal("inventory self tensor digest",
                  inventory.get("tensorInventorySha256"), digest)

    expected_base = expected_base_inventory(contract)
    actual_base = {
        row["name"]: (row["dtype"], tuple(row["shape"]))
        for row in tensors
        if not row["name"].startswith("model.visual.") and
        not row["name"].startswith("mtp.")
    }
    missing = sorted(set(expected_base) - set(actual_base))
    unexpected = sorted(set(actual_base) - set(expected_base))
    if missing or unexpected:
        raise ProfileError(
            "closed base inventory mismatch; "
            f"missing={missing[:3]} unexpected={unexpected[:3]}"
        )
    for name in sorted(expected_base):
        require_equal(f"{name} dtype/shape", actual_base[name],
                      expected_base[name])

    # Check the full committed JSON only after producing field-specific
    # failures above. This also pins top-level shard/file evidence that is not
    # duplicated in the tensor identity digest.
    require_equal("inventory fixture file SHA-256", file_sha256(inventory_path),
                  INVENTORY_FILE_SHA256)
    return tensors


def routed_destination(name: str, source_shape: tuple[int, ...]) -> dict:
    match = ROUTED_SOURCE_RE.fullmatch(name)
    if match is None:
        raise AssertionError(name)
    layer = int(match.group(1))
    role = match.group(2)
    if not 0 <= layer < QWEN4EXP_EXPERT_ADAPTER.layers:
        raise ProfileError(f"{name}: routed layer is outside 0..47")
    if role == "gate_up_proj":
        require_equal(f"{name} routed source shape", source_shape,
                      QWEN4EXP_EXPERT_ADAPTER.source_gate_up_shape)
        identities = (
            ("gate", QWEN4EXP_EXPERT_ADAPTER.destination_gate_shape,
             {"axis": 1, "end": 640, "kind": "split-transpose", "start": 0}),
            ("up", QWEN4EXP_EXPERT_ADAPTER.destination_up_shape,
             {"axis": 1, "end": 1280, "kind": "split-transpose", "start": 640}),
        )
    else:
        require_equal(f"{name} routed source shape", source_shape,
                      QWEN4EXP_EXPERT_ADAPTER.source_down_shape)
        identities = ((
            "down", QWEN4EXP_EXPERT_ADAPTER.destination_down_shape,
            {"kind": "expert-major-transpose"},
        ),)
    return {
        "codecStatus": QWEN4EXP_EXPERT_ADAPTER.codec_status,
        "family": QWEN4EXP_EXPERT_ADAPTER.family,
        "owner": "expert-major-v2",
        "records": [
            {
                "component": component,
                "identity": f"blk.{layer}.ffn_{component}_exps.weight",
                "logicalShape": list(shape),
                "sourceTransform": transform,
            }
            for component, shape, transform in identities
        ],
    }


def ple_destination(name: str) -> dict:
    match = PLE_SHARD_RE.fullmatch(name)
    if match is not None:
        shard = int(match.group(1))
        if not 0 <= shard < QWEN4EXP_PLE_ADAPTER.source_shards:
            raise ProfileError(f"{name}: PLE row shard is outside 0..127")
        role = "row-shard"
        physical_owner = "ple-row-store-v1"
    elif name.endswith((
            ".layer_multipliers", ".ngram_heads_offsets",
            ".ngram_heads_vocab_sizes")):
        role = "hash-control"
        physical_owner = "ple-manifest-metadata"
    else:
        role = "runtime-control"
        physical_owner = "base-gguf"
    return {
        "adapter": {
            "binaryEmission": QWEN4EXP_PLE_ADAPTER.binary_emission,
            "codecStatus": QWEN4EXP_PLE_ADAPTER.codec_status,
            "family": QWEN4EXP_PLE_ADAPTER.family,
            "geometryStatus": QWEN4EXP_PLE_ADAPTER.geometry_status,
            "hashId": QWEN4EXP_PLE_ADAPTER.hash_id,
            "magic": QWEN4EXP_PLE_ADAPTER.magic,
            "manifestHeaderBytes":
                QWEN4EXP_PLE_ADAPTER.manifest_header_bytes,
            "headRows": QWEN4EXP_PLE_ADAPTER.head_rows,
            "minimumPageAlignment":
                QWEN4EXP_PLE_ADAPTER.minimum_page_alignment,
            "pageDigestBytes": QWEN4EXP_PLE_ADAPTER.page_digest_bytes,
            "pageHeaderBytes": QWEN4EXP_PLE_ADAPTER.page_header_bytes,
            "paddingRows": (QWEN4EXP_PLE_ADAPTER.rows -
                            QWEN4EXP_PLE_ADAPTER.head_rows),
            "profileId": QWEN4EXP_PLE_ADAPTER.profile_id,
            "rowAlignment": QWEN4EXP_PLE_ADAPTER.row_alignment,
        },
        "owner": physical_owner,
        "role": role,
    }


def classify_and_map(row: dict, expected_base: set[str]) -> tuple[str, dict]:
    name = row["name"]
    if name.startswith("model.visual."):
        return "excluded-vision", {
            "owner": "excluded",
            "policy": "qwen4exp-base-v1-text-only",
        }
    if name.startswith("mtp."):
        return "excluded-MTP", {
            "owner": "excluded",
            "policy": "qwen4exp-base-v1-no-MTP-execution",
        }
    if name.startswith("model.language_model.layers.1.ple."):
        return "PLE", ple_destination(name)
    if ROUTED_SOURCE_RE.fullmatch(name):
        return "routed", routed_destination(name, tuple(row["shape"]))
    if name in expected_base:
        return "dense", {
            "codecStatus": "unselected-fail-closed",
            "logicalIdentity": name,
            "owner": "base-gguf",
        }
    raise ProfileError(f"{name}: no closed source-to-destination rule")


def make_report(inventory_path: Path = DEFAULT_INVENTORY,
                contract_path: Path = DEFAULT_CONTRACT) -> dict:
    inventory_path = inventory_path.resolve()
    contract_path = contract_path.resolve()
    contract = read_json(contract_path)
    inventory = read_json(inventory_path)
    validate_contract(contract, contract_path)
    tensors = validate_inventory(inventory, inventory_path, contract)
    expected_base = set(expected_base_inventory(contract))
    entries: list[dict] = []
    counts: dict[str, int] = {}
    bytes_by_classification: dict[str, int] = {}
    classified: set[str] = set()
    routed_records = 0
    for row in tensors:
        name = row["name"]
        if name in classified:
            raise ProfileError(f"{name}: source identity consumed more than once")
        classification, destination = classify_and_map(row, expected_base)
        classified.add(name)
        counts[classification] = counts.get(classification, 0) + 1
        byte_span = row["end"] - row["begin"]
        bytes_by_classification[classification] = \
            bytes_by_classification.get(classification, 0) + byte_span
        if classification == "routed":
            routed_records += len(destination["records"])
        entries.append({
            "classification": classification,
            "destination": destination,
            "source": {
                "byteExtent": [row["begin"], row["end"]],
                "byteSpan": byte_span,
                "dtype": row["dtype"],
                "identity": name,
                "shape": row["shape"],
                "shard": row["shard"],
            },
        })
    require_equal("consumed source identities", len(classified),
                  EXPECTED_TENSORS)
    require_equal("dry-run classifications", counts,
                  EXPECTED_CLASSIFICATIONS)
    require_equal("routed destination component tensors", routed_records, 144)

    report = {
        "entries": entries,
        "input": {
            "contractSha256": CONTRACT_FILE_SHA256,
            "inventoryFileSha256": INVENTORY_FILE_SHA256,
            "tensorInventorySha256": INVENTORY_SHA256,
        },
        "kind": "qwen4exp-conversion-dry-run",
        "profile": {
            "artifactProfileId": QWEN4EXP_EXPERT_ADAPTER.profile_id,
            "codecStatus": "routed-and-PLE-codecs-unselected",
            "expertStoreFamily": QWEN4EXP_EXPERT_ADAPTER.family,
            "repository": HF_REPOSITORY,
            "revision": HF_REVISION,
            "transformersCommit": TRANSFORMERS_COMMIT,
        },
        "schemaVersion": 1,
        "summary": {
            "classifications": counts,
            "owningShards": len(inventory["shardSha256"]),
            "routedDestinationComponentTensors": routed_records,
            "sourceBytes": EXPECTED_SOURCE_BYTES,
            "sourceBytesByClassification": bytes_by_classification,
            "sourceIdentities": len(classified),
        },
    }
    report["reportSha256"] = hashlib.sha256(canonical_json(report)).hexdigest()
    return report


def atomic_write(path: Path, data: bytes) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temp.exists():
        raise ProfileError(f"temporary report path already exists: {temp}")
    fd = os.open(temp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        written = 0
        while written < len(data):
            count = os.write(fd, data[written:])
            if count <= 0:
                raise OSError(f"short write at report byte {written}")
            written += count
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.replace(temp, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        temp.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="explicitly select the only supported header/index-only mode",
    )
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--write", type=Path, metavar="REPORT")
    output.add_argument("--check", type=Path, metavar="REPORT")
    args = parser.parse_args()
    try:
        encoded = canonical_json(
            make_report(args.inventory, args.contract), pretty=True
        )
        if args.write:
            atomic_write(args.write, encoded)
        elif args.check:
            try:
                current = args.check.read_bytes()
            except OSError as exc:
                raise ProfileError(f"cannot read report {args.check}: {exc}") from exc
            if current != encoded:
                raise ProfileError(f"dry-run report drifted: {args.check}")
        sys.stdout.buffer.write(encoded)
        return 0
    except (ProfileError, OSError) as exc:
        print(f"qwen4exp-profile: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
