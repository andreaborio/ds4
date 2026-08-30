#!/usr/bin/env python3
"""Offline validator for the pinned Qwen4Exp profile contract (Phase 0).

The contract at ``docs/contracts/qwen4exp-profile.json`` closes the first
Qwen4Exp profile against one pinned checkpoint revision.  This test never
touches the network, a checkpoint, or Metal: it re-derives every closed field
from the committed source inventory fixture and from the pinned Transformers
algorithms reproduced below, so a reviewer can disagree with any recorded
value using arithmetic instead of prose.

Three groups of evidence are checked:

* ``inventory``  -- digests, totals, classification and the full base-artifact
  tensor name/shape/type table reproduced from the role table.
* ``constants``  -- every derived constant recomputed from ``pinnedConfig``,
  then cross-checked against the tensor shapes that produced it.
* ``ple``        -- the PLE SplitMix multipliers, the 16 n-gram head primes and
  their offsets, padded row count, and the integer-hash safety bounds.

A battery of mutations then re-breaks each closed dimension to prove the
recomputation actually fails.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "docs" / "contracts" / "qwen4exp-profile.json"
FIXTURE_PATH = ROOT / "tests" / "qwen4exp" / "fixtures" / "qwen38flash-next-inventory-v1.json"

# Constants reproduced from the pinned modeling module (see the ADR and the
# pinned Transformers commit recorded in the contract).
MASK64 = (1 << 64) - 1
SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
SPLITMIX_M1 = 0xBF58476D1CE4E5B9
SPLITMIX_M2 = 0x94D049BB133111EB
PRIME_STEP = 10007
MAX_LONG = (1 << 63) - 1

CHECK_NAMES = (
    "identity",
    "source-pins",
    "inventory",
    "tensor-roles",
    "layer-types",
    "derived-constants",
    "ple-hash",
    "ple-extent",
    "expert-store",
    "norm-roles",
    "graph-facts",
    "config-rules",
    "license",
)


# ---------------------------------------------------------------------------
# pinned algorithms
# ---------------------------------------------------------------------------


def splitmix64(value: int) -> int:
    """Mirror of the pinned ``_splitmix64`` (increment, then two multiplies)."""

    value = (value + SPLITMIX_GAMMA) & MASK64
    value = ((value ^ (value >> 30)) * SPLITMIX_M1) & MASK64
    value = ((value ^ (value >> 27)) * SPLITMIX_M2) & MASK64
    return (value ^ (value >> 31)) & MASK64


def build_layer_multipliers(vocab_size: int, ngram_size: int, ple_layer_index: int, seed: int) -> list[int]:
    """Mirror of ``_build_layer_multipliers`` for the pinned PLE hash."""

    multiplier_max = MAX_LONG // max(vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + PRIME_STEP * ple_layer_index
    out: list[int] = []
    for index in range(ngram_size):
        value = (base_seed + SPLITMIX_GAMMA * (index + 1)) & MASK64
        out.append(2 * (splitmix64(value) % half_bound) + 1)
    return out


def is_prime(value: int) -> bool:
    """Mirror of the pinned trial-division ``_is_prime``."""

    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def find_nth_prime_after(start: int, count: int) -> int:
    """Mirror of the pinned ``_find_nth_prime_after``."""

    prime = start
    for _ in range(count):
        prime += 1
        while not is_prime(prime):
            prime += 1
    return prime


def build_head_tables(vocab_base: int, ngram_size: int, heads_per_ngram: int, ple_layer_index: int) -> tuple[list[int], list[int], int]:
    """Mirror of the pinned n-gram head prime and offset construction."""

    ngram_heads = (ngram_size - 1) * heads_per_ngram
    sizes: list[int] = []
    offsets: list[int] = []
    total = 0
    for head_idx in range(ngram_heads):
        global_head_idx = ple_layer_index * ngram_heads + head_idx
        size = find_nth_prime_after(vocab_base - 1, global_head_idx + 1)
        sizes.append(size)
        offsets.append(total)
        total += size
    return sizes, offsets, total


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def digest(tensors: list[dict]) -> str:
    """Recompute the canonical inventory digest over the tensor records."""

    blob = b""
    for row in sorted(tensors, key=lambda item: item["name"]):
        blob += "\0".join(
            (
                row["name"],
                row["dtype"],
                ",".join(str(value) for value in row["shape"]),
                row["shard"],
                str(row["begin"]),
                str(row["end"]),
            )
        ).encode("utf-8") + b"\n"
    return hashlib.sha256(blob).hexdigest()


def text_config(contract: dict) -> dict:
    return contract["pinnedConfig"]["text"]


def base_names(fixture: dict) -> list[str]:
    """Names kept in the base artifact (everything except mtp and vision)."""

    return [row["name"] for row in fixture["tensors"] if classify(fixture, row["name"]) == "base"]


def classify(fixture: dict, name: str) -> str:
    for label, prefix in (("mtp", "mtp."), ("vision", "model.visual.")):
        if name.startswith(prefix):
            return label
    return "base"


def span(fixture: dict, name: str) -> int:
    row = next(item for item in fixture["tensors"] if item["name"] == name)
    return row["end"] - row["begin"]


def dtype_of(fixture: dict, name: str) -> str:
    return next(item for item in fixture["tensors"] if item["name"] == name)["dtype"]


def role_entries(contract: dict) -> list[tuple[str, dict, list[int] | None]]:
    """Expand the role table into (kind, entry, layer list) triples.

    ``kind`` is one of the contract scope names; the layer list gives the
    runtime layer indices a per-layer path is expected at, or ``None`` for the
    global entries.  The 128 n-gram shard tensors carry ``count`` and are
    matched by pattern instead of enumeration.
    """

    roles = contract["tensorRoles"]
    cfg = text_config(contract)
    layer_types = cfg["layer_types"]
    runtime_layers = range(cfg["num_hidden_layers"])
    ple_layer = roles["pleRuntimeLayer"]
    linear = [index for index in runtime_layers if index < len(layer_types) and layer_types[index] == "linear_attention"]
    full = [index for index in runtime_layers if index < len(layer_types) and layer_types[index] == "full_attention"]
    scopes = (
        ("global", roles["global"], None),
        ("all", roles["all"], list(runtime_layers)),
        ("linear_attention", roles["linear_attention"], linear),
        ("full_attention", roles["full_attention"], full),
        ("ple", roles["ple"], [ple_layer]),
    )
    out: list[tuple[str, dict, list[int] | None]] = []
    for scope, entries, layers in scopes:
        for entry in entries:
            out.append((scope, entry, layers))
    return out


def role_paths(contract: dict) -> set[str]:
    """Every role path the contract knows about, with ``layers.N`` retained."""

    return {entry["path"] for _, entry, _ in role_entries(contract)}


def full_name(entry: dict) -> str:
    path = entry["path"]
    leaf = entry.get("leaf")
    return path if leaf is None else path + "." + leaf


def role_shape(contract: dict, path: str) -> tuple[str, tuple[int, ...]]:
    """Dtype and shape recorded for one role path (exact match required)."""

    for _, entry, _ in role_entries(contract):
        if entry["path"] == path:
            return entry["dtype"], tuple(entry["shape"])
    raise KeyError(path)


def ple_pattern(roles: dict) -> re.Pattern[str]:
    prefix = roles["pleLayerPattern"]
    return re.compile(
        "^"
        + re.escape(prefix)
        + r"ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$"
    )


def check_ple_name(roles: dict, name: str) -> bool:
    return name.startswith(roles["pleLayerPattern"])


def eq(errors: list[str], tag: str, label: str, got: object, want: object) -> None:
    if got != want:
        errors.append(f"{tag}: {label}: got {got!r}, closed value {want!r}")


def truthy_list(errors: list[str], tag: str, values: list[str], label: str) -> None:
    if not values:
        errors.append(f"{tag}: {label}: expected at least one rule")
    for value in values:
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{tag}: {label}: empty rule string")


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def check_identity(contract: dict, fixture: dict, errors: list[str]) -> int:
    before = len(errors)
    ident = contract["identity"]
    pins = contract["sourcePins"]
    cfg = contract["pinnedConfig"]
    eq(errors, "[identity.family]", "ggufArchitectureKey vs ggufArchSpelling",
       ident["ggufArchitectureKey"], ident["ggufArchSpelling"])
    eq(errors, "[identity.modelType]", "modelType vs pinnedConfig.topLevel",
       ident["modelType"], cfg["topLevel"]["model_type"])
    eq(errors, "[identity.textModelType]", "textModelType vs pinnedConfig.text",
       ident["textModelType"], text_config(contract)["model_type"])
    eq(errors, "[identity.architecture]", "architecture vs architectures[0]",
       ident["architecture"], cfg["topLevel"]["architectures"][0])
    eq(errors, "[identity.storeFamily]", "expertStoreFamily vs expertStore.family",
       ident["expertStoreFamily"], contract["expertStore"]["family"])
    eq(errors, "[identity.pleMagic]", "pleExtentMagic vs pleExtent.magic",
       ident["pleExtentMagic"], contract["pleExtent"]["magic"])
    eq(errors, "[identity.repo]", "hfRepository vs fixture.repository",
       ident["hfRepository"], fixture["repository"])
    eq(errors, "[identity.revision]", "hfRevision vs fixture.revision",
       ident["hfRevision"], fixture["revision"])
    eq(errors, "[identity.transformers]", "transformersCommit vs fixture",
       pins["transformersCommit"], fixture["transformersCommit"])
    expected = {
        "hebrusFamily": "qwen4exp",
        "ggufArchitectureKey": "qwen4exp",
        "artifactProfileId": "qwen4exp-base-v1",
        "expertStoreFamily": 4,
        "expertStoreFamilyName": "DS4_EXPERT_STORE_FAMILY_QWEN4EXP",
        "pleExtentMagic": "ds4.ple_rows.v1",
    }
    for key, value in expected.items():
        eq(errors, f"[identity.{key}]", "closed Qwen4Exp identity", ident[key], value)
    eq(errors, "[identity.schemaVersion]", "contract schema version", contract["schemaVersion"], 1)
    eq(errors, "[identity.kind]", "contract kind", contract["kind"], "qwen4exp-profile-contract")
    aliases = [alias.lower() for alias in ident["forbiddenAliases"]]
    eq(errors, "[identity.aliasUnique]", "forbidden aliases are unique", len(set(aliases)), len(aliases))
    for alias in ident["forbiddenAliases"]:
        if alias.lower() in {ident["ggufArchSpelling"].lower(), ident["hebrusFamily"].lower()}:
            errors.append(f"[identity.forbiddenAlias]: {alias} collides with the closed identity")
    truthy_list(errors, "[identity.rules]", ident["forbiddenAliases"], "forbidden aliases")
    return len(errors) - before


def check_source_pins(contract: dict, fixture: dict, errors: list[str]) -> int:
    before = len(errors)
    pins = contract["sourcePins"]
    eq(errors, "[sourcePins.fixturePath]", "fixture path is the committed canonical inventory",
       pins["inventoryFixture"], FIXTURE_PATH.relative_to(ROOT).as_posix())
    eq(errors, "[sourcePins.fixtureSchema]", "fixture schema version", fixture["schemaVersion"], 1)
    eq(errors, "[sourcePins.fixtureKind]", "fixture kind", fixture["kind"],
       "qwen38flash-next-source-inventory")
    files = {row["path"]: row for row in fixture["files"]}
    for path, digest_value in sorted(pins["fileSha256"].items()):
        row = files.get(path)
        if row is None:
            errors.append(f"[sourcePins.fileMissing]: {path} is absent from the fixture")
            continue
        eq(errors, f"[sourcePins.fileSha256.{path.replace('.', '_')}]", "digest",
           row["sha256"], digest_value)
    for path in fixture["files"]:
        if path["path"] == "LICENSE":
            eq(errors, "[sourcePins.licenseDigest]", "LICENSE has a recorded digest",
               bool(path["sha256"]), True)
    sources = {row["path"]: row for row in fixture["transformersSources"]}
    eq(errors, "[sourcePins.transformersPaths]", "pinned Transformers source set",
       sorted(sources), sorted(pins["transformersSourceSha256"]))
    for path, digest_value in sorted(pins["transformersSourceSha256"].items()):
        row = sources.get(path)
        if row is None:
            errors.append(f"[sourcePins.transformersMissing]: {path} absent from the fixture")
            continue
        eq(errors, f"[sourcePins.transformers.{Path(path).name}]", "source digest",
           row["sha256"], digest_value)
    eq(errors, "[sourcePins.status]", "contract status stays pinned-not-supported",
       contract["status"], "pinned-not-supported")
    truthy_list(errors, "[sourcePins.rules]", contract["exclusions"]["rules"], "exclusion rules")
    truthy_list(errors, "[sourcePins.gates]", contract["supportGates"], "support gates")
    return len(errors) - before


def check_inventory(contract: dict, fixture: dict, errors: list[str]) -> int:
    before = len(errors)
    pins = contract["sourcePins"]
    tensors = fixture["tensors"]
    names = [row["name"] for row in tensors]
    eq(errors, "[inventory.uniqueNames]", "tensor names are unique",
       len(set(names)), len(names))
    eq(errors, "[inventory.digest]", "recomputed inventory digest vs contract",
       digest(tensors), pins["inventorySha256"])
    eq(errors, "[inventory.digestSelf]", "fixture digest vs recomputed digest",
       fixture["tensorInventorySha256"], digest(tensors))
    total = sum(row["end"] - row["begin"] for row in tensors)
    eq(errors, "[inventory.totalBytes]", "sum of tensor spans vs fixture totalBytes",
       total, fixture["totalBytes"])
    eq(errors, "[inventory.sourceBytes]", "sum of tensor spans vs contract sourceBytes",
       total, pins["sourceBytes"])
    eq(errors, "[inventory.tensorCountFixture]", "fixture tensorCount",
       fixture["tensorCount"], len(tensors))
    eq(errors, "[inventory.tensorCountContract]", "contract tensorCount",
       pins["tensorCount"], len(tensors))
    eq(errors, "[inventory.tensorCountSelf]", "fixture tensorCount vs contract",
       fixture["tensorCount"], pins["tensorCount"])
    bytes_by_dtype: dict[str, int] = {}
    for row in tensors:
        bytes_by_dtype[row["dtype"]] = bytes_by_dtype.get(row["dtype"], 0) + (row["end"] - row["begin"])
    eq(errors, "[inventory.bytesByDtype]", "recomputed bytes by dtype vs contract",
       bytes_by_dtype, pins["bytesByDtype"])
    eq(errors, "[inventory.bytesByDtypeSelf]", "fixture bytes by dtype vs recomputation",
       fixture["bytesByDtype"], bytes_by_dtype)
    eq(errors, "[inventory.bytesSum]", "bytes-by-dtype sum vs totalBytes",
       sum(bytes_by_dtype.values()), total)
    classes: dict[str, int] = {}
    for name in names:
        label = classify(fixture, name)
        classes[label] = classes.get(label, 0) + 1
    ple_prefix = contract["tensorRoles"]["pleLayerPattern"]
    ple = sum(1 for name in names if name.startswith(ple_prefix))
    classes["ple"] = ple
    expected_classes = {k: v for k, v in sorted(classes.items())}
    eq(errors, "[inventory.classification]", "recomputed classification vs contract",
       expected_classes, {k: v for k, v in sorted(pins["classification"].items())})
    eq(errors, "[inventory.classificationSelf]", "fixture classification vs recomputation",
       {k: v for k, v in sorted(fixture["classification"].items())}, expected_classes)
    eq(errors, "[inventory.classificationSum]",
       "base + mtp + vision covers every tensor",
       classes["base"] + classes["mtp"] + classes["vision"], len(tensors))
    eq(errors, "[inventory.pleInBase]", "PLE tensors are inside the base classification",
       True if classes["ple"] <= classes["base"] else False, True)
    shards = sorted({row["shard"] for row in tensors})
    eq(errors, "[inventory.shardCount]", "distinct shards vs contract shardCount",
       len(shards), pins["shardCount"])
    eq(errors, "[inventory.shardCountSelf]", "fixture shardCount vs distinct shards",
       fixture["shardCount"], len(shards))
    eq(errors, "[inventory.shardDigestCount]", "shard digest entries vs distinct shards",
       len(fixture["shardSha256"]), len(shards))
    eq(errors, "[inventory.shardSet]", "shard digest keys cover every tensor shard",
       sorted(fixture["shardSha256"]), shards)
    for path, value in sorted(fixture["shardSha256"].items()):
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value or ""):
            errors.append(f"[inventory.shardDigest.{path.replace('.', '_')}]: not a sha256 hex digest: {value!r}")
    eq(errors, "[inventory.ngramRows]", "n-gram embedding rows vs contract padded rows",
       fixture["ngramEmbeddingRows"], contract["pleHash"]["paddedRows"])
    eq(errors, "[inventory.ngramCols]", "n-gram embedding columns vs PLE head width",
       fixture["ngramEmbeddingCols"], [contract["derivedConstants"]["ple_head_dim"]["value"]])
    ngram_rows = sum(row["shape"][0] for row in tensors if ".ngram_embedding.shard_" in row["name"])
    eq(errors, "[inventory.ngramRowsRecount]", "recounted n-gram shard rows vs fixture",
       ngram_rows, fixture["ngramEmbeddingRows"])
    dtype_width = {"BF16": 2, "I64": 8}
    for row in tensors:
        shape = row["shape"]
        if not shape or any(not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in shape):
            errors.append(f"[inventory.shape.{row['name']}]: dimensions must be positive integers: {shape!r}")
            continue
        if row["dtype"] not in dtype_width:
            errors.append(f"[inventory.dtype.{row['name']}]: unsupported source dtype {row['dtype']!r}")
            continue
        want_span = math.prod(shape) * dtype_width[row["dtype"]]
        eq(errors, f"[inventory.span.{row['name']}]", "byte span follows dtype and shape",
           row["end"] - row["begin"], want_span)
    return len(errors) - before


def check_tensor_roles(contract: dict, fixture: dict, errors: list[str]) -> int:
    before = len(errors)
    roles = contract["tensorRoles"]
    actual: dict[str, tuple[str, tuple[int, ...]]] = {}
    for row in fixture["tensors"]:
        if classify(fixture, row["name"]) != "base":
            continue
        actual[row["name"]] = (row["dtype"], tuple(row["shape"]))
    pattern = ple_pattern(roles)
    expected: dict[str, tuple[str, tuple[int, ...]]] = {}
    pattern_entries: list[dict] = []
    for scope, entry, layers in role_entries(contract):
        name = full_name(entry)
        if entry.get("count"):
            pattern_entries.append(entry)
            continue
        for layer in layers or []:
            expanded = name.replace("layers.N", f"layers.{layer}")
            if "layers.N" not in name and layer is None:
                expanded = name
            key = expanded
            if "layers.N" not in name:
                key = name
            if key in expected:
                errors.append(f"[tensorRoles.duplicate]: {key} is closed twice")
            expected[key] = (entry["dtype"], tuple(entry["shape"]))
        if "layers.N" not in name and layers is None:
            if name in expected:
                errors.append(f"[tensorRoles.duplicate]: {name} is closed twice")
            expected[name] = (entry["dtype"], tuple(entry["shape"]))
    for entry in pattern_entries:
        matched = sorted(n for n in actual if pattern.match(n))
        eq(errors, "[tensorRoles.pleShardCount]", "n-gram shard tensor count",
           len(matched), entry["count"])
        shard_indices = sorted(int(pattern.fullmatch(name).group(1)) for name in matched)
        eq(errors, "[tensorRoles.pleShardIndices]", "n-gram shard indices are contiguous",
           shard_indices, list(range(entry["count"])))
        for name in matched:
            if name in expected:
                errors.append(f"[tensorRoles.pleShardDuplicate]: {name} also closed literally")
            expected[name] = (entry["dtype"], tuple(entry["shape"]))
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing:
        errors.append(
            "[tensorRoles.missing]: base tensors closed but absent from the checkpoint: "
            + ", ".join(missing[:6])
            + (f" (+{len(missing) - 6} more)" if len(missing) > 6 else "")
        )
    if extra:
        errors.append(
            "[tensorRoles.unexpected]: base tensors present but not closed: "
            + ", ".join(extra[:6])
            + (f" (+{len(extra) - 6} more)" if len(extra) > 6 else "")
        )
    for name in sorted(set(expected) & set(actual)):
        eq(errors, f"[tensorRoles.shape.{name}]", "dtype and shape", actual[name], expected[name])
    eq(errors, "[tensorRoles.baseTotal]", "closed base tensor total",
       len(expected), roles["expectedBaseTensors"])
    eq(errors, "[tensorRoles.baseMatchesClassification]", "closed base total vs fixture classification",
       len(expected), contract["sourcePins"]["classification"]["base"])
    eq(errors, "[tensorRoles.expectedBase]", "closed base total vs checkpoint base tensors",
       len(expected), len(actual))
    for label, prefix_key in (("mtp", "mtpPrefix"), ("vision", "visionPrefix")):
        prefix = contract["exclusions"][prefix_key]
        names = [row["name"] for row in fixture["tensors"] if classify(fixture, row["name"]) == label]
        eq(errors, f"[tensorRoles.{label}Count]", f"{label} tensor count",
           len(names), contract["sourcePins"]["classification"][label])
        eq(errors, f"[tensorRoles.{label}ExclusionCount]", f"exclusions.{label}Tensors",
           contract["exclusions"][f"{label}Tensors"], len(names))
        bad = [n for n in names if not n.startswith(prefix)]
        if bad:
            errors.append(f"[tensorRoles.{label}Prefix]: {len(bad)} tensors miss {prefix}")
    mtp_names = [row["name"] for row in fixture["tensors"] if classify(fixture, row["name"]) == "mtp"]
    if any(".linear_attn." in n for n in mtp_names):
        errors.append("[tensorRoles.mtpLinearAttn]: the single MTP layer is full_attention, so GDN keys are unexpected")
    ple_prefix = roles["pleLayerPattern"]
    ple_layers = {n.split(".")[3] for n in actual if n.startswith(ple_prefix)}
    eq(errors, "[tensorRoles.pleLayer]", "PLE tensors appear at exactly one runtime layer",
       sorted(ple_layers), [str(roles["pleRuntimeLayer"])])
    return len(errors) - before


def check_layer_types(contract: dict, fixture: dict, errors: list[str]) -> int:
    before = len(errors)
    cfg = text_config(contract)
    interval = cfg["full_attention_interval"]
    layer_types = cfg["layer_types"]
    eq(errors, "[layerTypes.count]", "layer_types length vs num_hidden_layers",
       len(layer_types), cfg["num_hidden_layers"])
    counts: dict[str, int] = {}
    for index, kind in enumerate(layer_types):
        counts[kind] = counts.get(kind, 0) + 1
        want = "full_attention" if index % interval == interval - 1 else "linear_attention"
        eq(errors, f"[layerTypes.{index}]", "layer type follows the interval rule", kind, want)
    eq(errors, "[layerTypes.linearCount]", "linear_attention layers",
       counts.get("linear_attention", 0),
       sum(1 for i, k in enumerate(layer_types) if i % interval != interval - 1))
    eq(errors, "[layerTypes.fullCount]", "full_attention layers",
       counts.get("full_attention", 0),
       sum(1 for i, k in enumerate(layer_types) if i % interval == interval - 1))
    eq(errors, "[layerTypes.fullMultipleOfInterval]", "full_attention layers match the interval",
       counts.get("full_attention", 0), cfg["num_hidden_layers"] // interval)
    mtp = contract["pinnedConfig"]["mtp"]
    eq(errors, "[layerTypes.mtpCount]", "MTP layer_types length vs MTP layers",
       len(mtp["layer_types"]), mtp["num_hidden_layers"])
    for index, kind in enumerate(mtp["layer_types"]):
        eq(errors, f"[layerTypes.mtp.{index}]", "MTP layer is full_attention", kind, "full_attention")
    return len(errors) - before


def derived_rules(contract: dict, fixture: dict) -> dict[str, object]:
    """Recompute every derived constant from the pinned configuration."""

    cfg = text_config(contract)
    roles = contract["tensorRoles"]
    mtp = contract["pinnedConfig"]["mtp"]
    rope = cfg["rope_parameters"]
    padded = contract["pleHash"]["paddedRows"]
    out: dict[str, object] = {}
    out["wide_size"] = cfg["hidden_size"] * cfg["hc_count"]
    out["rotary_dim"] = int(cfg["head_dim"] * cfg["partial_rotary_factor"])
    out["mrope_section_sum"] = sum(rope["mrope_section"])
    out["attention_query_width"] = cfg["num_attention_heads"] * cfg["head_dim"] * 2
    out["attention_query_only_width"] = cfg["num_attention_heads"] * cfg["head_dim"]
    out["attention_kv_width"] = cfg["num_key_value_heads"] * cfg["head_dim"]
    query_path = "model.language_model.layers.N.self_attn.q_proj"
    out["attention_output_gate"] = (
        role_shape(contract, query_path)[1][0] == 2 * int(out["attention_query_only_width"])
    )
    out["indexer_qk_width"] = (cfg["indexer_n_heads"] + cfg["indexer_kv_heads"]) * cfg["indexer_head_dim"]
    out["indexer_query_width"] = cfg["indexer_n_heads"] * cfg["indexer_head_dim"]
    out["indexer_kv_width"] = cfg["indexer_kv_heads"] * cfg["indexer_head_dim"]
    out["indexer_block_topk"] = cfg["indexer_budget"] // cfg["indexer_compress_ratio"]
    out["indexer_selected_width"] = cfg["indexer_budget"] + cfg["indexer_compress_ratio"] - 1
    out["gdn_repeat_ratio"] = cfg["linear_num_value_heads"] // cfg["linear_num_key_heads"]
    out["gdn_key_width"] = cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"]
    out["gdn_value_width"] = cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"]
    out["gdn_qkv_width"] = 2 * int(out["gdn_key_width"]) + int(out["gdn_value_width"])
    out["gdn_conv_channels"] = int(out["gdn_qkv_width"])
    out["gdn_conv_padding"] = cfg["linear_conv_kernel_dim"] - 1
    out["gdn_gate_dim"] = cfg["linear_value_head_dim"]
    out["gdn_gate_activation"] = cfg["output_gate_type"] or cfg["hidden_act"]
    out["gdn_head_states"] = 2
    out["ple_ngram_heads"] = (cfg["ngram_size"] - 1) * cfg["heads_per_ngram"]
    out["ple_head_dim"] = cfg["ple_embed_dim"] // int(out["ple_ngram_heads"])
    out["ple_flattened_width"] = int(out["ple_ngram_heads"]) * int(out["ple_head_dim"])
    out["ple_key_proj_width"] = int(out["wide_size"])
    out["ple_conv_channels"] = int(out["wide_size"])
    out["ple_conv_dilation"] = cfg["ngram_size"]
    out["ple_conv_state_len"] = (cfg["ple_conv_kernel_size"] - 1) * cfg["ngram_size"]
    out["ple_row_bytes_bf16"] = int(out["ple_head_dim"]) * 2
    out["ple_parameters"] = padded * int(out["ple_head_dim"])
    out["expert_parameters"] = 3 * cfg["hidden_size"] * cfg["moe_intermediate_size"]
    out["expert_gate_up_rows"] = 2 * cfg["moe_intermediate_size"]
    out["routed_parameters_total"] = (
        int(out["expert_parameters"]) * cfg["num_experts"] * cfg["num_hidden_layers"]
    )
    out["active_routed_parameters_per_token"] = (
        int(out["expert_parameters"]) * cfg["num_experts_per_tok"] * cfg["num_hidden_layers"]
    )
    out["shared_expert_parameters"] = 3 * cfg["hidden_size"] * cfg["shared_expert_intermediate_size"]
    out["shared_expert_gate_outputs"] = 1
    out["routed_bytes_bf16"] = int(out["routed_parameters_total"]) * 2
    out["ple_bytes_bf16"] = int(out["ple_parameters"]) * 2
    out["routed_payload_gib_at_4bit"] = round(
        int(out["routed_parameters_total"]) * 4 / 8 / (2**30), 5
    )
    out["routed_payload_gib_at_2bit"] = round(
        int(out["routed_parameters_total"]) * 2 / 8 / (2**30), 5
    )
    del mtp, roles
    return out


def shape_agreement(contract: dict, fixture: dict, derived: dict[str, object]) -> list[str]:
    """Cross-check derived widths against the closed tensor shapes."""

    cfg = text_config(contract)
    errors: list[str] = []
    hidden = cfg["hidden_size"]
    wide = int(derived["wide_size"])
    table: list[tuple[str, tuple[int, ...]]] = [
        ("model.language_model.embed_tokens", (cfg["vocab_size"], hidden)),
        ("model.language_model.hyper_connection_mixer.hc_norm", (wide,)),
        ("model.language_model.hyper_connection_mixer.input_mix_weight_down", (cfg["hc_lowrank"], wide)),
        ("model.language_model.hyper_connection_mixer.input_mix_weight_up", (wide, cfg["hc_lowrank"])),
        ("model.language_model.layers.N.attn_hyper_connection.block_inject_weight", (cfg["hc_count"], wide)),
        ("model.language_model.layers.N.attn_hyper_connection.hc_norm", (wide,)),
        ("model.language_model.layers.N.attn_hyper_connection.input_mix_weight_down", (cfg["hc_lowrank"], wide)),
        ("model.language_model.layers.N.attn_hyper_connection.input_mix_weight_up", (wide, cfg["hc_lowrank"])),
        ("model.language_model.layers.N.mlp_hyper_connection.block_inject_weight", (cfg["hc_count"], wide)),
        ("model.language_model.layers.N.mlp.experts.down_proj", (cfg["num_experts"], hidden, cfg["moe_intermediate_size"])),
        ("model.language_model.layers.N.mlp.experts.gate_up_proj", (cfg["num_experts"], int(derived["expert_gate_up_rows"]), hidden)),
        ("model.language_model.layers.N.mlp.gate", (cfg["num_experts"], hidden)),
        ("model.language_model.layers.N.mlp.shared_expert.down_proj", (hidden, cfg["shared_expert_intermediate_size"])),
        ("model.language_model.layers.N.mlp.shared_expert.gate_proj", (cfg["shared_expert_intermediate_size"], hidden)),
        ("model.language_model.layers.N.mlp.shared_expert.up_proj", (cfg["shared_expert_intermediate_size"], hidden)),
        ("model.language_model.layers.N.mlp.shared_expert_gate", (int(derived["shared_expert_gate_outputs"]), hidden)),
        ("model.language_model.layers.N.linear_attn.A_log", (cfg["linear_num_value_heads"],)),
        ("model.language_model.layers.N.linear_attn.dt_bias", (cfg["linear_num_value_heads"],)),
        ("model.language_model.layers.N.linear_attn.conv1d", (int(derived["gdn_conv_channels"]), 1, cfg["linear_conv_kernel_dim"])),
        ("model.language_model.layers.N.linear_attn.in_proj_a", (cfg["linear_num_value_heads"], hidden)),
        ("model.language_model.layers.N.linear_attn.in_proj_b", (cfg["linear_num_value_heads"], hidden)),
        ("model.language_model.layers.N.linear_attn.in_proj_qkv", (int(derived["gdn_qkv_width"]), hidden)),
        ("model.language_model.layers.N.linear_attn.in_proj_z", (int(derived["gdn_value_width"]), hidden)),
        ("model.language_model.layers.N.linear_attn.norm", (int(derived["gdn_gate_dim"]),)),
        ("model.language_model.layers.N.linear_attn.out_proj", (hidden, int(derived["gdn_value_width"]))),
        ("model.language_model.layers.N.self_attn.q_proj", (int(derived["attention_query_width"]), hidden)),
        ("model.language_model.layers.N.self_attn.k_proj", (int(derived["attention_kv_width"]), hidden)),
        ("model.language_model.layers.N.self_attn.v_proj", (int(derived["attention_kv_width"]), hidden)),
        ("model.language_model.layers.N.self_attn.o_proj", (hidden, int(derived["attention_query_only_width"]))),
        ("model.language_model.layers.N.self_attn.q_norm", (cfg["head_dim"],)),
        ("model.language_model.layers.N.self_attn.k_norm", (cfg["head_dim"],)),
        ("model.language_model.layers.N.self_attn.indexer.index_qk_proj", (int(derived["indexer_qk_width"]), hidden)),
        ("model.language_model.layers.N.self_attn.indexer.q_layernorm", (cfg["indexer_head_dim"],)),
        ("model.language_model.layers.N.self_attn.indexer.k_layernorm", (cfg["indexer_head_dim"],)),
        ("model.language_model.layers.N.ple.key_proj", (int(derived["ple_key_proj_width"]), hidden)),
        ("model.language_model.layers.N.ple.value_proj", (hidden, hidden)),
        ("model.language_model.layers.N.ple.conv1d", (int(derived["ple_conv_channels"]), 1, cfg["ple_conv_kernel_size"])),
        ("model.language_model.layers.N.ple.norm_key", (wide,)),
        ("model.language_model.layers.N.ple.norm_query", (wide,)),
        ("model.language_model.layers.N.ple.norm_conv", (wide,)),
    ]
    for path, want in table:
        dtype, shape = role_shape(contract, path)
        eq(errors, f"[shape.{path.rsplit('.', 1)[-1]}]", "closed shape follows pinned dimensions",
           shape, tuple(want))
        eq(errors, f"[shape.{path.rsplit('.', 1)[-1]}.dtype]", "closed dtype", dtype, "BF16")
    lm_head = role_shape(contract, "lm_head")[1]
    eq(errors, "[shape.lm_head]", "untied head matches the vocabulary", lm_head, (cfg["vocab_size"], hidden))
    ple_buffers = [
        ("model.language_model.layers.N.ple.ple_embedding.layer_multipliers", (cfg["ngram_size"],), "I64"),
        ("model.language_model.layers.N.ple.ple_embedding.ngram_heads_offsets", (int(derived["ple_ngram_heads"]),), "I64"),
        ("model.language_model.layers.N.ple.ple_embedding.ngram_heads_vocab_sizes", (int(derived["ple_ngram_heads"]),), "I64"),
    ]
    for path, want, dtype in ple_buffers:
        got_dtype, got_shape = role_shape(contract, path)
        eq(errors, f"[shape.{path.rsplit('.', 2)[-1]}]", "PLE buffer shape", got_shape, tuple(want))
        eq(errors, f"[shape.{path.rsplit('.', 2)[-1]}.dtype]", "PLE buffer dtype", got_dtype, dtype)
    return errors


def check_derived(contract: dict, fixture: dict, errors: list[str]) -> int:
    before = len(errors)
    derived = derived_rules(contract, fixture)
    closed = contract["derivedConstants"]
    eq(errors, "[derived.ruleSet]", "every derived constant has a recomputation rule",
       sorted(closed), sorted(derived))
    for name, rule in sorted(derived.items()):
        entry = closed.get(name)
        if entry is None:
            errors.append(f"[derived.missing.{name.replace('_old', '')}]: {name} is closed without a value")
            continue
        eq(errors, f"[derived.{name}]", "recomputed value", entry["value"], rule)
        formula = entry["formula"]
        if not isinstance(formula, str) or not formula.strip():
            errors.append(f"[derived.{name}.formula]: formula is empty")
    errors.extend(shape_agreement(contract, fixture, derived))
    return len(errors) - before


def check_ple_hash(contract: dict, fixture: dict, errors: list[str]) -> int:
    before = len(errors)
    cfg = text_config(contract)
    hash_doc = contract["pleHash"]
    buffers = fixture["pleHashBuffers"]
    prefix = f"model.language_model.layers.{contract['tensorRoles']['pleRuntimeLayer']}.ple.ple_embedding."
    ngram_heads = int(derived_rules(contract, fixture)["ple_ngram_heads"])
    ple_layer_index = hash_doc["pleLayerIndex"]
    seed = hash_doc["seed"]
    eq(errors, "[ple.seed]", "seed matches the pinned config default",
       seed, contract["pinnedConfig"]["defaultsUsed"]["seed"])
    if not 0 <= ple_layer_index < len(cfg["ple_layer_ids"]):
        errors.append(
            f"[ple.layerIndex]: PLE ordinal {ple_layer_index} is outside the configured PLE layer list"
        )
    else:
        eq(errors, "[ple.layerIndex]", "PLE ordinal maps to the closed runtime layer",
           cfg["ple_layer_ids"][ple_layer_index] - 1, contract["tensorRoles"]["pleRuntimeLayer"])
    multipliers = build_layer_multipliers(cfg["vocab_size"], cfg["ngram_size"], ple_layer_index, seed)
    eq(errors, "[ple.multipliers]", "recomputed SplitMix multipliers",
       hash_doc["layerMultipliers"], multipliers)
    eq(errors, "[ple.multipliersBuffer]", "checkpoint buffer matches the recomputation",
       buffers[prefix + "layer_multipliers"], multipliers)
    sizes, offsets, total = build_head_tables(cfg["ngram_vocab_size_base"], cfg["ngram_size"], cfg["heads_per_ngram"], ple_layer_index)
    eq(errors, "[ple.headPrimes]", "recomputed head primes", hash_doc["headPrimes"], sizes)
    eq(errors, "[ple.headOffsets]", "recomputed head offsets", hash_doc["headOffsets"], offsets)
    eq(errors, "[ple.headPrimesBuffer]", "checkpoint head-size buffer matches the recomputation",
       buffers[prefix + "ngram_heads_vocab_sizes"], sizes)
    eq(errors, "[ple.headOffsetsBuffer]", "checkpoint offset buffer matches the recomputation",
       buffers[prefix + "ngram_heads_offsets"], offsets)
    eq(errors, "[ple.headCount]", "head arrays length", len(sizes), ngram_heads)
    for index, value in enumerate(sizes):
        if not is_prime(value):
            errors.append(f"[ple.prality.{index}]: {value} is not prime")
        if value % 2 == 0 and value != 2:
            errors.append(f"[ple.prality.{index}]: even head prime {value}")
        eq(errors, f"[ple.prime.{index}]", "prime is the (index+1)-th after the base",
           value, find_nth_prime_after(cfg["ngram_vocab_size_base"] - 1, index + 1))
        if index and value <= sizes[index - 1]:
            errors.append(f"[ple.primeOrder.{index}]: primes are not increasing")
    for index, value in enumerate(offsets):
        want = sum(sizes[:index])
        eq(errors, f"[ple.offset.{index}]", "offset is the prefix sum of head primes", value, want)
    eq(errors, "[ple.offsetZero]", "first head offset is zero", offsets[0], 0)
    eq(errors, "[ple.totalHeadRows]", "sum of head primes vs closed total head rows",
       sum(sizes), hash_doc["totalHeadRows"])
    divisor = cfg["make_ngram_vocab_size_divisible_by"]
    padded = math.ceil(sum(sizes) / divisor) * divisor
    eq(errors, "[ple.paddedRows]", "recomputed padded rows", hash_doc["paddedRows"], padded)
    eq(errors, "[ple.paddedMultiple]", "padded rows are divisible by the configured divisor",
       padded % divisor, 0)
    eq(errors, "[ple.padHeadroom]", "padding headroom is below one divisor",
       padded - sum(sizes), padded - sum(sizes) if padded - sum(sizes) < divisor else None)
    multiplier_max = MAX_LONG // cfg["vocab_size"]
    for index, value in enumerate(multipliers):
        if value % 2 == 0:
            errors.append(f"[ple.multiplierOdd.{index}]: {value} is even, the hash needs odd multipliers")
        if value >= multiplier_max:
            errors.append(f"[ple.multiplierBound.{index}]: {value} reaches max_long // vocab_size")
    max_product = (cfg["vocab_size"] - 1) * max(multipliers)
    eq(errors, "[ple.maxProduct]", "recomputed largest hash product",
       hash_doc["maxHashProduct"], max_product)
    eq(errors, "[ple.productBound]", "largest product stays below 2**63 - 1",
       max_product < MAX_LONG, True)
    eq(errors, "[ple.productBit63]", "bit 63 of the largest product stays clear",
       max_product & (1 << 63), 0)
    for index, value in enumerate(multipliers):
        signed = MAX_LONG % value
        unsigned = (MAX_LONG & MASK64) % value
        eq(errors, f"[ple.remainder.{index}]", "signed and unsigned remainders agree", signed, unsigned)
        for token in (0, 1, cfg["vocab_size"] - 1, cfg["eos_token_id"]):
            fold = token * value
            if fold >> 63:
                errors.append(f"[ple.foldBit63.{index}]: token {token} * multiplier sets bit 63")
            eq(errors, f"[ple.foldRemainder.{index}.{token}]", "fold remainder is positive",
               fold % value, (fold & MASK64) % value)
    truthy_list(errors, "[ple.hashSafety]", hash_doc["hashSafety"], "hash safety notes")
    derivation = hash_doc["derivation"]
    eq(errors, "[ple.derivation.gamma]", "SplitMix gamma constant",
       int(derivation["splitmix_gamma"], 16), SPLITMIX_GAMMA)
    eq(errors, "[ple.derivation.m1]", "SplitMix m1 constant",
       int(derivation["splitmix_m1"], 16), SPLITMIX_M1)
    eq(errors, "[ple.derivation.m2]", "SplitMix m2 constant",
       int(derivation["splitmix_m2"], 16), SPLITMIX_M2)
    eq(errors, "[ple.derivation.primeStep]", "prime step constant", derivation["prime_step"], PRIME_STEP)
    eq(errors, "[ple.derivation.maxLong]", "max_long constant", derivation["max_long"], MAX_LONG)
    eq(errors, "[ple.derivation.mask64]", "mask64 constant",
       int(derivation["mask64"], 16), MASK64)
    return len(errors) - before


def check_ple_extent(contract: dict, fixture: dict, errors: list[str]) -> int:
    before = len(errors)
    extent = contract["pleExtent"]
    roles = contract["tensorRoles"]
    derived = derived_rules(contract, fixture)
    ple_prefix = roles["pleLayerPattern"]
    cfg = text_config(contract)
    hash_doc = contract["pleHash"]
    row_alignment = extent["rowAlignment"]
    head_rows = extent["headRows"]
    padded_rows = (head_rows + row_alignment - 1) // row_alignment * row_alignment
    wire = extent["wireFormat"]
    eq(errors, "[pleExtent.rows]", "extent rows vs recomputed padded rows",
       extent["rows"], derived and hash_doc["paddedRows"])
    eq(errors, "[pleExtent.rowWidth]", "extent row width vs PLE head width",
       extent["rowWidth"], derived["ple_head_dim"])
    eq(errors, "[pleExtent.version]", "PLE wire version", extent["version"], 1)
    eq(errors, "[pleExtent.family]", "PLE family vs Qwen4Exp identity",
       extent["family"], contract["identity"]["expertStoreFamily"])
    eq(errors, "[pleExtent.profile]", "PLE profile vs artifact identity",
       extent["profileId"], contract["identity"]["artifactProfileId"])
    eq(errors, "[pleExtent.hashId]", "closed PLE hash identity",
       extent["hashId"], "SplitMix64-Qwen4Exp-v1")
    eq(errors, "[pleExtent.headRows]", "logical head rows vs PLE hash",
       head_rows, hash_doc["totalHeadRows"])
    eq(errors, "[pleExtent.rowAlignment]", "row alignment vs pinned divisor",
       row_alignment, cfg["make_ngram_vocab_size_divisible_by"])
    eq(errors, "[pleExtent.rowAlignmentExact]", "closed row alignment",
       row_alignment, 128)
    eq(errors, "[pleExtent.alignedRows]", "rows are align_up(head rows, 128)",
       extent["rows"], padded_rows)
    eq(errors, "[pleExtent.paddingRows]", "padding rows vs aligned head rows",
       extent["paddingRows"], extent["rows"] - head_rows)
    eq(errors, "[pleExtent.paddingRowsExact]", "closed padding row count",
       extent["paddingRows"], 90)
    eq(errors, "[pleExtent.rowsAligned]", "padded rows are row-aligned",
       extent["rows"] % row_alignment, 0)
    eq(errors, "[pleExtent.paddingBound]", "padding is below one alignment unit",
       0 <= extent["paddingRows"] < row_alignment, True)
    eq(errors, "[pleExtent.shards]", "extent shard count vs split_ngram_parts",
       extent["embeddingShardCount"], text_config(contract)["split_ngram_parts"])
    eq(errors, "[pleExtent.shardRows]", "shard rows times shard count covers the extent",
       extent["embeddingShardRows"] * extent["embeddingShardCount"], extent["rows"])
    eq(errors, "[pleExtent.shardRowsDerived]", "shard rows vs padded rows divided by parts",
       extent["embeddingShardRows"], contract["pleHash"]["paddedRows"] // extent["embeddingShardCount"])
    shard_names = [row["name"] for row in fixture["tensors"] if ".ngram_embedding.shard_" in row["name"]]
    eq(errors, "[pleExtent.shardTensors]", "checkpoint n-gram shard tensor count",
       len(shard_names), extent["embeddingShardCount"])
    shard_rows = {row["shape"][0] for row in fixture["tensors"] if ".ngram_embedding.shard_" in row["name"]}
    eq(errors, "[pleExtent.shardShapeRows]", "every shard stores the same row count",
       sorted(shard_rows), [extent["embeddingShardRows"]])
    embedding_names = [
        n for n in base_names(fixture)
        if n.startswith(ple_prefix) and ".ngram_embedding.shard_" in n
    ]
    ple_bytes = sum(span(fixture, n) for n in embedding_names)
    buffer_bytes = sum(
        span(fixture, n) for n in base_names(fixture) if n.startswith(ple_prefix) and dtype_of(fixture, n) == "I64"
    )
    eq(errors, "[pleExtent.bytes]", "embedding payload bytes inside the checkpoint",
       ple_bytes, derived["ple_bytes_bf16"])
    eq(errors, "[pleExtent.bufferBytes]", "hash buffer bytes",
       buffer_bytes, (text_config(contract)["ngram_size"] + 2 * int(derived["ple_ngram_heads"])) * 8)
    eq(errors, "[pleExtent.embeddingBytesFromGeometry]", "extent geometry matches checkpoint bytes",
       extent["rows"] * extent["rowWidth"] * 2, ple_bytes)
    eq(errors, "[pleExtent.magic]", "extent magic matches the closed identity",
       extent["magic"], contract["identity"]["pleExtentMagic"])
    expected_wire = {
        "endianness": "little",
        "headerBytes": 512,
        "pageHeaderBytes": 64,
        "minimumPageAlignment": 4096,
        "pageDigestAlgorithm": "SHA-256",
        "pageDigestBytes": 32,
        "pageDigestTable": "one digest per fixed page; no per-row index",
        "manifestDigest": "SHA-256(header with manifest digest zeroed || page-digest table || alignment padding)",
        "payloadDigest": "SHA-256 over every complete fixed-stride physical page in order",
        "codecBinding": "caller supplies one exact codec id/version/group-size/encoded-row-bytes descriptor",
    }
    eq(errors, "[pleExtent.wire.closed]", "closed PLE v1 wire format",
       wire, expected_wire)
    eq(errors, "[pleExtent.wire.digestBytes]", "SHA-256 digest width",
       wire["pageDigestBytes"], hashlib.sha256().digest_size)
    eq(errors, "[pleExtent.wire.headerPageRatio]", "page header divides manifest header",
       wire["headerBytes"] % wire["pageHeaderBytes"], 0)
    eq(errors, "[pleExtent.wire.pageAlignment]", "closed minimum page alignment",
       wire["minimumPageAlignment"], 4096)
    eq(errors, "[pleExtent.wire.alignmentPowerOfTwo]", "page alignment is a power of two",
       wire["minimumPageAlignment"] & (wire["minimumPageAlignment"] - 1), 0)
    expected_pending = [
        "the production codec id, group size and encoded row bytes require quality and Metal qualification",
        "production rows_per_page, page alignment and derived page_stride are frozen by the codec-specific artifact profile",
    ]
    eq(errors, "[pleExtent.pendingClosed]", "codec/page decisions remain explicitly pending",
       extent["pendingDecisions"], expected_pending)
    truthy_list(errors, "[pleExtent.rules]", extent["rules"], "extent rules")
    truthy_list(errors, "[pleExtent.pending]", extent["pendingDecisions"], "pending decisions")
    return len(errors) - before


def check_expert_store(contract: dict, fixture: dict, errors: list[str]) -> int:
    before = len(errors)
    store = contract["expertStore"]
    cfg = text_config(contract)
    derived = derived_rules(contract, fixture)
    eq(errors, "[expert.layers]", "expert store layers vs num_hidden_layers",
       store["layers"], cfg["num_hidden_layers"])
    eq(errors, "[expert.perLayer]", "experts per layer vs num_experts",
       store["expertsPerLayer"], cfg["num_experts"])
    eq(errors, "[expert.familyIdentity]", "store family matches the closed family name",
       store["family"], contract["identity"]["expertStoreFamily"])
    eq(errors, "[expert.version]", "store version", store["version"], 2)
    eq(errors, "[expert.tensor]", "store tensor name", store["tensor"], "ds4.expert_major.v2")
    eq(errors, "[expert.components]", "three routed components", len(store["components"]), 3)
    eq(errors, "[expert.componentNames]", "component names", store["components"], ["gate", "up", "down"])
    eq(errors, "[expert.routeRecords]", "minimum route records per token",
       store["minimumRouteRecordsPerToken"], cfg["num_hidden_layers"] * cfg["num_experts_per_tok"])
    raise_doc = store["raiseMaxExperts"]
    eq(errors, "[expert.max512]", "Qwen4Exp closes the structural maximum at 512",
       store["expertsPerLayer"], 512)
    eq(errors, "[expert.raise.field]", "shared C admission field",
       raise_doc["field"], "DS4_EXPERT_STORE_V2_MAX_EXPERTS")
    eq(errors, "[expert.raise.from]", "prior shared expert cap", raise_doc["from"], 384)
    eq(errors, "[expert.raise.to]", "raised expert cap equals num_experts", raise_doc["to"], cfg["num_experts"])
    eq(errors, "[expert.raise.phase]", "expert-cap raise is owned by Phase 2",
       raise_doc["phase"], 2)
    if not isinstance(raise_doc["note"], str) or not raise_doc["note"].strip():
        errors.append("[expert.raise.note]: admission-only note must be non-empty")
    if raise_doc["from"] >= raise_doc["to"]:
        errors.append("[expert.raise.from]: the current cap must be below the new cap")
    candidate = store["structuralCandidate"]
    eq(errors, "[expert.candidate.keys]", "closed structural candidate fields",
       set(candidate), {"storage", "logicalGgmlType", "groupSize",
                        "blockBytes", "status", "reason"})
    eq(errors, "[expert.candidate.storage]", "structural storage",
       candidate["storage"], "mlx-affine4")
    eq(errors, "[expert.candidate.logicalType]", "logical GGML descriptor type",
       candidate["logicalGgmlType"], "Q4_K")
    eq(errors, "[expert.candidate.group]", "affine group size",
       candidate["groupSize"], 64)
    eq(errors, "[expert.candidate.blockBytes]", "affine physical block bytes",
       candidate["blockBytes"], 36)
    eq(errors, "[expert.candidate.blockFormula]", "4-bit group plus BF16 scale and bias",
       candidate["blockBytes"], candidate["groupSize"] // 2 + 4)
    eq(errors, "[expert.candidate.status]", "candidate remains non-release-qualified",
       candidate["status"], "phase2-structural-not-release-qualified")
    if not isinstance(candidate["reason"], str) or "640" not in candidate["reason"]:
        errors.append("[expert.candidate.reason]: reason must name the 640-wide down row")
    routed = sum(
        span(fixture, n) for n in base_names(fixture)
        if ".mlp.experts." in n and not n.startswith(contract["tensorRoles"]["pleLayerPattern"])
    )
    eq(errors, "[expert.routedBytes]", "checkpoint routed payload bytes", routed, derived["routed_bytes_bf16"])
    shared = sum(span(fixture, n) for n in base_names(fixture) if ".mlp.shared_expert." in n)
    eq(errors, "[expert.sharedBytes]", "checkpoint shared expert bytes",
       shared, int(derived["shared_expert_parameters"]) * 2 * cfg["num_hidden_layers"])
    for name, bits in (("routed_payload_gib_at_4bit", 4), ("routed_payload_gib_at_2bit", 2)):
        recomputed = int(derived["routed_parameters_total"]) * bits / 8 / (2**30)
        closed_value = float(contract["derivedConstants"][name]["value"])
        if abs(recomputed - closed_value) > 1e-4:
            errors.append(f"[expert.{name}]: closed GiB {closed_value} != recomputed {recomputed:.5f}")
    return len(errors) - before


def check_norm_roles(contract: dict, fixture: dict, errors: list[str]) -> int:
    before = len(errors)
    conventions = contract["normConventions"]
    zero = set(conventions["zeroCentered"])
    conventional = set(conventions["conventional"])
    eq(errors, "[norms.disjoint]", "a norm role cannot be both conventions",
       len(zero & conventional), 0)
    paths = role_paths(contract)
    mtp_names = {
        tensor["name"].rsplit(".", 1)[0]
        for tensor in fixture["tensors"]
        if classify(fixture, tensor["name"]) == "mtp"
    }
    for pattern in sorted(zero | conventional):
        if pattern in paths:
            continue
        tail = pattern.replace("layers.N", "layers.0")
        if any(tail == n for n in mtp_names):
            continue
        errors.append(f"[norms.unknownRole]: {pattern} is not a closed tensor role")
    norms = {path for path in paths if re.search(r"(^|\.)(hc_)?(q_|k_)?norm(_[a-z]+)?$", path.split(".")[-1])}
    norms |= {path for path in paths if path.split(".")[-1].endswith("layernorm")}
    norms |= {path for path in paths if path.split(".")[-1] == "norm"}
    unclassified = sorted(norms - zero - conventional)
    if unclassified:
        errors.append("[norms.unclassified]: norm roles with no recorded convention: " + ", ".join(unclassified))
    eq(errors, "[norms.gatedConvention]", "GDN output norm uses the conventional convention",
       "model.language_model.layers.N.linear_attn.norm" in conventional, True)
    eq(errors, "[norms.gatedNotZero]", "GDN output norm is not zero-centered",
       "model.language_model.layers.N.linear_attn.norm" in zero, False)
    cfg = text_config(contract)
    eq(errors, "[norms.gatedActivation]", "conventional formula matches output_gate_type",
       conventions["formulaConventional"].count("sigmoid"), 1 if cfg["output_gate_type"] == "sigmoid" else 0)
    eq(errors, "[norms.zeroFormula]", "zero-centered formula applies (1 + weight)",
       "(1 + w)" in conventions["formulaZeroCentered"], True)
    eq(errors, "[norms.gatedFormula]", "conventional formula applies the weight then gates",
       "* w * sigmoid(g)" in conventions["formulaConventional"], True)
    eq(errors, "[norms.eps]", "rms_norm_eps is closed", cfg["rms_norm_eps"], 1e-06)
    truthy_list(errors, "[norms.rules]", conventions["rules"], "norm rules")
    return len(errors) - before


def check_graph_facts(contract: dict, fixture: dict, errors: list[str]) -> int:
    before = len(errors)
    facts = contract["graphFacts"]
    roles = contract["tensorRoles"]
    cfg = text_config(contract)
    names = base_names(fixture)
    norm_names = [n for n in names if re.fullmatch(r"model\.language_model\.(layers\.\d+\.)?norm\.weight", n)]
    eq(errors, "[graph.finalNormAbsence]", "no final model norm exists in the checkpoint",
       len(norm_names), 0)
    eq(errors, "[graph.finalNormFact]", "closed final-norm fact matches the checkpoint",
       facts["finalNorm"]["present"], bool(norm_names))
    inject = [n for n in names if ".block_inject_weight.weight" in n]
    eq(errors, "[graph.injectCount]", "block inject weights appear in both GR blocks of every layer",
       len(inject), 2 * cfg["num_hidden_layers"])
    eq(errors, "[graph.injectFact]", "closed injection fact matches the checkpoint",
       facts["perLayerGatedResidual"]["hasBlockInjectWeight"], bool(inject))
    final_inject = [n for n in inject if "hyper_connection_mixer" in n]
    eq(errors, "[graph.finalInject]", "the final mixer has no injection weight",
       len(final_inject), 0)
    eq(errors, "[graph.injectShape]", "injection weight shape",
       role_shape(contract, "model.language_model.layers.N.attn_hyper_connection.block_inject_weight")[1],
       (cfg["hc_count"], int(derived_rules(contract, fixture)["wide_size"])))
    eq(errors, "[graph.injectShapeFact]", "closed injection shape matches the role table",
       tuple(facts["perLayerGatedResidual"]["injectWeightShape"]),
       role_shape(contract, "model.language_model.layers.N.attn_hyper_connection.block_inject_weight")[1])
    gr_role_paths = role_paths(contract)
    eq(errors, "[graph.separateResidualBlocks]", "attention and MLP use distinct residual blocks",
       facts["perLayerGatedResidual"]["attnAndMlpSeparate"],
       all(
           f"model.language_model.layers.N.{block}_hyper_connection.block_inject_weight" in gr_role_paths
           for block in ("attn", "mlp")
       ))
    eq(errors, "[graph.outputGate]", "query projection includes the attention output gate",
       facts["attention"]["queryIncludesOutputGate"],
       role_shape(contract, "model.language_model.layers.N.self_attn.q_proj")[1][0]
       == 2 * int(derived_rules(contract, fixture)["attention_query_only_width"]))
    eq(errors, "[graph.outputGateWidth]", "query width is twice the head count times head dim",
       role_shape(contract, "model.language_model.layers.N.self_attn.q_proj")[1][0],
       cfg["num_attention_heads"] * cfg["head_dim"] * 2)
    eq(errors, "[graph.qkNorm]", "per-head query and key norms are closed",
       facts["attention"]["perHeadQkNorm"],
       all(
           path in gr_role_paths
           for path in (
               "model.language_model.layers.N.self_attn.q_norm",
               "model.language_model.layers.N.self_attn.k_norm",
           )
       ))
    eq(errors, "[graph.mropeRows]", "text positions carry text plus three M-RoPE rows",
       facts["attention"]["mropePositionRowsForText"], 4)
    qkv = role_shape(contract, "model.language_model.layers.N.linear_attn.in_proj_qkv")[1][0]
    derived = derived_rules(contract, fixture)
    eq(errors, "[graph.gdnSharedQK]", "GDN query and key share one projection width",
       qkv == 2 * int(derived["gdn_key_width"]) + int(derived["gdn_value_width"]),
       facts["gdn"]["queryAndKeyShareWidth"])
    eq(errors, "[graph.gdnSeparateZ]", "GDN keeps a separate value gate projection",
       any(True for path in roles["linear_attention"] if path["path"].endswith("in_proj_z")),
       facts["gdn"]["separateZProjection"])
    conv_role_present = any(".conv1d.weight" in full_name(path) for path in roles["linear_attention"])
    conv_names = [n for n in names if ".linear_attn.conv1d.weight" in n]
    has_bias = any(n.endswith("conv1d.bias") for n in fixture_names(fixture))
    eq(errors, "[graph.gdnNoConvBias]", "GDN convolution is bias-free",
       facts["gdn"]["biasFreeConv"], conv_role_present and not has_bias)
    eq(errors, "[graph.gdnConvBiasAbsent]", "no conv1d bias appears in the checkpoint", has_bias, False)
    a_logs = [n for n in names if n.endswith(".linear_attn.A_log")]
    dt_biases = [n for n in names if n.endswith(".linear_attn.dt_bias")]
    eq(errors, "[graph.gdnAlog]", "per-value-head A_log and dt_bias are closed",
       facts["gdn"]["perValueHeadAlogAndDtBias"],
       len(a_logs) == len(conv_names) == len(dt_biases) and bool(conv_names))
    eq(errors, "[graph.fusedGateUp]", "routed experts store a fused gate/up tensor",
       facts["moe"]["fusedGateUpTensor"],
       any(full_name(path).endswith("experts.gate_up_proj") for path in roles["all"]))
    eq(errors, "[graph.fusedDown]", "routed experts store a fused down tensor",
       facts["moe"]["fusedDownTensor"],
       any(full_name(path).endswith("experts.down_proj") for path in roles["all"]))
    eq(errors, "[graph.sharedGate]", "the shared expert keeps a sigmoid gate tensor",
       facts["moe"]["sharedExpertSigmoidGate"],
       any(".shared_expert_gate.weight" in n for n in names))
    eq(errors, "[graph.routerFloat32]", "router softmax is fixed to float32 by the pinned source",
       facts["moe"]["routerSoftmaxInFloat32"], True)
    eq(errors, "[graph.indexerUnderAttention]", "QSA indexer tensors live under self_attn",
       any(".self_attn.indexer." in n for n in names), facts["attention"]["indexerUnderSelfAttn"])
    eq(errors, "[graph.indexerNotUnderPle]", "PLE does not own an indexer subtree",
       any(".ple.indexer." in n for n in names), False)
    padding = facts["embeddingPaddingRow"]
    probe = fixture["paddingRowFacts"]
    eq(errors, "[graph.paddingFactToken]", "closed padding token matches the probe",
       padding["token"], probe["token"])
    eq(errors, "[graph.paddingToken]", "padding probe token is the closed shared bos/eos/pad id",
       probe["token"], cfg["eos_token_id"])
    eq(errors, "[graph.paddingTokenBos]", "bos_token_id equals the shared padding id",
       cfg["bos_token_id"], cfg["eos_token_id"])
    eq(errors, "[graph.paddingRowBytes]", "probe row bytes",
       probe["rowBytes"], cfg["hidden_size"] * 2)
    eq(errors, "[graph.paddingEmbedZero]", "embedding padding-row fact matches the probe",
       padding["embedAllZero"], probe["embedZero"])
    lm = next(row for row in probe["checked"] if row["name"] == "lm_head.weight")
    eq(errors, "[graph.paddingLmHeadZero]", "head padding-row fact matches the probe",
       padding["lmHeadAllZero"], lm["allZero"])
    eq(errors, "[graph.paddingProbeNames]", "both token tables were probed",
       sorted(row["name"] for row in probe["checked"]),
       sorted(["lm_head.weight", "model.language_model.embed_tokens.weight"]))
    truthy_list(errors, "[graph.evidence]", facts["finalNorm"]["evidence"], "final-norm evidence")
    truthy_list(errors, "[graph.paddingEvidence]", facts["embeddingPaddingRow"]["evidence"], "padding-row evidence")
    return len(errors) - before


def fixture_names(fixture: dict) -> list[str]:
    return [row["name"] for row in fixture["tensors"]]


def check_config_rules(contract: dict, fixture: dict, errors: list[str]) -> int:
    """Pinned ``Qwen4ExpTextConfig`` validation rules applied to the contract."""

    before = len(errors)
    cfg = text_config(contract)
    layers = cfg["layer_types"]
    ple_ids = cfg["ple_layer_ids"]
    eq(errors, "[configRules.pleCount]", "the contract closes exactly the pinned PLE layer count",
       len(ple_ids), 1)
    for value in ple_ids:
        if not 1 <= value <= cfg["num_hidden_layers"]:
            errors.append(f"[configRules.pleRange]: one-based PLE layer {value} is out of range")
        kind = layers[value - 1]
        eq(errors, f"[configRules.pleType.{value}]", "a one-based PLE layer is a linear_attention layer",
           kind, "linear_attention")
    eq(errors, "[configRules.pleRuntimeLayer]", "one-based ple_layer_ids maps to the runtime layer",
       ple_ids[0] - 1, contract["tensorRoles"]["pleRuntimeLayer"])
    eq(errors, "[configRules.ngramDivisor]", "divisor is a positive multiple of 2",
       cfg["make_ngram_vocab_size_divisible_by"] % 2, 0)
    eq(errors, "[configRules.interval]", "full attention interval is positive",
       cfg["full_attention_interval"] > 0, True)
    eq(errors, "[configRules.untied]", "the pinned config keeps input and output embeddings untied",
       cfg["tie_word_embeddings"], False)
    eq(errors, "[configRules.untiedTensors]", "both token tables are present",
       len([n for n in base_names(fixture) if n in ("lm_head.weight", "model.language_model.embed_tokens.weight")]), 2)
    eq(errors, "[configRules.attentionBias]", "projections keep the pinned biasless form",
       cfg["attention_bias"], False)
    eq(errors, "[configRules.headDivisible]", "GDN value heads divide key heads",
       cfg["linear_num_value_heads"] % cfg["linear_num_key_heads"], 0)
    eq(errors, "[configRules.embedDivisible]", "PLE embedding dim divides n-gram heads",
       cfg["ple_embed_dim"] % int(derived_rules(contract, fixture)["ple_ngram_heads"]), 0)
    eq(errors, "[configRules.indexerBudget]", "indexer budget divides the compress ratio",
       cfg["indexer_budget"] % cfg["indexer_compress_ratio"], 0)
    eq(errors, "[configRules.rotaryDivisible]", "rotary dim is even for half-rotation",
       int(derived_rules(contract, fixture)["rotary_dim"]) % 2, 0)
    eq(errors, "[configRules.mropeSum]", "mrope sections cover half the rotary width",
       sum(cfg["rope_parameters"]["mrope_section"]),
       int(derived_rules(contract, fixture)["rotary_dim"]) // 2)
    eq(errors, "[configRules.mropeSections]", "M-RoPE section allocation stays pinned",
       cfg["rope_parameters"]["mrope_section"], [11, 11, 10])
    eq(errors, "[configRules.ropeType]", "rope type stays the pinned default",
       cfg["rope_parameters"]["rope_type"], "default")
    eq(errors, "[configRules.mtpDedicated]", "MTP does not use dedicated embeddings in the pinned config",
       cfg["mtp_use_dedicated_embeddings"], False)
    return len(errors) - before


def check_license(contract: dict, fixture: dict, errors: list[str]) -> int:
    before = len(errors)
    license_doc = contract["license"]
    eq(errors, "[license.notApache]", "Apache-2.0 is not assumed", license_doc["assumedApache2"], False)
    eq(errors, "[license.reviewRequired]", "review is required before distribution",
       license_doc["reviewRequiredBeforeDistribution"], True)
    eq(errors, "[license.pendingAtPhase0]", "Phase 0 records the review as pending",
       license_doc["reviewStatus"], "pending")
    eq(errors, "[license.statusAlignment]", "an unresolved review keeps the profile unsupported",
       contract["status"] == "pinned-not-supported" and license_doc["reviewStatus"] == "pending", True)
    has_license = any(row["path"] == "LICENSE" and row["sha256"] for row in fixture["files"])
    eq(errors, "[license.digestRecorded]", "the pinned LICENSE digest is recorded", has_license, True)
    return len(errors) - before


CHECKS = {
    "identity": check_identity,
    "source-pins": check_source_pins,
    "inventory": check_inventory,
    "tensor-roles": check_tensor_roles,
    "layer-types": check_layer_types,
    "derived-constants": check_derived,
    "ple-hash": check_ple_hash,
    "ple-extent": check_ple_extent,
    "expert-store": check_expert_store,
    "norm-roles": check_norm_roles,
    "graph-facts": check_graph_facts,
    "config-rules": check_config_rules,
    "license": check_license,
}


def validate(contract: dict, fixture: dict, only: list[str] | None = None) -> list[str]:
    errors: list[str] = []
    for name in only or CHECK_NAMES:
        try:
            CHECKS[name](contract, fixture, errors)
        except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError) as exc:
            errors.append(
                f"[validator.{name}]: malformed contract or fixture: "
                f"{type(exc).__name__}: {exc}"
            )
    return errors


# ---------------------------------------------------------------------------
# mutation battery
# ---------------------------------------------------------------------------


def set_path(document: dict, path: tuple, value: object) -> None:
    node = document
    for key in path[:-1]:
        node = node[key]
    key = path[-1]
    if isinstance(node, list) and key == len(node):
        node.append(value)
    else:
        node[key] = value


def remove_entry(document: dict, path: tuple, predicate) -> None:
    node = document
    for key in path[:-1]:
        node = node[key]
    items = node[path[-1]]
    node[path[-1]] = [item for item in items if not predicate(item)]


MUTATIONS: list[tuple[str, str, object]] = [
    ("sourcePins.inventorySha256", "flip a digest digit", lambda d: set_path(
        d, ("sourcePins", "inventorySha256"), "a" * 64)),
    ("sourcePins.tensorCount", "shrink the tensor count", lambda d: set_path(
        d, ("sourcePins", "tensorCount"), 1657)),
    ("sourcePins.sourceBytes", "shrink the byte total", lambda d: set_path(
        d, ("sourcePins", "sourceBytes"), 359999963127)),
    ("sourcePins.bytesByDtype", "shrink BF16 bytes", lambda d: set_path(
        d, ("sourcePins", "bytesByDtype", "BF16"), 359999962847)),
    ("sourcePins.classificationBase", "shrink the base class count", lambda d: set_path(
        d, ("sourcePins", "classification", "base"), 1293)),
    ("sourcePins.classificationVision", "shrink the vision class count", lambda d: set_path(
        d, ("sourcePins", "classification", "vision"), 332)),
    ("sourcePins.classificationMtp", "shrink the mtp class count", lambda d: set_path(
        d, ("sourcePins", "classification", "mtp"), 30)),
    ("fileSha256", "wrong config.json digest", lambda d: set_path(
        d, ("sourcePins", "fileSha256", "config.json"), "f" * 64)),
    ("transformersSource", "wrong modeling source digest", lambda d: set_path(
        d, ("sourcePins", "transformersSourceSha256",
            "src/transformers/models/qwen4_exp/modeling_qwen4_exp.py"), "e" * 64)),
    ("identity.storeFamily", "wrong expert store family", lambda d: set_path(
        d, ("identity", "expertStoreFamily"), 3)),
    ("identity.textModelType", "wrong text model type", lambda d: set_path(
        d, ("identity", "textModelType"), "qwen3_text")),
    ("identity.pleMagic", "wrong PLE extent magic", lambda d: set_path(
        d, ("identity", "pleExtentMagic"), "ds4.ple_rows.v2")),
    ("pinnedConfig.numExperts", "shrink the expert count", lambda d: set_path(
        d, ("pinnedConfig", "text", "num_experts"), 384)),
    ("pinnedConfig.numLayers", "shrink the layer count", lambda d: set_path(
        d, ("pinnedConfig", "text", "num_hidden_layers"), 47)),
    ("pinnedConfig.hiddenSize", "shrink the hidden size", lambda d: set_path(
        d, ("pinnedConfig", "text", "hidden_size"), 2048)),
    ("pinnedConfig.pleLayerIds", "move PLE to another layer", lambda d: set_path(
        d, ("pinnedConfig", "text", "ple_layer_ids"), [3])),
    ("pinnedConfig.layerTypesPattern", "break the full-attention interval", lambda d: set_path(
        d, ("pinnedConfig", "text", "layer_types", 3), "linear_attention")),
    ("pinnedConfig.mrope", "change the pinned mrope section allocation", lambda d: set_path(
        d, ("pinnedConfig", "text", "rope_parameters", "mrope_section"), [12, 10, 10])),
    ("pinnedConfig.indexerBudget", "break the indexer budget divisor", lambda d: set_path(
        d, ("pinnedConfig", "text", "indexer_budget"), 2049)),
    ("pinnedConfig.tie", "claim tied embeddings", lambda d: set_path(
        d, ("pinnedConfig", "text", "tie_word_embeddings"), True)),
    ("pinnedConfig.seed", "change the PLE seed default", lambda d: set_path(
        d, ("pinnedConfig", "defaultsUsed", "seed"), 1235)),
    ("derived.gdn_repeat_ratio", "wrong GDN repeat ratio", lambda d: set_path(
        d, ("derivedConstants", "gdn_repeat_ratio", "value"), 2)),
    ("derived.indexer_selected_width", "wrong selected width", lambda d: set_path(
        d, ("derivedConstants", "indexer_selected_width", "value"), 2050)),
    ("derived.ple_head_dim", "wrong PLE head width", lambda d: set_path(
        d, ("derivedConstants", "ple_head_dim", "value"), 96)),
    ("derived.routed_parameters_total", "wrong routed parameter total", lambda d: set_path(
        d, ("derivedConstants", "routed_parameters_total", "value"), 120795955201)),
    ("derived.extraKey", "add an unclosed derived constant", lambda d: set_path(
        d, ("derivedConstants", "not_derived"), {"value": 1, "formula": "not derivable"})),
    ("tensorRoles.expertsShape", "wrong fused expert shape", lambda d: set_path(
        d, ("tensorRoles", "all", 5, "shape", 1), 640)),
    ("tensorRoles.dropLinearRole", "drop a GDN role", lambda d: remove_entry(
        d, ("tensorRoles", "linear_attention"), lambda item: item["path"].endswith("in_proj_z"))),
    ("tensorRoles.dropAllRole", "drop a shared role", lambda d: remove_entry(
        d, ("tensorRoles", "all"), lambda item: item["path"].endswith("block_inject_weight"))),
    ("tensorRoles.pleLayer", "move PLE to another layer", lambda d: set_path(
        d, ("tensorRoles", "pleRuntimeLayer"), 2)),
    ("tensorRoles.pleShardCount", "wrong n-gram shard count", lambda d: set_path(
        d, ("tensorRoles", "ple", 6, "count"), 64)),
    ("tensorRoles.expectedBase", "wrong base total", lambda d: set_path(
        d, ("tensorRoles", "expectedBaseTensors"), 1200)),
    ("pleHash.multipliers", "wrong SplitMix multiplier", lambda d: set_path(
        d, ("pleHash", "layerMultipliers", 0), 23703573157771)),
    ("pleHash.headPrimes", "wrong head prime", lambda d: set_path(
        d, ("pleHash", "headPrimes", 0), 20000001)),
    ("pleHash.headPrimesLength", "drop a head prime", lambda d: remove_entry(
        d, ("pleHash", "headPrimes"), lambda item: item == 20000171)),
    ("pleHash.headOffsets", "wrong head offset", lambda d: set_path(
        d, ("pleHash", "headOffsets", 1), 20000004)),
    ("pleHash.paddedRows", "shrink padded rows", lambda d: set_path(
        d, ("pleHash", "paddedRows"), 320001535)),
    ("pleHash.totalHeadRows", "shrink head rows", lambda d: set_path(
        d, ("pleHash", "totalHeadRows"), 320001445)),
    ("pleHash.maxProduct", "wrong largest product", lambda d: set_path(
        d, ("pleHash", "maxHashProduct"), 5886047582964040312)),
    ("pleHash.seed", "change the PLE seed", lambda d: set_path(
        d, ("pleHash", "seed"), 4242)),
    ("pleHash.derivationGamma", "wrong SplitMix gamma", lambda d: set_path(
        d, ("pleHash", "derivation", "splitmix_gamma"), "0x9e3779b97f4a7c16")),
    ("pleHash.pleLayerIndex", "wrong PLE layer index", lambda d: set_path(
        d, ("pleHash", "pleLayerIndex"), 1)),
    ("pleExtent.rows", "wrong extent rows", lambda d: set_path(
        d, ("pleExtent", "rows"), 320001535)),
    ("pleExtent.shardRows", "wrong shard rows", lambda d: set_path(
        d, ("pleExtent", "embeddingShardRows"), 2500011)),
    ("pleExtent.rowWidth", "wrong extent row width", lambda d: set_path(
        d, ("pleExtent", "rowWidth"), 128)),
    ("pleExtent.version", "wrong PLE wire version", lambda d: set_path(
        d, ("pleExtent", "version"), 2)),
    ("pleExtent.family", "wrong PLE family", lambda d: set_path(
        d, ("pleExtent", "family"), 3)),
    ("pleExtent.profileId", "wrong PLE artifact profile", lambda d: set_path(
        d, ("pleExtent", "profileId"), "qwen35moe-base-v1")),
    ("pleExtent.hashId", "wrong PLE hash identity", lambda d: set_path(
        d, ("pleExtent", "hashId"), "SplitMix64-Qwen4Exp-v2")),
    ("pleExtent.rowAlignment", "wrong PLE row alignment", lambda d: set_path(
        d, ("pleExtent", "rowAlignment"), 64)),
    ("pleExtent.headRows", "wrong logical PLE head row count", lambda d: set_path(
        d, ("pleExtent", "headRows"), 320001445)),
    ("pleExtent.paddingRows", "wrong explicit PLE padding", lambda d: set_path(
        d, ("pleExtent", "paddingRows"), 89)),
    ("pleExtent.wireEndianness", "wrong PLE wire endianness", lambda d: set_path(
        d, ("pleExtent", "wireFormat", "endianness"), "big")),
    ("pleExtent.wireHeader", "wrong PLE manifest header size", lambda d: set_path(
        d, ("pleExtent", "wireFormat", "headerBytes"), 256)),
    ("pleExtent.wirePageHeader", "wrong PLE page header size", lambda d: set_path(
        d, ("pleExtent", "wireFormat", "pageHeaderBytes"), 32)),
    ("pleExtent.wireAlignment", "wrong PLE minimum page alignment", lambda d: set_path(
        d, ("pleExtent", "wireFormat", "minimumPageAlignment"), 2048)),
    ("pleExtent.wireDigestAlgorithm", "wrong page digest algorithm", lambda d: set_path(
        d, ("pleExtent", "wireFormat", "pageDigestAlgorithm"), "BLAKE3")),
    ("pleExtent.wireDigestBytes", "wrong page digest width", lambda d: set_path(
        d, ("pleExtent", "wireFormat", "pageDigestBytes"), 64)),
    ("pleExtent.wireDigestTable", "invent a per-row digest index", lambda d: set_path(
        d, ("pleExtent", "wireFormat", "pageDigestTable"), "one digest per row")),
    ("pleExtent.wireManifestDigest", "change manifest digest coverage", lambda d: set_path(
        d, ("pleExtent", "wireFormat", "manifestDigest"), "SHA-256(header only)")),
    ("pleExtent.wirePayloadDigest", "change payload digest coverage", lambda d: set_path(
        d, ("pleExtent", "wireFormat", "payloadDigest"), "SHA-256(encoded rows)")),
    ("pleExtent.wireCodecBinding", "remove caller-supplied codec binding", lambda d: set_path(
        d, ("pleExtent", "wireFormat", "codecBinding"), "")),
    ("pleExtent.pendingCodec", "pretend the production codec is frozen", lambda d: set_path(
        d, ("pleExtent", "pendingDecisions", 0), "production codec is frozen")),
    ("pleExtent.pendingPage", "pretend production page geometry is frozen", lambda d: set_path(
        d, ("pleExtent", "pendingDecisions", 1), "production page geometry is frozen")),
    ("expertStore.perLayer", "wrong experts per layer", lambda d: set_path(
        d, ("expertStore", "expertsPerLayer"), 384)),
    ("expertStore.routeRecords", "wrong route records per token", lambda d: set_path(
        d, ("expertStore", "minimumRouteRecordsPerToken"), 481)),
    ("expertStore.raiseTo", "raise the cap to the wrong value", lambda d: set_path(
        d, ("expertStore", "raiseMaxExperts", "to"), 384)),
    ("expertStore.family", "wrong Qwen4Exp store family", lambda d: set_path(
        d, ("expertStore", "family"), 3)),
    ("expertStore.raiseField", "wrong shared expert-cap field", lambda d: set_path(
        d, ("expertStore", "raiseMaxExperts", "field"), "MAX_EXPERTS")),
    ("expertStore.raiseFrom", "wrong prior shared expert cap", lambda d: set_path(
        d, ("expertStore", "raiseMaxExperts", "from"), 383)),
    ("expertStore.raisePhase", "move the cap raise out of Phase 2", lambda d: set_path(
        d, ("expertStore", "raiseMaxExperts", "phase"), 3)),
    ("expertStore.raiseNote", "remove the admission-only cap note", lambda d: set_path(
        d, ("expertStore", "raiseMaxExperts", "note"), "")),
    ("expertStore.candidateStorage", "select the wrong structural storage", lambda d: set_path(
        d, ("expertStore", "structuralCandidate", "storage"), "ggml-k-quant")),
    ("expertStore.candidateLogicalType", "change the logical descriptor type", lambda d: set_path(
        d, ("expertStore", "structuralCandidate", "logicalGgmlType"), "IQ2_XS")),
    ("expertStore.candidateGroup", "change the affine group size", lambda d: set_path(
        d, ("expertStore", "structuralCandidate", "groupSize"), 128)),
    ("expertStore.candidateBlock", "change the affine physical block bytes", lambda d: set_path(
        d, ("expertStore", "structuralCandidate", "blockBytes"), 32)),
    ("expertStore.candidateStatus", "claim structural candidate is release-qualified", lambda d: set_path(
        d, ("expertStore", "structuralCandidate", "status"), "release-qualified")),
    ("expertStore.candidateReason", "drop the 640-wide-row reason", lambda d: set_path(
        d, ("expertStore", "structuralCandidate", "reason"), "structural candidate")),
    ("norms.dropGated", "unclassify the GDN norm", lambda d: set_path(
        d, ("normConventions", "conventional"), [])),
    ("norms.addUnknown", "classify an unknown role", lambda d: set_path(
        d, ("normConventions", "zeroCentered", len(contract_norms(d))), "model.norm.weight")),
    ("norms.zeroFormula", "drop the zero-centered offset", lambda d: set_path(
        d, ("normConventions", "formulaZeroCentered"), "y = x * rsqrt(mean(x^2) + eps) * w")),
    ("graph.finalNormPresent", "claim a final norm exists", lambda d: set_path(
        d, ("graphFacts", "finalNorm", "present"), True)),
    ("graph.paddingEmbedZero", "claim the padding row is zero", lambda d: set_path(
        d, ("graphFacts", "embeddingPaddingRow", "embedAllZero"), True)),
    ("graph.mropeRows", "wrong mrope position rows", lambda d: set_path(
        d, ("graphFacts", "attention", "mropePositionRowsForText"), 3)),
    ("exclusions.mtpTensors", "wrong excluded mtp count", lambda d: set_path(
        d, ("exclusions", "mtpTensors"), 30)),
    ("exclusions.visionPrefix", "wrong vision prefix", lambda d: set_path(
        d, ("exclusions", "visionPrefix"), "visual.")),
    ("license.assumedApache", "assume Apache-2.0", lambda d: set_path(
        d, ("license", "assumedApache2"), True)),
    ("license.reviewed", "claim the license review is done", lambda d: set_path(
        d, ("license", "reviewStatus"), "approved")),
    ("fixturePaddingProbe", "break the padding probe", lambda d, f: set_path(
        f, ("paddingRowFacts", "embedZero"), True)),
    ("fixtureClassification", "break the fixture classification", lambda d, f: set_path(
        f, ("classification", "base"), 1293)),
]


def contract_norms(document: dict) -> list[str]:
    return document["normConventions"]["zeroCentered"]


def run_mutations(contract: dict, fixture: dict) -> tuple[list[tuple[str, bool, str]], list[str]]:
    rows: list[tuple[str, bool, str]] = []
    problems: list[str] = []
    baseline = validate(contract, fixture)
    if baseline:
        problems.append("baseline contract fails: " + "; ".join(baseline[:5]))
        return rows, problems
    for label, _why, mutate in MUTATIONS:
        doc = copy.deepcopy(contract)
        doc_norms = doc
        doc = doc_norms
        fixed = copy.deepcopy(fixture)
        try:
            if mutate.__code__.co_argcount == 2:
                mutate(doc, fixed)
            else:
                mutate(doc)
        except (KeyError, IndexError, TypeError) as exc:
            rows.append((label, False, f"mutation itself failed: {exc}"))
            problems.append(f"{label}: mutation raised {exc}")
            continue
        errors = validate(doc, fixed)
        rows.append((label, bool(errors), errors[0] if errors else "no error"))
        if not errors:
            problems.append(f"{label}: mutation left the contract looking valid")
    return rows, problems


def main() -> int:
    contract = load(CONTRACT_PATH)
    fixture = load(FIXTURE_PATH)
    errors = validate(contract, fixture)
    for name in CHECK_NAMES:
        check_errors = validate(contract, fixture, only=[name])
        print(f"{'FAIL' if check_errors else 'ok  '} {name}")
    rows, problems = run_mutations(contract, fixture)
    rejected = sum(1 for _, was_rejected, _ in rows if was_rejected)
    print(f"mutations: {rejected} of {len(rows)} rejected")
    for error in errors:
        print("error: " + error, file=sys.stderr)
    for problem in problems:
        print("mutation-negative: " + problem, file=sys.stderr)
    if errors or problems:
        return 1
    print("qwen4exp contract validated against the pinned inventory")
    return 0


if __name__ == "__main__":
    sys.exit(main())
