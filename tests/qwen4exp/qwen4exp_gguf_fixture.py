#!/usr/bin/env python3
"""Build the sparse, structural-only Qwen4Exp GGUF admission fixture.

The fixture has production logical shapes and production PLE row geometry, but
it deliberately has no release-qualified weight codec.  It is accepted only by
a ``DS4_TEST_HOOKS`` build using the matching structural profile.  Large tensor
payloads are holes; only the GGUF directory and the two embedded manifests are
written.  Consequently a complete file hash would read roughly 168 GiB of
zeros and is intentionally not part of this test helper.

This module does not import Hebrus loader or converter code.  It derives the
physical inventory independently from the pinned source inventory and profile
contract, and writes the two documented little-endian store formats directly.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import random
import re
import struct
import tempfile
from collections.abc import Sequence
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "docs" / "contracts" / "qwen4exp-profile.json"
INVENTORY = (
    ROOT / "tests" / "qwen4exp" / "fixtures" /
    "qwen38flash-next-inventory-v1.json"
)

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
GGUF_DEFAULT_ALIGNMENT = 32
OWNER_PAGE_ALIGNMENT = 4096

GGUF_UINT8 = 0
GGUF_INT8 = 1
GGUF_UINT16 = 2
GGUF_INT16 = 3
GGUF_UINT32 = 4
GGUF_INT32 = 5
GGUF_FLOAT32 = 6
GGUF_BOOL = 7
GGUF_STRING = 8
GGUF_ARRAY = 9
GGUF_UINT64 = 10
GGUF_INT64 = 11
GGUF_FLOAT64 = 12

GGML_I8 = 24
GGML_BF16 = 30

EXPERT_TENSOR = "ds4.expert_major.v2"
EXPERT_MAGIC = b"DS4EXPV2"
EXPERT_HEADER_BYTES = 256
EXPERT_LAYER_BYTES = 224
EXPERT_COMPONENT_OFFSET = 32
EXPERT_COMPONENT_BYTES = 56
EXPERT_MANIFEST_DIGEST_OFFSET = 168
EXPERT_ALIGNMENT = 4096
EXPERT_FAMILY = 4
EXPERT_STORAGE_MLX_AFFINE4 = 1
EXPERT_GROUP_SIZE = 64
EXPERT_BLOCK_BYTES = 36
EXPERT_LAYERS = 48
EXPERTS = 512
EXPERTS_USED = 10

PLE_TENSOR = "ds4.ple_rows.v1"
PLE_MAGIC = b"DS4PLEV1"
PLE_PROFILE = "qwen4exp-base-v1"
PLE_HASH = "SplitMix64-Qwen4Exp-v1"
PLE_CODEC = "q4exp-fixture-bf16-v1"
PHYSICAL_PROFILE = "qwen4exp-phase3-fixture-bf16-v1"
BOUNDED_FUZZ_SEED = 0x5147455846555A5A
BOUNDED_FUZZ_CASES_PER_REGION = 12
BOUNDED_FUZZ_REGIONS = ("gguf", "expert", "ple")
PLE_HEADER_BYTES = 512
PLE_PAGE_HEADER_BYTES = 64
PLE_MANIFEST_DIGEST_OFFSET = 376
PLE_ROWS = 320_001_536
PLE_HEAD_ROWS = 320_001_446
PLE_ROW_WIDTH = 160
PLE_ROW_ALIGNMENT = 128
PLE_CODEC_VERSION = 1
PLE_CODEC_GROUP_SIZE = 1
PLE_ENCODED_ROW_BYTES = 320
PLE_ROWS_PER_PAGE = PLE_ROWS
PLE_PAGE_ALIGNMENT = 4096

SOURCE_TENSORS = 1658
SOURCE_BYTES = 359_999_963_128
SOURCE_INVENTORY_SHA256 = (
    "a639efc7a5147b04200e870d7e320335527f4361a8327b137feca2683b1dc434"
)
HF_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
TRANSFORMERS_COMMIT = "42ca97014c85d71a88ad60d55f08cb9fb4d26e2c"
TOKENIZER_SHA256 = (
    "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
)
TEMPLATE_SHA256 = (
    "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
)

PLE_MULTIPLIERS = (
    23_703_573_157_769,
    20_109_073_645_365,
    8_052_911_324_071,
)
PLE_PRIMES = (
    20_000_003, 20_000_023, 20_000_033, 20_000_047,
    20_000_059, 20_000_063, 20_000_069, 20_000_077,
    20_000_081, 20_000_093, 20_000_107, 20_000_147,
    20_000_153, 20_000_159, 20_000_161, 20_000_171,
)
PLE_OFFSETS = (
    0, 20_000_003, 40_000_026, 60_000_059,
    80_000_106, 100_000_165, 120_000_228, 140_000_297,
    160_000_374, 180_000_455, 200_000_548, 220_000_655,
    240_000_802, 260_000_955, 280_001_114, 300_001_275,
)
ROUTED_RE = re.compile(
    r"^model\.language_model\.layers\.\d+\.mlp\.experts\."
    r"(?:gate_up_proj|down_proj)$"
)
PLE_PREFIX = "model.language_model.layers.1.ple."
PLE_SHARD_RE = re.compile(
    r"^model\.language_model\.layers\.1\.ple\.ple_embedding\."
    r"ngram_embedding\.shard_\d+\.weight$"
)
PLE_HASH_SUFFIXES = (
    ".layer_multipliers",
    ".ngram_heads_offsets",
    ".ngram_heads_vocab_sizes",
)


@dataclasses.dataclass(frozen=True)
class Metadata:
    key: str
    value_type: int
    value: object


@dataclasses.dataclass(frozen=True)
class Tensor:
    name: str
    dims: tuple[int, ...]
    ggml_type: int
    size: int
    rel_offset: int = 0


@dataclasses.dataclass(frozen=True)
class FixtureSummary:
    path: Path
    file_bytes: int
    allocated_bytes: int
    metadata_count: int
    tensor_count: int
    dense_count: int
    header_bytes: int
    dense_bytes: int
    dense_page_bytes: int
    padding_bytes: int
    expert_offset: int
    expert_bytes: int
    ple_offset: int
    ple_bytes: int
    structural_sha256: str


@dataclasses.dataclass(frozen=True)
class Mutation:
    """One fail-closed mutation applied while rebuilding the same path."""

    name: str


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    return (value + alignment - 1) // alignment * alignment


def pack_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def pack_value(value_type: int, value: object) -> bytes:
    if value_type == GGUF_UINT32:
        return struct.pack("<I", int(value))
    if value_type == GGUF_INT32:
        return struct.pack("<i", int(value))
    if value_type == GGUF_UINT64:
        return struct.pack("<Q", int(value))
    if value_type == GGUF_INT64:
        return struct.pack("<q", int(value))
    if value_type == GGUF_FLOAT32:
        return struct.pack("<f", float(value))
    if value_type == GGUF_BOOL:
        return struct.pack("<B", 1 if value else 0)
    if value_type == GGUF_STRING:
        return pack_string(str(value))
    if value_type == GGUF_ARRAY:
        item_type, items = value  # type: ignore[misc]
        result = bytearray(struct.pack("<IQ", item_type, len(items)))
        for item in items:
            result += pack_value(item_type, item)
        return bytes(result)
    raise ValueError(f"unsupported fixture GGUF value type: {value_type}")


def metadata_bytes(entries: Sequence[Metadata]) -> bytes:
    result = bytearray()
    for entry in entries:
        result += pack_string(entry.key)
        result += struct.pack("<I", entry.value_type)
        result += pack_value(entry.value_type, entry.value)
    return bytes(result)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise AssertionError(f"expected JSON object: {path}")
    return value


def _is_dense_or_runtime_control(name: str) -> bool:
    if name.startswith("model.visual.") or name.startswith("mtp."):
        return False
    if ROUTED_RE.fullmatch(name):
        return False
    if name.startswith(PLE_PREFIX):
        if PLE_SHARD_RE.fullmatch(name) or name.endswith(PLE_HASH_SUFFIXES):
            return False
    return True


def dense_tensors() -> list[Tensor]:
    """Return the exact 1067 BF16 physical dense/control tensors.

    Safetensors records dimensions in framework order.  GGUF dimensions are
    the reverse order, matching Hebrus's existing tensor-directory convention.
    """

    contract = _load_json(CONTRACT)
    inventory = _load_json(INVENTORY)
    pins = contract["sourcePins"]
    if (pins["tensorCount"], pins["sourceBytes"],
            pins["inventorySha256"]) != (
                SOURCE_TENSORS, SOURCE_BYTES, SOURCE_INVENTORY_SHA256):
        raise AssertionError("Qwen4Exp contract source pins drifted")
    tensors: list[Tensor] = []
    for row in inventory["tensors"]:
        name = row["name"]
        if not _is_dense_or_runtime_control(name):
            continue
        if row["dtype"] != "BF16":
            raise AssertionError(f"non-BF16 physical dense tensor: {name}")
        source_dims = tuple(int(dim) for dim in row["shape"])
        elements = 1
        for dim in source_dims:
            if dim <= 0:
                raise AssertionError(f"invalid source dimension: {name}")
            elements *= dim
        tensors.append(Tensor(
            name=name,
            dims=tuple(reversed(source_dims)),
            ggml_type=GGML_BF16,
            size=elements * 2,
        ))
    tensors.sort(key=lambda tensor: tensor.name)
    if len(tensors) != 1067 or len({tensor.name for tensor in tensors}) != 1067:
        raise AssertionError(
            f"expected 1067 unique dense/control tensors, got {len(tensors)}"
        )
    descriptors = [{
        "sourceIdentity": tensor.name,
        "ggufDimensions": list(tensor.dims),
        "ggufType": "BF16",
    } for tensor in tensors]
    encoded = json.dumps(
        descriptors, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    expected = contract["admission"]["physicalFixture"]["dense"][
        "descriptorSha256"
    ]
    if hashlib.sha256(encoded).hexdigest() != expected:
        raise AssertionError("Qwen4Exp dense descriptor digest drifted")
    return tensors


def expert_store_geometry() -> tuple[int, int, int, int]:
    gate_bytes = 2560 // EXPERT_GROUP_SIZE * EXPERT_BLOCK_BYTES * 640
    down_bytes = 640 // EXPERT_GROUP_SIZE * EXPERT_BLOCK_BYTES * 2560
    record_bytes = gate_bytes * 2 + down_bytes
    layer_bytes = record_bytes * EXPERTS
    data_offset = align_up(
        EXPERT_HEADER_BYTES + EXPERT_LAYERS * EXPERT_LAYER_BYTES,
        EXPERT_ALIGNMENT,
    )
    store_bytes = data_offset + layer_bytes * EXPERT_LAYERS
    return record_bytes, layer_bytes, data_offset, store_bytes


def make_expert_manifest(mutation: str | None = None) -> bytes:
    record_bytes, layer_bytes, data_offset, store_bytes = \
        expert_store_geometry()
    descriptors = bytearray()
    component_geometry = (
        (2560, 640, 512),
        (2560, 640, 512),
        (640, 2560, 512),
    )
    expert_bytes = 2560 // EXPERT_GROUP_SIZE * EXPERT_BLOCK_BYTES * 640
    cursor = data_offset
    for layer in range(EXPERT_LAYERS):
        entry = bytearray(EXPERT_LAYER_BYTES)
        struct.pack_into(
            "<IIQQQ", entry, 0,
            layer, EXPERTS, record_bytes, cursor, layer_bytes,
        )
        record_offset = 0
        for role, dims in enumerate(component_geometry):
            role_bytes = expert_bytes
            struct.pack_into(
                "<IIIIQQQQQ", entry,
                EXPERT_COMPONENT_OFFSET + role * EXPERT_COMPONENT_BYTES,
                role, 12, 3, EXPERT_GROUP_SIZE,
                dims[0], dims[1], dims[2], role_bytes, record_offset,
            )
            record_offset += role_bytes
        if record_offset != record_bytes:
            raise AssertionError("expert component record geometry drifted")
        descriptors += entry
        cursor += layer_bytes
    if cursor != store_bytes:
        raise AssertionError("expert store size drifted")

    if mutation == "expert_component_offset":
        struct.pack_into(
            "<Q", descriptors,
            EXPERT_COMPONENT_OFFSET + 48,
            EXPERT_BLOCK_BYTES,
        )
    elif mutation == "expert_component_length":
        struct.pack_into(
            "<Q", descriptors,
            EXPERT_COMPONENT_OFFSET + 40,
            expert_bytes + EXPERT_BLOCK_BYTES,
        )

    header = bytearray(EXPERT_HEADER_BYTES)
    header[:8] = EXPERT_MAGIC
    family = 3 if mutation == "expert_family" else EXPERT_FAMILY
    version = 3 if mutation == "expert_header_version" else 2
    struct.pack_into(
        "<IIIIIIQQQQQQQ", header, 8,
        version, EXPERT_HEADER_BYTES, family, EXPERTS_USED,
        EXPERT_LAYERS, EXPERTS, SOURCE_TENSORS, EXPERT_LAYERS,
        len(descriptors), EXPERT_HEADER_BYTES, data_offset,
        store_bytes - data_offset, store_bytes,
    )
    struct.pack_into("<Q", header, 88, SOURCE_BYTES)
    header[96:128] = bytes.fromhex(SOURCE_INVENTORY_SHA256)
    # The sparse structural fixture intentionally does not authenticate the
    # unmaterialized routed payload.  The report must expose payloadVerified=false.
    header[128:160] = bytes(32)
    struct.pack_into(
        "<II", header, 160,
        EXPERT_STORAGE_MLX_AFFINE4, EXPERT_GROUP_SIZE,
    )
    digest_header = bytearray(header)
    digest_header[
        EXPERT_MANIFEST_DIGEST_OFFSET:
        EXPERT_MANIFEST_DIGEST_OFFSET + 32
    ] = bytes(32)
    digest = hashlib.sha256(digest_header + descriptors).digest()
    header[
        EXPERT_MANIFEST_DIGEST_OFFSET:
        EXPERT_MANIFEST_DIGEST_OFFSET + 32
    ] = digest
    if mutation == "expert_manifest_digest":
        header[EXPERT_MANIFEST_DIGEST_OFFSET] ^= 0x80
    return bytes(header + descriptors)


def ple_store_geometry() -> tuple[int, int, int, int, int]:
    page_count = 1
    page_stride = align_up(
        PLE_PAGE_HEADER_BYTES + PLE_ROWS_PER_PAGE * PLE_ENCODED_ROW_BYTES,
        PLE_PAGE_ALIGNMENT,
    )
    digest_offset = PLE_HEADER_BYTES
    digest_bytes = 32
    payload_offset = align_up(
        digest_offset + digest_bytes, PLE_PAGE_ALIGNMENT,
    )
    store_bytes = payload_offset + page_count * page_stride
    return page_count, page_stride, digest_offset, payload_offset, store_bytes


def _wire_id(value: str) -> bytes:
    encoded = value.encode("ascii")
    if not encoded or len(encoded) >= 32:
        raise AssertionError(f"invalid PLE wire identifier: {value!r}")
    return encoded + bytes(32 - len(encoded))


def make_ple_manifest(mutation: str | None = None) -> bytes:
    page_count, page_stride, digest_offset, payload_offset, store_bytes = \
        ple_store_geometry()
    digest_bytes = page_count * 32
    payload_bytes = page_count * page_stride
    header = bytearray(PLE_HEADER_BYTES)
    header[:8] = PLE_MAGIC
    struct.pack_into(
        "<I", header, 8,
        2 if mutation == "ple_header_version" else 1,
    )
    struct.pack_into("<I", header, 12, PLE_HEADER_BYTES)
    struct.pack_into("<I", header, 16, EXPERT_FAMILY)
    struct.pack_into("<I", header, 20, len(PLE_PRIMES))
    header[24:56] = _wire_id(PLE_PROFILE)
    header[56:88] = _wire_id(
        "SplitMix64-Qwen4Exp-v2"
        if mutation == "ple_hash_version" else PLE_HASH
    )
    header[88:120] = _wire_id(PLE_CODEC)
    struct.pack_into("<I", header, 120, PLE_CODEC_VERSION)
    struct.pack_into("<I", header, 124, PLE_CODEC_GROUP_SIZE)
    struct.pack_into("<I", header, 128, PLE_ENCODED_ROW_BYTES)
    struct.pack_into("<I", header, 132, PLE_ROWS_PER_PAGE)
    struct.pack_into("<I", header, 136, PLE_PAGE_ALIGNMENT)
    struct.pack_into("<I", header, 140, PLE_PAGE_HEADER_BYTES)
    struct.pack_into(
        "<Q", header, 144,
        PLE_ROWS - PLE_ROW_ALIGNMENT
        if mutation == "ple_manifest_rows" else PLE_ROWS,
    )
    struct.pack_into(
        "<I", header, 152,
        PLE_ROW_WIDTH + 1 if mutation == "ple_row_width" else PLE_ROW_WIDTH,
    )
    struct.pack_into("<I", header, 156, PLE_ROW_ALIGNMENT)
    struct.pack_into("<Q", header, 160, page_count)
    struct.pack_into("<Q", header, 168,
                     page_stride + (4096 if mutation == "ple_page_stride" else 0))
    struct.pack_into("<Q", header, 176, digest_offset)
    struct.pack_into("<Q", header, 184, digest_bytes)
    struct.pack_into("<Q", header, 192, payload_offset)
    struct.pack_into("<Q", header, 200, payload_bytes)
    struct.pack_into("<Q", header, 208, store_bytes)
    primes = list(PLE_PRIMES)
    offsets = list(PLE_OFFSETS)
    if mutation == "ple_head_prime":
        primes[7] += 2
    elif mutation == "ple_head_offset":
        offsets[7] += 1
    struct.pack_into("<16I", header, 216, *primes)
    struct.pack_into("<16I", header, 280, *offsets)
    # payload SHA is zero because the page payload is intentionally sparse and
    # unverified.  The manifest digest still authenticates this exact status.
    header[344:376] = bytes(32)
    digest_table_and_padding = bytes(payload_offset - PLE_HEADER_BYTES)
    digest_header = bytearray(header)
    digest_header[
        PLE_MANIFEST_DIGEST_OFFSET:PLE_MANIFEST_DIGEST_OFFSET + 32
    ] = bytes(32)
    manifest_digest = hashlib.sha256(
        digest_header + digest_table_and_padding
    ).digest()
    header[
        PLE_MANIFEST_DIGEST_OFFSET:PLE_MANIFEST_DIGEST_OFFSET + 32
    ] = manifest_digest
    if mutation == "ple_manifest_digest":
        header[PLE_MANIFEST_DIGEST_OFFSET] ^= 0x40
    return bytes(header) + digest_table_and_padding


def semantic_metadata(mutation: str | None = None) -> list[Metadata]:
    """Build the exact closed metadata set from the canonical contract."""

    contract = _load_json(CONTRACT)
    schema = contract["admission"]["metadataSchema"]
    if not schema["closed"] or len(schema["entries"]) != 108:
        raise AssertionError("Qwen4Exp admission metadata schema drifted")
    scalar_type = {
        "STRING": GGUF_STRING,
        "BOOL": GGUF_BOOL,
        "FLOAT32": GGUF_FLOAT32,
        "UINT32": GGUF_UINT32,
        "UINT64": GGUF_UINT64,
    }
    array_type = re.compile(r"^(UINT32|UINT64)\[(\d+)\]$")
    values: list[Metadata] = []
    for key, descriptor in schema["entries"].items():
        type_name = descriptor["type"]
        value = descriptor["value"]
        match = array_type.fullmatch(type_name)
        if match:
            item_name, count_text = match.groups()
            if not isinstance(value, list) or len(value) != int(count_text):
                raise AssertionError(f"metadata array contract drifted: {key}")
            values.append(Metadata(
                key, GGUF_ARRAY, (scalar_type[item_name], list(value)),
            ))
        else:
            if type_name not in scalar_type:
                raise AssertionError(f"unknown metadata contract type: {type_name}")
            values.append(Metadata(key, scalar_type[type_name], value))

    if mutation == "architecture":
        _replace_metadata(values, "general.architecture", "qwen4exp-invalid")
    elif mutation == "source_architecture":
        _replace_metadata(
            values, "ds4.model.source_architecture",
            "Qwen4ExpForCausalLM",
        )
    elif mutation == "profile":
        _replace_metadata(values, "ds4.model.profile_id", "qwen4exp-base-v2")
    elif mutation == "physical_profile":
        _replace_metadata(values, "ds4.model.physical_profile_id",
                          "qwen4exp-production-does-not-exist")
    elif mutation == "revision":
        _replace_metadata(values, "ds4.model.source_revision", "0" * 40)
    elif mutation == "context":
        _replace_metadata(values, "qwen4exp.text.max_position_embeddings",
                          262_143)
    elif mutation == "layer_pattern":
        entry = next(item for item in values
                     if item.key == "qwen4exp.text.layer_pattern")
        item_type, items = entry.value  # type: ignore[misc]
        changed = list(items)
        changed[7] = 0
        _replace_metadata(values, entry.key, (item_type, changed))
    elif mutation == "top_k":
        _replace_metadata(values, "qwen4exp.text.num_experts_per_tok", 9)
    elif mutation == "expert_count":
        _replace_metadata(values, "qwen4exp.text.num_experts", 511)
    elif mutation == "gr_rank":
        _replace_metadata(values, "qwen4exp.text.hc_lowrank", 319)
    elif mutation == "qsa_query_heads":
        _replace_metadata(values, "qwen4exp.text.num_attention_heads", 23)
    elif mutation == "qsa_kv_heads":
        _replace_metadata(values, "qwen4exp.text.num_key_value_heads", 3)
    elif mutation == "qsa_head_dim":
        _replace_metadata(values, "qwen4exp.text.head_dim", 255)
    elif mutation == "qsa_rotary_dim":
        _replace_metadata(values, "qwen4exp.text.partial_rotary_factor", 0.5)
    elif mutation == "qsa_index_query_heads":
        _replace_metadata(values, "qwen4exp.text.indexer_n_heads", 5)
    elif mutation == "qsa_index_kv_heads":
        _replace_metadata(values, "qwen4exp.text.indexer_kv_heads", 2)
    elif mutation == "qsa_index_head_dim":
        _replace_metadata(values, "qwen4exp.text.indexer_head_dim", 127)
    elif mutation == "gdn_key_heads":
        _replace_metadata(values, "qwen4exp.text.linear_num_key_heads", 15)
    elif mutation == "gdn_value_heads":
        _replace_metadata(values, "qwen4exp.text.linear_num_value_heads", 47)
    elif mutation == "gdn_key_dim":
        _replace_metadata(values, "qwen4exp.text.linear_key_head_dim", 127)
    elif mutation == "gdn_value_dim":
        _replace_metadata(values, "qwen4exp.text.linear_value_head_dim", 127)
    elif mutation in ("ple_insertion_layer", "ple_source_layer"):
        entry = next(item for item in values
                     if item.key == "qwen4exp.text.ple_layer_ids")
        item_type, _ = entry.value  # type: ignore[misc]
        layer = 3 if mutation == "ple_insertion_layer" else 1
        _replace_metadata(values, entry.key, (item_type, [layer]))
    elif mutation == "ple_rows":
        _replace_metadata(values, "qwen4exp.ple.rows", PLE_ROWS - 128)
    elif mutation == "ple_multiplier_high32":
        entry = next(item for item in values
                     if item.key == "qwen4exp.ple.layer_multipliers")
        item_type, items = entry.value  # type: ignore[misc]
        changed = list(items)
        changed[0] ^= 1 << 40
        _replace_metadata(values, entry.key, (item_type, changed))
    elif mutation == "tokenizer_digest":
        _replace_metadata(values, "ds4.tokenizer.digest", "0" * 64)
    elif mutation == "template_digest":
        _replace_metadata(values, "ds4.chat_template.digest", "0" * 64)
    elif mutation == "special_id":
        _replace_metadata(values, "qwen4exp.tokenizer.image_pad_token_id",
                          248_055)
    elif mutation == "mtp_policy":
        _replace_metadata(values, "ds4.mtp.present", True)
    elif mutation == "duplicate_kv":
        original = next(entry for entry in values
                        if entry.key == "qwen4exp.text.num_hidden_layers")
        values.append(original)
    elif mutation == "missing_metadata":
        index = next(i for i, entry in enumerate(values)
                     if entry.key == "qwen4exp.text.hidden_act")
        del values[index]
    elif mutation == "extra_metadata":
        values.append(Metadata("qwen4exp.unexpected", GGUF_UINT32, 1))
    elif mutation == "forbidden_metadata":
        index = next(i for i, entry in enumerate(values)
                     if entry.key == "qwen4exp.text.hidden_act")
        values[index] = Metadata(
            "qwen4exp.tokenizer.bos_token_id", GGUF_UINT32, 248_044,
        )
    elif mutation == "u64_wrong_type":
        index = next(i for i, entry in enumerate(values)
                     if entry.key == "qwen4exp.ple.layer_multipliers")
        _, items = values[index].value  # type: ignore[misc]
        values[index] = Metadata(
            values[index].key, GGUF_ARRAY, (GGUF_INT64, list(items)),
        )
    return values


def _replace_metadata(entries: list[Metadata], key: str, value: object) -> None:
    index = next(i for i, entry in enumerate(entries) if entry.key == key)
    entries[index] = dataclasses.replace(entries[index], value=value)


def _assign_offsets(tensors: Sequence[Tensor], data_pos: int) -> list[Tensor]:
    """Pack dense tensors at GGUF alignment and opaque stores at host pages.

    The frozen metadata set intentionally has no ``general.alignment``, hence
    GGUF v3's default alignment is 32 bytes.  Opaque stores have a stronger
    independent 4096-byte wire-alignment rule.  The sole bounded opaque-owner
    padding gap is the minimum needed to reach that stronger alignment.
    """

    result: list[Tensor] = []
    cursor = 0
    for tensor in tensors:
        if tensor.name in (EXPERT_TENSOR, PLE_TENSOR):
            cursor = align_up(data_pos + cursor, OWNER_PAGE_ALIGNMENT) - data_pos
        else:
            cursor = align_up(cursor, GGUF_DEFAULT_ALIGNMENT)
        result.append(dataclasses.replace(tensor, rel_offset=cursor))
        cursor += tensor.size
    return result


def physical_tensors(mutation: str | None = None,
                     data_pos: int = 0) -> list[Tensor]:
    dense = dense_tensors()
    _, _, _, expert_bytes = expert_store_geometry()
    _, _, _, _, ple_bytes = ple_store_geometry()
    tensors = dense + [
        Tensor(EXPERT_TENSOR, (expert_bytes,), GGML_I8, expert_bytes),
        Tensor(PLE_TENSOR, (ple_bytes,), GGML_I8, ple_bytes),
    ]
    if mutation == "ple_runtime_layer":
        tensors = [
            dataclasses.replace(
                tensor,
                name=tensor.name.replace(
                    "model.language_model.layers.1.ple.",
                    "model.language_model.layers.2.ple.",
                    1,
                ),
            ) if tensor.name.startswith(
                "model.language_model.layers.1.ple."
            ) else tensor
            for tensor in tensors
        ]
    elif mutation == "truncated_expert_store":
        index = next(i for i, tensor in enumerate(tensors)
                     if tensor.name == EXPERT_TENSOR)
        size = tensors[index].size - OWNER_PAGE_ALIGNMENT
        tensors[index] = dataclasses.replace(
            tensors[index], dims=(size,), size=size,
        )
    elif mutation == "truncated_ple_store":
        index = next(i for i, tensor in enumerate(tensors)
                     if tensor.name == PLE_TENSOR)
        size = tensors[index].size - PLE_PAGE_ALIGNMENT
        tensors[index] = dataclasses.replace(
            tensors[index], dims=(size,), size=size,
        )
    elif mutation == "missing_tensor":
        del tensors[0]
    elif mutation == "extra_tensor":
        tensors.insert(-2, Tensor("q4exp.unexpected.weight", (32,),
                                  GGML_BF16, 64))
    elif mutation == "canonical_routed":
        tensors[0] = Tensor("blk.0.ffn_gate_exps.weight", (1,), GGML_I8, 1)
    elif mutation == "vision_tensor":
        tensors[0] = Tensor(
            "model.visual.fixture.weight", (1,), GGML_BF16, 2,
        )
    elif mutation == "mtp_tensor":
        tensors[0] = Tensor("mtp.fixture.weight", (1,), GGML_BF16, 2)
    elif mutation == "second_expert_store":
        tensors[0] = Tensor(
            EXPERT_TENSOR, (expert_bytes,), GGML_I8, expert_bytes,
        )
    elif mutation == "second_ple_store":
        tensors[0] = Tensor(PLE_TENSOR, (ple_bytes,), GGML_I8, ple_bytes)
    elif mutation == "duplicate_tensor":
        tensors[1] = dataclasses.replace(tensors[0])
    elif mutation == "tensor_rank":
        tensors[0] = dataclasses.replace(
            tensors[0], dims=tensors[0].dims + (1,),
        )
    elif mutation == "tensor_dimension":
        dims = list(tensors[0].dims)
        dims[0] += 1
        tensors[0] = dataclasses.replace(tensors[0], dims=tuple(dims),
                                         size=tensors[0].size + 2)
    elif mutation == "tensor_type":
        tensors[0] = dataclasses.replace(tensors[0], ggml_type=GGML_I8,
                                         size=tensors[0].size // 2)
    elif mutation == "dimension_overflow":
        tensors[0] = dataclasses.replace(
            tensors[0], dims=(1 << 63, 3), size=tensors[0].size,
        )
    assigned = _assign_offsets(tensors, data_pos)
    if mutation == "exact_overlap":
        assigned[1] = dataclasses.replace(
            assigned[1], rel_offset=assigned[0].rel_offset,
        )
    elif mutation == "offset_overflow":
        assigned[0] = dataclasses.replace(
            assigned[0], rel_offset=(1 << 64) - 1,
        )
    elif mutation == "unaligned_tensor":
        assigned[0] = dataclasses.replace(
            assigned[0], rel_offset=assigned[0].rel_offset + 1,
        )
    elif mutation in ("page_isolation", "unaligned_store"):
        index = next(i for i, tensor in enumerate(assigned)
                     if tensor.name == PLE_TENSOR)
        assigned[index] = dataclasses.replace(
            assigned[index], rel_offset=assigned[index].rel_offset + 32,
        )
    elif mutation == "store_overlap":
        expert_index = next(i for i, tensor in enumerate(assigned)
                            if tensor.name == EXPERT_TENSOR)
        ple_index = next(i for i, tensor in enumerate(assigned)
                         if tensor.name == PLE_TENSOR)
        assigned[ple_index] = dataclasses.replace(
            assigned[ple_index],
            # The PLE manifest remains intact while landing in the sparse,
            # unauthenticated ExpertMajor payload rather than its manifest.
            rel_offset=(assigned[expert_index].rel_offset +
                        expert_store_geometry()[2]),
        )
    return assigned


def tensor_directory_bytes(tensors: Sequence[Tensor]) -> bytes:
    result = bytearray()
    for tensor in tensors:
        result += pack_string(tensor.name)
        result += struct.pack("<I", len(tensor.dims))
        for dim in tensor.dims:
            result += struct.pack("<Q", dim)
        result += struct.pack("<IQ", tensor.ggml_type, tensor.rel_offset)
    return bytes(result)


def _write_all_at(file, offset: int, data: bytes) -> None:
    file.seek(offset)
    written = file.write(data)
    if written != len(data):
        raise OSError(f"short fixture write at {offset}: {written}/{len(data)}")


def _structural_digest(path: Path, data_pos: int,
                       tensors: Sequence[Tensor]) -> str:
    """Hash bounded structural bytes, never sparse payload holes."""

    hash_value = hashlib.sha256()
    with path.open("rb") as file:
        hash_value.update(file.read(data_pos))
        for tensor in tensors:
            if tensor.name == EXPERT_TENSOR:
                file.seek(data_pos + tensor.rel_offset)
                hash_value.update(file.read(
                    EXPERT_HEADER_BYTES + EXPERT_LAYERS * EXPERT_LAYER_BYTES
                ))
            elif tensor.name == PLE_TENSOR:
                file.seek(data_pos + tensor.rel_offset)
                hash_value.update(file.read(PLE_PAGE_ALIGNMENT))
    return hash_value.hexdigest()


def _page_union_bytes(data_pos: int, tensors: Sequence[Tensor]) -> int:
    spans = sorted((
        (data_pos + tensor.rel_offset) & ~(OWNER_PAGE_ALIGNMENT - 1),
        align_up(data_pos + tensor.rel_offset + tensor.size,
                 OWNER_PAGE_ALIGNMENT),
    ) for tensor in tensors)
    total = 0
    previous_lo = previous_hi = 0
    for lo, hi in spans:
        if previous_hi and lo <= previous_hi:
            previous_hi = max(previous_hi, hi)
            continue
        if previous_hi:
            total += previous_hi - previous_lo
        previous_lo, previous_hi = lo, hi
    if previous_hi:
        total += previous_hi - previous_lo
    return total


def build_fixture(path: Path, mutation: str | Mutation | None = None) \
        -> FixtureSummary:
    mutation_name = mutation.name if isinstance(mutation, Mutation) else mutation
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = semantic_metadata(mutation_name)
    placeholder_tensors = physical_tensors(mutation_name)
    encoded_metadata = metadata_bytes(metadata)
    gguf_magic = 0 if mutation_name == "gguf_bad_magic" else GGUF_MAGIC
    gguf_version = 2 if mutation_name == "gguf_bad_version" else GGUF_VERSION
    header = struct.pack(
        "<IIQQ", gguf_magic, gguf_version,
        len(placeholder_tensors), len(metadata),
    )
    placeholder_directory = tensor_directory_bytes(placeholder_tensors)
    data_pos = align_up(
        len(header) + len(encoded_metadata) + len(placeholder_directory),
        GGUF_DEFAULT_ALIGNMENT,
    )
    tensors = physical_tensors(mutation_name, data_pos)
    encoded_directory = tensor_directory_bytes(tensors)
    prefix = header + encoded_metadata + encoded_directory
    if len(prefix) > data_pos:
        raise AssertionError("GGUF directory size changed while assigning offsets")
    prefix += bytes(data_pos - len(prefix))

    # Invalid-directory mutations retain the normal sparse allocation envelope;
    # this avoids trying to truncate at UINT64_MAX or materialize overlaps.
    normal_tensors = _assign_offsets(
        dense_tensors() + [
            Tensor(EXPERT_TENSOR, (expert_store_geometry()[3],), GGML_I8,
                   expert_store_geometry()[3]),
            Tensor(PLE_TENSOR, (ple_store_geometry()[4],), GGML_I8,
                   ple_store_geometry()[4]),
        ],
        data_pos,
    )
    normal_end = normal_tensors[-1].rel_offset + normal_tensors[-1].size
    serializable_ends = [
        tensor.rel_offset + tensor.size for tensor in tensors
        if tensor.rel_offset <= (1 << 63) - 1 and
        tensor.size <= (1 << 63) - 1 - tensor.rel_offset
    ]
    retain_normal_envelope = mutation_name not in {
        "truncated_expert_store", "truncated_ple_store",
    }
    envelope_ends = ([normal_end] if retain_normal_envelope else []) + \
        serializable_ends
    file_end = data_pos + max(envelope_ends)

    with path.open("w+b") as file:
        _write_all_at(file, 0, prefix)
        file.truncate(file_end)
        expert = next((tensor for tensor in tensors
                       if tensor.name == EXPERT_TENSOR), None)
        ple = next((tensor for tensor in tensors
                    if tensor.name == PLE_TENSOR), None)
        if expert is not None:
            manifest_mutation = mutation_name if mutation_name in {
                "expert_family", "expert_header_version",
                "expert_component_offset", "expert_component_length",
                "expert_manifest_digest",
            } else None
            _write_all_at(file, data_pos + expert.rel_offset,
                          make_expert_manifest(manifest_mutation))
        if ple is not None:
            manifest_mutation = mutation_name if mutation_name in {
                "ple_header_version", "ple_row_width", "ple_head_prime",
                "ple_head_offset", "ple_hash_version", "ple_manifest_rows",
                "ple_page_stride", "ple_manifest_digest",
            } else None
            _write_all_at(file, data_pos + ple.rel_offset,
                          make_ple_manifest(manifest_mutation))
        if mutation_name == "truncated_file":
            file.truncate(file_end - 1)
        file.flush()

    status = path.stat()
    allocated_bytes = status.st_blocks * 512
    # Metadata plus manifests are well below one MiB.  Leave ample filesystem
    # bookkeeping headroom while still catching accidental hole materialization.
    if allocated_bytes > 64 * 1024 * 1024:
        raise AssertionError(
            f"sparse fixture allocated {allocated_bytes} physical bytes"
        )
    expert = next((tensor for tensor in tensors
                   if tensor.name == EXPERT_TENSOR), None)
    ple = next((tensor for tensor in tensors if tensor.name == PLE_TENSOR), None)
    dense = [tensor for tensor in tensors
             if tensor.name not in (EXPERT_TENSOR, PLE_TENSOR)]
    dense_bytes = sum(tensor.size for tensor in dense)
    dense_page_bytes = _page_union_bytes(data_pos, dense)
    # Ownership accounting counts GGUF header/directory, tensor bytes, and all
    # inter-owner alignment gaps exactly once.
    padding_bytes = status.st_size - (
        data_pos + dense_bytes +
        (expert.size if expert else 0) + (ple.size if ple else 0)
    )
    return FixtureSummary(
        path=path,
        file_bytes=status.st_size,
        allocated_bytes=allocated_bytes,
        metadata_count=len(metadata),
        tensor_count=len(tensors),
        dense_count=sum(tensor.name not in (EXPERT_TENSOR, PLE_TENSOR)
                        for tensor in tensors),
        header_bytes=data_pos,
        dense_bytes=dense_bytes,
        dense_page_bytes=dense_page_bytes,
        padding_bytes=padding_bytes,
        expert_offset=(data_pos + expert.rel_offset) if expert else 0,
        expert_bytes=expert.size if expert else 0,
        ple_offset=(data_pos + ple.rel_offset) if ple else 0,
        ple_bytes=ple.size if ple else 0,
        structural_sha256=_structural_digest(path, data_pos, tensors),
    )


def _fuzz_xor_byte(file, offset: int, rng: random.Random,
                   bit_count: int = 8) -> None:
    file.seek(offset)
    original = file.read(1)
    if len(original) != 1:
        raise AssertionError(f"fuzz byte is outside fixture: {offset}")
    file.seek(offset)
    file.write(bytes((original[0] ^ (1 << rng.randrange(bit_count)),)))


def _fuzz_write(file, offset: int, fmt: str, value: int) -> None:
    encoded = struct.pack(fmt, value)
    file.seek(offset)
    if len(file.read(len(encoded))) != len(encoded):
        raise AssertionError(f"fuzz field is outside fixture: {offset}")
    file.seek(offset)
    file.write(encoded)


def build_bounded_fuzz_fixture(
        path: Path, region: str, case_index: int,
        seed: int = BOUNDED_FUZZ_SEED) -> FixtureSummary:
    """Apply one deterministic raw parser perturbation to a sparse fixture.

    The corpus deliberately touches only bounded directory or manifest bytes.
    It never reads, copies, or materializes any sparse tensor payload.
    """

    if region not in BOUNDED_FUZZ_REGIONS:
        raise ValueError(f"unknown bounded fuzz region: {region}")
    if not 0 <= case_index < BOUNDED_FUZZ_CASES_PER_REGION:
        raise ValueError(f"bounded fuzz case outside corpus: {case_index}")
    summary = build_fixture(path)
    region_id = BOUNDED_FUZZ_REGIONS.index(region) + 1
    rng = random.Random(seed ^ (region_id << 16) ^ case_index)

    encoded_metadata = metadata_bytes(semantic_metadata())
    directory_offset = 24 + len(encoded_metadata)
    first_tensor = dense_tensors()[0]
    first_name_bytes = first_tensor.name.encode("utf-8")
    first_name_offset = directory_offset + 8
    first_rank_offset = first_name_offset + len(first_name_bytes)
    first_dim_offset = first_rank_offset + 4
    first_type_offset = first_dim_offset + 8 * len(first_tensor.dims)
    first_rel_offset = first_type_offset + 4
    architecture_offset = 24 + encoded_metadata.index(
        pack_string("qwen4exp")
    ) + 8
    physical_profile_offset = 24 + encoded_metadata.index(
        PHYSICAL_PROFILE.encode("ascii")
    )

    with path.open("r+b") as file:
        if region == "gguf":
            if case_index == 0:
                _fuzz_xor_byte(file, rng.randrange(4), rng)
            elif case_index == 1:
                _fuzz_write(file, 4, "<I", GGUF_VERSION ^
                            (1 << rng.randrange(8)))
            elif case_index == 2:
                _fuzz_write(file, 8, "<Q", 1070)
            elif case_index == 3:
                _fuzz_write(file, 16, "<Q", 109)
            elif case_index == 4:
                _fuzz_xor_byte(file, architecture_offset +
                               rng.randrange(len("qwen4exp")), rng, 6)
            elif case_index == 5:
                _fuzz_xor_byte(file, physical_profile_offset +
                               rng.randrange(len(PHYSICAL_PROFILE)), rng, 6)
            elif case_index == 6:
                _fuzz_xor_byte(file, first_name_offset +
                               rng.randrange(len(first_name_bytes)), rng, 6)
            elif case_index == 7:
                _fuzz_write(file, first_rank_offset, "<I", 5)
            elif case_index == 8:
                _fuzz_write(file, first_dim_offset, "<Q", 0)
            elif case_index == 9:
                _fuzz_write(file, first_type_offset, "<I", GGML_I8)
            elif case_index == 10:
                _fuzz_write(file, first_rel_offset, "<Q", 1)
            else:
                _fuzz_write(file, directory_offset, "<Q", (1 << 64) - 1)
        elif region == "expert":
            base = summary.expert_offset
            if case_index == 0:
                _fuzz_xor_byte(file, base + rng.randrange(8), rng)
            elif case_index == 1:
                _fuzz_write(file, base + 8, "<I", 3)
            elif case_index == 2:
                _fuzz_write(file, base + 12, "<I", EXPERT_HEADER_BYTES - 1)
            elif case_index == 3:
                _fuzz_write(file, base + 16, "<I", EXPERT_FAMILY - 1)
            elif case_index == 4:
                _fuzz_write(file, base + 24, "<I", EXPERT_LAYERS - 1)
            elif case_index == 5:
                _fuzz_write(file, base + 28, "<I", EXPERTS - 1)
            elif case_index == 6:
                _fuzz_write(file, base + 80, "<Q",
                            summary.expert_bytes - OWNER_PAGE_ALIGNMENT)
            elif case_index == 7:
                _fuzz_xor_byte(file, base + EXPERT_MANIFEST_DIGEST_OFFSET +
                               rng.randrange(32), rng)
            elif case_index == 8:
                _fuzz_write(file, base + 160, "<I", 0)
            elif case_index == 9:
                _fuzz_write(file, base + EXPERT_HEADER_BYTES, "<I", 1)
            elif case_index == 10:
                _fuzz_write(file, base + EXPERT_HEADER_BYTES +
                            EXPERT_COMPONENT_OFFSET, "<I", 1)
            else:
                _fuzz_write(file, base + EXPERT_HEADER_BYTES +
                            EXPERT_COMPONENT_OFFSET + 48, "<Q",
                            EXPERT_BLOCK_BYTES)
        else:
            base = summary.ple_offset
            if case_index == 0:
                _fuzz_xor_byte(file, base + rng.randrange(8), rng)
            elif case_index == 1:
                _fuzz_write(file, base + 8, "<I", 2)
            elif case_index == 2:
                _fuzz_write(file, base + 12, "<I", PLE_HEADER_BYTES - 1)
            elif case_index == 3:
                _fuzz_write(file, base + 16, "<I", EXPERT_FAMILY - 1)
            elif case_index == 4:
                _fuzz_write(file, base + 20, "<I", len(PLE_PRIMES) - 1)
            elif case_index == 5:
                _fuzz_xor_byte(file, base + 24 + rng.randrange(24), rng, 6)
            elif case_index == 6:
                _fuzz_xor_byte(file, base + 56 + rng.randrange(24), rng, 6)
            elif case_index == 7:
                _fuzz_xor_byte(file, base + 88 + rng.randrange(24), rng, 6)
            elif case_index == 8:
                _fuzz_write(file, base + 120, "<I", PLE_CODEC_VERSION + 1)
            elif case_index == 9:
                _fuzz_write(file, base + 144, "<Q",
                            PLE_ROWS - PLE_ROW_ALIGNMENT)
            elif case_index == 10:
                _fuzz_xor_byte(file, base + PLE_MANIFEST_DIGEST_OFFSET +
                               rng.randrange(32), rng)
            else:
                _fuzz_xor_byte(file, base + PLE_HEADER_BYTES +
                               rng.randrange(32), rng)
        file.flush()
    return summary


def self_check(path: Path) -> FixtureSummary:
    first = build_fixture(path)
    if first.tensor_count != 1069 or first.dense_count != 1067:
        raise AssertionError(f"physical inventory mismatch: {first}")
    if (first.expert_offset % OWNER_PAGE_ALIGNMENT or
            first.ple_offset % PLE_PAGE_ALIGNMENT):
        raise AssertionError("store owner extent is not page aligned")
    if first.expert_offset + first.expert_bytes > first.ple_offset:
        raise AssertionError("ExpertMajor and PLE owner extents overlap")
    if first.ple_offset + first.ple_bytes != first.file_bytes:
        raise AssertionError("PLE is not the isolated final owner extent")
    second = build_fixture(path)
    if first.structural_sha256 != second.structural_sha256:
        raise AssertionError("fixture structural digest is nondeterministic")
    return second


MUTATIONS = (
    "architecture", "source_architecture", "profile", "physical_profile",
    "revision", "context", "layer_pattern", "top_k", "expert_count",
    "gr_rank", "qsa_query_heads", "qsa_kv_heads", "qsa_head_dim",
    "qsa_rotary_dim", "qsa_index_query_heads", "qsa_index_kv_heads",
    "qsa_index_head_dim", "gdn_key_heads", "gdn_value_heads",
    "gdn_key_dim", "gdn_value_dim", "ple_insertion_layer",
    "ple_source_layer", "ple_runtime_layer", "ple_rows",
    "ple_manifest_rows",
    "ple_multiplier_high32", "ple_head_prime", "ple_head_offset",
    "ple_hash_version",
    "u64_wrong_type", "tokenizer_digest", "template_digest", "special_id",
    "mtp_policy", "duplicate_kv", "missing_metadata", "extra_metadata",
    "forbidden_metadata", "missing_tensor", "extra_tensor",
    "canonical_routed", "vision_tensor", "mtp_tensor",
    "second_expert_store", "second_ple_store", "duplicate_tensor",
    "tensor_rank", "tensor_dimension", "tensor_type",
    "dimension_overflow", "offset_overflow", "gguf_bad_magic",
    "gguf_bad_version", "truncated_file", "unaligned_tensor",
    "exact_overlap", "page_isolation", "unaligned_store", "store_overlap",
    "truncated_expert_store", "expert_family", "expert_header_version",
    "expert_component_offset", "expert_component_length",
    "expert_manifest_digest", "truncated_ple_store", "ple_header_version",
    "ple_row_width", "ple_page_stride", "ple_manifest_digest",
)


def _summary_json(summary: FixtureSummary) -> str:
    value = dataclasses.asdict(summary)
    value["path"] = str(value["path"])
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path,
                        help="write or replace one sparse fixture")
    parser.add_argument("--mutation", choices=MUTATIONS)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if not args.write and not args.self_check:
        parser.error("one of --write or --self-check is required")
    if args.mutation and not args.write:
        parser.error("--mutation requires --write")

    if args.write:
        summary = build_fixture(args.write, args.mutation)
        print(_summary_json(summary), end="")
        return 0
    with tempfile.TemporaryDirectory(prefix="q4exp-gguf-fixture-") as tmp:
        summary = self_check(Path(tmp) / "qwen4exp-structural.gguf")
        print(_summary_json(summary), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
