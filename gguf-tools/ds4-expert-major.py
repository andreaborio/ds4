#!/usr/bin/env python3
"""Build and verify DS4 ExpertMajor v2 GGUFs.

The converter changes storage only: non-routed tensors and every GGUF metadata
record are copied byte-for-byte, while each routed layer becomes a sequence of
complete expert records (gate, up, down). The opaque store is self-describing;
DS4 reconstructs the canonical logical tensor inventory at load time.

The ``--dspark-support`` interface is reserved for a support GGUF generated
from the final 0731 shards. It currently fails closed because that artifact and
its SHA-256 pin do not exist. The older 8b3a... GGUF remains useful only as a
structural reference; matching its name, shape, or hash does not establish
final-checkpoint provenance.
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import errno
import hashlib
import json
import os
import re
import shutil
import stat as statlib
import struct
import sys
from pathlib import Path
from typing import BinaryIO, Iterator, Protocol


MAGIC = b"GGUF"
STORE_MAGIC = b"DS4EXPV2"
STORE_TENSOR = "ds4.expert_major.v2"
DSPARK_STORE_TENSOR = "ds4.dspark.expert_major.v2"
DSPARK_PREVIEW_REFERENCE_SHA256 = bytes.fromhex(
    "8b3adf5942bec22ae2ea867cd7079cf13530ba83ffcffaf00f5de48664a1a34e"
)
# Set only after generating and validating a support GGUF from the final 0731
# checkpoint shards 46-48. The preview reference above is deliberately not an
# accepted publication pin.
DSPARK_0731_FINAL_SUPPORT_SHA256: bytes | None = None
DSPARK_0731_PROVENANCE = {
    "dspark.source.revision":
        "7872f01b1d1fe23eabc4c98b48bffcef5a386062",
    "dspark.source.config_sha256":
        "6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023",
    "dspark.source.index_sha256":
        "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b",
    "dspark.source.shard46_sha256":
        "5db924ca907e0d93acd975bd5079c3662717f9ac709f23d079bd8f816d29d9dd",
    "dspark.source.shard47_sha256":
        "62816173f9f6e136b20b48e3b6f16613ac9ea02b5603f636928b253244a548bd",
    "dspark.source.shard48_sha256":
        "cc43742bd24ae6bcdea343a91442f6f66aed2cfebcc6b235470204851ce2f8a9",
}
DSPARK_0731_LAYER_COUNT = 43
DSPARK_0731_EMBEDDING = 4096
DSPARK_0731_VOCAB = 129280
DSPARK_0731_EXPERT_FF = 2048
DSPARK_0731_EXPERT_COUNT = 256
DSPARK_0731_EXPERT_USED = 6
STORE_VERSION = 2
STORE_FAMILY_DEEPSEEK4 = 1
STORE_FAMILY_GLM_DSA = 2
STORE_FAMILY_QWEN35_MOE = 3
STORE_STORAGE_GGML = 0
STORE_STORAGE_MLX_AFFINE4 = 1
STORE_FAMILIES = {
    STORE_FAMILY_DEEPSEEK4,
    STORE_FAMILY_GLM_DSA,
    STORE_FAMILY_QWEN35_MOE,
}
STORE_MAX_ROUTED_LAYERS = 79
STORE_MAX_MODEL_LAYER = 127
STORE_HEADER_BYTES = 256
STORE_LAYER_BYTES = 224
STORE_COMPONENT_BYTES = 56
STORE_COMPONENT_OFFSET = 32
STORE_ALIGNMENT = 4096
MANIFEST_DIGEST_OFFSET = 168
IO_BYTES = 8 * 1024 * 1024

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

# GGML type -> (block elements, block bytes). This is the v3 table DS4 uses.
TYPE_LAYOUT = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20),
    6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 40),
    10: (256, 84), 11: (256, 110), 12: (256, 144),
    13: (256, 176), 14: (256, 210), 15: (256, 292),
    16: (256, 66), 17: (256, 74), 18: (256, 98),
    19: (256, 110), 20: (256, 50), 21: (256, 110),
    22: (256, 82), 23: (256, 136), 24: (1, 1),
    25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8),
    29: (256, 56), 30: (1, 2),
}
ROUTED_TYPES = {
    10, 12, 13, 14, 16, 17, 18, 23,
}  # Q2_K, Q4_K, Q5_K, Q6_K, and the admitted IQ2/IQ3/IQ4 variants
TYPE_NAME = {
    10: "Q2_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 23: "IQ4_XS",
    24: "I8",
}
ROLE_NAME = ("gate", "up", "down")
ROUTED_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")
DSPARK_ROUTED_RE = re.compile(
    r"^mtp\.(\d+)\.ffn_(gate|up|down)_exps\.weight$"
)
DSPARK_STAGE_COUNT = 3
DSPARK_PROVENANCE_KEYS = tuple(DSPARK_0731_PROVENANCE)
DSPARK_METADATA_KEYS = (
    "dspark.block_size",
    "dspark.markov_rank",
    "dspark.noise_token_id",
    "dspark.target_layer_ids",
    "dspark.stage_count",
    "dspark.n_layers",
    *DSPARK_PROVENANCE_KEYS,
)
DSPARK_BLOCK_SUFFIXES = frozenset({
    "hc_attn_base.weight", "hc_attn_fn.weight", "hc_attn_scale.weight",
    "attn_sinks.weight", "attn_q_a.weight", "attn_q_a_norm.weight",
    "attn_q_b.weight", "attn_kv.weight", "attn_kv_a_norm.weight",
    "attn_output_a.weight", "attn_output_b.weight", "attn_norm.weight",
    "hc_ffn_base.weight", "hc_ffn_fn.weight", "hc_ffn_scale.weight",
    "ffn_gate_inp.weight", "exp_probs_b.bias", "ffn_norm.weight",
    "ffn_gate_exps.weight", "ffn_up_exps.weight",
    "ffn_down_exps.weight", "ffn_gate_shexp.weight",
    "ffn_up_shexp.weight", "ffn_down_shexp.weight",
})
DSPARK_STAGE0_SUFFIXES = frozenset({
    "main_proj.weight", "main_norm.weight",
})
DSPARK_FINAL_SUFFIXES = frozenset({
    "norm.weight", "hc_head_base.weight", "hc_head_fn.weight",
    "hc_head_scale.weight", "markov_head.markov_w1.weight",
    "markov_head.markov_w2.weight", "confidence_head.proj.weight",
})


class FormatError(RuntimeError):
    pass


class Digest(Protocol):
    def update(self, data: bytes, /) -> object:
        ...


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise FormatError("alignment must be positive")
    return (value + alignment - 1) // alignment * alignment


def read_exact(file: BinaryIO, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        chunk = file.read(size - len(result))
        if not chunk:
            raise FormatError("truncated GGUF metadata or tensor directory")
        result.extend(chunk)
    return bytes(result)


def u32(file: BinaryIO) -> int:
    return struct.unpack("<I", read_exact(file, 4))[0]


def u64(file: BinaryIO) -> int:
    return struct.unpack("<Q", read_exact(file, 8))[0]


def gguf_string(file: BinaryIO) -> str:
    size = u64(file)
    if size > 1 << 30:
        raise FormatError(f"unreasonable GGUF string length: {size}")
    return read_exact(file, size).decode("utf-8")


def pack_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def scalar_size(value_type: int) -> int:
    if value_type in (GGUF_UINT8, GGUF_INT8, GGUF_BOOL):
        return 1
    if value_type in (GGUF_UINT16, GGUF_INT16):
        return 2
    if value_type in (GGUF_UINT32, GGUF_INT32, GGUF_FLOAT32):
        return 4
    if value_type in (GGUF_UINT64, GGUF_INT64, GGUF_FLOAT64):
        return 8
    return 0


def skip_value(file: BinaryIO, value_type: int, depth: int = 0) -> None:
    if depth > 8:
        raise FormatError("GGUF array nesting is too deep")
    size = scalar_size(value_type)
    if size:
        file.seek(size, os.SEEK_CUR)
        return
    if value_type == GGUF_STRING:
        file.seek(u64(file), os.SEEK_CUR)
        return
    if value_type == GGUF_ARRAY:
        item_type = u32(file)
        count = u64(file)
        item_size = scalar_size(item_type)
        if item_size:
            file.seek(item_size * count, os.SEEK_CUR)
            return
        for _ in range(count):
            skip_value(file, item_type, depth + 1)
        return
    raise FormatError(f"unsupported GGUF metadata type {value_type}")


def read_metadata_value(file: BinaryIO, value_type: int):
    if value_type == GGUF_STRING:
        return gguf_string(file), None
    formats = {
        GGUF_UINT8: "<B", GGUF_INT8: "<b", GGUF_UINT16: "<H",
        GGUF_INT16: "<h", GGUF_UINT32: "<I", GGUF_INT32: "<i",
        GGUF_FLOAT32: "<f", GGUF_BOOL: "<?", GGUF_UINT64: "<Q",
        GGUF_INT64: "<q", GGUF_FLOAT64: "<d",
    }
    fmt = formats.get(value_type)
    if fmt:
        return (struct.unpack(fmt, read_exact(file, struct.calcsize(fmt)))[0],
                None)
    if value_type == GGUF_ARRAY:
        item_type = u32(file)
        count = u64(file)
        if count > 1 << 20:
            raise FormatError(f"unreasonable GGUF metadata array length: {count}")
        values = []
        for _ in range(count):
            value, nested_type = read_metadata_value(file, item_type)
            if nested_type is not None:
                raise FormatError("nested metadata arrays are not supported")
            values.append(value)
        return tuple(values), item_type
    skip_value(file, value_type)
    return None, None


@dataclasses.dataclass
class Tensor:
    name: str
    dims: tuple[int, ...]
    ggml_type: int
    rel_offset: int
    size: int
    abs_offset: int = 0
    new_rel_offset: int = 0


@dataclasses.dataclass
class GGUF:
    path: Path
    size: int
    version: int
    n_kv: int
    alignment: int
    kv_raw: bytes
    metadata: dict[str, object]
    metadata_types: dict[str, tuple[int, int | None]]
    metadata_records: tuple[tuple[str, bytes], ...]
    tensors: list[Tensor]
    data_offset: int


@dataclasses.dataclass(frozen=True)
class FDIdentity:
    device: int
    inode: int
    size: int
    mtime_ns: int


@dataclasses.dataclass
class Component:
    role: int
    tensor: Tensor
    expert_bytes: int
    record_offset: int
    block_elements: int | None = None


@dataclasses.dataclass
class Layer:
    index: int
    expert_count: int
    record_bytes: int
    data_offset: int
    data_size: int
    components: tuple[Component, Component, Component]


@dataclasses.dataclass
class StorePlan:
    source: GGUF
    family: int
    storage_format: int
    group_size: int
    layer_count: int
    expert_count: int
    expert_used_count: int
    source_tensor_count: int
    descriptor_bytes: bytes
    data_offset: int
    data_size: int
    store_size: int
    layers: list[Layer]


@dataclasses.dataclass
class SafeTensorShard:
    path: Path
    fd: int
    data_offset: int
    tensors: dict[str, dict[str, object]]


class MLXAffineSource:
    """Minimal mmap-free reader for the routed MLX safetensor slices."""

    def __init__(self, model_dir: Path):
        self.model_dir = model_dir.resolve()
        index_path = self.model_dir / "model.safetensors.index.json"
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            self.weight_map = dict(index["weight_map"])
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise FormatError(f"invalid MLX safetensor index: {exc}") from exc
        self.shards: dict[str, SafeTensorShard] = {}

    def close(self) -> None:
        for shard in self.shards.values():
            os.close(shard.fd)
        self.shards.clear()

    def _shard(self, name: str) -> SafeTensorShard:
        cached = self.shards.get(name)
        if cached is not None:
            return cached
        path = (self.model_dir / name).resolve()
        fd = os.open(path, os.O_RDONLY)
        try:
            header_bytes = struct.unpack("<Q", pread_exact(fd, 8, 0))[0]
            if header_bytes <= 0 or header_bytes > 1 << 30:
                raise FormatError(f"invalid safetensor header size in {path}")
            tensors = json.loads(
                pread_exact(fd, header_bytes, 8).decode("utf-8")
            )
        except BaseException:
            os.close(fd)
            raise
        shard = SafeTensorShard(path, fd, 8 + header_bytes, tensors)
        self.shards[name] = shard
        return shard

    def tensor(self, key: str, dtype: str,
               shape: tuple[int, ...]) -> tuple[SafeTensorShard, int, int]:
        shard_name = self.weight_map.get(key)
        if not isinstance(shard_name, str):
            raise FormatError(f"MLX tensor is missing: {key}")
        shard = self._shard(shard_name)
        entry = shard.tensors.get(key)
        if not isinstance(entry, dict) or entry.get("dtype") != dtype or \
                tuple(entry.get("shape", ())) != shape:
            raise FormatError(
                f"MLX tensor geometry differs for {key}: {entry}"
            )
        offsets = entry.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2 or \
                not all(isinstance(value, int) for value in offsets) or \
                offsets[0] < 0 or offsets[1] < offsets[0]:
            raise FormatError(f"invalid safetensor offsets for {key}")
        return shard, shard.data_offset + offsets[0], offsets[1] - offsets[0]

    def expert_bytes(self, key: str, dtype: str,
                     shape: tuple[int, ...], expert: int) -> bytes:
        shard, offset, size = self.tensor(key, dtype, shape)
        experts = shape[0]
        if expert < 0 or expert >= experts or size % experts:
            raise FormatError(f"invalid expert slice for {key}")
        stride = size // experts
        return pread_exact(shard.fd, stride, offset + expert * stride)


def tensor_nbytes(ggml_type: int, dims: tuple[int, ...]) -> int:
    layout = TYPE_LAYOUT.get(ggml_type)
    if not layout:
        raise FormatError(f"unsupported GGML tensor type {ggml_type}")
    elements = 1
    for dim in dims:
        if dim <= 0:
            raise FormatError("zero-sized GGUF tensor")
        elements *= dim
    block_elements, block_bytes = layout
    if elements % block_elements:
        raise FormatError(
            f"tensor element count {elements} is not aligned for type {ggml_type}"
        )
    return elements // block_elements * block_bytes


def lexical_absolute(path: Path) -> Path:
    """Make a path absolute without following its final symlink."""
    return Path(os.path.abspath(os.fspath(path)))


def fd_identity(fd: int, label: str) -> FDIdentity:
    info = os.fstat(fd)
    return stat_identity(info, label)


def stat_identity(info: os.stat_result, label: str) -> FDIdentity:
    if not statlib.S_ISREG(info.st_mode):
        raise FormatError(f"{label} is not a regular file")
    return FDIdentity(
        info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
    )


def open_input(path: Path, label: str) -> tuple[Path, int, FDIdentity]:
    absolute = lexical_absolute(path)
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    fd = os.open(absolute, flags)
    try:
        identity = fd_identity(fd, label)
    except BaseException:
        os.close(fd)
        raise
    return absolute, fd, identity


def require_fd_unchanged(fd: int, initial: FDIdentity, label: str) -> None:
    if fd_identity(fd, label) != initial:
        raise FormatError(f"{label} changed while its descriptor was in use")


def load_gguf_fd(path: Path, fd: int) -> GGUF:
    path = lexical_absolute(path)
    size = fd_identity(fd, f"GGUF {path}").size
    os.lseek(fd, 0, os.SEEK_SET)
    with os.fdopen(fd, "rb", buffering=0, closefd=False) as file:
        if read_exact(file, 4) != MAGIC:
            raise FormatError(f"{path} is not a GGUF")
        version = u32(file)
        n_tensors = u64(file)
        n_kv = u64(file)
        if version != 3:
            raise FormatError(f"only GGUF v3 is supported, got v{version}")
        kv_start = file.tell()
        metadata: dict[str, object] = {}
        metadata_types: dict[str, tuple[int, int | None]] = {}
        metadata_ranges: list[tuple[str, int, int]] = []
        alignment = 32
        wanted = {
            "general.architecture", "general.name", "general.alignment",
            "deepseek4.block_count", "deepseek4.embedding_length",
            "deepseek4.vocab_size", "deepseek4.expert_count",
            "deepseek4.expert_used_count",
            "deepseek4.expert_feed_forward_length",
            "glm-dsa.block_count", "glm-dsa.expert_count",
            "glm-dsa.expert_used_count",
            "glm-dsa.leading_dense_block_count",
            "glm-dsa.nextn_predict_layers",
            "qwen35moe.block_count", "qwen35moe.expert_count",
            "qwen35moe.expert_used_count",
            *DSPARK_METADATA_KEYS,
        }
        for _ in range(n_kv):
            record_start = file.tell()
            key = gguf_string(file)
            value_type = u32(file)
            if key in wanted:
                if key in metadata:
                    raise FormatError(f"duplicate GGUF metadata key: {key}")
                value, item_type = read_metadata_value(file, value_type)
                metadata[key] = value
                metadata_types[key] = (value_type, item_type)
                if key == "general.alignment":
                    alignment = int(value)
            else:
                skip_value(file, value_type)
            metadata_ranges.append((key, record_start, file.tell()))
        tensor_start = file.tell()
        file.seek(kv_start)
        kv_raw = read_exact(file, tensor_start - kv_start)
        metadata_records = tuple(
            (key, kv_raw[start - kv_start:end - kv_start])
            for key, start, end in metadata_ranges
        )
        file.seek(tensor_start)
        tensors: list[Tensor] = []
        for _ in range(n_tensors):
            name = gguf_string(file)
            ndim = u32(file)
            if ndim < 1 or ndim > 4:
                raise FormatError(f"unsupported rank {ndim} for {name}")
            dims = tuple(u64(file) for _ in range(ndim))
            ggml_type = u32(file)
            rel_offset = u64(file)
            tensors.append(Tensor(name, dims, ggml_type, rel_offset,
                                  tensor_nbytes(ggml_type, dims)))
        directory_end = file.tell()
        data_offset = align_up(directory_end, alignment)
    if alignment <= 0 or alignment & (alignment - 1):
        raise FormatError(f"invalid general.alignment: {alignment}")
    expected_rel_offset = 0
    for tensor in tensors:
        expected_rel_offset = align_up(expected_rel_offset, alignment)
        if tensor.rel_offset != expected_rel_offset:
            raise FormatError(
                f"non-canonical tensor offset for {tensor.name}: "
                f"expected {expected_rel_offset}, got {tensor.rel_offset}"
            )
        tensor.abs_offset = data_offset + tensor.rel_offset
        if tensor.abs_offset > size or tensor.size > size - tensor.abs_offset:
            raise FormatError(f"tensor points outside GGUF: {tensor.name}")
        expected_rel_offset += tensor.size
    expected_size = data_offset + align_up(expected_rel_offset, alignment)
    if size != expected_size:
        raise FormatError(
            f"GGUF byte range mismatch: expected {expected_size}, got {size}"
        )
    require_zero_range(
        fd, directory_end, data_offset - directory_end,
        "GGUF pre-data padding",
    )
    previous_end = data_offset
    for tensor in tensors:
        require_zero_range(
            fd, previous_end, tensor.abs_offset - previous_end,
            f"GGUF tensor padding before {tensor.name}",
        )
        previous_end = tensor.abs_offset + tensor.size
    require_zero_range(
        fd, previous_end, expected_size - previous_end,
        "GGUF terminal padding",
    )
    return GGUF(path, size, version, n_kv, alignment, kv_raw, metadata,
                metadata_types, metadata_records, tensors, data_offset)


def load_gguf(path: Path) -> GGUF:
    absolute, fd, identity = open_input(path, "GGUF input")
    try:
        result = load_gguf_fd(absolute, fd)
        require_fd_unchanged(fd, identity, "GGUF input")
        return result
    finally:
        os.close(fd)


def routed_inventory(gguf: GGUF, pattern: re.Pattern[str] = ROUTED_RE
                     ) -> dict[int, dict[int, Tensor]]:
    result: dict[int, dict[int, Tensor]] = {}
    role_index = {name: index for index, name in enumerate(ROLE_NAME)}
    for tensor in gguf.tensors:
        match = pattern.fullmatch(tensor.name)
        if not match:
            continue
        layer = int(match.group(1))
        role = role_index[match.group(2)]
        if role in result.setdefault(layer, {}):
            raise FormatError(f"duplicate routed tensor {tensor.name}")
        result[layer][role] = tensor
    return result


def dspark_metadata_records(source: GGUF) -> tuple[bytes, ...]:
    """Validate the standalone support contract and return raw DSpark KVs."""
    expected_keys = {
        "general.architecture", "general.name", "general.alignment",
        *DSPARK_METADATA_KEYS,
    }
    actual_keys = [key for key, _ in source.metadata_records]
    if len(actual_keys) != len(expected_keys) or set(actual_keys) != expected_keys:
        raise FormatError(
            "DSpark support metadata inventory mismatch; "
            f"expected={sorted(expected_keys)} actual={sorted(actual_keys)}"
        )
    expected_types = {
        "general.architecture": (GGUF_STRING, None),
        "general.name": (GGUF_STRING, None),
        "general.alignment": (GGUF_UINT32, None),
        "dspark.block_size": (GGUF_UINT32, None),
        "dspark.markov_rank": (GGUF_UINT32, None),
        "dspark.noise_token_id": (GGUF_UINT32, None),
        "dspark.target_layer_ids": (GGUF_ARRAY, GGUF_UINT32),
        "dspark.stage_count": (GGUF_UINT32, None),
        "dspark.n_layers": (GGUF_UINT32, None),
        **{key: (GGUF_STRING, None) for key in DSPARK_PROVENANCE_KEYS},
    }
    if source.metadata_types != expected_types:
        raise FormatError("DSpark support metadata types do not match the contract")
    if source.metadata.get("general.architecture") != "deepseek4-dspark":
        raise FormatError("DSpark support architecture must be deepseek4-dspark")
    if source.metadata.get("general.name") != \
            "DeepSeek V4 Flash DSpark support":
        raise FormatError("DSpark support model name does not match the contract")
    try:
        block_size = int(source.metadata["dspark.block_size"])
        markov_rank = int(source.metadata["dspark.markov_rank"])
        noise_token = int(source.metadata["dspark.noise_token_id"])
        target_layers = tuple(source.metadata["dspark.target_layer_ids"])
        stage_count = int(source.metadata["dspark.stage_count"])
        n_layers = int(source.metadata["dspark.n_layers"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FormatError("DSpark support metadata is incomplete") from exc
    if (block_size != 5 or markov_rank != 256 or noise_token != 128799 or
            target_layers != (40, 41, 42) or
            stage_count != DSPARK_STAGE_COUNT or n_layers != stage_count):
        raise FormatError("DSpark support metadata values are outside the contract")
    actual_provenance = {
        key: source.metadata.get(key) for key in DSPARK_PROVENANCE_KEYS
    }
    if actual_provenance != DSPARK_0731_PROVENANCE:
        raise FormatError(
            "DSpark support provenance does not match the independently "
            "pinned final 0731 source contract"
        )
    by_key = dict(source.metadata_records)
    return tuple(by_key[key] for key in DSPARK_METADATA_KEYS)


def validate_dspark_target_0731(target: StorePlan) -> None:
    source = target.source
    expected = {
        "deepseek4.block_count": DSPARK_0731_LAYER_COUNT,
        "deepseek4.embedding_length": DSPARK_0731_EMBEDDING,
        "deepseek4.vocab_size": DSPARK_0731_VOCAB,
        "deepseek4.expert_count": DSPARK_0731_EXPERT_COUNT,
        "deepseek4.expert_used_count": DSPARK_0731_EXPERT_USED,
        "deepseek4.expert_feed_forward_length": DSPARK_0731_EXPERT_FF,
    }
    for key, value in expected.items():
        if (source.metadata_types.get(key) != (GGUF_UINT32, None) or
                source.metadata.get(key) != value):
            raise FormatError(
                f"target metadata does not match final 0731: {key}={value}"
            )
    if (target.family != STORE_FAMILY_DEEPSEEK4 or
            target.layer_count != DSPARK_0731_LAYER_COUNT or
            target.expert_count != DSPARK_0731_EXPERT_COUNT or
            target.expert_used_count != DSPARK_0731_EXPERT_USED):
        raise FormatError("target ExpertMajor plan does not match final 0731")
    expected_types = (16, 16, 10)
    expected_dims = (
        (DSPARK_0731_EMBEDDING, DSPARK_0731_EXPERT_FF,
         DSPARK_0731_EXPERT_COUNT),
        (DSPARK_0731_EMBEDDING, DSPARK_0731_EXPERT_FF,
         DSPARK_0731_EXPERT_COUNT),
        (DSPARK_0731_EXPERT_FF, DSPARK_0731_EMBEDDING,
         DSPARK_0731_EXPERT_COUNT),
    )
    for layer in target.layers:
        for role, component in enumerate(layer.components):
            if (component.tensor.ggml_type != expected_types[role] or
                    component.tensor.dims != expected_dims[role]):
                raise FormatError(
                    "target routed tensor does not match final 0731: "
                    f"{component.tensor.name}"
                )


def reject_target_dspark_namespace(source: GGUF) -> None:
    for key, _ in source.metadata_records:
        lowered = key.lower()
        if (lowered == "dspark" or lowered.startswith("dspark.") or
                lowered.startswith("deepseek4.dspark") or
                ".dspark." in lowered):
            raise FormatError(
                f"target GGUF already owns a DSpark metadata alias: {key}"
            )
    for tensor in source.tensors:
        lowered = tensor.name.lower()
        if (lowered.startswith("mtp.") or "dspark" in lowered or
                lowered == DSPARK_STORE_TENSOR):
            raise FormatError(
                f"target GGUF already owns a DSpark tensor alias: {tensor.name}"
            )


def validate_dspark_static_tensors(source: GGUF, expert_count: int) -> None:
    """Validate the exact shape/type contract of all 72 final 0731 tensors."""
    if expert_count != DSPARK_0731_EXPERT_COUNT:
        raise FormatError("DSpark support expert count is not final 0731")
    by_name = {tensor.name: tensor for tensor in source.tensors}

    def tensor(stage: int, suffix: str) -> Tensor:
        return by_name[f"mtp.{stage}.{suffix}"]

    def expect(stage: int, suffix: str, dims: tuple[int, ...],
               ggml_type: int) -> None:
        candidate = tensor(stage, suffix)
        if candidate.dims != dims or candidate.ggml_type != ggml_type:
            raise FormatError(
                f"DSpark static tensor contract mismatch: {candidate.name} "
                f"type={candidate.ggml_type} dims={candidate.dims}; "
                f"expected type={ggml_type} dims={dims}"
            )

    block_contract = {
        "hc_attn_base.weight": ((24,), 0),
        "hc_attn_fn.weight": ((16384, 24), 1),
        "hc_attn_scale.weight": ((3,), 0),
        "attn_sinks.weight": ((64,), 0),
        "attn_q_a.weight": ((4096, 1024), 8),
        "attn_q_a_norm.weight": ((1024,), 0),
        "attn_q_b.weight": ((1024, 32768), 8),
        "attn_kv.weight": ((4096, 512), 8),
        "attn_kv_a_norm.weight": ((512,), 0),
        "attn_output_a.weight": ((4096, 8192), 8),
        "attn_output_b.weight": ((8192, 4096), 8),
        "attn_norm.weight": ((4096,), 0),
        "hc_ffn_base.weight": ((24,), 0),
        "hc_ffn_fn.weight": ((16384, 24), 1),
        "hc_ffn_scale.weight": ((3,), 0),
        "ffn_gate_inp.weight": ((4096, 256), 8),
        "exp_probs_b.bias": ((256,), 0),
        "ffn_norm.weight": ((4096,), 0),
        "ffn_gate_shexp.weight": ((4096, 2048), 8),
        "ffn_up_shexp.weight": ((4096, 2048), 8),
        "ffn_down_shexp.weight": ((2048, 4096), 8),
    }
    for stage in range(DSPARK_STAGE_COUNT):
        for suffix, (dims, ggml_type) in block_contract.items():
            expect(stage, suffix, dims, ggml_type)

    expect(0, "main_proj.weight", (12288, 4096), 8)
    expect(0, "main_norm.weight", (4096,), 0)
    final = DSPARK_STAGE_COUNT - 1
    expect(final, "norm.weight", (4096,), 0)
    expect(final, "hc_head_base.weight", (4,), 0)
    expect(final, "hc_head_fn.weight", (16384, 4), 1)
    expect(final, "hc_head_scale.weight", (1,), 0)
    expect(final, "markov_head.markov_w1.weight", (256, 129280), 8)
    expect(final, "markov_head.markov_w2.weight", (256, 129280), 8)
    expect(final, "confidence_head.proj.weight", (4352, 1), 8)


def make_dspark_store_plan(source: GGUF, target: StorePlan) -> StorePlan:
    """Plan the routed part of the three-stage final DSpark support GGUF."""
    dspark_metadata_records(source)
    validate_dspark_target_0731(target)
    if source.alignment != target.source.alignment:
        raise FormatError("DSpark support and target GGUF alignment differ")
    target_layers = tuple(source.metadata["dspark.target_layer_ids"])
    target_block_count = int(target.source.metadata["deepseek4.block_count"])
    if any(layer < 0 or layer >= target_block_count for layer in target_layers):
        raise FormatError("DSpark target layer metadata is outside the target model")
    expected_names: set[str] = set()
    for stage in range(DSPARK_STAGE_COUNT):
        suffixes = set(DSPARK_BLOCK_SUFFIXES)
        if stage == 0:
            suffixes.update(DSPARK_STAGE0_SUFFIXES)
        if stage == DSPARK_STAGE_COUNT - 1:
            suffixes.update(DSPARK_FINAL_SUFFIXES)
        expected_names.update(f"mtp.{stage}.{suffix}" for suffix in suffixes)
    actual_names = [tensor.name for tensor in source.tensors]
    if len(actual_names) != len(expected_names) or set(actual_names) != expected_names:
        missing = sorted(expected_names - set(actual_names))
        extra = sorted(set(actual_names) - expected_names)
        raise FormatError(
            "DSpark tensor inventory mismatch; "
            f"missing={missing} extra={extra}"
        )
    validate_dspark_static_tensors(source, target.expert_count)

    inventory = routed_inventory(source, DSPARK_ROUTED_RE)
    if set(inventory) != set(range(DSPARK_STAGE_COUNT)):
        raise FormatError("DSpark routed stages must be exactly mtp.0..2")
    data_offset = align_up(
        STORE_HEADER_BYTES + DSPARK_STAGE_COUNT * STORE_LAYER_BYTES,
        STORE_ALIGNMENT,
    )
    cursor = data_offset
    layers: list[Layer] = []
    routed_contract: tuple[tuple[tuple[int, ...], int], ...] | None = None
    for stage in range(DSPARK_STAGE_COUNT):
        by_role = inventory[stage]
        if set(by_role) != {0, 1, 2}:
            raise FormatError(f"DSpark stage {stage} does not have gate/up/down")
        components: list[Component] = []
        record_offset = 0
        for role in range(3):
            tensor = by_role[role]
            expected_type = (16, 16, 10)[role]
            expected_dims = (
                (4096, 2048, 256),
                (4096, 2048, 256),
                (2048, 4096, 256),
            )[role]
            if (tensor.ggml_type != expected_type or
                    tensor.dims != expected_dims):
                raise FormatError(
                    f"DSpark routed tensor does not match final 0731: "
                    f"{tensor.name} type={tensor.ggml_type} "
                    f"dims={tensor.dims}"
                )
            if tensor.dims[2] != target.expert_count or \
                    tensor.size % target.expert_count:
                raise FormatError(
                    f"DSpark expert dimension mismatch: {tensor.name}"
                )
            expert_bytes = tensor.size // target.expert_count
            components.append(Component(
                role, tensor, expert_bytes, record_offset
            ))
            record_offset += expert_bytes
        gate, up, down = (component.tensor for component in components)
        if gate.dims != up.dims or gate.ggml_type != up.ggml_type:
            raise FormatError(f"DSpark gate/up geometry differs at stage {stage}")
        if gate.dims[0] != down.dims[1] or gate.dims[1] != down.dims[0]:
            raise FormatError(f"DSpark gate/down dimensions disagree at stage {stage}")
        current_contract = tuple(
            (component.tensor.dims, component.tensor.ggml_type)
            for component in components
        )
        if routed_contract is None:
            routed_contract = current_contract
        elif current_contract != routed_contract:
            raise FormatError("DSpark routed geometry differs between stages")
        cursor = align_up(cursor, STORE_ALIGNMENT)
        layer_size = record_offset * target.expert_count
        layers.append(Layer(
            stage, target.expert_count, record_offset, cursor, layer_size,
            tuple(components),
        ))
        cursor += layer_size

    descriptor_bytes = b"".join(pack_layer(layer) for layer in layers)
    return StorePlan(
        source=source,
        family=STORE_FAMILY_DEEPSEEK4,
        storage_format=STORE_STORAGE_GGML,
        group_size=0,
        layer_count=DSPARK_STAGE_COUNT,
        expert_count=target.expert_count,
        expert_used_count=target.expert_used_count,
        source_tensor_count=len(source.tensors),
        descriptor_bytes=descriptor_bytes,
        data_offset=data_offset,
        data_size=cursor - data_offset,
        store_size=cursor,
        layers=layers,
    )


def make_store_plan(source: GGUF) -> StorePlan:
    architecture = source.metadata.get("general.architecture")
    try:
        if architecture == "deepseek4":
            family = STORE_FAMILY_DEEPSEEK4
            model_layer_count = int(source.metadata["deepseek4.block_count"])
            expert_count = int(source.metadata["deepseek4.expert_count"])
            expert_used_count = int(source.metadata["deepseek4.expert_used_count"])
            expected_layers = set(range(model_layer_count))
            family_name = "DeepSeek"
        elif architecture == "glm-dsa":
            family = STORE_FAMILY_GLM_DSA
            model_layer_count = int(source.metadata["glm-dsa.block_count"])
            expert_count = int(source.metadata["glm-dsa.expert_count"])
            expert_used_count = int(source.metadata["glm-dsa.expert_used_count"])
            leading_dense = int(
                source.metadata["glm-dsa.leading_dense_block_count"]
            )
            nextn_layers = int(source.metadata["glm-dsa.nextn_predict_layers"])
            if not (0 <= leading_dense < model_layer_count and
                    0 <= nextn_layers < model_layer_count):
                raise FormatError("GLM dense/NextN layer metadata is invalid")
            # Keep the GGUF self-contained. The NextN tail remains in the
            # store even when the current decode graph stops before it.
            expected_layers = set(range(leading_dense, model_layer_count))
            family_name = "GLM"
        elif architecture == "qwen35moe":
            family = STORE_FAMILY_QWEN35_MOE
            model_layer_count = int(source.metadata["qwen35moe.block_count"])
            expert_count = int(source.metadata["qwen35moe.expert_count"])
            expert_used_count = int(
                source.metadata["qwen35moe.expert_used_count"]
            )
            expected_layers = set(range(model_layer_count))
            family_name = "Qwen"
        else:
            raise FormatError(
                "expert-major v2 accepts deepseek4, glm-dsa, and "
                "qwen35moe GGUFs only"
            )
    except (KeyError, TypeError, ValueError) as exc:
        raise FormatError(
            f"{architecture or 'unknown'} layer/expert metadata is incomplete"
        ) from exc

    layer_count = len(expected_layers)
    if not (1 <= model_layer_count <= STORE_MAX_ROUTED_LAYERS and
            1 <= layer_count <= STORE_MAX_ROUTED_LAYERS and
            1 <= expert_count <= 384 and
            1 <= expert_used_count <= expert_count):
        raise FormatError(
            f"{family_name} layer or expert counts are outside v2 limits"
        )

    inventory = routed_inventory(source)
    if set(inventory) != expected_layers:
        missing = sorted(expected_layers - set(inventory))
        extra = sorted(set(inventory) - expected_layers)
        raise FormatError(f"routed layer inventory mismatch; missing={missing} extra={extra}")

    data_offset = align_up(STORE_HEADER_BYTES + layer_count * STORE_LAYER_BYTES,
                           STORE_ALIGNMENT)
    cursor = data_offset
    layers: list[Layer] = []
    for layer_index in sorted(expected_layers):
        by_role = inventory[layer_index]
        if set(by_role) != {0, 1, 2}:
            raise FormatError(f"layer {layer_index} does not have gate/up/down")
        components: list[Component] = []
        record_offset = 0
        for role in range(3):
            tensor = by_role[role]
            if tensor.ggml_type not in ROUTED_TYPES or len(tensor.dims) != 3:
                raise FormatError(
                    f"unsupported routed layout: {tensor.name} type={tensor.ggml_type} dims={tensor.dims}"
                )
            if tensor.dims[2] != expert_count or tensor.size % expert_count:
                raise FormatError(f"expert dimension mismatch: {tensor.name}")
            expert_bytes = tensor.size // expert_count
            components.append(Component(role, tensor, expert_bytes, record_offset))
            record_offset += expert_bytes
        gate, up, down = (component.tensor for component in components)
        if gate.dims != up.dims or gate.ggml_type != up.ggml_type:
            raise FormatError(f"gate/up geometry differs at layer {layer_index}")
        if gate.dims[0] != down.dims[1] or gate.dims[1] != down.dims[0] or \
                down.dims[2] != expert_count:
            raise FormatError(f"gate/down dimensions disagree at layer {layer_index}")
        cursor = align_up(cursor, STORE_ALIGNMENT)
        layer_size = record_offset * expert_count
        layers.append(Layer(layer_index, expert_count, record_offset,
                            cursor, layer_size, tuple(components)))
        cursor += layer_size

    descriptor_bytes = b"".join(pack_layer(layer) for layer in layers)
    return StorePlan(
        source=source,
        family=family,
        storage_format=STORE_STORAGE_GGML,
        group_size=0,
        layer_count=layer_count,
        expert_count=expert_count,
        expert_used_count=expert_used_count,
        source_tensor_count=len(source.tensors),
        descriptor_bytes=descriptor_bytes,
        data_offset=data_offset,
        data_size=cursor - data_offset,
        store_size=cursor,
        layers=layers,
    )


def pack_layer(layer: Layer) -> bytes:
    result = bytearray(STORE_LAYER_BYTES)
    struct.pack_into("<IIQQQ", result, 0, layer.index, layer.expert_count,
                     layer.record_bytes, layer.data_offset, layer.data_size)
    for component in layer.components:
        logical_block_elements, _ = TYPE_LAYOUT[component.tensor.ggml_type]
        block_elements = (
            component.block_elements
            if component.block_elements is not None
            else logical_block_elements
        )
        offset = STORE_COMPONENT_OFFSET + component.role * STORE_COMPONENT_BYTES
        struct.pack_into(
            "<IIIIQQQQQ", result, offset,
            component.role, component.tensor.ggml_type, 3, block_elements,
            *component.tensor.dims, component.expert_bytes,
            component.record_offset,
        )
    return bytes(result)


def make_header(plan: StorePlan, source_digest: bytes, payload_digest: bytes,
                manifest_digest: bytes = bytes(32)) -> bytes:
    if not all(len(digest) == 32 for digest in
               (source_digest, payload_digest, manifest_digest)):
        raise AssertionError("SHA-256 digest length")
    header = bytearray(STORE_HEADER_BYTES)
    header[0:8] = STORE_MAGIC
    struct.pack_into(
        "<IIIIIIQQQQQQQ", header, 8,
        STORE_VERSION, STORE_HEADER_BYTES, plan.family,
        plan.expert_used_count, plan.layer_count, plan.expert_count,
        plan.source_tensor_count, plan.layer_count, len(plan.descriptor_bytes),
        STORE_HEADER_BYTES, plan.data_offset, plan.data_size, plan.store_size,
    )
    struct.pack_into("<Q", header, 88, plan.source.size)
    header[96:128] = source_digest
    header[128:160] = payload_digest
    struct.pack_into("<II", header, 160,
                     plan.storage_format, plan.group_size)
    header[168:200] = manifest_digest
    return bytes(header)


def manifest_digest(header: bytes, descriptors: bytes) -> bytes:
    mutable = bytearray(header)
    mutable[MANIFEST_DIGEST_OFFSET:MANIFEST_DIGEST_OFFSET + 32] = bytes(32)
    return hashlib.sha256(mutable + descriptors).digest()


def hash_fd(fd: int, size: int, label: str) -> bytes:
    digest = hashlib.sha256()
    completed = 0
    last_percent = -1
    while completed < size:
        take = min(IO_BYTES, size - completed)
        data = pread_exact(fd, take, completed)
        digest.update(data)
        completed += len(data)
        percent = completed * 100 // size if size else 100
        if percent != last_percent and (percent == 100 or percent % 5 == 0):
            print(f"\r{label:<24} {percent:3d}%", end="", file=sys.stderr,
                  flush=True)
            last_percent = percent
    print(file=sys.stderr)
    return digest.digest()


def hash_file(path: Path, label: str) -> bytes:
    absolute, fd, identity = open_input(path, label)
    try:
        digest = hash_fd(fd, identity.size, label)
        require_fd_unchanged(fd, identity, label)
        return digest
    finally:
        os.close(fd)


def pread_exact(fd: int, size: int, offset: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        chunk = os.pread(fd, size - len(result), offset + len(result))
        if not chunk:
            raise FormatError(f"short read at {offset + len(result)}")
        result.extend(chunk)
    return bytes(result)


def pwrite_all(fd: int, data: bytes, offset: int) -> None:
    written = 0
    while written < len(data):
        count = os.pwrite(fd, data[written:], offset + written)
        if count <= 0:
            raise OSError(f"short write at {offset + written}")
        written += count


def require_zero_range(fd: int, offset: int, size: int, label: str) -> None:
    completed = 0
    while completed < size:
        take = min(IO_BYTES, size - completed)
        data = pread_exact(fd, take, offset + completed)
        if any(data):
            raise FormatError(f"non-zero {label} at byte {offset + completed}")
        completed += take


def copy_range(src_fd: int, src_offset: int, dst_fd: int, dst_offset: int,
               size: int, digest: Digest | None = None,
               compare_fd: int | None = None, compare_offset: int = 0) -> None:
    completed = 0
    while completed < size:
        take = min(IO_BYTES, size - completed)
        data = pread_exact(src_fd, take, src_offset + completed)
        if compare_fd is not None:
            other = pread_exact(compare_fd, take, compare_offset + completed)
            if data != other:
                raise FormatError(f"payload mismatch at byte {completed}")
        else:
            pwrite_all(dst_fd, data, dst_offset + completed)
        if digest is not None:
            digest.update(data)
        completed += take


def zeros(digest: Digest, size: int) -> None:
    block = bytes(min(IO_BYTES, size))
    while size:
        take = min(len(block), size)
        digest.update(block[:take])
        size -= take


def native_layout(source: GGUF, plan: StorePlan) -> tuple[list[Tensor], int, int]:
    routed = {component.tensor.name for layer in plan.layers
              for component in layer.components}
    tensors = [dataclasses.replace(tensor) for tensor in source.tensors
               if tensor.name not in routed]
    tensors.append(Tensor(STORE_TENSOR, (plan.store_size,), 24, 0,
                          plan.store_size))
    tensor_directory_bytes = sum(
        len(pack_string(tensor.name)) + 4 + 8 * len(tensor.dims) + 4 + 8
        for tensor in tensors
    )
    metadata_end = 4 + 4 + 8 + 8 + len(source.kv_raw) + tensor_directory_bytes
    data_offset = align_up(metadata_end, source.alignment)
    cursor = 0
    for tensor in tensors:
        cursor = align_up(cursor, source.alignment)
        tensor.new_rel_offset = cursor
        cursor += tensor.size
    return tensors, data_offset, data_offset + align_up(cursor, source.alignment)


def combined_layout(
        target: GGUF,
        target_plan: StorePlan,
        support: GGUF,
        support_plan: StorePlan,
        appended_metadata: tuple[bytes, ...],
        ) -> tuple[list[Tensor], int, int, bytes]:
    target_routed = {
        component.tensor.name
        for layer in target_plan.layers for component in layer.components
    }
    support_routed = {
        component.tensor.name
        for layer in support_plan.layers for component in layer.components
    }
    tensors = [
        dataclasses.replace(tensor) for tensor in target.tensors
        if tensor.name not in target_routed
    ]
    tensors.extend(
        dataclasses.replace(tensor) for tensor in support.tensors
        if tensor.name not in support_routed
    )
    names = [tensor.name for tensor in tensors]
    if len(names) != len(set(names)):
        raise FormatError("target and DSpark non-routed tensor names collide")
    tensors.extend((
        Tensor(STORE_TENSOR, (target_plan.store_size,), 24, 0,
               target_plan.store_size),
        Tensor(DSPARK_STORE_TENSOR, (support_plan.store_size,), 24, 0,
               support_plan.store_size),
    ))
    kv_raw = target.kv_raw + b"".join(appended_metadata)
    tensor_directory_bytes = sum(
        len(pack_string(tensor.name)) + 4 + 8 * len(tensor.dims) + 4 + 8
        for tensor in tensors
    )
    metadata_end = 4 + 4 + 8 + 8 + len(kv_raw) + tensor_directory_bytes
    data_offset = align_up(metadata_end, target.alignment)
    cursor = 0
    for tensor in tensors:
        cursor = align_up(cursor, target.alignment)
        tensor.new_rel_offset = cursor
        cursor += tensor.size
    return (tensors, data_offset,
            data_offset + align_up(cursor, target.alignment), kv_raw)


def write_native_header(fd: int, source: GGUF, tensors: list[Tensor],
                        data_offset: int, *, kv_raw: bytes | None = None,
                        n_kv: int | None = None) -> None:
    if kv_raw is None:
        kv_raw = source.kv_raw
    if n_kv is None:
        n_kv = source.n_kv
    parts = [MAGIC, struct.pack("<IQQ", source.version, len(tensors), n_kv),
             kv_raw]
    for tensor in tensors:
        parts.extend((
            pack_string(tensor.name), struct.pack("<I", len(tensor.dims)),
            struct.pack("<" + "Q" * len(tensor.dims), *tensor.dims),
            struct.pack("<IQ", tensor.ggml_type, tensor.new_rel_offset),
        ))
    header = b"".join(parts)
    if len(header) > data_offset:
        raise AssertionError("planned GGUF directory is too small")
    pwrite_all(fd, header, 0)


def check_space(destination: Path, required: int, reserve: int) -> None:
    free = shutil.disk_usage(destination.parent).free
    if free < required + reserve:
        raise FormatError(
            f"insufficient free space: need {required + reserve} bytes "
            f"({required} output + {reserve} reserve), have {free}"
        )


def paths_alias(left: Path, right: Path) -> bool:
    left = lexical_absolute(left)
    right = lexical_absolute(right)
    if left == right:
        return True
    try:
        left_info = os.lstat(left)
        right_info = os.lstat(right)
        return ((left_info.st_dev, left_info.st_ino) ==
                (right_info.st_dev, right_info.st_ino))
    except OSError:
        return False


def reject_destination_alias(destination: Path, *sources: Path) -> None:
    for source in sources:
        if paths_alias(destination, source):
            raise FormatError(
                f"destination aliases input GGUF: {lexical_absolute(source)}"
            )


def path_entry_exists(path: Path) -> bool:
    return os.path.lexists(path)


def unlink_owned_path(path: Path, identity: FDIdentity) -> bool:
    directory_fd = os.open(
        path.parent,
        os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        try:
            info = os.lstat(path.name, dir_fd=directory_fd)
        except FileNotFoundError:
            return True
        if ((info.st_dev, info.st_ino) !=
                (identity.device, identity.inode)):
            return False
        os.unlink(path.name, dir_fd=directory_fd)
        return True
    finally:
        os.close(directory_fd)


def install_temp(temp: Path, destination: Path, output_fd: int,
                 output_identity: FDIdentity,
                 output_digest: bytes) -> None:
    if temp.parent != destination.parent:
        raise AssertionError("temporary output must share destination directory")
    directory_fd = os.open(
        destination.parent,
        os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0),
    )
    installed_fd = -1
    installed_identity: FDIdentity | None = None
    cleanup_identity: FDIdentity | None = None
    failed = True
    try:
        copy_fallback = sys.platform != "darwin"
        if sys.platform == "darwin":
            libc = ctypes.CDLL(None, use_errno=True)
            clone = getattr(libc, "fclonefileat", None)
            if clone is None:
                copy_fallback = True
            else:
                clone.argtypes = (ctypes.c_int, ctypes.c_int,
                                  ctypes.c_char_p, ctypes.c_int)
                clone.restype = ctypes.c_int
                if clone(output_fd, directory_fd,
                         os.fsencode(destination.name), 0) != 0:
                    error = ctypes.get_errno()
                    if error == errno.EEXIST:
                        raise FormatError(
                            f"destination already exists: {destination}"
                        )
                    unsupported = {
                        errno.ENOTSUP,
                        getattr(errno, "EOPNOTSUPP", errno.ENOTSUP),
                        errno.ENOSYS,
                        errno.EXDEV,
                    }
                    if error not in unsupported:
                        raise OSError(
                            error, os.strerror(error), destination
                        )
                    copy_fallback = True
                else:
                    cloned_stat = os.lstat(
                        destination.name, dir_fd=directory_fd
                    )
                    cleanup_identity = stat_identity(
                        cloned_stat, "cloned output path"
                    )
                    installed_fd = os.open(
                        destination.name,
                        os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
                        dir_fd=directory_fd,
                    )
                    installed_identity = fd_identity(
                        installed_fd, "installed output descriptor"
                    )
                    if ((installed_identity.device,
                         installed_identity.inode) !=
                            (cleanup_identity.device,
                             cleanup_identity.inode)):
                        raise FormatError(
                            "cloned output identity changed before reopen"
                        )
        if copy_fallback:
            try:
                installed_fd = os.open(
                    destination.name,
                    os.O_RDWR | os.O_CREAT | os.O_EXCL |
                    os.O_CLOEXEC | os.O_NOFOLLOW,
                    0o644, dir_fd=directory_fd,
                )
            except FileExistsError as exc:
                raise FormatError(
                    f"destination already exists: {destination}"
                ) from exc
            installed_identity = fd_identity(
                installed_fd, "installed output descriptor"
            )
            cleanup_identity = installed_identity
            os.ftruncate(installed_fd, output_identity.size)
            copy_range(
                output_fd, 0, installed_fd, 0, output_identity.size
            )
            os.fsync(installed_fd)

        if installed_identity is None:
            installed_identity = fd_identity(
                installed_fd, "installed output descriptor"
            )
        cleanup_identity = installed_identity
        installed_digest = hash_fd(
            installed_fd, output_identity.size, "verify installed GGUF"
        )
        if installed_digest != output_digest:
            raise FormatError("installed output SHA-256 mismatch")
        installed_stat = os.lstat(destination.name, dir_fd=directory_fd)
        if (not statlib.S_ISREG(installed_stat.st_mode) or
                (installed_stat.st_dev, installed_stat.st_ino,
                 installed_stat.st_size) !=
                (installed_identity.device, installed_identity.inode,
                 output_identity.size)):
            raise FormatError("installed output identity changed unexpectedly")
        os.fsync(directory_fd)
        failed = False
    finally:
        if installed_fd >= 0:
            os.close(installed_fd)
        if failed and cleanup_identity is not None:
            unlink_owned_path(destination, cleanup_identity)
        os.close(directory_fd)

    if not unlink_owned_path(temp, output_identity):
        print(
            f"warning: temporary pathname was replaced; left intact: {temp}",
            file=sys.stderr,
        )


def require_digest_match(actual: bytes, expected: bytes) -> None:
    if len(actual) != 32 or len(expected) != 32 or actual != expected:
        raise FormatError(
            "DSpark support SHA-256 does not match the required artifact; "
            f"expected {expected.hex()}, got {actual.hex()}"
        )


def require_final_dspark_support_pin() -> bytes:
    pin = DSPARK_0731_FINAL_SUPPORT_SHA256
    if pin is None:
        raise FormatError(
            "final 0731 DSpark packaging is blocked: no SHA-256 pin exists "
            "for a support GGUF generated from final checkpoint shards "
            "46-48; preview support 8b3adf59...a1a34e is reference-only "
            "and rejected for publication"
        )
    if len(pin) != 32:
        raise FormatError("configured final 0731 DSpark SHA-256 pin is invalid")
    if pin == DSPARK_PREVIEW_REFERENCE_SHA256:
        raise FormatError(
            "preview DSpark SHA-256 cannot be used as the final 0731 "
            "production pin"
        )
    return pin


def parse_bytes(text: str) -> int:
    match = re.fullmatch(r"(\d+)(KiB|MiB|GiB)?", text)
    if not match:
        raise argparse.ArgumentTypeError(f"invalid byte quantity: {text}")
    value = int(match.group(1))
    multiplier = {None: 1, "KiB": 1 << 10, "MiB": 1 << 20,
                  "GiB": 1 << 30}[match.group(2)]
    return value * multiplier


def write_expert_store(plan: StorePlan, source_fd: int, output_fd: int,
                       store_abs: int, source_digest: bytes,
                       label: str) -> bytes:
    pwrite_all(output_fd, plan.descriptor_bytes,
               store_abs + STORE_HEADER_BYTES)
    payload_hash = hashlib.sha256()
    payload_cursor = plan.data_offset
    for ordinal, layer in enumerate(plan.layers, 1):
        zeros(payload_hash, layer.data_offset - payload_cursor)
        for expert in range(plan.expert_count):
            for component in layer.components:
                src_offset = (component.tensor.abs_offset +
                              expert * component.expert_bytes)
                dst_offset = (store_abs + layer.data_offset +
                              expert * layer.record_bytes +
                              component.record_offset)
                copy_range(source_fd, src_offset, output_fd, dst_offset,
                           component.expert_bytes, payload_hash)
        payload_cursor = layer.data_offset + layer.data_size
        print(f"\rwrite {label} layers {ordinal}/{plan.layer_count}",
              end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    if payload_cursor != plan.store_size:
        raise AssertionError("store payload plan did not reach its end")
    payload_digest = payload_hash.digest()
    provisional = make_header(plan, source_digest, payload_digest)
    final_header = make_header(
        plan, source_digest, payload_digest,
        manifest_digest(provisional, plan.descriptor_bytes),
    )
    pwrite_all(output_fd, final_header, store_abs)
    return payload_digest


def build(source_path: Path, destination: Path, reserve: int,
          verify_after: bool) -> None:
    destination = lexical_absolute(destination)
    reject_destination_alias(destination, source_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if path_entry_exists(destination):
        raise FormatError(f"destination already exists: {destination}")
    source_path, source_fd, source_identity = open_input(
        source_path, "source GGUF"
    )
    temp = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    output_fd = -1
    temp_owned = False
    owned_temp_identity: FDIdentity | None = None
    try:
        source = load_gguf_fd(source_path, source_fd)
        plan = make_store_plan(source)
        tensors, native_data_offset, output_size = native_layout(source, plan)
        check_space(destination, output_size, reserve)
        source_digest = hash_fd(
            source_fd, source_identity.size, "hash source GGUF"
        )
        output_fd = os.open(
            temp,
            os.O_CREAT | os.O_EXCL | os.O_RDWR |
            os.O_CLOEXEC | os.O_NOFOLLOW,
            0o644,
        )
        temp_owned = True
        owned_temp_identity = fd_identity(
            output_fd, "owned temporary output descriptor"
        )
        os.ftruncate(output_fd, output_size)
        write_native_header(output_fd, source, tensors, native_data_offset)
        by_name = {tensor.name: tensor for tensor in source.tensors}
        for index, tensor in enumerate(tensors[:-1], 1):
            original = by_name[tensor.name]
            copy_range(source_fd, original.abs_offset, output_fd,
                       native_data_offset + tensor.new_rel_offset, tensor.size)
            if index % 50 == 0 or index == len(tensors) - 1:
                print(f"\rcopy non-routed tensors {index}/{len(tensors) - 1}",
                      end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)

        store_tensor = tensors[-1]
        store_abs = native_data_offset + store_tensor.new_rel_offset
        payload_digest = write_expert_store(
            plan, source_fd, output_fd, store_abs, source_digest,
            "expert-major",
        )
        os.fsync(output_fd)
        if verify_after:
            native = load_gguf_fd(temp, output_fd)
            verify_open(
                source, source_fd, native, output_fd,
                source_digest=source_digest,
            )
        elif hash_fd(source_fd, source.size, "recheck source GGUF") != \
                source_digest:
            raise FormatError(
                "source GGUF changed while its descriptor was in use"
            )
        require_fd_unchanged(source_fd, source_identity, "source GGUF")
        output_digest = hash_fd(output_fd, output_size, "hash output GGUF")
        output_identity = fd_identity(output_fd, "verified output descriptor")
        install_temp(
            temp, destination, output_fd, output_identity, output_digest
        )
        temp_owned = False
        os.close(output_fd)
        output_fd = -1
    except BaseException:
        if output_fd >= 0:
            os.close(output_fd)
        if temp_owned and owned_temp_identity is not None:
            unlink_owned_path(temp, owned_temp_identity)
        raise
    finally:
        os.close(source_fd)
    print(f"installed atomically: {destination}")
    print(f"source_sha256: {source_digest.hex()}")
    print(f"payload_sha256: {payload_digest.hex()}")
    print(f"output_sha256: {output_digest.hex()}")
    print(f"output_bytes: {output_size}")


def build_combined(source_path: Path, support_path: Path, destination: Path,
                   reserve: int, verify_after: bool) -> None:
    destination = lexical_absolute(destination)
    reject_destination_alias(destination, source_path, support_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if path_entry_exists(destination):
        raise FormatError(f"destination already exists: {destination}")
    target_path, target_fd, target_identity = open_input(
        source_path, "target GGUF"
    )
    support_fd = -1
    temp = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    output_fd = -1
    temp_owned = False
    owned_temp_identity: FDIdentity | None = None
    try:
        support_path, support_fd, support_identity = open_input(
            support_path, "DSpark support GGUF"
        )
        target = load_gguf_fd(target_path, target_fd)
        reject_target_dspark_namespace(target)
        target_plan = make_store_plan(target)
        support = load_gguf_fd(support_path, support_fd)
        support_plan = make_dspark_store_plan(support, target_plan)
        final_support_pin = require_final_dspark_support_pin()
        target_metadata_keys = {key for key, _ in target.metadata_records}
        collisions = target_metadata_keys.intersection(DSPARK_METADATA_KEYS)
        if collisions:
            raise FormatError(
                "target GGUF already contains DSpark metadata: "
                f"{sorted(collisions)}"
            )
        appended_metadata = dspark_metadata_records(support)
        tensors, native_data_offset, output_size, kv_raw = combined_layout(
            target, target_plan, support, support_plan, appended_metadata,
        )
        check_space(destination, output_size, reserve)
        target_digest = hash_fd(
            target_fd, target_identity.size, "hash target GGUF"
        )
        support_digest = hash_fd(
            support_fd, support_identity.size, "hash DSpark support"
        )
        require_digest_match(support_digest, final_support_pin)
        output_fd = os.open(
            temp,
            os.O_CREAT | os.O_EXCL | os.O_RDWR |
            os.O_CLOEXEC | os.O_NOFOLLOW,
            0o644,
        )
        temp_owned = True
        owned_temp_identity = fd_identity(
            output_fd, "owned temporary output descriptor"
        )
        os.ftruncate(output_fd, output_size)
        write_native_header(
            output_fd, target, tensors, native_data_offset,
            kv_raw=kv_raw,
            n_kv=target.n_kv + len(appended_metadata),
        )
        target_routed = {
            component.tensor.name
            for layer in target_plan.layers for component in layer.components
        }
        support_routed = {
            component.tensor.name
            for layer in support_plan.layers for component in layer.components
        }
        target_non_routed = [
            tensor for tensor in target.tensors
            if tensor.name not in target_routed
        ]
        support_non_routed = [
            tensor for tensor in support.tensors
            if tensor.name not in support_routed
        ]
        combined_non_routed = target_non_routed + support_non_routed
        for index, (original, tensor) in enumerate(
                zip(combined_non_routed, tensors), 1):
            source_fd = (target_fd if index <= len(target_non_routed)
                         else support_fd)
            copy_range(
                source_fd, original.abs_offset, output_fd,
                native_data_offset + tensor.new_rel_offset, tensor.size,
            )
            if index % 50 == 0 or index == len(combined_non_routed):
                print(
                    f"\rcopy target/support non-routed tensors "
                    f"{index}/{len(combined_non_routed)}",
                    end="", file=sys.stderr, flush=True,
                )
        print(file=sys.stderr)

        target_store = tensors[-2]
        target_payload_digest = write_expert_store(
            target_plan, target_fd, output_fd,
            native_data_offset + target_store.new_rel_offset,
            target_digest, "target expert-major",
        )
        support_store = tensors[-1]
        support_payload_digest = write_expert_store(
            support_plan, support_fd, output_fd,
            native_data_offset + support_store.new_rel_offset,
            support_digest, "DSpark expert-major",
        )
        os.fsync(output_fd)
        if verify_after:
            native = load_gguf_fd(temp, output_fd)
            verify_combined_open(
                target, target_fd, support, support_fd, native, output_fd,
                target_digest=target_digest,
                support_digest=support_digest,
            )
        else:
            if hash_fd(target_fd, target.size, "recheck target GGUF") != \
                    target_digest:
                raise FormatError(
                    "target GGUF changed while its descriptor was in use"
                )
            if hash_fd(
                    support_fd, support.size, "recheck DSpark support") != \
                    support_digest:
                raise FormatError(
                    "DSpark support GGUF changed while its descriptor was in use"
                )
        require_fd_unchanged(target_fd, target_identity, "target GGUF")
        require_fd_unchanged(
            support_fd, support_identity, "DSpark support GGUF"
        )
        output_digest = hash_fd(output_fd, output_size, "hash output GGUF")
        output_identity = fd_identity(output_fd, "verified output descriptor")
        install_temp(
            temp, destination, output_fd, output_identity, output_digest
        )
        temp_owned = False
        os.close(output_fd)
        output_fd = -1
    except BaseException:
        if output_fd >= 0:
            os.close(output_fd)
        if temp_owned and owned_temp_identity is not None:
            unlink_owned_path(temp, owned_temp_identity)
        raise
    finally:
        if support_fd >= 0:
            os.close(support_fd)
        os.close(target_fd)
    print(f"installed atomically: {destination}")
    print(f"target_sha256: {target_digest.hex()}")
    print(f"target_payload_sha256: {target_payload_digest.hex()}")
    print(f"dspark_sha256: {support_digest.hex()}")
    print(f"dspark_payload_sha256: {support_payload_digest.hex()}")
    print(f"output_sha256: {output_digest.hex()}")
    print(f"output_bytes: {output_size}")


def parse_store(native: GGUF, tensor: Tensor,
                expected_name: str = STORE_TENSOR,
                native_fd: int | None = None,
                ) -> tuple[dict[str, object], list[Layer]]:
    if tensor.name != expected_name or tensor.ggml_type != 24 or \
            tensor.dims != (tensor.size,):
        raise FormatError("opaque expert-store tensor has an invalid GGUF descriptor")
    owned_fd = native_fd is None
    native_identity: FDIdentity | None = None
    if native_fd is None:
        _, native_fd, native_identity = open_input(
            native.path, "native GGUF"
        )
    try:
        header = pread_exact(native_fd, STORE_HEADER_BYTES, tensor.abs_offset)
        if header[:8] != STORE_MAGIC:
            raise FormatError("bad expert-store magic")
        values = struct.unpack_from("<IIIIIIQQQQQQQ", header, 8)
        (version, header_bytes, family, expert_used, layer_count,
         expert_count, source_tensors, descriptor_count, descriptor_bytes,
         descriptor_offset, data_offset, data_size, store_size) = values
        source_size = struct.unpack_from("<Q", header, 88)[0]
        storage_format, group_size = struct.unpack_from("<II", header, 160)
        storage_valid = (
            (storage_format == STORE_STORAGE_GGML and group_size == 0) or
            (storage_format == STORE_STORAGE_MLX_AFFINE4 and
             group_size == 64 and family == STORE_FAMILY_QWEN35_MOE)
        )
        if (version != STORE_VERSION or header_bytes != STORE_HEADER_BYTES or
                family not in STORE_FAMILIES or
                not 1 <= layer_count <= STORE_MAX_ROUTED_LAYERS or
                not 1 <= expert_count <= 384 or
                not 1 <= expert_used <= expert_count or
                source_tensors <= layer_count * 3 or source_size == 0 or
                descriptor_count != layer_count or
                descriptor_bytes != layer_count * STORE_LAYER_BYTES or
                descriptor_offset != STORE_HEADER_BYTES or
                data_offset % STORE_ALIGNMENT or data_offset + data_size != store_size or
                store_size != tensor.size or not storage_valid or
                any(header[200:])):
            raise FormatError("invalid expert-store header")
        descriptors = pread_exact(native_fd, descriptor_bytes,
                                  tensor.abs_offset + descriptor_offset)
        descriptor_end = descriptor_offset + descriptor_bytes
        require_zero_range(
            native_fd, tensor.abs_offset + descriptor_end,
            data_offset - descriptor_end, "expert-store pre-data padding",
        )
        recorded_manifest = header[168:200]
        if manifest_digest(header, descriptors) != recorded_manifest:
            raise FormatError("expert manifest SHA-256 mismatch")
        if owned_fd:
            assert native_identity is not None
            require_fd_unchanged(native_fd, native_identity, "native GGUF")
    finally:
        if owned_fd:
            os.close(native_fd)
    layers: list[Layer] = []
    previous_end = data_offset
    previous_layer_index = -1
    for il in range(layer_count):
        entry = descriptors[il * STORE_LAYER_BYTES:(il + 1) * STORE_LAYER_BYTES]
        layer_index, entry_experts, record_bytes, layer_offset, layer_size = \
            struct.unpack_from("<IIQQQ", entry)
        components: list[Component] = []
        record_cursor = 0
        for role in range(3):
            offset = STORE_COMPONENT_OFFSET + role * STORE_COMPONENT_BYTES
            (entry_role, ggml_type, ndim, block_elements, d0, d1, d2,
             expert_bytes, record_offset) = struct.unpack_from(
                "<IIIIQQQQQ", entry, offset
            )
            if (entry_role != role or ndim != 3 or
                    ggml_type not in ROUTED_TYPES or
                    d2 != expert_count or record_offset != record_cursor):
                raise FormatError(f"invalid component descriptor at layer {il} role {role}")
            if storage_format == STORE_STORAGE_GGML:
                descriptor_valid = (
                    TYPE_LAYOUT[ggml_type][0] == block_elements
                )
                expected = tensor_nbytes(ggml_type, (d0, d1, 1))
            elif storage_format == STORE_STORAGE_MLX_AFFINE4:
                descriptor_valid = (
                    ggml_type == 12 and
                    TYPE_LAYOUT[ggml_type][0] == block_elements
                )
                expected = tensor_nbytes(ggml_type, (d0, d1, 1))
            else:
                raise FormatError("unsupported expert-store storage format")
            if not descriptor_valid:
                raise FormatError(
                    f"invalid physical codec descriptor at layer {il} "
                    f"role {role}"
                )
            if expert_bytes != expected:
                raise FormatError(f"component byte size mismatch at layer {il} role {role}")
            synthetic = Tensor(
                f"blk.{layer_index}.ffn_{ROLE_NAME[role]}_exps.weight",
                               (d0, d1, d2), ggml_type, 0,
                               expert_bytes * expert_count)
            components.append(Component(
                role, synthetic, expert_bytes, record_offset, block_elements
            ))
            record_cursor += expert_bytes
        gate, up, down = (component.tensor for component in components)
        if (gate.dims != up.dims or gate.ggml_type != up.ggml_type or
                gate.dims[0] != down.dims[1] or
                gate.dims[1] != down.dims[0] or
                down.dims[2] != expert_count):
            raise FormatError(f"component geometry mismatch at layer {il}")
        if (layer_index <= previous_layer_index or
                layer_index > STORE_MAX_MODEL_LAYER or
                (family in (STORE_FAMILY_DEEPSEEK4,
                            STORE_FAMILY_QWEN35_MOE) and layer_index != il) or
                entry_experts != expert_count or
                record_bytes != record_cursor or layer_size != record_bytes * expert_count or
                layer_offset < previous_end or layer_offset % STORE_ALIGNMENT or
                layer_offset + layer_size > store_size or any(entry[200:])):
            raise FormatError(f"invalid layer descriptor {il}")
        layers.append(Layer(layer_index, expert_count, record_bytes, layer_offset,
                            layer_size, tuple(components)))
        previous_layer_index = layer_index
        previous_end = layer_offset + layer_size
    if previous_end != store_size:
        raise FormatError("expert descriptors do not cover the complete payload")
    manifest = {
        "header": header, "version": version, "family": family,
        "storage_format": storage_format, "group_size": group_size,
        "expert_used": expert_used, "layer_count": layer_count,
        "expert_count": expert_count, "source_tensors": source_tensors,
        "source_size": source_size, "data_offset": data_offset,
        "data_size": data_size, "store_size": store_size,
        "source_sha256": header[96:128], "payload_sha256": header[128:160],
        "manifest_sha256": recorded_manifest,
    }
    return manifest, layers


def verify_store_identity(source: GGUF, plan: StorePlan, native: GGUF,
                          store: Tensor, store_name: str,
                          identity_label: str, source_fd: int | None = None,
                          native_fd: int | None = None,
                          source_digest: bytes | None = None,
                          ) -> tuple[dict[str, object], list[Layer], bytes]:
    manifest, layers = parse_store(
        native, store, store_name, native_fd=native_fd
    )
    if (manifest["family"] != plan.family or
            manifest["storage_format"] != plan.storage_format or
            manifest["group_size"] != plan.group_size or
            manifest["layer_count"] != plan.layer_count or
            manifest["expert_count"] != plan.expert_count or
            manifest["expert_used"] != plan.expert_used_count or
            manifest["source_tensors"] != len(source.tensors) or
            manifest["source_size"] != source.size or
            manifest["data_offset"] != plan.data_offset or
            manifest["data_size"] != plan.data_size or
            manifest["store_size"] != plan.store_size):
        raise FormatError(
            f"{identity_label} expert-store identity does not match source GGUF"
        )
    if source_digest is None:
        owned_source_fd = source_fd is None
        source_identity: FDIdentity | None = None
        if source_fd is None:
            _, source_fd, source_identity = open_input(
                source.path, f"{identity_label} GGUF"
            )
        try:
            source_digest = hash_fd(
                source_fd, source.size, f"verify {identity_label} identity"
            )
            if owned_source_fd:
                assert source_identity is not None
                require_fd_unchanged(
                    source_fd, source_identity, f"{identity_label} GGUF"
                )
        finally:
            if owned_source_fd:
                os.close(source_fd)
    if source_digest != manifest["source_sha256"]:
        raise FormatError(
            f"{identity_label} GGUF SHA-256 does not match expert store"
        )
    return manifest, layers, source_digest


def verify_store_payload(plan: StorePlan, store: Tensor,
                         manifest: dict[str, object], layers: list[Layer],
                         source_fd: int, native_fd: int,
                         label: str) -> None:
    if len(layers) != len(plan.layers):
        raise FormatError("manifest layer count differs from store plan")
    payload_hash = hashlib.sha256()
    store_abs = store.abs_offset
    cursor = plan.data_offset
    for ordinal, (expected_layer, actual_layer) in enumerate(
            zip(plan.layers, layers), 1):
        canonical_offset = align_up(cursor, STORE_ALIGNMENT)
        if expected_layer.data_offset != canonical_offset:
            raise FormatError(
                f"non-canonical source store plan at layer {expected_layer.index}"
            )
        if (expected_layer.index != actual_layer.index or
                expected_layer.expert_count != actual_layer.expert_count or
                expected_layer.record_bytes != actual_layer.record_bytes or
                expected_layer.data_offset != actual_layer.data_offset or
                expected_layer.data_size != actual_layer.data_size):
            raise FormatError(
                "manifest layer layout differs at plan slot "
                f"{ordinal - 1}: expected layer {expected_layer.index} "
                f"offset={expected_layer.data_offset} "
                f"size={expected_layer.data_size}, got layer "
                f"{actual_layer.index} offset={actual_layer.data_offset} "
                f"size={actual_layer.data_size}"
            )
        gap = canonical_offset - cursor
        if gap:
            padding = pread_exact(native_fd, gap, store_abs + cursor)
            if any(padding):
                raise FormatError(
                    "non-zero or truncated layer padding before "
                    f"{actual_layer.index}"
                )
            payload_hash.update(padding)
        for expert in range(plan.expert_count):
            for expected_component, actual_component in zip(
                    expected_layer.components, actual_layer.components):
                if (expected_component.tensor.dims !=
                        actual_component.tensor.dims or
                        expected_component.tensor.ggml_type !=
                        actual_component.tensor.ggml_type or
                        expected_component.expert_bytes !=
                        actual_component.expert_bytes or
                        expected_component.record_offset !=
                        actual_component.record_offset):
                    raise FormatError(
                        "manifest geometry differs at layer "
                        f"{actual_layer.index} role {actual_component.role}"
                    )
                src_offset = (expected_component.tensor.abs_offset +
                              expert * expected_component.expert_bytes)
                packed_offset = (store_abs + actual_layer.data_offset +
                                 expert * actual_layer.record_bytes +
                                 actual_component.record_offset)
                copy_range(
                    native_fd, packed_offset, native_fd, 0,
                    actual_component.expert_bytes, payload_hash,
                    compare_fd=source_fd, compare_offset=src_offset,
                )
        cursor = expected_layer.data_offset + expected_layer.data_size
        print(f"\rverify {label} layers {ordinal}/{plan.layer_count}",
              end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    if cursor != plan.store_size or cursor != int(manifest["store_size"]):
        raise FormatError("verified payload did not reach store end")
    if payload_hash.digest() != manifest["payload_sha256"]:
        raise FormatError("expert payload SHA-256 mismatch")


def verify_non_routed(expected: list[Tensor], actual: list[Tensor],
                      source_fds: list[int], native_fd: int) -> None:
    if len(actual) != len(expected) or len(source_fds) != len(expected):
        raise FormatError("native non-routed tensor count mismatch")
    for expected_tensor, actual_tensor, source_fd in zip(
            expected, actual, source_fds):
        if (expected_tensor.name, expected_tensor.dims,
                expected_tensor.ggml_type, expected_tensor.size) != \
                (actual_tensor.name, actual_tensor.dims,
                 actual_tensor.ggml_type, actual_tensor.size):
            raise FormatError(
                f"non-routed descriptor mismatch: {expected_tensor.name}"
            )
        copy_range(
            source_fd, expected_tensor.abs_offset, native_fd, 0,
            expected_tensor.size, compare_fd=native_fd,
            compare_offset=actual_tensor.abs_offset,
        )


def verify_open(source: GGUF, source_fd: int, native: GGUF,
                native_fd: int,
                source_digest: bytes | None = None) -> None:
    authenticated_digest = source_digest
    plan = make_store_plan(source)
    if source.version != native.version or source.n_kv != native.n_kv or \
            source.kv_raw != native.kv_raw:
        raise FormatError("native GGUF metadata is not byte-identical to source")
    stores = [tensor for tensor in native.tensors if tensor.name == STORE_TENSOR]
    if len(stores) != 1 or any(
            tensor.name == DSPARK_STORE_TENSOR for tensor in native.tensors):
        raise FormatError("native GGUF must contain exactly one expert store")
    store = stores[0]
    manifest, layers, source_digest = verify_store_identity(
        source, plan, native, store, STORE_TENSOR, "source",
        source_fd, native_fd, source_digest,
    )

    routed_names = {component.tensor.name for layer in plan.layers
                    for component in layer.components}
    expected_non_routed = [tensor for tensor in source.tensors
                           if tensor.name not in routed_names]
    actual_non_routed = [tensor for tensor in native.tensors
                         if tensor.name != STORE_TENSOR]
    if len(actual_non_routed) != len(expected_non_routed):
        raise FormatError("native non-routed tensor count mismatch")
    verify_non_routed(
        expected_non_routed, actual_non_routed,
        [source_fd] * len(expected_non_routed), native_fd,
    )
    verify_store_payload(
        plan, store, manifest, layers, source_fd, native_fd,
        "expert-major",
    )
    if authenticated_digest is not None:
        final_digest = hash_fd(
            source_fd, source.size, "verify source identity"
        )
        if final_digest != authenticated_digest:
            raise FormatError("source GGUF changed while its descriptor was in use")
    print(f"valid DS4 expert-major v2 GGUF: {native.path}")
    print(f"source_sha256: {source_digest.hex()}")
    print(f"payload_sha256: {bytes(manifest['payload_sha256']).hex()}")


def verify(source_path: Path, native_path: Path) -> None:
    source_path, source_fd, source_identity = open_input(
        source_path, "source GGUF"
    )
    native_fd = -1
    try:
        native_path, native_fd, native_identity = open_input(
            native_path, "native GGUF"
        )
        source = load_gguf_fd(source_path, source_fd)
        native = load_gguf_fd(native_path, native_fd)
        verify_open(source, source_fd, native, native_fd)
        require_fd_unchanged(source_fd, source_identity, "source GGUF")
        require_fd_unchanged(native_fd, native_identity, "native GGUF")
    finally:
        if native_fd >= 0:
            os.close(native_fd)
        os.close(source_fd)


def verify_combined_open(
        target: GGUF, target_fd: int, support: GGUF, support_fd: int,
        native: GGUF, native_fd: int,
        target_digest: bytes | None = None,
        support_digest: bytes | None = None) -> None:
    authenticated_target_digest = target_digest
    authenticated_support_digest = support_digest
    reject_target_dspark_namespace(target)
    target_plan = make_store_plan(target)
    support_plan = make_dspark_store_plan(support, target_plan)
    final_support_pin = require_final_dspark_support_pin()
    appended_metadata = dspark_metadata_records(support)
    expected_kv = target.kv_raw + b"".join(appended_metadata)
    if (native.version != target.version or
            native.n_kv != target.n_kv + len(appended_metadata) or
            native.kv_raw != expected_kv):
        raise FormatError(
            "combined GGUF metadata is not the exact target metadata plus "
            "the DSpark records"
        )
    target_stores = [
        tensor for tensor in native.tensors if tensor.name == STORE_TENSOR
    ]
    support_stores = [
        tensor for tensor in native.tensors
        if tensor.name == DSPARK_STORE_TENSOR
    ]
    if len(target_stores) != 1 or len(support_stores) != 1:
        raise FormatError("combined GGUF must contain both expert stores once")
    target_store = target_stores[0]
    support_store = support_stores[0]
    target_manifest, target_layers, target_digest = verify_store_identity(
        target, target_plan, native, target_store, STORE_TENSOR, "target",
        target_fd, native_fd, target_digest,
    )
    support_manifest, support_layers, support_digest = verify_store_identity(
        support, support_plan, native, support_store, DSPARK_STORE_TENSOR,
        "DSpark support", support_fd, native_fd, support_digest,
    )
    require_digest_match(support_digest, final_support_pin)

    target_routed = {
        component.tensor.name
        for layer in target_plan.layers for component in layer.components
    }
    support_routed = {
        component.tensor.name
        for layer in support_plan.layers for component in layer.components
    }
    target_non_routed = [
        tensor for tensor in target.tensors if tensor.name not in target_routed
    ]
    support_non_routed = [
        tensor for tensor in support.tensors
        if tensor.name not in support_routed
    ]
    expected_non_routed = target_non_routed + support_non_routed
    actual_non_routed = [
        tensor for tensor in native.tensors
        if tensor.name not in (STORE_TENSOR, DSPARK_STORE_TENSOR)
    ]
    verify_non_routed(
        expected_non_routed, actual_non_routed,
        [target_fd] * len(target_non_routed) +
        [support_fd] * len(support_non_routed),
        native_fd,
    )
    verify_store_payload(
        target_plan, target_store, target_manifest, target_layers,
        target_fd, native_fd, "target expert-major",
    )
    verify_store_payload(
        support_plan, support_store, support_manifest, support_layers,
        support_fd, native_fd, "DSpark expert-major",
    )
    if authenticated_target_digest is not None:
        final_target_digest = hash_fd(
            target_fd, target.size, "verify target identity"
        )
        if final_target_digest != authenticated_target_digest:
            raise FormatError(
                "target GGUF changed while its descriptor was in use"
            )
    if authenticated_support_digest is not None:
        final_support_digest = hash_fd(
            support_fd, support.size, "verify DSpark identity"
        )
        if final_support_digest != authenticated_support_digest:
            raise FormatError(
                "DSpark support GGUF changed while its descriptor was in use"
            )
    print(f"valid combined DS4/DSpark ExpertMajor v2 GGUF: {native.path}")
    print(f"target_sha256: {target_digest.hex()}")
    print(f"dspark_sha256: {support_digest.hex()}")


def verify_combined(source_path: Path, support_path: Path,
                    native_path: Path) -> None:
    target_path, target_fd, target_identity = open_input(
        source_path, "target GGUF"
    )
    support_fd = -1
    native_fd = -1
    try:
        support_path, support_fd, support_identity = open_input(
            support_path, "DSpark support GGUF"
        )
        target = load_gguf_fd(target_path, target_fd)
        support = load_gguf_fd(support_path, support_fd)
        reject_target_dspark_namespace(target)
        target_plan = make_store_plan(target)
        make_dspark_store_plan(support, target_plan)
        require_final_dspark_support_pin()
        native_path, native_fd, native_identity = open_input(
            native_path, "combined GGUF"
        )
        native = load_gguf_fd(native_path, native_fd)
        verify_combined_open(
            target, target_fd, support, support_fd, native, native_fd
        )
        require_fd_unchanged(target_fd, target_identity, "target GGUF")
        require_fd_unchanged(
            support_fd, support_identity, "DSpark support GGUF"
        )
        require_fd_unchanged(native_fd, native_identity, "combined GGUF")
    finally:
        if native_fd >= 0:
            os.close(native_fd)
        if support_fd >= 0:
            os.close(support_fd)
        os.close(target_fd)


def clone_file(source: Path, destination: Path) -> None:
    """Create an APFS copy-on-write clone without temporarily duplicating 20GB."""
    libc = ctypes.CDLL(None, use_errno=True)
    clone = getattr(libc, "clonefile", None)
    if clone is None:
        raise FormatError("clonefile is unavailable; MLX repack requires macOS/APFS")
    clone.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int)
    clone.restype = ctypes.c_int
    if clone(os.fsencode(source), os.fsencode(destination), 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), destination)


def mlx_affine_component(
        mlx: MLXAffineSource,
        layer: int,
        expert: int,
        role: str,
        input_dim: int,
        output_dim: int) -> bytes:
    """Interleave one MLX affine4 matrix into 36-byte group-64 blocks."""
    try:
        import numpy as np
    except ImportError as exc:
        raise FormatError("numpy is required for MLX affine repacking") from exc

    if input_dim % 64:
        raise FormatError(f"MLX affine input width is not group-64: {input_dim}")
    groups = input_dim // 64
    prefix = (
        f"language_model.model.layers.{layer}.mlp.switch_mlp."
        f"{role}"
    )
    weights = mlx.expert_bytes(
        prefix + ".weight", "U32",
        (256, output_dim, input_dim // 8), expert,
    )
    scales = mlx.expert_bytes(
        prefix + ".scales", "BF16",
        (256, output_dim, groups), expert,
    )
    biases = mlx.expert_bytes(
        prefix + ".biases", "BF16",
        (256, output_dim, groups), expert,
    )
    expected_weights = output_dim * groups * 32
    expected_controls = output_dim * groups * 2
    if len(weights) != expected_weights or len(scales) != expected_controls or \
            len(biases) != expected_controls:
        raise FormatError(f"MLX affine byte geometry differs for {prefix}")

    packed = np.empty((output_dim, groups, 36), dtype=np.uint8)
    packed[:, :, :32] = np.frombuffer(weights, dtype=np.uint8).reshape(
        output_dim, groups, 32
    )
    packed[:, :, 32:34] = np.frombuffer(scales, dtype=np.uint8).reshape(
        output_dim, groups, 2
    )
    packed[:, :, 34:36] = np.frombuffer(biases, dtype=np.uint8).reshape(
        output_dim, groups, 2
    )
    return packed.tobytes()


def repack_mlx_affine(native_path: Path, mlx_dir: Path,
                      destination: Path, reserve: int) -> None:
    """Replace a Qwen v2 Q4_K payload with same-size MLX affine4 records."""
    native = load_gguf(native_path)
    stores = [tensor for tensor in native.tensors if tensor.name == STORE_TENSOR]
    if len(stores) != 1:
        raise FormatError("input must contain exactly one ExpertMajor v2 store")
    store_tensor = stores[0]
    manifest, layers = parse_store(native, store_tensor)
    if (manifest["family"] != STORE_FAMILY_QWEN35_MOE or
            manifest["storage_format"] != STORE_STORAGE_GGML or
            manifest["group_size"] != 0 or len(layers) != 40 or
            manifest["expert_count"] != 256 or
            any(component.tensor.ggml_type != 12
                for layer in layers for component in layer.components)):
        raise FormatError(
            "MLX affine repack needs a 40x256 Qwen Q4_K ExpertMajor v2 input"
        )

    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FormatError(f"destination already exists: {destination}")
    if shutil.disk_usage(destination.parent).free < reserve:
        raise FormatError(f"insufficient reserve space: need {reserve} bytes")
    temp = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    if temp.exists():
        raise FormatError(f"temporary path already exists: {temp}")

    mlx = MLXAffineSource(mlx_dir)
    output_fd = -1
    try:
        clone_file(native.path, temp)
        os.chmod(temp, 0o644)
        output_fd = os.open(temp, os.O_RDWR)
        store_abs = store_tensor.abs_offset
        payload_hash = hashlib.sha256()
        cursor = int(manifest["data_offset"])
        role_specs = (
            ("gate_proj", 2048, 512),
            ("up_proj", 2048, 512),
            ("down_proj", 512, 2048),
        )
        for ordinal, layer in enumerate(layers, 1):
            if layer.data_offset > cursor:
                gap = pread_exact(
                    output_fd, layer.data_offset - cursor,
                    store_abs + cursor,
                )
                if any(gap):
                    raise FormatError(
                        f"non-zero store padding before layer {layer.index}"
                    )
                payload_hash.update(gap)
            for expert in range(256):
                for component, (role, input_dim, output_dim) in zip(
                        layer.components, role_specs):
                    packed = mlx_affine_component(
                        mlx, layer.index, expert, role,
                        input_dim, output_dim,
                    )
                    if len(packed) != component.expert_bytes:
                        raise FormatError(
                            f"affine record size differs at layer {layer.index} "
                            f"expert {expert} role {role}"
                        )
                    offset = (
                        store_abs + layer.data_offset +
                        expert * layer.record_bytes +
                        component.record_offset
                    )
                    pwrite_all(output_fd, packed, offset)
                    payload_hash.update(packed)
            cursor = layer.data_offset + layer.data_size
            print(
                f"\rwrite MLX affine ExpertMajor layers {ordinal}/{len(layers)}",
                end="", file=sys.stderr, flush=True,
            )
        print(file=sys.stderr)
        if cursor != int(manifest["store_size"]):
            raise FormatError("affine repack did not cover the complete store")

        descriptors = pread_exact(
            output_fd, int(manifest["layer_count"]) * STORE_LAYER_BYTES,
            store_abs + STORE_HEADER_BYTES,
        )
        header = bytearray(manifest["header"])
        header[128:160] = payload_hash.digest()
        struct.pack_into(
            "<II", header, 160,
            STORE_STORAGE_MLX_AFFINE4, 64,
        )
        header[168:200] = bytes(32)
        header[168:200] = manifest_digest(bytes(header), descriptors)
        pwrite_all(output_fd, bytes(header), store_abs)
        os.fsync(output_fd)
        os.close(output_fd)
        output_fd = -1

        repacked = load_gguf(temp)
        repacked_store = next(
            tensor for tensor in repacked.tensors
            if tensor.name == STORE_TENSOR
        )
        repacked_manifest, _ = parse_store(repacked, repacked_store)
        if (repacked_manifest["storage_format"] !=
                STORE_STORAGE_MLX_AFFINE4 or
                repacked_manifest["group_size"] != 64 or
                repacked_manifest["payload_sha256"] != payload_hash.digest()):
            raise FormatError("repacked affine manifest did not validate")
        os.replace(temp, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        if output_fd >= 0:
            os.close(output_fd)
        temp.unlink(missing_ok=True)
        raise
    finally:
        mlx.close()
    print(f"installed atomically: {destination}")
    print("storage: mlx-affine4-g64")
    print(f"payload_sha256: {payload_hash.hexdigest()}")


def inspect(path: Path) -> None:
    source = load_gguf(path)
    plan = make_store_plan(source)
    _, _, output_size = native_layout(source, plan)
    routed_bytes = sum(layer.data_size for layer in plan.layers)
    print(f"architecture: {source.metadata['general.architecture']}")
    print(f"family: {plan.family}")
    print(f"layers: {plan.layer_count}")
    print(f"layer_ids: {plan.layers[0].index}..{plan.layers[-1].index}")
    print(f"experts: {plan.expert_count}")
    print(f"experts_used: {plan.expert_used_count}")
    print(f"routed_tensors: {plan.layer_count * 3}")
    print(f"routed_bytes: {routed_bytes}")
    print(f"store_bytes: {plan.store_size}")
    print(f"source_bytes: {source.size}")
    print(f"native_bytes: {output_size}")
    print(f"size_delta_bytes: {output_size - source.size}")
    classes: dict[tuple[int, int, int, int, int], int] = {}
    for layer in plan.layers:
        key = tuple(component.tensor.ggml_type for component in layer.components) + \
              (layer.record_bytes, plan.expert_count)
        classes[key] = classes.get(key, 0) + 1
    for key, count in sorted(classes.items()):
        gate_type, up_type, down_type, record_bytes, experts = key
        print("class: "
              f"layers={count} gate={TYPE_NAME.get(gate_type, gate_type)} "
              f"up={TYPE_NAME.get(up_type, up_type)} "
              f"down={TYPE_NAME.get(down_type, down_type)} "
              f"record_bytes={record_bytes} experts={experts}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("source", type=Path)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--reserve-bytes", type=parse_bytes,
                              default=1 << 30)
    build_parser.add_argument("--skip-verify", action="store_true",
                              help="diagnostic only; publication builds verify by default")
    build_parser.add_argument(
        "--dspark-support", type=Path,
        help="embed the final 0731 standalone DSpark support GGUF",
    )
    build_parser.add_argument("source", type=Path)
    build_parser.add_argument("destination", type=Path)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument(
        "--dspark-support", type=Path,
        help="verify the embedded final 0731 DSpark support GGUF",
    )
    verify_parser.add_argument("source", type=Path)
    verify_parser.add_argument("native", type=Path)
    affine_parser = subparsers.add_parser(
        "repack-mlx-affine",
        help="replace Qwen Q4_K expert records with local MLX affine4 groups",
    )
    affine_parser.add_argument("--reserve-bytes", type=parse_bytes,
                               default=1 << 30)
    affine_parser.add_argument("native", type=Path)
    affine_parser.add_argument("mlx_model", type=Path)
    affine_parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            inspect(args.source)
        elif args.command == "build":
            if args.dspark_support:
                build_combined(
                    args.source, args.dspark_support, args.destination,
                    args.reserve_bytes, not args.skip_verify,
                )
            else:
                build(args.source, args.destination, args.reserve_bytes,
                      not args.skip_verify)
        elif args.command == "verify":
            if args.dspark_support:
                verify_combined(
                    args.source, args.dspark_support, args.native,
                )
            else:
                verify(args.source, args.native)
        elif args.command == "repack-mlx-affine":
            repack_mlx_affine(
                args.native, args.mlx_model, args.destination,
                args.reserve_bytes,
            )
        return 0
    except (FormatError, OSError) as exc:
        print(f"ds4-expert-major: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
