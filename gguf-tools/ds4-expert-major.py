#!/usr/bin/env python3
"""Build and verify DS4 expert-major v2 GGUFs.

The converter changes storage only: non-routed tensors and every GGUF metadata
record are copied byte-for-byte, while each routed layer becomes a sequence of
complete expert records (gate, up, down). The opaque store is self-describing;
DS4 reconstructs the canonical logical tensor inventory at load time.
"""

from __future__ import annotations

import argparse
import ctypes
import dataclasses
import hashlib
import json
import os
import plistlib
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from typing import BinaryIO, Callable, Iterator, Protocol


MAGIC = b"GGUF"
STORE_MAGIC = b"DS4EXPV2"
STORE_TENSOR = "ds4.expert_major.v2"
STORE_VERSION = 2
STORE_FAMILY_DEEPSEEK4 = 1
STORE_FAMILY_GLM_DSA = 2
STORE_FAMILY_QWEN35_MOE = 3
STORE_STORAGE_GGML = 0
STORE_STORAGE_MLX_AFFINE4 = 1
STORE_STORAGE_MLX_AFFINE2 = 2
STORE_TYPE_MLX_AFFINE2 = 31
STORE_GROUP_PROFILE_AFFINE2_G32_U64_D64 = 0x00200040
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

DEEPSEEK_AFFINE2_ORIGIN = (
    "https://huggingface.co/mlx-community/DeepSeek-V4-Flash-2bit-DQ"
)
DEEPSEEK_AFFINE2_REVISION = "722bf559b7de93575b2320973cf2002e05bfe6c9"
DEEPSEEK_AFFINE2_CONFIG_SHA256 = (
    "b0d5c7c8d6471167b9ef6a4a97ad910a09bd1bc677e0483accdae0a21bf22f01"
)
DEEPSEEK_AFFINE2_INDEX_SHA256 = (
    "d1c2d929ab0a35be32cf18026bb31d6f99dad58d6c93a5a2abbe43791f9d6c30"
)
DEEPSEEK_AFFINE2_SOURCE_MANIFEST_SHA256 = (
    "cce807e30b9a1855be42dacdaf407d449115248fcfe32dad4bdd884aedf8e0cc"
)
DEEPSEEK_AFFINE2_TOTAL_SIZE = 96520315996
DEEPSEEK_AFFINE2_TOTAL_PARAMETERS = 284333146519
DEEPSEEK_AFFINE2_TENSOR_COUNT = 2610

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
ROUTED_TYPES = {10, 12, 13, 14, 16}  # Q2_K, Q4_K, Q5_K, Q6_K, IQ2_XXS
TYPE_NAME = {
    10: "Q2_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
    16: "IQ2_XXS", 24: "I8",
}
ROLE_NAME = ("gate", "up", "down")
ROUTED_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")


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
        return gguf_string(file)
    formats = {
        GGUF_UINT8: "<B", GGUF_INT8: "<b", GGUF_UINT16: "<H",
        GGUF_INT16: "<h", GGUF_UINT32: "<I", GGUF_INT32: "<i",
        GGUF_FLOAT32: "<f", GGUF_BOOL: "<?", GGUF_UINT64: "<Q",
        GGUF_INT64: "<q", GGUF_FLOAT64: "<d",
    }
    fmt = formats.get(value_type)
    if fmt:
        return struct.unpack(fmt, read_exact(file, struct.calcsize(fmt)))[0]
    skip_value(file, value_type)
    return None


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
    tensors: list[Tensor]
    data_offset: int


@dataclasses.dataclass
class Component:
    role: int
    tensor: Tensor
    expert_bytes: int
    record_offset: int


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


@dataclasses.dataclass(frozen=True)
class DeepSeekDonor:
    model_dir: Path
    origin: str
    revision: str
    config_sha256: str
    index_sha256: str
    source_digest: bytes
    source_size: int
    source_tensor_count: int
    gate_groups: tuple[int, ...]
    shard_oids: tuple[tuple[str, str, int], ...]
    hydrated: bool


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
            if not isinstance(tensors, dict):
                raise FormatError(f"invalid safetensor header in {path}")
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
        item_bytes = {"U32": 4, "BF16": 2}.get(dtype)
        elements = 1
        for dim in shape:
            if dim <= 0:
                raise FormatError(f"invalid safetensor shape for {key}")
            elements *= dim
        size = offsets[1] - offsets[0]
        if item_bytes is None or size != elements * item_bytes:
            raise FormatError(f"safetensor byte size differs for {key}")
        file_size = os.fstat(shard.fd).st_size
        absolute = shard.data_offset + offsets[0]
        if absolute > file_size or size > file_size - absolute:
            raise FormatError(f"safetensor extent exceeds shard for {key}")
        return shard, absolute, size

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


def load_gguf(path: Path) -> GGUF:
    path = path.resolve()
    size = path.stat().st_size
    with path.open("rb") as file:
        if read_exact(file, 4) != MAGIC:
            raise FormatError(f"{path} is not a GGUF")
        version = u32(file)
        n_tensors = u64(file)
        n_kv = u64(file)
        if version != 3:
            raise FormatError(f"only GGUF v3 is supported, got v{version}")
        kv_start = file.tell()
        metadata: dict[str, object] = {}
        alignment = 32
        wanted = {
            "general.architecture", "general.alignment",
            "deepseek4.block_count", "deepseek4.expert_count",
            "deepseek4.expert_used_count",
            "glm-dsa.block_count", "glm-dsa.expert_count",
            "glm-dsa.expert_used_count",
            "glm-dsa.leading_dense_block_count",
            "glm-dsa.nextn_predict_layers",
            "qwen35moe.block_count", "qwen35moe.expert_count",
            "qwen35moe.expert_used_count",
        }
        for _ in range(n_kv):
            key = gguf_string(file)
            value_type = u32(file)
            if key in wanted:
                value = read_metadata_value(file, value_type)
                metadata[key] = value
                if key == "general.alignment":
                    alignment = int(value)
            else:
                skip_value(file, value_type)
        tensor_start = file.tell()
        file.seek(kv_start)
        kv_raw = read_exact(file, tensor_start - kv_start)
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
        data_offset = align_up(file.tell(), alignment)
    if alignment <= 0 or alignment & (alignment - 1):
        raise FormatError(f"invalid general.alignment: {alignment}")
    for tensor in tensors:
        tensor.abs_offset = data_offset + tensor.rel_offset
        if tensor.abs_offset > size or tensor.size > size - tensor.abs_offset:
            raise FormatError(f"tensor points outside GGUF: {tensor.name}")
    return GGUF(path, size, version, n_kv, alignment, kv_raw, metadata,
                tensors, data_offset)


def routed_inventory(gguf: GGUF) -> dict[int, dict[int, Tensor]]:
    result: dict[int, dict[int, Tensor]] = {}
    role_index = {name: index for index, name in enumerate(ROLE_NAME)}
    for tensor in gguf.tensors:
        match = ROUTED_RE.fullmatch(tensor.name)
        if not match:
            continue
        layer = int(match.group(1))
        role = role_index[match.group(2)]
        if role in result.setdefault(layer, {}):
            raise FormatError(f"duplicate routed tensor {tensor.name}")
        result[layer][role] = tensor
    return result


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
        block_elements, _ = TYPE_LAYOUT[component.tensor.ggml_type]
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


def hash_file(path: Path, label: str) -> bytes:
    size = path.stat().st_size
    digest = hashlib.sha256()
    completed = 0
    last_percent = -1
    with path.open("rb", buffering=0) as file:
        while True:
            data = file.read(IO_BYTES)
            if not data:
                break
            digest.update(data)
            completed += len(data)
            percent = completed * 100 // size if size else 100
            if percent != last_percent and (percent == 100 or percent % 5 == 0):
                print(f"\r{label:<24} {percent:3d}%", end="", file=sys.stderr,
                      flush=True)
                last_percent = percent
    print(file=sys.stderr)
    return digest.digest()


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
    return tensors, data_offset, data_offset + cursor


def write_native_header(fd: int, source: GGUF, tensors: list[Tensor],
                        data_offset: int) -> None:
    parts = [MAGIC, struct.pack("<IQQ", source.version, len(tensors), source.n_kv),
             source.kv_raw]
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


def parse_bytes(text: str) -> int:
    match = re.fullmatch(r"(\d+)(KiB|MiB|GiB)?", text)
    if not match:
        raise argparse.ArgumentTypeError(f"invalid byte quantity: {text}")
    value = int(match.group(1))
    multiplier = {None: 1, "KiB": 1 << 10, "MiB": 1 << 20,
                  "GiB": 1 << 30}[match.group(2)]
    return value * multiplier


def build(source_path: Path, destination: Path, reserve: int,
          verify_after: bool) -> None:
    source = load_gguf(source_path)
    plan = make_store_plan(source)
    tensors, native_data_offset, output_size = native_layout(source, plan)
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    check_space(destination, output_size, reserve)
    source_digest = hash_file(source.path, "hash source GGUF")
    temp = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    if temp.exists():
        raise FormatError(f"temporary path already exists: {temp}")
    source_fd = os.open(source.path, os.O_RDONLY)
    output_fd = os.open(temp, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o644)
    try:
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
            print(f"\rwrite expert-major layers {ordinal}/{plan.layer_count}",
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
        os.fsync(output_fd)
        os.close(output_fd)
        output_fd = -1
        if verify_after:
            verify(source.path, temp)
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
        os.close(source_fd)
    print(f"installed atomically: {destination}")
    print(f"source_sha256: {source_digest.hex()}")
    print(f"payload_sha256: {payload_digest.hex()}")
    print(f"output_bytes: {output_size}")


def parse_store(native: GGUF, tensor: Tensor) -> tuple[dict[str, object], list[Layer]]:
    if tensor.name != STORE_TENSOR or tensor.ggml_type != 24 or \
            tensor.dims != (tensor.size,):
        raise FormatError("opaque expert-store tensor has an invalid GGUF descriptor")
    with native.path.open("rb", buffering=0) as file:
        file.seek(tensor.abs_offset)
        header = read_exact(file, STORE_HEADER_BYTES)
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
             group_size == 64 and family == STORE_FAMILY_QWEN35_MOE) or
            (storage_format == STORE_STORAGE_MLX_AFFINE2 and
             group_size == STORE_GROUP_PROFILE_AFFINE2_G32_U64_D64 and
             family == STORE_FAMILY_DEEPSEEK4)
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
        descriptors = pread_exact(file.fileno(), descriptor_bytes,
                                  tensor.abs_offset + descriptor_offset)
        recorded_manifest = header[168:200]
        if manifest_digest(header, descriptors) != recorded_manifest:
            raise FormatError("expert manifest SHA-256 mismatch")
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
            affine2 = storage_format == STORE_STORAGE_MLX_AFFINE2
            expected_block_elements = 32 if role == 0 else 64
            expected_block_bytes = 12 if role == 0 else 20
            if (entry_role != role or ndim != 3 or
                    d0 == 0 or d1 == 0 or block_elements == 0 or
                    d0 % block_elements != 0 or
                    (affine2
                     and ggml_type != STORE_TYPE_MLX_AFFINE2) or
                    (not affine2 and ggml_type not in ROUTED_TYPES) or
                    (storage_format == STORE_STORAGE_MLX_AFFINE4 and
                     ggml_type != 12) or
                    (affine2 and block_elements != expected_block_elements) or
                    (not affine2 and
                     TYPE_LAYOUT[ggml_type][0] != block_elements) or
                    d2 != expert_count or record_offset != record_cursor):
                raise FormatError(f"invalid component descriptor at layer {il} role {role}")
            expected = ((d0 // expected_block_elements) *
                        expected_block_bytes * d1) if affine2 else \
                tensor_nbytes(ggml_type, (d0, d1, 1))
            if expert_bytes != expected:
                raise FormatError(f"component byte size mismatch at layer {il} role {role}")
            synthetic = Tensor(
                f"blk.{layer_index}.ffn_{ROLE_NAME[role]}_exps.weight",
                               (d0, d1, d2), ggml_type, 0,
                               expert_bytes * expert_count)
            components.append(Component(role, synthetic, expert_bytes,
                                        record_offset))
            record_cursor += expert_bytes
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


def verify(source_path: Path, native_path: Path) -> None:
    source = load_gguf(source_path)
    plan = make_store_plan(source)
    native = load_gguf(native_path)
    if source.version != native.version or source.n_kv != native.n_kv or \
            source.kv_raw != native.kv_raw:
        raise FormatError("native GGUF metadata is not byte-identical to source")
    stores = [tensor for tensor in native.tensors if tensor.name == STORE_TENSOR]
    if len(stores) != 1:
        raise FormatError("native GGUF must contain exactly one expert store")
    store = stores[0]
    manifest, layers = parse_store(native, store)
    if (manifest["family"] != plan.family or
            manifest["storage_format"] != plan.storage_format or
            manifest["group_size"] != plan.group_size or
            manifest["layer_count"] != plan.layer_count or
            manifest["expert_count"] != plan.expert_count or
            manifest["expert_used"] != plan.expert_used_count or
            manifest["source_tensors"] != len(source.tensors) or
            manifest["source_size"] != source.size):
        raise FormatError("expert-store identity does not match source GGUF")
    source_digest = hash_file(source.path, "verify source identity")
    if source_digest != manifest["source_sha256"]:
        raise FormatError("source GGUF SHA-256 does not match expert store")

    routed_names = {component.tensor.name for layer in plan.layers
                    for component in layer.components}
    expected_non_routed = [tensor for tensor in source.tensors
                           if tensor.name not in routed_names]
    actual_non_routed = [tensor for tensor in native.tensors
                         if tensor.name != STORE_TENSOR]
    if len(actual_non_routed) != len(expected_non_routed):
        raise FormatError("native non-routed tensor count mismatch")
    source_fd = os.open(source.path, os.O_RDONLY)
    native_fd = os.open(native.path, os.O_RDONLY)
    try:
        for expected, actual in zip(expected_non_routed, actual_non_routed):
            if (expected.name, expected.dims, expected.ggml_type, expected.size) != \
                    (actual.name, actual.dims, actual.ggml_type, actual.size):
                raise FormatError(f"non-routed descriptor mismatch: {expected.name}")
            copy_range(source_fd, expected.abs_offset, native_fd, 0,
                       expected.size, compare_fd=native_fd,
                       compare_offset=actual.abs_offset)

        payload_hash = hashlib.sha256()
        store_abs = store.abs_offset
        cursor = int(manifest["data_offset"])
        for ordinal, (expected_layer, actual_layer) in enumerate(
                zip(plan.layers, layers), 1):
            if expected_layer.index != actual_layer.index:
                raise FormatError(
                    "manifest layer identity differs: "
                    f"expected {expected_layer.index}, got {actual_layer.index}"
                )
            gap = actual_layer.data_offset - cursor
            if gap:
                padding = pread_exact(native_fd, gap, store_abs + cursor)
                if any(padding):
                    raise FormatError(f"non-zero or truncated layer padding before {actual_layer.index}")
                payload_hash.update(padding)
            for expert in range(plan.expert_count):
                for expected_component, actual_component in zip(
                        expected_layer.components, actual_layer.components):
                    if (expected_component.tensor.dims != actual_component.tensor.dims or
                            expected_component.tensor.ggml_type != actual_component.tensor.ggml_type or
                            expected_component.expert_bytes != actual_component.expert_bytes or
                            expected_component.record_offset != actual_component.record_offset):
                        raise FormatError(
                            f"manifest geometry differs at layer {actual_layer.index} role {actual_component.role}"
                        )
                    src_offset = (expected_component.tensor.abs_offset +
                                  expert * expected_component.expert_bytes)
                    packed_offset = (store_abs + actual_layer.data_offset +
                                     expert * actual_layer.record_bytes +
                                     actual_component.record_offset)
                    copy_range(native_fd, packed_offset, native_fd, 0,
                               actual_component.expert_bytes, payload_hash,
                               compare_fd=source_fd, compare_offset=src_offset)
            cursor = actual_layer.data_offset + actual_layer.data_size
            print(f"\rverify expert-major layers {ordinal}/{plan.layer_count}",
                  end="", file=sys.stderr, flush=True)
        print(file=sys.stderr)
        if cursor != int(manifest["store_size"]):
            raise FormatError("verified payload did not reach store end")
        if payload_hash.digest() != manifest["payload_sha256"]:
            raise FormatError("expert payload SHA-256 mismatch")
    finally:
        os.close(source_fd)
        os.close(native_fd)
    print(f"valid DS4 expert-major v2 GGUF: {native.path}")
    print(f"source_sha256: {source_digest.hex()}")
    print(f"payload_sha256: {bytes(manifest['payload_sha256']).hex()}")


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


def interleave_mlx_affine2(
        weights: bytes,
        scales: bytes,
        biases: bytes,
        rows: int,
        input_dim: int,
        source_group_size: int,
        target_group_size: int) -> bytes:
    """Create exact affine2 blocks; g64->g32 duplicates BF16 controls losslessly."""
    if source_group_size not in (32, 64) or target_group_size not in (32, 64):
        raise FormatError("affine2 group size must be 32 or 64")
    if target_group_size > source_group_size or \
            source_group_size % target_group_size or \
            input_dim % source_group_size:
        raise FormatError("unsupported affine2 group normalization")
    source_groups = input_dim // source_group_size
    code_bytes = source_group_size // 4
    control_bytes = rows * source_groups * 2
    if len(weights) != rows * source_groups * code_bytes or \
            len(scales) != control_bytes or len(biases) != control_bytes:
        raise FormatError("affine2 source byte geometry differs")

    try:
        import numpy as np
    except ImportError as exc:
        raise FormatError("numpy is required for MLX affine repacking") from exc
    splits = source_group_size // target_group_size
    target_code_bytes = target_group_size // 4
    block_bytes = target_code_bytes + 4
    packed = np.empty(
        (rows, source_groups, splits, block_bytes), dtype=np.uint8
    )
    packed[:, :, :, :target_code_bytes] = np.frombuffer(
        weights, dtype=np.uint8
    ).reshape(rows, source_groups, splits, target_code_bytes)
    packed[:, :, :, target_code_bytes:target_code_bytes + 2] = np.frombuffer(
        scales, dtype=np.uint8
    ).reshape(rows, source_groups, 1, 2)
    packed[:, :, :, target_code_bytes + 2:target_code_bytes + 4] = \
        np.frombuffer(biases, dtype=np.uint8).reshape(
            rows, source_groups, 1, 2
        )
    return packed.tobytes()


def validate_deepseek_affine2_config(config: dict) -> tuple[int, ...]:
    """Validate the observed donor contract and return gate source groups."""
    if (config.get("torch_dtype") != "bfloat16" or
            config.get("hidden_size") != 4096 or
            config.get("moe_intermediate_size") != 2048 or
            config.get("n_routed_experts") != 256 or
            config.get("num_experts_per_tok") != 6 or
            config.get("num_hidden_layers") != 43):
        raise FormatError("DeepSeek affine2 donor shape/dtype differs")
    first = config.get("quantization")
    second = config.get("quantization_config")
    if not isinstance(first, dict) or first != second:
        raise FormatError("DeepSeek affine2 quantization maps differ")
    gate_groups = []
    for layer in range(43):
        for role in ("gate_proj", "up_proj", "down_proj"):
            key = f"model.layers.{layer}.ffn.switch_mlp.{role}"
            spec = first.get(key)
            expected_group = 64 if role != "gate_proj" else \
                (64 if layer == 42 else 32)
            if spec != {"group_size": expected_group,
                        "bits": 2, "mode": "affine"}:
                raise FormatError(
                    f"unsupported affine2 donor pattern at layer {layer} {role}"
                )
        gate_groups.append(64 if layer == 42 else 32)
    return tuple(gate_groups)


def deepseek_affine2_expected_tensors(
        gate_groups: tuple[int, ...],
        expert_count: int = 256,
        hidden_size: int = 4096,
        intermediate_size: int = 2048,
) -> list[tuple[str, str, tuple[int, ...]]]:
    expected: list[tuple[str, str, tuple[int, ...]]] = []
    for layer, gate_group in enumerate(gate_groups):
        role_shapes = (
            ("gate_proj", hidden_size, intermediate_size, gate_group),
            ("up_proj", hidden_size, intermediate_size, 64),
            ("down_proj", intermediate_size, hidden_size, 64),
        )
        for role, input_dim, rows, group in role_shapes:
            prefix = f"model.layers.{layer}.ffn.switch_mlp.{role}"
            expected.extend((
                (prefix + ".weight", "U32",
                 (expert_count, rows, input_dim // 16)),
                (prefix + ".scales", "BF16",
                 (expert_count, rows, input_dim // group)),
                (prefix + ".biases", "BF16",
                 (expert_count, rows, input_dim // group)),
            ))
    return expected


def _git(model_dir: Path, *args: str, binary: bool = False):
    result = subprocess.run(
        ["git", "-C", str(model_dir), *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if result.returncode:
        error = result.stderr.decode("utf-8", "replace").strip()
        raise FormatError(
            f"donor provenance check failed ({' '.join(args)}): "
            f"{error or 'git returned an error'}"
        )
    if binary:
        return result.stdout
    return result.stdout.decode("utf-8").strip()


def load_deepseek_affine2_donor(
        model_dir: Path,
        expected_revision: str,
        *,
        require_hydrated: bool,
        verify_shards: bool,
) -> DeepSeekDonor:
    """Validate the exact published donor, including committed LFS identities."""
    model_dir = model_dir.resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", expected_revision):
        raise FormatError("--expected-revision must be a full lowercase SHA-1")
    if expected_revision != DEEPSEEK_AFFINE2_REVISION:
        raise FormatError(
            "unsupported donor revision: this writer is pinned to "
            f"{DEEPSEEK_AFFINE2_REVISION}"
        )
    revision = _git(model_dir, "rev-parse", "HEAD")
    if revision != expected_revision:
        raise FormatError(
            f"donor revision differs: expected {expected_revision}, got {revision}"
        )
    origin = _git(model_dir, "config", "--get", "remote.origin.url").removesuffix(".git")
    if origin != DEEPSEEK_AFFINE2_ORIGIN:
        raise FormatError(
            f"donor origin differs: expected {DEEPSEEK_AFFINE2_ORIGIN}, got {origin}"
        )
    if subprocess.run(
            ["git", "-C", str(model_dir), "diff", "--quiet", "HEAD", "--",
             "config.json", "model.safetensors.index.json"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
        raise FormatError("donor config/index differ from the pinned revision")

    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    try:
        config_raw = config_path.read_bytes()
        index_raw = index_path.read_bytes()
        config = json.loads(config_raw)
        index = json.loads(index_raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise FormatError(f"invalid donor config/index: {exc}") from exc
    config_sha256 = hashlib.sha256(config_raw).hexdigest()
    index_sha256 = hashlib.sha256(index_raw).hexdigest()
    if config_sha256 != DEEPSEEK_AFFINE2_CONFIG_SHA256 or \
            index_sha256 != DEEPSEEK_AFFINE2_INDEX_SHA256:
        raise FormatError("donor config/index SHA-256 differs from the pinned release")
    gate_groups = validate_deepseek_affine2_config(config)
    weight_map = index.get("weight_map")
    metadata = index.get("metadata")
    if not isinstance(weight_map, dict) or \
            len(weight_map) != DEEPSEEK_AFFINE2_TENSOR_COUNT or \
            not isinstance(metadata, dict) or \
            metadata.get("total_size") != DEEPSEEK_AFFINE2_TOTAL_SIZE or \
            metadata.get("total_parameters") != DEEPSEEK_AFFINE2_TOTAL_PARAMETERS:
        raise FormatError("donor safetensor index identity differs")

    expected_tensors = deepseek_affine2_expected_tensors(gate_groups)
    missing = [name for name, _, _ in expected_tensors
               if not isinstance(weight_map.get(name), str)]
    if missing:
        raise FormatError(f"donor index misses routed tensor {missing[0]}")

    shard_pattern = re.compile(r"model-\d{5}-of-00019\.safetensors")
    shard_names = sorted(set(weight_map.values()))
    if len(shard_names) != 19 or any(
            not isinstance(name, str) or not shard_pattern.fullmatch(name)
            for name in shard_names):
        raise FormatError("donor index shard inventory differs")
    hydrated_names: list[str] = []
    pointer_names: list[str] = []
    lfs_bytes = 0
    shard_oids: list[tuple[str, str, int]] = []
    for name in shard_names:
        path = model_dir / name
        try:
            with path.open("rb") as file:
                head = file.read(256)
        except OSError as exc:
            raise FormatError(f"donor shard is missing: {name}") from exc
        committed = _git(model_dir, "show", f"HEAD:{name}", binary=True)
        match = re.search(rb"\nsize (\d+)\n?", committed)
        oid_match = re.search(rb"\noid sha256:([0-9a-f]{64})\n", committed)
        if not committed.startswith(
                b"version https://git-lfs.github.com/spec/v1\n") or \
                not match or not oid_match:
            raise FormatError(f"pinned Git object is not an LFS pointer: {name}")
        shard_size = int(match.group(1))
        shard_oid = oid_match.group(1).decode("ascii")
        shard_oids.append((name, shard_oid, shard_size))
        lfs_bytes += shard_size
        if head.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
            if path.read_bytes() != committed:
                raise FormatError(f"working-tree LFS pointer differs: {name}")
            pointer_names.append(name)
        else:
            if path.stat().st_size != shard_size:
                raise FormatError(
                    f"hydrated shard size differs for {name}: expected "
                    f"{shard_size}, got {path.stat().st_size}"
                )
            hydrated_names.append(name)
            if verify_shards:
                actual = hash_file(
                    path, f"hash donor shard {len(hydrated_names)}/19"
                ).hex()
                if actual != shard_oid:
                    raise FormatError(f"Git LFS SHA-256 mismatch for {name}")
    if pointer_names and hydrated_names:
        raise FormatError("partially hydrated donor is not a reproducible input")
    if (lfs_bytes < int(metadata["total_size"]) or
            lfs_bytes - int(metadata["total_size"]) >
                len(shard_names) * (1 << 20)):
        # Index total_size counts tensor data; LFS objects also include each
        # safetensor JSON header and its 8-byte length prefix.
        raise FormatError("Git LFS shard sizes differ from safetensor metadata")

    hydrated = len(hydrated_names) == len(shard_names)
    if require_hydrated and not hydrated:
        raise FormatError(
            "donor shards are Git LFS pointers; hydrate all 19 shards before writing"
        )

    # Once hydrated, verify every routed header entry and exact byte extent.
    if hydrated:
        source = MLXAffineSource(model_dir)
        try:
            for name, dtype, shape in expected_tensors:
                source.tensor(name, dtype, shape)
        finally:
            source.close()

    source_identity = {
        "schema": "ds4-deepseek-mlx-affine2-source-v1",
        "origin": origin,
        "revision": revision,
        "config_sha256": config_sha256,
        "index_sha256": index_sha256,
        "shards": [
            {"name": name, "oid_sha256": oid, "size": size}
            for name, oid, size in shard_oids
        ],
    }
    source_digest = hashlib.sha256(json.dumps(
        source_identity, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).digest()
    return DeepSeekDonor(
        model_dir=model_dir,
        origin=origin,
        revision=revision,
        config_sha256=config_sha256,
        index_sha256=index_sha256,
        source_digest=source_digest,
        source_size=int(metadata["total_size"]),
        source_tensor_count=len(weight_map),
        gate_groups=gate_groups,
        shard_oids=tuple(shard_oids),
        hydrated=hydrated,
    )


def pack_affine2_layer(layer: Layer) -> bytes:
    result = bytearray(STORE_LAYER_BYTES)
    struct.pack_into("<IIQQQ", result, 0, layer.index, layer.expert_count,
                     layer.record_bytes, layer.data_offset, layer.data_size)
    for component in layer.components:
        block_elements = 32 if component.role == 0 else 64
        offset = STORE_COMPONENT_OFFSET + component.role * STORE_COMPONENT_BYTES
        struct.pack_into(
            "<IIIIQQQQQ", result, offset,
            component.role, STORE_TYPE_MLX_AFFINE2, 3, block_elements,
            *component.tensor.dims, component.expert_bytes,
            component.record_offset,
        )
    return bytes(result)


def make_deepseek_affine2_store_plan(
        source_size: int,
        source_tensor_count: int,
        *,
        layer_count: int = 43,
        expert_count: int = 256,
        expert_used: int = 6,
        hidden_size: int = 4096,
        intermediate_size: int = 2048,
) -> StorePlan:
    if (source_size <= 0 or source_tensor_count <= layer_count * 3 or
            not 1 <= layer_count <= STORE_MAX_ROUTED_LAYERS or
            not 1 <= expert_count <= 384 or
            not 1 <= expert_used <= expert_count or
            hidden_size <= 0 or intermediate_size <= 0 or
            hidden_size % 64 or intermediate_size % 64):
        raise FormatError("invalid DeepSeek affine2 store geometry")
    source = GGUF(Path("<mlx-donor>"), source_size, 0, 0, STORE_ALIGNMENT,
                  b"", {}, [], 0)
    cursor = align_up(
        STORE_HEADER_BYTES + layer_count * STORE_LAYER_BYTES, STORE_ALIGNMENT
    )
    layers: list[Layer] = []
    dims_by_role = (
        (hidden_size, intermediate_size, expert_count),
        (hidden_size, intermediate_size, expert_count),
        (intermediate_size, hidden_size, expert_count),
    )
    for layer_index in range(layer_count):
        components: list[Component] = []
        record_offset = 0
        for role, dims in enumerate(dims_by_role):
            block_elements = 32 if role == 0 else 64
            block_bytes = 12 if role == 0 else 20
            expert_bytes = dims[0] // block_elements * block_bytes * dims[1]
            tensor = Tensor(
                f"blk.{layer_index}.ffn_{ROLE_NAME[role]}_exps.weight",
                dims, STORE_TYPE_MLX_AFFINE2, 0,
                expert_bytes * expert_count,
            )
            components.append(Component(
                role, tensor, expert_bytes, record_offset
            ))
            record_offset += expert_bytes
        layer_size = record_offset * expert_count
        layers.append(Layer(
            layer_index, expert_count, record_offset, cursor, layer_size,
            tuple(components),
        ))
        cursor += layer_size
    descriptors = b"".join(pack_affine2_layer(layer) for layer in layers)
    data_offset = layers[0].data_offset
    return StorePlan(
        source=source,
        family=STORE_FAMILY_DEEPSEEK4,
        storage_format=STORE_STORAGE_MLX_AFFINE2,
        group_size=STORE_GROUP_PROFILE_AFFINE2_G32_U64_D64,
        layer_count=layer_count,
        expert_count=expert_count,
        expert_used_count=expert_used,
        source_tensor_count=source_tensor_count,
        descriptor_bytes=descriptors,
        data_offset=data_offset,
        data_size=cursor - data_offset,
        store_size=cursor,
        layers=layers,
    )


def plan_deepseek_mlx_affine2(
        model_dir: Path,
        expected_revision: str) -> None:
    donor = load_deepseek_affine2_donor(
        model_dir, expected_revision,
        require_hydrated=False, verify_shards=False,
    )
    plan = make_deepseek_affine2_store_plan(
        donor.source_size, donor.source_tensor_count
    )

    gate_expert_bytes = plan.layers[0].components[0].expert_bytes
    up_expert_bytes = plan.layers[0].components[1].expert_bytes
    down_expert_bytes = plan.layers[0].components[2].expert_bytes
    record_bytes = plan.layers[0].record_bytes
    layer_bytes = plan.layers[0].data_size
    if (gate_expert_bytes, up_expert_bytes, down_expert_bytes,
            record_bytes) != (3145728, 2621440, 2621440, 8388608):
        raise FormatError("internal affine2 layout calculation differs")

    print("mode: plan-only")
    print(f"donor_origin: {donor.origin}")
    print(f"donor_revision: {donor.revision}")
    print(f"config_sha256: {donor.config_sha256}")
    print(f"index_sha256: {donor.index_sha256}")
    print(f"source_manifest_sha256: {donor.source_digest.hex()}")
    print(f"index_tensors: {donor.source_tensor_count}")
    print(f"routed_source_tensors: {len(deepseek_affine2_expected_tensors(donor.gate_groups))}")
    print(f"shards: {len(donor.shard_oids)}")
    print(f"hydrated_shards: {len(donor.shard_oids) if donor.hydrated else 0}")
    print(f"source_model_bytes: {donor.source_size}")
    print(f"source_shard_file_bytes: {sum(item[2] for item in donor.shard_oids)}")
    print(f"layers: {plan.layer_count}")
    print(f"experts: {plan.expert_count}")
    print(f"gate_expert_bytes: {gate_expert_bytes}")
    print(f"up_expert_bytes: {up_expert_bytes}")
    print(f"down_expert_bytes: {down_expert_bytes}")
    print(f"record_bytes: {record_bytes}")
    print(f"layer_bytes: {layer_bytes}")
    print(f"payload_bytes: {plan.data_size}")
    print(f"store_bytes: {plan.store_size}")


def deepseek_affine2_component(
        mlx,
        layer: int,
        expert: int,
        role: int,
        hidden_size: int,
        intermediate_size: int,
        gate_source_group: int,
) -> bytes:
    specs = (
        ("gate_proj", hidden_size, intermediate_size, gate_source_group, 32),
        ("up_proj", hidden_size, intermediate_size, 64, 64),
        ("down_proj", intermediate_size, hidden_size, 64, 64),
    )
    if not 0 <= role < len(specs):
        raise FormatError(f"invalid DeepSeek affine2 role {role}")
    role_name, input_dim, rows, source_group, target_group = specs[role]
    prefix = f"model.layers.{layer}.ffn.switch_mlp.{role_name}"
    weights = mlx.expert_bytes(
        prefix + ".weight", "U32",
        (mlx.expert_count, rows, input_dim // 16), expert,
    )
    scales = mlx.expert_bytes(
        prefix + ".scales", "BF16",
        (mlx.expert_count, rows, input_dim // source_group), expert,
    )
    biases = mlx.expert_bytes(
        prefix + ".biases", "BF16",
        (mlx.expert_count, rows, input_dim // source_group), expert,
    )
    return interleave_mlx_affine2(
        weights, scales, biases, rows, input_dim,
        source_group, target_group,
    )


def raw_store(path: Path) -> tuple[dict[str, object], list[Layer]]:
    path = path.resolve()
    size = path.stat().st_size
    fake = GGUF(path, size, 0, 0, STORE_ALIGNMENT, b"", {}, [], 0)
    tensor = Tensor(STORE_TENSOR, (size,), 24, 0, size, abs_offset=0)
    return parse_store(fake, tensor)


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _volume_is_internal(path: Path) -> bool | None:
    if sys.platform != "darwin":
        return None
    try:
        result = subprocess.run(
            ["/usr/sbin/diskutil", "info", "-plist", str(path)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        if result.returncode:
            return None
        info = plistlib.loads(result.stdout)
        value = info.get("Internal")
        return value if isinstance(value, bool) else None
    except (OSError, plistlib.InvalidFileException):
        return None


def check_deepseek_output_space(
        destination: Path, required: int, reserve: int) -> None:
    if reserve < 0:
        raise FormatError("reserve space cannot be negative")
    free = shutil.disk_usage(destination.parent).free
    if free < required + reserve:
        internal = _volume_is_internal(destination.parent)
        kind = "internal volume" if internal else \
            ("external volume" if internal is False else "destination volume")
        raise FormatError(
            f"insufficient free space on {kind}: need {required + reserve} "
            f"bytes ({required} store + {reserve} reserve), have {free}"
        )


def _checkpoint_paths(destination: Path) -> tuple[Path, Path]:
    partial = destination.with_name(f".{destination.name}.partial")
    state = destination.with_name(f".{destination.name}.resume.json")
    return partial, state


def _checkpoint_identity(
        plan: StorePlan,
        source_digest: bytes,
        gate_groups: tuple[int, ...],
) -> dict[str, object]:
    return {
        "schema": "ds4-deepseek-mlx-affine2-resume-v1",
        "source_manifest_sha256": source_digest.hex(),
        "descriptor_sha256": hashlib.sha256(plan.descriptor_bytes).hexdigest(),
        "store_size": plan.store_size,
        "data_offset": plan.data_offset,
        "layer_count": plan.layer_count,
        "expert_count": plan.expert_count,
        "gate_source_groups": list(gate_groups),
    }


def _write_checkpoint(path: Path, state: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if temporary.exists():
        raise FormatError(f"checkpoint temporary path already exists: {temporary}")
    fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        data = (json.dumps(state, sort_keys=True, indent=2) + "\n").encode()
        pwrite_all(fd, data, 0)
        os.fsync(fd)
    finally:
        os.close(fd)
    try:
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _hash_fd_range(
        fd: int, offset: int, size: int, digest: Digest) -> None:
    completed = 0
    while completed < size:
        take = min(IO_BYTES, size - completed)
        digest.update(pread_exact(fd, take, offset + completed))
        completed += take


def verify_deepseek_affine2_store_from_source(
        mlx,
        store_path: Path,
        plan: StorePlan,
        gate_groups: tuple[int, ...],
        source_digest: bytes,
        *,
        hidden_size: int = 4096,
        intermediate_size: int = 2048,
) -> bytes:
    manifest, layers = raw_store(store_path)
    if (manifest["family"] != STORE_FAMILY_DEEPSEEK4 or
            manifest["storage_format"] != STORE_STORAGE_MLX_AFFINE2 or
            manifest["group_size"] !=
                STORE_GROUP_PROFILE_AFFINE2_G32_U64_D64 or
            manifest["layer_count"] != plan.layer_count or
            manifest["expert_count"] != plan.expert_count or
            manifest["expert_used"] != plan.expert_used_count or
            manifest["source_tensors"] != plan.source_tensor_count or
            manifest["source_size"] != plan.source.size or
            manifest["source_sha256"] != source_digest or
            len(layers) != len(plan.layers)):
        raise FormatError("DeepSeek affine2 store identity differs")
    if len(gate_groups) != plan.layer_count:
        raise FormatError("DeepSeek affine2 gate group inventory differs")
    fd = os.open(store_path, os.O_RDONLY)
    payload_hash = hashlib.sha256()
    try:
        manifest_end = STORE_HEADER_BYTES + len(plan.descriptor_bytes)
        header_padding = pread_exact(
            fd, plan.data_offset - manifest_end, manifest_end
        )
        if any(header_padding):
            raise FormatError("non-zero DeepSeek affine2 manifest padding")
        cursor = plan.data_offset
        for ordinal, (expected_layer, actual_layer) in enumerate(
                zip(plan.layers, layers), 1):
            if pack_affine2_layer(actual_layer) != pack_affine2_layer(expected_layer):
                raise FormatError(
                    f"DeepSeek affine2 descriptor differs at layer {ordinal - 1}"
                )
            gap = actual_layer.data_offset - cursor
            if gap:
                padding = pread_exact(fd, gap, cursor)
                if any(padding):
                    raise FormatError(
                        f"non-zero padding before affine2 layer {actual_layer.index}"
                    )
                payload_hash.update(padding)
            for expert in range(plan.expert_count):
                for component in actual_layer.components:
                    expected = deepseek_affine2_component(
                        mlx, actual_layer.index, expert, component.role,
                        hidden_size, intermediate_size,
                        gate_groups[actual_layer.index],
                    )
                    if len(expected) != component.expert_bytes:
                        raise FormatError(
                            f"affine2 component size differs at layer "
                            f"{actual_layer.index} expert {expert} role "
                            f"{component.role}"
                        )
                    offset = (actual_layer.data_offset +
                              expert * actual_layer.record_bytes +
                              component.record_offset)
                    actual = pread_exact(fd, component.expert_bytes, offset)
                    payload_hash.update(actual)
                    if actual != expected:
                        raise FormatError(
                            f"affine2 payload differs at layer "
                            f"{actual_layer.index} expert {expert} role "
                            f"{component.role}"
                        )
            cursor = actual_layer.data_offset + actual_layer.data_size
            print(
                f"\rverify DeepSeek affine2 layers {ordinal}/{plan.layer_count}",
                end="", file=sys.stderr, flush=True,
            )
        print(file=sys.stderr)
        if cursor != plan.store_size:
            raise FormatError("DeepSeek affine2 verification did not cover the store")
    finally:
        os.close(fd)
    digest = payload_hash.digest()
    if digest != manifest["payload_sha256"]:
        raise FormatError("DeepSeek affine2 payload SHA-256 mismatch")
    return digest


def write_deepseek_affine2_store_from_source(
        mlx,
        destination: Path,
        plan: StorePlan,
        gate_groups: tuple[int, ...],
        source_digest: bytes,
        reserve: int,
        *,
        resume: bool,
        verify_after: bool,
        hidden_size: int = 4096,
        intermediate_size: int = 2048,
        progress_hook: Callable[[int], None] | None = None,
) -> bytes:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FormatError(f"destination already exists: {destination}")
    if len(gate_groups) != plan.layer_count or len(source_digest) != 32:
        raise FormatError("invalid DeepSeek affine2 writer identity")
    partial, state_path = _checkpoint_paths(destination)
    identity = _checkpoint_identity(plan, source_digest, gate_groups)
    completed_layers = 0
    fd = -1
    owns_partial = False
    payload_hash = hashlib.sha256()
    try:
        if partial.exists() or state_path.exists():
            if not resume or not partial.exists() or not state_path.exists():
                raise FormatError(
                    f"incomplete output exists; use --resume after checking "
                    f"{partial} and {state_path}"
                )
            try:
                checkpoint = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise FormatError(f"invalid resume checkpoint: {exc}") from exc
            completed_layers = checkpoint.pop("completed_layers", None)
            if checkpoint != identity or not isinstance(completed_layers, int) or \
                    not 0 <= completed_layers <= plan.layer_count:
                raise FormatError("resume checkpoint identity differs")
            if partial.stat().st_size != plan.store_size:
                raise FormatError("resume partial size differs")
            fd = os.open(partial, os.O_RDWR)
            provisional = make_header(plan, source_digest, bytes(32))
            provisional = make_header(
                plan, source_digest, bytes(32),
                manifest_digest(provisional, plan.descriptor_bytes),
            )
            expected_prefix = provisional + plan.descriptor_bytes
            if pread_exact(fd, len(expected_prefix), 0) != expected_prefix:
                raise FormatError("resume partial manifest differs")
            if any(pread_exact(
                    fd, plan.data_offset - len(expected_prefix),
                    len(expected_prefix))):
                raise FormatError("resume partial header padding differs")
            completed_end = plan.layers[completed_layers - 1].data_offset + \
                plan.layers[completed_layers - 1].data_size \
                if completed_layers else plan.data_offset
            check_deepseek_output_space(
                destination, plan.store_size - completed_end, reserve
            )
            _hash_fd_range(
                fd, plan.data_offset, completed_end - plan.data_offset,
                payload_hash,
            )
            print(
                f"resume DeepSeek affine2 at layer {completed_layers}/"
                f"{plan.layer_count}", file=sys.stderr,
            )
        else:
            check_deepseek_output_space(destination, plan.store_size, reserve)
            fd = os.open(partial, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o644)
            owns_partial = True
            os.ftruncate(fd, plan.store_size)
            provisional = make_header(plan, source_digest, bytes(32))
            provisional = make_header(
                plan, source_digest, bytes(32),
                manifest_digest(provisional, plan.descriptor_bytes),
            )
            pwrite_all(fd, provisional, 0)
            pwrite_all(fd, plan.descriptor_bytes, STORE_HEADER_BYTES)
            if resume:
                _write_checkpoint(
                    state_path, {**identity, "completed_layers": 0}
                )

        for ordinal in range(completed_layers, plan.layer_count):
            layer = plan.layers[ordinal]
            for expert in range(plan.expert_count):
                for component in layer.components:
                    packed = deepseek_affine2_component(
                        mlx, layer.index, expert, component.role,
                        hidden_size, intermediate_size,
                        gate_groups[layer.index],
                    )
                    if len(packed) != component.expert_bytes:
                        raise FormatError(
                            f"affine2 record size differs at layer {layer.index} "
                            f"expert {expert} role {component.role}"
                        )
                    offset = (layer.data_offset + expert * layer.record_bytes +
                              component.record_offset)
                    pwrite_all(fd, packed, offset)
                    payload_hash.update(packed)
            os.fsync(fd)
            if resume:
                _write_checkpoint(
                    state_path,
                    {**identity, "completed_layers": ordinal + 1},
                )
            print(
                f"\rwrite DeepSeek affine2 layers {ordinal + 1}/"
                f"{plan.layer_count}",
                end="", file=sys.stderr, flush=True,
            )
            if progress_hook is not None:
                progress_hook(ordinal + 1)
        print(file=sys.stderr)
        payload_digest = payload_hash.digest()
        provisional = make_header(plan, source_digest, payload_digest)
        header = make_header(
            plan, source_digest, payload_digest,
            manifest_digest(provisional, plan.descriptor_bytes),
        )
        pwrite_all(fd, header, 0)
        os.fsync(fd)
        os.close(fd)
        fd = -1
        if verify_after:
            verified = verify_deepseek_affine2_store_from_source(
                mlx, partial, plan, gate_groups, source_digest,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
            )
            if verified != payload_digest:
                raise FormatError("post-write affine2 digest differs")
        if destination.exists():
            raise FormatError(f"destination appeared during build: {destination}")
        os.replace(partial, destination)
        _fsync_directory(destination.parent)
        state_path.unlink(missing_ok=True)
        _fsync_directory(destination.parent)
    except BaseException:
        if fd >= 0:
            os.close(fd)
        if owns_partial and (not resume or not state_path.exists()):
            partial.unlink(missing_ok=True)
            state_path.unlink(missing_ok=True)
        raise
    return payload_digest


def write_deepseek_mlx_affine2(
        model_dir: Path,
        destination: Path,
        expected_revision: str,
        reserve: int,
        *,
        resume: bool,
        verify_after: bool,
) -> None:
    donor = load_deepseek_affine2_donor(
        model_dir, expected_revision,
        require_hydrated=True, verify_shards=False,
    )
    plan = make_deepseek_affine2_store_plan(
        donor.source_size, donor.source_tensor_count
    )
    # Reject an undersized destination before spending a full sequential pass
    # hashing the 19 hydrated source shards.
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial, state_path = _checkpoint_paths(destination)
    if partial.exists() or state_path.exists():
        if not resume or not partial.exists() or not state_path.exists():
            raise FormatError(
                f"incomplete output exists; use --resume after checking "
                f"{partial} and {state_path}"
            )
        # The exact remaining payload is checked after checkpoint validation.
        check_deepseek_output_space(destination, 0, reserve)
    else:
        check_deepseek_output_space(destination, plan.store_size, reserve)
    for ordinal, (name, oid, size) in enumerate(donor.shard_oids, 1):
        path = donor.model_dir / name
        if path.stat().st_size != size or \
                hash_file(path, f"hash donor shard {ordinal}/19").hex() != oid:
            raise FormatError(f"Git LFS SHA-256 mismatch for {name}")
    mlx = MLXAffineSource(donor.model_dir)
    mlx.expert_count = plan.expert_count
    try:
        payload_digest = write_deepseek_affine2_store_from_source(
            mlx, destination, plan, donor.gate_groups,
            donor.source_digest, reserve,
            resume=resume, verify_after=verify_after,
        )
    finally:
        mlx.close()
    print(f"installed atomically: {destination.resolve()}")
    print("storage: mlx-affine2-gate32-up64-down64")
    print(f"donor_origin: {donor.origin}")
    print(f"donor_revision: {donor.revision}")
    print(f"config_sha256: {donor.config_sha256}")
    print(f"index_sha256: {donor.index_sha256}")
    print(f"source_manifest_sha256: {donor.source_digest.hex()}")
    print(f"payload_sha256: {payload_digest.hex()}")
    print(f"store_bytes: {plan.store_size}")


def verify_deepseek_mlx_affine2(
        model_dir: Path,
        store_path: Path,
        expected_revision: str,
) -> None:
    donor = load_deepseek_affine2_donor(
        model_dir, expected_revision,
        require_hydrated=True, verify_shards=True,
    )
    plan = make_deepseek_affine2_store_plan(
        donor.source_size, donor.source_tensor_count
    )
    mlx = MLXAffineSource(donor.model_dir)
    mlx.expert_count = plan.expert_count
    try:
        digest = verify_deepseek_affine2_store_from_source(
            mlx, store_path, plan, donor.gate_groups, donor.source_digest
        )
    finally:
        mlx.close()
    print(f"valid DeepSeek MLX affine2 ExpertMajor v2 store: {store_path.resolve()}")
    print(f"source_manifest_sha256: {donor.source_digest.hex()}")
    print(f"payload_sha256: {digest.hex()}")


def validate_gguf_tensor_extents(source: GGUF) -> None:
    names: set[str] = set()
    previous_end = source.data_offset
    for tensor in sorted(source.tensors, key=lambda item: item.abs_offset):
        if tensor.name in names:
            raise FormatError(f"duplicate GGUF tensor name: {tensor.name}")
        names.add(tensor.name)
        if tensor.rel_offset % source.alignment or \
                tensor.abs_offset < previous_end:
            raise FormatError(
                f"overlapping or unaligned GGUF tensor: {tensor.name}"
            )
        previous_end = tensor.abs_offset + tensor.size


def deepseek_affine2_hybrid_inputs(
        source_path: Path,
        store_path: Path,
        *,
        allow_test_geometry: bool = False,
) -> tuple[GGUF, Tensor, dict[str, object], list[Layer],
           dict[str, object], list[Layer]]:
    source = load_gguf(source_path)
    if source.metadata.get("general.architecture") != "deepseek4":
        raise FormatError("affine2 embedding requires a DeepSeek4 GGUF")
    validate_gguf_tensor_extents(source)
    stores = [tensor for tensor in source.tensors if tensor.name == STORE_TENSOR]
    if len(stores) != 1 or any(ROUTED_RE.fullmatch(tensor.name)
                               for tensor in source.tensors):
        raise FormatError(
            "source GGUF must contain one ExpertMajor v2 store and no "
            "canonical routed tensors"
        )
    source_store = stores[0]
    source_manifest, source_layers = parse_store(source, source_store)
    replacement_manifest, replacement_layers = raw_store(store_path)
    try:
        metadata_identity = (
            int(source.metadata["deepseek4.block_count"]),
            int(source.metadata["deepseek4.expert_count"]),
            int(source.metadata["deepseek4.expert_used_count"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise FormatError("DeepSeek GGUF expert metadata is incomplete") from exc
    if (source_manifest["family"] != STORE_FAMILY_DEEPSEEK4 or
            replacement_manifest["family"] != STORE_FAMILY_DEEPSEEK4 or
            replacement_manifest["storage_format"] !=
                STORE_STORAGE_MLX_AFFINE2 or
            replacement_manifest["group_size"] !=
                STORE_GROUP_PROFILE_AFFINE2_G32_U64_D64 or
            metadata_identity != (
                replacement_manifest["layer_count"],
                replacement_manifest["expert_count"],
                replacement_manifest["expert_used"],
            ) or
            (source_manifest["layer_count"],
             source_manifest["expert_count"],
             source_manifest["expert_used"]) != metadata_identity or
            len(source_layers) != len(replacement_layers)):
        raise FormatError("replacement affine2 store identity differs from GGUF")
    if not allow_test_geometry:
        expected_dims = (
            (4096, 2048, 256),
            (4096, 2048, 256),
            (2048, 4096, 256),
        )
        if (metadata_identity != (43, 256, 6) or
                replacement_manifest["source_tensors"] !=
                    DEEPSEEK_AFFINE2_TENSOR_COUNT or
                replacement_manifest["source_size"] !=
                    DEEPSEEK_AFFINE2_TOTAL_SIZE or
                replacement_manifest["source_sha256"] != bytes.fromhex(
                    DEEPSEEK_AFFINE2_SOURCE_MANIFEST_SHA256
                ) or
                any(tuple(component.tensor.dims for component in layer.components)
                    != expected_dims for layer in replacement_layers)):
            raise FormatError(
                "replacement store is not the pinned DeepSeek affine2 donor"
            )
    for source_layer, replacement_layer in zip(
            source_layers, replacement_layers):
        if (source_layer.index != replacement_layer.index or
                source_layer.expert_count != replacement_layer.expert_count or
                any(old.tensor.dims != new.tensor.dims
                    for old, new in zip(source_layer.components,
                                        replacement_layer.components))):
            raise FormatError(
                f"replacement affine2 geometry differs at layer "
                f"{source_layer.index}"
            )
    return (source, source_store, source_manifest, source_layers,
            replacement_manifest, replacement_layers)


def replacement_gguf_layout(
        source: GGUF,
        replacement_store_bytes: int,
) -> tuple[list[Tensor], int, int]:
    tensors: list[Tensor] = []
    replaced = 0
    for tensor in source.tensors:
        if tensor.name == STORE_TENSOR:
            tensors.append(Tensor(
                STORE_TENSOR, (replacement_store_bytes,), 24, 0,
                replacement_store_bytes,
            ))
            replaced += 1
        else:
            tensors.append(dataclasses.replace(tensor))
    if replaced != 1:
        raise FormatError("GGUF replacement requires exactly one expert store")
    directory_bytes = sum(
        len(pack_string(tensor.name)) + 4 + 8 * len(tensor.dims) + 4 + 8
        for tensor in tensors
    )
    metadata_end = 4 + 4 + 8 + 8 + len(source.kv_raw) + directory_bytes
    data_offset = align_up(metadata_end, source.alignment)
    cursor = 0
    for tensor in tensors:
        cursor = align_up(cursor, source.alignment)
        tensor.new_rel_offset = cursor
        cursor += tensor.size
    return tensors, data_offset, data_offset + cursor


def _copy_standalone_store(
        source_fd: int,
        output_fd: int,
        output_offset: int,
        manifest: dict[str, object],
) -> tuple[bytes, bytes]:
    store_size = int(manifest["store_size"])
    payload_begin = int(manifest["data_offset"])
    payload_end = store_size
    store_hash = hashlib.sha256()
    payload_hash = hashlib.sha256()
    completed = 0
    while completed < store_size:
        take = min(IO_BYTES, store_size - completed)
        data = pread_exact(source_fd, take, completed)
        pwrite_all(output_fd, data, output_offset + completed)
        store_hash.update(data)
        begin = max(completed, payload_begin)
        end = min(completed + take, payload_end)
        if begin < end:
            payload_hash.update(data[begin - completed:end - completed])
        completed += take
    payload_digest = payload_hash.digest()
    if payload_digest != manifest["payload_sha256"]:
        raise FormatError("standalone affine2 store payload SHA-256 mismatch")
    return store_hash.digest(), payload_digest


def _compare_standalone_store(
        source_fd: int,
        output_fd: int,
        output_offset: int,
        manifest: dict[str, object],
) -> bytes:
    store_size = int(manifest["store_size"])
    payload_begin = int(manifest["data_offset"])
    digest = hashlib.sha256()
    completed = 0
    while completed < store_size:
        take = min(IO_BYTES, store_size - completed)
        source = pread_exact(source_fd, take, completed)
        embedded = pread_exact(output_fd, take, output_offset + completed)
        if source != embedded:
            raise FormatError(
                f"embedded affine2 store differs at byte {completed}"
            )
        begin = max(completed, payload_begin)
        end = min(completed + take, store_size)
        if begin < end:
            digest.update(source[begin - completed:end - completed])
        completed += take
    payload_digest = digest.digest()
    if payload_digest != manifest["payload_sha256"]:
        raise FormatError("standalone affine2 store payload SHA-256 mismatch")
    return payload_digest


def verify_deepseek_affine2_hybrid_gguf(
        source_path: Path,
        store_path: Path,
        output_path: Path,
        *,
        allow_test_geometry: bool = False,
) -> bytes:
    (source, _, _, _, replacement_manifest, _) = \
        deepseek_affine2_hybrid_inputs(
            source_path, store_path,
            allow_test_geometry=allow_test_geometry,
        )
    output = load_gguf(output_path)
    expected_tensors, expected_data_offset, expected_size = \
        replacement_gguf_layout(source, int(replacement_manifest["store_size"]))
    if (output.version != source.version or output.n_kv != source.n_kv or
            output.kv_raw != source.kv_raw or
            output.alignment != source.alignment or
            output.data_offset != expected_data_offset or
            output.size != expected_size or
            len(output.tensors) != len(expected_tensors)):
        raise FormatError("hybrid GGUF header or metadata differs")
    for expected, actual in zip(expected_tensors, output.tensors):
        if (expected.name, expected.dims, expected.ggml_type,
                expected.new_rel_offset, expected.size) != \
                (actual.name, actual.dims, actual.ggml_type,
                 actual.rel_offset, actual.size):
            raise FormatError(f"hybrid GGUF descriptor differs: {expected.name}")
    output_store = next(
        tensor for tensor in output.tensors if tensor.name == STORE_TENSOR
    )
    output_manifest, _ = parse_store(output, output_store)
    if output_manifest["header"] != replacement_manifest["header"]:
        raise FormatError("embedded affine2 store manifest differs")

    manifest_end = output.data_offset
    with output.path.open("rb", buffering=0) as file:
        file.seek(0)
        header_and_padding = read_exact(file, manifest_end)
    # The metadata and parsed descriptors above cover the meaningful header;
    # require deterministic zero padding up to the tensor data section.
    directory_end = 4 + 4 + 8 + 8 + len(output.kv_raw) + sum(
        len(pack_string(tensor.name)) + 4 + 8 * len(tensor.dims) + 4 + 8
        for tensor in output.tensors
    )
    if any(header_and_padding[directory_end:]):
        raise FormatError("hybrid GGUF header padding is non-zero")

    source_by_name = {tensor.name: tensor for tensor in source.tensors}
    source_fd = os.open(source.path, os.O_RDONLY)
    store_fd = os.open(store_path, os.O_RDONLY)
    output_fd = os.open(output.path, os.O_RDONLY)
    try:
        previous_end = output.data_offset
        for tensor in output.tensors:
            if tensor.abs_offset > previous_end:
                padding = pread_exact(
                    output_fd, tensor.abs_offset - previous_end, previous_end
                )
                if any(padding):
                    raise FormatError(
                        f"hybrid GGUF tensor padding is non-zero before "
                        f"{tensor.name}"
                    )
            if tensor.name == STORE_TENSOR:
                _compare_standalone_store(
                    store_fd, output_fd, tensor.abs_offset,
                    replacement_manifest,
                )
            else:
                original = source_by_name[tensor.name]
                copy_range(
                    source_fd, original.abs_offset, output_fd, 0,
                    tensor.size, compare_fd=output_fd,
                    compare_offset=tensor.abs_offset,
                )
            previous_end = tensor.abs_offset + tensor.size
        if previous_end != output.size:
            raise FormatError("hybrid GGUF has unexpected trailing bytes")
    finally:
        os.close(source_fd)
        os.close(store_fd)
        os.close(output_fd)
    return hash_file(output.path, "hash hybrid GGUF")


def plan_deepseek_affine2_hybrid_gguf(
        source_path: Path,
        store_path: Path,
        *,
        allow_test_geometry: bool = False,
) -> None:
    source, source_store, source_manifest, _, replacement_manifest, _ = \
        deepseek_affine2_hybrid_inputs(
            source_path, store_path,
            allow_test_geometry=allow_test_geometry,
        )
    _, data_offset, output_size = replacement_gguf_layout(
        source, int(replacement_manifest["store_size"])
    )
    print("mode: plan-only")
    print(f"source_gguf: {source.path}")
    print(f"source_bytes: {source.size}")
    print(f"source_store_bytes: {source_store.size}")
    print(f"source_store_storage: {source_manifest['storage_format']}")
    print(f"replacement_store: {store_path.resolve()}")
    print(f"replacement_store_bytes: {replacement_manifest['store_size']}")
    print(f"replacement_payload_sha256: {bytes(replacement_manifest['payload_sha256']).hex()}")
    print(f"output_data_offset: {data_offset}")
    print(f"output_bytes: {output_size}")
    print(f"size_delta_bytes: {output_size - source.size}")


def embed_deepseek_affine2_hybrid_gguf(
        source_path: Path,
        store_path: Path,
        destination: Path,
        reserve: int,
        *,
        verify_after: bool,
        allow_test_geometry: bool = False,
) -> None:
    source, _, _, _, replacement_manifest, _ = \
        deepseek_affine2_hybrid_inputs(
            source_path, store_path,
            allow_test_geometry=allow_test_geometry,
        )
    source_stat = source.path.stat()
    store_path = store_path.resolve()
    store_stat = store_path.stat()
    source_identity = (
        source_stat.st_dev, source_stat.st_ino, source_stat.st_size,
        source_stat.st_mtime_ns,
    )
    store_identity = (
        store_stat.st_dev, store_stat.st_ino, store_stat.st_size,
        store_stat.st_mtime_ns,
    )
    tensors, data_offset, output_size = replacement_gguf_layout(
        source, int(replacement_manifest["store_size"])
    )
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FormatError(f"destination already exists: {destination}")
    check_deepseek_output_space(destination, output_size, reserve)
    source_digest = hash_file(source.path, "hash source GGUF")
    temp = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    if temp.exists():
        raise FormatError(f"temporary path already exists: {temp}")
    source_fd = os.open(source.path, os.O_RDONLY)
    store_fd = os.open(store_path, os.O_RDONLY)
    output_fd = os.open(temp, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o644)
    store_digest = bytes(32)
    payload_digest = bytes(32)
    try:
        os.ftruncate(output_fd, output_size)
        write_native_header(output_fd, source, tensors, data_offset)
        source_by_name = {tensor.name: tensor for tensor in source.tensors}
        for ordinal, tensor in enumerate(tensors, 1):
            output_offset = data_offset + tensor.new_rel_offset
            if tensor.name == STORE_TENSOR:
                store_digest, payload_digest = _copy_standalone_store(
                    store_fd, output_fd, output_offset, replacement_manifest
                )
            else:
                original = source_by_name[tensor.name]
                copy_range(
                    source_fd, original.abs_offset, output_fd,
                    output_offset, tensor.size,
                )
            if ordinal % 50 == 0 or ordinal == len(tensors):
                print(
                    f"\rwrite hybrid GGUF tensors {ordinal}/{len(tensors)}",
                    end="", file=sys.stderr, flush=True,
                )
        print(file=sys.stderr)
        os.fsync(output_fd)
        os.close(output_fd)
        output_fd = -1
        if verify_after:
            output_digest = verify_deepseek_affine2_hybrid_gguf(
                source.path, store_path, temp,
                allow_test_geometry=allow_test_geometry,
            )
        else:
            output_digest = bytes(32)
        current_source = source.path.stat()
        current_store = store_path.stat()
        if source_identity != (
                current_source.st_dev, current_source.st_ino,
                current_source.st_size, current_source.st_mtime_ns) or \
                store_identity != (
                    current_store.st_dev, current_store.st_ino,
                    current_store.st_size, current_store.st_mtime_ns):
            raise FormatError("hybrid GGUF inputs changed during the build")
        if destination.exists():
            raise FormatError(f"destination appeared during build: {destination}")
        os.replace(temp, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        if output_fd >= 0:
            os.close(output_fd)
        temp.unlink(missing_ok=True)
        raise
    finally:
        os.close(source_fd)
        os.close(store_fd)
    print(f"installed atomically: {destination}")
    print(f"source_gguf_sha256: {source_digest.hex()}")
    print(f"standalone_store_sha256: {store_digest.hex()}")
    print(f"payload_sha256: {payload_digest.hex()}")
    if verify_after:
        print(f"output_gguf_sha256: {output_digest.hex()}")
    else:
        print("output_gguf_sha256: skipped")
    print(f"output_bytes: {output_size}")


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
    build_parser.add_argument("source", type=Path)
    build_parser.add_argument("destination", type=Path)
    verify_parser = subparsers.add_parser("verify")
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
    deepseek_affine_parser = subparsers.add_parser(
        "repack-deepseek-mlx-affine2",
        help=("write the pinned DeepSeek MLX affine2 routed weights as one "
              "standalone ExpertMajor v2 store"),
    )
    deepseek_affine_parser.add_argument(
        "--dry-run", action="store_true",
        help="validate provenance/index and print the layout without writing",
    )
    deepseek_affine_parser.add_argument(
        "--expected-revision", required=True,
        help="full pinned donor Git revision",
    )
    deepseek_affine_parser.add_argument(
        "--reserve-bytes", type=parse_bytes, default=1 << 30,
    )
    deepseek_affine_parser.add_argument(
        "--resume", action="store_true",
        help="checkpoint each completed layer and resume a matching partial",
    )
    deepseek_affine_parser.add_argument(
        "--skip-verify", action="store_true",
        help="diagnostic only; publication builds verify byte-for-byte by default",
    )
    deepseek_affine_parser.add_argument("mlx_model", type=Path)
    deepseek_affine_parser.add_argument("destination", type=Path, nargs="?")
    deepseek_verify_parser = subparsers.add_parser(
        "verify-deepseek-mlx-affine2",
        help="verify a standalone affine2 store against the pinned MLX donor",
    )
    deepseek_verify_parser.add_argument(
        "--expected-revision", required=True,
        help="full pinned donor Git revision",
    )
    deepseek_verify_parser.add_argument("mlx_model", type=Path)
    deepseek_verify_parser.add_argument("store", type=Path)
    hybrid_parser = subparsers.add_parser(
        "embed-deepseek-mlx-affine2",
        help="replace the routed store in a DeepSeek GGUF with affine2",
    )
    hybrid_parser.add_argument(
        "--dry-run", action="store_true",
        help="validate compatibility and print the rebuilt GGUF layout",
    )
    hybrid_parser.add_argument(
        "--reserve-bytes", type=parse_bytes, default=1 << 30,
    )
    hybrid_parser.add_argument(
        "--skip-verify", action="store_true",
        help="diagnostic only; publication builds verify byte-for-byte by default",
    )
    hybrid_parser.add_argument("source", type=Path)
    hybrid_parser.add_argument("store", type=Path)
    hybrid_parser.add_argument("destination", type=Path, nargs="?")
    hybrid_verify_parser = subparsers.add_parser(
        "verify-deepseek-mlx-affine2-gguf",
        help="verify a rebuilt DeepSeek affine2 GGUF byte-for-byte",
    )
    hybrid_verify_parser.add_argument("source", type=Path)
    hybrid_verify_parser.add_argument("store", type=Path)
    hybrid_verify_parser.add_argument("output", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            inspect(args.source)
        elif args.command == "build":
            build(args.source, args.destination, args.reserve_bytes,
                  not args.skip_verify)
        elif args.command == "verify":
            verify(args.source, args.native)
        elif args.command == "repack-mlx-affine":
            repack_mlx_affine(
                args.native, args.mlx_model, args.destination,
                args.reserve_bytes,
            )
        elif args.command == "repack-deepseek-mlx-affine2":
            if args.dry_run:
                if args.destination is not None:
                    raise FormatError("--dry-run does not accept a destination")
                plan_deepseek_mlx_affine2(
                    args.mlx_model, args.expected_revision,
                )
            else:
                if args.destination is None:
                    raise FormatError("destination is required unless --dry-run is used")
                write_deepseek_mlx_affine2(
                    args.mlx_model, args.destination, args.expected_revision,
                    args.reserve_bytes, resume=args.resume,
                    verify_after=not args.skip_verify,
                )
        elif args.command == "verify-deepseek-mlx-affine2":
            verify_deepseek_mlx_affine2(
                args.mlx_model, args.store, args.expected_revision,
            )
        elif args.command == "embed-deepseek-mlx-affine2":
            if args.dry_run:
                if args.destination is not None:
                    raise FormatError("--dry-run does not accept a destination")
                plan_deepseek_affine2_hybrid_gguf(args.source, args.store)
            else:
                if args.destination is None:
                    raise FormatError("destination is required unless --dry-run is used")
                embed_deepseek_affine2_hybrid_gguf(
                    args.source, args.store, args.destination,
                    args.reserve_bytes, verify_after=not args.skip_verify,
                )
        elif args.command == "verify-deepseek-mlx-affine2-gguf":
            digest = verify_deepseek_affine2_hybrid_gguf(
                args.source, args.store, args.output,
            )
            print(f"valid DeepSeek affine2 hybrid GGUF: {args.output.resolve()}")
            print(f"output_gguf_sha256: {digest.hex()}")
        return 0
    except (FormatError, OSError) as exc:
        print(f"ds4-expert-major: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
