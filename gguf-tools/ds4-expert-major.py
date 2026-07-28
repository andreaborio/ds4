#!/usr/bin/env python3
"""Build and verify DS4 expert-major v2 GGUFs.

The converter changes storage only: non-routed tensors and every GGUF metadata
record are copied byte-for-byte, while each routed layer becomes a sequence of
complete expert records (gate, up, down). The opaque store is self-describing;
DS4 reconstructs the canonical logical tensor inventory at load time.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import os
import re
import shutil
import struct
import sys
from pathlib import Path
from typing import BinaryIO, Iterator, Protocol


MAGIC = b"GGUF"
STORE_MAGIC = b"DS4EXPV2"
STORE_TENSOR = "ds4.expert_major.v2"
STORE_VERSION = 2
STORE_FAMILY_DEEPSEEK4 = 1
STORE_FAMILY_GLM_DSA = 2
STORE_FAMILY_QWEN35_MOE = 3
STORE_STORAGE_GGML = 0
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
            storage_format == STORE_STORAGE_GGML and group_size == 0
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
            if (entry_role != role or ndim != 3 or ggml_type not in ROUTED_TYPES or
                    TYPE_LAYOUT[ggml_type][0] != block_elements or
                    d2 != expert_count or record_offset != record_cursor):
                raise FormatError(f"invalid component descriptor at layer {il} role {role}")
            expected = tensor_nbytes(ggml_type, (d0, d1, 1))
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
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            inspect(args.source)
        elif args.command == "build":
            build(args.source, args.destination, args.reserve_bytes,
                  not args.skip_verify)
        elif args.command == "verify":
            verify(args.source, args.native)
        return 0
    except (FormatError, OSError) as exc:
        print(f"ds4-expert-major: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
