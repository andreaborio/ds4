"""Reviewed header-only schema for the final-0731 DSpark support GGUF.

The parser intentionally stops after GGUF metadata and the tensor directory.
It never maps or reads tensor payloads, so it can validate a real support file
without turning its multi-gigabyte weights into a test fixture.
"""

from __future__ import annotations

import argparse
import dataclasses
import struct
from pathlib import Path
from typing import BinaryIO, Mapping

from .metadata import DSparkMetadata, validate_0731_metadata


GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q2_K = 10
GGML_TYPE_IQ2_XXS = 16


class SupportSchemaError(ValueError):
    """The GGUF header does not describe the exact final support inventory."""


@dataclasses.dataclass(frozen=True)
class TensorSpec:
    ggml_type: int
    dimensions: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class SupportHeader:
    version: int
    metadata: Mapping[str, object]
    tensors: Mapping[str, TensorSpec]
    offsets: Mapping[str, int] = dataclasses.field(default_factory=dict)
    header_end: int = 0


_BLOCK_TENSORS: dict[str, TensorSpec] = {
    "hc_attn_base.weight": TensorSpec(GGML_TYPE_F32, (24,)),
    "hc_attn_fn.weight": TensorSpec(GGML_TYPE_F16, (16384, 24)),
    "hc_attn_scale.weight": TensorSpec(GGML_TYPE_F32, (3,)),
    "attn_sinks.weight": TensorSpec(GGML_TYPE_F32, (64,)),
    "attn_q_a.weight": TensorSpec(GGML_TYPE_Q8_0, (4096, 1024)),
    "attn_q_a_norm.weight": TensorSpec(GGML_TYPE_F32, (1024,)),
    "attn_q_b.weight": TensorSpec(GGML_TYPE_Q8_0, (1024, 32768)),
    "attn_kv.weight": TensorSpec(GGML_TYPE_Q8_0, (4096, 512)),
    "attn_kv_a_norm.weight": TensorSpec(GGML_TYPE_F32, (512,)),
    "attn_output_a.weight": TensorSpec(GGML_TYPE_Q8_0, (4096, 8192)),
    "attn_output_b.weight": TensorSpec(GGML_TYPE_Q8_0, (8192, 4096)),
    "attn_norm.weight": TensorSpec(GGML_TYPE_F32, (4096,)),
    "hc_ffn_base.weight": TensorSpec(GGML_TYPE_F32, (24,)),
    "hc_ffn_fn.weight": TensorSpec(GGML_TYPE_F16, (16384, 24)),
    "hc_ffn_scale.weight": TensorSpec(GGML_TYPE_F32, (3,)),
    "ffn_gate_inp.weight": TensorSpec(GGML_TYPE_Q8_0, (4096, 256)),
    "exp_probs_b.bias": TensorSpec(GGML_TYPE_F32, (256,)),
    "ffn_norm.weight": TensorSpec(GGML_TYPE_F32, (4096,)),
    "ffn_gate_exps.weight": TensorSpec(
        GGML_TYPE_IQ2_XXS, (4096, 2048, 256)
    ),
    "ffn_up_exps.weight": TensorSpec(
        GGML_TYPE_IQ2_XXS, (4096, 2048, 256)
    ),
    "ffn_down_exps.weight": TensorSpec(
        GGML_TYPE_Q2_K, (2048, 4096, 256)
    ),
    "ffn_gate_shexp.weight": TensorSpec(GGML_TYPE_Q8_0, (4096, 2048)),
    "ffn_up_shexp.weight": TensorSpec(GGML_TYPE_Q8_0, (4096, 2048)),
    "ffn_down_shexp.weight": TensorSpec(GGML_TYPE_Q8_0, (2048, 4096)),
}

_STAGE_EXTRAS: dict[int, dict[str, TensorSpec]] = {
    0: {
        "main_proj.weight": TensorSpec(GGML_TYPE_Q8_0, (12288, 4096)),
        "main_norm.weight": TensorSpec(GGML_TYPE_F32, (4096,)),
    },
    1: {},
    2: {
        "norm.weight": TensorSpec(GGML_TYPE_F32, (4096,)),
        "hc_head_base.weight": TensorSpec(GGML_TYPE_F32, (4,)),
        "hc_head_fn.weight": TensorSpec(GGML_TYPE_F16, (16384, 4)),
        "hc_head_scale.weight": TensorSpec(GGML_TYPE_F32, (1,)),
        "markov_head.markov_w1.weight": TensorSpec(
            GGML_TYPE_Q8_0, (256, 129280)
        ),
        "markov_head.markov_w2.weight": TensorSpec(
            GGML_TYPE_Q8_0, (256, 129280)
        ),
        "confidence_head.proj.weight": TensorSpec(
            GGML_TYPE_Q8_0, (4352, 1)
        ),
    },
}

_GGML_STORAGE: dict[int, tuple[int, int]] = {
    GGML_TYPE_F32: (1, 4),
    GGML_TYPE_F16: (1, 2),
    GGML_TYPE_Q8_0: (32, 34),
    GGML_TYPE_Q2_K: (256, 84),
    GGML_TYPE_IQ2_XXS: (256, 66),
}


def expected_tensor_schema() -> dict[str, TensorSpec]:
    """Return all 81 canonical tensor descriptors, keyed by GGUF name."""

    expected: dict[str, TensorSpec] = {}
    for stage in range(3):
        for suffix, spec in {**_BLOCK_TENSORS,
                             **_STAGE_EXTRAS[stage]}.items():
            expected[f"mtp.{stage}.{suffix}"] = spec
    if len(expected) != 81:
        raise AssertionError("internal DSpark support schema must contain 81 tensors")
    return expected


class _GGUFReader:
    def __init__(self, stream: BinaryIO):
        self.stream = stream

    def exact(self, size: int, label: str) -> bytes:
        data = self.stream.read(size)
        if len(data) != size:
            raise SupportSchemaError(f"truncated GGUF while reading {label}")
        return data

    def scalar(self, fmt: str, label: str) -> object:
        size = struct.calcsize("<" + fmt)
        return struct.unpack("<" + fmt, self.exact(size, label))[0]

    def string(self, label: str) -> str:
        size = int(self.scalar("Q", f"{label} length"))
        if size > 1 << 20:
            raise SupportSchemaError(f"unreasonable GGUF string length for {label}")
        try:
            return self.exact(size, label).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise SupportSchemaError(f"invalid UTF-8 in {label}") from exc

    def value(self, value_type: int, label: str) -> object:
        scalar_formats = {
            0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
            6: "f", 7: "?", 10: "Q", 11: "q", 12: "d",
        }
        if value_type in scalar_formats:
            return self.scalar(scalar_formats[value_type], label)
        if value_type == 8:
            return self.string(label)
        if value_type == 9:
            element_type = int(self.scalar("I", f"{label} array type"))
            count = int(self.scalar("Q", f"{label} array count"))
            if element_type == 9 or count > 1 << 20:
                raise SupportSchemaError(f"unsupported GGUF array in {label}")
            return [
                self.value(element_type, f"{label}[{index}]")
                for index in range(count)
            ]
        raise SupportSchemaError(
            f"unsupported GGUF metadata type {value_type} in {label}"
        )


def read_support_header(path: Path) -> SupportHeader:
    """Read metadata and tensor descriptors without touching tensor payloads."""

    with path.open("rb") as stream:
        reader = _GGUFReader(stream)
        if reader.exact(4, "magic") != b"GGUF":
            raise SupportSchemaError("not a GGUF file")
        version = int(reader.scalar("I", "version"))
        if version != 3:
            raise SupportSchemaError(
                f"final 0731 support requires GGUF version 3, got {version}"
            )
        tensor_count = int(reader.scalar("Q", "tensor count"))
        metadata_count = int(reader.scalar("Q", "metadata count"))
        if tensor_count > 4096 or metadata_count > 4096:
            raise SupportSchemaError("unreasonable GGUF header counts")

        metadata: dict[str, object] = {}
        for index in range(metadata_count):
            key = reader.string(f"metadata key {index}")
            if key in metadata:
                raise SupportSchemaError(f"duplicate GGUF metadata key {key}")
            value_type = int(reader.scalar("I", f"metadata type {key}"))
            metadata[key] = reader.value(value_type, key)

        tensors: dict[str, TensorSpec] = {}
        offsets: dict[str, int] = {}
        for index in range(tensor_count):
            name = reader.string(f"tensor name {index}")
            dimensions_count = int(reader.scalar("I", f"tensor rank {name}"))
            if dimensions_count > 4:
                raise SupportSchemaError(f"unsupported tensor rank for {name}")
            dimensions = tuple(
                int(reader.scalar("Q", f"tensor dimension {name}"))
                for _ in range(dimensions_count)
            )
            ggml_type = int(reader.scalar("I", f"tensor type {name}"))
            offset = int(reader.scalar("Q", f"tensor offset {name}"))
            if name in tensors:
                raise SupportSchemaError(f"duplicate GGUF tensor {name}")
            tensors[name] = TensorSpec(ggml_type, dimensions)
            offsets[name] = offset

        header_end = stream.tell()

    return SupportHeader(version, metadata, tensors, offsets, header_end)


def validate_support_header(header: SupportHeader) -> DSparkMetadata:
    """Fail closed unless metadata and all 81 descriptors match final 0731."""

    if header.version != 3:
        raise SupportSchemaError(
            f"final 0731 support requires GGUF version 3, got {header.version}"
        )
    semantic = validate_0731_metadata(header.metadata, flavor="support")
    expected = expected_tensor_schema()
    missing = sorted(set(expected) - set(header.tensors))
    unexpected = sorted(set(header.tensors) - set(expected))
    mismatched = sorted(
        name
        for name in set(expected) & set(header.tensors)
        if header.tensors[name] != expected[name]
    )
    errors: list[str] = []
    if missing:
        errors.append("missing tensors: " + ", ".join(missing))
    if unexpected:
        errors.append("unexpected tensors: " + ", ".join(unexpected))
    if mismatched:
        errors.append("descriptor mismatches: " + ", ".join(mismatched))
    if errors:
        raise SupportSchemaError("; ".join(errors))
    return semantic


def _tensor_nbytes(name: str, spec: TensorSpec) -> int:
    try:
        block_elements, block_bytes = _GGML_STORAGE[spec.ggml_type]
    except KeyError as exc:
        raise SupportSchemaError(
            f"cannot size unsupported GGML type {spec.ggml_type} for {name}"
        ) from exc
    elements = 1
    for dimension in spec.dimensions:
        if dimension <= 0:
            raise SupportSchemaError(f"non-positive tensor dimension for {name}")
        elements *= dimension
    if elements % block_elements:
        raise SupportSchemaError(
            f"tensor {name} element count is not divisible by its quant block"
        )
    return elements // block_elements * block_bytes


def validate_support_layout(header: SupportHeader, *, file_size: int) -> None:
    """Validate offsets and structural size without reading tensor payloads."""

    if isinstance(file_size, bool) or not isinstance(file_size, int) or file_size <= 0:
        raise SupportSchemaError("support file size must be a positive integer")
    if set(header.offsets) != set(header.tensors):
        raise SupportSchemaError("tensor offset directory does not match tensors")
    alignment = header.metadata.get("general.alignment")
    if isinstance(alignment, bool) or not isinstance(alignment, int) or alignment <= 0:
        raise SupportSchemaError("general.alignment must be a positive integer")
    data_start = (
        (int(header.header_end) + alignment - 1) // alignment * alignment
    )
    if data_start > file_size:
        raise SupportSchemaError("GGUF tensor data starts beyond end of file")

    extents: list[tuple[int, int, str]] = []
    for name, spec in header.tensors.items():
        offset = header.offsets[name]
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise SupportSchemaError(f"invalid tensor offset for {name}")
        if offset % alignment:
            raise SupportSchemaError(f"unaligned tensor offset for {name}")
        start = data_start + offset
        end = start + _tensor_nbytes(name, spec)
        if end > file_size:
            raise SupportSchemaError(f"tensor {name} extends beyond end of file")
        extents.append((start, end, name))

    extents.sort()
    previous_end = data_start
    previous_name = "GGUF header"
    for start, end, name in extents:
        if start < previous_end:
            raise SupportSchemaError(
                f"tensor payload overlap between {previous_name} and {name}"
            )
        previous_end = end
        previous_name = name
    structural_end = (
        (extents[-1][1] + alignment - 1) // alignment * alignment
        if extents else data_start
    )
    if structural_end != file_size:
        raise SupportSchemaError(
            "GGUF structural size does not equal the aligned final tensor end"
        )


def validate_support_file(path: Path) -> DSparkMetadata:
    header = read_support_header(path)
    semantic = validate_support_header(header)
    validate_support_layout(header, file_size=path.stat().st_size)
    return semantic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("support", type=Path)
    args = parser.parse_args()
    semantic = validate_support_file(args.support)
    print(
        f"DSpark support schema: OK: stages={semantic.stage_count} "
        f"tensors={len(expected_tensor_schema())} block={semantic.block_size}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
