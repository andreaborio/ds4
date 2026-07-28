#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import struct
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "gguf-tools" / "mixed" / "splice_mixed_expert_layers_gguf.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("ds4_mixed_splice", TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def pack_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def write_fixture(path: Path, tensors: list[tuple[str, tuple[int, ...], int, int]]) -> None:
    tool = load_tool()
    metadata = (
        pack_string("general.architecture") +
        struct.pack("<I", tool.GGUF_VALUE_STRING) +
        pack_string("qwen35moe") +
        pack_string("general.alignment") +
        struct.pack("<II", tool.GGUF_VALUE_UINT32, 32)
    )
    directory = bytearray()
    relative = 0
    payloads: list[tuple[int, bytes]] = []
    for ordinal, (name, dims, ggml_type, fill) in enumerate(tensors):
        relative = tool.pad_to(relative, 32)
        n_bytes = tool.tensor_nbytes(dims, ggml_type)
        payload = bytes([(fill + ordinal) % 251]) * n_bytes
        payloads.append((relative, payload))
        directory += pack_string(name)
        directory += struct.pack("<I", len(dims))
        directory += struct.pack("<" + "Q" * len(dims), *dims)
        directory += struct.pack("<IQ", ggml_type, relative)
        relative += n_bytes

    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 2)
    data_offset = tool.pad_to(len(header) + len(metadata) + len(directory), 32)
    with path.open("wb") as out:
        out.write(header)
        out.write(metadata)
        out.write(directory)
        out.write(bytes(data_offset - out.tell()))
        cursor = 0
        for offset, payload in payloads:
            out.write(bytes(offset - cursor))
            out.write(payload)
            cursor = offset + len(payload)


def tensor_payload(path: Path, tensor) -> bytes:
    with path.open("rb") as source:
        source.seek(tensor.data_offset)
        return source.read(tensor.n_bytes)


def main() -> int:
    tool = load_tool()
    routed = [
        ("blk.0.ffn_gate_exps.weight", (256, 1, 2)),
        ("blk.0.ffn_up_exps.weight", (256, 1, 2)),
        ("blk.0.ffn_down_exps.weight", (256, 1, 2)),
    ]
    with tempfile.TemporaryDirectory(prefix="ds4-mixed-splice-test-") as tmp:
        tmp_path = Path(tmp)
        base_path = tmp_path / "base.gguf"
        donor_path = tmp_path / "donor.gguf"
        output_path = tmp_path / "mixed.gguf"
        write_fixture(
            base_path,
            [("dense.weight", (256,), 13, 3)] +
            [(name, dims, 22, 11) for name, dims in routed],
        )
        write_fixture(
            donor_path,
            [("dense.weight", (256,), 8, 19)] +
            [(name, dims, ggml_type, 29)
             for (name, dims), ggml_type in zip(routed, (17, 18, 23))] +
            [("blk.1.extra.weight", (1,), 0, 41)],
        )

        base = tool.parse_gguf(base_path)
        donor = tool.parse_gguf(donor_path)
        plan = tool.build_plan(base, donor, {0})
        assert len(plan) == len(base.tensors)
        assert sum(item.source == "donor" for item in plan) == 3
        assert next(item for item in plan if item.name == "dense.weight").source == "base"

        tool.write_mixed(base, donor, plan, output_path, False)
        mixed = tool.parse_gguf(output_path)
        assert mixed.kv_blob == base.kv_blob
        assert [tensor.name for tensor in mixed.tensors] == [
            tensor.name for tensor in base.tensors
        ]
        assert mixed.tensor_by_name["dense.weight"].ggml_type == 13
        assert mixed.tensor_by_name[routed[0][0]].ggml_type == 17
        assert mixed.tensor_by_name[routed[1][0]].ggml_type == 18
        assert mixed.tensor_by_name[routed[2][0]].ggml_type == 23

        for tensor in mixed.tensors:
            expected_source = (
                donor if tensor.name.startswith("blk.0.ffn_") else base
            )
            expected = expected_source.tensor_by_name[tensor.name]
            assert tensor_payload(output_path, tensor) == tensor_payload(
                expected_source.path, expected
            )

        all_donor_path = tmp_path / "all-donor.gguf"
        all_donor_plan = tool.build_plan(base, donor, {0}, True)
        assert all(item.source == "donor" for item in all_donor_plan)
        tool.write_mixed(base, donor, all_donor_plan, all_donor_path, False)
        all_donor = tool.parse_gguf(all_donor_path)
        assert all_donor.tensor_count == base.tensor_count
        assert "blk.1.extra.weight" not in all_donor.tensor_by_name
        assert all_donor.tensor_by_name["dense.weight"].ggml_type == 8
        for tensor in all_donor.tensors:
            expected = donor.tensor_by_name[tensor.name]
            assert tensor_payload(all_donor_path, tensor) == tensor_payload(
                donor.path, expected
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
