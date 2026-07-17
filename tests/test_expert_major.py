#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "gguf-tools" / "ds4-expert-major.py"


def load_tool():
    spec = importlib.util.spec_from_file_location("ds4_expert_major", TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def pack_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def metadata_string(key: str, value: str) -> bytes:
    return pack_string(key) + struct.pack("<I", 8) + pack_string(value)


def metadata_u32(key: str, value: int) -> bytes:
    return pack_string(key) + struct.pack("<II", 4, value)


def fixture_byte(layer: int, role: int, expert: int, index: int) -> int:
    return (layer * 71 + role * 23 + expert * 7 + index) % 251


def write_fixture(path: Path) -> None:
    tool = load_tool()
    metadata = b"".join((
        metadata_string("general.architecture", "deepseek4"),
        metadata_u32("general.alignment", 32),
        metadata_u32("deepseek4.block_count", 2),
        metadata_u32("deepseek4.expert_count", 3),
        metadata_u32("deepseek4.expert_used_count", 2),
    ))
    tensors = [("token_embd.weight", (4,), 0, bytes(range(16)))]
    types = ((16, 16, 10), (12, 12, 12))
    for layer in range(2):
        for role, role_name in enumerate(tool.ROLE_NAME):
            ggml_type = types[layer][role]
            dims = (256, 256, 3)
            size = tool.tensor_nbytes(ggml_type, dims)
            expert_bytes = size // 3
            payload = bytearray(size)
            for expert in range(3):
                for index in range(expert_bytes):
                    payload[expert * expert_bytes + index] = fixture_byte(
                        layer, role, expert, index
                    )
            tensors.append((f"blk.{layer}.ffn_{role_name}_exps.weight",
                            dims, ggml_type, bytes(payload)))

    relative = 0
    directory = bytearray()
    offsets = []
    for name, dims, ggml_type, payload in tensors:
        relative = tool.align_up(relative, 32)
        offsets.append(relative)
        directory += pack_string(name)
        directory += struct.pack("<I", len(dims))
        directory += struct.pack("<" + "Q" * len(dims), *dims)
        directory += struct.pack("<IQ", ggml_type, relative)
        relative += len(payload)
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 5)
    data_offset = tool.align_up(len(header) + len(metadata) + len(directory), 32)
    with path.open("wb") as file:
        file.write(header)
        file.write(metadata)
        file.write(directory)
        file.write(bytes(data_offset - file.tell()))
        cursor = 0
        for (_, _, _, payload), offset in zip(tensors, offsets):
            file.write(bytes(offset - cursor))
            file.write(payload)
            cursor = offset + len(payload)


def run(*args: str, ok: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [sys.executable, str(TOOL), *args], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if ok and result.returncode != 0:
        raise AssertionError(f"command failed: {result.stderr}")
    if not ok and result.returncode == 0:
        raise AssertionError("corrupt fixture was accepted")
    return result


def main() -> int:
    tool = load_tool()
    probe = os.environ.get("DS4_EXPERT_STORE_PROBE")
    with tempfile.TemporaryDirectory(prefix="ds4-expert-major-test-") as tmp:
        tmp_path = Path(tmp)
        source = tmp_path / "source.gguf"
        native = tmp_path / "native.gguf"
        write_fixture(source)

        inspected = run("inspect", str(source))
        assert "layers: 2" in inspected.stdout
        assert "experts: 3" in inspected.stdout
        assert "IQ2_XXS" in inspected.stdout
        assert "Q4_K" in inspected.stdout
        assert "native_bytes:" in inspected.stdout
        assert "size_delta_bytes:" in inspected.stdout

        built = run("build", "--reserve-bytes", "0", str(source), str(native))
        assert "installed atomically" in built.stdout
        run("verify", str(source), str(native))

        native_gguf = tool.load_gguf(native)
        store = next(t for t in native_gguf.tensors if t.name == tool.STORE_TENSOR)
        if probe:
            subprocess.run([probe, str(native), str(store.abs_offset),
                            str(store.size)], check=True)

        bad_manifest = tmp_path / "bad-manifest.gguf"
        shutil.copyfile(native, bad_manifest)
        with bad_manifest.open("r+b") as file:
            file.seek(store.abs_offset + tool.STORE_HEADER_BYTES + 9)
            original = file.read(1)
            file.seek(-1, os.SEEK_CUR)
            file.write(bytes([original[0] ^ 0x80]))
        run("verify", str(source), str(bad_manifest), ok=False)
        if probe:
            rejected = subprocess.run(
                [probe, str(bad_manifest), str(store.abs_offset),
                 str(store.size)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            assert rejected.returncode != 0, "C reader accepted corrupt manifest"

        bad_reserved = tmp_path / "bad-reserved.gguf"
        shutil.copyfile(native, bad_reserved)
        with bad_reserved.open("r+b") as file:
            file.seek(store.abs_offset + 160)
            file.write(b"\x01")
        run("verify", str(source), str(bad_reserved), ok=False)
        if probe:
            rejected = subprocess.run(
                [probe, str(bad_reserved), str(store.abs_offset),
                 str(store.size)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            assert rejected.returncode != 0, "C reader accepted reserved header bits"

        bad_payload = tmp_path / "bad-payload.gguf"
        shutil.copyfile(native, bad_payload)
        with bad_payload.open("r+b") as file:
            file.seek(store.abs_offset + tool.STORE_ALIGNMENT)
            original = file.read(1)
            file.seek(-1, os.SEEK_CUR)
            file.write(bytes([original[0] ^ 0x40]))
        run("verify", str(source), str(bad_payload), ok=False)

    print("expert-major v2 converter and verifier: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
