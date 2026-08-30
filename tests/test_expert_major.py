#!/usr/bin/env python3

from __future__ import annotations

import dataclasses
import hashlib
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


def write_fixture(path: Path, architecture: str, *,
                  omit_metadata: frozenset[str] = frozenset(),
                  bad_geometry: bool = False,
                  reported_architecture: str | None = None) -> None:
    tool = load_tool()
    reported_architecture = reported_architecture or architecture
    if architecture == "deepseek4":
        metadata_items = (
            metadata_string("general.architecture", reported_architecture),
            metadata_u32("general.alignment", 32),
            metadata_u32("deepseek4.block_count", 2),
            metadata_u32("deepseek4.expert_count", 3),
            metadata_u32("deepseek4.expert_used_count", 2),
        )
        routed_layers = (0, 1)
        types = ((16, 16, 10), (12, 12, 12))
    elif architecture == "glm-dsa":
        metadata_items = (
            metadata_string("general.architecture", reported_architecture),
            metadata_u32("general.alignment", 32),
            metadata_u32("glm-dsa.block_count", 5),
            metadata_u32("glm-dsa.expert_count", 3),
            metadata_u32("glm-dsa.expert_used_count", 2),
            metadata_u32("glm-dsa.leading_dense_block_count", 3),
            metadata_u32("glm-dsa.nextn_predict_layers", 1),
        )
        routed_layers = (3, 4)
        types = ((10, 10, 10), (13, 13, 14))
    elif architecture == "qwen35moe":
        metadata_items = (
            metadata_string("general.architecture", reported_architecture),
            metadata_u32("general.alignment", 32),
            metadata_u32("qwen35moe.block_count", 2),
            metadata_u32("qwen35moe.expert_count", 3),
            metadata_u32("qwen35moe.expert_used_count", 2),
        )
        routed_layers = (0, 1)
        types = ((17, 17, 18), (18, 18, 23))
    elif architecture == "qwen4exp":
        metadata_items = (
            metadata_string("general.architecture", reported_architecture),
            metadata_u32("general.alignment", 32),
            metadata_u32("qwen4exp.block_count", 48),
            metadata_u32("qwen4exp.expert_count", 512),
            metadata_u32("qwen4exp.expert_used_count", 10),
        )
        # The Phase-2 source stub intentionally has no routed payload. The
        # converter must reject it before inventory traversal because no
        # Qwen4Exp release codec has been selected.
        routed_layers = ()
        types = ()
    else:
        raise AssertionError(f"unsupported fixture architecture: {architecture}")
    metadata_items = tuple(
        item for item in metadata_items
        if not any(pack_string(key) == item[:8 + len(key)]
                   for key in omit_metadata)
    )
    metadata = b"".join(metadata_items)
    tensors = [("token_embd.weight", (4,), 0, bytes(range(16)))]
    for layer_slot, layer in enumerate(routed_layers):
        for role, role_name in enumerate(tool.ROLE_NAME):
            ggml_type = types[layer_slot][role]
            dims = ((512, 128, 3)
                    if bad_geometry and architecture == "qwen35moe" and
                    layer_slot == 1 and role_name == "down"
                    else (256, 256, 3))
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
    header = b"GGUF" + struct.pack(
        "<IQQ", 3, len(tensors), len(metadata_items)
    )
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


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def write_affine_descriptor_fixture(
        path: Path, tool, *, family: int, block_elements: int,
        bad_qwen4_geometry: bool = False) -> tuple[object, object]:
    """Write header+descriptors only; the multi-GiB payload stays virtual."""

    if family == tool.STORE_FAMILY_QWEN4EXP:
        layer_count, expert_count, expert_used = 48, 512, 10
        gate_dims = (2496, 640, 512) if bad_qwen4_geometry else \
            (2560, 640, 512)
        down_dims = (640, gate_dims[0], 512)
    elif family == tool.STORE_FAMILY_QWEN35_MOE:
        layer_count, expert_count, expert_used = 2, 3, 2
        gate_dims = (256, 256, 3)
        down_dims = gate_dims
    else:
        raise AssertionError(family)

    data_offset = tool.align_up(
        tool.STORE_HEADER_BYTES + layer_count * tool.STORE_LAYER_BYTES,
        tool.STORE_ALIGNMENT,
    )
    cursor = data_offset
    layers = []
    for layer_index in range(layer_count):
        components = []
        record_offset = 0
        for role, dims in enumerate((gate_dims, gate_dims, down_dims)):
            elements = dims[0] * dims[1]
            if family == tool.STORE_FAMILY_QWEN4EXP:
                expert_bytes = elements // 64 * 36
            else:
                expert_bytes = tool.tensor_nbytes(12, (dims[0], dims[1], 1))
            tensor = tool.Tensor(
                f"blk.{layer_index}.ffn_{tool.ROLE_NAME[role]}_exps.weight",
                dims, 12, 0, expert_bytes * expert_count,
            )
            components.append(tool.Component(
                role, tensor, expert_bytes, record_offset, block_elements,
            ))
            record_offset += expert_bytes
        cursor = tool.align_up(cursor, tool.STORE_ALIGNMENT)
        layer_size = record_offset * expert_count
        layers.append(tool.Layer(
            layer_index, expert_count, record_offset, cursor, layer_size,
            tuple(components),
        ))
        cursor += layer_size

    source = tool.GGUF(
        path, 4096, 3, 0, 32, b"", {}, [], 0,
    )
    descriptors = b"".join(tool.pack_layer(layer) for layer in layers)
    plan = tool.StorePlan(
        source, family, tool.STORE_STORAGE_MLX_AFFINE4, 64,
        layer_count, expert_count, expert_used, 1658,
        descriptors, data_offset, cursor - data_offset, cursor, layers,
    )
    provisional = tool.make_header(plan, bytes(32), bytes(32))
    header = tool.make_header(
        plan, bytes(32), bytes(32),
        tool.manifest_digest(provisional, descriptors),
    )
    path.write_bytes(header + descriptors)
    # The descriptor declares the exact production geometry. Extending the
    # file sparsely lets the same bounded fixture pass the C extent checks
    # without allocating or writing the roughly 67 GiB routed payload.
    with path.open("r+b") as file:
        file.truncate(plan.store_size)
    tensor = tool.Tensor(
        tool.STORE_TENSOR, (plan.store_size,), 24, 0, plan.store_size,
        abs_offset=0,
    )
    native = tool.GGUF(
        path, path.stat().st_size, 3, 0, 32, b"", {}, [tensor], 0,
    )
    return native, tensor


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
    descriptors = tool.EXPERT_FAMILY_DESCRIPTORS
    assert isinstance(descriptors, tuple)
    assert [descriptor.family for descriptor in descriptors] == [1, 2, 3, 4]
    qwen4_descriptor = tool.family_descriptor(architecture="qwen4exp")
    assert (qwen4_descriptor.exact_layers,
            qwen4_descriptor.exact_experts,
            qwen4_descriptor.exact_experts_used) == (48, 512, 10)
    assert qwen4_descriptor.admitted_storage_formats == (
        tool.STORE_STORAGE_MLX_AFFINE4,
    )
    try:
        qwen4_descriptor.exact_experts = 384
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("ExpertMajor family descriptor is mutable")
    assert tool.STORE_MAX_EXPERTS == 512

    probe = os.environ.get("DS4_EXPERT_STORE_PROBE")
    with tempfile.TemporaryDirectory(prefix="ds4-expert-major-test-") as tmp:
        tmp_path = Path(tmp)
        source = tmp_path / "deepseek-source.gguf"
        native = tmp_path / "deepseek-native.gguf"
        write_fixture(source, "deepseek4")

        inspected = run("inspect", str(source))
        assert "layers: 2" in inspected.stdout
        assert "family: 1" in inspected.stdout
        assert "layer_ids: 0..1" in inspected.stdout
        assert "experts: 3" in inspected.stdout
        assert "IQ2_XXS" in inspected.stdout
        assert "Q4_K" in inspected.stdout
        assert "native_bytes:" in inspected.stdout
        assert "size_delta_bytes:" in inspected.stdout

        built = run("build", "--reserve-bytes", "0", str(source), str(native))
        assert "installed atomically" in built.stdout
        run("verify", str(source), str(native))
        assert sha256(source) == \
            "d059e0a1572cc1ebdab0c01e7efc720d8660d294d9836180c5393c5880b1ccc3"
        assert sha256(native) == \
            "89f8818ca36e6cf8db88eb1ba54cfd310e60acec58e67aba23468fe0d056f33e"

        atomic_target = tmp_path / "atomic-native.gguf"
        atomic_target.write_bytes(b"existing destination")
        original_verify = tool.verify
        try:
            def fail_verify(_source, _native):
                raise tool.FormatError("injected post-write verify failure")

            tool.verify = fail_verify
            try:
                tool.build(source, atomic_target, 0, True)
            except tool.FormatError as exc:
                assert "injected post-write verify failure" in str(exc)
            else:
                raise AssertionError("injected converter failure vanished")
        finally:
            tool.verify = original_verify
        assert atomic_target.read_bytes() == b"existing destination"
        assert not list(tmp_path.glob(".atomic-native.gguf.tmp.*"))
        tool.build(source, atomic_target, 0, True)
        assert sha256(atomic_target) == sha256(native)
        assert not list(tmp_path.glob(".atomic-native.gguf.tmp.*"))

        native_gguf = tool.load_gguf(native)
        store = next(t for t in native_gguf.tensors if t.name == tool.STORE_TENSOR)
        if probe:
            subprocess.run([probe, str(native), str(store.abs_offset),
                            str(store.size),
                            str(tool.STORE_FAMILY_DEEPSEEK4)], check=True)

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
                 str(store.size), str(tool.STORE_FAMILY_DEEPSEEK4)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            assert rejected.returncode != 0, "C reader accepted corrupt manifest"

        bad_storage = tmp_path / "bad-storage.gguf"
        shutil.copyfile(native, bad_storage)
        with bad_storage.open("r+b") as file:
            file.seek(store.abs_offset + 160)
            file.write(b"\x01")
        run("verify", str(source), str(bad_storage), ok=False)
        if probe:
            rejected = subprocess.run(
                [probe, str(bad_storage), str(store.abs_offset),
                 str(store.size), str(tool.STORE_FAMILY_DEEPSEEK4)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            assert rejected.returncode != 0, \
                "C reader accepted an unsupported storage format"

        bad_payload = tmp_path / "bad-payload.gguf"
        shutil.copyfile(native, bad_payload)
        with bad_payload.open("r+b") as file:
            file.seek(store.abs_offset + tool.STORE_ALIGNMENT)
            original = file.read(1)
            file.seek(-1, os.SEEK_CUR)
            file.write(bytes([original[0] ^ 0x40]))
        run("verify", str(source), str(bad_payload), ok=False)

        glm_source = tmp_path / "glm-source.gguf"
        glm_native = tmp_path / "glm-native.gguf"
        write_fixture(glm_source, "glm-dsa")
        glm_inspected = run("inspect", str(glm_source))
        assert "architecture: glm-dsa" in glm_inspected.stdout
        assert "family: 2" in glm_inspected.stdout
        assert "layers: 2" in glm_inspected.stdout
        assert "layer_ids: 3..4" in glm_inspected.stdout
        assert "Q2_K" in glm_inspected.stdout
        assert "Q5_K" in glm_inspected.stdout
        assert "Q6_K" in glm_inspected.stdout
        run("build", "--reserve-bytes", "0", str(glm_source),
            str(glm_native))
        run("verify", str(glm_source), str(glm_native))
        assert sha256(glm_source) == \
            "cff321ec7a3131836896c4a33148ac1255da002b58067e684d41dd940e8d1478"
        assert sha256(glm_native) == \
            "58aabe8367516b78a0b1e92e3a91cde011fc16375c8ae3edcfcca145590bd1d2"
        glm_gguf = tool.load_gguf(glm_native)
        glm_store = next(
            tensor for tensor in glm_gguf.tensors
            if tensor.name == tool.STORE_TENSOR
        )
        glm_manifest, glm_layers = tool.parse_store(glm_gguf, glm_store)
        assert glm_manifest["family"] == tool.STORE_FAMILY_GLM_DSA
        assert [layer.index for layer in glm_layers] == [3, 4]
        if probe:
            subprocess.run(
                [probe, str(glm_native), str(glm_store.abs_offset),
                 str(glm_store.size), str(tool.STORE_FAMILY_GLM_DSA)],
                check=True,
            )

        qwen_source = tmp_path / "qwen-source.gguf"
        qwen_native = tmp_path / "qwen-native.gguf"
        write_fixture(qwen_source, "qwen35moe")
        qwen_inspected = run("inspect", str(qwen_source))
        assert "architecture: qwen35moe" in qwen_inspected.stdout
        assert "family: 3" in qwen_inspected.stdout
        assert "layers: 2" in qwen_inspected.stdout
        assert "layer_ids: 0..1" in qwen_inspected.stdout
        assert "experts: 3" in qwen_inspected.stdout
        assert "IQ2_XS" in qwen_inspected.stdout
        assert "IQ3_XXS" in qwen_inspected.stdout
        assert "IQ4_XS" in qwen_inspected.stdout
        run("build", "--reserve-bytes", "0", str(qwen_source),
            str(qwen_native))
        run("verify", str(qwen_source), str(qwen_native))
        assert sha256(qwen_source) == \
            "645d00e4774324c1ee37129690094c9e7fac1c391c661aadb1d131914eaf9be8"
        assert sha256(qwen_native) == \
            "ba1ebc3860538b0f088848f5e76e680c4d4aa05084399487c1e36691269c0ce8"
        qwen_gguf = tool.load_gguf(qwen_native)
        qwen_store = next(
            tensor for tensor in qwen_gguf.tensors
            if tensor.name == tool.STORE_TENSOR
        )
        qwen_manifest, qwen_layers = tool.parse_store(qwen_gguf, qwen_store)
        assert qwen_manifest["family"] == tool.STORE_FAMILY_QWEN35_MOE
        assert [layer.index for layer in qwen_layers] == [0, 1]
        assert all(
            [component.role for component in layer.components] == [0, 1, 2]
            for layer in qwen_layers
        )

        qwen_bad_family = tmp_path / "qwen-bad-family.gguf"
        shutil.copyfile(qwen_native, qwen_bad_family)
        with qwen_bad_family.open("r+b") as file:
            store_abs = qwen_store.abs_offset
            file.seek(store_abs)
            header = bytearray(file.read(tool.STORE_HEADER_BYTES))
            struct.pack_into("<I", header, 16,
                             tool.STORE_FAMILY_DEEPSEEK4)
            descriptors = os.pread(
                file.fileno(),
                qwen_manifest["layer_count"] * tool.STORE_LAYER_BYTES,
                store_abs + tool.STORE_HEADER_BYTES,
            )
            header[tool.MANIFEST_DIGEST_OFFSET:
                   tool.MANIFEST_DIGEST_OFFSET + 32] = tool.manifest_digest(
                       header, descriptors
                   )
            file.seek(store_abs)
            file.write(header)
        family_result = run("verify", str(qwen_source),
                            str(qwen_bad_family), ok=False)
        assert "identity does not match" in family_result.stderr

        qwen_missing_metadata = tmp_path / "qwen-missing-metadata.gguf"
        write_fixture(
            qwen_missing_metadata, "qwen35moe",
            omit_metadata=frozenset({"qwen35moe.expert_used_count"}),
        )
        metadata_result = run("inspect", str(qwen_missing_metadata), ok=False)
        assert "metadata is incomplete" in metadata_result.stderr

        qwen_bad_geometry = tmp_path / "qwen-bad-geometry.gguf"
        write_fixture(qwen_bad_geometry, "qwen35moe", bad_geometry=True)
        geometry_result = run("inspect", str(qwen_bad_geometry), ok=False)
        assert "gate/down dimensions disagree" in geometry_result.stderr

        qwen_bad_architecture = tmp_path / "qwen-bad-architecture.gguf"
        write_fixture(
            qwen_bad_architecture, "qwen35moe",
            reported_architecture="badfamily",
        )
        architecture_result = run(
            "inspect", str(qwen_bad_architecture), ok=False
        )
        assert "accepts deepseek4, glm-dsa, qwen35moe, qwen4exp" in \
            architecture_result.stderr

        qwen_bad_payload = tmp_path / "qwen-bad-payload.gguf"
        shutil.copyfile(qwen_native, qwen_bad_payload)
        with qwen_bad_payload.open("r+b") as file:
            file.seek(qwen_store.abs_offset + qwen_layers[0].data_offset)
            original = file.read(1)
            file.seek(-1, os.SEEK_CUR)
            file.write(bytes([original[0] ^ 0x20]))
        payload_result = run("verify", str(qwen_source),
                             str(qwen_bad_payload), ok=False)
        assert "payload mismatch" in payload_result.stderr
        if probe:
            subprocess.run(
                [probe, str(qwen_native), str(qwen_store.abs_offset),
                 str(qwen_store.size), str(tool.STORE_FAMILY_QWEN35_MOE)],
                check=True,
            )

        qwen4_source = tmp_path / "qwen4exp-source-stub.gguf"
        write_fixture(qwen4_source, "qwen4exp")
        qwen4_result = run("inspect", str(qwen4_source), ok=False)
        assert "no release-qualified routed codec" in qwen4_result.stderr

        # Family 4 uses exact logical 48x512 geometry and the G64 affine
        # descriptor (64 elements / 36 bytes). The descriptor-only fixture
        # keeps the roughly 67 GiB logical payload sparse and allocates only
        # the manifest filesystem blocks.
        qwen4_manifest_path = tmp_path / "qwen4exp-affine-descriptor.bin"
        qwen4_native, qwen4_tensor = write_affine_descriptor_fixture(
            qwen4_manifest_path, tool,
            family=tool.STORE_FAMILY_QWEN4EXP,
            block_elements=64,
        )
        qwen4_manifest, qwen4_layers = tool.parse_store(
            qwen4_native, qwen4_tensor,
        )
        assert qwen4_manifest["family"] == tool.STORE_FAMILY_QWEN4EXP
        assert qwen4_manifest["storage_format"] == \
            tool.STORE_STORAGE_MLX_AFFINE4
        assert qwen4_manifest["group_size"] == 64
        assert (qwen4_manifest["layer_count"],
                qwen4_manifest["expert_count"],
                qwen4_manifest["expert_used"]) == (48, 512, 10)
        assert len(qwen4_layers) == 48
        assert all(
            component.block_elements == 64
            for layer in qwen4_layers for component in layer.components
        )
        qwen4_stat = qwen4_manifest_path.stat()
        assert qwen4_stat.st_size == qwen4_manifest["store_size"]
        assert qwen4_stat.st_blocks * 512 < 1 << 20
        if probe:
            subprocess.run(
                [probe, str(qwen4_manifest_path), "0",
                 str(qwen4_tensor.size), str(tool.STORE_FAMILY_QWEN4EXP),
                 str(tool.STORE_STORAGE_MLX_AFFINE4), "64"],
                check=True,
            )

        qwen4_bad_block = tmp_path / "qwen4exp-bad-block.bin"
        bad_native, bad_tensor = write_affine_descriptor_fixture(
            qwen4_bad_block, tool,
            family=tool.STORE_FAMILY_QWEN4EXP,
            block_elements=256,
        )
        try:
            tool.parse_store(bad_native, bad_tensor)
        except tool.FormatError as exc:
            assert "physical codec descriptor" in str(exc)
        else:
            raise AssertionError("family4 accepted legacy block_elements=256")

        qwen4_bad_geometry = tmp_path / "qwen4exp-bad-geometry.bin"
        bad_native, bad_tensor = write_affine_descriptor_fixture(
            qwen4_bad_geometry, tool,
            family=tool.STORE_FAMILY_QWEN4EXP,
            block_elements=64,
            bad_qwen4_geometry=True,
        )
        try:
            tool.parse_store(bad_native, bad_tensor)
        except tool.FormatError as exc:
            assert "Qwen4Exp affine descriptor" in str(exc)
        else:
            raise AssertionError("family4 accepted wrong affine geometry")

        # Family 3 remains byte-/descriptor-compatible with its existing
        # same-size Q4_K manifest convention: block_elements stays 256.
        legacy_path = tmp_path / "qwen35-affine-descriptor.bin"
        legacy_native, legacy_tensor = write_affine_descriptor_fixture(
            legacy_path, tool,
            family=tool.STORE_FAMILY_QWEN35_MOE,
            block_elements=256,
        )
        legacy_manifest, legacy_layers = tool.parse_store(
            legacy_native, legacy_tensor,
        )
        assert legacy_manifest["family"] == tool.STORE_FAMILY_QWEN35_MOE
        assert all(
            component.block_elements == 256
            for layer in legacy_layers for component in layer.components
        )
        legacy_bad_path = tmp_path / "qwen35-affine-bad-block.bin"
        bad_native, bad_tensor = write_affine_descriptor_fixture(
            legacy_bad_path, tool,
            family=tool.STORE_FAMILY_QWEN35_MOE,
            block_elements=64,
        )
        try:
            tool.parse_store(bad_native, bad_tensor)
        except tool.FormatError as exc:
            assert "physical codec descriptor" in str(exc)
        else:
            raise AssertionError("family3 accepted family4 block_elements=64")

    print("expert-major v2 converter and verifier: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
