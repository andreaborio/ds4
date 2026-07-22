#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import hashlib
import json
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
        types = ((12, 12, 12), (12, 12, 12))
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


def affine2_values(block: bytes, group: int) -> list[float]:
    code_bytes = group // 4
    scale = struct.unpack("<f", b"\0\0" + block[code_bytes:code_bytes + 2])[0]
    bias = struct.unpack("<f", b"\0\0" + block[code_bytes + 2:code_bytes + 4])[0]
    return [scale * ((block[i // 4] >> (2 * (i & 3))) & 3) + bias
            for i in range(group)]


def test_affine2_contract(tool) -> None:
    weights = bytes((i * 29 + 7) & 0xff for i in range(16))
    scale = struct.pack("<H", 0x3f40)  # BF16 0.75
    bias = struct.pack("<H", 0xbf00)   # BF16 -0.5
    g64 = tool.interleave_mlx_affine2(
        weights, scale, bias, 1, 64, 64, 64
    )
    g32 = tool.interleave_mlx_affine2(
        weights, scale, bias, 1, 64, 64, 32
    )
    assert len(g64) == 20 and len(g32) == 24
    assert g32[:8] == weights[:8] and g32[12:20] == weights[8:]
    assert g32[8:12] == scale + bias == g32[20:24]
    assert affine2_values(g64, 64) == (
        affine2_values(g32[:12], 32) + affine2_values(g32[12:], 32)
    )

    quant = {"group_size": 64, "bits": 4, "mode": "affine"}
    for layer in range(43):
        for role in ("gate_proj", "up_proj", "down_proj"):
            quant[f"model.layers.{layer}.ffn.switch_mlp.{role}"] = {
                "group_size": 64 if role != "gate_proj" or layer == 42 else 32,
                "bits": 2,
                "mode": "affine",
            }
    config = {
        "torch_dtype": "bfloat16", "hidden_size": 4096,
        "moe_intermediate_size": 2048, "n_routed_experts": 256,
        "num_experts_per_tok": 6, "num_hidden_layers": 43,
        "quantization": quant, "quantization_config": dict(quant),
    }
    groups = tool.validate_deepseek_affine2_config(config)
    assert groups == (32,) * 42 + (64,)
    bad = dict(config)
    bad_quant = dict(quant)
    bad_quant["model.layers.17.ffn.switch_mlp.gate_proj"] = {
        "group_size": 64, "bits": 2, "mode": "affine"
    }
    bad["quantization"] = bad_quant
    bad["quantization_config"] = dict(bad_quant)
    try:
        tool.validate_deepseek_affine2_config(bad)
    except tool.FormatError:
        pass
    else:
        raise AssertionError("unexpected donor gate/g64 pattern was accepted")


class SyntheticAffineSource:
    def __init__(self, expert_count: int):
        self.expert_count = expert_count
        self.reads: list[tuple[int, str, int]] = []

    def expert_bytes(self, key: str, dtype: str,
                     shape: tuple[int, ...], expert: int) -> bytes:
        assert shape[0] == self.expert_count
        assert 0 <= expert < self.expert_count
        layer = int(key.split(".")[2])
        item_bytes = {"U32": 4, "BF16": 2}[dtype]
        size = item_bytes
        for dim in shape[1:]:
            size *= dim
        self.reads.append((layer, key, expert))
        salt = (sum(key.encode()) + expert * 47 + layer * 89) & 0xff
        return bytes((salt + index * 13) & 0xff for index in range(size))


def write_safetensor_fixture(path: Path, *, truncated: bool = False) -> None:
    entries = {
        "weight": {"dtype": "U32", "shape": [2, 2, 1],
                   "data_offsets": [0, 16]},
        "scales": {"dtype": "BF16", "shape": [2, 2, 1],
                   "data_offsets": [16, 24]},
        "biases": {"dtype": "BF16", "shape": [2, 2, 1],
                   "data_offsets": [24, 32]},
    }
    header = json.dumps(entries, separators=(",", ":")).encode()
    payload = bytes(range(32 - int(truncated)))
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


def test_safetensor_reader(tool, tmp_path: Path) -> None:
    model = tmp_path / "mlx-reader"
    model.mkdir()
    shard = "model-00001-of-00001.safetensors"
    (model / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {key: shard for key in ("weight", "scales", "biases")}
    }))
    write_safetensor_fixture(model / shard)
    source = tool.MLXAffineSource(model)
    try:
        assert source.expert_bytes("weight", "U32", (2, 2, 1), 0) == bytes(range(8))
        assert source.expert_bytes("weight", "U32", (2, 2, 1), 1) == bytes(range(8, 16))
        assert source.expert_bytes("scales", "BF16", (2, 2, 1), 1) == \
            bytes(range(20, 24))
    finally:
        source.close()

    guarded = tmp_path / "mlx-reader-guarded"
    guarded.mkdir()
    shutil.copyfile(model / "model.safetensors.index.json",
                    guarded / "model.safetensors.index.json")
    guarded_shard = guarded / shard
    write_safetensor_fixture(guarded_shard)
    guarded_bytes = guarded_shard.read_bytes()
    expected_shards = ((
        shard, hashlib.sha256(guarded_bytes).hexdigest(), len(guarded_bytes),
    ),)
    source = tool.MLXAffineSource(
        guarded, expected_shards, verify_open_fds=True
    )
    try:
        source.expert_bytes("weight", "U32", (2, 2, 1), 0)
        with guarded_shard.open("r+b") as file:
            file.seek(-1, os.SEEK_END)
            original = file.read(1)
            file.seek(-1, os.SEEK_END)
            file.write(bytes([original[0] ^ 0x01]))
        try:
            source.verify_held_shards()
        except tool.FormatError as exc:
            assert "Git LFS shard changed while writing" in str(exc)
        else:
            raise AssertionError("held donor shard mutation was accepted")
    finally:
        source.close()

    broken = tmp_path / "mlx-reader-truncated"
    broken.mkdir()
    shutil.copyfile(model / "model.safetensors.index.json",
                    broken / "model.safetensors.index.json")
    write_safetensor_fixture(broken / shard, truncated=True)
    source = tool.MLXAffineSource(broken)
    try:
        try:
            source.tensor("biases", "BF16", (2, 2, 1))
        except tool.FormatError as exc:
            assert "extent exceeds shard" in str(exc)
        else:
            raise AssertionError("truncated safetensor extent was accepted")
    finally:
        source.close()


def test_affine2_writer(tool, tmp_path: Path, probe: str | None) -> Path:
    plan = tool.make_deepseek_affine2_store_plan(
        123456, 7, layer_count=2, expert_count=3, expert_used=2,
        hidden_size=256, intermediate_size=256,
    )
    groups = (32, 64)
    source_digest = hashlib.sha256(b"synthetic-affine2-donor").digest()
    assert plan.data_offset == 4096
    assert [component.expert_bytes for component in plan.layers[0].components] == \
        [24576, 20480, 20480]
    assert plan.layers[0].record_bytes == 65536
    assert plan.store_size == 397312

    no_space = tmp_path / "deepseek-affine2-no-space.store"
    no_space_source = SyntheticAffineSource(plan.expert_count)
    try:
        tool.write_deepseek_affine2_store_from_source(
            no_space_source, no_space, plan, groups, source_digest, 1 << 63,
            resume=False, verify_after=False,
            hidden_size=256, intermediate_size=256,
        )
    except tool.FormatError as exc:
        assert "insufficient free space" in str(exc)
    else:
        raise AssertionError("impossible affine2 free-space request was accepted")
    assert not no_space_source.reads and not no_space.exists()

    source = SyntheticAffineSource(plan.expert_count)
    output = tmp_path / "deepseek-affine2.store"
    digest = tool.write_deepseek_affine2_store_from_source(
        source, output, plan, groups, source_digest, 0,
        resume=False, verify_after=True,
        hidden_size=256, intermediate_size=256,
    )
    manifest, layers = tool.raw_store(output)
    assert manifest["payload_sha256"] == digest
    assert [layer.record_bytes for layer in layers] == [65536, 65536]
    assert not tool._checkpoint_paths(output)[0].exists()
    assert not tool._checkpoint_paths(output)[1].exists()
    if probe:
        subprocess.run([
            probe, str(output), "0", str(output.stat().st_size),
            str(tool.STORE_FAMILY_DEEPSEEK4),
            str(tool.STORE_STORAGE_MLX_AFFINE2), "2", "3", "2", "7",
        ], check=True)

    # Layer 1 is the synthetic equivalent of production layer 42: its source
    # gate is g64, while the store is normalized to two exact g32 blocks.
    gate = source.expert_bytes(
        "model.layers.1.ffn.switch_mlp.gate_proj.weight",
        "U32", (3, 256, 16), 0,
    )
    scales = source.expert_bytes(
        "model.layers.1.ffn.switch_mlp.gate_proj.scales",
        "BF16", (3, 256, 4), 0,
    )
    biases = source.expert_bytes(
        "model.layers.1.ffn.switch_mlp.gate_proj.biases",
        "BF16", (3, 256, 4), 0,
    )
    gate_offset = layers[1].data_offset
    with output.open("rb") as file:
        file.seek(gate_offset)
        first_row = file.read(24)
    assert first_row[:8] == gate[:8]
    assert first_row[12:20] == gate[8:16]
    assert first_row[8:12] == scales[:2] + biases[:2]
    assert first_row[20:24] == scales[:2] + biases[:2]

    corrupt = tmp_path / "deepseek-affine2-corrupt.store"
    shutil.copyfile(output, corrupt)
    with corrupt.open("r+b") as file:
        file.seek(layers[0].data_offset + 17)
        original = file.read(1)
        file.seek(-1, os.SEEK_CUR)
        file.write(bytes([original[0] ^ 0x40]))
    try:
        tool.verify_deepseek_affine2_store_from_source(
            source, corrupt, plan, groups, source_digest,
            hidden_size=256, intermediate_size=256,
        )
    except tool.FormatError:
        pass
    else:
        raise AssertionError("affine2 verifier accepted corrupt payload")

    resumable = tmp_path / "deepseek-affine2-resume.store"
    resume_source = SyntheticAffineSource(plan.expert_count)

    def interrupt_after_first_layer(completed: int) -> None:
        if completed == 1:
            raise RuntimeError("synthetic interruption")

    try:
        tool.write_deepseek_affine2_store_from_source(
            resume_source, resumable, plan, groups, source_digest, 0,
            resume=True, verify_after=False,
            hidden_size=256, intermediate_size=256,
            progress_hook=interrupt_after_first_layer,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("synthetic interruption was not raised")
    partial, state = tool._checkpoint_paths(resumable)
    assert partial.exists() and state.exists() and not resumable.exists()
    assert json.loads(state.read_text())["completed_layers"] == 1
    with partial.open("r+b") as file:
        corrupt_offset = plan.layers[0].data_offset + 17
        file.seek(corrupt_offset)
        original = file.read(1)
        file.seek(corrupt_offset)
        file.write(bytes([original[0] ^ 0x01]))
    try:
        tool.write_deepseek_affine2_store_from_source(
            resume_source, resumable, plan, groups, source_digest, 0,
            resume=True, verify_after=False,
            hidden_size=256, intermediate_size=256,
        )
    except tool.FormatError as exc:
        assert "resume partial layer 0 digest differs" in str(exc)
    else:
        raise AssertionError("resume accepted a corrupted completed prefix")
    with partial.open("r+b") as file:
        file.seek(corrupt_offset)
        file.write(original)
    resume_source.reads.clear()
    tool.write_deepseek_affine2_store_from_source(
        resume_source, resumable, plan, groups, source_digest, 0,
        resume=True, verify_after=False,
        hidden_size=256, intermediate_size=256,
    )
    assert resume_source.reads
    assert {layer for layer, _, _ in resume_source.reads} == {1}
    assert resumable.exists() and not partial.exists() and not state.exists()
    tool.verify_deepseek_affine2_store_from_source(
        resume_source, resumable, plan, groups, source_digest,
        hidden_size=256, intermediate_size=256,
    )

    cleanup = tmp_path / "deepseek-affine2-cleanup.store"
    cleanup_source = SyntheticAffineSource(plan.expert_count)
    try:
        tool.write_deepseek_affine2_store_from_source(
            cleanup_source, cleanup, plan, groups, source_digest, 0,
            resume=False, verify_after=False,
            hidden_size=256, intermediate_size=256,
            progress_hook=interrupt_after_first_layer,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("cleanup interruption was not raised")
    partial, state = tool._checkpoint_paths(cleanup)
    assert not cleanup.exists() and not partial.exists() and not state.exists()

    race = tmp_path / "deepseek-affine2-race.store"
    race_winner = b"race-winner-must-survive"

    def create_race_winner() -> None:
        race.write_bytes(race_winner)

    try:
        tool.write_deepseek_affine2_store_from_source(
            SyntheticAffineSource(plan.expert_count), race, plan, groups,
            source_digest, 0, resume=False, verify_after=False,
            hidden_size=256, intermediate_size=256,
            pre_install_hook=create_race_winner,
        )
    except OSError:
        pass
    else:
        raise AssertionError("standalone writer replaced a race winner")
    assert race.read_bytes() == race_winner
    race_partial, race_state = tool._checkpoint_paths(race)
    assert not race_partial.exists() and not race_state.exists()
    return output


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
    test_affine2_contract(tool)
    probe = os.environ.get("DS4_EXPERT_STORE_PROBE")
    with tempfile.TemporaryDirectory(prefix="ds4-expert-major-test-") as tmp:
        tmp_path = Path(tmp)
        test_safetensor_reader(tool, tmp_path)
        affine_store = test_affine2_writer(tool, tmp_path, probe)
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
        native_sha256 = hashlib.sha256(native.read_bytes()).hexdigest()

        native_gguf = tool.load_gguf(native)
        store = next(t for t in native_gguf.tensors if t.name == tool.STORE_TENSOR)
        if probe:
            subprocess.run([probe, str(native), str(store.abs_offset),
                            str(store.size),
                            str(tool.STORE_FAMILY_DEEPSEEK4)], check=True)

        hybrid = tmp_path / "deepseek-affine2-hybrid.gguf"
        tool.plan_deepseek_affine2_hybrid_gguf(
            native, affine_store, native_sha256, allow_test_geometry=True,
        )
        try:
            tool.plan_deepseek_affine2_hybrid_gguf(
                native, affine_store, "0" * 64, allow_test_geometry=True,
            )
        except tool.FormatError as exc:
            assert "source GGUF SHA-256 differs" in str(exc)
        else:
            raise AssertionError("hybrid planner accepted the wrong source SHA-256")
        production_gate = run(
            "embed-deepseek-mlx-affine2", "--dry-run",
            "--expected-source-sha256", native_sha256,
            str(native), str(affine_store), ok=False,
        )
        assert "not the pinned DeepSeek affine2 donor" in \
            production_gate.stderr
        tool.embed_deepseek_affine2_hybrid_gguf(
            native, affine_store, hybrid, 0, native_sha256,
            verify_after=True,
            allow_test_geometry=True,
        )
        assert tool.verify_deepseek_affine2_hybrid_gguf(
            native, affine_store, hybrid, native_sha256,
            allow_test_geometry=True,
        )
        hybrid_gguf = tool.load_gguf(hybrid)
        hybrid_stores = [
            tensor for tensor in hybrid_gguf.tensors
            if tensor.name == tool.STORE_TENSOR
        ]
        assert len(hybrid_stores) == 1
        assert hybrid_stores[0].size == affine_store.stat().st_size
        hybrid_manifest, hybrid_layers = tool.parse_store(
            hybrid_gguf, hybrid_stores[0]
        )
        assert hybrid_manifest["storage_format"] == \
            tool.STORE_STORAGE_MLX_AFFINE2
        assert [layer.record_bytes for layer in hybrid_layers] == \
            [65536, 65536]
        if probe:
            subprocess.run([
                probe, str(hybrid), str(hybrid_stores[0].abs_offset),
                str(hybrid_stores[0].size),
                str(tool.STORE_FAMILY_DEEPSEEK4),
                str(tool.STORE_STORAGE_MLX_AFFINE2),
                "2", "3", "2", "7",
            ], check=True)

        mismatched_plan = tool.make_deepseek_affine2_store_plan(
            123456, 8, layer_count=2, expert_count=3, expert_used=2,
            hidden_size=256, intermediate_size=256,
        )
        mismatched_store = tmp_path / "deepseek-affine2-count8.store"
        tool.write_deepseek_affine2_store_from_source(
            SyntheticAffineSource(mismatched_plan.expert_count),
            mismatched_store, mismatched_plan, (32, 64),
            hashlib.sha256(b"synthetic-count8-donor").digest(), 0,
            resume=False, verify_after=False,
            hidden_size=256, intermediate_size=256,
        )
        # The base GGUF and MLX donor are distinct source containers. Their
        # tensor inventory counts need not match; the embedded store preserves
        # the donor provenance while layer/expert geometry is checked against
        # the base model.
        tool.plan_deepseek_affine2_hybrid_gguf(
            native, mismatched_store, native_sha256,
            allow_test_geometry=True,
        )
        foreign_inventory_hybrid = \
            tmp_path / "deepseek-affine2-foreign-inventory.gguf"
        tool.embed_deepseek_affine2_hybrid_gguf(
            native, mismatched_store, foreign_inventory_hybrid, 0,
            native_sha256, verify_after=True, allow_test_geometry=True,
        )
        foreign_gguf = tool.load_gguf(foreign_inventory_hybrid)
        foreign_store = next(
            tensor for tensor in foreign_gguf.tensors
            if tensor.name == tool.STORE_TENSOR
        )
        foreign_manifest, _ = tool.parse_store(foreign_gguf, foreign_store)
        assert foreign_manifest["source_tensors"] == 8

        hybrid_race = tmp_path / "deepseek-affine2-hybrid-race.gguf"
        race_winner = b"hybrid-race-winner-must-survive"
        original_install = tool.install_no_replace

        def install_hybrid_after_race(temp: Path, destination: Path) -> None:
            destination.write_bytes(race_winner)
            original_install(temp, destination)

        tool.install_no_replace = install_hybrid_after_race
        try:
            try:
                tool.embed_deepseek_affine2_hybrid_gguf(
                    native, affine_store, hybrid_race, 0, native_sha256,
                    verify_after=False, allow_test_geometry=True,
                )
            except OSError:
                pass
            else:
                raise AssertionError("hybrid writer replaced a race winner")
        finally:
            tool.install_no_replace = original_install
        assert hybrid_race.read_bytes() == race_winner
        assert not list(tmp_path.glob(f".{hybrid_race.name}.tmp.*"))

        corrupt_store = tmp_path / "deepseek-affine2-embed-corrupt.store"
        shutil.copyfile(affine_store, corrupt_store)
        corrupt_manifest, _ = tool.raw_store(corrupt_store)
        with corrupt_store.open("r+b") as file:
            file.seek(int(corrupt_manifest["data_offset"]) + 31)
            original = file.read(1)
            file.seek(-1, os.SEEK_CUR)
            file.write(bytes([original[0] ^ 0x01]))
        rejected_hybrid = tmp_path / "rejected-affine2-hybrid.gguf"
        try:
            tool.embed_deepseek_affine2_hybrid_gguf(
                native, corrupt_store, rejected_hybrid, 0, native_sha256,
                verify_after=True, allow_test_geometry=True,
            )
        except tool.FormatError as exc:
            assert "payload SHA-256 mismatch" in str(exc)
        else:
            raise AssertionError("hybrid builder accepted corrupt affine2 store")
        assert not rejected_hybrid.exists()
        assert not list(tmp_path.glob(f".{rejected_hybrid.name}.tmp.*"))

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

        bad_reserved = tmp_path / "bad-reserved.gguf"
        shutil.copyfile(native, bad_reserved)
        with bad_reserved.open("r+b") as file:
            file.seek(store.abs_offset + 160)
            file.write(b"\x01")
        run("verify", str(source), str(bad_reserved), ok=False)
        if probe:
            rejected = subprocess.run(
                [probe, str(bad_reserved), str(store.abs_offset),
                 str(store.size), str(tool.STORE_FAMILY_DEEPSEEK4)],
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
        glm_native_sha256 = hashlib.sha256(glm_native.read_bytes()).hexdigest()
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
        assert "Q4_K" in qwen_inspected.stdout
        run("build", "--reserve-bytes", "0", str(qwen_source),
            str(qwen_native))
        run("verify", str(qwen_source), str(qwen_native))
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
        assert "accepts deepseek4, glm-dsa, and qwen35moe" in \
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

    print("expert-major v2 converter and verifier: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
