#!/usr/bin/env python3

from __future__ import annotations

import errno
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
FINAL_DSPARK_PROVENANCE = {
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


def metadata_u32_array(key: str, values: tuple[int, ...]) -> bytes:
    return (
        pack_string(key) + struct.pack("<IIQ", 9, 4, len(values)) +
        struct.pack("<" + "I" * len(values), *values)
    )


def fixture_byte(layer: int, role: int, expert: int, index: int) -> int:
    return (layer * 71 + role * 23 + expert * 7 + index) % 251


def write_fixture(path: Path, architecture: str, *,
                  omit_metadata: frozenset[str] = frozenset(),
                  bad_geometry: bool = False,
                  reported_architecture: str | None = None,
                  deepseek_layers: int = 2) -> None:
    tool = load_tool()
    reported_architecture = reported_architecture or architecture
    if architecture == "deepseek4":
        metadata_items = (
            metadata_string("general.architecture", reported_architecture),
            metadata_u32("general.alignment", 32),
            metadata_u32("deepseek4.block_count", deepseek_layers),
            metadata_u32("deepseek4.expert_count", 3),
            metadata_u32("deepseek4.expert_used_count", 2),
        )
        routed_layers = tuple(range(deepseek_layers))
        types = tuple(
            (16, 16, 10) if layer % 2 == 0 else (12, 12, 12)
            for layer in routed_layers
        )
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
        file.write(bytes(tool.align_up(cursor, 32) - cursor))


def write_sparse_gguf(
        path: Path, metadata_items: list[bytes],
        tensors: list[tuple[str, tuple[int, ...], int]]) -> None:
    tool = load_tool()
    alignment = 32
    metadata = b"".join(metadata_items)
    relative = 0
    directory = bytearray()
    for name, dims, ggml_type in tensors:
        relative = tool.align_up(relative, alignment)
        directory += pack_string(name)
        directory += struct.pack("<I", len(dims))
        directory += struct.pack("<" + "Q" * len(dims), *dims)
        directory += struct.pack("<IQ", ggml_type, relative)
        relative += tool.tensor_nbytes(ggml_type, dims)
    header = b"GGUF" + struct.pack(
        "<IQQ", 3, len(tensors), len(metadata_items)
    )
    data_offset = tool.align_up(
        len(header) + len(metadata) + len(directory), alignment
    )
    with path.open("wb") as file:
        file.write(header)
        file.write(metadata)
        file.write(directory)
        file.write(bytes(data_offset - file.tell()))
        file.truncate(data_offset + tool.align_up(relative, alignment))


def write_sparse_target_0731(path: Path, *,
                             metadata_override: tuple[str, int] | None = None,
                             dspark_alias: bool = False,
                             dspark_tensor_alias: bool = False,
                             bad_routed_shape: bool = False) -> None:
    tool = load_tool()
    values = {
        "deepseek4.block_count": 43,
        "deepseek4.embedding_length": 4096,
        "deepseek4.vocab_size": 129280,
        "deepseek4.expert_count": 256,
        "deepseek4.expert_used_count": 6,
        "deepseek4.expert_feed_forward_length": 2048,
    }
    if metadata_override:
        values[metadata_override[0]] = metadata_override[1]
    metadata_items = [
        metadata_string("general.architecture", "deepseek4"),
        metadata_u32("general.alignment", 32),
        *(metadata_u32(key, value) for key, value in values.items()),
    ]
    if dspark_alias:
        metadata_items.append(metadata_u32("deepseek4.dspark.stage_count", 3))
    tensors = [("token_embd.weight", (4096, 129280), 1)]
    for layer in range(43):
        gate_dims = ((4096, 1024, 256)
                     if bad_routed_shape and layer == 42
                     else (4096, 2048, 256))
        tensors.extend((
            (f"blk.{layer}.ffn_gate_exps.weight", gate_dims, 16),
            (f"blk.{layer}.ffn_up_exps.weight", (4096, 2048, 256), 16),
            (f"blk.{layer}.ffn_down_exps.weight", (2048, 4096, 256), 10),
        ))
    if dspark_tensor_alias:
        tensors.append(("mtp.legacy.weight", (4,), 24))
    write_sparse_gguf(path, metadata_items, tensors)


def write_dspark_fixture(
        path: Path, *,
        metadata_overrides: dict[
            str, int | str | tuple[int, ...]
        ] | None = None,
        omit_metadata: str | None = None,
        extra_metadata: bool = False,
        metadata_type_drift: str | None = None,
        omit_tensor: str | None = None,
        extra_tensor: bool = False,
        bad_static_shape: bool = False,
        alignment: int = 32) -> None:
    tool = load_tool()
    values: dict[str, int | str | tuple[int, ...]] = {
        "dspark.block_size": 5,
        "dspark.markov_rank": 256,
        "dspark.noise_token_id": 128799,
        "dspark.target_layer_ids": (40, 41, 42),
        "dspark.stage_count": 3,
        "dspark.n_layers": 3,
        **FINAL_DSPARK_PROVENANCE,
    }
    if metadata_overrides:
        values.update(metadata_overrides)
    def support_u32(key: str) -> bytes:
        value = int(values[key])
        if metadata_type_drift == key:
            return pack_string(key) + struct.pack("<Ii", 5, value)
        return metadata_u32(key, value)

    metadata_items = [
        metadata_string("general.architecture", "deepseek4-dspark"),
        metadata_string("general.name", "DeepSeek V4 Flash DSpark support"),
        metadata_u32("general.alignment", alignment),
        support_u32("dspark.block_size"),
        support_u32("dspark.markov_rank"),
        support_u32("dspark.noise_token_id"),
        metadata_u32_array(
            "dspark.target_layer_ids",
            tuple(values["dspark.target_layer_ids"]),
        ),
        support_u32("dspark.stage_count"),
        support_u32("dspark.n_layers"),
        *(metadata_string(key, str(values[key]))
          for key in tool.DSPARK_PROVENANCE_KEYS),
    ]
    if omit_metadata:
        encoded_key = pack_string(omit_metadata)
        metadata_items = [
            item for item in metadata_items if not item.startswith(encoded_key)
        ]
    if extra_metadata:
        metadata_items.append(metadata_u32("dspark.unexpected", 1))
    expected_names: set[str] = set()
    for stage in range(tool.DSPARK_STAGE_COUNT):
        suffixes = set(tool.DSPARK_BLOCK_SUFFIXES)
        if stage == 0:
            suffixes.update(tool.DSPARK_STAGE0_SUFFIXES)
        if stage == tool.DSPARK_STAGE_COUNT - 1:
            suffixes.update(tool.DSPARK_FINAL_SUFFIXES)
        expected_names.update(f"mtp.{stage}.{suffix}" for suffix in suffixes)
    if omit_tensor:
        expected_names.remove(omit_tensor)
    n_embd = 4096
    n_ff = 2048
    n_hc = 4
    hc_dim = n_embd * n_hc
    hc_mix_dim = 2 * n_hc + n_hc * n_hc
    n_head = 64
    n_head_dim = 512
    n_lora_q = 1024
    output_a_rows = 4096
    output_low_dim = 8192
    n_vocab = 129280
    block_layout = {
        "hc_attn_base.weight": ((hc_mix_dim,), 0),
        "hc_attn_fn.weight": ((hc_dim, hc_mix_dim), 1),
        "hc_attn_scale.weight": ((3,), 0),
        "attn_sinks.weight": ((n_head,), 0),
        "attn_q_a.weight": ((n_embd, n_lora_q), 8),
        "attn_q_a_norm.weight": ((n_lora_q,), 0),
        "attn_q_b.weight": ((n_lora_q, n_head * n_head_dim), 8),
        "attn_kv.weight": ((n_embd, n_head_dim), 8),
        "attn_kv_a_norm.weight": ((n_head_dim,), 0),
        "attn_output_a.weight": ((output_a_rows, output_low_dim), 8),
        "attn_output_b.weight": ((output_low_dim, n_embd), 8),
        "attn_norm.weight": ((n_embd,), 0),
        "hc_ffn_base.weight": ((hc_mix_dim,), 0),
        "hc_ffn_fn.weight": ((hc_dim, hc_mix_dim), 1),
        "hc_ffn_scale.weight": ((3,), 0),
        "ffn_gate_inp.weight": ((n_embd, 256), 8),
        "exp_probs_b.bias": ((256,), 0),
        "ffn_norm.weight": ((n_embd,), 0),
        "ffn_gate_shexp.weight": ((n_embd, n_ff), 8),
        "ffn_up_shexp.weight": ((n_embd, n_ff), 8),
        "ffn_down_shexp.weight": ((n_ff, n_embd), 8),
    }
    special_layout = {
        (0, "main_proj.weight"): ((3 * n_embd, n_embd), 8),
        (0, "main_norm.weight"): ((n_embd,), 0),
        (2, "norm.weight"): ((n_embd,), 0),
        (2, "hc_head_base.weight"): ((n_hc,), 0),
        (2, "hc_head_fn.weight"): ((hc_dim, n_hc), 1),
        (2, "hc_head_scale.weight"): ((1,), 0),
        (2, "markov_head.markov_w1.weight"): ((256, n_vocab), 8),
        (2, "markov_head.markov_w2.weight"): ((256, n_vocab), 8),
        (2, "confidence_head.proj.weight"): ((n_embd + 256, 1), 8),
    }
    tensors = []
    for name in sorted(expected_names):
        match = tool.DSPARK_ROUTED_RE.fullmatch(name)
        if match:
            role = match.group(2)
            dims = ((n_embd, n_ff, 256) if role != "down"
                    else (n_ff, n_embd, 256))
            ggml_type = 16 if role != "down" else 10
        else:
            _, stage_text, suffix = name.split(".", 2)
            stage = int(stage_text)
            layout = special_layout.get((stage, suffix))
            if layout is None:
                layout = block_layout[suffix]
            dims, ggml_type = layout
            if bad_static_shape and name == "mtp.2.confidence_head.proj.weight":
                dims = (n_embd, 1)
        tensors.append((name, dims, ggml_type))
    if extra_tensor:
        tensors.append(("mtp.1.unknown.weight", (4,), 24))
    if alignment != 32:
        # The generic sparse writer is canonical at 32 bytes; write a valid
        # 64-byte variant so the converter reaches the cross-file check.
        metadata = b"".join(metadata_items)
        relative = 0
        directory = bytearray()
        for name, dims, ggml_type in tensors:
            relative = tool.align_up(relative, alignment)
            directory += pack_string(name)
            directory += struct.pack("<I", len(dims))
            directory += struct.pack("<" + "Q" * len(dims), *dims)
            directory += struct.pack("<IQ", ggml_type, relative)
            relative += tool.tensor_nbytes(ggml_type, dims)
        header = b"GGUF" + struct.pack(
            "<IQQ", 3, len(tensors), len(metadata_items)
        )
        data_offset = tool.align_up(
            len(header) + len(metadata) + len(directory), alignment
        )
        with path.open("wb") as file:
            file.write(header)
            file.write(metadata)
            file.write(directory)
            file.write(bytes(data_offset - file.tell()))
            file.truncate(data_offset + tool.align_up(relative, alignment))
    else:
        write_sparse_gguf(path, metadata_items, tensors)


def write_sparse_dspark_store_container(path: Path, plan) -> None:
    tool = load_tool()
    tensor = tool.Tensor(
        tool.DSPARK_STORE_TENSOR, (plan.store_size,), 24, 0,
        plan.store_size, new_rel_offset=0,
    )
    directory = (
        pack_string(tensor.name) + struct.pack("<I", 1) +
        struct.pack("<Q", plan.store_size) + struct.pack("<IQ", 24, 0)
    )
    header = b"GGUF" + struct.pack("<IQQ", 3, 1, 0)
    data_offset = tool.align_up(len(header) + len(directory), 32)
    with path.open("wb") as file:
        file.write(header)
        file.write(directory)
        file.write(bytes(data_offset - file.tell()))
        file.truncate(data_offset + plan.store_size)
    fd = os.open(path, os.O_RDWR)
    try:
        tool.pwrite_all(
            fd, plan.descriptor_bytes,
            data_offset + tool.STORE_HEADER_BYTES,
        )
        provisional = tool.make_header(plan, bytes(32), bytes(32))
        store_header = tool.make_header(
            plan, bytes(32), bytes(32),
            tool.manifest_digest(provisional, plan.descriptor_bytes),
        )
        tool.pwrite_all(fd, store_header, data_offset)
    finally:
        os.close(fd)


def mutated_dspark_store_plan(plan, mutation: str):
    """Return a generic-v2-valid plan that violates one DSpark 0731 axis."""
    tool = load_tool()
    if mutation not in {"type", "record", "offset"}:
        raise AssertionError(f"unknown DSpark store mutation: {mutation}")

    cursor = plan.data_offset
    layers = []
    for stage, original_layer in enumerate(plan.layers):
        if mutation == "offset" and stage == 0:
            cursor += tool.STORE_ALIGNMENT
        cursor = tool.align_up(cursor, tool.STORE_ALIGNMENT)
        components = []
        record_offset = 0
        for role, original_component in enumerate(original_layer.components):
            dims = original_component.tensor.dims
            ggml_type = original_component.tensor.ggml_type
            if mutation == "type" and stage == 0 and role in (0, 1):
                ggml_type = 17
            if mutation == "record" and stage == 0:
                dims = ((4352, 2048, plan.expert_count)
                        if role in (0, 1)
                        else (2048, 4352, plan.expert_count))
            expert_bytes = tool.tensor_nbytes(
                ggml_type, (dims[0], dims[1], 1)
            )
            tensor = tool.dataclasses.replace(
                original_component.tensor,
                dims=dims,
                ggml_type=ggml_type,
                size=expert_bytes * plan.expert_count,
            )
            components.append(tool.Component(
                role, tensor, expert_bytes, record_offset
            ))
            record_offset += expert_bytes
        layer_size = record_offset * plan.expert_count
        layers.append(tool.Layer(
            stage, plan.expert_count, record_offset, cursor, layer_size,
            tuple(components),
        ))
        cursor += layer_size

    mutated = tool.dataclasses.replace(
        plan,
        descriptor_bytes=b"",
        data_size=cursor - plan.data_offset,
        store_size=cursor,
        layers=layers,
    )
    return tool.dataclasses.replace(
        mutated,
        descriptor_bytes=b"".join(tool.pack_layer(layer) for layer in layers),
    )


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
    assert tool.DSPARK_0731_PROVENANCE == FINAL_DSPARK_PROVENANCE
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
        baseline_native = native.read_bytes()
        baseline_native_sha256 = hashlib.sha256(baseline_native).digest()
        assert f"output_sha256: {baseline_native_sha256.hex()}" in built.stdout

        repeated = run(
            "build", "--reserve-bytes", "0", str(source), str(native),
            ok=False,
        )
        assert "destination already exists" in repeated.stderr
        assert hashlib.sha256(native.read_bytes()).digest() == \
            baseline_native_sha256

        sentinel = tmp_path / "destination-symlink-target.txt"
        sentinel.write_bytes(b"do not replace through this symlink")
        symlink_destination = tmp_path / "destination-symlink.gguf"
        symlink_destination.symlink_to(sentinel.name)
        symlink_rejected = run(
            "build", "--reserve-bytes", "0", str(source),
            str(symlink_destination), ok=False,
        )
        assert "destination already exists" in symlink_rejected.stderr
        assert symlink_destination.is_symlink()
        assert sentinel.read_bytes() == b"do not replace through this symlink"

        raced_temp = tmp_path / ".raced-install.tmp"
        raced_destination = tmp_path / "raced-install.gguf"
        raced_temp.write_bytes(b"converter output")
        raced_destination.write_bytes(b"racer wins")
        raced_fd = os.open(raced_temp, os.O_RDONLY | os.O_CLOEXEC)
        try:
            raced_identity = tool.fd_identity(raced_fd, "raced output")
            try:
                tool.install_temp(
                    raced_temp, raced_destination, raced_fd, raced_identity,
                    hashlib.sha256(b"converter output").digest(),
                )
            except tool.FormatError as exc:
                assert "destination already exists" in str(exc)
            else:
                raise AssertionError(
                    "no-clobber install overwrote a raced destination"
                )
        finally:
            os.close(raced_fd)
        assert raced_destination.read_bytes() == b"racer wins"
        assert raced_temp.read_bytes() == b"converter output"

        fallback_temp = tmp_path / ".unsupported-clone.tmp"
        fallback_destination = tmp_path / "unsupported-clone.gguf"
        fallback_bytes = b"copy fallback from verified descriptor"
        fallback_temp.write_bytes(fallback_bytes)
        fallback_fd = os.open(fallback_temp, os.O_RDONLY | os.O_CLOEXEC)

        class UnsupportedClone:
            argtypes = None
            restype = None

            def __call__(self, *_args) -> int:
                tool.ctypes.set_errno(errno.ENOTSUP)
                return -1

        class UnsupportedLibc:
            fclonefileat = UnsupportedClone()

        original_cdll = tool.ctypes.CDLL
        tool.ctypes.CDLL = lambda *_args, **_kwargs: UnsupportedLibc()
        try:
            fallback_identity = tool.fd_identity(
                fallback_fd, "unsupported-clone source"
            )
            tool.install_temp(
                fallback_temp, fallback_destination, fallback_fd,
                fallback_identity, hashlib.sha256(fallback_bytes).digest(),
            )
        finally:
            tool.ctypes.CDLL = original_cdll
            os.close(fallback_fd)
        assert fallback_destination.read_bytes() == fallback_bytes
        assert not fallback_temp.exists()

        absent_temp = tmp_path / ".absent-clone-symbol.tmp"
        absent_destination = tmp_path / "absent-clone-symbol.gguf"
        absent_bytes = b"copy fallback when fclonefileat is absent"
        absent_temp.write_bytes(absent_bytes)
        absent_fd = os.open(absent_temp, os.O_RDONLY | os.O_CLOEXEC)

        class LibcWithoutClone:
            pass

        tool.ctypes.CDLL = lambda *_args, **_kwargs: LibcWithoutClone()
        try:
            absent_identity = tool.fd_identity(
                absent_fd, "absent-clone source"
            )
            tool.install_temp(
                absent_temp, absent_destination, absent_fd,
                absent_identity, hashlib.sha256(absent_bytes).digest(),
            )
        finally:
            tool.ctypes.CDLL = original_cdll
            os.close(absent_fd)
        assert absent_destination.read_bytes() == absent_bytes
        assert not absent_temp.exists()

        if sys.platform == "darwin":
            # fclonefileat creates the destination before install_temp reopens
            # it. A reopen failure removes the still-owned clone.
            reopen_temp = tmp_path / ".clone-reopen-failure.tmp"
            reopen_destination = tmp_path / "clone-reopen-failure.gguf"
            reopen_bytes = b"verified clone before reopen failure"
            reopen_temp.write_bytes(reopen_bytes)
            reopen_fd = os.open(reopen_temp, os.O_RDONLY | os.O_CLOEXEC)
            original_open = tool.os.open

            def fail_clone_reopen(path: object, flags: int, *args: object,
                                  **kwargs: object) -> int:
                if (path == reopen_destination.name and
                        kwargs.get("dir_fd") is not None and
                        flags & os.O_NOFOLLOW and
                        not flags & os.O_CREAT):
                    raise OSError(errno.EIO, "forced clone reopen failure")
                return original_open(path, flags, *args, **kwargs)

            tool.os.open = fail_clone_reopen
            try:
                reopen_identity = tool.fd_identity(
                    reopen_fd, "clone-reopen source"
                )
                try:
                    tool.install_temp(
                        reopen_temp, reopen_destination, reopen_fd,
                        reopen_identity,
                        hashlib.sha256(reopen_bytes).digest(),
                    )
                except OSError as exc:
                    assert exc.errno == errno.EIO
                else:
                    raise AssertionError("forced clone reopen did not fail")
            finally:
                tool.os.open = original_open
                os.close(reopen_fd)
            assert not reopen_destination.exists()

            # If a foreign entry takes the pathname immediately before the
            # failed reopen, cleanup still targets only the clone identity.
            race_temp = tmp_path / ".clone-reopen-race.tmp"
            race_destination = tmp_path / "clone-reopen-race.gguf"
            race_owned = tmp_path / "clone-reopen-owned.gguf"
            race_bytes = b"owned clone displaced before reopen"
            race_temp.write_bytes(race_bytes)
            race_fd = os.open(race_temp, os.O_RDONLY | os.O_CLOEXEC)

            def replace_then_fail_reopen(
                    path: object, flags: int, *args: object,
                    **kwargs: object) -> int:
                if (path == race_destination.name and
                        kwargs.get("dir_fd") is not None and
                        flags & os.O_NOFOLLOW and
                        not flags & os.O_CREAT):
                    race_destination.rename(race_owned)
                    race_destination.write_bytes(b"foreign replacement")
                    raise OSError(errno.EIO, "forced clone reopen failure")
                return original_open(path, flags, *args, **kwargs)

            tool.os.open = replace_then_fail_reopen
            try:
                race_identity = tool.fd_identity(
                    race_fd, "clone-reopen-race source"
                )
                try:
                    tool.install_temp(
                        race_temp, race_destination, race_fd,
                        race_identity,
                        hashlib.sha256(race_bytes).digest(),
                    )
                except OSError as exc:
                    assert exc.errno == errno.EIO
                else:
                    raise AssertionError("raced clone reopen did not fail")
            finally:
                tool.os.open = original_open
                os.close(race_fd)
            assert race_destination.read_bytes() == b"foreign replacement"
            assert race_owned.read_bytes() == race_bytes

        post_clone_temp = tmp_path / ".post-clone-error.tmp"
        post_clone_destination = tmp_path / "post-clone-error.gguf"
        saved_clone = tmp_path / "post-clone-owned.gguf"
        post_clone_bytes = b"verified clone before post-install failure"
        post_clone_temp.write_bytes(post_clone_bytes)
        post_clone_fd = os.open(
            post_clone_temp, os.O_RDONLY | os.O_CLOEXEC
        )
        original_hash_fd_for_clone = tool.hash_fd

        def replace_destination_after_clone(
                fd: int, size: int, label: str) -> bytes:
            digest = original_hash_fd_for_clone(fd, size, label)
            if label == "verify installed GGUF":
                post_clone_destination.rename(saved_clone)
                post_clone_destination.write_bytes(b"foreign replacement")
                return bytes(32)
            return digest

        tool.hash_fd = replace_destination_after_clone
        try:
            post_clone_identity = tool.fd_identity(
                post_clone_fd, "post-clone source"
            )
            try:
                tool.install_temp(
                    post_clone_temp, post_clone_destination, post_clone_fd,
                    post_clone_identity,
                    hashlib.sha256(post_clone_bytes).digest(),
                )
            except tool.FormatError as exc:
                assert "installed output SHA-256 mismatch" in str(exc)
            else:
                raise AssertionError("post-clone mismatch was accepted")
        finally:
            tool.hash_fd = original_hash_fd_for_clone
            os.close(post_clone_fd)
        assert post_clone_destination.read_bytes() == b"foreign replacement"
        assert saved_clone.read_bytes() == post_clone_bytes

        # Installation reads the verified FD, never the replaceable temporary
        # pathname. A foreign replacement is retained by ownership-aware
        # cleanup; an independently existing destination is never clobbered.
        for destination_exists in (False, True):
            mode = "existing" if destination_exists else "new"
            substituted_temp = tmp_path / f".{mode}-substitute.tmp"
            saved_verified = tmp_path / f".{mode}-verified-inode"
            substituted_destination = tmp_path / f"{mode}-substitute.gguf"
            substituted_temp.write_bytes(b"verified output inode")
            substituted_fd = os.open(
                substituted_temp, os.O_RDONLY | os.O_CLOEXEC
            )
            try:
                substituted_identity = tool.fd_identity(
                    substituted_fd, f"{mode} verified output"
                )
                substituted_temp.rename(saved_verified)
                substituted_temp.write_bytes(b"attacker replacement!")
                if destination_exists:
                    substituted_destination.write_bytes(
                        b"preserve existing destination"
                    )
                try:
                    tool.install_temp(
                        substituted_temp, substituted_destination,
                        substituted_fd, substituted_identity,
                        hashlib.sha256(b"verified output inode").digest(),
                    )
                except tool.FormatError as exc:
                    if not destination_exists:
                        raise
                    assert "destination already exists" in str(exc)
                else:
                    if destination_exists:
                        raise AssertionError(
                            "existing destination was unexpectedly replaced"
                        )
                assert substituted_temp.read_bytes() == \
                    b"attacker replacement!"
                if destination_exists:
                    assert substituted_destination.read_bytes() == \
                        b"preserve existing destination"
                else:
                    assert substituted_destination.read_bytes() == \
                        b"verified output inode"
            finally:
                os.close(substituted_fd)

        source_symlink = tmp_path / "source-symlink.gguf"
        source_symlink.symlink_to(source.name)
        symlink_input_output = tmp_path / "source-symlink-output.gguf"
        run(
            "build", "--reserve-bytes", "0", str(source_symlink),
            str(symlink_input_output), ok=False,
        )
        assert not symlink_input_output.exists()

        # Authenticate, parse, hash, copy, and verify through one source FD.
        # The restore immediately before verification models an attacker that
        # made path-based authentication appear consistent around a swapped
        # inode; the built bytes must still come from the authenticated inode.
        stable_source = tmp_path / "fd-stable-source.gguf"
        replacement_source = tmp_path / "fd-stable-replacement.gguf"
        saved_source = tmp_path / "fd-stable-authenticated.gguf"
        stable_native = tmp_path / "fd-stable-native.gguf"
        shutil.copyfile(source, stable_source)
        shutil.copyfile(source, replacement_source)
        replacement_gguf = tool.load_gguf(replacement_source)
        routed = next(
            tensor for tensor in replacement_gguf.tensors
            if tool.ROUTED_RE.fullmatch(tensor.name)
        )
        with replacement_source.open("r+b", buffering=0) as file:
            original = os.pread(file.fileno(), 1, routed.abs_offset)
            os.pwrite(file.fileno(), bytes([original[0] ^ 0x5A]),
                      routed.abs_offset)

        original_hash_fd = tool.hash_fd
        original_hash_file = tool.hash_file
        original_verify_open = tool.verify_open
        original_verify = tool.verify
        swap_state = {"swapped": False, "restored": False}

        def swap_after_hash() -> None:
            if swap_state["swapped"]:
                return
            stable_source.rename(saved_source)
            replacement_source.rename(stable_source)
            swap_state["swapped"] = True

        def restore_before_verify() -> None:
            if not swap_state["swapped"] or swap_state["restored"]:
                return
            stable_source.unlink()
            saved_source.rename(stable_source)
            swap_state["restored"] = True

        def hash_fd_with_swap(fd: int, size: int, label: str) -> bytes:
            digest = original_hash_fd(fd, size, label)
            if label == "hash source GGUF":
                swap_after_hash()
            return digest

        def hash_file_with_swap(path: Path, label: str) -> bytes:
            digest = original_hash_file(path, label)
            if label == "hash source GGUF":
                swap_after_hash()
            return digest

        def verify_open_with_restore(*args, **kwargs) -> None:
            restore_before_verify()
            original_verify_open(*args, **kwargs)

        def verify_with_restore(*args, **kwargs) -> None:
            restore_before_verify()
            original_verify(*args, **kwargs)

        tool.hash_fd = hash_fd_with_swap
        tool.hash_file = hash_file_with_swap
        tool.verify_open = verify_open_with_restore
        tool.verify = verify_with_restore
        try:
            tool.build(stable_source, stable_native, 0, True)
        finally:
            restore_before_verify()
            tool.verify = original_verify
            tool.verify_open = original_verify_open
            tool.hash_file = original_hash_file
            tool.hash_fd = original_hash_fd
        assert swap_state == {"swapped": True, "restored": True}
        assert stable_native.read_bytes() == baseline_native

        same_destination = run(
            "build", "--reserve-bytes", "0", str(source), str(source),
            ok=False,
        )
        assert "destination aliases input" in same_destination.stderr
        hardlink_destination = tmp_path / "source-hardlink.gguf"
        os.link(source, hardlink_destination)
        hardlink_result = run(
            "build", "--reserve-bytes", "0", str(source),
            str(hardlink_destination), ok=False,
        )
        assert "destination aliases input" in hardlink_result.stderr

        native_gguf = tool.load_gguf(native)
        store = next(t for t in native_gguf.tensors if t.name == tool.STORE_TENSOR)
        if probe:
            subprocess.run([probe, str(native), str(store.abs_offset),
                            str(store.size),
                            str(tool.STORE_FAMILY_DEEPSEEK4)], check=True)

        combined_source = tmp_path / "deepseek-0731-source.gguf"
        dspark_support = tmp_path / "dspark-support.gguf"
        write_sparse_target_0731(combined_source)
        write_dspark_fixture(dspark_support)
        target_gguf = tool.load_gguf(combined_source)
        tool.reject_target_dspark_namespace(target_gguf)
        target_plan = tool.make_store_plan(target_gguf)
        tool.validate_dspark_target_0731(target_plan)
        support_gguf = tool.load_gguf(dspark_support)
        support_plan = tool.make_dspark_store_plan(support_gguf, target_plan)
        assert tool.DSPARK_PREVIEW_REFERENCE_SHA256.hex() == (
            "8b3adf5942bec22ae2ea867cd7079cf13530ba83ffcffaf00f5de48664a1a34e"
        )
        production_pin = tool.DSPARK_0731_FINAL_SUPPORT_SHA256
        assert production_pin is not None
        assert production_pin.hex() == (
            "aa2bd4b5b916e1aa0a01392d69cbdd9798a3f3050c29c22973c8ee4233af0413"
        )
        tool.DSPARK_0731_FINAL_SUPPORT_SHA256 = \
            tool.DSPARK_PREVIEW_REFERENCE_SHA256
        try:
            tool.require_final_dspark_support_pin()
        except tool.FormatError as exc:
            assert "preview DSpark SHA-256" in str(exc)
        else:
            raise AssertionError("preview DSpark digest became a production pin")
        finally:
            tool.DSPARK_0731_FINAL_SUPPORT_SHA256 = production_pin
        tool.require_digest_match(
            tool.DSPARK_PREVIEW_REFERENCE_SHA256,
            tool.DSPARK_PREVIEW_REFERENCE_SHA256,
        )
        try:
            tool.require_digest_match(
                tool.DSPARK_PREVIEW_REFERENCE_SHA256, production_pin,
            )
        except tool.FormatError as exc:
            assert "does not match the required artifact" in str(exc)
            assert production_pin.hex() in str(exc)
        else:
            raise AssertionError("preview support matched the final pin")
        assert tool.require_final_dspark_support_pin() == production_pin
        assert target_plan.layer_count == 43
        assert target_plan.expert_count == 256
        assert target_plan.expert_used_count == 6
        assert support_plan.layer_count == 3
        assert support_plan.source_tensor_count == 81
        assert support_plan.layers[0].record_bytes == 7077888
        assert [component.tensor.ggml_type
                for component in support_plan.layers[0].components] == [16, 16, 10]
        assert [component.tensor.dims
                for component in support_plan.layers[0].components] == [
                    (4096, 2048, 256),
                    (4096, 2048, 256),
                    (2048, 4096, 256),
                ]
        appended_metadata = tool.dspark_metadata_records(support_gguf)
        combined_tensors, _, _, combined_kv = tool.combined_layout(
            target_gguf, target_plan, support_gguf, support_plan,
            appended_metadata,
        )
        assert [tensor.name for tensor in combined_tensors[-2:]] == [
            tool.STORE_TENSOR, tool.DSPARK_STORE_TENSOR,
        ]
        assert len([tensor for tensor in combined_tensors
                    if tensor.name.startswith("mtp.")]) == 72
        assert combined_kv == target_gguf.kv_raw + b"".join(appended_metadata)

        blocked_output = tmp_path / "blocked-preview-combined.gguf"
        blocked_build = run(
            "build", "--reserve-bytes", "0",
            "--dspark-support", str(dspark_support),
            str(combined_source), str(blocked_output), ok=False,
        )
        assert "does not match the required artifact" in blocked_build.stderr
        assert production_pin.hex() in blocked_build.stderr
        assert not blocked_output.exists()
        # The same writer and plan produce identical target-store bytes at a
        # different outer GGUF offset, which is the combined-file invariant.
        target_digest = tool.hash_file(source, "fixture target identity")
        duplicate_store = tmp_path / "duplicate-target-store.bin"
        duplicate_offset = 4096
        duplicate_fd = os.open(
            duplicate_store, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o644
        )
        source_fd = os.open(source, os.O_RDONLY)
        try:
            os.ftruncate(duplicate_fd, duplicate_offset + store.size)
            duplicate_payload = tool.write_expert_store(
                tool.make_store_plan(tool.load_gguf(source)), source_fd,
                duplicate_fd, duplicate_offset, target_digest,
                "duplicate target expert-major",
            )
        finally:
            os.close(source_fd)
            os.close(duplicate_fd)
        with native.open("rb") as left, duplicate_store.open("rb") as right:
            left.seek(store.abs_offset)
            right.seek(duplicate_offset)
            assert left.read(store.size) == right.read(store.size)
        assert duplicate_payload.hex() in built.stdout

        sparse_aux = tmp_path / "sparse-dspark-store.gguf"
        write_sparse_dspark_store_container(sparse_aux, support_plan)
        sparse_aux_gguf = tool.load_gguf(sparse_aux)
        sparse_aux_store = sparse_aux_gguf.tensors[0]
        aux_manifest, aux_layers = tool.parse_store(
            sparse_aux_gguf, sparse_aux_store, tool.DSPARK_STORE_TENSOR,
        )
        assert aux_manifest["source_tensors"] == 81
        assert [layer.index for layer in aux_layers] == [0, 1, 2]
        for field in ("data_offset", "data_size", "store_size"):
            mismatched_plan = tool.dataclasses.replace(
                support_plan,
                **{field: getattr(support_plan, field) + tool.STORE_ALIGNMENT},
            )
            try:
                tool.verify_store_identity(
                    support_gguf, mismatched_plan, sparse_aux_gguf,
                    sparse_aux_store, tool.DSPARK_STORE_TENSOR, "DSpark",
                )
            except tool.FormatError as exc:
                assert "identity does not match" in str(exc)
            else:
                raise AssertionError(
                    f"store identity ignored planned {field}"
                )
        if probe:
            subprocess.run([
                probe, str(sparse_aux), str(sparse_aux_store.abs_offset),
                str(sparse_aux_store.size),
                str(tool.STORE_FAMILY_DEEPSEEK4), "0", "0",
                "dspark-0731",
            ], check=True)

        for mutation in ("type", "record", "offset"):
            rejected_plan = mutated_dspark_store_plan(
                support_plan, mutation
            )
            if mutation == "type":
                assert [component.tensor.ggml_type for component in
                        rejected_plan.layers[0].components] == [17, 17, 10]
            elif mutation == "record":
                assert rejected_plan.layers[0].record_bytes != \
                    support_plan.layers[0].record_bytes
            else:
                assert rejected_plan.layers[0].data_offset == \
                    support_plan.layers[0].data_offset + tool.STORE_ALIGNMENT
            rejected_path = tmp_path / f"bad-dspark-store-{mutation}.gguf"
            write_sparse_dspark_store_container(rejected_path, rejected_plan)
            rejected_gguf = tool.load_gguf(rejected_path)
            rejected_store = rejected_gguf.tensors[0]
            rejected_manifest, rejected_layers = tool.parse_store(
                rejected_gguf, rejected_store, tool.DSPARK_STORE_TENSOR,
            )
            assert len(rejected_layers) == tool.DSPARK_STAGE_COUNT
            try:
                tool.verify_store_payload(
                    support_plan, rejected_store, rejected_manifest,
                    rejected_layers, -1, -1, "rejected DSpark fixture",
                )
            except tool.FormatError as exc:
                assert "manifest layer layout differs" in str(exc)
            else:
                raise AssertionError(
                    f"store payload accepted {mutation} layout drift"
                )
            with rejected_path.open("rb", buffering=0) as file:
                rejected_header = os.pread(
                    file.fileno(), tool.STORE_HEADER_BYTES,
                    rejected_store.abs_offset,
                )
                rejected_descriptors = os.pread(
                    file.fileno(), len(rejected_plan.descriptor_bytes),
                    rejected_store.abs_offset + tool.STORE_HEADER_BYTES,
                )
            assert rejected_manifest["manifest_sha256"] == \
                tool.manifest_digest(
                    rejected_header, rejected_descriptors
                )
            if probe:
                subprocess.run([
                    probe, str(rejected_path),
                    str(rejected_store.abs_offset), str(rejected_store.size),
                    str(tool.STORE_FAMILY_DEEPSEEK4), "0", "0",
                    "dspark-0731-reject",
                ], check=True)

        for mutation, expected_error in (
            ("type", "component byte size mismatch"),
            ("record", "invalid layer descriptor"),
            ("offset", "invalid layer descriptor"),
        ):
            tampered_path = tmp_path / f"tampered-dspark-{mutation}.gguf"
            write_sparse_dspark_store_container(tampered_path, support_plan)
            tampered_gguf = tool.load_gguf(tampered_path)
            tampered_store = tampered_gguf.tensors[0]
            descriptor_count = support_plan.layer_count * \
                tool.STORE_LAYER_BYTES
            with tampered_path.open("r+b", buffering=0) as file:
                header = bytearray(os.pread(
                    file.fileno(), tool.STORE_HEADER_BYTES,
                    tampered_store.abs_offset,
                ))
                descriptors = bytearray(os.pread(
                    file.fileno(), descriptor_count,
                    tampered_store.abs_offset + tool.STORE_HEADER_BYTES,
                ))
                if mutation == "type":
                    struct.pack_into(
                        "<I", descriptors,
                        tool.STORE_COMPONENT_OFFSET + 4, 17,
                    )
                elif mutation == "record":
                    record_bytes = struct.unpack_from("<Q", descriptors, 8)[0]
                    struct.pack_into("<Q", descriptors, 8, record_bytes + 1)
                else:
                    layer_offset = struct.unpack_from(
                        "<Q", descriptors, 16
                    )[0]
                    struct.pack_into(
                        "<Q", descriptors, 16,
                        layer_offset + tool.STORE_ALIGNMENT,
                    )
                header[
                    tool.MANIFEST_DIGEST_OFFSET:
                    tool.MANIFEST_DIGEST_OFFSET + 32
                ] = bytes(32)
                header[
                    tool.MANIFEST_DIGEST_OFFSET:
                    tool.MANIFEST_DIGEST_OFFSET + 32
                ] = tool.manifest_digest(bytes(header), bytes(descriptors))
                file.seek(tampered_store.abs_offset)
                file.write(header)
                file.seek(
                    tampered_store.abs_offset + tool.STORE_HEADER_BYTES
                )
                file.write(descriptors)
            try:
                tool.parse_store(
                    tampered_gguf, tampered_store,
                    tool.DSPARK_STORE_TENSOR,
                )
            except tool.FormatError as exc:
                assert expected_error in str(exc)
            else:
                raise AssertionError(
                    f"generic store reader accepted {mutation} drift"
                )
            if probe:
                subprocess.run([
                    probe, str(tampered_path),
                    str(tampered_store.abs_offset), str(tampered_store.size),
                    str(tool.STORE_FAMILY_DEEPSEEK4), "0", "0",
                    "store-reject",
                ], check=True)

        descriptor_bytes = (
            support_plan.layer_count * tool.STORE_LAYER_BYTES
        )
        descriptor_start = (
            sparse_aux_store.abs_offset + tool.STORE_HEADER_BYTES
        )
        with sparse_aux.open("r+b") as file:
            original_header = os.pread(
                file.fileno(), tool.STORE_HEADER_BYTES,
                sparse_aux_store.abs_offset,
            )
            original_descriptors = os.pread(
                file.fileno(), descriptor_bytes, descriptor_start,
            )
            predata_byte = descriptor_start + descriptor_bytes
            file.seek(predata_byte)
            file.write(b"\x01")
        try:
            tool.parse_store(
                sparse_aux_gguf, sparse_aux_store,
                tool.DSPARK_STORE_TENSOR,
            )
        except tool.FormatError as exc:
            assert "pre-data padding" in str(exc)
        else:
            raise AssertionError("non-zero auxiliary pre-data padding accepted")
        with sparse_aux.open("r+b") as file:
            file.seek(predata_byte)
            file.write(b"\x00")

        overlapping = bytearray(original_descriptors)
        first_layer_offset = struct.unpack_from("<Q", overlapping, 16)[0]
        struct.pack_into(
            "<Q", overlapping, tool.STORE_LAYER_BYTES + 16,
            first_layer_offset,
        )
        overlap_header = bytearray(original_header)
        overlap_header[
            tool.MANIFEST_DIGEST_OFFSET:
            tool.MANIFEST_DIGEST_OFFSET + 32
        ] = bytes(32)
        overlap_header[
            tool.MANIFEST_DIGEST_OFFSET:
            tool.MANIFEST_DIGEST_OFFSET + 32
        ] = tool.manifest_digest(bytes(overlap_header), bytes(overlapping))
        with sparse_aux.open("r+b") as file:
            file.seek(sparse_aux_store.abs_offset)
            file.write(overlap_header)
            file.seek(descriptor_start)
            file.write(overlapping)
        try:
            tool.parse_store(
                sparse_aux_gguf, sparse_aux_store,
                tool.DSPARK_STORE_TENSOR,
            )
        except tool.FormatError as exc:
            assert "invalid layer descriptor" in str(exc)
        else:
            raise AssertionError("overlapping auxiliary descriptors accepted")
        with sparse_aux.open("r+b") as file:
            file.seek(sparse_aux_store.abs_offset)
            file.write(original_header)
            file.seek(descriptor_start)
            file.write(original_descriptors)

        alias_target = tmp_path / "target-with-dspark-alias.gguf"
        write_sparse_target_0731(alias_target, dspark_alias=True)
        try:
            tool.reject_target_dspark_namespace(tool.load_gguf(alias_target))
        except tool.FormatError as exc:
            assert "metadata alias" in str(exc)
        else:
            raise AssertionError("target DSpark metadata alias was accepted")

        alias_target_tensor = tmp_path / "target-with-dspark-tensor.gguf"
        write_sparse_target_0731(
            alias_target_tensor, dspark_tensor_alias=True
        )
        try:
            tool.reject_target_dspark_namespace(
                tool.load_gguf(alias_target_tensor)
            )
        except tool.FormatError as exc:
            assert "tensor alias" in str(exc)
        else:
            raise AssertionError("target DSpark tensor alias was accepted")

        for key, value in (
            ("deepseek4.block_count", 42),
            ("deepseek4.embedding_length", 4095),
            ("deepseek4.vocab_size", 129279),
            ("deepseek4.expert_count", 255),
            ("deepseek4.expert_used_count", 5),
            ("deepseek4.expert_feed_forward_length", 2047),
        ):
            bad_target = tmp_path / f"bad-target-{key.rsplit('.', 1)[-1]}.gguf"
            write_sparse_target_0731(
                bad_target, metadata_override=(key, value)
            )
            try:
                bad_target_plan = tool.make_store_plan(tool.load_gguf(bad_target))
                tool.validate_dspark_target_0731(bad_target_plan)
            except tool.FormatError:
                pass
            else:
                raise AssertionError(f"target metadata drift accepted: {key}")

        bad_target_shape = tmp_path / "bad-target-routed-shape.gguf"
        write_sparse_target_0731(bad_target_shape, bad_routed_shape=True)
        try:
            bad_target_plan = tool.make_store_plan(
                tool.load_gguf(bad_target_shape)
            )
            tool.validate_dspark_target_0731(bad_target_plan)
        except tool.FormatError:
            pass
        else:
            raise AssertionError("target routed geometry drift was accepted")

        target_alias_result = run(
            "build", "--dspark-support", str(dspark_support),
            str(combined_source), str(combined_source), ok=False,
        )
        assert "destination aliases input" in target_alias_result.stderr
        support_alias_result = run(
            "build", "--dspark-support", str(dspark_support),
            str(combined_source), str(dspark_support), ok=False,
        )
        assert "destination aliases input" in support_alias_result.stderr

        metadata_drifts: tuple[
            tuple[str, int | tuple[int, ...]], ...
        ] = (
            ("dspark.block_size", 4),
            ("dspark.markov_rank", 255),
            ("dspark.noise_token_id", 128798),
            ("dspark.target_layer_ids", (39, 40, 41)),
            ("dspark.stage_count", 2),
            ("dspark.n_layers", 2),
        )
        for key, value in metadata_drifts:
            bad_support = tmp_path / f"bad-{key.replace('.', '-')}.gguf"
            write_dspark_fixture(bad_support, metadata_overrides={key: value})
            try:
                tool.make_dspark_store_plan(
                    tool.load_gguf(bad_support), target_plan
                )
            except tool.FormatError as exc:
                assert "metadata values are outside" in str(exc)
            else:
                raise AssertionError(f"DSpark metadata drift accepted: {key}")

        bad_provenance = tmp_path / "bad-dspark-source-revision.gguf"
        write_dspark_fixture(
            bad_provenance,
            metadata_overrides={"dspark.source.revision": "unreviewed"},
        )
        try:
            tool.make_dspark_store_plan(
                tool.load_gguf(bad_provenance), target_plan
            )
        except tool.FormatError as exc:
            assert "independently pinned final 0731" in str(exc)
        else:
            raise AssertionError("self-declared DSpark provenance was accepted")

        bad_alignment = tmp_path / "bad-dspark-alignment.gguf"
        write_dspark_fixture(bad_alignment, alignment=64)
        try:
            tool.make_dspark_store_plan(tool.load_gguf(bad_alignment), target_plan)
        except tool.FormatError as exc:
            assert "alignment differ" in str(exc)
        else:
            raise AssertionError("DSpark alignment drift was accepted")

        for label, options, expected_error in (
            ("metadata-missing", {"omit_metadata": "dspark.markov_rank"},
             "metadata inventory mismatch"),
            ("metadata-extra", {"extra_metadata": True},
             "metadata inventory mismatch"),
            ("metadata-type", {"metadata_type_drift": "dspark.block_size"},
             "metadata types do not match"),
            ("tensor-missing", {"omit_tensor": "mtp.1.attn_norm.weight"},
             "tensor inventory mismatch"),
            ("tensor-extra", {"extra_tensor": True},
             "tensor inventory mismatch"),
            ("static-shape", {"bad_static_shape": True},
             "static tensor contract mismatch"),
        ):
            bad_support = tmp_path / f"bad-dspark-{label}.gguf"
            write_dspark_fixture(bad_support, **options)
            try:
                tool.make_dspark_store_plan(
                    tool.load_gguf(bad_support), target_plan
                )
            except tool.FormatError as exc:
                assert expected_error in str(exc)
            else:
                raise AssertionError(f"bad DSpark {label} fixture was accepted")

        trailing = tmp_path / "bad-trailing.gguf"
        shutil.copyfile(native, trailing)
        with trailing.open("ab") as file:
            file.write(b"\x00")
        trailing_result = run("verify", str(source), str(trailing), ok=False)
        assert "GGUF byte range mismatch" in trailing_result.stderr

        bad_gguf_padding = tmp_path / "bad-gguf-padding.gguf"
        shutil.copyfile(native, bad_gguf_padding)
        native_non_store = next(
            tensor for tensor in native_gguf.tensors
            if tensor.name != tool.STORE_TENSOR
        )
        next_tensor = next(
            tensor for tensor in native_gguf.tensors
            if tensor.abs_offset > native_non_store.abs_offset
        )
        padding_offset = native_non_store.abs_offset + native_non_store.size
        assert padding_offset < next_tensor.abs_offset
        with bad_gguf_padding.open("r+b") as file:
            file.seek(padding_offset)
            file.write(b"\x01")
        padding_result = run(
            "verify", str(source), str(bad_gguf_padding), ok=False
        )
        assert "GGUF tensor padding" in padding_result.stderr

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
