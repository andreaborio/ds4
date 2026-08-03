#!/usr/bin/env python3
"""Model-free runtime admission fixtures for the embedded DSpark 0731 store."""

from __future__ import annotations

import hashlib
import os
import struct
import subprocess
import sys
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path


UINT32 = 4
FLOAT32 = 6
BOOL = 7
STRING = 8
ARRAY = 9
UINT64 = 10

F32 = 0
F16 = 1
Q8_0 = 8
Q2_K = 10
IQ2_XXS = 16
I8 = 24
I32 = 26

TARGET_STORE = "ds4.expert_major.v2"
DSPARK_STORE = "ds4.dspark.expert_major.v2"
STORE_HEADER_BYTES = 256
STORE_LAYER_BYTES = 224
STORE_COMPONENT_BYTES = 56
STORE_COMPONENT_OFFSET = 32
STORE_ALIGNMENT = 4096
STORE_MANIFEST_DIGEST_OFFSET = 168
DSPARK_SUPPORT_SOURCE_BYTES = 5_989_114_912
DSPARK_SUPPORT_SOURCE_SHA256 = bytes.fromhex(
    "aa2bd4b5b916e1aa0a01392d69cbdd9798a3f3050c29c22973c8ee4233af0413"
)
DSPARK_SUPPORT_PAYLOAD_SHA256 = bytes.fromhex(
    "66398593c23efe9ac1be1c9bcc0f95087257e0b3e98087e892b6887ad3d80c95"
)
DSPARK_PROVENANCE = OrderedDict([
    ("dspark.source.revision",
     "7872f01b1d1fe23eabc4c98b48bffcef5a386062"),
    ("dspark.source.config_sha256",
     "6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023"),
    ("dspark.source.index_sha256",
     "98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b"),
    ("dspark.source.shard46_sha256",
     "5db924ca907e0d93acd975bd5079c3662717f9ac709f23d079bd8f816d29d9dd"),
    ("dspark.source.shard47_sha256",
     "62816173f9f6e136b20b48e3b6f16613ac9ea02b5603f636928b253244a548bd"),
    ("dspark.source.shard48_sha256",
     "cc43742bd24ae6bcdea343a91442f6f66aed2cfebcc6b235470204851ce2f8a9"),
])


@dataclass(frozen=True)
class Tensor:
    name: str
    dims: tuple[int, ...]
    value_type: int


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def pack_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def pack_scalar(value_type: int, value: object) -> bytes:
    if value_type == UINT32:
        return struct.pack("<I", int(value))
    if value_type == UINT64:
        return struct.pack("<Q", int(value))
    if value_type == FLOAT32:
        return struct.pack("<f", float(value))
    if value_type == BOOL:
        return struct.pack("<B", bool(value))
    if value_type == STRING:
        return pack_string(str(value))
    raise AssertionError(f"unsupported metadata type: {value_type}")


def pack_value(value_type: int, value: object) -> bytes:
    if value_type != ARRAY:
        return pack_scalar(value_type, value)
    item_type, items = value
    return (
        struct.pack("<IQ", item_type, len(items))
        + b"".join(pack_scalar(item_type, item) for item in items)
    )


def tensor_bytes(tensor: Tensor) -> int:
    elements = 1
    for dim in tensor.dims:
        elements *= dim
    block_elements, block_bytes = {
        F32: (1, 4),
        F16: (1, 2),
        Q8_0: (32, 34),
        Q2_K: (256, 84),
        IQ2_XXS: (256, 66),
        I8: (1, 1),
        I32: (1, 4),
    }[tensor.value_type]
    return ((elements + block_elements - 1) // block_elements) * block_bytes


def deepseek_metadata(*, include_dspark: bool, alias: bool = False,
                      case_alias: bool = False,
                      stage_count: int = 3,
                      target_layers: list[int] | None = None,
                      target_layers_type: int = UINT32,
                      provenance_omit: str | None = None,
                      provenance_altered: str | None = None,
                      provenance_bad_type: str | None = None,
                      provenance_case_alias: bool = False,
                      ) -> OrderedDict[str, tuple[int, object]]:
    ratios = [0 if layer < 2 else 4 if layer % 2 == 0 else 128
              for layer in range(43)]
    metadata: OrderedDict[str, tuple[int, object]] = OrderedDict([
        ("general.architecture", (STRING, "deepseek4")),
        ("general.name", (STRING, "DeepSeek V4 Flash 0731 synthetic")),
        ("general.alignment", (UINT32, 32)),
        ("deepseek4.block_count", (UINT32, 43)),
        ("deepseek4.context_length", (UINT64, 163840)),
        ("deepseek4.embedding_length", (UINT32, 4096)),
        ("deepseek4.vocab_size", (UINT32, 129280)),
        ("deepseek4.attention.head_count", (UINT32, 64)),
        ("deepseek4.attention.head_count_kv", (UINT32, 1)),
        ("deepseek4.attention.key_length", (UINT32, 512)),
        ("deepseek4.attention.value_length", (UINT32, 512)),
        ("deepseek4.rope.dimension_count", (UINT32, 64)),
        ("deepseek4.attention.q_lora_rank", (UINT32, 1024)),
        ("deepseek4.attention.output_lora_rank", (UINT32, 1024)),
        ("deepseek4.attention.output_group_count", (UINT32, 8)),
        ("deepseek4.expert_count", (UINT32, 256)),
        ("deepseek4.expert_used_count", (UINT32, 6)),
        ("deepseek4.expert_feed_forward_length", (UINT32, 2048)),
        ("deepseek4.expert_shared_count", (UINT32, 1)),
        ("deepseek4.expert_group_count", (UINT32, 0)),
        ("deepseek4.expert_group_used_count", (UINT32, 0)),
        ("deepseek4.hash_layer_count", (UINT32, 3)),
        ("deepseek4.attention.sliding_window", (UINT32, 128)),
        ("deepseek4.attention.indexer.head_count", (UINT32, 64)),
        ("deepseek4.attention.indexer.key_length", (UINT32, 128)),
        ("deepseek4.attention.indexer.top_k", (UINT32, 512)),
        ("deepseek4.hyper_connection.count", (UINT32, 4)),
        ("deepseek4.hyper_connection.sinkhorn_iterations", (UINT32, 20)),
        ("deepseek4.attention.compress_ratios", (ARRAY, (UINT32, ratios))),
        ("deepseek4.swiglu_clamp_exp", (ARRAY, (FLOAT32, [10.0] * 43))),
        ("deepseek4.rope.scaling.original_context_length", (UINT64, 65536)),
        ("deepseek4.rope.freq_base", (FLOAT32, 10000.0)),
        ("deepseek4.rope.scaling.factor", (FLOAT32, 16.0)),
        ("deepseek4.rope.scaling.yarn_beta_fast", (FLOAT32, 32.0)),
        ("deepseek4.rope.scaling.yarn_beta_slow", (FLOAT32, 1.0)),
        ("deepseek4.attention.compress_rope_freq_base", (FLOAT32, 160000.0)),
        ("deepseek4.expert_weights_scale", (FLOAT32, 1.5)),
        ("deepseek4.attention.layer_norm_rms_epsilon", (FLOAT32, 1.0e-6)),
        ("deepseek4.hyper_connection.epsilon", (FLOAT32, 1.0e-6)),
        ("deepseek4.expert_weights_norm", (BOOL, True)),
    ])
    if include_dspark:
        metadata.update([
            ("dspark.block_size", (UINT32, 5)),
            ("dspark.markov_rank", (UINT32, 256)),
            ("dspark.noise_token_id", (UINT32, 128799)),
            ("dspark.target_layer_ids", (
                ARRAY,
                (target_layers_type,
                 [40, 41, 42] if target_layers is None else target_layers),
            )),
            ("dspark.stage_count", (UINT32, stage_count)),
            ("dspark.n_layers", (UINT32, stage_count)),
        ])
        for key, value in DSPARK_PROVENANCE.items():
            if key == provenance_omit:
                continue
            if key == provenance_altered:
                value = "unreviewed"
            metadata[key] = (
                (UINT32, 7) if key == provenance_bad_type
                else (STRING, value)
            )
        if alias:
            metadata["deepseek4.dspark.block_size"] = (UINT32, 5)
        if case_alias:
            metadata["DeepSeek4.DSPark.block_size"] = (UINT32, 5)
        if provenance_case_alias:
            metadata["DSpark.Source.Revision"] = (
                STRING, DSPARK_PROVENANCE["dspark.source.revision"]
            )
    return metadata


def common_block(prefix: str, *, router_type: int) -> list[Tensor]:
    return [
        Tensor(prefix + "hc_attn_base.weight", (24,), F32),
        Tensor(prefix + "hc_attn_fn.weight", (16384, 24), F16),
        Tensor(prefix + "hc_attn_scale.weight", (3,), F32),
        Tensor(prefix + "attn_sinks.weight", (64,), F32),
        Tensor(prefix + "attn_q_a.weight", (4096, 1024), Q8_0),
        Tensor(prefix + "attn_q_a_norm.weight", (1024,), F32),
        Tensor(prefix + "attn_q_b.weight", (1024, 32768), Q8_0),
        Tensor(prefix + "attn_kv.weight", (4096, 512), Q8_0),
        Tensor(prefix + "attn_kv_a_norm.weight", (512,), F32),
        Tensor(prefix + "attn_output_a.weight", (4096, 8192), Q8_0),
        Tensor(prefix + "attn_output_b.weight", (8192, 4096), Q8_0),
        Tensor(prefix + "attn_norm.weight", (4096,), F32),
        Tensor(prefix + "hc_ffn_base.weight", (24,), F32),
        Tensor(prefix + "hc_ffn_fn.weight", (16384, 24), F16),
        Tensor(prefix + "hc_ffn_scale.weight", (3,), F32),
        Tensor(prefix + "ffn_gate_inp.weight", (4096, 256), router_type),
        Tensor(prefix + "exp_probs_b.bias", (256,), F32),
        Tensor(prefix + "ffn_norm.weight", (4096,), F32),
        Tensor(prefix + "ffn_gate_shexp.weight", (4096, 2048), Q8_0),
        Tensor(prefix + "ffn_up_shexp.weight", (4096, 2048), Q8_0),
        Tensor(prefix + "ffn_down_shexp.weight", (2048, 4096), Q8_0),
    ]


def target_non_routed_tensors() -> list[Tensor]:
    tensors = [
        Tensor("token_embd.weight", (4096, 129280), F16),
        Tensor("output_hc_base.weight", (4,), F32),
        Tensor("output_hc_fn.weight", (16384, 4), F16),
        Tensor("output_hc_scale.weight", (1,), F32),
        Tensor("output_norm.weight", (4096,), F32),
        Tensor("output.weight", (4096, 129280), Q8_0),
    ]
    for layer in range(43):
        prefix = f"blk.{layer}."
        tensors += common_block(prefix, router_type=F16)
        if layer < 3:
            tensors.append(Tensor(
                prefix + "ffn_gate_tid2eid.weight", (6, 129280), I32
            ))
        ratio = 0 if layer < 2 else 4 if layer % 2 == 0 else 128
        if ratio:
            compression_width = (2 if ratio == 4 else 1) * 512
            tensors += [
                Tensor(prefix + "attn_compressor_ape.weight",
                       (compression_width, ratio), F16),
                Tensor(prefix + "attn_compressor_kv.weight",
                       (4096, compression_width), F16),
                Tensor(prefix + "attn_compressor_gate.weight",
                       (4096, compression_width), F16),
                Tensor(prefix + "attn_compressor_norm.weight", (512,), F32),
            ]
        if ratio == 4:
            tensors += [
                Tensor(prefix + "indexer.attn_q_b.weight",
                       (1024, 8192), Q8_0),
                Tensor(prefix + "indexer.proj.weight", (4096, 64), F16),
                Tensor(prefix + "indexer_compressor_ape.weight",
                       (256, 4), F16),
                Tensor(prefix + "indexer_compressor_kv.weight",
                       (4096, 256), F16),
                Tensor(prefix + "indexer_compressor_gate.weight",
                       (4096, 256), F16),
                Tensor(prefix + "indexer_compressor_norm.weight", (128,), F32),
            ]
    return tensors


def dspark_static_tensors() -> list[Tensor]:
    tensors: list[Tensor] = []
    for stage in range(3):
        tensors += common_block(f"mtp.{stage}.", router_type=Q8_0)
    tensors += [
        Tensor("mtp.0.main_proj.weight", (12288, 4096), Q8_0),
        Tensor("mtp.0.main_norm.weight", (4096,), F32),
        Tensor("mtp.2.norm.weight", (4096,), F32),
        Tensor("mtp.2.hc_head_base.weight", (4,), F32),
        Tensor("mtp.2.hc_head_fn.weight", (16384, 4), F16),
        Tensor("mtp.2.hc_head_scale.weight", (1,), F32),
        Tensor("mtp.2.markov_head.markov_w1.weight", (256, 129280), Q8_0),
        Tensor("mtp.2.markov_head.markov_w2.weight", (256, 129280), Q8_0),
        Tensor("mtp.2.confidence_head.proj.weight", (4352, 1), Q8_0),
    ]
    assert len(tensors) == 72
    return tensors


def make_store(name: str, layers: int, source_tensors: int, *,
               expert_used: int = 6,
               source_size: int = 1,
               source_sha256: bytes = bytes(32),
               payload_sha256: bytes = bytes(32),
               bad_component_type: bool = False,
               bad_component_dim: bool = False) -> tuple[Tensor, bytes]:
    if len(source_sha256) != 32 or len(payload_sha256) != 32:
        raise ValueError("expert-store SHA-256 fields must contain 32 bytes")
    expert_count = 256
    data_offset = align_up(
        STORE_HEADER_BYTES + layers * STORE_LAYER_BYTES, STORE_ALIGNMENT
    )
    cursor = data_offset
    descriptors = bytearray()
    component_dims = (
        (4096, 2048, expert_count),
        (4096, 2048, expert_count),
        (2048, 4096, expert_count),
    )
    component_types = (IQ2_XXS, IQ2_XXS, Q2_K)
    for layer in range(layers):
        entry = bytearray(STORE_LAYER_BYTES)
        expert_sizes = [
            tensor_bytes(Tensor("", (dims[0], dims[1], 1), value_type))
            for dims, value_type in zip(component_dims, component_types)
        ]
        record_bytes = sum(expert_sizes)
        layer_bytes = record_bytes * expert_count
        struct.pack_into(
            "<IIQQQ", entry, 0,
            layer, expert_count, record_bytes, cursor, layer_bytes,
        )
        record_offset = 0
        for role, (dims, value_type, expert_bytes) in enumerate(zip(
            component_dims, component_types, expert_sizes
        )):
            encoded_type = F16 if bad_component_type and role == 0 \
                else value_type
            encoded_dims = ((dims[0] - 1, dims[1], dims[2])
                            if bad_component_dim and role == 0 else dims)
            struct.pack_into(
                "<IIIIQQQQQ", entry,
                STORE_COMPONENT_OFFSET + role * STORE_COMPONENT_BYTES,
                role, encoded_type, 3, 256,
                encoded_dims[0], encoded_dims[1], encoded_dims[2],
                expert_bytes, record_offset,
            )
            record_offset += expert_bytes
        assert record_bytes == 7_077_888
        descriptors += entry
        cursor += layer_bytes
    store_size = cursor
    header = bytearray(STORE_HEADER_BYTES)
    header[:8] = b"DS4EXPV2"
    struct.pack_into(
        "<IIIIIIQQQQQQQ", header, 8,
        2, STORE_HEADER_BYTES, 1, expert_used, layers, expert_count,
        source_tensors, layers, len(descriptors), STORE_HEADER_BYTES,
        data_offset, store_size - data_offset, store_size,
    )
    struct.pack_into("<Q", header, 88, source_size)
    header[96:128] = source_sha256
    header[128:160] = payload_sha256
    struct.pack_into("<II", header, 160, 0, 0)
    digest_header = bytearray(header)
    digest_header[
        STORE_MANIFEST_DIGEST_OFFSET:STORE_MANIFEST_DIGEST_OFFSET + 32
    ] = bytes(32)
    header[
        STORE_MANIFEST_DIGEST_OFFSET:STORE_MANIFEST_DIGEST_OFFSET + 32
    ] = hashlib.sha256(digest_header + descriptors).digest()
    return Tensor(name, (store_size,), I8), bytes(header + descriptors)


def write_fixture(path: Path, *, combined: bool,
                  metadata_alias: bool = False,
                  metadata_case_alias: bool = False,
                  metadata_stage_count: int = 3,
                  metadata_target_layers: list[int] | None = None,
                  metadata_target_layers_type: int = UINT32,
                  duplicate_metadata_key: str | None = None,
                  provenance_omit: str | None = None,
                  provenance_altered: str | None = None,
                  provenance_bad_type: str | None = None,
                  provenance_case_alias: bool = False,
                  support_expert_used: int = 6,
                  support_source_tensors: int = 81,
                  support_source_size: int = DSPARK_SUPPORT_SOURCE_BYTES,
                  support_bad_source_digest: bool = False,
                  support_bad_payload_digest: bool = False,
                  omit_static: str | None = None,
                  bad_static_name: bool = False,
                  bad_static_shape: bool = False,
                  bad_static_type: bool = False,
                  support_bad_component_type: bool = False,
                  support_bad_component_dim: bool = False,
                  overlap_stores: bool = False,
                  overlap_static_store: bool = False,
                  misaligned_static: bool = False,
                  outside_file_static: bool = False,
                  duplicate_support_store: bool = False,
                  duplicate_static_name: bool = False) -> tuple[int, int]:
    target_static = target_non_routed_tensors()
    target_source_count = len(target_static) + 43 * 3
    target_store, target_manifest = make_store(
        TARGET_STORE, 43, target_source_count
    )
    tensors = list(target_static)
    payloads = [(target_store, target_manifest)]
    if combined:
        support_static = dspark_static_tensors()
        if omit_static is not None:
            support_static = [item for item in support_static
                              if item.name != omit_static]
        if bad_static_shape:
            support_static = [
                Tensor(item.name, (4096, 4096), item.value_type)
                if item.name == "mtp.1.attn_q_a.weight" else item
                for item in support_static
            ]
        if bad_static_name:
            support_static = [
                Tensor("mtp.1.attention_norm.weight", item.dims,
                       item.value_type)
                if item.name == "mtp.1.attn_norm.weight" else item
                for item in support_static
            ]
        if bad_static_type:
            support_static = [
                Tensor(item.name, item.dims, F16)
                if item.name == "mtp.1.attn_norm.weight" else item
                for item in support_static
            ]
        if duplicate_static_name:
            support_static = [
                Tensor("mtp.0.hc_attn_base.weight", item.dims,
                       item.value_type)
                if item.name == "mtp.1.hc_attn_base.weight" else item
                for item in support_static
            ]
        tensors += support_static
        support_store, support_manifest = make_store(
            DSPARK_STORE, 3, support_source_tensors,
            expert_used=support_expert_used,
            source_size=support_source_size,
            source_sha256=(bytes(32) if support_bad_source_digest
                           else DSPARK_SUPPORT_SOURCE_SHA256),
            payload_sha256=(bytes(32) if support_bad_payload_digest
                            else DSPARK_SUPPORT_PAYLOAD_SHA256),
            bad_component_type=support_bad_component_type,
            bad_component_dim=support_bad_component_dim,
        )
        payloads.append((support_store, support_manifest))
    tensors.append(target_store)
    if combined:
        tensors.append(payloads[-1][0])
        if duplicate_support_store:
            tensors.append(payloads[-1][0])

    metadata = deepseek_metadata(
        include_dspark=combined, alias=metadata_alias,
        case_alias=metadata_case_alias,
        stage_count=metadata_stage_count,
        target_layers=metadata_target_layers,
        target_layers_type=metadata_target_layers_type,
        provenance_omit=provenance_omit,
        provenance_altered=provenance_altered,
        provenance_bad_type=provenance_bad_type,
        provenance_case_alias=provenance_case_alias,
    )
    metadata_entries = list(metadata.items())
    if duplicate_metadata_key is not None:
        metadata_entries.append(
            (duplicate_metadata_key, metadata[duplicate_metadata_key])
        )
    header = bytearray(struct.pack(
        "<IIQQ", 0x46554747, 3, len(tensors), len(metadata_entries)
    ))
    for key, (value_type, value) in metadata_entries:
        header += pack_string(key)
        header += struct.pack("<I", value_type)
        header += pack_value(value_type, value)

    offsets: list[int] = []
    cursor = 0
    for tensor in tensors:
        offsets.append(cursor)
        cursor = align_up(cursor + tensor_bytes(tensor), 32)
    target_index = next(
        index for index, tensor in enumerate(tensors)
        if tensor.name == TARGET_STORE
    )
    target_offset = offsets[target_index]
    support_indices = [
        index for index, tensor in enumerate(tensors)
        if tensor.name == DSPARK_STORE
    ]
    if combined:
        if overlap_stores:
            offsets[support_indices[0]] = target_offset + 16_384
        if overlap_static_store:
            static_index = next(
                index for index, tensor in enumerate(tensors)
                if tensor.name == "mtp.1.attn_norm.weight"
            )
            offsets[static_index] = offsets[support_indices[0]]
        if misaligned_static:
            static_index = next(
                index for index, tensor in enumerate(tensors)
                if tensor.name == "mtp.1.attn_norm.weight"
            )
            offsets[static_index] += 1
        if outside_file_static:
            static_index = next(
                index for index, tensor in enumerate(tensors)
                if tensor.name == "mtp.1.attn_norm.weight"
            )
            offsets[static_index] = cursor + 32
    for tensor, offset in zip(tensors, offsets):
        header += pack_string(tensor.name)
        header += struct.pack("<I", len(tensor.dims))
        header += struct.pack("<" + "Q" * len(tensor.dims), *tensor.dims)
        header += struct.pack("<IQ", tensor.value_type, offset)
    header += bytes((32 - len(header) % 32) % 32)
    data_offset = len(header)
    end = cursor
    with path.open("wb") as model:
        model.write(header)
        model.truncate(data_offset + end)
        model.seek(data_offset + target_offset)
        model.write(target_manifest)
        if combined:
            for support_index in support_indices:
                model.seek(data_offset + offsets[support_index])
                model.write(payloads[-1][1])
    return target_source_count, target_source_count + 81


def run(binary: Path, model: Path, lock: Path, *, inspect: bool,
        backend: str = "cpu"
        ) -> subprocess.CompletedProcess[str]:
    command = [str(binary), "-m", str(model), f"--{backend}"]
    command += ["--inspect"] if inspect else ["-p", "admission gate"]
    environment = os.environ.copy()
    environment["DS4_LOCK_FILE"] = str(lock)
    return subprocess.run(
        command, env=environment, text=True, capture_output=True, check=False
    )


def require(condition: bool, message: str,
            result: subprocess.CompletedProcess[str]) -> None:
    if condition:
        return
    raise AssertionError(
        f"{message}\nreturncode={result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} /path/to/ds4", file=sys.stderr)
        return 2
    binary = Path(sys.argv[1]).resolve()
    with tempfile.TemporaryDirectory(prefix="ds4-dspark-admission-") as name:
        root = Path(name)

        target = root / "target-only.gguf"
        target_count, combined_count = write_fixture(target, combined=False)
        target_result = run(binary, target, root / "target.lock", inspect=True)
        require(target_result.returncode == 0,
                "target-only ExpertMajor v2 inspect regressed", target_result)
        target_output = target_result.stdout + target_result.stderr
        require(f", {target_count} tensors" in target_output,
                "target-only logical tensor count changed", target_result)
        require("dspark support:" not in target_output,
                "target-only artifact was misidentified as DSpark", target_result)
        require("dspark capture:" not in target_output and
                "dspark cache-plan:" not in target_output,
                "target-only inspect acquired DSpark runtime state",
                target_result)
        require("SUPPORT descriptor installed" not in target_output,
                "inspect unexpectedly installed a Metal descriptor",
                target_result)
        target.unlink()

        combined = root / "combined.gguf"
        write_fixture(combined, combined=True)
        inspect_result = run(
            binary, combined, root / "combined-inspect.lock", inspect=True
        )
        output = inspect_result.stdout + inspect_result.stderr
        require(inspect_result.returncode == 0,
                "valid combined artifact was not inspectable", inspect_result)
        require("embedded ds4.dspark.expert_major.v2 (inspection-only)" in output,
                "DSpark inspect identity is missing", inspect_result)
        require("stages=3 block_size=5 markov_rank=256" in output,
                "DSpark inspect metadata is missing", inspect_result)
        require("source_tensors=81 static_tensors=72 routed_tensors=9" in output,
                "DSpark inspect inventory is missing", inspect_result)
        require(
            "dspark capture: post-layer-hc-mean width=4096 hc_lanes=4 "
            "taps=40,41,42 device_only=yes" in output,
            "DSpark post-layer capture contract is missing", inspect_result,
        )
        require(
            "dspark cache-plan: target_floor=259 support_floor=19 "
            "combined_floor=278" in output and
            "ownership=separate target_only_extra_records=0 "
            "implementation=fail-closed" in output,
            "DSpark cache floor/quota contract is missing", inspect_result,
        )
        require(f", {combined_count} tensors" in output,
                "nine DSpark routed identities were not expanded", inspect_result)
        require("SUPPORT descriptor installed" not in output,
                "combined inspect unexpectedly installed a Metal descriptor",
                inspect_result)

        inference_result = run(
            binary, combined, root / "combined-open.lock", inspect=False
        )
        inference_output = inference_result.stdout + inference_result.stderr
        require(inference_result.returncode != 0,
                "combined artifact reached inference", inference_result)
        require(
            "embedded DSpark support is inspection-only until its Metal graph "
            "and independently budgeted SSD cache are qualified" in
            inference_output,
            "combined artifact did not fail at the explicit engine gate",
            inference_result,
        )
        metal_inference_result = run(
            binary, combined, root / "combined-metal-open.lock",
            inspect=False, backend="metal",
        )
        metal_inference_output = (
            metal_inference_result.stdout + metal_inference_result.stderr
        )
        require(metal_inference_result.returncode != 0,
                "combined artifact reached Metal inference",
                metal_inference_result)
        if "metal backend requested but it is unavailable" not in \
                metal_inference_output:
            require(
                "embedded DSpark support is inspection-only until its Metal graph "
                "and independently budgeted SSD cache are qualified" in
                metal_inference_output,
                "combined artifact bypassed the gate with --metal",
                metal_inference_result,
            )
        combined.unlink()

        negative_cases = (
            ("metadata-alias", {"metadata_alias": True},
             "non-canonical alias"),
            ("metadata-case-alias", {"metadata_case_alias": True},
             "non-canonical alias"),
            ("provenance-missing", {
                "provenance_omit": "dspark.source.index_sha256"
             }, "source provenance inventory is incomplete"),
            ("provenance-altered", {
                "provenance_altered": "dspark.source.shard47_sha256"
             }, "source provenance does not match the pinned final 0731"),
            ("provenance-type", {
                "provenance_bad_type": "dspark.source.config_sha256"
             }, "source provenance does not match the pinned final 0731"),
            ("provenance-case-alias", {"provenance_case_alias": True},
             "non-canonical alias"),
            ("wrong-stage-count", {"metadata_stage_count": 2},
             "does not match the final 0731 contract"),
            ("wrong-target-layer-type", {
                "metadata_target_layers_type": UINT64
             }, "target-layer metadata is invalid"),
            ("wrong-target-layer-count", {
                "metadata_target_layers": [40, 41]
             }, "target-layer metadata is invalid"),
            ("wrong-target-layer-value", {
                "metadata_target_layers": [40, 41, 43]
             }, "does not match the final 0731 contract"),
            ("duplicate-contract-metadata", {
                "duplicate_metadata_key": "dspark.block_size"
             }, "inventory is incomplete or duplicated"),
            ("duplicate-provenance-metadata", {
                "duplicate_metadata_key": "dspark.source.revision"
             }, "source provenance inventory is incomplete or duplicated"),
            ("wrong-top-k", {"support_expert_used": 5},
             "manifest identity is invalid"),
            ("wrong-source-count", {"support_source_tensors": 82},
             "manifest identity is invalid"),
            ("wrong-source-size", {"support_source_size": 1},
             "manifest identity is invalid"),
            ("wrong-source-digest", {"support_bad_source_digest": True},
             "manifest identity is invalid"),
            ("wrong-payload-digest", {"support_bad_payload_digest": True},
             "manifest identity is invalid"),
            ("missing-static", {
                "omit_static": "mtp.1.attn_norm.weight"
             }, "store does not match GGUF metadata or tensor inventory"),
            ("wrong-static-shape", {"bad_static_shape": True},
             "static tensor contract mismatch"),
            ("wrong-static-type", {"bad_static_type": True},
             "static tensor contract mismatch"),
            ("wrong-static-name", {"bad_static_name": True},
             "static tensor contract mismatch"),
            ("wrong-routed-type", {"support_bad_component_type": True},
             "component geometry is invalid at layer 0 role 0"),
            ("wrong-routed-shape", {"support_bad_component_dim": True},
             "component geometry is invalid at layer 0 role 0"),
            ("overlapping-stores", {"overlap_stores": True},
             "target and DSpark expert stores overlap"),
            ("overlapping-static-store", {"overlap_static_store": True},
             "physical tensor ranges overlap"),
            ("misaligned-static", {"misaligned_static": True},
             "physical tensor range is invalid"),
            ("outside-file-static", {"outside_file_static": True},
             "tensor points outside GGUF file"),
            ("duplicate-support-store", {"duplicate_support_store": True},
             "tensor inventory"),
            ("duplicate-static-name", {"duplicate_static_name": True},
             "duplicate tensor name"),
        )
        for case, options, expected in negative_cases:
            model = root / f"{case}.gguf"
            write_fixture(model, combined=True, **options)
            result = run(binary, model, root / f"{case}.lock", inspect=True)
            require(result.returncode != 0,
                    f"invalid DSpark fixture was accepted: {case}", result)
            require(expected in result.stdout + result.stderr,
                    f"missing fail-closed diagnostic for {case}", result)
            model.unlink()

    print("DSpark embedded-store runtime admission: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
