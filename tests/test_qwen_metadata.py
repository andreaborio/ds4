#!/usr/bin/env python3
"""Exercise qwen35moe parser/dispatch using small synthetic GGUF files."""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import tempfile
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


UINT32 = 4
INT32 = 5
FLOAT32 = 6
BOOL = 7
STRING = 8
ARRAY = 9

TENSOR_F32 = 0
TENSOR_F16 = 1
TENSOR_Q8_0 = 8
TENSOR_Q4_K = 12
TENSOR_Q5_K = 13
TENSOR_Q6_K = 14
TENSOR_IQ2_XS = 17
TENSOR_IQ3_XXS = 18
TENSOR_IQ4_XS = 23
TENSOR_I8 = 24

EXPERT_STORE_TENSOR = "ds4.expert_major.v2"
EXPERT_STORE_V1_TENSOR = "ds4.expert_major.v1"
EXPERT_STORE_HEADER_BYTES = 256
EXPERT_STORE_LAYER_BYTES = 224
EXPERT_STORE_COMPONENT_BYTES = 56
EXPERT_STORE_COMPONENT_OFFSET = 32
EXPERT_STORE_ALIGNMENT = 4096
EXPERT_STORE_MANIFEST_DIGEST_OFFSET = 168

QWEN_CHAT_TEMPLATE_PATH = (
    Path(__file__).with_name("qwen") / "qwen36_chat_template.jinja"
)
# Text fixtures carry the repository newline; the GGUF metadata string does
# not. Strip exactly that terminator so the synthetic file remains byte-exact.
QWEN_CHAT_TEMPLATE = QWEN_CHAT_TEMPLATE_PATH.read_text(
    encoding="utf-8"
).removesuffix("\n")
QWEN_TOKENIZER_FIXTURE_PATH = (
    Path(__file__).with_name("qwen") / "qwen36_tokenizer_fixture.inc"
)


class RepeatedArray:
    def __init__(self, item_type: int, count: int, value: object) -> None:
        self.item_type = item_type
        self.count = count
        self.value = value


class SegmentedArray:
    def __init__(
        self, item_type: int, segments: tuple[tuple[int, object], ...]
    ) -> None:
        self.item_type = item_type
        self.segments = segments


@dataclass(frozen=True)
class Tensor:
    name: str
    dims: tuple[int, ...]
    value_type: int


def pack_string(value: str) -> bytes:
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data


def pack_scalar(value_type: int, value: object) -> bytes:
    if value_type == UINT32:
        return struct.pack("<I", int(value))
    if value_type == INT32:
        return struct.pack("<i", int(value))
    if value_type == FLOAT32:
        return struct.pack("<f", float(value))
    if value_type == BOOL:
        return struct.pack("<B", bool(value))
    if value_type == STRING:
        return pack_string(str(value))
    raise ValueError(f"unsupported scalar type {value_type}")


def pack_value(value_type: int, value: object) -> bytes:
    if value_type != ARRAY:
        return pack_scalar(value_type, value)

    if isinstance(value, RepeatedArray):
        item = pack_scalar(value.item_type, value.value)
        return (
            struct.pack("<IQ", value.item_type, value.count)
            + item * value.count
        )

    if isinstance(value, SegmentedArray):
        count = sum(segment_count for segment_count, _ in value.segments)
        return (
            struct.pack("<IQ", value.item_type, count)
            + b"".join(
                pack_scalar(value.item_type, item) * segment_count
                for segment_count, item in value.segments
            )
        )

    item_type, items = value
    return (
        struct.pack("<IQ", item_type, len(items))
        + b"".join(pack_scalar(item_type, item) for item in items)
    )


def qwen_metadata() -> OrderedDict[str, tuple[int, object]]:
    # These values are pinned to Qwen/Qwen3.6-35B-A3B.  The large tokenizer
    # arrays contain empty test strings: this test validates metadata shape and
    # family dispatch, while tokenizer byte-for-byte goldens live separately.
    return OrderedDict(
        [
            ("general.architecture", (STRING, "qwen35moe")),
            ("general.name", (STRING, "Qwen3.6 35B A3B synthetic")),
            ("qwen35moe.block_count", (UINT32, 40)),
            ("qwen35moe.context_length", (UINT32, 262144)),
            ("qwen35moe.embedding_length", (UINT32, 2048)),
            ("qwen35moe.attention.head_count", (UINT32, 16)),
            ("qwen35moe.attention.head_count_kv", (UINT32, 2)),
            ("qwen35moe.attention.key_length", (UINT32, 256)),
            ("qwen35moe.attention.value_length", (UINT32, 256)),
            ("qwen35moe.attention.layer_norm_rms_epsilon", (FLOAT32, 1.0e-6)),
            ("qwen35moe.rope.dimension_count", (UINT32, 64)),
            ("qwen35moe.rope.dimension_sections", (ARRAY, (INT32, [11, 11, 10, 0]))),
            ("qwen35moe.rope.freq_base", (FLOAT32, 10000000.0)),
            ("qwen35moe.expert_count", (UINT32, 256)),
            ("qwen35moe.expert_used_count", (UINT32, 8)),
            ("qwen35moe.expert_feed_forward_length", (UINT32, 512)),
            ("qwen35moe.expert_shared_feed_forward_length", (UINT32, 512)),
            ("qwen35moe.ssm.conv_kernel", (UINT32, 4)),
            ("qwen35moe.ssm.state_size", (UINT32, 128)),
            ("qwen35moe.ssm.group_count", (UINT32, 16)),
            ("qwen35moe.ssm.time_step_rank", (UINT32, 32)),
            ("qwen35moe.ssm.inner_size", (UINT32, 4096)),
            ("qwen35moe.full_attention_interval", (UINT32, 4)),
            ("tokenizer.ggml.model", (STRING, "gpt2")),
            ("tokenizer.ggml.pre", (STRING, "qwen35")),
            ("tokenizer.ggml.tokens", (ARRAY, RepeatedArray(STRING, 248320, ""))),
            (
                "tokenizer.ggml.token_type",
                (
                    ARRAY,
                    SegmentedArray(
                        INT32,
                        (
                            (248044, 1),
                            (14, 3),
                            (2, 4),
                            (6, 3),
                            (4, 4),
                            (7, 3),
                            (243, 5),
                        ),
                    ),
                ),
            ),
            ("tokenizer.ggml.merges", (ARRAY, RepeatedArray(STRING, 247587, ""))),
            ("tokenizer.ggml.bos_token_id", (UINT32, 248044)),
            ("tokenizer.ggml.padding_token_id", (UINT32, 248055)),
            ("tokenizer.ggml.eos_token_id", (UINT32, 248046)),
            ("tokenizer.ggml.add_bos_token", (BOOL, False)),
            (
                "tokenizer.chat_template",
                (STRING, QWEN_CHAT_TEMPLATE),
            ),
        ]
    )


def qwen_tensors() -> list[Tensor]:
    tensors = [
        Tensor("token_embd.weight", (2048, 248320), TENSOR_Q5_K),
        Tensor("output_norm.weight", (2048,), TENSOR_F32),
        Tensor("output.weight", (2048, 248320), TENSOR_Q4_K),
    ]
    for layer in range(40):
        prefix = f"blk.{layer}."
        routed_gate_type = (
            TENSOR_IQ3_XXS if layer == 1 else TENSOR_IQ2_XS
        )
        routed_down_type = (
            TENSOR_IQ4_XS
            if layer in (1, 34, 38, 39)
            else TENSOR_IQ3_XXS
        )
        shared_gate_type = TENSOR_Q6_K if layer == 1 else TENSOR_Q5_K
        shared_down_type = TENSOR_Q8_0 if layer == 1 else TENSOR_Q6_K
        tensors += [
            Tensor(prefix + "attn_norm.weight", (2048,), TENSOR_F32),
            Tensor(prefix + "post_attention_norm.weight", (2048,), TENSOR_F32),
        ]
        if (layer + 1) % 4 == 0:
            tensors += [
                Tensor(prefix + "attn_q.weight", (2048, 8192), TENSOR_Q5_K),
                Tensor(prefix + "attn_k.weight", (2048, 512), TENSOR_Q6_K),
                Tensor(prefix + "attn_v.weight", (2048, 512), TENSOR_Q6_K),
                Tensor(prefix + "attn_output.weight", (4096, 2048), TENSOR_Q5_K),
                Tensor(prefix + "attn_q_norm.weight", (256,), TENSOR_F32),
                Tensor(prefix + "attn_k_norm.weight", (256,), TENSOR_F32),
            ]
        else:
            recurrent_dense_type = (
                TENSOR_Q6_K if layer == 1 else TENSOR_Q5_K
            )
            tensors += [
                Tensor(
                    prefix + "attn_gate.weight",
                    (2048, 4096),
                    recurrent_dense_type,
                ),
                Tensor(
                    prefix + "attn_qkv.weight",
                    (2048, 8192),
                    recurrent_dense_type,
                ),
                Tensor(prefix + "ssm_a", (32,), TENSOR_F32),
                Tensor(prefix + "ssm_alpha.weight", (2048, 32), TENSOR_F32),
                Tensor(prefix + "ssm_beta.weight", (2048, 32), TENSOR_F32),
                Tensor(prefix + "ssm_conv1d.weight", (4, 8192), TENSOR_F32),
                Tensor(prefix + "ssm_dt.bias", (32,), TENSOR_F32),
                Tensor(prefix + "ssm_norm.weight", (128,), TENSOR_F32),
                Tensor(prefix + "ssm_out.weight", (4096, 2048), TENSOR_Q6_K),
            ]
        tensors += [
            Tensor(prefix + "ffn_gate_inp.weight", (2048, 256), TENSOR_F32),
            Tensor(
                prefix + "ffn_gate_exps.weight",
                (2048, 512, 256),
                routed_gate_type,
            ),
            Tensor(
                prefix + "ffn_up_exps.weight",
                (2048, 512, 256),
                routed_gate_type,
            ),
            Tensor(
                prefix + "ffn_down_exps.weight",
                (512, 2048, 256),
                routed_down_type,
            ),
            Tensor(prefix + "ffn_gate_inp_shexp.weight", (2048,), TENSOR_F32),
            Tensor(
                prefix + "ffn_gate_shexp.weight",
                (2048, 512),
                shared_gate_type,
            ),
            Tensor(
                prefix + "ffn_up_shexp.weight",
                (2048, 512),
                shared_gate_type,
            ),
            Tensor(
                prefix + "ffn_down_shexp.weight",
                (512, 2048),
                shared_down_type,
            ),
        ]
    assert len(tensors) == 733
    return tensors


def tensor_bytes(tensor: Tensor) -> int:
    elements = 1
    for dim in tensor.dims:
        elements *= dim
    block_elems, block_bytes = {
        TENSOR_F32: (1, 4),
        TENSOR_F16: (1, 2),
        TENSOR_Q8_0: (32, 34),
        TENSOR_Q4_K: (256, 144),
        TENSOR_Q5_K: (256, 176),
        TENSOR_Q6_K: (256, 210),
        TENSOR_IQ2_XS: (256, 74),
        TENSOR_IQ3_XXS: (256, 98),
        TENSOR_IQ4_XS: (256, 136),
        TENSOR_I8: (1, 1),
    }[tensor.value_type]
    return ((elements + block_elems - 1) // block_elems) * block_bytes


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def qwen_v2_store() -> tuple[Tensor, bytes]:
    layer_count = 40
    expert_count = 256
    expert_used_count = 8
    source_tensor_count = len(qwen_tensors())
    data_offset = align_up(
        EXPERT_STORE_HEADER_BYTES + layer_count * EXPERT_STORE_LAYER_BYTES,
        EXPERT_STORE_ALIGNMENT,
    )
    cursor = data_offset
    descriptors = bytearray()
    component_dims = (
        (2048, 512, expert_count),
        (2048, 512, expert_count),
        (512, 2048, expert_count),
    )
    for layer in range(layer_count):
        component_types = (
            (
                TENSOR_IQ3_XXS,
                TENSOR_IQ3_XXS,
                TENSOR_IQ4_XS,
            )
            if layer == 1
            else (
                TENSOR_IQ2_XS,
                TENSOR_IQ2_XS,
                TENSOR_IQ4_XS,
            )
            if layer in (34, 38, 39)
            else (
                TENSOR_IQ2_XS,
                TENSOR_IQ2_XS,
                TENSOR_IQ3_XXS,
            )
        )
        cursor = align_up(cursor, EXPERT_STORE_ALIGNMENT)
        entry = bytearray(EXPERT_STORE_LAYER_BYTES)
        record_bytes = 0
        component_bytes: list[int] = []
        for dims, component_type in zip(component_dims, component_types):
            one_expert = Tensor(
                "", (dims[0], dims[1], 1), component_type
            )
            component_bytes.append(tensor_bytes(one_expert))
            record_bytes += component_bytes[-1]
        layer_bytes = record_bytes * expert_count
        struct.pack_into(
            "<IIQQQ", entry, 0,
            layer, expert_count, record_bytes, cursor, layer_bytes,
        )
        record_offset = 0
        for role, (dims, component_type, expert_bytes) in enumerate(
            zip(component_dims, component_types, component_bytes)
        ):
            struct.pack_into(
                "<IIIIQQQQQ", entry,
                EXPERT_STORE_COMPONENT_OFFSET +
                role * EXPERT_STORE_COMPONENT_BYTES,
                role, component_type, 3, 256,
                dims[0], dims[1], dims[2], expert_bytes, record_offset,
            )
            record_offset += expert_bytes
        descriptors += entry
        cursor += layer_bytes

    store_size = cursor
    header = bytearray(EXPERT_STORE_HEADER_BYTES)
    header[:8] = b"DS4EXPV2"
    struct.pack_into(
        "<IIIIIIQQQQQQQ", header, 8,
        2, EXPERT_STORE_HEADER_BYTES, 3, expert_used_count,
        layer_count, expert_count, source_tensor_count, layer_count,
        len(descriptors), EXPERT_STORE_HEADER_BYTES, data_offset,
        store_size - data_offset, store_size,
    )
    struct.pack_into("<Q", header, 88, 1)
    manifest_header = bytearray(header)
    manifest_header[
        EXPERT_STORE_MANIFEST_DIGEST_OFFSET:
        EXPERT_STORE_MANIFEST_DIGEST_OFFSET + 32
    ] = bytes(32)
    digest = hashlib.sha256(manifest_header + descriptors).digest()
    header[
        EXPERT_STORE_MANIFEST_DIGEST_OFFSET:
        EXPERT_STORE_MANIFEST_DIGEST_OFFSET + 32
    ] = digest
    return Tensor(EXPERT_STORE_TENSOR, (store_size,), TENSOR_I8), \
        bytes(header + descriptors)


def qwen_native_tensors(version: int) -> tuple[list[Tensor], dict[str, bytes]]:
    non_routed = [
        tensor for tensor in qwen_tensors()
        if not tensor.name.endswith((
            ".ffn_gate_exps.weight",
            ".ffn_up_exps.weight",
            ".ffn_down_exps.weight",
        ))
    ]
    if version == 2:
        store, manifest = qwen_v2_store()
        return non_routed + [store], {store.name: manifest}
    if version == 1:
        store = Tensor(EXPERT_STORE_V1_TENSOR, (1,), TENSOR_I8)
        return non_routed + [store], {store.name: b"\x00"}
    raise AssertionError(f"unsupported native fixture version: {version}")


def write_gguf(
    path: Path,
    metadata: OrderedDict[str, tuple[int, object]],
    tensors: list[Tensor],
    tensor_payloads: dict[str, bytes] | None = None,
) -> None:
    data = bytearray()
    data += struct.pack("<IIQQ", 0x46554747, 3, len(tensors), len(metadata))
    for key, (value_type, value) in metadata.items():
        data += pack_string(key)
        data += struct.pack("<I", value_type)
        data += pack_value(value_type, value)

    # All synthetic tensors may overlap at relative offset zero: the loader
    # validates their names, types, and dimensions, but --inspect never reads
    # weight bytes.  A sparse tail equal to the largest tensor keeps this
    # model-free test below a megabyte of physical disk use.
    for tensor in tensors:
        data += pack_string(tensor.name)
        data += struct.pack("<I", len(tensor.dims))
        for dim in tensor.dims:
            data += struct.pack("<Q", dim)
        data += struct.pack("<IQ", tensor.value_type, 0)
    data += b"\x00" * ((32 - len(data) % 32) % 32)
    max_tensor_bytes = max(tensor_bytes(tensor) for tensor in tensors)
    with path.open("wb") as model:
        model.write(data)
        model.truncate(len(data) + max_tensor_bytes)
        for name, payload in (tensor_payloads or {}).items():
            tensor = next((item for item in tensors if item.name == name), None)
            if tensor is None or len(payload) > tensor_bytes(tensor):
                raise AssertionError(f"invalid synthetic payload for {name}")
            model.seek(len(data))
            model.write(payload)


def run_ds4(
    binary: Path,
    model: Path,
    lock: Path,
    inspect: bool = True,
    backend: str = "cpu",
    extra_args: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    command = [str(binary), "-m", str(model), f"--{backend}"]
    if inspect:
        command.append("--inspect")
    else:
        command += ["-p", "metadata gate"]
    command += extra_args
    env = os.environ.copy()
    env["DS4_LOCK_FILE"] = str(lock)
    env.pop("DS4_QWEN_EXPERIMENTAL_CPU", None)
    env.pop("DS4_QWEN_EXPERIMENTAL_METAL", None)
    return subprocess.run(command, env=env, text=True, capture_output=True, check=False)


def require(condition: bool, message: str, result: subprocess.CompletedProcess[str]) -> None:
    if condition:
        return
    raise AssertionError(
        f"{message}\nreturncode={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def compiled_backend(binary: Path) -> str:
    result = subprocess.run(
        [str(binary), "--capabilities=json"],
        text=True,
        capture_output=True,
        check=False,
    )
    require(result.returncode == 0, "capability query failed", result)
    try:
        document = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"capability query returned invalid JSON: {exc}") from exc
    backend = document.get("backend")
    if backend not in {"cpu", "metal"}:
        raise AssertionError(f"capability query returned invalid backend: {backend!r}")
    return backend


def check_frozen_reference() -> None:
    template = QWEN_CHAT_TEMPLATE.encode("utf-8")
    assert len(template) == 8057
    assert hashlib.sha256(template).hexdigest() == (
        "55d4931433fe502b794226ee7f4d206a6bdd436ac9f80eb7d8ebb4c639f9ea0c"
    )
    path = Path(__file__).with_name("qwen") / "qwen36_tokenizer_chat_golden.json"
    raw = path.read_bytes()
    golden_sha = hashlib.sha256(raw).hexdigest()
    assert golden_sha == (
        "d71b6e5d2e936c1e204b0cc1baf0945bed063a0aa4637be5e89a2c944281e2f6"
    )
    fixture = QWEN_TOKENIZER_FIXTURE_PATH.read_text(encoding="utf-8")
    fixture_golden_sha = re.search(
        r'^#define QWEN36_TOKENIZER_FIXTURE_GOLDEN_SHA256 "([0-9a-f]{64})"$',
        fixture,
        re.MULTILINE,
    )
    assert fixture_golden_sha is not None
    assert fixture_golden_sha.group(1) == golden_sha
    data = json.loads(raw)
    assert data["source"] == {
        "model": "Qwen/Qwen3.6-35B-A3B",
        "revision": "995ad96eacd98c81ed38be0c5b274b04031597b0",
    }
    assert data["collector"] == {
        "package_versions": {
            "transformers": "5.13.1",
            "tokenizers": "0.22.2",
            "Jinja2": "3.1.6",
            "huggingface-hub": "1.23.0",
        }
    }
    tokenizer = data["tokenizer"]
    assert tokenizer["chat_template_class"] == "Qwen2Tokenizer"
    assert tokenizer["encoding_source"] == (
        "tokenizer.json:qwen35 + tokenizer_config controls"
    )
    assert tokenizer["base_bpe_vocab_size"] == 248044
    assert tokenizer["tokenizer_json_vocab_size"] == 248070
    assert tokenizer["effective_vocab_size"] == 248077
    assert tokenizer["model_vocab_size"] == 248320
    assert tokenizer["unused_model_vocab_slots"] == 243
    assert tokenizer["bos_token_id"] is None
    assert tokenizer["eos_token_id"] == 248046
    assert tokenizer["pad_token_id"] == 248044
    assert tokenizer["special_token_ids"] == {
        "<|endoftext|>": 248044,
        "<|im_start|>": 248045,
        "<|im_end|>": 248046,
        "<|object_ref_start|>": 248047,
        "<|object_ref_end|>": 248048,
        "<|box_start|>": 248049,
        "<|box_end|>": 248050,
        "<|quad_start|>": 248051,
        "<|quad_end|>": 248052,
        "<|vision_start|>": 248053,
        "<|vision_end|>": 248054,
        "<|vision_pad|>": 248055,
        "<|image_pad|>": 248056,
        "<|video_pad|>": 248057,
        "<tool_call>": 248058,
        "</tool_call>": 248059,
        "<|fim_prefix|>": 248060,
        "<|fim_middle|>": 248061,
        "<|fim_suffix|>": 248062,
        "<|fim_pad|>": 248063,
        "<|repo_name|>": 248064,
        "<|file_sep|>": 248065,
        "<tool_response>": 248066,
        "</tool_response>": 248067,
        "<think>": 248068,
        "</think>": 248069,
        "<|audio_start|>": 248070,
        "<|audio_end|>": 248071,
        "<tts_pad>": 248072,
        "<tts_text_bos>": 248073,
        "<tts_text_eod>": 248074,
        "<tts_text_bos_single>": 248075,
        "<|audio_pad|>": 248076,
    }
    assert {case["name"] for case in data["text_vectors"]} == {
        "ascii",
        "italian",
        "cjk",
        "whitespace",
        "digits_and_contractions",
        "source_code",
        "emoji_zwj_and_nfc",
        "leading_combining_mark",
        "fim_specials",
        "thinking_specials",
        "tool_specials",
        "audio_control_tokens",
        "all_control_tokens",
        "literal_controls_as_data",
    }
    chat = {case["name"]: case for case in data["chat_vectors"]}
    assert set(chat) == {
        "plain_thinking",
        "plain_no_thinking",
        "system_and_user",
        "tools_prompt",
        "tool_roundtrip",
        "reasoning_before_last_query_stripped",
        "reasoning_after_last_query_preserved",
        "embedded_think_fallback",
        "typed_tool_arguments",
        "assistant_content_before_tool_call",
        "multiple_tool_calls",
        "grouped_tool_responses",
        "post_tool_new_user_strips_reasoning",
        "preserve_thinking",
        "tool_schema_unicode_and_order",
        "literal_controls_in_user_content_reference",
    }
    assert chat["plain_thinking"]["rendered"].endswith("<think>\n")
    assert "<think>\n\n</think>\n\n" in chat["plain_no_thinking"]["rendered"]
    assert "<function=get_weather>" in chat["tool_roundtrip"]["rendered"]
    assert all(case["token_ids"] for case in data["text_vectors"])
    assert all(case["token_ids"] for case in data["chat_vectors"])
    text = {case["name"]: case for case in data["text_vectors"]}
    assert text["leading_combining_mark"]["token_ids"] == [52033, 87383]
    assert text["audio_control_tokens"]["token_ids"] == [
        248070, 315, 9531, 248071
    ]
    assert text["all_control_tokens"]["token_ids"] == list(
        range(248044, 248077)
    )


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} /path/to/ds4", file=sys.stderr)
        return 2
    binary = Path(sys.argv[1]).resolve()
    backend = compiled_backend(binary)
    check_frozen_reference()

    with tempfile.TemporaryDirectory(prefix="ds4-qwen-metadata-") as tmp_name:
        tmp = Path(tmp_name)

        def check(
            name: str,
            mutate: Callable[[OrderedDict[str, tuple[int, object]]], None] | None,
            expected: str,
            success: bool = False,
            inspect: bool = True,
            tensor_mutate: Callable[[list[Tensor]], None] | None = None,
        ) -> None:
            metadata = qwen_metadata()
            if mutate is not None:
                mutate(metadata)
            tensors = qwen_tensors()
            if tensor_mutate is not None:
                tensor_mutate(tensors)
            model = tmp / f"{name}.gguf"
            write_gguf(model, metadata, tensors)
            result = run_ds4(binary, model, tmp / f"{name}.lock", inspect=inspect)
            model.unlink()
            combined = result.stdout + result.stderr
            require((result.returncode == 0) == success, f"unexpected result for {name}", result)
            require(expected in combined, f"missing diagnostic for {name}: {expected!r}", result)

        check("valid", None, "arch:  qwen35moe", success=True)
        check("valid-summary", None, "experts: count=256 used=8", success=True)
        check(
            "unknown-family",
            lambda m: m.__setitem__("general.architecture", (STRING, "qwen35moe_typo")),
            "unsupported GGUF architecture: qwen35moe_typo",
        )
        check(
            "missing-family",
            lambda m: m.pop("general.architecture"),
            "required string metadata key is missing: general.architecture",
        )
        check(
            "wrong-top-k",
            lambda m: m.__setitem__("qwen35moe.expert_used_count", (UINT32, 6)),
            "expected qwen35moe.expert_used_count=8",
        )
        check(
            "wrong-ssm-state",
            lambda m: m.__setitem__("qwen35moe.ssm.state_size", (UINT32, 64)),
            "expected qwen35moe.ssm.state_size=128",
        )
        check(
            "wrong-rms-epsilon",
            lambda m: m.__setitem__(
                "qwen35moe.attention.layer_norm_rms_epsilon", (FLOAT32, 0.0)
            ),
            "expected qwen35moe.attention.layer_norm_rms_epsilon=9.99999997e-07",
        )
        check(
            "wrong-attention-interval",
            lambda m: m.__setitem__("qwen35moe.full_attention_interval", (UINT32, 3)),
            "expected qwen35moe.full_attention_interval=4",
        )

        def corrupt_first_control_type(
            metadata: OrderedDict[str, tuple[int, object]],
        ) -> None:
            metadata["tokenizer.ggml.token_type"] = (
                ARRAY,
                SegmentedArray(
                    INT32,
                    (
                        (248045, 1),
                        (13, 3),
                        (2, 4),
                        (6, 3),
                        (4, 4),
                        (7, 3),
                        (243, 5),
                    ),
                ),
            )

        check(
            "wrong-first-control-token-type",
            corrupt_first_control_type,
            "expected tokenizer.ggml.token_type[248044]=3",
        )
        check(
            "wrong-chat-template",
            lambda m: m.__setitem__(
                "tokenizer.chat_template", (STRING, QWEN_CHAT_TEMPLATE + "\n")
            ),
            "chat_template does not match the pinned canonical template",
        )

        def add_recurrent_mask(
            metadata: OrderedDict[str, tuple[int, object]], *, valid: bool
        ) -> None:
            mask = [(layer + 1) % 4 != 0 for layer in range(40)]
            if not valid:
                mask[3] = True
            metadata["qwen35moe.attention.recurrent_layers"] = (ARRAY, (BOOL, mask))

        check(
            "valid-recurrent-mask",
            lambda m: add_recurrent_mask(m, valid=True),
            "arch:  qwen35moe",
            success=True,
        )
        check(
            "wrong-recurrent-mask",
            lambda m: add_recurrent_mask(m, valid=False),
            "recurrent_layers[3] does not match the required 3:1",
        )

        def add_mtp(metadata: OrderedDict[str, tuple[int, object]]) -> None:
            metadata["qwen35moe.block_count"] = (UINT32, 41)
            metadata["qwen35moe.nextn_predict_layers"] = (UINT32, 1)

        check("bundled-mtp", add_mtp, "must be converted with --no-mtp")

        def replace_tensor(
            tensors: list[Tensor],
            name: str,
            *,
            dims: tuple[int, ...] | None = None,
            value_type: int | None = None,
            replacement_name: str | None = None,
        ) -> None:
            for index, tensor in enumerate(tensors):
                if tensor.name == name:
                    tensors[index] = Tensor(
                        replacement_name or tensor.name,
                        dims or tensor.dims,
                        tensor.value_type if value_type is None else value_type,
                    )
                    return
            raise AssertionError(f"test tensor not found: {name}")

        check(
            "wrong-full-attention-shape",
            None,
            "blk.3.attn_output.weight has dim[0]=2048, expected 4096",
            tensor_mutate=lambda tensors: replace_tensor(
                tensors, "blk.3.attn_output.weight", dims=(2048, 4096)
            ),
        )
        check(
            "unsupported-community-down-quant",
            None,
            "blk.0.ffn_down_exps.weight has type q5_k, expected iq3_xxs "
            "in the selected Qwen routed layout",
            tensor_mutate=lambda tensors: replace_tensor(
                tensors, "blk.0.ffn_down_exps.weight", value_type=TENSOR_Q5_K
            ),
        )
        check(
            "missing-linear-attention-tensor",
            None,
            "required tensor is missing: blk.0.ssm_conv1d.weight",
            tensor_mutate=lambda tensors: replace_tensor(
                tensors,
                "blk.0.ssm_conv1d.weight",
                replacement_name="blk.0.ssm_conv1d.missing",
            ),
        )
        check(
            "canonical-runtime-closed",
            None,
            "Qwen inference requires a DS4 ExpertMajor v2 GGUF; "
            "canonical and v1 execution are no longer supported",
            inspect=False,
        )

        v1_tensors, v1_payloads = qwen_native_tensors(1)
        v1_model = tmp / "v1-runtime-closed.gguf"
        write_gguf(v1_model, qwen_metadata(), v1_tensors, v1_payloads)
        v1_result = run_ds4(
            binary, v1_model, tmp / "v1-runtime-closed.lock", inspect=False
        )
        v1_model.unlink()
        require(v1_result.returncode != 0,
                "Qwen native v1 inference was accepted", v1_result)
        require(
            "Qwen inference requires a DS4 ExpertMajor v2 GGUF; "
            "canonical and v1 execution are no longer supported" in
            v1_result.stdout + v1_result.stderr,
            "Qwen native v1 did not fail at the v2-only admission gate",
            v1_result,
        )

        v2_tensors, v2_payloads = qwen_native_tensors(2)
        v2_model = tmp / "v2-admission.gguf"
        write_gguf(v2_model, qwen_metadata(), v2_tensors, v2_payloads)
        v2_inspect = run_ds4(
            binary, v2_model, tmp / "v2-inspect.lock", inspect=True
        )
        require(v2_inspect.returncode == 0,
                "Qwen native v2 inspect failed", v2_inspect)
        require("arch:  qwen35moe" in v2_inspect.stdout + v2_inspect.stderr,
                "Qwen native v2 inspect summary is missing", v2_inspect)

        v2_cpu = run_ds4(
            binary, v2_model, tmp / "v2-cpu.lock", inspect=False
        )
        require(v2_cpu.returncode != 0,
                "Qwen native v2 CPU inference was accepted", v2_cpu)
        require(
            "Qwen ExpertMajor v2 inference requires the Metal runtime" in
            v2_cpu.stdout + v2_cpu.stderr,
            "Qwen native v2 CPU did not fail at the Metal admission gate",
            v2_cpu,
        )

        v2_metal = run_ds4(
            binary, v2_model, tmp / "v2-metal.lock", inspect=False,
            backend="metal", extra_args=("--quality",),
        )
        v2_metal_output = v2_metal.stdout + v2_metal.stderr
        require(v2_metal.returncode != 0,
                "Qwen native v2 quality guard was not enforced", v2_metal)
        if backend == "metal":
            require("does not support --quality yet" in v2_metal_output,
                    "Qwen native v2 did not reach Metal option validation",
                    v2_metal)
            require("DS4_QWEN_EXPERIMENTAL_METAL" not in v2_metal_output,
                    "Qwen native v2 still requires the experimental Metal flag",
                    v2_metal)
        else:
            require(
                "metal backend requested but it is unavailable in this build" in
                v2_metal_output,
                "CPU build did not reject the unavailable Metal backend",
                v2_metal,
            )
        v2_model.unlink()

    print("qwen metadata tests: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
