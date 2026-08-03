#!/usr/bin/env python3
"""Offline fail-closed tests for the final-0731 DSpark support quantizer."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
import subprocess
import tempfile
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "gguf-tools" / "deepseek4-quantize.c"
QUANTS = ROOT / "gguf-tools" / "quants.c"

SHARDS = (
    "model-00046-of-00048.safetensors",
    "model-00047-of-00048.safetensors",
    "model-00048-of-00048.safetensors",
)

COMMON = (
    "attn.attn_sink",
    "attn.kv_norm.weight",
    "attn.q_norm.weight",
    "attn.wkv.scale",
    "attn.wkv.weight",
    "attn.wo_a.scale",
    "attn.wo_a.weight",
    "attn.wo_b.scale",
    "attn.wo_b.weight",
    "attn.wq_a.scale",
    "attn.wq_a.weight",
    "attn.wq_b.scale",
    "attn.wq_b.weight",
    "attn_norm.weight",
    "ffn.gate.bias",
    "ffn.gate.weight",
    "ffn.shared_experts.w1.scale",
    "ffn.shared_experts.w1.weight",
    "ffn.shared_experts.w2.scale",
    "ffn.shared_experts.w2.weight",
    "ffn.shared_experts.w3.scale",
    "ffn.shared_experts.w3.weight",
    "ffn_norm.weight",
    "hc_attn_base",
    "hc_attn_fn",
    "hc_attn_scale",
    "hc_ffn_base",
    "hc_ffn_fn",
    "hc_ffn_scale",
)

STAGE_EXTRAS = {
    0: ("main_norm.weight", "main_proj.scale", "main_proj.weight"),
    1: (),
    2: (
        "confidence_head.proj.weight",
        "hc_head_base",
        "hc_head_fn",
        "hc_head_scale",
        "markov_head.markov_w1.weight",
        "markov_head.markov_w2.weight",
        "norm.weight",
    ),
}

OUTPUT_BLOCK = {
    "hc_attn_base.weight": (0, "24"),
    "hc_attn_fn.weight": (1, "16384x24"),
    "hc_attn_scale.weight": (0, "3"),
    "attn_sinks.weight": (0, "64"),
    "attn_q_a.weight": (8, "4096x1024"),
    "attn_q_a_norm.weight": (0, "1024"),
    "attn_q_b.weight": (8, "1024x32768"),
    "attn_kv.weight": (8, "4096x512"),
    "attn_kv_a_norm.weight": (0, "512"),
    "attn_output_a.weight": (8, "4096x8192"),
    "attn_output_b.weight": (8, "8192x4096"),
    "attn_norm.weight": (0, "4096"),
    "hc_ffn_base.weight": (0, "24"),
    "hc_ffn_fn.weight": (1, "16384x24"),
    "hc_ffn_scale.weight": (0, "3"),
    "ffn_gate_inp.weight": (8, "4096x256"),
    "exp_probs_b.bias": (0, "256"),
    "ffn_norm.weight": (0, "4096"),
    "ffn_gate_exps.weight": (16, "4096x2048x256"),
    "ffn_up_exps.weight": (16, "4096x2048x256"),
    "ffn_down_exps.weight": (10, "2048x4096x256"),
    "ffn_gate_shexp.weight": (8, "4096x2048"),
    "ffn_up_shexp.weight": (8, "4096x2048"),
    "ffn_down_shexp.weight": (8, "2048x4096"),
}

OUTPUT_EXTRAS = {
    0: {
        "main_proj.weight": (8, "12288x4096"),
        "main_norm.weight": (0, "4096"),
    },
    1: {},
    2: {
        "norm.weight": (0, "4096"),
        "hc_head_base.weight": (0, "4"),
        "hc_head_fn.weight": (1, "16384x4"),
        "hc_head_scale.weight": (0, "1"),
        "markov_head.markov_w1.weight": (8, "256x129280"),
        "markov_head.markov_w2.weight": (8, "256x129280"),
        "confidence_head.proj.weight": (8, "4352x1"),
    },
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def exact_weight_map() -> dict[str, str]:
    weights: dict[str, str] = {}
    for stage, shard in enumerate(SHARDS):
        for suffix in COMMON + STAGE_EXTRAS[stage]:
            weights[f"mtp.{stage}.{suffix}"] = shard
        for expert in range(256):
            for part in range(1, 4):
                weights[
                    f"mtp.{stage}.ffn.experts.{expert}.w{part}.weight"
                ] = shard
                weights[
                    f"mtp.{stage}.ffn.experts.{expert}.w{part}.scale"
                ] = shard
    assert len(weights) == 4_705
    return weights


def index_bytes(weights: dict[str, str]) -> bytes:
    return json.dumps(
        {"weight_map": weights}, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def compile_fixture_binary(
        output: Path, config: bytes, index: bytes,
        shard_payloads: tuple[bytes, bytes, bytes], *,
        tiny_geometry: bool = False) -> None:
    macros = {
        "DS4_DSPARK_TEST_CONFIG_SHA256": f'"{sha256(config)}"',
        "DS4_DSPARK_TEST_INDEX_SHA256": f'"{sha256(index)}"',
        "DS4_DSPARK_TEST_CONFIG_BYTES": str(len(config)),
        "DS4_DSPARK_TEST_INDEX_BYTES": str(len(index)),
        "DS4_DSPARK_TEST_INDEX_TENSORS": "4705",
    }
    for number, payload in zip((46, 47, 48), shard_payloads):
        macros[f"DS4_DSPARK_TEST_SHARD{number}_BYTES"] = str(len(payload))
        macros[f"DS4_DSPARK_TEST_SHARD{number}_SHA256"] = \
            f'"{sha256(payload)}"'
    command = [
        "cc", "-O2", "-Wall", "-Wextra", "-std=c11", "-D_GNU_SOURCE",
        "-DDS4_DSPARK_QUANTIZER_TEST_CONTRACT",
        *(["-DDS4_DSPARK_QUANTIZER_TEST_TINY_GEOMETRY"]
          if tiny_geometry else []),
        *(f"-D{key}={value}" for key, value in macros.items()),
        "-I", str(ROOT / "gguf-tools"), "-o", str(output),
        str(SOURCE), str(QUANTS), "-lm", "-pthread",
    ]
    subprocess.run(command, check=True)


def tensor_bytes(dtype: str, shape: tuple[int, ...], fill: int = 1) -> bytes:
    count = 1
    for dim in shape:
        count *= dim
    if dtype == "F32":
        return struct.pack("<f", float(fill)) * count
    if dtype == "F16":
        return struct.pack("<e", float(fill)) * count
    if dtype == "F8_E4M3":
        return bytes([0x38]) * count
    if dtype == "F8_E8M0":
        return bytes([127]) * count
    if dtype == "I8":
        return bytes([0x22]) * count
    raise AssertionError(f"unsupported synthetic dtype: {dtype}")


def add_tensor(tensors: dict[str, tuple[str, tuple[int, ...], bytes]],
               name: str, dtype: str, shape: tuple[int, ...]) -> None:
    tensors[name] = (dtype, shape, tensor_bytes(dtype, shape))


def tiny_stage_tensors(stage: int) -> dict[str, tuple[str, tuple[int, ...], bytes]]:
    prefix = f"mtp.{stage}."
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {}

    for suffix in ("attn.kv_norm.weight", "attn.q_norm.weight",
                   "attn_norm.weight", "ffn_norm.weight"):
        add_tensor(tensors, prefix + suffix, "F32", (128,))
    for suffix in ("attn.attn_sink", "hc_attn_base", "hc_attn_scale",
                   "hc_ffn_base", "hc_ffn_scale"):
        add_tensor(tensors, prefix + suffix, "F32", (2,))
    for suffix in ("hc_attn_fn", "hc_ffn_fn"):
        add_tensor(tensors, prefix + suffix, "F16", (2, 8))
    add_tensor(tensors, prefix + "ffn.gate.bias", "F32", (256,))
    add_tensor(tensors, prefix + "ffn.gate.weight", "F16", (256, 128))

    fp8_shapes = {
        "attn.wkv": (128, 128),
        "attn.wo_a": (128, 128),
        "attn.wo_b": (128, 128),
        "attn.wq_a": (128, 128),
        "attn.wq_b": (128, 128),
        "ffn.shared_experts.w1": (128, 128),
        "ffn.shared_experts.w2": (128, 128),
        "ffn.shared_experts.w3": (128, 128),
    }
    for suffix, shape in fp8_shapes.items():
        add_tensor(tensors, prefix + suffix + ".weight", "F8_E4M3", shape)
        add_tensor(
            tensors, prefix + suffix + ".scale", "F8_E8M0",
            (shape[0] // 128, shape[1] // 128),
        )

    for expert in range(256):
        for part in range(1, 4):
            base = prefix + f"ffn.experts.{expert}.w{part}"
            add_tensor(tensors, base + ".weight", "I8", (1, 128))
            add_tensor(tensors, base + ".scale", "F8_E8M0", (1, 8))

    if stage == 0:
        add_tensor(tensors, prefix + "main_norm.weight", "F32", (128,))
        add_tensor(
            tensors, prefix + "main_proj.weight", "F8_E4M3", (128, 128)
        )
        add_tensor(
            tensors, prefix + "main_proj.scale", "F8_E8M0", (1, 1)
        )
    if stage == 2:
        add_tensor(tensors, prefix + "norm.weight", "F32", (128,))
        add_tensor(tensors, prefix + "hc_head_base", "F32", (2,))
        add_tensor(tensors, prefix + "hc_head_fn", "F16", (2, 8))
        add_tensor(tensors, prefix + "hc_head_scale", "F32", (1,))
        add_tensor(
            tensors, prefix + "markov_head.markov_w1.weight",
            "F16", (4, 32),
        )
        add_tensor(
            tensors, prefix + "markov_head.markov_w2.weight",
            "F16", (4, 32),
        )
        add_tensor(
            tensors, prefix + "confidence_head.proj.weight",
            "F16", (1, 32),
        )
    expected = {0: 1568, 1: 1565, 2: 1572}[stage]
    assert len(tensors) == expected
    return tensors


def safetensors_bytes(
        tensors: dict[str, tuple[str, tuple[int, ...], bytes]]) -> bytes:
    header: dict[str, dict[str, object]] = {}
    payload = bytearray()
    for name in sorted(tensors):
        dtype, shape, data = tensors[name]
        begin = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [begin, len(payload)],
        }
    encoded = json.dumps(
        header, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(encoded)) + encoded + payload


def tiny_shards() -> tuple[bytes, bytes, bytes]:
    return tuple(
        safetensors_bytes(tiny_stage_tensors(stage)) for stage in range(3)
    )  # type: ignore[return-value]


def invoke(binary: Path, hf_dir: Path, *args: str,
           ok: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [str(binary), "--dspark-support-only", "--hf", str(hf_dir), *args],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if ok and result.returncode != 0:
        raise AssertionError(result.stderr)
    if not ok and result.returncode == 0:
        raise AssertionError("invalid DSpark source fixture was accepted")
    return result


def write_source(
        path: Path, config: bytes, index: bytes,
        shard_payloads: tuple[bytes, bytes, bytes], *, shards: bool) -> None:
    path.mkdir()
    (path / "config.json").write_bytes(config)
    (path / "model.safetensors.index.json").write_bytes(index)
    if shards:
        for name, payload in zip(SHARDS, shard_payloads):
            (path / name).write_bytes(payload)


def tiny_output_inventory() -> dict[str, tuple[int, tuple[int, ...]]]:
    block = {
        "hc_attn_base.weight": (0, (2,)),
        "hc_attn_fn.weight": (1, (8, 2)),
        "hc_attn_scale.weight": (0, (2,)),
        "attn_sinks.weight": (0, (2,)),
        "attn_q_a.weight": (8, (128, 128)),
        "attn_q_a_norm.weight": (0, (128,)),
        "attn_q_b.weight": (8, (128, 128)),
        "attn_kv.weight": (8, (128, 128)),
        "attn_kv_a_norm.weight": (0, (128,)),
        "attn_output_a.weight": (8, (128, 128)),
        "attn_output_b.weight": (8, (128, 128)),
        "attn_norm.weight": (0, (128,)),
        "hc_ffn_base.weight": (0, (2,)),
        "hc_ffn_fn.weight": (1, (8, 2)),
        "hc_ffn_scale.weight": (0, (2,)),
        "ffn_gate_inp.weight": (8, (128, 256)),
        "exp_probs_b.bias": (0, (256,)),
        "ffn_norm.weight": (0, (128,)),
        "ffn_gate_exps.weight": (16, (256, 1, 256)),
        "ffn_up_exps.weight": (16, (256, 1, 256)),
        "ffn_down_exps.weight": (10, (256, 1, 256)),
        "ffn_gate_shexp.weight": (8, (128, 128)),
        "ffn_up_shexp.weight": (8, (128, 128)),
        "ffn_down_shexp.weight": (8, (128, 128)),
    }
    extras = {
        0: {
            "main_proj.weight": (8, (128, 128)),
            "main_norm.weight": (0, (128,)),
        },
        1: {},
        2: {
            "norm.weight": (0, (128,)),
            "hc_head_base.weight": (0, (2,)),
            "hc_head_fn.weight": (1, (8, 2)),
            "hc_head_scale.weight": (0, (1,)),
            "markov_head.markov_w1.weight": (8, (32, 4)),
            "markov_head.markov_w2.weight": (8, (32, 4)),
            "confidence_head.proj.weight": (8, (32, 1)),
        },
    }
    result: dict[str, tuple[int, tuple[int, ...]]] = {}
    for stage in range(3):
        for suffix, contract in {**block, **extras[stage]}.items():
            result[f"mtp.{stage}.{suffix}"] = contract
    assert len(result) == 81
    return result


def inspect_support_gguf(path: Path) -> tuple[
        dict[str, object], dict[str, tuple[int, tuple[int, ...]]]]:
    data = path.read_bytes()
    position = 0

    def take(size: int) -> bytes:
        nonlocal position
        result = data[position:position + size]
        assert len(result) == size
        position += size
        return result

    def u32() -> int:
        return struct.unpack("<I", take(4))[0]

    def u64() -> int:
        return struct.unpack("<Q", take(8))[0]

    def string() -> str:
        return take(u64()).decode("utf-8")

    assert take(4) == b"GGUF"
    assert u32() == 3
    n_tensors = u64()
    n_kv = u64()
    assert n_tensors == 81
    assert n_kv == 15
    metadata: dict[str, object] = {}
    for _ in range(n_kv):
        key = string()
        value_type = u32()
        if value_type == 4:
            value: object = u32()
        elif value_type == 8:
            value = string()
        elif value_type == 9:
            assert u32() == 4
            value = tuple(u32() for _ in range(u64()))
        else:
            raise AssertionError(f"unexpected metadata type {value_type}")
        assert key not in metadata
        metadata[key] = value
    tensors: dict[str, tuple[int, tuple[int, ...]]] = {}
    offsets = []
    for _ in range(n_tensors):
        name = string()
        dims = tuple(u64() for _ in range(u32()))
        tensor_type = u32()
        offsets.append(u64())
        tensors[name] = (tensor_type, dims)
    assert len(tensors) == 81
    assert offsets == sorted(offsets) and len(offsets) == len(set(offsets))
    alignment = int(metadata["general.alignment"])
    data_offset = (position + alignment - 1) // alignment * alignment
    assert data[position:data_offset] == bytes(data_offset - position)
    assert data_offset < len(data)
    return metadata, tensors


def wait_for_temp(process: subprocess.Popen[str], output: Path) -> Path:
    temporary = Path(f"{output}.tmp.{process.pid}")
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if temporary.exists():
            return temporary
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(
                f"build exited before creating its temporary file:\n"
                f"{stdout}\n{stderr}"
            )
        time.sleep(0.001)
    process.kill()
    process.communicate()
    raise AssertionError("timed out waiting for DSpark temporary output")


def wait_for_path(process: subprocess.Popen[str], path: Path) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if path.exists():
            return
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(
                f"build exited before gate {path.name}:\n{stdout}\n{stderr}"
            )
        time.sleep(0.001)
    process.kill()
    process.communicate()
    raise AssertionError(f"timed out waiting for gate {path.name}")


def main() -> int:
    config = b'{"fixture":"final-0731-dspark"}\n'
    weights = exact_weight_map()
    index = index_bytes(weights)
    shard_payloads = (
        b"synthetic-final-shard-46\n",
        b"synthetic-final-shard-47\n",
        b"synthetic-final-shard-48\n",
    )
    with tempfile.TemporaryDirectory(
            prefix="ds4-dspark-support-quantizer-") as tmp:
        root = Path(tmp)
        binary = root / "deepseek4-quantize-test"
        compile_fixture_binary(binary, config, index, shard_payloads)

        dry_source = root / "dry-source"
        write_source(
            dry_source, config, index, shard_payloads, shards=False
        )
        dry = invoke(binary, dry_source, "--dry-run")
        assert "n_tensors: 81" in dry.stdout
        assert "approx_file_bytes: 5989114880" in dry.stdout, dry.stdout
        assert "dspark_source_tensors: 4705" in dry.stdout
        assert "dspark_dry_run: OK" in dry.stdout
        expected_output = {}
        for stage in range(3):
            for suffix, geometry in {
                    **OUTPUT_BLOCK, **OUTPUT_EXTRAS[stage]}.items():
                expected_output[f"mtp.{stage}.{suffix}"] = geometry
        actual_output = {}
        for line in dry.stdout.splitlines():
            if not line.startswith("dspark_tensor: "):
                continue
            name, type_text, dims_text = line.removeprefix(
                "dspark_tensor: "
            ).split()
            actual_output[name] = (
                int(type_text.removeprefix("type=")),
                dims_text.removeprefix("dims="),
            )
        assert len(expected_output) == 81
        assert actual_output == expected_output

        checked_source = root / "checked-source"
        write_source(
            checked_source, config, index, shard_payloads, shards=True
        )
        checked = invoke(binary, checked_source, "--check")
        assert "dspark_source_check: OK" in checked.stdout

        # Exercise the complete authenticated writer with real safetensors
        # headers/data and the production tensor/type/inventory contract. Only
        # geometry is reduced by the compile-time test contract.
        build_shards = tiny_shards()
        build_binary = root / "deepseek4-quantize-build-test"
        compile_fixture_binary(
            build_binary, config, index, build_shards, tiny_geometry=True
        )
        build_source = root / "build-source"
        write_source(build_source, config, index, build_shards, shards=True)
        support = root / "support.gguf"
        built = invoke(build_binary, build_source, "--out", str(support))
        assert "dspark_support_status: unqualified-authenticated-composer-input" in \
            built.stdout
        metadata, tensors = inspect_support_gguf(support)
        assert tensors == tiny_output_inventory()
        expected_provenance = {
            "dspark.source.revision": "synthetic-test-contract",
            "dspark.source.config_sha256": sha256(config),
            "dspark.source.index_sha256": sha256(index),
            "dspark.source.shard46_sha256": sha256(build_shards[0]),
            "dspark.source.shard47_sha256": sha256(build_shards[1]),
            "dspark.source.shard48_sha256": sha256(build_shards[2]),
        }
        assert {key: metadata[key] for key in expected_provenance} == \
            expected_provenance
        first_digest = sha256(support.read_bytes())
        assert f"dspark_support_sha256: {first_digest}" in built.stdout
        assert int(built.stdout.split("dspark_support_bytes: ", 1)[1]
                   .splitlines()[0]) == support.stat().st_size

        fallback_output = root / "fclone-unsupported-fallback.gguf"
        fallback_env = dict(os.environ)
        fallback_env["DS4_DSPARK_TEST_FORCE_FCLONE_UNSUPPORTED"] = "1"
        fallback = subprocess.run(
            [str(build_binary), "--dspark-support-only", "--hf",
             str(build_source), "--out", str(fallback_output)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=fallback_env,
        )
        assert fallback.returncode == 0, fallback.stderr
        assert sha256(fallback_output.read_bytes()) == first_digest

        # A successful clone is owned before its pathname is reopened. If the
        # reopen itself fails, cleanup removes that exact clone.
        reopen_failure_output = root / "clone-reopen-failure.gguf"
        reopen_failure_env = dict(os.environ)
        reopen_failure_env[
            "DS4_DSPARK_TEST_FORCE_CLONE_REOPEN_FAILURE"
        ] = "1"
        reopen_failure = subprocess.run(
            [str(build_binary), "--dspark-support-only", "--hf",
             str(build_source), "--out", str(reopen_failure_output)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=reopen_failure_env,
        )
        assert reopen_failure.returncode != 0, reopen_failure.stdout
        assert "open cloned output" in reopen_failure.stderr
        assert not reopen_failure_output.exists()
        assert not list(root.glob("clone-reopen-failure.gguf.tmp.*"))

        # If another writer replaces the pathname between clone-stat and
        # reopen, O_NOFOLLOW rejects it and cleanup preserves the foreign
        # entry because its device/inode no longer identify our clone.
        reopen_race_output = root / "clone-reopen-race.gguf"
        reopen_race_owned = root / "clone-reopen-owned.gguf"
        reopen_race_target = root / "clone-reopen-foreign-target"
        reopen_race_target.write_bytes(b"foreign reopen replacement\n")
        reopen_race_gates = root / "clone-reopen-race-gates"
        reopen_race_gates.mkdir()
        reopen_race_env = dict(os.environ)
        reopen_race_env[
            "DS4_DSPARK_TEST_CLONE_REOPEN_GATE"
        ] = str(reopen_race_gates)
        reopen_race = subprocess.Popen(
            [str(build_binary), "--dspark-support-only", "--hf",
             str(build_source), "--out", str(reopen_race_output)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=reopen_race_env,
        )
        reopen_race_temp = wait_for_temp(reopen_race, reopen_race_output)
        wait_for_path(
            reopen_race,
            reopen_race_gates / "after-clone-stat.ready",
        )
        reopen_race_output.replace(reopen_race_owned)
        reopen_race_output.symlink_to(reopen_race_target.name)
        (reopen_race_gates / "after-clone-stat.continue").touch()
        reopen_race_stdout, reopen_race_stderr = \
            reopen_race.communicate(timeout=30)
        assert reopen_race.returncode != 0, reopen_race_stdout
        assert "open cloned output" in reopen_race_stderr
        assert reopen_race_output.is_symlink()
        assert reopen_race_output.read_bytes() == \
            b"foreign reopen replacement\n"
        assert sha256(reopen_race_owned.read_bytes()) == first_digest
        assert not reopen_race_temp.exists()

        # A pre-existing destination is never clobbered; the tool has no
        # destructive replacement mode.
        rejected = invoke(
            build_binary, build_source, "--out", str(support), ok=False
        )
        assert "output exists" in rejected.stderr
        assert sha256(support.read_bytes()) == first_digest

        # A stale/symlink-like temp candidate is not ours: O_EXCL rejects it
        # and cleanup must not unlink it. The shell gate exposes the PID before
        # exec, so the test can create the exact candidate deterministically.
        stale_temp_output = root / "stale-temp-output.gguf"
        gated = subprocess.Popen(
            ["/bin/sh", "-c",
             'printf "%s\\n" "$$"; IFS= read -r _; exec "$@"',
             "dspark-temp-gate", str(build_binary),
             "--dspark-support-only", "--hf", str(build_source),
             "--out", str(stale_temp_output)],
            text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert gated.stdout is not None
        gated_pid = int(gated.stdout.readline().strip())
        stale_temp = Path(f"{stale_temp_output}.tmp.{gated_pid}")
        stale_temp_target = root / "not-owned-temp-target"
        stale_temp_target.write_bytes(b"not-owned-by-quantizer\n")
        stale_temp.symlink_to(stale_temp_target)
        gated_stdout, gated_stderr = gated.communicate(input="\n", timeout=30)
        assert gated.returncode != 0, gated_stdout
        assert "create exclusive output" in gated_stderr
        assert stale_temp.is_symlink()
        assert stale_temp.read_bytes() == b"not-owned-by-quantizer\n"
        assert not stale_temp_output.exists()

        # Reproduce the race where a destination appears after argument
        # validation. Atomic link installation must preserve the other file.
        race_output = root / "race-output.gguf"
        race_process = subprocess.Popen(
            [str(build_binary), "--dspark-support-only", "--hf",
             str(build_source), "--out", str(race_output)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        race_temp = wait_for_temp(race_process, race_output)
        race_output.write_bytes(b"other-writer\n")
        race_stdout, race_stderr = race_process.communicate(timeout=30)
        assert race_process.returncode != 0, race_stdout
        assert "output appeared during build" in race_stderr
        assert race_output.read_bytes() == b"other-writer\n"
        assert not race_temp.exists()

        # Replacing the completed temporary pathname cannot redirect install:
        # fclonefileat consumes the verified FD. Ownership-aware cleanup leaves
        # the foreign pathname intact.
        substitute_output = root / "temp-substitute.gguf"
        substitute_gates = root / "temp-substitute-gates"
        substitute_gates.mkdir()
        substitute_env = dict(os.environ)
        substitute_env["DS4_DSPARK_TEST_SWAP_GATE"] = str(substitute_gates)
        substitute_process = subprocess.Popen(
            [str(build_binary), "--dspark-support-only", "--hf",
             str(build_source), "--out", str(substitute_output)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=substitute_env,
        )
        substitute_temp = wait_for_temp(substitute_process, substitute_output)
        wait_for_path(
            substitute_process, substitute_gates / "after-open.ready"
        )
        (substitute_gates / "after-open.continue").touch()
        wait_for_path(
            substitute_process,
            substitute_gates / "before-final-auth.ready",
        )
        (substitute_gates / "before-final-auth.continue").touch()
        wait_for_path(
            substitute_process, substitute_gates / "before-install.ready"
        )
        verified_inode = root / "verified-temp-inode.gguf"
        substitute_temp.replace(verified_inode)
        replacement_bytes = bytearray(verified_inode.read_bytes())
        replacement_bytes[-1] ^= 1
        substitute_temp.write_bytes(replacement_bytes)
        (substitute_gates / "before-install.continue").touch()
        wait_for_path(
            substitute_process,
            substitute_gates / "after-installed-open.ready",
        )
        (substitute_gates / "after-installed-open.continue").touch()
        substitute_stdout, substitute_stderr = \
            substitute_process.communicate(timeout=30)
        assert substitute_process.returncode == 0, substitute_stderr
        assert sha256(substitute_output.read_bytes()) == first_digest
        assert substitute_temp.read_bytes() == replacement_bytes
        assert "temporary pathname was replaced; left intact" in \
            substitute_stderr

        # Once the cloned destination FD has a cleanup identity, a concurrent
        # replacement followed by a post-clone verification error must not
        # delete the foreign pathname.
        post_clone_output = root / "post-clone-replacement.gguf"
        post_clone_gates = root / "post-clone-replacement-gates"
        post_clone_gates.mkdir()
        post_clone_env = dict(os.environ)
        post_clone_env["DS4_DSPARK_TEST_SWAP_GATE"] = str(post_clone_gates)
        post_clone_process = subprocess.Popen(
            [str(build_binary), "--dspark-support-only", "--hf",
             str(build_source), "--out", str(post_clone_output)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=post_clone_env,
        )
        post_clone_temp = wait_for_temp(post_clone_process, post_clone_output)
        for phase in ("after-open", "before-final-auth", "before-install"):
            wait_for_path(
                post_clone_process, post_clone_gates / f"{phase}.ready"
            )
            (post_clone_gates / f"{phase}.continue").touch()
        wait_for_path(
            post_clone_process,
            post_clone_gates / "after-installed-open.ready",
        )
        saved_post_clone = root / "post-clone-owned.gguf"
        post_clone_output.replace(saved_post_clone)
        post_clone_output.write_bytes(b"foreign post-clone replacement\n")
        (post_clone_gates / "after-installed-open.continue").touch()
        post_clone_stdout, post_clone_stderr = \
            post_clone_process.communicate(timeout=30)
        assert post_clone_process.returncode != 0, post_clone_stdout
        assert "identity changed unexpectedly" in post_clone_stderr
        assert post_clone_output.read_bytes() == \
            b"foreign post-clone replacement\n"
        assert sha256(saved_post_clone.read_bytes()) == first_digest
        assert not post_clone_temp.exists()

        # Mutating an authenticated shard while it is consumed invalidates the
        # pre/post snapshot, removes the temporary file, and installs nothing.
        changing_source = root / "changing-source"
        shutil.copytree(build_source, changing_source)
        changing_output = root / "changing-output.gguf"
        changing_process = subprocess.Popen(
            [str(build_binary), "--dspark-support-only", "--hf",
             str(changing_source), "--out", str(changing_output)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        changing_temp = wait_for_temp(changing_process, changing_output)
        changing_shard = changing_source / SHARDS[2]
        changed = bytearray(changing_shard.read_bytes())
        changed[-1] ^= 1
        changing_shard.write_bytes(changed)
        changing_stdout, changing_stderr = changing_process.communicate(
            timeout=30
        )
        assert changing_process.returncode != 0, changing_stdout
        assert "SHA-256 mismatch" in changing_stderr or \
            "changed while in use" in changing_stderr
        assert not changing_output.exists()
        assert not changing_temp.exists()

        # Swap a shard path to different, valid bytes for the complete tensor
        # read, then restore the authenticated pathname before final auth. A
        # path-based pre/post scheme would accept the restored name while
        # having quantized the replacement. The fd-stable build must keep
        # reading the originally authenticated inode and match the baseline.
        swap_source = root / "swap-source"
        shutil.copytree(build_source, swap_source)
        swap_output = root / "swap-output.gguf"
        gate_dir = root / "swap-gates"
        gate_dir.mkdir()
        swap_env = dict(os.environ)
        swap_env["DS4_DSPARK_TEST_SWAP_GATE"] = str(gate_dir)
        swap_process = subprocess.Popen(
            [str(build_binary), "--dspark-support-only", "--hf",
             str(swap_source), "--out", str(swap_output)],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=swap_env,
        )
        wait_for_path(swap_process, gate_dir / "after-open.ready")
        shard_path = swap_source / SHARDS[0]
        saved_shard = root / "authenticated-shard-46"
        shard_path.replace(saved_shard)
        replacement = bytearray(build_shards[0])
        replacement[-1] ^= 1
        shard_path.write_bytes(replacement)
        (gate_dir / "after-open.continue").touch()
        wait_for_path(swap_process, gate_dir / "before-final-auth.ready")
        shard_path.unlink()
        saved_shard.replace(shard_path)
        (gate_dir / "before-final-auth.continue").touch()
        wait_for_path(swap_process, gate_dir / "before-install.ready")
        (gate_dir / "before-install.continue").touch()
        wait_for_path(swap_process, gate_dir / "after-installed-open.ready")
        (gate_dir / "after-installed-open.continue").touch()
        swap_stdout, swap_stderr = swap_process.communicate(timeout=30)
        assert swap_process.returncode == 0, swap_stderr
        assert sha256(swap_output.read_bytes()) == first_digest
        assert f"dspark_support_sha256: {first_digest}" in swap_stdout

        # A generation-time parser failure also cleans up the exclusive temp.
        malformed_shards = (b"bad", *build_shards[1:])
        malformed_binary = root / "deepseek4-quantize-malformed-test"
        compile_fixture_binary(
            malformed_binary, config, index, malformed_shards,
            tiny_geometry=True,
        )
        malformed_source = root / "malformed-source"
        write_source(
            malformed_source, config, index, malformed_shards, shards=True
        )
        malformed_output = root / "malformed-output.gguf"
        rejected = invoke(
            malformed_binary, malformed_source,
            "--out", str(malformed_output), ok=False,
        )
        assert "short read while reading safetensors header length" in \
            rejected.stderr
        assert not malformed_output.exists()
        assert not list(root.glob("malformed-output.gguf.tmp.*"))

        stale_config = root / "stale-config"
        shutil.copytree(checked_source, stale_config)
        (stale_config / "config.json").write_bytes(config + b"stale")
        rejected = invoke(binary, stale_config, "--dry-run", ok=False)
        assert "config size mismatch" in rejected.stderr

        stale_index = root / "stale-index"
        shutil.copytree(checked_source, stale_index)
        (stale_index / "model.safetensors.index.json").write_bytes(
            index + b"stale"
        )
        rejected = invoke(binary, stale_index, "--dry-run", ok=False)
        assert "index size mismatch" in rejected.stderr

        stale_shard = root / "stale-shard"
        shutil.copytree(checked_source, stale_shard)
        (stale_shard / SHARDS[1]).write_bytes(shard_payloads[1] + b"stale")
        rejected = invoke(binary, stale_shard, "--check", ok=False)
        assert "shard 47 size mismatch" in rejected.stderr

        missing_shard = root / "missing-shard"
        shutil.copytree(checked_source, missing_shard)
        (missing_shard / SHARDS[2]).unlink()
        rejected = invoke(binary, missing_shard, "--check", ok=False)
        assert "open authenticated input" in rejected.stderr and \
            SHARDS[2] in rejected.stderr

        bad_weights = dict(weights)
        del bad_weights["mtp.1.attn_norm.weight"]
        bad_weights["mtp.1.unexpected.weight"] = SHARDS[1]
        bad_index = index_bytes(bad_weights)
        bad_inventory_binary = root / "bad-inventory-quantizer"
        compile_fixture_binary(
            bad_inventory_binary, config, bad_index, shard_payloads
        )
        bad_inventory = root / "bad-inventory"
        write_source(
            bad_inventory, config, bad_index, shard_payloads, shards=False
        )
        rejected = invoke(
            bad_inventory_binary, bad_inventory, "--dry-run", ok=False
        )
        assert "missing DSpark tensor mtp.1.attn_norm.weight" in rejected.stderr

        bad_route_weights = dict(weights)
        bad_route_weights["mtp.0.attn_norm.weight"] = SHARDS[1]
        bad_route_index = index_bytes(bad_route_weights)
        bad_route_binary = root / "bad-route-quantizer"
        compile_fixture_binary(
            bad_route_binary, config, bad_route_index, shard_payloads
        )
        bad_route = root / "bad-route"
        write_source(
            bad_route, config, bad_route_index, shard_payloads, shards=False
        )
        rejected = invoke(bad_route_binary, bad_route, "--dry-run", ok=False)
        assert "invalid final 0731 DSpark index entry" in rejected.stderr

        alias = invoke(
            binary, checked_source,
            "--out", str(checked_source / "config.json"),
            ok=False,
        )
        assert "output exists" in alias.stderr

        fixed_recipe = invoke(
            binary, checked_source, "--dry-run", "--experts", "q4_k",
            ok=False,
        )
        assert "fixed inventory and quantization recipe" in fixed_recipe.stderr


    print("final-0731 DSpark support quantizer: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
