#!/usr/bin/env python3
"""Generate/check the compact Qwen4Exp Phase-6 sparse-QSA fixture.

Write mode executes the pinned Transformers QSA indexer and eager attention,
then cross-checks both against an independent NumPy transcription.  Check mode
is stdlib-only and verifies the checked-in JSON, C include and provenance byte
for byte.  No checkpoint weights are downloaded or loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import struct
import sys
import urllib.request


HF_REPOSITORY = "Qwen/Qwen3.8-Flash-Next"
HF_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
TRANSFORMERS_COMMIT = "42ca97014c85d71a88ad60d55f08cb9fb4d26e2c"
MODELING_SHA256 = "91e9b1e9c74efe373cd989fe1974a8fa305f4aad43628dbcbd03dac20437814f"
MODELING_URL = (
    "https://raw.githubusercontent.com/huggingface/transformers/"
    f"{TRANSFORMERS_COMMIT}/src/transformers/models/qwen4_exp/"
    "modeling_qwen4_exp.py"
)
PYTHON_VERSION = "3.13.13"
NUMPY_VERSION = "2.4.6"
TORCH_VERSION = "2.9.1"
TRANSFORMERS_VERSION = "5.16.0.dev0"
SEED = 0x51534136  # ASCII "QSA6"
ATOL = 2.0e-5
RTOL = 2.0e-5

# Filled after intentional regeneration.  They make the ordinary offline gate
# independent of NumPy, Torch and Transformers.
GOLDEN_JSON_SHA256 = "02ed2bbdd23f15d795a481b27e596ef442c934ff5fc4763281964370b1dd50cd"
GOLDEN_INC_SHA256 = "daf065acbd7f87cf00185f6a03e39b350816d5726e93708bdf0a5b60b6fa70d6"
ARRAY_PAYLOAD_SHA256 = "d20e1d711a2ae519fd7f2f2ad2ad9a4555b6d56fee994056365f5f81348eb4be"

ANCHOR = 37
VISIBLE = 14
COMPRESSION = 4
GROUP_BUDGET = 2
INDEX_DIM = 8
INDEX_HEADS = 4
N_ROT = 4
THETA = 10_000_000.0
EPSILON = 1.0e-6
ATTN_HEADS = 4
KV_HEADS = 2
KV_HEAD_DIM = 4
SELECTED = GROUP_BUDGET * COMPRESSION + VISIBLE % COMPRESSION

ARRAY_ORDER = (
    "raw_key",
    "query_norm_weight",
    "key_norm_weight",
    "group_position",
    "group_key",
    "index_query",
    "head_dot",
    "score",
    "selected_logical",
    "selected_position",
    "key",
    "value",
    "attention_query",
    "attention_output",
    "attention_weight",
    "tie_score",
    "tie_selected_group",
)


def load_capture_dependencies():
    import numpy as np
    import torch
    import transformers

    versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    expected = {
        "python": PYTHON_VERSION,
        "numpy": NUMPY_VERSION,
        "torch": TORCH_VERSION,
        "transformers": TRANSFORMERS_VERSION,
    }
    if versions != expected:
        raise RuntimeError(f"pinned capture environment required: {versions} != {expected}")
    return np, torch


def f32(np, values):
    return np.asarray(values, dtype=np.float32)


def u32(np, values):
    return np.asarray(values, dtype=np.uint32)


def partial_rope(np, values, positions):
    output = f32(np, values).copy()
    half = N_ROT // 2
    frequency = f32(
        np,
        1.0 / (THETA ** (np.arange(0, N_ROT, 2, dtype=np.float32) / np.float32(N_ROT))),
    )
    for token, position in enumerate(positions):
        angle = f32(np, np.float32(position) * frequency)
        cosine = np.cos(angle, dtype=np.float32)
        sine = np.sin(angle, dtype=np.float32)
        first = output[token, :half].copy()
        second = output[token, half:N_ROT].copy()
        output[token, :half] = f32(np, first * cosine - second * sine)
        output[token, half:N_ROT] = f32(np, second * cosine + first * sine)
    return output


def zero_centered_norm(np, values, weight):
    values = f32(np, values)
    variance = np.mean(values * values, axis=-1, keepdims=True, dtype=np.float32)
    inverse = np.float32(1.0) / np.sqrt(variance + np.float32(EPSILON), dtype=np.float32)
    return f32(np, values * inverse * (np.float32(1.0) + f32(np, weight)))


def independent_fixture(np, inputs):
    raw = inputs["raw_key"]
    group_position = inputs["group_position"]
    pooled = np.mean(
        raw[: (VISIBLE // COMPRESSION) * COMPRESSION].reshape(-1, COMPRESSION, INDEX_DIM),
        axis=1,
        dtype=np.float32,
    )
    group_key = partial_rope(
        np,
        zero_centered_norm(np, pooled, inputs["key_norm_weight"]),
        group_position,
    )
    query_pre = zero_centered_norm(
        np,
        inputs["query_pre"].reshape(VISIBLE, INDEX_HEADS, INDEX_DIM),
        inputs["query_norm_weight"],
    )
    query = partial_rope(
        np,
        query_pre.reshape(-1, INDEX_DIM),
        np.repeat(np.arange(ANCHOR, ANCHOR + VISIBLE, dtype=np.uint32), INDEX_HEADS),
    ).reshape(VISIBLE, INDEX_HEADS, INDEX_DIM)
    index_query = query[-1]
    head_dot = f32(np, group_key @ index_query.T)
    score = f32(
        np,
        np.sum(np.maximum(head_dot, np.float32(0.0)), axis=1, dtype=np.float32)
        / np.float32(math.sqrt(INDEX_DIM)),
    )
    group_id = np.arange(VISIBLE // COMPRESSION, dtype=np.uint32)
    chosen = np.lexsort((group_id, -score))[:GROUP_BUDGET]
    chosen = np.sort(chosen)
    selected_logical = []
    for group in chosen:
        selected_logical.extend(
            range(int(group) * COMPRESSION, (int(group) + 1) * COMPRESSION)
        )
    selected_logical.extend(
        range((VISIBLE // COMPRESSION) * COMPRESSION, VISIBLE)
    )
    selected_logical = u32(np, selected_logical)
    selected_position = u32(np, selected_logical + ANCHOR)

    query_attn = inputs["attention_query"]
    key = inputs["key"][selected_logical]
    value = inputs["value"][selected_logical]
    ratio = ATTN_HEADS // KV_HEADS
    output = np.zeros((ATTN_HEADS, KV_HEAD_DIM), dtype=np.float32)
    weights = np.zeros((ATTN_HEADS, SELECTED), dtype=np.float32)
    scale = np.float32(1.0 / math.sqrt(KV_HEAD_DIM))
    for head in range(ATTN_HEADS):
        kv_head = head // ratio
        logits = f32(np, key[:, kv_head] @ query_attn[head] * scale)
        probability = np.exp(logits - np.max(logits), dtype=np.float32)
        probability = f32(np, probability / np.sum(probability, dtype=np.float32))
        weights[head] = probability
        output[head] = f32(np, probability @ value[:, kv_head])

    tie_score = f32(np, [1.5, 1.5, 0.1])
    tie_selected_group = u32(np, [0, 1])
    return {
        "raw_key": raw,
        "query_norm_weight": inputs["query_norm_weight"],
        "key_norm_weight": inputs["key_norm_weight"],
        "group_position": group_position,
        "group_key": group_key,
        "index_query": index_query,
        "head_dot": head_dot,
        "score": score,
        "selected_logical": selected_logical,
        "selected_position": selected_position,
        "key": inputs["key"],
        "value": inputs["value"],
        "attention_query": query_attn,
        "attention_output": output,
        "attention_weight": weights,
        "tie_score": tie_score,
        "tie_selected_group": tie_selected_group,
    }


def capture_transformers(np, torch, inputs):
    import inspect
    from types import SimpleNamespace

    from transformers.models.qwen4_exp import modeling_qwen4_exp as model

    source_sha256 = hashlib.sha256(
        pathlib.Path(inspect.getfile(model)).read_bytes()
    ).hexdigest()
    if source_sha256 != MODELING_SHA256:
        raise RuntimeError(
            f"pinned Transformers source SHA-256 mismatch: {source_sha256}"
        )
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    config = SimpleNamespace(
        indexer_n_heads=INDEX_HEADS,
        indexer_kv_heads=1,
        indexer_head_dim=INDEX_DIM,
        indexer_budget=GROUP_BUDGET * COMPRESSION,
        indexer_compress_ratio=COMPRESSION,
        hidden_size=(INDEX_HEADS + 1) * INDEX_DIM,
        rms_norm_eps=EPSILON,
    )
    indexer = model.Qwen4ExpTextQSAIndexer(config, layer_idx=0)
    with torch.no_grad():
        indexer.index_qk_proj.weight.copy_(
            torch.eye(config.hidden_size, dtype=torch.float32)
        )
        indexer.q_layernorm.weight.copy_(
            torch.from_numpy(inputs["query_norm_weight"].copy())
        )
        indexer.k_layernorm.weight.copy_(
            torch.from_numpy(inputs["key_norm_weight"].copy())
        )

        hidden = np.concatenate([inputs["query_pre"], inputs["raw_key"]], axis=1)
        hidden_tensor = torch.from_numpy(hidden.copy()).unsqueeze(0)
        position = torch.arange(ANCHOR, ANCHOR + VISIBLE, dtype=torch.float32)
        inverse = 1.0 / (
            THETA ** (torch.arange(0, N_ROT, 2, dtype=torch.float32) / N_ROT)
        )
        frequency = position[:, None] * inverse[None, :]
        cosine = torch.cat([frequency.cos(), frequency.cos()], dim=-1).unsqueeze(0)
        sine = torch.cat([frequency.sin(), frequency.sin()], dim=-1).unsqueeze(0)
        causal = torch.tril(torch.ones(VISIBLE, VISIBLE, dtype=torch.bool))
        mask = indexer(hidden_tensor, (cosine, sine), causal[None, None], None)
        selected_logical = torch.nonzero(mask[0, 0, -1], as_tuple=False).flatten()

        qk = indexer.index_qk_proj(hidden_tensor)
        query_pre, raw_key = torch.split(
            qk, [INDEX_HEADS * INDEX_DIM, INDEX_DIM], dim=-1
        )
        query_pre = query_pre.reshape(1, VISIBLE, INDEX_HEADS, INDEX_DIM)
        query = indexer.q_layernorm(query_pre)
        query = model.apply_rotary_pos_emb(
            query, cos=cosine, sin=sine, unsqueeze_dim=2
        )
        pooled = raw_key.reshape(1, VISIBLE, INDEX_DIM)[0, :12]
        pooled = pooled.reshape(3, COMPRESSION, INDEX_DIM).float().mean(dim=1)
        pooled = indexer.k_layernorm(pooled)
        starts = torch.arange(0, 12, COMPRESSION, dtype=torch.long)
        group_key = model.apply_rotary_pos_emb(
            pooled.unsqueeze(1),
            cos=cosine[0].index_select(0, starts),
            sin=sine[0].index_select(0, starts),
        ).squeeze(1)
        head_dot = torch.matmul(query[0, -1].float(), group_key.float().T).T
        score = torch.relu(head_dot).sum(dim=-1) / math.sqrt(INDEX_DIM)

        selected_key = torch.from_numpy(
            inputs["key"][selected_logical.cpu().numpy()].copy()
        ).permute(1, 0, 2).unsqueeze(0)
        selected_value = torch.from_numpy(
            inputs["value"][selected_logical.cpu().numpy()].copy()
        ).permute(1, 0, 2).unsqueeze(0)
        attention_query = torch.from_numpy(inputs["attention_query"].copy())
        attention_query = attention_query.unsqueeze(0).unsqueeze(2)
        attention_module = SimpleNamespace(
            num_key_value_groups=ATTN_HEADS // KV_HEADS,
            training=False,
        )
        attention_output, attention_weight = model.eager_attention_forward(
            attention_module,
            attention_query,
            selected_key,
            selected_value,
            attention_mask=None,
            scaling=1.0 / math.sqrt(KV_HEAD_DIM),
        )

    return {
        "group_key": group_key.cpu().numpy().astype(np.float32),
        "index_query": query[0, -1].cpu().numpy().astype(np.float32),
        "head_dot": head_dot.cpu().numpy().astype(np.float32),
        "score": score.cpu().numpy().astype(np.float32),
        "selected_logical": selected_logical.cpu().numpy().astype(np.uint32),
        "selected_position": (
            selected_logical + ANCHOR
        ).cpu().numpy().astype(np.uint32),
        "attention_output": attention_output[0, 0].cpu().numpy().astype(np.float32),
        "attention_weight": attention_weight[0, :, 0].cpu().numpy().astype(np.float32),
    }


def build_arrays(np, torch):
    rng = np.random.Generator(np.random.PCG64(SEED))
    inputs = {
        "raw_key": f32(np, rng.uniform(-1.2, 1.2, (VISIBLE, INDEX_DIM))),
        "query_pre": f32(
            np, rng.uniform(-1.0, 1.0, (VISIBLE, INDEX_HEADS * INDEX_DIM))
        ),
        "query_norm_weight": f32(
            np, [0.05, -0.10, 0.15, 0.0, -0.20, 0.25, 0.10, -0.05]
        ),
        "key_norm_weight": f32(
            np, [0.0, 0.20, -0.10, 0.35, -0.25, 0.15, 0.05, -0.30]
        ),
        "group_position": u32(np, [ANCHOR, ANCHOR + 4, ANCHOR + 8]),
        "key": f32(
            np, rng.uniform(-1.1, 1.1, (VISIBLE, KV_HEADS, KV_HEAD_DIM))
        ),
        "value": f32(
            np, rng.uniform(-0.9, 0.9, (VISIBLE, KV_HEADS, KV_HEAD_DIM))
        ),
        "attention_query": f32(
            np, rng.uniform(-1.0, 1.0, (ATTN_HEADS, KV_HEAD_DIM))
        ),
    }
    arrays = independent_fixture(np, inputs)
    upstream = capture_transformers(np, torch, inputs)
    for name, value in upstream.items():
        expected = arrays[name]
        if value.dtype == np.uint32:
            if not np.array_equal(value, expected):
                raise RuntimeError(f"Transformers integer disagreement: {name}")
        elif not np.allclose(value, expected, atol=ATOL, rtol=RTOL):
            difference = float(np.max(np.abs(value - expected)))
            raise RuntimeError(f"Transformers F32 disagreement: {name}, max={difference}")
        arrays[name] = value
    if tuple(arrays) != ARRAY_ORDER:
        raise RuntimeError("fixture array order drift")
    if arrays["selected_logical"].size != SELECTED:
        raise RuntimeError("fixture did not exercise group cut plus raw tail")
    if np.array_equal(
        arrays["score"],
        np.maximum(np.sum(arrays["head_dot"], axis=1), np.float32(0.0))
        / np.float32(math.sqrt(INDEX_DIM)),
    ):
        raise RuntimeError("fixture does not distinguish sum(ReLU(dot_h))")
    return arrays


def array_bytes(array):
    return array.astype(array.dtype.newbyteorder("<"), copy=False).tobytes(order="C")


def array_record(array):
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array_bytes(array)).hexdigest(),
        "data": array.reshape(-1).tolist(),
    }


def render_json(arrays):
    document = {
        "format": "ds4-qwen4exp-qsa-golden-v1",
        "status": "model-free-not-support",
        "tolerance": {"float32_atol": ATOL, "float32_rtol": RTOL},
        "geometry": {
            "anchor": ANCHOR,
            "visible": VISIBLE,
            "compression": COMPRESSION,
            "group_budget": GROUP_BUDGET,
            "index_dim": INDEX_DIM,
            "index_heads": INDEX_HEADS,
            "n_rot": N_ROT,
            "theta": THETA,
            "attention_heads": ATTN_HEADS,
            "kv_heads": KV_HEADS,
            "kv_head_dim": KV_HEAD_DIM,
            "selected_width": SELECTED,
        },
        "arrays": {name: array_record(array) for name, array in arrays.items()},
    }
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def c_float(value):
    return float(value).hex() + "f"


def render_inc(arrays):
    lines = [
        "/* Generated by collect_qsa_reference.py; do not edit. */",
        "#ifndef DS4_QWEN4EXP_QSA_GOLDEN_INC",
        "#define DS4_QWEN4EXP_QSA_GOLDEN_INC",
        "#include <stdint.h>",
        "#define Q4E_QSA6_ATOL 2.0e-5f",
        "#define Q4E_QSA6_RTOL 2.0e-5f",
        "#define Q4E_QSA6_THETA 10000000.0f",
        "#define Q4E_QSA6_EPSILON 1.0e-6f",
        "enum {",
        f"    Q4E_QSA6_ANCHOR = {ANCHOR},",
        f"    Q4E_QSA6_VISIBLE = {VISIBLE},",
        f"    Q4E_QSA6_COMPRESSION = {COMPRESSION},",
        f"    Q4E_QSA6_GROUP_BUDGET = {GROUP_BUDGET},",
        f"    Q4E_QSA6_INDEX_DIM = {INDEX_DIM},",
        f"    Q4E_QSA6_INDEX_HEADS = {INDEX_HEADS},",
        f"    Q4E_QSA6_N_ROT = {N_ROT},",
        f"    Q4E_QSA6_ATTN_HEADS = {ATTN_HEADS},",
        f"    Q4E_QSA6_KV_HEADS = {KV_HEADS},",
        f"    Q4E_QSA6_KV_HEAD_DIM = {KV_HEAD_DIM},",
        f"    Q4E_QSA6_SELECTED = {SELECTED},",
        "};",
        "",
    ]
    for name, array in arrays.items():
        flat = array.reshape(-1)
        c_type = "uint32_t" if str(array.dtype) == "uint32" else "float"
        values = [
            f"{int(value)}u" if c_type == "uint32_t" else c_float(value)
            for value in flat
        ]
        lines.append(f"static const {c_type} q4e_qsa6_{name}[{flat.size}] = {{")
        for start in range(0, len(values), 6):
            lines.append("    " + ", ".join(values[start : start + 6]) + ",")
        lines.extend(["};", ""])
    lines.extend(["#endif", ""])
    return "\n".join(lines)


def render_provenance(arrays, json_text, inc_text):
    upstream = {
        "group_key",
        "index_query",
        "head_dot",
        "score",
        "selected_logical",
        "attention_output",
        "attention_weight",
    }
    controls = {"selected_position", "tie_score", "tie_selected_group"}
    origin = {}
    for name in arrays:
        if name in upstream:
            origin[name] = "pinned-transformers"
        elif name in controls:
            origin[name] = "contract-control"
        else:
            origin[name] = "deterministic-input"
    payload = b"".join(array_bytes(array) for array in arrays.values())
    document = {
        "format": "ds4-qwen4exp-qsa-provenance-v1",
        "status": "model-free-not-support",
        "hf_repository": HF_REPOSITORY,
        "hf_model_revision": HF_REVISION,
        "transformers_commit": TRANSFORMERS_COMMIT,
        "modeling_source_url": MODELING_URL,
        "modeling_source_sha256": MODELING_SHA256,
        "oracle": "pinned Transformers QSA indexer plus eager_attention_forward",
        "independent_cross_check": "explicit NumPy grouping, selection and attention",
        "generator_environment": {
            "python": PYTHON_VERSION,
            "numpy": NUMPY_VERSION,
            "torch": TORCH_VERSION,
            "transformers": TRANSFORMERS_VERSION,
            "device": "CPU",
            "dtype": "float32",
            "seed": SEED,
        },
        "array_origin": origin,
        "array_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "golden_json_sha256": hashlib.sha256(json_text.encode()).hexdigest(),
        "golden_inc_sha256": hashlib.sha256(inc_text.encode()).hexdigest(),
        "arrays": {
            name: hashlib.sha256(array_bytes(array)).hexdigest()
            for name, array in arrays.items()
        },
    }
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def build_outputs():
    np, torch = load_capture_dependencies()
    arrays = build_arrays(np, torch)
    json_text = render_json(arrays)
    inc_text = render_inc(arrays)
    provenance_text = render_provenance(arrays, json_text, inc_text)
    return json_text, inc_text, provenance_text


def verify_source():
    with urllib.request.urlopen(MODELING_URL, timeout=30) as response:
        source = response.read()
    actual = hashlib.sha256(source).hexdigest()
    if actual != MODELING_SHA256:
        raise RuntimeError(f"pinned modeling source digest mismatch: {actual}")


def offline_check(directory):
    json_path = directory / "qwen4exp_qsa_golden.json"
    inc_path = directory / "qwen4exp_qsa_golden.inc"
    provenance_path = directory / "qwen4exp_qsa_provenance.json"
    try:
        json_bytes = json_path.read_bytes()
        inc_bytes = inc_path.read_bytes()
        golden = json.loads(json_bytes)
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"fixture integrity read failed: {error}", file=sys.stderr)
        return 1
    if (
        provenance.get("hf_model_revision") != HF_REVISION
        or provenance.get("transformers_commit") != TRANSFORMERS_COMMIT
        or provenance.get("modeling_source_sha256") != MODELING_SHA256
    ):
        print("fixture provenance pin mismatch", file=sys.stderr)
        return 1
    json_sha256 = hashlib.sha256(json_bytes).hexdigest()
    inc_sha256 = hashlib.sha256(inc_bytes).hexdigest()
    if (
        json_sha256 != GOLDEN_JSON_SHA256
        or inc_sha256 != GOLDEN_INC_SHA256
        or json_sha256 != provenance.get("golden_json_sha256")
        or inc_sha256 != provenance.get("golden_inc_sha256")
    ):
        print("generated fixture file hash mismatch", file=sys.stderr)
        return 1
    records = golden.get("arrays", {})
    if (
        not isinstance(records, dict)
        or tuple(records) != tuple(sorted(ARRAY_ORDER))
        or set(provenance.get("array_origin", {})) != set(ARRAY_ORDER)
    ):
        print("fixture array inventory mismatch", file=sys.stderr)
        return 1
    payload_all = bytearray()
    for name in ARRAY_ORDER:
        record = records[name]
        try:
            dtype = record["dtype"]
            data = record["data"]
            shape = record["shape"]
            count = math.prod(shape)
            if count != len(data):
                raise ValueError("shape/data mismatch")
            if dtype == "float32":
                payload = b"".join(struct.pack("<f", value) for value in data)
            elif dtype == "uint32":
                payload = b"".join(struct.pack("<I", value) for value in data)
            else:
                raise ValueError(dtype)
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            print(f"invalid array record {name}: {error}", file=sys.stderr)
            return 1
        if hashlib.sha256(payload).hexdigest() != record.get("sha256"):
            print(f"array payload hash mismatch: {name}", file=sys.stderr)
            return 1
        if provenance.get("arrays", {}).get(name) != record.get("sha256"):
            print(f"array provenance hash mismatch: {name}", file=sys.stderr)
            return 1
        payload_all.extend(payload)
    if (
        hashlib.sha256(payload_all).hexdigest() != ARRAY_PAYLOAD_SHA256
        or provenance.get("array_payload_sha256") != ARRAY_PAYLOAD_SHA256
    ):
        print("aggregate array payload hash mismatch", file=sys.stderr)
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--verify-source", action="store_true")
    arguments = parser.parse_args()
    directory = pathlib.Path(__file__).resolve().parent
    if arguments.verify_source:
        verify_source()
    if not arguments.write:
        return offline_check(directory)
    json_text, inc_text, provenance_text = build_outputs()
    outputs = {
        directory / "qwen4exp_qsa_golden.json": json_text,
        directory / "qwen4exp_qsa_golden.inc": inc_text,
        directory / "qwen4exp_qsa_provenance.json": provenance_text,
    }
    for path, content in outputs.items():
        path.write_text(content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
