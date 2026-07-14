#!/usr/bin/env python3
"""Collect a deterministic Qwen3.6 full-attention scalar oracle.

The equations and fused Q/gate layout follow the official Transformers source
at commit 4626421dc6b741a329300682a6408246ee465490.  The GGUF norm weights are
post-conversion values: the pinned llama.cpp converter adds one to Qwen's
zero-centred RMSNorm parameters before writing them.

Text-only position ids have identical temporal, height, and width coordinates,
so Qwen's interleaved MRoPE reduces to split-half partial RoPE.  This collector
uses only Python's standard library and never downloads model weights.
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path


HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "qwen36_attention_golden.inc"

N_TOKEN = 3
N_QUERY_HEAD = 4
N_KV_HEAD = 2
HEAD_DIM = 6
N_ROT = 4
ROPE_THETA = 10_000_000.0
EPSILON = 1.0e-6
POSITION = [0, 2, 11]

Q_WEIGHT = [0.80, 1.10, 0.65, 1.25, 0.90, 1.05]
K_WEIGHT = [1.20, 0.75, 1.05, 0.85, 1.15, 0.95]


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def make_vector(count: int, phase: float, bias: float) -> list[float]:
    return [
        f32(0.85 * math.sin((i + 1) * phase) + 0.30 * math.cos((i + 2) * 0.19) + bias)
        for i in range(count)
    ]


QUERY = make_vector(N_TOKEN * N_QUERY_HEAD * HEAD_DIM, 0.37, 0.05)
GATE = make_vector(N_TOKEN * N_QUERY_HEAD * HEAD_DIM, 0.23, -0.15)
KEY = make_vector(N_TOKEN * N_KV_HEAD * HEAD_DIM, 0.41, -0.08)
VALUE = make_vector(N_TOKEN * N_KV_HEAD * HEAD_DIM, 0.29, 0.12)


def fuse_q_gate() -> list[float]:
    projection: list[float] = []
    for token in range(N_TOKEN):
        for head in range(N_QUERY_HEAD):
            base = (token * N_QUERY_HEAD + head) * HEAD_DIM
            projection.extend(QUERY[base : base + HEAD_DIM])
            projection.extend(GATE[base : base + HEAD_DIM])
    return projection


PROJECTION = fuse_q_gate()


def rms_norm(values: list[float], n_head: int, weight: list[float]) -> list[float]:
    output: list[float] = []
    for vector in range(N_TOKEN * n_head):
        base = vector * HEAD_DIM
        x = values[base : base + HEAD_DIM]
        total = 0.0
        for value in x:
            total = f32(total + f32(value * value))
        variance = f32(total / HEAD_DIM)
        inv_rms = f32(1.0 / math.sqrt(f32(variance + EPSILON)))
        for dim, value in enumerate(x):
            output.append(f32(f32(value * inv_rms) * weight[dim]))
    return output


def text_rope(values: list[float], n_head: int) -> list[float]:
    output = values.copy()
    half = N_ROT // 2
    for token, position in enumerate(POSITION):
        for head in range(n_head):
            base = (token * n_head + head) * HEAD_DIM
            for index in range(half):
                exponent = f32(f32(2.0 * index) / N_ROT)
                angle = f32(position / f32(ROPE_THETA**exponent))
                cosine = f32(math.cos(angle))
                sine = f32(math.sin(angle))
                a = output[base + index]
                b = output[base + index + half]
                output[base + index] = f32(f32(a * cosine) - f32(b * sine))
                output[base + index + half] = f32(f32(b * cosine) + f32(a * sine))
    return output


def sigmoid(value: float) -> float:
    if value >= 0.0:
        return f32(1.0 / f32(1.0 + f32(math.exp(-value))))
    exponential = f32(math.exp(value))
    return f32(exponential / f32(1.0 + exponential))


def causal_gqa(query: list[float], key: list[float], value: list[float]) -> list[float]:
    output = [0.0] * (N_TOKEN * N_QUERY_HEAD * HEAD_DIM)
    query_per_kv = N_QUERY_HEAD // N_KV_HEAD
    scale = f32(1.0 / math.sqrt(HEAD_DIM))
    for token in range(N_TOKEN):
        for query_head in range(N_QUERY_HEAD):
            kv_head = query_head // query_per_kv
            q_base = (token * N_QUERY_HEAD + query_head) * HEAD_DIM
            scores: list[float] = []
            for key_token in range(token + 1):
                k_base = (key_token * N_KV_HEAD + kv_head) * HEAD_DIM
                dot = 0.0
                for dim in range(HEAD_DIM):
                    dot = f32(dot + f32(query[q_base + dim] * key[k_base + dim]))
                scores.append(f32(dot * scale))
            maximum = max(scores)
            probability = [f32(math.exp(f32(score - maximum))) for score in scores]
            denominator = 0.0
            for item in probability:
                denominator = f32(denominator + item)
            probability = [f32(item / denominator) for item in probability]

            out_base = (token * N_QUERY_HEAD + query_head) * HEAD_DIM
            for key_token, item in enumerate(probability):
                v_base = (key_token * N_KV_HEAD + kv_head) * HEAD_DIM
                for dim in range(HEAD_DIM):
                    output[out_base + dim] = f32(
                        output[out_base + dim] + f32(item * value[v_base + dim])
                    )
    return output


QUERY_NORM = rms_norm(QUERY, N_QUERY_HEAD, Q_WEIGHT)
KEY_NORM = rms_norm(KEY, N_KV_HEAD, K_WEIGHT)
QUERY_ROPE = text_rope(QUERY_NORM, N_QUERY_HEAD)
KEY_ROPE = text_rope(KEY_NORM, N_KV_HEAD)
ATTENTION = causal_gqa(QUERY_ROPE, KEY_ROPE, VALUE)
GATED = [f32(value * sigmoid(gate)) for value, gate in zip(ATTENTION, GATE)]


def c_float(value: float) -> str:
    if value == 0.0:
        return "0.0f"
    return f"{value:.9g}f"


def emit_float_array(name: str, values: list[float]) -> str:
    lines = [f"static const float {name}[] = {{"]
    for offset in range(0, len(values), 6):
        row = ", ".join(c_float(value) for value in values[offset : offset + 6])
        lines.append(f"    {row},")
    lines.append("};")
    return "\n".join(lines)


def emit_u32_array(name: str, values: list[int]) -> str:
    body = ", ".join(f"{value}u" for value in values)
    return f"static const uint32_t {name}[] = {{{body}}};"


def collect() -> str:
    arrays = [
        emit_u32_array("qwen_attn_position", POSITION),
        emit_float_array("qwen_attn_projection", PROJECTION),
        emit_float_array("qwen_attn_query", QUERY),
        emit_float_array("qwen_attn_gate", GATE),
        emit_float_array("qwen_attn_key", KEY),
        emit_float_array("qwen_attn_value", VALUE),
        emit_float_array("qwen_attn_q_weight", Q_WEIGHT),
        emit_float_array("qwen_attn_k_weight", K_WEIGHT),
        emit_float_array("qwen_attn_query_norm", QUERY_NORM),
        emit_float_array("qwen_attn_key_norm", KEY_NORM),
        emit_float_array("qwen_attn_query_rope", QUERY_ROPE),
        emit_float_array("qwen_attn_key_rope", KEY_ROPE),
        emit_float_array("qwen_attn_output", ATTENTION),
        emit_float_array("qwen_attn_gated", GATED),
    ]
    header = """/* Generated by collect_attention_reference.py. */
enum {
    QWEN_ATTN_N_TOKEN = 3,
    QWEN_ATTN_N_QUERY_HEAD = 4,
    QWEN_ATTN_N_KV_HEAD = 2,
    QWEN_ATTN_HEAD_DIM = 6,
    QWEN_ATTN_N_ROT = 4,
};
#define QWEN_ATTN_ROPE_THETA 10000000.0f
#define QWEN_ATTN_EPSILON 1.0e-6f
"""
    return header + "\n" + "\n\n".join(arrays) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=FIXTURE)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    rendered = collect()
    if args.check:
        if not args.output.exists() or args.output.read_text() != rendered:
            raise SystemExit(f"Qwen attention fixture is stale: {args.output}")
        print("Qwen full-attention reference fixture matches collector")
        return 0

    args.output.write_text(rendered)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
