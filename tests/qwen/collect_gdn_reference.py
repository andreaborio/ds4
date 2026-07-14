#!/usr/bin/env python3
"""Collect a tiny deterministic oracle for Qwen3.6 Gated DeltaNet.

The equations follow the official Transformers 5.13.1 fallback implementation
and the V-head tiling performed by the official llama.cpp GGUF converter:
https://github.com/huggingface/transformers/blob/4626421dc6b741a329300682a6408246ee465490/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py
https://github.com/ggml-org/llama.cpp/blob/657e01125aa49577a62a5531fde24cbcc007006d/conversion/qwen.py

This collector intentionally uses only Python's standard library.  It is an
independent numerical oracle for the scalar C reference and future Metal
kernels; it never downloads model weights.
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path


HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "qwen36_gdn_golden.inc"

N_TOKEN = 4
N_CHANNEL = 4
KERNEL = 4
N_KEY_HEAD = 2
N_VALUE_HEAD = 4
KEY_DIM = 2
VALUE_DIM = 2

CONV_INPUT = [
    0.25, -0.50, 0.75, 1.00,
    -1.25, 0.20, 0.40, -0.80,
    0.60, 1.10, -0.30, 0.90,
    -0.45, 0.65, 1.25, -1.05,
]
CONV_WEIGHT = [
    0.10, -0.20, 0.30, 0.40,
    -0.50, 0.25, 0.75, -0.10,
    0.33, -0.66, 0.20, 0.50,
    -0.40, 0.10, -0.30, 0.70,
]

QUERY = [
    0.20, -0.40, 0.60, 0.30,
    -0.30, 0.80, -0.90, 0.20,
    0.70, 0.10, 0.40, -0.60,
    -0.20, 0.50, 0.30, 0.90,
]
KEY = [
    -0.50, 0.60, 0.25, -0.80,
    0.90, -0.20, -0.35, 0.70,
    -0.40, -0.70, 0.85, 0.15,
    0.65, -0.25, -0.75, 0.45,
]
VALUE = [
    0.25, -0.75, 0.50, 0.10, -0.35, 0.40, 0.80, -0.20,
    -0.20, 0.90, 0.30, -0.60, 0.55, 0.15, -0.45, 0.70,
    0.80, -0.40, -0.10, 0.70, 0.20, 0.60, -0.90, 0.05,
    0.40, -0.80, 0.60, 0.20, -0.30, 0.95, 0.15, -0.55,
]
ALPHA_LOGIT = [
    -0.30, 0.20, 25.0, -30.0,
    0.70, -0.50, 1.10, -1.20,
    -0.80, 0.40, 2.00, -2.50,
    -0.10, 0.90, -0.60, 1.70,
]
BETA_LOGIT = [
    -1.20, 1.10, -0.40, 0.80,
    0.30, -0.70, 1.50, -1.00,
    2.20, -0.20, -1.80, 0.55,
    0.20, -1.30, 0.75, -0.45,
]
SSM_A = [-0.50, -1.20, -0.80, -1.50]
DT_BIAS = [0.10, -0.20, 0.40, -0.60]
INITIAL_STATE = [
    0.10, -0.20, 0.30, 0.05,
    -0.15, 0.25, 0.40, -0.35,
    0.20, 0.15, -0.30, 0.45,
    -0.25, -0.10, 0.35, 0.05,
]
GATE = [
    -1.00, 0.50, 1.20, -0.70, 0.40, 1.10, -0.60, 0.20,
    0.00, 2.00, -1.50, 0.30, 0.90, -0.40, 1.30, -1.10,
    0.80, -0.20, 1.50, -2.00, 0.10, 0.70, -0.90, 1.80,
    -0.30, 1.40, 0.60, -0.80, 2.20, -0.10, 0.45, -1.70,
]
NORM_WEIGHT = [0.80, 1.20]
SHARED_INPUT = [1.00, -2.00, 3.00, -0.50, 0.25, 2.00]
SHARED_GATE_LOGIT = [-0.70, 1.20]

N_EXPERT = 256
N_SELECTED = 8


def sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def silu(x: float) -> float:
    return x * sigmoid(x)


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def causal_conv() -> tuple[list[float], list[float]]:
    history = KERNEL - 1
    state = [0.0] * (N_CHANNEL * history)
    output: list[float] = []
    for token in range(N_TOKEN):
        for channel in range(N_CHANNEL):
            state_base = channel * history
            weight_base = channel * KERNEL
            current = CONV_INPUT[token * N_CHANNEL + channel]
            total = current * CONV_WEIGHT[weight_base + KERNEL - 1]
            total += sum(
                state[state_base + k] * CONV_WEIGHT[weight_base + k]
                for k in range(history)
            )
            output.append(silu(total))
            state[state_base : state_base + history - 1] = state[
                state_base + 1 : state_base + history
            ]
            state[state_base + history - 1] = current
    return output, state


def softplus(x: float) -> float:
    if x > 20.0:
        return x
    if x < -20.0:
        return math.exp(x)
    return math.log1p(math.exp(x))


def controls() -> tuple[list[float], list[float]]:
    log_decay: list[float] = []
    beta: list[float] = []
    for token in range(N_TOKEN):
        for head in range(N_VALUE_HEAD):
            index = token * N_VALUE_HEAD + head
            beta.append(sigmoid(BETA_LOGIT[index]))
            log_decay.append(
                SSM_A[head] * softplus(ALPHA_LOGIT[index] + DT_BIAS[head])
            )
    return log_decay, beta


def gated_delta() -> tuple[list[float], list[float]]:
    state = INITIAL_STATE.copy()
    output: list[float] = []
    log_decay, beta = controls()
    q_scale = 1.0 / math.sqrt(KEY_DIM)
    for token in range(N_TOKEN):
        for value_head in range(N_VALUE_HEAD):
            # Official GGUF conversion tiles all V-side quantities.  This
            # modulo mapping intentionally differs from HF repeat_interleave.
            key_head = value_head % N_KEY_HEAD
            q_base = (token * N_KEY_HEAD + key_head) * KEY_DIM
            v_base = (token * N_VALUE_HEAD + value_head) * VALUE_DIM
            s_base = value_head * KEY_DIM * VALUE_DIM
            q = QUERY[q_base : q_base + KEY_DIM]
            k = KEY[q_base : q_base + KEY_DIM]
            q_inv = q_scale / math.sqrt(sum(x * x for x in q) + 1.0e-6)
            k_inv = 1.0 / math.sqrt(sum(x * x for x in k) + 1.0e-6)
            q = [x * q_inv for x in q]
            k = [x * k_inv for x in k]
            decay = math.exp(log_decay[token * N_VALUE_HEAD + value_head])
            for i in range(KEY_DIM * VALUE_DIM):
                state[s_base + i] *= decay
            for j in range(VALUE_DIM):
                memory = sum(
                    state[s_base + j * KEY_DIM + i] * k[i]
                    for i in range(KEY_DIM)
                )
                delta = (VALUE[v_base + j] - memory) * beta[
                    token * N_VALUE_HEAD + value_head
                ]
                for i in range(KEY_DIM):
                    state[s_base + j * KEY_DIM + i] += k[i] * delta
            for j in range(VALUE_DIM):
                output.append(
                    sum(
                        state[s_base + j * KEY_DIM + i] * q[i]
                        for i in range(KEY_DIM)
                    )
                )
    return output, state


def rmsnorm_gated(values: list[float]) -> list[float]:
    output: list[float] = []
    for vector in range(N_TOKEN * N_VALUE_HEAD):
        base = vector * VALUE_DIM
        x = values[base : base + VALUE_DIM]
        inv_rms = 1.0 / math.sqrt(
            sum(v * v for v in x) / VALUE_DIM + 1.0e-6
        )
        for dim in range(VALUE_DIM):
            output.append(
                x[dim]
                * inv_rms
                * NORM_WEIGHT[dim]
                * silu(GATE[base + dim])
            )
    return output


def shared_expert_gate() -> list[float]:
    output: list[float] = []
    dim = len(SHARED_INPUT) // len(SHARED_GATE_LOGIT)
    for vector, gate_logit in enumerate(SHARED_GATE_LOGIT):
        gate = sigmoid(gate_logit)
        output.extend(
            value * gate
            for value in SHARED_INPUT[vector * dim : (vector + 1) * dim]
        )
    return output


def router_reference() -> tuple[list[int], list[float]]:
    logits = [f32(-20.0 - i * 0.001) for i in range(N_EXPERT)]
    for expert, value in {
        201: 3.0,
        7: 2.5,
        88: 2.0,
        42: 1.5,
        111: 1.0,
        3: 0.5,
        17: 0.1,
        19: 0.0,
        23: -0.1,
    }.items():
        logits[expert] = f32(value)
    maximum = max(logits)
    probabilities = [f32(math.exp(f32(x - maximum))) for x in logits]
    total = 0.0
    for probability in probabilities:
        total = f32(total + probability)
    probabilities = [f32(x / total) for x in probabilities]
    selected = sorted(
        range(N_EXPERT), key=lambda i: (-probabilities[i], i)
    )[:N_SELECTED]
    selected_total = 0.0
    for expert in selected:
        selected_total = f32(selected_total + probabilities[expert])
    weights = [f32(probabilities[i] / selected_total) for i in selected]
    return selected, weights


def c_float(value: float) -> str:
    if value == 0.0:
        return "0.0f"
    rendered = f"{value:.9g}"
    if "." not in rendered and "e" not in rendered:
        rendered += ".0"
    return rendered + "f"


def c_array(name: str, values: list[float]) -> str:
    lines = [f"static const float {name}[{len(values)}] = {{"]
    for offset in range(0, len(values), 4):
        row = ", ".join(c_float(v) for v in values[offset : offset + 4])
        lines.append(f"    {row},")
    lines.append("};")
    return "\n".join(lines)


def c_int_array(name: str, values: list[int]) -> str:
    rendered = ", ".join(str(value) for value in values)
    return (
        f"static const int32_t {name}[{len(values)}] = {{\n"
        f"    {rendered},\n"
        "};"
    )


def render() -> str:
    conv_output, conv_state = causal_conv()
    delta_output, delta_state = gated_delta()
    gated_output = rmsnorm_gated(delta_output)
    shared_output = shared_expert_gate()
    router_ids, router_weights = router_reference()
    log_decay, beta = controls()
    arrays = [
        ("qwen_ref_conv_input", CONV_INPUT),
        ("qwen_ref_conv_weight", CONV_WEIGHT),
        ("qwen_ref_conv_output", conv_output),
        ("qwen_ref_conv_state", conv_state),
        ("qwen_ref_query", QUERY),
        ("qwen_ref_key", KEY),
        ("qwen_ref_value", VALUE),
        ("qwen_ref_alpha_logit", ALPHA_LOGIT),
        ("qwen_ref_beta_logit", BETA_LOGIT),
        ("qwen_ref_ssm_a", SSM_A),
        ("qwen_ref_dt_bias", DT_BIAS),
        ("qwen_ref_log_decay", log_decay),
        ("qwen_ref_beta", beta),
        ("qwen_ref_initial_state", INITIAL_STATE),
        ("qwen_ref_delta_output", delta_output),
        ("qwen_ref_delta_state", delta_state),
        ("qwen_ref_gate", GATE),
        ("qwen_ref_norm_weight", NORM_WEIGHT),
        ("qwen_ref_gated_output", gated_output),
        ("qwen_ref_shared_input", SHARED_INPUT),
        ("qwen_ref_shared_gate_logit", SHARED_GATE_LOGIT),
        ("qwen_ref_shared_output", shared_output),
        ("qwen_ref_router_weight", router_weights),
    ]
    header = f"""/* Generated by collect_gdn_reference.py.  Do not edit by hand. */
enum {{
    QWEN_REF_N_TOKEN = {N_TOKEN},
    QWEN_REF_N_CHANNEL = {N_CHANNEL},
    QWEN_REF_KERNEL = {KERNEL},
    QWEN_REF_N_KEY_HEAD = {N_KEY_HEAD},
    QWEN_REF_N_VALUE_HEAD = {N_VALUE_HEAD},
    QWEN_REF_KEY_DIM = {KEY_DIM},
    QWEN_REF_VALUE_DIM = {VALUE_DIM},
}};
"""
    rendered_arrays = [c_array(name, values) for name, values in arrays]
    rendered_arrays.append(c_int_array("qwen_ref_router_id", router_ids))
    return header + "\n" + "\n\n".join(rendered_arrays) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    content = render()
    if args.check:
        if not FIXTURE.exists() or FIXTURE.read_text() != content:
            raise SystemExit(f"{FIXTURE} is stale; refresh it intentionally")
        print("Qwen Gated DeltaNet reference fixture matches collector")
    else:
        print(content, end="")


if __name__ == "__main__":
    main()
