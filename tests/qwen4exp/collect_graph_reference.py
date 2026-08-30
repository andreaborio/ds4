#!/usr/bin/env python3
"""Generate/check the compact Qwen4Exp Phase-5 graph fixture.

Actual pinned Transformers modules provide the captured public arrays.  Two
independent NumPy runs (vectorized and explicit scalar dense loops) must agree
with them before output is written.  Persistent cache arrays that upstream
does not expose are labeled contract controls.  `--check` is entirely offline;
`--verify-source` is the opt-in network/source identity gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
import urllib.request
from dataclasses import dataclass

np = None


HF_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
TRANSFORMERS_COMMIT = "42ca97014c85d71a88ad60d55f08cb9fb4d26e2c"
MODELING_SHA256 = "91e9b1e9c74efe373cd989fe1974a8fa305f4aad43628dbcbd03dac20437814f"
GOLDEN_JSON_SHA256 = "ec24c0a872a21c9a6dedcff265ea604e3332a45bcf5ac3d98d455f4f62a7f150"
GOLDEN_INC_SHA256 = "934bcdd7362a7c5f60e7e9bb8820fdfe2ef0a29522fb9e594af22764c60b1669"
ARRAY_PAYLOAD_SHA256 = "a2307e4a035a031ac62ce8718281b3e410e9185c7bbac8efd7e1a9421b44dfcc"
ARRAY_ORDER = (
    "embedding", "ple_rows", "ple_output", "attention_mixed",
    "attention_output", "router_logits", "route_ids", "route_weights",
    "moe_output", "post_layer_wide", "final_hidden", "final_logits",
    "tokens", "final_wide", "final_activation", "final_gdn_conv",
    "final_gdn_recurrent", "final_qsa_key", "final_qsa_value",
    "final_qsa_raw_index", "final_qsa_position", "final_ple_history",
    "final_ple_conv",
)
MODELING_URL = (
    "https://raw.githubusercontent.com/huggingface/transformers/"
    f"{TRANSFORMERS_COMMIT}/src/transformers/models/qwen4_exp/"
    "modeling_qwen4_exp.py"
)
SEED = 0x4E585134
ATOL = 2.0e-5
RTOL = 2.0e-5

LAYERS = 4
GDN_LAYERS = 3
HIDDEN = 4
STREAMS = 4
WIDE = 16
RANK = 2
CONTEXT = 8
VOCAB = 13
EXPERTS = 512
TOP_K = 10
EXPERT_DIM = 3
GDN_KH = 1
GDN_VH = 3
GDN_KD = 2
GDN_VD = 2
GDN_CHANNELS = 10
QSA_QH = 2
QSA_HD = 2
QSA_IH = 2
QSA_ID = 2
COMPRESS = 4
BLOCK_BUDGET = 2
PLE_HEADS = 16
PLE_STATE = 9
PLE_PAD = 12
TOKENS_LIST = [2, 5, 7, 3, 9, 4]
PLE_PRIMES_LIST = [17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79]
TOKENS = None
PLE_PRIMES = None
PLE_OFFSETS = None
PLE_MULTIPLIERS = (23703573157769, 20109073645365, 8052911324071)


def load_numpy() -> None:
    global np, TOKENS, PLE_PRIMES, PLE_OFFSETS
    import numpy as numpy_module

    if sys.version_info[:2] != (3, 13):
        raise RuntimeError(f"Python 3.13 required, got {sys.version.split()[0]}")
    if numpy_module.__version__ != "2.4.6":
        raise RuntimeError(f"NumPy 2.4.6 required, got {numpy_module.__version__}")
    np = numpy_module
    TOKENS = np.asarray(TOKENS_LIST, dtype=np.uint32)
    PLE_PRIMES = np.asarray(PLE_PRIMES_LIST, dtype=np.uint32)
    PLE_OFFSETS = np.concatenate(
        [np.asarray([0], dtype=np.uint32), np.cumsum(PLE_PRIMES[:-1], dtype=np.uint32)]
    )


def f32(value: object) -> np.float32:
    return np.float32(value)


def splitmix64(value: int) -> int:
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


def weight(role: int, index: int) -> np.float32:
    mixed = SEED ^ ((role * 0xD6E8FEB86659FD93) & ((1 << 64) - 1))
    mixed ^= (index * 0xA0761D6478BD642F) & ((1 << 64) - 1)
    bits = splitmix64(mixed)
    signed = int((bits >> 40) & 0xFFFF) - 32768
    return f32(f32(signed) * f32(0.12 / 32768.0))


def vector(role: int, count: int) -> np.ndarray:
    return np.asarray([weight(role, index) for index in range(count)], dtype=np.float32)


def matrix(role: int, rows: int, columns: int) -> np.ndarray:
    return vector(role, rows * columns).reshape(rows, columns)


def dense(role: int, source: np.ndarray, rows: int, scalar: bool) -> np.ndarray:
    source = np.asarray(source, dtype=np.float32).reshape(-1)
    values = matrix(role, rows, source.size)
    if not scalar:
        return np.asarray(values @ source, dtype=np.float32)
    output = np.zeros(rows, dtype=np.float32)
    for row in range(rows):
        total = f32(0.0)
        for column in range(source.size):
            total = f32(total + f32(values[row, column] * source[column]))
        output[row] = total
    return output


def sigmoid(value: object) -> np.ndarray:
    x = np.asarray(value, dtype=np.float32)
    positive = x >= 0
    result = np.empty_like(x)
    result[positive] = f32(1.0) / (f32(1.0) + np.exp(-x[positive], dtype=np.float32))
    exponential = np.exp(x[~positive], dtype=np.float32)
    result[~positive] = exponential / (f32(1.0) + exponential)
    return result


def silu(value: object) -> np.ndarray:
    x = np.asarray(value, dtype=np.float32)
    return np.asarray(x * sigmoid(x), dtype=np.float32)


def softplus(value: object) -> np.ndarray:
    x = np.asarray(value, dtype=np.float32)
    return np.where(
        x > f32(20.0),
        x,
        np.where(x < f32(-20.0), np.exp(x), np.log1p(np.exp(x))),
    ).astype(np.float32)


def zcrms(value: np.ndarray, norm_role: int, group: int) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32).reshape(-1, group)
    norm_weight = vector(norm_role, value.size).reshape(value.shape)
    mean_square = np.mean(value * value, axis=1, keepdims=True, dtype=np.float32)
    inverse = f32(1.0) / np.sqrt(mean_square + f32(1.0e-6), dtype=np.float32)
    return np.asarray(value * inverse * (f32(1.0) + norm_weight), dtype=np.float32).reshape(-1)


def zcrms_shared(value: np.ndarray, norm_role: int, group: int) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32).reshape(-1, group)
    norm_weight = vector(norm_role, group)
    mean_square = np.mean(value * value, axis=1, keepdims=True, dtype=np.float32)
    inverse = f32(1.0) / np.sqrt(mean_square + f32(1.0e-6), dtype=np.float32)
    return np.asarray(value * inverse * (f32(1.0) + norm_weight), dtype=np.float32).reshape(-1)


def gr_roles(layer: int, phase: int) -> tuple[int, int, int, int]:
    base = 1000 + layer * 100 + phase * 10
    return base, base + 1, base + 2, base + 3


def gr_prepare(wide: np.ndarray, layer: int, phase: int, scalar: bool) -> tuple[np.ndarray, np.ndarray]:
    norm_role, down_role, up_role, inject_role = gr_roles(layer, phase)
    normalized = zcrms(wide, norm_role, HIDDEN)
    hidden = silu(dense(down_role, normalized, RANK, scalar) / f32(STREAMS))
    gates = sigmoid(dense(up_role, hidden, WIDE, scalar)).reshape(STREAMS, HIDDEN)
    mixed = np.mean(gates * normalized.reshape(STREAMS, HIDDEN), axis=0, dtype=np.float32)
    injection = f32(2.0) * sigmoid(
        dense(inject_role, normalized, STREAMS, scalar) / f32(STREAMS)
    )
    return mixed.astype(np.float32), injection.astype(np.float32)


def gr_apply(wide: np.ndarray, block: np.ndarray, injection: np.ndarray) -> np.ndarray:
    return np.asarray(
        wide.reshape(STREAMS, HIDDEN) + injection[:, None] * block[None, :],
        dtype=np.float32,
    ).reshape(-1)


def conv_step(
    source: np.ndarray, state: np.ndarray, role: int, kernel: int, dilation: int
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source, dtype=np.float32).reshape(-1)
    state = np.asarray(state, dtype=np.float32).copy().reshape(source.size, dilation * (kernel - 1))
    conv_weight = matrix(role, source.size, kernel)
    output = np.zeros(source.size, dtype=np.float32)
    for channel in range(source.size):
        total = f32(source[channel] * conv_weight[channel, kernel - 1])
        for tap in range(kernel - 1):
            total = f32(total + f32(state[channel, tap * dilation] * conv_weight[channel, tap]))
        output[channel] = silu(np.asarray([total], dtype=np.float32))[0]
        state[channel, :-1] = state[channel, 1:]
        state[channel, -1] = source[channel]
    return output, state.reshape(-1)


def ple_rows(token: int, history: list[int]) -> np.ndarray:
    previous1 = history[0] if len(history) >= 1 else PLE_PAD
    previous2 = history[1] if len(history) >= 2 else PLE_PAD
    bigram = ((token * PLE_MULTIPLIERS[0]) ^ (previous1 * PLE_MULTIPLIERS[1])) & ((1 << 64) - 1)
    trigram = (bigram ^ (previous2 * PLE_MULTIPLIERS[2])) & ((1 << 64) - 1)
    return np.asarray(
        [PLE_OFFSETS[head] + (bigram if head < 8 else trigram) % int(PLE_PRIMES[head]) for head in range(PLE_HEADS)],
        dtype=np.uint32,
    )


def ple_step(
    wide: np.ndarray,
    token: int,
    history: list[int],
    conv_state: np.ndarray,
    scalar: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], np.ndarray]:
    rows = ple_rows(token, history)
    embedding = np.asarray([weight(2000, int(row)) for row in rows], dtype=np.float32)
    key = zcrms(dense(2001, embedding, WIDE, scalar), 2003, HIDDEN).reshape(STREAMS, HIDDEN)
    value = dense(2002, embedding, HIDDEN, scalar)
    query = zcrms(wide, 2004, HIDDEN).reshape(STREAMS, HIDDEN)
    gate_dot = np.sum(key * query, axis=1, dtype=np.float32) / f32(math.sqrt(HIDDEN))
    signed_root = np.sign(gate_dot) * np.sqrt(np.maximum(np.abs(gate_dot), f32(1.0e-6)), dtype=np.float32)
    gated = (sigmoid(signed_root)[:, None] * value[None, :]).astype(np.float32).reshape(-1)
    gated_norm = zcrms(gated, 2005, HIDDEN)
    conv, next_conv = conv_step(gated_norm, conv_state, 2006, 4, 3)
    output = np.asarray(gated + conv, dtype=np.float32)
    if token == PLE_PAD:
        next_history: list[int] = []
    else:
        next_history = [token] + history[:1]
    return wide + output, output, rows, next_history, next_conv


def gdn_step(
    activation: np.ndarray,
    layer: int,
    conv_state: np.ndarray,
    recurrent: np.ndarray,
    scalar: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = 3000 + layer * 20
    qkv = dense(base, activation, GDN_CHANNELS, scalar)
    z = dense(base + 1, activation, GDN_VH * GDN_VD, scalar)
    b = dense(base + 2, activation, GDN_VH, scalar)
    a = dense(base + 3, activation, GDN_VH, scalar)
    conv, next_conv = conv_step(qkv, conv_state, base + 4, 4, 1)
    query = conv[:2].reshape(GDN_KH, GDN_KD)
    key = conv[2:4].reshape(GDN_KH, GDN_KD)
    value = conv[4:].reshape(GDN_VH, GDN_VD)
    a_log = vector(base + 5, GDN_VH)
    dt_bias = vector(base + 6, GDN_VH)
    log_decay = -np.exp(a_log, dtype=np.float32) * softplus(a + dt_bias)
    beta = sigmoid(b)
    next_recurrent = recurrent.copy().reshape(GDN_VH, GDN_KD, GDN_VD)
    core = np.zeros((GDN_VH, GDN_VD), dtype=np.float32)
    for head in range(GDN_VH):
        qhat = query[0] / np.sqrt(np.sum(query[0] * query[0], dtype=np.float32) + f32(1.0e-6), dtype=np.float32)
        qhat = qhat / f32(math.sqrt(GDN_KD))
        khat = key[0] / np.sqrt(np.sum(key[0] * key[0], dtype=np.float32) + f32(1.0e-6), dtype=np.float32)
        next_recurrent[head] *= np.exp(log_decay[head], dtype=np.float32)
        prediction = np.asarray(khat @ next_recurrent[head], dtype=np.float32)
        delta = (value[head] - prediction) * beta[head]
        next_recurrent[head] += khat[:, None] * delta[None, :]
        core[head] = np.asarray(qhat @ next_recurrent[head], dtype=np.float32)
    norm_weight = f32(1.0) + vector(base + 7, GDN_VD)
    variance = np.mean(core * core, axis=1, keepdims=True, dtype=np.float32)
    gated = core / np.sqrt(variance + f32(1.0e-6), dtype=np.float32)
    gated = gated * norm_weight[None, :] * sigmoid(z.reshape(GDN_VH, GDN_VD))
    block = dense(base + 8, gated.reshape(-1), HIDDEN, scalar)
    return block, next_conv, next_recurrent.reshape(-1)


def rope(values: np.ndarray, position: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).copy().reshape(-1, 2)
    angle = f32(position)
    cosine = f32(math.cos(float(angle)))
    sine = f32(math.sin(float(angle)))
    first = values[:, 0].copy()
    second = values[:, 1].copy()
    values[:, 0] = first * cosine - second * sine
    values[:, 1] = second * cosine + first * sine
    return values.reshape(-1)


def qsa_step(
    activation: np.ndarray,
    position: int,
    qsa_key: np.ndarray,
    qsa_value: np.ndarray,
    raw_index: np.ndarray,
    scalar: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base = 4000
    query = zcrms_shared(dense(base, activation, QSA_QH * QSA_HD, scalar), base + 10, QSA_HD)
    query = rope(query, position).reshape(QSA_QH, QSA_HD)
    gate_logit = dense(base + 3, activation, QSA_QH * QSA_HD, scalar)
    key = zcrms_shared(dense(base + 1, activation, QSA_HD, scalar), base + 11, QSA_HD)
    key = rope(key, position)
    value = dense(base + 2, activation, QSA_HD, scalar)
    index_query = zcrms_shared(dense(base + 4, activation, QSA_IH * QSA_ID, scalar), base + 6, QSA_ID)
    index_query = rope(index_query, position).reshape(QSA_IH, QSA_ID)
    raw = dense(base + 5, activation, QSA_ID, scalar)
    next_key = qsa_key.copy().reshape(CONTEXT, QSA_HD)
    next_value = qsa_value.copy().reshape(CONTEXT, QSA_HD)
    next_raw = raw_index.copy().reshape(CONTEXT, QSA_ID)
    next_key[position] = key
    next_value[position] = value
    next_raw[position] = raw

    visible = position + 1
    complete = visible // COMPRESS
    scores = np.zeros(complete, dtype=np.float32)
    if complete:
        groups = next_raw[: complete * COMPRESS].reshape(complete, COMPRESS, QSA_ID).mean(axis=1, dtype=np.float32)
        groups = zcrms_shared(groups.reshape(-1), base + 7, QSA_ID).reshape(complete, QSA_ID)
        for group in range(complete):
            groups[group] = rope(groups[group], group * COMPRESS)
            dots = index_query @ groups[group]
            scores[group] = np.sum(np.maximum(dots, f32(0.0)), dtype=np.float32) / f32(math.sqrt(QSA_ID))
        selected_group = sorted(range(complete), key=lambda item: (-float(scores[item]), item))[:BLOCK_BUDGET]
    else:
        selected_group = []
    selected = [group * COMPRESS + offset for group in selected_group for offset in range(COMPRESS)]
    selected.extend(range(complete * COMPRESS, visible))

    core = np.zeros((QSA_QH, QSA_HD), dtype=np.float32)
    scale = f32(1.0 / math.sqrt(QSA_HD))
    for head in range(QSA_QH):
        logits = np.asarray([np.dot(query[head], next_key[item]) * scale for item in selected], dtype=np.float32)
        probabilities = np.exp(logits - np.max(logits), dtype=np.float32)
        probabilities /= np.sum(probabilities, dtype=np.float32)
        for index, item in enumerate(selected):
            core[head] += probabilities[index] * next_value[item]
    gated = core.reshape(-1) * sigmoid(gate_logit)
    block = dense(base + 9, gated, HIDDEN, scalar)
    return block, next_key.reshape(-1), next_value.reshape(-1), next_raw.reshape(-1)


def router(activation: np.ndarray, layer: int, scalar: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits = dense(5000 + layer, activation, EXPERTS, scalar)
    maximum = np.max(logits)
    probability = np.exp(logits - maximum, dtype=np.float32)
    probability /= np.sum(probability, dtype=np.float32)
    ids = np.asarray(sorted(range(EXPERTS), key=lambda item: (-float(probability[item]), item))[:TOP_K], dtype=np.uint32)
    selected = probability[ids].astype(np.float32)
    selected /= np.sum(selected, dtype=np.float32)
    return logits, ids, selected


def moe(
    activation: np.ndarray,
    layer: int,
    ids: np.ndarray,
    route_weight: np.ndarray,
    scalar: bool,
) -> np.ndarray:
    routed = np.zeros(HIDDEN, dtype=np.float32)
    for slot, expert in enumerate(ids):
        base = 6000 + layer * 2000 + int(expert) * 3
        gate = dense(base, activation, EXPERT_DIM, scalar)
        up = dense(base + 1, activation, EXPERT_DIM, scalar)
        down = dense(base + 2, silu(gate) * up, HIDDEN, scalar)
        routed += route_weight[slot] * down
    shared_base = 15000 + layer * 10
    shared_gate = dense(shared_base, activation, EXPERT_DIM, scalar)
    shared_up = dense(shared_base + 1, activation, EXPERT_DIM, scalar)
    shared = dense(shared_base + 2, silu(shared_gate) * shared_up, HIDDEN, scalar)
    shared_scale = sigmoid(dense(shared_base + 3, activation, 1, scalar))[0]
    return np.asarray(routed + shared_scale * shared, dtype=np.float32)


@dataclass
class RunState:
    gdn_conv: np.ndarray
    gdn_recurrent: np.ndarray
    qsa_key: np.ndarray
    qsa_value: np.ndarray
    qsa_raw_index: np.ndarray
    qsa_position: np.ndarray
    ple_history: list[int]
    ple_conv: np.ndarray


def empty_state() -> RunState:
    return RunState(
        np.zeros((GDN_LAYERS, GDN_CHANNELS * 3), dtype=np.float32),
        np.zeros((GDN_LAYERS, GDN_VH * GDN_KD * GDN_VD), dtype=np.float32),
        np.zeros(CONTEXT * QSA_HD, dtype=np.float32),
        np.zeros(CONTEXT * QSA_HD, dtype=np.float32),
        np.zeros(CONTEXT * QSA_ID, dtype=np.float32),
        np.zeros(CONTEXT, dtype=np.uint32),
        [],
        np.zeros(WIDE * PLE_STATE, dtype=np.float32),
    )


def run_fixture(scalar: bool) -> dict[str, np.ndarray]:
    state = empty_state()
    captures: dict[str, list[np.ndarray]] = {
        "embedding": [],
        "ple_rows": [],
        "ple_output": [],
        "attention_mixed": [],
        "attention_output": [],
        "router_logits": [],
        "route_ids": [],
        "route_weights": [],
        "moe_output": [],
        "post_layer_wide": [],
        "final_hidden": [],
        "final_logits": [],
    }
    for position, token_value in enumerate(TOKENS):
        token = int(token_value)
        embedding = vector(100, VOCAB * HIDDEN).reshape(VOCAB, HIDDEN)[token].copy()
        captures["embedding"].append(embedding.copy())
        wide = np.tile(embedding, STREAMS).astype(np.float32)
        token_ple_rows = np.zeros(PLE_HEADS, dtype=np.uint32)
        token_ple_output = np.zeros(WIDE, dtype=np.float32)
        token_attn_mixed = []
        token_attn_output = []
        token_router_logits = []
        token_route_ids = []
        token_route_weights = []
        token_moe_output = []
        token_post_layer = []
        for layer in range(LAYERS):
            if layer == 1:
                wide, token_ple_output, token_ple_rows, state.ple_history, state.ple_conv = ple_step(
                    wide, token, state.ple_history, state.ple_conv, scalar
                )
            activation, injection = gr_prepare(wide, layer, 0, scalar)
            token_attn_mixed.append(activation.copy())
            if layer < GDN_LAYERS:
                block, state.gdn_conv[layer], state.gdn_recurrent[layer] = gdn_step(
                    activation, layer, state.gdn_conv[layer], state.gdn_recurrent[layer], scalar
                )
            else:
                block, state.qsa_key, state.qsa_value, state.qsa_raw_index = qsa_step(
                    activation, position, state.qsa_key, state.qsa_value, state.qsa_raw_index, scalar
                )
                state.qsa_position[position] = position
            token_attn_output.append(block.copy())
            wide = gr_apply(wide, block, injection)
            activation, injection = gr_prepare(wide, layer, 1, scalar)
            logits, ids, route_weights = router(activation, layer, scalar)
            block = moe(activation, layer, ids, route_weights, scalar)
            token_router_logits.append(logits.copy())
            token_route_ids.append(ids.copy())
            token_route_weights.append(route_weights.copy())
            token_moe_output.append(block.copy())
            wide = gr_apply(wide, block, injection)
            token_post_layer.append(wide.copy())
        final_hidden, _ = gr_prepare(wide, LAYERS, 2, scalar)
        final_logits = dense(17000, final_hidden, VOCAB, scalar)
        captures["ple_rows"].append(token_ple_rows)
        captures["ple_output"].append(token_ple_output)
        captures["attention_mixed"].append(np.asarray(token_attn_mixed))
        captures["attention_output"].append(np.asarray(token_attn_output))
        captures["router_logits"].append(np.asarray(token_router_logits))
        captures["route_ids"].append(np.asarray(token_route_ids))
        captures["route_weights"].append(np.asarray(token_route_weights))
        captures["moe_output"].append(np.asarray(token_moe_output))
        captures["post_layer_wide"].append(np.asarray(token_post_layer))
        captures["final_hidden"].append(final_hidden)
        captures["final_logits"].append(final_logits)

    result = {name: np.asarray(items) for name, items in captures.items()}
    result["tokens"] = TOKENS.copy()
    result["final_wide"] = result["post_layer_wide"][-1, -1].copy()
    result["final_activation"] = result["final_hidden"][-1].copy()
    result["final_gdn_conv"] = state.gdn_conv.reshape(-1).copy()
    result["final_gdn_recurrent"] = state.gdn_recurrent.reshape(-1).copy()
    result["final_qsa_key"] = state.qsa_key.copy()
    result["final_qsa_value"] = state.qsa_value.copy()
    result["final_qsa_raw_index"] = state.qsa_raw_index.copy()
    result["final_qsa_position"] = state.qsa_position.copy()
    history = state.ple_history + [PLE_PAD] * (2 - len(state.ple_history))
    result["final_ple_history"] = np.asarray(history + [len(state.ple_history)], dtype=np.uint32)
    result["final_ple_conv"] = state.ple_conv.copy()
    return result


def _torch_copy(parameter: object, values: np.ndarray) -> None:
    import torch

    with torch.no_grad():
        parameter.copy_(torch.from_numpy(np.asarray(values, dtype=np.float32)).reshape(parameter.shape))


def _assign_transformers_weights(model: object) -> None:
    import torch

    _torch_copy(model.model.embed_tokens.weight, matrix(100, VOCAB, HIDDEN))
    for layer_index, layer in enumerate(model.model.layers):
        for phase, hyper in enumerate((layer.attn_hyper_connection, layer.mlp_hyper_connection)):
            norm_role, down_role, up_role, inject_role = gr_roles(layer_index, phase)
            _torch_copy(hyper.hc_norm.weight, vector(norm_role, WIDE))
            _torch_copy(hyper.input_mix_weight_down.weight, matrix(down_role, RANK, WIDE))
            _torch_copy(hyper.input_mix_weight_up.weight, matrix(up_role, WIDE, RANK))
            _torch_copy(hyper.block_inject_weight.weight, matrix(inject_role, STREAMS, WIDE))

        router_role = 5000 + layer_index
        _torch_copy(layer.mlp.gate.weight, matrix(router_role, EXPERTS, HIDDEN))
        for expert in range(EXPERTS):
            expert_base = 6000 + layer_index * 2000 + expert * 3
            gate = matrix(expert_base, EXPERT_DIM, HIDDEN)
            up = matrix(expert_base + 1, EXPERT_DIM, HIDDEN)
            layer.mlp.experts.gate_up_proj.data[expert, :EXPERT_DIM].copy_(torch.from_numpy(gate))
            layer.mlp.experts.gate_up_proj.data[expert, EXPERT_DIM:].copy_(torch.from_numpy(up))
            layer.mlp.experts.down_proj.data[expert].copy_(
                torch.from_numpy(matrix(expert_base + 2, HIDDEN, EXPERT_DIM))
            )
        shared_base = 15000 + layer_index * 10
        _torch_copy(layer.mlp.shared_expert.gate_proj.weight, matrix(shared_base, EXPERT_DIM, HIDDEN))
        _torch_copy(layer.mlp.shared_expert.up_proj.weight, matrix(shared_base + 1, EXPERT_DIM, HIDDEN))
        _torch_copy(layer.mlp.shared_expert.down_proj.weight, matrix(shared_base + 2, HIDDEN, EXPERT_DIM))
        _torch_copy(layer.mlp.shared_expert_gate.weight, matrix(shared_base + 3, 1, HIDDEN))

        if layer_index < GDN_LAYERS:
            base = 3000 + layer_index * 20
            module = layer.linear_attn
            _torch_copy(module.in_proj_qkv.weight, matrix(base, GDN_CHANNELS, HIDDEN))
            _torch_copy(module.in_proj_z.weight, matrix(base + 1, GDN_VH * GDN_VD, HIDDEN))
            _torch_copy(module.in_proj_b.weight, matrix(base + 2, GDN_VH, HIDDEN))
            _torch_copy(module.in_proj_a.weight, matrix(base + 3, GDN_VH, HIDDEN))
            _torch_copy(module.conv1d.weight, matrix(base + 4, GDN_CHANNELS, 4).reshape(GDN_CHANNELS, 1, 4))
            _torch_copy(module.A_log, vector(base + 5, GDN_VH))
            _torch_copy(module.dt_bias, vector(base + 6, GDN_VH))
            _torch_copy(module.norm.weight, f32(1.0) + vector(base + 7, GDN_VD))
            _torch_copy(module.out_proj.weight, matrix(base + 8, HIDDEN, GDN_VH * GDN_VD))
        else:
            base = 4000
            module = layer.self_attn
            q = matrix(base, QSA_QH * QSA_HD, HIDDEN).reshape(QSA_QH, QSA_HD, HIDDEN)
            gate = matrix(base + 3, QSA_QH * QSA_HD, HIDDEN).reshape(QSA_QH, QSA_HD, HIDDEN)
            qg = np.concatenate([q, gate], axis=1).reshape(2 * QSA_QH * QSA_HD, HIDDEN)
            _torch_copy(module.q_proj.weight, qg)
            _torch_copy(module.k_proj.weight, matrix(base + 1, QSA_HD, HIDDEN))
            _torch_copy(module.v_proj.weight, matrix(base + 2, QSA_HD, HIDDEN))
            _torch_copy(module.q_norm.weight, vector(base + 10, QSA_HD))
            _torch_copy(module.k_norm.weight, vector(base + 11, QSA_HD))
            index_q = matrix(base + 4, QSA_IH * QSA_ID, HIDDEN)
            index_k = matrix(base + 5, QSA_ID, HIDDEN)
            _torch_copy(module.indexer.index_qk_proj.weight, np.concatenate([index_q, index_k], axis=0))
            _torch_copy(module.indexer.q_layernorm.weight, vector(base + 6, QSA_ID))
            _torch_copy(module.indexer.k_layernorm.weight, vector(base + 7, QSA_ID))
            _torch_copy(module.o_proj.weight, matrix(base + 9, HIDDEN, QSA_QH * QSA_HD))

    ple = model.model.layers[1].ple
    _torch_copy(
        ple.ple_embedding.ngram_embedding.weight,
        vector(2000, int(ple.ple_embedding.ngram_embedding.weight.numel())).reshape(
            ple.ple_embedding.ngram_embedding.weight.shape
        ),
    )
    with torch.no_grad():
        ple.ple_embedding.layer_multipliers.copy_(torch.tensor(PLE_MULTIPLIERS, dtype=torch.long))
    _torch_copy(ple.key_proj.weight, matrix(2001, WIDE, PLE_HEADS))
    _torch_copy(ple.value_proj.weight, matrix(2002, HIDDEN, PLE_HEADS))
    _torch_copy(ple.norm_key.weight, vector(2003, WIDE))
    _torch_copy(ple.norm_query.weight, vector(2004, WIDE))
    _torch_copy(ple.norm_conv.weight, vector(2005, WIDE))
    _torch_copy(ple.conv1d.weight, matrix(2006, WIDE, 4).reshape(WIDE, 1, 4))

    norm_role, down_role, up_role, _ = gr_roles(LAYERS, 2)
    mixer = model.model.hyper_connection_mixer
    _torch_copy(mixer.hc_norm.weight, vector(norm_role, WIDE))
    _torch_copy(mixer.input_mix_weight_down.weight, matrix(down_role, RANK, WIDE))
    _torch_copy(mixer.input_mix_weight_up.weight, matrix(up_role, WIDE, RANK))
    _torch_copy(model.lm_head.weight, matrix(17000, VOCAB, HIDDEN))


def run_transformers_fixture() -> dict[str, np.ndarray]:
    import inspect
    import types

    import torch
    import transformers
    from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig
    from transformers.models.qwen4_exp import modeling_qwen4_exp
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextExperts

    if torch.__version__.split("+")[0] != "2.9.1":
        raise RuntimeError(f"torch 2.9.1 required, got {torch.__version__}")
    source_path = pathlib.Path(inspect.getfile(modeling_qwen4_exp)).resolve()
    source_digest = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if source_digest != MODELING_SHA256:
        raise RuntimeError(f"loaded Transformers source digest mismatch: {source_digest}")
    if not transformers.__version__.startswith("5.16.0.dev0"):
        raise RuntimeError(f"unexpected Transformers build: {transformers.__version__}")

    config = Qwen4ExpTextConfig(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_hidden_layers=LAYERS,
        num_attention_heads=QSA_QH,
        num_key_value_heads=1,
        head_dim=QSA_HD,
        max_position_embeddings=CONTEXT,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=GDN_KD,
        linear_value_head_dim=GDN_VD,
        linear_num_key_heads=GDN_KH,
        linear_num_value_heads=GDN_VH,
        moe_intermediate_size=EXPERT_DIM,
        shared_expert_intermediate_size=EXPERT_DIM,
        num_experts_per_tok=TOP_K,
        num_experts=EXPERTS,
        layer_types=["linear_attention"] * GDN_LAYERS + ["full_attention"],
        hc_count=STREAMS,
        hc_lowrank=RANK,
        ple_layer_ids=[2],
        ple_embed_dim=PLE_HEADS,
        ple_conv_kernel_size=4,
        ngram_size=3,
        heads_per_ngram=8,
        ngram_vocab_size_base=17,
        make_ngram_vocab_size_divisible_by=1,
        seed=1234,
        split_ngram_parts=1,
        indexer_n_heads=QSA_IH,
        indexer_kv_heads=1,
        indexer_head_dim=QSA_ID,
        indexer_budget=BLOCK_BUDGET * COMPRESS,
        indexer_compress_ratio=COMPRESS,
        pad_token_id=PLE_PAD,
        bos_token_id=PLE_PAD,
        eos_token_id=PLE_PAD,
        output_gate_type="sigmoid",
        use_cache=False,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(SEED)
    model = Qwen4ExpForCausalLM(config).eval()
    _assign_transformers_weights(model)
    for layer in model.model.layers:
        # The integration wrapper selects grouped-mm on this host, whose tiny
        # 4-wide fixture violates its 16-byte stride. Execute the pinned class's
        # original scalar expert forward, retained in __wrapped__.
        layer.mlp.experts.forward = types.MethodType(
            Qwen4ExpTextExperts.forward.__wrapped__, layer.mlp.experts
        )

    attn_mixed: list[torch.Tensor] = []
    attn_output: list[torch.Tensor] = []
    router_logits: list[torch.Tensor] = []
    route_ids: list[torch.Tensor] = []
    route_weights: list[torch.Tensor] = []
    moe_output: list[torch.Tensor] = []
    post_layer: list[torch.Tensor] = []
    ple_output: list[torch.Tensor] = []
    handles = []

    def capture(sequence: list[torch.Tensor], select: int | None = None):
        def hook(_module: object, _inputs: object, output: object) -> None:
            selected = output[select] if select is not None else output
            sequence.append(selected.detach().cpu())
        return hook

    for layer_index, layer in enumerate(model.model.layers):
        handles.append(layer.attn_hyper_connection.register_forward_hook(capture(attn_mixed, 0)))
        attention = layer.linear_attn if layer_index < GDN_LAYERS else layer.self_attn
        handles.append(attention.register_forward_hook(capture(attn_output, 0 if layer_index == 3 else None)))

        def router_hook(_module: object, _inputs: object, output: object) -> None:
            router_logits.append(output[0].detach().cpu())
            route_weights.append(output[1].detach().cpu())
            route_ids.append(output[2].detach().cpu())

        handles.append(layer.mlp.gate.register_forward_hook(router_hook))
        handles.append(layer.mlp.register_forward_hook(capture(moe_output)))
        handles.append(layer.register_forward_hook(capture(post_layer)))
    handles.append(model.model.layers[1].ple.register_forward_hook(capture(ple_output)))

    input_ids = torch.tensor([TOKENS_LIST], dtype=torch.long)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=False)
        embedding = model.model.embed_tokens(input_ids)
        final_hidden = model.model.hyper_connection_mixer(post_layer[-1])
    for handle in handles:
        handle.remove()

    def tensor_array(value: torch.Tensor) -> np.ndarray:
        return value.detach().cpu().to(torch.float32).numpy().astype(np.float32)

    # Hooks run layer-major; transpose to the fixture's [token][layer] order.
    result = {
        "tokens": np.asarray(TOKENS_LIST, dtype=np.uint32),
        "embedding": tensor_array(embedding)[0],
        "ple_rows": np.asarray(
            [ple_rows(token, TOKENS_LIST[max(0, index - 2) : index][::-1]) for index, token in enumerate(TOKENS_LIST)],
            dtype=np.uint32,
        ),
        "ple_output": tensor_array(ple_output[0])[0],
        "attention_mixed": np.stack([tensor_array(item)[0] for item in attn_mixed], axis=1),
        "attention_output": np.stack([tensor_array(item)[0] for item in attn_output], axis=1),
        "router_logits": np.stack([tensor_array(item) for item in router_logits], axis=1),
        "route_ids": np.stack([item.detach().cpu().numpy().astype(np.uint32) for item in route_ids], axis=1),
        "route_weights": np.stack([tensor_array(item) for item in route_weights], axis=1),
        "moe_output": np.stack([tensor_array(item)[0] for item in moe_output], axis=1),
        "post_layer_wide": np.stack([tensor_array(item)[0] for item in post_layer], axis=1),
        "final_hidden": tensor_array(final_hidden)[0],
        "final_logits": tensor_array(outputs.logits)[0],
    }
    result["final_wide"] = result["post_layer_wide"][-1, -1].copy()
    result["final_activation"] = result["final_hidden"][-1].copy()
    return result


def array_bytes(array: np.ndarray) -> bytes:
    little = array.astype(array.dtype.newbyteorder("<"), copy=False)
    return little.tobytes(order="C")


def array_record(array: np.ndarray) -> dict[str, object]:
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array_bytes(array)).hexdigest(),
        "data": array.reshape(-1).tolist(),
    }


def render_json(arrays: dict[str, np.ndarray]) -> str:
    document = {
        "format": "ds4-qwen4exp-graph-golden-v1",
        "tolerance": {"float32_atol": ATOL, "float32_rtol": RTOL},
        "geometry": {
            "layers": LAYERS,
            "layer_pattern": ["GDN", "GDN", "GDN", "QSA"],
            "hidden": HIDDEN,
            "streams": STREAMS,
            "context": CONTEXT,
            "experts": EXPERTS,
            "experts_used": TOP_K,
            "ple_heads": PLE_HEADS,
        },
        "arrays": {name: array_record(array) for name, array in arrays.items()},
    }
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def c_float(value: np.float32) -> str:
    return float(np.float32(value)).hex() + "f"


def render_inc(arrays: dict[str, np.ndarray]) -> str:
    output = [
        "/* Generated by collect_graph_reference.py; do not edit. */",
        "#ifndef DS4_QWEN4EXP_GRAPH_GOLDEN_INC",
        "#define DS4_QWEN4EXP_GRAPH_GOLDEN_INC",
        "#define DS4_Q4E_GOLDEN_ATOL 2.0e-5f",
        "#define DS4_Q4E_GOLDEN_RTOL 2.0e-5f",
        "",
    ]
    for name, array in arrays.items():
        flat = array.reshape(-1)
        c_type = "uint32_t" if array.dtype == np.uint32 else "float"
        values = [str(int(value)) + "u" if c_type == "uint32_t" else c_float(value) for value in flat]
        output.append(f"static const {c_type} ds4_q4e_golden_{name}[{flat.size}] = {{")
        for start in range(0, len(values), 6):
            output.append("    " + ", ".join(values[start : start + 6]) + ",")
        output.extend(["};", ""])
    output.extend(["#endif", ""])
    return "\n".join(output)


def render_provenance(arrays: dict[str, np.ndarray], json_text: str, inc_text: str) -> str:
    payload = b"".join(array_bytes(array) for array in arrays.values())
    contract_control = {
        "tokens",
        "ple_rows",
        "final_gdn_conv",
        "final_gdn_recurrent",
        "final_qsa_key",
        "final_qsa_value",
        "final_qsa_raw_index",
        "final_qsa_position",
        "final_ple_history",
        "final_ple_conv",
    }
    document = {
        "format": "ds4-qwen4exp-graph-provenance-v1",
        "hf_model_revision": HF_REVISION,
        "transformers_commit": TRANSFORMERS_COMMIT,
        "modeling_source_url": MODELING_URL,
        "modeling_source_sha256": MODELING_SHA256,
        "oracle": "executed pinned Transformers Qwen4Exp modules/functions with deterministic F32 weights",
        "independent_cross_check": "explicit scalar-loop NumPy backend",
        "generator_environment": {
            "python": "3.13",
            "numpy": "2.4.6",
            "torch": "2.9.1",
            "transformers": "5.16.0.dev0 at pinned commit",
        },
        "contract_control_arrays": sorted(contract_control),
        "array_origin": {
            name: ("contract-control" if name in contract_control else "pinned-transformers")
            for name in arrays
        },
        "seed": SEED,
        "dtype": "float32 controls, activations, recurrence and logits",
        "device": "CPU",
        "input_ids": TOKENS.tolist(),
        "tolerance": {"atol": ATOL, "rtol": RTOL},
        "array_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "golden_json_sha256": hashlib.sha256(json_text.encode()).hexdigest(),
        "golden_inc_sha256": hashlib.sha256(inc_text.encode()).hexdigest(),
        "arrays": {name: array_record(array)["sha256"] for name, array in arrays.items()},
    }
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def build_outputs() -> tuple[str, str, str]:
    vectorized = run_fixture(scalar=False)
    scalar = run_fixture(scalar=True)
    if vectorized.keys() != scalar.keys():
        raise RuntimeError("oracle array inventories disagree")
    for name in vectorized:
        if vectorized[name].dtype == np.uint32:
            if not np.array_equal(vectorized[name], scalar[name]):
                raise RuntimeError(f"integer oracle disagreement: {name}")
        elif not np.allclose(vectorized[name], scalar[name], atol=ATOL, rtol=RTOL):
            difference = float(np.max(np.abs(vectorized[name] - scalar[name])))
            raise RuntimeError(f"F32 oracle disagreement: {name}, max={difference}")
    transformers_arrays = run_transformers_fixture()
    for name, upstream in transformers_arrays.items():
        control = vectorized[name]
        if upstream.dtype == np.uint32:
            if not np.array_equal(upstream, control):
                raise RuntimeError(f"Transformers integer disagreement: {name}")
        elif not np.allclose(upstream, control, atol=ATOL, rtol=RTOL):
            difference = float(np.max(np.abs(upstream - control)))
            raise RuntimeError(f"Transformers F32 disagreement: {name}, max={difference}")
        vectorized[name] = upstream
    json_text = render_json(vectorized)
    inc_text = render_inc(vectorized)
    provenance_text = render_provenance(vectorized, json_text, inc_text)
    return json_text, inc_text, provenance_text


def verify_source() -> None:
    with urllib.request.urlopen(MODELING_URL, timeout=30) as response:
        source = response.read()
    actual = hashlib.sha256(source).hexdigest()
    if actual != MODELING_SHA256:
        raise RuntimeError(f"pinned modeling source digest mismatch: {actual}")


def offline_check(directory: pathlib.Path) -> int:
    json_path = directory / "qwen4exp_graph_golden.json"
    inc_path = directory / "qwen4exp_graph_golden.inc"
    provenance_path = directory / "qwen4exp_graph_provenance.json"
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        json_bytes = json_path.read_bytes()
        inc_bytes = inc_path.read_bytes()
        golden = json.loads(json_bytes)
    except (OSError, json.JSONDecodeError) as error:
        print(f"fixture integrity read failed: {error}", file=sys.stderr)
        return 1
    if provenance.get("modeling_source_sha256") != MODELING_SHA256 or \
       provenance.get("transformers_commit") != TRANSFORMERS_COMMIT or \
       provenance.get("hf_model_revision") != HF_REVISION:
        print("fixture provenance pin mismatch", file=sys.stderr)
        return 1
    json_sha256 = hashlib.sha256(json_bytes).hexdigest()
    inc_sha256 = hashlib.sha256(inc_bytes).hexdigest()
    if json_sha256 != GOLDEN_JSON_SHA256 or inc_sha256 != GOLDEN_INC_SHA256 or \
       json_sha256 != provenance.get("golden_json_sha256") or \
       inc_sha256 != provenance.get("golden_inc_sha256"):
        print("generated fixture file hash mismatch", file=sys.stderr)
        return 1
    records = golden.get("arrays", {})
    if not isinstance(records, dict) or set(records) != set(ARRAY_ORDER) or \
       set(provenance.get("array_origin", {})) != set(ARRAY_ORDER):
        print("fixture array inventory mismatch", file=sys.stderr)
        return 1
    aggregate_payload = bytearray()
    for name in ARRAY_ORDER:
        record = records[name]
        try:
            dtype = record["dtype"]
            data = record["data"]
            if dtype == "float32":
                payload = b"".join(__import__("struct").pack("<f", value) for value in data)
            elif dtype == "uint32":
                payload = b"".join(__import__("struct").pack("<I", value) for value in data)
            else:
                raise ValueError(dtype)
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            print(f"invalid array record {name}: {error}", file=sys.stderr)
            return 1
        if hashlib.sha256(payload).hexdigest() != record.get("sha256"):
            print(f"array payload hash mismatch: {name}", file=sys.stderr)
            return 1
        aggregate_payload.extend(payload)
    if hashlib.sha256(aggregate_payload).hexdigest() != ARRAY_PAYLOAD_SHA256 or \
       provenance.get("array_payload_sha256") != ARRAY_PAYLOAD_SHA256:
        print("aggregate array payload hash mismatch", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="rewrite all generated fixtures")
    mode.add_argument("--check", action="store_true", help="offline byte-for-byte fixture check")
    parser.add_argument("--verify-source", action="store_true", help="verify pinned source identity over the network")
    arguments = parser.parse_args()
    directory = pathlib.Path(__file__).resolve().parent
    if arguments.verify_source:
        verify_source()
    if not arguments.write:
        return offline_check(directory)
    load_numpy()
    json_text, inc_text, provenance_text = build_outputs()
    outputs = {
        directory / "qwen4exp_graph_golden.json": json_text,
        directory / "qwen4exp_graph_golden.inc": inc_text,
        directory / "qwen4exp_graph_provenance.json": provenance_text,
    }
    for path, content in outputs.items():
        path.write_text(content, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
