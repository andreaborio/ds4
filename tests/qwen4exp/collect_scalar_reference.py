#!/usr/bin/env python3
"""Generate compact Qwen4Exp scalar reference vectors.

Write mode executes the pinned Transformers primitives and cross-checks them
against an independent NumPy transcription.  Offline check mode needs only
NumPy.  Neither mode imports Hebrus, loads weights, or allocates tensors at
model dimensions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "qwen4exp_scalar_golden.inc"
PROVENANCE = HERE / "qwen4exp_scalar_provenance.json"

HF_REPOSITORY = "Qwen/Qwen3.8-Flash-Next"
HF_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
TRANSFORMERS_COMMIT = "42ca97014c85d71a88ad60d55f08cb9fb4d26e2c"
TRANSFORMERS_SOURCE_SHA256 = (
    "91e9b1e9c74efe373cd989fe1974a8fa305f4aad43628dbcbd03dac20437814f"
)
CONFIG_SHA256 = "889658f2508e8c61d409b02e70e0d78d8d4452ec65aaafbe129805d213d2e74b"
INVENTORY_SHA256 = "a639efc7a5147b04200e870d7e320335527f4361a8327b137feca2683b1dc434"
TOKENIZER_SHA256 = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"
TOKENIZER_CONFIG_SHA256 = "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27"
CHAT_TEMPLATE_SHA256 = "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"
NUMPY_VERSION = "2.4.6"
PYTHON_VERSION = "3.13.13"
TORCH_VERSION = "2.9.1"
TRANSFORMERS_VERSION = "5.16.0.dev0"
SEED = 0x4E455854  # ASCII "NEXT"
DTYPE = "float32"
DEVICE = "cpu"
EPSILON = np.float32(1.0e-6)

MASK64 = (1 << 64) - 1
SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
SPLITMIX_M1 = 0xBF58476D1CE4E5B9
SPLITMIX_M2 = 0x94D049BB133111EB
PLE_SEED = 1234
PLE_PRIME_STEP = 10007
PLE_VOCAB = 248320
PLE_PAD_TOKEN = 248044
PLE_HEADS_PER_NGRAM = 8
PLE_HEADS = 16
PLE_NGRAM_SIZE = 3
PLE_BASE_VOCAB = 20_000_000


@dataclass(frozen=True)
class Array:
    name: str
    ctype: str
    values: np.ndarray


def f32(values: object) -> np.ndarray:
    return np.asarray(values, dtype="<f4")


def u32(values: object) -> np.ndarray:
    return np.asarray(values, dtype="<u4")


def u64(values: object) -> np.ndarray:
    return np.asarray(values, dtype="<u8")


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = f32(values)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive], dtype=np.float32))
    exponential = np.exp(values[~positive], dtype=np.float32)
    result[~positive] = exponential / (1.0 + exponential)
    return f32(result)


def silu(values: np.ndarray) -> np.ndarray:
    values = f32(values)
    return f32(values * sigmoid(values))


def softplus(values: np.ndarray) -> np.ndarray:
    values = f32(values)
    return f32(np.logaddexp(values, np.float32(0.0)))


def random_values(rng: np.random.Generator, shape: tuple[int, ...], scale: float = 1.0) -> np.ndarray:
    return f32(rng.uniform(-scale, scale, size=shape))


def zero_centered_norm(values: np.ndarray, weight: np.ndarray) -> np.ndarray:
    values = f32(values)
    weight = f32(weight)
    variance = np.mean(values * values, axis=-1, keepdims=True, dtype=np.float32)
    normalized = values * f32(1.0 / np.sqrt(variance + EPSILON, dtype=np.float32))
    return f32(normalized * (np.float32(1.0) + weight))


def conventional_gated_norm(values: np.ndarray, gate: np.ndarray, weight: np.ndarray) -> np.ndarray:
    values = f32(values)
    variance = np.mean(values * values, axis=-1, keepdims=True, dtype=np.float32)
    normalized = values * f32(1.0 / np.sqrt(variance + EPSILON, dtype=np.float32))
    return f32(f32(normalized * f32(weight)) * sigmoid(gate))


def gr_read(
    residual: np.ndarray,
    norm_weight: np.ndarray,
    down: np.ndarray,
    up: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_stream = residual.shape[0]
    normalized = zero_centered_norm(residual, norm_weight)
    wide = normalized.reshape(-1)
    hidden = silu(f32((down @ wide) / np.float32(n_stream)))
    mix_weight = sigmoid(f32(up @ hidden)).reshape(residual.shape)
    mixed = np.mean(f32(mix_weight * normalized), axis=0, dtype=np.float32)
    return f32(mixed), normalized, wide


def gr_vectors(rng: np.random.Generator) -> list[Array]:
    n_stream, dim, rank = 4, 3, 2
    residual = random_values(rng, (n_stream, dim), 1.4)
    norm_weight = random_values(rng, (n_stream, dim), 0.3)
    down = random_values(rng, (rank, n_stream * dim), 0.55)
    up = random_values(rng, (n_stream * dim, rank), 0.55)
    inject = random_values(rng, (n_stream, n_stream * dim), 0.45)
    block_output = f32([0.75, -0.35, 0.20])

    mixed, _normalized, wide = gr_read(residual, norm_weight, down, up)
    injection = f32(2.0 * sigmoid(f32((inject @ wide) / np.float32(n_stream))))
    applied = f32(residual + injection[:, None] * block_output[None, :])
    final, _normalized, _wide = gr_read(applied, norm_weight, down, up)

    return [
        Array("q4e_gr_residual", "float", residual),
        Array("q4e_gr_norm_weight", "float", norm_weight),
        Array("q4e_gr_down", "float", down),
        Array("q4e_gr_up", "float", up),
        Array("q4e_gr_inject", "float", inject),
        Array("q4e_gr_block_output", "float", block_output),
        Array("q4e_gr_mixed", "float", mixed),
        Array("q4e_gr_injection", "float", injection),
        Array("q4e_gr_applied", "float", applied),
        Array("q4e_gr_final", "float", final),
    ]


def gdn_vectors(rng: np.random.Generator) -> list[Array]:
    n_token, n_key_head, n_value_head = 2, 2, 6
    key_dim = value_dim = 2
    ratio = n_value_head // n_key_head
    query = random_values(rng, (n_token, n_key_head, key_dim), 1.1)
    key = random_values(rng, (n_token, n_key_head, key_dim), 1.1)
    value = random_values(rng, (n_token, n_value_head, value_dim), 0.9)
    alpha_logit = random_values(rng, (n_token, n_value_head), 2.0)
    beta_logit = random_values(rng, (n_token, n_value_head), 2.0)
    alpha_logit[0, 0], alpha_logit[0, 1] = np.float32(25.0), np.float32(-30.0)
    a_log = f32([-0.30, 0.10, 0.45, -0.75, 0.20, -0.55])
    dt_bias = f32([0.20, -0.40, 0.60, -0.80, 0.15, 0.50])
    log_decay = f32(-np.exp(a_log, dtype=np.float32) * softplus(alpha_logit + dt_bias))
    beta = sigmoid(beta_logit)
    state_initial = random_values(rng, (n_value_head, key_dim, value_dim), 0.25)
    state = state_initial.copy()
    output = np.zeros((n_token, n_value_head, value_dim), dtype="<f4")
    head_map = np.arange(n_value_head, dtype="<u4") // ratio

    for token in range(n_token):
        for value_head in range(n_value_head):
            key_head = int(head_map[value_head])
            q = query[token, key_head]
            k = key[token, key_head]
            q = f32(q / np.sqrt(np.sum(q * q, dtype=np.float32) + EPSILON, dtype=np.float32))
            k = f32(k / np.sqrt(np.sum(k * k, dtype=np.float32) + EPSILON, dtype=np.float32))
            q = f32(q / np.float32(math.sqrt(key_dim)))
            decay = np.float32(np.exp(log_decay[token, value_head], dtype=np.float32))
            state[value_head] = f32(state[value_head] * decay)
            prediction = f32(k @ state[value_head])
            delta = f32(beta[token, value_head] * (value[token, value_head] - prediction))
            state[value_head] = f32(state[value_head] + np.outer(k, delta))
            output[token, value_head] = f32(q @ state[value_head])

    conv_n_token, conv_n_channel, conv_kernel = 5, 4, 4
    conv_input = random_values(rng, (conv_n_token, conv_n_channel), 1.0)
    conv_weight = random_values(rng, (conv_n_channel, conv_kernel), 0.65)
    conv_state_initial = random_values(rng, (conv_n_channel, conv_kernel - 1), 0.4)
    conv_state = conv_state_initial.copy()
    conv_output = np.zeros_like(conv_input)
    for token in range(conv_n_token):
        for channel in range(conv_n_channel):
            total = np.float32(conv_input[token, channel] * conv_weight[channel, conv_kernel - 1])
            for tap in range(conv_kernel - 1):
                total = np.float32(total + np.float32(conv_state[channel, tap] * conv_weight[channel, tap]))
            conv_output[token, channel] = silu(f32([total]))[0]
        conv_state[:, :-1] = conv_state[:, 1:]
        conv_state[:, -1] = conv_input[token]

    return [
        Array("q4e_gdn_conv_input", "float", conv_input),
        Array("q4e_gdn_conv_weight", "float", conv_weight),
        Array("q4e_gdn_conv_state_initial", "float", conv_state_initial),
        Array("q4e_gdn_conv_output", "float", conv_output),
        Array("q4e_gdn_conv_state_final", "float", conv_state),
        Array("q4e_gdn_query", "float", query),
        Array("q4e_gdn_key", "float", key),
        Array("q4e_gdn_value", "float", value),
        Array("q4e_gdn_alpha_logit", "float", alpha_logit),
        Array("q4e_gdn_beta_logit", "float", beta_logit),
        Array("q4e_gdn_a_log", "float", a_log),
        Array("q4e_gdn_dt_bias", "float", dt_bias),
        Array("q4e_gdn_log_decay", "float", log_decay),
        Array("q4e_gdn_beta", "float", beta),
        Array("q4e_gdn_head_map", "uint32_t", head_map),
        Array("q4e_gdn_state_initial", "float", state_initial),
        Array("q4e_gdn_output", "float", output),
        Array("q4e_gdn_state_final", "float", state),
    ]


def router_one(logits: np.ndarray, n_selected: int) -> tuple[np.ndarray, np.ndarray]:
    logits = f32(logits)
    shifted = f32(logits - np.max(logits))
    probability = np.exp(shifted, dtype=np.float32)
    probability = f32(probability / np.sum(probability, dtype=np.float32))
    expert = np.arange(logits.size, dtype="<u4")
    order = np.lexsort((expert, -probability))[:n_selected]
    selected = expert[order]
    selected_probability = probability[order]
    weight = f32(selected_probability / np.sum(selected_probability, dtype=np.float32))
    return selected, weight


def router_vectors() -> list[Array]:
    expert = np.arange(512, dtype=np.float32)
    finite = f32(-4.0 + expert * np.float32(0.011) + np.sin(expert * np.float32(0.17)))
    for index, value in {
        7: 8.0,
        42: 7.75,
        88: 7.5,
        111: 7.25,
        201: 7.0,
        255: 6.75,
        300: 6.5,
        400: 6.25,
        499: 6.0,
        511: 5.75,
    }.items():
        finite[index] = np.float32(value)
    extreme = f32(np.linspace(-120.0, 120.0, 512, dtype=np.float32))
    extreme[0], extreme[511], extreme[256] = np.float32(119.75), np.float32(120.0), np.float32(-119.75)
    logits = f32(np.stack([finite, extreme]))
    ids, weights = zip(*(router_one(case, 10) for case in logits), strict=True)
    equal_logits = np.zeros(512, dtype="<f4")
    equal_id = u32(np.arange(10))
    equal_weight = f32([0.1] * 10)
    return [
        Array("q4e_router_equal_logits", "float", equal_logits),
        Array("q4e_router_equal_id", "uint32_t", equal_id),
        Array("q4e_router_equal_weight", "float", equal_weight),
        Array("q4e_router_upstream_logits", "float", logits),
        Array("q4e_router_upstream_id", "uint32_t", u32(np.stack(ids))),
        Array("q4e_router_upstream_weight", "float", f32(np.stack(weights))),
    ]


def partial_rope(values: np.ndarray, positions: np.ndarray, n_rot: int, theta: float) -> np.ndarray:
    output = f32(values).copy()
    half = n_rot // 2
    frequency = f32(1.0 / (theta ** (np.arange(0, n_rot, 2, dtype=np.float32) / np.float32(n_rot))))
    for token, position in enumerate(positions):
        angle = f32(np.float32(position) * frequency)
        cosine = np.cos(angle, dtype=np.float32)
        sine = np.sin(angle, dtype=np.float32)
        first = output[token, :half].copy()
        second = output[token, half:n_rot].copy()
        output[token, :half] = f32(first * cosine - second * sine)
        output[token, half:n_rot] = f32(second * cosine + first * sine)
    return output


def qsa_select(score: np.ndarray, visible_tokens: int, compression: int, budget: int) -> np.ndarray:
    n_group = visible_tokens // compression
    group_id = np.arange(n_group, dtype="<u4")
    chosen = np.lexsort((group_id, -f32(score)))[: min(budget, n_group)]
    selected: list[int] = []
    for group in group_id[chosen]:
        selected.extend(range(int(group) * compression, (int(group) + 1) * compression))
    selected.extend(range(n_group * compression, visible_tokens))
    return u32(selected)


def qsa_vectors(rng: np.random.Generator) -> list[Array]:
    compression, n_group, head_dim, n_rot, n_query_head = 4, 3, 8, 4, 4
    raw_key = random_values(rng, (n_group * compression, head_dim), 1.2)
    norm_weight = f32([0.0, 0.20, -0.10, 0.35, -0.25, 0.15, 0.05, -0.30])
    pooled = np.mean(raw_key.reshape(n_group, compression, head_dim), axis=1, dtype=np.float32)
    normalized = zero_centered_norm(pooled, norm_weight)
    group_position = u32(np.arange(n_group) * compression)
    group_key = partial_rope(normalized, group_position, n_rot, 10_000_000.0)
    query = random_values(rng, (n_query_head, head_dim), 1.0)
    head_dot = f32(group_key @ query.T)
    score = f32(np.sum(np.maximum(head_dot, np.float32(0.0)), axis=1, dtype=np.float32) / math.sqrt(head_dim))
    wrong_score = f32(np.maximum(np.sum(head_dot, axis=1, dtype=np.float32), np.float32(0.0)) / math.sqrt(head_dim))
    selected = qsa_select(score, visible_tokens=14, compression=compression, budget=2)
    tie_score = f32([1.5, 1.5, 0.1])
    tie_selected = qsa_select(tie_score, visible_tokens=14, compression=compression, budget=2)
    tail_before = u32([12, 13])
    tail_after = selected[-2:].copy()

    if np.array_equal(score, wrong_score):
        raise AssertionError("QSA fixture does not distinguish per-head ReLU")
    if not np.array_equal(tail_before, tail_after):
        raise AssertionError("QSA selection changed the incomplete tail")

    return [
        Array("q4e_qsa_raw_key", "float", raw_key),
        Array("q4e_qsa_norm_weight", "float", norm_weight),
        Array("q4e_qsa_group_position", "uint32_t", group_position),
        Array("q4e_qsa_group_key", "float", group_key),
        Array("q4e_qsa_query", "float", query),
        Array("q4e_qsa_head_dot", "float", head_dot),
        Array("q4e_qsa_score", "float", score),
        Array("q4e_qsa_wrong_relu_after_sum", "float", wrong_score),
        Array("q4e_qsa_selected", "uint32_t", selected),
        Array("q4e_qsa_tail_before", "uint32_t", tail_before),
        Array("q4e_qsa_tail_after", "uint32_t", tail_after),
        Array("q4e_qsa_tie_score", "float", tie_score),
        Array("q4e_qsa_tie_selected", "uint32_t", tie_selected),
    ]


def splitmix64(value: int) -> int:
    value = (value + SPLITMIX_GAMMA) & MASK64
    value = ((value ^ (value >> 30)) * SPLITMIX_M1) & MASK64
    value = ((value ^ (value >> 27)) * SPLITMIX_M2) & MASK64
    return (value ^ (value >> 31)) & MASK64


def ple_multipliers() -> np.ndarray:
    max_long = (1 << 63) - 1
    half_bound = max(1, (max_long // PLE_VOCAB) // 2)
    values = []
    for index in range(PLE_NGRAM_SIZE):
        mixed_seed = (PLE_SEED + SPLITMIX_GAMMA * (index + 1)) & MASK64
        values.append(2 * (splitmix64(mixed_seed) % half_bound) + 1)
    return u64(values)


def is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def nth_prime_after(start: int, count: int) -> int:
    value = start
    for _ in range(count):
        value += 1
        while not is_prime(value):
            value += 1
    return value


def ple_constants() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    multipliers = ple_multipliers()
    primes = u32([nth_prime_after(PLE_BASE_VOCAB - 1, i + 1) for i in range(PLE_HEADS)])
    offsets = np.zeros(PLE_HEADS, dtype="<u4")
    for i in range(1, PLE_HEADS):
        offsets[i] = offsets[i - 1] + primes[i - 1]
    return multipliers, primes, offsets


def ple_rows(
    current: int,
    previous1: int,
    previous2: int,
    multipliers: np.ndarray,
    primes: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    product0 = (current * int(multipliers[0])) & MASK64
    product1 = (previous1 * int(multipliers[1])) & MASK64
    product2 = (previous2 * int(multipliers[2])) & MASK64
    bigram = product0 ^ product1
    trigram = bigram ^ product2
    rows = []
    for head in range(PLE_HEADS):
        folded = bigram if head < PLE_HEADS_PER_NGRAM else trigram
        rows.append(int(offsets[head]) + folded % int(primes[head]))
    return u32(rows)


def ple_history_vectors() -> list[Array]:
    multipliers, primes, offsets = ple_constants()
    pinned_multipliers = u64([23703573157769, 20109073645365, 8052911324071])
    pinned_primes = u32([
        20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
        20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171,
    ])
    if not np.array_equal(multipliers, pinned_multipliers) or not np.array_equal(primes, pinned_primes):
        raise AssertionError("derived PLE constants drifted from the pinned checkpoint")

    token = u32([11, 12, PLE_PAD_TOKEN, 13, 14])
    history_before = np.zeros((len(token), 2), dtype="<u4")
    history_count_before = np.zeros(len(token), dtype="<u4")
    history_after = np.zeros((len(token), 2), dtype="<u4")
    history_count_after = np.zeros(len(token), dtype="<u4")
    rows = np.zeros((len(token), PLE_HEADS), dtype="<u4")
    recent: list[int] = []
    for index, current in enumerate(token):
        previous1 = recent[0] if len(recent) >= 1 else PLE_PAD_TOKEN
        previous2 = recent[1] if len(recent) >= 2 else PLE_PAD_TOKEN
        history_before[index] = u32([previous1, previous2])
        history_count_before[index] = len(recent)
        rows[index] = ple_rows(int(current), previous1, previous2, multipliers, primes, offsets)
        if current == PLE_PAD_TOKEN:
            recent = []
        else:
            recent = [int(current), *recent][:2]
        history_after[index] = u32([
            recent[0] if len(recent) >= 1 else PLE_PAD_TOKEN,
            recent[1] if len(recent) >= 2 else PLE_PAD_TOKEN,
        ])
        history_count_after[index] = len(recent)

    overflow_token = u64([0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFFFD])
    overflow_product = u64([
        (int(value) * int(multiplier)) & MASK64
        for value, multiplier in zip(overflow_token, multipliers, strict=True)
    ])
    overflow_fold = u64([
        int(overflow_product[0]) ^ int(overflow_product[1]),
        int(overflow_product[0]) ^ int(overflow_product[1]) ^ int(overflow_product[2]),
    ])

    return [
        Array("q4e_ple_multiplier", "uint64_t", multipliers),
        Array("q4e_ple_head_prime", "uint32_t", primes),
        Array("q4e_ple_head_offset", "uint32_t", offsets),
        Array("q4e_ple_token", "uint32_t", token),
        Array("q4e_ple_history_before", "uint32_t", history_before),
        Array("q4e_ple_history_count_before", "uint32_t", history_count_before),
        Array("q4e_ple_row", "uint32_t", rows),
        Array("q4e_ple_history_after", "uint32_t", history_after),
        Array("q4e_ple_history_count_after", "uint32_t", history_count_after),
        Array("q4e_ple_overflow_token", "uint64_t", overflow_token),
        Array("q4e_ple_overflow_product", "uint64_t", overflow_product),
        Array("q4e_ple_overflow_fold", "uint64_t", overflow_fold),
    ]


def ple_compute_vectors(rng: np.random.Generator) -> list[Array]:
    n_stream, dim = 4, 3
    query = f32([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.4, -0.8, 0.6],
    ])
    near = np.float32(1.0e-10 * math.sqrt(dim))
    key = f32([
        [0.0, 1.0, 0.0],
        [near, 0.0, 0.0],
        [-near, 0.0, 0.0],
        [-0.5, 0.25, 0.75],
    ])
    value = f32([0.60, -0.35, 0.90])
    dot = np.sum(f32(query * key), axis=1, dtype=np.float32) / np.float32(math.sqrt(dim))
    signed_root = f32(np.sign(dot) * np.sqrt(np.maximum(np.abs(dot), np.float32(1.0e-6)), dtype=np.float32))
    gate = sigmoid(signed_root)
    gated = f32(gate[:, None] * value[None, :])

    n_token, n_channel, kernel, dilation = 5, 2, 4, 3
    state_initial = random_values(rng, (n_channel, dilation * (kernel - 1)), 0.5)
    state = state_initial.copy()
    conv_input = random_values(rng, (n_token, n_channel), 1.0)
    conv_weight = random_values(rng, (n_channel, kernel), 0.7)
    conv_preact = np.zeros_like(conv_input)
    conv_output = np.zeros_like(conv_input)
    for token_index in range(n_token):
        for channel in range(n_channel):
            total = np.float32(0.0)
            for tap in range(kernel - 1):
                total = np.float32(total + np.float32(state[channel, tap * dilation] * conv_weight[channel, tap]))
            total = np.float32(total + np.float32(conv_input[token_index, channel] * conv_weight[channel, kernel - 1]))
            conv_preact[token_index, channel] = total
            conv_output[token_index, channel] = silu(f32([total]))[0]
        state[:, :-1] = state[:, 1:]
        state[:, -1] = conv_input[token_index]

    return [
        Array("q4e_ple_gate_query", "float", query),
        Array("q4e_ple_gate_key", "float", key),
        Array("q4e_ple_gate_value", "float", value),
        Array("q4e_ple_gate_signed_root", "float", signed_root),
        Array("q4e_ple_gate_sigmoid", "float", gate),
        Array("q4e_ple_gate_output", "float", gated),
        Array("q4e_ple_conv_input", "float", conv_input),
        Array("q4e_ple_conv_weight", "float", conv_weight),
        Array("q4e_ple_conv_state_initial", "float", state_initial),
        Array("q4e_ple_conv_preact", "float", conv_preact),
        Array("q4e_ple_conv_output", "float", conv_output),
        Array("q4e_ple_conv_state_final", "float", state),
    ]


def state_control_vectors() -> list[Array]:
    initial = f32([0.25, -0.50, 0.75, 1.00, -1.25, 1.50])
    copied = initial.copy()
    advanced = f32(initial + f32([0.1, 0.2, -0.3, 0.4, -0.5, 0.6]))
    rewound = copied.copy()
    reset = np.zeros_like(initial)
    return [
        Array("q4e_state_control_initial", "float", initial),
        Array("q4e_state_control_copied", "float", copied),
        Array("q4e_state_control_advanced", "float", advanced),
        Array("q4e_state_control_rewound", "float", rewound),
        Array("q4e_state_control_reset", "float", reset),
    ]


def build_arrays() -> list[Array]:
    rng = np.random.Generator(np.random.PCG64(SEED))
    norm_input = f32([
        [0.0, 0.75, -1.50, 2.25],
        [1.0e-20, -1.0e-20, 2.0e-20, -2.0e-20],
    ])
    norm_zero_weight = np.zeros(4, dtype="<f4")
    norm_weight = f32([0.20, -0.15, 0.00, 0.35])
    gated_input = f32([[0.25, -0.50, 1.00, -2.00], [1.50, 0.00, -0.75, 0.50]])
    gated_gate = f32([[-2.0, -0.25, 0.75, 3.0], [1.5, -1.5, 0.0, 0.25]])
    gated_weight = f32([1.0, 0.75, 1.25, 0.5])

    arrays = [
        Array("q4e_norm_input", "float", norm_input),
        Array("q4e_norm_zero_weight", "float", norm_zero_weight),
        Array("q4e_norm_zero_output", "float", zero_centered_norm(norm_input, norm_zero_weight)),
        Array("q4e_norm_weight", "float", norm_weight),
        Array("q4e_norm_output", "float", zero_centered_norm(norm_input, norm_weight)),
        Array("q4e_gated_norm_input", "float", gated_input),
        Array("q4e_gated_norm_gate", "float", gated_gate),
        Array("q4e_gated_norm_weight", "float", gated_weight),
        Array("q4e_gated_norm_output", "float", conventional_gated_norm(gated_input, gated_gate, gated_weight)),
    ]
    arrays.extend(gr_vectors(rng))
    arrays.extend(gdn_vectors(rng))
    arrays.extend(router_vectors())
    arrays.extend(qsa_vectors(rng))
    arrays.extend(ple_history_vectors())
    arrays.extend(ple_compute_vectors(rng))
    arrays.extend(state_control_vectors())
    return arrays


UPSTREAM_ORIGINS = {
    "q4e_norm_zero_output": "transformers:Qwen4ExpTextRMSNorm.forward",
    "q4e_norm_output": "transformers:Qwen4ExpTextRMSNorm.forward",
    "q4e_gated_norm_output": "transformers:Qwen4ExpTextRMSNormGated.forward(sigmoid)",
    "q4e_gr_mixed": "transformers:Qwen4ExpTextGatedResidual.forward",
    "q4e_gr_injection": "transformers:Qwen4ExpTextGatedResidual.forward",
    "q4e_gr_applied": "transformers:Qwen4ExpTextDecoderLayer injection update",
    "q4e_gr_final": "transformers:Qwen4ExpTextGatedResidual.forward(use_combine=False)",
    "q4e_gdn_conv_output": "transformers:causal_conv1d_update",
    "q4e_gdn_conv_state_final": "transformers:causal_conv1d_update",
    "q4e_gdn_log_decay": "transformers:Qwen4ExpTextGatedDeltaNet controls",
    "q4e_gdn_beta": "transformers:Qwen4ExpTextGatedDeltaNet controls",
    "q4e_gdn_output": "transformers:torch_recurrent_gated_delta_rule",
    "q4e_gdn_state_final": "transformers:torch_recurrent_gated_delta_rule",
    "q4e_router_upstream_id": "transformers:Qwen4ExpTextTopKRouter.forward",
    "q4e_router_upstream_weight": "transformers:Qwen4ExpTextTopKRouter.forward",
    "q4e_qsa_group_key": "transformers:Qwen4ExpTextQSAIndexer pooled-key path",
    "q4e_qsa_head_dot": "transformers:Qwen4ExpTextQSAIndexer score matmul",
    "q4e_qsa_score": "transformers:Qwen4ExpTextQSAIndexer per-head ReLU score",
    "q4e_qsa_selected": "transformers:Qwen4ExpTextQSAIndexer topk/expand/tail",
    "q4e_qsa_tail_after": "transformers:Qwen4ExpTextQSAIndexer tail append",
    "q4e_ple_multiplier": "transformers:_build_layer_multipliers",
    "q4e_ple_head_prime": "transformers:_find_nth_prime_after",
    "q4e_ple_head_offset": "transformers:Qwen4ExpTextNGramEmbedding offsets",
    "q4e_ple_row": "transformers:Qwen4ExpTextNGramEmbedding hash/remainder path",
    "q4e_ple_gate_signed_root": "transformers:Qwen4ExpTextPLELayer signed-root gate",
    "q4e_ple_gate_sigmoid": "transformers:Qwen4ExpTextPLELayer sigmoid gate",
    "q4e_ple_gate_output": "transformers:Qwen4ExpTextPLELayer gated value",
    "q4e_ple_conv_preact": "transformers:Qwen4ExpTextPLELayer dilated depthwise conv",
    "q4e_ple_conv_output": "transformers:Qwen4ExpTextPLELayer dilated depthwise conv+SiLU",
    "q4e_ple_conv_state_final": "transformers:Qwen4Exp PLE cache-state semantics",
}

CONTRACT_CONTROLS = {
    "q4e_gdn_head_map": "contract-control:exact 16-to-48 repeat-interleave analogue",
    "q4e_router_equal_logits": "contract-control:all-equal 512-expert logits",
    "q4e_router_equal_id": "contract-control:descending score then ascending expert ID",
    "q4e_router_equal_weight": "contract-control:renormalized equal top-10",
    "q4e_qsa_wrong_relu_after_sum": "contract-control:negative ReLU-after-sum comparator",
    "q4e_qsa_tail_before": "contract-control:incomplete causal tail identity",
    "q4e_qsa_tie_score": "contract-control:equal QSA block scores",
    "q4e_qsa_tie_selected": "contract-control:descending score then ascending group ID",
    "q4e_ple_history_before": "contract-control:transactional previous1/previous2 state",
    "q4e_ple_history_count_before": "contract-control:transactional PLE history count",
    "q4e_ple_history_after": "contract-control:current EOS resets successor history",
    "q4e_ple_history_count_after": "contract-control:current EOS resets successor count",
    "q4e_ple_overflow_token": "contract-control:out-of-profile uint64 wrap operands",
    "q4e_ple_overflow_product": "contract-control:uint64 modular multiplication",
    "q4e_ple_overflow_fold": "contract-control:uint64 XOR fold after wrap",
    "q4e_state_control_initial": "contract-control:tiny transactional state seed",
    "q4e_state_control_copied": "contract-control:state copy",
    "q4e_state_control_advanced": "contract-control:state advance after copy",
    "q4e_state_control_rewound": "contract-control:state rewind to copy",
    "q4e_state_control_reset": "contract-control:state reset",
}


def origin_for(name: str) -> tuple[str, str]:
    if name in UPSTREAM_ORIGINS:
        return "upstream-transformers", UPSTREAM_ORIGINS[name]
    if name in CONTRACT_CONTROLS:
        return "contract-control", CONTRACT_CONTROLS[name]
    return "deterministic-input", f"PCG64/fixed input for {name}"


def capture_upstream() -> tuple[list[Array], dict[str, str]]:
    """Run the pinned Transformers code and replace all authoritative outputs."""
    import inspect
    from types import SimpleNamespace

    import torch
    import torch.nn.functional as torch_f
    import transformers
    from transformers.models.qwen4_exp import modeling_qwen4_exp as model

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
        raise SystemExit(f"pinned capture environment required: expected {expected}, found {versions}")
    source_path = Path(inspect.getfile(model))
    source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if source_sha256 != TRANSFORMERS_SOURCE_SHA256:
        raise SystemExit(
            f"pinned Transformers source SHA-256 mismatch: {source_sha256} != {TRANSFORMERS_SOURCE_SHA256}"
        )

    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    arrays = build_arrays()
    by_name = {array.name: array for array in arrays}

    def tensor(name: str) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(by_name[name].values).copy())

    def replace(name: str, value: torch.Tensor | np.ndarray) -> None:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().contiguous().numpy()
        old = by_name[name]
        converted = f32(value) if old.ctype == "float" else (u32(value) if old.ctype == "uint32_t" else u64(value))
        if converted.shape != old.values.shape:
            raise AssertionError(f"upstream shape drift for {name}: {converted.shape} != {old.values.shape}")
        by_name[name] = Array(name, old.ctype, converted)

    with torch.no_grad():
        norm = model.Qwen4ExpTextRMSNorm(4, eps=1.0e-6)
        norm.weight.copy_(tensor("q4e_norm_zero_weight"))
        replace("q4e_norm_zero_output", norm(tensor("q4e_norm_input")))
        norm.weight.copy_(tensor("q4e_norm_weight"))
        replace("q4e_norm_output", norm(tensor("q4e_norm_input")))

        gated_norm = model.Qwen4ExpTextRMSNormGated(4, eps=1.0e-6, activation="sigmoid")
        gated_norm.weight.copy_(tensor("q4e_gated_norm_weight"))
        replace(
            "q4e_gated_norm_output",
            gated_norm(tensor("q4e_gated_norm_input"), tensor("q4e_gated_norm_gate")),
        )

        gr_config = SimpleNamespace(hc_count=4, hidden_size=3, hc_lowrank=2, rms_norm_eps=1.0e-6)
        gr = model.Qwen4ExpTextGatedResidual(gr_config, use_combine=True)
        gr.hc_norm.weight.copy_(tensor("q4e_gr_norm_weight").reshape(-1))
        gr.input_mix_weight_down.weight.copy_(tensor("q4e_gr_down"))
        gr.input_mix_weight_up.weight.copy_(tensor("q4e_gr_up"))
        gr.block_inject_weight.weight.copy_(tensor("q4e_gr_inject"))
        hyper_input = tensor("q4e_gr_residual").reshape(1, -1)
        mixed, original, injection = gr(hyper_input)
        replace("q4e_gr_mixed", mixed.squeeze(0))
        replace("q4e_gr_injection", injection.squeeze(0))
        applied = original.reshape(1, 4, 3) + injection.unsqueeze(-1) * tensor("q4e_gr_block_output")
        replace("q4e_gr_applied", applied.squeeze(0))
        final_gr = model.Qwen4ExpTextGatedResidual(gr_config, use_combine=False)
        final_gr.hc_norm.weight.copy_(gr.hc_norm.weight)
        final_gr.input_mix_weight_down.weight.copy_(gr.input_mix_weight_down.weight)
        final_gr.input_mix_weight_up.weight.copy_(gr.input_mix_weight_up.weight)
        replace("q4e_gr_final", final_gr(applied.reshape(1, -1)).squeeze(0))

        conv_input = tensor("q4e_gdn_conv_input").T.unsqueeze(0)
        conv_state = tensor("q4e_gdn_conv_state_initial").unsqueeze(0).clone()
        conv_output = model.causal_conv1d_update(
            conv_input,
            conv_state,
            tensor("q4e_gdn_conv_weight"),
            activation="silu",
        )
        replace("q4e_gdn_conv_output", conv_output.squeeze(0).T)
        replace("q4e_gdn_conv_state_final", conv_state.squeeze(0))

        alpha_logit = tensor("q4e_gdn_alpha_logit")
        beta_logit = tensor("q4e_gdn_beta_logit")
        a_log = tensor("q4e_gdn_a_log")
        dt_bias = tensor("q4e_gdn_dt_bias")
        log_decay = -a_log.float().exp() * torch_f.softplus(alpha_logit.float() + dt_bias)
        beta = beta_logit.sigmoid()
        replace("q4e_gdn_log_decay", log_decay)
        replace("q4e_gdn_beta", beta)
        query = tensor("q4e_gdn_query").unsqueeze(0).repeat_interleave(3, dim=2)
        key = tensor("q4e_gdn_key").unsqueeze(0).repeat_interleave(3, dim=2)
        value = tensor("q4e_gdn_value").unsqueeze(0)
        gdn_output, gdn_state = model.torch_recurrent_gated_delta_rule(
            query,
            key,
            value,
            g=log_decay.unsqueeze(0),
            beta=beta.unsqueeze(0),
            initial_state=tensor("q4e_gdn_state_initial").unsqueeze(0),
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        replace("q4e_gdn_output", gdn_output.squeeze(0))
        replace("q4e_gdn_state_final", gdn_state.squeeze(0))

        router_config = SimpleNamespace(
            num_experts_per_tok=10,
            num_experts=512,
            norm_topk_prob=True,
            hidden_size=1,
        )
        router = model.Qwen4ExpTextTopKRouter(router_config)
        router_ids = []
        router_weights = []
        for logits in tensor("q4e_router_upstream_logits"):
            router.weight.copy_(logits.reshape(512, 1))
            _logits, weights, ids = router(torch.ones(1, 1, dtype=torch.float32))
            router_ids.append(ids.squeeze(0))
            router_weights.append(weights.squeeze(0))
        replace("q4e_router_upstream_id", torch.stack(router_ids))
        replace("q4e_router_upstream_weight", torch.stack(router_weights))

        raw_key = tensor("q4e_qsa_raw_key")
        pooled = raw_key.reshape(3, 4, 8).float().mean(dim=1).to(raw_key.dtype)
        qsa_norm = model.Qwen4ExpTextRMSNorm(8, eps=1.0e-6)
        qsa_norm.weight.copy_(tensor("q4e_qsa_norm_weight"))
        pooled = qsa_norm(pooled)
        positions = tensor("q4e_qsa_group_position").float()
        inverse = 1.0 / (10_000_000.0 ** (torch.arange(0, 4, 2, dtype=torch.float32) / 4.0))
        frequency = positions[:, None] * inverse[None, :]
        cosine = torch.cat([frequency.cos(), frequency.cos()], dim=-1)
        sine = torch.cat([frequency.sin(), frequency.sin()], dim=-1)
        group_key = model.apply_rotary_pos_emb(pooled.unsqueeze(1), cos=cosine, sin=sine).squeeze(1)
        replace("q4e_qsa_group_key", group_key)
        head_dot = torch.matmul(tensor("q4e_qsa_query").float(), group_key.float().T).T
        score = torch.relu(head_dot).sum(dim=-1) / math.sqrt(8)
        replace("q4e_qsa_head_dot", head_dot)
        replace("q4e_qsa_score", score)
        chosen = score.topk(2, dim=0).indices
        block = torch.arange(12, dtype=torch.int64).reshape(3, 4)
        tail = tensor("q4e_qsa_tail_before").to(torch.int64)
        selected = torch.cat([block.index_select(0, chosen).flatten(), tail]).to(torch.int32)
        replace("q4e_qsa_selected", selected)
        replace("q4e_qsa_tail_after", selected[-2:])

        multipliers = model._build_layer_multipliers(PLE_VOCAB, PLE_NGRAM_SIZE, 0, PLE_SEED)
        primes = torch.tensor(
            [model._find_nth_prime_after(PLE_BASE_VOCAB - 1, head + 1) for head in range(PLE_HEADS)],
            dtype=torch.long,
        )
        offsets = torch.zeros(PLE_HEADS, dtype=torch.long)
        offsets[1:] = torch.cumsum(primes[:-1], dim=0)
        replace("q4e_ple_multiplier", multipliers)
        replace("q4e_ple_head_prime", primes)
        replace("q4e_ple_head_offset", offsets)
        ngram = model.Qwen4ExpTextNGramEmbedding.__new__(model.Qwen4ExpTextNGramEmbedding)
        torch.nn.Module.__init__(ngram)
        ngram.eos_token_id = PLE_PAD_TOKEN
        token = tensor("q4e_ple_token").to(torch.long).reshape(1, -1)
        shifted = [ngram._shift_right_ignore_eos(token, shift) for shift in range(PLE_NGRAM_SIZE)]
        row_blocks = []
        for order in (2, 3):
            start = (order - 2) * PLE_HEADS_PER_NGRAM
            end = start + PLE_HEADS_PER_NGRAM
            mixed = shifted[0] * multipliers[0]
            for position in range(1, order):
                mixed = torch.bitwise_xor(mixed, shifted[position] * multipliers[position])
            row_blocks.append(
                torch.remainder(mixed.unsqueeze(-1), primes[start:end]) + offsets[start:end]
            )
        replace("q4e_ple_row", torch.cat(row_blocks, dim=-1).squeeze(0))

        gate_query = tensor("q4e_ple_gate_query")
        gate_key = tensor("q4e_ple_gate_key")
        gate_value = tensor("q4e_ple_gate_value")
        gate_score = (gate_query * gate_key).sum(dim=-1) / math.sqrt(3)
        signed_root = gate_score.abs().clamp_min(1.0e-6).sqrt() * gate_score.sign()
        gate = torch.sigmoid(signed_root)
        replace("q4e_ple_gate_signed_root", signed_root)
        replace("q4e_ple_gate_sigmoid", gate)
        replace("q4e_ple_gate_output", gate.unsqueeze(-1) * gate_value)

        ple_input = tensor("q4e_ple_conv_input").T
        ple_state = tensor("q4e_ple_conv_state_initial")
        ple_full = torch.cat([ple_state, ple_input], dim=-1).unsqueeze(0)
        ple_weight = tensor("q4e_ple_conv_weight").unsqueeze(1)
        ple_preact = torch_f.conv1d(ple_full, ple_weight, groups=2, dilation=3).squeeze(0).T
        replace("q4e_ple_conv_preact", ple_preact)
        replace("q4e_ple_conv_output", torch_f.silu(ple_preact))
        replace("q4e_ple_conv_state_final", ple_full.squeeze(0)[:, -9:])

    return [by_name[array.name] for array in arrays], versions


def array_digest(arrays: list[Array]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(array.values)
        digest.update(array.name.encode("ascii") + b"\0")
        digest.update(values.dtype.str.encode("ascii") + b"\0")
        digest.update(",".join(str(value) for value in values.shape).encode("ascii") + b"\0")
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def array_sha256(array: Array) -> str:
    return hashlib.sha256(np.ascontiguousarray(array.values).tobytes(order="C")).hexdigest()


def c_float(value: np.float32) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("the fixture must contain only finite floats")
    rendered = format(number, ".9g")
    if "." not in rendered and "e" not in rendered:
        rendered += ".0"
    return rendered + "f"


def c_value(ctype: str, value: object) -> str:
    if ctype == "float":
        return c_float(np.float32(value))
    if ctype == "uint32_t":
        return f"{int(value)}u"
    if ctype == "uint64_t":
        return f"UINT64_C({int(value)})"
    raise ValueError(f"unsupported C type: {ctype}")


def emit_array(array: Array) -> str:
    flat = array.values.reshape(-1)
    width = 6 if array.ctype == "float" else 8
    lines = [f"static const {array.ctype} {array.name}[{flat.size}] = {{"]
    if np.all(flat == 0):
        lines.append(f"    {c_value(array.ctype, flat[0])},")
        lines.append("};")
        return "\n".join(lines)
    for start in range(0, flat.size, width):
        row = ", ".join(c_value(array.ctype, value) for value in flat[start : start + width])
        lines.append(f"    {row},")
    lines.append("};")
    return "\n".join(lines)


def render_fixture(arrays: list[Array]) -> str:
    digest = array_digest(arrays)
    header = f'''/* Generated by collect_scalar_reference.py.  Do not edit by hand. */
/* HF: {HF_REPOSITORY}@{HF_REVISION} */
/* Transformers: {TRANSFORMERS_COMMIT} */
#include <stdint.h>

#define Q4E_SCALAR_HF_REPOSITORY "{HF_REPOSITORY}"
#define Q4E_SCALAR_HF_REVISION "{HF_REVISION}"
#define Q4E_SCALAR_TRANSFORMERS_COMMIT "{TRANSFORMERS_COMMIT}"
#define Q4E_SCALAR_TRANSFORMERS_SOURCE_SHA256 "{TRANSFORMERS_SOURCE_SHA256}"
#define Q4E_SCALAR_CONFIG_SHA256 "{CONFIG_SHA256}"
#define Q4E_SCALAR_INVENTORY_SHA256 "{INVENTORY_SHA256}"
#define Q4E_SCALAR_TOKENIZER_SHA256 "{TOKENIZER_SHA256}"
#define Q4E_SCALAR_TOKENIZER_CONFIG_SHA256 "{TOKENIZER_CONFIG_SHA256}"
#define Q4E_SCALAR_CHAT_TEMPLATE_SHA256 "{CHAT_TEMPLATE_SHA256}"
#define Q4E_SCALAR_PYTHON_VERSION "{PYTHON_VERSION}"
#define Q4E_SCALAR_NUMPY_VERSION "{NUMPY_VERSION}"
#define Q4E_SCALAR_TORCH_VERSION "{TORCH_VERSION}"
#define Q4E_SCALAR_TRANSFORMERS_VERSION "{TRANSFORMERS_VERSION}"
#define Q4E_SCALAR_DTYPE "{DTYPE}"
#define Q4E_SCALAR_DEVICE "{DEVICE}"
#define Q4E_SCALAR_ARRAY_SHA256 "{digest}"

enum {{
    Q4E_NORM_N_VECTOR = 2,
    Q4E_NORM_DIM = 4,
    Q4E_GR_N_STREAM = 4,
    Q4E_GR_DIM = 3,
    Q4E_GR_RANK = 2,
    Q4E_GDN_N_TOKEN = 2,
    Q4E_GDN_N_KEY_HEAD = 2,
    Q4E_GDN_N_VALUE_HEAD = 6,
    Q4E_GDN_KEY_DIM = 2,
    Q4E_GDN_VALUE_DIM = 2,
    Q4E_GDN_REPEAT_RATIO = 3,
    Q4E_GDN_CONV_N_TOKEN = 5,
    Q4E_GDN_CONV_N_CHANNEL = 4,
    Q4E_GDN_CONV_KERNEL = 4,
    Q4E_ROUTER_N_UPSTREAM_CASE = 2,
    Q4E_ROUTER_N_EXPERT = 512,
    Q4E_ROUTER_N_SELECTED = 10,
    Q4E_QSA_COMPRESSION = 4,
    Q4E_QSA_N_GROUP = 3,
    Q4E_QSA_HEAD_DIM = 8,
    Q4E_QSA_N_ROT = 4,
    Q4E_QSA_N_QUERY_HEAD = 4,
    Q4E_QSA_VISIBLE_TOKEN = 14,
    Q4E_QSA_GROUP_BUDGET = 2,
    Q4E_QSA_N_SELECTED = 10,
    Q4E_PLE_N_TOKEN = 5,
    Q4E_PLE_N_HEAD = 16,
    Q4E_PLE_GATE_N_STREAM = 4,
    Q4E_PLE_GATE_DIM = 3,
    Q4E_PLE_CONV_N_TOKEN = 5,
    Q4E_PLE_CONV_N_CHANNEL = 2,
    Q4E_PLE_CONV_KERNEL = 4,
    Q4E_PLE_CONV_DILATION = 3,
    Q4E_PLE_CONV_STATE = 9,
    Q4E_STATE_CONTROL_VALUES = 6,
}};

#define Q4E_SCALAR_SEED UINT32_C({SEED})
#define Q4E_SCALAR_EPSILON 1.0e-6f
#define Q4E_QSA_THETA 10000000.0f
#define Q4E_PLE_PAD_TOKEN UINT32_C({PLE_PAD_TOKEN})
'''
    return header + "\n" + "\n\n".join(emit_array(array) for array in arrays) + "\n"


def dtype_name(array: Array) -> str:
    return {"float": "float32", "uint32_t": "uint32", "uint64_t": "uint64"}[array.ctype]


def provenance_document(arrays: list[Array]) -> dict[str, object]:
    records = []
    for array in arrays:
        authority, origin = origin_for(array.name)
        exact = array.ctype != "float" or authority != "upstream-transformers"
        records.append(
            {
                "name": array.name,
                "authority": authority,
                "origin": origin,
                "cType": array.ctype,
                "dtype": dtype_name(array),
                "shape": list(array.values.shape),
                "elements": int(array.values.size),
                "sha256": array_sha256(array),
                "numpyCrossCheck": (
                    {"mode": "exact"}
                    if exact
                    else {"mode": "allclose", "atol": 2.0e-5, "rtol": 2.0e-5}
                ),
            }
        )
    return {
        "schemaVersion": 1,
        "kind": "qwen4exp-scalar-provenance",
        "status": "model-free-not-support",
        "generatedBy": "tests/qwen4exp/collect_scalar_reference.py",
        "source": {
            "hfRepository": HF_REPOSITORY,
            "hfRevision": HF_REVISION,
            "transformersCommit": TRANSFORMERS_COMMIT,
            "transformersModelingSha256": TRANSFORMERS_SOURCE_SHA256,
            "configSha256": CONFIG_SHA256,
            "inventorySha256": INVENTORY_SHA256,
            "tokenizerSha256": TOKENIZER_SHA256,
            "tokenizerConfigSha256": TOKENIZER_CONFIG_SHA256,
            "chatTemplateSha256": CHAT_TEMPLATE_SHA256,
        },
        "capture": {
            "python": PYTHON_VERSION,
            "numpy": NUMPY_VERSION,
            "torch": TORCH_VERSION,
            "transformers": TRANSFORMERS_VERSION,
            "device": DEVICE,
            "dtype": DTYPE,
            "seedDecimal": SEED,
            "seedHex": f"0x{SEED:08x}",
            "torchDeterministicAlgorithms": True,
            "torchThreads": 1,
        },
        "fixtureArraySha256": array_digest(arrays),
        "arrays": records,
    }


def render_provenance(arrays: list[Array]) -> str:
    return json.dumps(provenance_document(arrays), indent=2, sort_keys=True) + "\n"


ARRAY_RE = re.compile(
    r"static const (float|uint32_t|uint64_t) ([a-zA-Z0-9_]+)\[(\d+)\] = \{\n(.*?)\n\};",
    re.DOTALL,
)


def parse_fixture(text: str, reference: list[Array]) -> list[Array]:
    shape_by_name = {array.name: array.values.shape for array in reference}
    expected_names = [array.name for array in reference]
    parsed: list[Array] = []
    for match in ARRAY_RE.finditer(text):
        ctype, name, count_text, body = match.groups()
        if name not in shape_by_name:
            raise SystemExit(f"unexpected scalar fixture array: {name}")
        count = int(count_text)
        tokens = [token.strip() for token in body.replace("\n", " ").split(",") if token.strip()]
        values: list[float | int] = []
        for token in tokens:
            if ctype == "float":
                values.append(float(token[:-1] if token.endswith("f") else token))
            elif ctype == "uint32_t":
                values.append(int(token[:-1] if token.endswith("u") else token))
            else:
                match_u64 = re.fullmatch(r"UINT64_C\((\d+)\)", token)
                if match_u64 is None:
                    raise SystemExit(f"invalid uint64 fixture token for {name}: {token}")
                values.append(int(match_u64.group(1)))
        if len(values) == 1 and count > 1 and values[0] == 0:
            values.extend([0] * (count - 1))
        if len(values) != count:
            raise SystemExit(f"fixture element count mismatch for {name}: {len(values)} != {count}")
        converted = f32(values) if ctype == "float" else (u32(values) if ctype == "uint32_t" else u64(values))
        shape = shape_by_name[name]
        if math.prod(shape) != count:
            raise SystemExit(f"fixture shape/count mismatch for {name}: {shape} != {count}")
        parsed.append(Array(name, ctype, converted.reshape(shape)))
    names = [array.name for array in parsed]
    if names != expected_names:
        raise SystemExit("fixture array order/set drifted")
    return parsed


def numpy_cross_check(actual: list[Array], independent: list[Array]) -> None:
    actual_by_name = {array.name: array for array in actual}
    if [array.name for array in actual] != [array.name for array in independent]:
        raise AssertionError("upstream and NumPy array sets differ")
    for expected in independent:
        observed = actual_by_name[expected.name]
        authority, _origin = origin_for(expected.name)
        if expected.ctype != "float" or authority != "upstream-transformers":
            if not np.array_equal(observed.values, expected.values):
                raise AssertionError(f"exact NumPy cross-check failed for {expected.name}")
        elif not np.allclose(observed.values, expected.values, atol=2.0e-5, rtol=2.0e-5):
            difference = float(np.max(np.abs(observed.values - expected.values)))
            raise AssertionError(f"NumPy cross-check failed for {expected.name}: max abs {difference}")


def check_offline(fixture: Path, provenance: Path) -> str:
    if np.__version__ != NUMPY_VERSION:
        raise SystemExit(f"offline check requires NumPy {NUMPY_VERSION}; found {np.__version__}")
    if not fixture.is_file() or not provenance.is_file():
        raise SystemExit("Qwen4Exp scalar fixture or provenance file is missing")
    independent = build_arrays()
    fixture_text = fixture.read_text(encoding="utf-8")
    arrays = parse_fixture(fixture_text, independent)
    numpy_cross_check(arrays, independent)
    if fixture_text != render_fixture(arrays):
        raise SystemExit(f"Qwen4Exp scalar fixture text drifted: {fixture}")
    provenance_text = provenance.read_text(encoding="utf-8")
    if provenance_text != render_provenance(arrays):
        raise SystemExit(f"Qwen4Exp scalar provenance drifted: {provenance}")
    return array_digest(arrays)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the checked-in C fixture")
    parser.add_argument("--check", action="store_true", help="fail if the checked-in fixture drifts")
    parser.add_argument("--output", type=Path, default=FIXTURE)
    parser.add_argument("--provenance", type=Path, default=PROVENANCE)
    args = parser.parse_args()
    if args.write == args.check:
        parser.error("use exactly one of --write or --check")

    if args.check:
        digest = check_offline(args.output, args.provenance)
        print(f"Qwen4Exp scalar fixture and provenance pass offline checks ({digest})")
        return 0

    arrays, versions = capture_upstream()
    numpy_cross_check(arrays, build_arrays())
    args.output.write_text(render_fixture(arrays), encoding="utf-8")
    args.provenance.write_text(render_provenance(arrays), encoding="utf-8")
    print(f"captured pinned upstream {versions}")
    print(f"wrote {args.output} and {args.provenance} ({array_digest(arrays)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
