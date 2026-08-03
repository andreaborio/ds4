"""Optional Apple-Silicon cross-checks for the NumPy DSpark oracle.

Importing :mod:`tools.dspark_oracle` never imports MLX.  These helpers load
``mlx.core`` only when explicitly called and do not depend on ``mlx-lm``.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from typing import Sequence

import numpy as np


# Operation-specific fixture limits for MLX 0.32.0 float32 Metal execution.
# They are deliberately not a global oracle tolerance: the synthetic Markov
# matmul measured 9.765625e-05 max absolute drift, while the confidence
# projection plus sigmoid measured 4.42771e-08.  The narrow ceilings below keep
# that Metal rounding visible and fail if either operation moves into a
# different numerical envelope.  NumPy float64 remains the fixture authority.
MLX_F32_MARKOV_MATMUL_MAX_ABS_DRIFT = 1.0e-4
MLX_F32_CONFIDENCE_MAX_ABS_DRIFT = 5.0e-8
MLX_F32_HC_MEAN_MAX_ABS_DRIFT = 1.0e-7
MLX_F32_MAIN_PROJECTION_MAX_ABS_DRIFT = 5.0e-8
MLX_F32_RAW_CONTEXT_MAIN_MAX_ABS_DRIFT = 1.0e-7
MLX_F32_HC_SPLIT_MAX_ABS_DRIFT = 1.0e-7
MLX_F32_HC_OUTPUT_MAX_ABS_DRIFT = 5.0e-7
# The raw-context finalizer publishes BF16 at every named boundary.  These
# measured MLX 0.32.0 / M5 Pro ceilings are intentionally separate: a one-ULP
# RoPE difference must not be hidden behind the coarser FP8-storage allowance.
# The final store allows one adjacent E4M3FN code step after an upstream BF16
# difference; the exact frozen NumPy digest remains authoritative.
MLX_BF16_CONTEXT_PROJECTED_MAX_ABS_DRIFT = 3.90625e-3
MLX_BF16_CONTEXT_NORMALIZED_MAX_ABS_DRIFT = 7.8125e-3
MLX_BF16_CONTEXT_ROPE_MAX_ABS_DRIFT = 7.8125e-3
MLX_BF16_CONTEXT_STORED_MAX_ABS_DRIFT = 6.25e-2
MLX_F32_CONTEXT_SCALE_MAX_ABS_DRIFT = 0.0
# The full attention output is BF16.  On the deterministic physical-ring
# fixture MLX differs from the NumPy online-softmax oracle in six lanes, each
# by at most one 2^-12 step after the final BF16 publication.
MLX_BF16_ATTENTION_OUTPUT_MAX_ABS_DRIFT = 2.44140625e-4
# The compact stage-zero chain uses deterministic synthetic projections and
# conditions V away from zero cancellation.  All publications before
# attention, and those after inverse RoPE, are exact between NumPy and MLX.
# C=2 is exact; the wrapped C=128 attention result differs in two lanes (ten at
# inverse RoPE), each by at most one 2^-12 BF16 step.
MLX_BF16_STAGE_ZERO_ATTENTION_MAX_ABS_DRIFT = 2.44140625e-4
# Payload-first 32/32/4 physical fixture: C=2 is exact and the wrapped C=128
# attention/inverse-RoPE boundaries differ in six adjacent BF16 lanes.
MLX_BF16_PHYSICAL_STAGE_ZERO_ATTENTION_MAX_ABS_DRIFT = 1.220703125e-4
# Payload-first FFN limits measured independently on MLX 0.32.0 / M5 Pro.
# Router matmul reduction drift changes 8/1,280 logits/probabilities and the
# six-way normalization changes all 30 weights.  Explicit BF16 before down
# bounds the propagated change to one adjacent code in 256 routed-mid lanes;
# down/routed-sum remain within one 2^-9 step and final MoE/HC publications are
# exact.  Keep these field-specific rather than hiding them in one tolerance.
MLX_FFN_OPERATION_MAX_ABS_DRIFT = {
    "hidden_input": 0.0,
    "hc_pre_output": 0.0,
    "ffn_normalized": 0.0,
    "router_logits": 1.5625e-2,
    "router_probabilities": 1.3456344604492188e-3,
    "expert_weights": 4.869699478149414e-5,
    "shared_gate": 0.0,
    "shared_up": 0.0,
    "shared_mid": 0.0,
    "shared_down": 0.0,
    "routed_gate": 0.0,
    "routed_up": 0.0,
    "routed_mid": 4.8828125e-4,
    "routed_down": 1.953125e-3,
    "routed_sum": 1.953125e-3,
    "moe_output": 0.0,
    "hc_post_output": 0.0,
}
EXPECTED_MLX_VERSION = "0.32.0"
EXPECTED_MLX_METAL_VERSION = "0.32.0"
DSPARK_TARGET_LAYER_IDS = (40, 41, 42)
DSPARK_RAW_CACHE_WIDTH = 512
DSPARK_RAW_CACHE_WINDOW = 128
DSPARK_ATTENTION_HEADS = 64
DSPARK_ATTENTION_BLOCK = 64
DSPARK_PROPOSAL_ROWS = 5
DSPARK_ROPE_WIDTH = 64
DSPARK_OUTPUT_GROUPS = 8


@dataclass(frozen=True)
class MLXStageContextKV:
    """Host-visible boundaries from the independent MLX Metal finalizer."""

    absolute_positions: np.ndarray
    projected: np.ndarray
    normalized: np.ndarray
    roped: np.ndarray
    stored: np.ndarray
    nonrope_scales: np.ndarray


@dataclass(frozen=True)
class MLXStageZeroAttentionHalf:
    """Host-visible MLX boundaries for the compact stage-zero oracle."""

    absolute_positions: np.ndarray
    hidden_input: np.ndarray
    hc_pre_output: np.ndarray
    attention_normalized: np.ndarray
    q_a: np.ndarray
    q_a_normalized: np.ndarray
    q_b: np.ndarray
    q_head_normalized: np.ndarray
    q_roped: np.ndarray
    kv_projected: np.ndarray
    kv_normalized: np.ndarray
    kv_roped: np.ndarray
    kv_stored: np.ndarray
    kv_nonrope_scales: np.ndarray
    attention_output: np.ndarray
    attention_inverse_roped: np.ndarray
    output_a: np.ndarray
    output_b: np.ndarray
    hc_post_output: np.ndarray


@dataclass(frozen=True)
class MLXStageFFNMoE:
    """Host-visible MLX boundaries for the payload-first FFN fixture."""

    hidden_input: np.ndarray
    hc_pre_output: np.ndarray
    ffn_normalized: np.ndarray
    router_logits: np.ndarray
    router_probabilities: np.ndarray
    selected_experts: np.ndarray
    expert_weights: np.ndarray
    shared_gate: np.ndarray
    shared_up: np.ndarray
    shared_mid: np.ndarray
    shared_down: np.ndarray
    routed_gate: np.ndarray
    routed_up: np.ndarray
    routed_mid: np.ndarray
    routed_down: np.ndarray
    routed_sum: np.ndarray
    moe_output: np.ndarray
    hc_post_output: np.ndarray


def _e4m3fn_positive_values() -> np.ndarray:
    """Build the positive finite E4M3FN codebook without NumPy-oracle reuse."""

    values: list[float] = []
    for code in range(127):
        exponent = code >> 3
        mantissa = code & 7
        if exponent == 0:
            values.append(mantissa * (2.0 ** -9))
        else:
            values.append((1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7)))
    return np.asarray(values, dtype=np.float32)


_E4M3FN_POSITIVE_VALUES = _e4m3fn_positive_values()
_E4M3FN_EVEN_CODE_PREFERENCE = np.asarray(
    [2 if (code & 1) == 0 else 1 for code in range(127)],
    dtype=np.int32,
)


def available() -> bool:
    try:
        return importlib.util.find_spec("mlx.core") is not None
    except ModuleNotFoundError:
        return False


def _mlx() -> object:
    try:
        import mlx.core as mx
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "MLX is optional and is not installed; install mlx only in the "
            "Apple-Silicon development environment"
        ) from exc
    return mx


def require_pinned_metal() -> tuple[str, str, str]:
    """Fail closed unless the parity lane is pinned MLX on the Metal GPU."""

    mx = _mlx()
    try:
        mlx_version = importlib_metadata.version("mlx")
        metal_version = importlib_metadata.version("mlx-metal")
    except importlib_metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "MLX parity requires both mlx and mlx-metal distributions"
        ) from exc
    if mlx_version != EXPECTED_MLX_VERSION:
        raise RuntimeError(
            f"MLX parity requires mlx=={EXPECTED_MLX_VERSION}, got {mlx_version}"
        )
    if metal_version != EXPECTED_MLX_METAL_VERSION:
        raise RuntimeError(
            "MLX parity requires mlx-metal=="
            f"{EXPECTED_MLX_METAL_VERSION}, got {metal_version}"
        )
    device = mx.default_device()
    if device.type != mx.gpu:
        raise RuntimeError(
            f"MLX parity requires the Metal GPU default device, got {device}"
        )
    # Force device creation here so a headless/sandboxed process fails at the
    # gate rather than halfway through an otherwise ambiguous parity run.
    probe = mx.array([0.0], dtype=mx.float32)
    mx.eval(probe)
    return mlx_version, metal_version, str(device)


def markov_step_bias(
    previous_tokens: Sequence[int] | np.ndarray,
    embedding: np.ndarray,
    projection: np.ndarray,
) -> np.ndarray:
    mx = _mlx()
    tokens = mx.array(np.asarray(previous_tokens, dtype=np.int32))
    # MLX's portable Apple GPU path is float32 here.  The canonical fixture is
    # still generated in NumPy float64; this adapter is a tolerance-based
    # device cross-check, not the source of expected values.
    embed = mx.array(np.asarray(embedding, dtype=np.float32))
    project = mx.array(np.asarray(projection, dtype=np.float32))
    result = embed[tokens] @ mx.transpose(project)
    mx.eval(result)
    return np.asarray(result, dtype=np.float64)


def confidence_schedule(
    block_hidden: np.ndarray,
    previous_tokens: Sequence[int] | np.ndarray,
    markov_embedding: np.ndarray,
    projection: Sequence[float] | np.ndarray,
) -> np.ndarray:
    mx = _mlx()
    hidden = mx.array(np.asarray(block_hidden, dtype=np.float32))
    tokens = mx.array(np.asarray(previous_tokens, dtype=np.int32))
    embed = mx.array(np.asarray(markov_embedding, dtype=np.float32))
    weights = mx.array(np.asarray(projection, dtype=np.float32).reshape(-1))
    features = mx.concatenate((hidden, embed[tokens]), axis=1)
    probabilities = mx.sigmoid(features @ weights)
    mx.eval(probabilities)
    return np.asarray(probabilities, dtype=np.float64)


def post_layer_hc_mean(hidden_states: np.ndarray) -> np.ndarray:
    """Run the exact four-lane DSpark capture primitive on MLX Metal."""

    mx = _mlx()
    source = np.asarray(hidden_states)
    if source.ndim != 3 or source.shape[0] == 0 or source.shape[1:] != (4, 4096):
        raise ValueError("hidden_states must have shape [token, 4, 4096]")
    if not np.all(np.isfinite(source)):
        raise ValueError("hidden_states must contain only finite values")
    hidden = mx.array(source.astype(np.float32, copy=False))
    result = mx.mean(hidden, axis=1)
    mx.eval(result)
    return np.asarray(result, dtype=np.float64)


def capture_target_hidden_rows(
    layer_hidden_states: Sequence[np.ndarray],
    layer_ids: Sequence[int],
    *,
    phase: str,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Cross-check frontier selection and retained history on MLX."""

    if tuple(layer_ids) != DSPARK_TARGET_LAYER_IDS:
        raise ValueError("target capture layers must be ordered 40, 41, 42")
    states = tuple(layer_hidden_states)
    if len(states) != 3:
        raise ValueError("target capture requires exactly three layer tensors")
    reduced = tuple(post_layer_hc_mean(state) for state in states)
    token_counts = {item.shape[0] for item in reduced}
    if len(token_counts) != 1:
        raise ValueError("target capture layers must have the same token count")
    token_count = next(iter(token_counts))
    if phase == "decode":
        if token_count != 1:
            raise ValueError("decode target capture requires exactly one token row")
        token_index = 0
    elif phase == "prefill":
        token_index = token_count - 1
    else:
        raise ValueError("target capture phase must be 'decode' or 'prefill'")
    history_start = max(0, token_count - 128)
    history = np.stack(
        [item[history_start:] for item in reduced], axis=1
    )
    return (
        np.stack([item[token_index] for item in reduced], axis=0),
        token_index,
        history,
    )


def main_project_and_norm(
    target_hidden: np.ndarray,
    projection: np.ndarray,
    norm_weight: Sequence[float] | np.ndarray,
    *,
    eps: float = 1.0e-6,
) -> np.ndarray:
    """Run capture concatenation, main projection, and RMSNorm on MLX."""

    mx = _mlx()
    target = mx.array(np.asarray(target_hidden, dtype=np.float32))
    project = mx.array(np.asarray(projection, dtype=np.float32))
    weight = mx.array(np.asarray(norm_weight, dtype=np.float32))
    flattened = mx.reshape(target, (*target.shape[:-2], -1))
    projected = flattened @ mx.transpose(project)
    variance = mx.mean(mx.square(projected), axis=-1, keepdims=True)
    result = projected * mx.rsqrt(variance + float(eps)) * weight
    mx.eval(result)
    return np.asarray(result, dtype=np.float64)


def _mlx_bfloat16_boundary(mx: object, value: object) -> object:
    """Publish a BF16 boundary and reopen it as float32 for the next op."""

    return value.astype(mx.bfloat16).astype(mx.float32)


def _mlx_rms_norm_bfloat16(
    mx: object,
    value: object,
    weight: object,
    *,
    eps: float,
) -> object:
    variance = mx.mean(mx.square(value), axis=-1, keepdims=True)
    return _mlx_bfloat16_boundary(
        mx, value * mx.rsqrt(variance + float(eps)) * weight
    )


def _mlx_q_head_norm_bfloat16(
    mx: object,
    value: object,
    *,
    eps: float,
) -> object:
    """Keep the pinned per-head Q expression at every BF16 boundary."""

    squared = _mlx_bfloat16_boundary(mx, mx.square(value))
    mean = _mlx_bfloat16_boundary(
        mx, mx.mean(squared, axis=-1, keepdims=True)
    )
    added = _mlx_bfloat16_boundary(mx, mean + float(eps))
    inverse_rms = _mlx_bfloat16_boundary(
        mx, mx.rsqrt(added)
    )
    return _mlx_bfloat16_boundary(mx, value * inverse_rms)


def _mlx_rope_tail_bfloat16(
    mx: object,
    value: object,
    positions: object,
    *,
    inverse: bool,
    rope_theta: float,
) -> object:
    """Rotate the final 64 lanes on MLX and reopen the BF16 tail."""

    tail = mx.reshape(
        value[..., -DSPARK_ROPE_WIDTH:],
        (*value.shape[:-1], DSPARK_ROPE_WIDTH // 2, 2),
    )
    frequency = 1.0 / mx.power(
        mx.array(float(rope_theta), dtype=mx.float32),
        mx.arange(0, DSPARK_ROPE_WIDTH, 2, dtype=mx.float32)
        / float(DSPARK_ROPE_WIDTH),
    )
    angle_shape = (
        (positions.shape[0],)
        + (1,) * (value.ndim - 2)
        + (DSPARK_ROPE_WIDTH // 2,)
    )
    angles = mx.reshape(positions[:, None] * frequency[None, :], angle_shape)
    cosine = mx.cos(angles)
    sine = mx.sin(angles)
    if inverse:
        sine = -sine
    first = tail[..., 0] * cosine - tail[..., 1] * sine
    second = tail[..., 0] * sine + tail[..., 1] * cosine
    rotated = _mlx_bfloat16_boundary(
        mx,
        mx.reshape(
            mx.stack((first, second), axis=-1),
            (*value.shape[:-1], DSPARK_ROPE_WIDTH),
        ),
    )
    return mx.concatenate(
        (value[..., :-DSPARK_ROPE_WIDTH], rotated), axis=-1
    )


def _mlx_round_e4m3fn(mx: object, value: object) -> object:
    """Round on MLX with clamp and nearest-even tie handling."""

    codebook = mx.array(_E4M3FN_POSITIVE_VALUES, dtype=mx.float32)
    preferences = mx.array(_E4M3FN_EVEN_CODE_PREFERENCE, dtype=mx.int32)
    magnitude = mx.minimum(mx.abs(value), mx.array(448.0, dtype=mx.float32))
    distances = mx.abs(magnitude[..., None] - codebook)
    minimum = mx.min(distances, axis=-1, keepdims=True)
    candidates = distances == minimum
    scores = candidates.astype(mx.int32) * preferences
    indices = mx.argmax(scores, axis=-1)
    rounded = mx.take(codebook, indices)
    return mx.where(value < 0.0, -rounded, rounded)


def direct_stage_context_kv(
    main_x: np.ndarray,
    projection: np.ndarray,
    norm_weight: Sequence[float] | np.ndarray,
    absolute_positions: Sequence[int] | np.ndarray,
    *,
    eps: float = 1.0e-6,
    rope_theta: float = 10000.0,
) -> MLXStageContextKV:
    """Cross-check the final raw-context KV boundaries on MLX Metal.

    This function deliberately re-expresses the finalizer instead of calling
    the NumPy reference.  It covers the post-Wkv BF16 boundary, RMSNorm BF16,
    RoPE on only the final 64 dimensions, seven independent 64-wide E4M3FN /
    UE8M0 groups over the 448-wide prefix, FP32 dequantization, and the final
    BF16 store.
    """

    main_source = np.asarray(main_x)
    project_source = np.asarray(projection)
    weight_source = np.asarray(norm_weight)
    positions_source = np.asarray(absolute_positions)
    if main_source.ndim != 2 or main_source.shape[0] == 0:
        raise ValueError("main_x must be a non-empty matrix")
    if project_source.shape != (DSPARK_RAW_CACHE_WIDTH, main_source.shape[1]):
        raise ValueError("context KV projection must have shape [512, hidden]")
    if weight_source.shape != (DSPARK_RAW_CACHE_WIDTH,):
        raise ValueError("context KV norm weight must have shape [512]")
    if (positions_source.ndim != 1 or
            positions_source.shape[0] != main_source.shape[0]):
        raise ValueError("absolute_positions must match main_x")
    if not np.issubdtype(positions_source.dtype, np.integer):
        raise ValueError("absolute_positions must contain integers")
    if np.any(positions_source < 0):
        raise ValueError("absolute_positions must be non-negative")
    if not all(np.all(np.isfinite(item)) for item in
               (main_source, project_source, weight_source)):
        raise ValueError("context KV inputs must be finite")
    if not np.isfinite(float(eps)) or eps <= 0.0:
        raise ValueError("context KV norm epsilon must be finite and positive")
    if not np.isfinite(float(rope_theta)) or rope_theta <= 0.0:
        raise ValueError("rope_theta must be finite and positive")

    mx = _mlx()
    main = mx.array(main_source.astype(np.float32, copy=False))
    project = mx.array(project_source.astype(np.float32, copy=False))
    weights = mx.array(weight_source.astype(np.float32, copy=False))
    positions = mx.array(positions_source.astype(np.float32, copy=False))

    projected = _mlx_bfloat16_boundary(mx, main @ mx.transpose(project))
    variance = mx.mean(mx.square(projected), axis=-1, keepdims=True)
    normalized = _mlx_bfloat16_boundary(
        mx, projected * mx.rsqrt(variance + float(eps)) * weights
    )

    tail = mx.reshape(normalized[:, 448:], (main_source.shape[0], 32, 2))
    frequency = 1.0 / mx.power(
        mx.array(float(rope_theta), dtype=mx.float32),
        mx.arange(0, 64, 2, dtype=mx.float32) / 64.0,
    )
    angles = positions[:, None] * frequency[None, :]
    cosine = mx.cos(angles)
    sine = mx.sin(angles)
    first = tail[..., 0] * cosine - tail[..., 1] * sine
    second = tail[..., 0] * sine + tail[..., 1] * cosine
    roped_tail = _mlx_bfloat16_boundary(
        mx, mx.reshape(mx.stack((first, second), axis=-1),
                       (main_source.shape[0], 64))
    )
    roped = mx.concatenate((normalized[:, :448], roped_tail), axis=1)

    grouped = mx.reshape(roped[:, :448], (main_source.shape[0], 7, 64))
    amax = mx.maximum(
        mx.max(mx.abs(grouped), axis=2),
        mx.array(1.0e-4, dtype=mx.float32),
    )
    scales = mx.power(
        mx.array(2.0, dtype=mx.float32),
        mx.ceil(mx.log2(amax / mx.array(448.0, dtype=mx.float32))),
    ).astype(mx.float32)
    quantized = _mlx_round_e4m3fn(mx, grouped / scales[..., None])
    stored_prefix = _mlx_bfloat16_boundary(
        mx, quantized.astype(mx.float32) * scales[..., None]
    )
    stored = mx.concatenate(
        (mx.reshape(stored_prefix, (main_source.shape[0], 448)), roped_tail),
        axis=1,
    )
    mx.eval(projected, normalized, roped, stored, scales)
    return MLXStageContextKV(
        np.array(positions_source, dtype=np.int64, copy=True),
        np.asarray(projected, dtype=np.float64),
        np.asarray(normalized, dtype=np.float64),
        np.asarray(roped, dtype=np.float64),
        np.asarray(stored, dtype=np.float64),
        np.asarray(scales, dtype=np.float64),
    )


def dspark_attention_official(
    queries: np.ndarray,
    committed_ring: np.ndarray,
    committed_count: int,
    draft_rows: np.ndarray,
    attention_sinks: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """Cross-check pinned physical-order sparse attention on MLX Metal.

    ``committed_ring`` is one stage's physical 128-by-512 ring.  The first
    ``committed_count`` physical slots are followed by five transient rows.
    The online 64-row softmax and every BF16 boundary are re-expressed here
    without importing the NumPy reference implementation.
    """

    query_source = np.asarray(queries)
    ring_source = np.asarray(committed_ring)
    draft_source = np.asarray(draft_rows)
    sink_source = np.asarray(attention_sinks)
    if query_source.shape != (
        DSPARK_PROPOSAL_ROWS, DSPARK_ATTENTION_HEADS, DSPARK_RAW_CACHE_WIDTH
    ):
        raise ValueError("queries must have shape [5, 64, 512]")
    if ring_source.shape != (DSPARK_RAW_CACHE_WINDOW, DSPARK_RAW_CACHE_WIDTH):
        raise ValueError("committed_ring must have shape [128, 512]")
    if (isinstance(committed_count, bool) or
            not isinstance(committed_count, (int, np.integer)) or
            int(committed_count) < 2 or
            int(committed_count) > DSPARK_RAW_CACHE_WINDOW):
        raise ValueError("committed_count must be inside [2, 128]")
    if draft_source.shape != (DSPARK_PROPOSAL_ROWS, DSPARK_RAW_CACHE_WIDTH):
        raise ValueError("draft_rows must have shape [5, 512]")
    if sink_source.shape != (DSPARK_ATTENTION_HEADS,):
        raise ValueError("attention_sinks must have shape [64]")
    if not all(np.all(np.isfinite(item)) for item in
               (query_source, ring_source, draft_source, sink_source)):
        raise ValueError("DSpark attention inputs must be finite")
    mx = _mlx()
    query = _mlx_bfloat16_boundary(
        mx, mx.array(query_source.astype(np.float32, copy=False))
    )
    ring = _mlx_bfloat16_boundary(
        mx, mx.array(ring_source.astype(np.float32, copy=False))
    )
    draft = _mlx_bfloat16_boundary(
        mx, mx.array(draft_source.astype(np.float32, copy=False))
    )
    sinks = mx.array(sink_source.astype(np.float32, copy=False))
    physical = mx.concatenate((ring[:int(committed_count)], draft), axis=0)
    running_max = mx.full(
        (DSPARK_PROPOSAL_ROWS, DSPARK_ATTENTION_HEADS),
        -float("inf"), dtype=mx.float32,
    )
    denominator = mx.zeros_like(running_max)
    numerator = mx.zeros(
        (DSPARK_PROPOSAL_ROWS,
         DSPARK_ATTENTION_HEADS,
         DSPARK_RAW_CACHE_WIDTH),
        dtype=mx.float32,
    )
    flat_query = mx.reshape(
        query,
        (DSPARK_PROPOSAL_ROWS * DSPARK_ATTENTION_HEADS,
         DSPARK_RAW_CACHE_WIDTH),
    )
    n_keys = int(committed_count) + DSPARK_PROPOSAL_ROWS
    for first in range(0, n_keys, DSPARK_ATTENTION_BLOCK):
        block = physical[first:min(first + DSPARK_ATTENTION_BLOCK, n_keys)]
        scores = mx.reshape(
            (flat_query @ mx.transpose(block))
            * float(DSPARK_RAW_CACHE_WIDTH ** -0.5),
            (DSPARK_PROPOSAL_ROWS, DSPARK_ATTENTION_HEADS, block.shape[0]),
        )
        previous_max = running_max
        running_max = mx.maximum(previous_max, mx.max(scores, axis=-1))
        previous_scale = mx.exp(previous_max - running_max)
        weights = mx.exp(scores - running_max[..., None])
        denominator = (
            denominator * previous_scale + mx.sum(weights, axis=-1)
        )
        rounded_weights = _mlx_bfloat16_boundary(mx, weights)
        block_numerator = mx.reshape(
            mx.reshape(
                rounded_weights,
                (DSPARK_PROPOSAL_ROWS * DSPARK_ATTENTION_HEADS,
                 block.shape[0]),
            ) @ block,
            (DSPARK_PROPOSAL_ROWS,
             DSPARK_ATTENTION_HEADS,
             DSPARK_RAW_CACHE_WIDTH),
        )
        numerator = (
            numerator * previous_scale[..., None] + block_numerator
        )
    denominator = denominator + mx.exp(sinks[None, :] - running_max)
    output = _mlx_bfloat16_boundary(
        mx, numerator / denominator[..., None]
    )
    mx.eval(output)
    return np.asarray(output, dtype=np.float64)


def stage_zero_attention_half(
    hidden_input: np.ndarray,
    hc_function_weight: np.ndarray,
    hc_scale: Sequence[float] | np.ndarray,
    hc_base: Sequence[float] | np.ndarray,
    attention_norm_weight: Sequence[float] | np.ndarray,
    q_a_weight: np.ndarray,
    q_a_norm_weight: Sequence[float] | np.ndarray,
    q_b_weight: np.ndarray,
    kv_weight: np.ndarray,
    kv_norm_weight: Sequence[float] | np.ndarray,
    absolute_positions: Sequence[int] | np.ndarray,
    physical_ring_rows: np.ndarray,
    committed_count: int,
    committed_token_start: int,
    other_stage_draft_rows: np.ndarray,
    attention_sinks: Sequence[float] | np.ndarray,
    output_a_weight: np.ndarray,
    output_b_weight: np.ndarray,
    *,
    norm_eps: float = 1.0e-6,
    hc_eps: float = 1.0e-6,
    hc_iterations: int = 20,
    rope_theta: float = 10000.0,
) -> MLXStageZeroAttentionHalf:
    """Re-express the compact final-geometry stage-zero seam on MLX Metal."""

    hidden_source = np.asarray(hidden_input, dtype=np.float32)
    positions_source = np.asarray(absolute_positions)
    hc_function_source = np.asarray(hc_function_weight, dtype=np.float32)
    hc_scale_source = np.asarray(hc_scale, dtype=np.float32)
    hc_base_source = np.asarray(hc_base, dtype=np.float32)
    attention_norm_source = np.asarray(
        attention_norm_weight, dtype=np.float32
    )
    q_a_source = np.asarray(q_a_weight, dtype=np.float32)
    q_a_norm_source = np.asarray(q_a_norm_weight, dtype=np.float32)
    q_b_source = np.asarray(q_b_weight, dtype=np.float32)
    kv_source = np.asarray(kv_weight, dtype=np.float32)
    kv_norm_source = np.asarray(kv_norm_weight, dtype=np.float32)
    output_a_source = np.asarray(output_a_weight, dtype=np.float32)
    output_b_source = np.asarray(output_b_weight, dtype=np.float32)
    ring_source = np.asarray(physical_ring_rows, dtype=np.float32)
    transient_source = np.asarray(other_stage_draft_rows, dtype=np.float32)
    sinks_source = np.asarray(attention_sinks, dtype=np.float32)
    if hidden_source.ndim != 3 or hidden_source.shape[:2] != (5, 4):
        raise ValueError("stage-zero hidden must have shape [5, 4, hidden]")
    if hidden_source.shape[2] == 0 or not np.all(np.isfinite(hidden_source)):
        raise ValueError("stage-zero hidden must be finite and non-empty")
    if (positions_source.shape != (5,) or
            not np.issubdtype(positions_source.dtype, np.integer) or
            np.any(positions_source < 0) or np.any(np.diff(positions_source) != 1)):
        raise ValueError("absolute_positions must be five consecutive integers")
    if (isinstance(committed_token_start, bool) or
            not isinstance(committed_token_start, (int, np.integer)) or
            int(committed_token_start) < 0):
        raise ValueError("committed_token_start must be non-negative")
    if (isinstance(committed_count, bool) or
            not isinstance(committed_count, (int, np.integer)) or
            int(committed_count) < 2 or int(committed_count) > 128):
        raise ValueError("committed_count must be inside [2, 128]")
    if int(committed_count) < 128 and int(committed_token_start) != 0:
        raise ValueError("partial official raw cache must start at position zero")
    if positions_source[0] != int(committed_token_start) + committed_count:
        raise ValueError("absolute_positions must start at committed cache end")
    if ring_source.shape != (DSPARK_RAW_CACHE_WINDOW, DSPARK_RAW_CACHE_WIDTH):
        raise ValueError("physical_ring_rows must have shape [128, 512]")
    if transient_source.shape != (3, 5, DSPARK_RAW_CACHE_WIDTH):
        raise ValueError("other_stage_draft_rows must have shape [3, 5, 512]")
    hidden_width = hidden_source.shape[2]
    q_rank = q_a_source.shape[0] if q_a_source.ndim == 2 else 0
    output_rank = output_a_source.shape[1] if output_a_source.ndim == 3 else 0
    if hc_function_source.shape != (24, 4 * hidden_width):
        raise ValueError("HC function weight must have shape [24, 4 * hidden]")
    if hc_scale_source.shape != (3,) or hc_base_source.shape != (24,):
        raise ValueError("HC scale/base geometry mismatch")
    if attention_norm_source.shape != (hidden_width,):
        raise ValueError("attention norm weight must match hidden width")
    if q_a_source.shape != (q_rank, hidden_width) or q_rank == 0:
        raise ValueError("q_a weight must have shape [q_rank, hidden]")
    if q_a_norm_source.shape != (q_rank,):
        raise ValueError("q_a norm weight must match q_rank")
    if q_b_source.shape != (
        DSPARK_ATTENTION_HEADS * DSPARK_RAW_CACHE_WIDTH, q_rank
    ):
        raise ValueError("q_b weight must have shape [64 * 512, q_rank]")
    if kv_source.shape != (DSPARK_RAW_CACHE_WIDTH, hidden_width):
        raise ValueError("KV weight must have shape [512, hidden]")
    if kv_norm_source.shape != (DSPARK_RAW_CACHE_WIDTH,):
        raise ValueError("KV norm weight must have shape [512]")
    if sinks_source.shape != (DSPARK_ATTENTION_HEADS,):
        raise ValueError("attention_sinks must have shape [64]")
    if output_a_source.shape != (
        DSPARK_OUTPUT_GROUPS,
        output_rank,
        (DSPARK_ATTENTION_HEADS // DSPARK_OUTPUT_GROUPS)
        * DSPARK_RAW_CACHE_WIDTH,
    ) or output_rank < 2:
        raise ValueError(
            "output_a weight must have shape [8, output_rank>=2, 8 * 512]"
        )
    if output_b_source.shape != (
        hidden_width, DSPARK_OUTPUT_GROUPS * output_rank
    ):
        raise ValueError("output_b weight must have shape [hidden, 8 * output_rank]")
    if not np.isfinite(float(norm_eps)) or norm_eps <= 0.0:
        raise ValueError("norm_eps must be finite and positive")
    if not np.isfinite(float(hc_eps)) or hc_eps <= 0.0:
        raise ValueError("hc_eps must be finite and positive")
    if (isinstance(hc_iterations, bool) or
            not isinstance(hc_iterations, (int, np.integer)) or
            int(hc_iterations) < 1):
        raise ValueError("hc_iterations must be a positive integer")
    if not np.isfinite(float(rope_theta)) or rope_theta <= 0.0:
        raise ValueError("rope_theta must be finite and positive")
    if not all(np.all(np.isfinite(item)) for item in (
        hc_function_source, hc_scale_source, hc_base_source,
        attention_norm_source, q_a_source, q_a_norm_source, q_b_source,
        kv_source, kv_norm_source, ring_source, transient_source,
        sinks_source, output_a_source, output_b_source,
    )):
        raise ValueError("stage-zero attention weights and inputs must be finite")

    mx = _mlx()
    hidden_boundary = _mlx_bfloat16_boundary(
        mx, mx.array(hidden_source)
    )
    mx.eval(hidden_boundary)
    hidden_boundary_host = np.asarray(hidden_boundary, dtype=np.float32)
    reduced, _pre, post, combination = hc_pre(
        hidden_boundary_host,
        hc_function_source,
        hc_scale_source,
        hc_base_source,
        norm_eps=norm_eps,
        hc_eps=hc_eps,
        iterations=hc_iterations,
    )
    positions = mx.array(positions_source.astype(np.float32, copy=False))
    hc_pre_output = _mlx_bfloat16_boundary(
        mx, mx.array(np.asarray(reduced, dtype=np.float32))
    )
    attention_normalized = _mlx_rms_norm_bfloat16(
        mx,
        hc_pre_output,
        mx.array(attention_norm_source),
        eps=norm_eps,
    )
    q_a = _mlx_bfloat16_boundary(
        mx, attention_normalized @ mx.transpose(mx.array(q_a_source))
    )
    q_a_normalized = _mlx_rms_norm_bfloat16(
        mx,
        q_a,
        mx.array(q_a_norm_source),
        eps=norm_eps,
    )
    q_b = _mlx_bfloat16_boundary(
        mx, q_a_normalized @ mx.transpose(mx.array(q_b_source))
    )
    q_b = mx.reshape(
        q_b,
        (DSPARK_PROPOSAL_ROWS, DSPARK_ATTENTION_HEADS,
         DSPARK_RAW_CACHE_WIDTH),
    )
    q_head_normalized = _mlx_q_head_norm_bfloat16(
        mx, q_b, eps=norm_eps
    )
    q_roped = _mlx_rope_tail_bfloat16(
        mx,
        q_head_normalized,
        positions,
        inverse=False,
        rope_theta=rope_theta,
    )
    mx.eval(
        hc_pre_output, attention_normalized, q_a, q_a_normalized,
        q_b, q_head_normalized, q_roped,
    )

    context = direct_stage_context_kv(
        np.asarray(attention_normalized, dtype=np.float32),
        kv_source,
        kv_norm_source,
        positions_source,
        eps=norm_eps,
        rope_theta=rope_theta,
    )
    draft_rows = np.array(transient_source, copy=True)
    draft_rows[0] = context.stored.astype(np.float32)
    attention_host = dspark_attention_official(
        np.asarray(q_roped, dtype=np.float32),
        ring_source,
        committed_count,
        draft_rows[0],
        sinks_source,
    )
    attention_output = mx.array(attention_host.astype(np.float32))
    inverse_roped = _mlx_rope_tail_bfloat16(
        mx,
        attention_output,
        positions,
        inverse=True,
        rope_theta=rope_theta,
    )
    grouped = mx.reshape(
        inverse_roped,
        (DSPARK_PROPOSAL_ROWS,
         DSPARK_OUTPUT_GROUPS,
         (DSPARK_ATTENTION_HEADS // DSPARK_OUTPUT_GROUPS)
         * DSPARK_RAW_CACHE_WIDTH),
    )
    output_a_weights = mx.array(output_a_source)
    output_a_parts = []
    for group in range(DSPARK_OUTPUT_GROUPS):
        output_a_parts.append(
            grouped[:, group] @ mx.transpose(output_a_weights[group])
        )
    output_a = _mlx_bfloat16_boundary(
        mx, mx.stack(output_a_parts, axis=1)
    )
    output_b = _mlx_bfloat16_boundary(
        mx,
        mx.reshape(output_a, (DSPARK_PROPOSAL_ROWS, -1))
        @ mx.transpose(mx.array(output_b_source)),
    )
    mx.eval(inverse_roped, output_a, output_b)
    post_host = hc_post(
        np.asarray(output_b, dtype=np.float32),
        hidden_boundary_host,
        post,
        combination,
    )
    hc_post_output = _mlx_bfloat16_boundary(
        mx, mx.array(post_host.astype(np.float32))
    )
    mx.eval(hc_post_output)

    def host(value: object) -> np.ndarray:
        return np.asarray(value, dtype=np.float64)

    return MLXStageZeroAttentionHalf(
        np.array(positions_source, dtype=np.int64, copy=True),
        host(hidden_boundary),
        host(hc_pre_output),
        host(attention_normalized),
        host(q_a),
        host(q_a_normalized),
        host(q_b),
        host(q_head_normalized),
        host(q_roped),
        context.projected,
        context.normalized,
        context.roped,
        context.stored,
        context.nonrope_scales,
        np.asarray(attention_host, dtype=np.float64),
        host(inverse_roped),
        host(output_a),
        host(output_b),
        host(hc_post_output),
    )


def _opened_payload_matrix(value: object, name: str) -> np.ndarray:
    opened = getattr(value, "dequantized", value)
    if callable(opened):
        opened = opened()
    matrix = np.asarray(opened, dtype=np.float32)
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must decode to one finite matrix")
    return matrix


def stage_ffn_moe_payload_first(
    hidden_input: np.ndarray,
    hc_function_weight: object,
    hc_scale: Sequence[float] | np.ndarray,
    hc_base: Sequence[float] | np.ndarray,
    ffn_norm_weight: Sequence[float] | np.ndarray,
    router_weight: object,
    selection_bias: Sequence[float] | np.ndarray,
    shared_gate_weight: object,
    shared_up_weight: object,
    shared_down_weight: object,
    routed_expert_weights: dict[int, dict[str, object]],
    *,
    norm_eps: float = 1.0e-6,
    hc_eps: float = 1.0e-6,
    hc_iterations: int = 20,
    swiglu_clamp: float = 10.0,
) -> MLXStageFFNMoE:
    """Independently re-express the official FFN dtype schedule on MLX."""

    hidden_source = np.asarray(hidden_input, dtype=np.float32)
    if hidden_source.shape != (5, 4, 4096):
        raise ValueError("FFN hidden input must have shape [5, 4, 4096]")
    function_source = _opened_payload_matrix(
        hc_function_weight, "FFN HC function"
    )
    norm_source = np.asarray(ffn_norm_weight, dtype=np.float32)
    router_source = _opened_payload_matrix(router_weight, "router weight")
    bias_source = np.asarray(selection_bias, dtype=np.float32)
    shared_gate_source = _opened_payload_matrix(
        shared_gate_weight, "shared gate"
    )
    shared_up_source = _opened_payload_matrix(shared_up_weight, "shared up")
    shared_down_source = _opened_payload_matrix(
        shared_down_weight, "shared down"
    )
    if (function_source.shape != (24, 4 * 4096) or
            norm_source.shape != (4096,) or
            router_source.shape != (256, 4096) or
            bias_source.shape != (256,) or
            shared_gate_source.shape != (256, 4096) or
            shared_up_source.shape != (256, 4096) or
            shared_down_source.shape != (4096, 256)):
        raise ValueError("FFN fixture weight geometry mismatch")
    if not all(np.all(np.isfinite(item)) for item in (
        hidden_source, function_source, norm_source, router_source,
        bias_source, shared_gate_source, shared_up_source, shared_down_source,
    )):
        raise ValueError("FFN fixture inputs must be finite")
    if (not np.isfinite(float(swiglu_clamp)) or swiglu_clamp < 0.0):
        raise ValueError("SwiGLU clamp must be finite and non-negative")

    mx = _mlx()

    def bf16(value: object) -> object:
        return _mlx_bfloat16_boundary(mx, value)

    def swiglu(gate: object, up: object) -> object:
        if swiglu_clamp > 1.0e-6:
            gate = mx.minimum(gate, float(swiglu_clamp))
            up = mx.clip(up, -float(swiglu_clamp), float(swiglu_clamp))
        return gate * mx.sigmoid(gate) * up

    hidden = bf16(mx.array(hidden_source))
    mx.eval(hidden)
    hidden_host = np.asarray(hidden, dtype=np.float32)
    reduced_host, _pre, post, combination = hc_pre(
        hidden_host,
        function_source,
        hc_scale,
        hc_base,
        norm_eps=norm_eps,
        hc_eps=hc_eps,
        iterations=hc_iterations,
    )
    hc_pre_output = bf16(mx.array(reduced_host.astype(np.float32)))
    ffn_normalized = _mlx_rms_norm_bfloat16(
        mx,
        hc_pre_output,
        mx.array(norm_source),
        eps=norm_eps,
    )
    router_logits = (
        ffn_normalized.astype(mx.float32)
        @ mx.transpose(mx.array(router_source, dtype=mx.float32))
    )
    router_probabilities = mx.sqrt(
        mx.maximum(router_logits, 0.0)
        + mx.log1p(mx.exp(-mx.abs(router_logits)))
    )
    mx.eval(ffn_normalized, router_logits, router_probabilities)
    raw_probabilities_host = np.asarray(
        router_probabilities, dtype=np.float32
    )
    expert_ids = np.arange(256, dtype=np.int32)
    probabilities_host = np.zeros((5, 256), dtype=np.float32)
    selected = np.full((5, 6), -1, dtype=np.int32)
    route_weights = np.zeros((5, 6), dtype=np.float32)
    for row in range(5):
        logits_host = np.asarray(router_logits[row], dtype=np.float32)
        row_probabilities = raw_probabilities_host[row]
        with np.errstate(over="ignore", invalid="ignore"):
            row_selection = row_probabilities + bias_source
        if (not np.all(np.isfinite(logits_host)) or
                not np.all(np.isfinite(row_probabilities)) or
                not np.all(np.isfinite(row_selection))):
            continue
        row_selected = np.lexsort(
            (expert_ids, -row_selection)
        )[:6]
        unbiased = row_probabilities[row_selected]
        denominator = np.sum(unbiased, dtype=np.float32)
        if not np.isfinite(denominator) or not denominator > 0.0:
            continue
        probabilities_host[row] = row_probabilities
        selected[row] = row_selected
        route_weights[row] = (
            unbiased / denominator * np.float32(1.5)
        )
    router_probabilities = mx.array(probabilities_host, dtype=mx.float32)

    normalized_host = np.asarray(ffn_normalized, dtype=np.float32)
    shared_gate = bf16(
        ffn_normalized @ mx.transpose(mx.array(shared_gate_source))
    )
    shared_up = bf16(
        ffn_normalized @ mx.transpose(mx.array(shared_up_source))
    )
    shared_mid = bf16(swiglu(shared_gate, shared_up))
    shared_down = bf16(
        shared_mid @ mx.transpose(mx.array(shared_down_source))
    )
    mx.eval(shared_gate, shared_up, shared_mid, shared_down)

    routed_gate = np.empty((5, 6, 256), dtype=np.float32)
    routed_up = np.empty_like(routed_gate)
    routed_mid = np.empty_like(routed_gate)
    routed_down = np.empty((5, 6, 4096), dtype=np.float32)
    for row in range(5):
        x = mx.array(normalized_host[row])
        for slot in range(6):
            expert = int(selected[row, slot])
            record = routed_expert_weights.get(expert)
            if record is None or set(record) != {"gate", "up", "down"}:
                raise ValueError(f"missing complete payload for expert {expert}")
            gate_matrix = _opened_payload_matrix(record["gate"], "routed gate")
            up_matrix = _opened_payload_matrix(record["up"], "routed up")
            if gate_matrix.shape != (256, 4096) or up_matrix.shape != (256, 4096):
                raise ValueError("routed gate/up compact geometry mismatch")
            gate = bf16(mx.array(gate_matrix) @ x)
            up = bf16(mx.array(up_matrix) @ x)
            mid = bf16(
                swiglu(gate, up) * float(route_weights[row, slot])
            )
            mx.eval(gate, up, mid)
            routed_gate[row, slot] = np.asarray(gate, dtype=np.float32)
            routed_up[row, slot] = np.asarray(up, dtype=np.float32)
            routed_mid[row, slot] = np.asarray(mid, dtype=np.float32)
            del gate_matrix, up_matrix
            down_matrix = _opened_payload_matrix(record["down"], "routed down")
            if down_matrix.shape != (4096, 256):
                raise ValueError("routed down compact geometry mismatch")
            down = bf16(mx.array(down_matrix) @ mid)
            mx.eval(down)
            routed_down[row, slot] = np.asarray(down, dtype=np.float32)

    routed_sum_rows = []
    for row in range(5):
        value = mx.zeros((4096,), dtype=mx.float32)
        for slot in np.argsort(selected[row], kind="stable"):
            value = value + mx.array(routed_down[row, slot])
        routed_sum_rows.append(value)
    routed_sum = mx.stack(routed_sum_rows)
    moe_output = bf16(routed_sum + shared_down)
    mx.eval(routed_sum, moe_output)
    post_host = hc_post(
        np.asarray(moe_output, dtype=np.float32),
        hidden_host,
        post,
        combination,
    )
    hc_post_output = bf16(mx.array(post_host.astype(np.float32)))
    mx.eval(hc_post_output)

    def host(value: object) -> np.ndarray:
        return np.asarray(value, dtype=np.float64)

    return MLXStageFFNMoE(
        host(hidden),
        host(hc_pre_output),
        host(ffn_normalized),
        host(router_logits),
        host(router_probabilities),
        np.array(selected, copy=True),
        route_weights.astype(np.float64),
        host(shared_gate),
        host(shared_up),
        host(shared_mid),
        host(shared_down),
        routed_gate.astype(np.float64),
        routed_up.astype(np.float64),
        routed_mid.astype(np.float64),
        routed_down.astype(np.float64),
        host(routed_sum),
        host(moe_output),
        host(hc_post_output),
    )


def hc_split_sinkhorn(
    mixes: np.ndarray,
    scale: Sequence[float] | np.ndarray,
    base: Sequence[float] | np.ndarray,
    *,
    iterations: int = 20,
    eps: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the four-lane HC split and Sinkhorn normalization on MLX."""

    mx = _mlx()
    values = mx.array(np.asarray(mixes, dtype=np.float32))
    scales = mx.array(np.asarray(scale, dtype=np.float32))
    bases = mx.array(np.asarray(base, dtype=np.float32))
    pre = mx.sigmoid(values[..., :4] * scales[0] + bases[:4]) + float(eps)
    post = 2.0 * mx.sigmoid(
        values[..., 4:8] * scales[1] + bases[4:8]
    )
    combination = mx.reshape(values[..., 8:], (*values.shape[:-1], 4, 4))
    combination = combination * scales[2] + mx.reshape(bases[8:], (4, 4))
    combination = combination - mx.max(combination, axis=-1, keepdims=True)
    combination = mx.exp(combination)
    combination = (
        combination / mx.sum(combination, axis=-1, keepdims=True)
        + float(eps)
    )
    combination = combination / (
        mx.sum(combination, axis=-2, keepdims=True) + float(eps)
    )
    for _ in range(iterations - 1):
        combination = combination / (
            mx.sum(combination, axis=-1, keepdims=True) + float(eps)
        )
        combination = combination / (
            mx.sum(combination, axis=-2, keepdims=True) + float(eps)
        )
    mx.eval(pre, post, combination)
    return tuple(
        np.asarray(item, dtype=np.float64)
        for item in (pre, post, combination)
    )


def hc_pre(
    hidden_states: np.ndarray,
    function_weight: np.ndarray,
    scale: Sequence[float] | np.ndarray,
    base: Sequence[float] | np.ndarray,
    *,
    norm_eps: float = 1.0e-6,
    hc_eps: float = 1.0e-6,
    iterations: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mx = _mlx()
    hidden = mx.array(np.asarray(hidden_states, dtype=np.float32))
    function = mx.array(np.asarray(function_weight, dtype=np.float32))
    flattened = mx.reshape(hidden, (*hidden.shape[:-2], -1))
    inverse_rms = mx.rsqrt(
        mx.mean(mx.square(flattened), axis=-1, keepdims=True)
        + float(norm_eps)
    )
    mixes = (flattened @ mx.transpose(function)) * inverse_rms
    mx.eval(mixes)
    pre, post, combination = hc_split_sinkhorn(
        np.asarray(mixes),
        scale,
        base,
        iterations=iterations,
        eps=hc_eps,
    )
    pre_mx = mx.array(np.asarray(pre, dtype=np.float32))
    reduced = mx.sum(pre_mx[..., :, None] * hidden, axis=-2)
    mx.eval(reduced)
    return (
        np.asarray(reduced, dtype=np.float64),
        pre,
        post,
        combination,
    )


def hc_post(
    branch_output: np.ndarray,
    residual: np.ndarray,
    post: np.ndarray,
    combination: np.ndarray,
) -> np.ndarray:
    mx = _mlx()
    branch = mx.array(np.asarray(branch_output, dtype=np.float32))
    saved = mx.array(np.asarray(residual, dtype=np.float32))
    post_weights = mx.array(np.asarray(post, dtype=np.float32))
    combination_weights = mx.array(np.asarray(combination, dtype=np.float32))
    # Keep the four-lane reduction explicit.  MLX's generic Metal matmul may
    # select reduced-precision accumulation even for this 4x4 operation,
    # obscuring the elementwise HC equation that the production kernel uses.
    combined = mx.sum(
        combination_weights[..., :, :, None] * saved[..., None, :, :],
        axis=-2,
    )
    result = post_weights[..., :, None] * branch[..., None, :] + combined
    mx.eval(result)
    return np.asarray(result, dtype=np.float64)


def hc_head(
    hidden_states: np.ndarray,
    function_weight: np.ndarray,
    scale: Sequence[float] | np.ndarray,
    base: Sequence[float] | np.ndarray,
    *,
    norm_eps: float = 1.0e-6,
    hc_eps: float = 1.0e-6,
) -> np.ndarray:
    mx = _mlx()
    hidden = mx.array(np.asarray(hidden_states, dtype=np.float32))
    function = mx.array(np.asarray(function_weight, dtype=np.float32))
    scale_value = mx.array(np.asarray(scale, dtype=np.float32).reshape(-1))
    bases = mx.array(np.asarray(base, dtype=np.float32))
    flattened = mx.reshape(hidden, (*hidden.shape[:-2], -1))
    inverse_rms = mx.rsqrt(
        mx.mean(mx.square(flattened), axis=-1, keepdims=True)
        + float(norm_eps)
    )
    mixes = (flattened @ mx.transpose(function)) * inverse_rms
    pre = mx.sigmoid(mixes * scale_value[0] + bases) + float(hc_eps)
    result = mx.sum(pre[..., :, None] * hidden, axis=-2)
    mx.eval(result)
    return np.asarray(result, dtype=np.float64)
