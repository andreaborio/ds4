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
EXPECTED_MLX_VERSION = "0.32.0"
EXPECTED_MLX_METAL_VERSION = "0.32.0"
DSPARK_TARGET_LAYER_IDS = (40, 41, 42)
DSPARK_RAW_CACHE_WIDTH = 512


@dataclass(frozen=True)
class MLXStageContextKV:
    """Host-visible boundaries from the independent MLX Metal finalizer."""

    absolute_positions: np.ndarray
    projected: np.ndarray
    normalized: np.ndarray
    roped: np.ndarray
    stored: np.ndarray
    nonrope_scales: np.ndarray


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
