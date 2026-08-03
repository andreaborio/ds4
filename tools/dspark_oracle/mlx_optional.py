"""Optional Apple-Silicon cross-checks for the NumPy DSpark oracle.

Importing :mod:`tools.dspark_oracle` never imports MLX.  These helpers load
``mlx.core`` only when explicitly called and do not depend on ``mlx-lm``.
"""

from __future__ import annotations

import importlib.util
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
MLX_F32_HC_SPLIT_MAX_ABS_DRIFT = 1.0e-7
MLX_F32_HC_OUTPUT_MAX_ABS_DRIFT = 5.0e-7
EXPECTED_MLX_VERSION = "0.32.0"
EXPECTED_MLX_METAL_VERSION = "0.32.0"


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
    hidden = mx.array(np.asarray(hidden_states, dtype=np.float32))
    result = mx.mean(hidden, axis=1)
    mx.eval(result)
    return np.asarray(result, dtype=np.float64)


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
