"""Optional Apple-Silicon cross-checks for the NumPy DSpark oracle.

Importing :mod:`tools.dspark_oracle` never imports MLX.  These helpers load
``mlx.core`` only when explicitly called and do not depend on ``mlx-lm``.
"""

from __future__ import annotations

import importlib.util
from typing import Sequence

import numpy as np


# Operation-specific fixture limits for MLX 0.32.0 float32 Metal execution.
# They are deliberately not a global oracle tolerance: the synthetic Markov
# matmul measured 9.765625e-05 max absolute drift, while the confidence
# projection plus sigmoid measured 3.43827e-08.  The narrow ceilings below keep
# that Metal rounding visible and fail if either operation moves into a
# different numerical envelope.  NumPy float64 remains the fixture authority.
MLX_F32_MARKOV_MATMUL_MAX_ABS_DRIFT = 1.0e-4
MLX_F32_CONFIDENCE_MAX_ABS_DRIFT = 5.0e-8
MLX_F32_HC_MEAN_MAX_ABS_DRIFT = 1.0e-6


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
