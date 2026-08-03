"""Small NumPy reference equations for DSpark fixture generation.

The functions here favor explicit validation and float64 arithmetic.  They are
not an inference implementation and are not imported by the Hebrus runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ConfidenceSchedule:
    """Per-position confidence probabilities and the scheduled prefix."""

    logits: np.ndarray
    probabilities: np.ndarray
    keep: int


@dataclass(frozen=True)
class SpeculativeSample:
    """Outcome of one exact speculative-sampling verification round."""

    accepted: int
    replacement_token: int
    committed_tokens: tuple[int, ...]
    acceptance_thresholds: np.ndarray
    target_row: int
    residual_probabilities: np.ndarray | None


# These are the numerical guards in DeepSpec's pinned evaluator.  They are
# deliberately named here: callers must not mistake them for a different
# proposal distribution or an adjustable product policy.
DRAFT_PROBABILITY_FLOOR = 1.0e-8
RESIDUAL_MASS_FLOOR = 1.0e-8


def post_layer_hc_mean(hidden_states: np.ndarray) -> np.ndarray:
    """Capture DSpark's post-layer ``mean(dim=2)`` in NumPy layout.

    The oracle uses ``[token, hc, hidden]`` rather than the model's framework
    dimension numbering.  Final 0731 capture is exactly four HC lanes and a
    4096-wide hidden row; accepting another shape would hide a graph tap or
    tensor-layout bug.
    """

    hidden = np.asarray(hidden_states, dtype=np.float64)
    if hidden.ndim != 3:
        raise ValueError("hidden_states must have shape [token, 4, 4096]")
    if hidden.shape[0] == 0 or hidden.shape[1:] != (4, 4096):
        raise ValueError("hidden_states must have shape [token, 4, 4096]")
    if not np.all(np.isfinite(hidden)):
        raise ValueError("hidden_states must contain only finite values")
    return np.mean(hidden, axis=1)


def _float_matrix(value: object, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 array")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _token_vector(value: object, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be a rank-1 array")
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"{name} must contain integer token ids")
    return raw.astype(np.int64, copy=False)


def _validate_token_range(tokens: np.ndarray, vocab: int, name: str) -> None:
    if np.any(tokens < 0) or np.any(tokens >= vocab):
        raise ValueError(f"{name} contains a token outside [0, {vocab})")


def markov_step_bias(
    previous_tokens: Sequence[int] | np.ndarray,
    embedding: np.ndarray,
    projection: np.ndarray,
) -> np.ndarray:
    """Return ``embedding[token] @ projection.T`` for each previous token.

    ``embedding`` and ``projection`` use the conventional NumPy layouts
    ``[vocab, rank]``.  The DSpark GGUF writer may store the same matrices with
    transposed logical dimensions; layout conversion belongs outside this
    semantic oracle.
    """

    embed = _float_matrix(embedding, "embedding")
    project = _float_matrix(projection, "projection")
    if embed.shape != project.shape:
        raise ValueError(
            "embedding and projection must both have shape [vocab, rank]"
        )
    tokens = _token_vector(previous_tokens, "previous_tokens")
    _validate_token_range(tokens, embed.shape[0], "previous_tokens")
    return embed[tokens] @ project.T


def markov_greedy_draft(
    base_logits: np.ndarray,
    first_previous_token: int,
    embedding: np.ndarray,
    projection: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a block greedily with the sequential Markov correction."""

    base = _float_matrix(base_logits, "base_logits")
    embed = _float_matrix(embedding, "embedding")
    project = _float_matrix(projection, "projection")
    if embed.shape != project.shape or base.shape[1] != embed.shape[0]:
        raise ValueError(
            "base logits must use the vocabulary shared by Markov weights"
        )
    if isinstance(first_previous_token, bool) or not isinstance(
        first_previous_token, (int, np.integer)
    ):
        raise ValueError("first_previous_token must be an integer token id")
    previous = int(first_previous_token)
    _validate_token_range(
        np.asarray([previous], dtype=np.int64), embed.shape[0], "first_previous_token"
    )

    drafted = np.empty(base.shape[0], dtype=np.int64)
    step_logits = np.empty_like(base)
    for position in range(base.shape[0]):
        bias = markov_step_bias(
            np.asarray([previous], dtype=np.int64), embed, project
        )[0]
        step_logits[position] = base[position] + bias
        drafted[position] = int(np.argmax(step_logits[position]))
        previous = int(drafted[position])
    return drafted, step_logits


def conditional_confidence(
    block_hidden: np.ndarray,
    previous_tokens: Sequence[int] | np.ndarray,
    markov_embedding: np.ndarray,
    projection: Sequence[float] | np.ndarray,
    *,
    threshold: float = 0.0,
) -> ConfidenceSchedule:
    """Evaluate DSpark confidence and stop at the first low-confidence slot.

    The confidence head consumes ``[hidden, markov_embedding(previous)]``.
    The evaluator applies ``sigmoid`` independently at each position and keeps
    the prefix before the first probability strictly below ``threshold``.  It
    does not multiply confidence probabilities together.  The separate
    speculative-verification acceptance mask does use a cumulative product of
    binary accept decisions; that is a different operation.
    """

    hidden = _float_matrix(block_hidden, "block_hidden")
    embed = _float_matrix(markov_embedding, "markov_embedding")
    tokens = _token_vector(previous_tokens, "previous_tokens")
    if tokens.shape[0] != hidden.shape[0]:
        raise ValueError("previous_tokens must have one entry per hidden row")
    _validate_token_range(tokens, embed.shape[0], "previous_tokens")

    weights = np.asarray(projection, dtype=np.float64)
    if weights.ndim == 2 and weights.shape[0] == 1:
        weights = weights[0]
    if weights.ndim != 1:
        raise ValueError("projection must be a vector or a [1, features] matrix")
    expected_features = hidden.shape[1] + embed.shape[1]
    if weights.shape[0] != expected_features:
        raise ValueError(
            f"projection has {weights.shape[0]} features; expected {expected_features}"
        )
    if not np.all(np.isfinite(weights)):
        raise ValueError("confidence projection must be finite")
    if not np.isfinite(float(threshold)) or threshold < 0.0 or threshold > 1.0:
        raise ValueError("threshold must be finite and inside [0, 1]")

    features = np.concatenate((hidden, embed[tokens]), axis=1)
    logits = features @ weights
    probabilities = np.empty_like(logits)
    positive = logits >= 0.0
    probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    negative_exp = np.exp(logits[~positive])
    probabilities[~positive] = negative_exp / (1.0 + negative_exp)
    if threshold <= 0.0:
        keep = probabilities.shape[0]
    else:
        below = np.flatnonzero(probabilities < float(threshold))
        keep = int(below[0]) if below.size else probabilities.shape[0]
    return ConfidenceSchedule(logits, probabilities, keep)


def _probability_matrix(value: object, name: str) -> np.ndarray:
    result = _float_matrix(value, name)
    if np.any(result < 0.0):
        raise ValueError(f"{name} contains a negative probability")
    sums = np.sum(result, axis=1, dtype=np.float64)
    if not np.allclose(sums, 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"every {name} row must sum to one")
    return result


def _uniform_vector(value: object, length: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,):
        raise ValueError(f"{name} must have shape [{length}]")
    if (
        not np.all(np.isfinite(result))
        or np.any(result < 0.0)
        or np.any(result >= 1.0)
    ):
        raise ValueError(f"{name} values must be finite and inside [0, 1)")
    return result


def categorical_from_uniform(probabilities: np.ndarray, uniform: float) -> int:
    """Deterministically invert a categorical CDF with a supplied uniform."""

    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.ndim != 1 or not np.all(np.isfinite(probs)) or np.any(probs < 0.0):
        raise ValueError("categorical probabilities must be a finite vector")
    if not np.isclose(np.sum(probs), 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("categorical probabilities must sum to one")
    if not np.isfinite(float(uniform)) or uniform < 0.0 or uniform >= 1.0:
        raise ValueError("categorical uniform must be inside [0, 1)")
    token = int(np.searchsorted(np.cumsum(probs), uniform, side="right"))
    return min(token, probs.shape[0] - 1)


def speculative_sample_exact(
    target_probabilities: np.ndarray,
    draft_tokens: Sequence[int] | np.ndarray,
    draft_probabilities: np.ndarray,
    acceptance_uniforms: Sequence[float] | np.ndarray,
    categorical_uniforms: Sequence[float] | np.ndarray,
) -> SpeculativeSample:
    """Apply exact speculative sampling with caller-supplied random draws.

    For a draft of length ``L``, target probabilities have ``L + 1`` rows: one
    row for each accept test and one bonus row.  The proposal has ``L`` rows.
    The accepted prefix uses ``min(1, p(x) / max(q(x), 1e-8))`` and stops at
    the first rejected position.  At rejection, replacement is sampled from
    normalized ``max(p - q, 0)``; matching the pinned evaluator, residual mass
    at or below ``1e-8`` falls back to the target row.  After full acceptance,
    replacement is sampled from the target bonus row.

    Supplying all uniforms makes fixtures deterministic while preserving the
    exact algorithm.  ``categorical_uniforms[k]`` belongs to target row ``k``;
    unused values do not affect the result.
    """

    target = _probability_matrix(target_probabilities, "target_probabilities")
    proposal = _probability_matrix(draft_probabilities, "draft_probabilities")
    tokens = _token_vector(draft_tokens, "draft_tokens")
    length = tokens.shape[0]
    if proposal.shape[0] != length:
        raise ValueError("draft_probabilities must have one row per draft token")
    if target.shape != (length + 1, proposal.shape[1]):
        raise ValueError(
            "target_probabilities must have L + 1 rows and share proposal vocabulary"
        )
    _validate_token_range(tokens, proposal.shape[1], "draft_tokens")
    accepts = _uniform_vector(acceptance_uniforms, length, "acceptance_uniforms")
    categoricals = _uniform_vector(
        categorical_uniforms, length + 1, "categorical_uniforms"
    )

    rows = np.arange(length, dtype=np.int64)
    q_selected = proposal[rows, tokens]
    p_selected = target[rows, tokens]
    thresholds = np.minimum(
        1.0,
        p_selected / np.maximum(q_selected, DRAFT_PROBABILITY_FLOOR),
    )

    accepted = length
    for position in range(length):
        if not accepts[position] < thresholds[position]:
            accepted = position
            break

    residual: np.ndarray | None
    if accepted < length:
        residual = np.maximum(target[accepted] - proposal[accepted], 0.0)
        residual_sum = float(np.sum(residual, dtype=np.float64))
        if residual_sum <= RESIDUAL_MASS_FLOOR:
            residual = target[accepted].copy()
            residual_sum = float(np.sum(residual, dtype=np.float64))
        residual = residual / residual_sum
        replacement = categorical_from_uniform(
            residual, float(categoricals[accepted])
        )
    else:
        residual = None
        replacement = categorical_from_uniform(
            target[length], float(categoricals[length])
        )

    committed = tuple(int(token) for token in tokens[:accepted]) + (replacement,)
    return SpeculativeSample(
        accepted=accepted,
        replacement_token=replacement,
        committed_tokens=committed,
        acceptance_thresholds=thresholds,
        target_row=accepted,
        residual_probabilities=residual,
    )
