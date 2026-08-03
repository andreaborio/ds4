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


@dataclass(frozen=True)
class HCSplit:
    """Hyper-Connection weights for one attention or FFN sublayer."""

    pre: np.ndarray
    post: np.ndarray
    combination: np.ndarray


@dataclass(frozen=True)
class StageSetup:
    """Shared main projection and repeated noisy block entering stage zero."""

    main_hidden: np.ndarray
    draft_hidden: np.ndarray


@dataclass(frozen=True)
class TargetCaptureRows:
    """The frontier tap and retained target history consumed by DSpark."""

    layer_ids: tuple[int, int, int]
    phase: str
    token_index: int
    absolute_token_position: int
    rows: np.ndarray
    history_token_start: int
    history_rows: np.ndarray


@dataclass(frozen=True)
class ProposalTokenLayout:
    """Pending target token, DSpark input positions, and output positions."""

    pending_token_id: int
    input_token_ids: np.ndarray
    input_positions: np.ndarray
    proposed_output_positions: np.ndarray


@dataclass(frozen=True)
class StageContextKV:
    """Direct target-derived DSpark context KV and its storage simulation."""

    absolute_positions: np.ndarray
    normalized: np.ndarray
    roped: np.ndarray
    stored: np.ndarray
    nonrope_scales: np.ndarray


@dataclass(frozen=True)
class StageChain:
    """Outputs of three ordered synthetic stages sharing one main input."""

    stage_outputs: tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass(frozen=True)
class MarkovDraft:
    """Sequential Markov proposal with explicit sampling probabilities."""

    tokens: np.ndarray
    corrected_logits: np.ndarray
    probabilities: np.ndarray


@dataclass(frozen=True)
class DraftHead:
    """Final HC head, sequential Markov logits, and confidence schedule."""

    hidden: np.ndarray
    base_logits: np.ndarray
    tokens: np.ndarray
    corrected_logits: np.ndarray
    confidence: ConfidenceSchedule


@dataclass(frozen=True)
class RawCacheState:
    """Three independent physical rings and their shared logical window."""

    capacity: int
    token_start: int
    length: int
    rows: np.ndarray


# These are the numerical guards in DeepSpec's pinned evaluator.  They are
# deliberately named here: callers must not mistake them for a different
# proposal distribution or an adjustable product policy.
DRAFT_PROBABILITY_FLOOR = 1.0e-8
RESIDUAL_MASS_FLOOR = 1.0e-8
DSPARK_STAGE_COUNT = 3
DSPARK_RAW_CACHE_WINDOW = 128
DSPARK_RAW_CACHE_WIDTH = 512
DSPARK_TARGET_LAYER_IDS = (40, 41, 42)


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


def capture_target_hidden_rows(
    layer_hidden_states: Sequence[np.ndarray],
    layer_ids: Sequence[int],
    *,
    phase: str,
    start_position: int = 0,
) -> TargetCaptureRows:
    """Capture the frontier tap and the retained target prompt history.

    Inputs are three graph taps in execution order, one after each of target
    layers 40, 41, and 42.  Every tap uses ``[token, 4, 4096]`` and is reduced
    across the four HC lanes.  ``rows`` is only the frontier tap used to start
    the next draft.  ``history_rows`` is token-major and retains every row from
    the newest 128 target positions; official prompt setup projects all of
    those rows into each stage's context KV rather than projecting only the
    frontier row.
    """

    ids = tuple(layer_ids)
    if ids != DSPARK_TARGET_LAYER_IDS:
        raise ValueError("target capture layers must be ordered 40, 41, 42")
    states = tuple(layer_hidden_states)
    if len(states) != DSPARK_STAGE_COUNT:
        raise ValueError("target capture requires exactly three layer tensors")
    reduced = tuple(post_layer_hc_mean(state) for state in states)
    token_counts = {item.shape[0] for item in reduced}
    if len(token_counts) != 1:
        raise ValueError("target capture layers must have the same token count")
    token_count = next(iter(token_counts))
    if (isinstance(start_position, bool) or
            not isinstance(start_position, (int, np.integer)) or
            start_position < 0):
        raise ValueError("start_position must be a non-negative integer")
    if phase == "decode":
        if token_count != 1:
            raise ValueError("decode target capture requires exactly one token row")
        token_index = 0
    elif phase == "prefill":
        token_index = token_count - 1
    else:
        raise ValueError("target capture phase must be 'decode' or 'prefill'")
    retained = min(token_count, DSPARK_RAW_CACHE_WINDOW)
    history_start_index = token_count - retained
    history_rows = np.stack(
        [item[history_start_index:] for item in reduced], axis=1
    )
    rows = np.stack(
        [item[token_index] for item in reduced], axis=0
    )
    return TargetCaptureRows(
        DSPARK_TARGET_LAYER_IDS,
        phase,
        token_index,
        int(start_position) + token_index,
        np.array(rows, copy=True),
        int(start_position) + history_start_index,
        np.array(history_rows, copy=True),
    )


def proposal_token_layout(
    last_target_position: int,
    pending_token_id: int,
    noise_token_id: int,
    *,
    block_size: int = 5,
) -> ProposalTokenLayout:
    """Declare the official shift between target output and DSpark rows.

    After target position ``p`` is evaluated, the sampled token ``y[p+1]`` is
    pending.  DSpark embeds ``[y[p+1], noise, noise, noise, noise]`` at absolute
    input positions ``p+1 .. p+5``.  Its five head rows propose outputs for
    ``p+2 .. p+6``; it does not re-predict the pending target token.
    """

    for value, name in (
        (last_target_position, "last_target_position"),
        (pending_token_id, "pending_token_id"),
        (noise_token_id, "noise_token_id"),
    ):
        if (isinstance(value, bool) or
                not isinstance(value, (int, np.integer)) or value < 0):
            raise ValueError(f"{name} must be a non-negative integer")
    if block_size != 5:
        raise ValueError("final 0731 requires block_size=5")
    input_ids = np.full(block_size, int(noise_token_id), dtype=np.int64)
    input_ids[0] = int(pending_token_id)
    first = int(last_target_position) + 1
    return ProposalTokenLayout(
        int(pending_token_id),
        input_ids,
        np.arange(first, first + block_size, dtype=np.int64),
        np.arange(first + 1, first + block_size + 1, dtype=np.int64),
    )


def concatenate_target_captures(target_hidden: np.ndarray) -> np.ndarray:
    """Flatten target rows in the only accepted order: 40, then 41, then 42."""

    target = np.asarray(target_hidden, dtype=np.float64)
    if target.ndim < 2 or target.shape[-2] != DSPARK_STAGE_COUNT:
        raise ValueError("target_hidden must have exactly three capture stages")
    if target.shape[-1] == 0:
        raise ValueError("target_hidden must have a non-empty hidden dimension")
    if not np.all(np.isfinite(target)):
        raise ValueError("target_hidden must be finite")
    return np.array(
        target.reshape(*target.shape[:-2], DSPARK_STAGE_COUNT * target.shape[-1]),
        copy=True,
    )


def rms_norm(
    hidden_states: np.ndarray,
    weight: Sequence[float] | np.ndarray,
    *,
    eps: float = 1.0e-6,
) -> np.ndarray:
    """Reference RMSNorm with float64 accumulation and no additive bias."""

    hidden = np.asarray(hidden_states, dtype=np.float64)
    weights = np.asarray(weight, dtype=np.float64)
    if hidden.ndim == 0 or hidden.shape[-1:] != weights.shape:
        raise ValueError("RMSNorm weight must match the final hidden dimension")
    if not np.all(np.isfinite(hidden)) or not np.all(np.isfinite(weights)):
        raise ValueError("RMSNorm inputs must be finite")
    if not np.isfinite(float(eps)) or eps <= 0.0:
        raise ValueError("RMSNorm epsilon must be finite and positive")
    variance = np.mean(np.square(hidden), axis=-1, keepdims=True)
    return hidden * (1.0 / np.sqrt(variance + float(eps))) * weights


def main_project_and_norm(
    target_hidden: np.ndarray,
    projection: np.ndarray,
    norm_weight: Sequence[float] | np.ndarray,
    *,
    eps: float = 1.0e-6,
) -> np.ndarray:
    """Concatenate captures 40/41/42, apply ``main_proj``, then RMSNorm.

    ``target_hidden`` uses ``[..., stage=3, hidden]``.  Projection uses the
    conventional linear layout ``[hidden, 3 * hidden]``; GGUF stores its
    dimensions in the reverse logical order ``[3 * hidden, hidden]``.
    """

    target = np.asarray(target_hidden, dtype=np.float64)
    project = _float_matrix(projection, "main projection")
    weights = np.asarray(norm_weight, dtype=np.float64)
    if target.ndim < 2 or target.shape[-2] != DSPARK_STAGE_COUNT:
        raise ValueError("target_hidden must have exactly three capture stages")
    hidden = target.shape[-1]
    if project.shape != (hidden, 3 * hidden):
        raise ValueError("main projection must have shape [hidden, 3 * hidden]")
    if weights.shape != (hidden,):
        raise ValueError("main norm weight must have shape [hidden]")
    flattened = concatenate_target_captures(target)
    return rms_norm(flattened @ project.T, weights, eps=eps)


def _round_bfloat16(value: np.ndarray) -> np.ndarray:
    """Round finite float32 values to BF16, returned as float32 storage."""

    result = np.asarray(value, dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError("BF16 simulation requires finite values")
    bits = np.array(result, copy=True).view(np.uint32)
    rounding = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    rounded = (bits + rounding) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def _e4m3fn_positive_values() -> tuple[np.ndarray, np.ndarray]:
    values: list[float] = []
    codes: list[int] = []
    for code in range(127):
        exponent = code >> 3
        mantissa = code & 7
        if exponent == 0:
            value = mantissa * (2.0 ** -9)
        else:
            value = (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))
        values.append(value)
        codes.append(code)
    return np.asarray(values, dtype=np.float64), np.asarray(codes, dtype=np.int64)


_E4M3FN_POSITIVE_VALUES, _E4M3FN_POSITIVE_CODES = _e4m3fn_positive_values()


def _round_float8_e4m3fn(value: np.ndarray) -> np.ndarray:
    """Round finite values to IEEE-like E4M3FN using nearest-even ties."""

    source = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(source)):
        raise ValueError("FP8 simulation requires finite values")
    magnitude = np.minimum(np.abs(source), 448.0)
    upper = np.searchsorted(_E4M3FN_POSITIVE_VALUES, magnitude, side="left")
    upper = np.minimum(upper, _E4M3FN_POSITIVE_VALUES.size - 1)
    lower = np.maximum(upper - 1, 0)
    low_distance = magnitude - _E4M3FN_POSITIVE_VALUES[lower]
    high_distance = _E4M3FN_POSITIVE_VALUES[upper] - magnitude
    use_upper = high_distance < low_distance
    ties = high_distance == low_distance
    use_upper |= ties & ((_E4M3FN_POSITIVE_CODES[upper] & 1) == 0)
    indices = np.where(use_upper, upper, lower)
    rounded = _E4M3FN_POSITIVE_VALUES[indices]
    return np.copysign(rounded, source)


def _simulate_official_nope_fp8(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Simulate official in-place E4M3FN Q/DQ over seven 64-wide groups."""

    rows = np.asarray(value, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] != 448:
        raise ValueError("non-RoPE KV must have shape [token, 448]")
    grouped = rows.reshape(rows.shape[0], 7, 64)
    amax = np.maximum(np.max(np.abs(grouped), axis=2), np.float32(1.0e-4))
    # Pinned scale_fmt=ue8m0 rounds amax/448 upward to an exact power of two.
    scales = np.exp2(np.ceil(np.log2(amax / np.float32(448.0)))).astype(
        np.float32
    )
    quantized = _round_float8_e4m3fn(
        grouped.astype(np.float64) / scales[..., None].astype(np.float64)
    )
    dequantized = _round_bfloat16(
        quantized.astype(np.float32) * scales[..., None]
    )
    return dequantized.reshape(rows.shape), scales


def direct_stage_context_kv(
    main_x: np.ndarray,
    projection: np.ndarray,
    norm_weight: Sequence[float] | np.ndarray,
    absolute_positions: Sequence[int] | np.ndarray,
    *,
    eps: float = 1.0e-6,
    rope_theta: float = 10000.0,
) -> StageContextKV:
    """Project official target history directly into one stage's context KV.

    This models ``kv_norm(Wkv(main_x))`` for all supplied target positions.
    The last 64 of 512 dimensions receive RoPE at their absolute positions;
    the first 448 are then quantize/dequantize simulated in seven 64-wide
    E4M3FN groups with UE8M0 power-of-two scales.  The returned arrays expose
    each boundary so a candidate path that incorrectly applies HC-pre or
    attention norm to target history cannot satisfy the fixture.

    NumPy cannot reproduce the quantized-weight GEMM that creates ``Wkv``.
    It does reproduce the official post-linear BF16 storage boundaries, RoPE,
    E4M3FN clamp/nearest-even conversion, and in-place BF16 dequantization.
    """

    main = _float_matrix(main_x, "main_x")
    project = _float_matrix(projection, "context KV projection")
    weights = np.asarray(norm_weight, dtype=np.float32)
    positions = _token_vector(absolute_positions, "absolute_positions")
    if project.shape != (DSPARK_RAW_CACHE_WIDTH, main.shape[1]):
        raise ValueError("context KV projection must have shape [512, hidden]")
    if weights.shape != (DSPARK_RAW_CACHE_WIDTH,):
        raise ValueError("context KV norm weight must have shape [512]")
    if positions.shape[0] != main.shape[0] or np.any(positions < 0):
        raise ValueError("absolute_positions must be non-negative and match main_x")
    if not np.all(np.isfinite(weights)):
        raise ValueError("context KV norm weight must be finite")
    if not np.isfinite(float(eps)) or eps <= 0.0:
        raise ValueError("context KV norm epsilon must be finite and positive")
    if not np.isfinite(float(rope_theta)) or rope_theta <= 0.0:
        raise ValueError("rope_theta must be finite and positive")

    projected = _round_bfloat16(main @ project.T)
    projected_f32 = projected.astype(np.float32)
    variance = np.mean(
        np.square(projected_f32), axis=-1, keepdims=True, dtype=np.float32
    )
    normalized = _round_bfloat16(
        projected_f32
        * (np.float32(1.0) / np.sqrt(variance + np.float32(eps)))
        * weights
    )

    roped = np.array(normalized, copy=True)
    tail = roped[:, -64:].reshape(roped.shape[0], 32, 2).astype(np.float32)
    frequency = np.float32(1.0) / np.power(
        np.float32(rope_theta),
        np.arange(0, 64, 2, dtype=np.float32) / np.float32(64.0),
        dtype=np.float32,
    )
    angles = positions.astype(np.float32)[:, None] * frequency[None, :]
    cosine = np.cos(angles).astype(np.float32)
    sine = np.sin(angles).astype(np.float32)
    first = tail[..., 0] * cosine - tail[..., 1] * sine
    second = tail[..., 0] * sine + tail[..., 1] * cosine
    tail[..., 0] = first
    tail[..., 1] = second
    roped[:, -64:] = _round_bfloat16(tail.reshape(roped.shape[0], 64))

    stored = np.array(roped, copy=True)
    stored[:, :-64], scales = _simulate_official_nope_fp8(stored[:, :-64])
    return StageContextKV(
        np.array(positions, copy=True),
        normalized.astype(np.float64),
        roped.astype(np.float64),
        stored.astype(np.float64),
        scales.astype(np.float64),
    )


def prepare_stage_zero(
    target_hidden: np.ndarray,
    pending_embedding: Sequence[float] | np.ndarray,
    noise_embedding: Sequence[float] | np.ndarray,
    main_projection: np.ndarray,
    main_norm_weight: Sequence[float] | np.ndarray,
    *,
    block_size: int = 5,
    hc_lanes: int = 4,
    eps: float = 1.0e-6,
) -> StageSetup:
    """Build stage zero from the pending target token plus four noise rows.

    ``pending_embedding`` belongs to ``y[p+1]`` sampled by target logits after
    evaluating position ``p``.  It is not an already accepted draft token and
    DSpark does not predict it again.
    """

    if block_size != 5 or hc_lanes != 4:
        raise ValueError("final 0731 requires block_size=5 and hc_lanes=4")
    main_hidden = main_project_and_norm(
        target_hidden, main_projection, main_norm_weight, eps=eps
    )
    if main_hidden.ndim != 1:
        raise ValueError("stage-zero setup expects one target position")
    pending = np.asarray(pending_embedding, dtype=np.float64)
    noise = np.asarray(noise_embedding, dtype=np.float64)
    if pending.shape != main_hidden.shape or noise.shape != main_hidden.shape:
        raise ValueError("pending/noise embeddings must match hidden width")
    if not np.all(np.isfinite(pending)) or not np.all(np.isfinite(noise)):
        raise ValueError("pending/noise embeddings must be finite")
    draft = np.repeat(noise[None, :], block_size, axis=0)
    draft[0] = pending
    draft_hidden = np.repeat(draft[:, None, :], hc_lanes, axis=1)
    return StageSetup(main_hidden, draft_hidden)


def run_synthetic_stage_chain(
    draft_hidden: np.ndarray,
    main_hidden: np.ndarray,
    stage_weights: np.ndarray,
    main_weights: np.ndarray,
    stage_biases: np.ndarray,
) -> StageChain:
    """Exercise only the ordered three-stage DSpark topology.

    This is deliberately not an approximation of attention or the MoE.  Each
    stage applies a distinct affine transform to the previous stage output and
    to the same immutable ``main_hidden``.  Closed fixtures can therefore catch
    stage reordering, skipping, reuse, or accidentally feeding a stage-local
    main value into the next stage without needing model weights.
    """

    hidden = np.asarray(draft_hidden, dtype=np.float64)
    main = np.asarray(main_hidden, dtype=np.float64)
    stages = np.asarray(stage_weights, dtype=np.float64)
    mains = np.asarray(main_weights, dtype=np.float64)
    biases = np.asarray(stage_biases, dtype=np.float64)
    if hidden.ndim < 2 or hidden.shape[-1] == 0:
        raise ValueError("draft_hidden must have rows and a hidden dimension")
    width = hidden.shape[-1]
    if main.shape != (width,):
        raise ValueError("main_hidden must have shape [hidden]")
    expected_matrices = (DSPARK_STAGE_COUNT, width, width)
    if stages.shape != expected_matrices or mains.shape != expected_matrices:
        raise ValueError("stage/main weights must have shape [3, hidden, hidden]")
    if biases.shape != (DSPARK_STAGE_COUNT, width):
        raise ValueError("stage_biases must have shape [3, hidden]")
    if not all(np.all(np.isfinite(item))
               for item in (hidden, main, stages, mains, biases)):
        raise ValueError("synthetic stage-chain inputs must be finite")

    outputs: list[np.ndarray] = []
    current = hidden
    for stage in range(DSPARK_STAGE_COUNT):
        current = (
            current @ stages[stage].T
            + main @ mains[stage].T
            + biases[stage]
        )
        outputs.append(np.array(current, copy=True))
    return StageChain(tuple(outputs))


def _stable_sigmoid(value: np.ndarray) -> np.ndarray:
    result = np.empty_like(value, dtype=np.float64)
    positive = value >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    negative_exp = np.exp(value[~positive])
    result[~positive] = negative_exp / (1.0 + negative_exp)
    return result


def hc_split_sinkhorn(
    mixes: np.ndarray,
    scale: Sequence[float] | np.ndarray,
    base: Sequence[float] | np.ndarray,
    *,
    hc_lanes: int = 4,
    iterations: int = 20,
    eps: float = 1.0e-6,
) -> HCSplit:
    """Split HC logits into pre/post weights and a Sinkhorn matrix."""

    values = np.asarray(mixes, dtype=np.float64)
    scales = np.asarray(scale, dtype=np.float64)
    bases = np.asarray(base, dtype=np.float64)
    mix_width = (2 + hc_lanes) * hc_lanes
    if values.ndim == 0 or values.shape[-1] != mix_width:
        raise ValueError("HC mixes have the wrong final dimension")
    if scales.shape != (3,) or bases.shape != (mix_width,):
        raise ValueError("HC scale/base geometry mismatch")
    if iterations < 1:
        raise ValueError("HC Sinkhorn iterations must be positive")
    if not np.isfinite(float(eps)) or eps <= 0.0:
        raise ValueError("HC epsilon must be finite and positive")
    if not all(np.all(np.isfinite(item)) for item in (values, scales, bases)):
        raise ValueError("HC inputs must be finite")

    pre_logits = values[..., :hc_lanes]
    post_logits = values[..., hc_lanes:2 * hc_lanes]
    combination_logits = values[..., 2 * hc_lanes:].reshape(
        *values.shape[:-1], hc_lanes, hc_lanes
    )
    pre = _stable_sigmoid(
        pre_logits * scales[0] + bases[:hc_lanes]
    ) + eps
    post = 2.0 * _stable_sigmoid(
        post_logits * scales[1] + bases[hc_lanes:2 * hc_lanes]
    )
    combination = (
        combination_logits * scales[2]
        + bases[2 * hc_lanes:].reshape(hc_lanes, hc_lanes)
    )
    combination -= np.max(combination, axis=-1, keepdims=True)
    combination = np.exp(combination)
    combination = (
        combination / np.sum(combination, axis=-1, keepdims=True) + eps
    )
    combination = combination / (
        np.sum(combination, axis=-2, keepdims=True) + eps
    )
    for _ in range(iterations - 1):
        combination = combination / (
            np.sum(combination, axis=-1, keepdims=True) + eps
        )
        combination = combination / (
            np.sum(combination, axis=-2, keepdims=True) + eps
        )
    return HCSplit(pre, post, combination)


def hc_pre(
    hidden_states: np.ndarray,
    function_weight: np.ndarray,
    scale: Sequence[float] | np.ndarray,
    base: Sequence[float] | np.ndarray,
    *,
    norm_eps: float = 1.0e-6,
    hc_eps: float = 1.0e-6,
    iterations: int = 20,
) -> tuple[np.ndarray, HCSplit]:
    """Reduce four HC lanes and retain weights required by ``hc_post``."""

    hidden = np.asarray(hidden_states, dtype=np.float64)
    function = _float_matrix(function_weight, "HC function weight")
    if hidden.ndim < 2 or hidden.shape[-2] != 4:
        raise ValueError("HC hidden state must have exactly four lanes")
    flat_width = hidden.shape[-2] * hidden.shape[-1]
    if function.shape != (24, flat_width):
        raise ValueError("HC function weight must have shape [24, 4 * hidden]")
    if not np.all(np.isfinite(hidden)):
        raise ValueError("HC hidden state must be finite")
    if not np.isfinite(float(norm_eps)) or norm_eps <= 0.0:
        raise ValueError("HC norm epsilon must be finite and positive")
    flattened = hidden.reshape(*hidden.shape[:-2], flat_width)
    inverse_rms = 1.0 / np.sqrt(
        np.mean(np.square(flattened), axis=-1, keepdims=True)
        + float(norm_eps)
    )
    split = hc_split_sinkhorn(
        (flattened @ function.T) * inverse_rms,
        scale,
        base,
        hc_lanes=4,
        iterations=iterations,
        eps=hc_eps,
    )
    reduced = np.sum(split.pre[..., :, None] * hidden, axis=-2)
    return reduced, split


def hc_post(
    branch_output: np.ndarray,
    residual: np.ndarray,
    split: HCSplit,
) -> np.ndarray:
    """Expand one branch result back into four HC lanes."""

    branch = np.asarray(branch_output, dtype=np.float64)
    saved = np.asarray(residual, dtype=np.float64)
    if saved.ndim < 2 or saved.shape[-2] != 4:
        raise ValueError("HC residual must have exactly four lanes")
    if branch.shape != saved.shape[:-2] + (saved.shape[-1],):
        raise ValueError("HC branch output must match residual hidden width")
    if split.post.shape != saved.shape[:-2] + (4,):
        raise ValueError("HC post weights do not match residual rows")
    if split.combination.shape != saved.shape[:-2] + (4, 4):
        raise ValueError("HC combination matrix does not match residual rows")
    combined = np.einsum("...ji,...id->...jd", split.combination, saved)
    return split.post[..., :, None] * branch[..., None, :] + combined


def hc_head(
    hidden_states: np.ndarray,
    function_weight: np.ndarray,
    scale: Sequence[float] | np.ndarray,
    base: Sequence[float] | np.ndarray,
    *,
    norm_eps: float = 1.0e-6,
    hc_eps: float = 1.0e-6,
) -> np.ndarray:
    """Collapse the final four HC lanes before norm/logits/confidence."""

    hidden = np.asarray(hidden_states, dtype=np.float64)
    function = _float_matrix(function_weight, "HC head function weight")
    scales = np.asarray(scale, dtype=np.float64).reshape(-1)
    bases = np.asarray(base, dtype=np.float64)
    if hidden.ndim < 2 or hidden.shape[-2] != 4:
        raise ValueError("HC head input must have exactly four lanes")
    flat_width = 4 * hidden.shape[-1]
    if function.shape != (4, flat_width):
        raise ValueError("HC head function must have shape [4, 4 * hidden]")
    if scales.shape != (1,) or bases.shape != (4,):
        raise ValueError("HC head scale/base geometry mismatch")
    if not all(np.all(np.isfinite(item))
               for item in (hidden, function, scales, bases)):
        raise ValueError("HC head inputs must be finite")
    if (not np.isfinite(float(norm_eps)) or norm_eps <= 0.0 or
            not np.isfinite(float(hc_eps)) or hc_eps <= 0.0):
        raise ValueError("HC head epsilons must be finite and positive")
    flattened = hidden.reshape(*hidden.shape[:-2], flat_width)
    inverse_rms = 1.0 / np.sqrt(
        np.mean(np.square(flattened), axis=-1, keepdims=True)
        + float(norm_eps)
    )
    mixes = (flattened @ function.T) * inverse_rms
    pre = _stable_sigmoid(mixes * scales[0] + bases) + float(hc_eps)
    return np.sum(pre[..., :, None] * hidden, axis=-2)


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


def markov_sampled_draft(
    base_logits: np.ndarray,
    first_previous_token: int,
    embedding: np.ndarray,
    projection: np.ndarray,
    sampling_uniforms: Sequence[float] | np.ndarray,
    *,
    temperature: float,
) -> MarkovDraft:
    """Sample a Markov-corrected block sequentially from supplied uniforms.

    The previous token is updated after every categorical draw, matching the
    official per-position loop.  Caller-provided uniforms make the stochastic
    path reproducible and keep RNG implementation details outside the oracle.
    """

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
        np.asarray([previous], dtype=np.int64),
        embed.shape[0],
        "first_previous_token",
    )
    if not np.isfinite(float(temperature)) or temperature <= 0.0:
        raise ValueError("sampled Markov temperature must be finite and positive")
    uniforms = _uniform_vector(
        sampling_uniforms, base.shape[0], "sampling_uniforms"
    )

    drafted = np.empty(base.shape[0], dtype=np.int64)
    corrected = np.empty_like(base)
    probabilities = np.empty_like(base)
    for position in range(base.shape[0]):
        bias = markov_step_bias(
            np.asarray([previous], dtype=np.int64), embed, project
        )[0]
        corrected[position] = base[position] + bias
        scaled = corrected[position] / float(temperature)
        scaled -= np.max(scaled)
        row = np.exp(scaled)
        row /= np.sum(row, dtype=np.float64)
        probabilities[position] = row
        drafted[position] = categorical_from_uniform(
            row, float(uniforms[position])
        )
        previous = int(drafted[position])
    return MarkovDraft(drafted, corrected, probabilities)


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


def finalize_draft_head(
    final_hc: np.ndarray,
    pending_token_id: int,
    output_projection: np.ndarray,
    norm_weight: Sequence[float] | np.ndarray,
    hc_function: np.ndarray,
    hc_scale: Sequence[float] | np.ndarray,
    hc_base: Sequence[float] | np.ndarray,
    markov_embedding: np.ndarray,
    markov_projection: np.ndarray,
    confidence_projection: Sequence[float] | np.ndarray,
    *,
    confidence_threshold: float,
    norm_eps: float = 1.0e-6,
    hc_eps: float = 1.0e-6,
) -> DraftHead:
    """Evaluate the final DSpark head with sequential Markov feedback.

    Confidence consumes the HC-collapsed hidden rows before the final RMSNorm,
    paired with W1 embeddings of ``[y[p+1], draft[0], ...]``.  Base logits
    consume the same rows after RMSNorm and the shared target output matrix.
    The first Markov correction therefore consumes the pending target token;
    the five returned tokens correspond to outputs ``p+2 .. p+6``.
    """

    collapsed = hc_head(
        final_hc,
        hc_function,
        hc_scale,
        hc_base,
        norm_eps=norm_eps,
        hc_eps=hc_eps,
    )
    if collapsed.ndim != 2 or collapsed.shape[0] != 5:
        raise ValueError("final 0731 draft head requires exactly five rows")
    output = _float_matrix(output_projection, "output projection")
    if output.shape[1] != collapsed.shape[-1]:
        raise ValueError("output projection hidden width mismatch")
    base_logits = rms_norm(collapsed, norm_weight, eps=norm_eps) @ output.T
    tokens, corrected = markov_greedy_draft(
        base_logits,
        pending_token_id,
        markov_embedding,
        markov_projection,
    )
    previous = np.concatenate((
        np.asarray([pending_token_id], dtype=np.int64),
        tokens[:-1],
    ))
    confidence = conditional_confidence(
        collapsed,
        previous,
        markov_embedding,
        confidence_projection,
        threshold=confidence_threshold,
    )
    return DraftHead(
        collapsed, base_logits, tokens, corrected, confidence
    )


def empty_raw_cache(
    capacity: int = DSPARK_RAW_CACHE_WINDOW,
    width: int = DSPARK_RAW_CACHE_WIDTH,
) -> RawCacheState:
    """Allocate the three exact final-0731 DSpark raw-KV rings."""

    if (isinstance(capacity, bool) or
            not isinstance(capacity, (int, np.integer)) or
            int(capacity) != DSPARK_RAW_CACHE_WINDOW):
        raise ValueError("final 0731 raw-cache capacity must be 128")
    if (isinstance(width, bool) or
            not isinstance(width, (int, np.integer)) or
            int(width) != DSPARK_RAW_CACHE_WIDTH):
        raise ValueError("final 0731 raw-cache width must be 512")
    return RawCacheState(
        capacity=DSPARK_RAW_CACHE_WINDOW,
        token_start=0,
        length=0,
        rows=np.zeros(
            (DSPARK_STAGE_COUNT,
             DSPARK_RAW_CACHE_WINDOW,
             DSPARK_RAW_CACHE_WIDTH),
            dtype=np.float64,
        ),
    )


def _validate_raw_cache_state(state: RawCacheState) -> None:
    if (
        state.capacity != DSPARK_RAW_CACHE_WINDOW
        or state.token_start < 0
        or state.length < 0
        or state.length > state.capacity
        or state.rows.ndim != 3
        or state.rows.shape[0] != DSPARK_STAGE_COUNT
        or state.rows.shape[1] != state.capacity
        or state.rows.shape[2] != DSPARK_RAW_CACHE_WIDTH
        or not np.all(np.isfinite(state.rows))
    ):
        raise ValueError("invalid three-stage raw-cache state")


def append_raw_cache(
    state: RawCacheState,
    token_position: int,
    stage_rows: np.ndarray,
) -> RawCacheState:
    """Commit one target-derived KV row to every stage-specific ring."""

    _validate_raw_cache_state(state)
    rows = np.asarray(stage_rows, dtype=np.float64)
    if rows.shape != (DSPARK_STAGE_COUNT, DSPARK_RAW_CACHE_WIDTH):
        raise ValueError("stage_rows must have shape [3, 512]")
    if not np.all(np.isfinite(rows)):
        raise ValueError("raw-cache rows must be finite")
    if (isinstance(token_position, bool) or
            not isinstance(token_position, (int, np.integer)) or
            token_position < 0):
        raise ValueError("token_position must be a non-negative integer")
    expected = state.token_start + state.length
    if state.length and token_position != expected:
        raise ValueError(
            f"raw-cache append position {token_position} does not follow {expected}"
        )
    if not state.length and token_position < state.token_start:
        raise ValueError("raw-cache append precedes its logical start")

    physical = token_position % state.capacity
    storage = np.array(state.rows, copy=True)
    storage[:, physical, :] = rows
    length = min(state.length + 1, state.capacity)
    token_start = token_position - length + 1
    return RawCacheState(
        state.capacity, token_start, length, storage
    )


def prefill_raw_cache(
    stage_rows: np.ndarray,
    *,
    start_position: int = 0,
) -> RawCacheState:
    """Populate all three stage rings from target-derived prefill KV rows.

    Input layout is exactly ``[token, stage=3, raw_width=512]``.  Once the
    final-0731 capacity of 128 is exceeded, physical slots wrap by absolute
    token position while the logical view retains only the newest window.
    """

    rows = np.asarray(stage_rows, dtype=np.float64)
    if (
        rows.ndim != 3
        or rows.shape[1:] != (DSPARK_STAGE_COUNT, DSPARK_RAW_CACHE_WIDTH)
        or rows.shape[0] == 0
    ):
        raise ValueError("prefill rows must have shape [token, 3, 512]")
    if not np.all(np.isfinite(rows)):
        raise ValueError("prefill raw-cache rows must be finite")
    if (isinstance(start_position, bool) or
            not isinstance(start_position, (int, np.integer)) or
            start_position < 0):
        raise ValueError("start_position must be a non-negative integer")
    storage = np.zeros(
        (DSPARK_STAGE_COUNT,
         DSPARK_RAW_CACHE_WINDOW,
         DSPARK_RAW_CACHE_WIDTH),
        dtype=np.float64,
    )
    positions = start_position + np.arange(rows.shape[0], dtype=np.int64)
    # A local mutable construction avoids copying the entire production-sized
    # ring for every prefill token.  The returned state owns this storage.
    for offset, position in enumerate(positions):
        storage[:, int(position) % DSPARK_RAW_CACHE_WINDOW, :] = rows[offset]
    length = min(rows.shape[0], DSPARK_RAW_CACHE_WINDOW)
    token_start = int(positions[-1]) - length + 1
    return RawCacheState(
        DSPARK_RAW_CACHE_WINDOW, token_start, length, storage
    )


def logical_raw_cache(state: RawCacheState) -> np.ndarray:
    """Return committed rows as ``[stage=3, logical_token, raw_width=512]``."""

    _validate_raw_cache_state(state)
    if state.length == 0:
        return np.empty(
            (DSPARK_STAGE_COUNT, 0, DSPARK_RAW_CACHE_WIDTH),
            dtype=np.float64,
        )
    positions = (
        np.arange(state.token_start, state.token_start + state.length)
        % state.capacity
    )
    return np.array(state.rows[:, positions, :], copy=True)


def proposal_raw_cache_view(
    state: RawCacheState,
    token_position: int,
    draft_rows: np.ndarray,
) -> np.ndarray:
    """Expose committed-through-``p`` context plus five candidate rows.

    The returned proposal view is temporary and does not mutate the committed
    rings.  ``token_position`` is the first candidate input position ``p+1``;
    the target-derived row for ``p`` is already in ``state``.  Supplying it a
    second time would duplicate the frontier, which is the historical antirez
    behavior rather than the pinned official model.  Rejection or abandonment
    of a proposal cannot advance cache ownership.
    """

    _validate_raw_cache_state(state)
    if (isinstance(token_position, bool) or
            not isinstance(token_position, (int, np.integer))):
        raise ValueError("token_position must be an integer")
    draft = np.asarray(draft_rows, dtype=np.float64)
    width = state.rows.shape[2]
    if (
        draft.ndim != 3
        or draft.shape[0] != DSPARK_STAGE_COUNT
        or draft.shape[2] != width
    ):
        raise ValueError("draft_rows must have shape [3, block, 512]")
    if draft.shape[1] != 5:
        raise ValueError("final 0731 proposal raw cache requires five draft rows")
    if token_position != state.token_start + state.length:
        raise ValueError("proposal position must equal committed cache end")
    if not np.all(np.isfinite(draft)):
        raise ValueError("proposal raw-cache rows must be finite")
    committed = logical_raw_cache(state)
    return np.concatenate((committed, draft), axis=1)


def commit_raw_cache_transaction(
    state: RawCacheState,
    start_position: int,
    target_stage_rows: np.ndarray,
    accepted_rows: int,
) -> RawCacheState:
    """Atomically append every accepted target-derived verifier row.

    The complete candidate batch is validated before a private copy is
    changed.  Only its accepted prefix is appended in absolute-position order;
    rejected rows remain transient.  Zero accepted rows is an explicit
    rollback.  This oracle models cache ownership, not token acceptance: the
    caller supplies rows captured from the authoritative target verifier.
    """

    _validate_raw_cache_state(state)
    rows = np.asarray(target_stage_rows, dtype=np.float64)
    if (
        rows.ndim != 3
        or rows.shape[1:] != (DSPARK_STAGE_COUNT, DSPARK_RAW_CACHE_WIDTH)
    ):
        raise ValueError("target_stage_rows must have shape [token, 3, 512]")
    if not np.all(np.isfinite(rows)):
        raise ValueError("target_stage_rows must be finite")
    if (isinstance(start_position, bool) or
            not isinstance(start_position, (int, np.integer)) or
            start_position < 0):
        raise ValueError("start_position must be a non-negative integer")
    if (isinstance(accepted_rows, bool) or
            not isinstance(accepted_rows, (int, np.integer)) or
            accepted_rows < 0 or accepted_rows > rows.shape[0]):
        raise ValueError("accepted_rows must select a valid target-row prefix")
    expected = state.token_start + state.length
    if int(start_position) != expected:
        raise ValueError(
            f"transaction start {start_position} does not follow {expected}"
        )
    if accepted_rows == 0:
        return state

    storage = np.array(state.rows, copy=True)
    token_start = state.token_start
    length = state.length
    for offset in range(int(accepted_rows)):
        position = int(start_position) + offset
        storage[:, position % state.capacity, :] = rows[offset]
        length = min(length + 1, state.capacity)
        token_start = position - length + 1
    return RawCacheState(state.capacity, token_start, length, storage)


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
