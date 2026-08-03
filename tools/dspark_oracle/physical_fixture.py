"""Payload-first fixtures for the stage-zero attention and FFN white boxes.

The fixtures are deliberately synthetic.  Their Q8_0 and F16 weights are first
serialized in the formats consumed by the runtime and then reopened from those
bytes; routed FFN weights exist only as valid IQ2_XXS/Q2_K payloads and are
decoded lazily.  Only reopened matrices enter the numerical oracle.  This
proves a test-hook contract, not parity with checkpoint weights.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .reference import RawCacheState, prefill_raw_cache


PHYSICAL_HIDDEN_WIDTH = 32
PHYSICAL_Q_RANK = 32
PHYSICAL_OUTPUT_RANK = 4
PHYSICAL_MODEL_ALIGNMENT = 64
Q8_0_BLOCK = 32
Q8_0_BLOCK_BYTES = 34
PHYSICAL_Q8_REQUIRED_PATHS = MappingProxyType({
    "q_a_weight": "generic_mm_half_staged",
    "q_b_weight": "generic_mm_half_staged",
    "kv_weight": "generic_mm_half_staged",
    "output_a_weight": "direct_grouped_q8_matvec_f32",
    "output_b_weight": "generic_mm_half_staged",
})


@dataclass(frozen=True)
class PackedWeight:
    """One serialized model weight and its payload-derived float32 view."""

    storage: str
    logical_shape: tuple[int, ...]
    payload: bytes
    dequantized: np.ndarray

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()


@dataclass(frozen=True)
class LazyPackedMatrix:
    """One valid low-bit GGML matrix decoded only when an expert is used."""

    storage: str
    logical_shape: tuple[int, int]
    payload: bytes

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()

    def dequantized(self) -> np.ndarray:
        if self.storage == "IQ2_XXS":
            return unpack_iq2_xxs(self.payload, self.logical_shape)
        if self.storage == "Q2_K":
            return unpack_q2_k(self.payload, self.logical_shape)
        raise ValueError(f"unsupported lazy fixture storage {self.storage}")


@dataclass(frozen=True)
class PhysicalFFNFixture:
    """Five-row, production-width FFN fixture with a compact 256-wide mid."""

    inputs: Mapping[str, object]
    packed_weights: Mapping[str, PackedWeight]
    routed_expert_weights: Mapping[int, Mapping[str, LazyPackedMatrix]]
    expected_selected: np.ndarray


@dataclass(frozen=True)
class PhysicalStageZeroFixture:
    """Complete deterministic input for one C=2 or C=128 white-box run."""

    committed_count: int
    inputs: Mapping[str, object]
    ideal_weights: Mapping[str, np.ndarray]
    packed_weights: Mapping[str, PackedWeight]
    model_blob: bytes
    model_offsets: Mapping[str, int]


FFN_HIDDEN_WIDTH = 4096
FFN_MID_WIDTH = 256
FFN_EXPERT_COUNT = 256
FFN_TOPK = 6
IQ2_XXS_BLOCK = 256
IQ2_XXS_BLOCK_BYTES = 66
Q2_K_BLOCK = 256
Q2_K_BLOCK_BYTES = 84
FFN_SELECTED_EXPERTS = np.asarray([
    [5, 0, 4, 1, 3, 2],
    [11, 6, 10, 7, 9, 8],
    [17, 12, 16, 13, 15, 14],
    [23, 18, 22, 19, 20, 21],
    [255, 24, 30, 27, 29, 28],
], dtype=np.int32)


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value)
    result.setflags(write=False)
    return result


def _round_away_from_zero(value: np.ndarray) -> np.ndarray:
    """Match C ``roundf`` for finite values inside the int8 range."""

    source = np.asarray(value, dtype=np.float32)
    rounded = np.copysign(
        np.floor(np.abs(source) + np.float32(0.5)), source
    )
    return rounded.astype(np.int16)


def pack_q8_0(value: np.ndarray) -> PackedWeight:
    """Serialize GGML Q8_0 exactly like ``ds4q_quantize_q8_0``.

    The int8 codes use the original float32 ``d = amax / 127`` and C
    round-away-from-zero.  Only after the codes are chosen is ``d`` rounded to
    the little-endian F16 scale stored in each 34-byte block.
    """

    source = np.ascontiguousarray(value, dtype=np.float32)
    if source.ndim < 2 or source.shape[-1] % Q8_0_BLOCK:
        raise ValueError("Q8_0 input must be a matrix with width divisible by 32")
    if not np.all(np.isfinite(source)):
        raise ValueError("Q8_0 input must be finite")
    blocks = source.reshape(-1, Q8_0_BLOCK)
    amax = np.max(np.abs(blocks), axis=1).astype(np.float32)
    scales_f32 = (amax / np.float32(127.0)).astype(np.float32)
    inverse = np.zeros_like(scales_f32)
    np.divide(
        np.float32(1.0), scales_f32,
        out=inverse, where=scales_f32 != 0.0,
    )
    scaled = np.multiply(
        blocks, inverse[:, None], dtype=np.float32
    )
    codes_i16 = _round_away_from_zero(scaled)
    if np.any(codes_i16 < -127) or np.any(codes_i16 > 127):
        raise ValueError("Q8_0 code escaped the signed 127 range")
    codes = codes_i16.astype(np.int8)
    scales_f16 = scales_f32.astype("<f2")

    packed = np.empty((blocks.shape[0], Q8_0_BLOCK_BYTES), dtype=np.uint8)
    packed[:, :2] = scales_f16.view(np.uint8).reshape(-1, 2)
    packed[:, 2:] = codes.view(np.uint8)
    payload = packed.tobytes(order="C")
    return unpack_q8_0(payload, source.shape)


def unpack_q8_0(payload: bytes, shape: tuple[int, ...]) -> PackedWeight:
    """Reopen a Q8_0 payload using its serialized F16 scales."""

    if len(shape) < 2 or shape[-1] % Q8_0_BLOCK:
        raise ValueError("Q8_0 shape width must be divisible by 32")
    element_count = int(np.prod(shape, dtype=np.int64))
    if element_count % Q8_0_BLOCK:
        raise ValueError("Q8_0 shape must contain complete blocks")
    block_count = element_count // Q8_0_BLOCK
    if len(payload) != block_count * Q8_0_BLOCK_BYTES:
        raise ValueError("Q8_0 payload size does not match shape")
    packed = np.frombuffer(payload, dtype=np.uint8).reshape(
        block_count, Q8_0_BLOCK_BYTES
    )
    scales = np.ascontiguousarray(packed[:, :2]).view("<f2").astype(
        np.float32
    ).reshape(-1)
    codes = packed[:, 2:].view(np.int8).astype(np.float32)
    reopened = np.multiply(codes, scales[:, None], dtype=np.float32).reshape(
        shape
    )
    if not np.all(np.isfinite(reopened)):
        raise ValueError("Q8_0 payload decoded a non-finite value")
    return PackedWeight("Q8_0", tuple(shape), bytes(payload), _readonly(reopened))


def pack_synthetic_iq2_xxs(
    shape: tuple[int, int], *, seed: int
) -> LazyPackedMatrix:
    """Build deterministic valid IQ2_XXS blocks without an ideal matrix.

    Every 8-value codebook index is zero, whose admitted GGML grid is eight
    equal magnitude-8 lanes.  Expert/row/block asymmetry comes from the eight
    independent 4-bit group scales and the four even-parity sign masks per
    group.  Constructing codes directly is intentional: only payload-decoded
    values exist, so no float matrix can leak around the low-bit contract.
    """

    if (len(shape) != 2 or shape[1] % IQ2_XXS_BLOCK):
        raise ValueError("IQ2_XXS shape width must be divisible by 256")
    rows, width = shape
    if rows <= 0:
        raise ValueError("IQ2_XXS shape must be non-empty")
    block_count = rows * (width // IQ2_XXS_BLOCK)
    block_index = np.arange(block_count, dtype=np.uint32)
    blocks = np.zeros((block_count, IQ2_XXS_BLOCK_BYTES), dtype=np.uint8)
    scales = np.where(
        ((block_index + np.uint32(seed)) & np.uint32(1)) == 0,
        np.float32(1.0 / 16.0),
        np.float32(1.0 / 32.0),
    ).astype("<f2")
    blocks[:, :2] = scales.view(np.uint8).reshape(-1, 2)
    for group32 in range(8):
        packed_signs = np.zeros(block_count, dtype=np.uint32)
        for group8 in range(4):
            sign_index = (
                np.uint32(seed * 17 + group32 * 11 + group8 * 23)
                + block_index * np.uint32(29)
            ) & np.uint32(127)
            packed_signs |= sign_index << np.uint32(7 * group8)
        scale_nibble = (
            np.uint32(seed + group32 * 3) + block_index
        ) & np.uint32(3)
        packed_signs |= scale_nibble << np.uint32(28)
        offset = 2 + group32 * 8
        # Four zero grid-index bytes are followed by sign/scale little endian.
        blocks[:, offset + 4:offset + 8] = packed_signs.astype(
            "<u4"
        ).view(np.uint8).reshape(-1, 4)
    return LazyPackedMatrix(
        "IQ2_XXS", shape, blocks.tobytes(order="C")
    )


def unpack_iq2_xxs(payload: bytes, shape: tuple[int, int]) -> np.ndarray:
    """Decode the deliberately narrow but physically valid IQ2 fixture."""

    if len(shape) != 2 or shape[1] % IQ2_XXS_BLOCK:
        raise ValueError("IQ2_XXS shape width must be divisible by 256")
    block_count = shape[0] * (shape[1] // IQ2_XXS_BLOCK)
    if len(payload) != block_count * IQ2_XXS_BLOCK_BYTES:
        raise ValueError("IQ2_XXS payload size does not match shape")
    blocks = np.frombuffer(payload, dtype=np.uint8).reshape(
        block_count, IQ2_XXS_BLOCK_BYTES
    )
    d = np.ascontiguousarray(blocks[:, :2]).view("<f2").astype(
        np.float32
    ).reshape(-1)
    result = np.empty((block_count, IQ2_XXS_BLOCK), dtype=np.float32)
    for group32 in range(8):
        offset = 2 + group32 * 8
        if np.any(blocks[:, offset:offset + 4] != 0):
            raise ValueError("synthetic IQ2_XXS payload escaped grid index zero")
        packed = np.ascontiguousarray(
            blocks[:, offset + 4:offset + 8]
        ).view("<u4").reshape(-1)
        magnitude = (
            d * np.float32(0.125)
            * (np.float32(2.0) * (packed >> np.uint32(28)).astype(np.float32)
               + np.float32(1.0))
            * np.float32(8.0)
        )
        for group8 in range(4):
            sign_index = (
                packed >> np.uint32(7 * group8)
            ) & np.uint32(127)
            parity = np.array(sign_index, copy=True)
            parity ^= parity >> np.uint32(4)
            parity ^= parity >> np.uint32(2)
            parity ^= parity >> np.uint32(1)
            parity &= np.uint32(1)
            sign_mask = sign_index | (parity << np.uint32(7))
            first = group32 * 32 + group8 * 8
            for lane in range(8):
                negative = (
                    sign_mask & np.uint32(1 << lane)
                ) != 0
                result[:, first + lane] = np.where(
                    negative, -magnitude, magnitude
                )
    if not np.all(np.isfinite(result)):
        raise ValueError("IQ2_XXS payload decoded a non-finite value")
    return result.reshape(shape)


def pack_synthetic_q2_k(
    shape: tuple[int, int], *, seed: int
) -> LazyPackedMatrix:
    """Build asymmetric valid Q2_K blocks for compact down projections."""

    if len(shape) != 2 or shape[1] % Q2_K_BLOCK:
        raise ValueError("Q2_K shape width must be divisible by 256")
    rows, width = shape
    if rows <= 0:
        raise ValueError("Q2_K shape must be non-empty")
    block_count = rows * (width // Q2_K_BLOCK)
    block_index = np.arange(block_count, dtype=np.uint32)
    blocks = np.zeros((block_count, Q2_K_BLOCK_BYTES), dtype=np.uint8)
    for group in range(16):
        scale = np.uint8(1 + ((seed + group) % 3))
        minimum = np.uint8(1 + ((seed + 2 * group) % 2))
        blocks[:, group] = scale | np.uint8(minimum << 4)
        q_base = 32 * (group // 8) + 16 * (group & 1)
        shift = ((group // 2) & 3) * 2
        for lane in range(16):
            q = (
                block_index
                + np.uint32(seed * 13 + group * 7 + lane * 3)
            ) & np.uint32(3)
            blocks[:, 16 + q_base + lane] |= (
                q << np.uint32(shift)
            ).astype(np.uint8)
    d = np.where(
        ((block_index + np.uint32(seed)) & np.uint32(1)) == 0,
        np.float32(1.0 / 128.0),
        np.float32(1.0 / 256.0),
    ).astype("<f2")
    dmin = np.full(block_count, np.float32(1.0 / 256.0), dtype="<f2")
    blocks[:, 80:82] = d.view(np.uint8).reshape(-1, 2)
    blocks[:, 82:84] = dmin.view(np.uint8).reshape(-1, 2)
    return LazyPackedMatrix("Q2_K", shape, blocks.tobytes(order="C"))


def unpack_q2_k(payload: bytes, shape: tuple[int, int]) -> np.ndarray:
    """Decode Q2_K with the same 16-value group addressing as Hebrus."""

    if len(shape) != 2 or shape[1] % Q2_K_BLOCK:
        raise ValueError("Q2_K shape width must be divisible by 256")
    block_count = shape[0] * (shape[1] // Q2_K_BLOCK)
    if len(payload) != block_count * Q2_K_BLOCK_BYTES:
        raise ValueError("Q2_K payload size does not match shape")
    blocks = np.frombuffer(payload, dtype=np.uint8).reshape(
        block_count, Q2_K_BLOCK_BYTES
    )
    d = np.ascontiguousarray(blocks[:, 80:82]).view("<f2").astype(
        np.float32
    ).reshape(-1)
    dmin = np.ascontiguousarray(blocks[:, 82:84]).view("<f2").astype(
        np.float32
    ).reshape(-1)
    result = np.empty((block_count, Q2_K_BLOCK), dtype=np.float32)
    for group in range(16):
        q_base = 32 * (group // 8) + 16 * (group & 1)
        shift = ((group // 2) & 3) * 2
        sc = blocks[:, group]
        scale = (sc & np.uint8(15)).astype(np.float32)
        minimum = (sc >> np.uint8(4)).astype(np.float32)
        for lane in range(16):
            q = (
                blocks[:, 16 + q_base + lane] >> np.uint8(shift)
            ) & np.uint8(3)
            result[:, group * 16 + lane] = (
                d * scale * q.astype(np.float32) - dmin * minimum
            )
    if not np.all(np.isfinite(result)):
        raise ValueError("Q2_K payload decoded a non-finite value")
    return result.reshape(shape)


def pack_f16(value: np.ndarray) -> PackedWeight:
    """Serialize and reopen a little-endian F16 tensor."""

    source = np.ascontiguousarray(value, dtype=np.float32)
    if source.ndim == 0 or not np.all(np.isfinite(source)):
        raise ValueError("F16 input must be a finite tensor")
    payload = source.astype("<f2").tobytes(order="C")
    reopened = np.frombuffer(payload, dtype="<f2").astype(np.float32).reshape(
        source.shape
    )
    if not np.all(np.isfinite(reopened)):
        raise ValueError("F16 payload decoded a non-finite value")
    return PackedWeight("F16", source.shape, payload, _readonly(reopened))


def pack_f32(value: np.ndarray) -> PackedWeight:
    """Serialize and reopen a little-endian F32 tensor."""

    source = np.ascontiguousarray(value, dtype="<f4")
    if source.ndim == 0 or not np.all(np.isfinite(source)):
        raise ValueError("F32 input must be a finite tensor")
    payload = source.tobytes(order="C")
    reopened = np.frombuffer(payload, dtype="<f4").reshape(source.shape)
    return PackedWeight("F32", source.shape, payload, _readonly(reopened))


def _single_q8_coefficient(
    rows: int, width: int, selector: np.ndarray, signs: np.ndarray
) -> np.ndarray:
    """Build rows with one live Q8 code plus payload-invisible dyadic dust.

    Every other lane in the selected 32-wide block receives +/-2^-10.  With
    d=2^-8 this is exactly one quarter of a code and therefore serializes to
    zero.  The dust proves that consumers use the reopened payload instead of
    accidentally retaining the ideal float matrix.  Entirely zero blocks,
    especially the 127 unused output-A blocks, remain untouched.
    """

    if selector.shape != (rows,) or signs.shape != (rows,):
        raise ValueError("Q8 selector/sign geometry mismatch")
    result = np.zeros((rows, width), dtype=np.float32)
    block_start = (selector // Q8_0_BLOCK) * Q8_0_BLOCK
    lane = selector % Q8_0_BLOCK
    for offset in range(1, Q8_0_BLOCK):
        dust_column = block_start + ((lane + offset) % Q8_0_BLOCK)
        result[np.arange(rows), dust_column] = (
            signs.astype(np.float32) * np.float32(1.0 / 1024.0)
        )
    result[np.arange(rows), selector] = (
        signs.astype(np.float32) * np.float32(127.0 / 256.0)
    )
    return result


def _ideal_weights() -> dict[str, np.ndarray]:
    hidden = PHYSICAL_HIDDEN_WIDTH
    q_rank = PHYSICAL_Q_RANK
    output_rank = PHYSICAL_OUTPUT_RANK

    hc_function = np.zeros((24, 4 * hidden), dtype=np.float32)
    hc_rows = np.arange(24)
    hc_function[hc_rows, (hc_rows * 17 + 3) % (4 * hidden)] = np.where(
        (hc_rows & 1) == 0, np.float32(0.125), np.float32(-0.0625)
    )

    q_a_rows = np.arange(q_rank)
    q_a = _single_q8_coefficient(
        q_rank,
        hidden,
        q_a_rows,
        np.where((q_a_rows & 1) == 0, 1, -1),
    )

    head = np.repeat(np.arange(64, dtype=np.int64), 512)
    dimension = np.tile(np.arange(512, dtype=np.int64), 64)
    q_selector = (head + dimension) % q_rank
    q_sign = np.where(
        (((dimension >> (head % 7)) ^ head) & 1) == 0, 1, -1
    )
    q_b = _single_q8_coefficient(
        64 * 512, q_rank, q_selector, q_sign
    )

    kv_rows = np.arange(512, dtype=np.int64)
    kv = _single_q8_coefficient(
        512,
        hidden,
        kv_rows % hidden,
        np.ones(512, dtype=np.int64),
    )

    output_rows = 8 * output_rank
    output_row = np.arange(output_rows, dtype=np.int64)
    output_selector = (
        ((output_row // output_rank) * 521 +
         (output_row % output_rank) * 977 + 37) % (8 * 512)
    )
    output_a_flat = _single_q8_coefficient(
        output_rows,
        8 * 512,
        output_selector,
        np.where((output_row & 1) == 0, 1, -1),
    )
    output_a = output_a_flat.reshape(8, output_rank, 8 * 512)

    output_b_rows = np.arange(hidden, dtype=np.int64)
    output_b = _single_q8_coefficient(
        hidden,
        8 * output_rank,
        (output_b_rows * 13 + 5) % (8 * output_rank),
        np.where((output_b_rows % 3) == 0, -1, 1),
    )

    return {
        "hc_function_weight": hc_function,
        "hc_scale": np.asarray([0.25, 0.125, 0.0625], dtype=np.float32),
        "hc_base": ((np.arange(24) % 7) - 3).astype(np.float32) / 16.0,
        "attention_norm_weight": (
            (np.arange(hidden) % 11).astype(np.float32) + 8.0
        ) / 16.0,
        "q_a_weight": q_a,
        "q_a_norm_weight": np.ones(q_rank, dtype=np.float32),
        "q_b_weight": q_b,
        "kv_weight": kv,
        "kv_norm_weight": (
            (np.arange(512) % 11).astype(np.float32) + 8.0
        ) / 512.0,
        "attention_sinks": np.linspace(
            -1.5, 1.125, 64, dtype=np.float32
        ),
        "output_a_weight": output_a,
        "output_b_weight": output_b,
    }


_WEIGHT_STORAGE = {
    "hc_function_weight": "F16",
    "hc_scale": "F32",
    "hc_base": "F32",
    "attention_norm_weight": "F32",
    "q_a_weight": "Q8_0",
    "q_a_norm_weight": "F32",
    "q_b_weight": "Q8_0",
    "kv_weight": "Q8_0",
    "kv_norm_weight": "F32",
    "attention_sinks": "F32",
    "output_a_weight": "Q8_0",
    "output_b_weight": "Q8_0",
}


def _pack_weights(
    ideal: Mapping[str, np.ndarray],
) -> dict[str, PackedWeight]:
    packers = {"Q8_0": pack_q8_0, "F16": pack_f16, "F32": pack_f32}
    return {
        name: packers[storage](ideal[name])
        for name, storage in _WEIGHT_STORAGE.items()
    }


def _model_blob(
    packed: Mapping[str, PackedWeight],
) -> tuple[bytes, dict[str, int]]:
    blob = bytearray()
    offsets: dict[str, int] = {}
    for name in _WEIGHT_STORAGE:
        aligned = (
            (len(blob) + PHYSICAL_MODEL_ALIGNMENT - 1)
            // PHYSICAL_MODEL_ALIGNMENT
            * PHYSICAL_MODEL_ALIGNMENT
        )
        blob.extend(b"\0" * (aligned - len(blob)))
        offsets[name] = aligned
        blob.extend(packed[name].payload)
    return bytes(blob), offsets


def _hidden_input() -> np.ndarray:
    token, lane, dimension = np.indices((5, 4, 32), dtype=np.int64)
    base = (
        np.float32(0.25)
        + ((token * 17 + lane * 11 + dimension * 7) % 17).astype(np.float32)
        / np.float32(64.0)
    )
    # Every raw lane is deliberately off its exact BF16 value.  The +/-2^-12
    # dust is below half an ULP throughout [0.25, 0.5], so the first required
    # BF16 publication reopens exactly ``base`` while skipping it is visible.
    dust_sign = np.where(
        ((token + lane + dimension) & 1) == 0,
        np.float32(1.0),
        np.float32(-1.0),
    )
    return base + dust_sign * np.float32(2.0 ** -12)


def _raw_rows() -> np.ndarray:
    token, stage, dimension = np.indices((130, 3, 512), dtype=np.int64)
    return (
        ((token * 19 + stage * 23 + dimension * 29) % 113).astype(np.float32)
        + np.float32(16.0)
    ) / np.float32(4096.0)


def _transient_rows() -> np.ndarray:
    stage, token, dimension = np.indices((3, 5, 512), dtype=np.int64)
    return (
        ((stage * 31 + token * 17 + dimension * 13) % 67).astype(np.float32)
        - np.float32(33.0)
    ) / np.float32(256.0)


def build_physical_stage_zero_fixture(
    committed_count: int,
) -> PhysicalStageZeroFixture:
    """Build one payload-derived C=2 or wrapped C=128 physical fixture."""

    if isinstance(committed_count, bool) or committed_count not in (2, 128):
        raise ValueError("physical fixture admits only C=2 or C=128")
    ideal = _ideal_weights()
    packed = _pack_weights(ideal)
    blob, offsets = _model_blob(packed)
    raw_source = _raw_rows()
    state: RawCacheState = prefill_raw_cache(
        raw_source[:2] if committed_count == 2 else raw_source,
        start_position=0,
    )
    positions = np.arange(
        state.token_start + state.length,
        state.token_start + state.length + 5,
        dtype=np.int64,
    )
    inputs: dict[str, object] = {
        "hidden_input": _hidden_input(),
        "absolute_positions": positions,
        "raw_cache": state,
        "other_stage_draft_rows": _transient_rows(),
    }
    inputs.update({
        name: weight.dequantized for name, weight in packed.items()
    })
    return PhysicalStageZeroFixture(
        committed_count,
        MappingProxyType(inputs),
        MappingProxyType({name: _readonly(value) for name, value in ideal.items()}),
        MappingProxyType(packed),
        blob,
        MappingProxyType(offsets),
    )


def payload_manifest(fixture: PhysicalStageZeroFixture) -> dict[str, object]:
    """Return the small C-handoff contract without embedding payload bytes."""

    state = fixture.inputs["raw_cache"]
    stage_zero_ring = np.asarray(state.rows[0], dtype="<f4").tobytes(order="C")
    raw_hidden = np.asarray(
        fixture.inputs["hidden_input"], dtype="<f4"
    ).tobytes(order="C")

    def weight_record(name: str, weight: PackedWeight) -> dict[str, object]:
        record: dict[str, object] = {
            "storage": weight.storage,
            "oracle_shape": list(weight.logical_shape),
            "offset": fixture.model_offsets[name],
            "bytes": len(weight.payload),
            "sha256": weight.sha256,
        }
        if weight.storage == "Q8_0":
            input_width = weight.logical_shape[-1]
            output_rows = int(np.prod(weight.logical_shape[:-1]))
            record.update({
                "gguf_ne": [input_width, output_rows],
                "row_bytes": input_width // Q8_0_BLOCK * Q8_0_BLOCK_BYTES,
                "output_rows": output_rows,
            })
        return record

    return {
        "fixture_version": 1,
        "geometry": {
            "proposal_rows": 5,
            "hc_lanes": 4,
            "hidden_width": PHYSICAL_HIDDEN_WIDTH,
            "q_rank": PHYSICAL_Q_RANK,
            "attention_heads": 64,
            "head_width": 512,
            "output_groups": 8,
            "output_rank": PHYSICAL_OUTPUT_RANK,
        },
        "alignment": PHYSICAL_MODEL_ALIGNMENT,
        "blob_bytes": len(fixture.model_blob),
        "blob_sha256": hashlib.sha256(fixture.model_blob).hexdigest(),
        # These are the production paths the native hook is required to use.
        # This manifest records the intended contract; the hook must capture
        # actual dispatch selection independently.
        "required_q8_paths": dict(PHYSICAL_Q8_REQUIRED_PATHS),
        "raw_hidden_input": {
            "shape": [5, 4, PHYSICAL_HIDDEN_WIDTH],
            "bytes": len(raw_hidden),
            "sha256": hashlib.sha256(raw_hidden).hexdigest(),
        },
        "stage_zero_raw_cache": {
            "capacity": state.capacity,
            "token_start": state.token_start,
            "length": state.length,
            "shape": [128, 512],
            "bytes": len(stage_zero_ring),
            "sha256": hashlib.sha256(stage_zero_ring).hexdigest(),
        },
        "weights": {
            name: weight_record(name, weight)
            for name, weight in fixture.packed_weights.items()
        },
    }


def _ffn_hidden_input() -> np.ndarray:
    hidden = np.zeros((5, 4, FFN_HIDDEN_WIDTH), dtype=np.float32)
    for token in range(5):
        for lane in range(4):
            base = np.float32(0.5 + lane / 16.0)
            dust = np.float32(2.0 ** -12) * (
                np.float32(1.0) if ((token + lane) & 1) == 0
                else np.float32(-1.0)
            )
            hidden[token, lane, token] = base + dust
    return hidden


def _ffn_ideal_weights() -> dict[str, np.ndarray]:
    hc_function = np.zeros((24, 4 * FFN_HIDDEN_WIDTH), dtype=np.float32)
    for row in range(24):
        lane = (row * 3 + 1) % 4
        dimension = row % 5
        hc_function[row, lane * FFN_HIDDEN_WIDTH + dimension] = np.float32(
            ((row % 7) - 3) / 32.0
        )

    router = np.zeros(
        (FFN_EXPERT_COUNT, FFN_HIDDEN_WIDTH), dtype=np.float32
    )
    bias = np.zeros(FFN_EXPERT_COUNT, dtype=np.float32)
    for token, experts in enumerate(FFN_SELECTED_EXPERTS):
        for slot, expert_value in enumerate(experts):
            expert = int(expert_value)
            router[expert, token] = np.float32(
                127 * (14 + (expert % 7)) / 4096.0
            )
            for dust_lane in range(5, 32):
                router[expert, dust_lane] = np.float32(
                    (1.0 if ((expert + dust_lane) & 1) == 0 else -1.0)
                    / 1024.0
                )
            bias[expert] = np.float32((5 - 2 * slot) * 0.4)

    shared_rows = np.arange(FFN_MID_WIDTH, dtype=np.int64)
    shared_gate = _single_q8_coefficient(
        FFN_MID_WIDTH,
        FFN_HIDDEN_WIDTH,
        (shared_rows * 37 + 3) % FFN_HIDDEN_WIDTH,
        np.where((shared_rows & 1) == 0, 1, -1),
    )
    # Make every fifth gate row consume one of the five live marker lanes so
    # the official upper-only gate clamp is exercised on every token.
    shared_gate[np.arange(5), :] = 0.0
    shared_gate[np.arange(5), np.arange(5)] = np.float32(127.0 / 256.0)
    shared_up = _single_q8_coefficient(
        FFN_MID_WIDTH,
        FFN_HIDDEN_WIDTH,
        (shared_rows * 53 + 1) % FFN_HIDDEN_WIDTH,
        np.where((shared_rows % 3) == 0, -1, 1),
    )
    shared_up[np.arange(5), :] = 0.0
    shared_up[np.arange(5), np.arange(5)] = np.float32(-127.0 / 256.0)
    down_rows = np.arange(FFN_HIDDEN_WIDTH, dtype=np.int64)
    shared_down = _single_q8_coefficient(
        FFN_HIDDEN_WIDTH,
        FFN_MID_WIDTH,
        (down_rows * 19 + 7) % FFN_MID_WIDTH,
        np.where((down_rows % 5) < 2, -1, 1),
    )
    return {
        "hc_ffn_function_weight": hc_function,
        "hc_ffn_scale": np.asarray([0.25, 0.125, 0.0625], dtype=np.float32),
        "hc_ffn_base": ((np.arange(24) % 9) - 4).astype(np.float32) / 32.0,
        "ffn_norm_weight": (
            (np.arange(FFN_HIDDEN_WIDTH) % 13).astype(np.float32) + 12.0
        ) / 16.0,
        "router_weight": router,
        "selection_bias": bias,
        "shared_gate_weight": shared_gate,
        "shared_up_weight": shared_up,
        "shared_down_weight": shared_down,
    }


def build_physical_ffn_fixture() -> PhysicalFFNFixture:
    """Build the five-row, 30-distinct-expert FFN payload transaction."""

    ideal = _ffn_ideal_weights()
    packed = {
        "hc_ffn_function_weight": pack_f16(
            ideal["hc_ffn_function_weight"]
        ),
        "hc_ffn_scale": pack_f32(ideal["hc_ffn_scale"]),
        "hc_ffn_base": pack_f32(ideal["hc_ffn_base"]),
        "ffn_norm_weight": pack_f32(ideal["ffn_norm_weight"]),
        "router_weight": pack_q8_0(ideal["router_weight"]),
        "selection_bias": pack_f32(ideal["selection_bias"]),
        "shared_gate_weight": pack_q8_0(ideal["shared_gate_weight"]),
        "shared_up_weight": pack_q8_0(ideal["shared_up_weight"]),
        "shared_down_weight": pack_q8_0(ideal["shared_down_weight"]),
    }
    routed: dict[int, Mapping[str, LazyPackedMatrix]] = {}
    for expert_value in FFN_SELECTED_EXPERTS.reshape(-1):
        expert = int(expert_value)
        record = {
            "gate": pack_synthetic_iq2_xxs(
                (FFN_MID_WIDTH, FFN_HIDDEN_WIDTH), seed=expert * 3 + 1
            ),
            "up": pack_synthetic_iq2_xxs(
                (FFN_MID_WIDTH, FFN_HIDDEN_WIDTH), seed=expert * 3 + 2
            ),
            "down": pack_synthetic_q2_k(
                (FFN_HIDDEN_WIDTH, FFN_MID_WIDTH), seed=expert * 3 + 3
            ),
        }
        routed[expert] = MappingProxyType(record)
    inputs: dict[str, object] = {
        "hidden_input": _ffn_hidden_input(),
        "hc_function_weight": packed["hc_ffn_function_weight"],
        "hc_scale": packed["hc_ffn_scale"].dequantized,
        "hc_base": packed["hc_ffn_base"].dequantized,
        "ffn_norm_weight": packed["ffn_norm_weight"].dequantized,
        "router_weight": packed["router_weight"],
        "selection_bias": packed["selection_bias"].dequantized,
        "shared_gate_weight": packed["shared_gate_weight"],
        "shared_up_weight": packed["shared_up_weight"],
        "shared_down_weight": packed["shared_down_weight"],
        "routed_expert_weights": MappingProxyType(routed),
    }
    return PhysicalFFNFixture(
        MappingProxyType(inputs),
        MappingProxyType(packed),
        MappingProxyType(routed),
        _readonly(np.array(FFN_SELECTED_EXPERTS, copy=True)),
    )


def ffn_payload_manifest(fixture: PhysicalFFNFixture) -> dict[str, object]:
    """Return stable identities for the compact-mid production-width fixture."""

    routed_digest = hashlib.sha256()
    routed_bytes = 0
    routed_records: dict[str, object] = {}
    for expert in sorted(fixture.routed_expert_weights):
        record = fixture.routed_expert_weights[expert]
        component_record: dict[str, object] = {}
        for role in ("gate", "up", "down"):
            weight = record[role]
            routed_digest.update(weight.payload)
            routed_bytes += len(weight.payload)
            component_record[role] = {
                "storage": weight.storage,
                "shape": list(weight.logical_shape),
                "bytes": len(weight.payload),
                "sha256": weight.sha256,
            }
        routed_records[str(expert)] = component_record
    return {
        "fixture_version": 1,
        "geometry": {
            "proposal_rows": 5,
            "hc_lanes": 4,
            "hidden_width": FFN_HIDDEN_WIDTH,
            "expert_count": FFN_EXPERT_COUNT,
            "topk": FFN_TOPK,
            "compact_mid_width": FFN_MID_WIDTH,
        },
        "selected_experts": fixture.expected_selected.tolist(),
        "non_routed": {
            name: {
                "storage": weight.storage,
                "shape": list(weight.logical_shape),
                "bytes": len(weight.payload),
                "sha256": weight.sha256,
            }
            for name, weight in fixture.packed_weights.items()
        },
        "routed_payload_bytes": routed_bytes,
        "routed_payload_sha256": routed_digest.hexdigest(),
        "routed_records": routed_records,
    }
