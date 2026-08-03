#!/usr/bin/env python3
"""Payload-first fixture for the disconnected three-stage DSpark proposal.

This fixture is intentionally narrower than a synthetic full transformer.  It
joins the already qualified attention/MoE checkpoints at their HC state seam,
then exercises the final 0731 HC/norm/output/Markov/confidence transaction.
Every model weight is serialized as Q8_0, F16, or F32 and reopened from those
bytes before it is exposed to the NumPy oracle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np


PROPOSAL_ROWS = 5
PROPOSAL_HC_LANES = 4
PROPOSAL_HIDDEN_WIDTH = 32
PROPOSAL_STAGE_COUNT = 3
PROPOSAL_VOCAB = 8
PROPOSAL_MARKOV_RANK = 256
PROPOSAL_PENDING_TOKEN = 0
PROPOSAL_CONFIDENCE_THRESHOLD = 0.5
Q8_0_BLOCK = 32
Q8_0_BLOCK_BYTES = 34


@dataclass(frozen=True)
class PackedWeight:
    """One standalone fixture payload and its byte-derived float32 view."""

    storage: str
    logical_shape: tuple[int, ...]
    payload: bytes
    dequantized: np.ndarray

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.payload).hexdigest()


@dataclass(frozen=True)
class PhysicalProposalFixture:
    """One deterministic three-stage/head proposal transaction."""

    inputs: Mapping[str, object]
    packed_weights: Mapping[str, PackedWeight]


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value)
    result.setflags(write=False)
    return result


def _round_away_from_zero(value: np.ndarray) -> np.ndarray:
    source = np.asarray(value, dtype=np.float32)
    return np.copysign(
        np.floor(np.abs(source) + np.float32(0.5)), source
    ).astype(np.int16)


def pack_q8_0(value: np.ndarray) -> PackedWeight:
    """Serialize/reopen the fixture's narrow GGML Q8_0 producer codec."""

    source = np.ascontiguousarray(value, dtype=np.float32)
    if source.ndim < 2 or source.shape[-1] % Q8_0_BLOCK:
        raise ValueError("Q8_0 fixture input width must be divisible by 32")
    if not np.all(np.isfinite(source)):
        raise ValueError("Q8_0 fixture input must be finite")
    blocks = source.reshape(-1, Q8_0_BLOCK)
    maximum = np.max(np.abs(blocks), axis=1).astype(np.float32)
    scale_f32 = (maximum / np.float32(127.0)).astype(np.float32)
    inverse = np.zeros_like(scale_f32)
    np.divide(
        np.float32(1.0), scale_f32,
        out=inverse, where=scale_f32 != np.float32(0.0),
    )
    codes_i16 = _round_away_from_zero(
        np.multiply(blocks, inverse[:, None], dtype=np.float32)
    )
    if np.any(codes_i16 < -127) or np.any(codes_i16 > 127):
        raise ValueError("Q8_0 fixture code escaped signed 127 range")
    packed = np.empty((blocks.shape[0], Q8_0_BLOCK_BYTES), dtype=np.uint8)
    packed[:, :2] = scale_f32.astype("<f2").view(np.uint8).reshape(-1, 2)
    packed[:, 2:] = codes_i16.astype(np.int8).view(np.uint8)
    payload = packed.tobytes(order="C")

    reopened_blocks = np.frombuffer(payload, dtype=np.uint8).reshape(
        -1, Q8_0_BLOCK_BYTES
    )
    reopened_scale = np.ascontiguousarray(
        reopened_blocks[:, :2]
    ).view("<f2").astype(np.float32).reshape(-1)
    reopened_codes = reopened_blocks[:, 2:].view(np.int8).astype(np.float32)
    reopened = np.multiply(
        reopened_codes, reopened_scale[:, None], dtype=np.float32
    ).reshape(source.shape)
    if not np.all(np.isfinite(reopened)):
        raise ValueError("Q8_0 fixture payload decoded non-finite data")
    return PackedWeight("Q8_0", source.shape, payload, _readonly(reopened))


def pack_f16(value: np.ndarray) -> PackedWeight:
    """Serialize/reopen the fixture's little-endian F16 producer codec."""

    source = np.ascontiguousarray(value, dtype=np.float32)
    if source.ndim == 0 or not np.all(np.isfinite(source)):
        raise ValueError("F16 fixture input must be a finite tensor")
    payload = source.astype("<f2").tobytes(order="C")
    reopened = np.frombuffer(payload, dtype="<f2").astype(np.float32).reshape(
        source.shape
    )
    if not np.all(np.isfinite(reopened)):
        raise ValueError("F16 fixture payload decoded non-finite data")
    return PackedWeight("F16", source.shape, payload, _readonly(reopened))


def pack_f32(value: np.ndarray) -> PackedWeight:
    """Serialize/reopen the fixture's little-endian F32 producer codec."""

    source = np.ascontiguousarray(value, dtype="<f4")
    if source.ndim == 0 or not np.all(np.isfinite(source)):
        raise ValueError("F32 fixture input must be a finite tensor")
    payload = source.tobytes(order="C")
    reopened = np.frombuffer(payload, dtype="<f4").reshape(source.shape)
    return PackedWeight("F32", source.shape, payload, _readonly(reopened))


def _initial_hc() -> np.ndarray:
    row = np.arange(PROPOSAL_ROWS, dtype=np.int64)[:, None, None]
    lane = np.arange(PROPOSAL_HC_LANES, dtype=np.int64)[None, :, None]
    dimension = np.arange(PROPOSAL_HIDDEN_WIDTH, dtype=np.int64)[None, None, :]
    raw = (row * 19 + lane * 13 + dimension * 7) % 31 - 15
    raw = np.where(raw == 0, 1, raw)
    return raw.astype(np.float32) / np.float32(16.0)


def _main_x() -> np.ndarray:
    dimension = np.arange(PROPOSAL_HIDDEN_WIDTH, dtype=np.int64)
    raw = (dimension * 11) % 29 - 14
    raw = np.where(raw == 0, -1, raw)
    return raw.astype(np.float32) / np.float32(16.0)


def _stage_weight(stage: int, *, main: bool) -> np.ndarray:
    row = np.arange(PROPOSAL_HIDDEN_WIDTH, dtype=np.int64)[:, None]
    column = np.arange(PROPOSAL_HIDDEN_WIDTH, dtype=np.int64)[None, :]
    if main:
        raw = (row * 5 + column * 19 + stage * 11) % 19 - 9
        divisor = np.float32(384.0)
    else:
        raw = (row * 17 + column * 13 + stage * 7) % 23 - 11
        divisor = np.float32(128.0)
    raw = np.where(raw == 0, stage + 1, raw)
    return raw.astype(np.float32) / divisor


def _stage_biases() -> np.ndarray:
    stage = np.arange(PROPOSAL_STAGE_COUNT, dtype=np.int64)[:, None]
    dimension = np.arange(PROPOSAL_HIDDEN_WIDTH, dtype=np.int64)[None, :]
    raw = (stage * 23 + dimension * 3) % 17 - 8
    raw = np.where(raw == 0, stage + 1, raw)
    return raw.astype(np.float32) / np.float32(128.0)


def _head_function() -> np.ndarray:
    row = np.arange(PROPOSAL_HC_LANES, dtype=np.int64)[:, None]
    column = np.arange(
        PROPOSAL_HC_LANES * PROPOSAL_HIDDEN_WIDTH, dtype=np.int64
    )[None, :]
    raw = (row * 29 + column * 7) % 31 - 15
    raw = np.where(raw == 0, row + 1, raw)
    return raw.astype(np.float32) / np.float32(256.0)


def _target_output() -> np.ndarray:
    token = np.arange(PROPOSAL_VOCAB, dtype=np.int64)[:, None]
    dimension = np.arange(PROPOSAL_HIDDEN_WIDTH, dtype=np.int64)[None, :]
    raw = (token * 13 + dimension * 5) % 17 - 8
    raw = np.where(raw == 0, token + 1, raw)
    return raw.astype(np.float32) / np.float32(512.0)


def _markov_w1() -> np.ndarray:
    weight = np.zeros(
        (PROPOSAL_VOCAB, PROPOSAL_MARKOV_RANK), dtype=np.float32
    )
    weight[np.arange(PROPOSAL_VOCAB), np.arange(PROPOSAL_VOCAB)] = 1.0
    return weight


def _markov_w2() -> np.ndarray:
    weight = np.zeros(
        (PROPOSAL_VOCAB, PROPOSAL_MARKOV_RANK), dtype=np.float32
    )
    # Previous token t makes t+1 dominant.  The final row wraps only so every
    # vocabulary row remains a live, valid payload; the five-row checkpoint
    # consumes previous ids 0..4 and must therefore draft 1..5.
    for previous in range(PROPOSAL_VOCAB):
        weight[(previous + 1) % PROPOSAL_VOCAB, previous] = 32.0
    return weight


def _confidence_weight() -> np.ndarray:
    weight = np.zeros(
        (1, PROPOSAL_HIDDEN_WIDTH + PROPOSAL_MARKOV_RANK), dtype=np.float32
    )
    # Make the HC hidden input observable without controlling the sign.  Q8_0
    # packing reopens these values at one exact code step beside the larger
    # Markov coefficients.
    weight[0, :PROPOSAL_HIDDEN_WIDTH] = np.where(
        (np.arange(PROPOSAL_HIDDEN_WIDTH) & 1) == 0, 1.0, -1.0
    ).astype(np.float32) / np.float32(32.0)
    # Previous ids are [pending=0, d0=1, d1=2, d2=3, d3=4].  Position three
    # is the first negative score while position four becomes positive again;
    # a cumulative or "last passing" policy therefore cannot satisfy it.
    weight[0, PROPOSAL_HIDDEN_WIDTH:PROPOSAL_HIDDEN_WIDTH + 5] = [
        4.0, 3.0, 2.0, -4.0, 4.0,
    ]
    return weight


def build_physical_proposal_fixture() -> PhysicalProposalFixture:
    """Build the compact payload-derived three-stage/head fixture."""

    ideal: dict[str, np.ndarray] = {}
    for stage in range(PROPOSAL_STAGE_COUNT):
        ideal[f"stage_weight_{stage}"] = _stage_weight(stage, main=False)
        ideal[f"stage_main_weight_{stage}"] = _stage_weight(stage, main=True)
    ideal.update({
        "stage_biases": _stage_biases(),
        "head_function_weight": _head_function(),
        "head_scale": np.asarray([0.75], dtype=np.float32),
        "head_base": np.asarray([-0.25, 0.125, 0.375, -0.5], dtype=np.float32),
        "head_norm_weight": (
            (np.arange(PROPOSAL_HIDDEN_WIDTH) % 9).astype(np.float32) + 8.0
        ) / np.float32(12.0),
        "target_output_weight": _target_output(),
        "markov_w1": _markov_w1(),
        "markov_w2": _markov_w2(),
        "confidence_weight": _confidence_weight(),
    })
    packed: dict[str, PackedWeight] = {}
    for name, value in ideal.items():
        if name == "head_function_weight":
            packed[name] = pack_f16(value)
        elif name in {
            "stage_biases", "head_scale", "head_base", "head_norm_weight"
        }:
            packed[name] = pack_f32(value)
        else:
            packed[name] = pack_q8_0(value)

    inputs: dict[str, object] = {
        "initial_hc": _readonly(_initial_hc()),
        "main_x": _readonly(_main_x()),
        "stage_weights": tuple(
            packed[f"stage_weight_{stage}"]
            for stage in range(PROPOSAL_STAGE_COUNT)
        ),
        "stage_main_weights": tuple(
            packed[f"stage_main_weight_{stage}"]
            for stage in range(PROPOSAL_STAGE_COUNT)
        ),
        "stage_biases": packed["stage_biases"].dequantized,
        "head_function_weight": packed["head_function_weight"],
        "head_scale": packed["head_scale"].dequantized,
        "head_base": packed["head_base"].dequantized,
        "head_norm_weight": packed["head_norm_weight"].dequantized,
        "target_output_weight": packed["target_output_weight"],
        "markov_w1": packed["markov_w1"],
        "markov_w2": packed["markov_w2"],
        "confidence_weight": packed["confidence_weight"],
        "pending_token_id": PROPOSAL_PENDING_TOKEN,
        "confidence_threshold": PROPOSAL_CONFIDENCE_THRESHOLD,
    }
    return PhysicalProposalFixture(
        MappingProxyType(inputs), MappingProxyType(packed)
    )


def proposal_payload_manifest(
    fixture: PhysicalProposalFixture,
) -> dict[str, object]:
    """Return stable input and payload identities for check mode."""

    payload = hashlib.sha256()
    payload_bytes = 0
    weights: dict[str, object] = {}
    for name in sorted(fixture.packed_weights):
        weight = fixture.packed_weights[name]
        name_bytes = name.encode("utf-8")
        payload.update(len(name_bytes).to_bytes(2, "little"))
        payload.update(name_bytes)
        payload.update(len(weight.payload).to_bytes(8, "little"))
        payload.update(weight.payload)
        payload_bytes += len(weight.payload)
        weights[name] = {
            "storage": weight.storage,
            "shape": list(weight.logical_shape),
            "bytes": len(weight.payload),
            "sha256": weight.sha256,
        }
    initial = np.asarray(fixture.inputs["initial_hc"], dtype="<f4").tobytes()
    main = np.asarray(fixture.inputs["main_x"], dtype="<f4").tobytes()
    return {
        "fixture_version": 1,
        "geometry": {
            "proposal_rows": PROPOSAL_ROWS,
            "hc_lanes": PROPOSAL_HC_LANES,
            "hidden_width": PROPOSAL_HIDDEN_WIDTH,
            "stage_count": PROPOSAL_STAGE_COUNT,
            "vocab": PROPOSAL_VOCAB,
            "markov_rank": PROPOSAL_MARKOV_RANK,
        },
        "pending_token": PROPOSAL_PENDING_TOKEN,
        "confidence_threshold": PROPOSAL_CONFIDENCE_THRESHOLD,
        "initial_hc_sha256": hashlib.sha256(initial).hexdigest(),
        "main_x_sha256": hashlib.sha256(main).hexdigest(),
        "payload_bytes": payload_bytes,
        "payload_sha256": payload.hexdigest(),
        "weights": weights,
    }


# This declaration was frozen from the reviewed deterministic generator and
# deliberately does not call the numerical proposal oracle.  ``--check`` makes
# payload or formula drift fail closed without duplicating generated weights.
FROZEN_PROPOSAL_MANIFEST_SHA256 = (
    "fb303e80eed40ec3756af0b3632766007c11ab90823410db770af6d64fe504d5"
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check", action="store_true",
        help="fail if the deterministic payload manifest changed",
    )
    args = parser.parse_args()
    forbidden_modules = {
        name for name in sys.modules
        if name in {
            "tools.dspark_oracle.reference",
            "tools.dspark_oracle.physical_fixture",
        }
        or name.endswith(".dspark_oracle.reference")
        or name.endswith(".dspark_oracle.physical_fixture")
    }
    if forbidden_modules:
        print(
            "DSpark proposal fixture imported a coupled oracle module: "
            + ", ".join(sorted(forbidden_modules)),
            flush=True,
        )
        return 2
    manifest = proposal_payload_manifest(build_physical_proposal_fixture())
    if args.check:
        manifest_bytes = json.dumps(
            manifest, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if hashlib.sha256(manifest_bytes).hexdigest() != (
            FROZEN_PROPOSAL_MANIFEST_SHA256
        ):
            print("DSpark proposal fixture is stale", flush=True)
            return 1
        print("DSpark proposal fixture is current")
        return 0
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
